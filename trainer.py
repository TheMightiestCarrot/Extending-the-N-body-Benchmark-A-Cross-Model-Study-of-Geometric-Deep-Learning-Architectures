import datetime
import json
import os
import pickle
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
from jsonargparse import namespace_to_dict
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.data import Batch
from tqdm import tqdm

import wandb
from helper_scripts.infer_self_feed import run_inference
from self_feed import MACROS_DIR_NAME, SelfFeedError, self_feed
from training.losses import (CentreOfMassLoss, EnergyLoss, 
                             MomentumLoss, TargetCommonLoss)
from utils.config_models import PrecisionMode
from utils.get_device import get_device
from utils.ks_utils import _combine_pvalues_fisher, _ks_p
from utils.utils_data import calculate_energies

ENERGY_ERROR_THRESHOLDS = [2.5, 5]  # relative error threshold for energy comparison


class Trainer:
    def __init__(
        self, model, dataloader, validation_dataloader=None, args=None
    ) -> None:
        self.args = args
        self.device = get_device(self.args.gpu_id)
        self.model = model.to(self.device)

        self.targets = self.args.target.split("+")
        if self.args.precision_mode == PrecisionMode.DOUBLE:
            self.model = self.model.double()
        else:
            self.model = self.model.float()
            if self.args.precision_mode == PrecisionMode.AUTOCAST:
                self.scaler = GradScaler()

        self.dataloader = dataloader
        self.optimizer = self.create_optimizer()
        self.lr_scheduler = self.create_lr_scheduler()

        # Determine atom types if available before metrics are created
        self.atom_type_indices = []
        self.atom_type_labels = {}
        if self.args.per_atom_loss:
            self.prepare_per_atom_loss()

        self.losses = self.initialize_losses()
        self.metrics = {"train": self.create_metrics()}
        self.validation_dataloader = validation_dataloader
        if validation_dataloader is not None:
            self.metrics["valid"] = self.create_metrics()

        self.training_start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print(f"Started training: {self.training_start_time}")

        self.step_count = 0
        subfolder = self.args.model_type
        name_suffix = "" if self.args.run_name is None else f"__{self.args.run_name}"
        self.save_dir_path = f"runs/{subfolder}/{self.training_start_time}{name_suffix}"
        os.makedirs(self.save_dir_path, exist_ok=True)
        self.best_metrics = {}

        if self.args.model_path:
            self.load_model_from_checkpoint()
        else:
            print("using default model weights")

        # num_neighbors is set by the model during creation

    def load_model_from_checkpoint(self):
        self._model_restoring_links()
        checkpoint = torch.load(
            self.args.model_path, map_location=self.device, weights_only=False
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint and self.lr_scheduler:
            self.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "step_count" in checkpoint:
            self.step_count = checkpoint["step_count"]
        if "best_metrics" in checkpoint:
            self.best_metrics = checkpoint["best_metrics"]
        print(f"Loaded model and optimizer state from {self.args.model_path}")

    def _model_restoring_links(self):
        restored_path = os.path.abspath(
            os.path.join(os.path.dirname(self.args.model_path))
        )
        # create symlink to restored model's folder in current folder
        restored_dirname = os.path.basename(os.path.normpath(restored_path))
        restored_from_path = os.path.abspath(
            os.path.join(self.save_dir_path, "restored_from", restored_dirname)
        )
        relative_restored_path = os.path.relpath(
            restored_path, start=os.path.dirname(restored_from_path)
        )
        os.makedirs(os.path.dirname(restored_from_path))
        os.symlink(relative_restored_path, restored_from_path, target_is_directory=True)

        # Create a symlink to the current model's folder in the restored model's folder
        restoring_path = os.path.abspath(
            os.path.join(
                restored_path, "restoring", os.path.basename(self.save_dir_path)
            )
        )
        relative_save_dir_path = os.path.relpath(
            self.save_dir_path, start=os.path.dirname(restoring_path)
        )
        os.makedirs(os.path.dirname(restoring_path), exist_ok=True)
        os.symlink(relative_save_dir_path, restoring_path, target_is_directory=True)

    def create_metrics(self):
        metrics = {"loss": torchmetrics.MeanMetric().to(self.device)}
        for target in self.targets:
            metrics[f"{target}_perc_error"] = torchmetrics.MeanMetric().to(self.device)

        if self.args.energy_loss:
            metrics["mae_energy"] = torchmetrics.MeanMetric().to(self.device)
            metrics["energy_perc_error"] = torchmetrics.MeanMetric().to(self.device)

        if self.args.com_loss:
            metrics["com_perc_error"] = torchmetrics.MeanMetric().to(self.device)

        for loss in self.losses:
            metrics[loss.name] = torchmetrics.MeanMetric().to(self.device)

        if self.args.per_atom_loss and self.atom_type_indices:
            if any(t in self.args.target for t in ["pos", "pos_dt"]):
                for idx in self.atom_type_indices:
                    metrics[f"position_loss_type_{self.atom_type_labels[idx]}"] = (
                        torchmetrics.MeanMetric().to(self.device)
                    )
            if any(t in self.args.target for t in ["vel", "vel_dt"]):
                for idx in self.atom_type_indices:
                    metrics[f"velocity_loss_type_{self.atom_type_labels[idx]}"] = (
                        torchmetrics.MeanMetric().to(self.device)
                    )

        return metrics

    def reset_metrics(self, partition):
        for metric in self.metrics[partition].values():
            metric.reset()

    def initialize_losses(self):
        losses = []

        losses.append(TargetCommonLoss(args=self.args))
        if self.args.com_loss:
            losses.append(CentreOfMassLoss(args=self.args))
        if self.args.energy_loss:
            losses.append(EnergyLoss(args=self.args))
        if self.args.momentum_loss:
            losses.append(
                MomentumLoss(weight=self.args.momentum_loss_weight, args=self.args)
            )

        return losses

    def create_optimizer(self):
        return torch.optim.AdamW(
            self.model.parameters(),
            weight_decay=1e-8,
            lr=self.args.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9,
        )

    def create_lr_scheduler(self):
        return LambdaLR(
            optimizer=self.optimizer,
            lr_lambda=lambda step: self._rate(
                step,
                factor=self.args.learning_rate_factor,
                warmup=self.args.learning_rate_warmup_steps,
            ),
        )

    def _rate(self, step, factor, warmup):
        if step == 0:
            step = 1
        return factor * (
            self.model.get_model_size() ** (-0.5)
            * min(step ** (-0.5), step * warmup ** (-1.5))
        )

    def _limit_gradients(self):
        if self.args.clip_gradients_value is not None:
            torch.nn.utils.clip_grad_value_(
                self.model.parameters(), clip_value=self.args.clip_gradients_value
            )
        if self.args.clip_gradients_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.args.clip_gradients_norm
            )

    def _gradient_isnan(self):
        for _name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm()
                if grad_norm.isnan():
                    return True
        return False

    def prepare_per_atom_loss(self):
        dataset = getattr(self.dataloader, "dataset", None)
        mapper = getattr(dataset, "atom_type_mapper", None)
        if mapper:
            mapper = {int(k): int(v) for k, v in mapper.items()}
            self.atom_type_indices = sorted(set(mapper.values()))
            try:
                from ase.data import chemical_symbols

                self.atom_type_labels = {
                    idx: chemical_symbols[atomic_num]
                    for atomic_num, idx in mapper.items()
                }
            except Exception:
                self.atom_type_labels = {
                    idx: str(atomic_num) for atomic_num, idx in mapper.items()
                }

    def train_one_step(self, data):
        # Zero grads up-front for this step
        self.optimizer.zero_grad()
        data = self.dataloader.preprocess_batch(data[0], self.device)

        took_step = False

        if self.args.precision_mode == PrecisionMode.AUTOCAST:
            # Forward under autocast
            with autocast():
                pred, loss = self.forward_pass(data)

            # Optional early abort on NaN/Inf activations reported by model
            self._pending_debug_stats = None
            if hasattr(self.model, "get_and_reset_debug_stats"):
                try:
                    self._pending_debug_stats = self.model.get_and_reset_debug_stats()
                except Exception:
                    self._pending_debug_stats = None
            if getattr(self.args, "abort_on_nan_activations", False) and self._pending_debug_stats:
                if any(
                    d.get("L0.inter.nan_or_inf", False)
                    or d.get("L0.mix.nan_or_inf", False)
                    or any(v for k, v in d.items() if k.endswith("nan_or_inf"))
                    for d in self._pending_debug_stats
                ):
                    # Skip backward/step if activations exploded
                    return pred, loss

            # Backward with GradScaler
            self.scaler.scale(loss).backward()

            # Unscale before gradient clipping and checks
            self.scaler.unscale_(self.optimizer)

            # Optionally drop step on NaN/Inf gradients
            if self.args.discard_nan_gradients and self._gradient_isnan():
                # Do not step or advance scheduler; clear grads and update scaler state
                try:
                    self.optimizer.zero_grad(set_to_none=True)
                except TypeError:
                    self.optimizer.zero_grad()
                # Still update scaler to adjust its scale heuristics
                self.scaler.update()
                return pred, loss

            # Clip after unscale
            self._limit_gradients()

            # Optimizer step via scaler
            self.scaler.step(self.optimizer)
            self.scaler.update()
            took_step = True

        else:
            # Standard precision forward
            pred, loss = self.forward_pass(data)

            # Optional early abort on NaN/Inf activations reported by model
            self._pending_debug_stats = None
            if hasattr(self.model, "get_and_reset_debug_stats"):
                try:
                    self._pending_debug_stats = self.model.get_and_reset_debug_stats()
                except Exception:
                    self._pending_debug_stats = None
            if getattr(self.args, "abort_on_nan_activations", False) and self._pending_debug_stats:
                if any(
                    d.get("L0.inter.nan_or_inf", False)
                    or d.get("L0.mix.nan_or_inf", False)
                    or any(v for k, v in d.items() if k.endswith("nan_or_inf"))
                    for d in self._pending_debug_stats
                ):
                    # Skip backward/step if activations exploded
                    return pred, loss

            # Backward to populate gradients
            loss.backward()

            # Optionally drop step on NaN/Inf gradients
            if self.args.discard_nan_gradients and self._gradient_isnan():
                try:
                    self.optimizer.zero_grad(set_to_none=True)
                except TypeError:
                    self.optimizer.zero_grad()
                return pred, loss

            # Clip and step optimizer
            self._limit_gradients()
            self.optimizer.step()
            took_step = True

        # Only advance LR schedule if we actually took an optimizer step
        if took_step:
            self.lr_scheduler.step()

        # Optional per-layer diagnostics to wandb
        if getattr(self.args, "debug_layer_stats_every", None) is not None:
            if self.step_count % int(self.args.debug_layer_stats_every) == 0:
                debug_stats = getattr(self, "_pending_debug_stats", None)
                if debug_stats:
                    # Flatten with per-layer keys
                    flat = {}
                    for i, d in enumerate(debug_stats):
                        for k, v in d.items():
                            flat[f"debug/{k}"] = v
                    try:
                        wandb.log(flat, step=self.step_count)
                    except Exception:
                        pass
                    try:
                        import os, json
                        out_path = os.path.join(self.save_dir_path, "layer_stats.jsonl")
                        with open(out_path, "a") as f:
                            record = {"step": int(self.step_count), **flat}
                            f.write(json.dumps(record) + "\n")
                    except Exception:
                        pass
                self._pending_debug_stats = None

        if (
            self.args.sync_cuda_cores
            and torch.cuda.is_available()
            and self.device.type == "cuda"
        ):
            torch.cuda.synchronize()
        return pred, loss

    def train_one_epoch(self):
        number_of_steps = (
            len(self.dataloader)
            if self.args.steps_per_epoch == -1
            else self.args.steps_per_epoch
        )
        number_of_steps_progress = tqdm(
            range(number_of_steps),
            desc=f"Epoch {self.step_count}",
            bar_format="{desc} | {bar:20} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}] | {postfix}",
            dynamic_ncols=True,
        )
        for _i in number_of_steps_progress:
            self.model.train()

            data, _ = self.dataloader.get_batch()
            t0 = time.time()

            pred, loss = self.train_one_step(data)
            t1 = time.time()
            dt = (t1 - t0) * 1000
            if loss != 0:
                # Fix: pass data[0] which is the actual graph, not the list
                graph_data = data[0] if isinstance(data, list) else data
                self.update_metrics(pred, graph_data, loss)
            number_of_steps_progress.set_postfix_str(
                self.results_str(dt, partition="train")
            )
        self.log_results(partition="train")
        self.reset_metrics("train")

        return self.model

    def validate_one_epoch(self, save_data=False):
        # Handle both fixed-length and streaming dataloaders
        if hasattr(self.validation_dataloader, "__len__"):
            num_validation_steps = len(self.validation_dataloader)
        else:
            print(
                "Validation not needed for OTF datasets (all samples are fresh). Returning."
            )
            return self.model

        number_of_steps_progress = tqdm(
            range(num_validation_steps),
            desc=f"Validation (epoch {self.step_count - 1})",
            bar_format="{desc} | {bar:20} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}] | {postfix}",
        )

        for batch_number in number_of_steps_progress:
            self.model.eval()

            dataloader = self.validation_dataloader

            data, _ = dataloader.get_batch()
            data = dataloader.preprocess_batch(data[0], self.device)
            t0 = time.time()
            with torch.no_grad():
                pred, loss = self.forward_pass(data, partition="valid")

            if (
                self.args.sync_cuda_cores
                and torch.cuda.is_available()
                and self.device.type == "cuda"
            ):
                torch.cuda.synchronize()
            t1 = time.time()
            dt = (t1 - t0) * 1000

            if save_data:
                self.save_data(data, pred, "valid", batch_number)
            self.update_metrics(pred, data, loss, partition="valid")
            number_of_steps_progress.set_postfix_str(
                self.results_str(dt, partition="valid")
            )
        log_dict = self.log_results(partition="valid")
        self.maybe_save_on_best_metric(log_dict)

        self.reset_metrics("valid")
        return self.model

    def save_data(self, data, pred, partition, batch_number):
        data_to_save = {"input": data, "prediction": pred}
        dir_to_save = os.path.join(
            self.save_dir_path, "predictions", f"epoch_{self.step_count - 1}"
        )
        os.makedirs(dir_to_save, exist_ok=True)
        print(f"saving data to {dir_to_save}")
        torch.save(
            data_to_save,
            os.path.join(
                dir_to_save,
                f"{partition}_batch_{batch_number}.pth",
            ),
        )

    def maybe_save_on_best_metric(self, log_dict):
        if "valid_loss" not in self.best_metrics:
            self.best_metrics["valid_loss"] = np.inf
        if log_dict["valid/loss"] < self.best_metrics["valid_loss"]:
            self.best_metrics["valid_loss"] = log_dict["valid/loss"]
            self.save_model(filename="model_best_valid_loss.pth")

    def forward_pass(self, data, partition="train"):
        metrics = self.metrics[partition]
        pred = self.model(data)
        pred = self.dataloader.postprocess_batch(pred, self.device)

        total_loss = 0
        for loss in self.losses:
            loss_result = loss.forward(pred=pred, data=data)
            total_loss += loss_result
            metrics[loss.name].update(loss_result)

        return pred, total_loss

    def update_metrics(self, pred, graph, loss, partition="train"):
        metrics = self.metrics[partition]
        metrics["loss"].update(loss)

        targets = self.args.target.split("+")
        for i, target in enumerate(targets):
            pred_quantity = pred[..., 3 * i : 3 * (i + 1)]
            target_quantity = graph.y[..., 3 * i : 3 * (i + 1)]

            err = pred_quantity.detach() - target_quantity
            err_l2 = torch.norm(err, dim=-1)
            target_l2 = torch.norm(target_quantity, dim=-1)

            percentage_error = (err_l2 / (target_l2 + 1e-12)).mean() * 100
            metrics[f"{target}_perc_error"].update(percentage_error)

            if (
                self.args.per_atom_loss
                and self.atom_type_indices
                and hasattr(graph, "atom_types")
            ):
                atom_types = graph.atom_types.squeeze()
                for idx in self.atom_type_indices:
                    mask = atom_types == idx
                    if mask.any():
                        loss_val = F.mse_loss(
                            pred_quantity[mask], target_quantity[mask]
                        )
                        if "pos" in target:
                            metrics[
                                f"position_loss_type_{self.atom_type_labels[idx]}"
                            ].update(1e10 * loss_val)
                        elif "vel" in target:
                            metrics[
                                f"velocity_loss_type_{self.atom_type_labels[idx]}"
                            ].update(1e10 * loss_val)

    def save_training_args(self):
        model_save_path = (
            f"{self.save_dir_path}/{self.args.dataset_name}_best_model.pth"
        )
        with open(
            os.path.join(os.path.dirname(model_save_path), "training_args.json"),
            "w",
        ) as f:
            args_dict = namespace_to_dict(self.args)
            json.dump({"args": args_dict}, f, indent=4)

    def save_model_params(self):
        with open(f"{self.save_dir_path}/model_params.json", "w") as f:
            json.dump(self.model.get_serializable_attributes(), f, indent=4)

    def save_dataset_attributes(self):
        dataset_save_path = f"{self.save_dir_path}/{self.args.dataset_name}_dataset"
        os.makedirs(dataset_save_path, exist_ok=True)
        attrs = self.dataloader.dataset.get_serializable_attributes()
        print(f"Training with attrs: {attrs}")
        # save dataset attributes to json
        with open(f"{dataset_save_path}/metadata.json", "w") as f:
            json.dump(attrs, f, indent=4)

        # Save the scaler
        if hasattr(self.dataloader.dataset, "scalers"):
            for scaler in self.dataloader.dataset.scalers:
                with open(f"{dataset_save_path}/scaler.pkl", "wb") as scaler_file:
                    pickle.dump(scaler, scaler_file)

    def create_wandb_run(self):
        WANDB_ID_FILENAME = "wandb_id"
        run_id = None
        if self.args.model_path is not None:
            run_id_path = os.path.join(
                os.path.dirname(self.args.model_path), WANDB_ID_FILENAME
            )
            if os.path.exists(run_id_path):
                with open(run_id_path, "r") as f:
                    run_id = f.read().strip()

        # allow environment variables to override project/entity/group/name
        project = os.getenv("WANDB_PROJECT", "nbody")
        entity = os.getenv("WANDB_ENTITY")
        group = os.getenv("WANDB_RUN_GROUP")
        job_type = os.getenv("WANDB_JOB_TYPE")
        run_name = os.getenv("WANDB_NAME", self.save_dir_path)

        run = wandb.init(
            project=project,
            entity=entity,
            group=group,
            job_type=job_type,
            config=self.args,
            name=run_name,
            id=run_id,
            resume="allow",
        )

        wandb.define_metric("train/*", step_metric="train/step")
        if self.args.do_validation:
            wandb.define_metric("valid/*", step_metric="valid/step")
        if self.args.test_macros_every is not None:
            wandb.define_metric("self_feed/*", step_metric="self_feed/step")
            # readable summaries for transformed p-values
            try:
                wandb.define_metric(
                    "self_feed/ks_*_neglog10", step_metric="self_feed/step", summary="min"
                )
                wandb.define_metric(
                    "self_feed/ks_*_log10", step_metric="self_feed/step", summary="max"
                )
                wandb.define_metric(
                    "self_feed/ks_combined_neglog10", step_metric="self_feed/step", summary="min"
                )
                wandb.define_metric(
                    "self_feed/ks_combined_log10", step_metric="self_feed/step", summary="max"
                )
            except Exception:
                pass

        # Save the run ID to a file for later use
        with open(os.path.join(self.save_dir_path, WANDB_ID_FILENAME), "w") as f:
            f.write(run.id)
        return run

    def save_model(self, save_path=None, filename="model.pth", final=False):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step_count": self.step_count,
            "best_metrics": self.best_metrics,
        }
        if self.lr_scheduler:
            checkpoint["scheduler_state_dict"] = self.lr_scheduler.state_dict()
        if save_path is None:
            save_path = self.save_dir_path

        os.makedirs(save_path, exist_ok=True)
        torch.save(checkpoint, os.path.join(save_path, filename))
        print(f"Model and optimizer state saved to {save_path}")
        if final:
            print(
                f"""
                To continue training, run 
                    python train.py --config {os.path.join(self.save_dir_path, "config.yaml")} --trainer.model_path {os.path.join(save_path, "model.pth")}
                    
                To run self feed, run 
                    python self_feed.py --config {os.path.join(self.save_dir_path, "config.yaml")} --trainer.model_path {os.path.join(save_path, "model.pth")}
                """
            )

    def log_results(self, partition: str = "train"):
        metrics = self.metrics[partition]
        log_dict = {
            f"{partition}/{name}": metric.compute().item()
            for name, metric in metrics.items()
        }
        log_dict = self.modify_log_dict(log_dict)
        log_dict[f"{partition}/step"] = (
            self.step_count if partition == "train" else self.step_count - 1
        )
        wandb.log(log_dict, commit=True)
        return log_dict

    def results_str(self, dt, partition: str = "train"):
        metrics = self.metrics[partition]
        print_str = ""
        for name, metric in metrics.items():
            print_str += f"{name.replace('_', ' ')}: {metric.compute().item():.5f} | "

        print_str += (
            f"lr={self.optimizer.param_groups[0]['lr']:.2e} | "
            f"{dt:.2f} ms @ {((self.args.batch_size) / (dt/1000)):.2f} ex/s"
        )
        print_str = self.modify_print_str(print_str)
        return print_str

    def modify_log_dict(self, log_dict):
        if self.args.per_atom_loss and self.atom_type_labels:
            updated = {}
            for key, value in log_dict.items():
                new_key = key
                for idx, label in self.atom_type_labels.items():
                    new_key = new_key.replace(f"_type_{idx}", f"_{label}")
                updated[new_key] = value
            return updated
        return log_dict

    def modify_print_str(self, print_str):
        if self.args.per_atom_loss and self.atom_type_labels:
            for idx, label in self.atom_type_labels.items():
                print_str = print_str.replace(f"_type_{idx}", f"_{label}")
        return print_str

    def self_feed_postprocess_common(
        self,
        *,
        energies: dict,
        steps_survived: int,
        save_dir: str,
        rdf_metrics: dict | None = None,
        vacf_metrics: dict | None = None,
        metrics_filename: str = "md_macro_metrics.json",
    ):
        """Shared post-processing for self-feed outputs across domains.

        Parameters
        - energies: dict with structure:
            {"simulation": {"potential": np.ndarray, "kinetic": np.ndarray, "total": np.ndarray},
             "self_feed": {"potential": np.ndarray, "kinetic": np.ndarray, "total": np.ndarray}}
          Each array should be 1D over time (steps); aggregating across batch is okay.
        - steps_survived: rollout length achieved by the self-feed generator
        - save_dir: checkpoint directory for this training step
        - rdf_metrics, vacf_metrics: optional macro metrics dicts
        - metrics_filename: file name used to persist compact metrics JSON for later plotting
        """
        os.makedirs(save_dir, exist_ok=True)

        # Energy thresholds metric (how many steps within threshold)
        sim_total = np.asarray(energies["simulation"]["total"]).reshape(-1)
        sf_total = np.asarray(energies["self_feed"]["total"]).reshape(-1)
        min_len = min(len(sim_total), len(sf_total))
        ratio = np.abs(sim_total[:min_len] / (sf_total[:min_len] + 1e-12))

        steps_metric = {}
        for t in ENERGY_ERROR_THRESHOLDS:
            mask = np.where((1.0 / t < ratio) & (ratio < t))[0]
            steps_metric[t] = int(mask[-1] + 1) if mask.size > 0 else 0

        print(
            f"Self feed energy within {'| '.join([f'{th*100:.0f}%: {steps_metric[th]} steps' for th in ENERGY_ERROR_THRESHOLDS])}"
        )

        # KS p-values (per-metric + combined)
        pvals = {}
        try:
            for key in ["total", "potential", "kinetic"]:
                pvals[f"energy_{key}"] = _ks_p(
                    energies["simulation"][key], energies["self_feed"][key]
                )
        except Exception:
            pass
        if rdf_metrics is not None:
            try:
                pvals["rdf"] = _ks_p(
                    rdf_metrics.get("rdf_simulation", []),
                    rdf_metrics.get("rdf_self_feed", []),
                )
            except Exception:
                pass
        if vacf_metrics is not None and isinstance(vacf_metrics.get("vacf", {}), dict):
            vacf = vacf_metrics["vacf"]
            try:
                pvals["vacf"] = _ks_p(
                    vacf.get("simulation", []), vacf.get("self_feed", [])
                )
            except Exception:
                pass
        p_combined = _combine_pvalues_fisher(list(pvals.values()))

        # persist compact data for ks-style post-hoc ranking
        try:
            to_json = {
                "energies": {
                    "simulation_total": np.asarray(energies["simulation"]["total"]).tolist(),
                    "self_feed_total": np.asarray(energies["self_feed"]["total"]).tolist(),
                    "simulation_potential": np.asarray(energies["simulation"]["potential"]).tolist(),
                    "self_feed_potential": np.asarray(energies["self_feed"]["potential"]).tolist(),
                    "simulation_kinetic": np.asarray(energies["simulation"]["kinetic"]).tolist(),
                    "self_feed_kinetic": np.asarray(energies["self_feed"]["kinetic"]).tolist(),
                },
                "ks_pvalues": {
                    **{k: (float(v) if v == v else float("nan")) for k, v in pvals.items()},
                    "combined": (
                        float(p_combined) if p_combined == p_combined else float("nan")
                    ),
                },
            }
            if rdf_metrics is not None:
                to_json["rdf"] = {
                    "simulation": np.asarray(rdf_metrics.get("rdf_simulation", [])).reshape(-1).tolist(),
                    "self_feed": np.asarray(rdf_metrics.get("rdf_self_feed", [])).reshape(-1).tolist(),
                    "mse_mean": rdf_metrics.get("mse_mean"),
                    "mse_median": rdf_metrics.get("mse_median"),
                }
            if vacf_metrics is not None and isinstance(vacf_metrics.get("vacf", {}), dict):
                vacf = vacf_metrics["vacf"]
                to_json["vacf"] = {
                    "simulation": np.asarray(vacf.get("simulation", [])).tolist(),
                    "self_feed": np.asarray(vacf.get("self_feed", [])).tolist(),
                    "mae": vacf_metrics.get("vacf_mae"),
                }
            with open(os.path.join(save_dir, metrics_filename), "w") as f:
                json.dump(to_json, f)
        except Exception:
            pass

        # Best checkpoint on primary threshold
        primary = ENERGY_ERROR_THRESHOLDS[0]
        if "self_feed_steps" not in self.best_metrics:
            self.best_metrics["self_feed_steps"] = 0
        if steps_metric[primary] >= self.best_metrics["self_feed_steps"]:
            self.best_metrics["self_feed_steps"] = steps_metric[primary]
            self.save_model(filename="model_best_self_feed.pth")

        # Log to W&B
        log_payload = {
            "self_feed/steps_survived": int(steps_survived),
            "self_feed/energy_steps_within_threshold": steps_metric[primary],
            "self_feed/step": self.step_count - 1,
        }
        if rdf_metrics is not None:
            log_payload["self_feed/rdf_mse_mean"] = rdf_metrics.get(
                "mse_mean", float("nan")
            )
            log_payload["self_feed/rdf_mse_median"] = rdf_metrics.get(
                "mse_median", float("nan")
            )
        if vacf_metrics is not None:
            log_payload["self_feed/vacf_mae"] = vacf_metrics.get(
                "vacf_mae", float("nan")
            )
        # add ks p-values to wandb log
        for key, val in pvals.items():
            if val == val and val > 0.0:
                safe = max(float(val), 1e-300)
                log_payload[f"self_feed/ks_{key}"] = safe
                log_payload[f"self_feed/ks_{key}_log10"] = float(np.log10(safe))
                log_payload[f"self_feed/ks_{key}_neglog10"] = float(-np.log10(safe))
            else:
                # clamp to tiny positive to enable log-scale plotting
                log_payload[f"self_feed/ks_{key}"] = float(1e-300)
                log_payload[f"self_feed/ks_{key}_log10"] = float(np.log10(1e-300))
                log_payload[f"self_feed/ks_{key}_neglog10"] = float(-np.log10(1e-300))
        if p_combined == p_combined and p_combined > 0.0:
            safe_c = max(float(p_combined), 1e-300)
            log_payload["self_feed/ks_combined"] = safe_c
            log_payload["self_feed/ks_combined_log10"] = float(np.log10(safe_c))
            log_payload["self_feed/ks_combined_neglog10"] = float(-np.log10(safe_c))
        else:
            log_payload["self_feed/ks_combined"] = float(1e-300)
            log_payload["self_feed/ks_combined_log10"] = float(np.log10(1e-300))
            log_payload["self_feed/ks_combined_neglog10"] = float(-np.log10(1e-300))
        wandb.log(log_payload, commit=True)

        # also log plotly charts mirroring ks_test output with log y-axis
        try:
            import plotly.graph_objects as go

            ckpt_root = os.path.join(self.save_dir_path, "checkpoints")
            step_dirs = sorted([d for d in os.listdir(ckpt_root) if d.isdigit()], key=int)

            steps = []
            combined_series = []
            per_metric = {k: [] for k in [
                "energy_total",
                "energy_potential",
                "energy_kinetic",
                "rdf",
                "vacf",
            ]}

            for d in step_dirs:
                path = os.path.join(ckpt_root, d, metrics_filename)
                if not os.path.exists(path):
                    continue
                try:
                    with open(path, "r") as f:
                        m = json.load(f)
                except Exception:
                    continue
                pks = m.get("ks_pvalues", {})
                c = pks.get("combined", None)
                if c is None or not (c == c) or c <= 0.0:
                    continue
                steps.append(int(d))
                combined_series.append(max(float(c), 1e-300))
                for k in per_metric.keys():
                    v = pks.get(k, np.nan)
                    if v == v and v > 0.0:
                        per_metric[k].append(max(float(v), 1e-300))
                    else:
                        per_metric[k].append(np.nan)

            if steps:
                fig_c = go.Figure()
                fig_c.add_trace(go.Scatter(x=steps, y=combined_series, mode="lines+markers", name="combined p"))
                fig_c.update_layout(
                    title=f"combined p-values vs checkpoints",
                    xaxis_title="checkpoint",
                    yaxis_title="combined p (fisher)",
                    template="plotly_white",
                    yaxis=dict(type="log", range=[-300, 0]),
                )
                wandb.log({"self_feed/ks_plot_combined": wandb.Plotly(fig_c)}, commit=False)

                fig_i = go.Figure()
                for k, ys in per_metric.items():
                    if all((y != y) for y in ys):
                        continue
                    fig_i.add_trace(go.Scatter(x=steps, y=ys, mode="lines+markers", name=k))
                fig_i.update_layout(
                    title="per-metric p-values vs checkpoints",
                    xaxis_title="checkpoint",
                    yaxis_title="p-value",
                    template="plotly_white",
                    yaxis=dict(type="log", range=[-300, 0]),
                )
                wandb.log({"self_feed/ks_plot_individual": wandb.Plotly(fig_i)}, commit=True)
        except Exception:
            pass

        return steps_metric

    def _compute_nbody_energies(self, loc: np.ndarray, vel: np.ndarray, G: float, softening: float):
        """Compute per-step energy time series averaged across batch for N-body trajectories.

        loc: (batch, steps, nodes, dims)
        vel: (batch, steps, nodes, dims)
        returns dict with keys potential/kinetic/total as 1D arrays over steps
        """
        # ensure numpy
        loc = np.asarray(loc)
        vel = np.asarray(vel)
        batch, steps, n, _ = loc.shape

        kinetic = np.zeros((batch, steps), dtype=np.float64)
        potential = np.zeros((batch, steps), dtype=np.float64)
        upper_idx = np.triu_indices(n, 1)

        for b in range(batch):
            L = loc[b]  # (steps, n, d)
            V = vel[b]
            kinetic[b] = 0.5 * np.sum(V * V, axis=(1, 2))  # unit masses

            # pairwise distances per step: (steps, n, n)
            dx = L[:, None, :, 0] - L[:, :, None, 0]
            dy = L[:, None, :, 1] - L[:, :, None, 1]
            if L.shape[2] > 2:
                dz = L[:, None, :, 2] - L[:, :, None, 2]
                dist2 = dx * dx + dy * dy + dz * dz
            else:
                dist2 = dx * dx + dy * dy
            inv_r = np.sqrt(dist2 + softening * softening)
            inv_r[inv_r > 0] = 1.0 / inv_r[inv_r > 0]
            # sum over upper triangle
            inv_sum = np.sum(inv_r[:, upper_idx[0], upper_idx[1]], axis=1)
            potential[b] = -G * inv_sum  # masses assumed 1

        # average across batch to obtain a representative time series
        pot_series = potential.mean(axis=0)
        kin_series = kinetic.mean(axis=0)
        tot_series = pot_series + kin_series
        return {"potential": pot_series, "kinetic": kin_series, "total": tot_series}

    def run_self_feed(self):
        """Run self-feeding for n-body simulations."""
        print(f"Running self feed (epoch {self.step_count - 1})")

        is_nbody = "nbody" in self.args.dataloader_type
        if not is_nbody:
            # This method should be overridden by subclasses for non-nbody dataloaders
            print(f"ℹ️ Self-feed not implemented for this trainer type, skipping.")
            return

        # optional cap on rollout length for speed-sensitive models (e.g., cgenn during HPO)
        max_steps = getattr(self.args, "self_feed_limit_steps", None)

        _, combined_locations, _ = run_inference(
            model_type=self.args.model_type,
            dataloader=self.dataloader,
            model_path=self.save_dir_path,
            model=self.model,
            save_dir=f"{self.save_dir_path}/checkpoints/{self.step_count}",
            print_step=True,
            n_bodies=None,
            plot_macros=True,
            num_neighbors=None,
            device=self.device,
            max_rollout_steps=max_steps,
        )

        steps = combined_locations.shape[2] - 1
        # compute simple energy macros and log like MD, without duplication
        try:
            dataset = getattr(self.dataloader, "dataset", None)
            # try to read interaction strength and softening; fall back to defaults
            if hasattr(dataset, "simulation"):
                G = float(getattr(dataset.simulation, "interaction_strength", 1.0))
                soft = float(getattr(dataset.simulation, "softening", 0.0))
            else:
                G = float(getattr(dataset, "interaction_strength", 1.0))
                soft = float(getattr(dataset, "softening", 0.0))

            sim_loc = combined_locations[0]  # (batch, steps, n, d)
            sf_loc = combined_locations[1]
            # velocities come from inference output; if unavailable, approximate by finite differences
            try:
                # helper_scripts.infer_self_feed.run_inference returns combined_velocities; fetch if possible
                # but our signature captured only positions; so we fallback
                raise RuntimeError
            except Exception:
                # finite difference (dt=1 in arbitrary units)
                def finite_diff(x):
                    # x: (batch, steps, n, d) -> (batch, steps, n, d)
                    v = np.zeros_like(x)
                    v[:, 1:, :, :] = x[:, 1:, :, :] - x[:, :-1, :, :]
                    v[:, 0, :, :] = v[:, 1, :, :]
                    return v

                sim_vel = finite_diff(sim_loc)
                sf_vel = finite_diff(sf_loc)

            energies = {
                "simulation": self._compute_nbody_energies(sim_loc, sim_vel, G, soft),
                "self_feed": self._compute_nbody_energies(sf_loc, sf_vel, G, soft),
            }

            save_dir = f"{self.save_dir_path}/checkpoints/{self.step_count}"
            self.self_feed_postprocess_common(
                energies=energies,
                steps_survived=steps,
                save_dir=save_dir,
                rdf_metrics=None,
                vacf_metrics=None,
                metrics_filename="nbody_macro_metrics.json",
            )
        except Exception as _:
            # if anything fails, at least log steps survived
            wandb.log(
                {
                    "self_feed/steps_survived": steps,
                    "self_feed/step": self.step_count - 1,
                },
                commit=True,
            )

        return steps

    def train(self):
        # torch.autograd.set_detect_anomaly(True)
        self.save_training_args()
        self.save_model_params()
        self.save_dataset_attributes()
        run = self.create_wandb_run()

        # Log parameter counts to wandb
        if hasattr(self, 'total_params') and hasattr(self, 'trainable_params'):
            wandb.log({
                "model/total_parameters": self.total_params,
                "model/trainable_parameters": self.trainable_params,
            }, commit=False)

        start_time = time.time()
        train_steps = self.args.train_steps
        while train_steps is None or self.step_count < train_steps:
            try:
                self.train_one_epoch()
                # we increase it here to track finished epochs in case of exception in later steps
                self.step_count += 1

                if self.step_count % self.args.save_model_every == 0:
                    self.save_model()

                if self.step_count % self.args.test_macros_every == 0:
                    try:
                        self.run_self_feed()
                    except SelfFeedError as e:
                        wandb.log(
                            {
                                "self_feed/steps_survived": e.steps_survived,
                                "self_feed/step": self.step_count - 1,
                            },
                            commit=True,
                        )
                        print(f"Couldn't run self-feed. Reason: {e}")
                        print("Continuing training...")
                    except Exception as e:
                        import traceback

                        print(f"Couldn't run self-feed. Reason: {e}")
                        print("Full stacktrace:")
                        traceback.print_exc()
                        print("Continuing training...")

                if (
                    self.validation_dataloader is not None
                    and self.step_count % self.args.validation_frequency == 0
                ):
                    self.validate_one_epoch()

            except KeyboardInterrupt:
                print("Training interrupted. Saving model...")
                self.save_model(final=True)
                break

            except Exception as e:
                print(e)
                self.save_model(final=True)
                run.alert(
                    title="Training crashed",
                    text=f"Model type: {self.args.model_type} Exception: {e}",
                )
                raise (e)

        end_time = time.time()
        print(
            f"Training for {self.step_count} steps took {end_time - start_time:.2f} seconds"
        )
