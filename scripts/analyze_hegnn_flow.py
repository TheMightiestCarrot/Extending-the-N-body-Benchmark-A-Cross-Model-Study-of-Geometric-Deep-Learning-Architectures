#!/usr/bin/env python3
"""
Collect forward and backward stability diagnostics for HEGNN.

This script loads a trained checkpoint, replays a configurable number of
mini-batches with instrumentation, and saves:
  - activation statistics per component/layer (JSONL + summary JSON)
  - gradient statistics per parameter/module (JSONL + summary JSON)
  - sampled gate values for histogram plots
  - matplotlib figures visualizing activation RMS trends, gate histograms,
    and gradient norms
  - batch-level metrics (loss, RMSE, relative errors)

The intent is to spot exploding/vanishing patterns before the self-feed
rollout blows up.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.losses import TargetCommonLoss
from utils.get_device import get_device
from utils.utils_train import create_model


def _to_namespace(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


def load_run_args(run_dir: Path) -> SimpleNamespace:
    args_path = run_dir / "training_args.json"
    if not args_path.exists():
        raise FileNotFoundError(f"Missing training args at {args_path}")
    with open(args_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if "args" not in payload:
        raise ValueError(f"`training_args.json` missing 'args' key: {args_path}")
    args_ns = _to_namespace(payload["args"])
    return args_ns


def import_from_path(class_path: str):
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def instantiate_dataloader(args: SimpleNamespace):
    if not hasattr(args, "dataloader") or not hasattr(args.dataloader, "class_path"):
        raise ValueError("Args must contain dataloader.class_path")
    dl_class = import_from_path(args.dataloader.class_path)
    return dl_class(args)


class StatCollector:
    def __init__(
        self,
        sample_buckets: Optional[Dict[str, Dict[str, Any]]] = None,
        max_bucket_samples: int = 200_000,
        sample_stride: int = 2048,
    ):
        self.records: List[Dict[str, Any]] = []
        self.sample_buckets = sample_buckets or {}
        self.samples: Dict[str, List[float]] = {k: [] for k in self.sample_buckets}
        self.max_bucket_samples = max_bucket_samples
        self.sample_stride = sample_stride

    def add(
        self,
        *,
        component: str,
        tensor: torch.Tensor | None,
        batch_idx: int,
        kind: str,
        component_type: str,
        category: str,
        layer_idx: Optional[int] = None,
        notes: Optional[str] = None,
        sample_bucket: Optional[str] = None,
    ):
        if tensor is None:
            return
        flat = tensor.detach()
        if flat.numel() == 0:
            return
        flat = flat.to(dtype=torch.float32).reshape(-1)
        finite_mask = torch.isfinite(flat)
        finite = flat[finite_mask]
        finite_count = int(finite.numel())
        if finite_count == 0:
            record = {
                "component": component,
                "component_type": component_type,
                "category": category,
                "layer_idx": layer_idx,
                "batch_idx": batch_idx,
                "kind": kind,
                "finite_frac": 0.0,
                "numel": int(flat.numel()),
                "dtype": str(tensor.dtype),
                "shape": tuple(tensor.shape),
                "notes": "all_nan_or_inf",
            }
            self.records.append(record)
            return

        stats_tensor = finite.cpu()
        abs_vals = stats_tensor.abs()
        mean_val = float(stats_tensor.mean().item())
        rms_val = float(torch.sqrt((stats_tensor**2).mean()).item())
        std_val = (
            float(stats_tensor.std(unbiased=False).item())
            if stats_tensor.numel() > 1
            else 0.0
        )
        record = {
            "component": component,
            "component_type": component_type,
            "category": category,
            "layer_idx": layer_idx,
            "batch_idx": batch_idx,
            "kind": kind,
            "mean": mean_val,
            "std": std_val,
            "min": float(stats_tensor.min().item()),
            "max": float(stats_tensor.max().item()),
            "max_abs": float(abs_vals.max().item()),
            "rms": rms_val,
            "l2_norm": float(torch.linalg.vector_norm(stats_tensor).item()),
            "l1_mean": float(abs_vals.mean().item()),
            "numel": int(flat.numel()),
            "finite_frac": float(finite_count / flat.numel()),
            "pct_abs_gt_1": float((abs_vals > 1.0).float().mean().item()),
            "pct_abs_gt_5": float((abs_vals > 5.0).float().mean().item()),
            "pct_abs_gt_10": float((abs_vals > 10.0).float().mean().item()),
            "pct_abs_lt_1e_3": float((abs_vals < 1e-3).float().mean().item()),
            "dtype": str(tensor.dtype),
            "shape": tuple(tensor.shape),
            "q01": float(torch.quantile(stats_tensor, torch.tensor(0.01)).item()),
            "q50": float(torch.quantile(stats_tensor, torch.tensor(0.5)).item()),
            "q99": float(torch.quantile(stats_tensor, torch.tensor(0.99)).item()),
            "notes": notes,
        }
        self.records.append(record)

        if (
            sample_bucket
            and sample_bucket in self.samples
            and len(self.samples[sample_bucket]) < self.max_bucket_samples
        ):
            bucket_conf = self.sample_buckets[sample_bucket]
            collect_abs = bucket_conf.get("abs", False)
            values = abs_vals if collect_abs else stats_tensor
            values = values.cpu().numpy()
            if values.size > self.sample_stride:
                idx = np.random.choice(
                    values.size, size=self.sample_stride, replace=False
                )
                values = values[idx]
            self.samples[sample_bucket].extend(values.tolist())


def attach_gate_hooks(model) -> List[Any]:
    handles = []

    def _make_hook(module):
        def _hook(_module, _inputs, output):
            setattr(_module, "_last_output", output)
        return _hook

    layers = getattr(model, "layers", [])
    for layer in layers:
        handles.append(layer.mlp_pos_basis.register_forward_hook(_make_hook(layer.mlp_pos_basis)))
        handles.append(layer.mlp_vel_basis.register_forward_hook(_make_hook(layer.mlp_vel_basis)))
    return handles


def instrumented_forward(
    model,
    data,
    *,
    batch_idx: int,
    collector: StatCollector,
):
    node_feat = getattr(data, "node_feat", None)
    if node_feat is None:
        node_feat = getattr(data, "x", None)
    if node_feat is None:
        raise ValueError("Batch must contain node_feat or x.")
    collector.add(
        component="input.node_feat",
        tensor=node_feat,
        batch_idx=batch_idx,
        kind="activation",
        component_type="node_feat_in",
        category="input",
        layer_idx=None,
    )
    node_feat = model.embedding(node_feat)
    collector.add(
        component="embedding.out",
        tensor=node_feat,
        batch_idx=batch_idx,
        kind="activation",
        component_type="node_feat",
        category="embedding",
        layer_idx=None,
    )

    node_pos = data.pos
    node_vel = data.vel
    edge_index = data.edge_index
    rel_pos = getattr(data, "rel_pos", None)
    if rel_pos is None:
        row, col = edge_index
        rel_pos = node_pos[row] - node_pos[col]
    collector.add(
        component="input.rel_pos",
        tensor=rel_pos,
        batch_idx=batch_idx,
        kind="activation",
        component_type="rel_pos",
        category="input",
        layer_idx=None,
    )
    diff_vel = node_vel[edge_index[0]] - node_vel[edge_index[1]]
    collector.add(
        component="input.diff_vel",
        tensor=diff_vel,
        batch_idx=batch_idx,
        kind="activation",
        component_type="diff_vel",
        category="input",
        layer_idx=None,
    )

    edge_length = getattr(data, "edge_length", None)
    if edge_length is None:
        edge_length = torch.linalg.vector_norm(rel_pos, dim=-1)
    else:
        edge_length = edge_length.squeeze(-1)
    radial = getattr(data, "hegnn_radial", None)
    if radial is None:
        radial = model.radial_basis(edge_length)
    collector.add(
        component="radial.basis",
        tensor=radial,
        batch_idx=batch_idx,
        kind="activation",
        component_type="radial",
        category="radial",
        layer_idx=None,
    )

    node_sh = model.sh_init(node_feat, rel_pos, edge_index, radial)
    collector.add(
        component="sh.init",
        tensor=node_sh,
        batch_idx=batch_idx,
        kind="activation",
        component_type="node_sh",
        category="sh",
        layer_idx=-1,
    )

    delta_pos = torch.zeros_like(node_pos)
    delta_vel = torch.zeros_like(node_vel)

    for layer_idx, layer in enumerate(model.layers):
        setattr(layer.mlp_pos_basis, "_last_output", None)
        setattr(layer.mlp_vel_basis, "_last_output", None)
        msg, edge_vec_pos, edge_vec_vel, diff_sh = layer.Msg(
            edge_index=edge_index,
            node_feat=node_feat,
            node_sh=node_sh,
            diff_pos=rel_pos,
            diff_vel=diff_vel,
            radial=radial,
        )
        collector.add(
            component=f"L{layer_idx}.msg",
            tensor=msg,
            batch_idx=batch_idx,
            kind="activation",
            component_type="msg",
            category="message",
            layer_idx=layer_idx,
        )
        pos_gate_tensor = getattr(layer.mlp_pos_basis, "_last_output", None)
        collector.add(
            component=f"L{layer_idx}.pos_gate",
            tensor=pos_gate_tensor,
            batch_idx=batch_idx,
            kind="activation",
            component_type="pos_gate",
            category="gate",
            layer_idx=layer_idx,
            sample_bucket="pos_gate",
            notes="pos gate scalars",
        )
        setattr(layer.mlp_pos_basis, "_last_output", None)
        vel_gate_tensor = getattr(layer.mlp_vel_basis, "_last_output", None)
        collector.add(
            component=f"L{layer_idx}.vel_gate",
            tensor=vel_gate_tensor,
            batch_idx=batch_idx,
            kind="activation",
            component_type="vel_gate",
            category="gate",
            layer_idx=layer_idx,
            sample_bucket="vel_gate",
            notes="vel gate scalars",
        )
        setattr(layer.mlp_vel_basis, "_last_output", None)
        collector.add(
            component=f"L{layer_idx}.edge_vec_pos",
            tensor=edge_vec_pos,
            batch_idx=batch_idx,
            kind="activation",
            component_type="edge_vec_pos",
            category="edge",
            layer_idx=layer_idx,
        )
        collector.add(
            component=f"L{layer_idx}.edge_vec_vel",
            tensor=edge_vec_vel,
            batch_idx=batch_idx,
            kind="activation",
            component_type="edge_vec_vel",
            category="edge",
            layer_idx=layer_idx,
        )
        collector.add(
            component=f"L{layer_idx}.diff_sh",
            tensor=diff_sh,
            batch_idx=batch_idx,
            kind="activation",
            component_type="diff_sh",
            category="sh",
            layer_idx=layer_idx,
        )

        msg_agg, pos_agg, vel_agg, sh_agg = layer.Agg(
            edge_index=edge_index,
            dim_size=node_feat.size(0),
            msg=msg,
            edge_vec_pos=edge_vec_pos,
            edge_vec_vel=edge_vec_vel,
            diff_sh=diff_sh,
        )
        collector.add(
            component=f"L{layer_idx}.msg_agg",
            tensor=msg_agg,
            batch_idx=batch_idx,
            kind="activation",
            component_type="msg_agg",
            category="aggregation",
            layer_idx=layer_idx,
        )
        collector.add(
            component=f"L{layer_idx}.pos_agg",
            tensor=pos_agg,
            batch_idx=batch_idx,
            kind="activation",
            component_type="pos_agg",
            category="aggregation",
            layer_idx=layer_idx,
        )
        collector.add(
            component=f"L{layer_idx}.vel_agg",
            tensor=vel_agg,
            batch_idx=batch_idx,
            kind="activation",
            component_type="vel_agg",
            category="aggregation",
            layer_idx=layer_idx,
        )
        collector.add(
            component=f"L{layer_idx}.sh_agg",
            tensor=sh_agg,
            batch_idx=batch_idx,
            kind="activation",
            component_type="sh_agg",
            category="aggregation",
            layer_idx=layer_idx,
        )

        node_feat, node_sh = layer.Upd(
            node_feat=node_feat, node_sh=node_sh, msg_agg=msg_agg, sh_agg=sh_agg
        )
        collector.add(
            component=f"L{layer_idx}.node_feat",
            tensor=node_feat,
            batch_idx=batch_idx,
            kind="activation",
            component_type="node_feat",
            category="node",
            layer_idx=layer_idx,
        )
        collector.add(
            component=f"L{layer_idx}.node_sh",
            tensor=node_sh,
            batch_idx=batch_idx,
            kind="activation",
            component_type="node_sh",
            category="node",
            layer_idx=layer_idx,
        )

        delta_pos = delta_pos + pos_agg
        delta_vel = delta_vel + vel_agg
        collector.add(
            component=f"L{layer_idx}.delta_pos",
            tensor=delta_pos,
            batch_idx=batch_idx,
            kind="activation",
            component_type="delta_pos",
            category="delta",
            layer_idx=layer_idx,
        )
        collector.add(
            component=f"L{layer_idx}.delta_vel",
            tensor=delta_vel,
            batch_idx=batch_idx,
            kind="activation",
            component_type="delta_vel",
            category="delta",
            layer_idx=layer_idx,
        )

    pos_input = torch.cat([node_feat, delta_pos], dim=-1)
    vel_input = torch.cat([node_feat, delta_vel, node_vel], dim=-1)
    collector.add(
        component="pos_head.input",
        tensor=pos_input,
        batch_idx=batch_idx,
        kind="activation",
        component_type="pos_head_in",
        category="head",
        layer_idx=None,
    )
    collector.add(
        component="vel_head.input",
        tensor=vel_input,
        batch_idx=batch_idx,
        kind="activation",
        component_type="vel_head_in",
        category="head",
        layer_idx=None,
    )

    pos_dt = model.pos_head(pos_input)
    vel_pred = model.vel_head(vel_input)
    collector.add(
        component="pos_head.output",
        tensor=pos_dt,
        batch_idx=batch_idx,
        kind="activation",
        component_type="pos_head_out",
        category="head",
        layer_idx=None,
    )
    collector.add(
        component="vel_head.output",
        tensor=vel_pred,
        batch_idx=batch_idx,
        kind="activation",
        component_type="vel_head_out",
        category="head",
        layer_idx=None,
    )

    target_map = {
        "pos_dt": pos_dt,
        "pos": node_pos + pos_dt,
        "vel": vel_pred,
        "vel_dt": vel_pred - node_vel,
    }

    outputs = []
    for target in model.targets:
        if target not in target_map:
            raise NotImplementedError(f"Unsupported target '{target}'")
        outputs.append(target_map[target])

    prediction = torch.cat(outputs, dim=-1)
    collector.add(
        component="model.output",
        tensor=prediction,
        batch_idx=batch_idx,
        kind="activation",
        component_type="output",
        category="output",
        layer_idx=None,
    )
    return prediction


def collect_gradient_stats(model, batch_idx: int) -> List[Dict[str, Any]]:
    records = []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach().to(dtype=torch.float32).cpu()
        grad_abs = grad.abs()
        param_data = param.detach().to(dtype=torch.float32).cpu()
        grad_norm = float(torch.linalg.vector_norm(grad).item())
        param_norm = float(torch.linalg.vector_norm(param_data).item())
        module_name, layer_idx = normalize_param_name(name)
        record = {
            "name": name,
            "module": module_name,
            "layer_idx": layer_idx,
            "batch_idx": batch_idx,
            "grad_mean": float(grad.mean().item()),
            "grad_std": float(grad.std(unbiased=False).item())
            if grad.numel() > 1
            else 0.0,
            "grad_max_abs": float(grad_abs.max().item()),
            "grad_l2": grad_norm,
            "grad_l1_mean": float(grad_abs.mean().item()),
            "param_l2": param_norm,
            "grad_param_ratio": float(
                grad_norm / (param_norm + 1e-12)
            ),
            "numel": int(param.numel()),
        }
        records.append(record)
    return records


def normalize_param_name(name: str) -> Tuple[str, Optional[int]]:
    parts = name.split(".")
    if not parts:
        return name, None
    if parts[0] == "layers" and len(parts) >= 3:
        layer_idx = int(parts[1])
        module = parts[2]
        return f"L{layer_idx}.{module}", layer_idx
    return parts[0], None


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_jsonl(path: Path, records: Iterable[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec))
            f.write("\n")


def aggregate_records(records: List[Dict[str, Any]], key_fields: Tuple[str, ...]):
    grouped = defaultdict(lambda: defaultdict(list))
    for rec in records:
        key = tuple(rec.get(field) for field in key_fields)
        bucket = grouped[key]
        for metric in (
            "mean",
            "std",
            "max_abs",
            "rms",
            "pct_abs_gt_1",
            "pct_abs_gt_5",
            "pct_abs_gt_10",
            "pct_abs_lt_1e_3",
        ):
            if metric in rec:
                bucket[metric].append(rec[metric])
        bucket["layer_idx"] = rec.get("layer_idx")
        bucket["category"] = rec.get("category")
        bucket["component_type"] = rec.get("component_type")
        bucket["component"] = rec.get("component")
    summary = []
    for key, bucket in grouped.items():
        entry = {
            "key": key,
            "component": bucket.get("component"),
            "component_type": bucket.get("component_type"),
            "category": bucket.get("category"),
            "layer_idx": bucket.get("layer_idx"),
        }
        for metric, values in bucket.items():
            if isinstance(values, list):
                entry[f"mean_{metric}"] = float(mean(values))
        summary.append(entry)
    return summary


def compute_batch_metrics(pred, data) -> Dict[str, float]:
    target = data.y
    num_targets = pred.shape[-1] // 3
    metrics = {}
    for idx in range(num_targets):
        pred_slice = pred[..., 3 * idx : 3 * (idx + 1)]
        target_slice = target[..., 3 * idx : 3 * (idx + 1)]
        err = pred_slice - target_slice
        rms = torch.sqrt(torch.mean(err**2))
        mae = torch.mean(err.abs())
        ref = torch.sqrt(torch.mean(target_slice**2))
        rel = (rms / (ref + 1e-12)).item()
        label = f"target_{idx}"
        metrics[f"{label}_rmse"] = float(rms.item())
        metrics[f"{label}_mae"] = float(mae.item())
        metrics[f"{label}_rel_rmse"] = float(rel)
        metrics[f"{label}_max_abs"] = float(err.abs().max().item())
    return metrics


def plot_activation_rms(summary, out_path: Path, num_layers: int):
    curves = defaultdict(lambda: [None] * num_layers)
    for rec in summary:
        layer_idx = rec.get("layer_idx")
        if layer_idx is None or layer_idx < 0:
            continue
        comp_type = rec.get("component_type")
        if comp_type not in {
            "msg",
            "msg_agg",
            "pos_gate",
            "vel_gate",
            "node_feat",
            "node_sh",
            "pos_agg",
            "vel_agg",
            "delta_pos",
            "delta_vel",
        }:
            continue
        curves[comp_type][layer_idx] = rec.get("mean_rms")

    plt.figure(figsize=(10, 6))
    for comp_type, values in curves.items():
        xs = list(range(num_layers))
        ys = [v if v is not None else np.nan for v in values]
        plt.plot(xs, ys, marker="o", label=comp_type)
    plt.xlabel("Layer index")
    plt.ylabel("Mean RMS")
    plt.title("Per-layer activation RMS")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_gate_histograms(samples: Dict[str, List[float]], out_path: Path):
    plt.figure(figsize=(10, 4))
    for idx, (bucket, values) in enumerate(samples.items(), start=1):
        plt.subplot(1, len(samples), idx)
        data = np.array(values)
        if data.size == 0:
            plt.title(f"{bucket} (no data)")
            continue
        plt.hist(data, bins=80, log=True, color="#1f77b4")
        plt.title(f"{bucket} gate\nμ={data.mean():.2f}, σ={data.std():.2f}")
        plt.xlabel("|value|" if bucket.endswith("_abs") else "value")
        plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_gradient_norms(grad_summary, out_path: Path, top_k: int = 15):
    sortable = []
    for rec in grad_summary:
        module = rec["module"]
        grad_l2 = rec.get("mean_grad_l2")
        ratio = rec.get("mean_grad_param_ratio")
        if grad_l2 is None or ratio is None:
            continue
        sortable.append((module, grad_l2, ratio))
    sortable.sort(key=lambda x: x[1], reverse=True)
    top = sortable[:top_k]
    if not top:
        return
    modules = [t[0] for t in top]
    grad_vals = [t[1] for t in top]
    ratios = [t[2] for t in top]

    fig, ax1 = plt.subplots(figsize=(11, 4))
    ax2 = ax1.twinx()
    indices = np.arange(len(top))
    ax1.bar(indices - 0.2, grad_vals, width=0.4, label="grad L2", color="#ff7f0e")
    ax2.bar(indices + 0.2, ratios, width=0.4, label="grad/param", color="#2ca02c")
    ax1.set_ylabel("Gradient L2 norm")
    ax2.set_ylabel("grad_norm / param_norm")
    ax1.set_xticks(indices)
    ax1.set_xticklabels(modules, rotation=45, ha="right")
    ax1.set_title("Top gradient norms by module")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="HEGNN flow diagnostics")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/hegnn"),
        help="Run directory containing checkpoint + training_args.json",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        required=True,
        help="Run subfolder name, e.g. 2025-11-18_18-39-38",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=4,
        help="Number of batches to analyze",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="model.pth",
        help="Checkpoint filename inside the run directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/hegnn_flow"),
        help="Directory to store stats and plots",
    )
    args = parser.parse_args()

    run_subdir = args.run_dir / args.timestamp
    if not run_subdir.exists():
        raise FileNotFoundError(f"Run subdir {run_subdir} does not exist")
    run_args = load_run_args(run_subdir)
    checkpoint_path = run_subdir / args.checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    run_args.model_path = str(checkpoint_path)

    ensure_dir(args.output_dir / args.timestamp)
    fig_dir = Path("figures") / "hegnn_flow" / args.timestamp
    ensure_dir(fig_dir)

    device = get_device(getattr(run_args, "gpu_id", "auto"))
    dataloader = instantiate_dataloader(run_args)
    model = create_model(run_args, dataloader).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    loss_fn = TargetCommonLoss(args=run_args)
    collector = StatCollector(
        sample_buckets={
            "pos_gate": {"abs": True},
            "vel_gate": {"abs": True},
        }
    )
    grad_records: List[Dict[str, Any]] = []
    batch_metrics: List[Dict[str, Any]] = []

    gate_hooks = attach_gate_hooks(model)
    try:
        for batch_idx in range(args.num_batches):
            model.zero_grad(set_to_none=True)
            batch_list, _ = dataloader.get_batch()
            graph = dataloader.preprocess_batch(
                batch_list[0], device=dataloader.device, training=False
            )
            prediction = instrumented_forward(
                model,
                graph,
                batch_idx=batch_idx,
                collector=collector,
            )
            loss = loss_fn(prediction, graph)
            loss.backward()
            grad_records.extend(collect_gradient_stats(model, batch_idx))
            batch_metric = compute_batch_metrics(prediction, graph)
            batch_metric["loss"] = float(loss.item())
            batch_metric["batch_idx"] = batch_idx
            batch_metrics.append(batch_metric)
    finally:
        for handle in gate_hooks:
            handle.remove()

    activation_records = collector.records
    stats_dir = args.output_dir / args.timestamp
    ensure_dir(stats_dir)

    activation_path = stats_dir / "activation_stats.jsonl"
    gradient_path = stats_dir / "gradient_stats.jsonl"
    batch_metric_path = stats_dir / "batch_metrics.json"

    save_jsonl(activation_path, activation_records)
    save_jsonl(gradient_path, grad_records)
    with open(batch_metric_path, "w", encoding="utf-8") as f:
        json.dump(batch_metrics, f, indent=2)

    activation_summary = aggregate_records(activation_records, ("component",))
    grad_summary = aggregate_gradients(grad_records)

    activation_summary_path = stats_dir / "activation_summary.json"
    grad_summary_path = stats_dir / "gradient_summary.json"
    with open(activation_summary_path, "w", encoding="utf-8") as f:
        json.dump(activation_summary, f, indent=2)
    with open(grad_summary_path, "w", encoding="utf-8") as f:
        json.dump(grad_summary, f, indent=2)

    plot_activation_rms(
        activation_summary,
        fig_dir / "activation_rms.png",
        num_layers=getattr(run_args, "num_layers", len(model.layers)),
    )
    plot_gate_histograms(collector.samples, fig_dir / "gate_hist.png")
    plot_gradient_norms(grad_summary, fig_dir / "gradient_norms.png")

    top_activation = sorted(
        activation_summary, key=lambda r: r.get("mean_rms", 0.0), reverse=True
    )[:10]
    top_activation_abs = sorted(
        activation_summary, key=lambda r: r.get("mean_pct_abs_gt_5", 0.0), reverse=True
    )[:10]
    top_grad = sorted(
        grad_summary, key=lambda r: r.get("mean_grad_l2", 0.0), reverse=True
    )[:10]
    summary_payload = {
        "activation_top_rms": top_activation,
        "activation_high_abs": top_activation_abs,
        "gradient_top_norms": top_grad,
        "batch_metrics": batch_metrics,
        "figures": {
            "activation_rms": str((fig_dir / "activation_rms.png").as_posix()),
            "gate_hist": str((fig_dir / "gate_hist.png").as_posix()),
            "gradient_norms": str((fig_dir / "gradient_norms.png").as_posix()),
        },
    }
    summary_path = stats_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print(f"[hegnn_flow] activation stats -> {activation_path}")
    print(f"[hegnn_flow] gradient stats -> {gradient_path}")
    print(f"[hegnn_flow] batch metrics -> {batch_metric_path}")
    print(f"[hegnn_flow] activation summary -> {activation_summary_path}")
    print(f"[hegnn_flow] gradient summary -> {grad_summary_path}")
    print(f"[hegnn_flow] summary -> {summary_path}")
    print(f"[hegnn_flow] figures written to {fig_dir}")


def aggregate_gradients(records: List[Dict[str, Any]]):
    grouped = defaultdict(lambda: defaultdict(list))
    layer_lookup = {}
    for rec in records:
        module = rec["module"]
        grouped[module]["grad_l2"].append(rec["grad_l2"])
        grouped[module]["grad_param_ratio"].append(rec["grad_param_ratio"])
        grouped[module]["grad_max_abs"].append(rec["grad_max_abs"])
        layer_lookup[module] = rec.get("layer_idx")
    summary = []
    for module, metrics in grouped.items():
        summary.append(
            {
                "module": module,
                "layer_idx": layer_lookup.get(module),
                "mean_grad_l2": float(mean(metrics["grad_l2"])),
                "mean_grad_param_ratio": float(mean(metrics["grad_param_ratio"])),
                "mean_grad_max_abs": float(mean(metrics["grad_max_abs"])),
            }
        )
    return summary


if __name__ == "__main__":
    main()
