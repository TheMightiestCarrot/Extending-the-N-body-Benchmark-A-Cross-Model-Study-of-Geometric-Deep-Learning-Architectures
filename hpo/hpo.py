import argparse
import json
import os
import time
import socket
import uuid
from dataclasses import dataclass
from typing import Dict, Any
import platform

import optuna
import torch
from tqdm import tqdm

from utils.config import parse_args as parse_repo_args, namespace_to_dict
from utils.utils_train import create_model, load_class_from_args
from utils.hpo_metrics import load_macro_pvalues_from_checkpoint


TARGET_PARAMS = {"param_small": 1_800_000, "param_medium": 10_000_000}
PARAM_TOLERANCE = 0.07  # ±7%


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def _write_json_atomic(path: str, payload: Any):
    """Write JSON atomically to avoid corruption on crashes."""
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        # best-effort cleanup
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def _system_metadata() -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    try:
        meta["hostname"] = socket.gethostname()
    except Exception:
        meta["hostname"] = "unknown"
    meta["pid"] = os.getpid()
    meta["platform"] = platform.platform()
    meta["python"] = platform.python_version()
    try:
        import torch as _torch

        meta["torch_version"] = getattr(_torch, "__version__", None)
        meta["cuda_available"] = bool(getattr(_torch.cuda, "is_available", lambda: False)())
        if meta["cuda_available"]:
            try:
                meta["cuda_version"] = getattr(_torch.version, "cuda", None)
            except Exception:
                meta["cuda_version"] = None
            try:
                meta["cudnn_version"] = getattr(_torch.backends.cudnn, "version", lambda: None)()
            except Exception:
                meta["cudnn_version"] = None
            try:
                device_count = _torch.cuda.device_count()
                meta["num_gpus"] = int(device_count)
                meta["gpus"] = [
                    {
                        "index": i,
                        "name": _torch.cuda.get_device_name(i),
                        "capability": _torch.cuda.get_device_capability(i),
                    }
                    for i in range(device_count)
                ]
            except Exception:
                pass
    except Exception:
        pass
    return meta


def set_trial_overrides(args, trial):
    # shared
    # base lr tuned around repo default (0.5) given inverse-sqrt schedule scaling
    args.learning_rate = trial.suggest_float("lr", 0.05, 2.0, log=True)
    args.learning_rate_factor = 1.0
    args.learning_rate_warmup_steps = 2048
    # regularization
    # weight decay is fixed in Trainer; could be added if exposed
    # dropout handled per-model where applicable

    # fixed checkpoint interval for macros; env overrides; no extra logic
    args.test_macros_every = int(os.getenv("HPO_TEST_MACROS_EVERY", "5"))
    # cap rollout length during HPO for speed (esp. cgenn); opt-out by unsetting env
    try:
        args.self_feed_limit_steps = int(os.getenv("HPO_SELF_FEED_LIMIT_STEPS", "20"))
    except Exception:
        args.self_feed_limit_steps = None

    # model-specific knobs (mapped to current repo args)
    if args.model_type == "ponita":
        args.hidden_features = trial.suggest_categorical(
            "hidden_features", [112, 128, 160, 192]
        )
        args.num_layers = trial.suggest_categorical("num_layers", [5, 6, 8, 10])
    elif args.model_type == "segnn":
        args.hidden_features = trial.suggest_categorical(
            "hidden_features", [48, 64, 96, 128]
        )
        args.num_layers = trial.suggest_categorical("num_layers", [5, 6, 8, 10])
        args.lmax_h = trial.suggest_categorical("lmax_h", [1, 2])
    elif args.model_type == "equiformer_v2":
        args.num_layers = trial.suggest_categorical("num_layers", [6, 8, 10])
        args.num_heads = trial.suggest_categorical("num_heads", [4, 8])
        base = trial.suggest_categorical("channel_base", [112, 128, 160, 192])
        args.attn_hidden_channels = base
        args.sphere_channels = base
        args.ffn_hidden_channels = base
        # lmax_list kept as provided in config.yaml; can be extended if needed
    elif args.model_type == "cgenn":
        args.hidden_features = trial.suggest_categorical(
            "hidden_features", [160, 192, 224, 256]
        )
        args.num_layers = trial.suggest_categorical("num_layers", [5, 6, 8, 10])
    elif args.model_type == "graph_transformer":  # vanilla transformer baseline
        args.hidden_features = trial.suggest_categorical(
            "hidden_features", [176, 192, 224, 256]
        )
        args.num_layers = trial.suggest_categorical("num_layers", [6, 8, 10])
        args.graph_transformer_num_heads = trial.suggest_categorical(
            "graph_transformer_num_heads", [4, 8]
        )

    # model-specific trainer pacing tweaks (primarily for cgenn throughput)
    original_steps_per_epoch = getattr(args, "steps_per_epoch", -1)
    if args.model_type == "cgenn":
        target_steps = int(os.getenv("HPO_CGENN_STEPS_PER_EPOCH", "10"))
        if target_steps > 0:
            try:
                args.steps_per_epoch = target_steps
            except Exception:
                pass
            if original_steps_per_epoch not in (-1, None, 0):
                scale = max(1, int(round(original_steps_per_epoch / float(target_steps))))
                try:
                    current = getattr(args, "test_macros_every", None)
                    if current is not None:
                        setattr(args, "test_macros_every", max(1, int(current * scale)))
                except Exception:
                    pass
                try:
                    current = getattr(args, "save_model_every", None)
                    if current is not None:
                        setattr(args, "save_model_every", max(1, int(current * scale)))
                except Exception:
                    pass
                try:
                    current = getattr(args, "validation_frequency", None)
                    if current is not None:
                        setattr(args, "validation_frequency", max(1, int(current * scale)))
                except Exception:
                    pass

    return args


def _initialize_lazy_and_count(model, train_loader, args) -> int:
    try:
        return count_params(model)
    except ValueError:
        # initialize lazy parameters with a real preprocessed batch
        batch_list, _ = train_loader.get_batch()
        dummy_batch = train_loader.preprocess_batch(batch_list[0], "cpu")
        model = model.to("cpu")
        if getattr(args, "precision_mode", "single") == "double":
            model = model.double()
        else:
            model = model.float()
        with torch.no_grad():
            _ = model(dummy_batch)
        return count_params(model)


def _quantize_width_for_model(args, key: str):
    base = 16
    width = int(getattr(args, key))
    # enforce head divisibility for attention models
    if args.model_type == "equiformer_v2":
        heads = getattr(args, "num_heads", 1) or 1
        if width % heads != 0:
            width = ((width + heads - 1) // heads) * heads
    if args.model_type == "graph_transformer":
        heads = getattr(args, "graph_transformer_num_heads", 1) or 1
        if width % heads != 0:
            width = ((width + heads - 1) // heads) * heads
    # snap to tensor-core friendly multiple
    width = max(base, ((width + base // 2) // base) * base)
    setattr(args, key, width)
    if args.model_type == "equiformer_v2":
        for attr in ("sphere_channels", "ffn_hidden_channels", "channel_base"):
            try:
                setattr(args, attr, width)
            except Exception:
                pass
    return width


def adjust_width_to_target(args, target: int, train_loader) -> int:
    # crude loop adjusting primary width-like knob to hit param target within tolerance
    lo, hi = 16, 1536
    key = "hidden_features"
    if args.model_type == "equiformer_v2":
        key = "attn_hidden_channels"

    for _ in range(8):
        model = create_model(args, train_loader)
        params = _initialize_lazy_and_count(model, train_loader, args)
        if abs(params - target) / target <= PARAM_TOLERANCE:
            return params
        if params > target:
            hi = getattr(args, key)
            setattr(args, key, max(lo, (lo + getattr(args, key)) // 2))
        else:
            lo = getattr(args, key)
            setattr(args, key, min(hi, (hi + getattr(args, key)) // 2))
        _quantize_width_for_model(args, key)
    model = create_model(args, train_loader)
    return _initialize_lazy_and_count(model, train_loader, args)


def run_short_training_and_score(
    args,
    trial_minutes: int,
    trial_log_dir: str = None,
    trial_number: int = None,
    mode_str: str = None,
    model_type: str = None,
    param_count: "int | None" = None,
) -> float:
    # build loaders
    dataloader_cls = load_class_from_args(args, "dataloader")
    train_loader = (
        dataloader_cls(args, partition="train")
        if args.do_validation
        else dataloader_cls(args)
    )
    valid_loader = (
        dataloader_cls(args, partition="valid") if args.do_validation else None
    )

    # build model/trainer
    model = create_model(args, train_loader)
    trainer_cls = load_class_from_args(args, "trainer")
    trainer = trainer_cls(
        model, train_loader, validation_dataloader=valid_loader, args=args
    )

    # require wandb api key and force online logging
    if not os.getenv("WANDB_API_KEY"):
        raise RuntimeError(
            "WANDB_API_KEY must be set for HPO runs. Please export WANDB_API_KEY and rerun."
        )
    os.environ["WANDB_MODE"] = "online"
    os.environ.setdefault("WANDB_SILENT", "true")
    # group hpo runs
    os.environ.setdefault("WANDB_PROJECT", "hpo")
    os.environ.setdefault(
        "WANDB_RUN_GROUP", f"hpo/{args.model_type}/{os.getenv('USER', 'user')}"
    )
    try:
        _ = trainer.create_wandb_run()
    except Exception as e:
        print(e)
        raise

    # also log exact config for this trial
    try:
        cfg_dict = namespace_to_dict(args)
        from wandb import sdk as _sdk

        if _sdk.wandb_run is not None:
            import wandb

            wandb.log({"hpo/config": cfg_dict}, commit=False)
    except Exception as e:
        print(e)

    # ensure run dir has dataset metadata before self-feed inference tries to load it
    try:
        trainer.save_training_args()
    except Exception as e:
        print(e)
    try:
        trainer.save_model_params()
    except Exception as e:
        print(e)
    try:
        trainer.save_dataset_attributes()
    except Exception as e:
        print(e)

    # limit training by wall-clock and by updates
    t0 = time.time()
    max_minutes = float(trial_minutes)
    max_updates = max(2 * args.test_macros_every, 6000)

    # force save interval so we get checkpoints to score
    save_every = args.test_macros_every

    # do training until time or updates reached; rely on trainer.train loop per-step saving
    trainer.args.train_steps = None  # open-ended; manual stop
    while True:
        trainer.train_one_epoch()
        trainer.step_count += 1
        if trainer.step_count % save_every == 0:
            try:
                trainer.run_self_feed()
            except Exception as e:
                print(e)
                pass
        if trainer.step_count >= max_updates:
            break
        if (time.time() - t0) / 60.0 >= max_minutes:
            break

    # find latest checkpoint directory and score
    debug = os.getenv("HPO_DEBUG", "0") not in ("0", "false", "False", None)
    ckpt_root = os.path.join(trainer.save_dir_path, "checkpoints")
    steps = (
        [d for d in os.listdir(ckpt_root) if d.isdigit()]
        if os.path.exists(ckpt_root)
        else []
    )
    if debug:
        print(f"[hpo] ckpt_root={ckpt_root}")
        print(f"[hpo] step_dirs_found={steps}")
    if not steps:
        # ensure at least one self-feed to materialize macros for scoring
        try:
            trainer.run_self_feed()
        except Exception as e:
            print(e)
            pass
        steps = (
            [d for d in os.listdir(ckpt_root) if d.isdigit()]
            if os.path.exists(ckpt_root)
            else []
        )
        if debug:
            print(f"[hpo] after self_feed, step_dirs_found={steps}")
        if not steps:
            return -690.0
    # evaluate several recent checkpoints to reduce variance
    mode = os.getenv("HPO_EVAL_MODE", "best_last_k")  # best_last_k | mean_last_k | median_last_k
    try:
        last_k = max(1, int(os.getenv("HPO_EVAL_LAST_K", "3")))
    except Exception as e:
        print(e)
        last_k = 3
    steps_sorted = sorted(steps, key=lambda s: int(s))
    cand_steps = steps_sorted[-last_k:] if last_k > 0 else steps_sorted

    # if requested, force a final long rollout at the latest checkpoint before scoring
    try:
        force_long = os.getenv("HPO_SELF_FEED_LONG_FINAL", "0").lower() in ("1", "true", "yes")
    except Exception:
        force_long = False
    if force_long and steps_sorted:
        latest = steps_sorted[-1]
        try:
            # temporarily override rollout cap for one long run
            old_cap = getattr(args, "self_feed_limit_steps", None)
            setattr(args, "self_feed_limit_steps", None)
            # re-run self-feed at latest step to materialize long macros
            # note: this uses current trainer state; acceptable for HPO diagnostics
            try:
                trainer.run_self_feed()
            except Exception as _e:
                print(_e)
            # restore cap
            setattr(args, "self_feed_limit_steps", old_cap)
        except Exception as _e:
            print(_e)

    scores = []
    for s in cand_steps:
        d = os.path.join(ckpt_root, s)
        if debug:
            try:
                files = sorted(os.listdir(d))
                print(f"[hpo] checking {d}, files={files}")
            except Exception as e:
                print(f"[hpo] cannot list {d}: {e}")
        _, comb = load_macro_pvalues_from_checkpoint(d)
        if comb == comb and comb > 0.0:
            scores.append(float(comb))

    if not scores:
        return -690.0

    import numpy as _np
    if mode == "mean_last_k":
        combined = float(_np.mean(scores))
    elif mode == "median_last_k":
        combined = float(_np.median(scores))
    else:
        combined = float(max(scores))
    if debug:
        print(f"[hpo] eval_mode={mode}, last_k={last_k}, scores={scores}, selected={combined}")

    # objective: log combined p-value for smoother optimization (avoid tiny positives)
    if not (combined == combined) or combined <= 0.0:
        return -690.0  # ~log(1e-300)
    import math

    final_logp = math.log(combined)

    # optional per-trial logging for crash resilience and post-hoc analyses
    if trial_log_dir is not None:
        try:
            os.makedirs(trial_log_dir, exist_ok=True)
            # config snapshot
            try:
                cfg_dict = namespace_to_dict(args)
                _write_json_atomic(os.path.join(trial_log_dir, "config.json"), cfg_dict)
            except Exception as _e:
                print(_e)
            # meta snapshot
            # vram info (best-effort)
            peak_vram_mb = None
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    peak_vram_mb = float(torch.cuda.max_memory_allocated() / 1e6)
            except Exception:
                pass

            meta = {
                "trial_number": trial_number,
                "mode": mode_str,
                "model_type": getattr(args, "model_type", model_type),
                "param_count": int(param_count) if param_count is not None else None,
                "minutes_run": float((time.time() - t0) / 60.0),
                "steps_run": int(getattr(trainer, "step_count", 0)),
                "steps_per_min": (
                    float(getattr(trainer, "step_count", 0)) / max(1e-6, float((time.time() - t0) / 60.0))
                ),
                "peak_vram_mb": peak_vram_mb,
                "save_dir_path": getattr(trainer, "save_dir_path", None),
                "ckpt_root": ckpt_root,
                "cand_steps": [int(x) for x in cand_steps],
                "scores_last_k": scores,
                "selected_score_p": combined,
                "selected_score_logp": final_logp,
                "eval_mode": mode,
                "last_k": last_k,
            }
            _write_json_atomic(os.path.join(trial_log_dir, "meta.json"), meta)
        except Exception as _e:
            print(_e)

    return final_logp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument(
        "--mode",
        type=str,
        choices=["param_small", "param_medium", "time_matched"],
        required=True,
    )
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--trial_minutes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument(
        "--models",
        type=str,
        default=None,
        help="comma-separated model types to sweep (e.g., 'ponita,segnn,equiformer_v2,cgenn,graph_transformer'). Defaults to model in config.",
    )
    ap.add_argument(
        "--dataloaders",
        type=str,
        default=None,
        help="comma-separated dataloader types aligned with --models (defaults to {model}_nbody).",
    )
    args_cli = ap.parse_args()

    # parse repo args to get typed namespace with defaults
    # temporarily sanitize sys.argv so repo parser doesn't see our hpo flags
    import sys

    saved_argv = sys.argv[:]
    try:
        repo_argv = [
            "train.py",
            "--config",
            args_cli.config,
            "--trainer_type",
            "trainer_nbody",
        ]
        sys.argv = repo_argv
        repo_args, _ = parse_repo_args()
    finally:
        sys.argv = saved_argv

    # determine models/dataloaders to sweep
    model_to_loader_auto = {
        "ponita": "ponita_nbody",
        "segnn": "segnn_nbody",
        "equiformer_v2": "equiformer_v2_nbody",
        "cgenn": "cgenn_nbody",
        "graph_transformer": "graph_transformer_nbody",
    }

    if args_cli.models is None:
        models_list = [repo_args.model_type]
    else:
        models_list = [m.strip() for m in args_cli.models.split(",") if m.strip()]

    if args_cli.dataloaders is None:
        dataloaders_list = [model_to_loader_auto.get(m, f"{m}_nbody") for m in models_list]
    else:
        dataloaders_list = [d.strip() for d in args_cli.dataloaders.split(",") if d.strip()]
        if len(dataloaders_list) != len(models_list):
            raise ValueError("--dataloaders must have same length as --models")

    base_out_dir = os.getenv("HPO_OUTDIR_ROOT", os.path.join("runs", "hpo_results"))
    run_tag = os.getenv("HPO_RUN_TAG")
    if not run_tag:
        try:
            host = socket.gethostname().split(".")[0]
        except Exception:
            host = "host"
        ts_run = time.strftime("%Y%m%d-%H%M%S")
        pid = os.getpid()
        short = uuid.uuid4().hex[:6]
        run_tag = f"{ts_run}_{host}_{pid}_{short}"
    out_dir = os.path.join(base_out_dir, run_tag)
    os.makedirs(out_dir, exist_ok=True)

    # best-effort convenience symlink to the latest run dir
    try:
        latest = os.path.join(base_out_dir, "latest")
        if os.path.islink(latest) or os.path.exists(latest):
            try:
                os.unlink(latest)
            except Exception:
                pass
        target_abs = os.path.abspath(out_dir)
        os.symlink(target_abs, latest)
    except Exception:
        pass

    # collect per-model results for final summary
    aggregated_results = []
    # write run-level system metadata for crash resilience
    try:
        _write_json_atomic(os.path.join(out_dir, "system.json"), _system_metadata())
    except Exception:
        pass

    # sweep loop per model
    for model_type, dataloader_type in zip(models_list, dataloaders_list):
        # rebuild repo args for this model/dataloader
        saved_argv = sys.argv[:]
        try:
            repo_argv = [
                "train.py",
                "--config",
                args_cli.config,
                "--trainer_type",
                "trainer_nbody",
                "--model_type",
                model_type,
                "--dataloader_type",
                dataloader_type,
            ]
            sys.argv = repo_argv
            repo_args_m, _ = parse_repo_args()
        finally:
            sys.argv = saved_argv

        def objective(trial: optuna.Trial):
            # copy base args to avoid mutation across trials
            local = type(repo_args_m)()
            local.__dict__.update(repo_args_m.__dict__)

            # ensure trainer/dataloader validation flags reasonable during trials
            setattr(local, "do_validation", False)
            local = set_trial_overrides(local, trial, args_cli.mode)

            # visibility
            try:
                print(f"[hpo] starting trial {trial.number + 1}/{args_cli.trials} for {local.model_type} | mode={args_cli.mode}")
            except Exception as e:
                print(e)

            # build a dataloader for width adjustment (if any) and parameter counting
            dl_cls = load_class_from_args(local, "dataloader")
            tmp_loader = dl_cls(local, partition="train") if local.do_validation else dl_cls(local)

            if args_cli.mode.startswith("param_"):
                target = TARGET_PARAMS[args_cli.mode]
                _ = adjust_width_to_target(local, target, tmp_loader)

            # always compute and record parameter count
            try:
                model_for_count = create_model(local, tmp_loader)
                param_count = _initialize_lazy_and_count(model_for_count, tmp_loader, local)
                trial.set_user_attr("param_count", int(param_count))
                try:
                    print(f"[hpo] param_count={int(param_count)}")
                except Exception as e:
                    print(e)
            except Exception as e:
                print(e)

            # log config
            try:
                cfg = namespace_to_dict(local)
                print(f"[hpo] trial_cfg={json.dumps(cfg, default=str)[:2000]}...")
            except Exception as e:
                print(e)

            # per-trial log dir
            trial_log_dir = os.path.join(out_dir, "trials", model_type, f"{trial.number:04d}")

            score = run_short_training_and_score(
                local,
                args_cli.trial_minutes,
                trial_log_dir=trial_log_dir,
                trial_number=trial.number,
                mode_str=args_cli.mode,
                model_type=model_type,
                param_count=param_count if 'param_count' in locals() else None,
            )
            trial.set_user_attr("score_log_p", float(score))
            try:
                trial.set_user_attr("trial_log_dir", trial_log_dir)
            except Exception:
                pass
            # persist lightweight trial summary for resilience
            try:
                trial_summary = {
                    "trial": trial.number,
                    "value_logp": float(score),
                    "params": dict(trial.params),
                    "user_attrs": dict(getattr(trial, "user_attrs", {})),
                    "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
                }
                _write_json_atomic(os.path.join(trial_log_dir, "trial_summary.json"), trial_summary)
            except Exception as _e:
                print(_e)
            return score

        # persistent storage so HPO is resumable after crashes
        storage_url = os.getenv(
            "HPO_STORAGE",
            f"sqlite:///{os.path.abspath(os.path.join(out_dir, 'study.db'))}"
        )
        study = optuna.create_study(
            direction="maximize",
            study_name=f"{model_type}-{args_cli.mode}",
            storage=storage_url,
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=args_cli.seed),
            pruner=optuna.pruners.MedianPruner(),
        )
        study.set_user_attr("_t0", time.time())
        study.set_user_attr("_model_type", model_type)

        # progress bar per model
        use_bar = os.getenv("HPO_PROGRESS", "bar").lower() == "bar"
        pbar = tqdm(total=args_cli.trials, desc=f"hpo:{model_type}", dynamic_ncols=True) if use_bar else None

        def _progress_cb(study: optuna.Study, trial: optuna.trial.FrozenTrial):
            try:
                t0 = study.user_attrs.get("_t0", time.time())
                elapsed = max(0.0, time.time() - t0)
                done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
                total = args_cli.trials
                per = elapsed / max(1, done)
                rem = max(0.0, (total - done) * per)
                if pbar is not None:
                    pbar.n = done
                    pbar.set_postfix_str(f"best={getattr(study, 'best_value', None)} eta={rem/60:.1f}m")
                    pbar.refresh()
                else:
                    print(f"[hpo] {model_type} progress {done}/{total} | elapsed {elapsed/60:.1f}m | eta {rem/60:.1f}m | best={getattr(study, 'best_value', None)}")

                # summarize trials
                completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
                rows = []
                for t in completed:
                    r = {"trial": t.number, "value": t.value}
                    r.update(t.params)
                    # include user attributes like param_count, score_log_p
                    try:
                        r.update(getattr(t, "user_attrs", {}))
                    except Exception:
                        pass
                    rows.append(r)
                try:
                    _write_json_atomic(
                        os.path.join(out_dir, f"hpo_{model_type}_{args_cli.mode}_trials.json"),
                        rows,
                    )
                    # status heartbeat for crash resilience
                    status = {
                        "model_type": model_type,
                        "mode": args_cli.mode,
                        "done": done,
                        "total": total,
                        "best": getattr(study, "best_value", None),
                        "eta_minutes": rem / 60.0,
                        "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
                    }
                    _write_json_atomic(os.path.join(out_dir, f"status_{model_type}_{args_cli.mode}.json"), status)
                except Exception as e:
                    print(e)
            except Exception as e:
                print(e)
                pass

        try:
            study.optimize(objective, n_trials=args_cli.trials, callbacks=[_progress_cb])
        finally:
            if pbar is not None:
                pbar.close()

        have_best = True
        try:
            print(f"best value (log p) for {model_type}:", study.best_value)
            print("best params:")
            for k, v in study.best_trial.params.items():
                print(f"  {k}: {v}")
            # also surface parameter count for the best trial
            try:
                _best_pc = getattr(study.best_trial, "user_attrs", {}).get("param_count")
                if _best_pc is not None:
                    print(f"param_count for best trial: {int(_best_pc)}")
            except Exception as e:
                print(e)
        except Exception as e:
            print(e)
            have_best = False
            print(f"no completed trials for {model_type}")

        payload = {"mode": args_cli.mode, "model_type": model_type}
        if have_best:
            payload.update({
                "best_value_logp": getattr(study, "best_value", None),
                "best_params": getattr(getattr(study, "best_trial", None), "params", {}),
                "best_param_count": getattr(getattr(study, "best_trial", None), "user_attrs", {}).get("param_count"),
            })
        _write_json_atomic(os.path.join(out_dir, f"hpo_{model_type}_{args_cli.mode}.json"), payload)

        # build per-model summary entry
        try:
            import math as _math
            completed_trials = [
                t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
            ]
            top_trials_sorted = sorted(
                completed_trials, key=lambda t: (t.value if t.value is not None else float("-inf")), reverse=True
            )
            top_k = []
            for t in top_trials_sorted[:3]:
                top_k.append(
                    {
                        "trial": t.number,
                        "value_logp": t.value,
                        "value_p": (float(_math.exp(t.value)) if (t.value is not None) else None),
                        "params": t.params,
                        "param_count": getattr(t, "user_attrs", {}).get("param_count"),
                    }
                )
            best_logp = getattr(study, "best_value", None) if have_best else None
            best_param_count = (
                getattr(getattr(study, "best_trial", None), "user_attrs", {}).get("param_count")
                if have_best else None
            )
            aggregated_results.append(
                {
                    "model_type": model_type,
                    "mode": args_cli.mode,
                    "num_completed_trials": len(completed_trials),
                    "best_value_logp": best_logp,
                    "best_value_p": (float(_math.exp(best_logp)) if (best_logp is not None) else None),
                    "best_params": (getattr(getattr(study, "best_trial", None), "params", {}) if have_best else {}),
                    "best_param_count": best_param_count,
                    "top_trials": top_k,
                }
            )
        except Exception as e:
            print(e)


    # print and persist aggregated summary across all models
    try:
        ts = time.strftime("%Y-%m-%d_%H-%M-%S")
        summary = {
            "timestamp": ts,
            "mode": args_cli.mode,
            "trials_per_model": args_cli.trials,
            "trial_minutes": args_cli.trial_minutes,
            "models": aggregated_results,
        }

        summary_json = os.path.join(out_dir, f"hpo_summary_{args_cli.mode}.json")
        _write_json_atomic(summary_json, summary)

        # pretty text summary
        def _fmt_num(x):
            try:
                if x is None:
                    return "na"
                return f"{float(x):.4f}"
            except Exception:
                return str(x)

        def _fmt_p(x):
            try:
                if x is None:
                    return "na"
                return f"{float(x):.3e}"
            except Exception:
                return str(x)

        def _fmt_int(x):
            try:
                if x is None:
                    return "na"
                return f"{int(x)}"
            except Exception:
                return str(x)

        lines = []
        lines.append(f"HPO SUMMARY ({args_cli.mode})")
        lines.append("")
        header = f"{'model':<18}{'best_logp':>14}{'best_p':>16}{'params':>12}{'trials':>10}"
        lines.append(header)
        lines.append("-" * len(header))
        for r in aggregated_results:
            lines.append(
                f"{r['model_type']:<18}{_fmt_num(r.get('best_value_logp')):>14}{_fmt_p(r.get('best_value_p')):>16}{_fmt_int(r.get('best_param_count')):>12}{r.get('num_completed_trials', 0):>10}"
            )
        lines.append("")
        lines.append("top trials by model (up to 3):")
        for r in aggregated_results:
            lines.append(f"- {r['model_type']}")
            for t in r.get("top_trials", []):
                lines.append(
                    f"    trial {t['trial']}: logp={_fmt_num(t.get('value_logp'))}, p={_fmt_p(t.get('value_p'))}, params={_fmt_int(t.get('param_count'))}"
                )

        summary_txt = os.path.join(out_dir, f"hpo_summary_{args_cli.mode}.txt")
        try:
            tmp_path = summary_txt + ".tmp"
            with open(tmp_path, "w") as f:
                f.write("\n".join(lines) + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, summary_txt)
        except Exception:
            pass

        # print to stdout
        print("\n".join(lines))
        print(f"saved summary to {summary_json} and {summary_txt}")
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()
