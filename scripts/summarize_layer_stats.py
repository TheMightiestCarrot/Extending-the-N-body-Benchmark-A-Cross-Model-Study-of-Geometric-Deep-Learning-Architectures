import argparse
import json
import os
from collections import defaultdict


def load_jsonl(path):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def summarize_run(run_dir: str, label: str) -> str:
    stats_path = os.path.join(run_dir, "layer_stats.jsonl")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"No layer_stats.jsonl at {run_dir}")

    # Track earliest NaN/Inf
    earliest_nan = None
    earliest_nan_keys = []

    # Track maxima per key
    max_values = {}
    max_steps = {}

    # Track per-layer maxima for a few canonical metrics
    per_layer = defaultdict(lambda: defaultdict(lambda: (float("-inf"), None)))

    for rec in load_jsonl(stats_path):
        step = rec.get("step")
        # detect any *_nan_or_inf
        nan_keys = [k for k, v in rec.items() if k.endswith("nan_or_inf") and v]
        if nan_keys and earliest_nan is None:
            earliest_nan = step
            earliest_nan_keys = nan_keys

        # update maxima
        for k, v in rec.items():
            if k == "step":
                continue
            try:
                val = float(v)
            except Exception:
                continue
            if (k not in max_values) or (val > max_values[k]):
                max_values[k] = val
                max_steps[k] = step
            # per-layer rollup: key structure debug/L{idx}.{stage}.{metric}
            if k.startswith("debug/L"):
                try:
                    _, rest = k.split("/", 1)
                    layer_and_rest = rest.split(".")
                    layer_id = layer_and_rest[0]  # e.g., L0
                    metric_name = ".".join(layer_and_rest[1:])
                    cur_val, _ = per_layer[layer_id][metric_name]
                    if val > cur_val:
                        per_layer[layer_id][metric_name] = (val, step)
                except Exception:
                    pass

    # Build markdown
    lines = []
    lines.append(f"## {label}")
    lines.append("")
    lines.append(f"Run dir: `{run_dir}`")
    if earliest_nan is not None:
        lines.append(f"- Earliest NaN/Inf at step {earliest_nan} in: {', '.join(sorted(set(earliest_nan_keys)))}")
    else:
        lines.append("- No NaN/Inf detected within captured steps.")

    # Top 6 maxima overall
    top_items = sorted(max_values.items(), key=lambda kv: kv[1], reverse=True)[:6]
    if top_items:
        lines.append("- Top metrics (max value @ step):")
        for k, v in top_items:
            lines.append(f"  - {k}: {v:.6g} @ {max_steps[k]}")

    # Per-layer brief
    if per_layer:
        lines.append("- Per-layer highlights:")
        for layer in sorted(per_layer.keys(), key=lambda s: int(s[1:]) if s[1:].isdigit() else 0):
            layer_items = per_layer[layer]
            # show a few interesting metrics if present
            show = []
            for name in (
                "inter.scalar_msg_max",
                "inter.vector_msg_norm_max",
                "mix.q_abs_max",
                "mix.mu_norm_max",
                "mix.dq_max",
                "mix.dmu_scale_max",
            ):
                if name in layer_items:
                    val, st = layer_items[name]
                    show.append(f"{name}={val:.6g}@{st}")
            if show:
                lines.append(f"  - {layer}: " + "; ".join(show))

    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("label")
    ap.add_argument("--append_to", default=None, help="Optional markdown file to append to")
    args = ap.parse_args()

    md = summarize_run(args.run_dir, args.label)
    if args.append_to:
        with open(args.append_to, "a") as f:
            f.write(md)
            f.write("\n\n---\n\n")
    else:
        print(md)


if __name__ == "__main__":
    main()

