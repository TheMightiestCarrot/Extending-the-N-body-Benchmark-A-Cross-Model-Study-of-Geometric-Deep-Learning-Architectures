import json
import math
import os
from typing import Dict, Tuple, List

import numpy as np


MACRO_FILES = {
    "collision_histogram": ("collision_distributions.json", "collision_histogram"),
    "group_collision_histogram": ("group_collision_distribution.json", "group_collision_count"),
    "leaving_count": ("leaving_distribution.json", "leaving_count"),
    "sharp_turn_count_30": ("sharp_turn_30_distribution.json", "sharp_turn_count_30"),
    "sharp_turn_count_45": ("sharp_turn_45_distribution.json", "sharp_turn_count_45"),
    "sticking_histogram": ("sticking_distributions.json", "sticking_histogram"),
}


def _ks_p(a: np.ndarray, b: np.ndarray) -> float:
    try:
        from scipy import stats
    except Exception:
        return float("nan")
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if a.size == 0 or b.size == 0:
        return float("nan")
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    _, p = stats.ks_2samp(a, b)
    return float(p)


def _fisher_combine(p_values: List[float]) -> float:
    vals = [p for p in p_values if p == p and p > 0.0]
    if not vals:
        return float("nan")
    # high-precision log-sum, fallback to python math
    try:
        from mpmath import mp, log
        from scipy.stats import chi2
        mp.dps = 200
        chi_stat = -2 * mp.fsum([log(mp.mpf(p)) for p in vals])
        chi_stat = float(chi_stat)
        dof = 2 * len(vals)
        combined = chi2.sf(chi_stat, dof)
        return float(max(combined, 1e-300))
    except Exception:
        s = -2.0 * sum(math.log(p) for p in vals)
        dof = 2 * len(vals)
        # crude fallback via normal approx of chi2
        from math import erf, sqrt
        z = (s - dof) / (sqrt(2.0 * dof))
        # approx survival fn
        p_tail = 0.5 * (1.0 - erf(z / math.sqrt(2.0)))
        return max(float(p_tail), 1e-300)


def load_macro_pvalues_from_checkpoint(ckpt_dir: str) -> Tuple[Dict[str, float], float]:
    """
    read the six macro jsons written by plot_macros and compute per-macro ks p-values.
    also include energy p-values if present in nbody_macro_metrics.json.
    returns (per_macro_p, combined_p)
    """
    plots_dir = ckpt_dir
    per_macro: Dict[str, float] = {}
    pvals: List[float] = []

    # macros from plot_macros
    for key, (fname, field_key) in MACRO_FILES.items():
        path = os.path.join(plots_dir, fname)
        if not os.path.exists(path):
            per_macro[key] = float("nan")
            continue
        try:
            with open(path, "r") as f:
                data = json.load(f)
            gt = np.array(data.get("ground truth", {}).get(field_key, []))
            pr = np.array(data.get("predicted", {}).get(field_key, []))
            p = _ks_p(gt, pr)
            per_macro[key] = p
            if p == p and p > 0.0:
                pvals.append(p)
        except Exception:
            per_macro[key] = float("nan")

    # energy ks from trainer’s compact metrics, if present
    energy_metrics_path = os.path.join(plots_dir, "nbody_macro_metrics.json")
    if os.path.exists(energy_metrics_path):
        try:
            with open(energy_metrics_path, "r") as f:
                m = json.load(f)
            ksp = m.get("ks_pvalues", {})
            for ek in ["energy_total", "energy_potential", "energy_kinetic"]:
                p = float(ksp.get(ek, float("nan")))
                per_macro[ek] = p
                if p == p and p > 0.0:
                    pvals.append(p)
        except Exception:
            pass

    combined = _fisher_combine(pvals)
    return per_macro, combined


