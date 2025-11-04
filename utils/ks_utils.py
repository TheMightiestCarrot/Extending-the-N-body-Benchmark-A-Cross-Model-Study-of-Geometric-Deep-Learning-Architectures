import numpy as np
from scipy import stats
from typing import List
from mpmath import log, mp
from scipy.stats import chi2

def _ks_p(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if a.size == 0 or b.size == 0 or np.all(np.isnan(a)) or np.all(np.isnan(b)):
        return np.nan
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        return np.nan
    _, p = stats.ks_2samp(a, b)
    return float(p)


def _combine_pvalues_fisher(p_values: List[float]) -> float:
    vals = [p for p in p_values if p == p and p > 0.0]  # drop nan, <=0
    if not vals:
        return np.nan
    mp.dps = 200
    chi_stat = -2 * mp.fsum([log(mp.mpf(p)) for p in vals])
    chi_stat = float(chi_stat)
    dof = 2 * len(vals)
    combined = chi2.sf(chi_stat, dof)
    return float(max(combined, 1e-300))