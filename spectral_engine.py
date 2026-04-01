"""
EvidenceSpectral: Random Matrix Theory analysis of trust component correlations.
Analyses 5x5 correlation matrix of trust components across 6,229 Cochrane MAs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

SCORES_PATH = Path(r"C:\Models\EvidenceScore\results\scores.csv")
GROUPS_PATH = Path(r"C:\Models\TrustGate\data\review_groups.csv")

COMPONENT_COLS = [
    "audit_score",
    "consistency_score",
    "robustness_score",
    "stability_score",
    "power_score",
]


def load_scores(path: Path = SCORES_PATH) -> pd.DataFrame:
    """Load scores CSV and return DataFrame with 5 component columns."""
    df = pd.read_csv(path)
    missing = [c for c in COMPONENT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in scores CSV: {missing}")
    return df[["ma_id", "review_id"] + COMPONENT_COLS].copy()


def load_groups(path: Path = GROUPS_PATH) -> pd.DataFrame:
    """Load review groups CSV."""
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# 2. Correlation matrix
# ---------------------------------------------------------------------------

def compute_correlation_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Compute Pearson correlation matrix (p x p) from component score columns.
    Returns a numpy array of shape (p, p).
    """
    X = df[COMPONENT_COLS].values.astype(float)
    corr = np.corrcoef(X, rowvar=False)
    return corr


# ---------------------------------------------------------------------------
# 3. Eigendecomposition
# ---------------------------------------------------------------------------

def eigen_decompose(corr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Eigendecompose a symmetric matrix.
    Returns (eigenvalues, eigenvectors) sorted by eigenvalue descending.
    Eigenvalues are real; imaginary parts discarded (should be ~0 for symmetric).
    """
    vals, vecs = np.linalg.eigh(corr)  # eigh for symmetric/hermitian
    # eigh returns ascending order; reverse to descending
    idx = np.argsort(vals)[::-1]
    vals = np.real(vals[idx])
    vecs = np.real(vecs[:, idx])
    return vals, vecs


# ---------------------------------------------------------------------------
# 4. Marchenko–Pastur bounds
# ---------------------------------------------------------------------------

def mp_bounds(p: int, n: int) -> tuple[float, float, float]:
    """
    Compute Marchenko-Pastur bounds for random matrix theory null distribution.

    Parameters
    ----------
    p : number of variables
    n : number of observations

    Returns
    -------
    (gamma, lambda_plus, lambda_minus)
      gamma      = p / n  (aspect ratio)
      lambda_+   = (1 + sqrt(gamma))^2  (upper bulk edge)
      lambda_-   = (1 - sqrt(gamma))^2  (lower bulk edge)
    """
    gamma = p / n
    sqrt_gamma = math.sqrt(gamma)
    lambda_plus = (1.0 + sqrt_gamma) ** 2
    lambda_minus = (1.0 - sqrt_gamma) ** 2
    return gamma, lambda_plus, lambda_minus


def count_signal_eigenvalues(eigenvalues: np.ndarray, lambda_plus: float) -> int:
    """Return count of eigenvalues exceeding the MP upper bound (signal dimensions)."""
    return int(np.sum(eigenvalues > lambda_plus))


# ---------------------------------------------------------------------------
# 5. Tracy–Widom test
# ---------------------------------------------------------------------------

def tracy_widom_stat(
    lambda_max: float,
    p: int,
    n: int,
) -> tuple[float, float, float]:
    """
    Compute Tracy-Widom (TW1) standardised statistic for the largest eigenvalue.

    Reference: Patterson et al. (2006) PLoS Genet; Johnstone (2001).

    mu_TW    = (sqrt(n-1) + sqrt(p))^2 / n
    sigma_TW = (sqrt(n-1) + sqrt(p)) / n * (1/sqrt(n-1) + 1/sqrt(p))^(1/3)
    s        = (lambda_max - mu_TW) / sigma_TW

    Returns
    -------
    (mu_TW, sigma_TW, s)  where s is the TW1 standardised statistic.
    Note: no closed-form CDF is bundled here; callers use the s value directly.
    Empirically: |s| > 2 indicates signal (significant departure from null).
    """
    mu_tw = (math.sqrt(n - 1) + math.sqrt(p)) ** 2 / n
    sigma_tw = (
        (math.sqrt(n - 1) + math.sqrt(p)) / n
        * (1.0 / math.sqrt(n - 1) + 1.0 / math.sqrt(p)) ** (1.0 / 3.0)
    )
    s = (lambda_max - mu_tw) / sigma_tw
    return mu_tw, sigma_tw, s


def tw_p_value_approx(s: float) -> float:
    """
    Approximation of TW1 p-value using the normal approximation.
    The TW1 distribution has mean ~-1.2065 and variance ~1.6078^2 in the
    standardised form; we use a normal approximation as a practical surrogate
    (accurate for screening, not precise hypothesis testing).
    TW1 mean ~ -1.2065, std ~ 1.6078 (from numerical tables).
    """
    # Rescale to approximately standard TW1 → then use normal tail
    TW1_MEAN = -1.2065
    TW1_STD = 1.6078
    z = (s - TW1_MEAN) / TW1_STD
    p_val = 1.0 - float(stats.norm.cdf(z))
    return p_val


# ---------------------------------------------------------------------------
# 6. Participation ratio
# ---------------------------------------------------------------------------

def participation_ratio(eigenvector: np.ndarray) -> float:
    """
    Participation ratio (PR) of an eigenvector.
    PR = (sum v_i^2)^2 / sum(v_i^4)
    Range: [1, p].  PR≈1 → localised on one component; PR≈p → spread uniformly.
    """
    v2 = eigenvector ** 2
    pr = (np.sum(v2) ** 2) / np.sum(v2 ** 2)
    return float(pr)


def participation_ratios(eigenvectors: np.ndarray) -> np.ndarray:
    """Compute PR for every column eigenvector. Returns array of length p."""
    return np.array([participation_ratio(eigenvectors[:, k]) for k in range(eigenvectors.shape[1])])


# ---------------------------------------------------------------------------
# 7. Per-domain correlation analysis
# ---------------------------------------------------------------------------

def merge_domains(df_scores: pd.DataFrame, df_groups: pd.DataFrame) -> pd.DataFrame:
    """
    Merge scores with review domain groups.
    scores.csv has review_id like 'CD013614'.
    review_groups.csv has review_id_prefix like 'CD013614'.
    """
    df_scores = df_scores.copy()
    df_scores["review_id_prefix"] = df_scores["review_id"].str.strip()
    merged = df_scores.merge(df_groups, on="review_id_prefix", how="left")
    return merged


def per_domain_correlations(
    df_merged: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """
    Compute 5x5 correlation matrices per review domain.
    Returns dict mapping domain name → correlation matrix.
    Only includes domains with >= 30 observations.
    """
    results: dict[str, np.ndarray] = {}
    if "review_group" not in df_merged.columns:
        return results
    for domain, grp in df_merged.groupby("review_group"):
        if len(grp) >= 30:
            corr = compute_correlation_matrix(grp)
            results[str(domain)] = corr
    return results


def domain_correlation_diff(
    domain_corrs: dict[str, np.ndarray],
    d1: str,
    d2: str,
) -> np.ndarray:
    """Return element-wise difference of two domain correlation matrices."""
    if d1 not in domain_corrs or d2 not in domain_corrs:
        raise KeyError(f"Domain(s) not found: {d1}, {d2}")
    return domain_corrs[d1] - domain_corrs[d2]


# ---------------------------------------------------------------------------
# 8. Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    scores_path: Path = SCORES_PATH,
    groups_path: Path = GROUPS_PATH,
) -> dict:
    """
    Full EvidenceSpectral pipeline.
    Returns a results dict suitable for JSON serialisation.
    """
    # --- load data
    df_scores = load_scores(scores_path)
    df_groups = load_groups(groups_path)
    n, p = len(df_scores), len(COMPONENT_COLS)

    # --- correlation matrix
    corr = compute_correlation_matrix(df_scores)

    # --- eigendecomposition
    eigenvalues, eigenvectors = eigen_decompose(corr)

    # --- MP bounds
    gamma, lambda_plus, lambda_minus = mp_bounds(p, n)
    n_signal = count_signal_eigenvalues(eigenvalues, lambda_plus)

    # --- Tracy-Widom
    mu_tw, sigma_tw, tw_s = tracy_widom_stat(eigenvalues[0], p, n)
    tw_p = tw_p_value_approx(tw_s)

    # --- Participation ratios
    prs = participation_ratios(eigenvectors)

    # --- Per-domain
    df_merged = merge_domains(df_scores, df_groups)
    domain_corrs = per_domain_correlations(df_merged)

    # --- First eigenvector loadings
    first_ev = eigenvectors[:, 0].tolist()

    results = {
        "n_obs": n,
        "n_vars": p,
        "component_names": COMPONENT_COLS,
        "correlation_matrix": corr.tolist(),
        "eigenvalues": eigenvalues.tolist(),
        "eigenvectors": eigenvectors.tolist(),
        "mp_gamma": gamma,
        "mp_lambda_plus": lambda_plus,
        "mp_lambda_minus": lambda_minus,
        "n_signal_eigenvalues": n_signal,
        "tw_mu": mu_tw,
        "tw_sigma": sigma_tw,
        "tw_s": tw_s,
        "tw_p_approx": tw_p,
        "participation_ratios": prs.tolist(),
        "first_eigenvector": first_ev,
        "domains_analysed": list(domain_corrs.keys()),
        "domain_correlations": {
            k: v.tolist() for k, v in domain_corrs.items()
        },
    }
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    results = run_pipeline()
    print("=" * 60)
    print("EvidenceSpectral Results")
    print("=" * 60)
    print(f"N observations : {results['n_obs']}")
    print(f"N variables    : {results['n_vars']}")
    print(f"Eigenvalues    : {[round(v, 4) for v in results['eigenvalues']]}")
    print(f"MP lambda+     : {results['mp_lambda_plus']:.4f}")
    print(f"MP lambda-     : {results['mp_lambda_minus']:.4f}")
    print(f"Signal dims    : {results['n_signal_eigenvalues']}")
    print(f"TW s-statistic : {results['tw_s']:.4f}")
    print(f"TW p (approx)  : {results['tw_p_approx']:.6f}")
    print(f"Domains found  : {len(results['domains_analysed'])}")
    print()
    print("First eigenvector loadings:")
    for name, load in zip(results["component_names"], results["first_eigenvector"]):
        print(f"  {name:25s}: {load:+.4f}")
    print()
    print("Participation ratios (per eigenvector):")
    for i, pr in enumerate(results["participation_ratios"]):
        print(f"  EV{i+1}: {pr:.3f}")

    # Save JSON
    out = Path(r"C:\Models\EvidenceSpectral\results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")
