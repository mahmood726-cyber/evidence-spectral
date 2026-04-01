"""
EvidenceSpectral — 25-test suite.
Groups:
  T1–T3   : Correlation matrix
  T4–T7   : Eigendecomposition
  T8–T11  : Marchenko-Pastur bounds
  T12–T14 : Tracy-Widom test
  T15–T18 : Participation ratio
  T19–T20 : Per-domain correlation differences
  T21–T25 : Pipeline integration
"""

import math
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure the engine module is importable even when run from this directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from spectral_engine import (
    compute_correlation_matrix,
    count_signal_eigenvalues,
    domain_correlation_diff,
    eigen_decompose,
    load_groups,
    load_scores,
    merge_domains,
    mp_bounds,
    participation_ratio,
    participation_ratios,
    per_domain_correlations,
    run_pipeline,
    tracy_widom_stat,
    tw_p_value_approx,
    COMPONENT_COLS,
    SCORES_PATH,
    GROUPS_PATH,
)

TOL = 1e-6


# ============================================================
# Helpers
# ============================================================

def _make_df(data: np.ndarray) -> pd.DataFrame:
    """Wrap a numpy array as a scores DataFrame."""
    return pd.DataFrame(data, columns=COMPONENT_COLS)


def _identity_df(n: int = 200) -> pd.DataFrame:
    """Create a DataFrame where each column is orthogonal (identity corr)."""
    rng = np.random.default_rng(seed=42)
    data = rng.standard_normal((n, 5))
    # Force zero correlation by using exactly orthogonal columns
    # Use QR decomposition to get orthogonal vectors then scale
    Q, _ = np.linalg.qr(rng.standard_normal((n, 5)))
    return pd.DataFrame(Q, columns=COMPONENT_COLS)


# ============================================================
# T1–T3: Correlation matrix
# ============================================================

class TestCorrelationMatrix:

    def test_T1_identity_data_gives_identity_corr(self):
        """T1: Exactly orthogonal columns → correlation matrix = identity."""
        # Build exactly orthogonal, zero-mean columns from Hadamard-style construction
        n = 1000
        # Column k: alternating +1/-1 pattern with period 2^k → guaranteed zero covariance
        # Use a direct algebraic construction: data = n x 5 with columns that are
        # exactly pairwise orthogonal *and* zero-mean.
        # Simplest: use the first 5 columns of a DFT-like real orthogonal basis.
        t = np.arange(n)
        cols = [np.sin(2 * np.pi * (k + 1) * t / n) for k in range(5)]
        data = np.column_stack(cols)
        df = _make_df(data)
        corr = compute_correlation_matrix(df)
        np.testing.assert_allclose(corr, np.eye(5), atol=1e-3)

    def test_T2_shape_is_p_by_p(self):
        """T2: Output shape is (p, p) = (5, 5)."""
        rng = np.random.default_rng(0)
        df = _make_df(rng.standard_normal((100, 5)))
        corr = compute_correlation_matrix(df)
        assert corr.shape == (5, 5)

    def test_T3_diagonal_is_one(self):
        """T3: Diagonal entries are exactly 1.0."""
        rng = np.random.default_rng(1)
        df = _make_df(rng.standard_normal((300, 5)))
        corr = compute_correlation_matrix(df)
        np.testing.assert_allclose(np.diag(corr), np.ones(5), atol=TOL)

    def test_T3b_known_two_variable_correlation(self):
        """T3b: Two-variable known correlation is recovered correctly."""
        n = 1000
        rng = np.random.default_rng(7)
        x = rng.standard_normal(n)
        y = 0.8 * x + math.sqrt(1 - 0.64) * rng.standard_normal(n)
        # Pad with 3 independent columns
        rest = rng.standard_normal((n, 3))
        data = np.column_stack([x, y, rest])
        df = _make_df(data)
        corr = compute_correlation_matrix(df)
        # corr[0,1] should be close to 0.8
        assert abs(corr[0, 1] - 0.8) < 0.03


# ============================================================
# T4–T7: Eigendecomposition
# ============================================================

class TestEigendecomposition:

    def test_T4_known_matrix_eigenvalues(self):
        """T4: Known symmetric matrix → eigenvalues match analytical values."""
        # 2x2 symmetric: [[2, 1],[1, 2]] → eigenvalues 3, 1
        A = np.array([[2.0, 1.0], [1.0, 2.0]])
        vals, _ = eigen_decompose(A)
        np.testing.assert_allclose(sorted(vals, reverse=True), [3.0, 1.0], atol=TOL)

    def test_T5_eigenvalues_sorted_descending(self):
        """T5: Returned eigenvalues are sorted descending."""
        rng = np.random.default_rng(2)
        A = rng.standard_normal((5, 5))
        corr = np.corrcoef(A.T)
        vals, _ = eigen_decompose(corr)
        for i in range(len(vals) - 1):
            assert vals[i] >= vals[i + 1] - TOL

    def test_T6_eigenvalues_sum_equals_p(self):
        """T6: Sum of eigenvalues of 5x5 correlation matrix = 5 (trace)."""
        rng = np.random.default_rng(3)
        df = _make_df(rng.standard_normal((500, 5)))
        corr = compute_correlation_matrix(df)
        vals, _ = eigen_decompose(corr)
        assert abs(np.sum(vals) - 5.0) < TOL

    def test_T7_eigenvectors_orthonormal(self):
        """T7: Eigenvector matrix is orthonormal (V^T V ≈ I)."""
        rng = np.random.default_rng(4)
        df = _make_df(rng.standard_normal((200, 5)))
        corr = compute_correlation_matrix(df)
        _, vecs = eigen_decompose(corr)
        product = vecs.T @ vecs
        np.testing.assert_allclose(product, np.eye(5), atol=1e-10)


# ============================================================
# T8–T11: Marchenko-Pastur bounds
# ============================================================

class TestMarchenkoPastur:

    def test_T8_gamma_correct(self):
        """T8: gamma = p/n for p=5, n=6229."""
        gamma, _, _ = mp_bounds(5, 6229)
        expected = 5 / 6229
        assert abs(gamma - expected) < TOL

    def test_T9_lambda_plus_near_1_for_small_gamma(self):
        """T9: For very small gamma (p<<n), lambda+ ≈ 1.0."""
        _, lp, lm = mp_bounds(5, 6229)
        # Both bounds should be within ~6% of 1.0 for gamma ≈ 0.0008
        assert abs(lp - 1.0) < 0.10
        assert abs(lm - 1.0) < 0.10

    def test_T10_lambda_plus_for_gamma_one(self):
        """T10: gamma=1 → lambda_+ = 4, lambda_- = 0."""
        gamma, lp, lm = mp_bounds(p=100, n=100)
        assert abs(gamma - 1.0) < TOL
        assert abs(lp - 4.0) < TOL
        assert abs(lm - 0.0) < TOL

    def test_T11_signal_count_correct(self):
        """T11: Eigenvalues=[1.841, 1.137, 0.905, 0.676, 0.441], lambda+=1.057 → 2 signal."""
        known_eigenvalues = np.array([1.841, 1.137, 0.905, 0.676, 0.441])
        _, lp, _ = mp_bounds(5, 6229)
        n_sig = count_signal_eigenvalues(known_eigenvalues, lp)
        assert n_sig == 2

    def test_T11b_lambda_plus_formula(self):
        """T11b: lambda+ = (1 + sqrt(gamma))^2 exactly."""
        gamma, lp, _ = mp_bounds(5, 6229)
        expected_lp = (1.0 + math.sqrt(gamma)) ** 2
        assert abs(lp - expected_lp) < TOL


# ============================================================
# T12–T14: Tracy-Widom test
# ============================================================

class TestTracyWidom:

    def test_T12_signal_eigenvalue_gives_large_s(self):
        """T12: A large eigenvalue (signal) → TW s > 2."""
        # Use eigenvalue 1.841 (clearly above MP bound for p=5, n=6229)
        _, _, s = tracy_widom_stat(1.841, p=5, n=6229)
        assert s > 2.0

    def test_T13_noise_eigenvalue_gives_small_s(self):
        """T13: A noise eigenvalue (below MP bound) → TW s < 2."""
        # Use eigenvalue 0.905 (below lambda_+ ≈ 1.057)
        _, _, s = tracy_widom_stat(0.905, p=5, n=6229)
        assert s < 2.0

    def test_T14_tw_mu_formula(self):
        """T14: mu_TW formula is (sqrt(n-1)+sqrt(p))^2/n."""
        p, n = 5, 6229
        mu, _, _ = tracy_widom_stat(1.5, p, n)
        expected = (math.sqrt(n - 1) + math.sqrt(p)) ** 2 / n
        assert abs(mu - expected) < TOL

    def test_T14b_tw_p_approx_signal(self):
        """T14b: Signal eigenvalue → TW p-value < 0.05."""
        _, _, s = tracy_widom_stat(1.841, p=5, n=6229)
        p_val = tw_p_value_approx(s)
        assert p_val < 0.05


# ============================================================
# T15–T18: Participation ratio
# ============================================================

class TestParticipationRatio:

    def test_T15_unit_vector_pr_equals_one(self):
        """T15: Unit vector localised on one component → PR = 1."""
        v = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        pr = participation_ratio(v)
        assert abs(pr - 1.0) < TOL

    def test_T16_uniform_vector_pr_equals_p(self):
        """T16: Uniform eigenvector → PR = p = 5."""
        p = 5
        v = np.ones(p) / math.sqrt(p)
        pr = participation_ratio(v)
        assert abs(pr - p) < TOL

    def test_T17_pr_in_range_1_to_p(self):
        """T17: PR is always in [1, p] for any normalised vector."""
        rng = np.random.default_rng(9)
        for _ in range(50):
            v = rng.standard_normal(5)
            v /= np.linalg.norm(v)
            pr = participation_ratio(v)
            assert 1.0 - TOL <= pr <= 5.0 + TOL

    def test_T18_batch_participation_ratios(self):
        """T18: batch participation_ratios returns array of length p."""
        rng = np.random.default_rng(5)
        df = _make_df(rng.standard_normal((200, 5)))
        corr = compute_correlation_matrix(df)
        _, vecs = eigen_decompose(corr)
        prs = participation_ratios(vecs)
        assert prs.shape == (5,)
        assert np.all(prs >= 1.0 - 1e-4)
        assert np.all(prs <= 5.0 + 1e-4)


# ============================================================
# T19–T20: Per-domain correlation differences
# ============================================================

class TestDomainCorrelations:

    def _make_merged_df(self) -> pd.DataFrame:
        """Create a minimal merged DataFrame with two review groups."""
        rng = np.random.default_rng(10)
        n1, n2 = 60, 80
        data1 = rng.standard_normal((n1, 5))
        data2 = rng.standard_normal((n2, 5)) + 0.5  # slightly shifted
        all_data = np.vstack([data1, data2])
        df = pd.DataFrame(all_data, columns=COMPONENT_COLS)
        df["review_id"] = ["CD000001"] * n1 + ["CD000002"] * n2
        df["review_id_prefix"] = df["review_id"]
        df["review_group"] = ["Cardiology"] * n1 + ["Respiratory"] * n2
        return df

    def test_T19_domain_corr_diff_shape(self):
        """T19: Domain correlation difference has shape (5, 5)."""
        df = self._make_merged_df()
        domain_corrs = per_domain_correlations(df)
        diff = domain_correlation_diff(domain_corrs, "Cardiology", "Respiratory")
        assert diff.shape == (5, 5)

    def test_T20_domain_corr_diff_antisymmetric(self):
        """T20: diff(A, B) = -diff(B, A)."""
        df = self._make_merged_df()
        domain_corrs = per_domain_correlations(df)
        diff_ab = domain_correlation_diff(domain_corrs, "Cardiology", "Respiratory")
        diff_ba = domain_correlation_diff(domain_corrs, "Respiratory", "Cardiology")
        np.testing.assert_allclose(diff_ab, -diff_ba, atol=TOL)


# ============================================================
# T21–T25: Pipeline integration (uses real data files)
# ============================================================

@pytest.mark.skipif(
    not SCORES_PATH.exists() or not GROUPS_PATH.exists(),
    reason="Real data files not available",
)
class TestPipelineIntegration:

    @pytest.fixture(scope="class")
    def results(self):
        return run_pipeline()

    def test_T21_pipeline_returns_dict(self, results):
        """T21: Pipeline returns a dict with required keys."""
        required_keys = [
            "n_obs", "n_vars", "eigenvalues", "correlation_matrix",
            "mp_lambda_plus", "n_signal_eigenvalues", "tw_s", "tw_p_approx",
            "participation_ratios", "first_eigenvector",
        ]
        for key in required_keys:
            assert key in results, f"Missing key: {key}"

    def test_T22_n_obs_matches_csv(self, results):
        """T22: n_obs matches actual row count of scores CSV."""
        df = pd.read_csv(SCORES_PATH)
        assert results["n_obs"] == len(df)

    def test_T23_signal_dimensions_two(self, results):
        """T23: Real data shows exactly 2 signal eigenvalues above MP bound."""
        assert results["n_signal_eigenvalues"] == 2

    def test_T24_largest_eigenvalue_above_MP(self, results):
        """T24: Largest eigenvalue exceeds lambda_+."""
        assert results["eigenvalues"][0] > results["mp_lambda_plus"]

    def test_T25_tw_stat_large_for_real_data(self, results):
        """T25: TW s-statistic is > 10 for the real 6229-MA dataset (very strong signal)."""
        assert results["tw_s"] > 10.0
