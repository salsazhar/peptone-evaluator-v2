"""Tests for Tanimoto similarity, diversity scoring, and duplicate detection."""

import numpy as np
import pandas as pd

from src.similarity import (
    compute_similarity_metrics,
    find_duplicates,
    top_similar_pairs,
    top_isolated_molecules,
)


class TestSimilarityMetrics:
    def test_returns_expected_keys(self, fp_matrix):
        result = compute_similarity_metrics(fp_matrix)
        expected = {
            "mean_pairwise_similarity", "median_pairwise_similarity",
            "mean_nn_similarity", "diversity_score",
            "nn_similarities", "sim_matrix",
        }
        assert expected.issubset(result.keys())

    def test_diversity_in_range(self, fp_matrix):
        result = compute_similarity_metrics(fp_matrix)
        assert 0.0 <= result["diversity_score"] <= 1.0

    def test_mean_pairwise_in_range(self, fp_matrix):
        result = compute_similarity_metrics(fp_matrix)
        assert 0.0 <= result["mean_pairwise_similarity"] <= 1.0

    def test_nn_similarities_length(self, fp_matrix):
        result = compute_similarity_metrics(fp_matrix)
        assert len(result["nn_similarities"]) == fp_matrix.shape[0]

    def test_sim_matrix_shape(self, fp_matrix):
        result = compute_similarity_metrics(fp_matrix)
        n = fp_matrix.shape[0]
        assert result["sim_matrix"].shape == (n, n)

    def test_diagonal_zeroed(self, fp_matrix):
        """Diagonal is 0 by design to exclude self-similarity from metrics."""
        result = compute_similarity_metrics(fp_matrix)
        diag = np.diag(result["sim_matrix"])
        np.testing.assert_array_almost_equal(diag, 0.0)

    def test_identical_molecules_similarity_one(self):
        """Two identical fingerprints should have similarity = 1."""
        fp = np.array([[1, 0, 1, 1, 0, 0, 1, 0]])
        fp_matrix = np.vstack([fp, fp])
        result = compute_similarity_metrics(fp_matrix)
        assert result["mean_pairwise_similarity"] == 1.0
        assert result["diversity_score"] == 0.0


class TestDuplicates:
    def test_finds_duplicates(self):
        smiles = pd.Series(["CCO", "CCO", "c1ccccc1", "c1ccccc1", "CC"])
        result = find_duplicates(smiles)
        assert result["duplicate_count"] == 2  # 2 extra copies

    def test_no_duplicates(self):
        smiles = pd.Series(["CCO", "c1ccccc1", "CC"])
        result = find_duplicates(smiles)
        assert result["duplicate_count"] == 0

    def test_empty_series(self):
        result = find_duplicates(pd.Series([], dtype=str))
        assert result["duplicate_count"] == 0
        assert result["duplicate_rate"] == 0.0


class TestTopSimilarPairs:
    def test_returns_dataframe(self, fp_matrix):
        smiles = [f"mol_{i}" for i in range(fp_matrix.shape[0])]
        result = top_similar_pairs(
            compute_similarity_metrics(fp_matrix)["sim_matrix"],
            smiles, top_n=5,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= 5
        assert "Tanimoto" in result.columns

    def test_single_molecule_returns_empty(self):
        sim = np.array([[1.0]])
        result = top_similar_pairs(sim, ["mol_0"], top_n=5)
        assert result.empty


class TestTopIsolated:
    def test_returns_correct_count(self, fp_matrix):
        metrics = compute_similarity_metrics(fp_matrix)
        smiles = [f"mol_{i}" for i in range(fp_matrix.shape[0])]
        result = top_isolated_molecules(metrics["nn_similarities"], smiles, top_n=3)
        assert len(result) <= 3
        assert "NN_Similarity" in result.columns
