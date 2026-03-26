"""Tests for Morgan fingerprint generation."""

import numpy as np
import pandas as pd

from src.chemistry import parse_smiles_column
from src.config import FP_NBITS, COL_VALID
from src.fingerprints import compute_morgan_fingerprints


class TestMorganFingerprints:
    def test_output_shape(self, enriched_df, fp_matrix):
        n_valid = enriched_df[COL_VALID].sum()
        assert fp_matrix.shape == (n_valid, FP_NBITS)

    def test_binary_values(self, fp_matrix):
        assert set(np.unique(fp_matrix)).issubset({0, 1})

    def test_no_valid_molecules_returns_none(self):
        df = parse_smiles_column(pd.DataFrame({"SMILES": ["bad", "worse"]}))
        result = compute_morgan_fingerprints(df)
        assert result is None

    def test_different_molecules_different_fps(self, enriched_df, fp_matrix):
        # Ethanol and aspirin should have different fingerprints
        assert not np.array_equal(fp_matrix[0], fp_matrix[2])

    def test_identical_smiles_identical_fps(self):
        """Two identical SMILES should produce identical fingerprints."""
        from src.chemistry import parse_smiles_column
        df = parse_smiles_column(pd.DataFrame({"SMILES": ["CCO", "CCO"]}))
        fps = compute_morgan_fingerprints(df)
        assert np.array_equal(fps[0], fps[1])
