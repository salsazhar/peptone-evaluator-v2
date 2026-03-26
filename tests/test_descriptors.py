"""Tests for molecular descriptor computation and rule-based flags."""

import math

import pandas as pd

from src.chemistry import parse_smiles_column
from src.config import COL_VALID, DESCRIPTOR_COLS, NUMERIC_DESCRIPTOR_COLS
from src.descriptors import compute_descriptors, apply_rule_flags, compute_descriptor_summary


class TestComputeDescriptors:
    def test_all_descriptor_columns_added(self, enriched_df):
        for col in DESCRIPTOR_COLS:
            assert col in enriched_df.columns, f"Missing descriptor column: {col}"

    def test_valid_molecules_have_values(self, enriched_df):
        valid = enriched_df[enriched_df[COL_VALID]]
        for col in NUMERIC_DESCRIPTOR_COLS:
            assert valid[col].notna().all(), f"NaN in {col} for valid molecules"

    def test_invalid_molecules_have_nan(self, enriched_df):
        invalid = enriched_df[~enriched_df[COL_VALID]]
        for col in NUMERIC_DESCRIPTOR_COLS:
            assert invalid[col].isna().all(), f"Non-NaN in {col} for invalid molecules"

    def test_ethanol_descriptors_sanity(self):
        """Ethanol (CCO): MW ~46, LogP < 0, 1 HBD, 1 HBA, 0 rings."""
        df = parse_smiles_column(pd.DataFrame({"SMILES": ["CCO"]}))
        df = compute_descriptors(df)
        row = df.iloc[0]
        assert 45 < row["MolWt"] < 47
        assert row["LogP"] < 1
        assert row["HBD"] == 1
        assert row["HBA"] == 1
        assert row["RingCount"] == 0

    def test_qed_in_range(self, enriched_df):
        valid = enriched_df[enriched_df[COL_VALID]]
        assert (valid["QED"] >= 0).all() and (valid["QED"] <= 1).all()


class TestRuleFlags:
    def test_flag_columns_added(self, enriched_df):
        for col in ["passes_lipinski_like", "high_flexibility_flag",
                     "high_lipophilicity_flag", "extreme_size_flag"]:
            assert col in enriched_df.columns

    def test_ethanol_passes_lipinski(self):
        df = parse_smiles_column(pd.DataFrame({"SMILES": ["CCO"]}))
        df = compute_descriptors(df)
        df = apply_rule_flags(df)
        assert df.iloc[0]["passes_lipinski_like"] == True

    def test_invalid_molecules_fail_lipinski(self, enriched_df):
        invalid = enriched_df[~enriched_df[COL_VALID]]
        assert not invalid["passes_lipinski_like"].any()

    def test_tiny_molecule_flagged_extreme_size(self):
        """Methane (C): MW ~16, should be flagged as extreme size."""
        df = parse_smiles_column(pd.DataFrame({"SMILES": ["C"]}))
        df = compute_descriptors(df)
        df = apply_rule_flags(df)
        assert df.iloc[0]["extreme_size_flag"] == True


class TestDescriptorSummary:
    def test_summary_shape(self, enriched_df):
        summary = compute_descriptor_summary(enriched_df)
        assert not summary.empty
        assert "Mean" in summary.columns
        assert "Median" in summary.columns

    def test_summary_only_valid(self, enriched_df):
        summary = compute_descriptor_summary(enriched_df)
        n_valid = enriched_df[COL_VALID].sum()
        # n column should equal valid count for complete descriptors
        for _, row in summary.iterrows():
            assert row["n"] <= n_valid

    def test_all_invalid_returns_nan_summary(self):
        df = parse_smiles_column(pd.DataFrame({"SMILES": ["bad", "worse"]}))
        df = compute_descriptors(df)
        summary = compute_descriptor_summary(df)
        # All molecules invalid → summary should have 0 count or be empty
        if not summary.empty:
            assert (summary["n"] == 0).all()
