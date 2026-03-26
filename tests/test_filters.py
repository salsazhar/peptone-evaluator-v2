"""Tests for the filtering pipeline."""

import pandas as pd

from src.config import COL_VALID, COL_UNIQUE
from src.filters import FilterSpec, apply_filters


class TestFilterSpec:
    def test_default_spec_passes_everything(self, enriched_df):
        spec = FilterSpec(valid_only=False)
        result = apply_filters(enriched_df, spec)
        assert len(result) == len(enriched_df)

    def test_valid_only(self, enriched_df):
        spec = FilterSpec(valid_only=True)
        result = apply_filters(enriched_df, spec)
        assert result[COL_VALID].all()

    def test_unique_only(self, enriched_df):
        from src.chemistry import get_unique_mask
        enriched_df[COL_UNIQUE] = get_unique_mask(enriched_df)
        spec = FilterSpec(valid_only=True, unique_only=True)
        result = apply_filters(enriched_df, spec)
        n_unique = enriched_df.loc[enriched_df[COL_VALID], "canonical_SMILES"].nunique()
        assert len(result) == n_unique

    def test_lipinski_only(self, enriched_df):
        spec = FilterSpec(valid_only=True, lipinski_only=True)
        result = apply_filters(enriched_df, spec)
        if not result.empty:
            assert result["passes_lipinski_like"].all()

    def test_mw_range_filter(self, enriched_df):
        spec = FilterSpec(valid_only=True, mw_range=(100.0, 200.0))
        result = apply_filters(enriched_df, spec)
        if not result.empty:
            assert (result["MolWt"] >= 100.0).all()
            assert (result["MolWt"] <= 200.0).all()

    def test_does_not_mutate_input(self, enriched_df):
        original_len = len(enriched_df)
        spec = FilterSpec(valid_only=True, lipinski_only=True)
        apply_filters(enriched_df, spec)
        assert len(enriched_df) == original_len

    def test_all_filtered_out_returns_empty(self, enriched_df):
        spec = FilterSpec(valid_only=True, mw_range=(99999.0, 99999.0))
        result = apply_filters(enriched_df, spec)
        assert result.empty

    def test_pic50_range_filter(self, enriched_df):
        spec = FilterSpec(valid_only=True, pic50_range=(6.0, 8.0))
        result = apply_filters(enriched_df, spec)
        if not result.empty and "pIC50" in result.columns:
            vals = result["pIC50"].dropna()
            assert (vals >= 6.0).all()
            assert (vals <= 8.0).all()
