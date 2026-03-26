"""Tests for Murcko scaffold decomposition and diversity metrics."""

import pandas as pd

from src.config import COL_VALID
from src.scaffolds import (
    get_murcko_scaffold,
    get_generic_scaffold,
    compute_scaffolds,
    scaffold_frequency,
    scaffold_diversity_stats,
)


class TestMurckoScaffold:
    def test_benzene_is_own_scaffold(self):
        assert get_murcko_scaffold("c1ccccc1") == "c1ccccc1"

    def test_aspirin_has_benzene_scaffold(self):
        scaffold = get_murcko_scaffold("CC(=O)Oc1ccccc1C(=O)O")
        assert scaffold is not None
        assert "c1ccccc1" in scaffold

    def test_acyclic_molecule(self):
        # Ethanol has no ring — scaffold should be empty or None
        scaffold = get_murcko_scaffold("CCO")
        # Murcko of acyclic = empty string or None depending on RDKit version
        assert scaffold is None or scaffold == ""

    def test_invalid_smiles(self):
        assert get_murcko_scaffold("not_a_molecule") is None


class TestGenericScaffold:
    def test_returns_string(self):
        result = get_generic_scaffold("c1ccccc1")
        assert isinstance(result, str)

    def test_invalid_smiles(self):
        assert get_generic_scaffold("???") is None


class TestComputeScaffolds:
    def test_columns_added(self, enriched_df):
        assert "murcko_scaffold" in enriched_df.columns
        assert "generic_scaffold" in enriched_df.columns

    def test_invalid_molecules_get_none(self, enriched_df):
        invalid = enriched_df[~enriched_df[COL_VALID]]
        assert invalid["murcko_scaffold"].isna().all()


class TestScaffoldFrequency:
    def test_returns_expected_columns(self, enriched_df):
        freq = scaffold_frequency(enriched_df)
        if not freq.empty:
            assert "Scaffold" in freq.columns
            assert "Count" in freq.columns
            assert "Fraction" in freq.columns

    def test_fractions_sum_to_one(self, enriched_df):
        freq = scaffold_frequency(enriched_df, top_n=100)
        if not freq.empty:
            assert abs(freq["Fraction"].sum() - 1.0) < 0.01


class TestScaffoldDiversity:
    def test_returns_expected_keys(self, enriched_df):
        stats = scaffold_diversity_stats(enriched_df)
        assert "unique_scaffolds" in stats
        assert "scaffold_ratio" in stats
        assert "singleton_fraction" in stats

    def test_ratio_in_range(self, enriched_df):
        stats = scaffold_diversity_stats(enriched_df)
        assert 0.0 <= stats["scaffold_ratio"] <= 1.0
