"""Tests for substructure search and highlighting."""

import pandas as pd

from src.substructure import (
    parse_smarts,
    parse_substructure_query,
    substructure_search,
    COMMON_SUBSTRUCTURES,
)
from src.chemistry import parse_smiles_column


class TestParseSmarts:
    def test_valid_smarts(self):
        mol = parse_smarts("[#6]")
        assert mol is not None

    def test_empty_string(self):
        assert parse_smarts("") is None

    def test_whitespace(self):
        assert parse_smarts("   ") is None

    def test_invalid_smarts(self):
        assert parse_smarts("not_smarts_at_all$$$") is None


class TestParseSubstructureQuery:
    def test_smarts_input(self):
        mol = parse_substructure_query("c1ccccc1")
        assert mol is not None

    def test_smiles_fallback(self):
        mol = parse_substructure_query("CCO")
        assert mol is not None

    def test_invalid_input(self):
        assert parse_substructure_query("$$$") is None


class TestSubstructureSearch:
    def test_benzene_found_in_aspirin(self):
        df = parse_smiles_column(pd.DataFrame({
            "SMILES": ["CC(=O)Oc1ccccc1C(=O)O", "CCO"],
        }))
        query = parse_substructure_query("c1ccccc1")
        matches = substructure_search(df, query)
        assert matches.iloc[0] == True   # aspirin contains benzene
        assert matches.iloc[1] == False  # ethanol does not

    def test_no_matches(self):
        df = parse_smiles_column(pd.DataFrame({"SMILES": ["CCO", "CC"]}))
        query = parse_substructure_query("c1ccccc1")
        matches = substructure_search(df, query)
        assert matches.sum() == 0

    def test_all_presets_are_valid(self):
        """Every preset pattern in COMMON_SUBSTRUCTURES should parse."""
        for name, smarts in COMMON_SUBSTRUCTURES.items():
            mol = parse_substructure_query(smarts)
            assert mol is not None, f"Preset '{name}' failed to parse: {smarts}"
