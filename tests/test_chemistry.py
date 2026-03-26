"""Tests for SMILES parsing, canonicalisation, and uniqueness detection."""

import pandas as pd

from src.chemistry import parse_smiles_column, get_unique_mask
from src.config import COL_CANONICAL, COL_VALID


class TestParseSmiles:
    def test_valid_smiles_parsed(self, parsed_df):
        valid = parsed_df[parsed_df[COL_VALID]]
        assert len(valid) == 8  # 5 unique valid + 3 duplicate valid

    def test_invalid_smiles_flagged(self, parsed_df):
        invalid = parsed_df[~parsed_df[COL_VALID]]
        assert len(invalid) == 2

    def test_canonical_smiles_added(self, parsed_df):
        assert COL_CANONICAL in parsed_df.columns
        valid = parsed_df[parsed_df[COL_VALID]]
        assert valid[COL_CANONICAL].notna().all()

    def test_invalid_smiles_get_none_canonical(self, parsed_df):
        invalid = parsed_df[~parsed_df[COL_VALID]]
        assert invalid[COL_CANONICAL].isna().all()

    def test_empty_dataframe(self, empty_df):
        result = parse_smiles_column(empty_df)
        assert len(result) == 0
        assert COL_VALID in result.columns

    def test_all_invalid(self):
        df = pd.DataFrame({"SMILES": ["bad", "worse", "also_bad"]})
        result = parse_smiles_column(df)
        assert result[COL_VALID].sum() == 0

    def test_preserves_other_columns(self):
        df = pd.DataFrame({"SMILES": ["CCO"], "pIC50": [6.5], "extra": ["x"]})
        result = parse_smiles_column(df)
        assert "extra" in result.columns
        assert result["extra"].iloc[0] == "x"


class TestUniqueMask:
    def test_duplicates_detected(self, parsed_df):
        mask = get_unique_mask(parsed_df)
        # VALID_SMILES has 5 unique, DUPLICATE_SMILES has 3 (all dupes of existing)
        # unique count should equal number of distinct canonical SMILES
        n_unique = parsed_df.loc[parsed_df[COL_VALID], COL_CANONICAL].nunique()
        assert mask.sum() == n_unique

    def test_invalid_molecules_not_unique(self, parsed_df):
        mask = get_unique_mask(parsed_df)
        invalid_idx = parsed_df[~parsed_df[COL_VALID]].index
        assert not mask[invalid_idx].any()

    def test_single_molecule_is_unique(self, single_mol_df):
        mask = get_unique_mask(single_mol_df)
        assert mask.sum() == 1
