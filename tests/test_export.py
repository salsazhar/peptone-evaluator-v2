"""Tests for SDF export."""

from src.chemistry import parse_smiles_column
from src.descriptors import compute_descriptors
from src.export import dataframe_to_sdf, sdf_bytes

import pandas as pd


class TestSdfExport:
    def test_sdf_contains_molecules(self):
        df = parse_smiles_column(pd.DataFrame({"SMILES": ["CCO", "c1ccccc1"]}))
        df = compute_descriptors(df)
        sdf = dataframe_to_sdf(df)
        # SDF records are separated by "$$$$"
        assert sdf.count("$$$$") == 2

    def test_sdf_skips_invalid(self):
        df = parse_smiles_column(pd.DataFrame({"SMILES": ["CCO", "bad"]}))
        df = compute_descriptors(df)
        sdf = dataframe_to_sdf(df, valid_only=True)
        assert sdf.count("$$$$") == 1

    def test_sdf_bytes_returns_bytes(self):
        df = parse_smiles_column(pd.DataFrame({"SMILES": ["CCO"]}))
        df = compute_descriptors(df)
        result = sdf_bytes(df)
        assert isinstance(result, bytes)
        assert b"$$$$" in result

    def test_all_invalid_produces_empty_sdf(self):
        df = parse_smiles_column(pd.DataFrame({"SMILES": ["bad", "worse"]}))
        df = compute_descriptors(df)
        sdf = dataframe_to_sdf(df, valid_only=True)
        assert isinstance(sdf, str)
        assert sdf.count("$$$$") == 0
