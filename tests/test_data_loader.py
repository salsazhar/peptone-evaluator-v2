"""Tests for CSV loading and column normalisation."""

import io
import pandas as pd
import pytest

from src.data_loader import load_csv, normalize_columns


class TestNormalizeColumns:
    def test_detects_smiles_case_insensitive(self):
        df = pd.DataFrame({"smiles": ["CCO"], "pic50": [6.5]})
        result, has_pic50 = normalize_columns(df)
        assert "SMILES" in result.columns
        assert has_pic50 is True
        assert "pIC50" in result.columns

    def test_raises_on_missing_smiles(self):
        df = pd.DataFrame({"activity": [6.5]})
        with pytest.raises(ValueError, match="(?i)smiles"):
            normalize_columns(df)

    def test_pic50_coerced_to_numeric(self):
        df = pd.DataFrame({"SMILES": ["CCO", "c1ccccc1"], "pIC50": ["6.5", "bad"]})
        result, has_pic50 = normalize_columns(df)
        assert has_pic50 is True
        assert pd.notna(result["pIC50"].iloc[0])
        assert pd.isna(result["pIC50"].iloc[1])

    def test_no_pic50_column(self):
        df = pd.DataFrame({"SMILES": ["CCO"]})
        result, has_pic50 = normalize_columns(df)
        assert has_pic50 is False


class TestLoadCsv:
    def test_loads_from_file_like(self):
        csv_text = "SMILES,pIC50\nCCO,6.5\nc1ccccc1,4.0\n"
        buf = io.StringIO(csv_text)
        df = load_csv(buf)
        assert len(df) == 2
        assert "SMILES" in df.columns
