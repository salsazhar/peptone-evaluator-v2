"""
Shared fixtures for Peptone Evaluator tests.

Provides small, deterministic molecule sets that exercise common edge cases:
valid SMILES, invalid SMILES, duplicates, Lipinski pass/fail, and single-molecule
datasets. Every test module imports from here — no fixture duplication.
"""

import pandas as pd
import pytest

from src.chemistry import parse_smiles_column
from src.descriptors import compute_descriptors, apply_rule_flags
from src.fingerprints import compute_morgan_fingerprints
from src.scaffolds import compute_scaffolds


# ---------------------------------------------------------------------------
# Raw SMILES fixtures
# ---------------------------------------------------------------------------

VALID_SMILES = [
    "CCO",                        # ethanol — tiny, passes Lipinski easily
    "c1ccccc1",                   # benzene
    "CC(=O)Oc1ccccc1C(=O)O",     # aspirin
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # ibuprofen
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",  # testosterone — steroid scaffold
]

INVALID_SMILES = ["not_a_molecule", "???"]

DUPLICATE_SMILES = ["CCO", "CCO", "c1ccccc1"]


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """DataFrame with a mix of valid, invalid, and duplicate SMILES + pIC50."""
    return pd.DataFrame({
        "SMILES": VALID_SMILES + INVALID_SMILES + DUPLICATE_SMILES,
        "pIC50": [6.5, 4.0, 7.2, 6.8, 5.5, None, None, 6.5, 6.5, 4.0],
    })


@pytest.fixture
def parsed_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """raw_df after SMILES parsing (adds mol, canonical_SMILES, is_valid)."""
    return parse_smiles_column(raw_df)


@pytest.fixture
def enriched_df(parsed_df: pd.DataFrame) -> pd.DataFrame:
    """parsed_df + descriptors + rule flags + scaffolds."""
    df = compute_descriptors(parsed_df)
    df = apply_rule_flags(df)
    df = compute_scaffolds(df)
    return df


@pytest.fixture
def fp_matrix(enriched_df: pd.DataFrame):
    """Morgan fingerprint matrix for the enriched dataset."""
    return compute_morgan_fingerprints(enriched_df)


@pytest.fixture
def empty_df() -> pd.DataFrame:
    """Empty DataFrame with SMILES column."""
    return pd.DataFrame({"SMILES": pd.Series([], dtype=str)})


@pytest.fixture
def single_mol_df() -> pd.DataFrame:
    """Single valid molecule."""
    return parse_smiles_column(pd.DataFrame({"SMILES": ["CCO"]}))
