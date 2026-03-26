"""
SMILES parsing, canonicalisation, and validity tracking.

Every call to Chem.MolFromSmiles goes through this module so that
RDKit logging is suppressed in one place.
"""

from __future__ import annotations

import pandas as pd
from rdkit import Chem, RDLogger

from .config import COL_SMILES, COL_CANONICAL, COL_VALID

# Suppress RDKit warnings globally when this module is imported
RDLogger.DisableLog("rdApp.*")


def _parse_one(smiles: str) -> tuple[Chem.Mol | None, str | None]:
    """Return (mol, canonical_smiles) or (None, None) for invalid SMILES."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None, None
    return mol, Chem.MolToSmiles(mol)


def parse_smiles_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add three columns to *df*:

    - ``mol``              – RDKit Mol object (None for invalid SMILES)
    - ``canonical_SMILES`` – canonical SMILES string (None for invalid)
    - ``is_valid``         – boolean flag

    Invalid rows are flagged, **not** dropped.
    """
    df = df.copy()
    parsed = df[COL_SMILES].apply(lambda s: _parse_one(s))
    df["mol"] = parsed.apply(lambda t: t[0])
    df[COL_CANONICAL] = parsed.apply(lambda t: t[1])
    df[COL_VALID] = df["mol"].notnull()
    return df


def get_unique_mask(df: pd.DataFrame) -> pd.Series:
    """
    Boolean Series that is True for the first occurrence of each
    canonical SMILES among valid molecules.
    """
    mask = pd.Series(False, index=df.index)
    valid = df[df[COL_VALID]]
    first_idx = valid.drop_duplicates(subset=[COL_CANONICAL]).index
    mask.loc[first_idx] = True
    return mask
