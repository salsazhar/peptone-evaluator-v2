"""
Substructure search and highlighting.

Supports SMARTS-based substructure queries against a molecular set.
All RDKit calls are encapsulated here.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from rdkit import Chem

from .config import COL_SMILES, COL_CANONICAL, COL_VALID


# ---------------------------------------------------------------------------
# SMARTS parsing
# ---------------------------------------------------------------------------

def parse_smarts(smarts: str) -> Chem.Mol | None:
    """Parse a SMARTS pattern. Returns None on failure."""
    if not smarts or not smarts.strip():
        return None
    return Chem.MolFromSmarts(smarts.strip())


def parse_substructure_query(query: str) -> Chem.Mol | None:
    """
    Accept either SMARTS or SMILES as a substructure query.

    Tries SMARTS first, falls back to SMILES if that fails.
    Returns None if both fail.
    """
    if not query or not query.strip():
        return None
    query = query.strip()
    mol = Chem.MolFromSmarts(query)
    if mol is not None:
        return mol
    mol = Chem.MolFromSmiles(query)
    return mol


# ---------------------------------------------------------------------------
# Substructure matching
# ---------------------------------------------------------------------------

def substructure_search(
    df: pd.DataFrame,
    query_mol: Chem.Mol,
) -> pd.Series:
    """
    Return a boolean Series indicating which rows contain the substructure.

    Expects df to have a COL_CANONICAL column and COL_VALID column.
    Invalid molecules are always False.
    """
    mask = pd.Series(False, index=df.index)

    for idx, row in df.iterrows():
        if not row.get(COL_VALID, False):
            continue
        smiles = row.get(COL_CANONICAL) or row.get(COL_SMILES)
        if not smiles:
            continue
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is not None and mol.HasSubstructMatch(query_mol):
            mask.at[idx] = True

    return mask


def batch_substructure_search(
    df: pd.DataFrame,
    query_mol: Chem.Mol,
    smiles_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split df into matches and non-matches.

    Returns (matches_df, non_matches_df).
    """
    col = smiles_col or COL_CANONICAL
    mask = substructure_search(df, query_mol)
    return df[mask].copy(), df[~mask].copy()


# ---------------------------------------------------------------------------
# Highlight atoms (SVG generation for display)
# ---------------------------------------------------------------------------

def highlight_substructure_svg(
    smiles: str,
    query_mol: Chem.Mol,
    size: tuple[int, int] = (350, 250),
) -> str | None:
    """
    Generate SVG string of the molecule with substructure atoms highlighted.

    Returns None if the molecule is invalid or doesn't contain the substructure.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    match = mol.GetSubstructMatch(query_mol)
    if not match:
        return None

    try:
        from rdkit.Chem import Draw
        drawer = Draw.MolDraw2DSVG(*size)
    except ImportError:
        return None

    drawer.DrawMolecule(
        mol,
        highlightAtoms=list(match),
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


# ---------------------------------------------------------------------------
# Common SMARTS patterns (convenience presets)
# ---------------------------------------------------------------------------

COMMON_SUBSTRUCTURES: dict[str, str] = {
    "Benzene ring": "c1ccccc1",
    "Phenol": "c1ccc(O)cc1",
    "Amine (primary)": "[NH2]",
    "Amine (secondary)": "[NH1]([#6])[#6]",
    "Amide": "[NX3][CX3](=[OX1])",
    "Carboxylic acid": "[CX3](=O)[OX2H1]",
    "Ester": "[CX3](=O)[OX2][#6]",
    "Sulfonamide": "[SX4](=[OX1])(=[OX1])([NX3])",
    "Halide (F/Cl/Br)": "[F,Cl,Br]",
    "Nitro group": "[NX3+](=O)[O-]",
    "Pyridine": "c1ccncc1",
    "Piperidine": "C1CCNCC1",
    "Piperazine": "C1CNCCN1",
    "Indole": "c1ccc2[nH]ccc2c1",
    "Imidazole": "c1cnc[nH]1",
}
