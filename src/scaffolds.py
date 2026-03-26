"""
Scaffold analysis via Murcko decomposition.

Decomposes molecules into their Murcko scaffolds and generic frameworks,
enabling analysis of scaffold diversity, frequency, and clustering.
"""

from __future__ import annotations

from collections import Counter

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from .config import COL_SMILES, COL_CANONICAL, COL_VALID


# ---------------------------------------------------------------------------
# Single-molecule scaffold extraction
# ---------------------------------------------------------------------------

def get_murcko_scaffold(smiles: str) -> str | None:
    """Return the Murcko scaffold SMILES for a given molecule."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(core)
    except Exception:
        return None


def get_generic_scaffold(smiles: str) -> str | None:
    """
    Return the generic Murcko framework SMILES.

    Generic = all atoms converted to carbon, all bonds to single.
    This groups structurally similar scaffolds more aggressively.
    """
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        generic = MurckoScaffold.MakeScaffoldGeneric(core)
        return Chem.MolToSmiles(generic)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Batch scaffold computation
# ---------------------------------------------------------------------------

def compute_scaffolds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add scaffold columns to the dataframe.

    Columns added:
    - murcko_scaffold: Murcko scaffold SMILES
    - generic_scaffold: Generic Murcko framework SMILES

    Operates on valid molecules only (uses COL_CANONICAL).
    """
    df = df.copy()
    smiles_source = COL_CANONICAL if COL_CANONICAL in df.columns else COL_SMILES

    df["murcko_scaffold"] = None
    df["generic_scaffold"] = None

    valid_mask = df[COL_VALID] if COL_VALID in df.columns else pd.Series(True, index=df.index)

    for idx in df[valid_mask].index:
        smi = df.at[idx, smiles_source]
        if smi:
            df.at[idx, "murcko_scaffold"] = get_murcko_scaffold(smi)
            df.at[idx, "generic_scaffold"] = get_generic_scaffold(smi)

    return df


# ---------------------------------------------------------------------------
# Scaffold statistics
# ---------------------------------------------------------------------------

def scaffold_frequency(
    df: pd.DataFrame,
    scaffold_col: str = "murcko_scaffold",
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Return a frequency table of the top N most common scaffolds.

    Columns: Scaffold, Count, Fraction, Example_SMILES.
    """
    valid = df[df[COL_VALID]] if COL_VALID in df.columns else df
    scaffolds = valid[scaffold_col].dropna()

    if scaffolds.empty:
        return pd.DataFrame(columns=["Scaffold", "Count", "Fraction", "Example_SMILES"])

    counts = Counter(scaffolds)
    total = len(scaffolds)

    rows = []
    smiles_col = COL_CANONICAL if COL_CANONICAL in valid.columns else COL_SMILES

    for scaffold, count in counts.most_common(top_n):
        example = valid.loc[valid[scaffold_col] == scaffold, smiles_col].iloc[0]
        rows.append({
            "Scaffold": scaffold,
            "Count": count,
            "Fraction": round(count / total, 4),
            "Example_SMILES": example,
        })

    return pd.DataFrame(rows)


def scaffold_diversity_stats(
    df: pd.DataFrame,
    scaffold_col: str = "murcko_scaffold",
) -> dict:
    """
    Compute scaffold diversity metrics.

    Returns dict with:
    - total_molecules: valid molecule count
    - unique_scaffolds: number of distinct scaffolds
    - scaffold_ratio: unique_scaffolds / total_molecules
    - singleton_scaffolds: scaffolds appearing exactly once
    - singleton_fraction: fraction of scaffolds that are singletons
    - top_scaffold: most frequent scaffold SMILES
    - top_scaffold_count: count of most frequent scaffold
    - top_scaffold_fraction: fraction of molecules in top scaffold
    """
    valid = df[df[COL_VALID]] if COL_VALID in df.columns else df
    scaffolds = valid[scaffold_col].dropna()

    if scaffolds.empty:
        return {
            "total_molecules": 0,
            "unique_scaffolds": 0,
            "scaffold_ratio": 0.0,
            "singleton_scaffolds": 0,
            "singleton_fraction": 0.0,
            "top_scaffold": None,
            "top_scaffold_count": 0,
            "top_scaffold_fraction": 0.0,
        }

    total = len(scaffolds)
    counts = Counter(scaffolds)
    unique = len(counts)
    singletons = sum(1 for c in counts.values() if c == 1)
    top_scaffold, top_count = counts.most_common(1)[0]

    return {
        "total_molecules": total,
        "unique_scaffolds": unique,
        "scaffold_ratio": round(unique / total, 4) if total else 0.0,
        "singleton_scaffolds": singletons,
        "singleton_fraction": round(singletons / unique, 4) if unique else 0.0,
        "top_scaffold": top_scaffold,
        "top_scaffold_count": top_count,
        "top_scaffold_fraction": round(top_count / total, 4) if total else 0.0,
    }
