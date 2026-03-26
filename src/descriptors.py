"""
Molecular descriptor computation and rule-based chemistry flags.

Computes 12 descriptors per molecule and 4 boolean rule flags.
All descriptor logic is isolated here — the UI never calls RDKit directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit.Chem import Descriptors, rdMolDescriptors, QED as QEDModule

from .config import (
    COL_VALID,
    DESCRIPTOR_COLS,
    LIPINSKI_MW_MAX,
    LIPINSKI_LOGP_MAX,
    LIPINSKI_HBD_MAX,
    LIPINSKI_HBA_MAX,
    LIPINSKI_ROTBOND_MAX,
    HIGH_FLEXIBILITY_ROTBOND,
    HIGH_LIPOPHILICITY_LOGP,
    EXTREME_SIZE_MW_LOW,
    EXTREME_SIZE_MW_HIGH,
)


# ---------------------------------------------------------------------------
# Per-molecule descriptor calculation
# ---------------------------------------------------------------------------

def _compute_one(mol) -> dict:
    """Compute all descriptors for a single RDKit Mol object."""
    if mol is None:
        return {col: (None if col == "MolFormula" else np.nan) for col in DESCRIPTOR_COLS}
    return {
        "MolWt": Descriptors.ExactMolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "HBD": rdMolDescriptors.CalcNumHBD(mol),
        "HBA": rdMolDescriptors.CalcNumHBA(mol),
        "RotatableBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "RingCount": rdMolDescriptors.CalcNumRings(mol),
        "FractionCsp3": rdMolDescriptors.CalcFractionCSP3(mol),
        "HeavyAtomCount": mol.GetNumHeavyAtoms(),
        "FormalCharge": sum(a.GetFormalCharge() for a in mol.GetAtoms()),
        "QED": QEDModule.qed(mol),
        "MolFormula": rdMolDescriptors.CalcMolFormula(mol),
    }


def compute_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add descriptor columns to *df*.

    Expects a ``mol`` column containing RDKit Mol objects (None for invalid).
    Invalid molecules receive NaN / None for all descriptors.
    """
    df = df.copy()
    desc_records = df["mol"].apply(_compute_one).tolist()
    desc_df = pd.DataFrame(desc_records, index=df.index)
    for col in DESCRIPTOR_COLS:
        df[col] = desc_df[col]
    return df


# ---------------------------------------------------------------------------
# Rule-based flags
# ---------------------------------------------------------------------------

def apply_rule_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add boolean flag columns based on simple, transparent rule thresholds.

    Flags (all True = flagged):
    - passes_lipinski_like: MW ≤ 500, LogP ≤ 5, HBD ≤ 5, HBA ≤ 10, RotBond ≤ 10
    - high_flexibility_flag: RotatableBonds > 10
    - high_lipophilicity_flag: LogP > 5
    - extreme_size_flag: MW < 150 or MW > 800
    """
    df = df.copy()

    valid = df[COL_VALID]

    df["passes_lipinski_like"] = (
        valid
        & (df["MolWt"] <= LIPINSKI_MW_MAX)
        & (df["LogP"] <= LIPINSKI_LOGP_MAX)
        & (df["HBD"] <= LIPINSKI_HBD_MAX)
        & (df["HBA"] <= LIPINSKI_HBA_MAX)
        & (df["RotatableBonds"] <= LIPINSKI_ROTBOND_MAX)
    )

    df["high_flexibility_flag"] = valid & (df["RotatableBonds"] > HIGH_FLEXIBILITY_ROTBOND)
    df["high_lipophilicity_flag"] = valid & (df["LogP"] > HIGH_LIPOPHILICITY_LOGP)
    df["extreme_size_flag"] = valid & (
        (df["MolWt"] < EXTREME_SIZE_MW_LOW) | (df["MolWt"] > EXTREME_SIZE_MW_HIGH)
    )

    return df


# ---------------------------------------------------------------------------
# Summary statistics table
# ---------------------------------------------------------------------------

def compute_descriptor_summary(
    df: pd.DataFrame,
    descriptor_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute summary statistics for each numeric descriptor.

    Returns a DataFrame with one row per descriptor and columns:
    Descriptor, n, Mean, Std, Median, Min, Max, Q1, Q3.
    Only includes valid molecules.
    """
    if descriptor_cols is None:
        from .config import NUMERIC_DESCRIPTOR_COLS
        descriptor_cols = NUMERIC_DESCRIPTOR_COLS

    valid = df[df[COL_VALID]] if COL_VALID in df.columns else df
    cols_present = [c for c in descriptor_cols if c in valid.columns]

    rows = []
    for col in cols_present:
        s = valid[col].dropna()
        if s.empty:
            continue
        rows.append({
            "Descriptor": col,
            "n": len(s),
            "Mean": round(float(s.mean()), 3),
            "Std": round(float(s.std()), 3),
            "Median": round(float(s.median()), 3),
            "Min": round(float(s.min()), 3),
            "Max": round(float(s.max()), 3),
            "Q1": round(float(s.quantile(0.25)), 3),
            "Q3": round(float(s.quantile(0.75)), 3),
        })
    return pd.DataFrame(rows)
