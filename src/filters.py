"""
DataFrame filtering logic.

Pure functions with no Streamlit dependency — testable in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .config import (
    COL_VALID,
    COL_CANONICAL,
    COL_PIC50,
    DEFAULT_MW_RANGE,
    DEFAULT_LOGP_RANGE,
    DEFAULT_TPSA_RANGE,
    DEFAULT_HBD_MAX,
    DEFAULT_HBA_MAX,
    DEFAULT_ROTBOND_MAX,
)


@dataclass
class FilterSpec:
    """Serialisable container for every user-configurable filter."""

    mw_range: tuple[float, float] = DEFAULT_MW_RANGE
    logp_range: tuple[float, float] = DEFAULT_LOGP_RANGE
    tpsa_range: tuple[float, float] = DEFAULT_TPSA_RANGE
    hbd_max: int = DEFAULT_HBD_MAX
    hba_max: int = DEFAULT_HBA_MAX
    rotbond_max: int = DEFAULT_ROTBOND_MAX
    pic50_range: tuple[float, float] | None = None  # None when column absent
    valid_only: bool = True
    unique_only: bool = False
    lipinski_only: bool = False


def apply_filters(df: pd.DataFrame, spec: FilterSpec) -> pd.DataFrame:
    """
    Return a filtered copy of *df* according to *spec*.

    Never modifies the input DataFrame.
    """
    mask = pd.Series(True, index=df.index)

    if spec.valid_only:
        mask &= df[COL_VALID]

    if spec.unique_only and COL_CANONICAL in df.columns:
        mask &= ~df.duplicated(subset=[COL_CANONICAL], keep="first") | ~df[COL_VALID]

    if spec.lipinski_only and "passes_lipinski_like" in df.columns:
        mask &= df["passes_lipinski_like"]

    # Numeric range filters (only applied to valid molecules)
    if "MolWt" in df.columns:
        mask &= df["MolWt"].between(*spec.mw_range) | ~df[COL_VALID]
    if "LogP" in df.columns:
        mask &= df["LogP"].between(*spec.logp_range) | ~df[COL_VALID]
    if "TPSA" in df.columns:
        mask &= df["TPSA"].between(*spec.tpsa_range) | ~df[COL_VALID]
    if "HBD" in df.columns:
        mask &= (df["HBD"] <= spec.hbd_max) | ~df[COL_VALID]
    if "HBA" in df.columns:
        mask &= (df["HBA"] <= spec.hba_max) | ~df[COL_VALID]
    if "RotatableBonds" in df.columns:
        mask &= (df["RotatableBonds"] <= spec.rotbond_max) | ~df[COL_VALID]

    if spec.pic50_range is not None and COL_PIC50 in df.columns:
        mask &= df[COL_PIC50].between(*spec.pic50_range) | df[COL_PIC50].isna()

    return df.loc[mask].reset_index(drop=True)
