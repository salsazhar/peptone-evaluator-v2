"""
Campaign comparison — current design round vs. reference set.

Computes overlap, novelty, and side-by-side descriptor statistics
to help evaluate whether a generative round is adding value.
No Streamlit dependency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import COL_CANONICAL, COL_VALID, NUMERIC_DESCRIPTOR_COLS


# ---------------------------------------------------------------------------
# Overlap detection
# ---------------------------------------------------------------------------

def compute_campaign_overlap(
    current_df: pd.DataFrame,
    reference_df: pd.DataFrame,
) -> dict:
    """
    Compare current round against reference set by canonical SMILES.

    Returns
    -------
    dict with:
        novel_count, overlap_count, novel_mask (bool Series aligned to
        current_df), overlap_mask (bool Series aligned to current_df).
    """
    current_valid = current_df[current_df[COL_VALID]]
    ref_valid = reference_df[reference_df[COL_VALID]]

    ref_smiles = set(ref_valid[COL_CANONICAL].dropna())
    current_canonical = current_valid[COL_CANONICAL]

    overlap_mask = pd.Series(False, index=current_df.index)
    overlap_mask.loc[current_valid.index] = current_canonical.isin(ref_smiles)

    novel_mask = pd.Series(False, index=current_df.index)
    novel_mask.loc[current_valid.index] = ~current_canonical.isin(ref_smiles)

    return {
        "novel_count": int(novel_mask.sum()),
        "overlap_count": int(overlap_mask.sum()),
        "novel_mask": novel_mask,
        "overlap_mask": overlap_mask,
        "ref_unique_count": len(ref_smiles),
    }


# ---------------------------------------------------------------------------
# Descriptor comparison
# ---------------------------------------------------------------------------

def compare_descriptor_stats(
    current_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    descriptor_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Side-by-side comparison of descriptor statistics.

    Returns DataFrame with columns:
        Descriptor, Current Mean, Current Median, Reference Mean,
        Reference Median, Delta Mean.
    """
    if descriptor_cols is None:
        descriptor_cols = NUMERIC_DESCRIPTOR_COLS

    cols_present = [c for c in descriptor_cols if c in current_df.columns and c in reference_df.columns]
    cur_valid = current_df[current_df[COL_VALID]]
    ref_valid = reference_df[reference_df[COL_VALID]]

    rows = []
    for col in cols_present:
        c_mean = cur_valid[col].mean()
        c_med = cur_valid[col].median()
        r_mean = ref_valid[col].mean()
        r_med = ref_valid[col].median()
        rows.append({
            "Descriptor": col,
            "Current Mean": round(c_mean, 3) if pd.notna(c_mean) else None,
            "Current Median": round(c_med, 3) if pd.notna(c_med) else None,
            "Reference Mean": round(r_mean, 3) if pd.notna(r_mean) else None,
            "Reference Median": round(r_med, 3) if pd.notna(r_med) else None,
            "Delta Mean": round(c_mean - r_mean, 3) if pd.notna(c_mean) and pd.notna(r_mean) else None,
        })
    return pd.DataFrame(rows)
