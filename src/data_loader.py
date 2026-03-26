"""
CSV loading and column normalisation.

Responsibilities:
- Read an uploaded file into a DataFrame
- Case-insensitive detection and renaming of SMILES / pIC50 columns
- Coerce pIC50 to numeric (NaN rows are kept, not dropped)
"""

from __future__ import annotations

import pandas as pd

from .config import COL_SMILES, COL_PIC50


def load_csv(uploaded_file) -> pd.DataFrame:
    """Read a Streamlit UploadedFile (or file-like) into a DataFrame."""
    return pd.read_csv(uploaded_file)


def normalize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """
    Detect SMILES and pIC50 columns case-insensitively and rename them
    to the canonical names defined in config.

    Returns
    -------
    (normalised_df, has_pic50)
        has_pic50 is True when a pIC50-like column was found.

    Raises
    ------
    ValueError
        If no SMILES-like column is found.
    """
    df = df.copy()
    col_lower = {c.lower(): c for c in df.columns}

    # --- SMILES ---
    if "smiles" not in col_lower:
        raise ValueError(
            f"No SMILES column found. Columns present: {list(df.columns)}"
        )
    df = df.rename(columns={col_lower["smiles"]: COL_SMILES})

    # --- pIC50 (optional) ---
    has_pic50 = "pic50" in col_lower
    if has_pic50:
        df = df.rename(columns={col_lower["pic50"]: COL_PIC50})
        df[COL_PIC50] = pd.to_numeric(df[COL_PIC50], errors="coerce")

    return df, has_pic50
