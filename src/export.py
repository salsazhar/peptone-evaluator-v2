"""
Export utilities — SDF and CSV writers.

Generates SDF (Structure-Data File) output from a processed DataFrame,
embedding molecular properties as SDF data fields. SDF is the standard
interchange format for chemistry tools (PyMOL, MOE, Maestro, etc.).
"""

from __future__ import annotations

import io
from typing import Sequence

from rdkit import Chem
from rdkit.Chem import AllChem

from .config import COL_SMILES, COL_CANONICAL, COL_VALID, NUMERIC_DESCRIPTOR_COLS


# ---------------------------------------------------------------------------
# SDF generation
# ---------------------------------------------------------------------------

# Columns to embed as SDF data fields (order matters for readability)
_SDF_PROPERTY_COLS: list[str] = [
    COL_SMILES,
    COL_CANONICAL,
    "pIC50",
    *NUMERIC_DESCRIPTOR_COLS,
    "MolFormula",
    "passes_lipinski_like",
    "murcko_scaffold",
    "priority_score",
]


def dataframe_to_sdf(
    df: "pd.DataFrame",
    property_cols: Sequence[str] | None = None,
    valid_only: bool = True,
    add_2d_coords: bool = True,
) -> str:
    """
    Convert a processed DataFrame to an SDF string.

    Parameters
    ----------
    df : DataFrame with at least COL_CANONICAL and COL_VALID columns.
    property_cols : columns to write as SDF data fields.
        Defaults to _SDF_PROPERTY_COLS (skipping missing columns).
    valid_only : if True, skip invalid molecules.
    add_2d_coords : if True, compute 2-D coordinates for depiction.

    Returns
    -------
    SDF-formatted string ready for file download.
    """
    import pandas as pd  # deferred to avoid circular import at module level

    if property_cols is None:
        property_cols = [c for c in _SDF_PROPERTY_COLS if c in df.columns]
    else:
        property_cols = [c for c in property_cols if c in df.columns]

    buf = io.StringIO()
    writer = Chem.SDWriter(buf)

    rows = df[df[COL_VALID]] if valid_only and COL_VALID in df.columns else df

    for _, row in rows.iterrows():
        smiles = row.get(COL_CANONICAL) or row.get(COL_SMILES)
        if not smiles:
            continue

        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            continue

        # Generate 2-D coordinates for structure depiction
        if add_2d_coords:
            AllChem.Compute2DCoords(mol)

        # Set molecule name to canonical SMILES (shown in SDF header)
        mol.SetProp("_Name", str(smiles))

        # Attach properties as SDF data fields
        for col in property_cols:
            val = row.get(col)
            if val is not None and not (isinstance(val, float) and pd.isna(val)):
                mol.SetProp(col, str(val))

        writer.write(mol)

    writer.close()
    return buf.getvalue()


def sdf_bytes(
    df: "pd.DataFrame",
    property_cols: Sequence[str] | None = None,
    valid_only: bool = True,
) -> bytes:
    """Return SDF content as UTF-8 bytes (for Streamlit download button)."""
    return dataframe_to_sdf(
        df, property_cols=property_cols, valid_only=valid_only,
    ).encode("utf-8")
