"""
Morgan fingerprint generation.

Produces a dense numpy bit-vector matrix from parsed molecules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit.Chem import AllChem

from .config import COL_VALID, FP_RADIUS, FP_NBITS


def compute_morgan_fingerprints(
    df: pd.DataFrame,
    radius: int = FP_RADIUS,
    n_bits: int = FP_NBITS,
) -> np.ndarray | None:
    """
    Compute Morgan fingerprints for **valid** molecules only.

    Parameters
    ----------
    df : DataFrame with ``mol`` and ``is_valid`` columns.
    radius : Morgan radius (default 2 → ECFP4 equivalent).
    n_bits : Fingerprint length in bits.

    Returns
    -------
    np.ndarray of shape ``(n_valid, n_bits)`` or ``None`` if no valid molecules.
    Row order matches ``df[df['is_valid']]``.
    """
    valid_mols = df.loc[df[COL_VALID], "mol"]
    if valid_mols.empty:
        return None

    fps = []
    for mol in valid_mols:
        bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        fps.append(np.array(bv))

    return np.vstack(fps)
