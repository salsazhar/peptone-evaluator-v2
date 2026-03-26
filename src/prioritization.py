"""
Prioritisation scoring, shortlisting, and diverse-representative selection.

Provides a transparent, rank-based composite score so scientists can
quickly identify molecules worth inspecting. No Streamlit dependency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    COL_VALID,
    COL_PIC50,
    COL_CANONICAL,
    COL_SMILES,
    PRIORITY_WEIGHTS,
    PRIORITY_WEIGHTS_NO_PIC50,
    SHORTLIST_DEFAULT,
    DIVERSE_REPS_DEFAULT,
    NUMERIC_DESCRIPTOR_COLS,
)


# ---------------------------------------------------------------------------
# Composite priority score
# ---------------------------------------------------------------------------

def compute_priority_scores(
    df: pd.DataFrame,
    nn_similarities: np.ndarray | None = None,
    valid_indices: np.ndarray | None = None,
    has_pic50: bool = False,
) -> pd.DataFrame:
    """
    Add a ``priority_score`` column to *df* for valid molecules.

    Scoring components (all rank-normalised to [0, 1]):
      - pIC50   — higher is better  (weight 0.3, or redistributed if absent)
      - QED     — higher is better  (weight 0.3)
      - diversity = 1 − nn_similarity  (weight 0.2)
      - passes_lipinski_like — binary  (weight 0.2)

    Invalid molecules get NaN for priority_score.
    """
    df = df.copy()
    weights = PRIORITY_WEIGHTS if has_pic50 else PRIORITY_WEIGHTS_NO_PIC50

    valid = df[COL_VALID]
    score = pd.Series(0.0, index=df.index)

    # --- pIC50 component ---
    if has_pic50 and COL_PIC50 in df.columns and "pic50" in weights:
        pic50_rank = df.loc[valid, COL_PIC50].rank(pct=True, na_option="bottom")
        score.loc[valid] += weights["pic50"] * pic50_rank.reindex(df.index, fill_value=0.0)

    # --- QED component ---
    if "QED" in df.columns and "qed" in weights:
        qed_rank = df.loc[valid, "QED"].rank(pct=True, na_option="bottom")
        score.loc[valid] += weights["qed"] * qed_rank.reindex(df.index, fill_value=0.0)

    # --- Diversity component (1 - nn_similarity) ---
    if nn_similarities is not None and valid_indices is not None and "diversity" in weights:
        diversity_series = pd.Series(np.nan, index=df.index)
        # nn_similarities aligns with the sampled valid molecules
        valid_idx_in_df = df.index[valid]
        if len(nn_similarities) == len(valid_idx_in_df):
            diversity_series.loc[valid_idx_in_df] = 1.0 - nn_similarities
        elif len(nn_similarities) <= len(valid_idx_in_df):
            # Sampled — assign to sampled indices only
            sampled_df_idx = valid_idx_in_df[valid_indices[:len(nn_similarities)]]
            diversity_series.loc[sampled_df_idx] = 1.0 - nn_similarities

        div_rank = diversity_series.rank(pct=True, na_option="bottom")
        score += weights["diversity"] * div_rank.fillna(0.0)

    # --- Lipinski component ---
    if "passes_lipinski_like" in df.columns and "lipinski" in weights:
        score.loc[valid] += weights["lipinski"] * df.loc[valid, "passes_lipinski_like"].astype(float)

    df["priority_score"] = np.where(valid, score, np.nan)
    return df


# ---------------------------------------------------------------------------
# Shortlist extraction
# ---------------------------------------------------------------------------

def get_shortlist(
    df: pd.DataFrame,
    top_n: int = SHORTLIST_DEFAULT,
) -> pd.DataFrame:
    """
    Return the top-N molecules by priority_score.

    Columns included: SMILES, canonical_SMILES, pIC50 (if present),
    key descriptors, priority_score.
    """
    if "priority_score" not in df.columns:
        return pd.DataFrame()

    scored = df.dropna(subset=["priority_score"]).nlargest(top_n, "priority_score")

    cols = [COL_SMILES, COL_CANONICAL]
    if COL_PIC50 in scored.columns:
        cols.append(COL_PIC50)
    cols += ["MolWt", "LogP", "TPSA", "QED", "passes_lipinski_like", "priority_score"]
    cols = [c for c in cols if c in scored.columns]

    return scored[cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Diverse representative selection (greedy max-min)
# ---------------------------------------------------------------------------

def select_diverse_representatives(
    fp_matrix: np.ndarray,
    valid_df: pd.DataFrame,
    n_representatives: int = DIVERSE_REPS_DEFAULT,
) -> pd.DataFrame:
    """
    Select *n_representatives* maximally diverse molecules using
    greedy max-min Tanimoto distance.

    Algorithm:
      1. Start with the molecule closest to the centroid.
      2. Iteratively pick the molecule with the largest minimum
         Tanimoto distance to any already-selected molecule.

    Returns a subset of *valid_df* with the selected rows.
    """
    n = fp_matrix.shape[0]
    if n <= n_representatives:
        return valid_df.reset_index(drop=True)

    fp = fp_matrix.astype(np.float64)
    norms = np.sum(fp, axis=1)

    # Helper: Tanimoto similarity between one vector and all rows
    def _tanimoto_one_vs_all(vec: np.ndarray, vec_norm: float) -> np.ndarray:
        dot = fp @ vec
        denom = norms + vec_norm - dot
        denom = np.where(denom == 0, 1.0, denom)
        return dot / denom

    # Seed: molecule closest to centroid
    centroid = fp.mean(axis=0)
    centroid_norm = centroid.sum()
    centroid_sim = _tanimoto_one_vs_all(centroid, centroid_norm)
    seed = int(np.argmax(centroid_sim))

    selected = [seed]
    # min_dist[i] = min distance from molecule i to any selected molecule
    min_dist = 1.0 - _tanimoto_one_vs_all(fp[seed], norms[seed])

    for _ in range(n_representatives - 1):
        # Exclude already selected
        min_dist[selected] = -1.0
        # Pick molecule with largest min distance
        next_idx = int(np.argmax(min_dist))
        selected.append(next_idx)
        # Update min distances
        new_dist = 1.0 - _tanimoto_one_vs_all(fp[next_idx], norms[next_idx])
        min_dist = np.minimum(min_dist, new_dist)

    result = valid_df.iloc[selected].reset_index(drop=True)

    cols = [COL_SMILES, COL_CANONICAL]
    if COL_PIC50 in result.columns:
        cols.append(COL_PIC50)
    cols += ["MolWt", "LogP", "TPSA", "QED"]
    cols = [c for c in cols if c in result.columns]

    return result[cols]
