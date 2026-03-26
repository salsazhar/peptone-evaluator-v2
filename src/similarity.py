"""
Tanimoto similarity and diversity metrics.

Computes pairwise similarity on Morgan fingerprint bit-vectors using
vectorised numpy operations.  Large datasets are randomly sampled to
keep computation tractable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    SIMILARITY_MAX_MOLECULES,
    DR_RANDOM_STATE,
    TOP_SIMILAR_PAIRS,
    TOP_ISOLATED,
)


# ---------------------------------------------------------------------------
# Core pairwise Tanimoto
# ---------------------------------------------------------------------------

def _pairwise_tanimoto(fp_matrix: np.ndarray) -> np.ndarray:
    """
    Compute the full pairwise Tanimoto similarity matrix.

    Uses the identity:
        T(A, B) = |A & B| / |A | B| = dot(A, B) / (|A| + |B| - dot(A, B))

    Parameters
    ----------
    fp_matrix : (n, d) binary array (float or bool).

    Returns
    -------
    (n, n) float64 similarity matrix with 0 on the diagonal.
    """
    fp = fp_matrix.astype(np.float64)
    dot = fp @ fp.T
    norms = np.sum(fp, axis=1)
    denom = norms[:, None] + norms[None, :] - dot
    denom = np.where(denom == 0, 1.0, denom)
    sim = dot / denom
    np.fill_diagonal(sim, 0.0)
    return sim


def nearest_neighbor_similarity(sim_matrix: np.ndarray) -> np.ndarray:
    """Max similarity to any other molecule for each row."""
    return np.max(sim_matrix, axis=1)


# ---------------------------------------------------------------------------
# High-level metrics (existing, extended)
# ---------------------------------------------------------------------------

def compute_similarity_metrics(
    fp_matrix: np.ndarray,
    sampled_smiles: list[str] | None = None,
    max_molecules: int = SIMILARITY_MAX_MOLECULES,
    random_state: int = DR_RANDOM_STATE,
) -> dict:
    """
    High-level similarity and diversity summary.

    If the dataset exceeds *max_molecules*, a random sample is used for
    the pairwise computation and the result is flagged.

    Returns
    -------
    dict with keys:
        mean_pairwise_similarity, median_pairwise_similarity,
        mean_nn_similarity, diversity_score,
        nn_similarities (1-D array), sim_matrix (2-D array),
        sampled_indices, was_sampled, sample_size.
    """
    n = fp_matrix.shape[0]
    was_sampled = n > max_molecules

    if was_sampled:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(n, size=max_molecules, replace=False)
        sample = fp_matrix[idx]
    else:
        idx = np.arange(n)
        sample = fp_matrix

    sim = _pairwise_tanimoto(sample)
    upper = sim[np.triu_indices_from(sim, k=1)]
    nn_sim = nearest_neighbor_similarity(sim)

    return {
        "mean_pairwise_similarity": float(np.mean(upper)) if upper.size > 0 else 0.0,
        "median_pairwise_similarity": float(np.median(upper)) if upper.size > 0 else 0.0,
        "mean_nn_similarity": float(np.mean(nn_sim)) if nn_sim.size > 0 else 0.0,
        "diversity_score": 1.0 - float(np.mean(upper)) if upper.size > 0 else 1.0,
        "nn_similarities": nn_sim,
        "sim_matrix": sim,
        "sampled_indices": idx,
        "was_sampled": was_sampled,
        "sample_size": sample.shape[0],
    }


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------

def find_duplicates(canonical_smiles: pd.Series) -> dict:
    """
    Count exact duplicates by canonical SMILES.

    Returns
    -------
    dict with duplicate_count and duplicate_rate.
    """
    total = len(canonical_smiles.dropna())
    unique = canonical_smiles.dropna().nunique()
    dup_count = total - unique
    return {
        "duplicate_count": dup_count,
        "duplicate_rate": dup_count / total if total > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Top similar pairs
# ---------------------------------------------------------------------------

def top_similar_pairs(
    sim_matrix: np.ndarray,
    smiles_list: list[str],
    top_n: int = TOP_SIMILAR_PAIRS,
) -> pd.DataFrame:
    """
    Return the *top_n* most similar molecule pairs.

    Returns DataFrame with columns: SMILES_A, SMILES_B, Tanimoto.
    """
    n = sim_matrix.shape[0]
    if n < 2:
        return pd.DataFrame(columns=["SMILES_A", "SMILES_B", "Tanimoto"])

    # Get upper-triangle indices and values
    tri_i, tri_j = np.triu_indices(n, k=1)
    tri_vals = sim_matrix[tri_i, tri_j]

    # Get top-N indices
    k = min(top_n, len(tri_vals))
    top_idx = np.argpartition(tri_vals, -k)[-k:]
    top_idx = top_idx[np.argsort(tri_vals[top_idx])[::-1]]

    rows = []
    for idx in top_idx:
        i, j = int(tri_i[idx]), int(tri_j[idx])
        rows.append({
            "SMILES_A": smiles_list[i],
            "SMILES_B": smiles_list[j],
            "Tanimoto": float(tri_vals[idx]),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Most isolated molecules
# ---------------------------------------------------------------------------

def top_isolated_molecules(
    nn_similarities: np.ndarray,
    smiles_list: list[str],
    top_n: int = TOP_ISOLATED,
) -> pd.DataFrame:
    """
    Return the *top_n* most isolated molecules (lowest NN similarity).

    Returns DataFrame with columns: SMILES, NN_Similarity.
    """
    k = min(top_n, len(nn_similarities))
    idx = np.argpartition(nn_similarities, k)[:k]
    idx = idx[np.argsort(nn_similarities[idx])]

    rows = []
    for i in idx:
        rows.append({
            "SMILES": smiles_list[int(i)],
            "NN_Similarity": float(nn_similarities[int(i)]),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-molecule nearest-neighbour table
# ---------------------------------------------------------------------------

def per_molecule_nn_table(
    sim_matrix: np.ndarray,
    smiles_list: list[str],
) -> pd.DataFrame:
    """
    For each molecule: its SMILES, nearest-neighbour SMILES, and NN similarity.

    Returns DataFrame with columns: SMILES, NN_SMILES, NN_Similarity.
    """
    nn_idx = np.argmax(sim_matrix, axis=1)
    nn_sim = np.max(sim_matrix, axis=1)

    rows = []
    for i in range(len(smiles_list)):
        rows.append({
            "SMILES": smiles_list[i],
            "NN_SMILES": smiles_list[int(nn_idx[i])],
            "NN_Similarity": float(nn_sim[i]),
        })
    return pd.DataFrame(rows)
