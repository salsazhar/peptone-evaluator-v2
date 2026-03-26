"""
Dimensionality reduction — PCA, t-SNE, and UMAP.

All methods operate on the Morgan fingerprint matrix (n_molecules × 1024)
and return 2-D coordinate arrays.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .config import (
    DR_RANDOM_STATE,
    TSNE_DEFAULT_PERPLEXITY,
    UMAP_DEFAULT_N_NEIGHBORS,
    UMAP_DEFAULT_MIN_DIST,
)


def reduce_pca(
    fp_matrix: np.ndarray,
    random_state: int = DR_RANDOM_STATE,
) -> np.ndarray:
    """PCA → 2 components. Returns (n, 2) array."""
    pca = PCA(n_components=2, random_state=random_state)
    return pca.fit_transform(fp_matrix)


def reduce_tsne(
    fp_matrix: np.ndarray,
    perplexity: int = TSNE_DEFAULT_PERPLEXITY,
    random_state: int = DR_RANDOM_STATE,
) -> np.ndarray:
    """t-SNE → 2 components. Returns (n, 2) array."""
    # Clamp perplexity to valid range
    perplexity = min(perplexity, max(1, fp_matrix.shape[0] - 1))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(fp_matrix)


def reduce_umap(
    fp_matrix: np.ndarray,
    n_neighbors: int = UMAP_DEFAULT_N_NEIGHBORS,
    min_dist: float = UMAP_DEFAULT_MIN_DIST,
    random_state: int = DR_RANDOM_STATE,
) -> np.ndarray:
    """UMAP → 2 components. Returns (n, 2) array."""
    import umap  # lazy import — umap-learn is an optional heavy dependency

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(n_neighbors, fp_matrix.shape[0] - 1),
        min_dist=min_dist,
        random_state=random_state,
        metric="jaccard",  # natural metric for binary fingerprints
    )
    return reducer.fit_transform(fp_matrix)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_METHODS = {
    "PCA": {
        "func": reduce_pca,
        "labels": ("PC1", "PC2"),
    },
    "t-SNE": {
        "func": reduce_tsne,
        "labels": ("t-SNE 1", "t-SNE 2"),
    },
    "UMAP": {
        "func": reduce_umap,
        "labels": ("UMAP 1", "UMAP 2"),
    },
}


def reduce(
    method: str,
    fp_matrix: np.ndarray,
    **kwargs,
) -> tuple[np.ndarray, str, str]:
    """
    Run the named dimensionality-reduction method.

    Returns
    -------
    (coords, x_label, y_label)
    """
    entry = _METHODS[method]
    coords = entry["func"](fp_matrix, **kwargs)
    return coords, entry["labels"][0], entry["labels"][1]
