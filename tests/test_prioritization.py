"""Tests for priority scoring, shortlisting, and diverse selection."""

import numpy as np
import pandas as pd

from src.config import COL_VALID
from src.prioritization import (
    compute_priority_scores,
    get_shortlist,
    select_diverse_representatives,
)
from src.similarity import compute_similarity_metrics


class TestPriorityScoring:
    def test_score_column_added(self, enriched_df, fp_matrix):
        metrics = compute_similarity_metrics(fp_matrix)
        valid_idx = enriched_df[enriched_df[COL_VALID]].index.to_numpy()
        result = compute_priority_scores(
            enriched_df, metrics["nn_similarities"], valid_idx, has_pic50=True,
        )
        assert "priority_score" in result.columns

    def test_scores_in_range(self, enriched_df, fp_matrix):
        metrics = compute_similarity_metrics(fp_matrix)
        valid_idx = enriched_df[enriched_df[COL_VALID]].index.to_numpy()
        result = compute_priority_scores(
            enriched_df, metrics["nn_similarities"], valid_idx, has_pic50=True,
        )
        valid_scores = result.loc[result[COL_VALID], "priority_score"].dropna()
        assert (valid_scores >= 0).all()
        assert (valid_scores <= 1).all()

    def test_invalid_molecules_get_nan(self, enriched_df, fp_matrix):
        metrics = compute_similarity_metrics(fp_matrix)
        valid_idx = enriched_df[enriched_df[COL_VALID]].index.to_numpy()
        result = compute_priority_scores(
            enriched_df, metrics["nn_similarities"], valid_idx, has_pic50=True,
        )
        invalid = result[~result[COL_VALID]]
        assert invalid["priority_score"].isna().all()

    def test_works_without_pic50(self, enriched_df, fp_matrix):
        df = enriched_df.drop(columns=["pIC50"], errors="ignore")
        metrics = compute_similarity_metrics(fp_matrix)
        valid_idx = df[df[COL_VALID]].index.to_numpy()
        result = compute_priority_scores(
            df, metrics["nn_similarities"], valid_idx, has_pic50=False,
        )
        assert "priority_score" in result.columns


class TestShortlist:
    def test_shortlist_respects_top_n(self, enriched_df, fp_matrix):
        metrics = compute_similarity_metrics(fp_matrix)
        valid_idx = enriched_df[enriched_df[COL_VALID]].index.to_numpy()
        scored = compute_priority_scores(
            enriched_df, metrics["nn_similarities"], valid_idx, has_pic50=True,
        )
        shortlist = get_shortlist(scored, top_n=3)
        assert len(shortlist) <= 3

    def test_shortlist_sorted_descending(self, enriched_df, fp_matrix):
        metrics = compute_similarity_metrics(fp_matrix)
        valid_idx = enriched_df[enriched_df[COL_VALID]].index.to_numpy()
        scored = compute_priority_scores(
            enriched_df, metrics["nn_similarities"], valid_idx, has_pic50=True,
        )
        shortlist = get_shortlist(scored, top_n=5)
        if len(shortlist) > 1:
            scores = shortlist["priority_score"].values
            assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


class TestDiverseRepresentatives:
    def test_returns_correct_count(self, enriched_df, fp_matrix):
        valid_df = enriched_df[enriched_df[COL_VALID]].copy()
        result = select_diverse_representatives(fp_matrix, valid_df, n_representatives=3)
        assert len(result) == 3

    def test_more_reps_than_molecules(self, enriched_df, fp_matrix):
        valid_df = enriched_df[enriched_df[COL_VALID]].copy()
        n = len(valid_df)
        result = select_diverse_representatives(fp_matrix, valid_df, n_representatives=n + 10)
        assert len(result) == n
