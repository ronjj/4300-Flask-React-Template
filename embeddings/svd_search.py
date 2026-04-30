"""
SVD-based player search and explainability.

Uses a pre-fitted sklearn TruncatedSVD on the same scaled feature matrix as
player_embeddings.npy. Compares cosine similarity in raw feature space vs
latent space, and explains matches via signed latent-dimension products.
"""
from __future__ import annotations

import os
from typing import Any

import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from embeddings.features import CANONICAL_FEATURE_COLUMNS

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SVD_BUNDLE_PATH = os.path.join(PROJECT_ROOT, "embeddings", "svd_bundle.joblib")

_SVD_CACHE: dict[str, Any] | None = None


def load_svd_bundle() -> dict[str, Any] | None:
    global _SVD_CACHE
    if _SVD_CACHE is not None:
        return _SVD_CACHE
    if not os.path.exists(SVD_BUNDLE_PATH):
        _SVD_CACHE = None
        return None
    _SVD_CACHE = joblib.load(SVD_BUNDLE_PATH)
    return _SVD_CACHE


def clear_svd_cache() -> None:
    global _SVD_CACHE
    _SVD_CACHE = None


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


def _component_feature_hints(
    components: np.ndarray,
    feature_names: list[str],
    dim: int,
    top_n: int = 4,
) -> tuple[list[str], list[str]]:
    """Top positive- and negative-loading original features for latent dimension `dim`."""
    if dim < 0 or dim >= components.shape[0]:
        return [], []
    loadings = components[dim]
    order = np.argsort(loadings)
    neg_feats = [feature_names[i] for i in order[:top_n]]
    pos_feats = [feature_names[i] for i in order[-top_n:][::-1]]
    return pos_feats, neg_feats


_FEATURE_BUCKETS: dict[str, tuple[str, ...]] = {
    "Finishing": ("goals", "expected_goals", "shots_on_target", "goals_per_90", "shots"),
    "Chance creation": ("assists", "key_passes", "expected_assists"),
    "Passing & progression": ("passes", "progressive_passes"),
    "Ball carrying / 1v1": ("dribbles_completed",),
    "Defending": ("tackles", "interceptions", "blocks", "clearances", "recoveries", "aerial_duels_won"),
    "Usage / availability": ("minutes_played", "appearances"),
    "Discipline": ("yellow_cards", "red_cards"),
    "Goalkeeping": ("saves", "clean_sheets", "save_percentage", "goals_against"),
    "Position bias": ("pos_Forward", "pos_Midfielder", "pos_Defender", "pos_Goalkeeper"),
}


def _dimension_human_label(
    components: np.ndarray,
    feature_names: list[str],
    dim: int,
    top_n: int = 6,
) -> tuple[str, str]:
    """
    Produce a human-readable label for a latent dimension from its loadings.

    Returns (label, detail) where label is a short phrase and detail is a
    "high vs low" description derived from top + and - loadings.
    """
    if dim < 0 or dim >= components.shape[0]:
        return "Latent factor", ""
    loadings = components[dim]
    order = np.argsort(loadings)
    neg_feats = [feature_names[i] for i in order[:top_n]]
    pos_feats = [feature_names[i] for i in order[-top_n:][::-1]]

    def bucket_scores(feats: list[str]) -> dict[str, float]:
        scores: dict[str, float] = {k: 0.0 for k in _FEATURE_BUCKETS}
        for feat in feats:
            if feat not in feature_names:
                continue
            idx = feature_names.index(feat)
            w = float(abs(loadings[idx]))
            for bucket, members in _FEATURE_BUCKETS.items():
                if feat in members:
                    scores[bucket] += w
        return scores

    pos_scores = bucket_scores(pos_feats)
    neg_scores = bucket_scores(neg_feats)

    def top_buckets(scores: dict[str, float], k: int = 2) -> list[str]:
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        ranked = [b for b, s in ranked if s > 0]
        return ranked[:k]

    # Pick exactly ONE descriptor for the dimension.
    #
    # Strategy: choose the single strongest bucket across the top positive and
    # negative loadings (by total absolute loading weight in that bucket).
    # This avoids confusing labels like "Defending + Finishing".
    combined_scores: dict[str, float] = {k: 0.0 for k in _FEATURE_BUCKETS}
    for bucket in combined_scores:
        combined_scores[bucket] = float(pos_scores.get(bucket, 0.0)) + float(
            neg_scores.get(bucket, 0.0)
        )
    ranked = sorted(combined_scores.items(), key=lambda kv: kv[1], reverse=True)
    label = ranked[0][0] if ranked and ranked[0][1] > 0 else "Latent factor"

    pos_detail = ", ".join(pos_feats[:3])
    neg_detail = ", ".join(neg_feats[:3])
    detail = f"High: {pos_detail} · Low: {neg_detail}"
    return label, detail


def explain_latent_alignment(
    query_latent: np.ndarray,
    player_latent: np.ndarray,
    bundle: dict[str, Any],
    top_k_each: int = 5,
) -> dict[str, Any]:
    """
    Explain cosine alignment via per-dimension products q_d * p_d (before norm).
    Positive product = same sign on that factor (reinforcing);
    negative = opposite sign (one high + one high- pulls apart).
    """
    svd = bundle["svd"]
    components: np.ndarray = svd.components_
    feature_names: list[str] = bundle.get("feature_names", list(CANONICAL_FEATURE_COLUMNS))

    q = np.asarray(query_latent, dtype=float).ravel()
    p = np.asarray(player_latent, dtype=float).ravel()
    products = q * p
    order = np.argsort(products)

    neg_dims = []
    for idx in order[:top_k_each]:
        if products[idx] >= 0:
            break
        pos_h, neg_h = _component_feature_hints(components, feature_names, int(idx))
        label, label_detail = _dimension_human_label(components, feature_names, int(idx))
        neg_dims.append(
            {
                "dim": int(idx),
                "label": label,
                "label_detail": label_detail,
                "query_activation": float(q[idx]),
                "player_activation": float(p[idx]),
                "contribution": float(products[idx]),
                "top_positive_loadings": pos_h,
                "top_negative_loadings": neg_h,
            }
        )

    pos_dims = []
    for idx in order[::-1][:top_k_each]:
        if products[idx] <= 0:
            break
        pos_h, neg_h = _component_feature_hints(components, feature_names, int(idx))
        label, label_detail = _dimension_human_label(components, feature_names, int(idx))
        pos_dims.append(
            {
                "dim": int(idx),
                "label": label,
                "label_detail": label_detail,
                "query_activation": float(q[idx]),
                "player_activation": float(p[idx]),
                "contribution": float(products[idx]),
                "top_positive_loadings": pos_h,
                "top_negative_loadings": neg_h,
            }
        )

    return {"positive_dimensions": pos_dims, "negative_dimensions": neg_dims}


def top_latent_activations(
    latent: np.ndarray,
    bundle: dict[str, Any],
    top_k: int = 8,
) -> list[dict[str, Any]]:
    """
    Return the strongest latent activations by absolute value.

    Each entry includes dim, activation, and a human label derived from loadings.
    """
    svd = bundle["svd"]
    components: np.ndarray = svd.components_
    feature_names: list[str] = bundle.get("feature_names", list(CANONICAL_FEATURE_COLUMNS))
    vec = np.asarray(latent, dtype=float).ravel()
    if vec.size == 0:
        return []
    order = np.argsort(np.abs(vec))[::-1]
    out: list[dict[str, Any]] = []
    for idx in order[: max(1, int(top_k))]:
        d = int(idx)
        label, label_detail = _dimension_human_label(components, feature_names, d)
        out.append(
            {
                "dim": d,
                "label": label,
                "label_detail": label_detail,
                "activation": float(vec[d]),
            }
        )
    return out


def top_latent_alignment(
    query_latent: np.ndarray,
    player_latent: np.ndarray,
    bundle: dict[str, Any],
    top_k: int = 8,
) -> list[dict[str, Any]]:
    """
    Return top aligned dimensions for a (query, player) pair.

    Each entry includes q, p, and product = q*p for the dimension, plus labels.
    """
    svd = bundle["svd"]
    components: np.ndarray = svd.components_
    feature_names: list[str] = bundle.get("feature_names", list(CANONICAL_FEATURE_COLUMNS))
    q = np.asarray(query_latent, dtype=float).ravel()
    p = np.asarray(player_latent, dtype=float).ravel()
    if q.size == 0 or p.size == 0 or q.shape != p.shape:
        return []
    products = q * p
    order = np.argsort(np.abs(products))[::-1]
    out: list[dict[str, Any]] = []
    for idx in order[: max(1, int(top_k))]:
        d = int(idx)
        label, label_detail = _dimension_human_label(components, feature_names, d)
        out.append(
            {
                "dim": d,
                "label": label,
                "label_detail": label_detail,
                "query_activation": float(q[d]),
                "player_activation": float(p[d]),
                "contribution": float(products[d]),
            }
        )
    return out


def rank_raw_vs_svd(
    prototype: np.ndarray,
    candidate_matrix: np.ndarray,
    bundle: dict[str, Any],
    top_k: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (raw_scores, svd_scores, raw_rank_idx_local, svd_rank_idx_local)
    for rows of candidate_matrix (same order as input).
    """
    proto = prototype.astype(float).reshape(1, -1)
    cand = candidate_matrix.astype(float)

    raw_scores = cosine_similarity(_normalize_rows(proto), _normalize_rows(cand))[0]

    svd = bundle["svd"]
    proto_lat = svd.transform(proto)
    cand_lat = svd.transform(cand)
    svd_scores = cosine_similarity(_normalize_rows(proto_lat), _normalize_rows(cand_lat))[0]

    raw_order = np.argsort(raw_scores)[::-1]
    svd_order = np.argsort(svd_scores)[::-1]
    return raw_scores, svd_scores, raw_order, svd_order


def svd_dimension_legend(bundle: dict[str, Any], max_dims: int = 16) -> list[dict[str, Any]]:
    """Short human-readable interpretation of each latent dimension from Vt loadings."""
    svd = bundle["svd"]
    components: np.ndarray = svd.components_
    feature_names: list[str] = bundle.get("feature_names", list(CANONICAL_FEATURE_COLUMNS))
    evr = getattr(svd, "explained_variance_ratio_", None)
    out: list[dict[str, Any]] = []
    k = min(max_dims, components.shape[0])
    for d in range(k):
        pos_h, neg_h = _component_feature_hints(components, feature_names, d, top_n=3)
        label, label_detail = _dimension_human_label(components, feature_names, d, top_n=6)
        entry: dict[str, Any] = {
            "dim": d,
            "label": label,
            "label_detail": label_detail,
            "top_positive_loadings": pos_h,
            "top_negative_loadings": neg_h,
        }
        if evr is not None and len(evr) > d:
            entry["explained_variance_ratio"] = float(evr[d])
        out.append(entry)
    return out
