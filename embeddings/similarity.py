from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from embeddings.features import CANONICAL_FEATURE_COLUMNS, DESCRIPTION_DISCRIMINATIVE_FEATURES
from embeddings.store import get_player_vector, load_player_metadata
from src.player_search import (
    career_start_year_for_embedding_player,
    nationality_filter_from_text,
    nationality_normalized_for_embedding_player,
    normalize_text,
    passes_max_age_under,
    query_max_age_under,
    reference_year_for_queries,
    region_nationality_allowlist_from_text,
)


POSITION_KEYWORDS = {
    "Forward": ("striker", "strikers", "forward", "forwards", "winger", "wingers"),
    "Midfielder": ("midfielder", "midfielders"),
    "Defender": ("defender", "defenders"),
    "Goalkeeper": ("goalkeeper", "goalkeepers", "keeper", "keepers"),
}


def _feature_indices(feature_names: list[str]) -> list[int]:
    return [CANONICAL_FEATURE_COLUMNS.index(name) for name in feature_names if name in CANONICAL_FEATURE_COLUMNS]


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms


def _batched_cosine(query_vector: np.ndarray, candidate_matrix: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    scores: list[np.ndarray] = []
    for start in range(0, len(candidate_matrix), batch_size):
        batch = candidate_matrix[start:start + batch_size]
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            scores.append(cosine_similarity(query_vector, batch)[0])
    return np.concatenate(scores, axis=0)


def parse_similarity_query(query_text: str) -> dict[str, Any]:
    """Parse a free-text similarity query into position, nationality, and era filters."""
    normalized = query_text.casefold()
    filters: dict[str, Any] = {}

    for position, keywords in POSITION_KEYWORDS.items():
        if any(re.search(rf"\b{re.escape(keyword)}\b", normalized) for keyword in keywords):
            filters["position"] = position
            break

    nationality_region = region_nationality_allowlist_from_text(query_text)
    if nationality_region:
        filters["nationality_region"] = nationality_region
    else:
        nationality = nationality_filter_from_text(query_text)
        if nationality:
            filters["nationality"] = nationality
            filters["nationality_normalized"] = normalize_text(nationality)

    decade_match = re.search(r"\b((?:19|20)\d0)s\b", normalized)
    if decade_match:
        start = int(decade_match.group(1))
        filters["era_start"] = start
        filters["era_end"] = start + 9

    max_age_under = query_max_age_under(query_text)
    if max_age_under is not None:
        filters["max_age_under"] = max_age_under
        filters["reference_year"] = reference_year_for_queries()

    return filters


def _get_player_position(player_name: str, metadata: dict[str, dict]) -> str:
    return metadata.get(player_name, {}).get("primary_position", "Unknown")


def build_description_prototype(
    filtered_names: list[str],
    matrix: np.ndarray,
    player_index: list[str],
    player_metadata: dict[str, dict],
    query_filters: dict[str, Any],
) -> np.ndarray:
    """Build a weighted exemplar prototype vector for description search."""
    if not filtered_names:
        raise ValueError("Cannot build a description prototype from an empty candidate set.")

    candidate_positions = [
        player_metadata.get(name, {}).get("primary_position")
        for name in filtered_names
        if player_metadata.get(name, {}).get("primary_position")
    ]
    active_position = query_filters.get("position")
    if active_position is None and candidate_positions:
        active_position = max(set(candidate_positions), key=candidate_positions.count)

    feature_names = DESCRIPTION_DISCRIMINATIVE_FEATURES.get(active_position or "", [])
    feature_idxs = _feature_indices(feature_names)
    index_lookup = {name: idx for idx, name in enumerate(player_index)}
    candidate_vectors = np.array([matrix[index_lookup[name]] for name in filtered_names], dtype=float)

    if not feature_idxs:
        return candidate_vectors.mean(axis=0)

    discriminative_scores = candidate_vectors[:, feature_idxs].sum(axis=1)
    sorted_order = np.argsort(discriminative_scores)[::-1]
    top_count = max(1, math.ceil(len(filtered_names) * 0.25))
    if len(filtered_names) < 4:
        top_count = max(1, math.ceil(len(filtered_names) * 0.5))
    exemplar_order = sorted_order[:top_count]
    exemplar_vectors = candidate_vectors[exemplar_order]
    exemplar_scores = discriminative_scores[exemplar_order]
    min_score = exemplar_scores.min()
    weights = exemplar_scores - min_score + 1.0
    return np.average(exemplar_vectors, axis=0, weights=weights)


def find_similar_players(
    query_name: str,
    matrix: np.ndarray,
    player_index: list[str],
    top_k: int = 10,
) -> list[dict]:
    """Find the top-k most similar players to a named player using cosine similarity."""
    metadata = load_player_metadata()
    query_vector = get_player_vector(query_name, matrix, player_index).reshape(1, -1)
    normalized_query = _normalize_vectors(query_vector.astype(float))
    normalized_matrix = _normalize_vectors(matrix.astype(float))
    similarities = _batched_cosine(normalized_query, normalized_matrix)
    query_idx = np.argmax(similarities)
    ranked_indices = np.argsort(similarities)[::-1]

    results: list[dict] = []
    for idx in ranked_indices:
        if idx == query_idx:
            continue
        normalized_name = player_index[idx]
        player_meta = metadata.get(normalized_name, {})
        results.append(
            {
                "player": player_meta.get("display_name", normalized_name),
                "lookup_key": normalized_name,
                "score": float(similarities[idx]),
                "position": player_meta.get("primary_position", "Unknown"),
            }
        )
        if len(results) >= top_k:
            break
    return results


def find_players_by_description(
    query_text: str,
    matrix: np.ndarray,
    player_index: list[str],
    player_metadata: dict[str, dict],
    top_k: int = 10,
) -> list[dict]:
    """Find players matching a free-text description via filtered prototype similarity."""
    filters = parse_similarity_query(query_text)
    filtered_names: list[str] = []
    nat_key = filters.get("nationality_normalized")
    nat_region = filters.get("nationality_region")
    max_age_under = filters.get("max_age_under")
    ref_year = int(filters.get("reference_year") or reference_year_for_queries())
    for name in player_index:
        meta = player_metadata.get(name, {})
        if filters.get("position") and meta.get("primary_position") != filters["position"]:
            continue
        era_start = filters.get("era_start")
        era_end = filters.get("era_end")
        if era_start is not None and era_end is not None:
            years = meta.get("season_years", [])
            if not any(era_start <= year <= era_end for year in years):
                continue
        pnat = nationality_normalized_for_embedding_player(name, meta)
        if nat_region:
            if pnat not in nat_region:
                continue
        elif nat_key:
            if pnat != nat_key:
                continue
        if max_age_under is not None:
            start = career_start_year_for_embedding_player(name, meta)
            if not passes_max_age_under(max_age_under, ref_year, start):
                continue
        filtered_names.append(name)

    if not filtered_names:
        return []

    prototype = build_description_prototype(filtered_names, matrix, player_index, player_metadata, filters).reshape(1, -1)
    index_lookup = {name: idx for idx, name in enumerate(player_index)}
    filtered_indices = [index_lookup[name] for name in filtered_names]
    filtered_matrix = matrix[filtered_indices]
    similarities = _batched_cosine(
        _normalize_vectors(prototype.astype(float)),
        _normalize_vectors(filtered_matrix.astype(float)),
    )
    ranked_local_indices = np.argsort(similarities)[::-1]

    results: list[dict] = []
    for local_idx in ranked_local_indices[:top_k]:
        normalized_name = filtered_names[local_idx]
        meta = player_metadata.get(normalized_name, {})
        results.append(
            {
                "player": meta.get("display_name", normalized_name),
                "lookup_key": normalized_name,
                "score": float(similarities[local_idx]),
                "position": meta.get("primary_position", "Unknown"),
            }
        )
    return results
