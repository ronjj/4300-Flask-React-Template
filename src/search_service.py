from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from embeddings.features import CANONICAL_FEATURE_COLUMNS
from embeddings.similarity import (
    build_description_prototype,
    find_players_by_description,
    find_similar_players,
    parse_similarity_query,
)
from embeddings.store import FEATURE_NAMES_JSON_PATH, get_player_index, load_embeddings, load_player_metadata
from embeddings.svd_search import (
    explain_latent_alignment,
    load_svd_bundle,
    rank_raw_vs_svd,
    svd_dimension_legend,
)

try:
    from player_search import (
        PLAYER_INDEX,
        boolean_search,
        career_start_year_for_embedding_player,
        find_player_by_name,
        league_filter_from_text,
        nationality_normalized_for_embedding_player,
        normalize_text,
        serialize_player,
        parse_query,
        passes_max_age_under,
        reference_year_for_queries,
    )
except ImportError:  # pragma: no cover - package-style import fallback
    from src.player_search import (
        PLAYER_INDEX,
        boolean_search,
        career_start_year_for_embedding_player,
        find_player_by_name,
        league_filter_from_text,
        nationality_normalized_for_embedding_player,
        normalize_text,
        serialize_player,
        parse_query,
        passes_max_age_under,
        reference_year_for_queries,
    )


LOGGER = logging.getLogger(__name__)

_FEATURE_NAME_CACHE: list[str] | None = None


def _load_feature_names(expected_dim: int) -> list[str]:
    global _FEATURE_NAME_CACHE
    if _FEATURE_NAME_CACHE is not None and len(_FEATURE_NAME_CACHE) == expected_dim:
        return _FEATURE_NAME_CACHE
    try:
        with open(FEATURE_NAMES_JSON_PATH, "r", encoding="utf-8") as handle:
            names = json.load(handle)
        if isinstance(names, list) and len(names) == expected_dim:
            _FEATURE_NAME_CACHE = [str(x) for x in names]
            return _FEATURE_NAME_CACHE
    except Exception:
        pass
    _FEATURE_NAME_CACHE = list(CANONICAL_FEATURE_COLUMNS) + [
        f"feature_{i}" for i in range(max(0, expected_dim - len(CANONICAL_FEATURE_COLUMNS)))
    ]
    _FEATURE_NAME_CACHE = _FEATURE_NAME_CACHE[:expected_dim]
    return _FEATURE_NAME_CACHE

SIMILAR_PLAYER_PATTERNS = (
    re.compile(r"^\s*(?:players?\s+)?most like\s+(.+?)\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?:players?\s+)?similar to\s+(.+?)\s*$", re.IGNORECASE),
    re.compile(r"^\s*who(?:'s| is)\s+like\s+(.+?)\s*$", re.IGNORECASE),
)
DESCRIPTION_HINTS = (
    "clinical",
    "creative",
    "defensive",
    "physical",
    "playmaker",
    "box-to-box",
    "prolific",
    "young",
)

_EMBEDDING_CACHE: dict[str, Any] | None = None


def _load_embeddings_bundle() -> dict[str, Any] | None:
    global _EMBEDDING_CACHE
    if _EMBEDDING_CACHE is not None:
        return _EMBEDDING_CACHE
    try:
        matrix, player_index = load_embeddings()
        player_metadata = load_player_metadata()
    except Exception as exc:  # pragma: no cover - runtime fallback
        LOGGER.warning("Embeddings unavailable for semantic search: %s", exc)
        _EMBEDDING_CACHE = None
        return None
    _EMBEDDING_CACHE = {
        "matrix": matrix,
        "player_index": player_index,
        "player_metadata": player_metadata,
    }
    return _EMBEDDING_CACHE


def _extract_similarity_target(query: str) -> str | None:
    for pattern in SIMILAR_PLAYER_PATTERNS:
        match = pattern.match(query)
        if match:
            return match.group(1).strip()
    return None


def _is_description_similarity_query(query: str) -> bool:
    normalized = query.casefold()
    return any(hint in normalized for hint in DESCRIPTION_HINTS) and not any(
        pattern.match(query) for pattern in SIMILAR_PLAYER_PATTERNS
    )


def _aggregate_embedding_result(result: Dict[str, Any]) -> Dict[str, Any]:
    display_name = result["player"]
    lookup_key = result.get("lookup_key") or display_name
    records = find_player_by_name(lookup_key) or []

    if not records:
        return {
            "player_id": None,
            "name": display_name,
            "nationality": None,
            "position": result.get("position"),
            "league": "Embedding profile",
            "team": None,
            "image": None,
            "goals": None,
            "assists": None,
            "appearances": None,
            "minutes": None,
            "shots_on_target": None,
            "dribbles_completed": None,
            "season_years": [],
            "seasons": [],
            "goals_per_game": None,
            "assists_per_game": None,
            "shot_on_target_ratio": None,
            "similarity_score": float(result.get("score", 0)),
            "search_mode": "embedding",
        }

    leagues = sorted({record.get("league") for record in records if record.get("league")})
    seasons = sorted({season for record in records for season in record.get("seasons", [])})
    season_years = sorted({year for record in records for year in record.get("season_years", [])})

    def first_non_null(field: str):
        for record in records:
            value = record.get(field)
            if value not in (None, "", []):
                return value
        return None

    def summed(field: str):
        values = [record.get(field) for record in records if record.get(field) is not None]
        return sum(values) if values else None

    return {
        "player_id": first_non_null("player_id"),
        "name": first_non_null("name") or display_name,
        "nationality": first_non_null("nationality"),
        "position": result.get("position") or first_non_null("position"),
        "league": ", ".join(leagues) if leagues else "Embedding profile",
        "team": first_non_null("team"),
        "image": first_non_null("image"),
        "goals": summed("goals"),
        "assists": summed("assists"),
        "appearances": summed("appearances"),
        "minutes": summed("minutes"),
        "shots_on_target": summed("shots_on_target"),
        "dribbles_completed": summed("dribbles_completed"),
        "season_years": season_years,
        "seasons": seasons,
        "goals_per_game": first_non_null("goals_per_game"),
        "assists_per_game": first_non_null("assists_per_game"),
        "shot_on_target_ratio": first_non_null("shot_on_target_ratio"),
        "similarity_score": float(result.get("score", 0)),
        "search_mode": "embedding",
    }


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


def _description_filtered_names(
    query: str,
    player_index: List[str],
    player_metadata: Dict[str, Any],
) -> List[str]:
    filters = parse_similarity_query(query)
    filtered_names: List[str] = []
    league = league_filter_from_text(query)
    nat_key = filters.get("nationality_normalized")
    nat_region = filters.get("nationality_region")
    max_age_under = filters.get("max_age_under")
    ref_year = int(filters.get("reference_year") or reference_year_for_queries())
    for name in player_index:
        meta = player_metadata.get(name, {})
        if league:
            leagues = meta.get("leagues") or []
            if league not in leagues:
                continue
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
    return filtered_names


def _pack_semantic_hit(
    embedding_bundle: Dict[str, Any],
    row_index: int,
    score: float,
    mode: str,
    svd_explain: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    player_index = embedding_bundle["player_index"]
    player_metadata = embedding_bundle["player_metadata"]
    norm_name = player_index[row_index]
    meta = player_metadata.get(norm_name, {})
    display = meta.get("display_name", norm_name)
    inner = {
        "player": display,
        "lookup_key": norm_name,
        "score": float(score),
        "position": meta.get("primary_position", "Unknown"),
    }
    agg = _aggregate_embedding_result(inner)
    agg["search_mode"] = mode
    if svd_explain is not None:
        agg["svd_explain"] = svd_explain
    return agg


def _dual_semantic_search(
    prototype: np.ndarray,
    candidate_indices: List[int],
    embedding_bundle: Dict[str, Any],
    mode: str,
    top_k: int = 10,
) -> Dict[str, Any]:
    """
    Rank candidates by raw cosine vs SVD latent cosine; attach per-hit latent explainability.
    """
    empty = {
        "mode": mode,
        "results": [],
        "results_svd": [],
        "results_without_svd": [],
        "svd_available": False,
        "svd_latent_dimensions": [],
    }
    if not candidate_indices:
        return empty

    matrix = embedding_bundle["matrix"]
    cand_mat = matrix[np.array(candidate_indices, dtype=int)]
    svd_bundle = load_svd_bundle()

    if svd_bundle is None:
        proto = prototype.astype(float).reshape(1, -1)
        sims = cosine_similarity(_normalize_rows(proto), _normalize_rows(cand_mat.astype(float)))[0]
        order = np.argsort(sims)[::-1][:top_k]
        results = [
            _pack_semantic_hit(embedding_bundle, candidate_indices[int(i)], sims[i], mode)
            for i in order
        ]
        return {
            "mode": mode,
            "results": results,
            "results_svd": None,
            "results_without_svd": None,
            "svd_available": False,
            "svd_latent_dimensions": [],
        }

    raw_scores, svd_scores, raw_order, svd_order = rank_raw_vs_svd(
        prototype, cand_mat, svd_bundle, top_k=top_k
    )
    svd_model = svd_bundle["svd"]
    proto_lat = svd_model.transform(prototype.reshape(1, -1))[0]
    cand_lat = svd_model.transform(cand_mat)

    raw_results: List[Dict[str, Any]] = []
    for loc in raw_order[:top_k]:
        loc = int(loc)
        raw_results.append(
            _pack_semantic_hit(
                embedding_bundle, candidate_indices[loc], raw_scores[loc], mode
            )
        )

    svd_results: List[Dict[str, Any]] = []
    for loc in svd_order[:top_k]:
        loc = int(loc)
        explain = explain_latent_alignment(proto_lat, cand_lat[loc], svd_bundle)
        svd_results.append(
            _pack_semantic_hit(
                embedding_bundle,
                candidate_indices[loc],
                svd_scores[loc],
                mode,
                svd_explain=explain,
            )
        )

    return {
        "mode": mode,
        "results": svd_results,
        "results_svd": svd_results,
        "results_without_svd": raw_results,
        "svd_available": True,
        "svd_latent_dimensions": svd_dimension_legend(svd_bundle),
    }


def search_players(query: str) -> Dict[str, Any]:
    """Route a search query to boolean or embedding-backed search."""
    semantic_target = _extract_similarity_target(query)
    embedding_bundle = _load_embeddings_bundle()

    if semantic_target and embedding_bundle is not None:
        if load_svd_bundle() is not None:
            try:
                q_idx = get_player_index(
                    semantic_target, embedding_bundle["player_index"]
                )
            except ValueError:
                pass
            else:
                matrix = embedding_bundle["matrix"]
                prototype = matrix[q_idx]
                candidate_indices = [
                    i for i in range(len(embedding_bundle["player_index"])) if i != q_idx
                ]
                return _dual_semantic_search(
                    prototype,
                    candidate_indices,
                    embedding_bundle,
                    "embedding_similarity",
                    top_k=10,
                )
        semantic_results = find_similar_players(
            semantic_target,
            embedding_bundle["matrix"],
            embedding_bundle["player_index"],
            top_k=10,
        )
        return {
            "mode": "embedding_similarity",
            "results": [_aggregate_embedding_result(r) for r in semantic_results],
            "results_svd": None,
            "results_without_svd": None,
            "svd_available": False,
            "svd_latent_dimensions": [],
        }

    if _is_description_similarity_query(query) and embedding_bundle is not None:
        if load_svd_bundle() is not None:
            pi = embedding_bundle["player_index"]
            pm = embedding_bundle["player_metadata"]
            matrix = embedding_bundle["matrix"]
            filtered_names = _description_filtered_names(query, pi, pm)
            if not filtered_names:
                bundle = load_svd_bundle()
                return {
                    "mode": "embedding_description",
                    "results": [],
                    "results_svd": [],
                    "results_without_svd": [],
                    "svd_available": bundle is not None,
                    "svd_latent_dimensions": svd_dimension_legend(bundle) if bundle else [],
                }
            index_lookup = {name: idx for idx, name in enumerate(pi)}
            candidate_indices = [index_lookup[n] for n in filtered_names]
            filters = parse_similarity_query(query)
            prototype = build_description_prototype(
                filtered_names, matrix, pi, pm, filters
            )
            return _dual_semantic_search(
                prototype,
                candidate_indices,
                embedding_bundle,
                "embedding_description",
                top_k=10,
            )
        semantic_results = find_players_by_description(
            query,
            embedding_bundle["matrix"],
            embedding_bundle["player_index"],
            embedding_bundle["player_metadata"],
            top_k=10,
        )
        return {
            "mode": "embedding_description",
            "results": [_aggregate_embedding_result(r) for r in semantic_results],
            "results_svd": None,
            "results_without_svd": None,
            "svd_available": False,
            "svd_latent_dimensions": [],
        }

    filters = parse_query(query)
    boolean_results = boolean_search(filters, PLAYER_INDEX["player_list"])

    # Always try to produce an SVD comparison when possible, even for boolean queries.
    svd_bundle = load_svd_bundle() if embedding_bundle is not None else None
    if embedding_bundle is not None and svd_bundle is not None:
        pi = embedding_bundle["player_index"]
        pm = embedding_bundle["player_metadata"]
        matrix = embedding_bundle["matrix"]
        index_lookup = {name: idx for idx, name in enumerate(pi)}

        # Apply the same boolean filters to build a candidate pool (not just top-20).
        candidates: list[dict[str, Any]] = PLAYER_INDEX["player_list"]

        league = filters.get("league")
        if league:
            candidates = [p for p in candidates if p.get("league") == league]

        nat_region = filters.get("nationality_region")
        if nat_region:
            candidates = [
                p for p in candidates if p.get("nationality_normalized") in nat_region
            ]
        else:
            nationality = filters.get("nationality")
            if nationality:
                nat_key = normalize_text(nationality)
                candidates = [
                    p
                    for p in candidates
                    if (p.get("nationality_normalized") or "") == nat_key
                ]

        positions = filters.get("positions")
        if positions:
            allowed = set(positions)
            candidates = [p for p in candidates if p.get("position") in allowed]

        season_start = filters.get("season_start")
        season_end = filters.get("season_end")
        if season_start is not None and season_end is not None:
            candidates = [
                p
                for p in candidates
                if any(
                    season_start <= year <= season_end
                    for year in p.get("season_years", [])
                )
            ]

        max_age_under = filters.get("max_age_under")
        if max_age_under is not None:
            ref_y = int(filters.get("reference_year") or reference_year_for_queries())
            candidates = [
                p
                for p in candidates
                if passes_max_age_under(
                    max_age_under,
                    ref_y,
                    career_start_year_for_embedding_player(
                        p.get("normalized_name") or "",
                        pm.get(p.get("normalized_name") or "", {}),
                    ),
                )
            ]

        candidate_indices: list[int] = []
        seen_candidate_idx: set[int] = set()
        allowed_meta_positions = set(filters.get("positions") or [])
        for p in candidates:
            key = p.get("normalized_name") or ""
            if key and key in index_lookup:
                if allowed_meta_positions:
                    meta_pos = (pm.get(key, {}) or {}).get("primary_position")
                    if meta_pos not in allowed_meta_positions:
                        continue
                idx = int(index_lookup[key])
                if idx in seen_candidate_idx:
                    continue
                seen_candidate_idx.add(idx)
                candidate_indices.append(idx)

        # Cap the candidate set to keep latency predictable.
        if len(candidate_indices) > 4000:
            candidate_indices = candidate_indices[:4000]

        if candidate_indices:
            sort_by = (filters.get("sort_by") or "goals").strip()
            fk_mode = sort_by == "freekick_score"
            tekky_mode = sort_by == "tekky_score"
            feature_names = _load_feature_names(int(matrix.shape[1]))
            if fk_mode:
                # Build a prototype from the embedding vectors of the best set-piece scorers.
                # This makes SVD reranking reflect the same concept as the non-SVD column.
                def fk_score(p: dict[str, Any]) -> float:
                    goals = p.get("set_piece_goals")
                    shots = p.get("freekick_shots")
                    try:
                        g = float(goals) if goals is not None else 0.0
                    except (TypeError, ValueError):
                        g = 0.0
                    try:
                        s = float(shots) if shots not in (None, "", 0) else 0.0
                    except (TypeError, ValueError):
                        s = 0.0
                    rate = (g / s) if s > 0 else 0.0
                    return g + 10.0 * rate

                exemplar_players = sorted(
                    (p for p in candidates if p.get("set_piece_goals") is not None),
                    key=fk_score,
                    reverse=True,
                )[:50]
                exemplar_indices: list[int] = []
                weights: list[float] = []
                for p in exemplar_players:
                    key = p.get("normalized_name") or ""
                    if key and key in index_lookup:
                        exemplar_indices.append(int(index_lookup[key]))
                        weights.append(max(0.0, fk_score(p)))
                if exemplar_indices:
                    ex_mat = matrix[np.array(exemplar_indices, dtype=int)]
                    w = np.asarray(weights, dtype=float)
                    if float(w.sum()) <= 0:
                        proto = ex_mat.mean(axis=0).astype(float)
                    else:
                        proto = np.average(ex_mat, axis=0, weights=w).astype(float)
                else:
                    cand_mat = matrix[np.array(candidate_indices, dtype=int)]
                    proto = cand_mat.mean(axis=0).astype(float)
                # Also restrict candidates to those with at least some set-piece signal.
                sp_set = set(exemplar_indices)
                candidate_indices = [i for i in candidate_indices if i in sp_set] or candidate_indices
            elif tekky_mode:
                def tekky_score(p: dict[str, Any]) -> float:
                    try:
                        dr = float(p.get("dribbles_completed") or 0.0)
                    except (TypeError, ValueError):
                        dr = 0.0
                    try:
                        kp = float(p.get("key_passes") or 0.0)
                    except (TypeError, ValueError):
                        kp = 0.0
                    try:
                        pp = float(p.get("progressive_passes") or 0.0)
                    except (TypeError, ValueError):
                        pp = 0.0
                    return dr * 1.0 + pp * 0.35 + kp * 0.75

                exemplar_players = sorted(candidates, key=tekky_score, reverse=True)[:50]
                exemplar_indices: list[int] = []
                weights: list[float] = []
                for p in exemplar_players:
                    key = p.get("normalized_name") or ""
                    if key and key in index_lookup:
                        exemplar_indices.append(int(index_lookup[key]))
                        weights.append(max(0.0, tekky_score(p)))
                if exemplar_indices:
                    ex_mat = matrix[np.array(exemplar_indices, dtype=int)]
                    w = np.asarray(weights, dtype=float)
                    if float(w.sum()) <= 0:
                        proto = ex_mat.mean(axis=0).astype(float)
                    else:
                        proto = np.average(ex_mat, axis=0, weights=w).astype(float)
                    ex_set = set(exemplar_indices)
                    candidate_indices = [i for i in candidate_indices if i in ex_set] or candidate_indices
                else:
                    cand_mat = matrix[np.array(candidate_indices, dtype=int)]
                    proto = cand_mat.mean(axis=0).astype(float)
            else:
                # Prototype: unit vector aligned to the feature when available.
                if sort_by in feature_names:
                    proto = np.zeros((len(feature_names),), dtype=float)
                    proto[feature_names.index(sort_by)] = 1.0
                else:
                    cand_mat = matrix[np.array(candidate_indices, dtype=int)]
                    proto = cand_mat.mean(axis=0).astype(float)

            response = _dual_semantic_search(
                proto,
                candidate_indices,
                embedding_bundle,
                "boolean_svd",
                top_k=10,
            )
            if fk_mode:
                ranked = sorted(
                    (p for p in candidates if p.get("set_piece_goals") is not None),
                    key=fk_score,
                    reverse=True,
                )[:10]
                response["results_without_svd"] = [
                    {**serialize_player(p), "similarity_score": None, "search_mode": "boolean_svd"}
                    for p in ranked
                ]
            if tekky_mode:
                ranked = sorted(candidates, key=tekky_score, reverse=True)[:10]
                response["results_without_svd"] = [
                    {**serialize_player(p), "similarity_score": None, "search_mode": "boolean_svd"}
                    for p in ranked
                ]
            if league or allowed_meta_positions:
                for key in ("results", "results_svd", "results_without_svd"):
                    hits = response.get(key) or []
                    for hit in hits:
                        hit["league"] = league
                        if len(allowed_meta_positions) == 1:
                            hit["position"] = next(iter(allowed_meta_positions))
            return response

    return {
        "mode": "boolean",
        "results": boolean_results,
        "results_svd": None,
        "results_without_svd": None,
        "svd_available": False,
        "svd_latent_dimensions": [],
    }
