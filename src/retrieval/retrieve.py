from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

try:
    from player_search import PLAYER_INDEX, normalize_text, passes_max_age_under
    from retrieval.query_understanding import STYLE_TO_STATS
except ImportError:  # pragma: no cover
    from src.player_search import PLAYER_INDEX, normalize_text, passes_max_age_under
    from src.retrieval.query_understanding import STYLE_TO_STATS


RETRIEVAL_CONFIDENCE_REWRITE_THRESHOLD = 0.55
RETRIEVAL_CONFIDENCE_WARN_THRESHOLD = 0.40
RETRIEVAL_CONFIDENCE_MIN_ACCEPT = 0.30
BASE_FEATURE_ORDER = [
    "goals",
    "assists",
    "shots_on_target",
    "dribbles_completed",
    "minutes",
    "appearances",
    "goals_per_game",
    "assists_per_game",
    "shot_on_target_ratio",
    "progressive_passes",
    "key_passes",
    "pass_completion",
    "tackles",
    "interceptions",
    "recoveries",
    "aerial_duels_won",
    "duels",
]
STAT_FAMILY_FEATURES = {
    "finishing": ["goals", "goals_per_game", "shot_on_target_ratio", "shots_on_target"],
    "progression_passing": ["progressive_passes", "key_passes", "pass_completion"],
    "chance_creation": ["key_passes", "assists", "assists_per_game", "progressive_passes"],
    "defensive_actions": ["tackles", "interceptions", "recoveries", "duels"],
    "retention": ["pass_completion", "dribbles_completed", "progressive_passes"],
    "duels": ["duels", "aerial_duels_won", "recoveries", "minutes"],
}
REQUESTED_STYLE_PRIMARY_STATS = {
    "progression_passing": ["progressive_passes", "key_passes"],
    "chance_creation": ["key_passes", "assists"],
    "defensive_actions": ["interceptions", "tackles"],
    "finishing": ["goals", "shots_on_target"],
    "retention": ["pass_completion", "progressive_passes"],
    "duels": ["duels", "aerial_duels_won"],
}
DEFAULT_NAMED_PLAYER_MINUTES_FLOOR = 600
MINUTES_RELIABILITY_SATURATION = 1800.0


def _safe_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(parsed):
        return 0.0
    return parsed


def _row_matches_filters(row: dict[str, Any], filters: dict[str, Any]) -> bool:
    positions = filters.get("positions")
    if positions and row.get("position") not in set(positions):
        return False

    leagues = filters.get("leagues")
    if leagues and row.get("league") not in set(leagues):
        return False

    teams = filters.get("teams")
    if teams and row.get("team") not in set(teams):
        return False

    nationality = filters.get("nationality")
    if nationality and normalize_text(row.get("nationality")) != normalize_text(nationality):
        return False

    nationality_region = filters.get("nationality_region")
    if nationality_region and row.get("nationality_normalized") not in set(nationality_region):
        return False

    minutes_min = filters.get("minutes_min")
    if minutes_min is not None and (_safe_float(row.get("minutes")) < float(minutes_min)):
        return False

    year_range = filters.get("year_range")
    if year_range:
        row_years = row.get("season_years") or []
        if not any(year_range[0] <= int(year) <= year_range[1] for year in row_years):
            return False

    max_age_under = filters.get("max_age_under")
    if max_age_under is not None:
        if not passes_max_age_under(
            max_age_under,
            int(filters.get("reference_year")),
            min(row.get("season_years") or [9999]),
        ):
            return False

    return True


def _candidate_rows(filters: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in PLAYER_INDEX["player_list"] if _row_matches_filters(row, filters)]


def _sanitize_vector(vector: np.ndarray) -> np.ndarray:
    sanitized = np.nan_to_num(vector.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(sanitized, -5.0, 5.0)


def _vector_from_row(row: dict[str, Any]) -> np.ndarray:
    stat_features = row.get("stat_features") or {}
    vector = np.array([_safe_float(stat_features.get(feature)) for feature in BASE_FEATURE_ORDER], dtype=float)
    return _sanitize_vector(vector)


def _vector_from_profile(profile: dict[str, Any]) -> np.ndarray:
    feature_means = profile.get("feature_means") or {}
    vector = np.array([_safe_float(feature_means.get(feature)) for feature in BASE_FEATURE_ORDER], dtype=float)
    return _sanitize_vector(vector)


def _is_aggregate_row(row: dict[str, Any]) -> bool:
    season_years = row.get("season_years") or []
    season_label = str(row.get("season_label") or "")
    return len(set(season_years)) > 1 or " – " in season_label or "--" in season_label


def _normalized_bucket_vectors(
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, np.ndarray], dict[str, tuple[np.ndarray, np.ndarray]]]:
    buckets: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in candidates:
        buckets[row.get("role_bucket") or row.get("position") or "Unknown"].append(_vector_from_row(row))

    normalized: dict[str, np.ndarray] = {}
    stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for bucket, vectors in buckets.items():
        matrix = np.vstack(vectors)
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        std = np.where(std == 0, 1.0, std)
        normalized_matrix = _sanitize_vector((matrix - mean) / std)
        normalized[bucket] = normalized_matrix
        stats[bucket] = (mean, std)
    return normalized, stats


def _style_weight_vector(style_descriptors: list[dict[str, Any]]) -> np.ndarray:
    weights = np.zeros(len(BASE_FEATURE_ORDER), dtype=float)
    if not style_descriptors:
        weights[: len(BASE_FEATURE_ORDER)] = 1.0
        return weights
    feature_index = {name: idx for idx, name in enumerate(BASE_FEATURE_ORDER)}
    for descriptor in style_descriptors:
        stat_family = descriptor.get("stat_family")
        features = STAT_FAMILY_FEATURES.get(stat_family, []) or descriptor.get("stats", [])
        for feature in features:
            if feature in feature_index:
                weights[feature_index[feature]] += 1.0
    if not weights.any():
        weights[: len(BASE_FEATURE_ORDER)] = 1.0
    return weights


def _query_vector(
    parsed_query: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    weights = _style_weight_vector(parsed_query.get("style_descriptors") or [])
    query_by_bucket: dict[str, np.ndarray] = {}
    target_players = {normalize_text(name) for name in parsed_query.get("entities", {}).get("players", [])}
    for bucket in {row.get("role_bucket") or row.get("position") or "Unknown" for row in candidates}:
        bucket_rows = [
            row for row in candidates if (row.get("role_bucket") or row.get("position") or "Unknown") == bucket
        ]
        target_rows = [row for row in bucket_rows if row.get("normalized_name") in target_players]
        if target_rows:
            target_vectors = np.vstack([_vector_from_row(row) for row in target_rows])
            query_by_bucket[bucket] = target_vectors.mean(axis=0)
        else:
            query_by_bucket[bucket] = weights
    return query_by_bucket


def _normalized_query_vector(
    raw_query_vector: np.ndarray,
    bucket_stats: tuple[np.ndarray, np.ndarray],
    descriptor_only: bool,
) -> np.ndarray:
    if descriptor_only:
        norm = np.linalg.norm(raw_query_vector)
        normalized = raw_query_vector if norm == 0 else raw_query_vector / norm
        return _sanitize_vector(normalized)
    mean, std = bucket_stats
    normalized = (raw_query_vector - mean) / std
    return _sanitize_vector(normalized)


def _descriptor_match_score(row: dict[str, Any], style_descriptors: list[dict[str, Any]]) -> float:
    if not style_descriptors:
        return 0.0
    stat_features = row.get("stat_features") or {}
    positive = 0
    total = 0
    for descriptor in style_descriptors:
        stat_family = descriptor.get("stat_family")
        features = STAT_FAMILY_FEATURES.get(stat_family, []) or descriptor.get("stats", [])
        if not features:
            continue
        total += len(features)
        positive += sum(1 for feature in features if _safe_float(stat_features.get(feature)) > 0)
    return positive / total if total else 0.0


def _style_stat_backing_score(row: dict[str, Any], style_descriptors: list[dict[str, Any]]) -> float:
    if not style_descriptors:
        return 0.0
    stat_features = row.get("stat_features") or {}
    scores: list[float] = []
    for descriptor in style_descriptors:
        primary_stats = REQUESTED_STYLE_PRIMARY_STATS.get(descriptor.get("stat_family"), descriptor.get("stats") or [])
        if not primary_stats:
            continue
        positive = sum(1 for stat in primary_stats if _safe_float(stat_features.get(stat)) > 0)
        scores.append(positive / len(primary_stats))
    return sum(scores) / len(scores) if scores else 0.0


def _metadata_soft_match_score(row: dict[str, Any], parsed_query: dict[str, Any]) -> float:
    score = 0.0
    filters = parsed_query.get("filters") or {}
    if filters.get("positions") and row.get("position") in set(filters["positions"]):
        score += 0.5
    if filters.get("leagues") and row.get("league") in set(filters["leagues"]):
        score += 0.3
    if filters.get("nationality") and normalize_text(row.get("nationality")) == normalize_text(filters["nationality"]):
        score += 0.2
    return min(score, 1.0)


def _minutes_reliability_score(row: dict[str, Any]) -> float:
    return min(_safe_float(row.get("minutes")) / MINUTES_RELIABILITY_SATURATION, 1.0)


def _effective_minutes_floor(parsed_query: dict[str, Any], filters: dict[str, Any]) -> int | None:
    explicit_floor = filters.get("minutes_min")
    if not parsed_query.get("is_named_player_similarity"):
        return int(explicit_floor) if explicit_floor is not None else None
    if parsed_query.get("has_low_sample_intent"):
        return int(explicit_floor) if explicit_floor is not None else None
    if explicit_floor is None:
        return DEFAULT_NAMED_PLAYER_MINUTES_FLOOR
    return max(int(explicit_floor), DEFAULT_NAMED_PLAYER_MINUTES_FLOOR)


def _prepare_candidate_rows(parsed_query: dict[str, Any], filters: dict[str, Any]) -> list[dict[str, Any]]:
    base_candidates = _candidate_rows(filters)
    if not base_candidates:
        return []

    candidates = list(base_candidates)
    if parsed_query.get("is_named_player_similarity"):
        anchor_role_bucket = parsed_query.get("anchor_role_bucket")
        if anchor_role_bucket and not filters.get("positions"):
            candidates = [
                row for row in candidates
                if (row.get("role_bucket") or row.get("position")) == anchor_role_bucket
            ]
        anchor_player = parsed_query.get("anchor_player_normalized_name")
        if anchor_player:
            candidates = [
                row for row in candidates
                if row.get("normalized_name") != anchor_player
            ]

    minutes_floor = _effective_minutes_floor(parsed_query, filters)
    if minutes_floor is not None:
        candidates = [
            row for row in candidates
            if _safe_float(row.get("minutes")) >= float(minutes_floor)
        ]
    return candidates


def _group_player_profiles(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("normalized_name")].append(row)

    profiles: dict[str, dict[str, Any]] = {}
    for normalized_name, player_rows in grouped.items():
        feature_means: dict[str, float] = {}
        for feature in BASE_FEATURE_ORDER:
            values = [
                _safe_float((row.get("stat_features") or {}).get(feature))
                for row in player_rows
                if (row.get("stat_features") or {}).get(feature) is not None
            ]
            feature_means[feature] = sum(values) / len(values) if values else 0.0
        exemplar = player_rows[0]
        profiles[normalized_name] = {
            "player_id": exemplar.get("player_id"),
            "player_name": exemplar.get("name"),
            "position": exemplar.get("position"),
            "league": exemplar.get("league"),
            "team": exemplar.get("team"),
            "role_bucket": exemplar.get("role_bucket"),
            "feature_means": feature_means,
            "rows": player_rows,
        }
    return profiles


def _profile_scores(
    profiles: dict[str, dict[str, Any]],
    parsed_query: dict[str, Any],
) -> dict[str, float]:
    if not profiles:
        return {}
    grouped_by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for profile in profiles.values():
        grouped_by_bucket[profile.get("role_bucket") or profile.get("position") or "Unknown"].append(profile)
    query_vectors = _query_vector(parsed_query, [row for profile in profiles.values() for row in profile["rows"]])
    target_players = parsed_query.get("entities", {}).get("players") or []
    descriptor_only = not bool(target_players)
    scores: dict[str, float] = {}
    for bucket, bucket_profiles in grouped_by_bucket.items():
        matrix = np.vstack([_vector_from_profile(profile) for profile in bucket_profiles])
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        std = np.where(std == 0, 1.0, std)
        normalized_matrix = _sanitize_vector((matrix - mean) / std)
        raw_query = query_vectors.get(bucket)
        if raw_query is None:
            raw_query = _style_weight_vector(parsed_query.get("style_descriptors") or [])
        normalized_query = _normalized_query_vector(raw_query, (mean, std), descriptor_only)
        normalized_query = _sanitize_vector(normalized_query)
        normalized_matrix = _sanitize_vector(normalized_matrix)
        row_scores = cosine_similarity(normalized_query.reshape(1, -1), normalized_matrix)[0]
        min_score = row_scores.min()
        max_score = row_scores.max()
        denom = max_score - min_score or 1.0
        for idx, profile in enumerate(bucket_profiles):
            scores[normalize_text(profile["player_name"])] = float((row_scores[idx] - min_score) / denom)
    return scores


def _style_matches(style_descriptors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"term": item["term"], "stat_family": item["stat_family"]} for item in style_descriptors]


def _build_row_hit(
    row: dict[str, Any],
    parsed_query: dict[str, Any],
    retrieval_score: float,
    stat_similarity: float = 0.0,
    descriptor_match_score: float | None = None,
    metadata_soft_match_score: float | None = None,
    minutes_reliability_score: float | None = None,
    style_stat_backing_score: float | None = None,
) -> dict[str, Any]:
    descriptor_score = (
        _descriptor_match_score(row, parsed_query.get("style_descriptors") or [])
        if descriptor_match_score is None
        else descriptor_match_score
    )
    metadata_score = (
        _metadata_soft_match_score(row, parsed_query)
        if metadata_soft_match_score is None
        else metadata_soft_match_score
    )
    minutes_score = (
        _minutes_reliability_score(row)
        if minutes_reliability_score is None
        else minutes_reliability_score
    )
    style_backing_score = (
        _style_stat_backing_score(row, parsed_query.get("style_descriptors") or [])
        if style_stat_backing_score is None
        else style_stat_backing_score
    )
    return {
        "evidence_id": f"row_{row.get('row_id')}",
        "source_type": "player_row",
        "player_id": row.get("player_id"),
        "player_name": row.get("name"),
        "normalized_name": row.get("normalized_name"),
        "season_id": row.get("season_id"),
        "season_label": row.get("season_label"),
        "team": row.get("team"),
        "league": row.get("league"),
        "position": row.get("position"),
        "retrieval_score": retrieval_score,
        "stat_similarity": stat_similarity,
        "descriptor_match_score": descriptor_score,
        "metadata_soft_match_score": metadata_score,
        "minutes_reliability_score": minutes_score,
        "style_stat_backing_score": style_backing_score,
        "style_matches": _style_matches(parsed_query.get("style_descriptors") or []),
        "raw_key_stats": {
            key: value
            for key, value in (row.get("stat_features") or {}).items()
            if value is not None and key in {"minutes", "goals", "assists", "key_passes", "progressive_passes", "tackles", "interceptions"}
        },
        "provenance": {
            "dataset": row.get("source_dataset"),
            "row_id": row.get("row_id"),
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
        },
        "is_aggregate_row": _is_aggregate_row(row),
        "anchor_role_bucket": parsed_query.get("anchor_role_bucket"),
    }


def _anchor_row_sort_key(
    row: dict[str, Any],
    parsed_query: dict[str, Any],
    filters: dict[str, Any],
) -> tuple[Any, ...]:
    year_range = filters.get("year_range")
    row_years = row.get("season_years") or []
    year_match = 1 if year_range and any(year_range[0] <= int(year) <= year_range[1] for year in row_years) else 0
    if year_range and row_years:
        in_range_years = [year for year in row_years if year_range[0] <= int(year) <= year_range[1]]
        year_distance = min(abs(int(year) - year_range[1]) for year in in_range_years) if in_range_years else 9999
    else:
        year_distance = 9999
    return (
        -year_match,
        -_style_stat_backing_score(row, parsed_query.get("style_descriptors") or []),
        -_descriptor_match_score(row, parsed_query.get("style_descriptors") or []),
        -_safe_float(row.get("minutes")),
        str(row.get("season_label") or ""),
        str(row.get("row_id") or ""),
    )


def build_anchor_evidence(
    parsed_query: dict[str, Any],
    filters: dict[str, Any],
) -> dict[str, Any] | None:
    if not parsed_query.get("is_named_player_similarity"):
        return None
    anchor_name = parsed_query.get("anchor_player_name")
    if not anchor_name:
        return None
    rows = [
        row
        for row in PLAYER_INDEX["player_list"]
        if row.get("normalized_name") == parsed_query.get("anchor_player_normalized_name")
        and _row_matches_filters(row, {**filters, "minutes_min": None})
    ]
    if not rows:
        rows = PLAYER_INDEX.get("players_by_name", {}).get(parsed_query.get("anchor_player_normalized_name"), [])
    if not rows:
        return None
    best_row = sorted(rows, key=lambda row: _anchor_row_sort_key(row, parsed_query, filters))[0]
    return _build_row_hit(
        best_row,
        parsed_query,
        retrieval_score=1.0,
        stat_similarity=1.0,
    )


def _rank_rows(
    parsed_query: dict[str, Any],
    filters: dict[str, Any],
    top_k: int,
) -> list[dict[str, Any]]:
    candidates = _prepare_candidate_rows(parsed_query, filters)
    if not candidates:
        return []
    normalized_by_bucket, bucket_stats = _normalized_bucket_vectors(candidates)
    query_vectors = _query_vector(parsed_query, candidates)
    target_players = parsed_query.get("entities", {}).get("players") or []
    descriptor_only = not bool(target_players)

    scored_hits: list[dict[str, Any]] = []
    for bucket, normalized_matrix in normalized_by_bucket.items():
        bucket_rows = [
            row for row in candidates if (row.get("role_bucket") or row.get("position") or "Unknown") == bucket
        ]
        raw_query = query_vectors[bucket]
        normalized_query = _normalized_query_vector(raw_query, bucket_stats[bucket], descriptor_only)
        normalized_query = _sanitize_vector(normalized_query)
        normalized_matrix = _sanitize_vector(normalized_matrix)
        similarities = cosine_similarity(normalized_query.reshape(1, -1), normalized_matrix)[0]
        min_score = similarities.min()
        max_score = similarities.max()
        denom = max_score - min_score or 1.0
        for idx, row in enumerate(bucket_rows):
            stat_similarity = float((similarities[idx] - min_score) / denom)
            descriptor_match_score = _descriptor_match_score(row, parsed_query.get("style_descriptors") or [])
            metadata_soft_match_score = _metadata_soft_match_score(row, parsed_query)
            minutes_reliability_score = _minutes_reliability_score(row)
            style_stat_backing_score = _style_stat_backing_score(row, parsed_query.get("style_descriptors") or [])
            final_row_score = (
                0.55 * stat_similarity
                + 0.10 * descriptor_match_score
                + 0.10 * metadata_soft_match_score
                + 0.20 * minutes_reliability_score
                + 0.05 * style_stat_backing_score
            )
            hit = _build_row_hit(
                row,
                parsed_query,
                retrieval_score=final_row_score,
                stat_similarity=stat_similarity,
                descriptor_match_score=descriptor_match_score,
                metadata_soft_match_score=metadata_soft_match_score,
                minutes_reliability_score=minutes_reliability_score,
                style_stat_backing_score=style_stat_backing_score,
            )
            scored_hits.append(hit)
    scored_hits.sort(
        key=lambda hit: (
            -float(hit["retrieval_score"]),
            -float(hit["descriptor_match_score"]),
            hit["player_name"].casefold(),
        )
    )
    return scored_hits[:top_k]


def _group_player_results(
    row_hits: list[dict[str, Any]],
    parsed_query: dict[str, Any],
    max_players: int,
    max_supporting_rows_per_player: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for hit in row_hits:
        grouped[normalize_text(hit["player_name"])].append(hit)

    profile_source_rows = [
        row
        for row in PLAYER_INDEX["player_list"]
        if normalize_text(row.get("name")) in grouped
    ]
    if parsed_query.get("is_named_player_similarity") and parsed_query.get("anchor_role_bucket"):
        anchor_role_bucket = parsed_query["anchor_role_bucket"]
        profile_source_rows = [
            row for row in profile_source_rows
            if (row.get("role_bucket") or row.get("position")) == anchor_role_bucket
        ]
    profiles = _group_player_profiles(profile_source_rows)
    profile_scores = _profile_scores(profiles, parsed_query)
    results: list[dict[str, Any]] = []
    for normalized_name, hits in grouped.items():
        hits.sort(key=lambda hit: float(hit["retrieval_score"]), reverse=True)
        best_row_score = float(hits[0]["retrieval_score"])
        avg_top2_row_score = (
            (best_row_score + float(hits[1]["retrieval_score"])) / 2.0
            if len(hits) > 1
            else best_row_score
        )
        player_profile_score = float(profile_scores.get(normalized_name, 0.0))
        final_player_score = (
            0.6 * best_row_score
            + 0.3 * avg_top2_row_score
            + 0.1 * player_profile_score
        )
        exemplar = hits[0]
        results.append(
            {
                "player_id": exemplar.get("player_id"),
                "player_name": exemplar.get("player_name"),
                "position": exemplar.get("position"),
                "league": exemplar.get("league"),
                "team": exemplar.get("team"),
                "final_player_score": final_player_score,
                "best_row_score": best_row_score,
                "avg_top2_row_score": avg_top2_row_score,
                "player_profile_score": player_profile_score,
                "supporting_rows": hits[:max_supporting_rows_per_player],
            }
        )
    results.sort(
        key=lambda result: (
            -float(result["final_player_score"]),
            -float(result["best_row_score"]),
            -float(result["avg_top2_row_score"]),
            result["player_name"].casefold(),
        )
    )
    return results[:max_players]


def _compute_retrieval_confidence(
    parser_confidence: float,
    row_hits: list[dict[str, Any]],
) -> float:
    if not row_hits:
        return 0.35 * parser_confidence
    top1 = float(row_hits[0]["retrieval_score"])
    top2 = float(row_hits[1]["retrieval_score"]) if len(row_hits) > 1 else 0.0
    normalized_top_score = max(0.0, min(top1, 1.0))
    score_margin = max(0.0, min(top1 - top2, 1.0))
    return (
        0.35 * parser_confidence
        + 0.45 * normalized_top_score
        + 0.20 * score_margin
    )


def retrieve_comparison_targets(
    parsed_query: dict[str, Any],
    filters: dict[str, Any],
    top_k_per_target: int = 8,
    max_supporting_rows_per_target: int = 2,
) -> dict[str, Any]:
    target_players = parsed_query.get("entities", {}).get("players") or []
    warnings: list[str] = []
    if len(target_players) < 2:
        warnings.append("Comparison queries require at least two named players; falling back to standard retrieval.")
        fallback = retrieve_ranked_players(
            parsed_query,
            filters,
            top_k=top_k_per_target,
            max_players=2,
            max_supporting_rows_per_player=max_supporting_rows_per_target,
        )
        fallback["warnings"].extend(warnings)
        return fallback

    rows = _candidate_rows(filters)
    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        normalized_name = normalize_text(row.get("name"))
        if normalized_name in {normalize_text(player) for player in target_players}:
            grouped_rows[normalized_name].append(row)

    comparison_hits: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for player in target_players:
        normalized_target = normalize_text(player)
        player_rows = grouped_rows.get(normalized_target, [])
        player_rows.sort(
            key=lambda row: (
                -_descriptor_match_score(row, parsed_query.get("style_descriptors") or []),
                -_safe_float(row.get("minutes")),
            )
        )
        supporting = player_rows[:max_supporting_rows_per_target]
        player_result = {
            "player_id": supporting[0].get("player_id") if supporting else None,
            "player_name": player,
            "position": supporting[0].get("position") if supporting else None,
            "league": supporting[0].get("league") if supporting else None,
            "team": supporting[0].get("team") if supporting else None,
            "final_player_score": 0.0,
            "best_row_score": 0.0,
            "avg_top2_row_score": 0.0,
            "player_profile_score": 0.0,
            "supporting_rows": [],
        }
        for idx, row in enumerate(supporting):
            hit = {
                **_build_row_hit(
                    row,
                    parsed_query,
                    retrieval_score=1.0 - (idx * 0.05),
                    stat_similarity=0.0,
                )
            }
            comparison_hits.append(hit)
            player_result["supporting_rows"].append(hit)
        results.append(player_result)

    retrieval_confidence = _compute_retrieval_confidence(parsed_query.get("confidence", 0.0), comparison_hits)
    return {
        "retrieval_mode": "player",
        "results": results,
        "hits": comparison_hits,
        "retrieval_confidence": retrieval_confidence,
        "warnings": warnings,
        "debug": {
            "candidate_count": len(rows),
            "comparison_targets": target_players,
        },
    }


def retrieve_ranked_players(
    parsed_query: dict[str, Any],
    filters: dict[str, Any],
    top_k: int = 12,
    max_players: int = 5,
    max_supporting_rows_per_player: int = 2,
) -> dict[str, Any]:
    row_hits = _rank_rows(parsed_query, filters, top_k=top_k)
    anchor_hit = build_anchor_evidence(parsed_query, filters)
    results = _group_player_results(
        row_hits,
        parsed_query,
        max_players=max_players,
        max_supporting_rows_per_player=max_supporting_rows_per_player,
    )
    retrieval_confidence = _compute_retrieval_confidence(parsed_query.get("confidence", 0.0), row_hits)
    warnings: list[str] = []
    top_hits = row_hits[: min(3, len(row_hits))]
    if any(hit.get("is_aggregate_row") for hit in top_hits) or (anchor_hit and anchor_hit.get("is_aggregate_row")):
        warnings.append("Evidence uses aggregate multi-year row data for some candidates.")
        retrieval_confidence = max(0.0, retrieval_confidence - 0.08)
    if retrieval_confidence < RETRIEVAL_CONFIDENCE_WARN_THRESHOLD:
        warnings.append("Low retrieval confidence; answer may be incomplete.")
    if not row_hits:
        warnings.append("No evidence rows matched the resolved filters.")
    return {
        "retrieval_mode": parsed_query.get("mode") or "season",
        "results": results,
        "hits": row_hits,
        "anchor_hit": anchor_hit,
        "retrieval_confidence": retrieval_confidence,
        "warnings": warnings,
        "debug": {
            "candidate_count": len(_prepare_candidate_rows(parsed_query, filters)),
            "row_hit_count": len(row_hits),
            "rewrite_threshold": RETRIEVAL_CONFIDENCE_REWRITE_THRESHOLD,
            "anchor_role_bucket": parsed_query.get("anchor_role_bucket"),
            "anchor_player_name": parsed_query.get("anchor_player_name"),
            "minutes_floor": _effective_minutes_floor(parsed_query, filters),
        },
    }
