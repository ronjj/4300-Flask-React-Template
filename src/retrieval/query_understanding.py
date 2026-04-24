from __future__ import annotations

import os
import re
from typing import Any

try:
    from player_search import (
        NATIONALITY_KEYWORDS,
        POSITION_GROUPS,
        PLAYER_INDEX,
        find_player_by_name,
        nationality_filter_from_text,
        normalize_text,
        query_max_age_under,
        reference_year_for_queries,
        region_nationality_allowlist_from_text,
    )
except ImportError:  # pragma: no cover
    from src.player_search import (
        NATIONALITY_KEYWORDS,
        POSITION_GROUPS,
        PLAYER_INDEX,
        find_player_by_name,
        nationality_filter_from_text,
        normalize_text,
        query_max_age_under,
        reference_year_for_queries,
        region_nationality_allowlist_from_text,
    )


STYLE_TO_STATS = {
    "clinical": ["finishing", "goals_per_game", "shot_on_target_ratio", "goals"],
    "finishing": ["finishing", "goals_per_game", "shot_on_target_ratio", "goals"],
    "progressive passing": ["progression_passing", "progressive_passes", "key_passes"],
    "progressive": ["progression_passing", "progressive_passes", "key_passes"],
    "creative": ["chance_creation", "key_passes", "assists", "assists_per_game"],
    "playmaker": ["chance_creation", "key_passes", "assists", "progressive_passes"],
    "defensive": ["defensive_actions", "tackles", "interceptions", "recoveries", "duels"],
    "ball-winning": ["defensive_actions", "tackles", "interceptions", "recoveries"],
    "press resistant": ["retention", "pass_completion", "progressive_passes", "dribbles_completed"],
    "physical": ["duels", "aerial_duels_won", "recoveries", "minutes"],
}
POSITION_PRIORITY = {
    "Midfielder": 0,
    "Defender": 1,
    "Forward": 2,
    "Goalkeeper": 3,
}
LOW_SAMPLE_KEYWORDS = (
    "young",
    "prospect",
    "emerging",
    "small sample",
    "limited minutes",
    "bench",
    "rotation",
)

COMPARISON_SPLIT_PATTERNS = (
    re.compile(r"\bvs\.?\b", re.IGNORECASE),
    re.compile(r"\bversus\b", re.IGNORECASE),
    re.compile(r"\bcompare\b", re.IGNORECASE),
)
DIRECT_COMPARISON_PATTERNS = (
    re.compile(r"^\s*compare\s+(?P<left>.+?)\s+vs\.?\s+(?P<right>.+?)\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?P<left>.+?)\s+vs\.?\s+(?P<right>.+?)\s*$", re.IGNORECASE),
    re.compile(r"^\s*compare\s+(?P<left>.+?)\s+and\s+(?P<right>.+?)\s*$", re.IGNORECASE),
    re.compile(r"^\s*difference\s+between\s+(?P<left>.+?)\s+and\s+(?P<right>.+?)\s*$", re.IGNORECASE),
    re.compile(r"^\s*who\s+is\s+better,?\s+(?P<left>.+?)\s+or\s+(?P<right>.+?)\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?P<left>.+?)\s+compared\s+to\s+(?P<right>.+?)\s*$", re.IGNORECASE),
)
SIMILARITY_PATTERNS = (
    re.compile(r"players?\s+like\s+(.+?)(?:\s+with|\s+from|\s+in|$)", re.IGNORECASE),
    re.compile(r"similar\s+to\s+(.+?)(?:\s+with|\s+from|\s+in|$)", re.IGNORECASE),
    re.compile(r"most\s+like\s+(.+?)(?:\s+with|\s+from|\s+in|$)", re.IGNORECASE),
)
LEAGUE_KEYWORDS = {
    "la liga": "La Liga",
    "premier league": "Premier League",
    "serie a": "Serie A",
}
POSITION_CANONICAL = {
    "cm": "Midfielder",
    "dm": "Midfielder",
    "am": "Midfielder",
    "st": "Forward",
    "cf": "Forward",
    "fw": "Forward",
    "cb": "Defender",
    "lb": "Defender",
    "rb": "Defender",
    "gk": "Goalkeeper",
}
UNSUPPORTED_STYLE_TERMS = (
    "progressive carries",
    "expected assists",
    "expected goals",
    "xg",
    "xa",
)
KNOWN_PLAYER_NAMES = sorted(
    {
        normalize_text(records[0].get("name"))
        for records in PLAYER_INDEX["players_by_name"].values()
        if records and records[0].get("name")
    },
    key=len,
    reverse=True,
)
DISPLAY_NAME_MAP = {
    normalize_text(records[0].get("name")): records[0].get("name")
    for records in PLAYER_INDEX["players_by_name"].values()
    if records and records[0].get("name")
}


def rewrite_player_chat_query(message: str) -> str:
    SPARK_API_KEY = os.getenv("SPARK_API_KEY")
    if not SPARK_API_KEY:
        return message

    from infosci_spark_client import LLMClient

    client = LLMClient(SPARK_API_KEY=SPARK_API_KEY)
    response = client.chat(
        [
            {
                "role": "system",
                "content": (
                    "Rewrite the user's soccer player search request into a compact retrieval query. "
                    "Keep the original intent, named players, leagues, eras, positions, nationalities, and style terms. "
                    "Do not answer the question. Do not add explanation. Output only the rewritten retrieval query."
                ),
            },
            {"role": "user", "content": message},
        ]
    )
    rewritten = (response.get("content") or "").strip()
    return rewritten or message


def _extract_year_range(query: str) -> tuple[list[int] | None, bool]:
    normalized = normalize_text(query)
    decade_match = re.search(r"\b((?:19|20)\d0)s\b", normalized)
    if decade_match:
        start = int(decade_match.group(1))
        return [start, start + 9], True

    range_match = re.search(r"\b((?:19|20)\d{2})\s*(?:to|-)\s*((?:19|20)\d{2})\b", normalized)
    if range_match:
        return [int(range_match.group(1)), int(range_match.group(2))], True

    years = re.findall(r"\b((?:19|20)\d{2})\b", normalized)
    if len(years) >= 2:
        return [int(years[0]), int(years[1])], True
    if len(years) == 1:
        year = int(years[0])
        return [year, year], True
    return None, False


def _extract_positions(query: str) -> list[str]:
    normalized = normalize_text(query)
    matches: set[str] = set()
    for position, keywords in POSITION_GROUPS.items():
        for keyword in keywords:
            if re.search(rf"\b{re.escape(keyword)}\b", normalized):
                matches.add(position)
                break
    for token, canonical in POSITION_CANONICAL.items():
        if re.search(rf"\b{re.escape(token)}\b", normalized):
            matches.add(canonical)
    return sorted(matches)


def _extract_leagues(query: str) -> list[str]:
    normalized = normalize_text(query)
    leagues = [league for key, league in LEAGUE_KEYWORDS.items() if key in normalized]
    return sorted(set(leagues))


def _extract_style_descriptors(query: str) -> tuple[list[dict[str, Any]], list[str]]:
    normalized = normalize_text(query)
    matches: list[dict[str, Any]] = []
    warnings: list[str] = []
    for term, mapped_stats in STYLE_TO_STATS.items():
        if term in normalized:
            matches.append({"term": term, "stat_family": mapped_stats[0], "stats": mapped_stats[1:]})
    for term in UNSUPPORTED_STYLE_TERMS:
        if term in normalized:
            warnings.append(f"Unsupported style descriptor for v1 scoring: {term}")
    deduped: list[dict[str, Any]] = []
    seen_terms: set[str] = set()
    for item in matches:
        if item["term"] in seen_terms:
            continue
        seen_terms.add(item["term"])
        deduped.append(item)
    return deduped, warnings


def _resolve_player_name(name_fragment: str) -> str | None:
    cleaned = re.sub(r"[^a-zA-Z0-9 .'-]", " ", name_fragment)
    cleaned = " ".join(cleaned.split())
    if not cleaned:
        return None
    rows = find_player_by_name(cleaned)
    if not rows:
        return None
    return rows[0].get("name")


def _extract_players(query: str) -> list[str]:
    normalized = normalize_text(query)
    resolved: list[str] = []

    comparison_players = _extract_comparison_players(query)
    if comparison_players:
        resolved.extend(comparison_players)

    if not resolved and any(pattern.search(query) for pattern in COMPARISON_SPLIT_PATTERNS):
        split_query = re.split(r"\bvs\.?\b|\bversus\b", query, maxsplit=1, flags=re.IGNORECASE)
        if len(split_query) == 2:
            left = re.sub(r"\bcompare\b", "", split_query[0], flags=re.IGNORECASE).strip()
            right = split_query[1].strip()
            for fragment in (left, right):
                resolved_name = _resolve_player_name(fragment)
                if resolved_name:
                    resolved.append(resolved_name)

    if not resolved:
        for pattern in SIMILARITY_PATTERNS:
            match = pattern.search(query)
            if match:
                resolved_name = _resolve_player_name(match.group(1))
                if resolved_name:
                    resolved.append(resolved_name)
                break

    if not resolved:
        for candidate in KNOWN_PLAYER_NAMES:
            if len(candidate) < 5:
                continue
            if re.search(rf"\b{re.escape(candidate)}\b", normalized):
                resolved.append(DISPLAY_NAME_MAP.get(candidate, candidate.title()))
                if len(resolved) >= 2:
                    break

    deduped: list[str] = []
    seen: set[str] = set()
    for name in resolved:
        key = normalize_text(name)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(name)
    return deduped


def _extract_comparison_players(query: str) -> list[str]:
    resolved: list[str] = []
    for pattern in DIRECT_COMPARISON_PATTERNS:
        match = pattern.search(query)
        if not match:
            continue
        left = match.group("left").strip()
        right = match.group("right").strip()
        for fragment in (left, right):
            resolved_name = _resolve_player_name(fragment)
            if resolved_name:
                resolved.append(resolved_name)
        break
    deduped: list[str] = []
    seen: set[str] = set()
    for name in resolved:
        key = normalize_text(name)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(name)
    return deduped


def _infer_intent(normalized: str, players: list[str]) -> str:
    if len(players) >= 2 and re.search(r"\b(vs|versus|compare|comparison)\b", normalized):
        return "comparison"
    if re.search(r"\b(explain|why|how)\b", normalized):
        return "explanation"
    if re.search(r"\b(best|top|rank|ranking)\b", normalized):
        return "ranking"
    return "similarity"


def _infer_mode(
    has_named_player: bool,
    has_explicit_era: bool,
    has_comparison_language: bool,
    has_similarity_language: bool,
    style_descriptor_count: int,
    player_count: int,
) -> str:
    if has_comparison_language and player_count >= 2:
        return "player"
    if has_named_player and has_explicit_era and style_descriptor_count > 0:
        return "hybrid"
    if has_explicit_era:
        return "season"
    if has_similarity_language and has_named_player:
        return "player"
    return "season"


def _infer_anchor_role(player_name: str | None) -> tuple[str | None, list[str]]:
    if not player_name:
        return None, []
    normalized_name = normalize_text(player_name)
    rows = PLAYER_INDEX["players_by_name"].get(normalized_name) or []
    if not rows:
        return None, []

    counts: dict[str, int] = {}
    minutes_by_position: dict[str, float] = {}
    for row in rows:
        position = row.get("position")
        if not position:
            continue
        counts[position] = counts.get(position, 0) + 1
        minutes_by_position[position] = minutes_by_position.get(position, 0.0) + float(row.get("minutes") or 0.0)

    if not counts:
        return None, []

    max_count = max(counts.values())
    candidates = [position for position, count in counts.items() if count == max_count]
    if len(candidates) > 1:
        max_minutes = max(minutes_by_position.get(position, 0.0) for position in candidates)
        candidates = [position for position in candidates if minutes_by_position.get(position, 0.0) == max_minutes]
    if len(candidates) > 1:
        candidates.sort(key=lambda position: POSITION_PRIORITY.get(position, 999))
    anchor_role = candidates[0]
    compatible_positions = sorted({row.get("position") for row in rows if row.get("position") == anchor_role})
    return anchor_role, compatible_positions


def parse_player_chat_query(message: str) -> dict[str, Any]:
    normalized = normalize_text(message)
    year_range, has_explicit_era = _extract_year_range(message)
    players = _extract_players(message)
    style_descriptors, warnings = _extract_style_descriptors(message)
    positions = _extract_positions(message)
    leagues = _extract_leagues(message)
    nationality_region = region_nationality_allowlist_from_text(message)
    nationality = None if nationality_region else nationality_filter_from_text(message)
    minutes_match = re.search(r"\b(?:at least|min(?:imum)?|over)\s+(\d{2,5})\s+minutes\b", normalized)
    minutes_min = int(minutes_match.group(1)) if minutes_match else None
    has_similarity_language = bool(re.search(r"\b(like|similar|most like)\b", normalized))
    has_comparison_language = bool(re.search(r"\b(vs|versus|compare|comparison)\b", normalized))
    has_low_sample_intent = any(keyword in normalized for keyword in LOW_SAMPLE_KEYWORDS)
    comparison_player_names = _extract_comparison_players(message)
    is_direct_player_comparison = len(comparison_player_names) >= 2
    anchor_player_name = None if is_direct_player_comparison else (players[0] if players else None)
    anchor_role_bucket, anchor_compatible_positions = _infer_anchor_role(anchor_player_name)

    filters: dict[str, Any] = {
        "year_range": year_range,
        "positions": positions or None,
        "minutes_min": minutes_min,
        "teams": [],
        "leagues": leagues,
        "nationality": nationality,
    }
    max_age_under = query_max_age_under(message)
    if max_age_under is not None:
        filters["max_age_under"] = max_age_under
        filters["reference_year"] = reference_year_for_queries()
    if nationality_region:
        filters["nationality_region"] = sorted(nationality_region)

    intent = _infer_intent(normalized, players)
    if is_direct_player_comparison:
        intent = "comparison"
    mode = _infer_mode(
        has_named_player=bool(players),
        has_explicit_era=has_explicit_era,
        has_comparison_language=has_comparison_language,
        has_similarity_language=has_similarity_language,
        style_descriptor_count=len(style_descriptors),
        player_count=len(players),
    )
    confidence = 0.35
    confidence += 0.2 if players else 0.0
    confidence += 0.15 if has_explicit_era else 0.0
    confidence += 0.15 if style_descriptors else 0.0
    confidence += 0.15 if positions else 0.0
    confidence = min(confidence, 0.95)

    return {
        "raw_query": message,
        "normalized_query": normalized,
        "intent": intent,
        "mode": mode,
        "comparison_target_count": len(players) if intent == "comparison" else 0,
        "comparison_player_names": comparison_player_names,
        "is_direct_player_comparison": is_direct_player_comparison,
        "entities": {
            "players": players,
            "teams": [],
            "leagues": leagues,
            "nationalities": [nationality] if nationality else [],
        },
        "filters": filters,
        "style_descriptors": [
            {"term": item["term"], "stat_family": item["stat_family"], "stats": item["stats"]}
            for item in style_descriptors
        ],
        "sort_preference": "goals" if intent == "ranking" else None,
        "anchor_player_name": anchor_player_name,
        "anchor_player_normalized_name": normalize_text(anchor_player_name) if anchor_player_name else None,
        "anchor_role_bucket": anchor_role_bucket,
        "anchor_compatible_positions": anchor_compatible_positions,
        "is_named_player_similarity": (not is_direct_player_comparison) and intent == "similarity" and bool(anchor_player_name),
        "has_low_sample_intent": has_low_sample_intent,
        "parser_flags": {
            "has_named_player": bool(players),
            "has_explicit_era": has_explicit_era,
            "has_comparison_language": has_comparison_language,
            "has_similarity_language": has_similarity_language,
        },
        "confidence": confidence,
        "warnings": warnings,
    }


def resolve_filters(
    parsed_filters: dict[str, Any],
    request_filters: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[str]]:
    resolved = dict(parsed_filters)
    warnings: list[str] = []
    request_filters = request_filters or {}
    for key, value in request_filters.items():
        if value in (None, "", [], {}):
            continue
        if key in resolved and resolved.get(key) not in (None, "", [], {}) and resolved.get(key) != value:
            warnings.append(
                f"Request filter {key}={value!r} overrode parsed {key}={resolved.get(key)!r}"
            )
        resolved[key] = value
    return resolved, warnings
