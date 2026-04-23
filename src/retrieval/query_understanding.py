from __future__ import annotations

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

COMPARISON_SPLIT_PATTERNS = (
    re.compile(r"\bvs\.?\b", re.IGNORECASE),
    re.compile(r"\bversus\b", re.IGNORECASE),
    re.compile(r"\bcompare\b", re.IGNORECASE),
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

    if any(pattern.search(query) for pattern in COMPARISON_SPLIT_PATTERNS):
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
