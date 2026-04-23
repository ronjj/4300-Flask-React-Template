from __future__ import annotations

import copy
import json
from typing import Any


REQUIRED_EVIDENCE_FIELDS = (
    "evidence_id",
    "source_type",
    "player_id",
    "player_name",
    "retrieval_mode",
    "retrieval_score",
    "rank",
    "matched_filters",
    "provenance",
)
OPTIONAL_DROP_ORDER = ("style_matches", "key_stats")


def build_evidence_record(
    hit: dict[str, Any],
    retrieval_mode: str,
    matched_filters: dict[str, Any],
    rank: int,
) -> dict[str, Any]:
    key_stats = {
        key: value
        for key, value in (hit.get("raw_key_stats") or {}).items()
        if value is not None
    }
    return {
        "evidence_id": hit["evidence_id"],
        "source_type": hit["source_type"],
        "player_id": hit.get("player_id"),
        "player_name": hit.get("player_name"),
        "season_id": hit.get("season_id"),
        "season_label": hit.get("season_label"),
        "team": hit.get("team"),
        "league": hit.get("league"),
        "position": hit.get("position"),
        "retrieval_mode": retrieval_mode,
        "retrieval_score": round(float(hit.get("retrieval_score", 0.0)), 6),
        "rank": rank,
        "matched_filters": matched_filters,
        "style_matches": hit.get("style_matches") or [],
        "key_stats": key_stats,
        "provenance": hit["provenance"],
    }


def truncate_evidence_for_budget(
    evidence: list[dict[str, Any]],
    max_chars: int,
) -> list[dict[str, Any]]:
    trimmed = copy.deepcopy(evidence)
    while len(json.dumps(trimmed, sort_keys=True)) > max_chars and trimmed:
        changed = False
        for record in trimmed:
            for field in OPTIONAL_DROP_ORDER:
                if record.get(field):
                    record.pop(field, None)
                    changed = True
                    break
            if changed:
                break
        if changed:
            continue
        trimmed.pop()
    return trimmed


def format_evidence_for_prompt(
    evidence: list[dict[str, Any]],
    max_chars: int = 5000,
) -> str:
    truncated = truncate_evidence_for_budget(evidence, max_chars=max_chars)
    chunks: list[str] = []
    for record in truncated:
        key_stats = ", ".join(
            f"{key}={value}"
            for key, value in sorted((record.get("key_stats") or {}).items())
        ) or "none"
        style_matches = ", ".join(
            f"{item.get('term')}->{item.get('stat_family')}"
            for item in record.get("style_matches") or []
        ) or "none"
        chunks.append(
            "\n".join(
                [
                    f"rank: {record['rank']}",
                    f"evidence_id: {record['evidence_id']}",
                    f"source_type: {record['source_type']}",
                    f"player_name: {record['player_name']}",
                    f"team: {record.get('team') or 'unknown'}",
                    f"league: {record.get('league') or 'unknown'}",
                    f"season_label: {record.get('season_label') or 'unknown'}",
                    f"score: {record['retrieval_score']}",
                    f"style_matches: {style_matches}",
                    f"key_stats: {key_stats}",
                    f"provenance_row_id: {record['provenance'].get('row_id')}",
                ]
            )
        )
    return "\n\n---\n\n".join(chunks)
