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
    anchor_evidence: dict[str, Any] | None = None,
    max_chars: int = 5000,
) -> str:
    candidate_budget = max_chars
    anchor_chunk: str | None = None
    if anchor_evidence:
        anchor_budget = max(600, min(max_chars // 3, 1800))
        anchor_record = truncate_evidence_for_budget([anchor_evidence], max_chars=anchor_budget)[0]
        anchor_chunk = _format_prompt_record(anchor_record, context_label="anchor")
        candidate_budget = max(800, max_chars - len(anchor_chunk) - 40)

    truncated = truncate_evidence_for_budget(evidence, max_chars=candidate_budget)
    candidate_chunks = [
        _format_prompt_record(record, context_label="candidate")
        for record in truncated
    ]
    sections: list[str] = []
    if anchor_chunk:
        sections.append(f"Anchor reference:\n{anchor_chunk}")
    sections.append(
        "Candidate evidence:\n"
        + ("\n\n---\n\n".join(candidate_chunks) if candidate_chunks else "No candidate evidence.")
    )
    return "\n\n====\n\n".join(sections)


def _prompt_key_stats(record: dict[str, Any]) -> dict[str, Any]:
    stats = record.get("raw_key_stats")
    if stats:
        return {
            key: value
            for key, value in stats.items()
            if value is not None
        }
    return {
        key: value
        for key, value in (record.get("key_stats") or {}).items()
        if value is not None
    }


def _format_prompt_record(record: dict[str, Any], context_label: str) -> str:
    key_stats = _prompt_key_stats(record)
    style_matches = record.get("style_matches") or []
    title = "Anchor" if context_label == "anchor" else "Candidate"
    lines = [
        f"{title}: {record['player_name']}",
        "",
        "key_stats:",
    ]
    preferred_order = ("progressive_passes", "key_passes", "assists", "interceptions", "tackles", "minutes")
    emitted: set[str] = set()
    for key in preferred_order:
        if key in key_stats:
            lines.append(f"- {key}: {key_stats[key]}")
            emitted.add(key)
    for key in sorted(key_stats):
        if key in emitted or key == "minutes":
            continue
        lines.append(f"- {key}: {key_stats[key]}")
    if len(lines) > 0 and lines[-1] == "key_stats:":
        lines.append("- none")
    lines.extend(
        [
            "",
            "Context:",
            f"- position: {record.get('position') or 'unknown'}",
            f"- team: {record.get('team') or 'unknown'}",
            f"- season: {record.get('season_label') or 'unknown'}",
            f"- league: {record.get('league') or 'unknown'}",
            "- style_matches:",
        ]
    )
    if style_matches:
        for item in style_matches:
            lines.append(f"- {item.get('term')}")
    else:
        lines.append("- none")
    if context_label == "candidate":
        if "progressive_passes" not in key_stats:
            lines.append("limitation: no progressive_passes recorded in this row")
        if not any(stat in key_stats for stat in ("key_passes", "assists")):
            lines.append("limitation: no key_passes or assists recorded in this row")
        if not any(stat in key_stats for stat in ("tackles", "interceptions")):
            lines.append("limitation: no tackles or interceptions recorded in this row")
    lines.extend(
        [
            f"aggregate_row: {bool(record.get('is_aggregate_row'))}",
            f"provenance_row_id: {record['provenance'].get('row_id')}",
        ]
    )
    return "\n".join(lines)
