from __future__ import annotations

import os
from typing import Any

try:
    from retrieval.evidence import format_evidence_for_prompt
except ImportError:  # pragma: no cover
    from src.retrieval.evidence import format_evidence_for_prompt


def build_no_evidence_answer(message: str) -> str:
    return (
        "I could not find enough grounded player evidence to answer that reliably. "
        f"Try narrowing the request, adding a league, era, position, or target player: '{message}'."
    )


def _normalize_stats(record: dict[str, Any]) -> dict[str, Any]:
    if record.get("raw_key_stats"):
        return {
            key: value
            for key, value in (record.get("raw_key_stats") or {}).items()
            if value is not None
        }
    return {
        key: value
        for key, value in (record.get("key_stats") or {}).items()
        if value is not None
    }


def build_grounded_messages(
    user_message: str,
    retrieval_mode: str,
    evidence: list[dict[str, Any]],
    anchor_evidence: dict[str, Any] | None = None,
    comparison_player_names: list[str] | None = None,
    max_prompt_chars: int = 5000,
) -> list[dict[str, Any]]:
    evidence_text = format_evidence_for_prompt(
        evidence,
        anchor_evidence=anchor_evidence,
        max_chars=max_prompt_chars,
    )
    named_similarity = anchor_evidence is not None
    direct_comparison = bool(comparison_player_names)
    comparison_names_text = ", ".join(comparison_player_names or [])
    return [
        {
            "role": "system",
            "content": (
                "You answer soccer player recommendation and comparison questions using only the provided evidence. "
                "Do not invent players, stats, teams, leagues, or seasons. "
                "If key stats are present in the evidence, use them. Do not say stats are unavailable unless the specific evidence record has no key_stats. "
                "When the evidence contains progressive_passes, treat it as evidence of progressive passing. "
                "When the evidence contains key_passes, treat it as evidence of chance creation. "
                "When the evidence contains interceptions or tackles, treat them as defensive activity evidence. "
                "When style_matches are present, use them as direct evidence of the requested play style. "
                "Do not claim evidence is absent if these fields are present. "
                "Mention uncertainty when rows are aggregate multi-year data or sample size is weak. "
                "For named-player similarity, the anchor player is the reference profile. "
                "Similarity is relative, not absolute. "
                "Compare candidate players against the anchor using the provided stats and minutes context. "
                "Identify the closest available matches among the candidates even if none fully matches the anchor. "
                "Do not reject all candidates solely because they are weaker than the anchor. "
                "Usually return the strongest available 3-5 candidates from the provided results. "
                "For each recommended candidate, explain what stats support the similarity and how the player falls short of the anchor. "
                "Candidate summaries must be stat-first. "
                "If multiple relevant stats are present for a candidate, include all of them before any qualitative judgment. "
                "If a stat is present, acknowledge it directly before any qualitative judgment. "
                "READ-BEFORE-WRITE RULE: Before writing any sentence about a candidate, you must first read that candidate's key_stats. "
                "Every candidate sentence MUST include at least one stat copied directly from that candidate's key_stats. "
                "All candidate claims must be derived directly from that candidate's key_stats. "
                "Before summarizing a candidate, read that candidate's key_stats. "
                "If a stat exists in key_stats, it must be treated as present evidence. "
                "If a relevant stat exists in key_stats, include its exact value in the answer. "
                "Prefer phrases like 'closest available matches', 'partial match', and 'shows evidence of ... though below the anchor'. "
                "Do not recommend the anchor player as a result. "
                "Do not claim the anchor player is missing if anchor evidence is present. "
                "Do not infer missing evidence from summaries or flags; use the raw candidate key_stats. "
                "Map progressive_passes to progressive passing evidence. "
                "Map key_passes or assists to chance creation evidence. "
                "Map tackles or interceptions to defensive activity evidence. "
                "Map minutes to sample size and reliability context. "
                "Do not omit available relevant stats when describing a candidate. "
                "Do not say a candidate has 'no stats recorded', 'no evidence', 'lacks data', or 'missing evidence' for a category if the relevant field exists in key_stats. "
                "Do not say a candidate lacks chance creation if key_passes or assists exists. "
                "Do not say a candidate lacks defensive activity if tackles or interceptions exists. "
                "Do not say a candidate lacks progressive passing if progressive_passes exists. "
                "Do not downplay or reinterpret present stats as weak or absent unless the numeric values clearly support that interpretation. "
                "Do not say 'no progressive passing evidence', 'no chance creation evidence', 'cannot compare', or 'no candidates match' unless those relevant fields are absent from all candidates. "
                "If a candidate lacks a requested stat field, describe that row as weaker or incomplete evidence rather than treating the whole query as unsupported. "
                "INVALID OUTPUT CONDITIONS: The answer is invalid if it says 'no stats recorded', 'no evidence', or 'missing data' for a stat present in key_stats. "
                "The answer is invalid if it omits relevant present stats while describing a candidate. "
                "The answer is invalid if it makes a claim about a candidate without citing at least one stat from that candidate's key_stats. "
                "For direct player comparisons, compare only the explicitly named players. "
                "Do not use 'closest available matches' language for direct player comparisons. "
                "Do not recommend or introduce unrelated players in a direct comparison answer. "
                "Use stat-first summaries for each named player, then a comparison section covering passing/progression, chance creation, defensive activity, and data limitations. "
                "If rows are aggregate or multi-year, mention that this limits precision but still answer from the available evidence. "
                "key_stats is non-empty if it contains any fields at all; partial key_stats (some but not all categories) are still valid evidence. "
                "Missing one stat category does not invalidate the rest of the row. "
                "Only say 'no stats recorded' if key_stats is completely empty. "
                "For defensive queries, prefer tackles and interceptions when available; if both are absent, note that direct defensive stats are unavailable in this row but still cite minutes, progressive_passes, goals, and assists. "
                "Do not refuse to summarize a candidate just because defensive stats are missing. "
                "If goals exist in key_stats, include that value in the candidate summary. "
                "For explanatory or 'why' queries, synthesize the answer from the available stats instead of listing empty profiles. "
                "Use available assists, key passes, progressive passes, goals, tackles, interceptions, and minutes naturally in the explanation. "
                "Keep the answer concise and grounded."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Retrieval mode: {retrieval_mode}\n"
                f"Named-player similarity: {'yes' if named_similarity else 'no'}\n"
                f"Direct player comparison: {'yes' if direct_comparison else 'no'}\n"
                f"Named comparison players: {comparison_names_text or 'none'}\n"
                "Answer style for named-player similarity:\n"
                "- Start by naming the anchor player.\n"
                "- Then list the closest available matches in retrieval order unless a lower-ranked candidate clearly has stronger stat-backed style evidence.\n"
                "- For each candidate, read the raw key_stats first and cite the exact supporting stats that are present.\n"
                "- Use this order inside each candidate summary: goals (if present), progressive passing stats, chance creation stats, defensive activity stats, minutes, then shortfall versus anchor.\n"
                "- If defensive stats (tackles, interceptions) are absent for a candidate, write 'direct defensive stats are unavailable in this row' but still cite all other present stats.\n"
                "- Each candidate bullet must cite at least one exact stat copied from key_stats.\n"
                "- If a candidate is weaker than the anchor, say so without rejecting the candidate.\n"
                "- Mention aggregate-data uncertainty briefly without refusing to answer.\n"
                "Answer style for direct player comparison:\n"
                "- Compare only the explicitly named players.\n"
                "- Do not use 'closest available matches'.\n"
                "- Use a side-by-side stat-first structure for each named player.\n"
                "- If evidence is missing for one player, say that clearly and do not substitute another player.\n"
                "Answer style for generic ranking or explanatory queries:\n"
                "- Use every available stat in each row.\n"
                "- Treat partial key_stats as valid evidence, not as empty rows.\n"
                "- If a category is unavailable for a row, say that category is unavailable for that row only.\n"
                "- For why/explanatory questions, explain the pattern using concrete examples from the evidence and their exact stats.\n"
                f"Evidence:\n{evidence_text}\n\n"
                f"User question: {user_message}"
            ),
        },
    ]


def _debug_log_messages(messages: list[dict[str, Any]]) -> None:
    system_prompt = next((message["content"] for message in messages if message.get("role") == "system"), "")
    user_prompt = next((message["content"] for message in messages if message.get("role") == "user"), "")
    print("===== SYSTEM PROMPT =====")
    print(system_prompt)
    print("===== USER PROMPT =====")
    print(user_prompt)
    print("===== END PROMPT =====")


def _candidate_records(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    ordered: list[dict[str, Any]] = []
    for record in evidence:
        player_name = str(record.get("player_name") or "")
        if not player_name or player_name in seen:
            continue
        seen.add(player_name)
        ordered.append(record)
    return ordered


def _candidate_stat_values(record: dict[str, Any]) -> list[str]:
    stats = _normalize_stats(record)
    return [
        str(value)
        for key, value in stats.items()
        if key in {"goals", "progressive_passes", "key_passes", "assists", "interceptions", "tackles", "minutes"}
    ]


def _query_is_explanatory(user_message: str) -> bool:
    lowered = user_message.casefold().strip()
    return lowered.startswith("why ") or lowered.startswith("how ") or " why " in lowered


def _stat_phrases(stats: dict[str, Any]) -> list[str]:
    phrases: list[str] = []
    if "assists" in stats:
        phrases.append(f"{stats['assists']} assists")
    if "key_passes" in stats:
        phrases.append(f"{stats['key_passes']} key passes")
    if "progressive_passes" in stats:
        phrases.append(f"{stats['progressive_passes']} progressive passes")
    if "goals" in stats:
        phrases.append(f"{stats['goals']} goals")
    if "tackles" in stats:
        phrases.append(f"{stats['tackles']} tackles")
    if "interceptions" in stats:
        phrases.append(f"{stats['interceptions']} interceptions")
    if "minutes" in stats:
        phrases.append(f"{stats['minutes']} minutes")
    return phrases


def _invalid_named_similarity_output(
    output: str,
    evidence: list[dict[str, Any]],
) -> tuple[bool, str | None]:
    lowered = output.casefold()
    banned_patterns = (
        "no stats recorded",
        "no recorded statistics",
        "missing data",
        "no evidence",
        "missing evidence",
        "lacks data",
        "key_stats fields for all candidates are empty",
    )
    candidates = _candidate_records(evidence)
    if any(pattern in lowered for pattern in banned_patterns):
        for record in candidates:
            if _candidate_stat_values(record):
                return True, "output used banned missing-data language despite available candidate stats"
    mentioned_candidates = 0
    for record in candidates:
        player_name = str(record.get("player_name") or "")
        if not player_name:
            continue
        if player_name.casefold() not in lowered:
            continue
        mentioned_candidates += 1
        stat_values = _candidate_stat_values(record)
        if stat_values and not any(value in output for value in stat_values):
            return True, f"output mentioned {player_name} without citing available stats"
    if candidates and mentioned_candidates == 0:
        return True, "output did not mention any candidate names"
    return False, None


def _deterministic_named_similarity_answer(
    user_message: str,
    evidence: list[dict[str, Any]],
    anchor_evidence: dict[str, Any],
) -> str:
    anchor_stats = _normalize_stats(anchor_evidence)
    anchor_parts: list[str] = []
    if "progressive_passes" in anchor_stats:
        anchor_parts.append(f"{anchor_stats['progressive_passes']} progressive passes")
    if "key_passes" in anchor_stats:
        anchor_parts.append(f"{anchor_stats['key_passes']} key passes")
    if "interceptions" in anchor_stats:
        anchor_parts.append(f"{anchor_stats['interceptions']} interceptions")
    if "tackles" in anchor_stats:
        anchor_parts.append(f"{anchor_stats['tackles']} tackles")
    if "minutes" in anchor_stats:
        anchor_parts.append(f"over {anchor_stats['minutes']} minutes")

    lines = [
        f"{anchor_evidence.get('player_name')} is the anchor, with " + ", ".join(anchor_parts) + ".",
    ]
    if any(bool(record.get("is_aggregate_row")) for record in evidence[:5]) or bool(anchor_evidence.get("is_aggregate_row")):
        lines.append("These results rely partly on aggregate or multi-year rows, so the comparison is directionally useful but not fully precise.")
    lines.append("")
    lines.append("Closest available matches:")
    for record in _candidate_records(evidence)[:5]:
        stats = _normalize_stats(record)
        stat_parts: list[str] = []
        if "goals" in stats:
            stat_parts.append(f"{stats['goals']} goals")
        if "progressive_passes" in stats:
            stat_parts.append(f"{stats['progressive_passes']} progressive passes")
        if "key_passes" in stats:
            stat_parts.append(f"{stats['key_passes']} key passes")
        elif "assists" in stats:
            stat_parts.append(f"{stats['assists']} assists")
        defensive_parts: list[str] = []
        if "interceptions" in stats:
            defensive_parts.append(f"{stats['interceptions']} interceptions")
        if "tackles" in stats:
            defensive_parts.append(f"{stats['tackles']} tackles")
        if defensive_parts:
            if len(defensive_parts) == 2:
                stat_parts.append(f"{defensive_parts[0]} and {defensive_parts[1]}")
            else:
                stat_parts.extend(defensive_parts)
        else:
            stat_parts.append("direct defensive stats unavailable")
        minutes_text = f" over {stats['minutes']} minutes" if "minutes" in stats else ""
        anchor_key_passes = anchor_stats.get("key_passes")
        comparison_note = "He is a partial midfield match and weaker than the anchor overall."
        if anchor_key_passes is not None and "key_passes" in stats:
            comparison_note = (
                f"He shows stat-backed similarity, but his chance creation is well below {anchor_evidence.get('player_name')}'s {anchor_key_passes} key passes."
            )
        lines.append(
            f"- {record.get('player_name')}: " + ", ".join(stat_parts) + f"{minutes_text}. {comparison_note}"
        )
    return "\n".join(lines)


def _deterministic_direct_comparison_answer(
    comparison_player_names: list[str],
    evidence: list[dict[str, Any]],
    user_message: str,
) -> str:
    grouped = {str(record.get("player_name") or ""): record for record in evidence}
    lines = [f"{' vs '.join(comparison_player_names)}"]
    if any(bool(record.get("is_aggregate_row")) for record in evidence):
        lines.append("Some rows are aggregate or multi-year, so this comparison is directionally useful but not fully precise.")
    for player in comparison_player_names:
        record = grouped.get(player)
        lines.append("")
        lines.append(f"{player}:")
        if not record:
            lines.append("- Evidence unavailable for the requested filters.")
            continue
        stats = _normalize_stats(record)
        if "minutes" in stats:
            lines.append(f"- minutes: {stats['minutes']}")
        if "progressive_passes" in stats:
            lines.append(f"- progressive_passes: {stats['progressive_passes']}")
        if "key_passes" in stats:
            lines.append(f"- key_passes: {stats['key_passes']}")
        elif "assists" in stats:
            lines.append(f"- assists: {stats['assists']}")
        if "interceptions" in stats or "tackles" in stats:
            defensive_parts = []
            if "interceptions" in stats:
                defensive_parts.append(f"interceptions={stats['interceptions']}")
            if "tackles" in stats:
                defensive_parts.append(f"tackles={stats['tackles']}")
            lines.append("- " + ", ".join(defensive_parts))
    lines.append("")
    lines.append("Comparison:")
    if len(comparison_player_names) >= 2 and all(grouped.get(player) for player in comparison_player_names[:2]):
        left = _normalize_stats(grouped[comparison_player_names[0]])
        right = _normalize_stats(grouped[comparison_player_names[1]])
        lines.append(
            f"- Passing/progression: {comparison_player_names[0]} has {left.get('progressive_passes', 'unknown')} progressive passes; "
            f"{comparison_player_names[1]} has {right.get('progressive_passes', 'unknown')}."
        )
        lines.append(
            f"- Chance creation: {comparison_player_names[0]} has {left.get('key_passes', left.get('assists', 'unknown'))}; "
            f"{comparison_player_names[1]} has {right.get('key_passes', right.get('assists', 'unknown'))}."
        )
        lines.append(
            f"- Defensive activity: {comparison_player_names[0]} has interceptions={left.get('interceptions', 'unknown')} and tackles={left.get('tackles', 'unknown')}; "
            f"{comparison_player_names[1]} has interceptions={right.get('interceptions', 'unknown')} and tackles={right.get('tackles', 'unknown')}."
        )
    else:
        lines.append("- Comparison is partial because evidence is missing for one of the requested players.")
    return "\n".join(lines)


def _invalid_direct_comparison_output(
    output: str,
    evidence: list[dict[str, Any]],
    comparison_player_names: list[str],
) -> tuple[bool, str | None]:
    lowered = output.casefold()
    if "closest available matches" in lowered:
        return True, "direct comparison output used similarity language"
    for player in comparison_player_names:
        record = next((item for item in evidence if str(item.get("player_name") or "") == player), None)
        if record and player.casefold() not in lowered:
            return True, f"direct comparison output omitted {player}"
    return False, None


def _invalid_generic_output(
    output: str,
    evidence: list[dict[str, Any]],
) -> tuple[bool, str | None]:
    lowered = output.casefold()
    banned_patterns = (
        "no stats recorded",
        "no recorded statistics",
        "missing data",
        "no evidence",
        "missing evidence",
        "lacks data",
        "key_stats fields for all candidates are empty",
        "key_stats: none",
    )
    records = _candidate_records(evidence)
    if any(pattern in lowered for pattern in banned_patterns):
        for record in records:
            if _candidate_stat_values(record):
                return True, "output used missing-stats language despite available evidence"
    all_stat_values = [value for record in records for value in _candidate_stat_values(record)]
    if all_stat_values and not any(value in output for value in all_stat_values):
        return True, "output did not cite any available stat values"
    mentioned_records = 0
    for record in records:
        player_name = str(record.get("player_name") or "")
        if not player_name or player_name.casefold() not in lowered:
            continue
        mentioned_records += 1
        stat_values = _candidate_stat_values(record)
        if stat_values and not any(value in output for value in stat_values):
            return True, f"output mentioned {player_name} without citing available stats"
    return False, None


def _deterministic_generic_answer(
    user_message: str,
    evidence: list[dict[str, Any]],
) -> str:
    records = _candidate_records(evidence)[:5]
    if not records:
        return build_no_evidence_answer(user_message)

    lines: list[str] = []
    if _query_is_explanatory(user_message):
        has_creation = any(any(stat in _normalize_stats(record) for stat in ("assists", "key_passes")) for record in records)
        has_progression = any("progressive_passes" in _normalize_stats(record) for record in records)
        has_defending = any(any(stat in _normalize_stats(record) for stat in ("tackles", "interceptions")) for record in records)
        explanation_parts: list[str] = []
        if has_creation:
            explanation_parts.append("they create chances through assists and key passes")
        if has_progression:
            explanation_parts.append("they also progress possession through progressive passes")
        if has_defending:
            explanation_parts.append("some rows also show defensive activity through tackles and interceptions")
        if explanation_parts:
            lines.append("The retrieved players look valuable because " + "; ".join(explanation_parts) + ".")
        else:
            lines.append("The retrieved players still show useful production from the available row-level stats, even though some categories are missing.")
    else:
        lines.append("Using the available evidence, these are the strongest stat-backed examples from the retrieved rows.")

    if any(bool(record.get("is_aggregate_row")) for record in records):
        lines.append("Some evidence rows are aggregate or multi-year, so the comparisons are directionally useful but not fully precise.")

    lines.append("")
    lines.append("Examples from the evidence:")
    for record in records:
        stats = _normalize_stats(record)
        stat_phrases = _stat_phrases(stats)
        if not stat_phrases:
            lines.append(f"- {record.get('player_name')}: key_stats unavailable for this row.")
            continue
        summary = f"- {record.get('player_name')}: " + ", ".join(stat_phrases) + "."
        if not any(stat in stats for stat in ("tackles", "interceptions")):
            summary += " Direct defensive stats are unavailable in this row, but the available stats still provide useful context."
        lines.append(summary)
    return "\n".join(lines)


def generate_grounded_answer(
    user_message: str,
    retrieval_mode: str,
    evidence: list[dict[str, Any]],
    retrieval_confidence: float,
    anchor_evidence: dict[str, Any] | None = None,
    comparison_player_names: list[str] | None = None,
    max_prompt_chars: int = 5000,
    include_debug: bool = False,
) -> tuple[str, dict[str, Any] | None]:
    if not evidence or retrieval_confidence < 0.30:
        return build_no_evidence_answer(user_message), None

    api_key = os.getenv("API_KEY")
    if not api_key:
        raise RuntimeError("API_KEY not set — add it to your .env file")

    from infosci_spark_client import LLMClient

    client = LLMClient(api_key=api_key)
    messages = build_grounded_messages(
        user_message=user_message,
        retrieval_mode=retrieval_mode,
        evidence=evidence,
        anchor_evidence=anchor_evidence,
        comparison_player_names=comparison_player_names,
        max_prompt_chars=max_prompt_chars,
    )
    _debug_log_messages(messages)
    response = client.chat(messages)
    answer = (response.get("content") or "").strip() or build_no_evidence_answer(user_message)
    debug_payload = {
        "system_prompt": next((message["content"] for message in messages if message.get("role") == "system"), ""),
        "user_prompt": next((message["content"] for message in messages if message.get("role") == "user"), ""),
        "messages": messages,
    } if include_debug else None
    if comparison_player_names:
        invalid, reason = _invalid_direct_comparison_output(answer, evidence, comparison_player_names)
        if invalid:
            answer = _deterministic_direct_comparison_answer(comparison_player_names, evidence, user_message)
            if debug_payload is not None:
                debug_payload["fallback_reason"] = reason
                debug_payload["fallback_used"] = True
        elif debug_payload is not None:
            debug_payload["fallback_used"] = False
    elif anchor_evidence is not None:
        invalid, reason = _invalid_named_similarity_output(answer, evidence)
        if invalid:
            answer = _deterministic_named_similarity_answer(user_message, evidence, anchor_evidence)
            if debug_payload is not None:
                debug_payload["fallback_reason"] = reason
                debug_payload["fallback_used"] = True
        elif debug_payload is not None:
            debug_payload["fallback_used"] = False
    else:
        invalid, reason = _invalid_generic_output(answer, evidence)
        if invalid:
            answer = _deterministic_generic_answer(user_message, evidence)
            if debug_payload is not None:
                debug_payload["fallback_reason"] = reason
                debug_payload["fallback_used"] = True
        elif debug_payload is not None:
            debug_payload["fallback_used"] = False
    return answer, debug_payload
