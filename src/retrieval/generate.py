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


def generate_grounded_answer(
    user_message: str,
    retrieval_mode: str,
    evidence: list[dict[str, Any]],
    retrieval_confidence: float,
    max_prompt_chars: int = 5000,
) -> str:
    if not evidence or retrieval_confidence < 0.30:
        return build_no_evidence_answer(user_message)

    api_key = os.getenv("API_KEY")
    if not api_key:
        raise RuntimeError("API_KEY not set — add it to your .env file")

    from infosci_spark_client import LLMClient

    client = LLMClient(api_key=api_key)
    evidence_text = format_evidence_for_prompt(evidence, max_chars=max_prompt_chars)
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You answer soccer player recommendation and comparison questions using only the provided evidence. "
                "Do not invent players, stats, teams, leagues, or seasons. "
                "If the evidence is incomplete, say what is missing. "
                "Keep the answer concise and grounded."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Retrieval mode: {retrieval_mode}\n"
                f"Evidence:\n{evidence_text}\n\n"
                f"User question: {user_message}"
            ),
        },
    ]
    response = client.chat(messages)
    return (response.get("content") or "").strip() or build_no_evidence_answer(user_message)
