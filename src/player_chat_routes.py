from __future__ import annotations

from typing import Any

from flask import jsonify, request

try:
    from retrieval.evidence import build_evidence_record
    from retrieval.generate import generate_grounded_answer
    from retrieval.query_understanding import parse_player_chat_query, resolve_filters, rewrite_player_chat_query
    from retrieval.retrieve import retrieve_comparison_targets, retrieve_ranked_players
except ImportError:  # pragma: no cover
    from src.retrieval.evidence import build_evidence_record
    from src.retrieval.generate import generate_grounded_answer
    from src.retrieval.query_understanding import parse_player_chat_query, resolve_filters, rewrite_player_chat_query
    from src.retrieval.retrieve import retrieve_comparison_targets, retrieve_ranked_players


def register_player_chat_route(app) -> None:
    @app.route("/api/player-chat", methods=["POST"])
    def player_chat():
        data = request.get_json() or {}
        message = (data.get("message") or "").strip()
        if not message:
            return jsonify({"error": "Message is required"}), 400

        rewritten_query = message
        rewrite_warning = None
        try:
            rewritten_query = rewrite_player_chat_query(message).strip() or message
        except RuntimeError as exc:
            rewrite_warning = f"LLM query rewrite unavailable: {exc}"
        except Exception as exc:  # pragma: no cover - defensive route handling
            rewrite_warning = f"LLM query rewrite failed: {exc}"

        parsed_query = parse_player_chat_query(rewritten_query)
        request_filters = data.get("filters") or {}
        resolved_filters, precedence_warnings = resolve_filters(
            parsed_query.get("filters") or {},
            request_filters,
        )
        parsed_query["filters"] = resolved_filters
        if data.get("mode") in {"auto", "player", "season", "hybrid"} and data.get("mode") != "auto":
            parsed_query["mode"] = data["mode"]

        top_k = int(data.get("top_k") or 12)
        max_players = int(data.get("max_players") or 5)
        max_supporting_rows_per_player = int(data.get("max_supporting_rows_per_player") or 2)

        if parsed_query.get("intent") == "comparison":
            retrieval = retrieve_comparison_targets(
                parsed_query,
                resolved_filters,
                top_k_per_target=top_k,
                max_supporting_rows_per_target=max_supporting_rows_per_player,
            )
        else:
            retrieval = retrieve_ranked_players(
                parsed_query,
                resolved_filters,
                top_k=top_k,
                max_players=max_players,
                max_supporting_rows_per_player=max_supporting_rows_per_player,
            )

        warnings = list(parsed_query.get("warnings") or [])
        if rewrite_warning:
            warnings.append(rewrite_warning)
        warnings.extend(precedence_warnings)
        warnings.extend(retrieval.get("warnings") or [])

        evidence = [
            build_evidence_record(
                hit,
                retrieval_mode=retrieval["retrieval_mode"],
                matched_filters=resolved_filters,
                rank=index + 1,
            )
            for index, hit in enumerate(retrieval.get("hits") or [])
        ]
        anchor_evidence = None
        if retrieval.get("anchor_hit"):
            anchor_evidence = build_evidence_record(
                retrieval["anchor_hit"],
                retrieval_mode=retrieval["retrieval_mode"],
                matched_filters=resolved_filters,
                rank=0,
            )
            anchor_evidence["normalized_name"] = retrieval["anchor_hit"].get("normalized_name")
            anchor_evidence["anchor_role_bucket"] = retrieval["anchor_hit"].get("anchor_role_bucket")
            anchor_evidence["is_aggregate_row"] = retrieval["anchor_hit"].get("is_aggregate_row", False)

        try:
            answer, generation_debug = generate_grounded_answer(
                user_message=message,
                retrieval_mode=retrieval["retrieval_mode"],
                evidence=evidence,
                retrieval_confidence=float(retrieval.get("retrieval_confidence", 0.0)),
                anchor_evidence=anchor_evidence,
                comparison_player_names=parsed_query.get("comparison_player_names"),
                include_debug=bool(data.get("debug")),
            )
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 500
        except Exception as exc:  # pragma: no cover - defensive route handling
            return jsonify({"error": f"Spark generation failed: {exc}"}), 502

        response: dict[str, Any] = {
            "answer": answer,
            "rewritten_query": rewritten_query,
            "retrieval_mode": retrieval["retrieval_mode"],
            "applied_filters": resolved_filters,
            "results": retrieval.get("results") or [],
            "evidence": evidence,
            "retrieval_confidence": retrieval.get("retrieval_confidence", 0.0),
            "warnings": warnings,
            "community_discussion": [],
            "debug": retrieval.get("debug") if data.get("debug") else None,
        }
        if data.get("debug") and anchor_evidence is not None:
            response["debug"] = response.get("debug") or {}
            response["debug"]["anchor_evidence"] = anchor_evidence
        if data.get("debug") and generation_debug is not None:
            response["debug"] = response.get("debug") or {}
            response["debug"]["generation"] = generation_debug
        return jsonify(response)
