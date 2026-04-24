from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from flask import Flask

from src.player_chat_routes import register_player_chat_route
from src.retrieval.evidence import format_evidence_for_prompt, truncate_evidence_for_budget
from src.retrieval.generate import (
    build_grounded_messages,
    generate_grounded_answer,
)
from src.retrieval.query_understanding import parse_player_chat_query, resolve_filters, rewrite_player_chat_query
from src.retrieval.retrieve import build_anchor_evidence, retrieve_ranked_players


class QueryUnderstandingTests(unittest.TestCase):
    @patch("src.retrieval.query_understanding.os.getenv", return_value="fake-key")
    @patch("infosci_spark_client.LLMClient")
    def test_rewrite_player_chat_query_returns_llm_rewrite(self, mock_client_cls, _mock_getenv) -> None:
        mock_client = mock_client_cls.return_value
        mock_client.chat.return_value = {"content": "creative Argentine midfielders in La Liga"}

        rewritten = rewrite_player_chat_query("Which Argentine midfielders in La Liga are the most creative?")

        self.assertEqual(rewritten, "creative Argentine midfielders in La Liga")

    @patch("src.retrieval.query_understanding.os.getenv", return_value="fake-key")
    @patch("infosci_spark_client.LLMClient")
    def test_rewrite_player_chat_query_falls_back_to_original_message_on_empty_output(self, mock_client_cls, _mock_getenv) -> None:
        mock_client = mock_client_cls.return_value
        mock_client.chat.return_value = {"content": "   "}

        original = "Which Argentine midfielders in La Liga are the most creative?"
        rewritten = rewrite_player_chat_query(original)

        self.assertEqual(rewritten, original)

    def test_similarity_query_parses_mode_entities_and_filters(self) -> None:
        parsed = parse_player_chat_query(
            "players like Luka Modric with progressive passing from the 2010s"
        )

        self.assertEqual(parsed["intent"], "similarity")
        self.assertEqual(parsed["mode"], "hybrid")
        self.assertIn("Luka Modric", parsed["entities"]["players"])
        self.assertEqual(parsed["filters"]["year_range"], [2010, 2019])
        self.assertTrue(any(item["term"] == "progressive passing" for item in parsed["style_descriptors"]))
        self.assertTrue(parsed["is_named_player_similarity"])
        self.assertEqual(parsed["anchor_player_name"], "Luka Modric")
        self.assertEqual(parsed["anchor_role_bucket"], "Midfielder")
        self.assertFalse(parsed["has_low_sample_intent"])

    def test_low_sample_intent_detected_from_keywords(self) -> None:
        parsed = parse_player_chat_query(
            "young prospects like Luka Modric with progressive passing and limited minutes"
        )

        self.assertTrue(parsed["has_low_sample_intent"])

    def test_anchor_role_inference_uses_minutes_then_priority_tiebreak(self) -> None:
        fake_index = {
            "players_by_name": {
                "test player": [
                    {"name": "Test Player", "position": "Defender", "minutes": 200},
                    {"name": "Test Player", "position": "Midfielder", "minutes": 500},
                ]
            }
        }
        with patch("src.retrieval.query_understanding.PLAYER_INDEX", fake_index), patch(
            "src.retrieval.query_understanding.find_player_by_name",
            return_value=[{"name": "Test Player"}],
        ):
            parsed = parse_player_chat_query("players like Test Player with progressive passing")

        self.assertEqual(parsed["anchor_role_bucket"], "Midfielder")

    def test_request_filters_override_parsed_filters_with_warning(self) -> None:
        resolved, warnings = resolve_filters(
            {
                "positions": ["Midfielder"],
                "year_range": [2010, 2019],
            },
            {"positions": ["Forward"]},
        )

        self.assertEqual(resolved["positions"], ["Forward"])
        self.assertEqual(resolved["year_range"], [2010, 2019])
        self.assertEqual(len(warnings), 1)
        self.assertIn("overrode parsed", warnings[0])

    def test_direct_player_comparison_patterns_parse_correctly(self) -> None:
        def fake_find(name: str) -> list[dict[str, Any]]:
            normalized = name.casefold()
            if "modric" in normalized:
                return [{"name": "Luka Modric"}]
            if "kroos" in normalized:
                return [{"name": "Toni Kroos"}]
            return []

        with patch("src.retrieval.query_understanding.find_player_by_name", side_effect=fake_find):
            for query in (
                "compare Luka Modric vs Toni Kroos in the 2010s",
                "Luka Modric vs Toni Kroos",
                "compare Luka Modric and Toni Kroos",
                "difference between Luka Modric and Toni Kroos",
            ):
                parsed = parse_player_chat_query(query)
                self.assertTrue(parsed["is_direct_player_comparison"])
                self.assertEqual(parsed["comparison_player_names"], ["Luka Modric", "Toni Kroos"])
                self.assertFalse(parsed["is_named_player_similarity"])


class EvidenceBudgetTests(unittest.TestCase):
    def test_truncation_drops_style_matches_before_key_stats_then_rows(self) -> None:
        evidence = [
            {
                "evidence_id": "ev1",
                "source_type": "player_row",
                "player_id": "1",
                "player_name": "One",
                "retrieval_mode": "season",
                "retrieval_score": 0.9,
                "rank": 1,
                "matched_filters": {"positions": ["Midfielder"]},
                "style_matches": [{"term": "creative", "stat_family": "chance_creation"}],
                "key_stats": {"goals": 5, "assists": 10},
                "provenance": {"row_id": "r1"},
            },
            {
                "evidence_id": "ev2",
                "source_type": "player_row",
                "player_id": "2",
                "player_name": "Two",
                "retrieval_mode": "season",
                "retrieval_score": 0.8,
                "rank": 2,
                "matched_filters": {"positions": ["Midfielder"]},
                "style_matches": [{"term": "creative", "stat_family": "chance_creation"}],
                "key_stats": {"goals": 3, "assists": 7},
                "provenance": {"row_id": "r2"},
            },
        ]

        trimmed = truncate_evidence_for_budget(evidence, max_chars=500)

        self.assertEqual(len(trimmed), 1)
        self.assertNotIn("style_matches", trimmed[0])
        self.assertIn("key_stats", trimmed[0])
        self.assertEqual(trimmed[0]["evidence_id"], "ev1")
        self.assertEqual(trimmed[0]["player_name"], "One")

    def test_prompt_format_includes_semantic_stat_labels(self) -> None:
        anchor_evidence = {
            "evidence_id": "ev_modric",
            "source_type": "player_row",
            "player_id": "64",
            "player_name": "Luka Modric",
            "retrieval_mode": "hybrid",
            "retrieval_score": 1.0,
            "rank": 0,
            "matched_filters": {"year_range": [2010, 2019]},
            "style_matches": [{"term": "progressive passing", "stat_family": "progression_passing"}],
            "key_stats": {
                "minutes": 22457,
                "progressive_passes": 10560,
                "key_passes": 496,
                "interceptions": 339,
                "tackles": 354,
            },
            "provenance": {"row_id": "laliga:64:2014-2024"},
            "is_aggregate_row": True,
        }
        evidence = [
            {
                "evidence_id": "ev_candidate",
                "source_type": "player_row",
                "player_id": "1",
                "player_name": "Candidate Mid",
                "retrieval_mode": "hybrid",
                "retrieval_score": 0.91,
                "rank": 1,
                "matched_filters": {"year_range": [2010, 2019]},
                "style_matches": [{"term": "progressive passing", "stat_family": "progression_passing"}],
                "key_stats": {
                    "minutes": 1800,
                    "progressive_passes": 440,
                    "key_passes": 41,
                    "interceptions": 45,
                    "tackles": 38,
                },
                "provenance": {"row_id": "laliga:1:2017-2018"},
            }
        ]

        prompt = format_evidence_for_prompt(evidence, anchor_evidence=anchor_evidence)
        self.assertIn("Anchor reference:", prompt)
        self.assertIn("Candidate evidence:", prompt)
        self.assertIn("Player: Candidate Mid", prompt)
        self.assertIn("Evidence ID: ev_candidate", prompt)
        self.assertIn("Position: unknown", prompt)
        self.assertIn("Team: unknown", prompt)
        self.assertIn("Season/row: unknown", prompt)
        self.assertIn("Style match: progressive passing -> progression_passing", prompt)
        self.assertIn("Key stats: 41 key passes, 440 progressive passes, 45 interceptions, 38 tackles, 1800 minutes", prompt)

    def test_generation_prompt_requires_using_progressive_passes_evidence(self) -> None:
        evidence = [
            {
                "evidence_id": "ev1",
                "source_type": "player_row",
                "player_id": "1",
                "player_name": "One",
                "retrieval_mode": "hybrid",
                "retrieval_score": 0.9,
                "rank": 1,
                "matched_filters": {},
                "style_matches": [{"term": "progressive passing", "stat_family": "progression_passing"}],
                "key_stats": {"progressive_passes": 20, "key_passes": 5, "minutes": 1600},
                "provenance": {"row_id": "r1"},
            }
        ]

        messages = build_grounded_messages(
            user_message="players like Luka Modric with progressive passing",
            retrieval_mode="hybrid",
            evidence=evidence,
            anchor_evidence={
                "evidence_id": "anchor_ev",
                "source_type": "player_row",
                "player_id": "64",
                "player_name": "Luka Modric",
                "retrieval_mode": "hybrid",
                "retrieval_score": 1.0,
                "rank": 0,
                "matched_filters": {},
                "style_matches": [{"term": "progressive passing", "stat_family": "progression_passing"}],
                "key_stats": {"progressive_passes": 100, "key_passes": 15, "minutes": 2200},
                "provenance": {"row_id": "anchor-row"},
                "is_aggregate_row": True,
            },
        )
        self.assertIn("Do not claim evidence is absent if these fields are present", messages[0]["content"])
        self.assertIn("Do not claim the anchor player is missing if anchor evidence is present", messages[0]["content"])
        self.assertIn("Similarity is relative, not absolute", messages[0]["content"])
        self.assertIn("Do not reject all candidates solely because they are weaker than the anchor", messages[0]["content"])
        self.assertIn("All candidate claims must be derived directly from that candidate's key_stats", messages[0]["content"])
        self.assertIn("If a relevant stat exists in key_stats, include its exact value in the answer", messages[0]["content"])
        self.assertIn("READ-BEFORE-WRITE RULE", messages[0]["content"])
        self.assertIn("Every candidate sentence MUST include at least one stat copied directly", messages[0]["content"])
        self.assertIn("INVALID OUTPUT CONDITIONS", messages[0]["content"])
        self.assertIn("Do not say a candidate has 'no stats recorded'", messages[0]["content"])
        self.assertIn("Do not say a candidate lacks defensive activity if tackles or interceptions exists", messages[0]["content"])
        self.assertIn("Do not say 'no progressive passing evidence'", messages[0]["content"])
        self.assertIn("progressive passes", messages[1]["content"])
        self.assertIn("Anchor reference:", messages[1]["content"])
        self.assertIn("Candidate evidence:", messages[1]["content"])
        self.assertIn("closest available matches", messages[1]["content"])
        self.assertIn("read the raw key_stats first", messages[1]["content"])
        self.assertIn("Key stats: 5 key passes, 20 progressive passes, 1600 minutes", messages[1]["content"])

    def test_candidate_prompt_block_includes_all_available_relevant_stats(self) -> None:
        prompt = format_evidence_for_prompt(
            [
                {
                    "evidence_id": "ev_canas",
                    "source_type": "player_row",
                    "player_id": "1639",
                    "player_name": "Jose Alberto Canas Ruiz-Herrera",
                    "retrieval_mode": "hybrid",
                    "retrieval_score": 0.79,
                    "rank": 1,
                    "matched_filters": {"year_range": [2010, 2019]},
                    "style_matches": [{"term": "progressive passing", "stat_family": "progression_passing"}],
                    "key_stats": {
                        "progressive_passes": 562,
                        "key_passes": 3,
                        "interceptions": 98,
                        "tackles": 73,
                        "minutes": 2897,
                    },
                    "provenance": {"row_id": "la-liga:1639:2014-2015"},
                }
            ],
            anchor_evidence={
                "evidence_id": "ev_modric",
                "source_type": "player_row",
                "player_id": "64",
                "player_name": "Luka Modric",
                "retrieval_mode": "hybrid",
                "retrieval_score": 1.0,
                "rank": 0,
                "matched_filters": {"year_range": [2010, 2019]},
                "style_matches": [{"term": "progressive passing", "stat_family": "progression_passing"}],
                "key_stats": {"progressive_passes": 10560, "key_passes": 496, "minutes": 22457},
                "provenance": {"row_id": "la-liga:64:2014-2024"},
            },
        )

        self.assertIn("Player: Jose Alberto Canas Ruiz-Herrera", prompt)
        self.assertIn("Key stats: 3 key passes, 562 progressive passes, 98 interceptions, 73 tackles, 2897 minutes", prompt)
        self.assertNotIn("candidate_progressive_passing_evidence_present", prompt)
        self.assertNotIn("candidate chance creation evidence:", prompt)
        self.assertNotIn("candidate defensive activity evidence:", prompt)
        self.assertNotIn("candidate_stat_backed_style_evidence", prompt)

    def test_modric_candidate_blocks_keep_raw_stats_for_canas_kranevitter_and_agbo(self) -> None:
        prompt = format_evidence_for_prompt(
            [
                {
                    "evidence_id": "ev_canas",
                    "source_type": "player_row",
                    "player_id": "1639",
                    "player_name": "Jose Alberto Canas Ruiz-Herrera",
                    "retrieval_mode": "hybrid",
                    "retrieval_score": 0.79,
                    "rank": 1,
                    "matched_filters": {"year_range": [2010, 2019]},
                    "style_matches": [{"term": "progressive passing", "stat_family": "progression_passing"}],
                    "key_stats": {"progressive_passes": 562, "key_passes": 3, "interceptions": 98, "tackles": 73, "minutes": 2897},
                    "provenance": {"row_id": "canas-row"},
                },
                {
                    "evidence_id": "ev_kranevitter",
                    "source_type": "player_row",
                    "player_id": "4214",
                    "player_name": "Matias Kranevitter",
                    "retrieval_mode": "hybrid",
                    "retrieval_score": 0.77,
                    "rank": 2,
                    "matched_filters": {"year_range": [2010, 2019]},
                    "style_matches": [{"term": "progressive passing", "stat_family": "progression_passing"}],
                    "key_stats": {"progressive_passes": 279, "key_passes": 3, "interceptions": 37, "tackles": 45, "minutes": 1608},
                    "provenance": {"row_id": "kranevitter-row"},
                },
                {
                    "evidence_id": "ev_agbo",
                    "source_type": "player_row",
                    "player_id": "1739",
                    "player_name": "Uche Henry Agbo",
                    "retrieval_mode": "hybrid",
                    "retrieval_score": 0.75,
                    "rank": 3,
                    "matched_filters": {"year_range": [2010, 2019]},
                    "style_matches": [{"term": "progressive passing", "stat_family": "progression_passing"}],
                    "key_stats": {"progressive_passes": 619, "key_passes": 27, "interceptions": 70, "tackles": 94, "minutes": 3016},
                    "provenance": {"row_id": "agbo-row"},
                },
            ],
            anchor_evidence={
                "evidence_id": "ev_modric",
                "source_type": "player_row",
                "player_id": "64",
                "player_name": "Luka Modric",
                "retrieval_mode": "hybrid",
                "retrieval_score": 1.0,
                "rank": 0,
                "matched_filters": {"year_range": [2010, 2019]},
                "style_matches": [{"term": "progressive passing", "stat_family": "progression_passing"}],
                "key_stats": {"progressive_passes": 10560, "key_passes": 496, "interceptions": 339, "tackles": 354, "minutes": 22457},
                "provenance": {"row_id": "modric-row"},
            },
        )
        self.assertIn("Key stats: 3 key passes, 562 progressive passes, 98 interceptions, 73 tackles, 2897 minutes", prompt)
        self.assertIn("Key stats: 3 key passes, 279 progressive passes, 37 interceptions, 45 tackles, 1608 minutes", prompt)
        self.assertIn("Key stats: 27 key passes, 619 progressive passes, 70 interceptions, 94 tackles, 3016 minutes", prompt)

    def test_prompt_marks_key_stats_unavailable_only_for_statless_row(self) -> None:
        prompt = format_evidence_for_prompt(
            [
                {
                    "evidence_id": "ev_full",
                    "source_type": "player_row",
                    "player_id": "1",
                    "player_name": "Stat Player",
                    "retrieval_mode": "hybrid",
                    "retrieval_score": 0.9,
                    "rank": 1,
                    "style_matches": [],
                    "key_stats": {"assists": 4, "minutes": 1200},
                    "provenance": {"row_id": "full-row"},
                },
                {
                    "evidence_id": "ev_empty",
                    "source_type": "player_row",
                    "player_id": "2",
                    "player_name": "Empty Player",
                    "retrieval_mode": "hybrid",
                    "retrieval_score": 0.8,
                    "rank": 2,
                    "style_matches": [],
                    "key_stats": {},
                    "provenance": {"row_id": "empty-row"},
                },
            ]
        )
        self.assertIn("Player: Stat Player", prompt)
        self.assertIn("Key stats: 4 assists, 1200 minutes", prompt)
        self.assertIn("Player: Empty Player", prompt)
        self.assertIn("key_stats: unavailable", prompt)
        self.assertNotIn("key_stats: none", prompt.casefold())


class RetrievalTests(unittest.TestCase):
    def test_within_position_normalization_still_exposes_raw_values(self) -> None:
        fake_index = {
            "player_list": [
                {
                    "player_id": "m1",
                    "name": "Mid One",
                    "normalized_name": "mid one",
                    "nationality": "Croatia",
                    "nationality_normalized": "croatia",
                    "position": "Midfielder",
                    "role_bucket": "Midfielder",
                    "league": "Serie A",
                    "team": "Alpha",
                    "season_years": [2018],
                    "minutes": 1000,
                    "season_id": "s1",
                    "season_label": "2018/2019",
                    "source_dataset": "serie_a",
                    "row_id": "serie-a:m1:2018-2019",
                    "stat_features": {
                        "goals": 4,
                        "assists": 9,
                        "shots_on_target": 10,
                        "dribbles_completed": 5,
                        "minutes": 1000,
                        "appearances": 30,
                        "goals_per_game": 0.13,
                        "assists_per_game": 0.30,
                        "shot_on_target_ratio": 0.5,
                        "progressive_passes": 80,
                        "key_passes": 40,
                        "pass_completion": 89,
                        "tackles": 20,
                        "interceptions": 15,
                        "recoveries": 50,
                        "aerial_duels_won": 5,
                        "duels": 40,
                    },
                },
                {
                    "player_id": "m2",
                    "name": "Mid Two",
                    "normalized_name": "mid two",
                    "nationality": "Spain",
                    "nationality_normalized": "spain",
                    "position": "Midfielder",
                    "role_bucket": "Midfielder",
                    "league": "Serie A",
                    "team": "Beta",
                    "season_years": [2018],
                    "minutes": 950,
                    "season_id": "s2",
                    "season_label": "2018/2019",
                    "source_dataset": "serie_a",
                    "row_id": "serie-a:m2:2018-2019",
                    "stat_features": {
                        "goals": 2,
                        "assists": 5,
                        "shots_on_target": 8,
                        "dribbles_completed": 4,
                        "minutes": 950,
                        "appearances": 28,
                        "goals_per_game": 0.07,
                        "assists_per_game": 0.18,
                        "shot_on_target_ratio": 0.4,
                        "progressive_passes": 60,
                        "key_passes": 30,
                        "pass_completion": 85,
                        "tackles": 18,
                        "interceptions": 14,
                        "recoveries": 45,
                        "aerial_duels_won": 4,
                        "duels": 35,
                    },
                },
                {
                    "player_id": "f1",
                    "name": "Forward One",
                    "normalized_name": "forward one",
                    "nationality": "Brazil",
                    "nationality_normalized": "brazil",
                    "position": "Forward",
                    "role_bucket": "Forward",
                    "league": "Serie A",
                    "team": "Gamma",
                    "season_years": [2018],
                    "minutes": 1100,
                    "season_id": "s3",
                    "season_label": "2018/2019",
                    "source_dataset": "serie_a",
                    "row_id": "serie-a:f1:2018-2019",
                    "stat_features": {
                        "goals": 14,
                        "assists": 2,
                        "shots_on_target": 20,
                        "dribbles_completed": 6,
                        "minutes": 1100,
                        "appearances": 29,
                        "goals_per_game": 0.48,
                        "assists_per_game": 0.07,
                        "shot_on_target_ratio": 0.62,
                        "progressive_passes": 8,
                        "key_passes": 6,
                        "pass_completion": 71,
                        "tackles": 4,
                        "interceptions": 2,
                        "recoveries": 9,
                        "aerial_duels_won": 7,
                        "duels": 18,
                    },
                },
            ]
        }
        parsed_query = {
            "mode": "season",
            "confidence": 0.8,
            "entities": {"players": []},
            "filters": {"positions": ["Midfielder"], "leagues": ["Serie A"]},
            "style_descriptors": [{"term": "creative", "stat_family": "chance_creation", "stats": ["key_passes", "assists"]}],
        }

        with patch("src.retrieval.retrieve.PLAYER_INDEX", fake_index):
            retrieval = retrieve_ranked_players(
                parsed_query,
                filters=parsed_query["filters"],
                top_k=5,
                max_players=3,
                max_supporting_rows_per_player=2,
            )

        self.assertGreater(retrieval["results"][0]["final_player_score"], retrieval["results"][1]["final_player_score"])
        top_support = retrieval["results"][0]["supporting_rows"][0]
        self.assertEqual(top_support["player_name"], "Mid One")
        self.assertEqual(top_support["raw_key_stats"]["progressive_passes"], 80)

    def test_named_player_similarity_restricts_to_anchor_bucket_and_penalizes_tiny_samples(self) -> None:
        fake_index = {
            "player_list": [
                {
                    "player_id": "anchor",
                    "name": "Luka Modric",
                    "normalized_name": "luka modric",
                    "nationality": "Croatia",
                    "nationality_normalized": "croatia",
                    "position": "Midfielder",
                    "role_bucket": "Midfielder",
                    "league": "La Liga",
                    "team": "Real Madrid",
                    "season_years": [2016],
                    "minutes": 2200,
                    "season_id": "anchor-season",
                    "season_label": "2016/2017",
                    "source_dataset": "la_liga",
                    "row_id": "la-liga:anchor:2016",
                    "stat_features": {
                        "goals": 4,
                        "assists": 10,
                        "shots_on_target": 12,
                        "dribbles_completed": 30,
                        "minutes": 2200,
                        "appearances": 33,
                        "goals_per_game": 0.12,
                        "assists_per_game": 0.30,
                        "shot_on_target_ratio": 0.45,
                        "progressive_passes": 120,
                        "key_passes": 55,
                        "pass_completion": 89,
                        "tackles": 28,
                        "interceptions": 20,
                        "recoveries": 70,
                        "aerial_duels_won": 8,
                        "duels": 40,
                    },
                },
                {
                    "player_id": "mid-good",
                    "name": "Midfielder Match",
                    "normalized_name": "midfielder match",
                    "nationality": "Spain",
                    "nationality_normalized": "spain",
                    "position": "Midfielder",
                    "role_bucket": "Midfielder",
                    "league": "La Liga",
                    "team": "Barcelona",
                    "season_years": [2017],
                    "minutes": 1800,
                    "season_id": "m1",
                    "season_label": "2017/2018",
                    "source_dataset": "la_liga",
                    "row_id": "la-liga:mid-good:2017",
                    "stat_features": {
                        "goals": 3,
                        "assists": 9,
                        "shots_on_target": 11,
                        "dribbles_completed": 28,
                        "minutes": 1800,
                        "appearances": 30,
                        "goals_per_game": 0.10,
                        "assists_per_game": 0.30,
                        "shot_on_target_ratio": 0.43,
                        "progressive_passes": 110,
                        "key_passes": 50,
                        "pass_completion": 88,
                        "tackles": 26,
                        "interceptions": 18,
                        "recoveries": 68,
                        "aerial_duels_won": 7,
                        "duels": 38,
                    },
                },
                {
                    "player_id": "mid-tiny",
                    "name": "Tiny Sample Mid",
                    "normalized_name": "tiny sample mid",
                    "nationality": "Spain",
                    "nationality_normalized": "spain",
                    "position": "Midfielder",
                    "role_bucket": "Midfielder",
                    "league": "La Liga",
                    "team": "Sevilla",
                    "season_years": [2018],
                    "minutes": 45,
                    "season_id": "m2",
                    "season_label": "2018/2019",
                    "source_dataset": "la_liga",
                    "row_id": "la-liga:mid-tiny:2018",
                    "stat_features": {
                        "goals": 0,
                        "assists": 1,
                        "shots_on_target": 1,
                        "dribbles_completed": 2,
                        "minutes": 45,
                        "appearances": 1,
                        "goals_per_game": 0.0,
                        "assists_per_game": 1.0,
                        "shot_on_target_ratio": 1.0,
                        "progressive_passes": 20,
                        "key_passes": 8,
                        "pass_completion": 91,
                        "tackles": 2,
                        "interceptions": 2,
                        "recoveries": 6,
                        "aerial_duels_won": 1,
                        "duels": 4,
                    },
                },
                {
                    "player_id": "forward",
                    "name": "Forward Mismatch",
                    "normalized_name": "forward mismatch",
                    "nationality": "Brazil",
                    "nationality_normalized": "brazil",
                    "position": "Forward",
                    "role_bucket": "Forward",
                    "league": "La Liga",
                    "team": "Valencia",
                    "season_years": [2017],
                    "minutes": 1900,
                    "season_id": "f1",
                    "season_label": "2017/2018",
                    "source_dataset": "la_liga",
                    "row_id": "la-liga:forward:2017",
                    "stat_features": {
                        "goals": 18,
                        "assists": 3,
                        "shots_on_target": 26,
                        "dribbles_completed": 18,
                        "minutes": 1900,
                        "appearances": 29,
                        "goals_per_game": 0.62,
                        "assists_per_game": 0.10,
                        "shot_on_target_ratio": 0.58,
                        "progressive_passes": 15,
                        "key_passes": 10,
                        "pass_completion": 74,
                        "tackles": 4,
                        "interceptions": 2,
                        "recoveries": 10,
                        "aerial_duels_won": 5,
                        "duels": 20,
                    },
                },
            ]
        }
        parsed_query = {
            "mode": "hybrid",
            "confidence": 0.8,
            "entities": {"players": ["Luka Modric"]},
            "filters": {"year_range": [2010, 2019], "positions": None, "leagues": [], "teams": [], "nationality": None, "minutes_min": None},
            "style_descriptors": [{"term": "progressive passing", "stat_family": "progression_passing", "stats": ["progressive_passes", "key_passes"]}],
            "anchor_player_name": "Luka Modric",
            "anchor_player_normalized_name": "luka modric",
            "anchor_role_bucket": "Midfielder",
            "anchor_compatible_positions": ["Midfielder"],
            "is_named_player_similarity": True,
            "has_low_sample_intent": False,
        }

        with patch("src.retrieval.retrieve.PLAYER_INDEX", fake_index):
            retrieval = retrieve_ranked_players(
                parsed_query,
                filters=parsed_query["filters"],
                top_k=10,
                max_players=5,
                max_supporting_rows_per_player=2,
            )

        result_names = [result["player_name"] for result in retrieval["results"]]
        self.assertIn("Midfielder Match", result_names)
        self.assertNotIn("Forward Mismatch", result_names)
        self.assertNotIn("Tiny Sample Mid", result_names)
        self.assertNotIn("Luka Modric", result_names)
        self.assertEqual(retrieval["results"][0]["position"], "Midfielder")
        self.assertEqual(retrieval["debug"]["minutes_floor"], 600)

    def test_anchor_evidence_prefers_year_range_style_stats_then_minutes(self) -> None:
        fake_index = {
            "player_list": [
                {
                    "player_id": "anchor-old",
                    "name": "Luka Modric",
                    "normalized_name": "luka modric",
                    "position": "Midfielder",
                    "role_bucket": "Midfielder",
                    "league": "La Liga",
                    "team": "Real Madrid",
                    "season_years": [2008],
                    "minutes": 2600,
                    "season_id": "old",
                    "season_label": "2008/2009",
                    "source_dataset": "la_liga",
                    "row_id": "la-liga:anchor:2008",
                    "stat_features": {"minutes": 2600, "key_passes": 10},
                },
                {
                    "player_id": "anchor-best",
                    "name": "Luka Modric",
                    "normalized_name": "luka modric",
                    "position": "Midfielder",
                    "role_bucket": "Midfielder",
                    "league": "La Liga",
                    "team": "Real Madrid",
                    "season_years": [2016],
                    "minutes": 2200,
                    "season_id": "best",
                    "season_label": "2016/2017",
                    "source_dataset": "la_liga",
                    "row_id": "la-liga:anchor:2016",
                    "stat_features": {"minutes": 2200, "progressive_passes": 120, "key_passes": 55},
                },
                {
                    "player_id": "anchor-weaker",
                    "name": "Luka Modric",
                    "normalized_name": "luka modric",
                    "position": "Midfielder",
                    "role_bucket": "Midfielder",
                    "league": "La Liga",
                    "team": "Real Madrid",
                    "season_years": [2017],
                    "minutes": 2400,
                    "season_id": "weaker",
                    "season_label": "2017/2018",
                    "source_dataset": "la_liga",
                    "row_id": "la-liga:anchor:2017",
                    "stat_features": {"minutes": 2400, "key_passes": 20},
                },
            ],
            "players_by_name": {},
        }
        parsed_query = {
            "entities": {"players": ["Luka Modric"]},
            "style_descriptors": [{"term": "progressive passing", "stat_family": "progression_passing", "stats": ["progressive_passes", "key_passes"]}],
            "anchor_player_name": "Luka Modric",
            "anchor_player_normalized_name": "luka modric",
            "anchor_role_bucket": "Midfielder",
            "is_named_player_similarity": True,
        }

        with patch("src.retrieval.retrieve.PLAYER_INDEX", fake_index):
            anchor = build_anchor_evidence(parsed_query, {"year_range": [2010, 2019], "positions": None, "leagues": [], "teams": [], "nationality": None, "minutes_min": None})

        self.assertIsNotNone(anchor)
        self.assertEqual(anchor["season_id"], "best")
        self.assertEqual(anchor["raw_key_stats"]["progressive_passes"], 120)

    def test_direct_comparison_retrieval_only_returns_requested_players(self) -> None:
        fake_index = {
            "player_list": [
                {
                    "player_id": "modric",
                    "name": "Luka Modric",
                    "normalized_name": "luka modric",
                    "position": "Midfielder",
                    "role_bucket": "Midfielder",
                    "league": "La Liga",
                    "team": "Real Madrid",
                    "season_years": [2016],
                    "minutes": 2200,
                    "season_id": "m",
                    "season_label": "2016/2017",
                    "source_dataset": "la_liga",
                    "row_id": "modric-row",
                    "stat_features": {"progressive_passes": 120, "key_passes": 55, "interceptions": 20, "tackles": 28, "minutes": 2200},
                },
                {
                    "player_id": "kroos",
                    "name": "Toni Kroos",
                    "normalized_name": "toni kroos",
                    "position": "Midfielder",
                    "role_bucket": "Midfielder",
                    "league": "La Liga",
                    "team": "Real Madrid",
                    "season_years": [2017],
                    "minutes": 2300,
                    "season_id": "k",
                    "season_label": "2017/2018",
                    "source_dataset": "la_liga",
                    "row_id": "kroos-row",
                    "stat_features": {"progressive_passes": 140, "key_passes": 65, "interceptions": 18, "tackles": 19, "minutes": 2300},
                },
                {
                    "player_id": "lamela",
                    "name": "E. Lamela",
                    "normalized_name": "e lamela",
                    "position": "Midfielder",
                    "role_bucket": "Midfielder",
                    "league": "Serie A",
                    "team": "Roma",
                    "season_years": [2012],
                    "minutes": 2000,
                    "season_id": "l",
                    "season_label": "2012/2013",
                    "source_dataset": "serie_a",
                    "row_id": "lamela-row",
                    "stat_features": {"progressive_passes": 99, "key_passes": 33, "minutes": 2000},
                },
            ]
        }
        parsed_query = {
            "comparison_player_names": ["Luka Modric", "Toni Kroos"],
            "entities": {"players": ["Luka Modric", "Toni Kroos"]},
            "confidence": 0.8,
            "style_descriptors": [],
        }
        with patch("src.retrieval.retrieve.PLAYER_INDEX", fake_index):
            from src.retrieval.retrieve import retrieve_comparison_targets
            retrieval = retrieve_comparison_targets(
                parsed_query,
                {"year_range": [2010, 2019], "positions": None, "leagues": [], "teams": [], "nationality": None, "minutes_min": None},
                top_k_per_target=4,
                max_supporting_rows_per_target=2,
            )
        names = {hit["player_name"] for hit in retrieval["hits"]}
        self.assertEqual(names, {"Luka Modric", "Toni Kroos"})
        self.assertNotIn("E. Lamela", names)

    def test_direct_comparison_missing_player_does_not_substitute_candidates(self) -> None:
        fake_index = {
            "player_list": [
                {
                    "player_id": "modric",
                    "name": "Luka Modric",
                    "normalized_name": "luka modric",
                    "position": "Midfielder",
                    "role_bucket": "Midfielder",
                    "league": "La Liga",
                    "team": "Real Madrid",
                    "season_years": [2016],
                    "minutes": 2200,
                    "season_id": "m",
                    "season_label": "2016/2017",
                    "source_dataset": "la_liga",
                    "row_id": "modric-row",
                    "stat_features": {"progressive_passes": 120, "key_passes": 55, "minutes": 2200},
                },
            ]
        }
        parsed_query = {
            "comparison_player_names": ["Luka Modric", "Toni Kroos"],
            "entities": {"players": ["Luka Modric", "Toni Kroos"]},
            "confidence": 0.8,
            "style_descriptors": [],
        }
        with patch("src.retrieval.retrieve.PLAYER_INDEX", fake_index):
            from src.retrieval.retrieve import retrieve_comparison_targets
            retrieval = retrieve_comparison_targets(
                parsed_query,
                {"year_range": [2010, 2019], "positions": None, "leagues": [], "teams": [], "nationality": None, "minutes_min": None},
            )
        hit_names = {hit["player_name"] for hit in retrieval["hits"]}
        self.assertEqual(hit_names, {"Luka Modric"})
        self.assertTrue(any("Toni Kroos" in warning for warning in retrieval["warnings"]))


class PlayerChatRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        app = Flask(
            __name__,
            static_folder=self.tmpdir.name,
            static_url_path="",
        )
        with open(os.path.join(self.tmpdir.name, "index.html"), "w", encoding="utf-8") as handle:
            handle.write("<html></html>")
        register_player_chat_route(app)
        self.client = app.test_client()

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    @patch("src.player_chat_routes.rewrite_player_chat_query", return_value="Luka Modric progressive passing midfielders 2010s")
    @patch("src.player_chat_routes.generate_grounded_answer", return_value=("grounded answer", {"system_prompt": "sys", "user_prompt": "usr"}))
    @patch("src.player_chat_routes.retrieve_ranked_players")
    def test_endpoint_returns_structured_payload(self, mock_retrieve, mock_generate, mock_rewrite) -> None:
        mock_retrieve.return_value = {
            "retrieval_mode": "hybrid",
            "results": [{"player_name": "Candidate Mid"}],
            "hits": [
                {
                    "evidence_id": "row_candidate",
                    "source_type": "player_row",
                    "player_id": "1",
                    "player_name": "Candidate Mid",
                    "normalized_name": "candidate mid",
                    "season_id": "season-1",
                    "season_label": "2017/2018",
                    "team": "Barcelona",
                    "league": "La Liga",
                    "position": "Midfielder",
                    "retrieval_score": 0.91,
                    "style_matches": [{"term": "progressive passing", "stat_family": "progression_passing"}],
                    "raw_key_stats": {"progressive_passes": 120, "key_passes": 50, "minutes": 1800},
                    "provenance": {"dataset": "la_liga", "row_id": "candidate-row", "retrieved_at": "now"},
                    "is_aggregate_row": False,
                }
            ],
            "anchor_hit": {
                "evidence_id": "row_anchor",
                "source_type": "player_row",
                "player_id": "64",
                "player_name": "Luka Modric",
                "normalized_name": "luka modric",
                "season_id": "anchor-season",
                "season_label": "2016/2017",
                "team": "Real Madrid",
                "league": "La Liga",
                "position": "Midfielder",
                "retrieval_score": 1.0,
                "style_matches": [{"term": "progressive passing", "stat_family": "progression_passing"}],
                "raw_key_stats": {"progressive_passes": 140, "key_passes": 60, "minutes": 2200},
                "provenance": {"dataset": "la_liga", "row_id": "anchor-row", "retrieved_at": "now"},
                "is_aggregate_row": True,
                "anchor_role_bucket": "Midfielder",
            },
            "retrieval_confidence": 0.8,
            "warnings": [],
            "debug": {"anchor_role_bucket": "Midfielder"},
        }
        response = self.client.post(
            "/api/player-chat",
            json={
                "message": "players like Luka Modric with progressive passing from the 2010s",
                "filters": {"year_range": [2020, 2021]},
                "debug": True,
            },
        )

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["answer"], "grounded answer")
        self.assertEqual(data["rewritten_query"], "Luka Modric progressive passing midfielders 2010s")
        self.assertIn("applied_filters", data)
        self.assertEqual(data["applied_filters"]["year_range"], [2020, 2021])
        self.assertTrue(any("overrode parsed" in warning for warning in data["warnings"]))
        self.assertIn("debug", data)
        self.assertIn("anchor_evidence", data["debug"])
        self.assertIn("generation", data["debug"])
        self.assertEqual(data["debug"]["generation"]["system_prompt"], "sys")
        mock_rewrite.assert_called_once_with("players like Luka Modric with progressive passing from the 2010s")
        self.assertIn("anchor_evidence", mock_generate.call_args.kwargs)

    @patch("src.player_chat_routes.rewrite_player_chat_query", side_effect=RuntimeError("API_KEY not set"))
    @patch("src.player_chat_routes.generate_grounded_answer", return_value=("grounded answer", None))
    @patch("src.player_chat_routes.retrieve_ranked_players")
    def test_endpoint_falls_back_to_original_message_when_rewrite_fails(self, mock_retrieve, _mock_generate, _mock_rewrite) -> None:
        mock_retrieve.return_value = {
            "retrieval_mode": "season",
            "results": [],
            "hits": [],
            "retrieval_confidence": 0.6,
            "warnings": [],
            "debug": None,
        }

        response = self.client.post(
            "/api/player-chat",
            json={"message": "best brazilian wingers"},
        )

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["rewritten_query"], "best brazilian wingers")
        self.assertTrue(any("LLM query rewrite unavailable" in warning for warning in data["warnings"]))

    @patch("src.retrieval.generate.os.getenv", return_value="fake-key")
    @patch("infosci_spark_client.LLMClient")
    def test_generate_grounded_answer_uses_deterministic_fallback_for_invalid_similarity_output(self, mock_client_cls, _mock_getenv) -> None:
        mock_client = mock_client_cls.return_value
        mock_client.chat.return_value = {
            "content": (
                "Jose Alberto Canas Ruiz-Herrera has no stats recorded. "
                "Matias Kranevitter has missing data."
            )
        }
        evidence = [
            {
                "evidence_id": "ev_canas",
                "source_type": "player_row",
                "player_id": "1639",
                "player_name": "Jose Alberto Canas Ruiz-Herrera",
                "retrieval_mode": "hybrid",
                "retrieval_score": 0.79,
                "rank": 1,
                "key_stats": {"progressive_passes": 562, "key_passes": 3, "interceptions": 98, "tackles": 73, "minutes": 2897},
                "style_matches": [{"term": "progressive passing", "stat_family": "progression_passing"}],
                "provenance": {"row_id": "canas-row"},
                "is_aggregate_row": True,
            },
            {
                "evidence_id": "ev_kran",
                "source_type": "player_row",
                "player_id": "4214",
                "player_name": "Matias Kranevitter",
                "retrieval_mode": "hybrid",
                "retrieval_score": 0.77,
                "rank": 2,
                "key_stats": {"progressive_passes": 279, "key_passes": 3, "interceptions": 37, "tackles": 45, "minutes": 1608},
                "style_matches": [{"term": "progressive passing", "stat_family": "progression_passing"}],
                "provenance": {"row_id": "kran-row"},
                "is_aggregate_row": True,
            },
        ]
        anchor_evidence = {
            "evidence_id": "ev_modric",
            "source_type": "player_row",
            "player_id": "64",
            "player_name": "Luka Modric",
            "retrieval_mode": "hybrid",
            "retrieval_score": 1.0,
            "rank": 0,
            "key_stats": {"progressive_passes": 10560, "key_passes": 496, "interceptions": 339, "tackles": 354, "minutes": 22457},
            "provenance": {"row_id": "modric-row"},
            "is_aggregate_row": True,
        }

        answer, debug = generate_grounded_answer(
            user_message="players like Luka Modric with progressive passing from the 2010s",
            retrieval_mode="hybrid",
            evidence=evidence,
            retrieval_confidence=0.8,
            anchor_evidence=anchor_evidence,
            include_debug=True,
        )

        self.assertIn("Jose Alberto Canas Ruiz-Herrera", answer)
        self.assertIn("562 progressive passes", answer)
        self.assertIn("3 key passes", answer)
        self.assertIn("98 interceptions", answer)
        self.assertIn("73 tackles", answer)
        self.assertTrue(debug["fallback_used"])
        self.assertIn("fallback_reason", debug)

    @patch("src.retrieval.generate.os.getenv", return_value="fake-key")
    @patch("infosci_spark_client.LLMClient")
    def test_generate_grounded_answer_uses_direct_comparison_fallback_when_output_uses_similarity_language(self, mock_client_cls, _mock_getenv) -> None:
        mock_client = mock_client_cls.return_value
        mock_client.chat.return_value = {"content": "The closest available matches are E. Lamela and J. Pastore."}
        evidence = [
            {
                "evidence_id": "ev_modric",
                "source_type": "player_row",
                "player_id": "64",
                "player_name": "Luka Modric",
                "retrieval_mode": "player",
                "retrieval_score": 1.0,
                "rank": 1,
                "key_stats": {"progressive_passes": 10560, "key_passes": 496, "interceptions": 339, "tackles": 354, "minutes": 22457},
                "provenance": {"row_id": "modric-row"},
            },
            {
                "evidence_id": "ev_kroos",
                "source_type": "player_row",
                "player_id": "8",
                "player_name": "Toni Kroos",
                "retrieval_mode": "player",
                "retrieval_score": 0.95,
                "rank": 2,
                "key_stats": {"progressive_passes": 9800, "key_passes": 430, "interceptions": 250, "tackles": 220, "minutes": 21000},
                "provenance": {"row_id": "kroos-row"},
            },
        ]
        answer, debug = generate_grounded_answer(
            user_message="compare Luka Modric vs Toni Kroos in the 2010s",
            retrieval_mode="player",
            evidence=evidence,
            retrieval_confidence=0.8,
            comparison_player_names=["Luka Modric", "Toni Kroos"],
            include_debug=True,
        )
        self.assertIn("Luka Modric vs Toni Kroos", answer)
        self.assertIn("Luka Modric", answer)
        self.assertIn("Toni Kroos", answer)
        self.assertNotIn("closest available matches", answer.casefold())
        self.assertTrue(debug["fallback_used"])

    def test_format_evidence_prefers_raw_key_stats_when_present(self) -> None:
        prompt = format_evidence_for_prompt(
            [
                {
                    "evidence_id": "ev_raw",
                    "source_type": "player_row",
                    "player_id": "1",
                    "player_name": "Raw Stats Player",
                    "retrieval_mode": "hybrid",
                    "retrieval_score": 0.8,
                    "rank": 1,
                    "raw_key_stats": {"progressive_passes": 999, "minutes": 1000},
                    "key_stats": {"progressive_passes": 1, "minutes": 5},
                    "provenance": {"row_id": "raw-row"},
                }
            ]
        )
        self.assertIn("Key stats: 999 progressive passes, 1000 minutes", prompt)
        self.assertNotIn("1 progressive passes", prompt)

    def test_endpoint_validates_message(self) -> None:
        response = self.client.post("/api/player-chat", json={})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"], "Message is required")


class PartialKeyStatsGenerationTests(unittest.TestCase):
    def test_system_prompt_contains_partial_stat_guidance(self) -> None:
        evidence = [
            {
                "evidence_id": "ev_amrabat",
                "source_type": "player_row",
                "player_id": "amrabat",
                "player_name": "S. Amrabat",
                "retrieval_mode": "hybrid",
                "retrieval_score": 0.85,
                "rank": 1,
                "matched_filters": {"leagues": ["Serie A"]},
                "style_matches": [],
                "key_stats": {"minutes": 9046, "progressive_passes": 30, "goals": 2, "assists": 2},
                "provenance": {"row_id": "serie-a:amrabat:agg"},
            }
        ]
        anchor_evidence = {
            "evidence_id": "ev_anchor_dm",
            "source_type": "player_row",
            "player_id": "anchor_dm",
            "player_name": "Anchor DM",
            "retrieval_mode": "hybrid",
            "retrieval_score": 1.0,
            "rank": 0,
            "matched_filters": {},
            "style_matches": [],
            "key_stats": {"minutes": 10000, "progressive_passes": 50, "goals": 3, "assists": 3},
            "provenance": {"row_id": "anchor-dm-row"},
        }
        messages = build_grounded_messages(
            user_message="best defensive midfielders in Serie A",
            retrieval_mode="hybrid",
            evidence=evidence,
            anchor_evidence=anchor_evidence,
        )
        system = messages[0]["content"]
        self.assertIn("partial key_stats (some but not all categories) are still valid evidence", system)
        self.assertIn("Missing one stat category does not invalidate the rest of the row", system)
        self.assertIn("Only say 'no stats recorded' if key_stats is completely empty", system)
        self.assertIn("Do not refuse to summarize a candidate just because defensive stats are missing", system)
        self.assertIn("If goals exist in key_stats, include that value in the candidate summary", system)

    @patch("src.retrieval.generate.os.getenv", return_value="fake-key")
    @patch("infosci_spark_client.LLMClient")
    def test_fallback_includes_goals_when_defensive_stats_absent(self, mock_client_cls, _mock_getenv) -> None:
        mock_client = mock_client_cls.return_value
        mock_client.chat.return_value = {"content": "S. Amrabat has no stats recorded."}
        evidence = [
            {
                "evidence_id": "ev_amrabat",
                "source_type": "player_row",
                "player_id": "amrabat",
                "player_name": "S. Amrabat",
                "retrieval_mode": "hybrid",
                "retrieval_score": 0.85,
                "rank": 1,
                "matched_filters": {},
                "style_matches": [],
                "key_stats": {"minutes": 9046, "progressive_passes": 30, "goals": 2, "assists": 2},
                "provenance": {"row_id": "serie-a:amrabat:agg"},
            }
        ]
        anchor_evidence = {
            "evidence_id": "ev_anchor_dm",
            "source_type": "player_row",
            "player_id": "anchor_dm",
            "player_name": "Anchor DM",
            "retrieval_mode": "hybrid",
            "retrieval_score": 1.0,
            "rank": 0,
            "matched_filters": {},
            "style_matches": [],
            "key_stats": {"minutes": 10000, "progressive_passes": 50, "goals": 3, "assists": 3},
            "provenance": {"row_id": "anchor-dm-row"},
        }
        answer, debug = generate_grounded_answer(
            user_message="best defensive midfielders in Serie A",
            retrieval_mode="hybrid",
            evidence=evidence,
            retrieval_confidence=0.8,
            anchor_evidence=anchor_evidence,
            include_debug=True,
        )
        self.assertTrue(debug["fallback_used"])
        self.assertIn("2 goals", answer)
        self.assertIn("9046", answer)

    @patch("src.retrieval.generate.os.getenv", return_value="fake-key")
    @patch("infosci_spark_client.LLMClient")
    def test_fallback_does_not_say_empty_stats_for_partial_records(self, mock_client_cls, _mock_getenv) -> None:
        mock_client = mock_client_cls.return_value
        mock_client.chat.return_value = {"content": "no recorded statistics for these players"}
        evidence = [
            {
                "evidence_id": "ev_amrabat",
                "source_type": "player_row",
                "player_id": "amrabat",
                "player_name": "S. Amrabat",
                "retrieval_mode": "hybrid",
                "retrieval_score": 0.85,
                "rank": 1,
                "matched_filters": {},
                "style_matches": [],
                "key_stats": {"minutes": 9046, "progressive_passes": 30, "goals": 2, "assists": 2},
                "provenance": {"row_id": "serie-a:amrabat:agg"},
            }
        ]
        anchor_evidence = {
            "evidence_id": "ev_anchor_dm",
            "source_type": "player_row",
            "player_id": "anchor_dm",
            "player_name": "Anchor DM",
            "retrieval_mode": "hybrid",
            "retrieval_score": 1.0,
            "rank": 0,
            "matched_filters": {},
            "style_matches": [],
            "key_stats": {"minutes": 10000, "progressive_passes": 50},
            "provenance": {"row_id": "anchor-dm-row"},
        }
        answer, debug = generate_grounded_answer(
            user_message="best defensive midfielders in Serie A",
            retrieval_mode="hybrid",
            evidence=evidence,
            retrieval_confidence=0.8,
            anchor_evidence=anchor_evidence,
            include_debug=True,
        )
        self.assertTrue(debug["fallback_used"])
        self.assertNotIn("no stats recorded", answer.casefold())
        self.assertNotIn("no recorded statistics", answer.casefold())

    @patch("src.retrieval.generate.os.getenv", return_value="fake-key")
    @patch("infosci_spark_client.LLMClient")
    def test_serie_a_defensive_midfielder_regression(self, mock_client_cls, _mock_getenv) -> None:
        mock_client = mock_client_cls.return_value
        mock_client.chat.return_value = {
            "content": "key_stats fields for all candidates are empty, no stats recorded."
        }
        evidence = [
            {
                "evidence_id": "ev_amrabat",
                "source_type": "player_row",
                "player_id": "amrabat",
                "player_name": "S. Amrabat",
                "retrieval_mode": "hybrid",
                "retrieval_score": 0.88,
                "rank": 1,
                "matched_filters": {"leagues": ["Serie A"]},
                "style_matches": [],
                "key_stats": {"minutes": 9046, "progressive_passes": 30, "goals": 2, "assists": 2},
                "provenance": {"row_id": "serie-a:amrabat:agg"},
            },
            {
                "evidence_id": "ev_crnigoj",
                "source_type": "player_row",
                "player_id": "crnigoj",
                "player_name": "D. Crnigoj",
                "retrieval_mode": "hybrid",
                "retrieval_score": 0.81,
                "rank": 2,
                "matched_filters": {"leagues": ["Serie A"]},
                "style_matches": [],
                "key_stats": {"minutes": 2461, "progressive_passes": 20, "goals": 3, "assists": 1},
                "provenance": {"row_id": "serie-a:crnigoj:agg"},
            },
            {
                "evidence_id": "ev_lopez",
                "source_type": "player_row",
                "player_id": "lopez",
                "player_name": "M. Lopez",
                "retrieval_mode": "hybrid",
                "retrieval_score": 0.77,
                "rank": 3,
                "matched_filters": {"leagues": ["Serie A"]},
                "style_matches": [],
                "key_stats": {"minutes": 8488, "progressive_passes": 223, "goals": 4, "assists": 1},
                "provenance": {"row_id": "serie-a:lopez:agg"},
            },
        ]
        anchor_evidence = {
            "evidence_id": "ev_anchor_dm",
            "source_type": "player_row",
            "player_id": "anchor_dm",
            "player_name": "Anchor DM",
            "retrieval_mode": "hybrid",
            "retrieval_score": 1.0,
            "rank": 0,
            "matched_filters": {},
            "style_matches": [],
            "key_stats": {"minutes": 12000, "progressive_passes": 80, "goals": 5, "assists": 4},
            "provenance": {"row_id": "anchor-dm-row"},
        }
        answer, debug = generate_grounded_answer(
            user_message="best defensive midfielders in Serie A",
            retrieval_mode="hybrid",
            evidence=evidence,
            retrieval_confidence=0.8,
            anchor_evidence=anchor_evidence,
            include_debug=True,
        )
        self.assertTrue(debug["fallback_used"])
        for value in ("9046", "30", "2461", "20", "8488", "223"):
            self.assertIn(value, answer, msg=f"Expected stat value '{value}' in fallback answer")
        self.assertIn("2 goals", answer)
        self.assertIn("3 goals", answer)
        self.assertIn("4 goals", answer)
        self.assertNotIn("no stats recorded", answer.casefold())
        self.assertNotIn("no recorded statistics", answer.casefold())
        self.assertNotIn("key_stats fields for all candidates are empty", answer.casefold())

    @patch("src.retrieval.generate.os.getenv", return_value="fake-key")
    @patch("infosci_spark_client.LLMClient")
    def test_generic_explanatory_query_fallback_uses_available_stats(self, mock_client_cls, _mock_getenv) -> None:
        mock_client = mock_client_cls.return_value
        mock_client.chat.return_value = {
            "content": "These players have no stats recorded and no evidence is available."
        }
        evidence = [
            {
                "evidence_id": "ev_banega",
                "source_type": "player_row",
                "player_id": "banega",
                "player_name": "E. Banega",
                "retrieval_mode": "hybrid",
                "retrieval_score": 0.92,
                "rank": 1,
                "matched_filters": {"nationality": "Argentina"},
                "style_matches": [{"term": "creative", "stat_family": "chance_creation"}],
                "key_stats": {"assists": 12, "key_passes": 87, "progressive_passes": 1128, "minutes": 5248},
                "provenance": {"row_id": "la-liga:banega:agg"},
                "is_aggregate_row": True,
            },
            {
                "evidence_id": "ev_robertone",
                "source_type": "player_row",
                "player_id": "robertone",
                "player_name": "Lucas Robertone",
                "retrieval_mode": "hybrid",
                "retrieval_score": 0.89,
                "rank": 2,
                "matched_filters": {"nationality": "Argentina"},
                "style_matches": [{"term": "creative", "stat_family": "chance_creation"}],
                "key_stats": {"assists": 6, "key_passes": 40, "progressive_passes": 700, "goals": 2, "minutes": 3900},
                "provenance": {"row_id": "la-liga:robertone:agg"},
                "is_aggregate_row": True,
            },
        ]

        answer, debug = generate_grounded_answer(
            user_message="why are creative Argentine midfielders valuable",
            retrieval_mode="hybrid",
            evidence=evidence,
            retrieval_confidence=0.8,
            include_debug=True,
        )

        self.assertTrue(debug["fallback_used"])
        self.assertIn("E. Banega", answer)
        self.assertIn("12 assists", answer)
        self.assertIn("87 key passes", answer)
        self.assertIn("1128 progressive passes", answer)
        self.assertIn("Lucas Robertone", answer)
        self.assertNotIn("no stats recorded", answer.casefold())


class EvalFixtureTests(unittest.TestCase):
    def test_eval_fixture_is_valid_json(self) -> None:
        with open("tests/fixtures/player_chat_eval.json", "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.assertGreaterEqual(len(payload), 10)
        self.assertTrue(all("query" in item for item in payload))


if __name__ == "__main__":
    unittest.main()
