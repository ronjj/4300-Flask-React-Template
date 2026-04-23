from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from flask import Flask

from src.player_chat_routes import register_player_chat_route
from src.retrieval.evidence import truncate_evidence_for_budget
from src.retrieval.query_understanding import parse_player_chat_query, resolve_filters
from src.retrieval.retrieve import retrieve_ranked_players


class QueryUnderstandingTests(unittest.TestCase):
    def test_similarity_query_parses_mode_entities_and_filters(self) -> None:
        parsed = parse_player_chat_query(
            "players like Luka Modric with progressive passing from the 2010s"
        )

        self.assertEqual(parsed["intent"], "similarity")
        self.assertEqual(parsed["mode"], "hybrid")
        self.assertIn("Luka Modric", parsed["entities"]["players"])
        self.assertEqual(parsed["filters"]["year_range"], [2010, 2019])
        self.assertTrue(any(item["term"] == "progressive passing" for item in parsed["style_descriptors"]))

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

        trimmed = truncate_evidence_for_budget(evidence, max_chars=250)

        self.assertEqual(len(trimmed), 1)
        self.assertNotIn("style_matches", trimmed[0])
        self.assertNotIn("key_stats", trimmed[0])
        self.assertEqual(trimmed[0]["evidence_id"], "ev1")
        self.assertEqual(trimmed[0]["player_name"], "One")


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

    @patch("src.player_chat_routes.generate_grounded_answer", return_value="grounded answer")
    def test_endpoint_returns_structured_payload(self, _mock_generate) -> None:
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
        self.assertIn("applied_filters", data)
        self.assertEqual(data["applied_filters"]["year_range"], [2020, 2021])
        self.assertTrue(any("overrode parsed" in warning for warning in data["warnings"]))
        self.assertIn("debug", data)

    def test_endpoint_validates_message(self) -> None:
        response = self.client.post("/api/player-chat", json={})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"], "Message is required")


class EvalFixtureTests(unittest.TestCase):
    def test_eval_fixture_is_valid_json(self) -> None:
        with open("tests/fixtures/player_chat_eval.json", "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.assertGreaterEqual(len(payload), 10)
        self.assertTrue(all("query" in item for item in payload))


if __name__ == "__main__":
    unittest.main()
