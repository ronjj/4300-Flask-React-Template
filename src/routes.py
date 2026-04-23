"""
Routes: React app serving plus FootySearch player search APIs.

To enable AI chat, set USE_LLM = True below. See llm_routes.py for AI code.
"""
import os

from flask import jsonify, request, send_from_directory

try:
    from player_search import find_player_by_name
    from search_service import search_players
except ImportError:  # pragma: no cover - package-style import fallback
    from src.player_search import find_player_by_name
    from src.search_service import search_players

# ── AI toggle ────────────────────────────────────────────────────────────────
USE_LLM = False
# USE_LLM = True
# ─────────────────────────────────────────────────────────────────────────────


def register_routes(app):
    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def serve(path):
        if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
            return send_from_directory(app.static_folder, path)
        return send_from_directory(app.static_folder, "index.html")

    @app.route("/api/config")
    def config():
        return jsonify({"use_llm": USE_LLM})

    @app.route("/api/search")
    def search():
        query = request.args.get("q", "")
        return jsonify(search_players(query))

    @app.route("/api/player")
    def player_lookup():
        name = request.args.get("name", "")
        players = find_player_by_name(name)
        if players is None:
            return jsonify({"player": None}), 404
        return jsonify(players)

    if USE_LLM:
        from llm_routes import register_chat_route

        register_chat_route(app, lambda query: search_players(query)["results"])
