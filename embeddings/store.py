from __future__ import annotations

import json
import os
from typing import Any

import joblib
import numpy as np
from rapidfuzz import process

from src.player_search import normalize_text


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "embeddings")
EMBEDDINGS_NPY_PATH = os.path.join(OUTPUT_DIR, "player_embeddings.npy")
PLAYER_INDEX_JSON_PATH = os.path.join(OUTPUT_DIR, "player_index.json")
PLAYER_METADATA_JSON_PATH = os.path.join(OUTPUT_DIR, "player_metadata.json")
SCALERS_JOBLIB_PATH = os.path.join(OUTPUT_DIR, "position_scalers.joblib")
FEATURE_NAMES_JSON_PATH = os.path.join(OUTPUT_DIR, "embedding_feature_names.json")


def save_embeddings(
    matrix: np.ndarray,
    player_index: list[str],
    player_metadata: dict[str, dict],
    scalers: dict[str, Any],
    feature_column_names: list[str] | None = None,
) -> None:
    """Save embeddings, player index, metadata, and fitted scalers to disk."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(EMBEDDINGS_NPY_PATH, matrix)
    with open(PLAYER_INDEX_JSON_PATH, "w", encoding="utf-8") as handle:
        json.dump(player_index, handle, ensure_ascii=False, indent=2)
    with open(PLAYER_METADATA_JSON_PATH, "w", encoding="utf-8") as handle:
        json.dump(player_metadata, handle, ensure_ascii=False, indent=2)
    joblib.dump(scalers, SCALERS_JOBLIB_PATH)
    if feature_column_names:
        with open(FEATURE_NAMES_JSON_PATH, "w", encoding="utf-8") as handle:
            json.dump(feature_column_names, handle, ensure_ascii=False, indent=2)


def load_embeddings() -> tuple[np.ndarray, list[str]]:
    """Load the embedding matrix and normalized player index from disk."""
    matrix = np.load(EMBEDDINGS_NPY_PATH)
    with open(PLAYER_INDEX_JSON_PATH, "r", encoding="utf-8") as handle:
        player_index = json.load(handle)
    return matrix, player_index


def load_player_metadata() -> dict[str, dict]:
    """Load serialized player metadata from disk."""
    with open(PLAYER_METADATA_JSON_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_position_scalers() -> dict[str, Any]:
    """Load fitted per-position scalers from disk."""
    return joblib.load(SCALERS_JOBLIB_PATH)


def get_player_index(name: str, player_index: list[str]) -> int:
    """Resolve a display or fuzzy name to the row index in player_index / matrix."""
    normalized_name = normalize_text(name)
    if not normalized_name:
        raise ValueError("Player name cannot be empty.")

    index_map = {player_name: idx for idx, player_name in enumerate(player_index)}
    if normalized_name in index_map:
        return index_map[normalized_name]

    fuzzy_match = process.extractOne(
        normalized_name,
        player_index,
        score_cutoff=80,
    )
    if fuzzy_match is None:
        raise ValueError(f"No player match found for '{name}'.")

    matched_name = fuzzy_match[0]
    return index_map[matched_name]


def get_player_vector(name: str, matrix: np.ndarray, player_index: list[str]) -> np.ndarray:
    """Return the vector for an exact or fuzzy-matched player name."""
    idx = get_player_index(name, player_index)
    return matrix[idx]
