from __future__ import annotations

import logging

import numpy as np

from embeddings.features import CANONICAL_FEATURE_COLUMNS, build_feature_matrix
from embeddings.preprocess import preprocess_player_stats
from embeddings.similarity import find_similar_players
from embeddings.store import load_embeddings, save_embeddings
from embeddings.text_embeddings import build_text_embeddings


logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Build and persist player embeddings, then run a quick similarity sanity check."""
    LOGGER.info("Preprocessing player stats")
    player_df, _, metadata = preprocess_player_stats()

    LOGGER.info("Building canonical feature matrix")
    matrix, player_index, player_metadata, scalers = build_feature_matrix(player_df, metadata)

    LOGGER.info("Building text embeddings (Wikipedia + Reddit)")
    text_bundle = build_text_embeddings(player_index)
    matrix = np.concatenate([matrix, text_bundle.matrix], axis=1)
    feature_names = list(CANONICAL_FEATURE_COLUMNS) + [f"text_{i}" for i in range(text_bundle.dim)]

    LOGGER.info("Saving embeddings and metadata")
    save_embeddings(
        matrix,
        player_index,
        player_metadata,
        scalers,
        feature_column_names=feature_names,
    )

    LOGGER.info("Reloading embeddings for smoke test")
    loaded_matrix, loaded_index = load_embeddings()

    for player_name in ("Harry Kane", "Virgil van Dijk"):
        LOGGER.info("Top 5 similar players for %s", player_name)
        for result in find_similar_players(player_name, loaded_matrix, loaded_index, top_k=5):
            print(result)


if __name__ == "__main__":
    main()
