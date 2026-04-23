"""
Fit TruncatedSVD on saved player_embeddings.npy and persist for search + explainability.

Run after build_embeddings.py:
  python -m embeddings.build_svd

Requires only numpy, scikit-learn, joblib (no pandas / rapidfuzz).
"""
from __future__ import annotations

import json
import logging
import os

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "embeddings")
EMBEDDINGS_NPY_PATH = os.path.join(OUTPUT_DIR, "player_embeddings.npy")
FEATURE_NAMES_JSON_PATH = os.path.join(OUTPUT_DIR, "embedding_feature_names.json")
SVD_BUNDLE_PATH = os.path.join(OUTPUT_DIR, "svd_bundle.joblib")


def main() -> None:
    if not os.path.exists(EMBEDDINGS_NPY_PATH):
        raise SystemExit(f"Missing {EMBEDDINGS_NPY_PATH}; run embeddings.build_embeddings first.")

    matrix = np.load(EMBEDDINGS_NPY_PATH)
    n_samples, n_features = matrix.shape

    if os.path.isfile(FEATURE_NAMES_JSON_PATH):
        with open(FEATURE_NAMES_JSON_PATH, "r", encoding="utf-8") as handle:
            feature_names = json.load(handle)
        if len(feature_names) != n_features:
            LOGGER.warning(
                "Feature name count %s != matrix columns %s; using generic labels",
                len(feature_names),
                n_features,
            )
            feature_names = [f"feature_{i}" for i in range(n_features)]
    else:
        LOGGER.warning("Missing %s; using generic feature_* labels", FEATURE_NAMES_JSON_PATH)
        feature_names = [f"feature_{i}" for i in range(n_features)]

    max_k = min(n_features - 1, n_samples - 1, 32)
    if max_k < 2:
        raise SystemExit("Not enough rows/columns for SVD (need at least 2 components cap).")
    n_components = max(2, min(16, max_k))

    LOGGER.info("Fitting TruncatedSVD with n_components=%s on matrix %s", n_components, matrix.shape)
    svd = TruncatedSVD(n_components=n_components, random_state=42, algorithm="randomized")
    svd.fit(matrix)

    bundle = {
        "svd": svd,
        "feature_names": feature_names,
        "n_samples": int(n_samples),
        "n_features": int(n_features),
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    joblib.dump(bundle, SVD_BUNDLE_PATH)
    LOGGER.info("Saved SVD bundle to %s", SVD_BUNDLE_PATH)
    evr = svd.explained_variance_ratio_
    LOGGER.info("Explained variance ratio (first 5): %s", evr[:5].tolist())


if __name__ == "__main__":
    main()
