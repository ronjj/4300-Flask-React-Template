from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from src.player_search import normalize_text


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "embeddings")

WIKI_PLAYER_GLOBS = (
    os.path.join(DATA_DIR, "wikipedia_data", "wiki_players_part*.csv"),
    os.path.join(DATA_DIR, "wikipedia_data", "wiki_sections_part*.csv"),
)
REDDIT_GLOB = os.path.join(DATA_DIR, "reddit", "all_leagues_reddit_part*.csv")

TEXT_EMBEDDINGS_NPY_PATH = os.path.join(OUTPUT_DIR, "player_text_embeddings.npy")
TEXT_MODEL_ID_PATH = os.path.join(OUTPUT_DIR, "player_text_model_id.txt")

DEFAULT_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TEXT_CHARS_PER_PLAYER = 4000
MAX_REDDIT_SNIPPETS_PER_PLAYER = 3
DEFAULT_TEXT_DIM_FALLBACK = 256


@dataclass(frozen=True)
class TextEmbeddingBundle:
    matrix: np.ndarray
    dim: int
    model_id: str


def _iter_glob(pattern: str) -> Iterable[str]:
    return [str(p) for p in sorted(Path(PROJECT_ROOT).glob(os.path.relpath(pattern, PROJECT_ROOT)))]


def _add_text(store: dict[str, list[str]], key: str, text: str) -> None:
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return
    bucket = store.setdefault(key, [])
    bucket.append(cleaned)


def load_player_text_corpus(player_index: list[str]) -> dict[str, str]:
    """
    Build a per-player text corpus from Wikipedia (summaries/sections) and Reddit snippets.

    Keys are normalized player names (matching embedding `player_index` entries).
    """
    wanted = set(player_index)
    chunks: dict[str, list[str]] = {}

    # Wikipedia: summaries and/or sections.
    for glob_pattern in WIKI_PLAYER_GLOBS:
        for path in _iter_glob(glob_pattern):
            if not os.path.exists(path):
                continue
            with open(path, newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    name = normalize_text(row.get("player_name") or row.get("wiki_title") or "")
                    if not name or name not in wanted:
                        continue
                    summary = row.get("wiki_summary") or row.get("section_text") or ""
                    _add_text(chunks, name, summary)

    # Reddit: take a few high-signal snippets per player (title + post text + comments).
    # These CSVs are large; keep the aggregation small and bounded.
    reddit_candidates: dict[str, list[tuple[int, str]]] = {}
    for path in _iter_glob(REDDIT_GLOB):
        if not os.path.exists(path):
            continue
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                name = normalize_text(row.get("player_name") or "")
                if not name or name not in wanted:
                    continue
                score = 0
                for field in ("comment_score", "post_score"):
                    try:
                        score = max(score, int(float(row.get(field) or 0)))
                    except (TypeError, ValueError):
                        continue
                text = row.get("post_title") or ""
                body = row.get("comment_body") or row.get("post_text") or ""
                merged = " ".join([text.strip(), body.strip()]).strip()
                merged = " ".join(merged.split())
                if not merged:
                    continue
                bucket = reddit_candidates.setdefault(name, [])
                bucket.append((score, merged))

    for name, items in reddit_candidates.items():
        items.sort(key=lambda x: x[0], reverse=True)
        for _, snippet in items[:MAX_REDDIT_SNIPPETS_PER_PLAYER]:
            _add_text(chunks, name, snippet)

    corpus: dict[str, str] = {}
    for name, parts in chunks.items():
        combined = "\n".join(parts)
        corpus[name] = combined[:MAX_TEXT_CHARS_PER_PLAYER]
    return corpus


def _load_cached_text_embeddings(expected_players: int, model_id: str) -> TextEmbeddingBundle | None:
    if not os.path.exists(TEXT_EMBEDDINGS_NPY_PATH) or not os.path.exists(TEXT_MODEL_ID_PATH):
        return None
    try:
        cached_model_id = Path(TEXT_MODEL_ID_PATH).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if cached_model_id != model_id:
        return None
    matrix = np.load(TEXT_EMBEDDINGS_NPY_PATH)
    if matrix.shape[0] != expected_players:
        return None
    return TextEmbeddingBundle(matrix=matrix, dim=int(matrix.shape[1]), model_id=model_id)


def build_text_embeddings(
    player_index: list[str],
    model_id: str = DEFAULT_MODEL_ID,
    force_rebuild: bool = False,
) -> TextEmbeddingBundle:
    """
    Encode per-player Wikipedia + Reddit text into a dense embedding matrix aligned to player_index.

    Caches results to disk under embeddings/ so rebuilds are fast.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not force_rebuild:
        cached = _load_cached_text_embeddings(len(player_index), model_id)
        if cached is not None:
            return cached

    corpus = load_player_text_corpus(player_index)
    texts = [corpus.get(name, "") for name in player_index]

    # Preferred: Sentence-BERT if the model is already cached / downloadable.
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        model = SentenceTransformer(model_id)
        vectors = model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        matrix = np.asarray(vectors, dtype=float)
        used_model_id = model_id
    except Exception:
        # Fallback: TF-IDF → SVD to a dense semantic space, fully offline.
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            min_df=2,
            stop_words="english",
        )
        tfidf = vectorizer.fit_transform(texts)
        k = min(DEFAULT_TEXT_DIM_FALLBACK, tfidf.shape[1] - 1) if tfidf.shape[1] > 1 else 1
        if k < 2:
            matrix = np.zeros((len(texts), DEFAULT_TEXT_DIM_FALLBACK), dtype=float)
        else:
            svd = TruncatedSVD(n_components=k, random_state=42)
            dense = svd.fit_transform(tfidf)
            norms = np.linalg.norm(dense, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            dense = dense / norms
            if k < DEFAULT_TEXT_DIM_FALLBACK:
                pad = np.zeros((dense.shape[0], DEFAULT_TEXT_DIM_FALLBACK - k), dtype=float)
                dense = np.concatenate([dense, pad], axis=1)
            matrix = dense.astype(float)
        used_model_id = f"tfidf-svd-{DEFAULT_TEXT_DIM_FALLBACK}"

    np.save(TEXT_EMBEDDINGS_NPY_PATH, matrix)
    Path(TEXT_MODEL_ID_PATH).write_text(used_model_id, encoding="utf-8")
    return TextEmbeddingBundle(matrix=matrix, dim=int(matrix.shape[1]), model_id=used_model_id)

