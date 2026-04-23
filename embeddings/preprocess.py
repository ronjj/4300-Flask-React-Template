from __future__ import annotations

import logging
import os
import re
from collections import Counter
from typing import Any

import pandas as pd

from src.player_search import (
    canonical_nationality,
    first_non_empty,
    normalize_text,
    primary_position,
)


LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
STATS_CSV_PATHS = [
    os.path.join(DATA_DIR, "prem_all_players.csv"),
    os.path.join(DATA_DIR, "laliga_all_players.csv"),
    os.path.join(DATA_DIR, "seriea_all_players.csv"),
]

POSITION_COLUMNS = ["position", "role_label", "role"]
PLAYER_NAME_COLUMNS = ["player_name", "display_name", "short_name", "known_name"]
SEASON_COLUMNS = ["season", "season_range", "seasons_played"]
NON_STAT_COLUMNS = {
    "normalized_name",
    "display_name",
    "league",
    "primary_position",
    "season_years",
    "_source_path",
    "_source_file",
    "_row_number",
}
IDENTIFIER_COLUMNS = {
    "player_id",
    "season_id",
    "provider_id",
    "team_id",
    "current_team_id",
    "opta_id",
    "shirt_number",
    "bib_number",
    "index",
    "subscription",
    "role",
    "seasons_played",
}
EXPLICIT_RATE_COLUMNS = {
    "expectedGoals",
    "expectedAssists",
    "expectedGoalsFreekick",
    "expectedGoalsOnTarget",
    "expectedGoalsOnTargetConceded",
    "goals-average-over-90-minutes",
    "assists-average-over-90-minutes",
    "accurate-pass-percentage",
    "accurate-shooting-percentage",
    "crosses-complete-percentage",
    "dribble-percentage",
    "goals-against-average",
    "penalty-kick-percentage",
    "tackle-percentage",
    "Passing Accuracy",
    "Aerials-Won-Perc",
    "XGEfficiency",
    "aerial-duels-won-perc",
    "bigChanceConvertPerc",
    "duels-won-perc",
    "groundDuelsWonPerc",
    "shotsOnGoalPerc",
    "succDribblingPerc",
    "tackles-won-perc",
    "turnoverPerc",
    "avg-distance-covered",
    "average-speed",
    "top-speed",
    "VelMedia",
    "VelMax",
}
RATE_HINTS = ("percentage", "perc", "per_90", "per90", "average", "ratio", "rate")
SEASON_TOKEN_RE = re.compile(r"(?P<start>(?:19|20)\d{2})(?:\s*/\s*(?P<end>\d{2,4}))?")
YEAR_RE = re.compile(r"(?:19|20)\d{2}")
LEAGUE_NAME_MAP = {
    "prem": "Premier League",
    "laliga": "La Liga",
    "seriea": "Serie A",
    "bundesliga": "Bundesliga",
    "ligue1": "Ligue 1",
}
MAX_SEASON_WARNINGS_PER_FILE = 10

_season_warning_counts: dict[str, int] = {}


def infer_league_name(path: str) -> str:
    """Infer a human-readable league label from a CSV filepath."""
    filename = os.path.basename(path).casefold()
    for token, league_name in LEAGUE_NAME_MAP.items():
        if token in filename:
            return league_name
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem.replace("_", " ").title()


def load_stats_csvs(csv_paths: list[str] | None = None) -> pd.DataFrame:
    """Load one or more league CSVs into a single DataFrame.

    Adds `league`, `_source_path`, `_source_file`, and `_row_number` columns if needed.
    """
    paths = csv_paths or STATS_CSV_PATHS
    frames: list[pd.DataFrame] = []

    for path in paths:
        if not os.path.exists(path):
            continue
        frame = pd.read_csv(path, low_memory=False)
        n = len(frame)
        meta = {
            "_source_path": [path] * n,
            "_source_file": [os.path.basename(path)] * n,
            "_row_number": list(range(1, n + 1)),
        }
        if "league" not in frame.columns:
            meta["league"] = [infer_league_name(path)] * n
        frame = pd.concat([frame.reset_index(drop=True), pd.DataFrame(meta)], axis=1)
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(f"No stats CSVs found from paths: {paths}")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    loaded_leagues = sorted(set(combined["league"].dropna().astype(str)))
    LOGGER.info("Loaded %s CSV file(s): %s", len(frames), ", ".join(loaded_leagues))
    return combined


def _select_first_present(row: pd.Series, columns: list[str]) -> Any:
    for column in columns:
        value = row.get(column)
        if pd.notna(value) and str(value).strip():
            return value
    return None


def _extract_years_from_text(text: str) -> list[int]:
    years: list[int] = []
    for match in SEASON_TOKEN_RE.finditer(text):
        start_year = int(match.group("start"))
        end_token = match.group("end")
        if not end_token:
            years.append(start_year)
            continue
        if len(end_token) == 2:
            century = start_year // 100
            end_year = century * 100 + int(end_token)
        else:
            end_year = int(end_token)
        years.extend(range(start_year, end_year + 1))
    if years:
        return sorted(set(years))
    return sorted(set(int(year) for year in YEAR_RE.findall(text)))


def _warn_unparseable_season(row: pd.Series) -> None:
    source_file = str(row.get("_source_file", "unknown"))
    count = _season_warning_counts.get(source_file, 0)
    if count >= MAX_SEASON_WARNINGS_PER_FILE:
        return
    LOGGER.warning(
        "Could not parse season data for %s row %s (%s)",
        source_file,
        row.get("_row_number", "?"),
        {column: row.get(column) for column in SEASON_COLUMNS if column in row.index},
    )
    _season_warning_counts[source_file] = count + 1


def extract_season_years_for_row(row: pd.Series) -> list[int]:
    """Extract season start years from a row using actual season columns only."""
    years: list[int] = []
    saw_nonempty_season_field = False
    for column in SEASON_COLUMNS:
        if column not in row.index:
            continue
        value = row.get(column)
        if pd.isna(value):
            continue
        text = str(value).strip()
        if not text:
            continue
        saw_nonempty_season_field = True
        years.extend(_extract_years_from_text(text))

    years = sorted(set(years))
    # No warning when all season fields are empty/NaN (e.g. aggregated Premier League CSV
    # with no per-row season). Warn only when values were present but yielded no years.
    if not years and saw_nonempty_season_field:
        _warn_unparseable_season(row)
    return years


def infer_primary_position(group: pd.DataFrame) -> str:
    """Infer the most common canonical position label across a player's rows."""
    positions = [value for value in group["position_canonical"].dropna().tolist() if value]
    if not positions:
        return "Unknown"
    return Counter(positions).most_common(1)[0][0]


def split_numeric_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split numeric columns into counting and rate-style stats."""
    numeric_columns: list[str] = []
    for column in df.columns:
        if column in NON_STAT_COLUMNS or column in IDENTIFIER_COLUMNS or column in SEASON_COLUMNS:
            continue
        converted = pd.to_numeric(df[column], errors="coerce")
        if converted.notna().any():
            numeric_columns.append(column)

    rate_columns: list[str] = []
    counting_columns: list[str] = []
    for column in numeric_columns:
        name = column.casefold()
        if column in EXPLICIT_RATE_COLUMNS or any(hint in name for hint in RATE_HINTS):
            rate_columns.append(column)
        else:
            counting_columns.append(column)
    return sorted(counting_columns), sorted(rate_columns)


def build_player_metadata(raw_df: pd.DataFrame, grouped_df: pd.DataFrame) -> dict[str, dict]:
    """Build player metadata derived strictly from raw season data."""
    metadata: dict[str, dict] = {}
    for normalized_name, group in raw_df.groupby("normalized_name", sort=True):
        if not normalized_name:
            continue
        display_name = grouped_df.loc[normalized_name, "display_name"]
        season_years = sorted({year for values in group["season_years"] for year in values})
        positions_seen = sorted({value for value in group["position_canonical"].dropna().tolist() if value})
        leagues = sorted({value for value in group["league"].dropna().astype(str).tolist() if value})
        nat_values = [v for v in group["nationality_canonical"].dropna().tolist() if v]
        nationality = Counter(nat_values).most_common(1)[0][0] if nat_values else None
        metadata[normalized_name] = {
            "display_name": display_name,
            "primary_position": grouped_df.loc[normalized_name, "primary_position"],
            "positions_seen": positions_seen,
            "leagues": leagues,
            "season_years": season_years,
            "career_start_year": min(season_years) if season_years else None,
            "career_end_year": max(season_years) if season_years else None,
            "nationality": nationality,
            "nationality_normalized": normalize_text(nationality) if nationality else "",
        }
    return metadata


def preprocess_player_stats(csv_paths: list[str] | None = None) -> tuple[pd.DataFrame, dict[str, str], dict[str, dict]]:
    """Load, normalize, and aggregate player stats into one row per normalized player name."""
    raw_df = load_stats_csvs(csv_paths)

    display_name = raw_df.apply(
        lambda row: _select_first_present(row, PLAYER_NAME_COLUMNS) or "Unknown Player",
        axis=1,
    )
    raw_df = pd.concat(
        [
            raw_df,
            pd.DataFrame(
                {
                    "display_name": display_name,
                    "normalized_name": display_name.map(normalize_text),
                },
                index=raw_df.index,
            ),
        ],
        axis=1,
    )
    raw_df = raw_df[raw_df["normalized_name"] != ""].copy()
    raw_df["position_canonical"] = raw_df.apply(
        lambda row: primary_position(_select_first_present(row, POSITION_COLUMNS)),
        axis=1,
    )
    raw_df["nationality_canonical"] = raw_df.apply(
        lambda row: canonical_nationality(
            first_non_empty(row.get("country"), row.get("nationality"))
        ),
        axis=1,
    )
    raw_df["season_years"] = raw_df.apply(extract_season_years_for_row, axis=1)

    counting_columns, rate_columns = split_numeric_columns(raw_df)
    stat_cols = counting_columns + rate_columns
    if stat_cols:
        raw_df[stat_cols] = raw_df[stat_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    raw_df = raw_df.copy()

    aggregate_spec = {column: "sum" for column in counting_columns}
    aggregate_spec.update({column: "mean" for column in rate_columns})
    grouped_numeric = raw_df.groupby("normalized_name", sort=True).agg(aggregate_spec)

    grouped_meta = raw_df.groupby("normalized_name", sort=True).apply(
        lambda group: pd.Series(
            {
                "display_name": Counter(group["display_name"]).most_common(1)[0][0],
                "primary_position": infer_primary_position(group),
            }
        ),
        include_groups=False,
    )

    grouped_df = grouped_meta.join(grouped_numeric, how="left").fillna(0)
    pos_cols = pd.DataFrame(
        {
            f"pos_{position}": (grouped_df["primary_position"] == position).astype(int)
            for position in ("Forward", "Midfielder", "Defender", "Goalkeeper")
        },
        index=grouped_df.index,
    )
    grouped_df = pd.concat([grouped_df, pos_cols], axis=1)

    display_name_map = grouped_df["display_name"].to_dict()
    metadata = build_player_metadata(raw_df, grouped_df)
    return grouped_df, display_name_map, metadata
