from __future__ import annotations

import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "embeddings")
SCALER_PATH = os.path.join(OUTPUT_DIR, "position_scalers.joblib")

UNIVERSAL_FEATURES = [
    "appearances",
    "minutes_played",
    "yellow_cards",
    "red_cards",
]

POSITION_FEATURE_GROUPS = {
    "Forward": [
        "goals",
        "assists",
        "shots",
        "shots_on_target",
        "expected_goals",
        "set_piece_goals",
        "freekick_shots",
        "goals_per_90",
        "dribbles_completed",
    ],
    "Midfielder": [
        "passes",
        "key_passes",
        "assists",
        "set_piece_goals",
        "interceptions",
        "dribbles_completed",
        "progressive_passes",
    ],
    "Defender": [
        "tackles",
        "clearances",
        "aerial_duels_won",
        "blocks",
        "interceptions",
        "recoveries",
    ],
    "Goalkeeper": [
        "saves",
        "clean_sheets",
        "goals_against",
        "save_percentage",
    ],
}

DESCRIPTION_DISCRIMINATIVE_FEATURES = {
    "Forward": ["goals", "expected_goals", "shots_on_target", "goals_per_90"],
    "Midfielder": ["assists", "key_passes", "progressive_passes", "interceptions"],
    "Defender": ["tackles", "interceptions", "clearances", "aerial_duels_won"],
    "Goalkeeper": ["saves", "clean_sheets", "save_percentage"],
}

POSITION_ONE_HOT_COLUMNS = [
    "pos_Forward",
    "pos_Midfielder",
    "pos_Defender",
    "pos_Goalkeeper",
]

RAW_TO_CANONICAL_FEATURES = {
    "goals": ["goals", "total_goals", "Goals", "Reti"],
    "assists": ["goalAssists", "total_assists", "assists", "Goal Assists", "Assists (Intentional)"],
    "shots": ["totalShots", "total_scoring_att", "total-scoring-attempts", "Total Shots", "Tiri"],
    "shots_on_target": ["shotsOnTargetIncGoals", "total_ontarget_attempt", "on-target-scoring-attempts", "Shots On Target ( inc goals )", "TiriInPorta"],
    "expected_goals": ["expectedGoals", "Xg"],
    "expected_goals_freekick": [
        "expectedGoalsFreekick",
        "expectedGoalsFreekick",
        "expected_goals_freekick",
        "total_att_freekick_goal",
        "total_att_freekick_target",
        "total_att_freekick_total",
    ],
    "set_piece_goals": [
        "setPiecesGoals",
        "setPieceGoals",
        "setPieceGoal",
        "total_att_freekick_goal",
        "total_direct_setpiece_goals",
    ],
    "freekick_shots": [
        "freekickTotal",
        "freeKickShots",
        "total_att_freekick_total",
    ],
    "expected_assists": ["expectedAssists"],
    "dribbles_completed": ["successfulDribbles", "successful-dribble", "succDribblingClose", "succDribblingOneOnOne"],
    "passes": ["totalPasses", "total_pass", "Total Passes", "Passaggi"],
    "progressive_passes": ["forwardPasses", "Forward Passes", "accurate-forward-pass"],
    "key_passes": ["keyPassesAttemptAssists", "Key Passes (Attempt Assists)", "PassChiave"],
    "interceptions": ["interceptions", "total_interception", "interception"],
    "tackles": ["totalTackles", "total_tackle", "tackle", "Contrasti"],
    "clearances": ["totalClearances", "total_clearance", "Total Clearances", "Clears"],
    "aerial_duels_won": ["aerialDuelsWon", "total_aerial_won", "Aerial Duels won", "AerDuels"],
    "blocks": ["blocks", "Blocked Shots"],
    "recoveries": ["recoveries", "Recoveries", "PalleRecuperate"],
    "saves": ["savesMade", "total_saves", "Saves Made", "Parate"],
    "clean_sheets": ["cleanSheets", "total_clean_sheet", "Clean Sheets"],
    "goals_against": ["goalsConceded", "total_goals_conceded", "goals-conceded", "RetiSubite"],
    "save_percentage": ["totalSavePerc", "Save Percentage"],
    "yellow_cards": ["yellowCards", "total_yellow_card", "yellow-cards", "Ammonizioni"],
    "red_cards": ["totalRedCards", "total_red_card", "red-cards", "Espulsioni"],
    "appearances": ["appearances", "gamesPlayed", "total_games", "games-played", "Games Played", "Presenze"],
    "minutes_played": ["timePlayed", "total_mins_played", "minutes-played", "Time Played", "Minuti"],
}

CANONICAL_FEATURE_COLUMNS = (
    UNIVERSAL_FEATURES
    + sorted({feature for features in POSITION_FEATURE_GROUPS.values() for feature in features})
    + POSITION_ONE_HOT_COLUMNS
)


def build_canonical_feature_frame(player_df: pd.DataFrame) -> pd.DataFrame:
    """Build a canonical feature frame from heterogeneous league stat columns."""
    feature_frame = pd.DataFrame(index=player_df.index)
    for canonical_name, raw_columns in RAW_TO_CANONICAL_FEATURES.items():
        available = [column for column in raw_columns if column in player_df.columns]
        if available:
            feature_frame[canonical_name] = player_df[available].sum(axis=1)
        else:
            feature_frame[canonical_name] = 0.0

    minutes = feature_frame["minutes_played"].replace(0, np.nan)
    feature_frame["goals_per_90"] = (feature_frame["goals"] / minutes) * 90
    if (feature_frame["save_percentage"] == 0).all():
        save_denominator = (feature_frame["saves"] + feature_frame["goals_against"]).replace(0, np.nan)
        feature_frame["save_percentage"] = (feature_frame["saves"] / save_denominator) * 100
    feature_frame["save_percentage"] = feature_frame["save_percentage"].fillna(0)
    feature_frame["goals_per_90"] = feature_frame["goals_per_90"].fillna(0)

    for one_hot_column in POSITION_ONE_HOT_COLUMNS:
        feature_frame[one_hot_column] = player_df.get(one_hot_column, 0).astype(float)

    for column in CANONICAL_FEATURE_COLUMNS:
        if column not in feature_frame.columns:
            feature_frame[column] = 0.0

    return feature_frame[CANONICAL_FEATURE_COLUMNS].fillna(0)


def fit_position_scalers(feature_frame: pd.DataFrame, metadata: dict[str, dict]) -> dict[str, StandardScaler]:
    """Fit one StandardScaler per primary position on numeric embedding features only."""
    scalers: dict[str, StandardScaler] = {}
    numeric_columns = [column for column in CANONICAL_FEATURE_COLUMNS if column not in POSITION_ONE_HOT_COLUMNS]

    for position in ("Forward", "Midfielder", "Defender", "Goalkeeper"):
        position_names = [
            name for name in feature_frame.index if metadata.get(name, {}).get("primary_position") == position
        ]
        if not position_names:
            continue
        scaler = StandardScaler()
        scaler.fit(feature_frame.loc[position_names, numeric_columns])
        scalers[position] = scaler
    return scalers


def apply_position_scalers(
    feature_frame: pd.DataFrame,
    metadata: dict[str, dict],
    scalers: dict[str, StandardScaler],
) -> np.ndarray:
    """Apply fitted per-position scalers and return a dense matrix."""
    numeric_columns = [column for column in CANONICAL_FEATURE_COLUMNS if column not in POSITION_ONE_HOT_COLUMNS]
    scaled = feature_frame.copy()

    for position, scaler in scalers.items():
        position_names = [
            name for name in feature_frame.index if metadata.get(name, {}).get("primary_position") == position
        ]
        if not position_names:
            continue
        scaled.loc[position_names, numeric_columns] = scaler.transform(
            feature_frame.loc[position_names, numeric_columns]
        )

    return scaled[CANONICAL_FEATURE_COLUMNS].to_numpy(dtype=float)


def build_feature_matrix(
    player_df: pd.DataFrame,
    metadata: dict[str, dict],
    fit_scalers: bool = True,
) -> tuple[np.ndarray, list[str], dict[str, dict], dict[str, StandardScaler]]:
    """Build the final embedding matrix and fitted scaler bundle."""
    feature_frame = build_canonical_feature_frame(player_df)
    scalers = fit_position_scalers(feature_frame, metadata) if fit_scalers else {}
    matrix = apply_position_scalers(feature_frame, metadata, scalers)
    return matrix, feature_frame.index.tolist(), metadata, scalers


def save_position_scalers(scalers: dict[str, Any], path: str = SCALER_PATH) -> None:
    """Persist fitted position scalers to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scalers, path)
