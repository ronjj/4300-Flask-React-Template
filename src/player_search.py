import csv
import difflib
import os
import re
import unicodedata
from datetime import date
from typing import Any, Dict, List, Optional


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

LEAGUE_SOURCES = (
    {
        "league": "La Liga",
        "path": os.path.join(DATA_DIR, "laliga_all_players.csv"),
    },
    {
        "league": "Premier League",
        "path": os.path.join(DATA_DIR, "prem_all_players.csv"),
    },
    {
        "league": "Serie A",
        "path": os.path.join(DATA_DIR, "seriea_all_players.csv"),
    },
)

POSITION_GROUPS = {
    "Forward": ("forward", "forwards", "striker", "strikers", "winger", "wingers", "fw", "st", "cf"),
    "Defender": ("defender", "defenders", "back", "backs", "cb", "lb", "rb", "df"),
    "Midfielder": ("midfielder", "midfielders", "midfield", "mf", "cm", "am", "dm"),
    "Goalkeeper": ("goalkeeper", "goalkeepers", "keeper", "keepers", "gk"),
}

NATIONALITY_KEYWORDS = {
    "spain": "Spain",
    "spanish": "Spain",
    "italy": "Italy",
    "italian": "Italy",
    "england": "England",
    "english": "England",
    "france": "France",
    "french": "France",
    "portugal": "Portugal",
    "portuguese": "Portugal",
    "argentina": "Argentina",
    "argentinian": "Argentina",
    "argentine": "Argentina",
    "brazil": "Brazil",
    "brazilian": "Brazil",
    "croatia": "Croatia",
    "croatian": "Croatia",
    "germany": "Germany",
    "german": "Germany",
    "belgium": "Belgium",
    "belgian": "Belgium",
    "netherlands": "Netherlands",
    "dutch": "Netherlands",
}

YEAR_PATTERN = re.compile(r"(19|20)\d{2}")

# No birth dates in league CSVs; earliest season is a lower bound on career start.
# If a player was at least this age in their first recorded season, their current age is at least
# (reference_year - career_start_year + MIN_SENIOR_DEBUT_AGE). Exclude when that is >= max_age_under.
MIN_SENIOR_DEBUT_AGE = 16
DEFAULT_YOUNG_MAX_AGE = 23

# Canonical country names; normalized keys must match `nationality_normalized` in CSV rows.
_AFRICA_COUNTRY_NAMES: tuple[str, ...] = (
    "Algeria",
    "Angola",
    "Benin",
    "Botswana",
    "Burkina Faso",
    "Burundi",
    "Cameroon",
    "Cape Verde",
    "Central African Republic",
    "Chad",
    "Comoros",
    "Congo",
    "Congo DR",
    "DR Congo",
    "Democratic Republic of the Congo",
    "Côte d'Ivoire",
    "Cote d'Ivoire",
    "Ivory Coast",
    "Djibouti",
    "Egypt",
    "Equatorial Guinea",
    "Eritrea",
    "Eswatini",
    "Ethiopia",
    "Gabon",
    "Gambia",
    "Ghana",
    "Guinea",
    "Guinea-Bissau",
    "Kenya",
    "Lesotho",
    "Liberia",
    "Libya",
    "Madagascar",
    "Malawi",
    "Mali",
    "Mauritania",
    "Mauritius",
    "Morocco",
    "Mozambique",
    "Namibia",
    "Niger",
    "Nigeria",
    "Rwanda",
    "São Tomé and Príncipe",
    "Senegal",
    "Seychelles",
    "Sierra Leone",
    "Somalia",
    "South Africa",
    "South Sudan",
    "Sudan",
    "Tanzania",
    "Togo",
    "Tunisia",
    "Uganda",
    "Zambia",
    "Zimbabwe",
)


def normalize_text(value: Optional[str]) -> str:
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKD", value)
    without_marks = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return " ".join(without_marks.casefold().split())


AFRICA_NATIONALITY_NORMALIZED = frozenset(
    normalize_text(n) for n in _AFRICA_COUNTRY_NAMES if normalize_text(n)
)


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> Optional[int]:
    parsed = safe_float(value)
    if parsed is None:
        return None
    return int(parsed)


def first_non_empty(*values: Any) -> Optional[str]:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def canonical_nationality(value: Optional[str]) -> Optional[str]:
    if not value:
        return None

    text = value.strip()
    key = normalize_text(text)
    aliases = {
        "pt": "Portugal",
        "ar": "Argentina",
        "br": "Brazil",
        "fr": "France",
        "es": "Spain",
        "it": "Italy",
        "gb-eng": "England",
        "england": "England",
    }
    return aliases.get(key, text)


def primary_position(value: Optional[str]) -> Optional[str]:
    if not value:
        return None

    normalized = normalize_text(value)
    if "goalkeeper" in normalized or normalized == "gk":
        return "Goalkeeper"
    if "defender" in normalized or normalized in {"df", "cb", "lb", "rb"}:
        return "Defender"
    if "midfielder" in normalized or normalized in {"mf", "cm", "am", "dm"}:
        return "Midfielder"
    if "forward" in normalized or normalized in {"fw", "st", "cf"}:
        return "Forward"
    if "winger" in normalized:
        return "Forward"
    return value.strip()


def choose_image(*values: Optional[str]) -> Optional[str]:
    for value in values:
        if not value:
            continue
        if "placeholder" in value:
            continue
        return value
    return None


def parse_years(text: Optional[str]) -> List[int]:
    if not text:
        return []
    return [int(match.group()) for match in YEAR_PATTERN.finditer(text)]


def expand_year_range(start: int, end: int) -> List[int]:
    if end < start:
        start, end = end, start
    return list(range(start, end + 1))


def extract_season_years(row: Dict[str, Any], league: str) -> List[int]:
    years: List[int] = []

    if league == "La Liga":
        season_range = row.get("season_range") or row.get("seasons_played") or ""
        parsed_years = parse_years(season_range)
        if len(parsed_years) >= 2:
            years.extend(expand_year_range(parsed_years[0], parsed_years[-1]))
        else:
            years.extend(parsed_years)
    elif league == "Serie A":
        season_range = row.get("season_range") or ""
        endpoints = parse_years(season_range)
        if len(endpoints) >= 2:
            years.extend(expand_year_range(endpoints[0], endpoints[1]))
        elif endpoints:
            years.append(endpoints[0])
        else:
            season = row.get("season") or ""
            season_years = parse_years(season)
            if season_years:
                years.append(season_years[0])
    elif league == "Premier League":
        season_range = row.get("season_range") or ""
        endpoints = parse_years(season_range)
        if len(endpoints) >= 2:
            years.extend(expand_year_range(endpoints[0], endpoints[1]))
        elif endpoints:
            years.append(endpoints[0])
        else:
            season = row.get("season") or row.get("season_range") or ""
            season_years = parse_years(season)
            years.extend(season_years)

    return sorted(set(years))


def rate_or_none(numerator: Optional[int], denominator: Optional[int]) -> Optional[float]:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def build_row_provenance_id(league: str, player_id: Optional[str], seasons: List[str]) -> str:
    season_label = seasons[0] if seasons else "unknown"
    base_player_id = player_id or "unknown"
    return f"{normalize_text(league)}:{base_player_id}:{normalize_text(season_label) or 'unknown'}"


def extract_stat_features(row: Dict[str, Any], league: str) -> Dict[str, Optional[float]]:
    if league == "La Liga":
        progressive_passes = safe_float(row.get("total_accurate_fwd_zone_pass"))
        key_passes = safe_float(row.get("total_att_assist"))
        pass_completion = rate_or_none(
            safe_int(row.get("total_accurate_pass")),
            safe_int(row.get("total_pass")),
        )
        tackles = safe_float(row.get("total_tackle"))
        interceptions = safe_float(row.get("total_interception"))
        recoveries = safe_float(row.get("total_recovery"))
        aerial_duels_won = safe_float(row.get("total_aerial_won"))
        duels = None
    elif league == "Premier League":
        progressive_passes = safe_float(row.get("forwardPasses"))
        key_passes = safe_float(row.get("keyPassesAttemptAssists"))
        pass_completion = rate_or_none(
            safe_int(row.get("totalPasses")) - safe_int(row.get("totalFwdZonePasses")) if safe_int(row.get("totalPasses")) is not None and safe_int(row.get("totalFwdZonePasses")) is not None else None,
            safe_int(row.get("totalPasses")),
        )
        if pass_completion is None:
            pass_completion = rate_or_none(
                safe_int(row.get("accuratePasses")),
                safe_int(row.get("totalPasses")),
            )
        tackles = safe_float(row.get("totalTackles"))
        interceptions = safe_float(row.get("interceptions"))
        recoveries = safe_float(row.get("recoveries"))
        aerial_duels_won = safe_float(row.get("aerialDuelsWon"))
        duels = safe_float(row.get("duelsWon"))
    else:
        progressive_passes = safe_float(row.get("Forward Passes"))
        key_passes = safe_float(row.get("Assists")) or safe_float(row.get("Goal Assists"))
        pass_completion = safe_float(row.get("accurate-pass-percentage"))
        tackles = safe_float(row.get("Tackles won")) or safe_float(row.get("Tackles"))
        interceptions = safe_float(row.get("Interceptions"))
        recoveries = safe_float(row.get("Recoveries"))
        aerial_duels_won = safe_float(row.get("Aerial Duels won"))
        duels = safe_float(row.get("Duels won"))

    return {
        "goals": safe_float(row.get("goals") or row.get("Goals") or row.get("total_goals")),
        "assists": safe_float(row.get("goalAssists") or row.get("assists") or row.get("Goal Assists") or row.get("total_assists")),
        "shots_on_target": safe_float(row.get("shotsOnTargetIncGoals") or row.get("Shots On Target ( inc goals )") or row.get("total_ontarget_attempt")),
        "dribbles_completed": safe_float(row.get("successfulDribbles") or row.get("successful-dribble") or row.get("total_dribbles_attempted")),
        "minutes": safe_float(row.get("timePlayed") or row.get("minutes-played") or row.get("Time Played") or row.get("total_mins_played")),
        "appearances": safe_float(row.get("gamesPlayed") or row.get("appearances") or row.get("games-played") or row.get("Games Played") or row.get("total_games")),
        "goals_per_game": None,
        "assists_per_game": None,
        "shot_on_target_ratio": None,
        "progressive_passes": progressive_passes,
        "key_passes": key_passes,
        "pass_completion": pass_completion,
        "tackles": tackles,
        "interceptions": interceptions,
        "recoveries": recoveries,
        "aerial_duels_won": aerial_duels_won,
        "duels": duels,
    }


def normalize_row(row: Dict[str, Any], league: str) -> Dict[str, Any]:
    player_id = first_non_empty(
        row.get("player_id"),
        row.get("provider_id"),
        row.get("opta_id"),
        row.get("id"),
    )

    if league == "La Liga":
        name = first_non_empty(row.get("player_name"), row.get("nickname")) or "Unknown Player"
        appearances = safe_int(row.get("total_games"))
        goals = safe_int(row.get("total_goals"))
        assists = safe_int(row.get("total_assists"))
        shots = safe_int(row.get("total_scoring_att"))
        shots_on_target = safe_int(row.get("total_ontarget_attempt"))
        dribbles = safe_int(row.get("total_dribbles_attempted"))
        seasons = [value for value in [row.get("season_range")] if value]
        team = row.get("team_name")
        nationality = canonical_nationality(row.get("country"))
        position = primary_position(row.get("position"))
        image = choose_image(row.get("player_image"))
        minutes = safe_int(row.get("total_mins_played"))
    elif league == "Premier League":
        name = (
            first_non_empty(
                f"{(row.get('first_name') or '').strip()} {(row.get('last_name') or '').strip()}".strip(),
                row.get("player_name"),
                row.get("known_name"),
            )
            or "Unknown Player"
        )
        appearances = safe_int(row.get("gamesPlayed") or row.get("appearances"))
        goals = safe_int(row.get("goals"))
        assists = safe_int(row.get("goalAssists"))
        shots = safe_int(row.get("totalShots"))
        shots_on_target = safe_int(row.get("shotsOnTargetIncGoals"))
        dribbles = safe_int(row.get("successfulDribbles"))
        season_range = row.get("season_range")
        seasons = [value for value in [season_range] if value]
        team = row.get("current_team_name")
        nationality = canonical_nationality(row.get("country"))
        position = primary_position(row.get("position"))
        image = choose_image(row.get("player_image"))
        minutes = safe_int(row.get("timePlayed"))
    else:
        name = first_non_empty(row.get("display_name"), row.get("short_name")) or "Unknown Player"
        appearances = safe_int(row.get("games-played") or row.get("Games Played"))
        goals = safe_int(row.get("goals") or row.get("Goals"))
        assists = safe_int(row.get("assists") or row.get("Goal Assists"))
        shots = safe_int(row.get("total-scoring-attempts") or row.get("Total Shots"))
        shots_on_target = safe_int(row.get("on-target-scoring-attempts") or row.get("Shots On Target ( inc goals )"))
        dribbles = safe_int(row.get("successful-dribble"))
        seasons = [value for value in [row.get("season_range"), row.get("season")] if value]
        team = row.get("team_name")
        nationality = canonical_nationality(row.get("nationality"))
        position = primary_position(row.get("role_label") or row.get("role"))
        image = choose_image(row.get("player_image"))
        minutes = safe_int(row.get("minutes-played") or row.get("Time Played"))

    normalized_name = normalize_text(name)
    season_years = extract_season_years(row, league)
    stat_features = extract_stat_features(row, league)
    stat_features["goals_per_game"] = rate_or_none(goals, appearances)
    stat_features["assists_per_game"] = rate_or_none(assists, appearances)
    stat_features["shot_on_target_ratio"] = rate_or_none(shots_on_target, shots)
    season_id = first_non_empty(row.get("season_id"), row.get("season"), row.get("season_range"))
    season_label = first_non_empty(row.get("season"), row.get("season_range"))

    return {
        "player_id": player_id,
        "name": name,
        "normalized_name": normalized_name,
        "nationality": nationality,
        "nationality_normalized": normalize_text(nationality),
        "position": position,
        "league": league,
        "team": team,
        "seasons": seasons,
        "season_years": season_years,
        "image": image,
        "appearances": appearances,
        "minutes": minutes,
        "goals": goals,
        "assists": assists,
        "shots": shots,
        "shots_on_target": shots_on_target,
        "dribbles_completed": dribbles,
        "goals_per_game": rate_or_none(goals, appearances),
        "assists_per_game": rate_or_none(assists, appearances),
        "shot_on_target_ratio": rate_or_none(shots_on_target, shots),
        "season_id": season_id,
        "season_label": season_label,
        "source_dataset": normalize_text(league).replace(" ", "_"),
        "row_id": build_row_provenance_id(league, player_id, seasons),
        "role_bucket": position or "Unknown",
        "stat_features": stat_features,
    }


def serialize_player(player: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "player_id": player.get("player_id"),
        "name": player.get("name"),
        "nationality": player.get("nationality"),
        "position": player.get("position"),
        "league": player.get("league"),
        "team": player.get("team"),
        "image": player.get("image"),
        "goals": player.get("goals"),
        "assists": player.get("assists"),
        "appearances": player.get("appearances"),
        "minutes": player.get("minutes"),
        "shots_on_target": player.get("shots_on_target"),
        "dribbles_completed": player.get("dribbles_completed"),
        "season_years": player.get("season_years", []),
        "seasons": player.get("seasons", []),
        "goals_per_game": player.get("goals_per_game"),
        "assists_per_game": player.get("assists_per_game"),
        "shot_on_target_ratio": player.get("shot_on_target_ratio"),
    }


def query_max_age_under(query: str) -> Optional[int]:
    """
    Parse an exclusive age ceiling (e.g. 'under 23' -> 23 means age must be < 23).
    'young' without a number defaults to DEFAULT_YOUNG_MAX_AGE.
    """
    text = normalize_text(query)
    if not text:
        return None
    for pattern in (
        r"\bunder\s+(\d{1,2})\b",
        r"\bbelow\s+(\d{1,2})\b",
        r"\byounger\s+than\s+(\d{1,2})\b",
        r"\bage\s+(\d{1,2})\s+or\s+younger\b",
    ):
        m = re.search(pattern, text)
        if m:
            n = int(m.group(1))
            if 1 <= n <= 99:
                return n
    if re.search(r"\byoung\b", text):
        return DEFAULT_YOUNG_MAX_AGE
    return None


def reference_year_for_queries() -> int:
    return date.today().year


def passes_max_age_under(
    max_age_under: Optional[int],
    reference_year: int,
    career_start_year: Optional[int],
) -> bool:
    if max_age_under is None:
        return True
    if career_start_year is None:
        return False
    youngest_plausible_now = reference_year - int(career_start_year) + MIN_SENIOR_DEBUT_AGE
    return youngest_plausible_now < max_age_under


def region_nationality_allowlist_from_text(query: str) -> Optional[frozenset[str]]:
    """If the query names a region (e.g. Africa), return allowed nationality_normalized values."""
    text = normalize_text(query)
    if not text:
        return None
    if re.search(r"\b(africa|african)\b", text):
        return AFRICA_NATIONALITY_NORMALIZED
    return None


def nationality_filter_from_text(query: str) -> Optional[str]:
    """Detect a canonical nationality from free text (keywords or 'from <country>')."""
    text = normalize_text(query)
    if not text:
        return None
    if region_nationality_allowlist_from_text(query):
        return None
    for keyword, nationality in NATIONALITY_KEYWORDS.items():
        if re.search(rf"\b{re.escape(keyword)}\b", text):
            return nationality
    from_match = re.search(
        r"\bfrom ([a-z]+(?: [a-z]+)?)(?=$|\b(?:in|during|between|with|and)\b|\d)",
        text,
    )
    if from_match:
        return from_match.group(1).title()
    return None


def parse_query(query: str) -> Dict[str, Any]:
    text = normalize_text(query)
    filters: Dict[str, Any] = {"sort_by": "goals"}

    if not text:
        return filters

    nationality_region = region_nationality_allowlist_from_text(query)
    if nationality_region:
        filters["nationality_region"] = nationality_region
    else:
        nationality = nationality_filter_from_text(query)
        if nationality:
            filters["nationality"] = nationality

    matched_positions: List[str] = []
    for position, keywords in POSITION_GROUPS.items():
        for keyword in keywords:
            if re.search(rf"\b{re.escape(keyword)}\b", text):
                matched_positions.append(position)
                break
    if matched_positions:
        filters["positions"] = sorted(set(matched_positions))

    decade_match = re.search(r"\b(19|20)\d0s\b", text)
    if decade_match:
        start = int(decade_match.group()[:4])
        filters["season_start"] = start
        filters["season_end"] = start + 9

    range_match = re.search(r"\b((?:19|20)\d{2})\s*(?:to|-)\s*((?:19|20)\d{2})\b", text)
    if range_match:
        filters["season_start"] = int(range_match.group(1))
        filters["season_end"] = int(range_match.group(2))

    if re.search(r"\bclinical\b", text):
        filters["sort_by"] = "shot_on_target_ratio"
    elif re.search(r"\bfast\b", text):
        filters["sort_by"] = "dribbles_completed"
    elif re.search(r"\b(best|top)\b", text):
        filters["sort_by"] = "goals"

    max_age_under = query_max_age_under(query)
    if max_age_under is not None:
        filters["max_age_under"] = max_age_under
        filters["reference_year"] = reference_year_for_queries()

    return filters


def boolean_search(filters: Dict[str, Any], players: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results = players

    nationality_region = filters.get("nationality_region")
    if nationality_region:
        results = [
            player
            for player in results
            if player.get("nationality_normalized") in nationality_region
        ]
    else:
        nationality = filters.get("nationality")
        if nationality:
            nationality_key = normalize_text(nationality)
            results = [
                player for player in results if player.get("nationality_normalized") == nationality_key
            ]

    positions = filters.get("positions")
    if positions:
        allowed = set(positions)
        results = [player for player in results if player.get("position") in allowed]

    season_start = filters.get("season_start")
    season_end = filters.get("season_end")
    if season_start is not None and season_end is not None:
        results = [
            player
            for player in results
            if any(season_start <= year <= season_end for year in player.get("season_years", []))
        ]

    max_age_under = filters.get("max_age_under")
    if max_age_under is not None:
        ref_y = int(filters.get("reference_year") or reference_year_for_queries())
        results = [
            player
            for player in results
            if passes_max_age_under(
                max_age_under,
                ref_y,
                career_start_year_for_normalized_key(player.get("normalized_name") or ""),
            )
        ]

    sort_by = filters.get("sort_by") or "goals"

    def descending_number(value: Any) -> float:
        if value is None:
            return float("inf")
        return -float(value)

    sorted_results = sorted(
        results,
        key=lambda player: (
            descending_number(player.get(sort_by)),
            descending_number(player.get("goals")),
            player.get("name", "").casefold(),
        ),
    )
    return [serialize_player(player) for player in sorted_results[:20]]


def load_player_index() -> Dict[str, Any]:
    players_by_name: Dict[str, List[Dict[str, Any]]] = {}
    player_list: List[Dict[str, Any]] = []
    for source in LEAGUE_SOURCES:
        if not os.path.exists(source["path"]):
            continue
        with open(source["path"], newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                normalized = normalize_row(row, source["league"])
                if not normalized["normalized_name"]:
                    continue
                players_by_name.setdefault(normalized["normalized_name"], []).append(normalized)
                player_list.append(normalized)

    return {
        "players_by_name": players_by_name,
        "player_list": player_list,
        "candidate_names": list(players_by_name.keys()),
    }


PLAYER_INDEX = load_player_index()


def career_start_year_for_normalized_key(normalized_name: str) -> Optional[int]:
    """
    Earliest calendar year seen in league rows. Premier League CSV rows are often
    career totals with no season_range, so we may only see another league's first year;
    single recent year + huge totals is treated as an unknown early start (veteran).
    """
    rows = PLAYER_INDEX["players_by_name"].get(normalized_name)
    if not rows:
        return None
    years = [y for r in rows for y in (r.get("season_years") or [])]
    if not years:
        return None
    ref_y = reference_year_for_queries()
    mn, mx = min(years), max(years)
    total_apps = sum(safe_int(r.get("appearances")) or 0 for r in rows)
    total_goals = sum(safe_int(r.get("goals")) or 0 for r in rows)
    year_span = mx - mn
    # Incomplete history: one recent season in data but massive totals (e.g. PL aggregate + one abroad year).
    if (
        mn == mx
        and mn >= ref_y - 2
        and (total_apps >= 100 or total_goals >= 35)
    ):
        return ref_y - 25
    # Few seasons in the extract vs very high minutes (e.g. only recent league years for a veteran).
    if year_span <= 4 and mn >= ref_y - 8 and total_apps >= 115:
        return ref_y - 25
    return mn


def career_start_year_for_embedding_player(
    normalized_player_key: str, embedding_meta: Dict[str, Any]
) -> Optional[int]:
    """Earliest season year: prefer live CSV index; embedding JSON can be incomplete."""
    from_csv = career_start_year_for_normalized_key(normalized_player_key)
    if from_csv is not None:
        return from_csv
    start = embedding_meta.get("career_start_year")
    if start is not None:
        return int(start)
    sy = embedding_meta.get("season_years") or []
    return min(sy) if sy else None


def nationality_normalized_for_embedding_player(
    normalized_player_key: str, embedding_meta: Dict[str, Any]
) -> str:
    """
    Normalized nationality for an embedding index key: prefer embedding metadata,
    else fall back to the live CSV player index (same key as normalized_name).
    """
    stored = embedding_meta.get("nationality_normalized")
    if stored:
        return stored
    nat = embedding_meta.get("nationality")
    if nat:
        return normalize_text(str(nat))
    rows = PLAYER_INDEX["players_by_name"].get(normalized_player_key)
    if not rows:
        return ""
    first = rows[0]
    return first.get("nationality_normalized") or normalize_text(first.get("nationality"))


def find_player_by_name(name: str) -> Optional[List[Dict[str, Any]]]:
    normalized_name = normalize_text(name)
    if not normalized_name:
        return None

    exact_matches = PLAYER_INDEX["players_by_name"].get(normalized_name)
    if exact_matches is not None:
        return [serialize_player(player) for player in exact_matches]

    fuzzy_matches = difflib.get_close_matches(
        normalized_name,
        PLAYER_INDEX["candidate_names"],
        n=1,
        cutoff=0.8,
    )
    if not fuzzy_matches:
        return None

    return [serialize_player(player) for player in PLAYER_INDEX["players_by_name"][fuzzy_matches[0]]]
