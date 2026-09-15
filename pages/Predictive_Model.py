from pathlib import Path
import base64
import json
import re
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


try:
    from sklearn.base import clone
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
    from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, r2_score, roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

APP_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = APP_DIR / "data"
LIVE_SOURCES_DIR = DATA_DIR / "live_sources"
PREDICTIVE_MODEL_DEBUG_LOG_PATH = LIVE_SOURCES_DIR / "predictive_model_projection_debug.json"

if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from site_nav import render_top_nav
from live_source_refresh import read_live_source_refresh_status
from season_utils import current_projected_finish, determine_active_nfl_season
import live_roster_scoring as roster_scoring


PRIMARY = "#F15A24"
SECONDARY = "#0073B7"
PAGE_BG = "#F6F8FB"
TEXT = "#07111f"
MUTED = "#64748B"
GRID = "#E5E7EB"
BRONCOS_LOGO_PATH = APP_DIR / "assets" / "broncos_logo_centered.png"
BRONCOS_LOGO_SOURCE = str(BRONCOS_LOGO_PATH)
HOME_FIELD_SCHEDULE_ADJUSTMENT = 1.0
AWAY_FIELD_SCHEDULE_ADJUSTMENT = 1.5
NEUTRAL_FIELD_SCHEDULE_ADJUSTMENT = 0.5

TEAM_ALIASES = {
    "ARZ": "ARI",
    "BLT": "BAL",
    "CLV": "CLE",
    "JAC": "JAX",
    "LAR": "LA",
    "STL": "LA",
    "SD": "LAC",
    "OAK": "LV",
    "WSH": "WAS",
}

TEAM_THEME_OVERRIDES = {
    "DEN": {"primary": PRIMARY, "secondary": SECONDARY},
}

DISPLAY_LABELS = {
    "season": "Season",
    "team": "Team",
    "current_wins": "Selected Season Wins",
    "current_losses": "Selected Season Losses",
    "current_ties": "Selected Season Ties",
    "next_season_wins": "Following-Season Wins",
    "wins_change": "Win Change Next Year",
    "predicted_next_wins": "Projected Following-Season Wins",
    "prob_10_plus_wins": "10+ Win Probability",
    "gini_score": "Gini Score",
    "gini_rank": "Gini Rank",
    "overall_estat": "Gini Score",
    "offense_estat": "Offense Gini",
    "defense_estat": "Defense Gini",
    "point_diff_per_game": "Point Differential per Game",
    "point_diff_rank": "Point Differential Rank",
    "off_adj_epa": "Adjusted Offensive EPA",
    "def_adj_epa": "Adjusted Defensive EPA",
    "success_margin": "Success Margin",
    "turnover_margin_per_game": "Turnover Margin per Game",
    "penalty_yards_margin_per_game": "Penalty Yard Margin per Game",
    "schedule_strength": "Schedule Strength",
    "sos_rank": "Schedule Strength Rank",
    "projection_schedule_strength": "Projection Schedule Strength",
    "projection_schedule_adjusted_strength": "Projection Schedule Strength",
    "projection_sos_rank": "Projection Schedule Rank",
    "projection_schedule_games": "Projection Schedule Games",
    "projection_home_games": "Projection Home Games",
    "projection_away_games": "Projection Away Games",
    "projection_neutral_games": "Projection Neutral Games",
    "projection_division_games": "Projection Division Games",
    "projection_conference_games": "Projection Conference Games",
    "projection_nonconference_games": "Projection Non-Conference Games",
    "projection_avg_rest_advantage": "Projection Rest Advantage",
    "projection_roster_score": "Live Leaderboard Roster Score",
    "projection_qb_score": "Roster QB Score",
    "projection_offense_skill_score": "Roster Skill Score",
    "projection_offensive_line_score": "Roster Offensive Line Score",
    "projection_defensive_front_score": "Roster Defensive Front Score",
    "projection_secondary_score": "Roster Secondary Score",
    "projection_premium_position_score": "Roster Premium Position Score",
    "projection_availability_score": "Roster Availability Score",
    "projection_rookie_projection_score": "Roster Rookie Score",
    "projection_continuity_score": "Roster Continuity Score",
    "offense_rank": "Offense Rank",
    "defense_rank": "Defense Rank",
    "balance_gap": "Offense/Defense Balance Gap",
    "elite_offense": "Elite Offense",
    "elite_defense": "Elite Defense",
    "two_way_team": "Two-Way Team",
    "top_gini_team": "Top Gini Team",
    "top_point_diff_team": "Top Point Differential Team",
    "super_square_profile": "Super Square Profile",
    "underperformer_signal": "Underperformer",
    "overperformer_signal": "Overperformer",
    "postseason_wins": "Postseason Wins",
    "postseason_games": "Postseason Games",
    "pythagorean_wins": "Expected Wins",
    "pythagorean_win_pct": "Expected Win %",
    "win_over_pythagorean": "Wins Above Expected",
    "epa_differential": "EPA Differential",
    "scoring_margin_signal": "Scoring Margin Signal",
}

BINARY_COLUMNS = {
    "elite_offense",
    "elite_defense",
    "two_way_team",
    "top_gini_team",
    "top_point_diff_team",
    "super_square_profile",
    "underperformer_signal",
    "overperformer_signal",
}

INTEGER_DISPLAY_COLUMNS = {
    "Season",
    "Selected Season Wins",
    "Following-Season Wins",
    "Projected Following-Season Wins",
    "Actual Following Season Wins",
    "Win Change Next Year",
    "Gini Rank",
    "Point Differential Rank",
    "Offense Rank",
    "Defense Rank",
    "Schedule Strength Rank",
    "Team-Seasons",
    "Projection Schedule Games",
    "Projection Home Games",
    "Projection Away Games",
    "Projection Neutral Games",
    "Projection Division Games",
    "Projection Conference Games",
    "Projection Non-Conference Games",
}

ONE_DECIMAL_DISPLAY_COLUMNS = {
    "Gini Score",
    "Offense Gini",
    "Defense Gini",
    "Point Differential per Game",
    "Schedule Strength",
    "Projection Schedule Strength",
    "Projection Rest Advantage",
    "Live Leaderboard Roster Score",
    "Roster QB Score",
    "Roster Skill Score",
    "Roster Offensive Line Score",
    "Roster Defensive Front Score",
    "Roster Secondary Score",
    "Roster Premium Position Score",
    "Roster Availability Score",
    "Roster Rookie Score",
    "Roster Continuity Score",
    "Average Following-Season Wins",
}

THREE_DECIMAL_DISPLAY_COLUMNS = {
    "Adjusted Offensive EPA",
    "Adjusted Defensive EPA",
    "Success Margin",
}


# -----------------------------
# Data helpers
# -----------------------------


BASE_DATA_FILES = {
    "team_season": DATA_DIR / "team_season_estat.csv",
    "team_game": DATA_DIR / "team_game_estat.csv",
    "games": DATA_DIR / "games_2005_onward.csv",
    "roster": DATA_DIR / "nfl_season_rosters_clean_2005_2025.csv",
    "team_assets": DATA_DIR / "teams_colors_logos.csv",
}

LIVE_ROSTER_SOURCE_FILES = {
    "weekly_rosters": LIVE_SOURCES_DIR / "weekly_rosters.csv",
    "player_weekly_stats": LIVE_SOURCES_DIR / "player_weekly_stats.csv",
    "player_season_stats": LIVE_SOURCES_DIR / "player_season_stats.csv",
    "player_snap_counts": LIVE_SOURCES_DIR / "snap_counts.csv",
    "injuries": LIVE_SOURCES_DIR / "injuries.csv",
    "transactions": LIVE_SOURCES_DIR / "transactions.csv",
    "depth_charts": LIVE_SOURCES_DIR / "depth_charts.csv",
    "draft_picks": LIVE_SOURCES_DIR / "draft_picks.csv",
    "contracts": LIVE_SOURCES_DIR / "contracts.csv",
    "players": LIVE_SOURCES_DIR / "players.csv",
}

LIVE_ROSTER_READ_KEYS = {
    "weekly_rosters",
    "player_weekly_stats",
    "player_season_stats",
    "player_snap_counts",
    "injuries",
    "transactions",
    "depth_charts",
}


def normalize_team(value):
    team = "" if pd.isna(value) else str(value).strip().upper()
    return TEAM_ALIASES.get(team, team)


def relative_path_label(path):
    path = Path(path)
    try:
        return str(path.relative_to(APP_DIR)).replace("\\", "/")
    except Exception:
        return path.name


def infer_schedule_year_from_path(path):
    match = re.search(r"(?:schedule|schedules)_(\d{4})", Path(path).stem, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def schedule_source_paths(include_base=False):
    paths = []
    if include_base:
        paths.append(BASE_DATA_FILES["games"])

    paths.append(LIVE_SOURCES_DIR / "schedules.csv")

    for root in (DATA_DIR, LIVE_SOURCES_DIR):
        if not root.exists():
            continue
        for pattern in ("schedule_*.csv", "schedules_*.csv"):
            paths.extend(root.glob(pattern))

    unique_paths = []
    seen = set()
    for path in paths:
        path = Path(path)
        key = str(path).lower()
        if key not in seen and path.exists():
            seen.add(key)
            unique_paths.append(path)
    return unique_paths


def live_roster_source_paths():
    return list(LIVE_ROSTER_SOURCE_FILES.values())


def season_roster_source_paths():
    return roster_scoring.season_roster_source_paths(DATA_DIR)


def data_file_signature():
    paths = list(BASE_DATA_FILES.values()) + schedule_source_paths(include_base=False) + live_roster_source_paths() + season_roster_source_paths()
    signature = []
    for path in paths:
        path = Path(path)
        if path.exists():
            stat = path.stat()
            signature.append((relative_path_label(path), True, int(stat.st_mtime_ns), int(stat.st_size)))
        else:
            signature.append((relative_path_label(path), False, None, None))
    return tuple(sorted(signature))


def schedule_source_priority(path):
    path = Path(path)
    name = path.name.lower()
    if infer_schedule_year_from_path(path) is not None:
        return 0
    if path.parent == LIVE_SOURCES_DIR and name == "schedules.csv":
        return 5
    if name == "games_2005_onward.csv":
        return 50
    return 20


def load_schedule_source_frames(base_games):
    sources = []
    messages = []

    if isinstance(base_games, pd.DataFrame) and not base_games.empty:
        sources.append(
            {
                "path": str(BASE_DATA_FILES["games"]),
                "label": "Historical/base schedule file",
                "priority": schedule_source_priority(BASE_DATA_FILES["games"]),
                "inferred_year": None,
                "df": base_games.copy(),
            }
        )

    for path in schedule_source_paths(include_base=False):
        try:
            schedule_df = pd.read_csv(path, low_memory=False)
        except Exception as exc:
            messages.append(f"Could not read schedule file {relative_path_label(path)}: {exc}")
            continue

        sources.append(
            {
                "path": str(path),
                "label": "Actual schedule file",
                "priority": schedule_source_priority(path),
                "inferred_year": infer_schedule_year_from_path(path),
                "df": schedule_df,
            }
        )

    sources.sort(key=lambda source: (source["priority"], relative_path_label(source["path"])))
    return sources, messages


@st.cache_data(show_spinner=False)
def load_data(file_signature=None):
    data = {}
    messages = []

    for key, path in BASE_DATA_FILES.items():
        if not path.exists():
            data[key] = pd.DataFrame()
            messages.append(f"Missing file: {path.name}")
            continue

        try:
            data[key] = pd.read_csv(path)
        except Exception as exc:
            data[key] = pd.DataFrame()
            messages.append(f"Could not read {path.name}: {exc}")

    season_rosters, season_roster_messages = roster_scoring.load_season_roster_sources(DATA_DIR)
    data["season_rosters"] = season_rosters
    messages.extend(season_roster_messages)

    future_data = {}
    future_sources = {}
    for key, path in LIVE_ROSTER_SOURCE_FILES.items():
        if key not in LIVE_ROSTER_READ_KEYS:
            continue

        if not path.exists():
            future_data[key] = pd.DataFrame()
            continue

        try:
            future_data[key] = pd.read_csv(path, low_memory=False)
            future_sources[key] = relative_path_label(path)
        except Exception as exc:
            future_data[key] = pd.DataFrame()
            messages.append(f"Could not read live roster source {relative_path_label(path)}: {exc}")

    data["future_data"] = future_data
    data["future_sources"] = future_sources

    schedule_sources, schedule_messages = load_schedule_source_frames(data.get("games", pd.DataFrame()))
    data["schedule_sources"] = schedule_sources
    messages.extend(schedule_messages)

    return data, messages


def clean_numeric_columns(df, columns):
    output = df.copy()
    for column in columns:
        if column in output.columns:
            output[column] = pd.to_numeric(output[column], errors="coerce")
    return output


def standardize_team_season(df):
    output = df.copy()
    if "team" in output.columns:
        output["team"] = output["team"].map(normalize_team)
    if "season" in output.columns:
        output["season"] = pd.to_numeric(output["season"], errors="coerce")
        output = output.dropna(subset=["season"])
        output["season"] = output["season"].astype(int)
    return output


def first_existing_column(df, candidates):
    for column in candidates:
        if column in df.columns:
            return column
    return None


def team_context_maps(team_assets):
    if not isinstance(team_assets, pd.DataFrame) or team_assets.empty:
        return {}, {}

    team_col = first_existing_column(team_assets, ["team_abbr", "team", "abbr"])
    conf_col = first_existing_column(team_assets, ["team_conf", "conference", "conf"])
    division_col = first_existing_column(team_assets, ["team_division", "division", "div"])

    if team_col is None:
        return {}, {}

    context = team_assets.copy()
    context["_team"] = context[team_col].map(normalize_team)

    conf_lookup = (
        context.dropna(subset=["_team"]).set_index("_team")[conf_col].astype(str).to_dict()
        if conf_col is not None
        else {}
    )
    division_lookup = (
        context.dropna(subset=["_team"]).set_index("_team")[division_col].astype(str).to_dict()
        if division_col is not None
        else {}
    )
    return conf_lookup, division_lookup


def regular_schedule_rows_for_year(schedule_source, schedule_year):
    schedule = schedule_source.get("df", pd.DataFrame())
    if not isinstance(schedule, pd.DataFrame) or schedule.empty:
        return pd.DataFrame()

    schedule = schedule.copy()
    season_col = first_existing_column(schedule, ["season", "game_season"])
    inferred_year = schedule_source.get("inferred_year")

    if season_col is not None:
        season_values = pd.to_numeric(schedule[season_col], errors="coerce")
        schedule = schedule[season_values.eq(int(schedule_year))].copy()
    elif inferred_year != int(schedule_year):
        return pd.DataFrame()

    if schedule.empty:
        return pd.DataFrame()

    game_type_col = first_existing_column(schedule, ["game_type", "season_type"])
    if game_type_col is not None:
        game_type = schedule[game_type_col].astype(str).str.upper().str.strip()
        schedule = schedule[game_type.str.startswith("REG")].copy()

    home_col = first_existing_column(schedule, ["home_team", "home"])
    away_col = first_existing_column(schedule, ["away_team", "away"])
    if home_col is None or away_col is None:
        return pd.DataFrame()

    schedule["_home_team"] = schedule[home_col].map(normalize_team)
    schedule["_away_team"] = schedule[away_col].map(normalize_team)
    schedule = schedule[(schedule["_home_team"] != "") & (schedule["_away_team"] != "")].copy()
    return schedule


def schedule_completion_status(schedule, schedule_year):
    if schedule.empty:
        return False, "No regular-season schedule rows found."

    expected_games = expected_regular_season_games(schedule_year)
    teams = pd.concat([schedule["_home_team"], schedule["_away_team"]], ignore_index=True)
    team_counts = teams.value_counts()
    if len(team_counts) < 30:
        return False, f"Only {len(team_counts)} teams found in the schedule."
    if team_counts.min() < expected_games:
        return False, f"At least one team has fewer than {expected_games} scheduled games."
    return True, f"{len(schedule)} games across {len(team_counts)} teams."


def choose_schedule_for_year(schedule_sources, schedule_year):
    incomplete_messages = []

    for source in schedule_sources or []:
        schedule = regular_schedule_rows_for_year(source, schedule_year)
        complete, detail = schedule_completion_status(schedule, schedule_year)
        source_path = relative_path_label(source.get("path", "schedule"))

        if complete:
            return schedule, source, detail, incomplete_messages

        if not schedule.empty:
            incomplete_messages.append(f"{source_path}: {detail}")

    return pd.DataFrame(), None, "", incomplete_messages


def available_schedule_years(schedule_sources):
    years = set()
    for source in schedule_sources or []:
        schedule = source.get("df", pd.DataFrame())
        if not isinstance(schedule, pd.DataFrame) or schedule.empty:
            continue

        season_col = first_existing_column(schedule, ["season", "game_season"])
        if season_col is not None:
            values = pd.to_numeric(schedule[season_col], errors="coerce").dropna().astype(int)
            years.update(values.unique().tolist())
        elif source.get("inferred_year") is not None:
            years.add(int(source["inferred_year"]))

    return sorted(years)


def strength_lookup_for_feature_year(model_df, feature_year):
    if model_df.empty or "season" not in model_df.columns or "team" not in model_df.columns:
        return {}, pd.NA, None, ""

    available_years = sorted(
        pd.to_numeric(model_df["season"], errors="coerce")
        .dropna()
        .astype(int)
        .loc[lambda seasons: seasons <= int(feature_year)]
        .unique()
        .tolist()
    )
    if not available_years:
        return {}, pd.NA, None, ""

    strength_year = available_years[-1]
    strength_df = model_df[pd.to_numeric(model_df["season"], errors="coerce").eq(strength_year)].copy()

    strength_col = first_existing_column(
        strength_df,
        ["gini_score", "custom_estat", "overall_estat", "scoring_margin_signal", "point_diff_per_game"],
    )
    if strength_col is None:
        return {}, pd.NA, strength_year, ""

    strength_df["_team"] = strength_df["team"].map(normalize_team)
    strength_df["_strength"] = pd.to_numeric(strength_df[strength_col], errors="coerce")
    strength_df = strength_df.dropna(subset=["_team", "_strength"])
    if strength_df.empty:
        return {}, pd.NA, strength_year, display_label(strength_col)

    league_avg = float(strength_df["_strength"].mean())
    lookup = strength_df.drop_duplicates("_team").set_index("_team")["_strength"].to_dict()
    return lookup, league_avg, strength_year, display_label(strength_col)


def schedule_location_for_team(row, is_home_team):
    location = str(row.get("location", "")).strip().lower()
    if "neutral" in location:
        return "neutral"
    return "home" if is_home_team else "away"


def venue_difficulty_adjustment(location):
    if location == "home":
        return -HOME_FIELD_SCHEDULE_ADJUSTMENT
    if location == "away":
        return AWAY_FIELD_SCHEDULE_ADJUSTMENT
    return NEUTRAL_FIELD_SCHEDULE_ADJUSTMENT


def calculate_projection_schedule_features(
    schedule,
    schedule_year,
    feature_year,
    strength_lookup,
    league_avg_strength,
    strength_year,
    strength_label,
    team_assets,
    source,
    source_detail,
):
    conf_lookup, division_lookup = team_context_maps(team_assets)
    div_col = first_existing_column(schedule, ["div_game", "division_game"])
    week_col = first_existing_column(schedule, ["week", "game_week", "week_number"])
    home_rest_col = first_existing_column(schedule, ["home_rest"])
    away_rest_col = first_existing_column(schedule, ["away_rest"])
    entries = []

    for _, row in schedule.iterrows():
        home = normalize_team(row.get("_home_team"))
        away = normalize_team(row.get("_away_team"))
        if not home or not away:
            continue

        game_week = pd.to_numeric(row.get(week_col), errors="coerce") if week_col else pd.NA
        div_game_value = pd.to_numeric(row.get(div_col), errors="coerce") if div_col else pd.NA
        home_rest = pd.to_numeric(row.get(home_rest_col), errors="coerce") if home_rest_col else pd.NA
        away_rest = pd.to_numeric(row.get(away_rest_col), errors="coerce") if away_rest_col else pd.NA

        for team, opponent, is_home_team in ((home, away, True), (away, home, False)):
            opponent_strength = strength_lookup.get(opponent, league_avg_strength)
            location = schedule_location_for_team(row, is_home_team)
            venue_adjustment = venue_difficulty_adjustment(location)

            if pd.notna(home_rest) and pd.notna(away_rest):
                rest_advantage = float(home_rest - away_rest) if is_home_team else float(away_rest - home_rest)
            else:
                rest_advantage = pd.NA

            rest_adjustment = -0.15 * float(rest_advantage) if pd.notna(rest_advantage) else 0.0

            team_conf = conf_lookup.get(team)
            opponent_conf = conf_lookup.get(opponent)
            team_division = division_lookup.get(team)
            opponent_division = division_lookup.get(opponent)
            conference_game = int(team_conf == opponent_conf) if team_conf and opponent_conf else pd.NA
            if pd.notna(div_game_value):
                division_game = int(float(div_game_value) == 1)
            elif team_division and opponent_division:
                division_game = int(team_division == opponent_division)
            else:
                division_game = pd.NA

            entries.append(
                {
                    "season": int(feature_year),
                    "team": team,
                    "projection_year": int(schedule_year),
                    "projection_week": game_week,
                    "projection_opponent": opponent,
                    "projection_opponent_strength": opponent_strength,
                    "projection_location": location,
                    "projection_venue_adjustment": venue_adjustment,
                    "projection_rest_advantage": rest_advantage,
                    "projection_adjusted_opponent_strength": opponent_strength + venue_adjustment + rest_adjustment,
                    "projection_division_game": division_game,
                    "projection_conference_game": conference_game,
                }
            )

    if not entries:
        return pd.DataFrame()

    games_df = pd.DataFrame(entries)
    grouped = (
        games_df.groupby(["season", "team"], as_index=False)
        .agg(
            projection_schedule_games=("projection_opponent", "count"),
            projection_schedule_strength=("projection_opponent_strength", "mean"),
            projection_schedule_adjusted_strength=("projection_adjusted_opponent_strength", "mean"),
            projection_home_games=("projection_location", lambda values: int((values == "home").sum())),
            projection_away_games=("projection_location", lambda values: int((values == "away").sum())),
            projection_neutral_games=("projection_location", lambda values: int((values == "neutral").sum())),
            projection_division_games=("projection_division_game", "sum"),
            projection_conference_games=("projection_conference_game", "sum"),
            projection_avg_rest_advantage=("projection_rest_advantage", "mean"),
        )
    )
    grouped["projection_nonconference_games"] = grouped["projection_schedule_games"] - grouped["projection_conference_games"].fillna(0)
    grouped["projection_sos_rank"] = grouped["projection_schedule_adjusted_strength"].rank(ascending=False, method="min")
    grouped["projection_schedule_year"] = int(schedule_year)
    grouped["projection_schedule_feature_year"] = int(feature_year)
    grouped["projection_opponent_strength_year"] = int(strength_year) if strength_year is not None else pd.NA
    grouped["projection_opponent_strength_source"] = f"{strength_year} {strength_label}".strip()
    grouped["projection_schedule_source"] = relative_path_label(source.get("path", "schedule"))
    grouped["projection_schedule_status"] = (
        f"Using actual {schedule_year} schedule from {grouped['projection_schedule_source'].iloc[0]} "
        f"({source_detail}); opponent strength source: {grouped['projection_opponent_strength_source'].iloc[0]}."
    )
    return grouped


def add_projection_schedule_features(model_df, schedule_sources, team_assets):
    if model_df.empty or "season" not in model_df.columns or "team" not in model_df.columns:
        return model_df, []

    output = model_df.copy()
    feature_frames = []
    notes = []
    seasons = sorted(pd.to_numeric(output["season"], errors="coerce").dropna().astype(int).unique())

    for feature_year in seasons:
        schedule_year = feature_year + 1
        schedule, source, source_detail, incomplete_messages = choose_schedule_for_year(schedule_sources, schedule_year)
        if source is None or schedule.empty:
            if incomplete_messages:
                notes.append(
                    f"Projection schedule fallback for {schedule_year}: actual schedule rows were incomplete. "
                    + " | ".join(incomplete_messages[:2])
                )
            continue

        strength_lookup, league_avg_strength, strength_year, strength_label = strength_lookup_for_feature_year(output, feature_year)
        if not strength_lookup or pd.isna(league_avg_strength):
            notes.append(f"Projection schedule fallback for {schedule_year}: no opponent strength data was available through {feature_year}.")
            continue

        schedule_features = calculate_projection_schedule_features(
            schedule,
            schedule_year,
            feature_year,
            strength_lookup,
            league_avg_strength,
            strength_year,
            strength_label,
            team_assets,
            source,
            source_detail,
        )
        if not schedule_features.empty:
            feature_frames.append(schedule_features)

    if not feature_frames:
        return output, notes

    schedule_feature_df = pd.concat(feature_frames, ignore_index=True)
    output = output.merge(schedule_feature_df, on=["season", "team"], how="left")
    return output, notes


def empty_player_stats():
    return {
        "weekly": pd.DataFrame(),
        "seasonal": pd.DataFrame(),
        "snaps": pd.DataFrame(),
        "injuries": pd.DataFrame(),
    }


def roster_file_source_label(roster_file):
    if roster_file == "nfl_season_rosters_clean_2005_2025.csv":
        return relative_path_label(BASE_DATA_FILES["roster"])
    if roster_file and roster_file != "No roster file found":
        roster_path = Path(str(roster_file))
        if roster_path.is_absolute() or roster_path.exists():
            return relative_path_label(roster_path)
        return f"data/{roster_path.name}"
    return None


def player_stat_source_files(player_stats):
    sources = []
    if not player_stats.get("weekly", pd.DataFrame()).empty:
        sources.append(relative_path_label(LIVE_ROSTER_SOURCE_FILES["player_weekly_stats"]))
    if not player_stats.get("seasonal", pd.DataFrame()).empty:
        sources.append(relative_path_label(LIVE_ROSTER_SOURCE_FILES["player_season_stats"]))
    if not player_stats.get("snaps", pd.DataFrame()).empty:
        sources.append(relative_path_label(LIVE_ROSTER_SOURCE_FILES["player_snap_counts"]))
    return sources


def add_projection_roster_features(model_df, historical_roster=None, season_rosters=None, future_data=None):
    if model_df.empty or "season" not in model_df.columns or "team" not in model_df.columns:
        return model_df, []

    output = model_df.copy()
    notes = [
        "Using the shared Live Leaderboard roster score for projection roster context; historical backtests use local season rosters and current/future projections use live roster/depth/player files when available."
    ]
    feature_frames = []
    seasons = sorted(pd.to_numeric(output["season"], errors="coerce").dropna().astype(int).unique())
    if not seasons:
        return output, notes

    live_context_start_year = int(max(seasons)) + 1
    live_player_stats, _ = roster_scoring.load_player_stats(future_data or {})
    live_player_stats = roster_scoring.standardize_player_stats(live_player_stats)
    blank_player_stats = empty_player_stats()

    for feature_year in seasons:
        roster_context_year = int(feature_year) + 1
        team_universe = sorted(output.loc[output["season"].eq(feature_year), "team"].dropna().astype(str).unique())
        if not team_universe:
            continue

        use_live_sources = roster_context_year >= live_context_start_year
        context_future_data = future_data if use_live_sources else {}
        player_stats = live_player_stats if use_live_sources else blank_player_stats

        local_roster, roster_file, _ = roster_scoring.load_local_roster(
            roster_context_year,
            historical_roster if historical_roster is not None else pd.DataFrame(),
            season_rosters or {},
        )
        roster_context, context_info = roster_scoring.build_roster_context(local_roster, context_future_data, roster_context_year)
        roster_scores = roster_scoring.calculate_team_roster_score(
            roster_context,
            roster_context_year,
            team_universe,
            historical_roster=historical_roster if historical_roster is not None else pd.DataFrame(),
            transactions=roster_scoring.load_transaction_feed(context_future_data),
            player_stats=player_stats,
        )
        if roster_scores.empty:
            continue

        source_files = set(context_info.get("source_files", []))
        roster_source = roster_file_source_label(roster_file)
        if roster_source and not context_info.get("used_live_roster"):
            source_files.add(roster_source)
        if use_live_sources:
            source_files.update(player_stat_source_files(player_stats))
        source_summary = ", ".join(sorted(source_files)) if source_files else "neutral roster defaults"

        rename_map = {
            column: f"projection_{column}"
            for column in roster_scoring.ROSTER_SCORE_COLUMNS
            if column != "team"
        }
        roster_scores = roster_scores.rename(columns=rename_map)
        roster_scores["season"] = int(feature_year)
        roster_scores["projection_roster_context_year"] = int(roster_context_year)
        roster_scores["projection_roster_source_files"] = source_summary
        roster_scores["projection_roster_status"] = (
            f"{context_info.get('status', 'Using Live Leaderboard roster score.')} "
            f"Source files: {source_summary}."
        )
        roster_scores["projection_roster_used_live_sources"] = bool(use_live_sources)

        unique_roster_scores = pd.to_numeric(roster_scores.get("projection_roster_score"), errors="coerce").round(4).nunique(dropna=True)
        if use_live_sources:
            notes.append(f"Using Live Leaderboard roster score for {roster_context_year} projections from {source_summary}.")
        if unique_roster_scores <= 1:
            notes.append(f"Projection roster scores for {roster_context_year} are not varying across teams; using neutral roster defaults for that context.")

        feature_frames.append(roster_scores)

    if not feature_frames:
        notes.append("Projection roster score fallback: no roster context rows were available.")
        return output, notes

    roster_feature_df = pd.concat(feature_frames, ignore_index=True)
    output = output.merge(roster_feature_df, on=["season", "team"], how="left")
    return output, notes


def calculate_team_records(team_game, games):
    record_cols = ["season", "team", "current_wins", "current_losses", "current_ties", "scored_games"]
    postseason_cols = ["season", "team", "postseason_wins", "postseason_games"]

    records = pd.DataFrame(columns=record_cols)
    postseason_records = pd.DataFrame(columns=postseason_cols)
    notes = []

    if not team_game.empty and {"season", "team", "points_for", "points_against"}.issubset(team_game.columns):
        game_df = standardize_team_season(team_game)
        if "season_type" in game_df.columns:
            game_df = game_df[game_df["season_type"].astype(str).str.upper().eq("REG")]

        game_df = clean_numeric_columns(game_df, ["points_for", "points_against"])
        game_df = game_df.dropna(subset=["season", "team", "points_for", "points_against"])

        if not game_df.empty:
            game_df["win"] = (game_df["points_for"] > game_df["points_against"]).astype(int)
            game_df["loss"] = (game_df["points_for"] < game_df["points_against"]).astype(int)
            game_df["tie"] = (game_df["points_for"] == game_df["points_against"]).astype(int)

            records = (
                game_df.groupby(["season", "team"], as_index=False)
                .agg(
                    current_wins=("win", "sum"),
                    current_losses=("loss", "sum"),
                    current_ties=("tie", "sum"),
                    scored_games=("game_id", "nunique") if "game_id" in game_df.columns else ("win", "size"),
                    points_for_total=("points_for", "sum"),
                    points_against_total=("points_against", "sum"),
                )
            )
    else:
        notes.append("Regular-season wins could not be calculated because team game scores are missing.")

    if not games.empty and {"season", "game_type", "home_team", "away_team", "home_score", "away_score"}.issubset(games.columns):
        playoff_df = games.copy()
        playoff_df["season"] = pd.to_numeric(playoff_df["season"], errors="coerce")
        playoff_df = playoff_df.dropna(subset=["season"])
        playoff_df["season"] = playoff_df["season"].astype(int)
        playoff_df["game_type"] = playoff_df["game_type"].astype(str).str.upper()
        playoff_df = playoff_df[~playoff_df["game_type"].eq("REG")]
        playoff_df = clean_numeric_columns(playoff_df, ["home_score", "away_score"])
        playoff_df = playoff_df.dropna(subset=["home_score", "away_score", "home_team", "away_team"])

        if not playoff_df.empty:
            winners = []
            appearances = []
            for _, row in playoff_df.iterrows():
                home = str(row["home_team"]).strip()
                away = str(row["away_team"]).strip()
                appearances.append({"season": row["season"], "team": home, "postseason_games": 1})
                appearances.append({"season": row["season"], "team": away, "postseason_games": 1})

                if row["home_score"] > row["away_score"]:
                    winners.append({"season": row["season"], "team": home, "postseason_wins": 1})
                elif row["away_score"] > row["home_score"]:
                    winners.append({"season": row["season"], "team": away, "postseason_wins": 1})

            appearance_df = pd.DataFrame(appearances)
            winner_df = pd.DataFrame(winners)

            postseason_records = (
                appearance_df.groupby(["season", "team"], as_index=False)["postseason_games"]
                .sum()
                .merge(
                    winner_df.groupby(["season", "team"], as_index=False)["postseason_wins"].sum(),
                    on=["season", "team"],
                    how="left",
                )
            )
            postseason_records["postseason_wins"] = postseason_records["postseason_wins"].fillna(0)
        else:
            notes.append("Postseason prediction is limited because no playoff rows were found in the games file.")
    else:
        notes.append("Postseason prediction is limited because the games file does not include complete playoff fields.")

    return records, postseason_records, notes


@st.cache_data(show_spinner=False)
def build_model_dataset(team_season, team_game, games, schedule_sources=None, team_assets=None, historical_roster=None, season_rosters=None, future_data=None):
    notes = []
    missing_columns = []

    if team_season.empty:
        return pd.DataFrame(), [], ["team_season_estat.csv is required for the predictive model."]

    required = {"season", "team"}
    if not required.issubset(team_season.columns):
        missing = sorted(required - set(team_season.columns))
        return pd.DataFrame(), missing, [f"Missing required season columns: {', '.join(missing)}"]

    model_df = standardize_team_season(team_season)

    numeric_candidates = [
        "games",
        "points_for",
        "points_against",
        "point_diff",
        "off_epa_per_play",
        "off_adj_epa",
        "off_success_rate",
        "def_epa_allowed_per_play",
        "def_adj_epa",
        "def_success_allowed",
        "penalty_yards_margin",
        "turnover_margin",
        "point_diff_per_game",
        "turnover_margin_per_game",
        "penalty_yards_margin_per_game",
        "success_margin",
        "yards_per_play",
        "offense_estat",
        "defense_estat",
        "overall_estat",
        "custom_estat",
        "schedule_strength",
        "overall_rank",
        "offense_rank",
        "defense_rank",
        "sos_rank",
    ]
    model_df = clean_numeric_columns(model_df, numeric_candidates)

    records, postseason_records, record_notes = calculate_team_records(team_game, games)
    notes.extend(record_notes)

    if not records.empty:
        model_df = model_df.merge(records, on=["season", "team"], how="left")
    else:
        missing_columns.append("current_wins")

    # Add accuracy-focused carry-forward features.
    # These use the full completed season profile to predict the following/selected year.
    if "points_for" not in model_df.columns and "points_for_total" in model_df.columns:
        model_df["points_for"] = model_df["points_for_total"]
    elif "points_for" in model_df.columns and "points_for_total" in model_df.columns:
        model_df["points_for"] = model_df["points_for"].fillna(model_df["points_for_total"])

    if "points_against" not in model_df.columns and "points_against_total" in model_df.columns:
        model_df["points_against"] = model_df["points_against_total"]
    elif "points_against" in model_df.columns and "points_against_total" in model_df.columns:
        model_df["points_against"] = model_df["points_against"].fillna(model_df["points_against_total"])

    if "scored_games" in model_df.columns:
        games_played_for_model = pd.to_numeric(model_df["scored_games"], errors="coerce")
    elif "games" in model_df.columns:
        games_played_for_model = pd.to_numeric(model_df["games"], errors="coerce")
    else:
        games_played_for_model = pd.Series(pd.NA, index=model_df.index)

    if {"points_for", "points_against"}.issubset(model_df.columns):
        pf = pd.to_numeric(model_df["points_for"], errors="coerce")
        pa = pd.to_numeric(model_df["points_against"], errors="coerce")
        exponent = 2.37
        denominator = (pf.clip(lower=0) ** exponent) + (pa.clip(lower=0) ** exponent)
        model_df["pythagorean_win_pct"] = ((pf.clip(lower=0) ** exponent) / denominator).where(denominator > 0)
        model_df["pythagorean_wins"] = model_df["pythagorean_win_pct"] * games_played_for_model

        if "point_diff" not in model_df.columns:
            model_df["point_diff"] = pf - pa
        else:
            model_df["point_diff"] = pd.to_numeric(model_df["point_diff"], errors="coerce").fillna(pf - pa)

        if "point_diff_per_game" not in model_df.columns:
            model_df["point_diff_per_game"] = model_df["point_diff"] / games_played_for_model.replace(0, pd.NA)
        else:
            model_df["point_diff_per_game"] = pd.to_numeric(model_df["point_diff_per_game"], errors="coerce").fillna(
                model_df["point_diff"] / games_played_for_model.replace(0, pd.NA)
            )

    if {"current_wins", "pythagorean_wins"}.issubset(model_df.columns):
        model_df["win_over_pythagorean"] = pd.to_numeric(model_df["current_wins"], errors="coerce") - pd.to_numeric(model_df["pythagorean_wins"], errors="coerce")

    if {"off_adj_epa", "def_adj_epa"}.issubset(model_df.columns):
        model_df["epa_differential"] = pd.to_numeric(model_df["off_adj_epa"], errors="coerce") + pd.to_numeric(model_df["def_adj_epa"], errors="coerce")

    if {"point_diff_per_game", "success_margin"}.issubset(model_df.columns):
        model_df["scoring_margin_signal"] = (
            pd.to_numeric(model_df["point_diff_per_game"], errors="coerce")
            + 25 * pd.to_numeric(model_df["success_margin"], errors="coerce")
        )

    if not postseason_records.empty:
        model_df = model_df.merge(postseason_records, on=["season", "team"], how="left")
        model_df["postseason_wins"] = model_df["postseason_wins"].fillna(0)
        model_df["postseason_games"] = model_df["postseason_games"].fillna(0)
    else:
        model_df["postseason_wins"] = pd.NA
        model_df["postseason_games"] = pd.NA

    if "custom_estat" in model_df.columns:
        model_df["gini_score"] = model_df["custom_estat"]
    elif "overall_estat" in model_df.columns:
        model_df["gini_score"] = model_df["overall_estat"]
    elif "estat_raw" in model_df.columns:
        model_df["gini_score"] = pd.to_numeric(model_df["estat_raw"], errors="coerce")
    else:
        missing_columns.append("gini_score")

    if "gini_score" in model_df.columns:
        model_df["gini_rank"] = model_df.groupby("season")["gini_score"].rank(ascending=False, method="min")
    elif "overall_rank" in model_df.columns:
        model_df["gini_rank"] = model_df["overall_rank"]

    if "point_diff_per_game" in model_df.columns:
        model_df["point_diff_rank"] = model_df.groupby("season")["point_diff_per_game"].rank(ascending=False, method="min")

    if {"offense_rank", "defense_rank"}.issubset(model_df.columns):
        model_df["balance_gap"] = (model_df["offense_rank"] - model_df["defense_rank"]).abs()
        model_df["elite_offense"] = (model_df["offense_rank"] <= 8).astype(int)
        model_df["elite_defense"] = (model_df["defense_rank"] <= 8).astype(int)
        model_df["two_way_team"] = ((model_df["offense_rank"] <= 12) & (model_df["defense_rank"] <= 12)).astype(int)
        model_df["strong_unit"] = ((model_df["offense_rank"] <= 12) | (model_df["defense_rank"] <= 12)).astype(int)
    else:
        missing_columns.append("offense_rank/defense_rank")

    if "gini_rank" in model_df.columns:
        model_df["top_gini_team"] = (model_df["gini_rank"] <= 6).astype(int)

    if "point_diff_rank" in model_df.columns:
        model_df["top_point_diff_team"] = (model_df["point_diff_rank"] <= 6).astype(int)

    if {"top_gini_team", "top_point_diff_team", "strong_unit"}.issubset(model_df.columns):
        model_df["super_square_profile"] = (
            (model_df["top_gini_team"] == 1)
            & (model_df["top_point_diff_team"] == 1)
            & (model_df["strong_unit"] == 1)
        ).astype(int)

    if "current_wins" in model_df.columns:
        model_df["wins_rank"] = model_df.groupby("season")["current_wins"].rank(ascending=False, method="min")
        if "gini_rank" in model_df.columns:
            model_df["profile_win_gap"] = model_df["wins_rank"] - model_df["gini_rank"]
            model_df["underperformer_signal"] = (model_df["profile_win_gap"] >= 4).astype(int)
            model_df["overperformer_signal"] = (model_df["profile_win_gap"] <= -4).astype(int)

        model_df = model_df.sort_values(["team", "season"])
        model_df["next_season"] = model_df.groupby("team")["season"].shift(-1)
        model_df["next_season_wins"] = model_df.groupby("team")["current_wins"].shift(-1)
        model_df["regular_season_complete"] = model_df.apply(
            lambda row: row_has_completed_regular_season(row, row.get("season")), axis=1
        )
        model_df["next_regular_season_complete"] = model_df.groupby("team")["regular_season_complete"].shift(-1)
        incomplete_target = (
            model_df["next_season"].ne(model_df["season"] + 1)
            | ~model_df["next_regular_season_complete"].fillna(False).astype(bool)
        )
        model_df.loc[incomplete_target, "next_season_wins"] = pd.NA
        model_df["wins_change"] = model_df["next_season_wins"] - model_df["current_wins"]
        model_df["strong_next_season"] = (model_df["next_season_wins"] >= 10).astype("Int64")
    else:
        notes.append("Next-season targets are unavailable because current-season wins could not be calculated.")

    model_df, schedule_notes = add_projection_schedule_features(model_df, schedule_sources or [], team_assets)
    notes.extend(schedule_notes)

    model_df, roster_notes = add_projection_roster_features(
        model_df,
        historical_roster=historical_roster,
        season_rosters=season_rosters,
        future_data=future_data,
    )
    notes.extend(roster_notes)

    return model_df, sorted(set(missing_columns)), notes


def display_label(column):
    return DISPLAY_LABELS.get(column, str(column).replace("_", " ").title())


def yes_no(value):
    if pd.isna(value):
        return "-"
    return "Yes" if int(value) == 1 else "No"


def profile_text(value, positive_label, negative_label):
    if pd.isna(value):
        return "-"
    return positive_label if int(value) == 1 else negative_label


def get_feature_candidates(model_df):
    # Regular-season model: full previous-season profile predicts the selected target year.
    # Pythagorean/scoreboard features are intentionally first because they backtested as the
    # most stable carry-forward signals, while Gini/EPA explain team quality underneath.
    candidates = [
        "current_wins",
        "pythagorean_wins",
        "win_over_pythagorean",
        "point_diff_per_game",
        "points_for",
        "points_against",
        "gini_score",
        "gini_rank",
        "offense_estat",
        "defense_estat",
        "off_adj_epa",
        "def_adj_epa",
        "epa_differential",
        "success_margin",
        "schedule_strength",
        "sos_rank",
        "projection_schedule_strength",
        "projection_schedule_adjusted_strength",
        "projection_sos_rank",
        "projection_schedule_games",
        "projection_home_games",
        "projection_away_games",
        "projection_neutral_games",
        "projection_division_games",
        "projection_conference_games",
        "projection_nonconference_games",
        "projection_avg_rest_advantage",
        "projection_roster_score",
        "projection_qb_score",
        "projection_offense_skill_score",
        "projection_offensive_line_score",
        "projection_defensive_front_score",
        "projection_secondary_score",
        "projection_premium_position_score",
        "projection_availability_score",
        "projection_rookie_projection_score",
        "projection_continuity_score",
        "turnover_margin_per_game",
        "penalty_yards_margin_per_game",
        "offense_rank",
        "defense_rank",
        "balance_gap",
        "two_way_team",
        "super_square_profile",
        "underperformer_signal",
        "overperformer_signal",
        "scoring_margin_signal",
    ]
    return [column for column in candidates if column in model_df.columns]


def get_core_pythagorean_features(model_df):
    candidates = [
        "current_wins",
        "pythagorean_wins",
        "win_over_pythagorean",
        "point_diff_per_game",
        "points_for",
        "points_against",
        "projection_roster_score",
        "projection_schedule_adjusted_strength",
        "projection_sos_rank",
        "projection_home_games",
        "projection_away_games",
        "projection_neutral_games",
    ]
    return [column for column in candidates if column in model_df.columns]


def get_playoff_feature_candidates(model_df):
    # Postseason model: selected/current season profile predicts selected/current season playoff success.
    candidates = [
        "current_wins",
        "pythagorean_wins",
        "win_over_pythagorean",
        "point_diff_per_game",
        "gini_score",
        "gini_rank",
        "offense_estat",
        "defense_estat",
        "off_adj_epa",
        "def_adj_epa",
        "epa_differential",
        "success_margin",
        "schedule_strength",
        "offense_rank",
        "defense_rank",
        "balance_gap",
        "two_way_team",
        "top_gini_team",
        "top_point_diff_team",
        "super_square_profile",
        "underperformer_signal",
        "overperformer_signal",
        "turnover_margin_per_game",
        "penalty_yards_margin_per_game",
    ]
    return [column for column in candidates if column in model_df.columns]


def limit_to_last_target_seasons(df, max_seasons=20):
    if df.empty or "season" not in df.columns:
        return df
    seasons = sorted(df["season"].dropna().astype(int).unique())
    if len(seasons) <= max_seasons:
        return df
    keep = set(seasons[-max_seasons:])
    return df[df["season"].isin(keep)].copy()


@st.cache_data(show_spinner=False)
def train_models(model_df):
    if not SKLEARN_AVAILABLE:
        return {
            "available": False,
            "message": "scikit-learn is not available in this environment.",
        }

    if model_df.empty or "next_season_wins" not in model_df.columns:
        return {
            "available": False,
            "message": "Following-season wins are not available for modeling.",
        }

    all_features = get_feature_candidates(model_df)
    core_features = get_core_pythagorean_features(model_df)
    if not all_features:
        return {
            "available": False,
            "message": "No usable team traits were found for modeling.",
        }

    training_df = model_df.dropna(subset=["next_season_wins"]).copy()
    training_df = clean_numeric_columns(training_df, all_features + ["next_season_wins", "strong_next_season"])
    training_df = training_df.dropna(subset=["next_season_wins"])

    if len(training_df) < 80 or training_df["season"].nunique() < 8:
        return {
            "available": False,
            "message": "The modeling dataset is too small for a useful rolling backtest.",
            "features": all_features,
            "rows": len(training_df),
        }

    model_configs = []
    if len(core_features) >= 3:
        model_configs.extend(
            [
                (
                    "Pythagorean Ridge",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                            ("model", Ridge(alpha=5.0)),
                        ]
                    ),
                    core_features,
                ),
                (
                    "Pythagorean Elastic Net",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                            ("model", ElasticNet(alpha=0.05, l1_ratio=0.15, random_state=42, max_iter=20000)),
                        ]
                    ),
                    core_features,
                ),
            ]
        )

    model_configs.extend(
        [
            (
                "Expanded Ridge",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        ("model", Ridge(alpha=8.0)),
                    ]
                ),
                all_features,
            ),
            (
                "Expanded Elastic Net",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        ("model", ElasticNet(alpha=0.08, l1_ratio=0.25, random_state=42, max_iter=20000)),
                    ]
                ),
                all_features,
            ),
            (
                "Gradient Boosting",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        (
                            "model",
                            GradientBoostingRegressor(
                                n_estimators=180,
                                learning_rate=0.035,
                                max_depth=2,
                                random_state=42,
                            ),
                        ),
                    ]
                ),
                all_features,
            ),
            (
                "Random Forest",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        (
                            "model",
                            RandomForestRegressor(
                                n_estimators=450,
                                min_samples_leaf=4,
                                random_state=42,
                            ),
                        ),
                    ]
                ),
                all_features,
            ),
        ]
    )

    seasons = sorted(training_df["season"].dropna().astype(int).unique())
    min_train_seasons = 8
    backtest_rows = []
    model_scores = []

    for name, model, features in model_configs:
        predictions = []
        for test_season in seasons[min_train_seasons:]:
            train_df = training_df[training_df["season"] < test_season].copy()
            test_df = training_df[training_df["season"] == test_season].copy()
            if train_df.empty or test_df.empty:
                continue

            fitted = clone(model)
            fitted.fit(train_df[features], train_df["next_season_wins"])
            preds = fitted.predict(test_df[features]).clip(0, 17)

            fold = test_df[["season", "team", "current_wins", "next_season_wins"]].copy()
            fold["Model"] = name
            fold["predicted_next_wins"] = preds
            predictions.append(fold)

        if not predictions:
            continue

        pred_df = pd.concat(predictions, ignore_index=True)
        errors = pred_df["next_season_wins"] - pred_df["predicted_next_wins"]
        mae = float(errors.abs().mean())
        rmse = float((errors.pow(2).mean()) ** 0.5)
        within_two = float((errors.abs() <= 2).mean())
        corr = float(pred_df[["next_season_wins", "predicted_next_wins"]].corr().iloc[0, 1]) if len(pred_df) > 2 else None
        model_scores.append(
            {
                "Model": name,
                "Rolling MAE": mae,
                "Rolling RMSE": rmse,
                "Within 2 Wins": within_two,
                "Correlation": corr,
                "Features Used": len(features),
            }
        )
        backtest_rows.append(pred_df)

    if not model_scores:
        return {
            "available": False,
            "message": "The rolling backtest could not produce any predictions.",
            "features": all_features,
        }

    model_results_df = pd.DataFrame(model_scores).sort_values(["Rolling MAE", "Rolling RMSE"], ascending=True)
    best_model_name = str(model_results_df.iloc[0]["Model"])
    best_model, best_features = next((clone(model), features) for name, model, features in model_configs if name == best_model_name)

    production_model = best_model
    production_model.fit(training_df[best_features], training_df["next_season_wins"])

    combined_backtest = pd.concat(backtest_rows, ignore_index=True)
    best_backtest = combined_backtest[combined_backtest["Model"] == best_model_name].copy()

    baseline_source = best_backtest["current_wins"].fillna(training_df["current_wins"].median())
    baseline_mae = float(mean_absolute_error(best_backtest["next_season_wins"], baseline_source))
    model_mae = float(model_results_df.iloc[0]["Rolling MAE"])
    metrics = {
        "model_mae": model_mae,
        "baseline_mae": baseline_mae,
        "mae_edge": baseline_mae - model_mae,
        "rmse": float(model_results_df.iloc[0]["Rolling RMSE"]),
        "within_two_wins": float(model_results_df.iloc[0]["Within 2 Wins"]),
        "correlation": model_results_df.iloc[0]["Correlation"],
        "best_model_name": best_model_name,
    }

    best_estimator = production_model.named_steps["model"]
    if hasattr(best_estimator, "feature_importances_"):
        importance_values = best_estimator.feature_importances_
    elif hasattr(best_estimator, "coef_"):
        importance_values = abs(best_estimator.coef_)
    else:
        importance_values = [0] * len(best_features)

    importances = pd.DataFrame(
        {
            "Feature": best_features,
            "Team Trait": [display_label(feature) for feature in best_features],
            "Importance": importance_values,
        }
    ).sort_values("Importance", ascending=False)

    classifier = None
    class_metrics = {}
    class_test_df = best_backtest.copy()
    if "strong_next_season" in training_df.columns:
        class_df = training_df.dropna(subset=["strong_next_season"]).copy()
        if class_df["strong_next_season"].nunique() == 2:
            classifier_spec = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=350,
                            min_samples_leaf=4,
                            random_state=42,
                        ),
                    ),
                ]
            )
            classifier = clone(classifier_spec)
            classifier.fit(class_df[best_features], class_df["strong_next_season"].astype(int))
            try:
                class_test_df = best_backtest.copy()
                class_test_df["strong_next_season"] = (class_test_df["next_season_wins"] >= 10).astype(int)
                class_test_df["prob_10_plus_wins"] = classifier.predict_proba(class_test_df[best_features])[:, 1] if set(best_features).issubset(class_test_df.columns) else pd.NA
            except Exception:
                pass

    return {
        "available": True,
        "reg_model": production_model,
        "classifier": classifier,
        "features": best_features,
        "metrics": metrics,
        "class_metrics": class_metrics,
        "importances": importances,
        "model_results": model_results_df,
        "train_seasons": seasons[:-1],
        "test_seasons": seasons[8:],
        "test_df": best_backtest,
        "class_test_df": class_test_df,
        "training_rows": len(training_df),
        "train_rows": len(training_df),
        "test_rows": len(best_backtest),
    }


@st.cache_data(show_spinner=False)
def train_playoff_models(model_df):
    if not SKLEARN_AVAILABLE:
        return {
            "available": False,
            "message": "scikit-learn is not available in this environment.",
        }

    if model_df.empty or "postseason_wins" not in model_df.columns:
        return {
            "available": False,
            "message": "Postseason wins are not available for playoff modeling.",
        }

    features = get_playoff_feature_candidates(model_df)
    if not features:
        return {
            "available": False,
            "message": "No usable team traits were found for playoff modeling.",
        }

    training_df = model_df.dropna(subset=["postseason_wins"]).copy()
    training_df = clean_numeric_columns(training_df, features + ["postseason_wins", "postseason_games"])
    training_df = training_df.dropna(subset=["postseason_wins"])
    training_df["won_playoff_game"] = (training_df["postseason_wins"] >= 1).astype(int)

    if len(training_df) < 80 or training_df["season"].nunique() < 8:
        return {
            "available": False,
            "message": "The playoff dataset is too small for a useful rolling backtest.",
            "features": features,
            "rows": len(training_df),
        }

    model_configs = [
        (
            "Playoff Ridge",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", Ridge(alpha=4.0)),
                ]
            ),
        ),
        (
            "Playoff Elastic Net",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", ElasticNet(alpha=0.05, l1_ratio=0.20, random_state=42, max_iter=20000)),
                ]
            ),
        ),
        (
            "Playoff Gradient Boosting",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        GradientBoostingRegressor(
                            n_estimators=150,
                            learning_rate=0.035,
                            max_depth=2,
                            random_state=42,
                        ),
                    ),
                ]
            ),
        ),
        (
            "Playoff Random Forest",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        RandomForestRegressor(
                            n_estimators=450,
                            min_samples_leaf=4,
                            random_state=42,
                        ),
                    ),
                ]
            ),
        ),
    ]

    seasons = sorted(training_df["season"].dropna().astype(int).unique())
    min_train_seasons = 8
    model_scores = []
    backtest_rows = []

    for name, model in model_configs:
        predictions = []
        for test_season in seasons[min_train_seasons:]:
            train_df = training_df[training_df["season"] < test_season].copy()
            test_df = training_df[training_df["season"] == test_season].copy()
            if train_df.empty or test_df.empty:
                continue

            fitted = clone(model)
            fitted.fit(train_df[features], train_df["postseason_wins"])
            preds = fitted.predict(test_df[features]).clip(0, 4)

            fold = test_df[["season", "team", "postseason_wins", "postseason_games"]].copy()
            fold["Model"] = name
            fold["predicted_postseason_wins"] = preds
            predictions.append(fold)

        if not predictions:
            continue

        pred_df = pd.concat(predictions, ignore_index=True)
        errors = pred_df["postseason_wins"] - pred_df["predicted_postseason_wins"]
        mae = float(errors.abs().mean())
        rmse = float((errors.pow(2).mean()) ** 0.5)
        playoff_team_df = pred_df[pd.to_numeric(pred_df["postseason_games"], errors="coerce").fillna(0) > 0]
        playoff_team_mae = float((playoff_team_df["postseason_wins"] - playoff_team_df["predicted_postseason_wins"]).abs().mean()) if not playoff_team_df.empty else mae
        model_scores.append(
            {
                "Model": name,
                "Rolling MAE": mae,
                "Rolling RMSE": rmse,
                "Playoff Team MAE": playoff_team_mae,
            }
        )
        backtest_rows.append(pred_df)

    if not model_scores:
        return {
            "available": False,
            "message": "The rolling playoff backtest could not produce any predictions.",
            "features": features,
        }

    model_results_df = pd.DataFrame(model_scores).sort_values(["Rolling MAE", "Playoff Team MAE"], ascending=True)
    best_model_name = str(model_results_df.iloc[0]["Model"])
    best_model = next(clone(model) for name, model in model_configs if name == best_model_name)
    best_model.fit(training_df[features], training_df["postseason_wins"])

    combined_backtest = pd.concat(backtest_rows, ignore_index=True)
    best_backtest = combined_backtest[combined_backtest["Model"] == best_model_name].copy()
    baseline_mae = float(mean_absolute_error(best_backtest["postseason_wins"], [0] * len(best_backtest)))
    metrics = {
        "model_mae": float(model_results_df.iloc[0]["Rolling MAE"]),
        "baseline_mae": baseline_mae,
        "mae_edge": baseline_mae - float(model_results_df.iloc[0]["Rolling MAE"]),
        "rmse": float(model_results_df.iloc[0]["Rolling RMSE"]),
        "playoff_team_mae": float(model_results_df.iloc[0]["Playoff Team MAE"]),
        "best_model_name": best_model_name,
    }

    classifier = None
    class_metrics = {}
    class_results = pd.DataFrame()

    if training_df["won_playoff_game"].nunique() == 2:
        classifier_specs = {
            "Logistic Regression": Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(C=0.6, class_weight="balanced", max_iter=20000)),
                ]
            ),
            "Random Forest": Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=350,
                            min_samples_leaf=5,
                            random_state=42,
                            class_weight="balanced",
                        ),
                    ),
                ]
            ),
            "Gradient Boosting": Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        GradientBoostingClassifier(
                            n_estimators=120,
                            learning_rate=0.035,
                            max_depth=2,
                            random_state=42,
                        ),
                    ),
                ]
            ),
        }

        class_rows = []
        for name, model in classifier_specs.items():
            probs = []
            actuals = []
            for test_season in seasons[min_train_seasons:]:
                train_df = training_df[training_df["season"] < test_season].copy()
                test_df = training_df[training_df["season"] == test_season].copy()
                if train_df.empty or test_df.empty or train_df["won_playoff_game"].nunique() < 2:
                    continue
                fitted = clone(model)
                fitted.fit(train_df[features], train_df["won_playoff_game"].astype(int))
                probs.extend(fitted.predict_proba(test_df[features])[:, 1].tolist())
                actuals.extend(test_df["won_playoff_game"].astype(int).tolist())

            if not probs:
                continue
            predicted = [1 if prob >= 0.5 else 0 for prob in probs]
            try:
                auc_value = float(roc_auc_score(actuals, probs))
            except ValueError:
                auc_value = None
            class_rows.append(
                {
                    "Model": name,
                    "Accuracy": float(accuracy_score(actuals, predicted)),
                    "AUC": auc_value,
                }
            )

        if class_rows:
            class_results = pd.DataFrame(class_rows)
            class_results["_sort_auc"] = class_results["AUC"].fillna(-1)
            class_results = class_results.sort_values(["_sort_auc", "Accuracy"], ascending=False).drop(columns="_sort_auc")
            best_classifier_name = str(class_results.iloc[0]["Model"])
            classifier = clone(classifier_specs[best_classifier_name])
            classifier.fit(training_df[features], training_df["won_playoff_game"].astype(int))
            class_metrics = {
                "best_classifier_name": best_classifier_name,
                "accuracy": float(class_results.iloc[0]["Accuracy"]),
                "auc": class_results.iloc[0]["AUC"],
            }

    return {
        "available": True,
        "reg_model": best_model,
        "classifier": classifier,
        "features": features,
        "metrics": metrics,
        "class_metrics": class_metrics,
        "model_results": model_results_df,
        "class_results": class_results,
        "train_seasons": seasons[:-1],
        "test_seasons": seasons[8:],
        "test_df": best_backtest,
        "training_rows": len(training_df),
        "train_rows": len(training_df),
        "test_rows": len(best_backtest),
    }


def build_team_name_lookup(team_assets):
    if team_assets.empty or not {"team_abbr", "team_name"}.issubset(team_assets.columns):
        return {}

    lookup = {}
    for _, row in team_assets.dropna(subset=["team_abbr"]).iterrows():
        lookup[str(row["team_abbr"])] = str(row.get("team_name", row["team_abbr"]))
    return lookup


def build_team_theme_lookup(team_assets):
    if team_assets.empty or "team_abbr" not in team_assets.columns:
        return TEAM_THEME_OVERRIDES.copy()

    lookup = {}
    for _, row in team_assets.dropna(subset=["team_abbr"]).iterrows():
        abbr = str(row["team_abbr"]).strip()
        primary = row.get("team_color", PRIMARY)
        secondary = row.get("team_color2", SECONDARY)
        if not isinstance(primary, str) or not primary.startswith("#"):
            primary = PRIMARY
        if not isinstance(secondary, str) or not secondary.startswith("#"):
            secondary = SECONDARY
        lookup[abbr] = {"primary": primary, "secondary": secondary}

    lookup.update(TEAM_THEME_OVERRIDES)
    return lookup


def get_team_theme(team, team_theme_lookup):
    return team_theme_lookup.get(str(team), {"primary": PRIMARY, "secondary": SECONDARY})


def build_team_logo_lookup(team_assets):
    if team_assets.empty or "team_abbr" not in team_assets.columns:
        return {"DEN": BRONCOS_LOGO_SOURCE}

    logo_column = "team_logo_espn" if "team_logo_espn" in team_assets.columns else None
    if logo_column is None:
        return {}

    lookup = {}
    for _, row in team_assets.dropna(subset=["team_abbr"]).iterrows():
        abbr = str(row["team_abbr"]).strip()
        logo = row.get(logo_column, "")
        if isinstance(logo, str) and logo.startswith("http"):
            lookup[abbr] = logo
    lookup["DEN"] = BRONCOS_LOGO_SOURCE
    return lookup


@st.cache_data(show_spinner=False)
def image_file_to_data_uri(path):
    if not isinstance(path, str) or not path:
        return ""

    image_path = Path(path)
    if not image_path.is_absolute():
        image_path = APP_DIR / image_path
    if not image_path.exists():
        return ""

    encoded = base64.b64encode(image_path.read_bytes()).decode()
    return f"data:image/png;base64,{encoded}"


def team_display_name(team, team_name_lookup, include_abbr=True):
    team = str(team)
    full_name = team_name_lookup.get(team, team)
    if include_abbr and full_name != team:
        return f"{full_name} ({team})"
    return full_name


def render_metric_card(label, value, helper="", accent=PRIMARY):
    st.markdown(
        f"""
<div class="predict-card metric-card" style="border-left-color:{accent};">
    <div class="metric-label">{label}</div>
    <div class="metric-value">{value}</div>
    <div class="metric-helper">{helper}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_takeaway(text, title="Key Takeaway", accent=PRIMARY):
    st.markdown(
        f"""
<div class="predict-card takeaway-card" style="border-left-color:{accent};">
    <div class="takeaway-title">{title}</div>
    <div class="takeaway-text">{text}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_caption(text):
    st.markdown(
        f"""
<div class="chart-caption">
    <span>What this shows:</span> {text}
</div>
""",
        unsafe_allow_html=True,
    )


def format_display_table(df, fmt=None):
    display_df = df.copy()

    for column in display_df.columns:
        if column in {"Super Square Profile", "Overperformer", "Underperformer", "Elite Offense", "Elite Defense", "Top Gini Team", "Top Point Differential Team"}:
            display_df[column] = display_df[column].apply(yes_no)
        elif column == "Two-Way Team":
            display_df[column] = display_df[column].apply(lambda value: profile_text(value, "Two-Way Team", "Not Two-Way"))

    if fmt:
        for column, formatter in fmt.items():
            if column in display_df.columns:
                display_df[column] = display_df[column].apply(lambda value: formatter.format(value) if pd.notna(value) else "-")
    else:
        for column in display_df.columns:
            if column in INTEGER_DISPLAY_COLUMNS:
                display_df[column] = display_df[column].apply(lambda value: f"{float(value):.0f}" if pd.notna(value) and str(value) != "-" else value)
            elif column in ONE_DECIMAL_DISPLAY_COLUMNS:
                display_df[column] = display_df[column].apply(lambda value: f"{float(value):.1f}" if pd.notna(value) and str(value) != "-" else value)
            elif column in THREE_DECIMAL_DISPLAY_COLUMNS:
                display_df[column] = display_df[column].apply(lambda value: f"{float(value):.3f}" if pd.notna(value) and str(value) != "-" else value)

    return display_df


def rename_for_display(df, team_name_lookup=None):
    display_df = df.copy()
    if team_name_lookup and "team" in display_df.columns:
        display_df["team"] = display_df["team"].apply(lambda team: team_display_name(team, team_name_lookup))
    return display_df.rename(columns={column: display_label(column) for column in display_df.columns})


def show_clean_table(df, fmt=None, max_height=460):
    display_df = format_display_table(df, fmt=fmt)

    st.dataframe(
        display_df,
        hide_index=True,
        use_container_width=True,
        height=max_height,
    )


def chart_layout(fig, height=480, x_title=None, y_title=None, legend_title=None, accent=PRIMARY):
    fig.update_layout(
        height=height,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color=TEXT, size=12),
        margin=dict(l=72, r=48, t=96, b=105),
        title=dict(
            font=dict(size=15, color=TEXT),
            x=0.01,
            xanchor="left",
            y=0.97,
            yanchor="top",
        ),
        xaxis=dict(
            gridcolor=GRID,
            zeroline=False,
            automargin=True,
        ),
        yaxis=dict(
            gridcolor=GRID,
            zeroline=False,
            automargin=True,
        ),
        hoverlabel=dict(bgcolor="#FFFFFF", font_color=TEXT, bordercolor=accent),
        uniformtext_minsize=10,
        uniformtext_mode="show",
    )

    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)

    if x_title:
        fig.update_xaxes(title_text=x_title)
    if y_title:
        fig.update_yaxes(title_text=y_title)
    if legend_title:
        fig.update_layout(legend_title_text=legend_title)
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.24,
                xanchor="center",
                x=0.5,
            )
        )
    return fig


def percent_text(value):
    if pd.isna(value):
        return "-"
    return f"{100 * float(value):.1f}%"


def number_text(value, decimals=1):
    if pd.isna(value):
        return "-"
    return f"{float(value):.{decimals}f}"


def win_text(value, decimals=0):
    if value is None or pd.isna(value):
        return "Not available yet"
    return f"{float(value):.{decimals}f}"


def expected_regular_season_games(season):
    try:
        season = int(season)
    except Exception:
        return 17
    return 17 if season >= 2021 else 16


def row_has_completed_regular_season(row, season):
    if row is None:
        return False
    games_played = row.get("scored_games", pd.NA)
    if pd.isna(games_played):
        wins = row.get("current_wins", pd.NA)
        losses = row.get("current_losses", pd.NA)
        ties = row.get("current_ties", 0)
        if pd.isna(wins) or pd.isna(losses):
            return False
        games_played = float(wins) + float(losses) + (0 if pd.isna(ties) else float(ties))
    return float(games_played) >= expected_regular_season_games(season)


def get_prediction_for_row(row, model_bundle):
    if not model_bundle.get("available"):
        return None, None

    features = model_bundle["features"]
    X = pd.DataFrame([{feature: row.get(feature, pd.NA) for feature in features}])
    predicted_wins = float(model_bundle["reg_model"].predict(X)[0])

    probability = None
    classifier = model_bundle.get("classifier")
    if classifier is not None:
        probability = float(classifier.predict_proba(X)[0, 1])

    return predicted_wins, probability


def debug_number(value, decimals=3):
    if value is None or pd.isna(value):
        return None
    try:
        return round(float(value), decimals)
    except Exception:
        return None


def write_projection_debug_log(selected_year, model_df, model_bundle):
    feature_year = int(selected_year) - 1
    if model_df.empty or "season" not in model_df.columns or not model_bundle.get("available"):
        return None

    target_slice = model_df[model_df["season"].eq(int(selected_year))]
    target_games = (
        pd.to_numeric(target_slice["scored_games"], errors="coerce")
        if "scored_games" in target_slice.columns
        else pd.Series(dtype=float)
    )
    if target_games.notna().any() and target_games.max() > 0 and PREDICTIVE_MODEL_DEBUG_LOG_PATH.exists():
        try:
            existing = json.loads(PREDICTIVE_MODEL_DEBUG_LOG_PATH.read_text(encoding="utf-8"))
            if int(existing.get("selected_year", 0)) == int(selected_year):
                return "Preserved the original/preseason projection reference after the season began."
        except Exception:
            pass

    feature_slice = model_df[model_df["season"] == feature_year].copy()
    if feature_slice.empty:
        return None

    rows = []
    for _, row in feature_slice.sort_values("team").iterrows():
        predicted_wins, probability = get_prediction_for_row(row, model_bundle)
        rows.append(
            {
                "team": str(row.get("team")),
                "selected_year": int(selected_year),
                "feature_year": int(feature_year),
                "schedule_year_used": debug_number(row.get("projection_schedule_year"), 0),
                "roster_context_year": debug_number(row.get("projection_roster_context_year"), 0),
                "roster_score": debug_number(row.get("projection_roster_score")),
                "schedule_difficulty": debug_number(row.get("projection_schedule_adjusted_strength")),
                "projected_wins": debug_number(predicted_wins),
                "ten_plus_win_chance": debug_number(probability),
                "roster_status": str(row.get("projection_roster_status", "")),
                "schedule_status": str(row.get("projection_schedule_status", "")),
            }
        )

    roster_scores = [row["roster_score"] for row in rows if row["roster_score"] is not None]
    payload = {
        "selected_year": int(selected_year),
        "feature_year": int(feature_year),
        "model_features": list(model_bundle.get("features", [])),
        "uses_live_leaderboard_roster_score": "projection_roster_score" in model_bundle.get("features", []),
        "roster_score_unique_count": len(set(roster_scores)),
        "rows": rows,
    }

    try:
        PREDICTIVE_MODEL_DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        PREDICTIVE_MODEL_DEBUG_LOG_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as exc:
        return f"Could not write predictive model debug log: {exc}"

    return None


def frozen_preseason_projection(selected_year, team):
    """Read the pre-season snapshot without recalculating it from in-season results."""
    if not PREDICTIVE_MODEL_DEBUG_LOG_PATH.exists():
        return None
    try:
        payload = json.loads(PREDICTIVE_MODEL_DEBUG_LOG_PATH.read_text(encoding="utf-8"))
        if int(payload.get("selected_year", 0)) != int(selected_year):
            return None
        for row in payload.get("rows", []):
            if str(row.get("team", "")).upper() == str(team).upper():
                return row
    except Exception:
        return None
    return None


def get_playoff_prediction_for_row(row, playoff_bundle):
    if not playoff_bundle.get("available"):
        return None, None

    features = playoff_bundle["features"]
    X = pd.DataFrame([{feature: row.get(feature, pd.NA) for feature in features}])
    projected_wins = float(playoff_bundle["reg_model"].predict(X)[0])

    probability = None
    classifier = playoff_bundle.get("classifier")
    if classifier is not None:
        probability = float(classifier.predict_proba(X)[0, 1])

    return projected_wins, probability


def playoff_outlook_label(row, projected_wins, probability):
    projected = projected_wins if projected_wins is not None and pd.notna(projected_wins) else 0
    prob = probability if probability is not None and pd.notna(probability) else 0

    if projected >= 1.3 or prob >= 0.58:
        return "Deep-run contender"
    if projected >= 0.8 or prob >= 0.38:
        return "Playoff win threat"
    if projected >= 0.35 or prob >= 0.20:
        return "Outside chance"
    return "Long-shot profile"


def playoff_profile_interpretation(row):
    reasons = []
    wins = row.get("current_wins")
    gini_rank = row.get("gini_rank")
    point_diff = row.get("point_diff_per_game")
    offense_rank = row.get("offense_rank")
    defense_rank = row.get("defense_rank")
    schedule_strength = row.get("schedule_strength")

    if pd.notna(wins):
        if wins >= 12:
            reasons.append("regular-season record already looked like a top contender")
        elif wins >= 10:
            reasons.append("regular-season record put it in the usual playoff mix")
        else:
            reasons.append("regular-season record left less margin for a playoff run")

    if pd.notna(gini_rank):
        if gini_rank <= 6:
            reasons.append("Gini profile ranked with the league's best teams")
        elif gini_rank <= 12:
            reasons.append("Gini profile was strong enough to support a playoff win")
        elif gini_rank > 18:
            reasons.append("underlying team quality lagged behind typical playoff winners")

    if pd.notna(point_diff):
        if point_diff >= 6:
            reasons.append("scoring margin showed strong week-to-week control")
        elif point_diff <= 0:
            reasons.append("scoring margin did not show consistent separation")

    if pd.notna(offense_rank) and pd.notna(defense_rank):
        if offense_rank <= 12 and defense_rank <= 12:
            reasons.append("both offense and defense were top-12 units")
        elif offense_rank <= 8:
            reasons.append("offense had enough strength to drive an upset path")
        elif defense_rank <= 8:
            reasons.append("defense had enough strength to keep playoff games close")

    if pd.notna(schedule_strength) and schedule_strength >= 100:
        reasons.append("schedule context suggests the profile was tested")

    if not reasons:
        reasons.append("profile landed near the historical middle of the playoff inputs")

    return reasons[:4]


def risk_label(row, predicted_wins, probability):
    reasons = []
    selected_wins = row.get("current_wins")
    gini_rank = row.get("gini_rank")
    pd_rank = row.get("point_diff_rank")
    balance_gap = row.get("balance_gap")
    point_diff = row.get("point_diff_per_game")

    if row.get("super_square_profile", 0) == 1:
        reasons.append("it matched the Super Square profile")
    if row.get("two_way_team", 0) == 1:
        reasons.append("it had a balanced offense-defense profile")
    if pd.notna(point_diff) and point_diff > 5:
        reasons.append("it controlled scoring margin")
    if row.get("underperformer_signal", 0) == 1:
        reasons.append("its Gini profile was stronger than its selected-season win total")
    if row.get("overperformer_signal", 0) == 1:
        reasons.append("its selected-season win total looked stronger than its underlying profile")

    high_profile = pd.notna(gini_rank) and gini_rank <= 8 and pd.notna(pd_rank) and pd_rank <= 8
    balanced = row.get("two_way_team", 0) == 1 or (pd.notna(balance_gap) and balance_gap <= 8)
    weaker_underlying = (
        pd.notna(gini_rank)
        and gini_rank > 14
        and (pd.isna(point_diff) or point_diff < 2)
    )
    one_sided = (
        pd.notna(balance_gap)
        and balance_gap >= 14
        and (
            row.get("elite_offense", 0) == 1
            or row.get("elite_defense", 0) == 1
        )
    )

    if high_profile and balanced:
        label = "Strong carry-forward profile"
    elif pd.notna(selected_wins) and selected_wins >= 10 and weaker_underlying:
        label = "Regression risk"
    elif row.get("underperformer_signal", 0) == 1 and high_profile:
        label = "Improvement candidate"
    elif one_sided:
        label = "Volatile profile"
    else:
        label = "Stable profile"

    if not reasons:
        if predicted_wins is not None and pd.notna(selected_wins):
            if predicted_wins > selected_wins + 1:
                reasons.append("similar historical teams tended to improve the following season")
            elif predicted_wins < selected_wins - 1:
                reasons.append("similar historical teams tended to give back some wins the following season")
            else:
                reasons.append("similar historical teams tended to land close to the same win range")
        else:
            reasons.append("its profile landed near the historical middle of the model inputs")

    explanation = (
        f"This team projects as a {label.lower()} because "
        + ", ".join(reasons[:3])
        + ". This projection is based on how similar team profiles performed in past seasons. "
        + "It is not a guarantee, and it does not fully know future injuries, quarterback changes, coaching changes, or roster movement."
    )

    return label, explanation


def profile_interpretation(row, predicted_wins, season_df):
    selected_wins = row.get("current_wins")
    gini_score = row.get("gini_score")
    point_diff = row.get("point_diff_per_game")
    offense_rank = row.get("offense_rank")
    defense_rank = row.get("defense_rank")
    balance_gap = row.get("balance_gap")

    league_avg_gini = season_df["gini_score"].mean() if "gini_score" in season_df.columns else pd.NA
    reasons = []

    if pd.notna(gini_score) and pd.notna(league_avg_gini):
        gap = gini_score - league_avg_gini
        if gap >= 5:
            reasons.append(f"its Gini Score was {gap:.1f} points above the selected-season league average")
        elif gap <= -5:
            reasons.append(f"its Gini Score was {abs(gap):.1f} points below the selected-season league average")
        else:
            reasons.append("its Gini Score was near the selected-season league average")

    if pd.notna(point_diff):
        if point_diff >= 5:
            reasons.append(f"it had a strong scoring margin at {point_diff:.1f} points per game")
        elif point_diff <= -3:
            reasons.append(f"its point differential was negative at {point_diff:.1f} points per game")
        else:
            reasons.append("its scoring margin was close to even")

    if pd.notna(offense_rank) and pd.notna(defense_rank):
        if offense_rank <= 12 and defense_rank <= 12:
            reasons.append("both offense and defense ranked in the top 12")
        elif offense_rank <= 10 and defense_rank > 20:
            reasons.append("the offense was strong, but the defense lagged behind")
        elif defense_rank <= 10 and offense_rank > 20:
            reasons.append("the defense was strong, but the offense lagged behind")
        elif offense_rank > 20 and defense_rank > 20:
            reasons.append("both offense and defense ranked outside the top 20")
        elif pd.notna(balance_gap) and balance_gap <= 6:
            reasons.append("the offense and defense profile was fairly balanced")

    if row.get("underperformer_signal", 0) == 1:
        reasons.append("the team looked better by profile than by record")
    elif row.get("overperformer_signal", 0) == 1:
        reasons.append("the team looked stronger by record than by profile")
    elif pd.notna(selected_wins):
        if selected_wins >= 11:
            reasons.append("the selected-season record was already strong")
        elif selected_wins <= 5:
            reasons.append("the selected-season record was weak")

    if not reasons:
        reasons.append("its profile was close to the historical middle of the dataset")

    if predicted_wins is not None and pd.notna(selected_wins):
        if predicted_wins >= selected_wins + 2:
            reasons.append("similar historical profiles tended to improve the following season")
        elif predicted_wins <= selected_wins - 2:
            reasons.append("similar historical profiles tended to give back wins the following season")
        else:
            reasons.append("similar historical profiles usually stayed near the same win range")

    return reasons[:4]


# -----------------------------
# Page styling
# -----------------------------


st.markdown(
    f"""
<style>
.stApp,
[data-testid="stAppViewContainer"] {{
    background: {PAGE_BG} !important;
    color: {TEXT};
    overflow-x: hidden;
}}

[data-testid="stSidebar"],
[data-testid="collapsedControl"] {{
    display: none !important;
}}

.block-container {{
    position: relative;
    z-index: 2;
    padding-top: 0rem !important;
    padding-bottom: 2.6rem !important;
}}

.predict-bg {{
    position: fixed;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    pointer-events: none;
    z-index: 0;
}}

.predict-bg::before {{
    content: "";
    position: absolute;
    top: -18%;
    left: -10%;
    width: 145%;
    height: 145%;
    background: radial-gradient(ellipse at center, rgba(241, 90, 36, 0.22) 0%, rgba(241, 90, 36, 0.07) 30%, transparent 62%);
    animation: predictGlowPrimary 24s ease-in-out infinite;
}}

.predict-bg::after {{
    content: "";
    position: absolute;
    bottom: -18%;
    right: -10%;
    width: 145%;
    height: 145%;
    background: radial-gradient(ellipse at center, rgba(0, 115, 183, 0.20) 0%, rgba(0, 115, 183, 0.07) 30%, transparent 62%);
    animation: predictGlowSecondary 26s ease-in-out infinite;
}}

@keyframes predictGlowPrimary {{
    0% {{ opacity: 0.45; transform: translate(-8%, 0%) scale(1); }}
    50% {{ opacity: 0.8; transform: translate(78%, 28%) scale(1.24); }}
    100% {{ opacity: 0.45; transform: translate(-8%, 0%) scale(1); }}
}}

@keyframes predictGlowSecondary {{
    0% {{ opacity: 0.75; transform: translate(8%, 0%) scale(1); }}
    50% {{ opacity: 0.62; transform: translate(-58%, -18%) scale(1.22); }}
    100% {{ opacity: 0.75; transform: translate(8%, 0%) scale(1); }}
}}

.predict-page {{
    position: relative;
    z-index: 2;
    max-width: 1500px;
    margin: -1.55rem auto 0 auto;
    padding: 0 1.1rem 2.5rem 1.1rem;
}}

.predict-hero {{
    position: relative;
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(320px, 0.55fr);
    gap: 1.4rem;
    align-items: center;
    margin: 0.15rem 0 1.15rem 0;
    padding: 1.8rem 1.9rem;
    border-radius: 20px;
    overflow: hidden;
    background:
        radial-gradient(circle at 92% 12%, {PRIMARY}44 0%, transparent 28%),
        radial-gradient(circle at 78% 85%, {SECONDARY}38 0%, transparent 30%),
        linear-gradient(135deg, #07111f 0%, #172033 58%, #273447 100%);
    border: 1px solid rgba(255,255,255,0.14);
    box-shadow: 0 22px 46px rgba(15, 23, 42, 0.18);
}}

.predict-hero::before {{
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 5px;
    background: linear-gradient(90deg, {PRIMARY}, {SECONDARY});
}}

.predict-kicker {{
    color: rgba(255,255,255,0.72);
    font-size: 0.8rem;
    font-weight: 900;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.55rem;
}}

.predict-title {{
    color: #FFFFFF;
    font-size: clamp(2.3rem, 4vw, 3.45rem);
    line-height: 1;
    font-weight: 950;
    letter-spacing: 0;
    margin-bottom: 0.75rem;
}}

.predict-subtitle {{
    max-width: 860px;
    color: rgba(255,255,255,0.86);
    font-size: 1rem;
    line-height: 1.68;
}}

.research-card {{
    justify-self: end;
    width: min(100%, 520px);
    min-height: 188px;
    padding: 1.15rem 1.2rem 1.22rem 1.2rem;
    border-radius: 18px;
    background: rgba(255,255,255,0.13);
    border: 1px solid rgba(255,255,255,0.24);
    border-left: 5px solid {PRIMARY};
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.16), 0 14px 30px rgba(0,0,0,0.12);
    backdrop-filter: blur(10px);
}}

.research-label {{
    color: rgba(255,255,255,0.68);
    font-size: 0.72rem;
    font-weight: 950;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.45rem;
}}

.research-stat {{
    color: #FFFFFF;
    font-size: 2rem;
    line-height: 1.1;
    font-weight: 950;
    margin-bottom: 0.45rem;
}}

.research-text {{
    color: rgba(255,255,255,0.83);
    font-size: 0.95rem;
    line-height: 1.55;
}}

.research-detail-text {{
    margin-top: 0.85rem;
    color: rgba(255,255,255,0.88);
    font-size: 0.88rem;
    line-height: 1.58;
    font-weight: 760;
}}

.predict-card {{
    background: rgba(255,255,255,0.92);
    border: 1px solid rgba(15, 23, 42, 0.10);
    border-left: 5px solid {PRIMARY};
    border-radius: 16px;
    box-shadow: 0 12px 28px rgba(15, 23, 42, 0.065);
    backdrop-filter: blur(9px);
    overflow: hidden;
}}

.metric-card {{
    min-height: 118px;
    padding: 0.95rem 1.05rem;
}}

.metric-label {{
    color: {MUTED};
    font-size: 0.74rem;
    line-height: 1.2;
    font-weight: 950;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.45rem;
}}

.metric-value {{
    color: {TEXT};
    font-size: 1.72rem;
    line-height: 1.15;
    font-weight: 950;
    margin-bottom: 0.45rem;
}}

.metric-helper {{
    color: {MUTED};
    font-size: 0.84rem;
    line-height: 1.4;
    font-weight: 750;
}}

.explain-box {{
    padding: 1rem 1.1rem;
    margin: 0.85rem 0 1rem 0;
    color: #334155;
    font-size: 0.94rem;
    line-height: 1.65;
}}

.explain-title {{
    color: {TEXT};
    font-size: 1rem;
    font-weight: 950;
    margin-bottom: 0.4rem;
}}

.takeaway-card {{
    padding: 0.95rem 1.05rem;
    margin: 0.65rem 0 1rem 0;
}}

.takeaway-title {{
    color: {TEXT};
    font-size: 0.82rem;
    line-height: 1.2;
    font-weight: 950;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.35rem;
}}

.takeaway-text {{
    color: #334155;
    font-size: 0.94rem;
    line-height: 1.58;
    font-weight: 700;
}}

.chart-caption {{
    margin: 0.4rem 0 1.05rem 0;
    padding: 0.75rem 0.9rem;
    border-radius: 12px;
    background: rgba(255,255,255,0.78);
    border: 1px solid rgba(15, 23, 42, 0.09);
    color: #475569;
    font-size: 0.88rem;
    line-height: 1.55;
    box-shadow: 0 8px 18px rgba(15, 23, 42, 0.035);
}}

.chart-caption span {{
    color: {TEXT};
    font-weight: 950;
}}

.feature-group-grid {{
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.9rem;
    margin: 0.75rem 0 1rem 0;
}}

.feature-group-card {{
    padding: 1rem 1.05rem;
}}

.feature-group-title {{
    color: {TEXT};
    font-size: 0.96rem;
    line-height: 1.2;
    font-weight: 950;
    margin-bottom: 0.65rem;
}}

.feature-list {{
    margin: 0;
    padding-left: 1.05rem;
    color: #334155;
    font-size: 0.9rem;
    line-height: 1.65;
}}

.team-accent-badge {{
    display: inline-flex;
    align-items: center;
    width: fit-content;
    padding: 0.3rem 0.65rem;
    border-radius: 999px;
    color: #ffffff;
    font-size: 0.78rem;
    line-height: 1;
    font-weight: 950;
    margin-bottom: 0.7rem;
}}

.section-title {{
    color: {TEXT};
    font-size: 1.35rem;
    line-height: 1.2;
    font-weight: 950;
    letter-spacing: -0.02em;
    margin: 0.45rem 0 0.65rem 0;
}}

.section-title::after {{
    content: "";
    display: block;
    width: 58px;
    height: 4px;
    border-radius: 999px;
    margin-top: 0.45rem;
    background: linear-gradient(90deg, {PRIMARY}, {SECONDARY});
}}

.prediction-summary {{
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(280px, 0.45fr);
    gap: 1rem;
    margin: 0.75rem 0 1.25rem 0;
}}

.profile-panel {{
    padding: 1.05rem 1.15rem;
}}

.profile-heading {{
    color: {TEXT};
    font-size: 1.2rem;
    font-weight: 950;
    margin-bottom: 0.35rem;
}}

.profile-body {{
    color: #334155;
    font-size: 0.95rem;
    line-height: 1.65;
}}

div[data-testid="stPlotlyChart"] {{
    background: #FFFFFF !important;
    border: 1px solid rgba(15, 23, 42, 0.10);
    border-radius: 18px;
    padding: 0.75rem 0.9rem;
    box-shadow: 0 14px 32px rgba(15, 23, 42, 0.07);
}}

div[data-testid="stSelectbox"] label p {{
    color: {TEXT} !important;
    font-size: 0.94rem !important;
    font-weight: 900 !important;
}}

div[data-testid="stSelectbox"] [data-baseweb="select"] > div {{
    background-color: #FFFFFF !important;
    border: 1px solid #D1D5DB !important;
    border-radius: 10px !important;
    box-shadow: 0 6px 16px rgba(15, 23, 42, 0.045) !important;
}}

.stTabs [data-baseweb="tab-list"] {{
    gap: 0.45rem;
    background: rgba(255,255,255,0.76);
    border: 1px solid rgba(15, 23, 42, 0.10);
    border-radius: 14px;
    padding: 0.35rem;
    box-shadow: 0 8px 20px rgba(15, 23, 42, 0.045);
}}

.stTabs [data-baseweb="tab"] {{
    border-radius: 10px;
    padding: 0.72rem 0.85rem;
    font-size: 0.92rem;
    font-weight: 850;
    color: #334155;
}}

.stTabs [aria-selected="true"] {{
    background: #FFFFFF;
    color: {PRIMARY} !important;
    box-shadow: 0 6px 16px rgba(15, 23, 42, 0.075);
}}

.stTabs [data-baseweb="tab-highlight"] {{
    background-color: {PRIMARY} !important;
    height: 3px;
}}

@media (max-width: 900px) {{
    .predict-hero,
    .prediction-summary {{
        grid-template-columns: 1fr;
    }}

    .feature-group-grid {{
        grid-template-columns: 1fr;
    }}

    .research-card {{
        justify-self: start;
        width: 100%;
    }}
}}

@media (max-width: 700px) {{
    .predict-page {{
        padding-left: 0.2rem;
        padding-right: 0.2rem;
    }}

    .predict-hero {{
        padding: 1.18rem;
        border-radius: 16px;
    }}

    .hero-title-row {{
        align-items: flex-start;
        gap: 0.75rem;
    }}

    .hero-title-logo {{
        width: 76px;
        height: 76px;
        flex-basis: 76px;
        border-radius: 16px;
    }}

    .hero-title-logo img,
    .hero-title-logo .predict-team-logo,
    .hero-title-logo .broncos-logo-img {{
        width: 62px !important;
        height: 62px !important;
        max-width: 62px !important;
        max-height: 62px !important;
    }}

    .predict-title {{
        font-size: clamp(1.65rem, 8.5vw, 2.15rem);
    }}

    .metric-value {{
        font-size: 1.45rem;
    }}

    .stTabs [data-baseweb="tab"] {{
        padding: 0.58rem 0.65rem;
        font-size: 0.84rem;
    }}
}}
</style>
<div class="predict-bg" aria-hidden="true"></div>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# Load and prepare data
# -----------------------------


file_signature = data_file_signature()
data, load_messages = load_data(file_signature)
team_season = data["team_season"]

recorded_refresh_status = read_live_source_refresh_status()

team_game = data["team_game"]
games = data["games"]
team_assets = data["team_assets"]
schedule_sources = data.get("schedule_sources", [])
historical_roster = data.get("roster", pd.DataFrame())
season_rosters = data.get("season_rosters", {})
future_data = data.get("future_data", {})

model_df, skipped_columns, build_notes = build_model_dataset(
    team_season,
    team_game,
    games,
    schedule_sources=schedule_sources,
    team_assets=team_assets,
    historical_roster=historical_roster,
    season_rosters=season_rosters,
    future_data=future_data,
)
model_bundle = train_models(model_df)
playoff_model_bundle = train_playoff_models(model_df)

team_name_lookup = build_team_name_lookup(team_assets)
team_theme_lookup = build_team_theme_lookup(team_assets)
team_logo_lookup = build_team_logo_lookup(team_assets)

seasons_available = sorted(model_df["season"].dropna().astype(int).unique()) if not model_df.empty and "season" in model_df.columns else []
profile_seasons = seasons_available[-20:] if len(seasons_available) > 20 else seasons_available
analysis_df = (
    limit_to_last_target_seasons(model_df.dropna(subset=["next_season_wins"]).copy(), 20)
    if "next_season_wins" in model_df.columns
    else pd.DataFrame()
)
analysis_seasons = sorted(analysis_df["season"].dropna().astype(int).unique()) if not analysis_df.empty else []
completed_target_rows = int(len(analysis_df))
postseason_rows_available = bool(
    "postseason_games" in model_df.columns
    and pd.to_numeric(model_df["postseason_games"], errors="coerce").fillna(0).sum() > 0
)


if load_messages:
    for message in load_messages:
        st.warning(message)

if model_df.empty:
    st.error("The predictive model page could not build a team-season dataset from the available files.")
    st.stop()

latest_completed_season = seasons_available[-1]
active_season = determine_active_nfl_season()
future_prediction_year = active_season if active_season - 1 in seasons_available else latest_completed_season + 1
actual_schedule_years = available_schedule_years(schedule_sources)
scheduled_projection_years = [year for year in actual_schedule_years if year - 1 in seasons_available]
prediction_year_options = sorted(set(seasons_available + [future_prediction_year] + scheduled_projection_years))
season_options = sorted(prediction_year_options, reverse=True)

default_season = future_prediction_year
session_season_value = st.session_state.get(
    "predictor_selected_season_control",
    st.session_state.get("predictor_selected_season", default_season),
)
try:
    session_season = int(session_season_value)
except Exception:
    session_season = default_season
selected_season = session_season if session_season in prediction_year_options else default_season
st.session_state["predictor_selected_season"] = selected_season
if st.session_state.get("predictor_selected_season_control") not in season_options:
    st.session_state.pop("predictor_selected_season_control", None)

season_slice = model_df[model_df["season"] == selected_season].copy()
feature_year_for_controls = selected_season - 1
feature_slice_for_controls = model_df[model_df["season"] == feature_year_for_controls].copy()
team_source_slice = feature_slice_for_controls if not feature_slice_for_controls.empty else season_slice

teams_available = sorted(team_source_slice["team"].dropna().astype(str).unique())
default_team = "DEN" if "DEN" in teams_available else teams_available[0]
team_label_lookup = {
    team: team_display_name(team, team_name_lookup, include_abbr=False)
    for team in teams_available
}
team_abbr_lookup = {label: team for team, label in team_label_lookup.items()}
team_options = list(team_label_lookup.values())
control_team_value = st.session_state.get("predictor_selected_team_control")
if control_team_value in team_abbr_lookup:
    selected_team = team_abbr_lookup[control_team_value]
else:
    stored_team = str(st.session_state.get("predictor_selected_team", default_team))
    selected_team = stored_team if stored_team in teams_available else default_team
selected_team_label = team_label_lookup.get(selected_team, selected_team)
st.session_state["predictor_selected_team"] = selected_team
if st.session_state.get("predictor_selected_team_control") not in team_options:
    st.session_state.pop("predictor_selected_team_control", None)
projection_debug_log_message = write_projection_debug_log(selected_season, model_df, model_bundle)
if projection_debug_log_message:
    build_notes.append(projection_debug_log_message)
selected_label = team_display_name(selected_team, team_name_lookup, include_abbr=False)
selected_team_name = team_name_lookup.get(selected_team, selected_team)
selected_theme = get_team_theme(selected_team, team_theme_lookup)
page_primary = selected_theme["primary"]
page_secondary = selected_theme["secondary"]
st._config.set_option("theme.primaryColor", page_primary)
selected_logo = team_logo_lookup.get(selected_team, "")
selected_logo_source = (
    image_file_to_data_uri(selected_logo)
    if selected_logo and not selected_logo.startswith("http")
    else selected_logo
)
selected_logo_class = "predict-team-logo broncos-logo-img" if selected_team == "DEN" else "predict-team-logo"
hero_logo_html = (
    f'<img class="{selected_logo_class}" src="{selected_logo_source}" alt="{selected_team} logo">'
    if selected_logo_source
    else f'<div class="predict-team-fallback">{selected_team}</div>'
)

st.markdown(
    f"""
<style>
.predict-bg::before {{
    background: radial-gradient(
        ellipse at center,
        {page_primary}35 0%,
        {page_primary}13 30%,
        transparent 62%
    ) !important;
}}

.predict-bg::after {{
    background: radial-gradient(
        ellipse at center,
        {page_secondary}35 0%,
        {page_secondary}13 30%,
        transparent 62%
    ) !important;
}}

.predict-hero {{
    background:
        radial-gradient(circle at 92% 12%, {page_primary}44 0%, transparent 28%),
        radial-gradient(circle at 78% 85%, {page_secondary}38 0%, transparent 30%),
        linear-gradient(135deg, #07111f 0%, #172033 58%, #273447 100%) !important;
}}

.predict-hero::before {{
    background: linear-gradient(90deg, {page_primary}, {page_secondary}) !important;
}}

.research-card,
.predict-card {{
    border-left-color: {page_primary};
}}

.section-title::after {{
    background: linear-gradient(90deg, {page_primary}, {page_secondary}) !important;
}}

.stTabs [aria-selected="true"] {{
    color: {page_primary} !important;
}}

.stTabs [data-baseweb="tab-highlight"] {{
    background-color: {page_primary} !important;
}}

.predict-controls-title {{
    color: {TEXT};
    font-size: 1.02rem;
    font-weight: 950;
    margin: 0.9rem 0 0.45rem 0;
}}

.selected-team-strip {{
    display: flex;
    align-items: center;
    gap: 0.95rem;
    margin: 0.85rem 0 1.15rem 0;
    padding: 0.95rem 1rem;
    border-radius: 16px;
    background: rgba(255,255,255,0.92);
    border: 1px solid rgba(15, 23, 42, 0.10);
    border-left: 5px solid {page_primary};
    box-shadow: 0 12px 28px rgba(15, 23, 42, 0.065);
}}

.selected-team-strip img,
.predict-team-logo {{
    width: 66px;
    height: 66px;
    display: block;
    object-fit: contain;
    object-position: center center;
    filter: drop-shadow(0 8px 16px rgba(15, 23, 42, 0.16));
}}

.predict-team-logo.broncos-logo-img {{
    width: 78px;
    height: 78px;
}}

.hero-title-row {{
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 0.75rem;
}}

.hero-title-logo {{
    width: 104px;
    height: 104px;
    flex: 0 0 104px;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 0;
    border-radius: 22px;
    background:
        radial-gradient(circle at 30% 22%, rgba(255,255,255,0.24), rgba(255,255,255,0.10) 58%, rgba(255,255,255,0.07));
    border: 1px solid rgba(255,255,255,0.26);
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.18),
        0 12px 26px rgba(0,0,0,0.18);
    overflow: hidden;
    line-height: 0;
}}

.hero-title-logo img,
.hero-title-logo .predict-team-logo {{
    width: 82px !important;
    height: 82px !important;
    max-width: 82px !important;
    max-height: 82px !important;
    object-fit: contain;
    object-position: center center;
    display: block;
    margin: auto;
    filter: drop-shadow(0 8px 12px rgba(0,0,0,0.24));
}}

.hero-title-logo .broncos-logo-img {{
    width: 88px !important;
    height: 88px !important;
    max-width: 88px !important;
    max-height: 88px !important;
    transform: translateX(3px);
}}

.hero-title-text {{
    min-width: 0;
}}

.hero-title-text .predict-title {{
    margin-bottom: 0;
}}

.predict-team-fallback {{
    width: 66px;
    height: 66px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #FFFFFF;
    background: {page_primary};
    font-size: 1rem;
    font-weight: 950;
}}

.selected-team-kicker {{
    color: {MUTED};
    font-size: 0.74rem;
    font-weight: 950;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.2rem;
}}

.selected-team-main {{
    color: {TEXT};
    font-size: 1.2rem;
    line-height: 1.15;
    font-weight: 950;
}}

.selected-team-sub {{
    color: {MUTED};
    font-size: 0.86rem;
    line-height: 1.35;
    margin-top: 0.24rem;
    font-weight: 750;
}}
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# Header
# -----------------------------


render_top_nav("Predictive Model", page_primary, page_secondary)

st.markdown('<div class="predict-page">', unsafe_allow_html=True)

st.markdown(
    f"""
<div class="predict-hero">
<div>
<div class="predict-kicker">Historical Team Forecasting</div>

<div class="hero-title-row">
<div class="hero-title-logo">
{hero_logo_html}
</div>
<div class="hero-title-text">
<div class="predict-title">{selected_season} {selected_team_name}</div>
</div>
</div>

<div class="predict-subtitle">
Regular-season outlook and playoff upside for the selected team profile.
Change the team or year below and the full page theme will update with it.
</div>
</div>

<div class="research-card">
<div class="research-label">Forecast Inputs</div>
<div class="research-stat" style="font-size:1.45rem;">
{selected_season - 1} → {selected_season}
</div>
<div class="research-text">
Uses the completed {selected_season - 1} full-season profile to forecast {selected_season} regular-season wins.
Playoff projections unlock only after the selected regular season is complete.
</div>
<div class="research-detail-text">
The forecast reads the Gini profile, expected-wins profile, scoring efficiency, and season trend together instead of treating any one input as the whole answer.
</div>
</div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="predict-controls-title">Predictor Controls</div>', unsafe_allow_html=True)
control_team_col, control_season_col = st.columns([2.35, 0.9], gap="large")

control_team_col.selectbox(
    "Team",
    team_options,
    index=team_options.index(selected_team_label) if selected_team_label in team_options else 0,
    key="predictor_selected_team_control",
)

control_season_col.selectbox(
    "Season",
    season_options,
    index=season_options.index(selected_season),
    key="predictor_selected_season_control",
)


# -----------------------------
# Predictor tools
# -----------------------------


st.markdown(
    f"""
<div style="
    display:flex;
    flex-direction:column;
    align-items:center;
    justify-content:center;
    width:100%;
    margin:0.45rem 0 0.85rem 0;
">
    <div style="
        text-align:center;
        color:{TEXT};
        font-size:1.55rem;
        line-height:1.2;
        font-weight:950;
        letter-spacing:-0.02em;
        margin:0;
    ">
        {selected_season} Predictor Tool
    </div>
    <div style="
        width:64px;
        height:4px;
        border-radius:999px;
        margin-top:0.32rem;
        background:linear-gradient(90deg, {page_primary}, {page_secondary});
    "></div>
</div>
""",
    unsafe_allow_html=True,
)

tab_regular, tab_playoffs = st.tabs(
    [
        "Regular Season Predictor Tool",
        "Playoffs Predictor Tool",
    ]
)


with tab_regular:
    prediction_target_year = selected_season
    feature_year = selected_season - 1

    feature_slice = model_df[model_df["season"] == feature_year].copy()
    target_slice = model_df[model_df["season"] == prediction_target_year].copy()

    selected_rows = feature_slice[feature_slice["team"] == selected_team]
    target_rows = target_slice[target_slice["team"] == selected_team]

    if selected_rows.empty:
        st.warning(f"No {feature_year} team-season row found for this selection. The model needs the completed prior season to project {prediction_target_year}.")
    else:
        selected_row = selected_rows.iloc[0].copy()
        target_row = target_rows.iloc[0] if not target_rows.empty else None
        projection_schedule_year_value = pd.to_numeric(selected_row.get("projection_schedule_year"), errors="coerce")
        uses_actual_schedule = pd.notna(projection_schedule_year_value) and int(projection_schedule_year_value) == int(prediction_target_year)
        if uses_actual_schedule:
            schedule_status_text = str(selected_row.get("projection_schedule_status", f"Using actual {prediction_target_year} schedule."))
            projection_input_helper = f"Uses {feature_year} full-season profile + actual {prediction_target_year} schedule"
        else:
            schedule_status_text = (
                f"No complete actual {prediction_target_year} schedule was found for projection inputs. "
                "The model is using the prior completed profile without a selected-year schedule overlay."
            )
            projection_input_helper = f"Uses {feature_year} full-season profile"

        roster_score_value = pd.to_numeric(selected_row.get("projection_roster_score"), errors="coerce")
        roster_context_year_value = pd.to_numeric(selected_row.get("projection_roster_context_year"), errors="coerce")
        uses_roster_score = pd.notna(roster_score_value)
        if uses_roster_score:
            roster_context_year_label = int(roster_context_year_value) if pd.notna(roster_context_year_value) else prediction_target_year
            roster_status_text = str(
                selected_row.get(
                    "projection_roster_status",
                    f"Using Live Leaderboard roster score for {roster_context_year_label} roster context.",
                )
            )
            projection_input_helper += " + Live Leaderboard roster score"
        else:
            roster_status_text = (
                f"No Live Leaderboard roster score was available for {prediction_target_year}. "
                "The model is using the prior completed profile without a selected-year roster overlay."
            )

        predicted_wins, probability = get_prediction_for_row(selected_row, model_bundle)
        preseason_reference = frozen_preseason_projection(prediction_target_year, selected_team)
        if preseason_reference is not None:
            frozen_wins = pd.to_numeric(preseason_reference.get("projected_wins"), errors="coerce")
            frozen_probability = pd.to_numeric(preseason_reference.get("ten_plus_win_chance"), errors="coerce")
            if pd.notna(frozen_wins):
                predicted_wins = float(frozen_wins)
                projection_input_helper = "Frozen original/preseason projection reference"
            if pd.notna(frozen_probability):
                probability = float(frozen_probability)
        projected_wins = max(0, min(17, predicted_wins)) if predicted_wins is not None else None

        model_mae = model_bundle.get("metrics", {}).get("model_mae") if model_bundle.get("available") else None
        range_text = "Not available yet."
        if projected_wins is not None and model_mae is not None:
            lower_bound = max(0, round(projected_wins - model_mae))
            upper_bound = min(17, round(projected_wins + model_mae))
            range_text = f"{lower_bound} to {upper_bound} wins"

        actual_selected_year_wins = (
            target_row.get("current_wins")
            if target_row is not None
            else pd.NA
        )
        actual_selected_year_losses = (
            target_row.get("current_losses")
            if target_row is not None
            else pd.NA
        )
        actual_selected_year_ties = (
            target_row.get("current_ties")
            if target_row is not None
            else pd.NA
        )
        selected_year_games_played = (
            target_row.get("scored_games")
            if target_row is not None
            else pd.NA
        )
        full_regular_season_games = expected_regular_season_games(prediction_target_year)
        model_result_label = "Current Projected Finish"
        model_result_helper = "Banked results + frozen preseason rate for remaining games"
        current_finish_value = None

        if pd.isna(actual_selected_year_wins):
            actual_wins_text = "Not available yet."
            model_result_text = "Awaiting season start."
            games_played_value = 0
        else:
            actual_wins_value = float(actual_selected_year_wins)
            games_played_value = float(selected_year_games_played) if pd.notna(selected_year_games_played) else 0
            actual_wins_text = win_text(actual_selected_year_wins, 0)

            if projected_wins is None or pd.isna(projected_wins):
                model_result_text = "Projection unavailable."
            elif games_played_value <= 0:
                current_finish_value = float(projected_wins)
                model_result_text = f"{current_finish_value:.1f} wins"
                model_result_helper = "No completed games; matches the frozen preseason projection"
            elif games_played_value > 0 and games_played_value < full_regular_season_games:
                ties_value = (
                    float(actual_selected_year_ties)
                    if pd.notna(actual_selected_year_ties)
                    else 0.0
                )
                current_finish_value = current_projected_finish(
                    actual_wins_value,
                    ties_value,
                    games_played_value,
                    projected_wins,
                    prediction_target_year,
                )
                model_result_text = f"{current_finish_value:.1f} wins"
                remaining_games = max(0, full_regular_season_games - int(games_played_value))
                outlook_difference = current_finish_value - float(projected_wins)
                if abs(outlook_difference) < 0.05:
                    comparison_text = "matching the original projection"
                elif outlook_difference > 0:
                    comparison_text = f"{outlook_difference:.1f} above the original projection"
                else:
                    comparison_text = f"{abs(outlook_difference):.1f} below the original projection"
                model_result_helper = (
                    f"{int(games_played_value)} final, {remaining_games} remaining; {comparison_text}"
                )
            else:
                model_result_label = "Final Result vs. Projection"
                model_result_helper = ""
                ties_value = (
                    float(actual_selected_year_ties)
                    if pd.notna(actual_selected_year_ties)
                    else 0.0
                )
                current_finish_value = actual_wins_value + 0.5 * ties_value
                difference = current_finish_value - float(projected_wins)

                if abs(difference) < 0.05:
                    model_result_text = "Actual matched the projection."
                elif difference > 0:
                    model_result_text = f"Actual finished {difference:.1f} wins above projection."
                else:
                    model_result_text = f"Actual finished {abs(difference):.1f} wins below projection."

        label, explanation = risk_label(selected_row, projected_wins, probability)
        profile_reasons = profile_interpretation(selected_row, projected_wins, feature_slice)

        expected_wins = selected_row.get("pythagorean_wins", pd.NA)
        wins_above_expected = selected_row.get("win_over_pythagorean", pd.NA)
        if pd.notna(expected_wins) and pd.notna(wins_above_expected):
            if float(wins_above_expected) >= 1.5:
                profile_reasons.insert(0, f"record regression flag: {feature_year} finished {float(wins_above_expected):.1f} wins above expected")
            elif float(wins_above_expected) <= -1.5:
                profile_reasons.insert(0, f"improvement flag: {feature_year} finished {abs(float(wins_above_expected)):.1f} wins below expected")
            else:
                profile_reasons.insert(0, f"expected-wins profile was stable at {float(expected_wins):.1f} wins")

        if projected_wins is not None and pd.notna(projected_wins):
            profile_reasons.append(f"rolling backtest model projects {float(projected_wins):.1f} wins for {prediction_target_year}")

        if uses_actual_schedule:
            schedule_rank = pd.to_numeric(selected_row.get("projection_sos_rank"), errors="coerce")
            schedule_strength = pd.to_numeric(selected_row.get("projection_schedule_adjusted_strength"), errors="coerce")
            schedule_games = pd.to_numeric(selected_row.get("projection_schedule_games"), errors="coerce")
            schedule_reason = f"actual {prediction_target_year} schedule is included"
            if pd.notna(schedule_rank) and pd.notna(schedule_strength):
                schedule_reason = (
                    f"actual {prediction_target_year} schedule ranks {int(schedule_rank)} in difficulty "
                    f"at {float(schedule_strength):.1f} adjusted opponent strength"
                )
            if pd.notna(schedule_games):
                schedule_reason += f" across {int(schedule_games)} games"
            profile_reasons.insert(1 if profile_reasons else 0, schedule_reason)

        if uses_roster_score:
            profile_reasons.insert(
                2 if len(profile_reasons) >= 2 else len(profile_reasons),
                f"Live Leaderboard roster score is {float(roster_score_value):.1f} for {prediction_target_year} roster context",
            )

        reasons_html = "".join(f"<li>{reason}</li>" for reason in profile_reasons[:5])

        selected_theme = get_team_theme(selected_team, team_theme_lookup)
        team_primary = selected_theme["primary"]
        team_secondary = selected_theme["secondary"]

        metric_cols = st.columns(4)
        with metric_cols[0]:
            record_parts = []
            if pd.notna(actual_selected_year_wins):
                record_parts.append(str(int(float(actual_selected_year_wins))))
                if pd.notna(actual_selected_year_losses):
                    record_parts.append(str(int(float(actual_selected_year_losses))))
                if pd.notna(actual_selected_year_ties) and float(actual_selected_year_ties) > 0:
                    record_parts.append(str(int(float(actual_selected_year_ties))))
            record_helper = (
                f"Actual record: {'-'.join(record_parts)}"
                if len(record_parts) >= 2
                else f"Actual {prediction_target_year} wins"
            )
            render_metric_card(
                f"{prediction_target_year} Wins",
                actual_wins_text,
                record_helper,
                accent=team_primary,
            )
        with metric_cols[1]:
            render_metric_card(
                f"Original / Preseason Projected {prediction_target_year} Wins",
                win_text(projected_wins, 1) if projected_wins is not None else "Model unavailable",
                projection_input_helper,
                accent=team_secondary,
            )
        with metric_cols[2]:
            render_metric_card(
                f"Likely {prediction_target_year} Range",
                range_text,
                "Uses rolling backtest error",
                accent=team_primary,
            )
        with metric_cols[3]:
            render_metric_card(
                model_result_label,
                model_result_text,
                model_result_helper,
                accent=team_secondary,
            )

        st.markdown(
            f"""
<div class="prediction-summary">
    <div class="predict-card profile-panel" style="border-left-color:{team_primary};">
        <div class="team-accent-badge" style="background:{team_primary};">{feature_year} Input Profile</div>
        <div class="profile-heading">{selected_label}</div>
        <div class="profile-body">
            <ul>{reasons_html}</ul>
        </div>
    </div>
    <div class="predict-card profile-panel" style="border-left-color:{team_secondary};">
        <div class="team-accent-badge" style="background:{team_secondary};">Regular Season Outlook</div>
        <div class="profile-heading">{label}</div>
        <div class="profile-body">{explanation}</div>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

        comparison_values = [
            (f"Original {prediction_target_year} Projection", projected_wins),
        ]
        if current_finish_value is not None and games_played_value < full_regular_season_games:
            comparison_values.append((f"Current {prediction_target_year} Outlook", current_finish_value))
        if pd.notna(actual_selected_year_wins):
            comparison_values.append((f"Wins Earned Through {int(games_played_value)} Game{'s' if games_played_value != 1 else ''}", actual_selected_year_wins))

        comparison_df = pd.DataFrame(comparison_values, columns=["Win View", "Wins"]).dropna()
        if not comparison_df.empty:
            fig = px.bar(
                comparison_df,
                x="Win View",
                y="Wins",
                color="Win View",
                color_discrete_sequence=[team_secondary, team_primary, "#94A3B8"],
                title=f"{selected_label}: {prediction_target_year} Regular-Season Projection",
                labels={"Win View": "Win View", "Wins": "Wins"},
            )
            max_chart_wins = pd.to_numeric(comparison_df["Wins"], errors="coerce").max()
            y_max = max(18, float(max_chart_wins) + 2.0) if pd.notna(max_chart_wins) else 18

            fig.update_traces(
                texttemplate="%{y:.1f}",
                textposition="outside",
                cliponaxis=False,
                showlegend=False,
                width=0.42,
            )
            fig.update_yaxes(range=[0, y_max])
            fig.update_layout(showlegend=False)

            st.plotly_chart(
                chart_layout(fig, height=430, y_title="Wins", accent=team_primary),
                use_container_width=True,
            )

        if model_bundle.get("available"):
            with st.expander("Model check"):
                feature_text = ", ".join(display_label(feature) for feature in model_bundle.get("features", []))
                within_two = model_bundle.get("metrics", {}).get("within_two_wins")
                within_two_line = f"<br>Within two wins: {within_two * 100:.1f}%." if within_two is not None and pd.notna(within_two) else ""
                probability_line = (
                    f"<br>{prediction_target_year} 10+ win chance: {percent_text(probability)}."
                    if probability is not None and pd.notna(probability)
                    else ""
                )
                st.markdown(
                    f"""
<div class="predict-card explain-box">
    The regular-season model uses: {feature_text}.
    <br><br>
    Schedule input: {schedule_status_text}
    <br><br>
    Roster input: {roster_status_text}
    <br><br>
    Best rolling-backtest model: {model_bundle['metrics']['best_model_name']}.
    <br>Rolling MAE: {model_bundle['metrics']['model_mae']:.2f} wins.
    Simple previous-wins baseline MAE: {model_bundle['metrics']['baseline_mae']:.2f} wins.
    {within_two_line}
    {probability_line}
</div>
""",
                    unsafe_allow_html=True,
                )
        else:
            st.info(model_bundle.get("message", "The regular-season model could not be trained from the available files."))


with tab_playoffs:
    st.markdown('<div class="section-title">Playoffs Predictor Tool</div>', unsafe_allow_html=True)

    playoff_season = selected_season
    playoff_slice = model_df[model_df["season"] == playoff_season].copy()
    playoff_selected_label = selected_label
    playoff_selected_team = selected_team

    playoff_rows = playoff_slice[playoff_slice["team"] == playoff_selected_team]
    playoff_theme = get_team_theme(playoff_selected_team, team_theme_lookup)
    team_primary = playoff_theme["primary"]
    team_secondary = playoff_theme["secondary"]

    if playoff_rows.empty:
        st.info(
            f"The {playoff_season} playoff predictor will unlock after the {playoff_season} regular season exists in the model data. "
            "Postseason predictions use the current selected season only, not the prior season."
        )
    else:
        playoff_row = playoff_rows.iloc[0]
        season_complete = row_has_completed_regular_season(playoff_row, playoff_season)
        actual_playoff_games = pd.to_numeric(playoff_row.get("postseason_games"), errors="coerce")
        actual_playoff_wins = playoff_row.get("postseason_wins")

        if not season_complete:
            st.info(
                f"The {playoff_season} postseason predictor is locked until the regular season is complete. "
                "This keeps the playoff model honest because it uses current-year regular-season stats and results."
            )
        elif pd.isna(actual_playoff_games) or float(actual_playoff_games) <= 0:
            metric_cols = st.columns(4)
            with metric_cols[0]:
                render_metric_card(
                    f"{playoff_season} Wins",
                    win_text(playoff_row.get("current_wins"), 0),
                    "Final regular-season wins",
                    accent=team_primary,
                )
            with metric_cols[1]:
                render_metric_card(
                    "Postseason Status",
                    "Did not qualify",
                    "No playoff games recorded",
                    accent=team_secondary,
                )
            with metric_cols[2]:
                render_metric_card(
                    "Projected Playoff Wins",
                    "0",
                    "No postseason path",
                    accent=team_primary,
                )
            with metric_cols[3]:
                render_metric_card(
                    "Chance of a Playoff Win",
                    "0.0%",
                    "Did not reach playoffs",
                    accent=team_secondary,
                )
            st.markdown(
                f"""
<div class="prediction-summary">
    <div class="predict-card profile-panel" style="border-left-color:{team_primary};">
        <div class="team-accent-badge" style="background:{team_primary};">{playoff_season} Playoff Profile</div>
        <div class="profile-heading">{playoff_selected_label}</div>
        <div class="profile-body">This team did not qualify for the postseason, so the playoff projection is not run as an active playoff-team forecast.</div>
    </div>
    <div class="predict-card profile-panel" style="border-left-color:{team_secondary};">
        <div class="team-accent-badge" style="background:{team_secondary};">Playoff Outlook</div>
        <div class="profile-heading">Season ended before playoffs</div>
        <div class="profile-body">The postseason model is designed to evaluate completed regular-season playoff profiles, not future playoff guesses before qualification is known.</div>
    </div>
</div>
""",
                unsafe_allow_html=True,
            )
        else:
            projected_playoff_wins, playoff_win_probability = get_playoff_prediction_for_row(playoff_row, playoff_model_bundle)
            projected_playoff_wins = max(0, min(4, projected_playoff_wins)) if projected_playoff_wins is not None else None
            playoff_mae = playoff_model_bundle.get("metrics", {}).get("playoff_team_mae") if playoff_model_bundle.get("available") else None
            if playoff_mae is None:
                playoff_mae = playoff_model_bundle.get("metrics", {}).get("model_mae") if playoff_model_bundle.get("available") else None
            playoff_range = "Not available yet"
            if projected_playoff_wins is not None and playoff_mae is not None:
                lower_bound = max(0, round(projected_playoff_wins - playoff_mae))
                upper_bound = min(4, round(projected_playoff_wins + playoff_mae))
                playoff_range = f"{lower_bound} to {upper_bound} playoff wins"

            outlook = playoff_outlook_label(playoff_row, projected_playoff_wins, playoff_win_probability)
            playoff_reasons = playoff_profile_interpretation(playoff_row)
            playoff_reasons_html = "".join(f"<li>{reason}</li>" for reason in playoff_reasons)

            metric_cols = st.columns(5)
            with metric_cols[0]:
                render_metric_card(
                    f"{playoff_season} Wins",
                    win_text(playoff_row.get("current_wins"), 0),
                    "Final regular-season wins",
                    accent=team_primary,
                )
            with metric_cols[1]:
                render_metric_card(
                    "Projected Playoff Wins",
                    win_text(projected_playoff_wins, 1) if projected_playoff_wins is not None else "Model unavailable",
                    playoff_model_bundle["metrics"]["best_model_name"] if playoff_model_bundle.get("available") else "Projection unavailable",
                    accent=team_secondary,
                )
            with metric_cols[2]:
                render_metric_card("Likely Range", playoff_range, "Uses playoff-team backtest error", accent=team_primary)
            with metric_cols[3]:
                render_metric_card(
                    "Chance of a Playoff Win",
                    percent_text(playoff_win_probability),
                    "At least one postseason win",
                    accent=team_secondary,
                )
            with metric_cols[4]:
                render_metric_card(
                    "Actual Playoff Result",
                    f"{win_text(actual_playoff_wins, 0)} wins",
                    f"{win_text(actual_playoff_games, 0)} games played",
                    accent=team_primary,
                )

            st.markdown(
                f"""
<div class="prediction-summary">
    <div class="predict-card profile-panel" style="border-left-color:{team_primary};">
        <div class="team-accent-badge" style="background:{team_primary};">{playoff_season} Playoff Profile</div>
        <div class="profile-heading">{playoff_selected_label}</div>
        <div class="profile-body">
            <ul>{playoff_reasons_html}</ul>
        </div>
    </div>
    <div class="predict-card profile-panel" style="border-left-color:{team_secondary};">
        <div class="team-accent-badge" style="background:{team_secondary};">Playoff Outlook</div>
        <div class="profile-heading">{outlook}</div>
        <div class="profile-body">
            The playoff model uses the selected season's completed regular-season profile only. It stays locked before the season is complete so the forecast does not mix prior-year assumptions with current-year playoff outcomes.
        </div>
    </div>
</div>
""",
                unsafe_allow_html=True,
            )

            comparison_values = [
                ("Projected Playoff Wins", projected_playoff_wins),
            ]
            if pd.notna(actual_playoff_wins):
                comparison_values.append(("Actual Playoff Wins", actual_playoff_wins))
            comparison_df = pd.DataFrame(comparison_values, columns=["Playoff View", "Wins"]).dropna()
            if not comparison_df.empty:
                fig = px.bar(
                    comparison_df,
                    x="Playoff View",
                    y="Wins",
                    color="Playoff View",
                    color_discrete_sequence=[team_secondary, team_primary],
                    title=f"{playoff_selected_label}: Playoff Projection",
                    labels={"Playoff View": "Playoff View", "Wins": "Playoff Wins"},
                )
            max_chart_wins = pd.to_numeric(comparison_df["Wins"], errors="coerce").max()
            y_max = max(4.5, float(max_chart_wins) + 0.8) if pd.notna(max_chart_wins) else 4.5

            fig.update_traces(
                texttemplate="%{y:.1f}",
                textposition="outside",
                cliponaxis=False,
                showlegend=False,
                width=0.42,
            )
            fig.update_yaxes(range=[0, y_max])
            fig.update_layout(showlegend=False)

            st.plotly_chart(
                chart_layout(fig, height=420, y_title="Playoff Wins", accent=team_primary),
                use_container_width=True,
            )

            if playoff_model_bundle.get("available"):
                with st.expander("Model check"):
                    feature_text = ", ".join(display_label(feature) for feature in playoff_model_bundle.get("features", []))
                    auc_text = playoff_model_bundle.get("class_metrics", {}).get("auc")
                    auc_line = f"<br>Playoff-win classifier AUC: {auc_text:.2f}." if auc_text is not None and pd.notna(auc_text) else ""
                    st.markdown(
                        f"""
<div class="predict-card explain-box">
    The playoff model uses: {feature_text}.
    <br><br>
    Best rolling-backtest playoff model: {playoff_model_bundle['metrics']['best_model_name']}.
    <br>All-team playoff-wins MAE: {playoff_model_bundle['metrics']['model_mae']:.2f}.
    <br>Playoff-team MAE: {playoff_model_bundle['metrics']['playoff_team_mae']:.2f}.
    Simple zero-win baseline MAE: {playoff_model_bundle['metrics']['baseline_mae']:.2f}.
    {auc_line}
</div>
""",
                        unsafe_allow_html=True,
                    )
            else:
                st.info(playoff_model_bundle.get("message", "The playoff model could not be trained from the available files."))


st.markdown("</div>", unsafe_allow_html=True)
