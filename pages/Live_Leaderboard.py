from pathlib import Path
from datetime import datetime, timezone, timedelta, time
from time import perf_counter
import base64
import html
import importlib
import json
import sys

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None

try:
    from sklearn.base import clone
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

APP_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = APP_DIR / "data"
LIVE_SOURCES_DIR = DATA_DIR / "live_sources"
SNAPSHOT_PATH = DATA_DIR / "live_leaderboard_snapshot.csv"
WEEKLY_HISTORY_PATH = DATA_DIR / "live_leaderboard_weekly_history.csv"
LIVE_SOURCE_REFRESH_LOG_PATH = DATA_DIR / "live_sources" / "last_live_source_refresh.txt"
LIVE_SOURCE_REFRESH_STATUS_PATH = DATA_DIR / "live_sources" / "last_live_source_refresh_status.json"
DAILY_RELOAD_TIME = time(23, 59)
WEEKLY_SNAPSHOT_CUTOFF_WEEKDAY = 1  # Tuesday, matching datetime.weekday()
WEEKLY_SNAPSHOT_CUTOFF_TIME = time(23, 59)

# nflverse live source pulls should use the latest published public season.
# Local season roster files can still provide the current/future roster context.
NFLVERSE_LIVE_SOURCE_FALLBACK_SEASON = datetime.now().year - 1

if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from site_nav import render_top_nav
from live_source_refresh import (
    record_model_recalculation,
    refresh_live_sources_if_needed as run_live_source_refresh_if_needed,
)
from season_utils import comparable_ranked_populations, select_live_performance_population
import live_roster_scoring as roster_scoring


PRIMARY = "#F15A24"
SECONDARY = "#0073B7"
NAVY = "#07111f"
PAGE_BG = "#F6F8FB"
TEXT = "#07111f"
MUTED = "#64748B"

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

CURRENT_TEAM_NAME_OVERRIDES = {
    "LA": "Los Angeles Rams",
    "LAC": "Los Angeles Chargers",
    "LV": "Las Vegas Raiders",
}

LIVE_SOURCE_CACHE_FILES = {
    "player_weekly_stats": LIVE_SOURCES_DIR / "player_weekly_stats.csv",
    "player_season_stats": LIVE_SOURCES_DIR / "player_season_stats.csv",
    "player_prior_season_stats": LIVE_SOURCES_DIR / "player_prior_season_stats.csv",
    "players": LIVE_SOURCES_DIR / "players.csv",
    "player_snap_counts": LIVE_SOURCES_DIR / "snap_counts.csv",
    "weekly_rosters": LIVE_SOURCES_DIR / "weekly_rosters.csv",
    "injuries": LIVE_SOURCES_DIR / "injuries.csv",
    "transactions": LIVE_SOURCES_DIR / "transactions.csv",
    "draft_picks": LIVE_SOURCES_DIR / "draft_picks.csv",
    "contracts": LIVE_SOURCES_DIR / "contracts.csv",
    "depth_charts": LIVE_SOURCES_DIR / "depth_charts.csv",
    "schedules": LIVE_SOURCES_DIR / "schedules.csv",
    "teams": LIVE_SOURCES_DIR / "teams.csv",
}

LEADERBOARD_CACHE_SIGNATURE_EXCLUDED_PATHS = {
    SNAPSHOT_PATH,
    WEEKLY_HISTORY_PATH,
    LIVE_SOURCE_REFRESH_LOG_PATH,
    LIVE_SOURCE_REFRESH_STATUS_PATH,
    LIVE_SOURCES_DIR / "predictive_model_projection_debug.json",
}

SUPER_BOWL_WINNERS = {
    2005: "PIT",
    2006: "IND",
    2007: "NYG",
    2008: "PIT",
    2009: "NO",
    2010: "GB",
    2011: "NYG",
    2012: "BAL",
    2013: "SEA",
    2014: "NE",
    2015: "DEN",
    2016: "NE",
    2017: "PHI",
    2018: "NE",
    2019: "KC",
    2020: "TB",
    2021: "LA",
    2022: "KC",
    2023: "KC",
    2024: "PHI",
    2025: "SEA",
}

QUADRANT_LABELS = {
    "q1_probability": "Q1 - Non-Contenders",
    "q2_probability": "Q2 - Middle Tier",
    "q3_probability": "Q3 - Playoff Contenders",
    "q4_probability": "Q4 - Super Bowl Contenders",
}

QUADRANT_REQUIREMENTS = [
    ("pd_pos_share_rank", "low", "control"),
    ("thru12_estat_rank", "low", "control"),
    ("late12_epa", "high", "control"),
    ("success_rank", "low", "control"),
    ("thru8_estat", "high", "control"),
    ("thru14_epa_rank", "low", "control"),
    ("thru10_unit", "high", "pressure"),
    ("late14_unit_rank", "low", "pressure"),
]

SNAPSHOT_COLUMNS = [
    "snapshot_timestamp",
    "season",
    "team",
    "live_market_score",
    "live_rank",
    "current_gini_score",
    "performance_score",
    "roster_score",
    "projected_wins",
    "projected_wins_low",
    "projected_wins_high",
    "most_likely_quadrant",
    "highest_quadrant_probability",
    "q1_probability",
    "q2_probability",
    "q3_probability",
    "q4_probability",
]


def eastern_timezone():
    if ZoneInfo is not None:
        try:
            return ZoneInfo("America/New_York")
        except Exception:
            pass
    return timezone(timedelta(hours=-4), name="ET")


ET = eastern_timezone()
WEEKLY_SNAPSHOT_CUTOFFS = {}


def now_et():
    return datetime.now(ET)


def coerce_et(dt):
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=ET)
    return dt.astimezone(ET)


def parse_et_timestamp(value):
    if not value:
        return None
    try:
        return coerce_et(datetime.fromisoformat(str(value).strip()))
    except Exception:
        return None


def current_hour_window_et(moment=None):
    return coerce_et(moment or now_et()).replace(minute=0, second=0, microsecond=0)


def scheduled_daily_reload_at(day):
    return datetime.combine(day, DAILY_RELOAD_TIME).replace(tzinfo=ET)


def latest_daily_reload_due_time(moment=None):
    moment = coerce_et(moment or now_et())
    today_due = scheduled_daily_reload_at(moment.date())
    if moment >= today_due:
        return today_due
    return today_due - timedelta(days=1)


def next_daily_reload_time(moment=None):
    moment = coerce_et(moment or now_et())
    today_due = scheduled_daily_reload_at(moment.date())
    if moment < today_due:
        return today_due
    return today_due + timedelta(days=1)


def daily_snapshot_key(moment=None):
    return latest_daily_reload_due_time(moment).date().isoformat()


def milliseconds_until_next_daily_reload(moment=None):
    moment = coerce_et(moment or now_et())
    delay = next_daily_reload_time(moment) - moment
    return max(1000, int(delay.total_seconds() * 1000))


def schedule_daily_live_leaderboard_reload(hour=0, minute=5):
    eastern = ZoneInfo("America/New_York") if ZoneInfo is not None else ET
    moment = datetime.now(eastern)
    next_reload = moment.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if moment >= next_reload:
        next_reload += timedelta(days=1)

    delay_ms = max(1000, int((next_reload - moment).total_seconds() * 1000))

    # This reload only helps an open local browser pick up CSVs after the 11:59 PM scheduled refresh.
    components.html(
        f"""
<script>
(function() {{
  window.setTimeout(function() {{
    try {{
      window.parent.location.reload();
    }} catch (error) {{
      window.location.reload();
    }}
  }}, {delay_ms});
}})();
</script>
""",
        height=0,
        scrolling=False,
    )


def format_et_timestamp(dt, include_seconds=True):
    date_text = dt.strftime("%B %d, %Y")
    time_fmt = "%I:%M:%S %p" if include_seconds else "%I:%M %p"
    return f"{date_text} | {dt.strftime(time_fmt).lstrip('0')} ET"


def escape(value):
    return html.escape("" if pd.isna(value) else str(value))


def number(value, decimals=1):
    numeric = pd.to_numeric(value, errors="coerce")
    return "-" if pd.isna(numeric) else f"{numeric:.{decimals}f}"


def pct(value, decimals=0):
    numeric = pd.to_numeric(value, errors="coerce")
    return "-" if pd.isna(numeric) else f"{numeric * 100:.{decimals}f}%"


def ordinal(value):
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "-"
    number_value = int(numeric)
    if 10 <= number_value % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(number_value % 10, "th")
    return f"{number_value}{suffix}"


def safe_read_csv(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(), f"Missing file: {path.name}"
    try:
        return pd.read_csv(path), ""
    except Exception as exc:
        return pd.DataFrame(), f"Could not read {path.name}: {exc}"


def first_existing_source(key, fallback_path):
    cache_path = LIVE_SOURCE_CACHE_FILES.get(key)
    if cache_path is not None and cache_path.exists():
        return cache_path, "Cached nflverse file"
    return fallback_path, "Local CSV file"


def leaderboard_base_files():
    return {
        "team_season": DATA_DIR / "team_season_estat.csv",
        "team_game": DATA_DIR / "team_game_estat.csv",
        "games": DATA_DIR / "games_2005_onward.csv",
        "historical_roster": DATA_DIR / "nfl_season_rosters_clean_2005_2025.csv",
        "team_assets": DATA_DIR / "teams_colors_logos.csv",
    }


def leaderboard_future_fallback_files():
    return {
        "player_weekly_stats": DATA_DIR / "player_weekly_stats.csv",
        "player_season_stats": DATA_DIR / "player_season_stats.csv",
        "player_prior_season_stats": DATA_DIR / "player_prior_season_stats.csv",
        "players": DATA_DIR / "players.csv",
        "player_snap_counts": DATA_DIR / "player_snap_counts.csv",
        "weekly_rosters": DATA_DIR / "weekly_rosters.csv",
        "injuries": DATA_DIR / "injuries.csv",
        "transactions": DATA_DIR / "transactions.csv",
        "depth_charts": DATA_DIR / "depth_charts.csv",
        "draft_picks": DATA_DIR / "draft_picks.csv",
        "contracts": DATA_DIR / "contracts.csv",
        "schedules": DATA_DIR / "games_2005_onward.csv",
        "teams": DATA_DIR / "teams_colors_logos.csv",
    }


def relative_source_label(path):
    path = Path(path)
    try:
        return str(path.relative_to(APP_DIR)).replace("\\", "/")
    except Exception:
        return str(path)


def source_file_signature_entry(path):
    path = Path(path)
    if not path.exists():
        return (relative_source_label(path), False, None, None)
    stat = path.stat()
    return (relative_source_label(path), True, int(stat.st_mtime_ns), int(stat.st_size))


def live_leaderboard_source_paths():
    paths = list(leaderboard_base_files().values())
    paths.extend(roster_scoring.season_roster_source_paths(DATA_DIR))
    paths.extend(LIVE_SOURCE_CACHE_FILES.values())
    paths.extend(leaderboard_future_fallback_files().values())

    unique_paths = []
    seen = set()
    excluded = {str(Path(path).resolve(strict=False)).lower() for path in LEADERBOARD_CACHE_SIGNATURE_EXCLUDED_PATHS}
    for path in paths:
        path = Path(path)
        resolved = str(path.resolve(strict=False)).lower()
        name = path.name.lower()
        if resolved in excluded or "debug" in name or "timing" in name:
            continue
        key = str(path).lower()
        if key not in seen:
            seen.add(key)
            unique_paths.append(path)
    return unique_paths


def live_leaderboard_file_signature():
    return tuple(sorted(source_file_signature_entry(path) for path in live_leaderboard_source_paths()))


@st.cache_data(show_spinner=False)
def load_data(file_signature=None):
    files = leaderboard_base_files()
    future_fallback_files = leaderboard_future_fallback_files()
    data, future_data, messages, future_sources = {}, {}, [], {}
    for key, path in files.items():
        data[key], message = safe_read_csv(path)
        if message:
            messages.append(message)
    season_rosters, season_roster_messages = roster_scoring.load_season_roster_sources(DATA_DIR)
    data["season_rosters"] = season_rosters
    messages.extend(season_roster_messages)
    for key, fallback_path in future_fallback_files.items():
        path, source_type = first_existing_source(key, fallback_path)
        future_sources[key] = {"path": str(path), "source_type": source_type, "exists": path.exists()}
        future_data[key], message = safe_read_csv(path) if path.exists() else (pd.DataFrame(), "")
        if message:
            messages.append(message)
    season_roster_paths = [Path(info["path"]) for info in season_rosters.values() if isinstance(info, dict) and info.get("path")]
    mtimes = [path.stat().st_mtime for path in list(files.values()) + season_roster_paths + [Path(info["path"]) for info in future_sources.values()] if path.exists()]
    load_time = now_et()
    source_mtime = datetime.fromtimestamp(max(mtimes), ET) if mtimes else load_time
    future_status = {key: info["exists"] for key, info in future_sources.items()}
    return data, future_data, future_status, future_sources, messages, load_time, source_mtime


@st.cache_data(show_spinner=False)
def build_live_leaderboard_payload(file_signature):
    timings = {}

    def mark_timing(label, start_time):
        timings[label] = round(perf_counter() - start_time, 4)

    step_start = perf_counter()
    data, future_data, future_status, future_sources, load_messages, load_time, source_mtime = load_data(file_signature)
    mark_timing("load_data", step_start)

    team_season = data.get("team_season", pd.DataFrame())
    team_game = data.get("team_game", pd.DataFrame())
    historical_roster = data.get("historical_roster", pd.DataFrame())
    season_rosters = data.get("season_rosters", {})
    team_assets = data.get("team_assets", pd.DataFrame())

    if team_season.empty:
        return {
            "leaderboard": pd.DataFrame(),
            "selected_season": None,
            "weekly_snapshot_key": "",
            "weekly_snapshot_year": None,
            "weekly_snapshot_period": "",
            "roster_file": "No roster file found",
            "production_found": False,
            "future_status": future_status,
            "future_sources": future_sources,
            "load_messages": load_messages,
            "load_time": load_time,
            "source_mtime": source_mtime,
            "projected_meta": {},
            "has_snapshot": False,
            "build_timings": timings,
            "team_season_empty": True,
        }

    selected_season = int(get_latest_available_season(team_season, season_rosters))
    schedule_sources = [
        future_data.get("schedules", pd.DataFrame()),
        data.get("games", pd.DataFrame()),
    ]
    schedule_frames = [df for df in schedule_sources if isinstance(df, pd.DataFrame) and not df.empty]
    schedule_for_period = pd.concat(schedule_frames, ignore_index=True, sort=False) if schedule_frames else pd.DataFrame()
    weekly_snapshot_key, weekly_snapshot_year, weekly_snapshot_period = current_nfl_period_key(
        selected_season,
        schedule_for_period,
    )

    step_start = perf_counter()
    player_stats, production_found = roster_scoring.load_player_stats(future_data)
    player_stats = roster_scoring.standardize_player_stats(player_stats)
    performance_season = get_selected_performance_season(int(selected_season), team_season)
    schedule_teams = set()
    for column in ("home_team", "away_team"):
        if column in schedule_for_period.columns:
            schedule_teams.update(schedule_for_period[column].dropna().map(normalize_team).tolist())
    fallback_performance, _ = build_performance_scores(team_season, performance_season)
    team_universe = sorted(schedule_teams or set(fallback_performance["team"].dropna().tolist()))
    performance_df, performance_all, _ = build_blended_current_performance(
        team_season,
        int(selected_season),
        team_universe,
    )
    mark_timing("performance_scores", step_start)

    step_start = perf_counter()
    local_roster, roster_file, _ = roster_scoring.load_local_roster(int(selected_season), historical_roster, season_rosters)
    transactions = roster_scoring.load_transaction_feed(future_data)
    roster_context, _ = roster_scoring.build_roster_context(local_roster, future_data, int(selected_season))
    selected_roster_scores = roster_scoring.calculate_team_roster_score(
        roster_context,
        int(selected_season),
        team_universe,
        historical_roster=historical_roster,
        transactions=transactions,
        player_stats=player_stats,
    )
    mark_timing("selected_roster_score", step_start)

    step_start = perf_counter()
    historical_roster_scores = []
    if not historical_roster.empty and "season" in historical_roster.columns:
        hist_season_values = pd.to_numeric(historical_roster["season"], errors="coerce")
        hist_seasons = sorted(hist_season_values.dropna().astype(int).unique())
        for roster_season in hist_seasons:
            season_roster = historical_roster[hist_season_values == roster_season].copy()
            teams_for_season = sorted(performance_all.loc[performance_all["season"] == roster_season, "team"].dropna().unique().tolist())
            if teams_for_season:
                season_scores = roster_scoring.calculate_team_roster_score(
                    season_roster,
                    int(roster_season),
                    teams_for_season,
                    historical_roster=historical_roster,
                    transactions=pd.DataFrame(),
                )
                season_scores["season"] = int(roster_season)
                historical_roster_scores.append(season_scores)
    roster_scores_all = pd.concat(historical_roster_scores, ignore_index=True) if historical_roster_scores else pd.DataFrame()
    mark_timing("historical_roster_scores", step_start)

    step_start = perf_counter()
    records = calculate_current_wins(team_game)
    projected_wins, projected_meta = calculate_projected_wins(
        performance_all,
        records,
        roster_scores_all,
        selected_roster_scores,
        int(selected_season) - 1 if int(selected_season) - 1 in set(performance_all["season"]) else performance_season,
        int(selected_season),
    )
    mark_timing("projected_wins", step_start)

    step_start = perf_counter()
    quadrants = calculate_quadrant_probabilities(team_game, int(selected_season), team_universe)
    mark_timing("quadrant_probabilities", step_start)

    step_start = perf_counter()
    leaderboard = merge_leaderboard_scores(performance_df, selected_roster_scores, projected_wins, quadrants, team_assets, int(selected_season))
    record_columns = ["team", "current_wins", "current_losses", "current_ties"]
    leaderboard = leaderboard.merge(
        records[records["season"] == int(selected_season)][record_columns],
        on="team",
        how="left",
    )
    mark_timing("final_merge", step_start)

    timings["total_payload_build"] = round(sum(timings.values()), 4)
    print(f"Live Leaderboard payload build timings: {timings}")

    return {
        "leaderboard": leaderboard,
        "selected_season": int(selected_season),
        "weekly_snapshot_key": weekly_snapshot_key,
        "weekly_snapshot_year": weekly_snapshot_year,
        "weekly_snapshot_period": weekly_snapshot_period,
        "roster_file": roster_file,
        "production_found": production_found,
        "future_status": future_status,
        "future_sources": future_sources,
        "load_messages": load_messages,
        "load_time": load_time,
        "source_mtime": source_mtime,
        "projected_meta": projected_meta,
        "has_snapshot": False,
        "build_timings": timings,
        "team_season_empty": False,
    }


def is_nflreadpy_available():
    try:
        importlib.import_module("nflreadpy")
        return True
    except Exception:
        return False


def _nflreadpy_module():
    return importlib.import_module("nflreadpy")


def _call_nflreadpy_loader(loader_names, seasons=None):
    nfl = _nflreadpy_module()
    last_error = None
    for name in loader_names:
        loader = getattr(nfl, name, None)
        if loader is None:
            continue
        try:
            if seasons is None:
                return loader()
            try:
                return loader(seasons=seasons)
            except TypeError:
                return loader(seasons)
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise AttributeError(f"No nflreadpy loader found for: {', '.join(loader_names)}")


def load_nflverse_rosters(seasons):
    return _call_nflreadpy_loader(["load_rosters", "import_rosters"], seasons)


def load_nflverse_weekly_rosters(seasons):
    return _call_nflreadpy_loader(["load_weekly_rosters", "load_rosters_weekly", "import_weekly_rosters"], seasons)


def load_nflverse_player_stats(seasons):
    return _call_nflreadpy_loader(["load_player_stats", "load_player_stats_seasons", "import_player_stats"], seasons)


def load_nflverse_snap_counts(seasons):
    return _call_nflreadpy_loader(["load_snap_counts", "import_snap_counts"], seasons)


def load_nflverse_injuries(seasons):
    return _call_nflreadpy_loader(["load_injuries", "import_injuries"], seasons)


def load_nflverse_schedules(seasons):
    return _call_nflreadpy_loader(["load_schedules", "import_schedules"], seasons)

def get_nflverse_source_season(selected_season):
    """
    Try to use the selected season once nflreadpy has schedule rows for it.
    Until then, use current year - 1 as the fallback season.
    """
    selected_season = int(selected_season)

    if not is_nflreadpy_available():
        return min(selected_season, NFLVERSE_LIVE_SOURCE_FALLBACK_SEASON)

    try:
        schedule_df = convert_nflreadpy_frame_to_pandas(
            load_nflverse_schedules([selected_season])
        )

        if not schedule_df.empty and "season" in schedule_df.columns:
            seasons_found = pd.to_numeric(
                schedule_df["season"],
                errors="coerce",
            ).dropna().astype(int)

            if seasons_found.eq(selected_season).any():
                return selected_season

    except Exception:
        pass

    return min(selected_season, NFLVERSE_LIVE_SOURCE_FALLBACK_SEASON)

def load_nflverse_teams():
    return _call_nflreadpy_loader(["load_teams", "import_teams"], None)

def load_nflverse_players():
    return _call_nflreadpy_loader(["load_players", "import_players"], None)

def load_nflverse_draft_picks(seasons):
    return _call_nflreadpy_loader(["load_draft_picks", "import_draft_picks"], seasons)

def load_nflverse_transactions(seasons):
    return _call_nflreadpy_loader(["load_transactions", "import_transactions"], seasons)


def load_nflverse_contracts():
    return _call_nflreadpy_loader(["load_contracts", "import_contracts"], None)


def load_nflverse_depth_charts(seasons):
    return _call_nflreadpy_loader(["load_depth_charts", "import_depth_charts"], seasons)


def convert_nflreadpy_frame_to_pandas(df):
    """
    Converts nflreadpy output into a pandas DataFrame.

    nflreadpy may return pandas, Polars, or another dataframe-like object.
    This makes the saving logic more reliable.
    """
    if df is None:
        return pd.DataFrame()

    if isinstance(df, pd.DataFrame):
        return df

    if hasattr(df, "to_pandas"):
        try:
            return df.to_pandas()
        except Exception:
            pass

    if hasattr(df, "to_dataframe"):
        try:
            return df.to_dataframe()
        except Exception:
            pass

    try:
        return pd.DataFrame(df)
    except Exception:
        return pd.DataFrame()


def update_local_data_from_nflverse(seasons):
    if not is_nflreadpy_available():
        return False, "nflreadpy is not installed yet, so the app is continuing with local files.", []

    LIVE_SOURCES_DIR.mkdir(parents=True, exist_ok=True)

    loaders = [
        ("weekly_rosters", LIVE_SOURCE_CACHE_FILES["weekly_rosters"], load_nflverse_weekly_rosters),
        ("player_season_stats", LIVE_SOURCE_CACHE_FILES["player_season_stats"], load_nflverse_player_stats),
        ("players", LIVE_SOURCE_CACHE_FILES["players"], lambda _: load_nflverse_players()),
        ("player_snap_counts", LIVE_SOURCE_CACHE_FILES["player_snap_counts"], load_nflverse_snap_counts),
        ("injuries", LIVE_SOURCE_CACHE_FILES["injuries"], load_nflverse_injuries),
        ("transactions", LIVE_SOURCE_CACHE_FILES["transactions"], load_nflverse_transactions),
        ("draft_picks", LIVE_SOURCE_CACHE_FILES["draft_picks"], load_nflverse_draft_picks),
        ("contracts", LIVE_SOURCE_CACHE_FILES["contracts"], lambda _: load_nflverse_contracts()),
        ("depth_charts", LIVE_SOURCE_CACHE_FILES["depth_charts"], load_nflverse_depth_charts),
        ("schedules", LIVE_SOURCE_CACHE_FILES["schedules"], load_nflverse_schedules),
        ("teams", LIVE_SOURCE_CACHE_FILES["teams"], lambda _: load_nflverse_teams()),
    ]

    saved_files = []
    errors = []

    for label, path, loader in loaders:
        try:
            raw_df = loader(seasons)
            df = convert_nflreadpy_frame_to_pandas(raw_df)

            if not df.empty:
                path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(path, index=False)
                saved_files.append(path.name)
            else:
                errors.append(f"{label}: loader returned an empty dataframe")

        except Exception as exc:
            errors.append(f"{label}: {exc}")

    if saved_files:
        detail = f"Updated local nflverse cache: {', '.join(saved_files)}."

        if errors:
            detail += f" Some sources were unavailable: {'; '.join(errors[:4])}."

        return True, detail, saved_files

    if errors:
        return False, (
            "nflreadpy was available, but no files were saved. "
            f"Details: {'; '.join(errors[:6])}"
        ), []

    return False, "nflreadpy was available, but no public rows were returned. Continuing with local files.", []

def read_live_source_refresh_status():
    if not LIVE_SOURCE_REFRESH_STATUS_PATH.exists():
        return {}

    try:
        status = json.loads(LIVE_SOURCE_REFRESH_STATUS_PATH.read_text())
        return status if isinstance(status, dict) else {}
    except Exception:
        return {}


def read_last_successful_live_source_refresh():
    status = read_live_source_refresh_status()

    last_success = parse_et_timestamp(status.get("last_successful_refresh_time"))
    if last_success is not None:
        return last_success

    if status:
        if bool(status.get("ok", False)):
            return parse_et_timestamp(status.get("refresh_time"))
        return None

    if not LIVE_SOURCE_REFRESH_LOG_PATH.exists():
        return None

    try:
        return parse_et_timestamp(LIVE_SOURCE_REFRESH_LOG_PATH.read_text().strip())
    except Exception:
        return None


def read_last_live_source_refresh_attempt():
    status = read_live_source_refresh_status()
    return parse_et_timestamp(status.get("refresh_time"))


def should_refresh_live_sources():
    """
    Controls live source refresh timing by ET hour window.
    This is separate from weekly leaderboard snapshot saving.
    """
    current_window = current_hour_window_et()
    last_success = read_last_successful_live_source_refresh()

    if last_success is not None and current_hour_window_et(last_success) >= current_window:
        return False

    last_attempt = read_last_live_source_refresh_attempt()
    if last_attempt is not None and current_hour_window_et(last_attempt) >= current_window:
        return False

    return True


def mark_live_sources_refreshed(ok=False, message=""):
    LIVE_SOURCE_REFRESH_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    refresh_time = now_et()
    last_success = refresh_time if ok else read_last_successful_live_source_refresh()

    if ok:
        LIVE_SOURCE_REFRESH_LOG_PATH.write_text(refresh_time.isoformat())

    status = {
        "refresh_time": refresh_time.isoformat(),
        "last_successful_refresh_time": last_success.isoformat() if last_success else "",
        "ok": bool(ok),
        "message": str(message),
    }

    LIVE_SOURCE_REFRESH_STATUS_PATH.write_text(json.dumps(status, indent=2))

def read_last_live_source_refresh_status():
    if not LIVE_SOURCE_REFRESH_STATUS_PATH.exists():
        return False, "No previous live source refresh status found."

    try:
        status = read_live_source_refresh_status()
        ok = bool(status.get("ok", False))
        message = status.get("message", "No refresh message found.")
        refresh_time = status.get("refresh_time", "")
        last_success = status.get("last_successful_refresh_time", "")

        if refresh_time:
            detail = f"{message} Last attempt: {refresh_time}"
            if last_success:
                detail += f" Last successful refresh: {last_success}"
            return ok, detail

        return ok, message

    except Exception as exc:
        return False, f"Could not read previous live source refresh status: {exc}"


def refresh_live_sources_if_needed(selected_season):
    """
    Legacy inline refresh helper retained for compatibility; the page now calls
    the shared daily refresh pipeline imported above.
    Keep this delegate so any future local call still respects the 11:59 PM ET
    daily window instead of the old page-open/hourly refresh behavior.
    """
    return run_live_source_refresh_if_needed(selected_season)

def normalize_team(team):
    value = "" if pd.isna(team) else str(team).strip().upper()
    return TEAM_ALIASES.get(value, value)


def clean_numeric_columns(df, columns):
    output = df.copy()
    for column in columns:
        if column in output.columns:
            output[column] = pd.to_numeric(output[column], errors="coerce")
    return output


def build_team_name_lookup(team_assets):
    if team_assets.empty or not {"team_abbr", "team_name"}.issubset(team_assets.columns):
        return CURRENT_TEAM_NAME_OVERRIDES.copy()
    lookup = {
        normalize_team(row["team_abbr"]): str(row["team_name"])
        for _, row in team_assets.dropna(subset=["team_abbr"]).iterrows()
    }
    lookup.update(CURRENT_TEAM_NAME_OVERRIDES)
    return lookup


def build_team_logo_lookup(team_assets):
    if team_assets.empty or not {"team_abbr", "team_logo_espn"}.issubset(team_assets.columns):
        return {}
    return {
        normalize_team(row["team_abbr"]): str(row["team_logo_espn"])
        for _, row in team_assets.dropna(subset=["team_abbr"]).iterrows()
        if pd.notna(row.get("team_logo_espn"))
    }


def build_team_color_lookup(team_assets):
    lookup = {}
    if team_assets.empty or "team_abbr" not in team_assets.columns:
        return lookup
    for _, row in team_assets.dropna(subset=["team_abbr"]).iterrows():
        team = normalize_team(row["team_abbr"])
        primary = row.get("team_color", PRIMARY)
        secondary = row.get("team_color2", SECONDARY)
        lookup[team] = {
            "primary": primary if isinstance(primary, str) and primary.startswith("#") else PRIMARY,
            "secondary": secondary if isinstance(secondary, str) and secondary.startswith("#") else SECONDARY,
        }
    lookup["DEN"] = {"primary": PRIMARY, "secondary": SECONDARY}
    return lookup


def get_latest_available_season(team_season, season_rosters):
    seasons = []
    if not team_season.empty and "season" in team_season.columns:
        seasons.extend(pd.to_numeric(team_season["season"], errors="coerce").dropna().astype(int).tolist())
    if isinstance(season_rosters, dict):
        seasons.extend(int(year) for year in season_rosters.keys() if str(year).isdigit())
        for entry in season_rosters.values():
            frame = entry.get("df", pd.DataFrame()) if isinstance(entry, dict) else entry
            if isinstance(frame, pd.DataFrame) and not frame.empty and "season" in frame.columns:
                seasons.extend(pd.to_numeric(frame["season"], errors="coerce").dropna().astype(int).tolist())
    elif isinstance(season_rosters, pd.DataFrame) and not season_rosters.empty and "season" in season_rosters.columns:
        seasons.extend(pd.to_numeric(season_rosters["season"], errors="coerce").dropna().astype(int).tolist())
    return max(seasons) if seasons else datetime.now().year


def get_selected_performance_season(selected_season, team_season):
    seasons = sorted(pd.to_numeric(team_season["season"], errors="coerce").dropna().astype(int).unique())
    if selected_season in seasons:
        return selected_season
    earlier = [season for season in seasons if season <= selected_season]
    return max(earlier) if earlier else max(seasons)


def normalize_score(series, center=85, spread=10, lower=60, upper=120):
    numeric = pd.to_numeric(series, errors="coerce")
    std = numeric.std()
    if pd.isna(std) or std == 0:
        return pd.Series(center, index=numeric.index)
    return (center + spread * ((numeric - numeric.mean()) / std)).clip(lower, upper).fillna(center)


def zscore_by_season(df, column):
    values = pd.to_numeric(df[column], errors="coerce")
    mean = values.groupby(df["season"]).transform("mean")
    std = values.groupby(df["season"]).transform("std").replace(0, np.nan)
    return ((values - mean) / std).fillna(0)


def image_file_to_data_uri(path):
    try:
        file_path = Path(path)
        if not file_path.is_absolute():
            file_path = APP_DIR / file_path
        if not file_path.exists():
            return ""
        return "data:image/png;base64," + base64.b64encode(file_path.read_bytes()).decode()
    except Exception:
        return ""


def logo_source(logo):
    if not isinstance(logo, str) or not logo:
        return ""
    return logo if logo.startswith("http") or logo.startswith("data:") else image_file_to_data_uri(logo)


def logo_html(team, logo, class_name):
    src = logo_source(logo)
    if src:
        return f'<img class="{class_name}" src="{escape(src)}" alt="{escape(team)} logo">'
    return f'<div class="{class_name} logo-fallback">{escape(team)}</div>'


def reconstruct_gini_score_if_needed(df):
    output = df.copy()
    if "custom_estat" in output.columns:
        output["current_gini_score"] = pd.to_numeric(output["custom_estat"], errors="coerce")
        return output, "custom_estat"
    if "overall_estat" in output.columns:
        output["current_gini_score"] = pd.to_numeric(output["overall_estat"], errors="coerce")
        return output, "overall_estat"
    if "estat_raw" in output.columns:
        output["current_gini_score"] = pd.to_numeric(output["estat_raw"], errors="coerce")
        return output, "estat_raw"
    weights = {
        "offense_estat": 0.22,
        "defense_estat": 0.22,
        "point_diff_per_game": 0.14,
        "success_margin": 0.12,
        "turnover_margin_per_game": 0.08,
        "penalty_yards_margin_per_game": 0.05,
        "schedule_strength": 0.07,
        "off_adj_epa": 0.05,
        "def_adj_epa": 0.05,
    }
    score = pd.Series(0.0, index=output.index)
    used = 0
    for column, weight in weights.items():
        if column not in output.columns:
            continue
        z = zscore_by_season(output, column)
        score += weight * (-z if column == "penalty_yards_margin_per_game" else z)
        used += weight
    output["current_gini_score"] = 100 + 15 * (score / used) if used else 100.0
    return output, "reconstructed_gini_style_score"


def build_performance_scores(team_season, season=None):
    output = team_season.copy()
    output["season"] = pd.to_numeric(output["season"], errors="coerce")
    output = output.dropna(subset=["season"])
    output["season"] = output["season"].astype(int)
    output["team"] = output["team"].apply(normalize_team)
    numeric_cols = [
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
        "estat_raw",
        "schedule_strength",
        "overall_rank",
        "offense_rank",
        "defense_rank",
        "sos_rank",
    ]
    output = clean_numeric_columns(output, numeric_cols)
    output, source = reconstruct_gini_score_if_needed(output)
    output["performance_score"] = output.groupby("season")["current_gini_score"].transform(
        lambda values: normalize_score(values, center=85, spread=10, lower=60, upper=120)
    )
    if {"offense_rank", "defense_rank"}.issubset(output.columns):
        output["balance_gap"] = (output["offense_rank"] - output["defense_rank"]).abs()
    else:
        output["balance_gap"] = 16
    if season is not None:
        output = output[output["season"] == int(season)].copy()
    return output, source


def build_blended_current_performance(team_season, selected_season, team_universe):
    """Use one comparable performance season for the live ranking population."""
    performance_all, source = build_performance_scores(team_season)
    current = select_live_performance_population(performance_all, selected_season, team_universe)
    return current, performance_all, source


@st.cache_data(show_spinner=False)
def calculate_current_wins(team_game):
    cols = ["season", "team", "current_wins", "current_losses", "current_ties", "scored_games"]
    if team_game.empty or not {"season", "team", "points_for", "points_against"}.issubset(team_game.columns):
        return pd.DataFrame(columns=cols)
    games = team_game.copy()
    games["season"] = pd.to_numeric(games["season"], errors="coerce")
    games = games.dropna(subset=["season"])
    games["season"] = games["season"].astype(int)
    games["team"] = games["team"].apply(normalize_team)
    if "season_type" in games.columns:
        games = games[games["season_type"].astype(str).str.upper().eq("REG")]
    games = clean_numeric_columns(games, ["points_for", "points_against"]).dropna(subset=["points_for", "points_against"])
    games["win"] = (games["points_for"] > games["points_against"]).astype(int)
    games["loss"] = (games["points_for"] < games["points_against"]).astype(int)
    games["tie"] = (games["points_for"] == games["points_against"]).astype(int)
    return games.groupby(["season", "team"], as_index=False).agg(
        current_wins=("win", "sum"),
        current_losses=("loss", "sum"),
        current_ties=("tie", "sum"),
        scored_games=("game_id", "nunique") if "game_id" in games.columns else ("win", "size"),
    )


def load_local_roster(selected_season, historical_roster, season_rosters):
    return roster_scoring.load_local_roster(selected_season, historical_roster, season_rosters)


def load_live_roster_feed():
    return pd.DataFrame()


def load_transaction_feed(future_data=None):
    return future_data.get("transactions", pd.DataFrame()).copy() if future_data else pd.DataFrame()


def load_player_stats(future_data):
    stats = {
        "weekly": future_data.get("player_weekly_stats", pd.DataFrame()).copy(),
        "seasonal": future_data.get("player_season_stats", pd.DataFrame()).copy(),
        "snaps": future_data.get("player_snap_counts", pd.DataFrame()).copy(),
        "injuries": future_data.get("injuries", pd.DataFrame()).copy(),
    }
    production_found = not stats["weekly"].empty or not stats["seasonal"].empty
    return stats, production_found


def standardize_player_stats(player_stats):
    standardized = {}
    for key, df in player_stats.items():
        if df.empty:
            standardized[key] = df
            continue
        output = df.copy()
        output.columns = [str(column).strip().lower() for column in output.columns]
        standardized[key] = output
    return standardized


def first_existing_column(df, candidates):
    for column in candidates:
        if column in df.columns:
            return column
    return None


def build_player_match_key(df):
    id_col = first_existing_column(df, ["gsis_id", "player_id", "player_gsis_id", "nfl_id"])
    name_col = first_existing_column(df, ["full_name", "player_name", "player_display_name", "name"])
    if id_col is not None:
        key = df[id_col].fillna("").astype(str).str.lower().str.strip()
    elif name_col is not None:
        key = df[name_col].fillna("").astype(str).str.lower().str.strip()
    else:
        key = pd.Series("", index=df.index)
    return key


def derive_player_production_scores(stats_df):
    if stats_df.empty:
        return pd.DataFrame(columns=["player_key", "production_score", "games_played"])

    stats = stats_df.copy()
    stats["player_key"] = build_player_match_key(stats)
    stats = stats[stats["player_key"].astype(str).str.len() > 0].copy()
    if stats.empty:
        return pd.DataFrame(columns=["player_key", "production_score", "games_played"])

    weighted_columns = {
        "passing_yards": 0.006,
        "pass_yards": 0.006,
        "passing_tds": 1.6,
        "pass_tds": 1.6,
        "interceptions": -2.4,
        "sacks": 2.0,
        "qb_hits": 1.3,
        "tackles_for_loss": 1.5,
        "rushing_yards": 0.010,
        "rush_yards": 0.010,
        "receiving_yards": 0.010,
        "rec_yards": 0.010,
        "receptions": 0.12,
        "targets": 0.04,
        "receiving_tds": 1.2,
        "rushing_tds": 1.2,
        "touchdowns": 1.2,
        "fantasy_points": 0.18,
        "fantasy_points_ppr": 0.15,
        "tackles": 0.16,
        "def_tackles": 0.16,
        "passes_defended": 1.2,
        "pass_defended": 1.2,
        "epa": 10.0,
        "passing_epa": 10.0,
        "receiving_epa": 10.0,
        "rushing_epa": 10.0,
    }
    raw = pd.Series(0.0, index=stats.index)
    used = False
    for column, weight in weighted_columns.items():
        if column in stats.columns:
            raw += pd.to_numeric(stats[column], errors="coerce").fillna(0) * weight
            used = True
    if not used:
        return pd.DataFrame(columns=["player_key", "production_score", "games_played"])

    games_col = first_existing_column(stats, ["games", "games_played", "recent_team_games", "week"])
    games_played = pd.to_numeric(stats[games_col], errors="coerce").fillna(0) if games_col else pd.Series(0, index=stats.index)
    player_scores = pd.DataFrame({"player_key": stats["player_key"], "production_raw": raw, "games_played": games_played})
    player_scores = player_scores.groupby("player_key", as_index=False).agg(
        production_raw=("production_raw", "sum"),
        games_played=("games_played", "max"),
    )
    player_scores["production_score"] = normalize_score(player_scores["production_raw"], center=75, spread=12, lower=45, upper=100)
    return player_scores[["player_key", "production_score", "games_played"]]


def build_draft_lookup(historical_roster):
    if historical_roster.empty or not {"full_name", "draft_number"}.issubset(historical_roster.columns):
        return {}
    draft = historical_roster.dropna(subset=["full_name", "draft_number"]).copy()
    draft["name_key"] = draft["full_name"].astype(str).str.lower().str.strip()
    draft["draft_number"] = pd.to_numeric(draft["draft_number"], errors="coerce")
    draft = draft.dropna(subset=["draft_number"])
    return draft.groupby("name_key")["draft_number"].min().to_dict()


def standardize_roster_columns(roster, selected_season, historical_roster=None):
    output = roster.copy()
    if output.empty:
        return output
    for column in ["season", "team", "position", "depth_chart_position", "status", "full_name", "draft_number", "years_exp", "entry_year", "rookie_year", "gsis_id"]:
        if column not in output.columns:
            output[column] = pd.NA
    output["season"] = pd.to_numeric(output["season"], errors="coerce").fillna(selected_season).astype(int)
    output["team"] = output["team"].apply(normalize_team)
    output["position"] = output["position"].fillna("").astype(str).str.upper().str.strip()
    output["depth_chart_position"] = output["depth_chart_position"].fillna(output["position"]).astype(str).str.upper().str.strip()
    output["status"] = output["status"].fillna("ACT").astype(str).str.upper().str.strip()
    output["full_name"] = output["full_name"].fillna("").astype(str).str.strip()
    output["name_key"] = output["full_name"].str.lower()
    output["draft_number"] = pd.to_numeric(output["draft_number"], errors="coerce")
    if historical_roster is not None and not historical_roster.empty:
        output["draft_number"] = output["draft_number"].fillna(output["name_key"].map(build_draft_lookup(historical_roster)))
    for column in ["years_exp", "entry_year", "rookie_year"]:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    output["years_exp"] = output["years_exp"].fillna(selected_season - output["entry_year"])
    output["years_exp"] = output["years_exp"].fillna(selected_season - output["rookie_year"])
    output["years_exp"] = output["years_exp"].clip(lower=0, upper=22).fillna(3)
    return output


def position_group(position, depth_position=""):
    pos = str(depth_position or position).upper()
    base = str(position).upper()
    if pos == "QB" or base == "QB":
        return "qb"
    if pos in {"WR", "RB", "FB", "TE"} or base in {"WR", "RB", "FB", "TE"}:
        return "skill"
    if pos in {"OT", "T", "G", "C", "OL", "OG"} or base in {"OT", "T", "G", "C", "OL", "OG"}:
        return "offensive_line"
    if pos in {"EDGE", "DL", "DT", "DE", "OLB", "NT"} or base in {"EDGE", "DL", "DT", "DE", "OLB", "NT"}:
        return "defensive_front"
    if pos in {"CB", "S", "DB", "FS", "SS"} or base in {"CB", "S", "DB", "FS", "SS"}:
        return "secondary"
    if pos in {"K", "P", "LS"} or base in {"K", "P", "LS"}:
        return "special_teams"
    return "other"


def is_premium_position(position, depth_position=""):
    pos = str(depth_position or position).upper()
    base = str(position).upper()
    return pos in {"QB", "WR", "OT", "T", "EDGE", "DE", "DT", "DL", "CB", "DB"} or base in {"QB", "WR", "OT", "T", "EDGE", "DE", "DT", "DL", "CB", "DB"}


def position_importance(position, depth_position=""):
    pos = str(depth_position or position).upper()
    group = position_group(position, depth_position)

    if pos == "QB":
        return 1.14
    if pos in {"EDGE", "DE", "OT", "T", "CB"}:
        return 1.08
    if pos in {"WR", "DL", "DT", "DB"}:
        return 1.04
    if group in {"offensive_line", "defensive_front", "secondary"}:
        return 1.02
    if group == "special_teams":
        return 0.78
    return 0.98 if group == "skill" else 0.92


def calculate_availability_multiplier(status):
    value = str(status).upper().strip()
    if value in {"ACT", "ACTIVE", "FULL"}:
        return 1.0
    if value in {"QUESTIONABLE", "QUE"}:
        return 0.82
    if value in {"DOUBTFUL", "DBT"}:
        return 0.35
    if value in {"IR", "RES", "PUP", "SUS", "RSR"}:
        return 0.14
    if value == "RSN":
        return 0.25
    if value in {"DEV", "EXE", "PRACTICE", "PS"}:
        return 0.35
    if value == "RFA":
        return 0.72
    if value == "UDF":
        return 0.62
    if value in {"UFA", "CUT", "WAIVED", "RELEASED", "NWT", "OUT"}:
        return 0.08
    return 0.88


def calculate_experience_score(row):
    exp = pd.to_numeric(row.get("years_exp"), errors="coerce")
    exp = 3 if pd.isna(exp) else max(0, min(float(exp), 22))
    if exp <= 1:
        score = 58 + 5 * exp
    elif exp <= 4:
        score = 65 + 6 * (exp - 1)
    elif exp <= 8:
        score = 83 + 1.5 * (8 - abs(6 - exp))
    else:
        score = 86 - 2.2 * (exp - 8)
    return float(np.clip(score, 52, 90))


def calculate_draft_capital_score(row):
    pick = pd.to_numeric(row.get("draft_number"), errors="coerce")
    if pd.isna(pick):
        return 48.0
    pick = float(pick)
    if pick <= 32:
        return float(95 - ((pick - 1) / 31) * 10)
    if pick <= 64:
        return float(85 - ((pick - 33) / 31) * 10)
    if pick <= 100:
        return float(78 - ((pick - 65) / 35) * 10)
    if pick <= 170:
        return float(68 - ((pick - 101) / 69) * 10)
    if pick <= 260:
        return float(55 - ((pick - 171) / 89) * 10)
    return 45.0


def calculate_rookie_projection_score(row):
    return float(np.clip(calculate_draft_capital_score(row) * position_importance(row.get("position"), row.get("depth_chart_position")), 40, 98))


def calculate_player_score_blend(player_row, current_week=None):
    fallback = 0.45 * calculate_draft_capital_score(player_row) + 0.40 * calculate_experience_score(player_row) + 0.15 * calculate_rookie_projection_score(player_row)
    production = pd.to_numeric(player_row.get("production_score"), errors="coerce")
    if pd.isna(production):
        return float(np.clip(fallback, 35, 98))
    games_played = pd.to_numeric(player_row.get("games_played"), errors="coerce")
    games_played = 0 if pd.isna(games_played) else games_played
    fallback_weight = 0.90 if games_played <= 2 else 0.65 if games_played <= 5 else 0.40 if games_played <= 9 else 0.15
    return float(np.clip(fallback_weight * fallback + (1 - fallback_weight) * production, 35, 100))


def build_player_stat_scores(roster, player_stats):
    output = roster.copy()
    output["production_score"] = pd.NA
    output["games_played"] = pd.NA
    seasonal_scores = derive_player_production_scores(player_stats.get("seasonal", pd.DataFrame()))
    weekly_scores = derive_player_production_scores(player_stats.get("weekly", pd.DataFrame()))
    score_frames = [df for df in [seasonal_scores, weekly_scores] if not df.empty]
    if not score_frames:
        return output, False

    scores = pd.concat(score_frames, ignore_index=True)
    scores = scores.groupby("player_key", as_index=False).agg(
        production_score=("production_score", "max"),
        games_played=("games_played", "max"),
    )
    roster_keys = output[["name_key"]].copy()
    roster_keys["player_key"] = output.get("gsis_id", pd.Series("", index=output.index)).fillna("").astype(str).str.lower().str.strip()
    roster_keys.loc[roster_keys["player_key"].eq(""), "player_key"] = roster_keys.loc[roster_keys["player_key"].eq(""), "name_key"]
    output["player_key"] = roster_keys["player_key"]
    output = output.merge(scores, on="player_key", how="left", suffixes=("", "_from_stats"))
    output["production_score"] = output["production_score_from_stats"].combine_first(output["production_score"])
    output["games_played"] = output["games_played_from_stats"].combine_first(output["games_played"])
    output = output.drop(columns=["production_score_from_stats", "games_played_from_stats"], errors="ignore")
    return output, output["production_score"].notna().any()


def weighted_top_average(values, top_n, neutral=70):
    numeric = pd.to_numeric(values, errors="coerce").dropna().sort_values(ascending=False).head(top_n)

    if numeric.empty:
        return float(neutral)

    top_average = float(np.average(numeric, weights=np.linspace(1.0, 0.42, len(numeric))))
    group_depth_average = float(numeric.mean())

    # Blend top-end talent with depth so one or two players do not carry the whole group.
    return float(np.clip(0.72 * top_average + 0.28 * group_depth_average, 45, 100))


def calculate_continuity_score(team_roster, previous_roster):
    if team_roster.empty or previous_roster.empty:
        return 75.0
    current_ids = set(team_roster.get("gsis_id", pd.Series(dtype=str)).dropna().astype(str))
    previous_ids = set(previous_roster.get("gsis_id", pd.Series(dtype=str)).dropna().astype(str))
    if current_ids and previous_ids:
        ratio = len(current_ids.intersection(previous_ids)) / max(len(previous_ids), 1)
    else:
        current_names = set(team_roster["name_key"].dropna().astype(str))
        previous_names = set(previous_roster["name_key"].dropna().astype(str))
        ratio = len(current_names.intersection(previous_names)) / max(len(previous_names), 1)
    return float(50 + 50 * np.clip(ratio, 0, 1))


def build_position_group_scores(team_roster):
    active = team_roster.copy()
    active["weighted_player_score"] = active["player_score"] * active["availability_multiplier"]
    return {
        "qb_score": weighted_top_average(active.loc[active["position_group"] == "qb", "weighted_player_score"], 2, 68),
        "offense_skill_score": weighted_top_average(active.loc[active["position_group"] == "skill", "weighted_player_score"], 8, 70),
        "offensive_line_score": weighted_top_average(active.loc[active["position_group"] == "offensive_line", "weighted_player_score"], 8, 70),
        "defensive_front_score": weighted_top_average(active.loc[active["position_group"] == "defensive_front", "weighted_player_score"], 9, 70),
        "secondary_score": weighted_top_average(active.loc[active["position_group"] == "secondary", "weighted_player_score"], 8, 70),
        "special_teams_score": weighted_top_average(active.loc[active["position_group"] == "special_teams", "weighted_player_score"], 3, 72),
        "premium_position_score": weighted_top_average(active.loc[active["premium_position"], "weighted_player_score"], 10, 70),
    }


def calculate_transaction_impact(team=None, transactions=None):
    # Future transaction feeds can update this with capped signed/traded/cut/IR impact.
    return 0.0


@st.cache_data(show_spinner=False)
def calculate_team_roster_score(roster, selected_season, team_universe, historical_roster=None, transactions=None, player_stats=None):
    return roster_scoring.calculate_team_roster_score(
        roster,
        selected_season,
        team_universe,
        historical_roster=historical_roster,
        transactions=transactions,
        player_stats=player_stats,
    )

    columns = [
        "team",
        "roster_score",
        "qb_score",
        "offense_skill_score",
        "offensive_line_score",
        "defensive_front_score",
        "secondary_score",
        "special_teams_score",
        "premium_position_score",
        "availability_score",
        "rookie_projection_score",
        "transaction_impact_score",
        "experience_score",
        "draft_capital_score",
        "continuity_score",
    ]
    neutral = {column: 75.0 for column in columns if column != "team"}
    if roster.empty:
        return pd.DataFrame([{**{"team": team}, **neutral} for team in team_universe])
    standardized = standardize_roster_columns(roster, selected_season, historical_roster)
    standardized, _ = build_player_stat_scores(standardized, player_stats or {"seasonal": pd.DataFrame(), "weekly": pd.DataFrame()})
    standardized["availability_multiplier"] = standardized["status"].apply(calculate_availability_multiplier)
    standardized["position_group"] = standardized.apply(lambda row: position_group(row["position"], row["depth_chart_position"]), axis=1)
    standardized["premium_position"] = standardized.apply(lambda row: is_premium_position(row["position"], row["depth_chart_position"]), axis=1)
    standardized["draft_capital_score"] = standardized.apply(calculate_draft_capital_score, axis=1)
    standardized["experience_score"] = standardized.apply(calculate_experience_score, axis=1)
    standardized["rookie_player_score"] = standardized.apply(calculate_rookie_projection_score, axis=1)
    standardized["player_score"] = standardized.apply(calculate_player_score_blend, axis=1)
    standardized["player_score"] = (
        standardized["player_score"]
        * standardized.apply(lambda row: position_importance(row["position"], row["depth_chart_position"]), axis=1)
    ).clip(35, 100)
    previous_all = pd.DataFrame()
    if historical_roster is not None and not historical_roster.empty:
        previous_all = standardize_roster_columns(historical_roster, selected_season - 1, historical_roster)
        previous_all = previous_all[previous_all["season"] == selected_season - 1].copy()
    rows = []
    for team in team_universe:
        team_roster = standardized[standardized["team"] == team].copy()
        if team_roster.empty:
            rows.append({**{"team": team}, **neutral})
            continue
        group_scores = build_position_group_scores(team_roster)
        availability_score = float(np.clip(50 + 50 * team_roster["availability_multiplier"].mean(), 50, 100))
        rookie_mask = (team_roster["years_exp"] <= 1) | (team_roster["entry_year"] >= selected_season - 1)
        rookie_score = weighted_top_average(team_roster.loc[rookie_mask, "rookie_player_score"], 5, 65)
        experience_score = weighted_top_average(team_roster["experience_score"], 30, 72)
        draft_capital_score = weighted_top_average(team_roster["draft_capital_score"], 20, 60)
        previous_roster = previous_all[previous_all["team"] == team].copy() if not previous_all.empty else pd.DataFrame()
        continuity_score = calculate_continuity_score(team_roster, previous_roster)
        transaction_impact = calculate_transaction_impact(team, transactions)
        base = (
            0.18 * group_scores["qb_score"]
            + 0.17 * group_scores["offensive_line_score"]
            + 0.13 * group_scores["offense_skill_score"]
            + 0.17 * group_scores["defensive_front_score"]
            + 0.14 * group_scores["secondary_score"]
            + 0.08 * group_scores["premium_position_score"]
            + 0.08 * availability_score
            + 0.03 * rookie_score
            + 0.02 * continuity_score
        )
        rows.append(
            {
                "team": team,
                "roster_score": float(np.clip(base + transaction_impact, 50, 100)),
                **group_scores,
                "availability_score": availability_score,
                "rookie_projection_score": float(np.clip(rookie_score, 40, 98)),
                "transaction_impact_score": float(transaction_impact),
                "experience_score": float(experience_score),
                "draft_capital_score": float(draft_capital_score),
                "continuity_score": float(continuity_score),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def merge_live_roster_updates(local_roster, live_roster):
    return local_roster if live_roster is None or live_roster.empty else local_roster


def calculate_projected_wins_score(projected_wins):
    return float(np.clip(50 + (pd.to_numeric(projected_wins, errors="coerce") / 17) * 50, 50, 100))


@st.cache_data(show_spinner=False)
def calculate_projected_wins(performance_all, records, roster_scores_all, selected_roster_scores, performance_season, selected_season):
    fallback_columns = ["team", "projected_wins", "projected_wins_low", "projected_wins_high", "projected_wins_range", "projected_wins_score"]
    selected_features = performance_all[performance_all["season"] == performance_season].copy()
    selected_features = selected_features.merge(records, on=["season", "team"], how="left")
    selected_features = selected_features.merge(selected_roster_scores, on="team", how="left")
    if performance_all.empty or records.empty:
        selected_features["projected_wins"] = 8.5
        meta = {"features": [], "model_available": False}
    else:
        model_df = performance_all.merge(records, on=["season", "team"], how="left")
        model_df = model_df.merge(roster_scores_all, on=["season", "team"], how="left")
        model_df = model_df.sort_values(["team", "season"])
        model_df["next_season"] = model_df.groupby("team")["season"].shift(-1)
        model_df["next_season_wins"] = model_df.groupby("team")["current_wins"].shift(-1)
        model_df.loc[model_df["next_season"] != model_df["season"] + 1, "next_season_wins"] = pd.NA
        candidates = [
            "current_wins",
            "current_gini_score",
            "performance_score",
            "offense_estat",
            "defense_estat",
            "point_diff_per_game",
            "off_adj_epa",
            "def_adj_epa",
            "success_margin",
            "turnover_margin_per_game",
            "penalty_yards_margin_per_game",
            "schedule_strength",
            "offense_rank",
            "defense_rank",
            "balance_gap",
            "roster_score",
            "qb_score",
            "premium_position_score",
            "availability_score",
        ]
        features = [column for column in candidates if column in model_df.columns and column in selected_features.columns]
        training_df = model_df.dropna(subset=["next_season_wins"]).copy()
        training_df = training_df[training_df["season"] < performance_season].copy()
        if training_df["season"].nunique() < 6 or len(training_df) < 80:
            training_df = model_df.dropna(subset=["next_season_wins"]).copy()
        training_df = clean_numeric_columns(training_df, features + ["next_season_wins"])
        selected_features = clean_numeric_columns(selected_features, features + ["current_wins"])
        if not SKLEARN_AVAILABLE or not features or training_df["season"].nunique() < 6:
            selected_features["projected_wins"] = selected_features["current_wins"].fillna(8.5).clip(0, 17)
            meta = {"features": features, "model_available": False}
        else:
            seasons = sorted(training_df["season"].dropna().astype(int).unique())
            test_count = max(3, min(5, len(seasons) // 4))
            test_seasons, train_seasons = seasons[-test_count:], seasons[:-test_count]
            train_df = training_df[training_df["season"].isin(train_seasons)].copy()
            test_df = training_df[training_df["season"].isin(test_seasons)].copy()
            specs = {
                "Ridge Regression": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", Ridge(alpha=4.0))]),
                "Gradient Boosting": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", GradientBoostingRegressor(n_estimators=160, learning_rate=0.04, max_depth=2, random_state=42))]),
                "Random Forest": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", RandomForestRegressor(n_estimators=320, min_samples_leaf=4, random_state=42))]),
            }
            fitted, results = {}, []
            for name, model in specs.items():
                model.fit(train_df[features], train_df["next_season_wins"])
                preds = model.predict(test_df[features]).clip(0, 17)
                mae = float(mean_absolute_error(test_df["next_season_wins"], preds))
                results.append({"name": name, "mae": mae})
                fitted[name] = model
            best = min(results, key=lambda item: item["mae"])
            baseline = test_df["current_wins"].fillna(train_df["current_wins"].median())
            baseline_mae = float(mean_absolute_error(test_df["next_season_wins"], baseline))
            production_model = clone(fitted[best["name"]])
            production_model.fit(training_df[features], training_df["next_season_wins"])
            model_predictions = pd.Series(production_model.predict(selected_features[features]).clip(0, 17), index=selected_features.index)
            baseline_selected = selected_features["current_wins"].fillna(training_df["current_wins"].median()).clip(0, 17)
            selected_features["projected_wins"] = (0.65 * model_predictions + 0.35 * baseline_selected).clip(0, 17) if best["mae"] > baseline_mae else model_predictions
            meta = {
                "features": features,
                "model_available": True,
                "model_name": best["name"],
                "model_mae": best["mae"],
                "baseline_mae": baseline_mae,
                "use_blend": best["mae"] > baseline_mae,
                "train_seasons": train_seasons,
                "test_seasons": test_seasons,
            }
    mae = float(meta.get("model_mae", 2.2)) if isinstance(meta, dict) else 2.2
    mae = max(1.4, min(3.5, mae))
    selected_features["projected_wins"] = selected_features["projected_wins"].fillna(8.5).clip(0, 17)
    selected_features["projected_wins_low"] = (selected_features["projected_wins"] - mae).round().clip(0, 17).astype(int)
    selected_features["projected_wins_high"] = (selected_features["projected_wins"] + mae).round().clip(0, 17).astype(int)
    selected_features["projected_wins_range"] = selected_features.apply(lambda row: f"Range: {row['projected_wins_low']}-{row['projected_wins_high']}", axis=1)
    selected_features["projected_wins_score"] = selected_features["projected_wins"].apply(calculate_projected_wins_score)
    return selected_features[fallback_columns], meta


def rank_metric_by_season(df, metric_col, rank_col, ascending=False):
    df[rank_col] = df.groupby("season")[metric_col].rank(ascending=ascending, method="min")
    return df


def calculate_checkpoint_metrics(team_game, seasons):
    required = {"season", "season_type", "week", "team", "estat_game", "off_adj_epa", "def_adj_epa", "point_diff", "success_margin"}
    if team_game.empty or not required.issubset(team_game.columns):
        return pd.DataFrame(columns=["season", "team"])
    regular = team_game.copy()
    regular["season"] = pd.to_numeric(regular["season"], errors="coerce")
    regular = regular.dropna(subset=["season"])
    regular["season"] = regular["season"].astype(int)
    regular["team"] = regular["team"].apply(normalize_team)
    regular = regular[(regular["season"].isin(seasons)) & regular["season_type"].astype(str).str.upper().eq("REG")].copy()
    regular = clean_numeric_columns(regular, ["week", "estat_game", "off_adj_epa", "def_adj_epa", "point_diff", "success_margin"])
    regular["epa_diff_game"] = regular["off_adj_epa"] + regular["def_adj_epa"]
    regular["best_epa_unit_game"] = regular[["off_adj_epa", "def_adj_epa"]].max(axis=1)
    checkpoint_metrics = regular[["season", "team"]].drop_duplicates().copy()
    for week in (5, 6, 8, 10, 12, 14):
        checkpoint = regular[regular["week"] <= week].groupby(["season", "team"], as_index=False).agg(
            checkpoint_estat=("estat_game", "mean"),
            checkpoint_epa=("epa_diff_game", "mean"),
            checkpoint_unit=("best_epa_unit_game", "mean"),
            checkpoint_point_diff=("point_diff", "mean"),
            checkpoint_success=("success_margin", "mean"),
        )
        prefix = f"thru{week}"
        for suffix, source_col in [("estat", "checkpoint_estat"), ("epa", "checkpoint_epa"), ("unit", "checkpoint_unit"), ("pd", "checkpoint_point_diff"), ("success", "checkpoint_success")]:
            checkpoint[f"{prefix}_{suffix}"] = checkpoint[source_col]
            rank_metric_by_season(checkpoint, source_col, f"{prefix}_{suffix}_rank")
        keep = ["season", "team"] + [column for column in checkpoint.columns if column.startswith(f"{prefix}_")]
        checkpoint_metrics = checkpoint_metrics.merge(checkpoint[keep], on=["season", "team"], how="left")
    for start_week in (10, 12, 14):
        checkpoint = regular[regular["week"] >= start_week].groupby(["season", "team"], as_index=False).agg(
            checkpoint_estat=("estat_game", "mean"),
            checkpoint_epa=("epa_diff_game", "mean"),
            checkpoint_unit=("best_epa_unit_game", "mean"),
            checkpoint_point_diff=("point_diff", "mean"),
            checkpoint_success=("success_margin", "mean"),
        )
        prefix = f"late{start_week}"
        for suffix, source_col in [("estat", "checkpoint_estat"), ("epa", "checkpoint_epa"), ("unit", "checkpoint_unit"), ("pd", "checkpoint_point_diff"), ("success", "checkpoint_success")]:
            checkpoint[f"{prefix}_{suffix}"] = checkpoint[source_col]
            rank_metric_by_season(checkpoint, source_col, f"{prefix}_{suffix}_rank")
        keep = ["season", "team"] + [column for column in checkpoint.columns if column.startswith(f"{prefix}_")]
        checkpoint_metrics = checkpoint_metrics.merge(checkpoint[keep], on=["season", "team"], how="left")
    return checkpoint_metrics


def calculate_super_square_inputs(team_game):
    required = {"season", "season_type", "team", "game_id", "point_diff", "off_adj_epa", "def_adj_epa", "success_margin", "estat_game"}
    if team_game.empty or not required.issubset(team_game.columns):
        return pd.DataFrame()
    seasons = sorted(pd.to_numeric(team_game["season"], errors="coerce").dropna().astype(int).unique())
    regular = team_game.copy()
    regular["season"] = pd.to_numeric(regular["season"], errors="coerce")
    regular = regular.dropna(subset=["season"])
    regular["season"] = regular["season"].astype(int)
    regular["team"] = regular["team"].apply(normalize_team)
    regular = regular[regular["season_type"].astype(str).str.upper().eq("REG")].copy()
    regular = clean_numeric_columns(regular, ["point_diff", "off_adj_epa", "def_adj_epa", "success_margin", "estat_game"])
    regular["epa_diff_game"] = regular["off_adj_epa"] + regular["def_adj_epa"]
    regular["best_epa_unit_game"] = regular[["off_adj_epa", "def_adj_epa"]].max(axis=1)
    metrics = regular.groupby(["season", "team"], as_index=False).agg(
        games=("game_id", "nunique"),
        point_diff_per_game=("point_diff", "mean"),
        off_epa=("off_adj_epa", "mean"),
        def_epa=("def_adj_epa", "mean"),
        success_margin=("success_margin", "mean"),
        estat=("estat_game", "mean"),
        pd_pos_share=("point_diff", lambda values: (values > 0).mean()),
    )
    metrics["EPA Differential"] = metrics["off_epa"] + metrics["def_epa"]
    metrics["off_z"] = zscore_by_season(metrics, "off_epa")
    metrics["def_z"] = zscore_by_season(metrics, "def_epa")
    metrics["pd_z"] = zscore_by_season(metrics, "point_diff_per_game")
    metrics["success_z"] = zscore_by_season(metrics, "success_margin")
    metrics["Offense EStat"] = 100 + 15 * metrics["off_z"]
    metrics["Defense EStat"] = 100 + 15 * metrics["def_z"]
    metrics["Gini Score"] = 100 + 15 * (0.35 * metrics["off_z"] + 0.35 * metrics["def_z"] + 0.15 * metrics["pd_z"] + 0.15 * metrics["success_z"])
    rank_metric_by_season(metrics, "Gini Score", "gini_rank")
    rank_metric_by_season(metrics, "point_diff_per_game", "point_diff_rank")
    rank_metric_by_season(metrics, "EPA Differential", "epa_diff_rank")
    rank_metric_by_season(metrics, "success_margin", "success_rank")
    rank_metric_by_season(metrics, "pd_pos_share", "pd_pos_share_rank")
    rank_metric_by_season(metrics, "Offense EStat", "offense_rank_calc")
    rank_metric_by_season(metrics, "Defense EStat", "defense_rank_calc")
    metrics["best_unit_rank"] = metrics[["offense_rank_calc", "defense_rank_calc"]].min(axis=1)
    metrics = metrics.merge(calculate_checkpoint_metrics(team_game, seasons), on=["season", "team"], how="left")
    return metrics


def research_quadrant_config(super_square_inputs):
    rows = []
    for season, team in SUPER_BOWL_WINNERS.items():
        match = super_square_inputs[(super_square_inputs["season"] == season) & (super_square_inputs["team"] == normalize_team(team))]
        if not match.empty:
            rows.append(match.iloc[0])
    champions = pd.DataFrame(rows)
    requirements = []
    for column, direction, axis in QUADRANT_REQUIREMENTS:
        if champions.empty or column not in champions.columns:
            cutoff = 100.0 if direction == "high" else 16.0
        elif direction == "low":
            cutoff = float(pd.to_numeric(champions[column], errors="coerce").max())
        else:
            cutoff = float(pd.to_numeric(champions[column], errors="coerce").min())
        requirements.append({"column": column, "direction": direction, "axis": axis, "cutoff": cutoff})
    return {"requirements": requirements, "x_cut": 100.0, "y_cut": 100.0}


def gate_score(rule_df, requirement):
    column = requirement["column"]
    if column not in rule_df.columns:
        return pd.Series(80.0, index=rule_df.index)
    series = pd.to_numeric(rule_df[column], errors="coerce")
    cutoff = float(requirement["cutoff"])
    if requirement["direction"] == "low":
        score = 100 + 5 * (cutoff - series)
    else:
        season_std = series.groupby(rule_df["season"]).transform("std").replace(0, np.nan)
        fallback_scale = max(abs(cutoff), 0.05)
        score = 100 + 18 * ((series - cutoff) / season_std.fillna(fallback_scale).clip(lower=fallback_scale))
    return score.clip(30, 170).fillna(80)


def calculate_control_profile_score(super_square_inputs, config):
    output = super_square_inputs.copy()
    controls = []
    for index, requirement in enumerate(config["requirements"], start=1):
        score_col = f"gate_{index}_score"
        output[score_col] = gate_score(output, requirement)
        if requirement["axis"] == "control":
            controls.append(score_col)
    output["Control Profile Score"] = output[controls].min(axis=1) if controls else 85.0
    return output


def calculate_unit_pressure_score(super_square_inputs, config):
    output = super_square_inputs.copy()
    pressure = []
    for index, requirement in enumerate(config["requirements"], start=1):
        score_col = f"gate_{index}_score"
        if score_col not in output.columns:
            output[score_col] = gate_score(output, requirement)
        if requirement["axis"] == "pressure":
            pressure.append(score_col)
    output["Unit Pressure Score"] = output[pressure].min(axis=1) if pressure else 85.0
    return output


def sigmoid(value):
    return 1 / (1 + np.exp(-np.clip(value, -40, 40)))


def assign_most_likely_quadrant(df):
    output = df.copy()
    cols = ["q1_probability", "q2_probability", "q3_probability", "q4_probability"]
    output["highest_quadrant_probability"] = output[cols].max(axis=1)
    output["most_likely_quadrant"] = output[cols].idxmax(axis=1).map(QUADRANT_LABELS)
    return output


def calculate_quadrant_probability_score(row):
    return float(np.clip(row["q1_probability"] * 60 + row["q2_probability"] * 75 + row["q3_probability"] * 88 + row["q4_probability"] * 100, 50, 100))


@st.cache_data(show_spinner=False)
def calculate_quadrant_probabilities(team_game, performance_season, team_universe):
    inputs = calculate_super_square_inputs(team_game)
    if inputs.empty:
        output = pd.DataFrame({"team": team_universe})
        output[["q1_probability", "q2_probability", "q3_probability", "q4_probability"]] = 0.25
        output["Control Profile Score"] = 85.0
        output["Unit Pressure Score"] = 85.0
    else:
        config = research_quadrant_config(inputs)
        scored = calculate_unit_pressure_score(calculate_control_profile_score(inputs, config), config)
        output = scored[(scored["season"] == performance_season) & (scored["team"].isin(team_universe))].copy()
        pc = sigmoid((pd.to_numeric(output["Control Profile Score"], errors="coerce").fillna(85) - 100) / 8)
        pp = sigmoid((pd.to_numeric(output["Unit Pressure Score"], errors="coerce").fillna(85) - 100) / 8)
        output["q1_probability"] = (1 - pc) * (1 - pp)
        output["q2_probability"] = pc * (1 - pp)
        output["q3_probability"] = (1 - pc) * pp
        output["q4_probability"] = pc * pp
        total = output[["q1_probability", "q2_probability", "q3_probability", "q4_probability"]].sum(axis=1).replace(0, 1)
        for column in ["q1_probability", "q2_probability", "q3_probability", "q4_probability"]:
            output[column] = output[column] / total
    output = assign_most_likely_quadrant(output)
    output["quadrant_probability_score"] = output.apply(calculate_quadrant_probability_score, axis=1)
    return output[["team", "most_likely_quadrant", "highest_quadrant_probability", "q1_probability", "q2_probability", "q3_probability", "q4_probability", "quadrant_probability_score", "Control Profile Score", "Unit Pressure Score"]]


def calculate_live_market_score(row):
    return float(
        np.clip(
            0.65 * row["performance_score"]
            + 0.10 * row["roster_score"]
            + 0.15 * row["projected_wins_score"]
            + 0.10 * row["quadrant_probability_score"],
            45,
            120,
        )
    )


def merge_leaderboard_scores(performance_df, roster_scores, projected_wins, quadrants, team_assets, selected_season):
    names, logos, colors = build_team_name_lookup(team_assets), build_team_logo_lookup(team_assets), build_team_color_lookup(team_assets)
    output = performance_df.merge(roster_scores, on="team", how="left")
    output = output.merge(projected_wins, on="team", how="left")
    output = output.merge(quadrants, on="team", how="left")
    output["team_name"] = output["team"].map(names).fillna(output["team"])
    output["team_logo"] = output["team"].map(logos).fillna("")
    output["team_primary_color"] = output["team"].map(lambda team: colors.get(team, {}).get("primary", PRIMARY))
    output["team_secondary_color"] = output["team"].map(lambda team: colors.get(team, {}).get("secondary", SECONDARY))
    output["season"] = selected_season
    for column, fallback in [
        ("roster_score", 75),
        ("projected_wins", 8.5),
        ("projected_wins_score", 75),
        ("projected_wins_low", 6),
        ("projected_wins_high", 11),
        ("quadrant_probability_score", 75),
        ("q1_probability", 0.25),
        ("q2_probability", 0.25),
        ("q3_probability", 0.25),
        ("q4_probability", 0.25),
        ("highest_quadrant_probability", 0.25),
    ]:
        output[column] = pd.to_numeric(output.get(column, fallback), errors="coerce").fillna(fallback)
    output["projected_wins_range"] = output.get("projected_wins_range", "").fillna("")
    output["most_likely_quadrant"] = output.get("most_likely_quadrant", "Q2 - Middle Tier").fillna("Q2 - Middle Tier")
    output["live_market_score"] = output.apply(calculate_live_market_score, axis=1)
    output = output.sort_values("live_market_score", ascending=False).reset_index(drop=True)
    output["live_rank"] = np.arange(1, len(output) + 1)
    return output


def load_previous_snapshot():
    if not SNAPSHOT_PATH.exists():
        return pd.DataFrame(), False
    snapshot, _ = safe_read_csv(SNAPSHOT_PATH)
    return snapshot, not snapshot.empty


def _set_neutral_weekly_movement(output):
    output["previous_week_rank"] = output.get("live_rank", 0)
    output["previous_week_score"] = output.get("live_market_score", 0.0)
    output["weekly_rank_change"] = 0
    output["weekly_score_change"] = 0.0
    output["weekly_movement_arrow"] = "\u2192"
    output["previous_rank"] = output["previous_week_rank"]
    output["previous_score"] = output["previous_week_score"]
    output["rank_change"] = output["weekly_rank_change"]
    output["score_change"] = output["weekly_score_change"]
    output["movement_arrow"] = output["weekly_movement_arrow"]
    return output


def calculate_weekly_movement(leaderboard, weekly_snapshot):
    output = leaderboard.copy()
    required = {"season", "team", "live_rank", "live_market_score"}
    if output.empty or weekly_snapshot.empty or not required.issubset(weekly_snapshot.columns):
        return _set_neutral_weekly_movement(output)

    previous = weekly_snapshot[weekly_snapshot["season"] == output["season"].iloc[0]].copy()
    if previous.empty or not comparable_ranked_populations(output, previous):
        return _set_neutral_weekly_movement(output)

    previous = previous[["team", "live_rank", "live_market_score"]].rename(
        columns={
            "live_rank": "previous_week_rank",
            "live_market_score": "previous_week_score",
        }
    )
    output = output.merge(previous, on="team", how="left")
    output["previous_week_rank"] = pd.to_numeric(output["previous_week_rank"], errors="coerce").fillna(output["live_rank"])
    output["previous_week_score"] = pd.to_numeric(output["previous_week_score"], errors="coerce").fillna(output["live_market_score"])
    output["weekly_rank_change"] = pd.to_numeric(output["previous_week_rank"], errors="coerce") - pd.to_numeric(output["live_rank"], errors="coerce")
    output["weekly_score_change"] = pd.to_numeric(output["live_market_score"], errors="coerce") - pd.to_numeric(output["previous_week_score"], errors="coerce")
    output["weekly_movement_arrow"] = np.select(
        [
            output["weekly_rank_change"] > 0,
            output["weekly_rank_change"] < 0,
            output["weekly_score_change"] > 0.05,
            output["weekly_score_change"] < -0.05,
        ],
        ["\u2191", "\u2193", "\u2191", "\u2193"],
        default="\u2192",
    )
    output["previous_rank"] = output["previous_week_rank"]
    output["previous_score"] = output["previous_week_score"]
    output["rank_change"] = output["weekly_rank_change"]
    output["score_change"] = output["weekly_score_change"]
    output["movement_arrow"] = output["weekly_movement_arrow"]
    return output


def calculate_rank_movement(leaderboard, previous_snapshot):
    return calculate_weekly_movement(leaderboard, previous_snapshot)


def should_save_daily_snapshot():
    current_time = now_et()
    latest_due = latest_daily_reload_due_time(current_time)
    if current_time - latest_due > timedelta(hours=1):
        return False

    snapshot, has_snapshot = load_previous_snapshot()

    if not has_snapshot:
        return True

    snapshot_day = latest_due.date().isoformat()
    if "snapshot_day" in snapshot.columns and snapshot["snapshot_day"].astype(str).eq(snapshot_day).any():
        return False

    if "snapshot_timestamp" in snapshot.columns:
        timestamps = pd.to_datetime(snapshot["snapshot_timestamp"], errors="coerce", utc=True)
        if timestamps.notna().any():
            latest_snapshot_time = timestamps.max().to_pydatetime().astimezone(ET)
            if latest_snapshot_time >= latest_due:
                return False

    return True


def save_live_snapshot(leaderboard, period_key):
    if not should_save_daily_snapshot():
        return False

    snapshot = leaderboard.copy()
    snapshot["snapshot_timestamp"] = now_et().isoformat()
    snapshot["snapshot_day"] = daily_snapshot_key()
    snapshot["snapshot_week"] = period_key

    live_columns = ["snapshot_day", "snapshot_week"] + SNAPSHOT_COLUMNS
    for column in live_columns:
        if column not in snapshot.columns:
            snapshot[column] = pd.NA

    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    snapshot[live_columns].to_csv(SNAPSHOT_PATH, index=False)
    return True


def first_matching_column(df, candidates):
    if df is None or not hasattr(df, "columns"):
        return None
    column_lookup = {str(column).lower(): column for column in df.columns}
    for candidate in candidates:
        match = column_lookup.get(candidate.lower())
        if match is not None:
            return match
    return None


def snapshot_cutoff_at(day):
    return datetime.combine(day, WEEKLY_SNAPSHOT_CUTOFF_TIME).replace(tzinfo=ET)


def current_calendar_week_cutoff(moment):
    moment = coerce_et(moment)
    monday = moment.date() - timedelta(days=moment.weekday())
    cutoff_day = monday + timedelta(days=WEEKLY_SNAPSHOT_CUTOFF_WEEKDAY)
    return snapshot_cutoff_at(cutoff_day)


def weekly_snapshot_window_key(cutoff):
    return coerce_et(cutoff).isoformat()


def weekly_snapshot_already_saved(history, week_key, save_cutoff):
    if history.empty:
        return False

    window_key = weekly_snapshot_window_key(save_cutoff)

    if "snapshot_window" in history.columns and history["snapshot_window"].astype(str).eq(window_key).any():
        return True

    if "snapshot_timestamp" in history.columns:
        timestamps = pd.to_datetime(history["snapshot_timestamp"], errors="coerce", utc=True)
        window_start = save_cutoff.astimezone(timezone.utc)
        window_end = (save_cutoff + timedelta(days=7)).astimezone(timezone.utc)
        in_window = (timestamps >= window_start) & (timestamps < window_end)

        if "snapshot_week" in history.columns:
            in_window = in_window & history["snapshot_week"].astype(str).eq(str(week_key))

        if in_window.any():
            return True

    if "snapshot_week" in history.columns:
        return history["snapshot_week"].astype(str).eq(str(week_key)).any()

    return False


def next_weekly_snapshot_cutoff(start_date):
    days_until_cutoff = (WEEKLY_SNAPSHOT_CUTOFF_WEEKDAY - start_date.weekday()) % 7
    if days_until_cutoff == 0:
        days_until_cutoff = 7
    return snapshot_cutoff_at(start_date + timedelta(days=days_until_cutoff))


def schedule_date_values(values):
    parsed = pd.to_datetime(values, errors="coerce")
    try:
        if parsed.dt.tz is not None:
            parsed = parsed.dt.tz_convert(ET).dt.tz_localize(None)
        return parsed.dt.date
    except Exception:
        fallback = pd.to_datetime(values.astype(str).str.slice(0, 10), errors="coerce")
        return fallback.dt.date


def snapshot_period_sort_key(value):
    text = str(value).strip()
    parts = text.split("-")

    # Legacy calendar-week format, e.g. 2026-W22.
    if len(parts) == 2 and parts[1].lower().startswith("w"):
        year = pd.to_numeric(parts[0], errors="coerce")
        week = pd.to_numeric(parts[1][1:], errors="coerce")

        if pd.isna(year) or pd.isna(week):
            return None

        try:
            week_start = datetime.fromisocalendar(int(year), int(week), 1).date().isoformat()
        except ValueError:
            week_start = f"{int(week):02d}"

        return (int(year), 0, 0, week_start)

    # Old offseason format, if it already exists in your CSV.
    if len(parts) == 2 and parts[1].lower() == "offseason":
        year = pd.to_numeric(parts[0], errors="coerce")
        if pd.isna(year):
            return None
        return (int(year), 0, 0, "0000-00-00")

    # New weekly offseason format:
    # YYYY-Offseason-YYYY-MM-DD
    if len(parts) == 5 and parts[1].lower() == "offseason":
        year = pd.to_numeric(parts[0], errors="coerce")
        cutoff_date = "-".join(parts[2:5])

        if pd.isna(year):
            return None

        return (int(year), 0, 0, cutoff_date)

    # Preseason format:
    # YYYY-Preseason-Week-WW
    if len(parts) == 4 and parts[1].lower() == "preseason" and parts[2].lower() == "week":
        year = pd.to_numeric(parts[0], errors="coerce")
        week = pd.to_numeric(parts[3], errors="coerce")

        if pd.isna(year) or pd.isna(week):
            return None

        return (int(year), 1, int(week), "0000-00-00")

    # Regular season format:
    # YYYY-Week-WW
    if len(parts) == 3 and parts[1].lower() == "week":
        year = pd.to_numeric(parts[0], errors="coerce")
        week = pd.to_numeric(parts[2], errors="coerce")

        if pd.isna(year) or pd.isna(week):
            return None

        return (int(year), 2, int(week), "0000-00-00")

    # Playoff formats generated by current_nfl_period_key:
    # YYYY-Wild-Card-Week, YYYY-Divisional-Week, YYYY-Conference-Championship, YYYY-Super-Bowl
    if len(parts) >= 2:
        year = pd.to_numeric(parts[0], errors="coerce")
        if not pd.isna(year):
            playoff_label = "-".join(parts[1:]).lower()
            playoff_order = {
                "wild-card-week": 1,
                "divisional-week": 2,
                "conference-championship": 3,
                "super-bowl": 4,
            }
            if playoff_label.startswith("playoff-week-"):
                week = pd.to_numeric(playoff_label.replace("playoff-week-", ""), errors="coerce")
                if not pd.isna(week):
                    return (int(year), 3, int(week), "0000-00-00")
            if playoff_label in playoff_order:
                return (int(year), 3, playoff_order[playoff_label], "0000-00-00")

    return None


def current_nfl_period_key(selected_season, schedule_df):
    season_value = pd.to_numeric(pd.Series([selected_season]), errors="coerce").iloc[0]
    season = int(season_value) if not pd.isna(season_value) else now_et().year
    today = now_et().date()

    def offseason_period():
        cutoff = current_calendar_week_cutoff(now_et())
        key = f"{season}-Offseason-{cutoff.date().isoformat()}"
        WEEKLY_SNAPSHOT_CUTOFFS[key] = cutoff
        return key, season, "Offseason"

    if not isinstance(schedule_df, pd.DataFrame) or schedule_df.empty:
        return offseason_period()

    schedule = schedule_df.copy()

    season_col = first_matching_column(schedule, ["season", "game_season"])
    if season_col is not None:
        season_values = pd.to_numeric(schedule[season_col], errors="coerce")
        schedule = schedule[season_values.eq(season)].copy()

    date_col = first_matching_column(schedule, ["gameday", "game_date", "date"])
    week_col = first_matching_column(schedule, ["week", "game_week", "week_number"])
    type_col = first_matching_column(schedule, ["season_type", "game_type"])

    if schedule.empty or date_col is None or week_col is None:
        return offseason_period()

    schedule["_game_date"] = schedule_date_values(schedule[date_col])
    schedule["_week"] = pd.to_numeric(schedule[week_col], errors="coerce")

    if type_col is not None:
        schedule["_season_type"] = schedule[type_col].astype(str).str.strip().str.upper()
    else:
        schedule["_season_type"] = "REG"

    schedule = schedule.dropna(subset=["_game_date", "_week"]).copy()

    if schedule.empty:
        return offseason_period()

    # Preseason
    preseason = schedule[schedule["_season_type"].str.startswith("PRE")].copy()
    if not preseason.empty:
        preseason_starts = preseason.groupby("_week")["_game_date"].min().sort_index()
        preseason_started = preseason_starts[preseason_starts <= today]

        regular = schedule[schedule["_season_type"].str.startswith("REG")].copy()
        regular_start = regular["_game_date"].min() if not regular.empty else None

        if not preseason_started.empty and (regular_start is None or today < regular_start):
            current_week_index = preseason_started.index.max()
            current_week = int(current_week_index)
            week_start_date = preseason_started.loc[current_week_index]

            key = f"{season}-Preseason-Week-{current_week:02d}"
            WEEKLY_SNAPSHOT_CUTOFFS[key] = next_weekly_snapshot_cutoff(week_start_date)

            return key, season, f"Preseason Week {current_week}"

    # Regular season
    regular = schedule[schedule["_season_type"].str.startswith("REG")].copy()
    if not regular.empty:
        regular_starts = regular.groupby("_week")["_game_date"].min().sort_index()
        regular_started = regular_starts[regular_starts <= today]

        postseason = schedule[
            schedule["_season_type"].str.startswith("POST")
            | schedule["_season_type"].str.startswith("PLAYOFF")
        ].copy()
        postseason_start = postseason["_game_date"].min() if not postseason.empty else None

        if not regular_started.empty and (postseason_start is None or today < postseason_start):
            current_week_index = regular_started.index.max()
            current_week = int(current_week_index)
            week_start_date = regular_started.loc[current_week_index]

            key = f"{season}-Week-{current_week:02d}"
            WEEKLY_SNAPSHOT_CUTOFFS[key] = next_weekly_snapshot_cutoff(week_start_date)

            return key, season, f"Week {current_week}"

    # Playoffs
    postseason = schedule[
        schedule["_season_type"].str.startswith("POST")
        | schedule["_season_type"].str.startswith("PLAYOFF")
    ].copy()

    if not postseason.empty:
        postseason_starts = postseason.groupby("_week")["_game_date"].min().sort_index()
        postseason_started = postseason_starts[postseason_starts <= today]

        if not postseason_started.empty:
            current_week_index = postseason_started.index.max()
            current_week = int(current_week_index)
            week_start_date = postseason_started.loc[current_week_index]

            playoff_labels = {
                19: "Wild Card Week",
                20: "Divisional Week",
                21: "Conference Championship",
                22: "Super Bowl",
            }

            playoff_label = playoff_labels.get(current_week, f"Playoff Week {current_week}")

            key = f"{season}-{playoff_label.replace(' ', '-')}"
            WEEKLY_SNAPSHOT_CUTOFFS[key] = next_weekly_snapshot_cutoff(week_start_date)

            return key, season, playoff_label

    return offseason_period()


def load_weekly_history():
    if not WEEKLY_HISTORY_PATH.exists():
        return pd.DataFrame(), False
    history, _ = safe_read_csv(WEEKLY_HISTORY_PATH)
    return history, not history.empty


def get_latest_weekly_movement_snapshot(current_week):
    history, has_history = load_weekly_history()

    if not has_history or "snapshot_week" not in history.columns:
        return pd.DataFrame(), False

    current_sort_key = snapshot_period_sort_key(current_week)
    if current_sort_key is None:
        return pd.DataFrame(), False

    history = history.copy()
    history["_snapshot_sort_key"] = history["snapshot_week"].apply(snapshot_period_sort_key)
    history = history[history["_snapshot_sort_key"].notna()].copy()
    candidates = history[history["_snapshot_sort_key"].apply(lambda sort_key: sort_key <= current_sort_key)].copy()

    if candidates.empty:
        return pd.DataFrame(), False

    latest_sort_key = max(candidates["_snapshot_sort_key"])
    latest_snapshot = candidates[candidates["_snapshot_sort_key"].apply(lambda sort_key: sort_key == latest_sort_key)].copy()

    if "snapshot_timestamp" in latest_snapshot.columns:
        timestamps = pd.to_datetime(latest_snapshot["snapshot_timestamp"], errors="coerce", utc=True)
        if timestamps.notna().any():
            latest_timestamp = timestamps.max()
            latest_snapshot = latest_snapshot[timestamps.eq(latest_timestamp)].copy()

    latest_snapshot = latest_snapshot.drop(columns=["_snapshot_sort_key"], errors="ignore")

    return latest_snapshot, not latest_snapshot.empty


def get_latest_prior_week_snapshot(current_week):
    return get_latest_weekly_movement_snapshot(current_week)


def should_save_weekly_snapshot(week_key):
    if not week_key:
        return False
    if snapshot_period_sort_key(week_key) is None:
        return False

    current_time = now_et()
    save_cutoff = WEEKLY_SNAPSHOT_CUTOFFS.get(str(week_key), current_calendar_week_cutoff(current_time))

    if current_time < save_cutoff:
        return False

    history, has_history = load_weekly_history()

    if has_history and weekly_snapshot_already_saved(history, week_key, save_cutoff):
        return False

    return True


def save_weekly_snapshot_if_needed(leaderboard, week_key):
    if not should_save_weekly_snapshot(week_key):
        return False, week_key

    history, has_history = load_weekly_history()
    save_cutoff = WEEKLY_SNAPSHOT_CUTOFFS.get(str(week_key), current_calendar_week_cutoff(now_et()))

    if has_history and weekly_snapshot_already_saved(history, week_key, save_cutoff):
        return False, week_key

    snapshot = leaderboard.copy()
    snapshot["snapshot_timestamp"] = now_et().isoformat()
    snapshot["snapshot_window"] = weekly_snapshot_window_key(save_cutoff)
    snapshot["snapshot_week"] = week_key

    weekly_columns = ["snapshot_window", "snapshot_week"] + SNAPSHOT_COLUMNS

    for column in weekly_columns:
        if column not in snapshot.columns:
            snapshot[column] = pd.NA

    new_rows = snapshot[weekly_columns].copy()

    if has_history and not history.empty:
        history = pd.concat([history, new_rows], ignore_index=True)
    else:
        history = new_rows

    WEEKLY_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(WEEKLY_HISTORY_PATH, index=False)

    return True, week_key


def generate_status_label(row):
    projected_delta = row.get("projected_wins", 8.5) - row.get("current_wins", 8.5)
    if row.get("live_market_score", 0) >= 96:
        return "Elite Market"
    if row.get("most_likely_quadrant") == "Q4 - Super Bowl Contenders" and row.get("highest_quadrant_probability", 0) >= 0.34:
        return "Q4 Threat"
    if projected_delta >= 2 and row.get("roster_score", 0) >= 78:
        return "High Upside"
    if row.get("movement_arrow") == "\u2191":
        return "Rising"
    if projected_delta <= -2:
        return "Regression Watch"
    if row.get("roster_score", 0) >= row.get("performance_score", 0) + 8:
        return "Roster Boost"
    if row.get("live_market_score", 0) < 72:
        return "Watch List"
    return "Stable"


def generate_movement_reason(row, has_snapshot):
    if not has_snapshot:
        return "No compatible weekly baseline yet"
    rank_change = pd.to_numeric(row.get("rank_change"), errors="coerce")
    score_change = pd.to_numeric(row.get("score_change"), errors="coerce")
    if pd.notna(rank_change) and rank_change < 0:
        return "Current profile slipped versus the weekly baseline"
    if pd.notna(rank_change) and rank_change == 0 and (pd.isna(score_change) or abs(score_change) < 0.05):
        return "No rank movement since the weekly baseline"
    if pd.notna(score_change) and score_change < 0:
        return "Live score softened since the weekly baseline"
    if row.get("roster_score", 0) >= 82:
        return "Roster score lifted profile"
    if row.get("current_gini_score", 0) >= 106:
        return "Strong Gini performance profile"
    if row.get("projected_wins", 0) >= 10:
        return "Projected wins improved"
    if row.get("most_likely_quadrant") in {"Q3 - Playoff Contenders", "Q4 - Super Bowl Contenders"}:
        return "Moved closer to a stronger quadrant profile"
    return "Current profile improved versus the weekly baseline"


def render_live_status_strip(selected_season, weekly_snapshot_period, static_time, leaderboard, has_snapshot=True):
    fallback = format_et_timestamp(static_time, include_seconds=True)
    fallback_dt = static_time.replace(tzinfo=ET) if static_time.tzinfo is None else static_time.astimezone(ET)
    next_refresh_fallback_dt = next_daily_reload_time(fallback_dt)
    next_refresh_fallback = next_refresh_fallback_dt.strftime("%I:%M %p").lstrip("0") + " ET"

    best = leaderboard.sort_values("live_rank", ascending=True).iloc[0]
    worst = leaderboard.sort_values("live_rank", ascending=False).iloc[0]

    risers = leaderboard[pd.to_numeric(leaderboard["rank_change"], errors="coerce") > 0].copy()
    fallers = leaderboard[pd.to_numeric(leaderboard["rank_change"], errors="coerce") < 0].copy()

    if risers.empty:
        riser = best
    else:
        riser = risers.sort_values("rank_change", ascending=False).iloc[0]

    if fallers.empty:
        faller = worst
    else:
        faller = fallers.sort_values("rank_change", ascending=True).iloc[0]

    def movement_amount(row):
        value = pd.to_numeric(row.get("rank_change", 0), errors="coerce")
        return 0 if pd.isna(value) else int(abs(value))

    def card_style(row):
        return (
            f'--card-accent:{escape(row.get("team_primary_color", PRIMARY))};'
            f'--card-accent2:{escape(row.get("team_secondary_color", SECONDARY))};'
        )

    riser_change = movement_amount(riser)
    faller_change = movement_amount(faller)

    riser_text = f"+{riser_change}" if riser_change else "0"
    faller_text = f"-{faller_change}" if faller_change else "0"
    riser_label = "Riser of Week" if has_snapshot else "Movement Pending"
    faller_label = "Faller of Week" if has_snapshot else "Movement Pending"
    movement_unit = "spots" if has_snapshot else "no baseline"
    if not has_snapshot:
        riser_text = "—"
        faller_text = "—"

    def record_text(row):
        wins = pd.to_numeric(row.get("current_wins"), errors="coerce")
        losses = pd.to_numeric(row.get("current_losses"), errors="coerce")
        ties = pd.to_numeric(row.get("current_ties"), errors="coerce")

        if pd.isna(wins) or pd.isna(losses):
            return "Record unavailable"

        wins = int(wins)
        losses = int(losses)
        ties = 0 if pd.isna(ties) else int(ties)

        if ties > 0:
            return f"{wins}-{losses}-{ties}"

        return f"{wins}-{losses}"

    best_record = record_text(best)
    worst_record = record_text(worst)

    components.html(
        f"""
<style>
html, body {{
    margin:0;
    padding:0;
    background:transparent;
    overflow:hidden;
    font-family:"Inter", "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, Arial, sans-serif;
}}

/* ── Combined header box ── */
.header-box {{
    position:relative;
    border-radius:20px;
    border:1px solid rgba(255,255,255,.16);
    background:
        radial-gradient(circle at 88% 12%, rgba(241,90,36,.34), transparent 30%),
        radial-gradient(circle at 78% 88%, rgba(0,115,183,.30), transparent 32%),
        linear-gradient(135deg, #07111f, #172033 58%, #263447);
    box-shadow:0 18px 42px rgba(15,23,42,.16);
    overflow:hidden;
}}

.header-box::before {{
    content:"";
    position:absolute;
    top:0;
    left:0;
    right:0;
    height:5px;
    background:linear-gradient(90deg, #F15A24, #0073B7);
    z-index:2;
}}

/* ── Hero section ── */
.hero-section {{
    padding:1.15rem 1.55rem 1rem;
}}

.hero-kicker {{
    color:rgba(255,255,255,.70);
    font-size:.70rem;
    font-weight:850;
    text-transform:uppercase;
    letter-spacing:.15em;
    margin-bottom:.48rem;
}}

.hero-title {{
    color:#fff;
    font-size:2.65rem;
    line-height:.98;
    font-weight:900;
    letter-spacing:-.035em;
    margin-bottom:.56rem;
}}

.hero-copy {{
    max-width:880px;
    color:rgba(255,255,255,.84);
    font-size:.94rem;
    line-height:1.48;
    font-weight:500;
}}

/* ── Divider ── */
.section-divider {{
    height:1px;
    background:rgba(255,255,255,.10);
    margin:0 1.55rem;
}}

/* ── Card row ── */
.status-strip {{
    display:grid;
    grid-template-columns:minmax(500px, .92fr) minmax(650px, 1.08fr);
    gap:1rem;
    align-items:stretch;
    padding:.85rem 1.55rem 1.35rem;
}}

.status-left {{
    display:grid;
    grid-template-columns:140px 140px minmax(250px, 1fr);
    gap:.72rem;
    align-items:stretch;
}}

.status-right {{
    display:grid;
    grid-template-columns:repeat(4, minmax(0, 1fr));
    gap:.72rem;
    align-items:stretch;
}}

/* ── Left cards ── */
.status-card {{
    position:relative;
    z-index:1;
    display:flex;
    flex-direction:column;
    justify-content:center;
    box-sizing:border-box;
    max-width:100%;
    min-width:0;
    min-height:94px;
    padding:.78rem .88rem;
    border-radius:14px;
    background:rgba(255,255,255,.105);
    border:1px solid rgba(255,255,255,.18);
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,.13),
        0 10px 24px rgba(0,0,0,.10);
    backdrop-filter:blur(16px);
}}

.status-live {{
    align-items:flex-start;
}}

.status-label {{
    color:rgba(255,255,255,.66);
    font-size:.64rem;
    font-weight:850;
    text-transform:uppercase;
    letter-spacing:.13em;
    margin-bottom:.36rem;
}}

.status-value {{
    color:#FFFFFF;
    font-size:1.05rem;
    font-weight:850;
    line-height:1.12;
    letter-spacing:-.01em;
}}

.live-time-row {{
    display:flex;
    align-items:center;
    flex-wrap:wrap;
    gap:.5rem;
    box-sizing:border-box;
    width:100%;
    max-width:100%;
    min-width:0;
}}

.live-clock-pill,
.next-refresh-pill {{
    display:inline-flex;
    align-items:center;
    box-sizing:border-box;
    max-width:100%;
    min-width:0;
    min-height:32px;
    padding:0 .86rem;
    border-radius:999px;
    color:#FFFFFF;
    background:rgba(255,255,255,.14);
    border:1px solid rgba(255,255,255,.24);
    box-shadow:0 10px 22px rgba(0,0,0,.16);
    white-space:normal;
    overflow-wrap:anywhere;
    font-size:clamp(.78rem, 1.4vw, .88rem);
    font-weight:850;
}}

.next-refresh-pill {{
    background:linear-gradient(135deg, rgba(241,90,36,.24), rgba(0,115,183,.18));
    border-color:rgba(255,255,255,.28);
}}

/* ── Broadcast-style headline cards ── */
.headline-card {{
    position:relative;
    z-index:1;
    display:grid;
    grid-template-rows:18px 64px minmax(30px, auto) 30px;
    align-items:center;
    justify-items:center;
    text-align:center;
    min-height:132px;
    padding:.76rem .68rem .86rem;
    border-radius:15px;
    background:
        linear-gradient(180deg, rgba(255,255,255,.145), rgba(255,255,255,.075)),
        linear-gradient(135deg, rgba(255,255,255,.04), rgba(255,255,255,.10));
    border:1px solid rgba(255,255,255,.18);
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,.16),
        0 12px 24px rgba(0,0,0,.12);
    backdrop-filter:blur(16px);
    overflow:hidden;
}}

.headline-card::before {{
    content:"";
    position:absolute;
    top:0;
    left:0;
    right:0;
    height:4px;
    border-radius:15px 15px 0 0;
    background:linear-gradient(90deg, var(--card-accent), var(--card-accent2));
    opacity:.95;
}}

.headline-label {{
    color:rgba(255,255,255,.68);
    font-size:.62rem;
    font-weight:850;
    text-transform:uppercase;
    letter-spacing:.13em;
    margin:0;
    align-self:end;
}}

.headline-logo-wrap {{
    width:60px;
    height:60px;
    display:flex;
    align-items:center;
    justify-content:center;
    margin:0 auto;
    flex-shrink:0;
    border-radius:999px;
    background:
        radial-gradient(circle at 30% 22%, rgba(255,255,255,1), rgba(255,255,255,.88) 58%, rgba(226,232,240,.92));
    border:1px solid rgba(255,255,255,.78);
    box-shadow:
        0 10px 20px rgba(0,0,0,.18),
        inset 0 1px 0 rgba(255,255,255,.85);
}}

.headline-logo {{
    display:block;
    max-width:51px;
    max-height:51px;
    object-fit:contain;
    object-position:center center;
    filter:drop-shadow(0 5px 7px rgba(15,23,42,.22)) saturate(1.08) contrast(1.04);
}}

.headline-logo.logo-fallback {{
    width:46px;
    height:46px;
    border-radius:50%;
    background:#07111f;
    color:white;
    font-size:.68rem;
    font-weight:900;
    display:flex;
    align-items:center;
    justify-content:center;
}}

.headline-team {{
    color:#FFFFFF;
    font-size:.82rem;
    font-weight:850;
    line-height:1.16;
    letter-spacing:-.01em;
    max-width:100%;
    min-height:30px;
    display:flex;
    align-items:center;
    justify-content:center;
    text-wrap:balance;
}}

.movement-spacer {{
    min-height:28px;
    width:100%;
}}

.record-badge {{
    display:inline-flex;
    align-items:center;
    justify-content:center;
    min-height:28px;
    padding:0 .62rem;
    border-radius:999px;
    color:#FFFFFF;
    background:rgba(255,255,255,.13);
    border:1px solid rgba(255,255,255,.24);
    box-shadow:
        0 8px 16px rgba(0,0,0,.12),
        inset 0 1px 0 rgba(255,255,255,.18);
    font-size:.76rem;
    font-weight:900;
    letter-spacing:-.01em;
    margin-top:.38rem;
}}

.record-badge span {{
    color:rgba(255,255,255,.68);
    font-size:.62rem;
    font-weight:850;
    text-transform:uppercase;
    letter-spacing:.07em;
    margin-right:.32rem;
}}

.headline-team {{
    color:#FFFFFF;
    font-size:.82rem;
    font-weight:850;
    line-height:1.16;
    letter-spacing:-.01em;
    max-width:100%;
    min-height:1.85em;
    display:flex;
    align-items:center;
    justify-content:center;
    text-wrap:balance;
}}

/* ── Modern movement indicators ── */
.movement-indicator {{
    display:inline-flex;
    align-items:center;
    justify-content:center;
    gap:.28rem;
    min-height:28px;
    padding:.16rem .5rem .16rem .22rem;
    border-radius:999px;
    color:#FFFFFF;
    font-size:.76rem;
    font-weight:900;
    margin-top:.38rem;
    flex-shrink:0;
    box-shadow:
        0 10px 18px rgba(0,0,0,.16),
        inset 0 1px 0 rgba(255,255,255,.20);
}}

.movement-icon {{
    width:22px;
    height:22px;
    display:inline-flex;
    align-items:center;
    justify-content:center;
    border-radius:999px;
    background:rgba(255,255,255,.20);
    font-size:.80rem;
    line-height:1;
}}

.movement-value {{
    font-size:.82rem;
    line-height:1;
    letter-spacing:-.02em;
}}

.movement-label {{
    color:rgba(255,255,255,.76);
    font-size:.62rem;
    font-weight:800;
    text-transform:uppercase;
    letter-spacing:.06em;
}}

.movement-up {{
    background:linear-gradient(135deg, #15803D, #22C55E);
    border:1px solid rgba(187,247,208,.46);
}}

.movement-down {{
    background:linear-gradient(135deg, #B91C1C, #EF4444);
    border:1px solid rgba(254,202,202,.46);
}}

/* ── Responsive ── */
@media(max-width:1180px) {{
    .status-strip {{
        grid-template-columns:1fr;
    }}

    .status-left {{
        grid-template-columns:140px 140px minmax(220px,1fr);
    }}
}}

@media(max-width:760px) {{
    .hero-section {{
        padding:1.1rem 1.15rem .95rem;
    }}

    .hero-title {{
        font-size:2.15rem;
    }}

    .status-strip {{
        grid-template-columns:1fr;
        padding:.85rem 1.15rem 1rem;
    }}

    .section-divider {{
        margin:0 1.15rem;
    }}

    .status-left,
    .status-right {{
        grid-template-columns:1fr 1fr;
    }}

    .live-clock-pill,
    .next-refresh-pill {{
        flex:1 1 100%;
        justify-content:center;
    }}
}}
</style>

<div class="header-box">
    <div class="hero-section">
        <div class="hero-kicker">Live Team Market</div>
        <div class="hero-title">Live Leaderboard</div>
        <div class="hero-copy">A current-moment ranking using Gini performance, roster strength, projected wins, Super Square profile, and weekly movement. During the season, teams enter after their first completed game.</div>
    </div>

    <div class="section-divider"></div>

    <div class="status-strip">
        <div class="status-left">
            <div class="status-card">
                <div class="status-label">Current Season</div>
                <div class="status-value">{escape(selected_season)}</div>
            </div>

            <div class="status-card">
                <div class="status-label">Week / Ranked</div>
                <div class="status-value">{escape(weekly_snapshot_period)} · {len(leaderboard)} teams</div>
            </div>

            <div class="status-card status-live">
                <div class="status-label">Last Page Load</div>
                <div class="live-time-row">
                    <div id="live-clock-text" class="status-value live-clock-pill">{escape(fallback)}</div>
                    <div id="next-refresh-text" class="status-value next-refresh-pill">Next Refresh: {escape(next_refresh_fallback)}</div>
                </div>
            </div>
        </div>

        <div class="status-right">
            <div class="headline-card" style="{card_style(best)}">
                <div class="headline-label">Current Best</div>
                <div class="headline-logo-wrap">{logo_html(best["team"], best.get("team_logo", ""), "headline-logo")}</div>
                <div class="headline-team">{escape(best["team_name"])}</div>
                <div class="record-badge"><span>Record</span>{escape(best_record)}</div>
            </div>

            <div class="headline-card" style="{card_style(worst)}">
                <div class="headline-label">Current Worst</div>
                <div class="headline-logo-wrap">{logo_html(worst["team"], worst.get("team_logo", ""), "headline-logo")}</div>
                <div class="headline-team">{escape(worst["team_name"])}</div>
                <div class="record-badge"><span>Record</span>{escape(worst_record)}</div>
            </div>

            <div class="headline-card" style="{card_style(riser)}">
                <div class="headline-label">{escape(riser_label)}</div>
                <div class="headline-logo-wrap">{logo_html(riser["team"], riser.get("team_logo", ""), "headline-logo")}</div>
                <div class="headline-team">{escape(riser["team_name"])}</div>
                <div class="movement-indicator movement-up">
                    <span class="movement-icon">↗</span>
                    <span class="movement-value">{escape(riser_text)}</span>
                    <span class="movement-label">{escape(movement_unit)}</span>
                </div>
            </div>

            <div class="headline-card" style="{card_style(faller)}">
                <div class="headline-label">{escape(faller_label)}</div>
                <div class="headline-logo-wrap">{logo_html(faller["team"], faller.get("team_logo", ""), "headline-logo")}</div>
                <div class="headline-team">{escape(faller["team_name"])}</div>
                <div class="movement-indicator movement-down">
                    <span class="movement-icon">↘</span>
                    <span class="movement-value">{escape(faller_text)}</span>
                    <span class="movement-label">{escape(movement_unit)}</span>
                </div>
            </div>
        </div>
    </div>
</div>

<script>
(function() {{
  const clockTarget = document.getElementById("live-clock-text");
  const refreshTarget = document.getElementById("next-refresh-text");
  if (!clockTarget || !refreshTarget) {{
    return;
  }}

  function updateClock() {{
    clockTarget.textContent = "{escape(fallback)}";
    refreshTarget.textContent = "Next Refresh: {escape(next_refresh_fallback)}";
    return;
    try {{
      const now = new Date();

      const dateText = new Intl.DateTimeFormat("en-US", {{
        timeZone:"America/New_York",
        month:"short",
        day:"numeric",
        year:"numeric"
      }}).format(now);

      const timeText = new Intl.DateTimeFormat("en-US", {{
        timeZone:"America/New_York",
        hour:"numeric",
        minute:"2-digit",
        second:"2-digit",
        hour12:true
      }}).format(now);

      const easternParts = new Intl.DateTimeFormat("en-US", {{
        timeZone:"America/New_York",
        year:"numeric",
        month:"numeric",
        day:"numeric",
        hour:"2-digit",
        minute:"numeric",
        second:"numeric",
        hourCycle:"h23"
      }}).formatToParts(now).reduce(function(parts, part) {{
        parts[part.type] = part.value;
        return parts;
      }}, {{}});

      const refreshText = "11:59 PM";

      clockTarget.textContent = dateText + " | " + timeText + " ET";
      refreshTarget.textContent = "Next Refresh: " + refreshText + " ET";
    }} catch (err) {{
      clockTarget.textContent = "{escape(fallback)}";
      refreshTarget.textContent = "Next Refresh: {escape(next_refresh_fallback)}";
    }}
  }}

  updateClock();
}})();
</script>
""",
        height=375,
        scrolling=False,
    )


def render_leaderboard(leaderboard):
    rows = []
    for _, row in leaderboard.iterrows():
        team, move = row["team"], row.get("movement_arrow", "\u2192")
        rank_change = row.get("rank_change")
        movement_text = f"{move} {abs(int(0 if pd.isna(rank_change) else rank_change))}"
        score_change = row.get("score_change")
        score_text = "+0.0" if pd.isna(score_change) else f"{score_change:+.1f}"
        move_class = "move-up" if move == "\u2191" else "move-down" if move == "\u2193" else "move-flat"
        rows.append(
            f"""
<div class="leader-row" style="--team-primary:{escape(row.get("team_primary_color", PRIMARY))};--team-secondary:{escape(row.get("team_secondary_color", SECONDARY))};">
  <div class="rank-cell">{int(row["live_rank"])}</div>
  <div class="team-cell"><div class="row-logo-wrap">{logo_html(team, row.get("team_logo", ""), "row-logo")}</div><div><div class="team-name">{escape(row["team_name"])}</div><div class="team-abbr">{escape(team)}</div></div></div>
  <div class="score-cell"><div class="score-value">{number(row["live_market_score"])}</div><div class="score-label">Live Score</div></div>
  <div class="mini-cell"><b>{number(row["current_gini_score"])}</b><span>Gini</span></div>
  <div class="mini-cell"><b>{number(row["roster_score"])}</b><span>Roster</span></div>
  <div class="mini-cell"><b>{number(row["projected_wins"])}</b><span>Wins</span></div>
  <div class="quad-cell"><b>{escape(str(row["most_likely_quadrant"]).replace(" - ", " "))}</b><span>{pct(row["highest_quadrant_probability"])} current profile probability</span></div>
  <div class="move-cell {move_class}"><b>{escape(movement_text)}</b><span>{escape(score_text)}</span></div>
  <div class="status-pill">{escape(row["status_label"])}</div>
</div>
"""
        )
    st.markdown(f'<div class="section-heading">Current Leaderboard · {len(leaderboard)} teams with a completed game</div><div class="leaderboard-wrap">{"".join(rows)}</div>', unsafe_allow_html=True)


def mover_card(row, reason):
    team = row["team"]
    rank_change = row.get("rank_change")
    move = row.get("movement_arrow", "\u2192")
    movement_text = f"{move} {abs(int(0 if pd.isna(rank_change) else rank_change))}"
    score_change = row.get("score_change")
    return f"""
<div class="mover-item"><div class="row-logo-wrap small">{logo_html(team, row.get("team_logo", ""), "row-logo")}</div>
<div class="mover-main"><div class="team-name">{escape(row["team_name"])}</div><div class="mover-reason">{escape(reason)}</div></div>
<div class="mover-stat"><b>{escape(movement_text)}</b><span>{escape("+0.0" if pd.isna(score_change) else f"{score_change:+.1f}")}</span></div></div>
"""


def render_biggest_movers(leaderboard, has_snapshot):
    if has_snapshot:
        risers = leaderboard[pd.to_numeric(leaderboard["rank_change"], errors="coerce") > 0].sort_values("rank_change", ascending=False).head(5)
        fallers = leaderboard[pd.to_numeric(leaderboard["rank_change"], errors="coerce") < 0].sort_values("rank_change").head(5)
    else:
        risers, fallers = leaderboard.head(5), leaderboard.tail(5).sort_values("live_rank")
    if risers.empty:
        risers = leaderboard.head(3)
    if fallers.empty:
        fallers = leaderboard.tail(3)
    riser_html = "".join(mover_card(row, generate_movement_reason(row, has_snapshot)) for _, row in risers.iterrows())
    faller_html = "".join(mover_card(row, generate_movement_reason(row, has_snapshot)) for _, row in fallers.iterrows())
    st.markdown(f'<div class="movers-grid"><div class="market-card"><div class="card-title">Biggest Weekly Risers</div>{riser_html}</div><div class="market-card"><div class="card-title">Biggest Weekly Fallers</div>{faller_html}</div></div>', unsafe_allow_html=True)


def roster_strengths(row):
    metrics = pd.Series(
        {
            "QB Score": row.get("qb_score"),
            "Offensive Line": row.get("offensive_line_score"),
            "Skill Positions": row.get("offense_skill_score"),
            "Defensive Front": row.get("defensive_front_score"),
            "Secondary": row.get("secondary_score"),
            "Premium Position Depth": row.get("premium_position_score"),
            "Availability": row.get("availability_score"),
            "Rookie Projection": row.get("rookie_projection_score"),
            "Experience": row.get("experience_score"),
            "Draft Capital": row.get("draft_capital_score"),
            "Continuity": row.get("continuity_score"),
        },
        dtype="float64",
    ).dropna()
    return (metrics.idxmax(), metrics.idxmin()) if not metrics.empty else ("Roster profile", "Roster profile")


def render_team_detail(leaderboard, production_found):
    labels = [f"#{int(row.live_rank)} {row.team_name} ({row.team})" for row in leaderboard.itertuples()]
    selected_label = st.selectbox("Team Detail", labels, index=0, key="live_leaderboard_team_detail", width="stretch")
    selected_team = selected_label.split("(")[-1].replace(")", "").strip()
    row = leaderboard[leaderboard["team"] == selected_team].iloc[0]
    strength, concern = roster_strengths(row)
    production_note = "Established players are anchored by prior-season NFL production, with current-season production phased in as games accumulate. Draft capital is only a fallback when NFL production is unavailable." if production_found else "Player production stats were not found, so the roster score is currently using draft capital, experience, availability, continuity, and position-depth proxies. Once player stat files are added, the model will shift toward actual player production."
    explanation = f"The {row['team_name']} rank {ordinal(row['live_rank'])} because their Gini profile and roster context combine into a {row['live_market_score']:.1f} Live Market Score. Their projected win range is {int(row['projected_wins_low'])} to {int(row['projected_wins_high'])} wins, and their current profile points most strongly toward {row['most_likely_quadrant']} with a {row['highest_quadrant_probability'] * 100:.0f}% probability. Their roster score is strongest in {strength} and most limited by {concern}. This is a current team-market rating, not a guarantee."
    metric_items = [
        ("Live Market Score", row.get("live_market_score")),
        ("Current Gini Score", row.get("current_gini_score")),
        ("Roster Score", row.get("roster_score")),
        ("Projected Wins", row.get("projected_wins")),
        ("Performance Score", row.get("performance_score")),
        ("QB Score", row.get("qb_score")),
        ("Offensive Line", row.get("offensive_line_score")),
        ("Skill Positions", row.get("offense_skill_score")),
        ("Defensive Front", row.get("defensive_front_score")),
        ("Secondary", row.get("secondary_score")),
        ("Premium Position", row.get("premium_position_score")),
        ("Availability", row.get("availability_score")),
        ("Rookie Projection", row.get("rookie_projection_score")),
        ("Experience", row.get("experience_score")),
        ("Draft Capital", row.get("draft_capital_score")),
        ("Continuity", row.get("continuity_score")),
        ("Transaction Impact", row.get("transaction_impact_score")),
    ]
    metrics_html = "".join(f'<div class="detail-metric"><span>{escape(label)}</span><b>{number(value)}</b></div>' for label, value in metric_items)
    st.markdown(
        f"""
<div class="team-detail-card" style="--team-primary:{escape(row.get("team_primary_color", PRIMARY))};--team-secondary:{escape(row.get("team_secondary_color", SECONDARY))};">
<div class="detail-header"><div class="detail-logo-wrap">{logo_html(row["team"], row.get("team_logo", ""), "detail-logo")}</div><div><div class="detail-kicker">Live Team Market</div><div class="detail-title">#{int(row["live_rank"])} {escape(row["team_name"])}</div><div class="detail-subtitle">{escape(row["status_label"])} | {escape(row["most_likely_quadrant"])} | {pct(row["highest_quadrant_probability"])} current profile probability</div></div></div>
<div class="detail-grid">{metrics_html}</div>
<div class="prob-row"><div><b>Q1</b><span>{pct(row["q1_probability"])}</span></div><div><b>Q2</b><span>{pct(row["q2_probability"])}</span></div><div><b>Q3</b><span>{pct(row["q3_probability"])}</span></div><div><b>Q4</b><span>{pct(row["q4_probability"])}</span></div></div>
<div class="plain-explain">{escape(explanation)}</div><div class="plain-explain muted-explain">{escape(production_note)}</div></div>
""",
        unsafe_allow_html=True,
    )


def extract_refresh_message_field(message, label):
    if not message or label not in message:
        return ""

    tail = str(message).split(label, 1)[1].strip()
    stop_labels = [
        " Daily refresh window:",
        " Next scheduled window:",
        " Last successful refresh:",
        " Last attempt:",
        " Previous result:",
    ]
    stop_positions = [tail.find(stop_label) for stop_label in stop_labels if tail.find(stop_label) > 0]
    if stop_positions:
        tail = tail[: min(stop_positions)].strip()
    return tail.rstrip(".").strip()


def parse_refresh_display_time(value):
    if not value:
        return None

    parsed = parse_et_timestamp(value)
    if parsed is not None:
        return parsed

    text = str(value).strip()
    if text.lower() in {"none recorded", "not recorded", "unknown"}:
        return None

    for suffix in (" ET", "ET"):
        if text.endswith(suffix):
            text = text[: -len(suffix)].strip()
            break

    for fmt in ("%Y-%m-%d %I:%M %p", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=ET)
        except ValueError:
            pass

    return None


def format_refresh_display_time(value):
    parsed = parse_refresh_display_time(value)
    if parsed is None:
        return "Not recorded"
    return format_et_timestamp(parsed, include_seconds=False)


def summarize_refresh_reason(message, status):
    text = str(message or status.get("reason", "") or "").strip()
    lower = text.lower()

    if "already fresh" in lower or "are fresh for the" in lower:
        return "Live source CSVs are already fresh for the latest daily window."
    if "already attempted" in lower:
        return "Refresh already ran or was attempted for the latest daily window."
    if "manual refresh forced" in lower:
        return "Manual refresh was requested."
    if "no previous successful" in lower:
        return "No previous successful refresh was recorded."
    if "no cacheable files were updated" in lower:
        return "Refresh ran, but no cacheable source files were updated."
    if "nflreadpy could not be imported" in lower:
        return "Refresh could not run because nflreadpy is unavailable."
    if status.get("reason"):
        return str(status.get("reason"))
    return text or "Refresh status unavailable."


def refresh_detail_rows(live_source_refresh_ok, live_source_refresh_message):
    status = read_live_source_refresh_status()
    message = str(live_source_refresh_message or status.get("message", "") or "")

    action = "Not attempted"
    lower_message = message.lower()
    if lower_message.startswith("skipped") or "skipped live source refresh" in lower_message:
        action = "Skipped"
    elif live_source_refresh_ok:
        action = "Refreshed"
    elif "failed" in lower_message or "no cacheable files were updated" in lower_message:
        action = "Failed"

    latest_window = (
        extract_refresh_message_field(message, "Daily refresh window:")
        or status.get("daily_refresh_window_label")
        or status.get("daily_refresh_due_time")
        or ""
    )
    next_window = (
        extract_refresh_message_field(message, "Next scheduled window:")
        or status.get("next_daily_refresh_due_time")
        or ""
    )
    last_success = (
        extract_refresh_message_field(message, "Last successful refresh:")
        or status.get("last_successful_refresh_time")
        or ""
    )
    last_attempt = (
        extract_refresh_message_field(message, "Last attempt:")
        or status.get("refresh_time")
        or ""
    )

    current_season = pd.to_numeric(status.get("current_season"), errors="coerce")
    source_notes = []
    source_labels = {
        "weekly_rosters": "Weekly rosters",
        "player_season_stats": "Player production",
        "player_prior_season_stats": "Prior-season player production",
        "player_snap_counts": "Snap counts",
        "injuries": "Injuries",
        "draft_picks": "Draft picks",
        "depth_charts": "Depth charts",
        "schedules": "Schedules/results",
    }
    for source_key, result in (status.get("sources") or {}).items():
        if not isinstance(result, dict):
            continue
        label = source_labels.get(source_key, str(source_key).replace("_", " ").title())
        source_season = pd.to_numeric(result.get("season"), errors="coerce")
        if not result.get("ok"):
            source_notes.append(f"{label}: refresh failed; prior local data retained.")
        elif pd.notna(current_season) and pd.notna(source_season) and int(source_season) != int(current_season):
            source_notes.append(
                f"{label}: {int(source_season)} fallback retained because {int(current_season)} data was unavailable."
            )
    source_note = " ".join(source_notes) or "All season-specific source files were refreshed for the current season."

    rows = [
        ("Refresh Status", action),
        ("Reason", summarize_refresh_reason(message, status)),
        ("Latest Window", format_refresh_display_time(latest_window)),
        ("Next Window", format_refresh_display_time(next_window)),
        ("Last Successful Refresh", format_refresh_display_time(last_success)),
        ("Last Attempt", format_refresh_display_time(last_attempt)),
        ("Data Through", f"Week {status.get('data_through_week')}" if status.get("data_through_week") is not None else "Not recorded"),
        ("Latest Completed Game", str(status.get("latest_completed_game_date") or "Not recorded")),
        ("Completed Regular-Season Games", str(status.get("completed_regular_season_game_count", "Not recorded"))),
        ("PBP Rows", str(status.get("pbp_row_count", "Not recorded"))),
        ("Source Notes", source_note),
    ]

    technical = {
        "display_message": message,
        "status_file_message": status.get("message", ""),
        "status_reason": status.get("reason", ""),
        "refresh_time": status.get("refresh_time", ""),
        "last_successful_refresh_time": status.get("last_successful_refresh_time", ""),
        "daily_refresh_window_label": status.get("daily_refresh_window_label", ""),
        "next_daily_refresh_due_time": status.get("next_daily_refresh_due_time", ""),
    }
    return rows, technical


def render_roster_tracker_plan(
    roster_file,
    load_time,
    future_status,
    production_found,
    snapshot_exists,
    live_source_refresh_ok=False,
    live_source_refresh_message="Live source refresh status unavailable.",
):
    refresh_rows, technical_refresh = refresh_detail_rows(
        live_source_refresh_ok,
        live_source_refresh_message,
    )

    refresh_rows_html = "".join(
        [
            (
                '<div class="refresh-detail-row">'
                f'<div class="refresh-detail-label">{escape(label)}</div>'
                f'<div class="refresh-detail-value">{escape(value)}</div>'
                '</div>'
            )
            for label, value in refresh_rows
        ]
    )

    source_rows = [
        ("Player production", "Yes" if production_found else "No"),
        ("Injuries", "Yes" if future_status.get("injuries") else "No"),
        ("Draft picks", "Yes" if future_status.get("draft_picks") else "No"),
        ("Contracts", "Yes" if future_status.get("contracts") else "No"),
        ("Depth charts", "Yes" if future_status.get("depth_charts") else "No"),
        ("Snapshot file", "Yes" if snapshot_exists else "No"),
    ]

    source_rows_html = "".join(
        [
            (
                '<div class="source-found-row">'
                f'<span>{escape(label)}</span>'
                f'<b class="{"source-found-yes" if value == "Yes" else "source-found-no"}">{escape(value)}</b>'
                '</div>'
            )
            for label, value in source_rows
        ]
    )

    roster_file_display = Path(str(roster_file)).name if roster_file else "No roster file found"

    refresh_card_html = (
        '<div class="tracker-plan">'
        '<div class="refresh-detail-card">'
        '<div class="refresh-card-title">Live Source Refresh</div>'
        f'<div class="refresh-detail-grid">{refresh_rows_html}</div>'
        '<div class="refresh-short-note">Unattended refreshes need an external scheduler while Streamlit is closed.</div>'
        '</div>'
        '</div>'
    )

    source_card_html = (
        '<div class="source-files-card">'
        '<div>'
        '<div class="refresh-card-title">Source Files Found</div>'
        f'<div class="refresh-card-subtitle">Roster source: {escape(roster_file_display)}</div>'
        f'<div class="refresh-card-subtitle">Last loaded: {escape(format_et_timestamp(load_time, include_seconds=False))}</div>'
        '</div>'
        f'<div class="source-found-grid">{source_rows_html}</div>'
        '</div>'
    )

    with st.expander("Roster + Data Source Details", expanded=False):
        st.markdown(refresh_card_html, unsafe_allow_html=True)
        st.markdown(source_card_html, unsafe_allow_html=True)

def render_source_status_card(nflreadpy_available, future_sources, roster_file, production_found):
    source_text = "nflreadpy available." if nflreadpy_available else "nflreadpy not installed. Using local CSV files."
    layer_text = "Player production layer active" if production_found else "Roster proxy layer active"
    used_files = [
        "data/team_season_estat.csv",
        "data/team_game_estat.csv",
        "data/games_2005_onward.csv",
        f"data/{roster_file}",
        "data/teams_colors_logos.csv",
    ]
    for key in ["player_weekly_stats", "player_season_stats", "player_prior_season_stats", "player_snap_counts", "injuries", "transactions", "draft_picks", "contracts", "depth_charts"]:
        info = future_sources.get(key, {})
        if info.get("exists"):
            path = Path(info.get("path", ""))
            try:
                used_files.append(str(path.relative_to(APP_DIR)).replace("\\", "/"))
            except Exception:
                used_files.append(path.name)
    files_html = "".join(f"<span>{escape(path)}</span>" for path in used_files)
    st.markdown(
        f"""
<div class="source-status-card">
    <div>
        <div class="source-status-title">Free Data Source Status</div>
        <div class="source-status-main">{escape(source_text)}</div>
        <div class="source-status-sub">{escape(layer_text)}</div>
    </div>
    <div class="source-file-list">{files_html}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_free_data_source_plan(selected_season, nflreadpy_available):
    with st.expander("Free Live Data Source Plan", expanded=False):
        st.markdown(
            f"""
<div class="tracker-plan">
<p><b>Current version:</b> This page uses local CSV files first so the app stays stable without internet access.</p>
<p><b>Future free source:</b> nflverse through <code>nflreadpy</code> when it is installed and online loading is available.</p>
<p><b>Roster updates:</b> Roster data and weekly rosters can refresh into <code>data/live_sources/</code>, then the page can read the cached files before the older local CSV fallbacks.</p>
<p><b>Game and player updates:</b> Schedules/game data can update during the season. Player stats and Next Gen Stats can update after games or nightly when available.</p>
<p><b>Availability and starter weighting:</b> Injury data and weekly rosters can support the availability score. Snap counts can help weight starters more than backups.</p>
<p><b>No paid API:</b> No API keys, paid endpoints, or scraping are required.</p>
<p><b>Status:</b> {"nflreadpy available." if nflreadpy_available else "nflreadpy not installed. Using local CSV files."}</p>
</div>
""",
            unsafe_allow_html=True,
        )
        if st.button("Refresh Local nflverse Cache", key="refresh_nflverse_cache", use_container_width=True):
            ok, message = run_live_source_refresh_if_needed(int(selected_season), force=True)
            if ok:
                st.success(message)
                st.cache_data.clear()
            else:
                st.info(message)


def render_css():
    st.markdown(
        f"""
<style>
.stApp,[data-testid="stAppViewContainer"]{{
    background:
        radial-gradient(circle at 8% 8%, rgba(241,90,36,.16), transparent 28%),
        radial-gradient(circle at 92% 12%, rgba(0,115,183,.16), transparent 30%),
        radial-gradient(circle at 50% 92%, rgba(0,115,183,.10), transparent 34%),
        linear-gradient(180deg, #F7FAFD 0%, #EEF4F8 48%, #F8FAFC 100%) !important;
    color:{TEXT};
}}

[data-testid="stAppViewContainer"]::before{{
    content:"";
    position:fixed;
    inset:0;
    pointer-events:none;
    z-index:0;
    background:
        linear-gradient(120deg, rgba(255,255,255,.40), transparent 35%, rgba(255,255,255,.28) 65%, transparent 100%);
    opacity:.55;
}}

[data-testid="stHeader"]{{
    background:transparent!important;
}}

.block-container{{
    position:relative;
    z-index:1;
}}
[data-testid="stSidebar"],[data-testid="collapsedControl"]{{display:none!important;}}
.block-container{{
    position:relative;
    z-index:1;
    padding-top:0!important;
    padding-bottom:2.5rem!important;
}}
.live-page{{max-width:1520px;margin:0 auto;padding:.35rem 1rem 2.5rem;}}
.leader-hero{{position:relative;overflow:hidden;margin:.9rem 0 1rem;padding:2rem;border-radius:20px;border:1px solid rgba(255,255,255,.16);background:radial-gradient(circle at 88% 12%,rgba(241,90,36,.34),transparent 30%),radial-gradient(circle at 78% 88%,rgba(0,115,183,.30),transparent 32%),linear-gradient(135deg,#07111f,#172033 58%,#263447);box-shadow:0 24px 52px rgba(15,23,42,.20);}}
.leader-hero:before{{content:"";position:absolute;top:0;left:0;right:0;height:5px;background:linear-gradient(90deg,{PRIMARY},{SECONDARY});}}
.hero-kicker{{color:rgba(255,255,255,.70);font-size:.78rem;font-weight:950;text-transform:uppercase;letter-spacing:.08em;margin-bottom:.55rem;}}
.hero-title{{color:#fff;font-size:clamp(2.3rem,4.5vw,3.65rem);line-height:1;font-weight:950;margin-bottom:.72rem;}}
.hero-copy{{max-width:920px;color:rgba(255,255,255,.86);font-size:1.02rem;line-height:1.66;}}
.market-pills{{display:flex;flex-wrap:wrap;gap:.55rem;margin-top:1rem;}}
.market-pills span{{display:inline-flex;align-items:center;min-height:30px;padding:.4rem .72rem;border-radius:999px;background:rgba(255,255,255,.13);border:1px solid rgba(255,255,255,.22);color:rgba(255,255,255,.88);font-size:.82rem;font-weight:850;}}
.current-season-card{{
    min-height:62px;
    display:flex;
    align-items:center;
    justify-content:flex-start;
    gap:1rem;
    padding:.85rem 1rem;
    border-radius:16px;
    background:rgba(255,255,255,.86);
    border:1px solid rgba(15,23,42,.10);
    box-shadow:0 12px 28px rgba(15,23,42,.06);
    backdrop-filter:blur(14px);
    margin:.65rem 0 .75rem;
}}

.current-season-left{{
    display:flex;
    align-items:center;
    gap:2.6rem;
    min-width:0;
    flex-wrap:wrap;
}}

.current-season-value{{
    color:{TEXT};
    font-size:1rem;
    font-weight:950;
}}
.section-heading{{color:{TEXT};font-size:1.32rem;font-weight:950;margin:1.1rem 0 .7rem;}}
.section-heading:after{{content:"";display:block;width:58px;height:4px;border-radius:999px;margin-top:.45rem;background:linear-gradient(90deg,{PRIMARY},{SECONDARY});}}
.leaderboard-wrap{{display:grid;gap:.55rem;}}
.leader-row{{display:grid;grid-template-columns:48px minmax(220px,1.6fr) 112px 88px 88px 88px minmax(220px,1.25fr) 82px 126px;gap:.7rem;align-items:center;min-height:76px;padding:.7rem .78rem;border-radius:16px;background:rgba(255,255,255,.92);border:1px solid rgba(15,23,42,.10);border-left:5px solid var(--team-primary);box-shadow:0 10px 24px rgba(15,23,42,.055);}}
.rank-cell{{display:flex;align-items:center;justify-content:center;width:42px;height:42px;border-radius:12px;color:white;background:linear-gradient(135deg,var(--team-primary),var(--team-secondary));font-size:1.05rem;font-weight:950;}}
.team-cell,.detail-header,.mover-item{{display:flex;align-items:center;gap:.72rem;min-width:0;}}
.row-logo-wrap{{width:48px;height:48px;flex:0 0 48px;display:flex;align-items:center;justify-content:center;}}
.row-logo-wrap.small{{width:42px;height:42px;flex-basis:42px;}}
.row-logo,.team-logo{{display:block;max-width:46px;max-height:46px;object-fit:contain;object-position:center center;filter:drop-shadow(0 6px 10px rgba(15,23,42,.14));}}
.logo-fallback{{width:42px;height:42px;border-radius:50%;background:{NAVY};color:white;font-size:.75rem;font-weight:950;display:flex;align-items:center;justify-content:center;}}
.team-name{{color:{TEXT};font-size:.98rem;line-height:1.15;font-weight:950;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
.team-abbr,.score-label,.mini-cell span,.quad-cell span,.move-cell span,.mover-reason,.detail-subtitle{{color:{MUTED};font-size:.76rem;line-height:1.25;font-weight:750;}}
.score-value{{color:{TEXT};font-size:1.38rem;line-height:1;font-weight:950;}}
.mini-cell b,.quad-cell b,.move-cell b{{display:block;color:{TEXT};font-size:.96rem;line-height:1.15;font-weight:950;}}
.quad-cell b{{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
.move-up b{{color:#15803D;}}.move-down b{{color:#DC2626;}}.move-flat b{{color:#475569;}}
.status-pill{{justify-self:start;display:inline-flex;align-items:center;justify-content:center;min-height:30px;padding:.35rem .62rem;border-radius:999px;background:rgba(15,23,42,.06);color:{TEXT};font-size:.78rem;font-weight:950;white-space:nowrap;}}
.movers-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1rem;margin:1rem 0 1.25rem;}}
.market-card,.team-detail-card{{padding:1rem;border-radius:16px;background:rgba(255,255,255,.88);border:1px solid rgba(15,23,42,.10);box-shadow:0 12px 28px rgba(15,23,42,.065);}}
.card-title{{color:{TEXT};font-size:1rem;font-weight:950;margin-bottom:.75rem;}}
.mover-item{{padding:.62rem 0;border-top:1px solid rgba(15,23,42,.08);}}
.mover-main{{flex:1 1 auto;min-width:0;}}.mover-stat{{min-width:58px;text-align:right;}}.mover-stat b{{display:block;color:{TEXT};font-weight:950;}}.mover-stat span{{color:{MUTED};font-size:.76rem;font-weight:800;}}
.team-detail-card{{border-left:5px solid var(--team-primary);margin:.7rem 0 1.1rem;}}
.detail-logo-wrap{{width:76px;height:76px;flex:0 0 76px;display:flex;align-items:center;justify-content:center;}}.detail-logo{{display:block;max-width:72px;max-height:72px;object-fit:contain;object-position:center center;}}
.detail-kicker{{color:{MUTED};font-size:.74rem;text-transform:uppercase;letter-spacing:.08em;font-weight:950;margin-bottom:.22rem;}}.detail-title{{color:{TEXT};font-size:1.45rem;line-height:1.12;font-weight:950;}}
.detail-grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.58rem;margin-top:1rem;}}.detail-metric{{min-height:68px;padding:.68rem .72rem;border-radius:12px;background:rgba(248,250,252,.92);border:1px solid rgba(15,23,42,.08);}}.detail-metric span{{display:block;color:{MUTED};font-size:.72rem;font-weight:900;line-height:1.2;}}.detail-metric b{{display:block;color:{TEXT};font-size:1.12rem;line-height:1.2;font-weight:950;margin-top:.25rem;}}
.prob-row{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.55rem;margin-top:.85rem;}}.prob-row div{{padding:.65rem;border-radius:12px;background:linear-gradient(135deg,rgba(7,17,31,.92),rgba(0,115,183,.76));color:white;}}.prob-row b,.prob-row span{{display:block;}}.prob-row b{{font-size:.78rem;font-weight:950;}}.prob-row span{{font-size:1.2rem;font-weight:950;margin-top:.18rem;}}
.plain-explain{{margin-top:.9rem;color:#334155;font-size:.93rem;line-height:1.62;font-weight:700;}}.muted-explain{{color:{MUTED};font-weight:650;}}.tracker-plan p{{color:#334155;font-size:.94rem;line-height:1.6;margin:0 0 .55rem;}}
.refresh-detail-card,.source-files-card{{
    margin:.8rem 0 1rem;
    padding:1rem;
    border-radius:16px;
    background:rgba(255,255,255,.90);
    border:1px solid rgba(15,23,42,.10);
    border-left:5px solid {PRIMARY};
    box-shadow:0 12px 28px rgba(15,23,42,.06);
    backdrop-filter:blur(14px);
}}
.source-files-card{{
    display:grid;
    grid-template-columns:minmax(220px,.45fr) minmax(0,1fr);
    gap:1rem;
    border-left-color:{SECONDARY};
}}
.refresh-card-title{{
    color:{TEXT};
    font-size:1rem;
    font-weight:950;
    margin-bottom:.7rem;
}}
.refresh-card-subtitle{{
    color:#475569;
    font-size:.82rem;
    line-height:1.35;
    font-weight:800;
    margin:.18rem 0;
}}
.refresh-detail-grid{{
    display:grid;
    grid-template-columns:repeat(2,minmax(0,1fr));
    gap:.55rem;
}}
.refresh-detail-row,.source-found-row{{
    min-height:58px;
    padding:.66rem .72rem;
    border-radius:12px;
    background:rgba(248,250,252,.92);
    border:1px solid rgba(15,23,42,.08);
}}
.refresh-detail-label{{
    color:{MUTED};
    font-size:.70rem;
    font-weight:950;
    text-transform:uppercase;
    letter-spacing:.07em;
    margin-bottom:.18rem;
}}
.refresh-detail-value{{
    color:{TEXT};
    font-size:.92rem;
    line-height:1.35;
    font-weight:900;
}}
.refresh-short-note{{
    margin-top:.72rem;
    color:#475569;
    font-size:.82rem;
    font-weight:800;
}}
.source-found-grid{{
    display:grid;
    grid-template-columns:repeat(2,minmax(0,1fr));
    gap:.5rem;
}}
.source-found-row{{
    min-height:42px;
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:.7rem;
}}
.source-found-row span{{
    color:#334155;
    font-size:.86rem;
    font-weight:850;
}}
.source-found-row b{{
    display:inline-flex;
    align-items:center;
    justify-content:center;
    min-width:44px;
    min-height:24px;
    border-radius:999px;
    font-size:.74rem;
    font-weight:950;
}}
.source-found-yes{{
    color:#166534;
    background:rgba(22,101,52,.10);
    border:1px solid rgba(22,101,52,.18);
}}
.source-found-no{{
    color:#991B1B;
    background:rgba(153,27,27,.09);
    border:1px solid rgba(153,27,27,.16);
}}
.technical-refresh-log{{
    margin:.25rem 0 1rem;
    border-radius:14px;
    border:1px solid rgba(15,23,42,.10);
    background:rgba(255,255,255,.78);
    overflow:hidden;
}}
.technical-refresh-log summary{{
    cursor:pointer;
    padding:.72rem .86rem;
    color:{TEXT};
    font-size:.88rem;
    font-weight:950;
}}
.technical-refresh-log pre{{
    margin:0;
    padding:.8rem .9rem;
    max-height:220px;
    overflow:auto;
    white-space:pre-wrap;
    word-break:break-word;
    border-top:1px solid rgba(15,23,42,.08);
    background:#F8FAFC;
    color:#334155;
    font-size:.76rem;
    line-height:1.45;
}}
.source-status-card{{display:grid;grid-template-columns:minmax(260px,.55fr) minmax(0,1fr);gap:1rem;align-items:start;margin:.8rem 0 1rem;padding:1rem;border-radius:16px;background:rgba(255,255,255,.90);border:1px solid rgba(15,23,42,.10);border-left:5px solid {SECONDARY};box-shadow:0 12px 28px rgba(15,23,42,.06);}}
.source-status-title{{color:{MUTED};font-size:.74rem;font-weight:950;text-transform:uppercase;letter-spacing:.08em;margin-bottom:.25rem;}}
.source-status-main{{color:{TEXT};font-size:1.05rem;font-weight:950;line-height:1.2;}}
.source-status-sub{{color:{PRIMARY};font-size:.86rem;font-weight:900;margin-top:.25rem;}}
.source-file-list{{display:flex;flex-wrap:wrap;gap:.42rem;justify-content:flex-end;}}
.source-file-list span{{display:inline-flex;align-items:center;min-height:28px;padding:.32rem .55rem;border-radius:999px;background:#F8FAFC;border:1px solid rgba(15,23,42,.08);color:#334155;font-size:.76rem;font-weight:800;}}
.snapshot-status-card{{
    min-height:56px;
    padding:.78rem 1rem;
    border-radius:16px;
    background:rgba(255,255,255,.86);
    border:1px solid rgba(15,23,42,.10);
    box-shadow:0 12px 28px rgba(15,23,42,.06);
    backdrop-filter:blur(14px);
}}

.snapshot-status-label{{
    color:{MUTED};
    font-size:.72rem;
    font-weight:950;
    text-transform:uppercase;
    letter-spacing:.08em;
}}

.snapshot-status-value{{
    color:{TEXT};
    font-size:1.05rem;
    font-weight:950;
    margin-top:.1rem;
}}

.snapshot-status-meta{{
    color:#475569;
    font-size:.78rem;
    font-weight:800;
    margin-top:.05rem;
}}
.live-page{{
    max-width:1520px;
    margin:-3.35rem auto 0;
    padding:0 1rem 2.5rem;
}}

.leader-hero{{
    position:relative;
    overflow:hidden;
    margin:0;
    padding:1.45rem 1.65rem;
    border-radius:20px 20px 0 0;
    border:1px solid rgba(255,255,255,.16);
    border-bottom:0;
    background:
        radial-gradient(circle at 88% 12%,rgba(241,90,36,.34),transparent 30%),
        radial-gradient(circle at 78% 88%,rgba(0,115,183,.30),transparent 32%),
        linear-gradient(135deg,#07111f,#172033 58%,#263447);
    box-shadow:0 18px 42px rgba(15,23,42,.16);
}}

.hero-kicker{{
    color:rgba(255,255,255,.72);
    font-size:.72rem;
    font-weight:950;
    text-transform:uppercase;
    letter-spacing:.13em;
    margin-bottom:.6rem;
}}

.hero-title{{
    color:#fff;
    font-size:clamp(2.1rem,3.8vw,3.1rem);
    line-height:1;
    font-weight:950;
    margin-bottom:.65rem;
}}

.hero-copy{{
    max-width:860px;
    color:rgba(255,255,255,.84);
    font-size:.95rem;
    line-height:1.55;
}}

.market-pills{{
    display:flex;
    flex-wrap:wrap;
    gap:.5rem;
    margin-top:1.05rem;
    align-items:center;
}}

.market-pills span{{
    display:inline-flex;
    align-items:center;
    justify-content:center;
    min-height:30px;
    padding:0 13px;
    border-radius:999px;
    background:rgba(255,255,255,.13);
    border:1px solid rgba(255,255,255,.20);
    color:#fff;
    font-size:.76rem;
    font-weight:900;
    white-space:nowrap;
}}

.market-pills .live-time-pill{{
    background:rgba(0,115,183,.24);
    border-color:rgba(255,255,255,.28);
    min-height:44px;
    padding:0 1.15rem;
    font-size:.96rem;
    letter-spacing:0;
    box-shadow:0 10px 24px rgba(0,0,0,.12);
}}

.current-season-card{{
    min-height:56px;
    display:flex;
    align-items:center;
    justify-content:flex-start;
    gap:1rem;
    padding:.78rem 1rem;
    border-radius:16px;
    background:rgba(255,255,255,.86);
    border:1px solid rgba(15,23,42,.10);
    box-shadow:0 12px 28px rgba(15,23,42,.06);
    backdrop-filter:blur(14px);
}}

.compact-season-card{{
    max-width:520px;
}}

.current-season-left{{
    display:flex;
    align-items:center;
    gap:3.5rem;
    min-width:0;
}}

.current-season-label{{
    color:{MUTED};
    font-size:.72rem;
    font-weight:950;
    text-transform:uppercase;
    letter-spacing:.08em;
}}

.current-season-value{{
    color:{TEXT};
    font-size:1.05rem;
    font-weight:950;
}}

.current-season-meta{{
    color:#475569;
    font-size:.82rem;
    font-weight:800;
}}

div[data-testid="stSelectbox"] label p{{color:{TEXT}!important;font-size:.94rem!important;font-weight:900!important;}}div[data-testid="stSelectbox"] [data-baseweb="select"]>div{{background-color:white!important;border:1px solid #D1D5DB!important;border-radius:10px!important;box-shadow:0 6px 16px rgba(15,23,42,.045)!important;}}
@media(max-width:1100px){{.leader-row{{grid-template-columns:44px minmax(190px,1fr) 92px 74px 74px 74px;}}.quad-cell,.move-cell,.status-pill{{grid-column:span 2;}}.detail-grid{{grid-template-columns:repeat(2,minmax(0,1fr));}}}}
@media(max-width:780px){{
    .live-page{{margin:-1.05rem auto 0;padding:0 .25rem 2rem;}}
    .leader-hero{{padding:1.25rem 1rem;border-radius:16px 16px 0 0;}}
    .hero-title{{font-size:clamp(1.85rem,10vw,2.45rem);}}
    .hero-copy{{font-size:.9rem;}}
    .market-pills .live-time-pill{{width:100%;white-space:normal;text-align:center;line-height:1.25;padding:.65rem .85rem;}}
    .current-season-left{{gap:1rem;flex-wrap:wrap;}}
    .leader-row{{grid-template-columns:42px minmax(0,1fr);gap:.55rem;border-radius:14px;}}
    .team-name{{white-space:normal;}}
    .score-cell,.mini-cell,.quad-cell,.move-cell,.status-pill{{grid-column:2;}}
    .movers-grid,.detail-grid,.prob-row,.source-status-card,.source-files-card,.refresh-detail-grid,.source-found-grid{{grid-template-columns:1fr;}}
    .source-file-list{{justify-content:flex-start;}}
}}
</style>
""",
        unsafe_allow_html=True,
    )


render_css()
render_top_nav("Live Leaderboard", PRIMARY, SECONDARY)

# The application entry point owns the daily due-check and browser reload so the
# same authoritative pipeline serves every page without page-specific downloads.

file_signature = live_leaderboard_file_signature()
initial_data, _, _, _, _, _, _ = load_data(file_signature)
initial_team_season = initial_data.get("team_season", pd.DataFrame())
initial_season_rosters = initial_data.get("season_rosters", {})

recorded_refresh_status = read_live_source_refresh_status()
live_source_refresh_ok = bool(recorded_refresh_status.get("ok", False))
live_source_refresh_message = str(
    recorded_refresh_status.get("message", "No authoritative current-season refresh is recorded.")
)

payload = build_live_leaderboard_payload(file_signature)
print(f"Live Leaderboard payload timings (cached or rebuilt): {payload.get('build_timings', {})}")

leaderboard = payload["leaderboard"]
selected_season = payload["selected_season"]
weekly_snapshot_key = payload["weekly_snapshot_key"]
weekly_snapshot_period = payload["weekly_snapshot_period"]
roster_file = payload["roster_file"]
production_found = payload["production_found"]
future_status = payload["future_status"]
future_sources = payload["future_sources"]
load_messages = payload["load_messages"]
load_time = payload["load_time"]
source_mtime = payload["source_mtime"]
projected_meta = payload["projected_meta"]
has_snapshot = payload["has_snapshot"]
nflreadpy_available = is_nflreadpy_available()

for message in load_messages:
    st.warning(message)

if payload.get("team_season_empty") or leaderboard.empty:
    st.error("Live Leaderboard needs team_season_estat.csv to build the Gini foundation.")
    st.stop()

weekly_snapshot_saved, weekly_snapshot_key = save_weekly_snapshot_if_needed(leaderboard, weekly_snapshot_key)
weekly_movement_snapshot, has_snapshot = get_latest_weekly_movement_snapshot(weekly_snapshot_key)
if has_snapshot and not comparable_ranked_populations(leaderboard, weekly_movement_snapshot):
    weekly_movement_snapshot = pd.DataFrame()
    has_snapshot = False
leaderboard = calculate_weekly_movement(leaderboard, weekly_movement_snapshot)
leaderboard["status_label"] = leaderboard.apply(generate_status_label, axis=1)
save_live_snapshot(leaderboard, weekly_snapshot_key)

st.markdown('<div class="live-page">', unsafe_allow_html=True)

render_start = perf_counter()

render_live_status_strip(
    selected_season,
    weekly_snapshot_period,
    now_et(),
    leaderboard,
    has_snapshot,
)

# Keep technical data-source details out of the public top section.
# They are shown later in the bottom details expander.
# render_source_status_card(nflreadpy_available, future_sources, roster_file, production_found)

render_leaderboard(leaderboard)
render_biggest_movers(leaderboard, has_snapshot)
st.markdown('<div class="section-heading">Team Detail</div>', unsafe_allow_html=True)
render_team_detail(leaderboard, production_found)
render_roster_tracker_plan(
    roster_file,
    load_time,
    future_status,
    production_found,
    SNAPSHOT_PATH.exists(),
    live_source_refresh_ok=live_source_refresh_ok,
    live_source_refresh_message=live_source_refresh_message,
)
# render_free_data_source_plan(performance_season, nflreadpy_available)
st.markdown("</div>", unsafe_allow_html=True)
print(f"Live Leaderboard render timing: {round(perf_counter() - render_start, 4)}s")
