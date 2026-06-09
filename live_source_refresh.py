from datetime import datetime, time, timedelta, timezone
import importlib
import json
from pathlib import Path

import pandas as pd

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
LIVE_SOURCES_DIR = DATA_DIR / "live_sources"
LIVE_SOURCE_REFRESH_LOG_PATH = LIVE_SOURCES_DIR / "last_live_source_refresh.txt"
LIVE_SOURCE_REFRESH_STATUS_PATH = LIVE_SOURCES_DIR / "last_live_source_refresh_status.json"
DAILY_REFRESH_TIME = time(23, 59)
REFRESH_PIPELINE_VERSION = 2

LIVE_SOURCE_CACHE_FILES = {
    "player_weekly_stats": LIVE_SOURCES_DIR / "player_weekly_stats.csv",
    "player_season_stats": LIVE_SOURCES_DIR / "player_season_stats.csv",
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

REQUIRED_LIVE_SOURCE_KEYS = (
    "injuries",
    "player_snap_counts",
    "weekly_rosters",
    "player_season_stats",
    "schedules",
    "players",
    "teams",
    "contracts",
    "depth_charts",
    "draft_picks",
)

SEASONAL_FALLBACK_YEAR = datetime.now().year - 1


def eastern_timezone():
    if ZoneInfo is not None:
        try:
            return ZoneInfo("America/New_York")
        except Exception:
            pass
    return timezone(timedelta(hours=-4), name="ET")


ET = eastern_timezone()


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


def _iso_or_empty(dt):
    return dt.isoformat() if dt else ""


def _source_path(key):
    return LIVE_SOURCE_CACHE_FILES[key]


def first_existing_source(key, fallback_path):
    cache_path = LIVE_SOURCE_CACHE_FILES.get(key)
    if cache_path is not None and cache_path.exists():
        return cache_path, "Cached nflverse file"
    return fallback_path, "Local CSV file"


def scheduled_refresh_for_day(day):
    return datetime.combine(day, DAILY_REFRESH_TIME).replace(tzinfo=ET)


def latest_daily_refresh_due_time(moment=None):
    moment = coerce_et(moment or now_et())
    today_due = scheduled_refresh_for_day(moment.date())
    if moment >= today_due:
        return today_due
    return today_due - timedelta(days=1)


def next_daily_refresh_time(moment=None):
    moment = coerce_et(moment or now_et())
    today_due = scheduled_refresh_for_day(moment.date())
    if moment < today_due:
        return today_due
    return today_due + timedelta(days=1)


def format_refresh_window_time(dt):
    dt = coerce_et(dt)
    return dt.strftime("%Y-%m-%d %I:%M %p ET") if dt else "unknown"


def live_source_refresh_window_context(reference_time=None):
    reference_time = coerce_et(reference_time or now_et())
    latest_due = latest_daily_refresh_due_time(reference_time)
    return {
        "reference_time": reference_time,
        "latest_due": latest_due,
        "latest_due_label": format_refresh_window_time(latest_due),
        "next_due": next_daily_refresh_time(reference_time),
        "next_due_label": format_refresh_window_time(next_daily_refresh_time(reference_time)),
    }


def get_live_source_file_state(reference_time=None, freshness_cutoff=None):
    reference_time = coerce_et(reference_time or now_et())
    freshness_cutoff = coerce_et(freshness_cutoff or latest_daily_refresh_due_time(reference_time))
    state = {}

    for key in REQUIRED_LIVE_SOURCE_KEYS:
        path = _source_path(key)
        exists = path.exists()
        modified_time = None
        age_seconds = None

        if exists:
            modified_time = datetime.fromtimestamp(path.stat().st_mtime, ET)
            age_seconds = max(0.0, (reference_time - modified_time).total_seconds())

        state[key] = {
            "path": str(path),
            "file": path.name,
            "exists": exists,
            "modified_time": _iso_or_empty(modified_time),
            "age_seconds": age_seconds,
            "freshness_cutoff": freshness_cutoff.isoformat(),
            "stale": (not exists) or modified_time is None or modified_time < freshness_cutoff,
        }

    return state


def read_live_source_refresh_status():
    if not LIVE_SOURCE_REFRESH_STATUS_PATH.exists():
        return {}

    try:
        status = json.loads(LIVE_SOURCE_REFRESH_STATUS_PATH.read_text(encoding="utf-8"))
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
        return parse_et_timestamp(LIVE_SOURCE_REFRESH_LOG_PATH.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def read_last_live_source_refresh_attempt():
    status = read_live_source_refresh_status()
    return parse_et_timestamp(status.get("refresh_time"))


def should_refresh_live_sources(force=False):
    reference_time = now_et()
    latest_due = latest_daily_refresh_due_time(reference_time)
    file_state = get_live_source_file_state(reference_time, freshness_cutoff=latest_due)
    stale_keys = [key for key, info in file_state.items() if info["stale"]]
    window_label = format_refresh_window_time(latest_due)

    if force:
        return True, f"Manual refresh forced for the {window_label} daily window.", file_state

    status = read_live_source_refresh_status()
    last_success = read_last_successful_live_source_refresh()

    if not stale_keys:
        if last_success is not None and last_success >= latest_due:
            return (
                False,
                f"All required live source CSV files are fresh for the {window_label} daily window; "
                f"last successful refresh was {format_refresh_window_time(last_success)}.",
                file_state,
            )
        return (
            False,
            f"All required live source CSV files are already fresh for the {window_label} daily window; "
            "no additional page-open refresh is needed.",
            file_state,
        )

    last_attempt = read_last_live_source_refresh_attempt()
    attempted_by_current_pipeline = status.get("refresh_pipeline_version") == REFRESH_PIPELINE_VERSION

    if attempted_by_current_pipeline and last_attempt is not None and last_attempt >= latest_due:
        return (
            False,
            f"Live source refresh already attempted for the {window_label} daily window at "
            f"{format_refresh_window_time(last_attempt)}.",
            file_state,
        )

    if last_success is None:
        return True, f"No previous successful live source refresh is recorded for the {window_label} daily window.", file_state

    if last_success < latest_due:
        return (
            True,
            f"Last successful live source refresh ({format_refresh_window_time(last_success)}) is before "
            f"the {window_label} daily window.",
            file_state,
        )

    return True, f"At least one required live source CSV is stale or missing for the {window_label} daily window.", file_state


def is_nflreadpy_available():
    try:
        importlib.import_module("nflreadpy")
        return True
    except Exception:
        return False


def _nflreadpy_module():
    return importlib.import_module("nflreadpy")


def convert_nflreadpy_frame_to_pandas(df):
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

    if hasattr(df, "collect"):
        try:
            collected = df.collect()
            if hasattr(collected, "to_pandas"):
                return collected.to_pandas()
        except Exception:
            pass

    try:
        return pd.DataFrame(df)
    except Exception:
        return pd.DataFrame()


def _call_nflreadpy_loader(nfl, loader_names, seasons=None, kwargs=None):
    kwargs = kwargs or {}
    last_error = None

    for name in loader_names:
        loader = getattr(nfl, name, None)
        if loader is None:
            continue

        try:
            if seasons is None:
                return loader(**kwargs)
            try:
                return loader(seasons=seasons, **kwargs)
            except TypeError:
                return loader(seasons, **kwargs)
        except Exception as exc:
            last_error = exc

    if last_error is not None:
        raise last_error

    raise AttributeError(f"No nflreadpy loader found for: {', '.join(loader_names)}")


def _coerce_selected_season(seasons):
    if seasons is None:
        return datetime.now().year

    if isinstance(seasons, int):
        return seasons

    if isinstance(seasons, (list, tuple, set)):
        values = [int(value) for value in seasons if value is not None]
        return max(values) if values else datetime.now().year

    return int(seasons)


def _season_candidates(selected_season):
    selected_season = int(selected_season)
    fallback = min(selected_season, SEASONAL_FALLBACK_YEAR)
    candidates = [selected_season]

    if fallback not in candidates:
        candidates.append(fallback)

    return candidates


SOURCE_LOADERS = (
    {
        "key": "weekly_rosters",
        "loader_names": ("load_rosters_weekly", "load_weekly_rosters", "import_weekly_rosters", "load_rosters"),
        "seasonal": True,
    },
    {
        "key": "player_season_stats",
        "loader_names": ("load_player_stats", "load_player_stats_seasons", "import_player_stats"),
        "seasonal": True,
        "kwargs": {"summary_level": "week"},
    },
    {
        "key": "players",
        "loader_names": ("load_players", "import_players"),
        "seasonal": False,
    },
    {
        "key": "player_snap_counts",
        "loader_names": ("load_snap_counts", "import_snap_counts"),
        "seasonal": True,
    },
    {
        "key": "injuries",
        "loader_names": ("load_injuries", "import_injuries"),
        "seasonal": True,
    },
    {
        "key": "draft_picks",
        "loader_names": ("load_draft_picks", "import_draft_picks"),
        "seasonal": True,
    },
    {
        "key": "contracts",
        "loader_names": ("load_contracts", "import_contracts"),
        "seasonal": False,
    },
    {
        "key": "depth_charts",
        "loader_names": ("load_depth_charts", "import_depth_charts"),
        "seasonal": True,
    },
    {
        "key": "schedules",
        "loader_names": ("load_schedules", "import_schedules"),
        "seasonal": True,
    },
    {
        "key": "teams",
        "loader_names": ("load_teams", "import_teams"),
        "seasonal": False,
    },
)


def _save_dataframe_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    df.to_csv(temp_path, index=False)
    temp_path.replace(path)


def _refresh_source(nfl, source, selected_season):
    key = source["key"]
    path = _source_path(key)
    loader_names = source["loader_names"]
    kwargs = source.get("kwargs", {})
    attempts = []

    if source.get("seasonal", True):
        season_attempts = _season_candidates(selected_season)
    else:
        season_attempts = [None]

    for season in season_attempts:
        try:
            loader_seasons = [int(season)] if season is not None else None
            raw_df = _call_nflreadpy_loader(nfl, loader_names, seasons=loader_seasons, kwargs=kwargs)
            df = convert_nflreadpy_frame_to_pandas(raw_df)

            if df.empty:
                attempts.append(
                    {
                        "season": season,
                        "ok": False,
                        "error": "loader returned an empty dataframe",
                    }
                )
                continue

            _save_dataframe_csv(df, path)
            modified_time = datetime.fromtimestamp(path.stat().st_mtime, ET)

            return {
                "ok": True,
                "file": path.name,
                "path": str(path),
                "rows": int(len(df)),
                "season": season,
                "modified_time": modified_time.isoformat(),
                "error": "",
                "attempts": attempts,
            }

        except Exception as exc:
            attempts.append(
                {
                    "season": season,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    exact_error = "; ".join(
        f"season {attempt['season']}: {attempt['error']}"
        if attempt.get("season") is not None
        else attempt["error"]
        for attempt in attempts
    )

    return {
        "ok": False,
        "file": path.name,
        "path": str(path),
        "rows": 0,
        "season": None,
        "modified_time": "",
        "error": exact_error or "No loader attempts were made.",
        "attempts": attempts,
    }


def _missing_nflreadpy_results(error):
    return {
        key: {
            "ok": False,
            "file": _source_path(key).name,
            "path": str(_source_path(key)),
            "rows": 0,
            "season": None,
            "modified_time": "",
            "error": error,
            "attempts": [],
        }
        for key in REQUIRED_LIVE_SOURCE_KEYS
    }


def update_local_data_from_nflverse(seasons=None, return_details=False):
    selected_season = _coerce_selected_season(seasons)
    LIVE_SOURCES_DIR.mkdir(parents=True, exist_ok=True)

    try:
        nfl = _nflreadpy_module()
    except Exception as exc:
        message = f"nflreadpy could not be imported: {type(exc).__name__}: {exc}"
        results = _missing_nflreadpy_results(message)
        if return_details:
            return False, message, [], results
        return False, message, []

    source_results = {}
    saved_files = []

    for source in SOURCE_LOADERS:
        result = _refresh_source(nfl, source, selected_season)
        source_results[source["key"]] = result
        if result["ok"]:
            saved_files.append(result["file"])

    errors = [
        f"{key}: {result['error']}"
        for key, result in source_results.items()
        if not result["ok"]
    ]

    if saved_files:
        detail = f"Updated live source CSVs: {', '.join(saved_files)}."
        if errors:
            detail += f" Source errors: {'; '.join(errors)}."
        if return_details:
            return True, detail, saved_files, source_results
        return True, detail, saved_files

    if errors:
        detail = f"nflreadpy ran, but no live source CSVs were saved. Source errors: {'; '.join(errors)}."
        if return_details:
            return False, detail, [], source_results
        return False, detail, []

    detail = "nflreadpy ran, but no public rows were returned. Continuing with local files."
    if return_details:
        return False, detail, [], source_results
    return False, detail, []


def mark_live_sources_refreshed(ok=False, message="", source_results=None, selected_season=None, reason=""):
    LIVE_SOURCES_DIR.mkdir(parents=True, exist_ok=True)
    refresh_time = now_et()
    window_context = live_source_refresh_window_context(refresh_time)
    previous_success = read_last_successful_live_source_refresh()
    last_success = refresh_time if ok else previous_success
    source_results = source_results or {}

    if ok:
        LIVE_SOURCE_REFRESH_LOG_PATH.write_text(refresh_time.isoformat(), encoding="utf-8")

    required_failures = {
        key: result
        for key, result in source_results.items()
        if key in REQUIRED_LIVE_SOURCE_KEYS and not result.get("ok", False)
    }

    status = {
        "refresh_pipeline_version": REFRESH_PIPELINE_VERSION,
        "refresh_time": refresh_time.isoformat(),
        "last_successful_refresh_time": _iso_or_empty(last_success),
        "ok": bool(ok),
        "complete": bool(ok and not required_failures),
        "updated_any": any(result.get("ok", False) for result in source_results.values()),
        "selected_season": int(selected_season) if selected_season is not None else None,
        "attempted": True,
        "skipped": False,
        "daily_refresh_due_time": window_context["latest_due"].isoformat(),
        "daily_refresh_window_label": window_context["latest_due_label"],
        "next_daily_refresh_due_time": window_context["next_due"].isoformat(),
        "message": str(message),
        "reason": str(reason),
        "sources": source_results,
        "freshness": get_live_source_file_state(refresh_time),
    }

    LIVE_SOURCE_REFRESH_STATUS_PATH.write_text(json.dumps(status, indent=2), encoding="utf-8")


def read_last_live_source_refresh_status():
    if not LIVE_SOURCE_REFRESH_STATUS_PATH.exists():
        return False, "No previous live source refresh status found."

    try:
        status = read_live_source_refresh_status()
        ok = bool(status.get("ok", False))
        message = status.get("message", "No refresh message found.")
        refresh_time = status.get("refresh_time", "")
        last_success = status.get("last_successful_refresh_time", "")
        window_label = status.get("daily_refresh_window_label", "")
        next_due = parse_et_timestamp(status.get("next_daily_refresh_due_time"))

        if refresh_time:
            detail = f"{message} Last attempt: {refresh_time}"
            if last_success:
                detail += f" Last successful refresh: {last_success}"
            if window_label:
                detail += f" Daily refresh window: {window_label}"
            if next_due:
                detail += f" Next scheduled window: {format_refresh_window_time(next_due)}"
            return ok, detail

        return ok, message

    except Exception as exc:
        return False, f"Could not read previous live source refresh status: {exc}"


def record_model_recalculation(page_name, message):
    status = read_live_source_refresh_status()
    if not status:
        return

    recalculations = status.get("model_recalculation", {})
    if not isinstance(recalculations, dict):
        recalculations = {}

    recalculations[str(page_name)] = {
        "recalculated_time": now_et().isoformat(),
        "message": str(message),
    }
    status["model_recalculation"] = recalculations
    LIVE_SOURCE_REFRESH_STATUS_PATH.write_text(json.dumps(status, indent=2), encoding="utf-8")


def refresh_live_sources_if_needed(selected_season=None, force=False):
    selected_season = _coerce_selected_season(selected_season)
    should_refresh, reason, _ = should_refresh_live_sources(force=force)
    window_context = live_source_refresh_window_context()

    def status_tail():
        last_success = read_last_successful_live_source_refresh()
        last_attempt = read_last_live_source_refresh_attempt()
        parts = [
            f"Daily refresh window: {window_context['latest_due_label']}.",
            f"Next scheduled window: {window_context['next_due_label']}.",
        ]
        if last_success is None:
            parts.append("Last successful refresh: none recorded.")
        else:
            parts.append(f"Last successful refresh: {format_refresh_window_time(last_success)}.")
        if last_attempt is not None:
            parts.append(f"Last attempt: {format_refresh_window_time(last_attempt)}.")
        return " ".join(parts)

    if not should_refresh:
        last_ok, last_message = read_last_live_source_refresh_status()
        if last_message == "No previous live source refresh status found.":
            return False, f"Skipped live source refresh. {reason} {status_tail()}"
        return False, f"Skipped live source refresh. {reason} {status_tail()} Previous result: {last_message}"

    updated_any, message, saved_files, source_results = update_local_data_from_nflverse(
        selected_season,
        return_details=True,
    )
    required_failures = {
        key: result
        for key, result in source_results.items()
        if key in REQUIRED_LIVE_SOURCE_KEYS and not result.get("ok", False)
    }
    complete = bool(updated_any and not required_failures)

    if updated_any:
        seasons_used = sorted(
            {
                int(result["season"])
                for result in source_results.values()
                if result.get("ok") and result.get("season") is not None
            }
        )
        season_text = ", ".join(str(season) for season in seasons_used) if seasons_used else "static"
        full_message = (
            f"Live sources refreshed for dashboard season {selected_season}; "
            f"nflreadpy source seasons used: {season_text}. {message}"
        )
    else:
        full_message = (
            f"Live source refresh attempted for dashboard season {selected_season}, "
            f"but no cacheable files were updated. {message}"
        )

    mark_live_sources_refreshed(
        ok=complete,
        message=full_message,
        source_results=source_results,
        selected_season=selected_season,
        reason=reason,
    )
    return updated_any, f"{full_message} {status_tail()}"
