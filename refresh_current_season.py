"""One authoritative, fail-safe current-season NFL refresh pipeline."""

from __future__ import annotations

import argparse
import csv
from io import BytesIO
from io import StringIO
import hashlib
import json
import os
from pathlib import Path
from threading import Lock
import uuid

import numpy as np
import pandas as pd
import requests

from gini_metrics import REQUIRED_TEAM_SEASON_COLUMNS, build_team_metrics_from_pbp
from live_source_refresh import (
    LIVE_SOURCE_REFRESH_LOG_PATH,
    LIVE_SOURCE_REFRESH_STATUS_PATH,
    LIVE_SOURCES_DIR,
    REFRESH_PIPELINE_VERSION,
    latest_daily_refresh_due_time,
    next_daily_refresh_time,
    now_et,
    read_live_source_refresh_status,
    update_local_data_from_nflverse,
)
from season_utils import (
    completed_game_summary,
    completed_regular_season_games,
    determine_active_nfl_season,
    regular_season_completion_status,
)


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
TEAM_GAME_PATH = DATA_DIR / "team_game_estat.csv"
TEAM_SEASON_PATH = DATA_DIR / "team_season_estat.csv"
GAMES_PATH = DATA_DIR / "games_2005_onward.csv"
PIPELINE_VERSION = REFRESH_PIPELINE_VERSION
_AUTO_REFRESH_LOCK = Lock()


def _to_pandas(frame):
    if isinstance(frame, pd.DataFrame):
        return frame
    if hasattr(frame, "to_pandas"):
        return frame.to_pandas()
    return pd.DataFrame(frame)


def _download_with_nflreadpy(season):
    import nflreadpy

    pbp = _to_pandas(nflreadpy.load_pbp([int(season)]))
    schedule = _to_pandas(nflreadpy.load_schedules([int(season)]))
    return pbp, schedule, "nflreadpy"


def _download_from_stable_nflverse(season):
    pbp_url = (
        "https://github.com/nflverse/nflverse-data/releases/download/"
        f"pbp/play_by_play_{int(season)}.parquet"
    )
    schedule_url = (
        "https://github.com/nflverse/nflverse-data/releases/download/"
        "schedules/games.parquet"
    )
    headers = {"User-Agent": "Gini-NFL-current-season-refresh/3"}
    pbp_response = requests.get(pbp_url, headers=headers, timeout=90)
    pbp_response.raise_for_status()
    schedule_response = requests.get(schedule_url, headers=headers, timeout=90)
    schedule_response.raise_for_status()
    pbp = pd.read_parquet(BytesIO(pbp_response.content))
    schedule = pd.read_parquet(BytesIO(schedule_response.content))
    schedule = schedule[pd.to_numeric(schedule["season"], errors="coerce").eq(int(season))].copy()
    return pbp, schedule, "nflverse parquet fallback"


def download_current_season_data(season):
    errors = []
    for loader in (_download_with_nflreadpy, _download_from_stable_nflverse):
        try:
            pbp, schedule, source = loader(season)
            if pbp.empty:
                raise ValueError("PBP source returned zero rows.")
            if schedule.empty:
                raise ValueError("Schedule source returned zero rows.")
            return pbp, schedule, source, errors
        except Exception as exc:
            errors.append(f"{loader.__name__}: {type(exc).__name__}: {exc}")
    raise RuntimeError("; ".join(errors))


def _read_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Required working data file is missing: {path}")
    return pd.read_csv(path, low_memory=False)


def replace_season_rows(existing, replacement, season):
    """Replace one season while retaining every historical row and column."""
    if "season" not in existing.columns or "season" not in replacement.columns:
        raise ValueError("Both existing and replacement data must include a season column.")
    historical = existing[~pd.to_numeric(existing["season"], errors="coerce").eq(int(season))].copy()
    aligned = replacement.copy()
    for column in existing.columns:
        if column not in aligned.columns:
            aligned[column] = pd.NA
    extra_columns = [column for column in aligned.columns if column not in existing.columns]
    if extra_columns:
        aligned = aligned.drop(columns=extra_columns)
    aligned = aligned[existing.columns]
    return pd.concat([historical, aligned], ignore_index=True), historical


def _assert_historical_unchanged(before, after, season, label):
    before_hist = before[~pd.to_numeric(before["season"], errors="coerce").eq(int(season))].reset_index(drop=True)
    after_hist = after[~pd.to_numeric(after["season"], errors="coerce").eq(int(season))].reset_index(drop=True)
    pd.testing.assert_frame_equal(before_hist, after_hist, check_dtype=False, check_exact=True, obj=label)


def _validate_current_outputs(
    season,
    pbp,
    schedule,
    team_game,
    team_season,
    merged_team_game,
    merged_team_season,
    merged_games,
    existing_team_game,
    existing_team_season,
    existing_games,
):
    completed = completed_regular_season_games(schedule, season)
    if completed.empty:
        raise ValueError(f"No completed regular-season games were found for {season}.")
    if pbp.empty:
        raise ValueError("PBP validation failed: zero rows.")
    if not pd.to_numeric(pbp.get("season"), errors="coerce").dropna().eq(int(season)).all():
        raise ValueError("PBP validation failed: unexpected season rows were returned.")

    scheduled_teams = set(completed["home_team"]).union(set(completed["away_team"]))
    output_teams = set(team_game["team"].dropna().astype(str))
    if not scheduled_teams.issubset(output_teams):
        raise ValueError(f"Team-game output is missing completed-game teams: {sorted(scheduled_teams - output_teams)}")
    if len(team_game) != len(completed) * 2:
        raise ValueError(f"Expected two team-game rows per completed game; found {len(team_game)} rows for {len(completed)} games.")
    if team_season.empty or team_season["team"].nunique() != len(output_teams):
        raise ValueError("Team-season output does not contain one row for every team with a completed game.")
    if team_season.duplicated(["season", "team"]).any():
        raise ValueError("Duplicate season/team rows were found in team-season output.")
    if team_game.duplicated(["season", "game_id", "team"]).any():
        raise ValueError("Duplicate season/game/team rows were found in team-game output.")

    missing = sorted(REQUIRED_TEAM_SEASON_COLUMNS - set(team_season.columns))
    if missing:
        raise ValueError(f"Team-season output is missing required Gini columns: {missing}")
    finite_columns = [
        "off_adj_epa", "def_adj_epa", "success_margin", "point_diff_per_game",
        "turnover_margin_per_game", "penalty_yards_margin_per_game", "offense_estat",
        "defense_estat", "overall_estat", "schedule_strength",
    ]
    for column in finite_columns:
        values = pd.to_numeric(team_season[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values).all():
            raise ValueError(f"Team-season column {column} contains missing or non-finite values.")
    team_count = len(team_season)
    for column in ["overall_rank", "offense_rank", "defense_rank", "sos_rank"]:
        ranks = pd.to_numeric(team_season[column], errors="coerce")
        if ranks.isna().any() or ranks.lt(1).any() or ranks.gt(team_count).any():
            raise ValueError(f"Invalid values were found in rank column {column}.")

    _assert_historical_unchanged(existing_team_game, merged_team_game, season, "team-game history")
    _assert_historical_unchanged(existing_team_season, merged_team_season, season, "team-season history")
    _assert_historical_unchanged(existing_games, merged_games, season, "schedule history")
    if merged_team_season.duplicated(["season", "team"]).any():
        raise ValueError("Merged team-season output contains duplicate season/team rows.")
    if merged_team_game.duplicated(["season", "game_id", "team"]).any():
        raise ValueError("Merged team-game output contains duplicate season/game/team rows.")


def _atomic_write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temp, path)


def _season_replacement_bytes(path, replacement, season):
    """Preserve non-active-season CSV lines byte-for-byte."""
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig")
    newline = "\r\n" if b"\r\n" in raw else "\n"
    lines = text.splitlines()
    if not lines:
        raise ValueError(f"Cannot preserve history in empty file {path.name}.")
    header = next(csv.reader([lines[0]]))
    if "season" not in header:
        raise ValueError(f"Cannot preserve history in {path.name}: no season column.")
    season_index = header.index("season")
    historical_lines = []
    for line in lines[1:]:
        fields = next(csv.reader([line]))
        if len(fields) <= season_index or str(fields[season_index]).strip() != str(int(season)):
            historical_lines.append(line)

    aligned = replacement.copy()
    for column in header:
        if column not in aligned.columns:
            aligned[column] = pd.NA
    aligned = aligned[header]
    buffer = StringIO()
    aligned.to_csv(buffer, index=False, header=False, lineterminator=newline)
    replacement_text = buffer.getvalue().rstrip("\r\n")
    output_lines = [lines[0], *historical_lines]
    if replacement_text:
        output_lines.extend(replacement_text.splitlines())
    return (newline.join(output_lines) + newline).encode("utf-8")


def _atomic_replace_csvs(outputs):
    staged = {}
    backups = {}
    try:
        for path, payload in outputs.items():
            frame, serialized = payload if isinstance(payload, tuple) else (payload, None)
            temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
            if serialized is None:
                frame.to_csv(temp, index=False)
            else:
                temp.write_bytes(serialized)
            check = pd.read_csv(temp, low_memory=False)
            if len(check) != len(frame) or list(check.columns) != list(frame.columns):
                raise ValueError(f"Staged validation failed for {path.name}.")
            staged[path] = temp
            backups[path] = path.read_bytes() if path.exists() else None

        replaced = []
        try:
            for path, temp in staged.items():
                os.replace(temp, path)
                replaced.append(path)
        except Exception:
            for path in replaced:
                backup = backups[path]
                if backup is None:
                    path.unlink(missing_ok=True)
                else:
                    restore = path.with_name(f".{path.name}.{uuid.uuid4().hex}.restore")
                    restore.write_bytes(backup)
                    os.replace(restore, path)
            raise
    finally:
        for temp in staged.values():
            temp.unlink(missing_ok=True)


def _frame_digest(frame):
    hashed = pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    return hashlib.sha256(hashed).hexdigest()


def should_refresh_current_season(force=False, reference_time=None):
    reference_time = reference_time or now_et()
    due = latest_daily_refresh_due_time(reference_time)
    status = read_live_source_refresh_status()
    if force:
        return True, "Manual refresh forced."
    if status.get("refresh_pipeline_version") != PIPELINE_VERSION:
        return True, "No successful refresh from the authoritative current-season pipeline is recorded."

    attempt = pd.to_datetime(status.get("pipeline_attempt_time") or status.get("refresh_time"), errors="coerce", utc=True)
    success = pd.to_datetime(status.get("last_successful_refresh_time"), errors="coerce", utc=True)
    due_utc = pd.Timestamp(due).tz_convert("UTC")
    if pd.notna(success) and success >= due_utc:
        return False, "The authoritative pipeline already succeeded for the latest daily window."
    if pd.notna(attempt) and attempt >= due_utc:
        return False, "The authoritative pipeline already attempted the latest daily window; its recorded result remains authoritative."
    return True, "The authoritative pipeline has not attempted the latest daily window."


def _base_status(attempt_time, season, previous_status, reason):
    previous_success = (
        previous_status.get("last_successful_refresh_time", "")
        if previous_status.get("refresh_pipeline_version") == PIPELINE_VERSION
        else ""
    )
    return {
        "refresh_pipeline_version": PIPELINE_VERSION,
        "pipeline_attempt_time": attempt_time.isoformat(),
        "refresh_time": attempt_time.isoformat(),
        "last_successful_refresh_time": previous_success,
        "ok": False,
        "complete": False,
        "attempted": True,
        "skipped": False,
        "selected_season": int(season),
        "current_season": int(season),
        "reason": reason,
        "message": "Current-season refresh started.",
        "daily_refresh_due_time": latest_daily_refresh_due_time(attempt_time).isoformat(),
        "next_daily_refresh_due_time": next_daily_refresh_time(attempt_time).isoformat(),
        "steps": {},
        "sources": {},
    }


def refresh_current_season(season=None, force=False, refresh_live_context=True):
    season = int(season or determine_active_nfl_season(now_et()))
    attempt_time = now_et()
    previous_status = read_live_source_refresh_status()
    due, reason = should_refresh_current_season(force=force, reference_time=attempt_time)
    if not due:
        skipped_status = dict(previous_status)
        skipped_status["skipped"] = True
        return False, f"Skipped current-season refresh. {reason}", skipped_status

    status = _base_status(attempt_time, season, previous_status, reason)
    try:
        pbp, schedule, core_source, fallback_errors = download_current_season_data(season)
        status["steps"]["pbp_download"] = {"ok": True, "rows": int(len(pbp)), "source": core_source}
        status["steps"]["schedule_results_download"] = {"ok": True, "rows": int(len(schedule)), "source": core_source}
        if fallback_errors:
            status["steps"]["download_fallback_attempts"] = fallback_errors

        summary = completed_game_summary(schedule, season)
        completion = regular_season_completion_status(schedule, season)
        if summary["completed_regular_season_game_count"] > 0 and pbp.empty:
            raise ValueError("Completed games exist, but the PBP download is empty.")

        team_game_current, team_season_current = build_team_metrics_from_pbp(pbp, schedule, season)
        status["steps"]["gini_team_game_rebuild"] = {"ok": True, "rows": int(len(team_game_current))}
        status["steps"]["gini_team_season_rebuild"] = {"ok": True, "rows": int(len(team_season_current))}

        existing_team_game = _read_csv(TEAM_GAME_PATH)
        existing_team_season = _read_csv(TEAM_SEASON_PATH)
        existing_games = _read_csv(GAMES_PATH)
        merged_team_game, team_game_history = replace_season_rows(existing_team_game, team_game_current, season)
        merged_team_season, team_season_history = replace_season_rows(existing_team_season, team_season_current, season)
        merged_games, games_history = replace_season_rows(existing_games, schedule, season)

        _validate_current_outputs(
            season, pbp, schedule, team_game_current, team_season_current,
            merged_team_game, merged_team_season, merged_games,
            existing_team_game, existing_team_season, existing_games,
        )
        status["steps"]["validation"] = {
            "ok": True,
            "historical_team_game_rows": int(len(team_game_history)),
            "historical_team_season_rows": int(len(team_season_history)),
            "historical_schedule_rows": int(len(games_history)),
            "historical_team_game_digest": _frame_digest(team_game_history),
            "historical_team_season_digest": _frame_digest(team_season_history),
            "historical_schedule_digest": _frame_digest(games_history),
        }

        _atomic_replace_csvs(
            {
                TEAM_GAME_PATH: (
                    merged_team_game,
                    _season_replacement_bytes(TEAM_GAME_PATH, team_game_current, season),
                ),
                TEAM_SEASON_PATH: (
                    merged_team_season,
                    _season_replacement_bytes(TEAM_SEASON_PATH, team_season_current, season),
                ),
                GAMES_PATH: (
                    merged_games,
                    _season_replacement_bytes(GAMES_PATH, schedule, season),
                ),
                LIVE_SOURCES_DIR / "schedules.csv": schedule,
            }
        )
        status["steps"]["atomic_output_replacement"] = {"ok": True}

        context_ok = False
        context_message = "Live roster/context refresh was not requested."
        context_results = {}
        if refresh_live_context:
            context_ok, context_message, _, context_results = update_local_data_from_nflverse(
                season, return_details=True
            )
        status["sources"] = context_results
        status["steps"]["live_roster_context_refresh"] = {
            "ok": bool(context_ok),
            "message": context_message,
        }
        status["steps"]["live_leaderboard_recalculation"] = {
            "ok": True,
            "message": "Authoritative team-season inputs are current; the page cache signature will rebuild on load.",
        }
        status["steps"]["predictive_model_inputs"] = {
            "ok": True,
            "message": "Completed regular-season results and current schedule are available without counting future games.",
        }

        success_time = now_et()
        status.update(summary)
        status.update(
            {
                "last_successful_refresh_time": success_time.isoformat(),
                "ok": True,
                "complete": bool(context_ok or not refresh_live_context),
                "updated_any": True,
                "pbp_row_count": int(len(pbp)),
                "completed_regular_season_game_count": int(summary["completed_regular_season_game_count"]),
                "current_season_team_game_rows": int(len(team_game_current)),
                "current_season_team_season_rows": int(len(team_season_current)),
                "regular_season_complete": bool(completion["complete"]),
                "regular_season_completion_message": completion["reason"],
                "historical_data_unchanged": True,
                "message": (
                    f"Authoritative {season} refresh succeeded through Week {summary['data_through_week']} "
                    f"({summary['completed_regular_season_game_count']} completed regular-season games; "
                    f"{len(pbp)} PBP rows). {context_message}"
                ),
            }
        )
        LIVE_SOURCE_REFRESH_LOG_PATH.write_text(success_time.isoformat(), encoding="utf-8")
        _atomic_write_json(LIVE_SOURCE_REFRESH_STATUS_PATH, status)
        return True, status["message"], status
    except Exception as exc:
        status["ok"] = False
        status["complete"] = False
        status["updated_any"] = False
        status["error"] = f"{type(exc).__name__}: {exc}"
        status["message"] = f"Authoritative {season} refresh failed; previous known-good summary data was retained. {status['error']}"
        _atomic_write_json(LIVE_SOURCE_REFRESH_STATUS_PATH, status)
        return False, status["message"], status


def refresh_current_season_if_due(season=None):
    """Run the authoritative pipeline at most once per due window per process.

    Streamlit sessions can rerun concurrently. The process lock makes the
    metadata freshness check and any resulting refresh one critical section, so
    multiple visitors cannot start duplicate nflverse downloads.
    """
    with _AUTO_REFRESH_LOCK:
        return refresh_current_season(season=season, force=False)


def validate_existing_outputs(season=None):
    season = int(season or determine_active_nfl_season(now_et()))
    team_game = _read_csv(TEAM_GAME_PATH)
    team_season = _read_csv(TEAM_SEASON_PATH)
    games = _read_csv(GAMES_PATH)
    current_games = completed_regular_season_games(games, season)
    current_team_game = team_game[pd.to_numeric(team_game["season"], errors="coerce").eq(season)]
    current_team_season = team_season[pd.to_numeric(team_season["season"], errors="coerce").eq(season)]
    if current_games.empty or current_team_game.empty or current_team_season.empty:
        raise ValueError(f"Existing outputs do not contain completed {season} performance data.")
    if len(current_team_game) != len(current_games) * 2:
        raise ValueError("Existing team-game rows are inconsistent with completed schedule results.")
    if current_team_season.duplicated(["season", "team"]).any():
        raise ValueError("Existing team-season output contains duplicates.")
    return {
        **completed_game_summary(games, season),
        "current_season_team_game_rows": int(len(current_team_game)),
        "current_season_team_season_rows": int(len(current_team_season)),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("season", nargs="?", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-live-context", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.validate_only:
        print(json.dumps(validate_existing_outputs(args.season), indent=2))
        return
    ok, message, status = refresh_current_season(
        args.season,
        force=args.force,
        refresh_live_context=not args.skip_live_context,
    )
    print(message)
    print(json.dumps(status, indent=2))
    if not ok and not status.get("skipped"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
