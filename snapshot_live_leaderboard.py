"""Freeze a complete NFL week's leaderboard after the authoritative data refresh.

The versioned weekly archive owns rank movement. Daily/browser refreshes never
replace a finalized week, and the refresh baseline is not a weekly snapshot.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from live_leaderboard_weekly import (
    finalizable_week_key,
    finalized_week_snapshot,
    freeze_weekly_snapshot,
    write_weekly_history,
)


APP_DIR = Path(__file__).resolve().parent
GAMES_PATH = APP_DIR / "data" / "games_2005_onward.csv"
WEEKLY_HISTORY_PATH = APP_DIR / "data" / "live_leaderboard_weekly_history.csv"
REFRESH_BASELINE_PATH = APP_DIR / "data" / "live_leaderboard_refresh_baseline.csv"
PAGE_PATH = APP_DIR / "pages" / "Live_Leaderboard.py"


def latest_completed_regular_season_key(games: pd.DataFrame) -> str:
    """Return the season/week key for the latest completed regular-season game."""
    required = {"season", "week", "home_score", "away_score"}
    missing = sorted(required.difference(games.columns))
    if missing:
        raise ValueError(f"Schedule is missing required columns: {', '.join(missing)}")

    completed = games.copy()
    if "game_type" in completed.columns:
        completed = completed[completed["game_type"].astype(str).str.upper().eq("REG")]
    elif "season_type" in completed.columns:
        completed = completed[completed["season_type"].astype(str).str.upper().eq("REG")]

    completed["season"] = pd.to_numeric(completed["season"], errors="coerce")
    completed["week"] = pd.to_numeric(completed["week"], errors="coerce")
    completed["home_score"] = pd.to_numeric(completed["home_score"], errors="coerce")
    completed["away_score"] = pd.to_numeric(completed["away_score"], errors="coerce")
    completed = completed.dropna(subset=["season", "week", "home_score", "away_score"])

    if completed.empty:
        raise ValueError("No completed regular-season games are available for a weekly baseline.")

    season = int(completed["season"].max())
    week = int(completed.loc[completed["season"].eq(season), "week"].max())
    return f"{season}-Week-{week:02d}"


def replace_weekly_snapshot(
    history: pd.DataFrame,
    snapshot: pd.DataFrame,
    snapshot_week: str,
    snapshot_timestamp: str,
) -> pd.DataFrame:
    """Replace one period's history row without leaving duplicate ranks."""
    current = history.copy()
    if not current.empty and "snapshot_week" in current.columns:
        current = current[~current["snapshot_week"].astype(str).eq(snapshot_week)].copy()

    replacement = snapshot.copy()
    replacement["snapshot_week"] = snapshot_week
    replacement["snapshot_timestamp"] = snapshot_timestamp
    replacement["snapshot_window"] = snapshot_timestamp

    preferred_columns = ["snapshot_window", "snapshot_week"] + [
        column for column in replacement.columns if column not in {"snapshot_window", "snapshot_week"}
    ]
    if current.empty:
        all_columns = [column for column in preferred_columns if column != "snapshot_day"]
    else:
        # Existing history owns the schema. Daily-only snapshot metadata must not
        # silently add a new column and rewrite every historical row.
        all_columns = list(current.columns)

    for column in all_columns:
        if column not in current.columns:
            current[column] = pd.NA
        if column not in replacement.columns:
            replacement[column] = pd.NA

    combined = pd.concat(
        [current[all_columns], replacement[all_columns]],
        ignore_index=True,
    )
    return combined


def build_prior_week_snapshot(timeout_seconds: int = 240) -> tuple[str, int]:
    """Compatibility entry point: finalize processed data, never the pre-refresh view."""
    games = pd.read_csv(GAMES_PATH, low_memory=False)
    team_game = pd.read_csv(APP_DIR / "data" / "team_game_estat.csv", low_memory=False)
    moment = datetime.now(timezone.utc)
    snapshot_week = finalizable_week_key(games, team_game, moment)
    if snapshot_week is None:
        return "", 0
    history = (
        pd.read_csv(WEEKLY_HISTORY_PATH, low_memory=False)
        if WEEKLY_HISTORY_PATH.exists()
        else pd.DataFrame()
    )
    existing = finalized_week_snapshot(history, snapshot_week, games)
    if not existing.empty:
        return snapshot_week, len(existing)
    from streamlit.testing.v1 import AppTest
    prior_capture = os.environ.get("LIVE_LEADERBOARD_CAPTURE_ONLY")
    os.environ["LIVE_LEADERBOARD_CAPTURE_ONLY"] = "1"
    try:
        app = AppTest.from_file(str(PAGE_PATH)).run(timeout=timeout_seconds)
    finally:
        if prior_capture is None:
            os.environ.pop("LIVE_LEADERBOARD_CAPTURE_ONLY", None)
        else:
            os.environ["LIVE_LEADERBOARD_CAPTURE_ONLY"] = prior_capture
    if app.exception:
        raise RuntimeError("; ".join(str(error.value) for error in app.exception))
    payload = app.session_state["live_leaderboard_capture_payload"]
    if payload["weekly_snapshot_key"] != snapshot_week:
        # A delayed run must not relabel a later calendar week's roster/model view.
        return "", 0
    snapshot = payload["leaderboard"]
    updated, changed = freeze_weekly_snapshot(history, snapshot, snapshot_week, games, team_game, moment)
    if changed:
        write_weekly_history(updated, WEEKLY_HISTORY_PATH)
    return snapshot_week, len(snapshot)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args()

    snapshot_week, team_count = build_prior_week_snapshot(args.timeout)
    if snapshot_week:
        print(f"Finalized archive: {snapshot_week}, {team_count} teams (existing finalized weeks stay frozen).")
    else:
        print("No NFL week is ready to finalize: waiting for complete, processed results and the 6am ET cutoff.")


if __name__ == "__main__":
    main()
