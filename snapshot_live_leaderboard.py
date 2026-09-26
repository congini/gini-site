"""Persist the final leaderboard from the most recently completed NFL week.

The scheduled refresh runs this script before replacing the source data. That
preserves the ranking users actually saw at the end of the prior week, so the
next leaderboard can report true week-over-week rank movement.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from streamlit.testing.v1 import AppTest


APP_DIR = Path(__file__).resolve().parent
GAMES_PATH = APP_DIR / "data" / "games_2005_onward.csv"
WEEKLY_HISTORY_PATH = APP_DIR / "data" / "live_leaderboard_weekly_history.csv"
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
    """Replace one week's baseline so reruns cannot leave duplicate or stale ranks."""
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
    games = pd.read_csv(GAMES_PATH, low_memory=False)
    snapshot_week = latest_completed_regular_season_key(games)

    with TemporaryDirectory(prefix="gini-leaderboard-baseline-") as temp_dir:
        snapshot_path = Path(temp_dir) / "live_leaderboard_snapshot.csv"
        os.environ["LIVE_LEADERBOARD_FORCE_SNAPSHOT"] = "1"
        os.environ["LIVE_LEADERBOARD_SNAPSHOT_PATH"] = str(snapshot_path)
        try:
            app = AppTest.from_file(str(PAGE_PATH))
            app.run(timeout=timeout_seconds)
        finally:
            os.environ.pop("LIVE_LEADERBOARD_FORCE_SNAPSHOT", None)
            os.environ.pop("LIVE_LEADERBOARD_SNAPSHOT_PATH", None)

        if app.exception:
            messages = "; ".join(str(exception.value) for exception in app.exception)
            raise RuntimeError(f"Live Leaderboard snapshot failed: {messages}")
        if not snapshot_path.exists():
            raise RuntimeError("Live Leaderboard did not produce a snapshot file.")

        snapshot = pd.read_csv(snapshot_path, low_memory=False)
    required = {"team", "live_rank", "live_market_score"}
    missing = sorted(required.difference(snapshot.columns))
    if missing:
        raise ValueError(f"Leaderboard snapshot is missing required columns: {', '.join(missing)}")
    if snapshot.empty or snapshot["team"].duplicated().any():
        raise ValueError("Leaderboard snapshot must contain one row per ranked team.")

    timestamp = datetime.now().astimezone().isoformat()
    history = (
        pd.read_csv(WEEKLY_HISTORY_PATH, low_memory=False)
        if WEEKLY_HISTORY_PATH.exists()
        else pd.DataFrame()
    )
    updated = replace_weekly_snapshot(history, snapshot, snapshot_week, timestamp)
    WEEKLY_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    updated.to_csv(WEEKLY_HISTORY_PATH, index=False)
    return snapshot_week, len(snapshot)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args()

    snapshot_week, team_count = build_prior_week_snapshot(args.timeout)
    print(f"Saved {snapshot_week} leaderboard baseline for {team_count} teams.")


if __name__ == "__main__":
    main()
