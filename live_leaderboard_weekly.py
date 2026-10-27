"""Frozen NFL-week movement and snapshot persistence.

Kept separate from the long-lived history-view module so an in-place Streamlit
update can import new weekly APIs even while the previous history module is cached.
"""
from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

import pandas as pd

from live_leaderboard_history import parse_week_key

NFL_TEAMS = frozenset("ARI ATL BAL BUF CAR CHI CIN CLE DAL DEN DET GB HOU IND JAX KC LA LAC LV MIA MIN NE NO NYG NYJ PHI PIT SEA SF TB TEN WAS".split())
EASTERN = ZoneInfo("America/New_York")


def _regular_schedule(schedule, season):
    required = {"season", "week", "home_team", "away_team", "home_score", "away_score", "gameday"}
    if schedule is None or not required.issubset(schedule.columns):
        return pd.DataFrame()
    games = schedule[pd.to_numeric(schedule["season"], errors="coerce").eq(season)].copy()
    kind = next((c for c in ("game_type", "season_type") if c in games), None)
    if kind:
        games = games[games[kind].astype(str).str.upper().eq("REG")].copy()
    games["week"] = pd.to_numeric(games["week"], errors="coerce")
    return games


def week_finalization_cutoff(schedule, key):
    """A week may freeze after all scheduled results exist and the next 6am ET processing window."""
    parsed = parse_week_key(key)
    if not parsed:
        return None
    games = _regular_schedule(schedule, parsed[0])
    if games.empty:
        return None
    games = games[games["week"].eq(parsed[1])]
    if games.empty or games[["home_score", "away_score"]].apply(pd.to_numeric, errors="coerce").isna().any().any():
        return None
    dates = pd.to_datetime(games["gameday"], errors="coerce")
    if dates.isna().any():
        return None
    last_day = pd.Timestamp(dates.max().date()).tz_localize(EASTERN)
    return last_day + pd.DateOffset(days=1, hours=6)


def valid_rank_population(frame, season):
    """Require the same complete league, never a partial or duplicated ranking."""
    if frame is None or not {"team", "season", "live_rank"}.issubset(frame.columns):
        return False
    ranks = pd.to_numeric(frame["live_rank"], errors="coerce")
    return (len(frame) == 32 and set(frame["team"]) == NFL_TEAMS
            and not frame["team"].duplicated().any()
            and pd.to_numeric(frame["season"], errors="coerce").eq(season).all()
            and set(ranks) == set(range(1, 33)))


def finalized_week_snapshot(history, key, schedule=None):
    """Read an exact frozen NFL week. Replays and premature legacy captures are not authoritative."""
    parsed = parse_week_key(key)
    if not parsed or history is None or "snapshot_week" not in history:
        return pd.DataFrame()
    rows = history[history["snapshot_week"].astype(str).eq(key)].copy()
    if rows.empty:
        return rows
    if "snapshot_kind" in rows and rows["snapshot_kind"].eq("reconstructed").any():
        return pd.DataFrame()
    states = rows.get("snapshot_status", pd.Series("", index=rows.index)).fillna("")
    explicit = states.eq("finalized")
    if explicit.any():
        rows = rows[explicit].copy()
    elif states.ne("").any():
        return pd.DataFrame()
    if "snapshot_timestamp" not in rows:
        return pd.DataFrame()
    timestamps = pd.to_datetime(rows["snapshot_timestamp"], errors="coerce", utc=True)
    if timestamps.isna().any():
        return pd.DataFrame()
    # Once final, the first finalized capture owns the week permanently.
    chosen_time = timestamps.min() if explicit.any() else timestamps.max()
    rows = rows[timestamps.eq(chosen_time)].copy()
    if not valid_rank_population(rows, parsed[0]):
        return pd.DataFrame()
    if not explicit.any():
        cutoff = week_finalization_cutoff(schedule, key)
        if cutoff is None or chosen_time < cutoff.tz_convert("UTC"):
            return pd.DataFrame()
    return rows


def previous_finalized_week_snapshot(history, current_key, schedule=None):
    parsed = parse_week_key(current_key)
    if not parsed or parsed[1] <= 1:
        return pd.DataFrame()
    return finalized_week_snapshot(history, f"{parsed[0]}-Week-{parsed[1]-1:02d}", schedule)


def calculate_weekly_rank_movement(leaderboard, history, current_key, schedule=None, *, historical=False):
    """Previous NFL week's frozen rank minus this view's rank; missing evidence stays missing."""
    output = leaderboard.copy()
    for column in ("previous_week_rank", "weekly_rank_change"):
        output[column] = pd.Series(pd.NA, index=output.index, dtype="Int64")
    output["weekly_movement_available"] = False
    parsed = parse_week_key(current_key)
    output["weekly_comparison_key"] = f"{parsed[0]}-Week-{parsed[1]-1:02d}" if parsed and parsed[1] > 1 else ""
    if not parsed or not valid_rank_population(output, parsed[0]):
        return output
    if historical and finalized_week_snapshot(history, current_key, schedule).empty:
        return output
    previous = previous_finalized_week_snapshot(history, current_key, schedule)
    if previous.empty:
        return output
    output["previous_week_rank"] = output["team"].map(previous.set_index("team")["live_rank"]).astype("Int64")
    output["weekly_rank_change"] = output["previous_week_rank"] - pd.to_numeric(output["live_rank"]).astype("Int64")
    output["weekly_movement_available"] = True
    return output


def finalizable_week_key(schedule, team_game, moment):
    """Only freeze the latest evidence week after every result has reached the processed team-game data."""
    if schedule is None or schedule.empty or "season" not in schedule:
        return None
    season = int(pd.to_numeric(schedule["season"], errors="coerce").max())
    games = _regular_schedule(schedule, season)
    if games.empty:
        return None
    completed = games.dropna(subset=["home_score", "away_score"])
    if completed.empty:
        return None
    week = int(completed["week"].max())
    key = f"{season}-Week-{week:02d}"
    cutoff = week_finalization_cutoff(schedule, key)
    timestamp = pd.Timestamp(moment)
    timestamp = timestamp.tz_localize(EASTERN) if timestamp.tzinfo is None else timestamp
    if cutoff is None or timestamp < cutoff:
        return None
    required = {"season", "week", "game_id", "team", "points_for", "points_against"}
    if team_game is None or not required.issubset(team_game.columns) or "game_id" not in games:
        return None
    through_week = games[games["week"].le(week)]
    if through_week[["home_score", "away_score"]].isna().any().any():
        return None
    processed = team_game[pd.to_numeric(team_game["season"]).eq(season)].copy()
    if "season_type" in processed:
        processed = processed[processed["season_type"].astype(str).str.upper().eq("REG")]
    if pd.to_numeric(processed["week"]).gt(week).any():
        return None
    expected = {(r.game_id, team) for r in through_week.itertuples() for team in (r.home_team, r.away_team)}
    processed = processed.dropna(subset=["points_for", "points_against"])
    actual = set(zip(processed["game_id"], processed["team"]))
    if actual != expected or processed.duplicated(["game_id", "team"]).any():
        return None
    return key


def freeze_weekly_snapshot(history, snapshot, key, schedule, team_game, moment):
    """Idempotently finalize a complete, processed NFL week; daily reruns cannot replace it."""
    if key != finalizable_week_key(schedule, team_game, moment):
        raise ValueError("Cannot finalize an incomplete or unprocessed NFL week.")
    if not valid_rank_population(snapshot, parse_week_key(key)[0]):
        raise ValueError("A finalized leaderboard requires all 32 teams with unique ranks 1-32.")
    existing = finalized_week_snapshot(history, key, schedule)
    if not existing.empty:
        return history.copy(), False
    replacement = snapshot.copy()
    replacement["snapshot_week"] = key
    replacement["snapshot_timestamp"] = pd.Timestamp(moment).isoformat()
    replacement["snapshot_window"] = replacement["snapshot_timestamp"]
    replacement["snapshot_status"] = "finalized"
    replacement["snapshot_kind"] = "captured"
    replacement["data_through_week"] = parse_week_key(key)[1]
    previous = history[~history["snapshot_week"].astype(str).eq(key)].copy() if "snapshot_week" in history else history.copy()
    return pd.concat([previous, replacement], ignore_index=True, sort=False), True


def write_weekly_history(history, path):
    """Publish the persisted archive atomically; the refresh workflow versions this CSV in Git."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        history.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
