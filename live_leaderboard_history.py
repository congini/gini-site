"""Pure helpers for Live Leaderboard snapshot and team-history views."""

from __future__ import annotations

import re
import os
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


WEEK_KEY_PATTERN = re.compile(r"^(?P<season>\d{4})-Week-(?P<week>\d{1,2})$")
NFL_TEAMS = frozenset("ARI ATL BAL BUF CAR CHI CIN CLE DAL DEN DET GB HOU IND JAX KC LA LAC LV MIA MIN NE NO NYG NYJ PHI PIT SEA SF TB TEN WAS".split())
EASTERN = ZoneInfo("America/New_York")


def parse_week_key(value) -> tuple[int, int] | None:
    match = WEEK_KEY_PATTERN.fullmatch(str(value or "").strip())
    if not match:
        return None
    return int(match.group("season")), int(match.group("week"))


def week_label(value) -> str:
    parsed = parse_week_key(value)
    return f"Week {parsed[1]}" if parsed else str(value)


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


def available_snapshot_keys(history: pd.DataFrame, season: int) -> list[str]:
    """Return recorded in-season leaderboard snapshots, newest first."""
    if history is None or history.empty or "snapshot_week" not in history.columns:
        return []
    keys = []
    for value in history["snapshot_week"].dropna().astype(str).unique():
        parsed = parse_week_key(value)
        if parsed and parsed[0] == int(season):
            keys.append((parsed[1], value))
    return [value for _, value in sorted(keys, reverse=True)]


def combine_recorded_and_reconstructed(recorded, reconstructed):
    """Fill missing weeks only; captured snapshots always take precedence."""
    captured = recorded.copy()
    captured["snapshot_kind"] = "captured"
    replay = reconstructed.copy()
    if replay.empty:
        return captured
    if "snapshot_week" not in replay:
        raise ValueError("Reconstructed history must identify each snapshot week.")
    captured_keys = set(captured.get("snapshot_week", pd.Series(dtype=str)).astype(str))
    replay = replay[~replay["snapshot_week"].astype(str).isin(captured_keys)].copy()
    replay["snapshot_kind"] = "reconstructed"
    return pd.concat([captured, replay], ignore_index=True, sort=False)


def latest_period_snapshot(history: pd.DataFrame, snapshot_key: str) -> pd.DataFrame:
    """Return one complete, latest-timestamp snapshot for a recorded week."""
    if history is None or history.empty or "snapshot_week" not in history.columns:
        return pd.DataFrame()
    snapshot = history[history["snapshot_week"].astype(str).eq(str(snapshot_key))].copy()
    if snapshot.empty:
        return snapshot
    if "snapshot_timestamp" in snapshot.columns:
        timestamps = pd.to_datetime(snapshot["snapshot_timestamp"], errors="coerce", utc=True)
        if timestamps.notna().any():
            snapshot = snapshot[timestamps.eq(timestamps.max())].copy()
    return snapshot.drop_duplicates("team", keep="last")


def records_through_week(team_game: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """Calculate regular-season records using only games through the selected week."""
    columns = ["team", "current_wins", "current_losses", "current_ties", "scored_games"]
    required = {"season", "week", "team", "points_for", "points_against"}
    if team_game is None or team_game.empty or not required.issubset(team_game.columns):
        return pd.DataFrame(columns=columns)

    games = team_game.copy()
    if "season_type" in games.columns:
        games = games[games["season_type"].astype(str).str.upper().eq("REG")]
    games["season"] = pd.to_numeric(games["season"], errors="coerce")
    games["week"] = pd.to_numeric(games["week"], errors="coerce")
    games["points_for"] = pd.to_numeric(games["points_for"], errors="coerce")
    games["points_against"] = pd.to_numeric(games["points_against"], errors="coerce")
    games = games[
        games["season"].eq(int(season))
        & games["week"].le(int(week))
        & games["points_for"].notna()
        & games["points_against"].notna()
    ].copy()
    if games.empty:
        return pd.DataFrame(columns=columns)

    games["current_wins"] = games["points_for"].gt(games["points_against"]).astype(int)
    games["current_losses"] = games["points_for"].lt(games["points_against"]).astype(int)
    games["current_ties"] = games["points_for"].eq(games["points_against"]).astype(int)
    records = (
        games.groupby("team", as_index=False)
        .agg(
            current_wins=("current_wins", "sum"),
            current_losses=("current_losses", "sum"),
            current_ties=("current_ties", "sum"),
            scored_games=("week", "size"),
        )
    )
    return records[columns]


def snapshot_leaderboard(
    history: pd.DataFrame,
    snapshot_key: str,
    current_leaderboard: pd.DataFrame,
    team_game: pd.DataFrame,
    season: int,
) -> pd.DataFrame:
    """Hydrate a recorded ranking snapshot with current display assets and week-specific records."""
    snapshot = latest_period_snapshot(history, snapshot_key)
    parsed = parse_week_key(snapshot_key)
    if snapshot.empty or not parsed:
        return pd.DataFrame()

    asset_columns = [
        "team",
        "team_name",
        "team_logo",
        "team_primary_color",
        "team_secondary_color",
    ]
    assets = current_leaderboard[[c for c in asset_columns if c in current_leaderboard.columns]].copy()
    assets = assets.drop_duplicates("team", keep="last")
    records = records_through_week(team_game, season, parsed[1])
    # Replay snapshots may already carry records/assets. Always hydrate from the
    # selected week's games and display assets without merge suffix collisions.
    replaced_columns = (set(assets.columns) | set(records.columns)) - {"team"}
    snapshot = snapshot.drop(columns=list(replaced_columns), errors="ignore")
    snapshot = snapshot.merge(assets, on="team", how="left", validate="one_to_one")
    snapshot = snapshot.merge(records, on="team", how="left", validate="one_to_one")
    snapshot["live_rank"] = pd.to_numeric(snapshot["live_rank"], errors="coerce")
    snapshot = snapshot.dropna(subset=["live_rank"]).sort_values("live_rank").reset_index(drop=True)
    snapshot["live_rank"] = snapshot["live_rank"].astype(int)
    snapshot["snapshot_week"] = snapshot_key
    return snapshot


def build_team_history(
    history: pd.DataFrame,
    current_leaderboard: pd.DataFrame,
    team_game: pd.DataFrame,
    season: int,
    current_snapshot_key: str,
) -> pd.DataFrame:
    """Build one row per team and captured week, enriched with that week's opponent and result."""
    periods = []
    for snapshot_key in reversed(available_snapshot_keys(history, season)):
        snapshot = latest_period_snapshot(history, snapshot_key)
        parsed = parse_week_key(snapshot_key)
        if snapshot.empty or not parsed:
            continue
        snapshot = snapshot[[c for c in ["team", "live_rank", "live_market_score", "snapshot_kind"] if c in snapshot.columns]].copy()
        snapshot["snapshot_week"] = snapshot_key
        snapshot["week"] = parsed[1]
        periods.append(snapshot)

    current_parsed = parse_week_key(current_snapshot_key)
    if current_parsed and current_leaderboard is not None and not current_leaderboard.empty:
        current = current_leaderboard[["team", "live_rank", "live_market_score"]].copy()
        current["snapshot_week"] = current_snapshot_key
        current["snapshot_kind"] = "current"
        current["week"] = current_parsed[1]
        periods = [frame for frame in periods if not frame["snapshot_week"].astype(str).eq(current_snapshot_key).any()]
        periods.append(current)

    if not periods:
        return pd.DataFrame(
            columns=["team", "snapshot_week", "week", "live_rank", "live_market_score", "opponent", "result", "score_line", "home_away"]
        )

    output = pd.concat(periods, ignore_index=True)
    output["live_rank"] = pd.to_numeric(output["live_rank"], errors="coerce")
    output["live_market_score"] = pd.to_numeric(output["live_market_score"], errors="coerce")

    game_columns = ["team", "week", "opponent", "points_for", "points_against"]
    team_game_frame = team_game if isinstance(team_game, pd.DataFrame) else pd.DataFrame()
    optional_columns = [column for column in ["home_away", "is_home"] if column in team_game_frame.columns]
    games = team_game_frame.copy()
    if not games.empty and {"season", *game_columns}.issubset(games.columns):
        if "season_type" in games.columns:
            games = games[games["season_type"].astype(str).str.upper().eq("REG")]
        games["season"] = pd.to_numeric(games["season"], errors="coerce")
        games["week"] = pd.to_numeric(games["week"], errors="coerce")
        games = games[games["season"].eq(int(season))]
        games = games[game_columns + optional_columns].drop_duplicates(["team", "week"], keep="last")
        output = output.merge(games, on=["team", "week"], how="left")
    else:
        for column in game_columns[2:] + optional_columns:
            output[column] = pd.NA

    points_for = pd.to_numeric(output.get("points_for"), errors="coerce")
    points_against = pd.to_numeric(output.get("points_against"), errors="coerce")
    output["result"] = np.select(
        [points_for.gt(points_against), points_for.lt(points_against), points_for.eq(points_against)],
        ["W", "L", "T"],
        default="Bye",
    )
    output["score_line"] = [
        f"{int(scored)}-{int(allowed)}" if pd.notna(scored) and pd.notna(allowed) else ""
        for scored, allowed in zip(points_for, points_against)
    ]
    if "home_away" not in output.columns:
        if "is_home" in output.columns:
            output["home_away"] = np.where(output["is_home"].fillna(False), "vs", "at")
        else:
            output["home_away"] = ""
    else:
        output["home_away"] = output["home_away"].astype(str).str.upper().map({"HOME": "vs", "AWAY": "at"}).fillna("")
    output.loc[output["opponent"].isna(), "opponent"] = "BYE"
    output.loc[output["opponent"].eq("BYE"), "home_away"] = ""
    return output.sort_values(["team", "week"]).reset_index(drop=True)
