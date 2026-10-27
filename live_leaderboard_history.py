"""Pure helpers for Live Leaderboard snapshot and team-history views."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


WEEK_KEY_PATTERN = re.compile(r"^(?P<season>\d{4})-Week-(?P<week>\d{1,2})$")


def parse_week_key(value) -> tuple[int, int] | None:
    match = WEEK_KEY_PATTERN.fullmatch(str(value or "").strip())
    if not match:
        return None
    return int(match.group("season")), int(match.group("week"))


def week_label(value) -> str:
    parsed = parse_week_key(value)
    return f"Week {parsed[1]}" if parsed else str(value)


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
