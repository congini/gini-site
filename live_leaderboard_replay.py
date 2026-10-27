"""Cut off local inputs for a reproducible historical Live Leaderboard replay.

Replays use the current scoring methodology and revised local source data. They
are reconstructed estimates, never presented as rankings captured at the time.
"""

from __future__ import annotations

import pandas as pd

from gini_metrics import apply_default_gini_scores, finalize_schedule_adjusted_gini
from season_utils import normalize_team


RECONSTRUCTION_VERSION = "weekly-replay-v1"


def rebuild_week_metrics(team_game, season, week):
    """Recompute opponent adjustments and Gini using only games through week.

Stored adjusted EPA includes the latest opponent season averages, so merely
filtering those adjusted columns would leak later games into earlier weeks.
This follows the authoritative builder's aggregation and scoring functions.
"""
    required = {
        "season", "season_type", "week", "game_id", "team", "opponent",
        "points_for", "points_against", "off_epa_per_play",
        "def_epa_allowed_per_play", "off_success_rate", "def_success_allowed",
        "takeaways", "turnovers", "penalty_yards_margin",
    }
    if not required.issubset(team_game.columns):
        raise ValueError(f"Missing game inputs: {sorted(required - set(team_game.columns))}")
    seasons = pd.to_numeric(team_game["season"], errors="coerce")
    weeks = pd.to_numeric(team_game["week"], errors="coerce")
    games = team_game[
        seasons.eq(season) & weeks.le(week)
        & team_game["season_type"].astype(str).str.upper().eq("REG")
    ].copy()
    games = games.dropna(subset=["points_for", "points_against"])
    if games.empty or games.duplicated(["game_id", "team"]).any():
        raise ValueError("Replay requires unique completed team-game rows.")
    if not pd.to_numeric(games["week"]).eq(week).any():
        raise ValueError(f"No completed games are available for Week {week}.")
    for column in ("team", "opponent"):
        games[column] = games[column].map(normalize_team)

    units = games.groupby(["season", "team"], as_index=False).agg(
        opp_season_off_epa_per_play=("off_epa_per_play", "mean"),
        opp_season_def_epa_allowed_per_play=("def_epa_allowed_per_play", "mean"),
    ).rename(columns={"team": "opponent"})
    games = games.drop(columns=["opp_season_off_epa_per_play", "opp_season_def_epa_allowed_per_play"], errors="ignore")
    games = games.merge(units, on=["season", "opponent"], how="left", validate="many_to_one")
    games["off_adj_epa"] = games["off_epa_per_play"] - games["opp_season_def_epa_allowed_per_play"]
    games["def_adj_epa"] = games["opp_season_off_epa_per_play"] - games["def_epa_allowed_per_play"]
    games["point_diff"] = games["points_for"] - games["points_against"]
    games["turnover_margin"] = games["takeaways"] - games["turnovers"]
    games["success_margin"] = games["off_success_rate"] - games["def_success_allowed"]
    games["estat_game"] = (
        .40 * games["off_adj_epa"].fillna(0) + .40 * games["def_adj_epa"].fillna(0)
        + .10 * games["success_margin"].fillna(0) + .07 * games["turnover_margin"].fillna(0)
        + .03 * games["penalty_yards_margin"].fillna(0) / 50
    )
    aggregations = {
        "games": ("game_id", "nunique"),
        **{column: (column, "sum") for column in [
            "points_for", "points_against", "point_diff", "turnover_margin", "penalty_yards_margin",
        ]},
        **{column: (column, "mean") for column in [
            "off_epa_per_play", "def_epa_allowed_per_play", "off_adj_epa", "def_adj_epa",
            "off_success_rate", "def_success_allowed",
        ]},
        "estat_raw": ("estat_game", "mean"),
    }
    for column in ["off_plays", "off_epa_total", "yards_gained", "pass_attempts", "rush_attempts",
                   "turnovers", "yards_allowed", "takeaways", "penalties_committed", "penalty_yards_committed"]:
        if column in games:
            aggregations[column] = (column, "sum")
    if "pass_rate" in games:
        aggregations["pass_rate"] = ("pass_rate", "mean")
    summary = games.groupby(["season", "team"], as_index=False).agg(**aggregations)
    for column in ["point_diff", "turnover_margin", "penalty_yards_margin"]:
        summary[f"{column}_per_game"] = summary[column] / summary["games"]
    summary["success_margin"] = summary["off_success_rate"] - summary["def_success_allowed"]
    if {"yards_gained", "off_plays"}.issubset(summary):
        summary["yards_per_play"] = summary["yards_gained"] / summary["off_plays"].replace(0, float("nan"))
    summary = finalize_schedule_adjusted_gini(apply_default_gini_scores(summary), games)
    return games, summary


def restrict_week_rows(frame, season, week, *, require_week=False):
    """Keep prior seasons and current-season rows no later than the cutoff."""
    if frame is None or frame.empty:
        return pd.DataFrame() if frame is None else frame.copy()
    if "season" not in frame:
        raise ValueError("A historical weekly input must identify its season.")
    seasons = pd.to_numeric(frame["season"], errors="coerce")
    if "week" not in frame:
        if require_week and seasons.eq(season).any():
            raise ValueError("Current-season aggregate has no week cutoff; cannot replay it safely.")
        return frame[seasons.lt(season)].copy()
    weeks = pd.to_numeric(frame["week"], errors="coerce")
    return frame[seasons.lt(season) | (seasons.eq(season) & weeks.le(week))].copy()


def prepare_replay_inputs(data, future_data, season, week):
    """Copy and restrict every time-varying scoring source before page scoring."""
    games, summary = rebuild_week_metrics(data["team_game"], season, week)
    output, context = dict(data), dict(future_data)
    prior_games = data["team_game"][pd.to_numeric(data["team_game"]["season"]).lt(season)].copy()
    output["team_game"] = pd.concat([prior_games, games], ignore_index=True)
    prior_summary = data["team_season"][pd.to_numeric(data["team_season"]["season"]).lt(season)].copy()
    output["team_season"] = pd.concat([prior_summary, summary], ignore_index=True)
    for key in ["games"]:
        output[key] = restrict_week_rows(data[key], season, week)
    for key in ["schedules", "weekly_rosters", "injuries", "player_weekly_stats", "player_season_stats", "player_snap_counts"]:
        context[key] = restrict_week_rows(future_data.get(key, pd.DataFrame()), season, week, require_week=True)

    # Use the latest available roster week per team, including teams on a bye.
    roster = context["weekly_rosters"]
    if roster.empty:
        raise ValueError("Historical weekly rosters are required for a replay.")
    roster = roster[pd.to_numeric(roster["season"]).eq(season)].copy()
    roster = roster[pd.to_numeric(roster["week"]).eq(roster.groupby("team")["week"].transform("max"))].copy()
    expected = set(summary["team"])
    if set(roster["team"].map(normalize_team)) != expected:
        raise ValueError("Historical rosters do not cover every ranked team.")
    context["weekly_rosters"] = roster
    sources = dict(data.get("season_rosters", {}))
    sources[season] = {"df": roster, "label": f"weekly_rosters through Week {week}"}
    output["season_rosters"] = sources
    historical = data.get("historical_roster", pd.DataFrame())
    if not historical.empty:
        output["historical_roster"] = historical[pd.to_numeric(historical["season"]).lt(season)].copy()

    # Prior production must actually be from a prior season. The fixed preseason
    # projection is read by the existing app, then updated using cut-off records.
    context["player_prior_season_stats"] = restrict_week_rows(
        future_data.get("player_prior_season_stats", pd.DataFrame()), season, 0
    )
    date_column = "game_date" if "game_date" in games else "gameday"
    final_date = pd.to_datetime(games[date_column], errors="coerce", utc=True).max().date()
    cutoff = (pd.Timestamp(final_date, tz="America/New_York") + pd.Timedelta(days=1)
              - pd.Timedelta(nanoseconds=1)).tz_convert("UTC")
    depth = future_data.get("depth_charts", pd.DataFrame()).copy()
    if not depth.empty:
        date_col = next((c for c in ["dt", "date", "depth_chart_date"] if c in depth), None)
        if date_col is None:
            raise ValueError("Depth chart has no historical timestamp.")
        dates = pd.to_datetime(depth[date_col], errors="coerce", utc=True)
        depth = depth[dates.le(cutoff)].copy()
    context["depth_charts"] = depth
    # Transactions without dates cannot be assigned to a historical week.
    transactions = future_data.get("transactions", pd.DataFrame()).copy()
    if not transactions.empty:
        date_col = next((c for c in ["date", "transaction_date", "dt"] if c in transactions), None)
        if date_col is None:
            raise ValueError("Transactions have no historical timestamp.")
        transactions = transactions[pd.to_datetime(transactions[date_col], errors="coerce", utc=True).le(cutoff)].copy()
    context["transactions"] = transactions
    metadata = {
        "snapshot_kind": "reconstructed", "reconstruction_version": RECONSTRUCTION_VERSION,
        "data_through_week": int(week), "data_cutoff": cutoff.isoformat(),
        "game_count": int(games["game_id"].nunique()),
        "roster_rows": len(roster), "depth_rows": len(depth),
        "method": "Current Live Leaderboard formula, week-limited local data, current frozen preseason reference",
    }
    return output, context, metadata
