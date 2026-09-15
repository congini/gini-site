"""Shared NFL season, completion, and pacing helpers."""

from __future__ import annotations

from datetime import datetime

import pandas as pd


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


def normalize_team(value):
    team = "" if pd.isna(value) else str(value).strip().upper()
    return TEAM_ALIASES.get(team, team)


def determine_active_nfl_season(moment=None):
    """Return the season currently being prepared or played.

    January and February still belong to the prior NFL season. From March onward,
    nflverse schedule/roster data is organized under the current calendar year.
    """
    moment = moment or datetime.now()
    return int(moment.year - 1 if moment.month <= 2 else moment.year)


def expected_regular_season_games(season):
    return 17 if int(season) >= 2021 else 16


def _season_rows(schedule, season):
    if schedule is None or schedule.empty:
        return pd.DataFrame()

    rows = schedule.copy()
    season_col = "season" if "season" in rows.columns else "game_season" if "game_season" in rows.columns else None
    if season_col is None:
        return pd.DataFrame()
    rows = rows[pd.to_numeric(rows[season_col], errors="coerce").eq(int(season))].copy()

    type_col = "game_type" if "game_type" in rows.columns else "season_type" if "season_type" in rows.columns else None
    if type_col is not None:
        rows = rows[rows[type_col].astype(str).str.upper().str.startswith("REG")].copy()
    return rows


def completed_regular_season_games(schedule, season):
    """Return only regular-season schedule rows with both final scores present."""
    rows = _season_rows(schedule, season)
    required = {"home_team", "away_team", "home_score", "away_score"}
    if rows.empty or not required.issubset(rows.columns):
        return pd.DataFrame(columns=rows.columns)

    rows["home_score"] = pd.to_numeric(rows["home_score"], errors="coerce")
    rows["away_score"] = pd.to_numeric(rows["away_score"], errors="coerce")
    rows = rows.dropna(subset=["home_team", "away_team", "home_score", "away_score"]).copy()
    rows["home_team"] = rows["home_team"].map(normalize_team)
    rows["away_team"] = rows["away_team"].map(normalize_team)
    if "game_id" in rows.columns:
        rows = rows.drop_duplicates("game_id", keep="last")
    return rows


def regular_season_completion_status(schedule, season):
    """Describe completion from actual schedule/results, never a calendar cutoff."""
    scheduled = _season_rows(schedule, season)
    completed = completed_regular_season_games(schedule, season)
    expected_per_team = expected_regular_season_games(season)

    if scheduled.empty or not {"home_team", "away_team"}.issubset(scheduled.columns):
        return {
            "complete": False,
            "scheduled_games": 0,
            "completed_games": 0,
            "teams": 0,
            "reason": "No regular-season schedule rows found.",
        }

    scheduled = scheduled.copy()
    scheduled["home_team"] = scheduled["home_team"].map(normalize_team)
    scheduled["away_team"] = scheduled["away_team"].map(normalize_team)
    teams = pd.concat([scheduled["home_team"], scheduled["away_team"]], ignore_index=True)
    counts = teams[teams.ne("")].value_counts()
    schedule_complete = (
        len(counts) >= 32
        and not counts.empty
        and int(counts.min()) >= expected_per_team
        and len(completed) == len(scheduled)
    )
    if schedule_complete:
        reason = f"All {len(completed)} regular-season games have final scores."
    elif len(completed) < len(scheduled):
        reason = f"{len(completed)} of {len(scheduled)} regular-season games have final scores."
    else:
        reason = f"The schedule does not contain {expected_per_team} games for every NFL team."

    return {
        "complete": bool(schedule_complete),
        "scheduled_games": int(len(scheduled)),
        "completed_games": int(len(completed)),
        "teams": int(len(counts)),
        "reason": reason,
    }


def completed_game_summary(schedule, season):
    completed = completed_regular_season_games(schedule, season)
    if completed.empty:
        return {
            "data_through_week": None,
            "latest_completed_game_date": None,
            "completed_regular_season_game_count": 0,
        }

    weeks = pd.to_numeric(completed.get("week"), errors="coerce")
    date_col = "gameday" if "gameday" in completed.columns else "game_date" if "game_date" in completed.columns else None
    dates = pd.to_datetime(completed[date_col], errors="coerce") if date_col else pd.Series(dtype="datetime64[ns]")
    return {
        "data_through_week": int(weeks.max()) if weeks.notna().any() else None,
        "latest_completed_game_date": dates.max().date().isoformat() if dates.notna().any() else None,
        "completed_regular_season_game_count": int(len(completed)),
    }


def calculate_regular_season_records(schedule, season=None):
    seasons = [int(season)] if season is not None else sorted(
        pd.to_numeric(schedule.get("season"), errors="coerce").dropna().astype(int).unique()
    )
    rows = []
    for year in seasons:
        completed = completed_regular_season_games(schedule, year)
        for _, game in completed.iterrows():
            home_score = float(game["home_score"])
            away_score = float(game["away_score"])
            for team, points_for, points_against in (
                (game["home_team"], home_score, away_score),
                (game["away_team"], away_score, home_score),
            ):
                rows.append(
                    {
                        "season": year,
                        "team": team,
                        "win": int(points_for > points_against),
                        "loss": int(points_for < points_against),
                        "tie": int(points_for == points_against),
                    }
                )
    if not rows:
        return pd.DataFrame(columns=["season", "team", "current_wins", "current_losses", "current_ties", "scored_games"])
    return pd.DataFrame(rows).groupby(["season", "team"], as_index=False).agg(
        current_wins=("win", "sum"),
        current_losses=("loss", "sum"),
        current_ties=("tie", "sum"),
        scored_games=("win", "size"),
    )


def current_win_pace(wins, games_played, season):
    games_played = float(games_played)
    if games_played <= 0:
        return None
    return (float(wins) / games_played) * expected_regular_season_games(season)


def current_projected_finish(wins, ties, games_played, preseason_projected_wins, season):
    """Blend banked results with the frozen preseason rate for games remaining.

    A straight-line pace is mathematically valid but far too volatile early in a
    season (an 0-1 team becomes a 0-win pace). This outlook gives completed games
    their actual value and applies the original preseason expectation only to the
    unplayed portion of the schedule. Ties count as half a win.
    """
    if preseason_projected_wins is None or pd.isna(preseason_projected_wins):
        return None

    total_games = float(expected_regular_season_games(season))
    played = min(total_games, max(0.0, float(games_played)))
    wins_value = 0.0 if wins is None or pd.isna(wins) else float(wins)
    ties_value = 0.0 if ties is None or pd.isna(ties) else float(ties)
    earned_wins = max(0.0, wins_value) + 0.5 * max(0.0, ties_value)
    earned_wins = min(played, earned_wins)
    preseason_win_rate = min(1.0, max(0.0, float(preseason_projected_wins) / total_games))
    remaining_games = max(0.0, total_games - played)
    return min(total_games, earned_wins + remaining_games * preseason_win_rate)


def select_live_performance_population(performance_all, selected_season, team_universe=None):
    """Select a comparable leaderboard population for the live season.

    Before any current-season game is complete, the latest prior season supplies a
    preseason baseline for every available team. As soon as current-season results
    exist, only teams with completed-game performance rows are ranked; prior-year
    rows are never mixed into an in-season table.
    """
    if performance_all is None or performance_all.empty:
        return pd.DataFrame(columns=getattr(performance_all, "columns", []))

    selected_season = int(selected_season)
    output = performance_all.copy()
    output["season"] = pd.to_numeric(output["season"], errors="coerce")
    universe = set(team_universe or output["team"].dropna().astype(str).tolist())
    current = output[
        output["season"].eq(selected_season) & output["team"].isin(universe)
    ].copy()
    if not current.empty:
        current["performance_source_season"] = selected_season
        return current

    prior = output[
        output["season"].lt(selected_season) & output["team"].isin(universe)
    ].sort_values("season")
    if prior.empty:
        return current
    prior = prior.groupby("team", as_index=False).tail(1).copy()
    prior["performance_source_season"] = prior["season"].astype(int)
    prior["season"] = selected_season
    return prior.reset_index(drop=True)


def comparable_ranked_populations(current, previous):
    """Return whether two ranking frames contain the same non-empty team set."""
    if current is None or previous is None or current.empty or previous.empty:
        return False
    if "team" not in current.columns or "team" not in previous.columns:
        return False
    current_teams = set(current["team"].dropna().map(normalize_team))
    previous_teams = set(previous["team"].dropna().map(normalize_team))
    return bool(current_teams) and current_teams == previous_teams
