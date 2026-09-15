"""Authoritative Gini/EStat calculations shared by builders and the dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd

from season_utils import completed_regular_season_games, normalize_team


# These are the active Gini Dashboard defaults. Changing them changes the model.
BASELINE_WEIGHTS = {
    "Offense": 0.30,
    "Defense": 0.30,
    "Point Diff": 0.15,
    "Success Margin": 0.12,
    "Turnovers": 0.06,
    "Penalties": 0.02,
    "Schedule Strength": 0.05,
}

GINI_COMPONENT_COLUMNS = {
    "Offense": "off_z",
    "Defense": "def_z",
    "Point Diff": "pd_z",
    "Success Margin": "success_z",
    "Turnovers": "turnover_z",
    "Penalties": "penalty_z",
    "Schedule Strength": "schedule_z",
}

REQUIRED_TEAM_SEASON_COLUMNS = {
    "season", "team", "games", "off_adj_epa", "def_adj_epa", "success_margin",
    "point_diff_per_game", "turnover_margin_per_game", "penalty_yards_margin_per_game",
    "offense_estat", "defense_estat", "overall_estat", "schedule_strength",
    "overall_rank", "offense_rank", "defense_rank", "sos_rank",
}


def normalize_weights(weights):
    total = float(sum(weights.values()))
    return {key: (float(value) / total if total else 0.0) for key, value in weights.items()}


def recompute_overall(df, weights=None):
    weights = normalize_weights(weights or BASELINE_WEIGHTS)
    score = pd.Series(0.0, index=df.index)
    for label, column in GINI_COMPONENT_COLUMNS.items():
        values = pd.to_numeric(df[column], errors="coerce").fillna(0) if column in df.columns else 0.0
        score = score + weights[label] * values
    return 100 + 15 * score


def zscore_by_season(df, column):
    values = pd.to_numeric(df[column], errors="coerce")
    mean = values.groupby(df["season"]).transform("mean")
    std = values.groupby(df["season"]).transform("std").replace(0, np.nan)
    return ((values - mean) / std).fillna(0)


def apply_default_gini_scores(team_season):
    output = team_season.copy()
    for target, source in (
        ("off_z", "off_adj_epa"),
        ("def_z", "def_adj_epa"),
        ("pd_z", "point_diff_per_game"),
        ("success_z", "success_margin"),
        ("turnover_z", "turnover_margin_per_game"),
        ("penalty_z", "penalty_yards_margin_per_game"),
    ):
        output[target] = zscore_by_season(output, source)

    output["offense_estat"] = 100 + 15 * output["off_z"]
    output["defense_estat"] = 100 + 15 * output["def_z"]

    # First pass supplies opponent strength. The final pass below adds schedule_z.
    output["schedule_z"] = 0.0
    output["overall_estat"] = recompute_overall(output, BASELINE_WEIGHTS)
    return output


def finalize_schedule_adjusted_gini(team_season, team_game):
    output = team_season.copy()
    opponent_scores = output[["season", "team", "overall_estat"]].rename(
        columns={"team": "opponent", "overall_estat": "opponent_overall_estat"}
    )
    games = team_game.merge(opponent_scores, on=["season", "opponent"], how="left")
    sos = games.groupby(["season", "team"], as_index=False).agg(
        schedule_strength=("opponent_overall_estat", "mean")
    )
    output = output.drop(columns=["schedule_strength"], errors="ignore").merge(
        sos, on=["season", "team"], how="left"
    )
    output["schedule_z"] = zscore_by_season(output, "schedule_strength")
    output["schedule_strength_z"] = output["schedule_z"]
    output["overall_estat"] = recompute_overall(output, BASELINE_WEIGHTS)
    output["overall_rank"] = output.groupby("season")["overall_estat"].rank(ascending=False, method="min").astype(int)
    output["offense_rank"] = output.groupby("season")["offense_estat"].rank(ascending=False, method="min").astype(int)
    output["defense_rank"] = output.groupby("season")["defense_estat"].rank(ascending=False, method="min").astype(int)
    output["sos_rank"] = output.groupby("season")["schedule_strength"].rank(ascending=False, method="min").astype(int)
    return output.drop(columns=["schedule_z"], errors="ignore")


def build_team_metrics_from_pbp(pbp, schedule, season):
    """Build active-season team-game and team-season outputs from completed games."""
    season = int(season)
    completed = completed_regular_season_games(schedule, season)
    if completed.empty:
        raise ValueError(f"No completed {season} regular-season games were found.")
    if "game_id" not in completed.columns:
        raise ValueError("Schedule/results data does not include game_id.")

    plays = pbp.copy()
    defaults = {
        "season": season, "season_type": "REG", "week": pd.NA, "game_id": "",
        "game_date": pd.NA, "home_team": "", "away_team": "", "posteam": pd.NA,
        "defteam": pd.NA, "play_type": pd.NA, "epa": 0, "success": 0,
        "yards_gained": 0, "pass_attempt": 0, "rush_attempt": 0,
        "interception": 0, "fumble_lost": 0, "touchdown": 0, "penalty": 0,
        "penalty_team": pd.NA, "penalty_yards": 0, "qb_kneel": 0, "qb_spike": 0,
    }
    for column, default in defaults.items():
        if column not in plays.columns:
            plays[column] = default

    plays["season"] = pd.to_numeric(plays["season"], errors="coerce")
    plays = plays[plays["season"].eq(season)].copy()
    completed_ids = set(completed["game_id"].dropna().astype(str))
    plays = plays[plays["game_id"].astype(str).isin(completed_ids)].copy()
    plays = plays[plays["posteam"].notna() & plays["defteam"].notna()].copy()
    plays = plays[plays["season_type"].astype(str).str.upper().isin(["REG", "POST"])].copy()
    plays = plays[plays["play_type"].isin(["pass", "run", "no_play"])].copy()
    plays = plays[pd.to_numeric(plays["qb_kneel"], errors="coerce").fillna(0).eq(0)]
    plays = plays[pd.to_numeric(plays["qb_spike"], errors="coerce").fillna(0).eq(0)]
    if plays.empty:
        raise ValueError(f"The {season} PBP download has no usable plays for completed games.")

    for column in [
        "epa", "success", "yards_gained", "pass_attempt", "rush_attempt", "interception",
        "fumble_lost", "touchdown", "penalty", "penalty_yards",
    ]:
        plays[column] = pd.to_numeric(plays[column], errors="coerce").fillna(0)
    for column in ["home_team", "away_team", "posteam", "defteam", "penalty_team"]:
        plays[column] = plays[column].map(normalize_team)

    keys = ["season", "season_type", "week", "game_id", "game_date", "home_team", "away_team", "posteam", "defteam"]
    offense = plays.groupby(keys, dropna=False).agg(
        off_plays=("epa", "count"), off_epa_total=("epa", "sum"), off_epa_per_play=("epa", "mean"),
        off_success_rate=("success", "mean"), yards_gained=("yards_gained", "sum"),
        pass_attempts=("pass_attempt", "sum"), rush_attempts=("rush_attempt", "sum"),
        interceptions=("interception", "sum"), fumbles_lost=("fumble_lost", "sum"),
        touchdowns=("touchdown", "sum"),
    ).reset_index()
    offense["turnovers"] = offense["interceptions"] + offense["fumbles_lost"]
    offense = offense.drop(columns=["interceptions", "fumbles_lost"]).rename(columns={"posteam": "team", "defteam": "opponent"})

    defense = plays.groupby(
        ["season", "season_type", "week", "game_id", "defteam", "posteam"], dropna=False
    ).agg(
        def_plays=("epa", "count"), def_epa_allowed_total=("epa", "sum"),
        def_epa_allowed_per_play=("epa", "mean"), def_success_allowed=("success", "mean"),
        yards_allowed=("yards_gained", "sum"), takeaways_int=("interception", "sum"),
        takeaways_fumble=("fumble_lost", "sum"),
    ).reset_index().rename(columns={"defteam": "team", "posteam": "opponent"})
    defense["takeaways"] = defense["takeaways_int"] + defense["takeaways_fumble"]
    defense = defense.drop(columns=["takeaways_int", "takeaways_fumble"])

    penalties = plays[plays["penalty_team"].ne("")].groupby(
        ["season", "week", "game_id", "penalty_team"], dropna=False
    ).agg(
        penalties_committed=("penalty", "sum"), penalty_yards_committed=("penalty_yards", "sum")
    ).reset_index().rename(columns={"penalty_team": "team"})

    team_game = offense.merge(
        defense, on=["season", "season_type", "week", "game_id", "team", "opponent"], how="left"
    ).merge(penalties, on=["season", "week", "game_id", "team"], how="left")
    team_game[["penalties_committed", "penalty_yards_committed"]] = team_game[
        ["penalties_committed", "penalty_yards_committed"]
    ].fillna(0)
    team_game["is_home"] = team_game["team"].eq(team_game["home_team"]).astype(int)
    team_game["home_away"] = np.where(team_game["is_home"].eq(1), "Home", "Away")
    team_game["pass_rate"] = team_game["pass_attempts"] / (team_game["pass_attempts"] + team_game["rush_attempts"]).replace(0, np.nan)

    home_scores = completed[["game_id", "home_team", "home_score", "away_score"]].rename(
        columns={"home_team": "team", "home_score": "points_for", "away_score": "points_against"}
    )
    away_scores = completed[["game_id", "away_team", "away_score", "home_score"]].rename(
        columns={"away_team": "team", "away_score": "points_for", "home_score": "points_against"}
    )
    team_game = team_game.merge(pd.concat([home_scores, away_scores], ignore_index=True), on=["game_id", "team"], how="left")
    if team_game[["points_for", "points_against"]].isna().any().any():
        raise ValueError("At least one PBP team-game row could not be matched to a completed final score.")

    season_team = team_game.groupby(["season", "team"], as_index=False).agg(
        season_off_epa_per_play=("off_epa_per_play", "mean"),
        season_def_epa_allowed_per_play=("def_epa_allowed_per_play", "mean"),
    )
    opponent = season_team.rename(columns={
        "team": "opponent", "season_off_epa_per_play": "opp_season_off_epa_per_play",
        "season_def_epa_allowed_per_play": "opp_season_def_epa_allowed_per_play",
    })
    team_game = team_game.merge(opponent, on=["season", "opponent"], how="left")
    team_game["off_adj_epa"] = team_game["off_epa_per_play"] - team_game["opp_season_def_epa_allowed_per_play"]
    team_game["def_adj_epa"] = team_game["opp_season_off_epa_per_play"] - team_game["def_epa_allowed_per_play"]
    team_game["point_diff"] = team_game["points_for"] - team_game["points_against"]
    team_game["turnover_margin"] = team_game["takeaways"] - team_game["turnovers"]
    opponent_penalties = team_game[["season", "week", "game_id", "team", "penalty_yards_committed"]].rename(
        columns={"team": "opponent", "penalty_yards_committed": "opp_penalty_yards_committed"}
    )
    team_game = team_game.merge(opponent_penalties, on=["season", "week", "game_id", "opponent"], how="left")
    team_game["penalty_yards_margin"] = team_game["opp_penalty_yards_committed"].fillna(0) - team_game["penalty_yards_committed"].fillna(0)
    team_game["success_margin"] = team_game["off_success_rate"] - team_game["def_success_allowed"]
    team_game["estat_game"] = (
        0.40 * team_game["off_adj_epa"].fillna(0)
        + 0.40 * team_game["def_adj_epa"].fillna(0)
        + 0.10 * team_game["success_margin"].fillna(0)
        + 0.07 * team_game["turnover_margin"].fillna(0)
        + 0.03 * (team_game["penalty_yards_margin"].fillna(0) / 50)
    )

    team_season = team_game.groupby(["season", "team"], as_index=False).agg(
        games=("game_id", "nunique"), points_for=("points_for", "sum"), points_against=("points_against", "sum"),
        point_diff=("point_diff", "sum"), off_plays=("off_plays", "sum"), off_epa_total=("off_epa_total", "sum"),
        off_epa_per_play=("off_epa_per_play", "mean"), off_adj_epa=("off_adj_epa", "mean"),
        off_success_rate=("off_success_rate", "mean"), yards_gained=("yards_gained", "sum"),
        pass_attempts=("pass_attempts", "sum"), rush_attempts=("rush_attempts", "sum"), pass_rate=("pass_rate", "mean"),
        turnovers=("turnovers", "sum"), def_epa_allowed_per_play=("def_epa_allowed_per_play", "mean"),
        def_adj_epa=("def_adj_epa", "mean"), def_success_allowed=("def_success_allowed", "mean"),
        yards_allowed=("yards_allowed", "sum"), takeaways=("takeaways", "sum"),
        penalties_committed=("penalties_committed", "sum"), penalty_yards_committed=("penalty_yards_committed", "sum"),
        penalty_yards_margin=("penalty_yards_margin", "sum"), turnover_margin=("turnover_margin", "sum"),
        estat_raw=("estat_game", "mean"),
    )
    team_season["point_diff_per_game"] = team_season["point_diff"] / team_season["games"].replace(0, np.nan)
    team_season["turnover_margin_per_game"] = team_season["turnover_margin"] / team_season["games"].replace(0, np.nan)
    team_season["penalty_yards_margin_per_game"] = team_season["penalty_yards_margin"] / team_season["games"].replace(0, np.nan)
    team_season["success_margin"] = team_season["off_success_rate"] - team_season["def_success_allowed"]
    team_season["yards_per_play"] = team_season["yards_gained"] / team_season["off_plays"].replace(0, np.nan)
    team_season = finalize_schedule_adjusted_gini(apply_default_gini_scores(team_season), team_game)

    team_game = team_game.sort_values(["season", "week", "team"]).reset_index(drop=True)
    team_season = team_season.sort_values(["season", "overall_rank", "team"]).reset_index(drop=True)
    return team_game, team_season
