"""
Build lightweight data files for the NFL EStat / Gini dashboard.
Input: nflverse play_by_play_YYYY.csv files, games.csv.xlsx, roster_2026.csv.xlsx
Output: data/team_game_estat.csv, data/team_season_estat.csv, data/games_2005_onward.csv, data/roster_2026.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

PBP_FILES = sorted(BASE_DIR.glob("play_by_play_*.csv"))

USECOLS = [
    "season", "season_type", "week", "game_id", "game_date", "home_team", "away_team",
    "posteam", "defteam", "play_type", "epa", "success", "yards_gained", "pass_attempt",
    "rush_attempt", "interception", "fumble_lost", "touchdown", "penalty", "penalty_team",
    "penalty_yards", "qb_kneel", "qb_spike"
]


def safe_read_csv(path: Path, usecols=None):
    try:
        return pd.read_csv(path, usecols=lambda c: c in usecols if usecols else None, low_memory=False)
    except Exception as exc:
        print(f"Standard read failed for {path.name}: {exc}")
        print("Retrying with Python engine and skipping malformed rows...")
        return pd.read_csv(path, usecols=lambda c: c in usecols if usecols else None, engine="python", on_bad_lines="skip")


def zscore_by_season(df: pd.DataFrame, col: str) -> pd.Series:
    mean = df.groupby("season")[col].transform("mean")
    std = df.groupby("season")[col].transform("std").replace(0, np.nan)
    return ((df[col] - mean) / std).fillna(0)


all_team_games = []

for path in PBP_FILES:
    print(f"Processing {path.name}...")
    pbp = safe_read_csv(path, usecols=USECOLS)

    # Keep regular and playoff scrimmage/no-play penalty rows with team context.
    pbp = pbp[pbp["posteam"].notna() & pbp["defteam"].notna()].copy()
    pbp = pbp[pbp["season_type"].isin(["REG", "POST"])]
    pbp = pbp[pbp["play_type"].isin(["pass", "run", "no_play"])]
    if "qb_kneel" in pbp.columns:
        pbp = pbp[pbp["qb_kneel"].fillna(0) == 0]
    if "qb_spike" in pbp.columns:
        pbp = pbp[pbp["qb_spike"].fillna(0) == 0]

    numeric_cols = ["epa", "success", "yards_gained", "pass_attempt", "rush_attempt", "interception", "fumble_lost", "touchdown", "penalty", "penalty_yards"]
    for col in numeric_cols:
        if col in pbp.columns:
            pbp[col] = pd.to_numeric(pbp[col], errors="coerce").fillna(0)

    # Offense by team-game
    off = (
        pbp.groupby(["season", "season_type", "week", "game_id", "game_date", "home_team", "away_team", "posteam", "defteam"], dropna=False)
        .agg(
            off_plays=("epa", "count"),
            off_epa_total=("epa", "sum"),
            off_epa_per_play=("epa", "mean"),
            off_success_rate=("success", "mean"),
            yards_gained=("yards_gained", "sum"),
            pass_attempts=("pass_attempt", "sum"),
            rush_attempts=("rush_attempt", "sum"),
            turnovers=("interception", "sum"),
            fumbles_lost=("fumble_lost", "sum"),
            touchdowns=("touchdown", "sum"),
        )
        .reset_index()
    )
    off["turnovers"] = off["turnovers"] + off["fumbles_lost"]
    off = off.drop(columns=["fumbles_lost"])
    off = off.rename(columns={"posteam": "team", "defteam": "opponent"})

    # Defensive row is the same plays from defteam perspective.
    deff = (
        pbp.groupby(["season", "season_type", "week", "game_id", "defteam", "posteam"], dropna=False)
        .agg(
            def_plays=("epa", "count"),
            def_epa_allowed_total=("epa", "sum"),
            def_epa_allowed_per_play=("epa", "mean"),
            def_success_allowed=("success", "mean"),
            yards_allowed=("yards_gained", "sum"),
            takeaways_int=("interception", "sum"),
            takeaways_fumble=("fumble_lost", "sum"),
        )
        .reset_index()
        .rename(columns={"defteam": "team", "posteam": "opponent"})
    )
    deff["takeaways"] = deff["takeaways_int"] + deff["takeaways_fumble"]
    deff = deff.drop(columns=["takeaways_int", "takeaways_fumble"])

    # Penalties committed by team-game.
    penalties = pbp[pbp["penalty_team"].notna()].copy()
    pen = (
        penalties.groupby(["season", "week", "game_id", "penalty_team"], dropna=False)
        .agg(
            penalties_committed=("penalty", "sum"),
            penalty_yards_committed=("penalty_yards", "sum"),
        )
        .reset_index()
        .rename(columns={"penalty_team": "team"})
    )

    team_game = off.merge(
        deff,
        on=["season", "season_type", "week", "game_id", "team", "opponent"],
        how="left"
    ).merge(
        pen,
        on=["season", "week", "game_id", "team"],
        how="left"
    )

    team_game["penalties_committed"] = team_game["penalties_committed"].fillna(0)
    team_game["penalty_yards_committed"] = team_game["penalty_yards_committed"].fillna(0)
    team_game["is_home"] = (team_game["team"] == team_game["home_team"]).astype(int)
    team_game["home_away"] = np.where(team_game["is_home"] == 1, "Home", "Away")
    team_game["pass_rate"] = team_game["pass_attempts"] / (team_game["pass_attempts"] + team_game["rush_attempts"]).replace(0, np.nan)
    all_team_games.append(team_game)

team_game = pd.concat(all_team_games, ignore_index=True)

# Load game scores/schedule info.
games_path = BASE_DIR / "games.csv.xlsx"
if games_path.exists():
    games = pd.read_excel(games_path)
    games_2005 = games[games["season"] >= 2005].copy()
    games_2005.to_csv(DATA_DIR / "games_2005_onward.csv", index=False)

    home_scores = games_2005[["game_id", "home_team", "home_score", "away_score"]].rename(columns={"home_team": "team", "home_score": "points_for", "away_score": "points_against"})
    away_scores = games_2005[["game_id", "away_team", "away_score", "home_score"]].rename(columns={"away_team": "team", "away_score": "points_for", "home_score": "points_against"})
    scores = pd.concat([home_scores, away_scores], ignore_index=True)
    team_game = team_game.merge(scores, on=["game_id", "team"], how="left")
else:
    team_game["points_for"] = np.nan
    team_game["points_against"] = np.nan

# Opponent-adjusted game measures using season averages.
season_team_avgs = (
    team_game.groupby(["season", "team"], as_index=False)
    .agg(
        season_off_epa_per_play=("off_epa_per_play", "mean"),
        season_def_epa_allowed_per_play=("def_epa_allowed_per_play", "mean"),
    )
)
opp = season_team_avgs.rename(columns={
    "team": "opponent",
    "season_off_epa_per_play": "opp_season_off_epa_per_play",
    "season_def_epa_allowed_per_play": "opp_season_def_epa_allowed_per_play",
})
team_game = team_game.merge(opp, on=["season", "opponent"], how="left")
team_game["off_adj_epa"] = team_game["off_epa_per_play"] - team_game["opp_season_def_epa_allowed_per_play"]
team_game["def_adj_epa"] = team_game["opp_season_off_epa_per_play"] - team_game["def_epa_allowed_per_play"]
team_game["point_diff"] = team_game["points_for"] - team_game["points_against"]
team_game["turnover_margin"] = team_game["takeaways"] - team_game["turnovers"]

# Opponent penalty yards for margin.
opp_pen = team_game[["season", "week", "game_id", "team", "penalty_yards_committed"]].rename(columns={"team": "opponent", "penalty_yards_committed": "opp_penalty_yards_committed"})
team_game = team_game.merge(opp_pen, on=["season", "week", "game_id", "opponent"], how="left")
team_game["penalty_yards_margin"] = team_game["opp_penalty_yards_committed"].fillna(0) - team_game["penalty_yards_committed"].fillna(0)
team_game["success_margin"] = team_game["off_success_rate"] - team_game["def_success_allowed"]
team_game["estat_game"] = (
    0.40 * team_game["off_adj_epa"].fillna(0)
    + 0.40 * team_game["def_adj_epa"].fillna(0)
    + 0.10 * team_game["success_margin"].fillna(0)
    + 0.07 * team_game["turnover_margin"].fillna(0)
    + 0.03 * (team_game["penalty_yards_margin"].fillna(0) / 50)
)

# Season-level table.
team_season = (
    team_game.groupby(["season", "team"], as_index=False)
    .agg(
        games=("game_id", "nunique"),
        points_for=("points_for", "sum"),
        points_against=("points_against", "sum"),
        point_diff=("point_diff", "sum"),
        off_plays=("off_plays", "sum"),
        off_epa_total=("off_epa_total", "sum"),
        off_epa_per_play=("off_epa_per_play", "mean"),
        off_adj_epa=("off_adj_epa", "mean"),
        off_success_rate=("off_success_rate", "mean"),
        yards_gained=("yards_gained", "sum"),
        pass_attempts=("pass_attempts", "sum"),
        rush_attempts=("rush_attempts", "sum"),
        pass_rate=("pass_rate", "mean"),
        turnovers=("turnovers", "sum"),
        def_epa_allowed_per_play=("def_epa_allowed_per_play", "mean"),
        def_adj_epa=("def_adj_epa", "mean"),
        def_success_allowed=("def_success_allowed", "mean"),
        yards_allowed=("yards_allowed", "sum"),
        takeaways=("takeaways", "sum"),
        penalties_committed=("penalties_committed", "sum"),
        penalty_yards_committed=("penalty_yards_committed", "sum"),
        penalty_yards_margin=("penalty_yards_margin", "sum"),
        turnover_margin=("turnover_margin", "sum"),
        estat_raw=("estat_game", "mean"),
    )
)

team_season["point_diff_per_game"] = team_season["point_diff"] / team_season["games"].replace(0, np.nan)
team_season["turnover_margin_per_game"] = team_season["turnover_margin"] / team_season["games"].replace(0, np.nan)
team_season["penalty_yards_margin_per_game"] = team_season["penalty_yards_margin"] / team_season["games"].replace(0, np.nan)
team_season["success_margin"] = team_season["off_success_rate"] - team_season["def_success_allowed"]
team_season["yards_per_play"] = team_season["yards_gained"] / team_season["off_plays"].replace(0, np.nan)

# Default standardized component scores by season.
team_season["off_z"] = zscore_by_season(team_season, "off_adj_epa")
team_season["def_z"] = zscore_by_season(team_season, "def_adj_epa")
team_season["pd_z"] = zscore_by_season(team_season, "point_diff_per_game")
team_season["success_z"] = zscore_by_season(team_season, "success_margin")
team_season["turnover_z"] = zscore_by_season(team_season, "turnover_margin_per_game")
team_season["penalty_z"] = zscore_by_season(team_season, "penalty_yards_margin_per_game")

team_season["offense_estat"] = 100 + 15 * team_season["off_z"]
team_season["defense_estat"] = 100 + 15 * team_season["def_z"]
team_season["overall_estat"] = 100 + 15 * (
    0.30 * team_season["off_z"]
    + 0.30 * team_season["def_z"]
    + 0.15 * team_season["pd_z"]
    + 0.15 * team_season["success_z"]
    + 0.07 * team_season["turnover_z"]
    + 0.03 * team_season["penalty_z"]
)

# Strength of schedule: average opponent overall_estat by games played.
opp_scores = team_season[["season", "team", "overall_estat"]].rename(columns={"team": "opponent", "overall_estat": "opponent_overall_estat"})
team_game_sos = team_game.merge(opp_scores, on=["season", "opponent"], how="left")
sos = team_game_sos.groupby(["season", "team"], as_index=False).agg(schedule_strength=("opponent_overall_estat", "mean"))
team_season = team_season.merge(sos, on=["season", "team"], how="left")
team_season["schedule_strength_z"] = zscore_by_season(team_season, "schedule_strength")

# Rank fields.
team_season["overall_rank"] = team_season.groupby("season")["overall_estat"].rank(ascending=False, method="min").astype(int)
team_season["offense_rank"] = team_season.groupby("season")["offense_estat"].rank(ascending=False, method="min").astype(int)
team_season["defense_rank"] = team_season.groupby("season")["defense_estat"].rank(ascending=False, method="min").astype(int)
team_season["sos_rank"] = team_season.groupby("season")["schedule_strength"].rank(ascending=False, method="min").astype(int)

# Sort and save.
team_game = team_game.sort_values(["season", "week", "team"]).reset_index(drop=True)
team_season = team_season.sort_values(["season", "overall_rank"]).reset_index(drop=True)

team_game.to_csv(DATA_DIR / "team_game_estat.csv", index=False)
team_season.to_csv(DATA_DIR / "team_season_estat.csv", index=False)

# Roster passthrough.
roster_path = BASE_DIR / "roster_2026.csv.xlsx"
if roster_path.exists():
    roster = pd.read_excel(roster_path)
    roster.to_csv(DATA_DIR / "roster_2026.csv", index=False)

print("Done.")
print(f"Team-game rows: {len(team_game):,}")
print(f"Team-season rows: {len(team_season):,}")
print(f"Saved files in: {DATA_DIR}")
