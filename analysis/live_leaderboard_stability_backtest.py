"""Historical reliability study for the Live Leaderboard performance layer.

This script intentionally rebuilds season-to-date Gini inputs at every regular-
season week.  It does not modify the Gini Dashboard or production data.  The
output is an inspectable comparison of raw performance against candidate
sample-size adjustments for the Live Leaderboard only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gini_metrics import BASELINE_WEIGHTS, apply_default_gini_scores, finalize_schedule_adjusted_gini
from live_leaderboard_stabilization import (
    LIVE_PERFORMANCE_CENTER,
    LIVE_PERFORMANCE_PRIOR_GAMES,
    LIVE_PERFORMANCE_PRIOR_RETENTION,
)
from season_utils import normalize_team


TEAM_GAME_PATH = ROOT / "data" / "team_game_estat.csv"
LIVE_HISTORY_PATH = ROOT / "data" / "live_leaderboard_weekly_history.csv"

PERFORMANCE_CENTER = LIVE_PERFORMANCE_CENTER
LIVE_PERFORMANCE_WEIGHT = 0.65
COMPONENT_COLUMNS = {
    "EPA offense": ("off_z", "Offense"),
    "EPA defense": ("def_z", "Defense"),
    "Point differential": ("pd_z", "Point Diff"),
    "Success margin": ("success_z", "Success Margin"),
    "Turnover margin": ("turnover_z", "Turnovers"),
    "Penalty margin": ("penalty_z", "Penalties"),
    "Schedule strength": ("schedule_strength_z", "Schedule Strength"),
}


@dataclass(frozen=True)
class Candidate:
    prior_retention: float
    k: float


def normalize_performance(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    std = numeric.std()
    if pd.isna(std) or std == 0:
        return pd.Series(PERFORMANCE_CENTER, index=numeric.index)
    return (PERFORMANCE_CENTER + 10 * ((numeric - numeric.mean()) / std)).clip(60, 120)


def cumulative_gini(team_games: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    games = team_games[
        team_games["season"].eq(season)
        & team_games["season_type"].astype(str).str.upper().eq("REG")
        & team_games["week"].le(week)
    ].copy()
    if games.empty:
        return pd.DataFrame()

    for column in ("team", "opponent"):
        games[column] = games[column].map(normalize_team)

    current_units = games.groupby("team", as_index=False).agg(
        current_off_epa=("off_epa_per_play", "mean"),
        current_def_epa_allowed=("def_epa_allowed_per_play", "mean"),
    )
    opponent_units = current_units.rename(
        columns={
            "team": "opponent",
            "current_off_epa": "opponent_off_epa",
            "current_def_epa_allowed": "opponent_def_epa_allowed",
        }
    )
    games = games.drop(
        columns=["opp_season_off_epa_per_play", "opp_season_def_epa_allowed_per_play"],
        errors="ignore",
    ).merge(opponent_units, on="opponent", how="left")
    games["off_adj_epa"] = games["off_epa_per_play"] - games["opponent_def_epa_allowed"]
    games["def_adj_epa"] = games["opponent_off_epa"] - games["def_epa_allowed_per_play"]
    games["point_diff"] = games["points_for"] - games["points_against"]
    games["turnover_margin"] = games["takeaways"] - games["turnovers"]
    games["success_margin"] = games["off_success_rate"] - games["def_success_allowed"]

    summary = games.groupby(["season", "team"], as_index=False).agg(
        games=("game_id", "nunique"),
        points_for=("points_for", "sum"),
        points_against=("points_against", "sum"),
        point_diff=("point_diff", "sum"),
        off_adj_epa=("off_adj_epa", "mean"),
        off_success_rate=("off_success_rate", "mean"),
        def_adj_epa=("def_adj_epa", "mean"),
        def_success_allowed=("def_success_allowed", "mean"),
        turnover_margin=("turnover_margin", "sum"),
        penalty_yards_margin=("penalty_yards_margin", "sum"),
    )
    summary["point_diff_per_game"] = summary["point_diff"] / summary["games"]
    summary["turnover_margin_per_game"] = summary["turnover_margin"] / summary["games"]
    summary["penalty_yards_margin_per_game"] = summary["penalty_yards_margin"] / summary["games"]
    summary["success_margin"] = summary["off_success_rate"] - summary["def_success_allowed"]

    scored = finalize_schedule_adjusted_gini(apply_default_gini_scores(summary), games)
    scored["week"] = int(week)
    scored["current_gini_score"] = scored["overall_estat"]
    scored["raw_performance_score"] = normalize_performance(scored["current_gini_score"])
    scored["wins"] = (
        games.assign(win=games["points_for"].gt(games["points_against"]).astype(int))
        .groupby("team")["win"]
        .sum()
        .reindex(scored["team"])
        .to_numpy()
    )

    for label, (column, weight_label) in COMPONENT_COLUMNS.items():
        scored[f"component::{label}"] = 15 * BASELINE_WEIGHTS[weight_label] * scored[column]
    return scored


def build_weekly_history(team_games: pd.DataFrame) -> pd.DataFrame:
    frames = []
    regular = team_games[team_games["season_type"].astype(str).str.upper().eq("REG")].copy()
    regular["season"] = pd.to_numeric(regular["season"], errors="coerce").astype("Int64")
    regular["week"] = pd.to_numeric(regular["week"], errors="coerce").astype("Int64")
    for season in sorted(regular["season"].dropna().astype(int).unique()):
        season_games = regular[regular["season"].eq(season)]
        for week in sorted(season_games["week"].dropna().astype(int).unique()):
            frame = cumulative_gini(regular, season, week)
            if not frame.empty:
                frames.append(frame)
    history = pd.concat(frames, ignore_index=True)
    return history.sort_values(["season", "week", "team"]).reset_index(drop=True)


def attach_prior(history: pd.DataFrame, prior_retention: float) -> pd.DataFrame:
    output = history.copy()
    final = (
        output.sort_values("week")
        .groupby(["season", "team"], as_index=False)
        .tail(1)[["season", "team", "raw_performance_score", "wins", "games"]]
    )
    prior = final.rename(
        columns={
            "season": "prior_season",
            "raw_performance_score": "prior_final_performance",
            "wins": "prior_wins",
            "games": "prior_games",
        }
    )
    prior["season"] = prior["prior_season"] + 1
    output = output.merge(
        prior[["season", "team", "prior_final_performance", "prior_wins", "prior_games"]],
        on=["season", "team"],
        how="left",
    )
    output["baseline_performance_score"] = PERFORMANCE_CENTER + prior_retention * (
        output["prior_final_performance"].fillna(PERFORMANCE_CENTER) - PERFORMANCE_CENTER
    )
    return output


def add_candidate_scores(history: pd.DataFrame, candidate: Candidate) -> pd.DataFrame:
    output = attach_prior(history, candidate.prior_retention)
    games = pd.to_numeric(output["games"], errors="coerce").fillna(0).clip(lower=0)
    reliability = games / (games + candidate.k) if candidate.k > 0 else pd.Series(1.0, index=output.index)
    output["performance_reliability"] = reliability.clip(0, 1)
    output["stabilized_performance_score"] = (
        output["performance_reliability"] * output["raw_performance_score"]
        + (1 - output["performance_reliability"]) * output["baseline_performance_score"]
    )

    # A neutral historical proxy holds the non-performance 35% at the league
    # center.  Historical roster/quadrant/projection snapshots do not exist for
    # every season, and holding them constant cleanly isolates the one proposed
    # production change: stabilizing the 65% performance input.
    output["raw_live_proxy"] = (
        LIVE_PERFORMANCE_WEIGHT * output["raw_performance_score"]
        + (1 - LIVE_PERFORMANCE_WEIGHT) * PERFORMANCE_CENTER
    )
    output["stabilized_live_proxy"] = (
        LIVE_PERFORMANCE_WEIGHT * output["stabilized_performance_score"]
        + (1 - LIVE_PERFORMANCE_WEIGHT) * PERFORMANCE_CENTER
    )
    output["raw_live_rank"] = output.groupby(["season", "week"])["raw_live_proxy"].rank(
        ascending=False, method="min"
    )
    output["stabilized_live_rank"] = output.groupby(["season", "week"])["stabilized_live_proxy"].rank(
        ascending=False, method="min"
    )
    return output


def ranked_correlation(x: pd.Series, y: pd.Series) -> float:
    valid = pd.concat([x, y], axis=1).dropna()
    if len(valid) < 3:
        return float("nan")
    return float(valid.iloc[:, 0].rank().corr(valid.iloc[:, 1].rank()))


def evaluate_candidate(history: pd.DataFrame, candidate: Candidate) -> dict[str, float]:
    scored = add_candidate_scores(history, candidate)
    final = (
        scored.sort_values("week")
        .groupby(["season", "team"], as_index=False)
        .tail(1)[["season", "team", "wins", "games", "raw_performance_score"]]
        .rename(columns={"wins": "final_wins", "games": "final_games", "raw_performance_score": "final_performance"})
    )
    scored = scored.merge(final, on=["season", "team"], how="left")
    scored["final_win_pct"] = scored["final_wins"] / scored["final_games"].replace(0, np.nan)
    scored = scored[scored["season"].ge(scored["season"].min() + 1)].copy()

    group = scored.groupby(["season", "team"], sort=False)
    for column in ("raw_live_proxy", "stabilized_live_proxy", "raw_live_rank", "stabilized_live_rank"):
        scored[f"previous_{column}"] = group[column].shift(1)
    transitions = scored.dropna(subset=["previous_raw_live_proxy"]).copy()
    transitions["raw_score_move"] = (transitions["raw_live_proxy"] - transitions["previous_raw_live_proxy"]).abs()
    transitions["stable_score_move"] = (
        transitions["stabilized_live_proxy"] - transitions["previous_stabilized_live_proxy"]
    ).abs()
    transitions["raw_rank_move"] = (transitions["raw_live_rank"] - transitions["previous_raw_live_rank"]).abs()
    transitions["stable_rank_move"] = (
        transitions["stabilized_live_rank"] - transitions["previous_stabilized_live_rank"]
    ).abs()

    early = scored[scored["week"].between(2, 8)]
    bad = early[early["final_win_pct"].le(0.35)]
    good = early[early["final_win_pct"].ge(0.60)]
    return {
        "prior_retention": candidate.prior_retention,
        "k": candidate.k,
        "avg_score_move": transitions["stable_score_move"].mean(),
        "p95_score_move": transitions["stable_score_move"].quantile(0.95),
        "max_score_move": transitions["stable_score_move"].max(),
        "avg_rank_move": transitions["stable_rank_move"].mean(),
        "p95_rank_move": transitions["stable_rank_move"].quantile(0.95),
        "future_win_corr": ranked_correlation(early["stabilized_live_proxy"], early["final_win_pct"]),
        "final_quality_corr": ranked_correlation(early["stabilized_live_proxy"], early["final_performance"]),
        "bad_team_top8_rate": float(bad["stabilized_live_rank"].le(8).mean()),
        "good_team_top8_rate": float(good["stabilized_live_rank"].le(8).mean()),
        "raw_avg_score_move": transitions["raw_score_move"].mean(),
        "raw_p95_score_move": transitions["raw_score_move"].quantile(0.95),
        "raw_avg_rank_move": transitions["raw_rank_move"].mean(),
        "raw_p95_rank_move": transitions["raw_rank_move"].quantile(0.95),
        "row_count": float(len(scored)),
    }


def driver_summary(history: pd.DataFrame) -> pd.DataFrame:
    columns = [f"component::{label}" for label in COMPONENT_COLUMNS]
    ordered = history.sort_values(["season", "team", "week"]).copy()
    deltas = ordered.groupby(["season", "team"])[columns].diff().abs()
    summary = pd.DataFrame(
        {
            "driver": [column.split("::", 1)[1] for column in columns],
            "mean_abs_gini_points": [deltas[column].mean() for column in columns],
            "p95_abs_gini_points": [deltas[column].quantile(0.95) for column in columns],
        }
    )
    total = summary["mean_abs_gini_points"].sum()
    summary["share_of_component_motion"] = summary["mean_abs_gini_points"] / total
    return summary.sort_values("mean_abs_gini_points", ascending=False)


def current_refresh_component_changes() -> pd.DataFrame:
    if not LIVE_HISTORY_PATH.exists():
        return pd.DataFrame()
    history = pd.read_csv(LIVE_HISTORY_PATH)
    week_rows = history[history["snapshot_week"].astype(str).str.contains("Week-", na=False)].copy()
    if week_rows.empty:
        return pd.DataFrame()
    week_rows["week_number"] = pd.to_numeric(
        week_rows["snapshot_week"].astype(str).str.extract(r"Week-(\d+)")[0], errors="coerce"
    )
    available = sorted(week_rows["week_number"].dropna().unique())
    if len(available) < 2:
        return pd.DataFrame()
    previous_week, current_week = available[-2], available[-1]
    previous = week_rows[week_rows["week_number"].eq(previous_week)].copy()
    current = week_rows[week_rows["week_number"].eq(current_week)].copy()
    previous = previous.sort_values("snapshot_timestamp").drop_duplicates("team", keep="last")
    current = current.sort_values("snapshot_timestamp").drop_duplicates("team", keep="last")
    merged = current.merge(previous, on="team", suffixes=("_new", "_previous"))
    for suffix in ("new", "previous"):
        merged[f"quadrant_score_{suffix}"] = (
            60 * merged[f"q1_probability_{suffix}"]
            + 75 * merged[f"q2_probability_{suffix}"]
            + 88 * merged[f"q3_probability_{suffix}"]
            + 100 * merged[f"q4_probability_{suffix}"]
        )
    components = {
        "performance": ("performance_score", 0.65),
        "roster": ("roster_score", 0.10),
        "projected wins": ("projected_wins", 0.15 * 50 / 17),
        "quadrant": ("quadrant_score", 0.10),
    }
    for label, (column, weight) in components.items():
        merged[f"{label}_score_delta"] = merged[f"{column}_new"] - merged[f"{column}_previous"]
        merged[f"{label}_live_contribution"] = weight * merged[f"{label}_score_delta"]
    merged["observed_live_delta"] = merged["live_market_score_new"] - merged["live_market_score_previous"]
    merged["rank_change"] = merged["live_rank_previous"] - merged["live_rank_new"]
    return merged.sort_values("observed_live_delta", key=lambda values: values.abs(), ascending=False)


def example_transitions(scored: pd.DataFrame, limit: int = 8) -> pd.DataFrame:
    ordered = scored.sort_values(["season", "team", "week"]).copy()
    group = ordered.groupby(["season", "team"], sort=False)
    for column in ("raw_live_rank", "stabilized_live_rank", "raw_live_proxy", "stabilized_live_proxy"):
        ordered[f"previous_{column}"] = group[column].shift(1)
    examples = ordered.dropna(subset=["previous_raw_live_rank"]).copy()
    examples["raw_rank_move"] = (examples["raw_live_rank"] - examples["previous_raw_live_rank"]).abs()
    examples["stabilized_rank_move"] = (
        examples["stabilized_live_rank"] - examples["previous_raw_live_rank"]
    ).abs()
    examples["rank_move_reduction"] = examples["raw_rank_move"] - examples["stabilized_rank_move"]
    examples = examples.sort_values(["rank_move_reduction", "raw_rank_move"], ascending=False).head(limit)
    return examples[
        [
            "season", "week", "team", "games",
            "previous_raw_live_rank", "raw_live_rank", "stabilized_live_rank",
            "previous_raw_live_proxy", "raw_live_proxy", "stabilized_live_proxy",
            "raw_performance_score", "stabilized_performance_score", "performance_reliability",
        ]
    ]


def main() -> None:
    team_games = pd.read_csv(TEAM_GAME_PATH, low_memory=False)
    team_games["season"] = pd.to_numeric(team_games["season"], errors="coerce")
    team_games["week"] = pd.to_numeric(team_games["week"], errors="coerce")
    history = build_weekly_history(team_games)

    print("\nACTUAL TOP-LEVEL WEIGHTS")
    print("performance=65%, roster=10%, projected_wins=15%, quadrant=10%")
    print("\nRAW GINI WEEK-TO-WEEK DRIVER MOTION")
    print(driver_summary(history).to_string(index=False, formatters={"share_of_component_motion": "{:.1%}".format}))

    candidates = [
        Candidate(retention, k)
        for retention in (0.50, 0.75, 1.00)
        for k in (0.0, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0)
    ]
    results = pd.DataFrame(evaluate_candidate(history, candidate) for candidate in candidates)
    print("\nCANDIDATE RELIABILITY RESULTS")
    print(
        results.sort_values(["future_win_corr", "avg_rank_move"], ascending=[False, True])[
            [
                "prior_retention", "k", "avg_score_move", "p95_score_move", "avg_rank_move",
                "p95_rank_move", "future_win_corr", "final_quality_corr", "bad_team_top8_rate",
                "good_team_top8_rate", "raw_avg_score_move", "raw_avg_rank_move",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )

    selected = Candidate(
        prior_retention=LIVE_PERFORMANCE_PRIOR_RETENTION,
        k=LIVE_PERFORMANCE_PRIOR_GAMES,
    )
    selected_scored = add_candidate_scores(history, selected)
    print(
        f"\nDIAGNOSTIC EXAMPLES FOR retention={selected.prior_retention:.2f}, "
        f"k={selected.k:.1f}"
    )
    print(example_transitions(selected_scored).to_string(index=False, float_format=lambda value: f"{value:.2f}"))

    current = current_refresh_component_changes()
    if not current.empty:
        print("\nLATEST AVAILABLE CURRENT-SEASON REFRESH COMPONENT CHANGES")
        columns = [
            "team", "week_number_previous", "week_number_new", "rank_change", "observed_live_delta",
            "performance_live_contribution", "roster_live_contribution",
            "projected wins_live_contribution", "quadrant_live_contribution",
        ]
        print(current[columns].head(10).to_string(index=False, float_format=lambda value: f"{value:.2f}"))


if __name__ == "__main__":
    main()
