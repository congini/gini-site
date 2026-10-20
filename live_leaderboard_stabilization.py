"""Sample-size stabilization for the Live Leaderboard performance input.

The raw Gini and normalized performance values remain unchanged.  This module
only creates the performance value consumed by the Live Market Score.  The
historically selected reliability curve is games / (games + 2): one game's
evidence receives 33% weight, five games receive 71%, and 17 games receive
89%.  The prior is the previous season's final normalized performance regressed
halfway toward the league center so it steadies small samples without freezing
teams at last year's level.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


LIVE_PERFORMANCE_CENTER = 85.0
LIVE_PERFORMANCE_PRIOR_RETENTION = 0.50
LIVE_PERFORMANCE_PRIOR_GAMES = 2.0


def performance_reliability(games_played, prior_games=LIVE_PERFORMANCE_PRIOR_GAMES):
    """Return the share of Live Performance supported by current-season games."""
    games = pd.to_numeric(games_played, errors="coerce").fillna(0).clip(lower=0)
    if prior_games <= 0:
        return pd.Series(1.0, index=games.index, dtype=float)
    return (games / (games + float(prior_games))).clip(0, 1)


def stabilize_live_performance(
    current_performance,
    all_performance,
    *,
    center=LIVE_PERFORMANCE_CENTER,
    prior_retention=LIVE_PERFORMANCE_PRIOR_RETENTION,
    prior_games=LIVE_PERFORMANCE_PRIOR_GAMES,
):
    """Add raw and stabilized Live Performance columns without changing Gini.

    The previous season's final normalized performance is an existing Gini
    signal, not a new rating.  Missing priors fall back to the league center.
    """
    current = current_performance.copy()
    if current.empty:
        for column in (
            "raw_performance_score",
            "performance_baseline_score",
            "performance_reliability",
            "stabilized_performance_score",
        ):
            current[column] = pd.Series(dtype=float)
        return current

    raw = pd.to_numeric(current.get("performance_score"), errors="coerce")
    current["raw_performance_score"] = raw

    history = all_performance.copy()
    required = {"season", "team", "performance_score"}
    if history.empty or not required.issubset(history.columns):
        prior = pd.DataFrame(columns=["season", "team", "prior_final_performance_score"])
    else:
        history["season"] = pd.to_numeric(history["season"], errors="coerce")
        history["performance_score"] = pd.to_numeric(history["performance_score"], errors="coerce")
        prior = history.dropna(subset=["season", "team", "performance_score"])[
            ["season", "team", "performance_score"]
        ].copy()
        prior["season"] = prior["season"].astype(int) + 1
        prior = prior.rename(columns={"performance_score": "prior_final_performance_score"})
        prior = prior.drop_duplicates(["season", "team"], keep="last")

    current["season"] = pd.to_numeric(current["season"], errors="coerce").astype("Int64")
    current = current.merge(prior, on=["season", "team"], how="left", sort=False)
    prior_score = pd.to_numeric(current["prior_final_performance_score"], errors="coerce").fillna(center)
    current["performance_baseline_score"] = center + float(prior_retention) * (prior_score - center)

    game_column = "games" if "games" in current.columns else "scored_games"
    games = current.get(game_column, pd.Series(0, index=current.index))
    current["performance_reliability"] = performance_reliability(games, prior_games)
    current["stabilized_performance_score"] = (
        current["performance_reliability"] * current["raw_performance_score"]
        + (1 - current["performance_reliability"]) * current["performance_baseline_score"]
    )
    current["stabilized_performance_score"] = current["stabilized_performance_score"].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(current["raw_performance_score"]).fillna(center)
    return current.drop(columns=["prior_final_performance_score"], errors="ignore")
