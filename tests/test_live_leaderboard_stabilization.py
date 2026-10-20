import pandas as pd
import pytest

from live_leaderboard_stabilization import (
    performance_reliability,
    stabilize_live_performance,
)


def test_reliability_fades_the_adjustment_as_games_accumulate():
    reliability = performance_reliability(pd.Series([1, 5, 17]))
    assert reliability.tolist() == pytest.approx([1 / 3, 5 / 7, 17 / 19])


def test_stabilization_preserves_raw_score_and_uses_regressed_prior():
    current = pd.DataFrame(
        {"season": [2026], "team": ["DEN"], "games": [1], "performance_score": [105.0]}
    )
    history = pd.DataFrame(
        {
            "season": [2025, 2026],
            "team": ["DEN", "DEN"],
            "games": [17, 1],
            "performance_score": [95.0, 105.0],
        }
    )

    result = stabilize_live_performance(current, history)

    assert result.loc[0, "performance_score"] == 105.0
    assert result.loc[0, "raw_performance_score"] == 105.0
    assert result.loc[0, "performance_baseline_score"] == 90.0
    assert result.loc[0, "performance_reliability"] == pytest.approx(1 / 3)
    assert result.loc[0, "stabilized_performance_score"] == 95.0


def test_missing_prior_uses_league_center_without_mutating_input():
    current = pd.DataFrame(
        {"season": [2026], "team": ["EXP"], "games": [5], "performance_score": [91.0]}
    )
    before = current.copy(deep=True)

    result = stabilize_live_performance(current, pd.DataFrame())

    pd.testing.assert_frame_equal(current, before)
    assert result.loc[0, "performance_baseline_score"] == 85.0
    assert result.loc[0, "stabilized_performance_score"] == pytest.approx(89.2857142857)
