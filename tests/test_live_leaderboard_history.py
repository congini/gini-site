import pandas as pd

from live_leaderboard_history import (
    available_snapshot_keys,
    build_team_history,
    records_through_week,
    snapshot_leaderboard,
)


def test_available_snapshot_keys_uses_only_recorded_in_season_weeks():
    history = pd.DataFrame(
        {
            "snapshot_week": [
                "2026-Offseason",
                "2026-Week-01",
                "2026-Week-03",
                "2025-Week-18",
            ]
        }
    )

    assert available_snapshot_keys(history, 2026) == ["2026-Week-03", "2026-Week-01"]


def test_records_through_week_excludes_later_games():
    games = pd.DataFrame(
        [
            {"season": 2026, "season_type": "REG", "week": 1, "team": "DEN", "points_for": 24, "points_against": 17},
            {"season": 2026, "season_type": "REG", "week": 2, "team": "DEN", "points_for": 20, "points_against": 23},
            {"season": 2026, "season_type": "REG", "week": 3, "team": "DEN", "points_for": 30, "points_against": 10},
        ]
    )

    result = records_through_week(games, 2026, 2).iloc[0]

    assert result["current_wins"] == 1
    assert result["current_losses"] == 1
    assert result["scored_games"] == 2


def test_snapshot_leaderboard_uses_saved_ranks_and_week_specific_record():
    history = pd.DataFrame(
        [
            {
                "snapshot_week": "2026-Week-02",
                "snapshot_timestamp": "2026-09-22T06:00:00-04:00",
                "team": "DEN",
                "live_rank": 21,
                "live_market_score": 78.2,
                "current_gini_score": 99.0,
                "roster_score": 82.0,
                "projected_wins": 9.3,
            }
        ]
    )
    current = pd.DataFrame(
        [
            {
                "team": "DEN",
                "team_name": "Denver Broncos",
                "team_logo": "den.png",
                "team_primary_color": "#FB4F14",
                "team_secondary_color": "#002244",
            }
        ]
    )
    games = pd.DataFrame(
        [
            {"season": 2026, "season_type": "REG", "week": 1, "team": "DEN", "points_for": 24, "points_against": 17},
            {"season": 2026, "season_type": "REG", "week": 2, "team": "DEN", "points_for": 20, "points_against": 23},
            {"season": 2026, "season_type": "REG", "week": 3, "team": "DEN", "points_for": 30, "points_against": 10},
        ]
    )

    result = snapshot_leaderboard(history, "2026-Week-02", current, games, 2026).iloc[0]

    assert result["live_rank"] == 21
    assert result["team_name"] == "Denver Broncos"
    assert result["current_wins"] == 1
    assert result["current_losses"] == 1


def test_team_history_adds_current_week_and_opponent_results():
    history = pd.DataFrame(
        [
            {
                "snapshot_week": "2026-Week-01",
                "snapshot_timestamp": "2026-09-15T06:00:00-04:00",
                "team": "DEN",
                "live_rank": 27,
                "live_market_score": 72.5,
            }
        ]
    )
    current = pd.DataFrame(
        [{"team": "DEN", "live_rank": 10, "live_market_score": 85.4}]
    )
    games = pd.DataFrame(
        [
            {
                "season": 2026,
                "season_type": "REG",
                "week": 1,
                "team": "DEN",
                "opponent": "KC",
                "points_for": 10,
                "points_against": 31,
                "home_away": "AWAY",
            },
            {
                "season": 2026,
                "season_type": "REG",
                "week": 7,
                "team": "DEN",
                "opponent": "LV",
                "points_for": 24,
                "points_against": 17,
                "home_away": "HOME",
            },
        ]
    )

    result = build_team_history(history, current, games, 2026, "2026-Week-07")

    assert result[["week", "live_rank"]].to_dict("records") == [
        {"week": 1, "live_rank": 27},
        {"week": 7, "live_rank": 10},
    ]
    assert result[["opponent", "result", "score_line", "home_away"]].to_dict("records") == [
        {"opponent": "KC", "result": "L", "score_line": "10-31", "home_away": "at"},
        {"opponent": "LV", "result": "W", "score_line": "24-17", "home_away": "vs"},
    ]
