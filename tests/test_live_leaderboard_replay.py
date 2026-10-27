from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backfill_live_leaderboard import missing_weeks, validate_reconstruction
from live_leaderboard_history import combine_recorded_and_reconstructed, snapshot_leaderboard
from live_leaderboard_replay import prepare_replay_inputs, rebuild_week_metrics, restrict_week_rows


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def local_games():
    return pd.read_csv(ROOT / "data" / "team_game_estat.csv", low_memory=False)


def test_rebuild_reconciles_with_authoritative_latest_season(local_games):
    current = pd.read_csv(ROOT / "data" / "team_season_estat.csv")
    season = int(local_games["season"].max())
    week = int(local_games.loc[local_games["season"].eq(season), "week"].max())
    _, rebuilt = rebuild_week_metrics(local_games, season, week)
    reference = current[current["season"].eq(season)].sort_values("team").reset_index(drop=True)
    rebuilt = rebuilt.sort_values("team").reset_index(drop=True)
    columns = ["overall_estat", "off_adj_epa", "def_adj_epa", "schedule_strength", "games"]
    np.testing.assert_allclose(rebuilt[columns], reference[columns], rtol=1e-10, atol=1e-10)


def test_later_games_and_stored_opponent_adjustments_cannot_leak(local_games):
    season = int(local_games["season"].max())
    changed = local_games.copy()
    future = changed["season"].eq(season) & changed["week"].gt(4)
    for column in ["off_epa_per_play", "def_epa_allowed_per_play", "points_for", "points_against"]:
        changed.loc[future, column] = 999999
    for column in ["off_adj_epa", "def_adj_epa", "estat_game", "opp_season_off_epa_per_play", "opp_season_def_epa_allowed_per_play"]:
        changed[column] = -999999
    base_games, base_scores = rebuild_week_metrics(local_games, season, 4)
    new_games, new_scores = rebuild_week_metrics(changed, season, 4)
    pd.testing.assert_frame_equal(base_scores, new_scores)
    pd.testing.assert_frame_equal(base_games[["off_adj_epa", "def_adj_epa", "estat_game"]], new_games[["off_adj_epa", "def_adj_epa", "estat_game"]])


def test_unknown_current_season_aggregate_is_rejected():
    with pytest.raises(ValueError, match="no week cutoff"):
        restrict_week_rows(pd.DataFrame({"season": [2026], "passing_yards": [1500]}), 2026, 4, require_week=True)


def test_replay_context_excludes_later_production_rosters_and_depth(local_games):
    season = int(local_games["season"].max())
    data = {
        "team_game": local_games,
        "team_season": pd.read_csv(ROOT / "data" / "team_season_estat.csv"),
        "games": pd.read_csv(ROOT / "data" / "games_2005_onward.csv"),
    }
    future = {key: pd.read_csv(ROOT / "data" / "live_sources" / f"{filename}.csv", low_memory=False) for key, filename in [
        ("weekly_rosters", "weekly_rosters"), ("injuries", "injuries"),
        ("player_season_stats", "player_season_stats"), ("player_snap_counts", "snap_counts"),
        ("depth_charts", "depth_charts"), ("schedules", "schedules"),
    ]}
    _, context, meta = prepare_replay_inputs(data, future, season, 4)
    for key in ["weekly_rosters", "injuries", "player_season_stats", "player_snap_counts", "schedules"]:
        assert pd.to_numeric(context[key]["week"]).max() <= 4
    roster = context["weekly_rosters"]
    assert roster.groupby("team")["week"].nunique().eq(1).all()
    assert roster["team"].nunique() == 32
    assert pd.to_datetime(context["depth_charts"]["dt"], utc=True).max() <= pd.Timestamp(meta["data_cutoff"])
    assert meta["snapshot_kind"] == "reconstructed"


def test_backfill_fills_gaps_without_replacing_captured_snapshots(local_games):
    season = int(local_games["season"].max())
    captured = pd.DataFrame({"snapshot_week": [f"{season}-Week-03"], "team": ["DEN"], "live_rank": [21]})
    replay = pd.DataFrame({"snapshot_week": [f"{season}-Week-03", f"{season}-Week-04"], "team": ["DEN", "DEN"], "live_rank": [5, 15]})
    merged = combine_recorded_and_reconstructed(captured, replay)
    assert merged["live_rank"].tolist() == [21, 15]
    assert merged["snapshot_kind"].tolist() == ["captured", "reconstructed"]
    assert 4 in missing_weeks(captured, local_games, season)
    assert 3 not in missing_weeks(captured, local_games, season)


def test_replay_record_fields_are_hydrated_without_merge_collisions():
    saved = pd.DataFrame([{"snapshot_week": "2026-Week-04", "team": "DEN", "live_rank": 12, "current_wins": 99}])
    assets = pd.DataFrame([{"team": "DEN", "team_name": "Denver Broncos"}])
    games = pd.DataFrame([{"season": 2026, "week": 1, "team": "DEN", "points_for": 24, "points_against": 17}])
    view = snapshot_leaderboard(saved, "2026-Week-04", assets, games, 2026)
    assert view.iloc[0]["current_wins"] == 1
    assert not any(column.endswith(("_x", "_y")) for column in view)
