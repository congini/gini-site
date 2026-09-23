from __future__ import annotations

from datetime import datetime, timedelta
import json
import os
from pathlib import Path

import pandas as pd

import gini_metrics
import live_roster_scoring
from snapshot_live_leaderboard import (
    latest_completed_regular_season_key,
    replace_weekly_snapshot,
)
import live_source_refresh
import refresh_current_season
from season_utils import (
    calculate_regular_season_records,
    comparable_ranked_populations,
    completed_regular_season_games,
    current_projected_finish,
    current_win_pace,
    is_prior_snapshot_period,
    regular_season_completion_status,
    select_live_performance_population,
    weekly_rank_change,
)


def _game(game_id, home, away, home_score=None, away_score=None, week=1, season=2026):
    return {
        "game_id": game_id,
        "season": season,
        "game_type": "REG",
        "week": week,
        "gameday": f"{season}-09-01",
        "home_team": home,
        "away_team": away,
        "home_score": home_score,
        "away_score": away_score,
    }


def test_future_games_are_not_counted_and_pacing_uses_completed_regular_games():
    schedule = pd.DataFrame(
        [
            _game("played", "SEA", "NE", 13, 10),
            _game("future", "DEN", "KC"),
            {**_game("post", "SEA", "DEN", 24, 21), "game_type": "POST"},
        ]
    )
    completed = completed_regular_season_games(schedule, 2026)
    records = calculate_regular_season_records(schedule, 2026)

    assert completed["game_id"].tolist() == ["played"]
    assert set(records["team"]) == {"SEA", "NE"}
    assert records.loc[records["team"].eq("SEA"), "current_wins"].iloc[0] == 1
    assert current_win_pace(1, 1, 2026) == 17


def test_current_projected_finish_banks_results_without_one_game_overreaction():
    preseason_projection = 11.26

    winless_after_one = current_projected_finish(0, 0, 1, preseason_projection, 2026)
    unbeaten_after_one = current_projected_finish(1, 0, 1, preseason_projection, 2026)
    tied_after_one = current_projected_finish(0, 1, 1, preseason_projection, 2026)

    assert winless_after_one == 16 * preseason_projection / 17
    assert unbeaten_after_one == 1 + 16 * preseason_projection / 17
    assert tied_after_one == 0.5 + 16 * preseason_projection / 17
    assert current_projected_finish(0, 0, 0, preseason_projection, 2026) == preseason_projection
    assert current_projected_finish(10, 0, 17, preseason_projection, 2026) == 10


def test_super_square_completion_gate_uses_results_not_calendar():
    teams = [f"T{i:02d}" for i in range(32)]
    rotation = teams[1:]
    fixed = teams[0]
    games = []
    game_number = 0
    for week in range(1, 18):
        lineup = [fixed, *rotation]
        for index in range(16):
            game_number += 1
            games.append(_game(f"g{game_number}", lineup[index], lineup[-index - 1], 20, 17, week=week))
        rotation = [rotation[-1], *rotation[:-1]]
    schedule = pd.DataFrame(games)

    assert regular_season_completion_status(schedule, 2026)["complete"] is True
    schedule.loc[schedule.index[-1], "away_score"] = pd.NA
    assert regular_season_completion_status(schedule, 2026)["complete"] is False


def test_active_season_replacement_preserves_history_and_prevents_duplicates():
    existing = pd.DataFrame(
        [
            {"season": 2025, "team": "SEA", "value": 1.234567890123},
            {"season": 2026, "team": "SEA", "value": 2.0},
        ]
    )
    replacement = pd.DataFrame(
        [
            {"season": 2026, "team": "SEA", "value": 3.0},
            {"season": 2026, "team": "NE", "value": 4.0},
        ]
    )
    merged, history = refresh_current_season.replace_season_rows(existing, replacement, 2026)

    pd.testing.assert_frame_equal(history.reset_index(drop=True), existing.iloc[[0]].reset_index(drop=True))
    assert len(merged[merged["season"].eq(2026)]) == 2
    assert not merged.duplicated(["season", "team"]).any()


def test_freshness_uses_pipeline_metadata_not_file_mtime(tmp_path, monkeypatch):
    source_path = tmp_path / "schedules.csv"
    source_path.write_text("season\n2026\n", encoding="utf-8")
    os.utime(source_path, (0, 0))
    status_path = tmp_path / "status.json"
    now = live_source_refresh.now_et()
    status_path.write_text(
        json.dumps(
            {
                "refresh_pipeline_version": live_source_refresh.REFRESH_PIPELINE_VERSION,
                "last_successful_refresh_time": now.isoformat(),
                "sources": {"schedules": {"ok": True, "rows": 272, "season": 2026}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(live_source_refresh, "LIVE_SOURCE_REFRESH_STATUS_PATH", status_path)
    monkeypatch.setattr(live_source_refresh, "LIVE_SOURCE_CACHE_FILES", {"schedules": source_path})
    monkeypatch.setattr(live_source_refresh, "REQUIRED_LIVE_SOURCE_KEYS", ("schedules",))

    state = live_source_refresh.get_live_source_file_state(now, now - timedelta(hours=1))
    assert state["schedules"]["stale"] is False
    assert state["schedules"]["freshness_authority"] == "pipeline_metadata"

    status_path.write_text(json.dumps({"refresh_pipeline_version": 2, "ok": True}), encoding="utf-8")
    os.utime(source_path, None)
    state = live_source_refresh.get_live_source_file_state(now, now - timedelta(hours=1))
    assert state["schedules"]["stale"] is True


def test_failed_download_retains_previous_known_good_outputs(tmp_path, monkeypatch):
    paths = {
        "TEAM_GAME_PATH": tmp_path / "team_game.csv",
        "TEAM_SEASON_PATH": tmp_path / "team_season.csv",
        "GAMES_PATH": tmp_path / "games.csv",
    }
    for name, path in paths.items():
        path.write_text(f"season,value\n2025,{name}\n", encoding="utf-8")
        monkeypatch.setattr(refresh_current_season, name, path)
    status_path = tmp_path / "status.json"
    log_path = tmp_path / "success.txt"
    monkeypatch.setattr(refresh_current_season, "LIVE_SOURCE_REFRESH_STATUS_PATH", status_path)
    monkeypatch.setattr(refresh_current_season, "LIVE_SOURCE_REFRESH_LOG_PATH", log_path)
    before = {name: path.read_bytes() for name, path in paths.items()}

    def fail_download(season):
        raise ConnectionError("synthetic download failure")

    monkeypatch.setattr(refresh_current_season, "download_current_season_data", fail_download)
    ok, message, status = refresh_current_season.refresh_current_season(2026, force=True)

    assert ok is False
    assert "retained" in message
    assert status["ok"] is False
    assert all(path.read_bytes() == before[name] for name, path in paths.items())


def test_contract_export_keeps_active_context_without_large_history_blobs():
    contracts = pd.DataFrame(
        [
            {
                "player": "Active Player",
                "team": "LA",
                "is_active": True,
                "apy": 25_000_000,
                "gsis_id": "active",
                "season_history": "large historical payload",
                "contract_history": "large contract payload",
            },
            {
                "player": "Inactive Player",
                "team": "LA",
                "is_active": False,
                "apy": 1_000_000,
                "gsis_id": "inactive",
                "season_history": "unused",
                "contract_history": "unused",
            },
        ]
    )

    compact = live_source_refresh.compact_contracts_frame(contracts)

    assert compact["player"].tolist() == ["Active Player"]
    assert "apy" in compact.columns
    assert "gsis_id" in compact.columns
    assert "season_history" not in compact.columns
    assert "contract_history" not in compact.columns


def test_app_level_due_refresh_delegates_to_authoritative_pipeline(monkeypatch):
    calls = []

    def fake_refresh(season=None, force=False):
        calls.append((season, force))
        return False, "already current", {"skipped": True}

    monkeypatch.setattr(refresh_current_season, "refresh_current_season", fake_refresh)
    ok, message, status = refresh_current_season.refresh_current_season_if_due(2026)

    assert calls == [(2026, False)]
    assert ok is False
    assert message == "already current"
    assert status["skipped"] is True


def test_builder_and_dashboard_share_active_formula():
    weights = gini_metrics.BASELINE_WEIGHTS
    assert weights == {
        "Offense": 0.30,
        "Defense": 0.30,
        "Point Diff": 0.15,
        "Success Margin": 0.12,
        "Turnovers": 0.06,
        "Penalties": 0.02,
        "Schedule Strength": 0.05,
    }
    root = Path(__file__).resolve().parents[1]
    dashboard_source = (root / "pages" / "Gini_Dashboard.py").read_text(encoding="utf-8")
    builder_source = (root / "build_nfl_estat_data.py").read_text(encoding="utf-8")
    assert "from gini_metrics import BASELINE_WEIGHTS" in dashboard_source
    assert "apply_default_gini_scores" in builder_source

    components = pd.DataFrame(
        [{"off_z": 1, "def_z": 2, "pd_z": 3, "success_z": 4, "turnover_z": 5, "penalty_z": 6, "schedule_z": 7}]
    )
    expected = 100 + 15 * sum(weights[label] * components[gini_metrics.GINI_COMPONENT_COLUMNS[label]].iloc[0] for label in weights)
    assert gini_metrics.recompute_overall(components).iloc[0] == expected


def test_established_player_production_supersedes_draft_position():
    common = {
        "position": "WR",
        "depth_chart_position": "WR",
        "years_exp": 4,
        "production_score": 80.0,
        "prior_production_score": 94.0,
        "games_played": 1,
    }
    first_rounder = live_roster_scoring.calculate_player_score_blend({**common, "draft_number": 10})
    fifth_rounder = live_roster_scoring.calculate_player_score_blend({**common, "draft_number": 177})

    assert first_rounder == fifth_rounder
    assert first_rounder == 0.925 * 94.0 + 0.075 * 80.0


def test_current_player_production_phases_in_without_one_game_overreaction():
    base = {
        "position": "WR",
        "depth_chart_position": "WR",
        "draft_number": 177,
        "years_exp": 4,
        "production_score": 100.0,
        "prior_production_score": 60.0,
    }
    after_one = live_roster_scoring.calculate_player_score_blend({**base, "games_played": 1})
    after_ten = live_roster_scoring.calculate_player_score_blend({**base, "games_played": 10})

    assert after_one == 63.0
    assert after_ten == 90.0


def test_draft_capital_remains_a_fallback_for_players_without_nfl_production():
    common = {
        "position": "WR",
        "depth_chart_position": "WR",
        "years_exp": 0,
        "production_score": pd.NA,
        "prior_production_score": pd.NA,
    }
    early_pick = live_roster_scoring.calculate_player_score_blend({**common, "draft_number": 10})
    late_pick = live_roster_scoring.calculate_player_score_blend({**common, "draft_number": 220})
    assert early_pick > late_pick


def test_live_leaderboard_does_not_mix_current_and_prior_season_performance():
    performance = pd.DataFrame(
        [
            {"season": 2025, "team": "LA", "current_gini_score": 101.0},
            {"season": 2025, "team": "NE", "current_gini_score": 99.0},
            {"season": 2026, "team": "LA", "current_gini_score": 105.0},
        ]
    )
    in_season = select_live_performance_population(performance, 2026, ["LA", "NE"])
    preseason = select_live_performance_population(performance[performance["season"].eq(2025)], 2026, ["LA", "NE"])

    assert in_season[["season", "team"]].to_dict("records") == [{"season": 2026, "team": "LA"}]
    assert set(preseason["team"]) == {"LA", "NE"}
    assert set(preseason["season"]) == {2026}
    assert set(preseason["performance_source_season"]) == {2025}


def test_weekly_rank_movement_requires_the_same_team_population():
    current = pd.DataFrame({"team": ["LA", "NE", "SEA", "SF"]})
    compatible = pd.DataFrame({"team": ["SF", "SEA", "NE", "LA"]})
    incompatible = pd.DataFrame({"team": ["LA", "NE", "SEA", "SF", "BUF"]})

    assert comparable_ranked_populations(current, compatible) is True
    assert comparable_ranked_populations(current, incompatible) is False


def test_weekly_rank_movement_uses_the_previous_week_rank():
    assert weekly_rank_change(21, 16) == 5
    assert weekly_rank_change(8, 11) == -3
    assert weekly_rank_change(12, 12) == 0


def test_weekly_movement_baseline_excludes_the_current_period():
    current = (2026, 2, 2, "0000-00-00")
    assert is_prior_snapshot_period((2026, 2, 1, "0000-00-00"), current) is True
    assert is_prior_snapshot_period(current, current) is False


def test_latest_completed_regular_season_key_ignores_unplayed_and_postseason_games():
    games = pd.DataFrame(
        [
            {"season": 2026, "week": 1, "game_type": "REG", "home_score": 24, "away_score": 17},
            {"season": 2026, "week": 2, "game_type": "REG", "home_score": 20, "away_score": 21},
            {"season": 2026, "week": 3, "game_type": "REG", "home_score": None, "away_score": None},
            {"season": 2026, "week": 20, "game_type": "POST", "home_score": 30, "away_score": 27},
        ]
    )

    assert latest_completed_regular_season_key(games) == "2026-Week-02"


def test_replace_weekly_snapshot_removes_a_premature_baseline():
    history = pd.DataFrame(
        [
            {"snapshot_week": "2026-Week-01", "team": "DEN", "live_rank": 5},
            {"snapshot_week": "2026-Offseason", "team": "DEN", "live_rank": 7},
        ]
    )
    corrected = pd.DataFrame(
        [
            {"team": "DEN", "live_rank": 27, "live_market_score": 74.2},
            {"team": "SEA", "live_rank": 1, "live_market_score": 98.0},
        ]
    )

    result = replace_weekly_snapshot(
        history,
        corrected,
        "2026-Week-01",
        "2026-09-15T06:00:00-04:00",
    )

    week_one = result[result["snapshot_week"].eq("2026-Week-01")]
    assert week_one[["team", "live_rank"]].to_dict("records") == [
        {"team": "DEN", "live_rank": 27},
        {"team": "SEA", "live_rank": 1},
    ]


def test_weekly_roster_snapshots_are_deduplicated_and_departed_players_removed():
    roster = pd.DataFrame(
        [
            {"season": 2026, "week": 1, "team": "LA", "gsis_id": "p1", "full_name": "Player One", "position": "WR", "status": "ACT"},
            {"season": 2026, "week": 2, "team": "LA", "gsis_id": "p1", "full_name": "Player One", "position": "WR", "status": "ACT"},
            {"season": 2026, "week": 2, "team": "LA", "gsis_id": "p2", "full_name": "Player Two", "position": "WR", "status": "CUT"},
        ]
    )
    standardized = live_roster_scoring.standardize_roster_columns(roster, 2026)
    assert standardized["gsis_id"].tolist() == ["p1"]
    assert standardized["week"].tolist() == [2]


def test_deployed_tab_styling_is_version_tolerant_and_runtime_is_pinned():
    root = Path(__file__).resolve().parents[1]
    nav_source = (root / "site_nav.py").read_text(encoding="utf-8")
    requirements = (root / "requirements.txt").read_text(encoding="utf-8").splitlines()

    assert '.stTabs [role="tablist"]' in nav_source
    assert '.stTabs [role="tab"]' in nav_source
    assert 'div[data-testid="stTabs"] [role="tab"]' in nav_source
    assert "streamlit==1.58.0" in requirements
