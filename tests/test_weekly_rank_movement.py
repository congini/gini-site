"""Weekly movement must be reproducible from frozen NFL weeks, not refresh timing."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import snapshot_live_leaderboard as snapshot_pipeline

from live_leaderboard_history import (
    NFL_TEAMS, calculate_weekly_rank_movement, finalizable_week_key,
    finalized_week_snapshot, freeze_weekly_snapshot,
    previous_finalized_week_snapshot, write_weekly_history,
)


def ranking(week=6, season=2026, status="finalized"):
    return pd.DataFrame({
        "team": sorted(NFL_TEAMS), "season": season, "live_rank": range(1, 33),
        "live_market_score": [100 - i / 2 for i in range(32)],
        "current_gini_score": 105.0, "roster_score": 84.0, "projected_wins": 9.5,
        "snapshot_week": f"{season}-Week-{week:02d}",
        "snapshot_timestamp": "2026-11-10T11:00:00+00:00",
        "snapshot_status": status, "snapshot_kind": "captured",
    })


def schedule_and_processed(weeks=2):
    teams = sorted(NFL_TEAMS)
    schedule, processed = [], []
    for week in range(1, weeks + 1):
        day = (pd.Timestamp("2026-09-14") + pd.Timedelta(days=7 * (week - 1))).date().isoformat()
        for index in range(0, 32, 2):
            game_id = f"2026_{week:02d}_{index}"
            home, away = teams[index:index + 2]
            schedule.append(dict(season=2026, week=week, game_id=game_id,
                                 game_type="REG", gameday=day, home_team=home,
                                 away_team=away, home_score=21, away_score=17))
            for team, opponent, scored, allowed in [(home, away, 21, 17), (away, home, 17, 21)]:
                processed.append(dict(season=2026, week=week, game_id=game_id,
                                      season_type="REG", team=team, opponent=opponent,
                                      points_for=scored, points_against=allowed))
    return pd.DataFrame(schedule), pd.DataFrame(processed)


def test_all_32_teams_use_previous_rank_minus_current_without_changing_scores():
    prior = ranking()
    current = ranking(7)
    current.loc[[1, 4], "live_rank"] = [5, 2]
    before = current.copy(deep=True)
    result = calculate_weekly_rank_movement(current, prior, "2026-Week-07")
    assert result.loc[4, "weekly_rank_change"] == 3  # fifth to second
    assert result.loc[1, "weekly_rank_change"] == -3  # second to fifth
    assert result.loc[2, "weekly_rank_change"] == 0
    assert result["weekly_movement_available"].all()
    assert result["weekly_rank_change"].sum() == 0
    pd.testing.assert_frame_equal(result[before.columns], before)


@pytest.mark.parametrize("prior", [pd.DataFrame(), ranking(5), ranking(6, 2025), ranking(7)])
def test_missing_exact_previous_week_is_unavailable_instead_of_zero_or_an_older_week(prior):
    result = calculate_weekly_rank_movement(ranking(7), prior, "2026-Week-07")
    assert result["weekly_rank_change"].isna().all()
    assert not result["weekly_movement_available"].any()


def test_first_week_never_uses_last_season_or_offseason():
    assert previous_finalized_week_snapshot(ranking(18, 2025), "2026-Week-01").empty


@pytest.mark.parametrize("failure", ["partial", "duplicate_team", "duplicate_rank", "wrong_season", "replay", "provisional"])
def test_invalid_previous_population_disables_comparison_for_every_team(failure):
    prior = ranking()
    if failure == "partial":
        prior = prior.iloc[:-1]
    elif failure == "duplicate_team":
        prior.loc[1, "team"] = prior.loc[0, "team"]
    elif failure == "duplicate_rank":
        prior.loc[1, "live_rank"] = 1
    elif failure == "wrong_season":
        prior.loc[1, "season"] = 2025
    elif failure == "replay":
        prior["snapshot_kind"] = "reconstructed"
    else:
        prior["snapshot_status"] = "provisional"
    result = calculate_weekly_rank_movement(ranking(7), prior, "2026-Week-07")
    assert result["weekly_rank_change"].isna().all()
    assert not result["weekly_movement_available"].any()


def test_daily_refresh_and_current_week_snapshots_cannot_reset_the_frozen_comparison():
    history = ranking()
    current = ranking(7)
    current.loc[[1, 4], "live_rank"] = [5, 2]
    before = calculate_weekly_rank_movement(current, history, "2026-Week-07")
    daily = current.copy()
    daily["snapshot_timestamp"] = "2026-11-11T11:00:00+00:00"
    after = calculate_weekly_rank_movement(current, pd.concat([history, daily]), "2026-Week-07")
    pd.testing.assert_series_equal(before["weekly_rank_change"], after["weekly_rank_change"])


def test_historical_week_uses_its_own_preceding_week_not_the_live_baseline():
    history = pd.concat([ranking(5), ranking(6), ranking(7)], ignore_index=True)
    historic = ranking(6)
    historic.loc[[0, 3], "live_rank"] = [4, 1]
    result = calculate_weekly_rank_movement(historic, history, "2026-Week-06", historical=True)
    assert result.loc[3, "weekly_rank_change"] == 3
    assert result["weekly_comparison_key"].eq("2026-Week-05").all()


def test_estimated_historical_view_cannot_present_estimates_as_actual_movement():
    previous = ranking(5)
    result = calculate_weekly_rank_movement(ranking(6), previous, "2026-Week-06", historical=True)
    assert not result["weekly_movement_available"].any()


def test_legacy_capture_must_have_been_saved_after_the_week_was_complete():
    schedule, _ = schedule_and_processed()
    legacy = ranking(2).drop(columns=["snapshot_status"])
    legacy["snapshot_timestamp"] = "2026-09-20T16:00:00-04:00"
    assert finalized_week_snapshot(legacy, "2026-Week-02", schedule).empty
    legacy["snapshot_timestamp"] = "2026-09-22T06:05:00-04:00"
    assert len(finalized_week_snapshot(legacy, "2026-Week-02", schedule)) == 32


def test_finalization_waits_for_all_games_processed_and_the_morning_cutoff():
    schedule, processed = schedule_and_processed()
    assert finalizable_week_key(schedule, processed, "2026-09-22T06:00:00-04:00") == "2026-Week-02"
    assert finalizable_week_key(schedule, processed, "2026-09-22T05:59:00-04:00") is None
    assert finalizable_week_key(schedule, processed.iloc[:-1], "2026-09-22T07:00:00-04:00") is None
    schedule.loc[schedule.index[-1], "away_score"] = None
    assert finalizable_week_key(schedule, processed, "2026-09-22T07:00:00-04:00") is None


def test_finalized_snapshots_persist_and_daily_reruns_cannot_rewrite_them(tmp_path):
    schedule, processed = schedule_and_processed()
    original = ranking(2)
    frozen, saved = freeze_weekly_snapshot(pd.DataFrame(), original, "2026-Week-02",
                                          schedule, processed, "2026-09-22T07:00:00-04:00")
    assert saved
    path = tmp_path / "live_leaderboard_weekly_history.csv"
    write_weekly_history(frozen, path)
    first_bytes = path.read_bytes()
    restored = pd.read_csv(path)
    changed = original.copy()
    changed.loc[[0, 1], "live_rank"] = [2, 1]
    rerun, saved = freeze_weekly_snapshot(restored, changed, "2026-Week-02",
                                         schedule, processed, "2026-09-23T07:00:00-04:00")
    assert not saved
    pd.testing.assert_frame_equal(restored, rerun)
    assert len(finalized_week_snapshot(restored, "2026-Week-02", schedule)) == 32
    assert path.read_bytes() == first_bytes


def test_a_provisional_week_can_be_finalized_once_without_duplicate_ranks():
    schedule, processed = schedule_and_processed()
    premature = ranking(2, status="provisional")
    frozen, saved = freeze_weekly_snapshot(premature, ranking(2), "2026-Week-02",
                                          schedule, processed, "2026-09-22T07:00:00-04:00")
    assert saved and len(frozen) == 32
    assert frozen["snapshot_status"].eq("finalized").all()


def test_post_refresh_capture_saves_once_and_leaves_refresh_baseline_untouched(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    schedule, processed = schedule_and_processed()
    (tmp_path / "data").mkdir()
    schedule.to_csv(tmp_path / "games.csv", index=False)
    processed.to_csv(tmp_path / "data/team_game_estat.csv", index=False)
    archive = tmp_path / "weekly.csv"
    baseline = tmp_path / "refresh_baseline.csv"
    baseline.write_text("existing daily baseline", encoding="utf-8")
    monkeypatch.setattr(snapshot_pipeline, "APP_DIR", tmp_path)
    monkeypatch.setattr(snapshot_pipeline, "GAMES_PATH", tmp_path / "games.csv")
    monkeypatch.setattr(snapshot_pipeline, "WEEKLY_HISTORY_PATH", archive)
    monkeypatch.setattr(snapshot_pipeline, "REFRESH_BASELINE_PATH", baseline)
    class Clock:
        @staticmethod
        def now(zone):
            return pd.Timestamp("2026-09-22T12:00:00Z").to_pydatetime()
    monkeypatch.setattr(snapshot_pipeline, "datetime", Clock)
    calls = []
    def capture(path):
        calls.append(path)
        assert snapshot_pipeline.os.environ["LIVE_LEADERBOARD_CAPTURE_ONLY"] == "1"
        payload = {"weekly_snapshot_key": "2026-Week-02", "leaderboard": ranking(2)}
        return SimpleNamespace(run=lambda **kwargs: SimpleNamespace(exception=[], session_state={"live_leaderboard_capture_payload": payload}))
    monkeypatch.setattr(AppTest, "from_file", capture)
    monkeypatch.setenv("LIVE_LEADERBOARD_CAPTURE_ONLY", "restore-this-value")
    assert snapshot_pipeline.build_prior_week_snapshot() == ("2026-Week-02", 32)
    saved = archive.read_bytes()
    assert snapshot_pipeline.build_prior_week_snapshot() == ("2026-Week-02", 32)
    assert archive.read_bytes() == saved
    assert len(calls) == 1
    assert baseline.read_text(encoding="utf-8") == "existing daily baseline"
    assert snapshot_pipeline.os.environ["LIVE_LEADERBOARD_CAPTURE_ONLY"] == "restore-this-value"


def test_legacy_page_entry_points_never_read_the_refresh_baseline():
    source = Path("pages/Live_Leaderboard.py").read_text(encoding="utf-8")
    names = {"get_latest_weekly_movement_snapshot", "get_latest_prior_week_snapshot",
             "calculate_weekly_movement", "calculate_rank_movement",
             "should_save_weekly_snapshot", "save_weekly_snapshot_if_needed"}
    tree = ast.Module(body=[n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
    namespace = dict(pd=pd, DATA_DIR=Path("data"),
                     previous_finalized_week_snapshot=previous_finalized_week_snapshot,
                     calculate_weekly_rank_movement=calculate_weekly_rank_movement,
                     load_weekly_history=lambda: (ranking(), True),
                     safe_read_csv=lambda path: (pd.DataFrame(), ""))
    def reject_baseline():
        raise AssertionError("A refresh baseline must never be read for weekly movement")
    namespace["load_refresh_baseline"] = reject_baseline
    exec(compile(tree, str(Path("pages/Live_Leaderboard.py")), "exec"), namespace)
    for name in ("get_latest_weekly_movement_snapshot", "get_latest_prior_week_snapshot"):
        snapshot, found = namespace[name]("2026-Week-07")
        assert found and snapshot["snapshot_week"].eq("2026-Week-06").all()
    assert namespace["save_weekly_snapshot_if_needed"](ranking(7), "2026-Week-07") == (False, "2026-Week-07")
