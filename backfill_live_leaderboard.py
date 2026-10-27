"""Reconstruct missing leaderboard weeks locally without replacing captured history.

Run: python backfill_live_leaderboard.py --season 2026
Uses the actual page scoring pipeline with week-limited data, stopping before
the page can write its daily or weekly snapshots. No downloads are performed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from live_leaderboard_history import available_snapshot_keys, parse_week_key
from live_leaderboard_replay import RECONSTRUCTION_VERSION


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "data" / "live_leaderboard_reconstructed_history.csv"
RECEIPT = ROOT / "analysis" / "live_leaderboard_reconstruction.json"
SNAPSHOT_FIELDS = [
    "season", "team", "live_market_score", "live_rank", "current_gini_score",
    "performance_score", "raw_performance_score", "stabilized_performance_score",
    "performance_baseline_score", "performance_reliability", "roster_score",
    "projected_wins", "projected_wins_score", "projected_wins_low", "projected_wins_high",
    "most_likely_quadrant", "highest_quadrant_probability", "q1_probability",
    "q2_probability", "q3_probability", "q4_probability", "quadrant_probability_score",
    "games", "current_wins", "current_losses", "current_ties", "scored_games",
]


def missing_weeks(captured, games, season):
    regular = games[
        pd.to_numeric(games["season"]).eq(season)
        & games["season_type"].astype(str).str.upper().eq("REG")
    ].dropna(subset=["points_for", "points_against"])
    if regular.empty:
        return []
    latest = int(pd.to_numeric(regular["week"]).max())
    saved = {parse_week_key(key)[1] for key in available_snapshot_keys(captured, season)}
    # The latest week has a live view; only fill completed earlier periods.
    return [w for w in range(1, latest) if w not in saved]


def validate_reconstruction(frame, season, week, expected_teams):
    if len(frame) != len(expected_teams) or set(frame["team"]) != set(expected_teams):
        raise ValueError("Reconstruction does not contain every expected team exactly once.")
    if frame["team"].duplicated().any() or set(frame["live_rank"]) != set(range(1, len(frame) + 1)):
        raise ValueError("Reconstruction has duplicate teams or non-contiguous ranks.")
    numeric = ["live_market_score", "current_gini_score", "roster_score", "projected_wins"]
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Reconstruction has missing or non-finite scoring inputs.")
    expected = (
        .65 * frame["stabilized_performance_score"] + .10 * frame["roster_score"]
        + .15 * frame["projected_wins_score"] + .10 * frame["quadrant_probability_score"]
    ).clip(45, 120)
    if not np.allclose(frame["live_market_score"], expected):
        raise ValueError("Reconstruction does not reconcile to the production scoring weights.")
    if not frame["season"].eq(season).all() or frame["games"].gt(week).any():
        raise ValueError("Reconstruction contains games outside its week/season cutoff.")
    if not frame["scored_games"].eq(frame["current_wins"] + frame["current_losses"] + frame["current_ties"]).all():
        raise ValueError("Historical records do not reconcile to completed games.")


def replay_week(season, week, timeout):
    from streamlit.testing.v1 import AppTest

    key = f"{season}-Week-{week:02d}"
    previous = os.environ.get("LIVE_LEADERBOARD_REPLAY_WEEK")
    os.environ["LIVE_LEADERBOARD_REPLAY_WEEK"] = key
    try:
        app = AppTest.from_file(str(ROOT / "pages" / "Live_Leaderboard.py"))
        app.run(timeout=timeout)
        if app.exception:
            raise RuntimeError("; ".join(str(error.value) for error in app.exception))
        payload = app.session_state["live_leaderboard_replay_payload"]
    finally:
        if previous is None:
            os.environ.pop("LIVE_LEADERBOARD_REPLAY_WEEK", None)
        else:
            os.environ["LIVE_LEADERBOARD_REPLAY_WEEK"] = previous
    frame = payload["leaderboard"]
    snapshot = frame[[c for c in SNAPSHOT_FIELDS if c in frame]].copy()
    timestamp = datetime.now(timezone.utc).isoformat()
    snapshot["snapshot_week"] = key
    snapshot["snapshot_timestamp"] = timestamp
    snapshot["snapshot_window"] = timestamp
    metadata = payload["reconstruction_metadata"]
    for name in ["snapshot_kind", "reconstruction_version", "data_through_week", "data_cutoff"]:
        snapshot[name] = metadata[name]
    return snapshot, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args()
    captured_path = ROOT / "data" / "live_leaderboard_weekly_history.csv"
    captured = pd.read_csv(captured_path)
    games = pd.read_csv(ROOT / "data" / "team_game_estat.csv", low_memory=False)
    expected_teams = set(games.loc[pd.to_numeric(games["season"]).eq(args.season), "team"])
    weeks = missing_weeks(captured, games, args.season)
    existing = pd.read_csv(OUTPUT) if OUTPUT.exists() else pd.DataFrame()
    existing_keys = set(existing.get("snapshot_week", pd.Series(dtype=str)).astype(str))
    weeks = [w for w in weeks if f"{args.season}-Week-{w:02d}" not in existing_keys]
    if not weeks:
        print("No missing leaderboard weeks require reconstruction.", flush=True)
        return
    source_paths = [ROOT / "data" / "team_game_estat.csv", ROOT / "data" / "team_season_estat.csv"]
    source_paths += list((ROOT / "data" / "live_sources").glob("*.csv"))
    source_paths += [ROOT / "data" / "live_sources" / "predictive_model_projection_debug.json"]
    source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths if p.exists()}
    captured_hash = hashlib.sha256(captured_path.read_bytes()).hexdigest()
    frames, diagnostics = [], []
    for week in weeks:
        print(f"Reconstructing {args.season} Week {week}...", flush=True)
        snapshot, metadata = replay_week(args.season, week, args.timeout)
        validate_reconstruction(snapshot, args.season, week, expected_teams)
        frames.append(snapshot)
        diagnostics.append({**metadata, "week": week, "team_count": len(snapshot),
                            "top_five": snapshot.head(5)[["team", "live_rank", "live_market_score"]].to_dict("records")})
    if hashlib.sha256(captured_path.read_bytes()).hexdigest() != captured_hash:
        raise RuntimeError("Captured history changed while replaying; stop before publishing backfill.")
    combined = pd.concat([existing, *frames], ignore_index=True, sort=False)
    if combined.duplicated(["snapshot_week", "team"]).any():
        raise ValueError("Reconstructed archive has duplicate team-week entries.")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temp = OUTPUT.with_suffix(".csv.tmp")
    combined.to_csv(temp, index=False)
    os.replace(temp, OUTPUT)
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8")) if RECEIPT.exists() else {"runs": []}
    receipt["runs"].append({"season": args.season, "version": RECONSTRUCTION_VERSION,
                            "generated_at": datetime.now(timezone.utc).isoformat(),
                            "source_sha256": source_hashes, "captured_history_sha256": captured_hash,
                            "limitation": "Current methodology and revised inputs; not original captured rankings.",
                            "weeks": diagnostics})
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(f"Saved {len(frames) * len(expected_teams)} reconstructed team-week rows to {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
