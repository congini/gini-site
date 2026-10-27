# Weekly rank movement audit — October 27, 2026

This records the local review before deployment approval. No source-data refreshes were performed during that review; deployment preserves the latest data already on main.

## Verified causes of the earlier failures

1. `get_latest_weekly_movement_snapshot()` returned `live_leaderboard_refresh_baseline.csv` before inspecting weekly history. That file currently contains 32 Week 1 rows. The selector did not verify that its period was the immediately preceding NFL week.
2. Its fallback selected the latest *available* prior period. Missing weeks could therefore silently turn a weekly comparison into a multiweek comparison.
3. `snapshot_live_leaderboard.py` ran before the source refresh and replaced that period's rows on each invocation. A pre-refresh view was being treated as a finalized weekly archive, and reruns could change the baseline.
4. Browser sessions could save a weekly snapshot using a calendar cutoff. The archived Week 3 capture is dated September 26, before that week's final scheduled game. It is not accepted as a finalized baseline.
5. Missing comparisons were converted to zero by copying current ranks into previous ranks. This made unavailable data look like a verified unchanged ranking.

## New behavior

- Formula: `previous_week_rank - current_rank`.
- The exact preceding NFL week and the same season are required. No fallback to older weeks, offseason periods, daily baselines, or reconstructed estimates.
- Both populations must contain all 32 NFL teams, with unique teams and ranks 1–32. Invalid comparisons are unavailable for the entire population.
- Live views use current ranks against the preceding frozen week. Historical views require finalized captures for both compared weeks.
- Available unchanged ranks display a gray dash. Missing evidence also displays a neutral dash, with an unavailable tooltip and distinct internal availability flag.
- The collapsed classification badge is replaced by small green/red SVG arrows and the absolute movement, or a gray dash. The existing badge slot, card geometry, fonts, logos, plus control, and expanded contents are retained.
- Raw Gini, Live Score, roster, wins, and record calculations are untouched.

## Snapshot lifecycle and persistence

The refresh workflow now invokes the finalizer **after** the authoritative refresh and validation. A week can be finalized only when every scheduled regular-season game through that week has a score, each game/team pair exists in processed team-game data, and the morning 6am ET processing cutoff after the final scheduled game has passed. A delayed run cannot relabel a different calendar week's model view.

The finalizer reads a capture-only payload, writes the weekly CSV atomically, and records `snapshot_status=finalized` and the NFL data cutoff. Once finalized, a week is immutable on daily reruns. Browser visits no longer create weekly captures. The existing workflow continues to commit the weekly archive as generated data when this change is eventually deployed, preserving it across Streamlit restarts and deployments.

Legacy captures are accepted only if their timestamp is after the completed week's cutoff and their population is valid. The original snapshot CSVs were preserved byte-for-byte during this task; no missing historical ranking was invented or promoted to a captured result.

## Current local archive

| NFL week | Captured rows | Accepted as a finalized baseline |
|---|---:|---|
| 1 | 32 | Yes |
| 2 | 32 | Yes |
| 3 | 32 | No — captured before the week finished |
| 4 | 0 | No — reconstructed display history only |
| 5 | 32 | Yes |
| 6 | 0 | No — reconstructed display history only |
| 7 | 0 | No — current live period |

Consequently, the current Week 7 view correctly shows unavailable dashes because an authoritative captured Week 6 snapshot is missing. Week 2 can show actual movement against Week 1. Reconstructed weeks remain browsable, but their estimates are not represented as actual weekly movement.

## Verification

56 tests passed using the deployed Streamlit 1.58 runtime. Coverage includes simulated rises/falls/ties, complete 32-team comparisons, score preservation, exact-week/season selection, historical views, missing/partial/duplicate snapshots, incomplete game data, morning cutoffs, frozen snapshot writes, reload persistence, daily reruns, refresh-baseline independence, native UI rendering, and week-button navigation.

Local preview: `http://127.0.0.1:8512/live-leaderboard`. Automatic source refresh is disabled only in the preview process, and its daily snapshot output is redirected to a local backup file.

## Deployment verification after approval

57 checks passed on the latest main checkout using Streamlit 1.58. The newer scheduled-refresh data and existing workflow schedule were preserved. Final desktop verification confirmed all 32 cards remain 76px high, with shared value/label centers and centered rank, logo, movement, and expand controls. Mobile layout remained unchanged at 390px.
