# Live Leaderboard design QA

- Final result: passed
- Route: `http://localhost:8502/live-leaderboard`
- Desktop viewport: 1540 x 900 CSS pixels
- Narrow viewport checked: approximately 510 x 632 CSS pixels
- Density assumption: standard desktop browser scaling

## References

- Hero reference: `C:/Users/cmang/AppData/Local/Temp/codex-clipboard-e6ac32ee-ff8e-438f-b7bf-1ffacf3ffff5.png`
- Details-expander reference: `C:/Users/cmang/AppData/Local/Temp/codex-clipboard-6d28fdbd-5690-409d-8515-b920a0ea5510.png`
- Movement reference: `C:/Users/cmang/AppData/Local/Temp/codex-clipboard-d0723455-fc80-47bd-a45a-0cfd45253a99.png`
- Implementation capture: Codex in-app browser capture of `http://localhost:8502/live-leaderboard` (the browser API did not expose a filesystem-backed screenshot path).

## Findings

- The Last Page Load card is absent.
- The remaining six hero cards fill the available row at desktop width and reflow cleanly at the narrow breakpoint.
- The Roster + Data Source Details expander is absent from the rendered DOM.
- Movement arrows use rank movement only: green up, red down, and the absolute number of places moved.
- Denver renders as `↑ 16`, matching its move from 27th in the corrected Week 1 baseline to 11th in Week 2.
- Week 1 and Week 2 baselines each contain 32 rows and 32 unique teams.

## Comparison history

1. Initial implementation removed the requested hero/detail elements and changed movement to rank deltas.
2. Visual QA exposed Denver as `↓ 6`; data tracing showed the stored Week 1 snapshot had been captured before Week 1 was complete.
3. Reconstructed the final Week 1 leaderboard from the validated September 15 data, replaced the premature snapshot, and added a pre-refresh baseline step to the scheduled workflow.
4. Final desktop and narrow checks passed, including DOM assertions for the removed elements and Denver's corrected `↑ 16` movement.
