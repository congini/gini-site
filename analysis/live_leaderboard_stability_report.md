# Live Leaderboard stability and responsive-layout review

This report covers only the Live Leaderboard layer. The raw Gini Dashboard and
its season-to-date methodology remain unchanged.

## What caused the volatility

The production blend is confirmed as 65% performance, 10% roster, 15%
projected wins, and 10% quadrant probability. In the latest saved current-season
transition (Week 3 to Week 5), the performance contribution explained nearly
all of the largest Live Score changes. For example, New Orleans moved -10.57
points overall, of which -9.82 came from performance; Green Bay moved +9.59,
of which +9.12 came from performance.

Across season-to-date weekly reconstructions from 2005 through 2026, mean
absolute raw Gini component motion was:

| Driver | Mean absolute Gini points | Share of component motion |
|---|---:|---:|
| EPA defense | 1.252 | 32.2% |
| EPA offense | 1.120 | 28.8% |
| Point differential | 0.483 | 12.4% |
| Success margin | 0.396 | 10.2% |
| Schedule strength | 0.318 | 8.2% |
| Turnover margin | 0.241 | 6.2% |
| Penalty margin | 0.081 | 2.1% |

Opponent-adjusted offensive and defensive EPA therefore account for about 61%
of week-to-week component motion. Point differential and success margin are the
next largest contributors. Turnovers and penalties are not the main source.
League-relative normalization also makes these signals move more sharply when
only a few games exist.

## Selected stabilization

The Live Leaderboard now retains both the Raw Performance Score and a separate
Live Performance Score:

```text
prior baseline = 85 + 0.50 × (prior-season final performance - 85)
reliability = games played / (games played + 2)
Live Performance = reliability × Raw Performance
                 + (1 - reliability) × prior baseline
```

The current-season share is 33% after one game, 71% after five, 83% after ten,
and 89% after 17. Missing prior-season data falls back to the league center of
85. The 65/10/15/10 top-level weights are unchanged.

`k=2` was selected over the marginally higher-correlation `k=3` result because
their future-win correlations were effectively tied (0.6290 versus 0.6293),
while `k=2` hands control back to current-season evidence faster and reaches 89%
current weight by Week 17 instead of 85%.

## Historical test results

The controlled historical backtest holds the non-performance 35% at the league
center because complete week-by-week roster, projected-wins, and quadrant
snapshots do not exist for every season. This isolates the only production
change being proposed: the 65% performance input.

| Measure | Raw | Stabilized | Result |
|---|---:|---:|---:|
| Average weekly score movement | 1.542 | 1.051 | 31.8% lower |
| 95th-percentile score movement | 5.038 | 3.063 | 39.2% lower |
| Average weekly rank movement | 2.366 | 2.098 | 11.4% lower |
| 95th-percentile rank movement | 8 | 7 | 1 place lower |
| Future-win rank correlation | 0.602 | 0.629 | improved |
| End-of-season quality correlation | 0.706 | 0.742 | improved |
| Bad-team early top-eight rate | 4.61% | 4.10% | improved |
| Good-team early top-eight rate | 51.55% | 53.44% | improved |

The stabilized version reduces isolated jumps without making genuinely strong
teams less identifiable or artificially propping up poor teams.

## Diagnostic examples

The following examples show the previous board followed by the new raw result
and the result with the selected stabilization. Scores are the controlled
historical Live Score used in the backtest.

| Season / week | Team | Games | Previous rank | Raw new rank | Stabilized new rank | Previous score | Raw new score | Stabilized new score |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2016 / 2 | SEA | 2 | 7 | 28 | 16 | 89.39 | 77.85 | 84.34 |
| 2018 / 2 | IND | 2 | 28 | 7 | 18 | 80.02 | 89.50 | 84.81 |
| 2013 / 2 | SF | 2 | 11 | 26 | 16 | 86.70 | 80.55 | 85.32 |
| 2025 / 3 | CAR | 3 | 23 | 8 | 18 | 81.64 | 89.26 | 85.10 |
| 2010 / 2 | NE | 2 | 7 | 20 | 11 | 90.56 | 83.29 | 86.33 |

These are intentionally large one-week changes. Stabilization still moves every
team in the correct direction, but it does not allow one result to erase the
prior body of evidence.

## Responsive verification

The Live Leaderboard was visually checked in the local Streamlit app at the
requested viewports:

| Viewport | Horizontal overflow | Hero behavior | Team-entry behavior |
|---|---|---|---|
| 1440px desktop | none | six-card desktop row | compact horizontal rows |
| 768px tablet | none | headline cards reflow | intentional two-tier grid |
| 430px mobile | none | two-column stacked cards | compact five-section card |
| 390px mobile | none | two-column stacked cards | compact five-section card |

The hero is now ordinary responsive page content rather than a fixed-height
iframe, eliminating clipped headline cards. Mobile entries preserve rank, logo,
team, Live Score, movement, Gini, roster, projected wins, status, and quadrant
information without horizontal scrolling. Team Detail metrics and quadrant
cards use a two-column mobile grid.
