# NFL EStat / Gini Dashboard

This is a Streamlit dashboard for Conor Mangini's NFL EStat / Gini project.

## Files

- `1_About_Me.py` — Streamlit entry page
- `pages/Gini_Dashboard.py` — main Gini dashboard page
- `build_nfl_estat_data.py` — rebuilds the summary data from the raw nflverse play-by-play files
- `requirements.txt` — packages to install
- `data/team_season_estat.csv` — one row per team per season
- `data/team_game_estat.csv` — one row per team per game
- `data/games_2005_onward.csv` — schedule/results file
- `data/roster_2026.csv` — 2026 roster file

## How to run

Put the files in one folder, then install the requirements:

```bash
pip install -r requirements.txt
```

Run the dashboard:

```bash
streamlit run 1_About_Me.py
```

## Recommended workflow

Use the raw play-by-play CSVs only to rebuild the smaller summary files. The dashboard should run from the summary files because they are much faster than the full 1M+ row combined CSV.

## Model summary

The default score combines:

- opponent-adjusted offensive EPA/play
- opponent-adjusted defensive EPA/play
- point differential per game
- success margin
- turnover margin
- penalty margin
- schedule strength

The dashboard also includes sliders so the user can test different weights.


