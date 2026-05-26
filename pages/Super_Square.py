from pathlib import Path
from io import BytesIO
import base64
import sys

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from PIL import Image, ImageOps


st.set_page_config(
    page_title="Super Square",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------
# PATH SETUP
# -----------------------------

APP_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = APP_DIR / "data"

if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from site_nav import render_top_nav


PRIMARY = "#F15A24"
SECONDARY = "#0073B7"


SUPER_BOWL_WINNERS = {
    2005: "PIT",
    2006: "IND",
    2007: "NYG",
    2008: "PIT",
    2009: "NO",
    2010: "GB",
    2011: "NYG",
    2012: "BAL",
    2013: "SEA",
    2014: "NE",
    2015: "DEN",
    2016: "NE",
    2017: "PHI",
    2018: "NE",
    2019: "KC",
    2020: "TB",
    2021: "LA",
    2022: "KC",
    2023: "KC",
    2024: "PHI",
    2025: "SEA",
}


@st.cache_data
def load_super_square_data():
    team_season = pd.read_csv(DATA_DIR / "team_season_estat.csv")
    assets_path = DATA_DIR / "teams_colors_logos.csv"
    team_assets = pd.read_csv(assets_path) if assets_path.exists() else pd.DataFrame()
    return team_season, team_assets


@st.cache_data(show_spinner=False)
def logo_url_to_data_uri(url, grayscale=False):
    if not isinstance(url, str) or not url.startswith("http"):
        return ""

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        image = Image.open(BytesIO(response.content)).convert("RGBA")

        if grayscale:
            alpha = image.getchannel("A")
            image = ImageOps.grayscale(image).convert("RGBA")
            image.putalpha(alpha)

        buffer = BytesIO()
        image.save(buffer, format="PNG")

        encoded = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{encoded}"

    except Exception:
        return ""


def build_team_lookups(team_assets):
    logo_lookup = {}
    name_lookup = {}

    if team_assets.empty or "team_abbr" not in team_assets.columns:
        return logo_lookup, name_lookup

    for _, row in team_assets.dropna(subset=["team_abbr"]).iterrows():
        abbr = str(row.get("team_abbr"))

        if "team_logo_espn" in team_assets.columns and pd.notna(row.get("team_logo_espn")):
            logo_lookup[abbr] = str(row.get("team_logo_espn"))

        if "team_name" in team_assets.columns and pd.notna(row.get("team_name")):
            name_lookup[abbr] = str(row.get("team_name"))
        else:
            name_lookup[abbr] = abbr

    return logo_lookup, name_lookup


def get_score_column(df):
    if "custom_estat" in df.columns:
        return "custom_estat"
    if "overall_estat" in df.columns:
        return "overall_estat"
    raise ValueError("No Gini/EStat score column found.")


# -----------------------------
# LOAD DATA
# -----------------------------

team_season, team_assets = load_super_square_data()
logo_lookup, name_lookup = build_team_lookups(team_assets)


# -----------------------------
# PAGE CSS
# -----------------------------

st.markdown(
    f"""
<style>
.stApp,
[data-testid="stAppViewContainer"] {{
    background:
        radial-gradient(circle at 8% 18%, rgba(0,115,183,0.11), transparent 34%),
        radial-gradient(circle at 92% 22%, rgba(241,90,36,0.10), transparent 36%),
        #F6F8FB !important;
    overflow-x: hidden;
}}

.block-container {{
    position: relative;
    z-index: 1;
    padding-top: 0.35rem !important;
    padding-bottom: 2.8rem !important;
}}

.super-page {{
    position: relative;
    z-index: 1;
    max-width: 1500px;
    margin: -1.85rem auto 0 auto;
    padding: 0 1.5rem 3rem 1.5rem;
}}

.super-hero {{
    position: relative;
    overflow: hidden;
    display: grid;
    grid-template-columns: minmax(0, 0.95fr) minmax(600px, 1.05fr);
    gap: 1.5rem;
    align-items: center;
    margin-top: 0.35rem;
    margin-bottom: 1rem;
    padding: 2.35rem 2.6rem;
    border-radius: 20px;
    background:
        radial-gradient(circle at 88% 20%, rgba(241,90,36,0.28), transparent 32%),
        radial-gradient(circle at 70% 92%, rgba(0,115,183,0.22), transparent 30%),
        linear-gradient(135deg, #07111f 0%, #172033 56%, #273447 100%);
    border: 1px solid rgba(255,255,255,0.14);
    box-shadow: 0 22px 46px rgba(15, 23, 42, 0.16);
}}

.super-hero::before {{
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    height: 5px;
    width: 100%;
    background: linear-gradient(90deg, {PRIMARY}, {SECONDARY});
}}

.super-kicker {{
    color: {PRIMARY};
    font-size: 0.8rem;
    font-weight: 950;
    text-transform: uppercase;
    letter-spacing: 0.10em;
    margin-bottom: 0.55rem;
}}

.super-title {{
    color: #FFFFFF;
    font-size: clamp(2.15rem, 4vw, 3.2rem);
    line-height: 1;
    font-weight: 950;
    margin-bottom: 0.72rem;
}}

.super-subtitle {{
    max-width: 920px;
    color: rgba(255,255,255,0.84);
    font-size: 0.98rem;
    line-height: 1.68;
}}

.super-proof-card {{
    position: relative;
    justify-self: stretch;
    width: 100%;
    max-width: none;
    min-height: 150px;
    padding: 1.35rem 1.7rem;
    border-radius: 17px;
    background:
        linear-gradient(135deg, rgba(255,255,255,0.18), rgba(255,255,255,0.075)),
        linear-gradient(135deg, rgba(7,17,31,0.08), rgba(255,255,255,0.03));
    border: 1px solid rgba(255,255,255,0.24);
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.14),
        0 16px 34px rgba(0,0,0,0.13);
    backdrop-filter: blur(12px);
    overflow: hidden;
}}

.super-proof-card::before {{
    content: "";
    position: absolute;
    left: 0;
    top: 0;
    width: 5px;
    height: 100%;
    background: linear-gradient(180deg, {PRIMARY}, {SECONDARY});
    opacity: 0.95;
}}

.super-proof-card::after {{
    content: "";
    position: absolute;
    right: -55px;
    bottom: -65px;
    width: 150px;
    height: 150px;
    border-radius: 999px;
    background: rgba(255,255,255,0.055);
    pointer-events: none;
}}

.super-proof-label {{
    color: rgba(255,255,255,0.66);
    font-size: 0.72rem;
    line-height: 1.2;
    font-weight: 950;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.45rem;
}}

.super-proof-number {{
    color: #FFFFFF;
    font-size: 2.05rem;
    line-height: 1;
    font-weight: 950;
    letter-spacing: -0.04em;
    margin-bottom: 0.4rem;
}}

.super-proof-text {{
    color: rgba(255,255,255,0.9);
    font-size: 1rem;
    line-height: 1.6;
    font-weight: 750;
    max-width: none;
    width: 100%;
}}

.super-proof-text span {{
    color: #FFFFFF;
    font-weight: 950;
}}

.super-proof-label,
.super-proof-number,
.super-proof-text {{
    position: relative;
    z-index: 1;
}}

.super-controls-card-marker,
.super-reveal-field-marker {{
    display: none;
}}

div[data-testid="stHorizontalBlock"]:has(.super-controls-card-marker) {{
    display: grid;
    grid-template-columns: minmax(360px, 720px) minmax(260px, 340px);
    align-items: start;
    justify-content: center;
    gap: 1.25rem;

    width: min(100%, 1120px) !important;
    margin: 0.85rem auto 1rem auto;
    padding: 0.9rem 1.05rem 1rem 1.05rem;

    border-radius: 16px;
    background: rgba(255,255,255,0.92);
    border: 1px solid rgba(15, 23, 42, 0.10);
    border-left: 5px solid {PRIMARY};
    box-shadow: 0 12px 28px rgba(15, 23, 42, 0.065);
    backdrop-filter: blur(9px);
    overflow: hidden;
    transform: translateX(5px);
}}

div[data-testid="stHorizontalBlock"]:has(.super-controls-card-marker) > div {{
    width: 100% !important;
    min-width: 0 !important;
}}

div[data-testid="stHorizontalBlock"]:has(.super-controls-card-marker) > div:has(.super-reveal-field-marker) {{
    padding-top: 2.53rem !important;
}}

div[data-testid="stHorizontalBlock"]:has(.super-controls-card-marker) div[data-testid="stSelectbox"] {{
    margin-bottom: 0 !important;
}}

div[data-testid="stHorizontalBlock"]:has(.super-controls-card-marker) div[data-testid="stSelectbox"] > label {{
    margin-bottom: 0.3rem !important;
}}

.super-controls-title {{
    color: #07111f;
    font-size: 1.05rem;
    line-height: 1.15;
    font-weight: 950;
    letter-spacing: -0.02em;
    margin: 0 0 0.5rem 0;
}}

div[data-testid="stHorizontalBlock"]:has(.super-controls-card-marker) div[data-testid="stCheckbox"] {{
    margin: 0 !important;
    padding: 0 !important;
    min-height: 40px !important;
    display: flex !important;
    align-items: center !important;
}}

div[data-testid="stHorizontalBlock"]:has(.super-controls-card-marker) div[data-testid="stCheckbox"] label {{
    margin: 0 !important;
    padding: 0 !important;
    min-height: 40px !important;
    display: flex !important;
    align-items: center !important;
    gap: 0.34rem !important;
}}

div[data-testid="stHorizontalBlock"]:has(.super-controls-card-marker) div[data-testid="stCheckbox"] label > span:first-child {{
    transform: translateY(-2px);
}}

div[data-testid="stHorizontalBlock"]:has(.super-controls-card-marker) div[data-testid="stCheckbox"] label > div {{
    padding-left: 0 !important;
}}

div[data-testid="stHorizontalBlock"]:has(.super-controls-card-marker) div[data-testid="stCheckbox"] p {{
    margin: 0 !important;
    color: #07111f !important;
    font-size: 0.95rem !important;
    font-weight: 850 !important;
    line-height: 1.2 !important;
}}

.super-info-box {{
    margin: 1rem auto 1.2rem auto;
    width: min(100%, 1120px);
    padding: 1.05rem 1.15rem;
    border-radius: 16px;
    background: rgba(255,255,255,0.88);
    border: 1px solid rgba(15, 23, 42, 0.10);
    border-left: 5px solid {SECONDARY};
    box-shadow: 0 12px 28px rgba(15, 23, 42, 0.055);
    color: #334155;
    font-size: 0.94rem;
    line-height: 1.65;
}}

.super-info-title {{
    color: #07111f;
    font-size: 1.02rem;
    font-weight: 950;
    margin-bottom: 0.45rem;
}}

.super-info-box b {{
    color: #07111f;
}}

@media screen and (max-width: 900px) {{
    div[data-testid="stHorizontalBlock"]:has(.super-controls-card-marker) {{
        grid-template-columns: 1fr;
        width: 100% !important;
        gap: 0.75rem;
        transform: none;
    }}

    div[data-testid="stHorizontalBlock"]:has(.super-controls-card-marker) div[data-testid="stCheckbox"] {{
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }}
}}

.square-summary {{
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 0.75rem;
    flex-wrap: wrap;
    width: min(100%, 1120px);
    margin: 0.6rem auto 1rem auto;
}}

.super-note {{
    width: min(100%, 1120px);
    margin: 0.85rem auto 0.75rem auto;
    color: #64748B;
    font-size: 0.86rem;
    line-height: 1.55;
    text-align: center;
}}

.square-pill {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-height: 36px;
    padding: 0.55rem 0.8rem;
    border-radius: 999px;
    background: rgba(255,255,255,0.86);
    border: 1px solid rgba(15,23,42,0.10);
    color: #07111f;
    font-size: 0.84rem;
    font-weight: 850;
    box-shadow: 0 8px 18px rgba(15, 23, 42, 0.045);
}}

.square-pill span {{
    color: {PRIMARY};
    font-weight: 950;
}}

div[data-testid="stSelectbox"] label p,
div[data-testid="stCheckbox"] label p {{
    color: #07111f !important;
    font-size: 0.95rem !important;
    font-weight: 850 !important;
}}

div[data-testid="stSelectbox"] [data-baseweb="select"] > div {{
    background-color: #FFFFFF !important;
    border: 1px solid #D1D5DB !important;
    border-radius: 10px !important;
    box-shadow: 0 6px 16px rgba(15, 23, 42, 0.045) !important;
}}

div[data-testid="stPlotlyChart"] {{
    background: #FFFFFF !important;
    border: 1px solid rgba(15, 23, 42, 0.10);
    border-radius: 18px;
    padding: 0.75rem 0.9rem;
    box-shadow: 0 16px 36px rgba(15, 23, 42, 0.075);
}}

.super-expander-note {{
    color: #334155;
    font-size: 0.95rem;
    line-height: 1.65;
}}

@media screen and (max-width: 900px) {{
    .super-hero {{
        grid-template-columns: 1fr;
        padding: 2rem;
    }}

    .super-proof-card {{
        justify-self: start;
    }}

    .super-page {{
        padding-left: 1rem;
        padding-right: 1rem;
    }}
}}
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# NAV + HERO
# -----------------------------

render_top_nav("Super Square", PRIMARY, SECONDARY)

st.markdown('<div class="super-page">', unsafe_allow_html=True)

st.markdown(
"""
<div class="super-hero">
<div class="super-hero-copy">
<div class="super-kicker">Historical Super Bowl winner profile</div>
<div class="super-title">Super Square</div>
<div class="super-subtitle">
The Super Square is a selective regular-season contender zone built from the traits that consistently show up in championship teams. Across the last 20 completed seasons in this dashboard, every Super Bowl winner came from inside the Super Square.
</div>
</div>

<div class="super-proof-card">
<div class="super-proof-label">Historical Signal</div>
<div class="super-proof-number">20 Seasons</div>
<div class="super-proof-text">
Over the last 20 completed seasons, every Super Bowl champion came from inside the <span>Super Square</span>.
</div>
</div>
</div>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# CONTROLS
# -----------------------------

available_seasons = sorted(
    [
        int(season)
        for season in team_season["season"].dropna().unique()
        if 2005 <= int(season) <= 2025
    ],
    reverse=True,
)

season_col, reveal_col = st.columns([1, 1], gap="large")

with season_col:
    st.markdown(
        '<div class="super-controls-card-marker"></div><div class="super-controls-title">Super Square Setup</div>',
        unsafe_allow_html=True,
    )

    selected_season = st.selectbox(
        "Season",
        available_seasons,
        index=0,
        key="super_square_selected_season",
        width="stretch",
    )

with reveal_col:
    st.markdown('<div class="super-reveal-field-marker"></div>', unsafe_allow_html=True)

    reveal_winner = st.checkbox(
        "Reveal Super Bowl winner",
        value=False,
        key="super_square_reveal_winner",
        help="Greys out every logo except the team that won the Super Bowl after this regular season.",
    )

st.markdown(
    """
<div class="super-note">
    The 2005 NFL season maps to the Super Bowl played in 2006. This page uses completed regular seasons from 2005 through 2025.
</div>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# SUPER SQUARE LOGIC
# -----------------------------

season_df = team_season[team_season["season"] == selected_season].copy()

try:
    score_col = get_score_column(season_df)
except ValueError as err:
    st.error(str(err))
    st.stop()

required_cols = [score_col, "point_diff_per_game", "offense_estat", "defense_estat"]
missing_cols = [column for column in required_cols if column not in season_df.columns]

if missing_cols:
    st.error(f"Missing required columns for Super Square: {missing_cols}")
    st.stop()

season_df["Team"] = season_df["team"]
season_df["Team Name"] = season_df["Team"].map(name_lookup).fillna(season_df["Team"])
season_df["Gini Score"] = pd.to_numeric(season_df[score_col], errors="coerce")
season_df["Point Differential per Game"] = pd.to_numeric(
    season_df["point_diff_per_game"],
    errors="coerce",
)
season_df["Offense EStat"] = pd.to_numeric(season_df["offense_estat"], errors="coerce")
season_df["Defense EStat"] = pd.to_numeric(season_df["defense_estat"], errors="coerce")

season_df = season_df.dropna(
    subset=[
        "Gini Score",
        "Point Differential per Game",
        "Offense EStat",
        "Defense EStat",
    ]
).copy()

season_df["gini_rank"] = season_df["Gini Score"].rank(
    ascending=False,
    method="min",
).astype(int)

season_df["pd_rank"] = season_df["Point Differential per Game"].rank(
    ascending=False,
    method="min",
).astype(int)

season_df["offense_rank_calc"] = season_df["Offense EStat"].rank(
    ascending=False,
    method="min",
).astype(int)

season_df["defense_rank_calc"] = season_df["Defense EStat"].rank(
    ascending=False,
    method="min",
).astype(int)

season_df["best_unit_rank"] = season_df[
    ["offense_rank_calc", "defense_rank_calc"]
].min(axis=1)

SQUARE_RANK = 6
CUSP_RANK = 10
BEST_UNIT_LIMIT = 12

season_df["Inside Super Square"] = (
    (season_df["gini_rank"] <= SQUARE_RANK)
    & (season_df["pd_rank"] <= SQUARE_RANK)
    & (season_df["best_unit_rank"] <= BEST_UNIT_LIMIT)
)

season_df["On the Cusp"] = (
    ~season_df["Inside Super Square"]
    & (
        ((season_df["gini_rank"] <= CUSP_RANK) & (season_df["pd_rank"] <= CUSP_RANK))
        | ((season_df["gini_rank"] <= SQUARE_RANK) & (season_df["pd_rank"] <= 12))
        | ((season_df["gini_rank"] <= 12) & (season_df["pd_rank"] <= SQUARE_RANK))
    )
)


def status_label(row):
    if row["Inside Super Square"]:
        return "Inside Super Square"
    if row["On the Cusp"]:
        return "On the Cusp"
    return "Outside"


season_df["Status"] = season_df.apply(status_label, axis=1)

champion = SUPER_BOWL_WINNERS.get(selected_season)
season_df["Is Champion"] = season_df["Team"] == champion

inside_count = int(season_df["Inside Super Square"].sum())
cusp_count = int(season_df["On the Cusp"].sum())

winner_text = "Hidden"
if reveal_winner and champion:
    winner_name = season_df.loc[season_df["Team"] == champion, "Team Name"]
    winner_text = winner_name.iloc[0] if not winner_name.empty else champion

st.markdown(
    f"""
<div class="square-summary">
    <div class="square-pill">Inside Super Square:&nbsp;<span>{inside_count}</span></div>
    <div class="square-pill">On the Cusp:&nbsp;<span>{cusp_count}</span></div>
    <div class="square-pill">Super Bowl Winner:&nbsp;<span>{winner_text}</span></div>
</div>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# CHART
# -----------------------------

x_values = season_df["Gini Score"]
y_values = season_df["Point Differential per Game"]

x_span = max(x_values.max() - x_values.min(), 1)
y_span = max(y_values.max() - y_values.min(), 1)

x_pad = x_span * 0.08
y_pad = y_span * 0.12

x_min = x_values.min() - x_pad
x_max = x_values.max() + x_pad
y_min = y_values.min() - y_pad
y_max = y_values.max() + y_pad

x_cut = season_df.loc[season_df["gini_rank"] <= SQUARE_RANK, "Gini Score"].min()
y_cut = season_df.loc[
    season_df["pd_rank"] <= SQUARE_RANK,
    "Point Differential per Game",
].min()

x_cusp = season_df.loc[season_df["gini_rank"] <= CUSP_RANK, "Gini Score"].min()
y_cusp = season_df.loc[
    season_df["pd_rank"] <= CUSP_RANK,
    "Point Differential per Game",
].min()

fig = px.scatter(
    season_df,
    x="Gini Score",
    y="Point Differential per Game",
    hover_name="Team Name",
    custom_data=[
        "Team",
        "Status",
        "gini_rank",
        "pd_rank",
        "offense_rank_calc",
        "defense_rank_calc",
        "best_unit_rank",
    ],
    title=f"{selected_season} Super Square",
)

fig.update_traces(
    marker=dict(
        size=18,
        color="rgba(0,0,0,0.01)",
        line=dict(width=0),
    ),
    hovertemplate=(
        "<b>%{hovertext}</b><br>"
        "Team: %{customdata[0]}<br>"
        "Status: %{customdata[1]}<br>"
        "Gini Rank: %{customdata[2]}<br>"
        "Point Diff Rank: %{customdata[3]}<br>"
        "Offense Rank: %{customdata[4]}<br>"
        "Defense Rank: %{customdata[5]}<br>"
        "Best Unit Rank: %{customdata[6]}<extra></extra>"
    ),
)

fig.add_shape(
    type="rect",
    x0=x_cusp,
    x1=x_max,
    y0=y_cusp,
    y1=y_max,
    line=dict(color=SECONDARY, width=2, dash="dot"),
    fillcolor="rgba(0,115,183,0.035)",
    layer="below",
)

fig.add_shape(
    type="rect",
    x0=x_cut,
    x1=x_max,
    y0=y_cut,
    y1=y_max,
    line=dict(color=PRIMARY, width=3),
    fillcolor="rgba(241,90,36,0.075)",
    layer="below",
)

fig.add_annotation(
    x=x_cut + (x_max - x_cut) * 0.52,
    y=y_max - y_pad * 0.22,
    text="SUPER SQUARE",
    showarrow=False,
    font=dict(size=16, color="#07111f", family="Arial Black"),
    bgcolor="rgba(255,255,255,0.86)",
    bordercolor=PRIMARY,
    borderwidth=1,
    borderpad=5,
)

fig.add_annotation(
    x=x_cusp,
    y=y_cusp,
    text="Cusp Zone",
    showarrow=True,
    arrowhead=2,
    ax=-35,
    ay=35,
    font=dict(size=12, color="#334155"),
    bgcolor="rgba(255,255,255,0.84)",
    bordercolor=SECONDARY,
    borderwidth=1,
    borderpad=4,
)

logo_df = season_df.copy()
logo_df["Logo URL"] = logo_df["Team"].map(logo_lookup).fillna("")

logo_df["Draw Order"] = 0
logo_df.loc[logo_df["On the Cusp"], "Draw Order"] = 1
logo_df.loc[logo_df["Inside Super Square"], "Draw Order"] = 2
logo_df.loc[logo_df["Is Champion"], "Draw Order"] = 3
logo_df = logo_df.sort_values("Draw Order")

base_logo_width = x_span * 0.052
base_logo_height = y_span * 0.085

for _, row in logo_df.iterrows():
    is_champion = bool(row["Is Champion"])

    if reveal_winner and champion:
        grayscale = not is_champion
        opacity = 1.0 if is_champion else 0.13
        size_boost = 1.40 if is_champion else 0.82
    else:
        if row["Inside Super Square"]:
            grayscale = False
            opacity = 1.0
            size_boost = 1.14
        elif row["On the Cusp"]:
            grayscale = False
            opacity = 0.70
            size_boost = 0.98
        else:
            grayscale = True
            opacity = 0.25
            size_boost = 0.86

    logo_source = logo_url_to_data_uri(row["Logo URL"], grayscale=grayscale)

    if not logo_source:
        continue

    fig.add_layout_image(
        dict(
            source=logo_source,
            xref="x",
            yref="y",
            x=row["Gini Score"],
            y=row["Point Differential per Game"],
            sizex=base_logo_width * size_boost,
            sizey=base_logo_height * size_boost,
            xanchor="center",
            yanchor="middle",
            sizing="contain",
            opacity=opacity,
            layer="above",
        )
    )

if reveal_winner and champion:
    winner_row = season_df[season_df["Team"] == champion]
    if not winner_row.empty:
        winner = winner_row.iloc[0]
        fig.add_annotation(
            x=winner["Gini Score"],
            y=winner["Point Differential per Game"],
            text=f"{winner['Team Name']}<br>Super Bowl Champion",
            showarrow=True,
            arrowhead=2,
            ax=50,
            ay=-52,
            font=dict(size=13, color="#07111f"),
            bgcolor="rgba(255,255,255,0.94)",
            bordercolor=PRIMARY,
            borderwidth=2,
            borderpad=6,
        )

fig.update_layout(
    height=650,
    plot_bgcolor="#FFFFFF",
    paper_bgcolor="#FFFFFF",
    title=dict(
        text=f"{selected_season} Super Square",
        font=dict(color="#07111f", size=18),
        x=0.01,
        xanchor="left",
    ),
    xaxis=dict(
        title="Gini Score",
        range=[x_min, x_max],
        showgrid=True,
        gridcolor="#E5E7EB",
        zeroline=False,
        tickfont=dict(color="#64748B"),
        title_font=dict(color="#334155"),
    ),
    yaxis=dict(
        title="Point Differential per Game",
        range=[y_min, y_max],
        showgrid=True,
        gridcolor="#E5E7EB",
        zeroline=False,
        tickfont=dict(color="#64748B"),
        title_font=dict(color="#334155"),
    ),
    margin=dict(l=65, r=35, t=60, b=60),
    showlegend=False,
)

st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# EXPLANATION
# -----------------------------

st.markdown(
    """
<div class="super-info-box">
    <div class="super-info-title">How the Super Square works</div>
    The Super Square is intentionally selective. A team is inside the square when it clears all three filters:
    <br><br>
    <b>1. Top 6 in Gini Score</b><br>
    <b>2. Top 6 in point differential per game</b><br>
    <b>3. At least one strong unit</b>, meaning offense or defense ranks top 12.
    <br><br>
    The dotted outer area is the cusp zone. Those teams are close to the historical contender profile,
    but they miss at least one of the stricter Super Square cutoffs.
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("</div>", unsafe_allow_html=True)
