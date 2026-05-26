from pathlib import Path
from io import BytesIO
import base64
import html
import sys
import uuid

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageOps


st.set_page_config(
    page_title="NFL EStat / Gini Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

APP_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = APP_DIR / "data"

if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from site_nav import render_top_nav

TEAM_THEME_OVERRIDES = {
    "ARI": {"primary": "#97233F", "secondary": "#000000"},
    "ATL": {"primary": "#A71930", "secondary": "#000000"},
    "BAL": {"primary": "#241773", "secondary": "#9E7C0C"},
    "BUF": {"primary": "#00338D", "secondary": "#C60C30"},
    "CAR": {"primary": "#0085CA", "secondary": "#101820"},
    "CHI": {"primary": "#0B162A", "secondary": "#C83803"},
    "CIN": {"primary": "#FB4F14", "secondary": "#000000"},
    "CLE": {"primary": "#311D00", "secondary": "#FF3C00"},
    "DAL": {"primary": "#003594", "secondary": "#869397"},
    "DEN": {"primary": "#F15A24", "secondary": "#0073B7"},
    "DET": {"primary": "#0076B6", "secondary": "#B0B7BC"},
    "GB": {"primary": "#203731", "secondary": "#FFB612"},
    "HOU": {"primary": "#03202F", "secondary": "#A71930"},
    "IND": {"primary": "#002C5F", "secondary": "#A2AAAD"},
    "JAX": {"primary": "#006778", "secondary": "#D7A22A"},
    "KC": {"primary": "#E31837", "secondary": "#FFB81C"},
    "LA": {"primary": "#003594", "secondary": "#FFD100"},
    "LAC": {"primary": "#0080C6", "secondary": "#FFC20E"},
    "LV": {"primary": "#000000", "secondary": "#A5ACAF"},
    "MIA": {"primary": "#008E97", "secondary": "#FC4C02"},
    "MIN": {"primary": "#4F2683", "secondary": "#FFC62F"},
    "NE": {"primary": "#002244", "secondary": "#C60C30"},
    "NO": {"primary": "#D3BC8D", "secondary": "#101820"},
    "NYG": {"primary": "#0B2265", "secondary": "#A71930"},
    "NYJ": {"primary": "#125740", "secondary": "#000000"},
    "PHI": {"primary": "#004C54", "secondary": "#A5ACAF"},
    "PIT": {"primary": "#000000", "secondary": "#FFB612"},
    "SEA": {"primary": "#002244", "secondary": "#69BE28"},
    "SF": {"primary": "#AA0000", "secondary": "#B3995D"},
    "TB": {"primary": "#D50A0A", "secondary": "#34302B"},
    "TEN": {"primary": "#4B92DB", "secondary": "#C8102E"},
    "WAS": {"primary": "#5A1414", "secondary": "#FFB612"},
}

BASELINE_WEIGHTS = {
    "Offense": 0.30,
    "Defense": 0.30,
    "Point Diff": 0.15,
    "Success Margin": 0.12,
    "Turnovers": 0.06,
    "Penalties": 0.02,
    "Schedule Strength": 0.05,
}

LABELS = {
    "team": "Team",
    "season": "Season",
    "week": "Week",
    "game_id": "Game ID",
    "opponent": "Opponent",
    "home_away": "Home/Away",
    "points_for": "Points For",
    "points_against": "Points Against",
    "offense_estat": "Offense EStat",
    "defense_estat": "Defense EStat",
    "custom_estat": "Custom EStat",
    "custom_rank": "Rank",
    "overall_estat": "Default EStat",
    "offense_rank": "Offense Rank",
    "defense_rank": "Defense Rank",
    "off_epa_per_play": "Offensive EPA per Play",
    "off_adj_epa": "Opponent-Adjusted Offensive EPA per Play",
    "def_epa_allowed_per_play": "Defensive EPA Allowed per Play",
    "def_adj_epa": "Opponent-Adjusted Defensive EPA",
    "off_success_rate": "Offensive Success Rate",
    "def_success_allowed": "Defensive Success Rate Allowed",
    "success_margin": "Success Margin",
    "point_diff": "Point Differential",
    "point_diff_per_game": "Point Differential per Game",
    "turnover_margin": "Turnover Margin",
    "turnover_margin_per_game": "Turnover Margin per Game",
    "penalty_yards_margin": "Penalty Yard Margin",
    "penalty_yards_margin_per_game": "Penalty Yard Margin per Game",
    "schedule_strength": "Schedule Strength",
    "sos_rank": "Strength of Schedule Rank",
    "estat_game": "Game EStat",
    "opp_adj_estat_game": "Matchup-Adjusted Game EStat",
    "opponent_strength": "Opponent Strength",
    "opponent_strength_adj": "Opponent Strength Adjustment",
    "gameday": "Game Date",
    "weekday": "Day",
    "gametime": "Game Time",
    "away_team": "Away Team",
    "away_score": "Away Score",
    "home_team": "Home Team",
    "home_score": "Home Score",
    "stadium": "Stadium",
    "roof": "Roof",
    "surface": "Surface",
    "spread_line": "Spread",
    "total_line": "Total",
}


@st.cache_data
def load_data():
    team_season = pd.read_csv(DATA_DIR / "team_season_estat.csv")
    team_game = pd.read_csv(DATA_DIR / "team_game_estat.csv")
    games_path = DATA_DIR / "games_2005_onward.csv"
    roster_path = DATA_DIR / "nfl_season_rosters_clean_2005_2025.csv"
    assets_path = DATA_DIR / "teams_colors_logos.csv"

    games = pd.read_csv(games_path) if games_path.exists() else pd.DataFrame()
    roster = pd.read_csv(roster_path) if roster_path.exists() else pd.DataFrame()
    team_assets = pd.read_csv(assets_path) if assets_path.exists() else pd.DataFrame()
    return team_season, team_game, games, roster, team_assets


def normalize_weights(weights):
    total = sum(weights.values())
    if total == 0:
        return {key: 0 for key in weights}
    return {key: value / total for key, value in weights.items()}


def recompute_overall(df, weights):
    w = normalize_weights(weights)
    return 100 + 15 * (
        w["Offense"] * df["off_z"]
        + w["Defense"] * df["def_z"]
        + w["Point Diff"] * df["pd_z"]
        + w["Success Margin"] * df["success_z"]
        + w["Turnovers"] * df["turnover_z"]
        + w["Penalties"] * df["penalty_z"]
        + w["Schedule Strength"] * df["schedule_z"]
    )


def readable_text_color(hex_color):
    if not isinstance(hex_color, str) or not hex_color.startswith("#"):
        return "#000000"

    value = hex_color.lstrip("#")
    if len(value) != 6:
        return "#000000"

    try:
        red = int(value[0:2], 16)
        green = int(value[2:4], 16)
        blue = int(value[4:6], 16)
    except ValueError:
        return "#000000"

    brightness = (red * 299 + green * 587 + blue * 114) / 1000
    return "#000000" if brightness > 155 else "#FFFFFF"


def get_team_theme(team, team_assets):
    if team in TEAM_THEME_OVERRIDES:
        primary = TEAM_THEME_OVERRIDES[team]["primary"]
        secondary = TEAM_THEME_OVERRIDES[team]["secondary"]
        logo = ""
        if not team_assets.empty and "team_abbr" in team_assets.columns:
            match = team_assets[team_assets["team_abbr"] == team]
            if not match.empty:
                logo = match.iloc[0].get("team_logo_espn", "")
        return {
            "primary": primary,
            "secondary": secondary,
            "logo": logo,
            "text": readable_text_color(primary),
        }

    default_theme = {
        "primary": "#1F77B4",
        "secondary": "#D9EAF7",
        "logo": "",
        "text": "#FFFFFF",
    }
    if team_assets.empty or "team_abbr" not in team_assets.columns:
        return default_theme

    match = team_assets[team_assets["team_abbr"] == team]
    if match.empty:
        return default_theme

    row = match.iloc[0]
    primary = row.get("team_color", default_theme["primary"])
    secondary = row.get("team_color2", default_theme["secondary"])
    logo = row.get("team_logo_espn", "")

    if not isinstance(primary, str) or not primary.startswith("#"):
        primary = default_theme["primary"]
    if not isinstance(secondary, str) or not secondary.startswith("#"):
        secondary = default_theme["secondary"]

    return {
        "primary": primary,
        "secondary": secondary,
        "logo": logo,
        "text": readable_text_color(primary),
    }


def build_team_name_lookup(team_assets):
    if team_assets.empty or not {"team_abbr", "team_name"}.issubset(team_assets.columns):
        return {}

    lookup = {}
    for _, row in team_assets.dropna(subset=["team_abbr"]).iterrows():
        abbr = row.get("team_abbr")
        name = row.get("team_name", abbr)
        if pd.notna(abbr):
            lookup[str(abbr)] = str(name) if pd.notna(name) else str(abbr)
    return lookup

def get_team_record(team_game_df, season, team):
    if team_game_df.empty:
        return "-"

    required_cols = {"season", "team", "points_for", "points_against"}
    if not required_cols.issubset(team_game_df.columns):
        return "-"

    games_played = team_game_df[
        (team_game_df["season"] == season)
        & (team_game_df["team"] == team)
    ].copy()

    games_played["points_for"] = pd.to_numeric(games_played["points_for"], errors="coerce")
    games_played["points_against"] = pd.to_numeric(games_played["points_against"], errors="coerce")

    games_played = games_played.dropna(subset=["points_for", "points_against"])

    if games_played.empty:
        return "-"

    wins = int((games_played["points_for"] > games_played["points_against"]).sum())
    losses = int((games_played["points_for"] < games_played["points_against"]).sum())
    ties = int((games_played["points_for"] == games_played["points_against"]).sum())

    if ties > 0:
        return f"{wins}-{losses}-{ties}"

    return f"{wins}-{losses}"


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

def rank_badge(rank, value, suffix=""):
    if rank <= 12:
        return "&#8593;", f"{value}{suffix}", "#15803D", "#DCFCE7"
    if rank <= 21:
        return "-", f"{value}{suffix}", "#6B7280", "#F0F2F6"
    return "&#8595;", f"{value}{suffix}", "#DC2626", "#FEE2E2"


def ordinal_rank(number):
    number = int(number)
    if 10 <= number % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"


def metric_rank_info(df, column, selected_team, higher_is_better=True):
    metric_df = df[["team", column]].dropna().copy()
    if metric_df.empty:
        return "-", "#6B7280", "#F0F2F6"

    metric_df["rank"] = metric_df[column].rank(
        ascending=not higher_is_better,
        method="min",
    ).astype(int)
    selected_rank = metric_df.loc[metric_df["team"] == selected_team, "rank"]
    if selected_rank.empty:
        return "-", "#6B7280", "#F0F2F6"

    rank_value = int(selected_rank.iloc[0])
    if rank_value <= 12:
        return ordinal_rank(rank_value), "#15803D", "#DCFCE7"
    if rank_value <= 21:
        return ordinal_rank(rank_value), "#6B7280", "#F0F2F6"
    return ordinal_rank(rank_value), "#DC2626", "#FEE2E2"


def schedule_difficulty_info(df, selected_team):
    metric_df = df[["team", "schedule_strength"]].dropna().copy()
    if metric_df.empty:
        return "-", "#6B7280", "#F0F2F6"

    metric_df["rank"] = metric_df["schedule_strength"].rank(
        ascending=False,
        method="min",
    ).astype(int)
    selected_rank = metric_df.loc[metric_df["team"] == selected_team, "rank"]
    if selected_rank.empty:
        return "-", "#6B7280", "#F0F2F6"

    rank_value = int(selected_rank.iloc[0])
    if rank_value == 1:
        return "1st - Hardest Schedule", "#DC2626", "#FEE2E2"
    if rank_value <= 12:
        return f"{ordinal_rank(rank_value)} - Hard Schedule", "#DC2626", "#FEE2E2"
    if rank_value <= 21:
        return f"{ordinal_rank(rank_value)} - Average Schedule", "#6B7280", "#F0F2F6"
    if rank_value == 32:
        return "32nd - Easiest Schedule", "#15803D", "#DCFCE7"
    return f"{ordinal_rank(rank_value)} - Easy Schedule", "#15803D", "#DCFCE7"


def apply_schedule_z_score(season_df):
    if "schedule_strength" not in season_df.columns:
        season_df["schedule_z"] = 0
        return season_df

    schedule_std = season_df["schedule_strength"].std()
    if pd.notna(schedule_std) and schedule_std != 0:
        season_df["schedule_z"] = (
            season_df["schedule_strength"] - season_df["schedule_strength"].mean()
        ) / schedule_std
    else:
        season_df["schedule_z"] = 0
    return season_df


def format_table(df, fmt=None):
    display_df = df.copy()
    if fmt:
        for column, format_string in fmt.items():
            if column in display_df.columns:
                display_df[column] = display_df[column].apply(
                    lambda value: format_string.format(value) if pd.notna(value) else "-"
                )
    return display_df


def format_whole_number(value):
    numeric_value = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric_value):
        return "-"
    return f"{int(numeric_value)}"


def dataframe_to_html_table(df, highlight_column=None, highlight_value=None):
    header_html = "".join(
        f"<th>{html.escape(str(column))}</th>" for column in df.columns
    )
    rows_html = []

    highlight_mask = pd.Series(False, index=df.index)
    if highlight_column in df.columns:
        highlight_mask = df[highlight_column].astype(str) == str(highlight_value)

    for row_index, row in df.iterrows():
        row_class = ' class="selected-team-row"' if bool(highlight_mask.loc[row_index]) else ""
        cells_html = "".join(
            f"<td>{html.escape(str(value)) if pd.notna(value) else '-'}</td>"
            for value in row
        )
        rows_html.append(f"<tr{row_class}>{cells_html}</tr>")

    return (
        '<table border="1" class="dataframe clean-table">'
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody>"
        "</table>"
    )


def show_clean_table(
    df,
    fmt=None,
    max_height=420,
    extra_class="",
    top_margin_px=8,
    highlight_column=None,
    highlight_value=None,
    scroll_to_highlight=False,
    scroll_offset_rows=4,
):
    table_id = f"clean-table-{uuid.uuid4().hex}"
    table_class = f"clean-table-wrap {extra_class}".strip()
    extra_margin_css = (
        f".clean-table-outer.{extra_class} {{ margin-top: {top_margin_px}px !important; }}"
        if extra_class
        else ""
    )

    table_html = dataframe_to_html_table(
        format_table(df, fmt),
        highlight_column=highlight_column,
        highlight_value=highlight_value,
    )

    st.markdown(
        f"""
<style>
.clean-table-outer {{
    border: 1px solid #D9DEE7;
    border-radius: 10px;
    overflow: hidden;
    margin-top: 8px;
    background: #FFFFFF !important;
    box-shadow: 0 12px 28px rgba(15, 23, 42, 0.08);
}}

.clean-table-wrap {{
    overflow-x: auto;
    overflow-y: auto;
    max-height: {max_height}px;
    background: #FFFFFF !important;
    position: relative;
    isolation: isolate;
    overscroll-behavior: contain;
}}

.clean-table {{
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 14px;
    color: #111827;
    background: #FFFFFF !important;
}}

.clean-table thead {{
    background: #F8FAFC !important;
    position: sticky;
    top: 0;
    z-index: 200;
}}

.clean-table thead th {{
    position: sticky;
    top: 0;
    z-index: 210;
    background: #F8FAFC !important;
    background-color: #F8FAFC !important;
    background-clip: padding-box;
    color: #111827;
    font-weight: 700;
    text-align: left;
    padding: 10px 12px;
    border-right: 1px solid #E5E7EB;
    border-bottom: 1px solid #E5E7EB;
    box-shadow:
        inset 0 -3px 0 0 {selected_team_color},
        0 -10px 0 10px #F8FAFC,
        0 6px 10px rgba(248, 250, 252, 0.96);
    white-space: nowrap;
}}

.clean-table tbody {{
    background: #FFFFFF !important;
}}

.clean-table tbody tr {{
    background-color: #FFFFFF !important;
}}

.clean-table tbody tr:nth-child(even) {{
    background-color: #F8FAFC !important;
}}

.clean-table tbody td {{
    padding: 9px 12px;
    border-right: 1px solid #E5E7EB;
    border-bottom: 1px solid #E5E7EB;
    white-space: nowrap;
    color: #111827;
    background-color: inherit !important;
}}

.clean-table tbody tr:hover {{
    background-color: #EEF2F7 !important;
}}

.clean-table tbody tr.selected-team-row td {{
    background-color: inherit !important;
    border-top: 2px solid {selected_team_color};
    border-bottom: 2px solid {selected_team_color2};
}}

.clean-table tbody tr.selected-team-row td:first-child {{
    border-left: 4px solid {selected_team_color};
}}

.clean-table tbody tr.selected-team-row td:last-child {{
    border-right: 2px solid {selected_team_color2};
}}

/* Schedule tables are short, so do not use sticky headers or scrolling.
   This prevents row text from bleeding under the header and removes the extra bottom space. */
.clean-table-outer.schedule-table {{
    display: inline-block !important;
    width: 100% !important;
    height: auto !important;
    margin-bottom: 0 !important;
}}

.clean-table-outer.schedule-table .clean-table-wrap {{
    overflow-x: visible !important;
    overflow-y: visible !important;
    max-height: none !important;
    height: auto !important;
    display: block !important;
}}

.clean-table-outer.schedule-table .clean-table {{
    margin-bottom: 0 !important;
}}

.clean-table-outer.schedule-table .clean-table thead {{
    position: static !important;
    top: auto !important;
    z-index: auto !important;
}}

.clean-table-outer.schedule-table .clean-table thead th {{
    position: static !important;
    top: auto !important;
    z-index: auto !important;
}}

{extra_margin_css}
</style>

<div class="clean-table-outer {extra_class}">
    <div id="{table_id}" class="{table_class}">
        {table_html}
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    if scroll_to_highlight:
        components.html(
            f"""
<script>
(function() {{
    const tableId = "{table_id}";
    const offsetRows = {int(scroll_offset_rows)};
    let attempts = 0;

    function scrollToSelectedRow() {{
        attempts += 1;
        const wrapper = window.parent.document.getElementById(tableId);
        if (!wrapper) {{
            return attempts > 25;
        }}

        const selectedRow = wrapper.querySelector("tbody tr.selected-team-row");
        if (!selectedRow) {{
            return true;
        }}

        const header = wrapper.querySelector("thead");
        const headerHeight = header ? header.getBoundingClientRect().height : 0;
        const rowHeight = selectedRow.getBoundingClientRect().height || 40;
        const maxScroll = Math.max(0, wrapper.scrollHeight - wrapper.clientHeight);
        const targetScroll = selectedRow.offsetTop - headerHeight - (offsetRows * rowHeight);

        wrapper.scrollTop = Math.max(0, Math.min(targetScroll, maxScroll));
        return true;
    }}

    const timer = window.setInterval(function() {{
        if (scrollToSelectedRow()) {{
            window.clearInterval(timer);
        }}
    }}, 80);
}})();
</script>
""",
            height=0,
        )


def show_rank_metric(column, label, rank_value, delta_info):
    symbol, delta_text, badge_text_color, badge_bg_color = delta_info
    column.markdown(
        f"""
<div class="metric-card">
    <div class="metric-label">{label}</div>
    <div class="metric-value">#{int(rank_value)}</div>
    <div class="metric-badge" style="background:{badge_bg_color}; color:{badge_text_color};">
        {symbol} {delta_text}
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


def show_deep_dive_metric(
    column,
    label,
    value,
    rank_text=None,
    badge_text_color="#6B7280",
    badge_bg_color="#F0F2F6",
):
    rank_html = ""
    if rank_text:
        rank_html = (
            f"<div class='metric-badge' style='background:{badge_bg_color}; "
            f"color:{badge_text_color};'>{rank_text}</div>"
        )
    column.markdown(
        f"""
<div class="metric-card">
    <div class="metric-label">{label}</div>
    <div class="metric-value">{value}</div>
    {rank_html}
</div>
""",
        unsafe_allow_html=True,
    )


team_season, team_game, games, roster, team_assets = load_data()

if team_season.empty:
    st.error("No team season data found. Check the data folder.")
    st.stop()

seasons = sorted(team_season["season"].dropna().unique().astype(int), reverse=True)

if "gini_selected_season" not in st.session_state or st.session_state["gini_selected_season"] not in seasons:
    st.session_state["gini_selected_season"] = seasons[0]

selected_season = int(st.session_state["gini_selected_season"])

season_df = team_season[team_season["season"] == selected_season].copy()
season_df = apply_schedule_z_score(season_df)

teams = sorted(season_df["team"].dropna().unique())
default_team_index = teams.index("KC") if "KC" in teams else 0
team_name_lookup = build_team_name_lookup(team_assets)
team_label_lookup = {team: team_name_lookup.get(team, team) for team in teams}
team_abbr_lookup = {label: team for team, label in team_label_lookup.items()}

if "gini_selected_team" not in st.session_state or st.session_state["gini_selected_team"] not in teams:
    st.session_state["gini_selected_team"] = teams[default_team_index]

selected_team = st.session_state["gini_selected_team"]

theme = get_team_theme(selected_team, team_assets)
selected_team_color = theme["primary"]
selected_team_color2 = theme["secondary"]
selected_team_logo = theme["logo"]

selected_team_name = team_label_lookup.get(selected_team, selected_team)
selected_team_record = get_team_record(team_game, selected_season, selected_team)

if st.session_state.get("theme_team") != selected_team:
    st.session_state["theme_team"] = selected_team
    st._config.set_option("theme.primaryColor", selected_team_color)
    st.rerun()

st._config.set_option("theme.primaryColor", selected_team_color)

page_bg = "#F6F8FB"
card_bg = "#FFFFFF"
text_color = "#111827"
muted_text = "#6B7280"
border_color = "#E5E7EB"
neutral_bar = "#C9D3DF"

# Solid Plotly chart backgrounds for readability
plot_bg = "#FFFFFF"
paper_bg = "#FFFFFF"
grid_color = "#E5E7EB"

st.markdown(
    f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-color: {page_bg};
    color: {text_color};
    overflow-x: hidden;
}}

.team-bg-canvas {{
    position: fixed;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    pointer-events: none;
    z-index: 0;
    overflow: visible;
}}

.team-bg-canvas::before {{
    content: "";
    position: absolute;
    top: -18%;
    left: -10%;
    width: 145%;
    height: 145%;
    background: radial-gradient(
        ellipse at center,
        {selected_team_color}35 0%,
        {selected_team_color}14 30%,
        transparent 62%
    );
    animation: teamGlowPrimary 24s ease-in-out infinite;
}}

.team-bg-canvas::after {{
    content: "";
    position: absolute;
    bottom: -18%;
    right: -10%;
    width: 145%;
    height: 145%;
    background: radial-gradient(
        ellipse at center,
        {selected_team_color2}32 0%,
        {selected_team_color2}12 30%,
        transparent 62%
    );
    animation: teamGlowSecondary 26s ease-in-out infinite;
}}

@keyframes teamGlowPrimary {{
    0%   {{ opacity: 0.45; transform: translate(-8%, 0%) scale(1); }}
    25%  {{ opacity: 0.85; transform: translate(38%, -18%) scale(1.18); }}
    50%  {{ opacity: 0.62; transform: translate(78%, 28%) scale(1.28); }}
    75%  {{ opacity: 0.82; transform: translate(28%, 58%) scale(1.08); }}
    100% {{ opacity: 0.45; transform: translate(-8%, 0%) scale(1); }}
}}

@keyframes teamGlowSecondary {{
    0%   {{ opacity: 0.78; transform: translate(8%, 0%) scale(1); }}
    25%  {{ opacity: 0.42; transform: translate(-28%, 38%) scale(1.18); }}
    50%  {{ opacity: 0.68; transform: translate(-58%, -18%) scale(1.25); }}
    75%  {{ opacity: 0.42; transform: translate(18%, -48%) scale(1.08); }}
    100% {{ opacity: 0.78; transform: translate(8%, 0%) scale(1); }}
}}

[data-testid="stAppViewContainer"] > .main {{
    position: relative;
    z-index: 1;
}}

.block-container {{
    position: relative;
    z-index: 2;
    padding-top: 2.2rem !important;
    padding-bottom: 2.8rem !important;
}}

[data-testid="stSidebar"] {{
    z-index: 3;
    background: linear-gradient(180deg, #EEF3F8 0%, #F8FAFC 100%) !important;
}}

[data-testid="stSidebar"],
[data-testid="collapsedControl"] {{
    display: none !important;
}}

[data-testid="stHeader"] {{
    background: rgba(255,255,255,0.96) !important;
    backdrop-filter: blur(10px);
    border-bottom: 1px solid rgba(15, 23, 42, 0.08);
    box-shadow: 0 8px 22px rgba(15, 23, 42, 0.045);
}}

.site-top-nav {{
    position: sticky;
    top: 0;
    z-index: 20;

    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;

    margin: 0.35rem 0 1.25rem 0;
    padding: 0.75rem 1rem;

    border: 1px solid rgba(15, 23, 42, 0.10);
    border-radius: 16px;

    background: rgba(255,255,255,0.92);
    box-shadow: 0 10px 28px rgba(15, 23, 42, 0.065);
    backdrop-filter: blur(14px);
}}

.site-top-brand {{
    display: inline-flex;
    align-items: center;
    gap: 0.55rem;
    color: #07111f;
    font-size: 0.92rem;
    line-height: 1.15;
    font-weight: 950;
    padding: 0.4rem 0.55rem;
}}

.site-top-brand::before {{
    content: "";
    width: 10px;
    height: 10px;
    border-radius: 999px;
    background: {selected_team_color};
    box-shadow: 0 0 0 4px {selected_team_color}1f;
}}

.site-nav-links {{
    display: flex;
    align-items: center;
    justify-content: flex-end;
    gap: 0.25rem;
    flex-wrap: wrap;
}}

.site-nav-link {{
    position: relative;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-height: 34px;
    padding: 0.42rem 0.72rem;
    border-radius: 999px;
    color: #334155 !important;
    font-size: 0.84rem;
    line-height: 1.15;
    font-weight: 850;
    text-decoration: none !important;
}}

.site-nav-link:hover {{
    background: #FFFFFF;
    color: {selected_team_color} !important;
}}

.site-nav-link.active {{
    background: #FFFFFF;
    color: {selected_team_color} !important;
    box-shadow: 0 6px 16px rgba(15, 23, 42, 0.075);
}}

.site-nav-link.active::after {{
    content: "";
    position: absolute;
    left: 0.9rem;
    right: 0.9rem;
    bottom: 0.22rem;
    height: 2px;
    border-radius: 999px;
    background: {selected_team_color};
}}

div[data-testid="stSelectbox"] label p {{
    color: #07111f !important;
    font-size: 0.92rem !important;
    font-weight: 850 !important;
    letter-spacing: -0.01em !important;
}}

.dashboard-setup-card-marker {{
    display: none;
}}

div[data-testid="stHorizontalBlock"]:has(.dashboard-setup-card-marker) {{
    display: grid;
    grid-template-columns: minmax(0, 1.65fr) minmax(360px, 0.8fr);
    align-items: end;
    gap: 1.35rem;
    margin: 0.95rem 0 0.9rem 0;
    padding: 1rem 1.2rem 1.15rem 1.2rem;
    background: rgba(255,255,255,0.82);
    border: 1px solid rgba(15, 23, 42, 0.10);
    border-left: 5px solid {selected_team_color};
    border-radius: 16px;
    box-shadow: 0 12px 28px rgba(15, 23, 42, 0.065);
    backdrop-filter: blur(9px);
    overflow: hidden;
}}

div[data-testid="stHorizontalBlock"]:has(.dashboard-setup-card-marker) > div {{
    width: 100% !important;
    min-width: 0 !important;
}}

div[data-testid="stHorizontalBlock"]:has(.dashboard-setup-card-marker) div[data-testid="stSelectbox"] {{
    width: 100% !important;
}}

div[data-testid="stHorizontalBlock"]:has(.dashboard-setup-card-marker) div[data-baseweb="select"] {{
    width: 100% !important;
    min-width: 100% !important;
}}

.dashboard-setup-title {{
    color: #07111f;
    font-size: 1.05rem;
    line-height: 1.15;
    font-weight: 950;
    letter-spacing: -0.02em;
    margin: 0 0 0.65rem 0;
}}

.dashboard-setup-season-spacer {{
    height: 1.8rem;
}}

@media (max-width: 900px) {{
    div[data-testid="stHorizontalBlock"]:has(.dashboard-setup-card-marker) {{
        grid-template-columns: 1fr;
        gap: 0.7rem;
    }}

    .dashboard-setup-season-spacer {{
        display: none;
    }}
}}

.weights-intro {{
    color: #475569;
    font-size: 0.88rem;
    line-height: 1.55;
    margin-bottom: 0.85rem;
}}

div[data-testid="stExpander"] {{
    background: rgba(255,255,255,0.78) !important;
    border: 1px solid rgba(15, 23, 42, 0.10) !important;
    border-left: 5px solid {selected_team_color} !important;
    border-radius: 16px !important;
    box-shadow: 0 12px 28px rgba(15, 23, 42, 0.065) !important;
    backdrop-filter: blur(9px);
    overflow: hidden;
    margin: 0.5rem 0 1.05rem 0;
}}

div[data-testid="stExpander"]:hover {{
    border-color: rgba(15, 23, 42, 0.16) !important;
    box-shadow: 0 16px 34px rgba(15, 23, 42, 0.085) !important;
}}

div[data-testid="stExpander"] details {{
    background: transparent !important;
}}

div[data-testid="stExpander"] details summary {{
    background: linear-gradient(90deg, rgba(255,255,255,0.92), rgba(255,255,255,0.64)) !important;
    color: #07111f !important;
    font-weight: 950 !important;
    padding: 0.88rem 1rem !important;
}}

div[data-testid="stExpander"] details summary p {{
    color: #07111f !important;
    font-weight: 950 !important;
}}

div[data-testid="stExpander"] details[open] summary {{
    border-bottom: 1px solid rgba(15, 23, 42, 0.08);
}}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label {{
    color: #111827 !important;
}}

.weight-label-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #FFFFFF;
    border: 1px solid rgba(15, 23, 42, 0.10);
    border-bottom: 0;
    border-radius: 12px 12px 0 0;
    padding: 0.58rem 0.72rem 0.15rem 0.72rem;
    margin-top: 0.58rem;
    color: #111827;
    font-size: 0.8rem;
    font-weight: 850;
    box-shadow: 0 8px 18px rgba(15, 23, 42, 0.045);
}}

.weight-label-row span:last-child {{
    color: {selected_team_color};
    font-weight: 950;
}}

div[data-testid="stExpander"] div[data-testid="stSlider"] {{
    background: #FFFFFF;
    border: 1px solid rgba(15, 23, 42, 0.10);
    border-top: 0;
    border-radius: 0 0 12px 12px;
    padding: 0 0.72rem 0.62rem 0.72rem;
    margin-bottom: 0.2rem;
    box-shadow: 0 8px 18px rgba(15, 23, 42, 0.045);
}}

div[data-testid="stSelectbox"] [data-baseweb="select"] > div {{
    background-color: #FFFFFF !important;
    border: 1px solid #D1D5DB !important;
    border-radius: 10px !important;
    box-shadow: 0 6px 16px rgba(15, 23, 42, 0.045) !important;
}}

div[data-testid="stSelectbox"] [data-baseweb="select"] > div:hover {{
    border-color: {selected_team_color} !important;
}}

div[data-testid="stSelectbox"] [data-baseweb="select"] input {{
    color: #111827 !important;
}}

div[data-testid="stSelectbox"] [data-baseweb="select"] svg {{
    color: #111827 !important;
}}

div[data-testid="stSelectbox"] [data-baseweb="select"] span {{
    color: #111827 !important;
}}

[data-baseweb="popover"] {{
    background-color: #FFFFFF !important;
}}

[data-baseweb="popover"] [role="listbox"],
[data-baseweb="menu"] {{
    background-color: #FFFFFF !important;
    border: 1px solid #D1D5DB !important;
    border-radius: 10px !important;
    box-shadow: 0 14px 32px rgba(15, 23, 42, 0.14) !important;
}}

[data-baseweb="menu"] li {{
    background-color: #FFFFFF !important;
    color: #111827 !important;
}}

[data-baseweb="menu"] li:hover {{
    background-color: #F8FAFC !important;
}}

div[data-testid="stTextInput"] [data-baseweb="input"] {{
    background-color: #FFFFFF !important;
    border: 1px solid #D1D5DB !important;
    border-radius: 10px !important;
    box-shadow: 0 6px 16px rgba(15, 23, 42, 0.045) !important;
}}

div[data-testid="stTextInput"] [data-baseweb="input"]:hover {{
    border-color: {selected_team_color} !important;
}}

div[data-testid="stTextInput"] input {{
    background-color: #FFFFFF !important;
    color: #111827 !important;
}}

div[data-testid="stTextInput"] input::placeholder {{
    color: #9CA3AF !important;
}}

/* Tighten selectbox and text input spacing so roster filters sit closer to tables */
div[data-testid="stSelectbox"],
div[data-testid="stTextInput"] {{
    margin-bottom: 0.25rem !important;
}}

div[data-testid="stSelectbox"] > div,
div[data-testid="stTextInput"] > div {{
    margin-bottom: 0 !important;
}}

div[data-testid="stSelectbox"] div[data-baseweb="select"],
div[data-testid="stTextInput"] div[data-baseweb="input"] {{
    margin-bottom: 0 !important;
}}

h1, h2, h3, h4, h5, h6, p, label {{
    color: {text_color} !important;
}}

.dashboard-hero {{
    position: relative;
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    gap: 1.25rem;
    align-items: center;
    margin: 0.25rem 0 1.1rem 0;
    padding: 1.65rem 1.75rem;
    border-radius: 20px;
    overflow: hidden;
    background:
        radial-gradient(circle at 92% 12%, {selected_team_color}44 0%, transparent 28%),
        radial-gradient(circle at 78% 85%, {selected_team_color2}38 0%, transparent 30%),
        linear-gradient(135deg, #07111f 0%, #172033 58%, #273447 100%);
    border: 1px solid rgba(255, 255, 255, 0.14);
    box-shadow: 0 22px 46px rgba(15, 23, 42, 0.18);
}}

.dashboard-hero::before {{
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 5px;
    background: linear-gradient(90deg, {selected_team_color}, {selected_team_color2});
}}

.hero-kicker {{
    color: rgba(255,255,255,0.72);
    font-size: 0.8rem;
    font-weight: 900;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.5rem;
}}

.hero-title {{
    color: #FFFFFF;
    font-size: clamp(2.1rem, 4vw, 3.25rem);
    line-height: 1;
    font-weight: 950;
    letter-spacing: 0;
    margin-bottom: 0.65rem;
}}

.hero-subtitle {{
    max-width: 760px;
    color: rgba(255,255,255,0.84);
    font-size: 0.98rem;
    line-height: 1.65;
}}

.hero-team-card {{
    position: relative;
    justify-self: stretch;
    width: 100%;
    display: flex;
    align-items: center;
    gap: 1.35rem;
    min-height: 126px;
    padding: 1.12rem 1.45rem;
    border-radius: 17px;
    background:
        linear-gradient(90deg, rgba(255,255,255,0.17), rgba(255,255,255,0.10)),
        radial-gradient(circle at 85% 20%, {selected_team_color}22 0%, transparent 34%);
    border: 1px solid rgba(255,255,255,0.26);
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.16),
        0 16px 34px rgba(0,0,0,0.14);
    backdrop-filter: blur(10px);
    overflow: hidden;
}}

.hero-team-card::before {{
    content: "";
    position: absolute;
    left: 0;
    top: 0;
    width: 5px;
    height: 100%;
    background: linear-gradient(180deg, {selected_team_color}, {selected_team_color2});
    opacity: 0.95;
}}

.hero-logo-wrap {{
    width: 98px;
    height: 98px;
    flex: 0 0 98px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 15px;
    background: rgba(255,255,255,0.11);
    border: 1px solid rgba(255,255,255,0.18);
}}

.hero-logo-wrap img {{
    width: 86px;
    height: 86px;
    object-fit: contain;
    filter: drop-shadow(0 8px 15px rgba(0,0,0,0.28));
}}

.hero-team-fallback {{
    width: 86px;
    height: 86px;
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(255,255,255,0.14);
    color: #FFFFFF;
    font-size: 1.05rem;
    font-weight: 950;
    border: 1px solid rgba(255,255,255,0.25);
}}

.hero-team-content {{
    min-width: 0;
    flex: 1;
}}

.hero-team-kicker {{
    color: rgba(255,255,255,0.66);
    font-size: 0.7rem;
    line-height: 1.2;
    font-weight: 950;
    text-transform: uppercase;
    letter-spacing: 0.075em;
    margin-bottom: 0.28rem;
}}

.hero-team-main {{
    color: #FFFFFF;
    font-size: clamp(1.28rem, 2.1vw, 1.72rem);
    line-height: 1.12;
    font-weight: 950;
    max-width: none;
    overflow-wrap: anywhere;
    text-wrap: balance;
}}

.hero-team-record {{
    margin-top: 0.5rem;
    color: rgba(255,255,255,0.82);
    font-size: 0.92rem;
    line-height: 1.25;
    font-weight: 850;
    letter-spacing: -0.01em;
}}

.hero-team-record span {{
    color: #FFFFFF;
    font-weight: 950;
}}

@media (max-width: 900px) {{
    .dashboard-hero {{
        grid-template-columns: 1fr;
        padding: 1.35rem;
    }}

    .hero-team-card {{
        justify-self: start;
    }}
}}

.metric-help {{
    width: 100%;
    background: #FFFFFF !important;
    border: 1px solid #D1D5DB;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
    margin-bottom: 0.75rem;
}}

.metric-help summary {{
    cursor: pointer;
    list-style: none;
    background: #F8FAFC !important;
    padding: 0.9rem 1.05rem;
    font-size: 0.95rem;
    font-weight: 750;
    color: #111827;
    display: flex;
    align-items: center;
    gap: 0.65rem;
}}

.metric-help summary::-webkit-details-marker {{
    display: none;
}}

.metric-help summary::before {{
    content: "›";
    font-size: 1.25rem;
    line-height: 1;
    color: #111827;
    transform: rotate(0deg);
    transition: transform 0.18s ease;
}}

.metric-help[open] summary {{
    border-bottom: 1px solid #E5E7EB;
}}

.metric-help[open] summary::before {{
    transform: rotate(90deg);
}}

.metric-help-body {{
    background: #FFFFFF !important;
    padding: 1rem 1.15rem;
    color: #111827;
    font-size: 0.95rem;
    line-height: 1.65;
}}

.metric-help-body p {{
    margin: 0 0 0.85rem 0;
    padding: 0;
    color: #111827;
}}

.metric-help-body p:last-child {{
    margin-bottom: 0;
}}

.stTabs [data-baseweb="tab-list"] {{
    gap: 0.45rem;
    background: rgba(255,255,255,0.72);
    border: 1px solid rgba(15, 23, 42, 0.10);
    border-radius: 14px;
    padding: 0.35rem;
    box-shadow: 0 8px 20px rgba(15, 23, 42, 0.045);
}}

.stTabs [data-baseweb="tab"] {{
    border-radius: 10px;
    padding: 0.72rem 0.85rem;
    font-size: 0.93rem;
    font-weight: 850;
    color: #334155;
}}

.stTabs [data-baseweb="tab-highlight"] {{
    background-color: {selected_team_color} !important;
    height: 3px;
}}

.stTabs [aria-selected="true"] {{
    background: #FFFFFF;
    color: {selected_team_color} !important;
    box-shadow: 0 6px 16px rgba(15, 23, 42, 0.075);
}}

:root,
html,
body,
[data-testid="stApp"] {{
    --primary-color: {selected_team_color} !important;
}}

div[data-testid="stSlider"] label,
div[data-testid="stSlider"] p {{
    color: {selected_team_color} !important;
}}

div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div:nth-child(1) {{
    height: 5px !important;
    min-height: 5px !important;
    border-radius: 999px !important;
}}

div[data-testid="stSlider"] div[role="slider"] {{
    background-color: {selected_team_color2} !important;
    box-shadow: 0 0 0 1.5px {selected_team_color} !important;
}}

div[data-testid="stSlider"] div[role="slider"]:hover,
div[data-testid="stSlider"] div[role="slider"]:focus {{
    background-color: {selected_team_color2} !important;
}}

div[data-testid="stCheckbox"]:has(input:disabled) label,
div[data-testid="stCheckbox"]:has(input:disabled) label span,
div[data-testid="stCheckbox"]:has(input:disabled) p {{
    color: #9CA3AF !important;
    opacity: 1 !important;
}}

.metric-card {{
    position: relative;
    overflow: hidden;
    background: rgba(255, 255, 255, 0.94);
    border: 1px solid rgba(15, 23, 42, 0.10);
    border-left: 5px solid {selected_team_color};
    box-shadow: 0 12px 28px rgba(15, 23, 42, 0.075);
    padding: 0.95rem 1.05rem 0.98rem 1rem;
    border-radius: 14px;
    min-height: 128px;
    backdrop-filter: blur(5px);
}}

.metric-card::before {{
    content: "";
    position: absolute;
    left: 5px;
    top: 0;
    width: calc(100% - 5px);
    height: 4px;
    background: linear-gradient(90deg, {selected_team_color2}, transparent);
    opacity: 0.75;
}}

.metric-label {{
    position: relative;
    z-index: 1;
    font-size: 13px;
    line-height: 1.25;
    font-weight: 850;
    color: #111827;
    margin-bottom: 0.48rem;
}}

.metric-value {{
    position: relative;
    z-index: 1;
    font-size: 34px;
    line-height: 1.1;
    color: #111827;
    margin-bottom: 0.62rem;
    font-weight: 850;
}}

.metric-badge {{
    position: relative;
    z-index: 1;
    display: inline-flex;
    align-items: center;
    width: fit-content;
    max-width: 100%;
    padding: 0.22rem 0.52rem;
    border-radius: 999px;
    font-size: 13px;
    line-height: 1.25;
    font-weight: 750;
    white-space: normal;
}}

div[data-testid="stPlotlyChart"] {{
    background: #FFFFFF !important;
    border: 1px solid rgba(15, 23, 42, 0.10);
    border-radius: 16px;
    padding: 0.65rem 0.85rem;
    box-shadow: 0 14px 32px rgba(15, 23, 42, 0.07);
    display: flex;
    justify-content: center;
    align-items: center;
}}

/* Make dashboard dropdown labels and selected values easier to read */
div[data-testid="stSelectbox"] label p {{
    color: #07111f !important;
    font-size: 1rem !important;
    font-weight: 850 !important;
}}

div[data-testid="stSelectbox"] [data-baseweb="select"] div {{
    color: #07111f !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
}}

div[data-testid="stSelectbox"] [data-baseweb="select"] span {{
    color: #07111f !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
}}

div[data-testid="stSelectbox"] [data-baseweb="select"] input {{
    color: #07111f !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
}}

</style>
""",

    unsafe_allow_html=True,
)

st.markdown(
    '<div class="team-bg-canvas" aria-hidden="true"></div>',
    unsafe_allow_html=True,
)

render_top_nav("Gini Dashboard", selected_team_color, selected_team_color2)

hero_logo_html = (
    f'<img src="{selected_team_logo}" alt="{selected_team} logo">'
    if selected_team_logo
    else f'<div class="hero-team-fallback">{selected_team}</div>'
)

st.markdown(
    f"""
<div class="dashboard-hero">
    <div>
        <div class="hero-kicker">Interactive NFL team evaluation model</div>
        <div class="hero-title">Gini Dashboard</div>
        <div class="hero-subtitle">
            Compare team strength through offense, defense, scoring margin, consistency,
            turnovers, penalties, and schedule context. Use the controls below to select a team
            and tune the model.
        </div>
    </div>
    <div class="hero-team-card">
        <div class="hero-logo-wrap">
            {hero_logo_html}
        </div>
        <div class="hero-team-content">
            <div class="hero-team-kicker">Team Profile</div>
            <div class="hero-team-main">{selected_season} {selected_team_name}</div>
            <div class="hero-team-record">Record: <span>{selected_team_record}</span></div>
        </div>
    </div>
""",
    unsafe_allow_html=True,
)

team_options = [team_label_lookup[team] for team in teams]
current_team_label = team_label_lookup.get(selected_team, selected_team)

control_team_col, control_season_col = st.columns([2.35, 0.9], gap="large")

with control_team_col:
    st.markdown(
        '<div class="dashboard-setup-card-marker"></div><div class="dashboard-setup-title">Dashboard Setup</div>',
        unsafe_allow_html=True,
    )
    selected_team_label = st.selectbox(
        "Team",
        team_options,
        index=team_options.index(current_team_label) if current_team_label in team_options else 0,
        key=f"gini_selected_team_label_{selected_season}",
        width="stretch",
    )

with control_season_col:
    st.markdown(
        '<div class="dashboard-setup-season-spacer"></div>',
        unsafe_allow_html=True,
    )
    selected_season_choice = st.selectbox(
        "Season",
        seasons,
        index=seasons.index(selected_season),
        key="gini_selected_season_control",
        width="stretch",
    )

new_selected_team = team_abbr_lookup.get(selected_team_label, selected_team_label)

if int(selected_season_choice) != selected_season or new_selected_team != selected_team:
    st.session_state["gini_selected_season"] = int(selected_season_choice)
    st.session_state["gini_selected_team"] = new_selected_team
    st.rerun()

reset_counter_key = f"reset_counter_{selected_team}"
if reset_counter_key not in st.session_state:
    st.session_state[reset_counter_key] = 0

reset_counter = st.session_state[reset_counter_key]

def model_weight_slider(column, label, key_suffix):
    widget_key = f"weight_{key_suffix}_{selected_team}_{reset_counter}"
    current_value = float(st.session_state.get(widget_key, BASELINE_WEIGHTS[label]))
    column.markdown(
        f"""
<div class="weight-label-row">
    <span>{label}</span>
    <span>{current_value:.2f}</span>
</div>
""",
        unsafe_allow_html=True,
    )
    return column.slider(
        label,
        0.0,
        1.0,
        BASELINE_WEIGHTS[label],
        0.01,
        key=widget_key,
        label_visibility="collapsed",
    )

with st.expander("Customize Model Weights", expanded=False):
    st.markdown(
        '<div class="weights-intro">Adjust the model emphasis only when you want to test a different football philosophy. The dashboard updates rankings from these weights.</div>',
        unsafe_allow_html=True,
    )
    w1, w2, w3, w4 = st.columns(4)
    w5, w6, w7, w_reset = st.columns(4)

    weights = {
        "Offense": model_weight_slider(w1, "Offense", "Offense"),
        "Defense": model_weight_slider(w2, "Defense", "Defense"),
        "Point Diff": model_weight_slider(w3, "Point Diff", "Point_Diff"),
        "Success Margin": model_weight_slider(w4, "Success Margin", "Success_Margin"),
        "Turnovers": model_weight_slider(w5, "Turnovers", "Turnovers"),
        "Penalties": model_weight_slider(w6, "Penalties", "Penalties"),
        "Schedule Strength": model_weight_slider(w7, "Schedule Strength", "Schedule_Strength"),
    }

    st.markdown("<div style='height: 1.78rem;'></div>", unsafe_allow_html=True)
    if w_reset.button("Reset weights", use_container_width=True):
        st.session_state[reset_counter_key] += 1
        st.rerun()

season_df["custom_estat"] = recompute_overall(season_df, weights)
season_df["custom_rank"] = season_df["custom_estat"].rank(
    ascending=False,
    method="min",
).astype(int)
season_df = season_df.sort_values("custom_rank")

team_row = season_df[season_df["team"] == selected_team].iloc[0]

custom_average = season_df["custom_estat"].mean()
gini_average_delta = float(team_row["custom_estat"] - custom_average)

if abs(gini_average_delta) < 0.05:
    gini_context = f"Ranked #{int(team_row['custom_rank'])} - at {selected_season} Gini avg"
    gini_context_color = "#6B7280"
    gini_context_bg = "#F0F2F6"
elif gini_average_delta > 0:
    gini_context = (
        f"Ranked #{int(team_row['custom_rank'])} - "
        f"{gini_average_delta:.1f} above {selected_season} Gini avg"
    )
    gini_context_color = "#15803D"
    gini_context_bg = "#DCFCE7"
else:
    gini_context = (
        f"Ranked #{int(team_row['custom_rank'])} - "
        f"{abs(gini_average_delta):.1f} below {selected_season} Gini avg"
    )
    gini_context_color = "#DC2626"
    gini_context_bg = "#FEE2E2"

offense_rank_value = int(team_row["offense_rank"])
defense_rank_value = int(team_row["defense_rank"])
balance_gap = abs(offense_rank_value - defense_rank_value)

if offense_rank_value <= 12 and defense_rank_value <= 12:
    team_balance = "Two-Way"
    balance_text_color = "#15803D"
    balance_bg_color = "#DCFCE7"
elif balance_gap <= 5:
    team_balance = "Balanced"
    balance_text_color = "#6B7280"
    balance_bg_color = "#F0F2F6"
elif offense_rank_value < defense_rank_value:
    team_balance = "Offense-Led"
    balance_text_color = selected_team_color
    balance_bg_color = "#F8FAFC"
else:
    team_balance = "Defense-Led"
    balance_text_color = selected_team_color
    balance_bg_color = "#F8FAFC"

balance_badge = f"Off #{offense_rank_value} / Def #{defense_rank_value}"

top_sos_rank, top_sos_text_color, top_sos_bg_color = schedule_difficulty_info(
    season_df,
    selected_team,
)

k1, k2, k3, k4 = st.columns(4)
k1.markdown(
    f"""
<div class="metric-card">
    <div class="metric-label">{selected_team} Gini Score</div>
    <div class="metric-value">{team_row["custom_estat"]:.1f}</div>
    <div class="metric-badge" style="background:{gini_context_bg}; color:{gini_context_color};">
        {gini_context}
    </div>
</div>
""",
    unsafe_allow_html=True,
)
show_rank_metric(
    k2,
    f"{selected_team} Offense",
    team_row["offense_rank"],
    rank_badge(int(team_row["offense_rank"]), f"{team_row['offense_estat']:.1f}", " EStat"),
)
show_rank_metric(
    k3,
    f"{selected_team} Defense",
    team_row["defense_rank"],
    rank_badge(int(team_row["defense_rank"]), f"{team_row['defense_estat']:.1f}", " EStat"),
)
show_deep_dive_metric(
    k4,
    "Schedule Difficulty",
    f"{team_row['schedule_strength']:.1f}",
    top_sos_rank,
    top_sos_text_color,
    top_sos_bg_color,
)

tab_rankings, tab_scatter, tab_team, tab_schedule = st.tabs(
    [
        "Overall Rankings",
        "Offense vs Defense",
        "Team Deep Dive",
        "Schedule + Roster",
    ]
)

with tab_rankings:
    st.subheader(f"{selected_season} Overall Rankings")
    selected_team_rank = int(team_row["custom_rank"])
    default_top_n = max(15, selected_team_rank)
    top_n = st.slider(
        "How many teams to show?",
        5,
        32,
        default_top_n,
        key=f"top_n_{selected_season}_{selected_team}",
    )

    chart_df = season_df.head(top_n).sort_values("custom_estat", ascending=True)
    chart_df = chart_df.rename(
        columns={
            "team": "Team",
            "custom_estat": "Custom EStat",
            "custom_rank": "Rank",
            "offense_estat": "Offense EStat",
            "defense_estat": "Defense EStat",
            "point_diff_per_game": "Point Differential per Game",
            "schedule_strength": "Schedule Strength",
        }
    )
    chart_df["Bar Color"] = chart_df["Team"].apply(
        lambda team: selected_team_color if team == selected_team else neutral_bar
    )
    chart_df["Bar Outline"] = chart_df["Team"].apply(
        lambda team: selected_team_color2 if team == selected_team else "#FFFFFF"
    )
    chart_df["Outline Width"] = chart_df["Team"].apply(
        lambda team: 2 if team == selected_team else 0
    )

    fig = px.bar(
        chart_df,
        x="Custom EStat",
        y="Team",
        orientation="h",
        text="Custom EStat",
        hover_data={
            "Rank": True,
            "Offense EStat": ":.1f",
            "Defense EStat": ":.1f",
            "Point Differential per Game": ":.1f",
            "Schedule Strength": ":.1f",
        },
        title=f"Top {top_n} Teams by Custom EStat",
    )
    fig.update_traces(
        marker_color=chart_df["Bar Color"],
        marker_line_color=chart_df["Bar Outline"],
        marker_line_width=chart_df["Outline Width"],
        texttemplate="%{text:.1f}",
        textposition="outside",
        cliponaxis=False,
        width=0.72,
    )
    fig.update_layout(
        height=560,
        plot_bgcolor=plot_bg,
        paper_bgcolor=paper_bg,
        font=dict(color="#1F2937", size=12),
        title=dict(font=dict(color="#111827", size=15), x=0.0, xanchor="left"),
        xaxis=dict(
            title="Custom EStat",
            title_font=dict(color="#64748B", size=12),
            tickfont=dict(color="#64748B", size=11),
            showgrid=False,
            zeroline=False,
            showline=False,
        ),
        yaxis=dict(
            title="Team",
            title_font=dict(color="#64748B", size=12),
            tickfont=dict(color="#64748B", size=11),
            showgrid=False,
            zeroline=False,
            showline=False,
        ),
        margin=dict(l=65, r=70, t=55, b=45),
        bargap=0.28,
    )
    st.plotly_chart(fig, use_container_width=True)

    display_cols = [
        "custom_rank",
        "team",
        "custom_estat",
        "overall_estat",
        "offense_estat",
        "defense_estat",
        "point_diff_per_game",
        "off_adj_epa",
        "def_adj_epa",
        "success_margin",
        "turnover_margin_per_game",
        "schedule_strength",
        "sos_rank",
    ]
    rankings_table = season_df[display_cols].rename(
        columns={
            "custom_rank": "Rank",
            "team": "Team",
            "custom_estat": "Custom EStat",
            "overall_estat": "Default EStat",
            "offense_estat": "Offense",
            "defense_estat": "Defense",
            "point_diff_per_game": "Pt Diff/G",
            "off_adj_epa": "Adj Off EPA/Play",
            "def_adj_epa": "Adj Def EPA/Play",
            "success_margin": "Success Margin",
            "turnover_margin_per_game": "TO Margin/G",
            "schedule_strength": "Schedule Strength",
            "sos_rank": "SOS Rank",
        }
    )
    show_clean_table(
        rankings_table,
        fmt={
            "Custom EStat": "{:.1f}",
            "Default EStat": "{:.1f}",
            "Offense": "{:.1f}",
            "Defense": "{:.1f}",
            "Pt Diff/G": "{:.1f}",
            "Adj Off EPA/Play": "{:.3f}",
            "Adj Def EPA/Play": "{:.3f}",
            "Success Margin": "{:.3f}",
            "TO Margin/G": "{:.2f}",
            "Schedule Strength": "{:.1f}",
        },
        max_height=420,
        highlight_column="Team",
        highlight_value=selected_team,
        scroll_to_highlight=True,
        scroll_offset_rows=4,
    )

with tab_scatter:
    st.subheader("Offensive vs Defensive Performance")
    st.write(
        "Teams in the upper-right are balanced: strong offense and strong defense "
        "after opponent adjustment."
    )
    scatter_df = season_df.copy()

    logo_lookup = {}
    if not team_assets.empty and {"team_abbr", "team_logo_espn"}.issubset(team_assets.columns):
        logo_lookup = (
            team_assets
            .dropna(subset=["team_abbr", "team_logo_espn"])
            .set_index("team_abbr")["team_logo_espn"]
            .to_dict()
        )

    scatter_df["Logo URL"] = scatter_df["team"].map(logo_lookup).fillna("")

    # Use positive point differential to size logos.
    # Teams with negative point differential stay at the minimum logo size.
    scatter_df["bubble_size"] = scatter_df["point_diff_per_game"].clip(lower=0) + 1

    min_bubble_size = scatter_df["bubble_size"].min()
    max_bubble_size = scatter_df["bubble_size"].max()

    if pd.notna(max_bubble_size) and max_bubble_size != min_bubble_size:
        scatter_df["Logo Scale"] = 1.00 + 1.25 * (
            (scatter_df["bubble_size"] - min_bubble_size)
            / (max_bubble_size - min_bubble_size)
        )
    else:
        scatter_df["Logo Scale"] = 1.25

    scatter_df["Logo Source"] = scatter_df.apply(
        lambda row: logo_url_to_data_uri(
            row["Logo URL"],
            grayscale=row["team"] != selected_team,
        ),
        axis=1,
    )

    scatter_df["Logo Opacity"] = scatter_df["team"].apply(
        lambda team: 1.0 if team == selected_team else 0.42
    )

    scatter_df = scatter_df.rename(columns=LABELS)

    fig = px.scatter(
        scatter_df,
        x="Offense EStat",
        y="Defense EStat",
        size="bubble_size",
        size_max=48,
        hover_name="Team",
        hover_data={
            "Rank": True,
            "Custom EStat": ":.1f",
            "Opponent-Adjusted Offensive EPA per Play": ":.3f",
            "Opponent-Adjusted Defensive EPA": ":.3f",
            "Schedule Strength": ":.1f",
            "Point Differential per Game": ":.1f",
            "bubble_size": False,
            "Logo URL": False,
            "Logo Source": False,
            "Logo Scale": False,
            "Logo Opacity": False,
        },
        title=f"{selected_season} Offense vs Defense EStat",
    )

    # Hide the original Plotly bubbles. Logos become the visible bubbles.
    fig.update_traces(
        marker=dict(
            color="rgba(0,0,0,0)",
            opacity=0.01,
            line=dict(width=0),
        ),
        showlegend=False,
    )

    logo_df = scatter_df[
        scatter_df["Logo Source"].astype(str).str.len() > 0
    ].copy()

    if not logo_df.empty:
        x_span = max(
            logo_df["Offense EStat"].max() - logo_df["Offense EStat"].min(),
            1,
        )
        y_span = max(
            logo_df["Defense EStat"].max() - logo_df["Defense EStat"].min(),
            1,
        )

        base_logo_width = x_span * 0.065
        base_logo_height = y_span * 0.090

        # Draw selected team last so it sits on top if logos overlap.
        logo_df["Selected Sort"] = logo_df["Team"].apply(
            lambda team: 1 if team == selected_team else 0
        )
        logo_df = logo_df.sort_values("Selected Sort")

        for _, row in logo_df.iterrows():
            selected_boost = 1.12 if row["Team"] == selected_team else 1.0

            fig.add_layout_image(
                dict(
                    source=row["Logo Source"],
                    xref="x",
                    yref="y",
                    x=row["Offense EStat"],
                    y=row["Defense EStat"],
                    sizex=base_logo_width * row["Logo Scale"] * selected_boost,
                    sizey=base_logo_height * row["Logo Scale"] * selected_boost,
                    xanchor="center",
                    yanchor="middle",
                    sizing="contain",
                    opacity=row["Logo Opacity"],
                    layer="above",
                )
            )

    fig.add_hline(y=100, line_dash="dash", opacity=0.35)
    fig.add_vline(x=100, line_dash="dash", opacity=0.35)

    fig.update_layout(
        title=dict(
            text=f"{selected_season} Offense vs Defense EStat",
            font=dict(color="#111827", size=15),
            x=0.01,
            y=0.98,
            xanchor="left",
            yanchor="top",
        ),
        xaxis_title="Offense EStat",
        yaxis_title="Defense EStat",
        height=600,
        margin=dict(l=65, r=35, t=42, b=55),
        plot_bgcolor=plot_bg,
        paper_bgcolor=paper_bg,
        font=dict(color=text_color),
        xaxis=dict(
            gridcolor=grid_color,
            linecolor=border_color,
            tickfont=dict(color=muted_text),
        ),
        yaxis=dict(
            gridcolor=grid_color,
            linecolor=border_color,
            tickfont=dict(color=muted_text),
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f"""<div style="
margin-top: 0.85rem;
margin-bottom: 1.4rem;
padding: 1rem 1.15rem;
border-radius: 14px;
background: rgba(255,255,255,0.58);
border: 1px solid rgba(15, 23, 42, 0.10);
border-left: 5px solid {selected_team_color};
box-shadow: 0 10px 24px rgba(15, 23, 42, 0.055);
backdrop-filter: blur(5px);
">
<div style="
font-size: 0.95rem;
font-weight: 900;
color: #07111f;
margin-bottom: 0.45rem;
">
How to Read This Chart
</div>

<div style="
font-size: 0.92rem;
line-height: 1.65;
color: #334155;
">
This chart compares every team by <b>Offense EStat</b> and <b>Defense EStat</b>.
Teams farther to the right performed better offensively, while teams higher on the chart
performed better defensively. The upper-right area represents the most balanced teams.
Logo size is based on <b>positive point differential per game</b>, so teams that consistently
outscored opponents appear larger. The selected team stays in full color, while the rest of
the league is shown in gray to keep the focus on the team you selected.
</div>

<div style="
margin-top: 0.75rem;
padding-top: 0.7rem;
border-top: 1px solid rgba(15, 23, 42, 0.08);
font-size: 0.82rem;
line-height: 1.55;
color: #64748b;
">
Quick read: upper-right = strong offense and defense. Larger logo = stronger positive scoring margin.
Full-color logo = selected team.
</div>
</div>""",
        unsafe_allow_html=True,
    )

with tab_team:
    st.subheader(f"{selected_team} Deep Dive - {selected_season}")
    team_games = team_game[
        (team_game["season"] == selected_season) & (team_game["team"] == selected_team)
    ].copy()
    team_games = team_games.sort_values(["week", "game_id"])

    matchup_difficulty_weight = 0.20
    opponent_strength_lookup = season_df.set_index("team")["custom_estat"].to_dict()
    league_avg_strength = season_df["custom_estat"].mean()
    team_games["opponent_strength"] = team_games["opponent"].map(opponent_strength_lookup)
    team_games["opponent_strength"] = team_games["opponent_strength"].fillna(league_avg_strength)
    team_games["opponent_strength_adj"] = (
        team_games["opponent_strength"] - league_avg_strength
    )
    team_games["matchup_difficulty_adjustment"] = (
        matchup_difficulty_weight * team_games["opponent_strength_adj"]
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    pd_rank, pd_text_color, pd_bg_color = metric_rank_info(
        season_df,
        "point_diff_per_game",
        selected_team,
        higher_is_better=True,
    )
    off_rank, off_text_color, off_bg_color = metric_rank_info(
        season_df,
        "off_adj_epa",
        selected_team,
        higher_is_better=True,
    )
    def_rank, def_text_color, def_bg_color = metric_rank_info(
        season_df,
        "def_adj_epa",
        selected_team,
        higher_is_better=True,
    )
    to_rank, to_text_color, to_bg_color = metric_rank_info(
        season_df,
        "turnover_margin_per_game",
        selected_team,
        higher_is_better=True,
    )
    show_deep_dive_metric(
        c1,
        "Team Balance",
        team_balance,
        balance_badge,
        balance_text_color,
        balance_bg_color,
    )
    show_deep_dive_metric(
        c2,
        "Point Diff/G",
        f"{team_row['point_diff_per_game']:.1f}",
        pd_rank,
        pd_text_color,
        pd_bg_color,
    )
    show_deep_dive_metric(
        c3,
        "Adj Off EPA",
        f"{team_row['off_adj_epa']:.3f}",
        off_rank,
        off_text_color,
        off_bg_color,
    )
    show_deep_dive_metric(
        c4,
        "Adj Def EPA",
        f"{team_row['def_adj_epa']:.3f}",
        def_rank,
        def_text_color,
        def_bg_color,
    )
    show_deep_dive_metric(
        c5,
        "TO Margin/G",
        f"{team_row['turnover_margin_per_game']:.2f}",
        to_rank,
        to_text_color,
        to_bg_color,
    )

    metric_options = [
        "estat_game",
        "off_epa_per_play",
        "def_epa_allowed_per_play",
        "off_adj_epa",
        "def_adj_epa",
        "success_margin",
        "point_diff",
        "turnover_margin",
        "penalty_yards_margin",
    ]

    st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)
    metric_col, matchup_col = st.columns([1.4, 1])
    metric_col.markdown(
        "<div style='font-weight:700; color:#111827; margin-bottom:6px;'>Weekly Metric</div>",
        unsafe_allow_html=True,
    )
    metric_choice_label = metric_col.selectbox(
        "Weekly Metric",
        [LABELS.get(metric, metric) for metric in metric_options],
        index=0,
        label_visibility="collapsed",
    )
    metric_choice = {
        LABELS.get(metric, metric): metric for metric in metric_options
    }[metric_choice_label]

    matchup_available = metric_choice == "estat_game"
    matchup_col.markdown("<div style='height:38px;'></div>", unsafe_allow_html=True)
    include_matchup_difficulty = matchup_col.checkbox(
        "Matchup Difficulty",
        value=False,
        disabled=not matchup_available,
        help="Only available for Game EStat. Adjusts the weekly score based on opponent strength.",
    )

    if metric_choice == "estat_game" and include_matchup_difficulty:
        metric_choice = "opp_adj_estat_game"
        metric_choice_label = LABELS["opp_adj_estat_game"]

    # Put weekly Game EStat on a cleaner user-facing scale.
    # 0.0 raw Game EStat = 70.0
    # 0.5 raw Game EStat = 100.0
    chart_games = team_games.copy()
    chart_games["estat_game_display"] = 70 + 60 * chart_games["estat_game"]
    chart_games["opp_adj_estat_game_display"] = (
        chart_games["estat_game_display"]
        + chart_games["matchup_difficulty_adjustment"]
    )

    if metric_choice == "estat_game":
        chart_metric = "estat_game_display"
        chart_metric_label = "Game EStat"
    elif metric_choice == "opp_adj_estat_game":
        chart_metric = "opp_adj_estat_game_display"
        chart_metric_label = "Matchup-Adjusted Game EStat"
    else:
        chart_metric = metric_choice
        chart_metric_label = metric_choice_label

    # Keep raw column names for Plotly references; use labels only for display text.
    line_df = chart_games.copy()

    if metric_choice in ["turnover_margin", "penalty_yards_margin", "point_diff"]:
        metric_hover_format = ":.0f"
    elif metric_choice in ["estat_game", "opp_adj_estat_game"]:
        metric_hover_format = ":.1f"
    else:
        metric_hover_format = ":.3f"

    weekly_hover_data = {
        "week": True,
        chart_metric: metric_hover_format,
        "opponent": True,
        "home_away": True,
        "points_for": True,
        "points_against": True,
    }
    weekly_labels = {
        "week": "Week",
        chart_metric: chart_metric_label,
        "opponent": "Opponent",
        "home_away": "Home/Away",
        "points_for": "Points For",
        "points_against": "Points Against",
    }
    if metric_choice == "opp_adj_estat_game":
        weekly_hover_data.update(
            {
                "estat_game_display": ":.1f",
                "matchup_difficulty_adjustment": ":.1f",
                "opponent_strength": ":.1f",
            }
        )
        weekly_labels.update(
            {
                "estat_game_display": "Base Game EStat",
                "matchup_difficulty_adjustment": "Matchup Adjustment",
                "opponent_strength": "Opponent Strength",
            }
        )

    fig = px.line(
        line_df,
        x="week",
        y=chart_metric,
        markers=True,
        title=f"{selected_team} Weekly {chart_metric_label}",
        hover_data=weekly_hover_data,
        labels=weekly_labels,
    )
    fig.update_traces(
        line=dict(color=selected_team_color, width=4),
        marker=dict(
            color=selected_team_color2,
            size=10,
            line=dict(color=selected_team_color, width=2),
        ),
    )
    fig.update_layout(
        height=420,
        xaxis_title="Week",
        yaxis_title=chart_metric_label,
        plot_bgcolor=plot_bg,
        paper_bgcolor=paper_bg,
        font=dict(color=text_color),
        hoverlabel=dict(
            bgcolor=card_bg,
            font_size=13,
            font_color=text_color,
            bordercolor=selected_team_color,
        ),
    )
    max_week = int(line_df["week"].max()) if not line_df.empty else 18

    fig.update_xaxes(
        showgrid=False,
        linecolor=border_color,
        tickfont=dict(color=muted_text),
        tickmode="linear",
        tick0=1,
        dtick=1,
        range=[0.5, max_week + 0.5],
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=grid_color,
        linecolor=border_color,
        tickfont=dict(color=muted_text),
    )
    st.plotly_chart(fig, use_container_width=True)

    if metric_choice in ["estat_game", "opp_adj_estat_game"]:
        st.markdown(
            f"""<div style="
margin-top: 0.85rem;
margin-bottom: 1.2rem;
padding: 0.9rem 1.05rem;
border-radius: 12px;
background: #FFFFFF;
border: 1px solid rgba(15, 23, 42, 0.10);
border-left: 5px solid {selected_team_color};
box-shadow: 0 8px 20px rgba(15, 23, 42, 0.055);
font-size: 0.92rem;
line-height: 1.6;
color: #334155;
">
Weekly Game EStat is shown on a cleaner display scale that starts from a 70 baseline.
When matchup difficulty is turned on, the chart stays on that same display scale and adds
a small opponent-strength adjustment so the matchup effect can be compared directly.
</div>""",
            unsafe_allow_html=True,
        )

    selected_metric_for_help = metric_choice

    def render_metric_help(title, body_html, expanded=False):
        open_attr = " open" if expanded else ""
        st.markdown(
            f"""<details class="metric-help"{open_attr}><summary>{title}</summary><div class="metric-help-body">{body_html}</div></details>""",
            unsafe_allow_html=True,
        )

    st.markdown("#### Understanding the Metrics")

    render_metric_help(
        "Game EStat",
        "<p><b>Game EStat</b> is the main weekly score for how well a team played in one game.</p>"
        "<p>Instead of only looking at the final score, Game EStat combines efficiency, consistency, turnovers, penalties, and overall game control.</p>"
        "<p>In the weekly chart and game log, Game EStat is shown on a 70-based readability scale. Higher numbers mean stronger performances, and scores around 100 or better should read as strong weekly games.</p>",
        expanded=selected_metric_for_help in ["estat_game", "opp_adj_estat_game"],
    )

    render_metric_help(
        "Offensive EPA per Play",
        "<p><b>Offensive EPA per Play</b> measures how much value the offense creates on an average play.</p>"
        "<p>Higher is better for offense.</p>",
        expanded=selected_metric_for_help in ["off_epa_per_play", "off_adj_epa"],
    )

    render_metric_help(
        "Defensive EPA Allowed per Play",
        "<p><b>Defensive EPA Allowed per Play</b> measures how much value the defense allows the opposing offense to create.</p>"
        "<p>Lower is better in the raw stat because it means the defense is allowing less value per play.</p>",
        expanded=selected_metric_for_help in ["def_epa_allowed_per_play", "def_adj_epa"],
    )

    render_metric_help(
        "Opponent-Adjusted EPA",
        "<p><b>Opponent-Adjusted EPA</b> adds context to offensive and defensive performance by comparing what a team did against what that opponent usually allows or produces.</p>"
        "<p>This helps separate a strong performance against a good opponent from the same surface result against a weaker opponent.</p>",
        expanded=selected_metric_for_help in ["off_adj_epa", "def_adj_epa"],
    )

    render_metric_help(
        "Success Rate and Success Margin",
        "<p><b>Success Margin</b> compares the team's offensive success rate to the opponent's offensive success rate.</p>"
        "<p>A positive value means the team was more consistent play-to-play than its opponent.</p>",
        expanded=selected_metric_for_help == "success_margin",
    )

    render_metric_help(
        "Point Differential",
        "<p><b>Point Differential</b> shows how much a team won or lost by.</p>"
        "<p>It is useful, but this dashboard pairs it with efficiency stats so the final score is not the only signal.</p>",
        expanded=selected_metric_for_help == "point_diff",
    )

    render_metric_help(
        "Turnover Margin",
        "<p><b>Turnover Margin</b> shows whether a team gained or lost extra possessions.</p>"
        "<p>Positive values mean the team forced more turnovers than it committed.</p>",
        expanded=selected_metric_for_help == "turnover_margin",
    )

    render_metric_help(
        "Penalty Yard Margin",
        "<p><b>Penalty Yard Margin</b> shows whether penalties helped or hurt a team's field position.</p>"
        "<p>Positive values mean the opponent gave up more penalty yards.</p>",
        expanded=selected_metric_for_help == "penalty_yards_margin",
    )

    render_metric_help(
        "Matchup Difficulty",
        "<p><b>Matchup Difficulty</b> adjusts Game EStat based on opponent strength. A strong performance against a good team gets more credit than the same performance against a weaker team.</p>"
        "<p>When matchup difficulty is turned on, the weekly score stays on the same 70-based display scale so it can be read the same way as regular Game EStat.</p>",
        expanded=selected_metric_for_help == "opp_adj_estat_game",
    )

    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)
    st.markdown(f"#### {selected_team} Game Log - {selected_season}")
    game_cols = [
        "week",
        "opponent",
        "home_away",
        "points_for",
        "points_against",
        "point_diff",
        "off_epa_per_play",
        "off_adj_epa",
        "def_epa_allowed_per_play",
        "def_adj_epa",
        "off_success_rate",
        "def_success_allowed",
        "turnover_margin",
        "penalty_yards_margin",
        "estat_game_display",
    ]
    game_log_table = chart_games[game_cols].rename(
        columns={
            **LABELS,
            "estat_game_display": "Game EStat",
        }
    )
    show_clean_table(
        game_log_table,
        fmt={
            "Points For": "{:.0f}",
            "Points Against": "{:.0f}",
            "Point Differential": "{:.0f}",
            "Turnover Margin": "{:.0f}",
            "Penalty Yard Margin": "{:.0f}",
            "Offensive EPA per Play": "{:.3f}",
            "Opponent-Adjusted Offensive EPA per Play": "{:.3f}",
            "Defensive EPA Allowed per Play": "{:.3f}",
            "Opponent-Adjusted Defensive EPA": "{:.3f}",
            "Offensive Success Rate": "{:.1%}",
            "Defensive Success Rate Allowed": "{:.1%}",
            "Game EStat": "{:.1f}",
        },
        max_height=420,
    )

with tab_schedule:
    st.subheader("Schedule + Season Roster")
    if not games.empty:
        st.markdown(f"**Schedule for {selected_team} - {selected_season} Season**")
        sched = games[
            (games["season"] == selected_season)
            & ((games["home_team"] == selected_team) | (games["away_team"] == selected_team))
        ].copy()
        if sched.empty:
            st.info(f"No schedule rows found for {selected_team} in {selected_season}.")
        else:
            sched = sched.sort_values(["week", "gameday"])
            sched_cols = [
                column
                for column in [
                    "week",
                    "gameday",
                    "weekday",
                    "gametime",
                    "away_team",
                    "away_score",
                    "home_team",
                    "home_score",
                    "stadium",
                    "roof",
                    "surface",
                    "spread_line",
                    "total_line",
                ]
                if column in sched.columns
            ]

            # Clean schedule table display values
            sched_display = sched[sched_cols].copy()

            if "away_score" in sched_display.columns:
                sched_display["away_score"] = sched_display["away_score"].apply(format_whole_number)

            if "home_score" in sched_display.columns:
                sched_display["home_score"] = sched_display["home_score"].apply(format_whole_number)

            if "gametime" in sched_display.columns:
                sched_display["gametime"] = sched_display["gametime"].apply(
                    lambda value: str(value).strip()[:5] if pd.notna(value) else "-"
                )

            if "roof" in sched_display.columns:
                sched_display["roof"] = sched_display["roof"].apply(
                    lambda value: str(value).strip().capitalize() if pd.notna(value) else "-"
                )

            if "surface" in sched_display.columns:
                def clean_surface(value):
                    if pd.isna(value):
                        return "-"
                    surface_value = str(value).strip().lower()
                    if surface_value == "grass":
                        return "Grass"
                    if "turf" in surface_value:
                        return "Turf"
                    return surface_value.capitalize()

                sched_display["surface"] = sched_display["surface"].apply(clean_surface)

            show_clean_table(
                sched_display.rename(columns=LABELS),
                max_height=620,
                extra_class="schedule-table",
            )
    else:
        st.info("No schedule file found in the data folder.")

    st.markdown(
        """
<div style="
    height: 2.25rem;
    border-top: 1px solid rgba(15, 23, 42, 0.12);
    margin-top: 0.6rem;
"></div>
""",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""<div style="
font-size: 1.35rem;
line-height: 1.15;
font-weight: 800;
color: #111827;
margin: 0 0 -0.45rem 0;
">
{selected_team} Season Roster - {selected_season}
</div>""",
        unsafe_allow_html=True,
    )
    if not roster.empty:
        st.markdown("<div style='height: 0rem;'></div>", unsafe_allow_html=True)

        roster_df = roster[
            (roster["season"] == selected_season) & (roster["team"] == selected_team)
        ].copy()
        position_options = ["All"]
        if "position" in roster_df.columns:
            position_options += sorted(roster_df["position"].dropna().unique().tolist())

        c_pos, c_search = st.columns([1, 1], gap="medium")
        roster_position = c_pos.selectbox(
            "Position",
            position_options,
        )
        name_search = c_search.text_input("Player search")

        st.markdown("<div style='height: 0.05rem;'></div>", unsafe_allow_html=True)

        if roster_position != "All" and "position" in roster_df.columns:
            roster_df = roster_df[roster_df["position"] == roster_position]
        if name_search and "full_name" in roster_df.columns:
            roster_df = roster_df[
                roster_df["full_name"].str.contains(name_search, case=False, na=False)
            ]

        status_labels = {
            "ACT": "Active",
            "INA": "Inactive",
            "RES": "Reserve / Injured",
            "PUP": "Physically Unable to Perform",
            "NON": "Non-Football Injury",
            "SUS": "Suspended",
            "CUT": "Cut",
            "DEV": "Practice Squad",
        }
        if "status" in roster_df.columns:
            roster_df["status"] = roster_df["status"].replace(status_labels)

        roster_cols = [
            column
            for column in [
                "season",
                "team",
                "position",
                "depth_chart_position",
                "jersey_number",
                "status",
                "full_name",
                "height",
                "weight",
                "college",
                "years_exp",
                "entry_year",
                "draft_club",
            ]
            if column in roster_df.columns
        ]
        display_roster = roster_df[roster_cols].rename(
            columns={
                "season": "Season",
                "team": "Team",
                "position": "Position",
                "depth_chart_position": "Depth Chart Position",
                "jersey_number": "Jersey Number",
                "status": "Roster Status",
                "full_name": "Player Name",
                "height": "Height",
                "weight": "Weight",
                "college": "College",
                "years_exp": "Years Experience",
                "entry_year": "Entry Year",
                "draft_club": "Draft Team",
            }
        )

        # Clean roster table display values
        for column in ["Jersey Number", "Height", "Weight", "Entry Year"]:
            if column in display_roster.columns:
                display_roster[column] = display_roster[column].apply(format_whole_number)

        if "Draft Team" in display_roster.columns:
            display_roster["Draft Team"] = display_roster["Draft Team"].apply(
                lambda value: "Undrafted" if pd.isna(value) or str(value).strip() == "" else str(value).strip()
            )

        show_clean_table(
            display_roster,
            max_height=420,
            extra_class="roster-table-tight",
            top_margin_px=-8,
        )

        st.markdown("<div style='height: 0.55rem;'></div>", unsafe_allow_html=True)

        st.caption(
            "Note: This is a season roster, not an exact 53-man roster snapshot. "
            "It includes players listed with the team during that season."
        )
    else:
        st.info("No roster file found in the data folder.")

st.divider()
st.caption(
    "Data prepared from nflverse "
    "play-by-play, game, and roster files."
)
