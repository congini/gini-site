import html

import streamlit as st
from pathlib import Path
import sys


st.set_page_config(
    page_title="Gini Metric Terminology",
    layout="wide",
    initial_sidebar_state="collapsed",
)

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from site_nav import render_top_nav

PRIMARY = "#F15A24"
SECONDARY = "#0073B7"


def esc(value):
    return html.escape(str(value))


def render_entry(entry):
    name = esc(entry["name"])
    meaning = esc(entry["meaning"])
    reading = esc(entry.get("reading", ""))

    reading_html = ""
    if reading:
        reading_html = (
            '<div class="term-reading">'
            '<div class="term-reading-label">How to read it</div>'
            f"<div>{reading}</div>"
            "</div>"
        )

    st.markdown(
        f"""
<div class="term-card">
    <div class="term-name-row">
        <div class="term-marker"></div>
        <div class="term-name">{name}</div>
    </div>
    <div class="term-body">
        <div class="term-body-label">What it means</div>
        <div class="term-meaning">{meaning}</div>
    </div>
{reading_html}
</div>
""",
        unsafe_allow_html=True,
    )


def render_section(title, intro, entries):
    st.markdown(
        f"""
<div class="section-title">{esc(title)}</div>
<div class="section-intro">{esc(intro)}</div>
""",
        unsafe_allow_html=True,
    )

    for entry in entries:
        render_entry(entry)


current_dashboard_entries = [
    {
        "name": "Gini Metric",
        "meaning": "The dashboard's main team-quality score. It summarizes how well a team performed across the major areas that tend to decide football games: offense, defense, scoreboard control, consistency, turnovers, penalties, and schedule context.",
        "reading": "Around 100 is roughly average for that season. Higher scores point to stronger overall performance after accounting for the league environment in that season.",
    },
    {
        "name": "Model Weights",
        "meaning": "The controls inside the Customize Model Weights panel that let users test different football philosophies, such as valuing offense more, defense more, or schedule difficulty more.",
        "reading": "Changing the sliders changes the custom rankings on the page, but it does not change the underlying game data.",
    },
    {
        "name": "Team Profile",
        "meaning": "The summary panel in the dashboard hero that shows the currently selected season and full team name.",
        "reading": "Use this as the quick confirmation of which team-season view is active before reading the rankings, charts, and tables below.",
    },
    {
        "name": "Dashboard Setup",
        "meaning": "The control area below the hero where users select the team and season shown throughout the dashboard.",
        "reading": "The team dropdown uses full team names. The season dropdown changes the year, and both controls update the dashboard view.",
    },
    {
        "name": "Score Scale",
        "meaning": "The dashboard uses different public scales depending on what is being shown. Season ratings are centered around league average, weekly Game EStat is shown on a cleaner display scale, and schedule strength is treated as context.",
        "reading": "For season ratings, around 100 means roughly average for that season and higher is better. For weekly Game EStat, around 100 or better is a strong game. For Schedule Strength, higher means tougher opponents rather than a better team.",
    },
    {
        "name": "Selected-Team Gini Card",
        "meaning": "The main score card for the team selected in the dashboard controls.",
        "reading": "The badge combines the team's selected-season rank with how far its Gini score sits above or below that season's Gini average.",
    },
    {
        "name": "Schedule Difficulty",
        "meaning": "The top-card view of how difficult the selected team's schedule was.",
        "reading": "Higher means a tougher schedule. It is context for the team's results, not a team-quality grade by itself.",
    },
    {
        "name": "Team Balance",
        "meaning": "A quick deep-dive label that compares the selected team's offensive and defensive ranks.",
        "reading": "Two-Way means both units are strong. Balanced means offense and defense are close in rank. Offense-Led or Defense-Led points to the stronger side.",
    },
    {
        "name": "Custom EStat",
        "meaning": "The season-level team rating produced by the current model weight settings.",
        "reading": "It uses the same general season-rating scale as the main Gini Metric: around 100 is roughly average for that season, and higher is better.",
    },
    {
        "name": "Default EStat",
        "meaning": "The prebuilt season-level score saved with the dashboard data. It gives every team a consistent baseline before any user changes the model weight sliders.",
        "reading": "Use Default EStat as the baseline season view. Around 100 is roughly average for that season. Use Custom EStat when you want the rankings based on your selected weights.",
    },
    {
        "name": "Overall Rankings Table",
        "meaning": "The full 1-32 team table in the Overall Rankings tab. It shows every team, while the selected team is outlined in the active team colors.",
        "reading": "When possible, the table opens with the selected team around the fifth visible row, but users can still scroll through the full league table.",
    },
    {
        "name": "Super Square",
        "meaning": "A historical Super Bowl contender zone built from regular-season team data. A team is inside the Super Square when it ranks top 6 in Gini Score, top 6 in point differential per game, and has at least one top-12 unit on offense or defense.",
        "reading": "Inside the square means the team matched the dashboard's strongest historical contender profile for that season. It is a signal, not a prediction guarantee.",
    },
    {
        "name": "On the Cusp",
        "meaning": "A Super Square label for teams that are close to the contender zone but do not meet all of the inside-square requirements.",
        "reading": "Cusp teams usually have a strong partial profile, such as being near the top in Gini Score or point differential, but they are missing at least one part of the full Super Square filter.",
    },
    {
        "name": "Reveal Super Bowl Winner",
        "meaning": "The optional Super Square checkbox that visually identifies the team that won the Super Bowl after the selected regular season.",
        "reading": "Turning it on helps compare the champion against the Super Square zone for that season without changing the chart's filtering rules.",
    },
    {
        "name": "Offense EStat",
        "meaning": "A season-relative offensive score that reflects how much value a team's offense created while considering the defenses it faced.",
        "reading": "Around 100 is roughly average for that season. Higher means the offense was stronger than average for its season context.",
    },
    {
        "name": "Defense EStat",
        "meaning": "A season-relative defensive score that reflects how well a defense limited opponents while considering the offenses it faced.",
        "reading": "Around 100 is roughly average for that season. Higher means better defensive performance in the dashboard's scoring direction.",
    },
    {
        "name": "Season Standardization",
        "meaning": "A context step that compares teams to the league environment they actually played in before showing them on the dashboard scale.",
        "reading": "This helps compare teams across years without pretending every NFL season had the same scoring environment.",
    },
    {
        "name": "Game EStat",
        "meaning": "A game-level performance score that looks beyond the final score by evaluating how well a team played underneath the result.",
        "reading": "Weekly Game EStat is shown on a cleaner 70-based display scale. Higher scores point to stronger all-around games, while scores around 100 or better should read as strong weekly performances.",
    },
    {
        "name": "Matchup-Adjusted Game EStat",
        "meaning": "An optional weekly view that gives extra context for the strength of the opponent in that game.",
        "reading": "It stays on the same 70-based display scale as regular Game EStat and adds a small opponent-strength adjustment so the matchup effect can be compared directly.",
    },
]


game_level_entries = [
    {
        "name": "EPA",
        "meaning": "Expected Points Added estimates whether a play improved or hurt a team's scoring outlook.",
        "reading": "Positive offensive EPA is good for the offense. Lower EPA allowed is generally better for a defense.",
    },
    {
        "name": "Offensive EPA per Play",
        "meaning": "A per-play view of how much value the offense creates when it has the ball.",
        "reading": "Higher usually points to a more efficient offense.",
    },
    {
        "name": "Defensive EPA Allowed per Play",
        "meaning": "A per-play view of how much value the defense allows opposing offenses to create.",
        "reading": "Lower raw values are better, while dashboard-adjusted defensive scores are oriented so higher is better.",
    },
    {
        "name": "Opponent-Adjusted Offensive EPA",
        "meaning": "An offensive efficiency measure that gives context for the quality of the defense faced.",
        "reading": "Strong production against a strong defense is treated differently than similar production against a weak defense.",
    },
    {
        "name": "Opponent-Adjusted Defensive EPA",
        "meaning": "A defensive efficiency measure that gives context for the quality of the offense faced.",
        "reading": "It rewards defenses for limiting offenses that usually create more value.",
    },
    {
        "name": "Success Rate",
        "meaning": "The share of plays that keep a team on schedule based on down, distance, and game situation.",
        "reading": "It is a consistency measure, not just an explosiveness measure.",
    },
    {
        "name": "Defensive Success Rate Allowed",
        "meaning": "How often the defense allows the opposing offense to stay on schedule.",
        "reading": "Lower is better because it means the defense is creating more difficult situations.",
    },
    {
        "name": "Success Margin",
        "meaning": "A comparison of a team's offensive consistency against the consistency it allowed on defense.",
        "reading": "Positive values generally mean the team controlled the game more reliably from play to play.",
    },
    {
        "name": "Point Differential",
        "meaning": "How much a team won or lost by in a game.",
        "reading": "It is important, but it can be noisy without efficiency and opponent context.",
    },
    {
        "name": "Turnover Margin",
        "meaning": "Whether a team gained or lost extra possessions through takeaways and giveaways.",
        "reading": "Positive is generally good because it means the team created more possession value.",
    },
    {
        "name": "Penalty Yard Margin",
        "meaning": "Whether penalties created a field-position advantage or disadvantage.",
        "reading": "Positive is generally good because the opponent gave away more penalty yardage.",
    },
    {
        "name": "Pass Rate",
        "meaning": "How often a team chooses or is forced into passing relative to its offensive play mix.",
        "reading": "It helps describe offensive identity, game script, and style rather than pure team quality.",
    },
]


season_level_entries = [
    {
        "name": "Point Differential per Game",
        "meaning": "A season-level view of scoring margin adjusted to a game-by-game lens.",
        "reading": "Higher usually reflects stronger scoreboard control across the season.",
    },
    {
        "name": "Turnover Margin per Game",
        "meaning": "A season-level view of possession advantage adjusted to a game-by-game lens.",
        "reading": "It can explain why a team's record differs from its efficiency profile.",
    },
    {
        "name": "Penalty Yard Margin per Game",
        "meaning": "A season-level view of penalty-field-position advantage adjusted to a game-by-game lens.",
        "reading": "It adds discipline and hidden-yardage context without overpowering the core efficiency metrics.",
    },
    {
        "name": "Yards per Play",
        "meaning": "A basic offensive efficiency stat that shows how much yardage a team gains on a typical play.",
        "reading": "It is useful context, but EPA-based metrics usually capture game value more directly.",
    },
    {
        "name": "Schedule Strength",
        "meaning": "A season-level view of how difficult a team's opponents were.",
        "reading": "Higher means the team faced a tougher set of opponents. It is a context score, not a direct team-quality grade.",
    },
    {
        "name": "Strength of Schedule Rank",
        "meaning": "A rank of schedule difficulty across the league for the selected season.",
        "reading": "A top schedule rank means the team faced one of the hardest schedules.",
    },
    {
        "name": "Overall Rank, Offense Rank, Defense Rank",
        "meaning": "Season-relative placements for overall team quality, offensive performance, and defensive performance.",
        "reading": "Rank 1 is best. Ties may share the same placement.",
    },
]


context_entries = [
    {
        "name": "Schedule Fields",
        "meaning": "Game metadata such as week, date, teams, scores, venue, roof, surface, and timing.",
        "reading": "These fields help users understand the environment around each result.",
    },
    {
        "name": "Betting Market Fields",
        "meaning": "Market context such as spread, total, and moneyline information carried in the schedule file.",
        "reading": "These fields are useful for future analysis but are not presented as core public model ingredients.",
    },
    {
        "name": "Rest and Division Fields",
        "meaning": "Context about rest timing and whether a game was played inside the division.",
        "reading": "These fields can support future matchup analysis.",
    },
    {
        "name": "Roster Fields",
        "meaning": "Player and team roster attributes such as season, team, position, player name, jersey, status, body profile, college, and draft background.",
        "reading": "The roster page is season-level context, not an exact active-roster snapshot for every game day.",
    },
]


st.markdown(
    f"""
<style>
.stApp,
[data-testid="stAppViewContainer"] {{
    background: #ffffff !important;
    overflow-x: hidden;
}}

.bg-canvas {{
    position: fixed;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    pointer-events: none;
    z-index: 0;
    overflow: visible;
}}

.bg-canvas::before {{
    content: "";
    position: absolute;
    top: -20%;
    left: -10%;
    width: 150%;
    height: 150%;
    background: radial-gradient(ellipse at center, rgba(241, 90, 36, 0.28) 0%, rgba(241, 90, 36, 0.08) 30%, transparent 60%);
    animation: glowPulseOrange 22s ease-in-out infinite;
}}

.bg-canvas::after {{
    content: "";
    position: absolute;
    bottom: -20%;
    right: -10%;
    width: 150%;
    height: 150%;
    background: radial-gradient(ellipse at center, rgba(0, 115, 183, 0.24) 0%, rgba(0, 115, 183, 0.08) 30%, transparent 60%);
    animation: glowPulseBlue 24s ease-in-out infinite;
}}

@keyframes glowPulseOrange {{
    0%   {{ opacity: 0.5; transform: translate(-10%, 0%)   scale(1);    }}
    25%  {{ opacity: 1;   transform: translate(40%, -20%)  scale(1.2);  }}
    50%  {{ opacity: 0.7; transform: translate(80%, 30%)   scale(1.3);  }}
    75%  {{ opacity: 1;   transform: translate(30%, 60%)   scale(1.1);  }}
    100% {{ opacity: 0.5; transform: translate(-10%, 0%)   scale(1);    }}
}}

@keyframes glowPulseBlue {{
    0%   {{ opacity: 1;   transform: translate(10%, 0%)    scale(1);    }}
    25%  {{ opacity: 0.5; transform: translate(-30%, 40%)  scale(1.2);  }}
    50%  {{ opacity: 0.8; transform: translate(-60%, -20%) scale(1.28); }}
    75%  {{ opacity: 0.5; transform: translate(20%, -50%)  scale(1.1);  }}
    100% {{ opacity: 1;   transform: translate(10%, 0%)    scale(1);    }}
}}

[data-testid="stAppViewContainer"] > .main {{
    position: relative;
    z-index: 1;
}}

[data-testid="stSidebar"] {{
    z-index: 2;
}}

[data-testid="stHeader"] {{
    background: rgba(255,255,255,0.72) !important;
    backdrop-filter: blur(8px);
}}

.block-container {{
    position: relative;
    z-index: 1;
    padding-top: 0.35rem !important;
    padding-bottom: 2.6rem !important;
}}

.terminology-page {{
    position: relative;
    z-index: 1;
    max-width: 1500px;
    margin: -3.25rem auto 0 auto;
    padding: 0 1.5rem 3rem 1.5rem;
}}

.terminology-hero {{
    position: relative;
    background: linear-gradient(135deg, {SECONDARY} 0%, #172131 58%, #273444 100%);
    color: #ffffff;
    border-radius: 18px;
    padding: 2.6rem 3rem;
    box-shadow: 0 22px 45px rgba(0, 0, 0, 0.16);
    border: 1px solid rgba(255,255,255,0.13);
    overflow: hidden;
    margin-top: -0.75rem;
    margin-bottom: 1.4rem;
}}

.terminology-hero::before {{
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    height: 6px;
    width: 100%;
    background: linear-gradient(90deg, {PRIMARY}, rgba(255,255,255,0.35), {SECONDARY});
}}

.terminology-kicker {{
    color: {PRIMARY};
    font-size: 0.82rem;
    font-weight: 900;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.7rem;
}}

.terminology-title {{
    font-size: 2.55rem;
    line-height: 1.05;
    font-weight: 950;
    margin-bottom: 0.9rem;
}}

.terminology-text {{
    max-width: 980px;
    color: rgba(255,255,255,0.92);
    font-size: 1.02rem;
    line-height: 1.75;
}}

.section-title {{
    font-size: 1.45rem;
    font-weight: 950;
    margin: 2.2rem 0 0.4rem 0;
    color: #07111f;
}}

.section-title::after {{
    content: "";
    display: block;
    width: 58px;
    height: 4px;
    border-radius: 999px;
    margin-top: 0.45rem;
    background: linear-gradient(90deg, {PRIMARY}, {SECONDARY});
}}

.section-intro {{
    max-width: 950px;
    color: #475569;
    font-size: 0.96rem;
    line-height: 1.7;
    margin-bottom: 1rem;
}}

.term-card {{
    position: relative;
    background: linear-gradient(180deg, #ffffff 0%, #fbfcfe 100%);
    border: 1px solid rgba(15, 23, 42, 0.11);
    border-radius: 12px;
    padding: 1.15rem 1.2rem 1.05rem 1.2rem;
    margin-bottom: 0.85rem;
    box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
    overflow: hidden;
}}

.term-card::before {{
    content: "";
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 5px;
    background: linear-gradient(180deg, {PRIMARY}, {SECONDARY});
}}

.term-card:hover {{
    border-color: rgba(0, 115, 183, 0.28);
    box-shadow: 0 14px 32px rgba(15, 23, 42, 0.085);
}}

.term-name-row {{
    display: flex;
    align-items: center;
    gap: 0.55rem;
    margin-bottom: 0.8rem;
}}

.term-marker {{
    width: 9px;
    height: 9px;
    border-radius: 999px;
    background: {PRIMARY};
    box-shadow: 0 0 0 4px rgba(241, 90, 36, 0.11);
}}

.term-name {{
    color: #07111f;
    font-size: 1.06rem;
    font-weight: 950;
    line-height: 1.25;
}}

.term-body {{
    padding-left: 1.05rem;
}}

.term-body-label,
.term-reading-label {{
    color: {SECONDARY};
    font-size: 0.72rem;
    font-weight: 900;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.32rem;
}}

.term-meaning {{
    color: #233044;
    line-height: 1.65;
    font-size: 0.96rem;
}}

.term-reading {{
    margin-top: 0.85rem;
    margin-left: 1.05rem;
    padding: 0.78rem 0.9rem;
    border-radius: 10px;
    background: #F8FAFC;
    border: 1px solid rgba(0, 34, 68, 0.10);
    border-left: 4px solid {SECONDARY};
    color: #334155;
    font-size: 0.9rem;
    line-height: 1.55;
}}

.stTabs [data-baseweb="tab-list"] {{
    gap: 0.25rem;
    border-bottom: 1px solid #E5E7EB;
}}

.stTabs [data-baseweb="tab"] {{
    color: #111827;
    font-weight: 750;
    padding: 0.75rem 0.8rem;
}}

.stTabs [aria-selected="true"] {{
    color: {SECONDARY} !important;
}}

.stTabs [data-baseweb="tab-highlight"] {{
    background-color: {PRIMARY} !important;
    height: 3px;
}}

@media screen and (max-width: 900px) {{
    .terminology-hero {{
        padding: 2rem;
    }}

    .terminology-title {{
        font-size: 2.05rem;
    }}

    .term-body,
    .term-reading {{
        margin-left: 0;
        padding-left: 0;
    }}
}}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="bg-canvas" aria-hidden="true"></div>', unsafe_allow_html=True)
render_top_nav("Terminology", PRIMARY, SECONDARY)
st.markdown('<div class="terminology-page">', unsafe_allow_html=True)

st.markdown(
    """
<div class="terminology-hero">
    <div class="terminology-kicker">Gini Metric Documentation</div>
    <div class="terminology-title">Terminology</div>
    <div class="terminology-text">
        A public-facing guide to the stats, labels, and football concepts used throughout the dashboard.
        This page explains what each term means and how to read it in the context of the dashboard.
    </div>
</div>
""",
    unsafe_allow_html=True,
)

tab_current, tab_game, tab_season, tab_context = st.tabs(
    [
        "Current Model",
        "Game Stats",
        "Season Stats",
        "Context",
    ]
)

with tab_current:
    render_section(
        "Current Dashboard Terms",
        "The public labels and scoring concepts used in the current Gini Metric experience.",
        current_dashboard_entries,
    )

with tab_game:
    render_section(
        "Game-Level Stats",
        "Terms from play-by-play data that power the weekly chart, game log, and team deep dive.",
        game_level_entries,
    )

with tab_season:
    render_section(
        "Season-Level Stats",
        "Team-season aggregates used for rankings, top cards, deep-dive cards, and schedule context.",
        season_level_entries,
    )

with tab_context:
    render_section(
        "Schedule, Market, and Roster Context",
        "Supporting fields that help explain game environments, team context, and future analysis options.",
        context_entries,
    )

st.markdown("</div>", unsafe_allow_html=True)
