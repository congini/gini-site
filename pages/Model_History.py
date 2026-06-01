import html
from pathlib import Path
import sys

import streamlit as st


st.set_page_config(
    page_title="Gini Metric Model History",
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


def render_stage(stage):
    finding_items = "".join(
        f"<li>{esc(item)}</li>" for item in stage.get("findings", [])
    )
    impact_items = "".join(
        f"<li>{esc(item)}</li>" for item in stage.get("impact", [])
    )

    st.markdown(
        f"""
<div class="history-stage">
    <div class="stage-rail">
        <div class="stage-dot"></div>
        <div class="stage-line"></div>
    </div>
    <div class="stage-card">
        <div class="stage-kicker">{esc(stage["label"])}</div>
        <div class="stage-title">{esc(stage["name"])}</div>
        <div class="stage-purpose">{esc(stage["purpose"])}</div>
        <div class="stage-grid">
            <div class="stage-panel">
                <div class="panel-label">What it found</div>
                <ul>{finding_items}</ul>
            </div>
            <div class="stage-panel">
                <div class="panel-label">What carried forward</div>
                <ul>{impact_items}</ul>
            </div>
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_finding(title, body):
    st.markdown(
        f"""
<div class="finding-card">
    <div class="finding-title-row">
        <div class="finding-dot"></div>
        <div class="finding-title">{esc(title)}</div>
    </div>
    <div class="finding-body">{esc(body)}</div>
</div>
""",
        unsafe_allow_html=True,
    )


model_stages = [
    {
        "label": "Foundation | Original Gini/EStat",
        "name": "Turning Team Quality Into a Readable Score",
        "purpose": "The project started with one question: how can NFL teams be compared in a way that sees beyond wins, losses, and final scores?",
        "findings": [
            "Records can hide whether a team is actually controlling games.",
            "Efficiency, point differential, turnovers, penalties, and schedule context all change the story.",
            "A single score is most useful when users can still see the ingredients behind it.",
        ],
        "impact": [
            "Created the EStat/Gini foundation used across the site.",
            "Made season-relative scoring a core design choice so teams are judged against their actual league year.",
            "Set the goal for every later page: explain football performance clearly, not just numerically.",
        ],
    },
    {
        "label": "Core Tool | Gini Dashboard",
        "name": "The Main Team Evaluation Workspace",
        "purpose": "The Gini Dashboard became the home base for evaluating any team-season, comparing rankings, and testing different football philosophies.",
        "findings": [
            "Users need the top-line score plus the offense, defense, schedule, and weekly game details behind it.",
            "Custom weights make the model more useful because different users value football ingredients differently.",
            "Team logos, colors, charts, and sortable tables make the model easier to explore.",
        ],
        "impact": [
            "Established the main ranking table, offense/defense chart, team deep dive, schedule, and roster view.",
            "Kept the public model readable while preserving the deeper formulas underneath.",
            "Turned the Gini score into an interactive evaluation tool instead of a static spreadsheet output.",
        ],
    },
    {
        "label": "Contender Tool | Super Square",
        "name": "Studying Championship Profiles",
        "purpose": "Super Square was added to answer a different question: which regular-season profiles look like the profiles past Super Bowl champions tended to have?",
        "findings": [
            "Championship teams usually show both broad control and at least one pressure point that can stress opponents.",
            "A contender profile is stronger when it combines Gini/EStat checkpoints, point differential, success margin, EPA, and unit strength.",
            "Some teams sit close to the profile without fully clearing it, which makes cusp teams useful to track.",
        ],
        "impact": [
            "Created a championship-contender profile tool separate from the main rankings.",
            "Gave users a way to compare a selected season against historical champion patterns.",
            "Made the site broader than team ranking by adding a postseason-contender lens.",
        ],
    },
    {
        "label": "Current-Moment Tool | Live Leaderboard",
        "name": "Live Roster and Market-Style Ranking",
        "purpose": "The Live Leaderboard was built for the current NFL moment, where roster strength, projected wins, live source refreshes, and movement matter more than historical season browsing.",
        "findings": [
            "A current team rating needs roster context, not just last season's performance.",
            "Local nflverse/nflreadpy source files can support refreshed schedules, rosters, injuries, snap counts, transactions, and player production.",
            "Movement is most useful when it compares the current board to the most recent live or weekly snapshot.",
        ],
        "impact": [
            "Added Live Market Score, roster score, projected wins, quadrant probabilities, and refresh movement.",
            "Separated current-moment ranking from the historical Gini Dashboard.",
            "Created weekly history while still allowing the page to refresh local data every five minutes.",
        ],
    },
    {
        "label": "Forecasting Tool | Predictive Model",
        "name": "Regular-Season and Playoff Outlooks",
        "purpose": "The Predictive Model extends the platform from evaluation into forecasting by using completed team profiles to project wins and postseason outlooks.",
        "findings": [
            "Forecasts are clearer when the page separates prediction target year from feature year.",
            "Expected wins, win pace, regression risk, and improvement signals help explain why a forecast moves.",
            "The playoff predictor should stay locked until the regular season is complete so it does not mix future guesses with current-year playoff outcomes.",
        ],
        "impact": [
            "Added a regular-season wins predictor with likely range and model-result language.",
            "Added a playoff predictor that evaluates completed regular-season playoff profiles.",
            "Completed the site's four-part structure: evaluate, rank live, profile contenders, and forecast outcomes.",
        ],
    },
]

key_findings = [
    (
        "The tools answer different questions",
        "Gini Dashboard evaluates team quality, Live Leaderboard ranks the current market, Super Square studies contender profiles, and Predictive Model forecasts future outcomes.",
    ),
    (
        "Evaluation comes before prediction",
        "The forecasting page is stronger because it starts from the same team-quality language built in the Gini and Super Square work.",
    ),
    (
        "Current context matters",
        "The Live Leaderboard exists because roster movement, injuries, depth, and refreshed source files can change how a team should be viewed today.",
    ),
    (
        "Contender profile is not the same as ranking",
        "Super Square is not just a top-team list; it asks whether a team matches the historical shape of championship contenders.",
    ),
    (
        "Plain language is part of the model",
        "The site works best when the numbers are paired with labels, cards, tables, and explanations that a normal football fan can read quickly.",
    ),
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
    padding-bottom: 2.8rem !important;
}}

.history-page {{
    position: relative;
    z-index: 1;
    max-width: 1500px;
    margin: -3.25rem auto 0 auto;
    padding: 0 1.5rem 3rem 1.5rem;
}}

.history-hero {{
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, {SECONDARY} 0%, #172131 58%, #273444 100%);
    color: #ffffff;
    border-radius: 18px;
    padding: 2.6rem 3rem;
    border: 1px solid rgba(255,255,255,0.13);
    box-shadow: 0 22px 45px rgba(0, 0, 0, 0.16);
    margin-top: -0.75rem;
    margin-bottom: 1.6rem;
}}

.history-hero::before {{
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    height: 6px;
    width: 100%;
    background: linear-gradient(90deg, {PRIMARY}, rgba(255,255,255,0.35), {SECONDARY});
}}

.history-kicker {{
    color: {PRIMARY};
    font-size: 0.82rem;
    font-weight: 900;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.7rem;
}}

.history-title {{
    font-size: clamp(2.05rem, 5vw, 2.55rem);
    line-height: 1.05;
    font-weight: 950;
    margin-bottom: 0.9rem;
}}

.history-text {{
    max-width: 960px;
    color: rgba(255,255,255,0.92);
    font-size: 1.02rem;
    line-height: 1.75;
}}

.section-title {{
    font-size: 1.45rem;
    font-weight: 950;
    margin: 2.2rem 0 0.45rem 0;
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

.history-stage {{
    display: grid;
    grid-template-columns: 36px minmax(0, 1fr);
    gap: 0.9rem;
    margin-bottom: 1.05rem;
}}

.stage-rail {{
    position: relative;
    display: flex;
    justify-content: center;
}}

.stage-dot {{
    position: relative;
    z-index: 2;
    width: 14px;
    height: 14px;
    margin-top: 1.35rem;
    border-radius: 999px;
    background: {PRIMARY};
    box-shadow: 0 0 0 5px rgba(241, 90, 36, 0.12);
}}

.stage-line {{
    position: absolute;
    top: 1.35rem;
    bottom: -1.2rem;
    width: 2px;
    background: linear-gradient(180deg, {PRIMARY}, rgba(0,115,183,0.35));
}}

.stage-card {{
    position: relative;
    background: linear-gradient(180deg, #ffffff 0%, #fbfcfe 100%);
    border: 1px solid rgba(15, 23, 42, 0.11);
    border-radius: 12px;
    padding: 1.15rem 1.25rem 1.1rem 1.25rem;
    box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
}}

.stage-kicker {{
    color: {SECONDARY};
    font-size: 0.72rem;
    font-weight: 900;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.25rem;
}}

.stage-title {{
    color: #07111f;
    font-size: 1.2rem;
    font-weight: 950;
    margin-bottom: 0.45rem;
}}

.stage-purpose {{
    color: #233044;
    line-height: 1.65;
    font-size: 0.96rem;
    margin-bottom: 1rem;
}}

.stage-grid {{
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.85rem;
}}

.stage-panel {{
    background: #F8FAFC;
    border: 1px solid rgba(0, 34, 68, 0.10);
    border-radius: 10px;
    padding: 0.8rem 0.9rem;
}}

.panel-label {{
    color: {SECONDARY};
    font-size: 0.72rem;
    font-weight: 900;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}}

.stage-panel ul {{
    margin: 0;
    padding-left: 1.05rem;
    color: #334155;
    line-height: 1.55;
    font-size: 0.9rem;
}}

.finding-grid {{
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 0.85rem;
    margin-top: 1rem;
}}

.finding-card {{
    background: #ffffff;
    border: 1px solid rgba(15, 23, 42, 0.10);
    border-radius: 12px;
    padding: 0.95rem 1rem;
    box-shadow: 0 8px 18px rgba(15, 23, 42, 0.035);
}}

.finding-card:hover {{
    border-color: rgba(0, 115, 183, 0.20);
    box-shadow: 0 12px 24px rgba(15, 23, 42, 0.06);
}}

.finding-title-row {{
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.55rem;
}}

.finding-dot {{
    width: 7px;
    height: 7px;
    flex: 0 0 auto;
    border-radius: 999px;
    background: {SECONDARY};
    box-shadow: 0 0 0 3px rgba(0, 115, 183, 0.08);
}}

.finding-title {{
    color: #07111f;
    font-size: 0.98rem;
    font-weight: 950;
    line-height: 1.25;
}}

.finding-body {{
    color: #475569;
    font-size: 0.9rem;
    line-height: 1.58;
}}

@media screen and (max-width: 1050px) {{
    .finding-grid {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }}
}}

@media screen and (max-width: 800px) {{
    .history-page {{
        margin-top: -0.75rem;
        padding-left: 0.25rem;
        padding-right: 0.25rem;
    }}

    .history-hero {{
        padding: 1.35rem;
        border-radius: 16px;
    }}

    .stage-grid,
    .finding-grid {{
        grid-template-columns: 1fr;
    }}
}}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="bg-canvas" aria-hidden="true"></div>', unsafe_allow_html=True)
render_top_nav("Model History", PRIMARY, SECONDARY)
st.markdown('<div class="history-page">', unsafe_allow_html=True)

st.markdown(
    """
<div class="history-hero">
    <div class="history-kicker">Football Analytics Platform</div>
    <div class="history-title">Model History</div>
    <div class="history-text">
        The Gini Metric started as a team-quality score, then grew into a full NFL analytics
        platform. The site now has four connected tools: the Gini Dashboard for team evaluation,
        Live Leaderboard for current-moment ranking, Super Square for championship-profile study,
        and Predictive Model for regular-season and playoff outlooks.
    </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="section-title">Evolution of the Model</div>
<div class="section-intro">
    Each stage answered a different football question. Together, the tools move from evaluating
    what happened, to ranking the current league, to identifying contender profiles, to forecasting
    what could happen next.
</div>
""",
    unsafe_allow_html=True,
)

for stage in model_stages:
    render_stage(stage)

st.markdown(
    """
<div class="section-title">What the Models Taught Me</div>
<div class="section-intro">
    These are the user-facing lessons that shaped the current four-page analytics experience.
</div>
<div class="finding-grid">
""",
    unsafe_allow_html=True,
)

for title, body in key_findings:
    render_finding(title, body)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
