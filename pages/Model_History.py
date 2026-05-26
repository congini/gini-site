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
        "label": "2022 | Foundation",
        "name": "Original Gini Stat",
        "purpose": "The first version was built to answer a simple football question: which teams were actually playing well beyond their record and final scores?",
        "findings": [
            "Wins and losses can hide how well a team really played.",
            "A team can look strong in the standings while relying on fragile advantages.",
            "A team can look ordinary in the standings while showing stronger underlying performance.",
        ],
        "impact": [
            "The model needed to care about context, not just outcomes.",
            "Team quality had to include both scoreboard results and what happened underneath them.",
            "The Gini project became a way to explain football performance more clearly.",
        ],
    },
    {
        "label": "2023 | Simplification",
        "name": "Simple Gini Stat",
        "purpose": "The simple version made the model easier to read by focusing on a cleaner overall view of team quality.",
        "findings": [
            "Balanced teams usually created more believable profiles than teams carried by only one side of the ball.",
            "Offense and defense needed to be visible separately before being understood together.",
            "A single number is useful only when users can still see what is driving it.",
        ],
        "impact": [
            "The dashboard needed top-line rankings plus component views.",
            "Offense and defense became central public-facing pillars.",
            "The model moved toward a clearer story instead of only a bigger spreadsheet.",
        ],
    },
    {
        "label": "2024 | Expansion",
        "name": "Pass and Run Splits",
        "purpose": "The pass/run expansion explored how teams created value, not just whether they created value.",
        "findings": [
            "Passing and rushing details explain matchup pressure and team identity.",
            "A team can be efficient overall while still having a clear weakness in how it wins.",
            "Sub-scores are useful for scouting the shape of a team, but they should not overwhelm the main evaluation.",
        ],
        "impact": [
            "The public dashboard kept team details and roster context available without making them the whole model.",
            "The model history preserved passing and rushing ideas as a deeper analysis layer.",
            "The current dashboard favors cleaner interpretation over showing every internal split.",
        ],
    },
    {
        "label": "2024 | Context",
        "name": "Advanced Gini Stat",
        "purpose": "The advanced version added more context around schedule difficulty, team details, and results context.",
        "findings": [
            "Schedule difficulty can completely change how a season should be judged.",
            "Opponent context is one of the biggest differences between a surface-level ranking and a useful one.",
            "More inputs can improve the picture, but too much detail makes the model harder to explain publicly.",
        ],
        "impact": [
            "Schedule strength became a core part of the dashboard conversation.",
            "The model needed a public version that was powerful but easier to understand.",
            "The advanced work pushed the current dashboard toward a clearer public presentation.",
        ],
    },
    {
        "label": "2025 | Current",
        "name": "Gini Metric",
        "purpose": "The current dashboard turns the project into an interactive tool for exploring teams, seasons, and football philosophies.",
        "findings": [
            "Users should be able to test different football beliefs without rebuilding the model.",
            "A clean dashboard can make advanced team evaluation easier to trust.",
            "The best public version explains what the numbers mean in clear football language.",
        ],
        "impact": [
            "The site now separates dashboard terms, model history, and team exploration.",
            "Custom weights let users explore how rankings change under different priorities.",
            "The Gini Metric is presented as a football evaluation system, not just a spreadsheet output.",
        ],
    },
]

key_findings = [
    (
        "Record is not the full story",
        "Standings matter, but they can miss efficiency, opponent difficulty, turnovers, game script, and hidden yardage.",
    ),
    (
        "Context changes interpretation",
        "A strong performance against a strong opponent should not be read the same way as the same surface result against weaker competition.",
    ),
    (
        "Balance matters",
        "Teams with credible offense and defense tend to create more stable profiles than teams relying too heavily on one phase.",
    ),
    (
        "Details explain the result",
        "Passing, rushing, penalties, turnovers, and schedule context help explain how a team got to its result.",
    ),
    (
        "Public clarity matters",
        "The dashboard should help users understand the model's football meaning quickly and confidently.",
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
    font-size: 2.55rem;
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
    .history-hero {{
        padding: 2rem;
    }}

    .history-title {{
        font-size: 2.05rem;
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
    <div class="history-kicker">Gini Metric Documentation</div>
    <div class="history-title">Model History</div>
    <div class="history-text">
        The Gini Metric grew out of a long-running question: how can NFL teams be compared
        in a way that respects record, efficiency, opponent quality, team details, and context?
        This page explains what each stage of the project taught and how those lessons shaped
        the current dashboard.
    </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="section-title">Evolution of the Model</div>
<div class="section-intro">
    Each version of the project answered a different football question. The result is a dashboard
    that is easier to explore, easier to explain, and better suited for comparing teams across seasons.
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
    These are the public-facing lessons that shaped the current Gini Metric dashboard.
</div>
<div class="finding-grid">
""",
    unsafe_allow_html=True,
)

for title, body in key_findings:
    render_finding(title, body)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
