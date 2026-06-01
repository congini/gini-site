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


gini_dashboard_entries = [
    {"name": "Gini Metric", "meaning": "The main team-quality score on the Gini Dashboard. It blends offense, defense, point differential, consistency, turnovers, penalties, and schedule context into one season-relative rating.", "reading": "Around 100 is roughly average for that season. Higher means the team graded better in the selected model view."},
    {"name": "Custom EStat", "meaning": "The Gini score recalculated from the current Customize Model Weights sliders.", "reading": "Use this when you want the rankings to reflect your football philosophy, such as heavier offense, defense, or schedule emphasis."},
    {"name": "Default EStat", "meaning": "The stored baseline EStat that comes from the prepared season data before user slider changes.", "reading": "Use it as the consistent baseline score. Custom EStat is the interactive version."},
    {"name": "Offense EStat", "meaning": "A season-relative offensive score built from opponent-adjusted offensive performance.", "reading": "Higher is better. It shows whether the offense created more value than the league average in that season."},
    {"name": "Defense EStat", "meaning": "A season-relative defensive score built so stronger defense points upward on the dashboard scale.", "reading": "Higher is better. It rewards defenses that limit opponent value after context is applied."},
    {"name": "Game EStat", "meaning": "A game-level performance score that combines efficiency, consistency, turnovers, penalties, and game control.", "reading": "The dashboard displays it on a cleaner weekly scale. Higher numbers mean stronger single-game performance."},
    {"name": "Matchup-Adjusted Game EStat", "meaning": "A Game EStat view that gives extra context for opponent strength.", "reading": "Use it to see whether a weekly performance should be read differently because of the opponent."},
    {"name": "Schedule Strength", "meaning": "A measure of how difficult a team's opponents were.", "reading": "Higher means a tougher schedule. It is context, not a direct team-quality grade."},
    {"name": "Strength of Schedule Rank", "meaning": "The league rank of schedule difficulty in the selected season.", "reading": "Rank 1 means the hardest schedule. Larger numbers mean easier relative schedule paths."},
    {"name": "Success Margin", "meaning": "A consistency measure comparing a team's offensive success rate to what it allowed on defense.", "reading": "Positive values usually mean the team was more reliable play to play than its opponent."},
    {"name": "EPA", "meaning": "Expected Points Added estimates whether a play improved or hurt expected scoring value.", "reading": "Positive offensive EPA is good. For defense, allowing less EPA is better, and dashboard scores are oriented so stronger defense reads higher."},
    {"name": "Opponent-Adjusted EPA", "meaning": "EPA interpreted through the quality of the opponent faced.", "reading": "It helps separate strong production against a strong opponent from similar production against a weaker one."},
    {"name": "Turnover Margin", "meaning": "The possession swing created by takeaways versus giveaways.", "reading": "Positive is generally good because the team gained more possession value than it lost."},
    {"name": "Penalty Yard Margin", "meaning": "The field-position swing created by penalty yards.", "reading": "Positive usually means the opponent gave away more penalty yardage than the selected team did."},
]


live_leaderboard_entries = [
    {"name": "Live Market Score", "meaning": "The Live Leaderboard's current-moment team rating. It combines current Gini profile, roster score, projected wins, and Super Square-style profile probability.", "reading": "Higher means the team currently looks stronger in the live market-style ranking."},
    {"name": "Current Gini Score", "meaning": "The performance foundation used by the Live Leaderboard for the selected current season.", "reading": "It anchors the live rating in football performance before roster and projection layers are added."},
    {"name": "Performance Score", "meaning": "A normalized score that translates the team's Gini profile into the live leaderboard blend.", "reading": "Higher means the team's performance profile is helping its live rank."},
    {"name": "Roster Score", "meaning": "A live roster-strength layer using available roster, depth, production, availability, and player-context signals.", "reading": "Higher means the roster layer is lifting the team's current outlook."},
    {"name": "Projected Wins", "meaning": "The current win projection used by the Live Leaderboard.", "reading": "Use it as a directional outlook, not a guarantee. It helps translate team quality into expected season result."},
    {"name": "Projected Wins Range", "meaning": "The lower-to-upper band around projected wins.", "reading": "A wider band means more uncertainty. A narrow band means the model sees a more stable outlook."},
    {"name": "Quadrant Probability", "meaning": "The probability that a team fits one of the four predictive profile buckets.", "reading": "Higher Q3 or Q4 probability points toward a stronger playoff or contender profile."},
    {"name": "Current Best", "meaning": "The best-ranked or strongest live profile at the current refresh.", "reading": "Use it to see who leads the board right now, not who led a previous saved week."},
    {"name": "Current Worst", "meaning": "The weakest live profile at the current refresh.", "reading": "Use it to see who is currently at the bottom of the market-style board."},
    {"name": "Riser of the Week", "meaning": "A team moving up relative to the most recent saved or rolling live snapshot.", "reading": "It highlights movement, not just absolute strength."},
    {"name": "Faller of the Week", "meaning": "A team moving down relative to the most recent saved or rolling live snapshot.", "reading": "It flags teams whose current live profile has softened."},
    {"name": "Snapshot", "meaning": "A saved copy of the leaderboard at a point in time.", "reading": "Snapshots let the page compare current rank and score to a prior board."},
    {"name": "Weekly History", "meaning": "The once-per-football-period archive stored in live_leaderboard_weekly_history.csv.", "reading": "It should grow only when a weekly snapshot is due, not on every five-minute refresh."},
    {"name": "Live Source Refresh", "meaning": "The process of rereading local source files and optional nflverse cache files.", "reading": "Refresh updates the page view from local data. It does not automatically mean a new weekly history row was saved."},
    {"name": "nflreadpy / nflverse Source Files", "meaning": "Free nflverse data files that can be cached locally through nflreadpy.", "reading": "They support refreshed schedules, weekly rosters, injuries, transactions, snap counts, and player production when available."},
    {"name": "Injuries, Weekly Rosters, Snap Counts, Schedules, Draft Picks, Contracts, Depth Charts", "meaning": "Roster and context files that can strengthen the live roster layer over time.", "reading": "More complete source files make the live roster score more current and less dependent on proxy signals."},
]


super_square_entries = [
    {"name": "Super Square", "meaning": "A championship-contender profile tool built from historical regular-season team patterns.", "reading": "Inside the square means the team matched the historical profile gates. It is a signal, not a Super Bowl guarantee."},
    {"name": "Championship Profile", "meaning": "The combined set of traits that past champions tended to show before the playoffs.", "reading": "Teams that match more of the profile look more like historical contenders."},
    {"name": "Control Profile", "meaning": "The broad team-control side of Super Square: scoreboard control, Gini/EStat checkpoints, success margin, and EPA strength.", "reading": "A higher control score means the team looked more stable and complete."},
    {"name": "Unit Pressure", "meaning": "The unit-strength side of Super Square, focused on whether a team had a strong enough offense or defense to pressure opponents.", "reading": "A high unit pressure score means at least one side of the ball looked championship-caliber."},
    {"name": "Q1, Q2, Q3, Q4", "meaning": "Four profile buckets ranging from non-contender shapes to Super Bowl contender shapes.", "reading": "Q4 is strongest. Q1 is weakest. Q2 and Q3 sit in the middle tiers."},
    {"name": "Q4 Super Bowl Contender", "meaning": "The strongest quadrant profile on the Super Square page.", "reading": "A Q4 team has the closest regular-season shape to high-end contender patterns."},
    {"name": "Super Square Requirements", "meaning": "The specific gates a team must clear to be considered inside the Super Square.", "reading": "Requirements are learned from historical champion floors and then applied to every team-season."},
    {"name": "Cusp Teams", "meaning": "Teams that come close to the Super Square profile but do not fully clear every gate.", "reading": "Cusp teams are worth watching, but they are not officially inside the square."},
    {"name": "Gini Rank", "meaning": "A team's rank by the Super Square Gini/EStat profile within its season.", "reading": "Lower rank numbers are stronger. Rank 1 is best."},
    {"name": "Point Differential Rank", "meaning": "A team's season rank by scoring margin.", "reading": "Strong point differential usually supports the control-profile side."},
    {"name": "Success Rank", "meaning": "A team's rank by consistency and success-margin signals.", "reading": "A strong rank means the team won more plays and situations, not only final scores."},
    {"name": "EPA Differential", "meaning": "The gap between value created by the offense and value allowed by the defense.", "reading": "Positive or high differential points to stronger underlying team quality."},
    {"name": "Best Unit", "meaning": "The stronger side of the ball for a team, usually offense or defense.", "reading": "Super Square uses this to judge whether the team had a pressure point that could carry into postseason football."},
    {"name": "Historical Champion Pattern", "meaning": "The common statistical shape found across past Super Bowl champions.", "reading": "It helps compare current teams to proven championship paths without treating history as destiny."},
]


predictive_model_entries = [
    {"name": "Regular Season Predictor", "meaning": "The tool that uses completed prior-year team profiles to forecast regular-season wins.", "reading": "It is designed for season outlooks, not single-game picks."},
    {"name": "Prediction Target Year", "meaning": "The season being projected.", "reading": "If the target is 2026, the model is estimating the 2026 regular-season result."},
    {"name": "Feature Year", "meaning": "The completed season used as the input profile for the forecast.", "reading": "For a 2026 projection, the feature year is usually 2025."},
    {"name": "Projected Wins", "meaning": "The model's central win estimate.", "reading": "Read it as the middle of the forecast, not an exact prediction."},
    {"name": "Likely Range", "meaning": "The win band around the projection using model error.", "reading": "It shows a reasonable high-low expectation based on backtest misses."},
    {"name": "Model Result", "meaning": "The plain-language comparison between projection and actual or current pace.", "reading": "It tells whether the team is tracking above, below, or near the forecast."},
    {"name": "Actual Wins", "meaning": "The real wins recorded for the target season once games are available.", "reading": "Use it to judge how the projection is aging as the season progresses or after it ends."},
    {"name": "Win Pace", "meaning": "The current record scaled to a full regular season.", "reading": "Useful during an unfinished season because actual wins are not final yet."},
    {"name": "Expected Wins / Pythagorean Wins", "meaning": "A scoreboard-based estimate of how many games a team probably should have won from its points scored and allowed.", "reading": "It helps spot teams whose record may be stronger or weaker than their point profile."},
    {"name": "Wins Above Expected", "meaning": "Actual wins minus expected or Pythagorean wins.", "reading": "Large positive values can suggest regression risk. Large negative values can suggest improvement potential."},
    {"name": "Regression Risk", "meaning": "A label for teams whose prior record may have been better than their underlying profile.", "reading": "It warns that the next season could be tougher than the record alone implies."},
    {"name": "Improvement Candidate", "meaning": "A label for teams whose underlying profile looked better than their record.", "reading": "It flags teams that may rebound if close-game or scoring luck normalizes."},
    {"name": "Rolling Backtest", "meaning": "A time-aware test where older seasons train the model and later seasons test it.", "reading": "It avoids testing the model on information it would not have known yet."},
    {"name": "MAE", "meaning": "Mean Absolute Error, or the average miss in wins.", "reading": "Lower MAE means the model missed by fewer wins on average."},
    {"name": "Playoffs Predictor", "meaning": "The postseason tool that evaluates completed regular-season playoff profiles.", "reading": "It is separate from the regular-season predictor and focuses on playoff wins/outlook."},
    {"name": "Postseason Lock", "meaning": "The rule that keeps the playoff predictor closed until the regular season is complete.", "reading": "It prevents the tool from mixing current-year playoff forecasting with incomplete regular-season data."},
    {"name": "Projected Playoff Wins", "meaning": "The playoff model's central estimate of postseason wins.", "reading": "It is capped to the realistic playoff range and should be read as an outlook, not a bracket guarantee."},
    {"name": "Chance of a Playoff Win", "meaning": "The estimated chance a playoff team wins at least one postseason game.", "reading": "Higher values mean the model sees a stronger chance of advancing beyond a playoff appearance."},
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
    font-size: clamp(2.05rem, 5vw, 2.55rem);
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
    .terminology-page {{
        margin-top: -0.75rem;
        padding-left: 0.25rem;
        padding-right: 0.25rem;
    }}

    .terminology-hero {{
        padding: 1.35rem;
        border-radius: 16px;
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
    <div class="terminology-kicker">Platform Documentation</div>
    <div class="terminology-title">Terminology</div>
    <div class="terminology-text">
        A page-by-page guide to the stats, labels, and football concepts used across the site.
        Each tab explains the language for one tool so users can read the numbers without needing
        to know the underlying spreadsheets first.
    </div>
</div>
""",
    unsafe_allow_html=True,
)

tab_gini, tab_live, tab_square, tab_predictive = st.tabs(
    [
        "Gini Dashboard",
        "Live Leaderboard",
        "Super Square",
        "Predictive Model",
    ]
)

with tab_gini:
    render_section(
        "Gini Dashboard Terms",
        "Terms used by the main team evaluation page: rankings, EStat scores, weekly game view, schedule context, and the model-weight controls.",
        gini_dashboard_entries,
    )

with tab_live:
    render_section(
        "Live Leaderboard Terms",
        "Terms used by the current-moment page: Live Market Score, roster strength, projected wins, movement, snapshots, and refreshed local source files.",
        live_leaderboard_entries,
    )

with tab_square:
    render_section(
        "Super Square Terms",
        "Terms used by the championship-profile page: contender quadrants, control profile, unit pressure, historical champion patterns, and cusp teams.",
        super_square_entries,
    )

with tab_predictive:
    render_section(
        "Predictive Model Terms",
        "Terms used by the forecasting page: regular-season projections, feature years, likely ranges, backtests, regression labels, and playoff predictor locks.",
        predictive_model_entries,
    )

st.markdown("</div>", unsafe_allow_html=True)
