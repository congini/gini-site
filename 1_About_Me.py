import streamlit as st
import base64
import sys
from pathlib import Path

st.set_page_config(
    page_title="About Me",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------
# ABOUT ME PAGE
# -----------------------------

# Uses your existing Broncos/team colors if they already exist in the app.
# If your variables have different names, only change these two lines.
primary = locals().get("primary_color", locals().get("team_primary", "#F15A24"))
secondary = locals().get("secondary_color", locals().get("team_secondary", "#0073B7"))

# Keep headshot.jpeg inside the main project assets folder:
# nfl_estat_dashboard_project/
#     assets/
#         headshot.jpeg
#     pages/
#         About_Me.py
APP_DIR = Path(__file__).resolve().parent
ASSETS_DIR = APP_DIR / "assets"
headshot_path = ASSETS_DIR / "headshot.jpeg"

if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from site_nav import render_top_nav

def image_to_base64(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        return ""

headshot_base64 = image_to_base64(headshot_path)

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
            padding-top: 0.35rem !important;
            padding-bottom: 2.5rem !important;
        }}

        .about-page {{
            max-width: 1500px;
            margin: 0 auto;
            padding: 0 clamp(0.75rem, 2vw, 1.5rem) 3rem clamp(0.75rem, 2vw, 1.5rem);
        }}

        .fade-slide {{
            animation: fadeSlideUp 0.75s ease both;
        }}

        .fade-slide-delay {{
            animation: fadeSlideUp 0.95s ease both;
        }}

        @keyframes fadeSlideUp {{
            from {{
                opacity: 0;
                transform: translateY(18px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}

        .hero-grid {{
            display: grid;
            grid-template-columns: minmax(0, 2.2fr) minmax(320px, 0.85fr);
            gap: 2rem;
            align-items: stretch;
            margin-bottom: 3.25rem;
        }}

        .hero-card {{
            position: relative;
            background: linear-gradient(135deg, {secondary} 0%, #172131 58%, #273444 100%);
            color: white;
            border-radius: 22px;
            padding: 2.75rem 3rem;
            box-shadow: 0 22px 45px rgba(0, 0, 0, 0.18);
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.13);
            min-height: 340px;
            height: auto;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }}

        .hero-card::before {{
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            height: 6px;
            width: 100%;
            background: linear-gradient(90deg, {primary}, rgba(255,255,255,0.35), {secondary});
        }}

        .hero-card::after {{
            content: "";
            position: absolute;
            right: -70px;
            bottom: -70px;
            width: 210px;
            height: 210px;
            border-radius: 50%;
            border: 28px solid rgba(255,255,255,0.045);
        }}

        .hero-kicker {{
            color: {primary};
            font-size: 0.85rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.75rem;
        }}

        .hero-title {{
            font-size: clamp(2rem, 4.2vw, 2.65rem);
            line-height: 1.05;
            font-weight: 900;
            margin-bottom: 1rem;
            letter-spacing: -0.03em;
        }}

        .hero-text {{
            font-size: 1.08rem;
            line-height: 1.75;
            max-width: 900px;
            color: rgba(255,255,255,0.92);
            margin-bottom: 1.4rem;
        }}

        .pill-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.75rem;
            margin-top: 0.4rem;
        }}

        .about-pill {{
            padding: 0.48rem 0.9rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.10);
            border: 1px solid rgba(255,255,255,0.20);
            color: white;
            font-size: 0.82rem;
            font-weight: 700;
            backdrop-filter: blur(8px);
        }}

        .hero-statement {{
            max-width: 760px;
            margin-top: 0.25rem;
            padding-left: 1rem;
            border-left: 4px solid {primary};
            color: rgba(255,255,255,0.88);
            font-size: 1rem;
            line-height: 1.6;
            font-weight: 750;
        }}

        .headshot-card {{
            position: relative;
            border-radius: 22px;
            background: #ffffff;
            border: 1px solid rgba(0,0,0,0.08);
            box-shadow: 0 18px 42px rgba(0, 0, 0, 0.13);
            padding: 0.75rem;
            overflow: hidden;
            height: clamp(280px, 42vw, 340px);
            box-sizing: border-box;
        }}

        .headshot-card::before {{
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            height: 5px;
            width: 100%;
            background: linear-gradient(90deg, {primary}, {secondary});
            z-index: 3;
        }}

        .headshot-img {{
            position: absolute;
            inset: 0.75rem;
            width: calc(100% - 1.5rem) !important;
            height: calc(100% - 1.5rem) !important;
            object-fit: cover !important;
            object-position: center top;
            border-radius: 15px;
            display: block;
            filter: saturate(1.02) contrast(1.02);
            z-index: 1;
        }}

        .section-title {{
            font-size: 1.55rem;
            font-weight: 900;
            margin: 2.1rem 0 1.1rem 0;
            color: #07111f;
            letter-spacing: -0.025em;
        }}

        .section-title::after {{
            content: "";
            display: block;
            width: 58px;
            height: 4px;
            border-radius: 999px;
            margin-top: 0.45rem;
            background: linear-gradient(90deg, {primary}, {secondary});
        }}

        .card-grid {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 1.05rem;
            margin-bottom: 1.4rem;
        }}

        .about-card {{
            background: #ffffff;
            border-radius: 18px;
            padding: 1.45rem 1.55rem;
            border: 1px solid rgba(0, 34, 68, 0.16);
            box-shadow: 0 10px 25px rgba(15, 23, 42, 0.055);
            transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease;
            position: relative;
            overflow: hidden;
        }}

        .about-card::before {{
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 4px;
            background: linear-gradient(90deg, {primary}, rgba(251,79,20,0.18));
        }}

        .about-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 18px 36px rgba(15, 23, 42, 0.12);
            border-color: {secondary};
        }}

        .about-card h3 {{
            font-size: 1rem;
            font-weight: 900;
            margin-bottom: 0.75rem;
            color: #07111f;
        }}

        .about-card p {{
            font-size: 0.95rem;
            line-height: 1.65;
            color: #233044;
            margin: 0;
        }}

        .question-grid {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.9rem 1rem;
            margin-bottom: 1.6rem;
        }}

        .question-box {{
            background: #f8fafc;
            border: 1px solid rgba(0, 34, 68, 0.08);
            border-left: 5px solid {primary};
            border-radius: 14px;
            padding: 1.05rem 1.2rem;
            font-size: 0.96rem;
            color: #111827;
            box-shadow: 0 5px 14px rgba(15, 23, 42, 0.035);
            transition: transform 0.22s ease, box-shadow 0.22s ease, background 0.22s ease;
        }}

        .question-box:hover {{
            transform: translateY(-3px);
            background: #ffffff;
            box-shadow: 0 12px 24px rgba(15, 23, 42, 0.10);
        }}

        .estat-feature {{
            background: linear-gradient(135deg, rgba(251, 79, 20, 0.08), rgba(255,255,255,0.96));
            border: 1px solid rgba(251, 79, 20, 0.34);
            border-radius: 18px;
            padding: 1.5rem 1.6rem;
            margin: 1.5rem 0 2.4rem 0;
            box-shadow: 0 10px 28px rgba(251, 79, 20, 0.07);
            position: relative;
            overflow: hidden;
        }}

        .estat-feature::before {{
            content: "";
            position: absolute;
            left: 0;
            top: 0;
            width: 6px;
            height: 100%;
            background: {primary};
        }}

        .estat-feature h3 {{
            font-size: 1.08rem;
            font-weight: 900;
            margin-bottom: 0.75rem;
            color: #07111f;
        }}

        .estat-feature p {{
            font-size: 0.97rem;
            line-height: 1.7;
            color: #233044;
            margin: 0;
        }}

        .context-grid {{
            display: grid;
            grid-template-columns: minmax(0, 1.25fr) minmax(320px, 0.85fr);
            gap: 1.4rem;
            align-items: stretch;
            margin-bottom: 2.2rem;
        }}

        .context-text {{
            font-size: 0.98rem;
            line-height: 1.78;
            color: #233044;
            background: #ffffff;
            border-radius: 18px;
            padding: 1.55rem 1.65rem;
            border: 1px solid rgba(0, 34, 68, 0.16);
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.075);
            position: relative;
            overflow: hidden;
            transition: transform 0.22s ease, box-shadow 0.22s ease;
        }}

        .context-text p {{
            margin-top: 0;
            margin-bottom: 1rem;
        }}

        .context-text p:last-child {{
            margin-bottom: 0;
        }}

        .core-card {{
            background: #ffffff;
            border-radius: 18px;
            padding: 1.55rem 1.65rem;
            border: 1px solid rgba(0, 34, 68, 0.16);
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.075);
            position: relative;
            overflow: hidden;
            transition: transform 0.22s ease, box-shadow 0.22s ease;
        }}

        .context-text:hover,
        .core-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 18px 38px rgba(15, 23, 42, 0.12);
        }}

        .context-text::before,
        .core-card::before {{
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            height: 5px;
            width: 100%;
            background: linear-gradient(90deg, {primary}, {secondary});
        }}

        .core-card h3 {{
            font-size: 1.05rem;
            font-weight: 900;
            margin-bottom: 0.75rem;
            color: #07111f;
        }}

        .core-card p {{
            font-size: 0.96rem;
            line-height: 1.72;
            color: #233044;
            margin: 0;
        }}

        .headshot-overlay {{
            position: absolute;
            left: 0.75rem;
            right: 0.75rem;
            bottom: 0.75rem;
            z-index: 2;
            padding: 1rem 1.1rem;
            border-radius: 0 0 15px 15px;
            background: linear-gradient(180deg, rgba(7, 17, 31, 0.05), rgba(7, 17, 31, 0.86));
            color: #ffffff;
        }}

        .headshot-name {{
            font-size: 1rem;
            font-weight: 900;
            line-height: 1.2;
        }}

        .headshot-role {{
            font-size: 0.78rem;
            font-weight: 700;
            color: rgba(255,255,255,0.82);
            margin-top: 0.25rem;
        }}

        .quick-facts-grid {{
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.9rem;
            margin: 1.25rem 0 2rem 0;
            position: relative;
            z-index: 3;
        }}

        .quick-fact-card {{
            background: #ffffff;
            border: 1px solid rgba(0, 34, 68, 0.14);
            border-left: 5px solid {primary};
            border-radius: 16px;
            padding: 1rem 1.1rem;
            box-shadow: 0 12px 26px rgba(15, 23, 42, 0.08);
        }}

        .quick-fact-label {{
            font-size: 0.72rem;
            font-weight: 900;
            color: {secondary};
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.35rem;
        }}

        .quick-fact-value {{
            font-size: 0.95rem;
            font-weight: 900;
            color: #07111f;
        }}

        .about-footer {{
            border-top: 1px solid rgba(0,0,0,0.12);
            padding-top: 1.25rem;
            margin-top: 2rem;
            color: #64748b;
            font-size: 0.82rem;
        }}

        @media screen and (max-width: 950px) {{
            .hero-grid {{
                grid-template-columns: 1fr;
            }}

            .card-grid {{
                grid-template-columns: 1fr;
            }}

            .question-grid {{
                grid-template-columns: 1fr;
            }}

            .context-grid {{
                grid-template-columns: 1fr;
            }}

            .quick-facts-grid {{
                grid-template-columns: repeat(2, minmax(0, 1fr));
                margin-top: 1.1rem;
            }}

            .hero-card {{
                padding: 2.1rem;
            }}
        }}

        @media screen and (max-width: 640px) {{
            .about-page {{
                margin-top: -0.8rem !important;
            }}

            .hero-card {{
                min-height: 0;
                padding: 1.45rem;
                border-radius: 16px;
            }}

            .hero-text,
            .hero-statement {{
                font-size: 0.94rem;
                line-height: 1.58;
            }}

            .headshot-card {{
                height: 300px;
                border-radius: 16px;
            }}

            .quick-facts-grid {{
                grid-template-columns: 1fr;
            }}

            .about-card,
            .context-text,
            .core-card,
            .estat-feature {{
                padding: 1.1rem 1.15rem;
                border-radius: 14px;
            }}
        }}
    </style>

    <div class="bg-canvas" aria-hidden="true"></div>
    """,
    unsafe_allow_html=True
)

render_top_nav("About Me", primary, secondary)

st.markdown(
    """
    <style>
        .site-top-nav {
            margin-top: 0 !important;
            margin-bottom: 0.35rem !important;
        }

        .about-page {
            margin-top: -3.25rem !important;
        }

        .hero-grid {
            margin-top: -1rem !important;
            margin-bottom: 1.05rem !important;
        }

        .hero-card,
        .headshot-card {
            margin-top: -1rem !important;
        }

        .quick-facts-grid {
            margin: 0.85rem 0 0.65rem 0 !important;
        }

        .section-title {
            margin-top: 1rem !important;
        }

        @media screen and (max-width: 640px) {
            .about-page {
                margin-top: -0.75rem !important;
            }

            .hero-grid,
            .hero-card,
            .headshot-card {
                margin-top: 0 !important;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="about-page">', unsafe_allow_html=True)

# -----------------------------
# HERO SECTION
# -----------------------------

hero_left, hero_right = st.columns([1.55, 0.45])

with hero_left:
    st.markdown(
        f"""
<div class="hero-card fade-slide">
    <div class="hero-kicker">Football Analytics Portfolio</div>
    <div class="hero-title">About Me</div>
    <div class="hero-text">
        I’m Conor Mangini, and I built the Gini Metric to evaluate NFL teams with more context than wins,
losses, and final scores alone. The goal is to make football analytics easier to explore, easier to
understand, and more useful for comparing teams across seasons.
    </div>
        <div class="hero-statement">
        A context-driven football metric for comparing teams across seasons, styles, and schedules.
    </div>
</div>
""",
        unsafe_allow_html=True
    )

with hero_right:
    if headshot_base64:
        st.markdown(
            f"""
<div class="headshot-card fade-slide-delay">
    <img class="headshot-img" src="data:image/jpeg;base64,{headshot_base64}" alt="Conor Mangini headshot">
    <div class="headshot-overlay">
        <div class="headshot-name">Conor Mangini</div>
        <div class="headshot-role">Sports Analytics | Gini Metric Creator</div>
    </div>
</div>
""",
            unsafe_allow_html=True
        )
    else:
        st.warning("Headshot not found. Make sure it is saved as assets/headshot.jpeg.")

st.markdown(
    """
<div class="quick-facts-grid fade-slide">
    <div class="quick-fact-card">
        <div class="quick-fact-label">Focus</div>
        <div class="quick-fact-value">NFL Team Evaluation</div>
    </div>
    <div class="quick-fact-card">
        <div class="quick-fact-label">Model</div>
        <div class="quick-fact-value">Gini Metric</div>
    </div>
    <div class="quick-fact-card">
        <div class="quick-fact-label">Built With</div>
        <div class="quick-fact-value">Python + Streamlit</div>
    </div>
    <div class="quick-fact-card">
        <div class="quick-fact-label">Background</div>
        <div class="quick-fact-value">Business Analytics</div>
    </div>
</div>
""",
    unsafe_allow_html=True
)

# -----------------------------
# BACKGROUND
# -----------------------------

st.markdown('<div class="section-title fade-slide">Background</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="card-grid fade-slide">
        <div class="about-card">
            <h3>Who I Am</h3>
            <p>
                I recently graduated from Quinnipiac University through the 3+1 Business Analytics program.
                My main interests are sports analytics, operations, scouting, and using data to better explain performance.
            </p>
        </div>
        <div class="about-card">
            <h3>Why I Built This</h3>
            <p>
                I wanted a better way to evaluate NFL teams beyond record and box-score results.
                A team’s performance should be judged with context, including opponent strength, efficiency,
                consistency, penalty ratios, and schedule difficulty.
            </p>
        </div>
        <div class="about-card">
            <h3>The Bigger Goal</h3>
            <p>
                This site is meant to serve as a football analytics platform. The long-term goal is to help
                people explore teams, compare seasons, understand trends, and think about football through a sharper analytical lens.
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# QUESTIONS SECTION
# -----------------------------

st.markdown('<div class="section-title fade-slide">What This Site Is Built To Answer</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="question-grid fade-slide">
        <div class="question-box">Which teams were actually the most efficient?</div>
        <div class="question-box">How did a team perform week by week?</div>
        <div class="question-box">Which teams were carried by offense or defense?</div>
        <div class="question-box">Did the final score match the underlying performance?</div>
        <div class="question-box">Which teams played a harder or easier schedule?</div>
        <div class="question-box">How do rankings change when the model weights change?</div>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# ABOUT Gini Metric
# -----------------------------

st.markdown(
    """
    <div class="estat-feature fade-slide">
        <h3>About Gini Metric</h3>
        <p>
            Gini Metric is my custom NFL team evaluation model. It combines offensive efficiency, defensive efficiency,
point differential, success margin, turnovers, penalties, and schedule strength into one score. The goal is
not to replace watching games or traditional football analysis. The goal is to add context, especially when
a team’s record does not fully match how well they actually played.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# WHY CONTEXT MATTERS
# -----------------------------

st.markdown('<div class="section-title fade-slide">Why Context Matters</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="context-grid fade-slide">
        <div class="context-text">
            <p>
                Football performance can be misleading when it is judged only by wins and losses.
                Some teams benefit from easier schedules, turnover luck, or close-game variance.
                Other teams may look worse in the standings but perform better in efficiency-based metrics.
            </p>
            <p>
                This dashboard is built around that idea. It gives users a way to compare teams through multiple
                angles instead of relying on one number. The model is also adjustable, so users can test different
                football philosophies instead of being locked into one fixed view.
            </p>
        </div>
        <div class="core-card">
            <h3>Core Idea</h3>
            <p>
                A team’s performance should be judged by how well it played, who it played against,
                and how consistently it created advantages.
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# FOOTER
# -----------------------------

st.markdown(
    """
    <div class="about-footer">
        Built by Conor Mangini as part of a larger football analytics project focused on team evaluation,
        efficiency, and opponent-adjusted performance.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)
