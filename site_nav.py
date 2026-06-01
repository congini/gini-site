import html
import streamlit as st
import streamlit.components.v1 as components


NAV_ITEMS = [
    ("About Me", "/"),
    ("Gini Dashboard", "/gini-dashboard"),
    ("Live Leaderboard", "/live-leaderboard"),
    ("Super Square", "/super-square"),
    ("Predictive Model", "/predictive-model"),
    ("Model History", "/model-history"),
    ("Terminology", "/terminology"),
]


def render_top_nav(active_page, primary="#F15A24", secondary="#0073B7"):
    safe_primary = html.escape(str(primary))
    safe_secondary = html.escape(str(secondary))

    links_html = "\n".join(
        f'<a class="site-nav-link{" active" if label == active_page else ""}" href="{href}" target="_self">{label}</a>'
        for label, href in NAV_ITEMS
    )

    st.markdown(
        f"""
<style>
[data-testid="stSidebar"],
[data-testid="collapsedControl"] {{
    display: none !important;
    visibility: hidden !important;
    width: 0 !important;
    min-width: 0 !important;
}}

[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
.stDeployButton {{
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    min-height: 0 !important;
    pointer-events: none !important;
}}

.block-container {{
    padding-top: 0.35rem !important;
    width: 100% !important;
    max-width: min(100%, 1560px) !important;
    padding-left: clamp(0.75rem, 2vw, 2rem) !important;
    padding-right: clamp(0.75rem, 2vw, 2rem) !important;
}}

html,
body,
.stApp,
[data-testid="stAppViewContainer"] {{
    width: 100% !important;
    max-width: 100%;
    overflow-x: hidden;
}}

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section.main,
main {{
    width: 100% !important;
    max-width: 100vw !important;
    overflow-x: hidden !important;
    overflow-x: clip !important;
}}

.block-container,
[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"],
iframe {{
    max-width: 100% !important;
}}

img,
svg,
canvas {{
    max-width: 100%;
}}

.bg-canvas,
.team-bg-canvas,
.predict-bg {{
    position: fixed !important;
    inset: 0 !important;
    width: 100vw !important;
    height: 100vh !important;
    max-width: 100vw !important;
    max-height: 100vh !important;
    overflow: hidden !important;
    pointer-events: none !important;
    contain: layout paint;
}}

div[data-testid="stPlotlyChart"] {{
    max-width: 100%;
    overflow: hidden;
}}

.stTabs [data-baseweb="tab-list"] {{
    flex-wrap: wrap;
    row-gap: 0.35rem;
}}

.stTabs [data-baseweb="tab"] {{
    white-space: normal;
}}

:root {{
    --gini-primary: {safe_primary};
    --gini-secondary: {safe_secondary};
    --gini-ink: #07111f;
    --gini-text: #122033;
    --gini-muted: #64748b;
    --gini-card: rgba(255,255,255,0.84);
    --gini-card-strong: rgba(255,255,255,0.94);
    --gini-border: rgba(15,23,42,0.10);
    --gini-border-strong: rgba(15,23,42,0.16);
    --gini-soft-bg: rgba(248,250,252,0.72);
    --gini-shadow: 0 18px 42px rgba(15,23,42,0.095);
    --gini-shadow-soft: 0 10px 26px rgba(15,23,42,0.07);
    --gini-radius: 18px;
}}

@keyframes giniFadeUp {{
    from {{
        opacity: 0;
        transform: translateY(14px);
    }}
    to {{
        opacity: 1;
        transform: translateY(0);
    }}
}}

@keyframes giniSheen {{
    0% {{ background-position: 0% 50%; }}
    50% {{ background-position: 100% 50%; }}
    100% {{ background-position: 0% 50%; }}
}}

@keyframes giniPulse {{
    0%, 100% {{ box-shadow: 0 0 0 0 rgba(241,90,36,0.20); }}
    50% {{ box-shadow: 0 0 0 7px rgba(241,90,36,0.00); }}
}}

@keyframes giniFloat {{
    0%, 100% {{ transform: translateY(0); }}
    50% {{ transform: translateY(-4px); }}
}}

.stApp,
[data-testid="stAppViewContainer"] {{
    text-rendering: optimizeLegibility;
}}

.block-container > div:first-child {{
    animation: giniFadeUp 0.42s ease-out both;
}}

:where(
    .dashboard-hero,
    .leader-hero,
    .super-hero,
    .predict-hero,
    .history-hero,
    .terminology-hero,
    .hero-card
) {{
    position: relative;
    isolation: isolate;
    overflow: hidden;
    border-color: rgba(255,255,255,0.20) !important;
    box-shadow: 0 22px 52px rgba(15,23,42,0.14) !important;
    animation: giniFadeUp 0.52s ease-out both;
}}

:where(
    .dashboard-hero,
    .leader-hero,
    .super-hero,
    .predict-hero,
    .history-hero,
    .terminology-hero,
    .hero-card
)::after {{
    content: "";
    position: absolute;
    inset: 1px;
    border-radius: inherit;
    pointer-events: none;
    z-index: 0;
    background:
        linear-gradient(115deg, transparent 0%, rgba(255,255,255,0.20) 38%, transparent 56%),
        linear-gradient(135deg, rgba(255,255,255,0.10), transparent 42%);
    background-size: 220% 100%, 100% 100%;
    animation: giniSheen 9s ease-in-out infinite;
    opacity: 0.62;
}}

:where(
    .dashboard-hero,
    .leader-hero,
    .super-hero,
    .predict-hero,
    .history-hero,
    .terminology-hero,
    .hero-card
) > * {{
    position: relative;
    z-index: 1;
}}

:where(
    .metric-card,
    .market-card,
    .predict-card,
    .stage-card,
    .finding-card,
    .term-card,
    .about-card,
    .context-text,
    .core-card,
    .estat-feature,
    .quick-fact-card,
    .question-box,
    .super-proof-card,
    .super-info-box,
    .source-status-card,
    .team-detail-card,
    .leader-row,
    .detail-metric
) {{
    position: relative;
    max-width: 100%;
    border-color: var(--gini-border) !important;
    box-shadow: var(--gini-shadow-soft) !important;
    transition:
        transform 0.18s ease,
        box-shadow 0.18s ease,
        border-color 0.18s ease,
        background 0.18s ease;
}}

:where(
    .metric-card,
    .market-card,
    .predict-card,
    .stage-card,
    .finding-card,
    .term-card,
    .about-card,
    .context-text,
    .core-card,
    .estat-feature,
    .quick-fact-card,
    .question-box,
    .super-proof-card,
    .super-info-box,
    .source-status-card,
    .team-detail-card,
    .leader-row
):hover {{
    transform: translateY(-3px);
    border-color: rgba(241,90,36,0.24) !important;
    box-shadow: var(--gini-shadow) !important;
}}

:where(
    .metric-card,
    .market-card,
    .predict-card,
    .stage-card,
    .finding-card,
    .term-card,
    .about-card,
    .context-text,
    .core-card,
    .estat-feature,
    .quick-fact-card,
    .question-box,
    .super-info-box,
    .team-detail-card
)::after {{
    content: "";
    position: absolute;
    left: 12px;
    right: 12px;
    top: 0;
    height: 1px;
    pointer-events: none;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.82), transparent);
    opacity: 0.74;
}}

:where(
    .hero-logo-wrap,
    .headline-logo-wrap,
    .row-logo-wrap,
    .detail-logo-wrap,
    .hero-title-logo,
    .headshot-card
) {{
    animation: giniFloat 6.5s ease-in-out infinite;
    will-change: transform;
}}

:where(.status-pill, .refresh-badge, .metric-badge, .team-accent-badge, .square-pill, .record-badge) {{
    max-width: 100%;
    overflow-wrap: anywhere;
    box-shadow: 0 8px 18px rgba(15,23,42,0.08);
}}

:where(.live-clock-pill, .next-refresh-pill, .live-clock-value) {{
    animation: giniPulse 3.5s ease-in-out infinite;
}}

div[data-testid="stPlotlyChart"] {{
    border-color: var(--gini-border) !important;
    box-shadow: 0 16px 36px rgba(15,23,42,0.075) !important;
    transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
}}

div[data-testid="stPlotlyChart"]:hover {{
    transform: translateY(-2px);
    border-color: rgba(241,90,36,0.18) !important;
    box-shadow: 0 20px 44px rgba(15,23,42,0.10) !important;
}}

.stTabs [data-baseweb="tab-list"] {{
    gap: 0.35rem;
    padding-bottom: 0.35rem;
}}

.stTabs [data-baseweb="tab"] {{
    border-radius: 999px;
    transition: background 0.16s ease, color 0.16s ease, transform 0.16s ease;
}}

.stTabs [data-baseweb="tab"]:hover {{
    transform: translateY(-1px);
    background: rgba(255,255,255,0.74);
}}

.stButton > button,
button[kind="secondary"],
button[kind="primary"] {{
    border-radius: 999px !important;
    border: 1px solid rgba(15,23,42,0.12) !important;
    box-shadow: 0 8px 18px rgba(15,23,42,0.075) !important;
    transition: transform 0.16s ease, box-shadow 0.16s ease, border-color 0.16s ease !important;
}}

.stButton > button:hover,
button[kind="secondary"]:hover,
button[kind="primary"]:hover {{
    transform: translateY(-1px);
    border-color: rgba(241,90,36,0.28) !important;
    box-shadow: 0 12px 24px rgba(15,23,42,0.10) !important;
}}

div[data-testid="stDataFrame"],
div[data-testid="stTable"],
div[data-testid="stDataFrameResizable"] {{
    max-width: 100%;
    overflow: hidden;
    border-radius: 14px;
    border: 1px solid var(--gini-border);
    box-shadow: 0 12px 28px rgba(15,23,42,0.055);
}}

div[data-testid="stDataFrame"] * {{
    overflow-wrap: anywhere;
}}

@media (prefers-reduced-motion: reduce) {{
    *,
    *::before,
    *::after {{
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        scroll-behavior: auto !important;
        transition-duration: 0.01ms !important;
    }}
}}

body .site-top-nav {{
    position: fixed !important;
    top: 0.35rem !important;
    left: 0.5rem !important;
    right: 0.5rem !important;
    z-index: 2147483000 !important;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0;
    width: auto !important;
    max-width: none !important;
    margin: 0 !important;
    min-height: 48px;
    padding: 0.42rem 0.7rem;
    box-sizing: border-box;
    border: 1px solid rgba(15, 23, 42, 0.10);
    border-radius: 16px;
    background: rgba(255,255,255,0.82);
    box-shadow: 0 10px 28px rgba(15, 23, 42, 0.075);
    backdrop-filter: blur(14px);
    transform: translateY(0);
    opacity: 1;
    transition:
        transform 0.24s ease,
        opacity 0.18s ease,
        box-shadow 0.18s ease;
    will-change: transform;
}}

body .site-top-nav-spacer {{
    height: 0.1rem;
}}

body .site-top-nav.site-nav-hidden {{
    transform: translateY(calc(-100% - 1rem));
    opacity: 0;
    pointer-events: none;
}}

body .site-top-nav.site-nav-visible {{
    transform: translateY(0);
    opacity: 1;
    pointer-events: auto;
}}

.site-nav-links {{
    display: flex;
    align-items: center;
    justify-content: space-evenly;
    gap: 0.45rem;
    flex-wrap: wrap;
    width: 100%;
}}

.site-nav-link {{
    position: relative;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    flex: 1 1 0;
    min-height: 34px;
    padding: 0.42rem 1rem;
    border-radius: 999px;
    color: #334155 !important;
    font-size: 0.94rem;
    line-height: 1.15;
    font-weight: 900;
    text-decoration: none !important;
    text-align: center;
    white-space: nowrap;
}}

.site-nav-link:hover {{
    background: rgba(255,255,255,0.92);
    color: {safe_primary} !important;
}}

.site-nav-link.active {{
    background: #FFFFFF;
    color: {safe_primary} !important;
    box-shadow: 0 6px 16px rgba(15, 23, 42, 0.075);
}}

.site-nav-link.active::after {{
    content: none;
    display: none;
}}

@media (max-width: 760px) {{
    body .site-top-nav {{
        position: static !important;
        align-items: center;
        left: 0.35rem !important;
        right: 0.35rem !important;
        width: auto !important;
        min-height: 58px;
        margin: 0 0 0.75rem 0 !important;
        padding: 0.42rem 0.5rem;
        border-radius: 14px;
    }}

    body .site-top-nav-spacer {{
        height: 0.15rem;
    }}

    .site-nav-links {{
        justify-content: center;
        gap: 0.3rem;
    }}

    .site-nav-link {{
        flex: 1 1 42%;
        min-height: 30px;
        padding: 0.34rem 0.45rem;
        font-size: 0.78rem;
    }}

    .block-container {{
        padding-left: 0.7rem !important;
        padding-right: 0.7rem !important;
    }}

    div[data-testid="stHorizontalBlock"] {{
        flex-wrap: wrap !important;
        gap: 0.65rem !important;
    }}

    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {{
        flex: 1 1 100% !important;
        width: 100% !important;
        min-width: 0 !important;
    }}
}}
</style>

<nav class="site-top-nav"><div class="site-nav-links">{links_html}</div></nav>
<div class="site-top-nav-spacer" aria-hidden="true"></div>
""",
        unsafe_allow_html=True,
    )

    components.html(
        """
<script>
(function() {
    const parentWindow = window.parent;
    const parentDocument = parentWindow.document;
    let lastScrollY = 0;
    let initialized = false;

    function getScrollY() {
        const doc = parentDocument.documentElement;
        const body = parentDocument.body;
        const appView = parentDocument.querySelector('[data-testid="stAppViewContainer"]');
        const main = parentDocument.querySelector('section.main, [data-testid="stMain"]');
        return Math.max(
            parentWindow.scrollY || 0,
            parentWindow.pageYOffset || 0,
            doc ? doc.scrollTop || 0 : 0,
            body ? body.scrollTop || 0 : 0,
            appView ? appView.scrollTop || 0 : 0,
            main ? main.scrollTop || 0 : 0
        );
    }

    function setVisible(nav, visible) {
        nav.classList.toggle("site-nav-hidden", !visible);
        nav.classList.toggle("site-nav-visible", visible);
    }

    function updateNavVisibility() {
        const navs = parentDocument.querySelectorAll(".site-top-nav");
        if (!navs.length) {
            return;
        }

        const currentScrollY = getScrollY();
        if (!initialized) {
            lastScrollY = currentScrollY;
            initialized = true;
            navs.forEach(function(nav) { setVisible(nav, true); });
            return;
        }

        const delta = currentScrollY - lastScrollY;
        if (currentScrollY < 24 || delta < -4) {
            navs.forEach(function(nav) { setVisible(nav, true); });
        } else if (delta > 4 && currentScrollY > 90) {
            navs.forEach(function(nav) { setVisible(nav, false); });
        }

        lastScrollY = currentScrollY;
    }

    parentWindow.removeEventListener("scroll", parentWindow.__giniNavScrollHandler || function() {});
    if (parentDocument.__giniNavScrollHandler) {
        parentDocument.removeEventListener("scroll", parentDocument.__giniNavScrollHandler, true);
    }

    parentWindow.__giniNavScrollHandler = updateNavVisibility;
    parentDocument.__giniNavScrollHandler = updateNavVisibility;
    parentWindow.addEventListener("scroll", updateNavVisibility, { passive: true });
    parentDocument.addEventListener("scroll", updateNavVisibility, true);

    updateNavVisibility();
})();
</script>
""",
        height=0,
    )
