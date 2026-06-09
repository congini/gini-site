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

DEFAULT_PRIMARY = "#F15A24"
DEFAULT_SECONDARY = "#0073B7"


def _safe_hex_color(value, fallback):
    text = str(value or "").strip()
    if text.startswith("#") and len(text) == 4:
        text = "#" + "".join(char * 2 for char in text[1:])

    if (
        text.startswith("#")
        and len(text) == 7
        and all(char in "0123456789abcdefABCDEF" for char in text[1:])
    ):
        return text

    return fallback


def render_global_background(primary_color=DEFAULT_PRIMARY, secondary_color=DEFAULT_SECONDARY):
    safe_primary = _safe_hex_color(primary_color, DEFAULT_PRIMARY)
    safe_secondary = _safe_hex_color(secondary_color, DEFAULT_SECONDARY)

    st.markdown(
        f"""
<style>
:root {{
    --estat-ambient-primary: {safe_primary};
    --estat-ambient-secondary: {safe_secondary};
}}

html,
body {{
    background: #F8FAFC !important;
}}

.stApp,
[data-testid="stAppViewContainer"] {{
    color-scheme: light;
    background:
        linear-gradient(118deg, transparent 0 41%, {safe_secondary}0B 41.15%, transparent 42.3%, transparent 58%, {safe_primary}0B 58.15%, transparent 59.4%),
        radial-gradient(circle at 13% 9%, {safe_primary}18 0%, transparent 28%),
        radial-gradient(circle at 88% 13%, {safe_secondary}18 0%, transparent 30%),
        radial-gradient(circle at 52% 98%, rgba(0, 115, 183, 0.08) 0%, transparent 34%),
        linear-gradient(180deg, #FFFFFF 0%, #F7FAFC 48%, #EEF5FA 100%) !important;
    background-size:
        145% 145%,
        132% 132%,
        138% 138%,
        130% 130%,
        100% 100%;
    background-position:
        0 0,
        0 0,
        0 0,
        0 0,
        0 0;
    animation: estatAmbientTextureDrift 30s ease-in-out infinite alternate !important;
    will-change: background-position;
    overflow-x: hidden;
}}

[data-testid="stAppViewContainer"] {{
    position: relative;
    isolation: isolate;
}}

[data-testid="stAppViewContainer"]::before,
[data-testid="stAppViewContainer"]::after {{
    content: "";
    position: fixed;
    pointer-events: none;
    z-index: 0;
    filter: blur(34px);
    transform: translate3d(0, 0, 0);
    will-change: transform, opacity;
}}

[data-testid="stAppViewContainer"]::before {{
    top: -34vmax;
    left: -24vmax;
    width: 76vmax;
    height: 76vmax;
    border-radius: 999px;
    background:
        radial-gradient(ellipse at center, {safe_primary}36 0%, {safe_primary}16 34%, transparent 66%);
    opacity: 0.58;
    animation: estatAmbientPrimary 24s ease-in-out infinite alternate !important;
    animation-play-state: running !important;
}}

[data-testid="stAppViewContainer"]::after {{
    right: -26vmax;
    bottom: -34vmax;
    width: 82vmax;
    height: 82vmax;
    border-radius: 999px;
    background:
        radial-gradient(ellipse at center, {safe_secondary}34 0%, {safe_secondary}14 34%, transparent 68%);
    opacity: 0.52;
    animation: estatAmbientSecondary 28s ease-in-out infinite alternate !important;
    animation-play-state: running !important;
}}

.estat-ambient-motion {{
    position: fixed;
    inset: -14vh -12vw;
    z-index: 0;
    pointer-events: none;
    overflow: hidden;
    contain: paint;
}}

.estat-ambient-dots {{
    position: absolute;
    inset: 0;
    z-index: 2;
    pointer-events: none;
    opacity: 0.42;
    background-image:
        radial-gradient(circle, rgba(15, 23, 42, 0.17) 1px, transparent 1.35px),
        radial-gradient(circle, rgba(15, 23, 42, 0.12) 0.85px, transparent 1.2px);
    background-size: 28px 28px, 44px 44px;
    background-position: 0 0, 14px 18px;
    animation: estatAmbientDotDrift 34s ease-in-out infinite alternate !important;
    transform: translate3d(0, 0, 0);
    will-change: background-position, transform;
}}

.estat-ambient-glow {{
    position: absolute;
    display: block;
    z-index: 1;
    border-radius: 999px;
    filter: blur(36px);
    transform: translate3d(0, 0, 0);
    will-change: transform, opacity;
}}

.estat-ambient-glow-primary {{
    top: -12vmax;
    left: -10vmax;
    width: 54vmax;
    height: 54vmax;
    background:
        radial-gradient(ellipse at center, {safe_primary}30 0%, {safe_primary}14 36%, transparent 68%);
    opacity: 0.62;
    animation: estatAmbientPrimary 24s ease-in-out infinite alternate !important;
}}

.estat-ambient-glow-secondary {{
    right: -12vmax;
    bottom: -14vmax;
    width: 58vmax;
    height: 58vmax;
    background:
        radial-gradient(ellipse at center, {safe_secondary}30 0%, {safe_secondary}14 36%, transparent 70%);
    opacity: 0.58;
    animation: estatAmbientSecondary 28s ease-in-out infinite alternate !important;
}}

.bg-canvas,
.team-bg-canvas,
.predict-bg {{
    display: none !important;
}}

[data-testid="stAppViewContainer"] > .main,
[data-testid="stMain"],
section.main,
.block-container {{
    position: relative;
    z-index: 1;
}}

[data-testid="stHeader"] {{
    background: transparent !important;
}}

@keyframes estatAmbientPrimary {{
    0% {{
        opacity: 0.48;
        transform: translate3d(-3vw, -2vh, 0) scale(1);
    }}
    50% {{
        opacity: 0.74;
        transform: translate3d(18vw, 10vh, 0) scale(1.08);
    }}
    100% {{
        opacity: 0.56;
        transform: translate3d(8vw, 28vh, 0) scale(1.03);
    }}
}}

@keyframes estatAmbientSecondary {{
    0% {{
        opacity: 0.52;
        transform: translate3d(3vw, 2vh, 0) scale(1);
    }}
    50% {{
        opacity: 0.72;
        transform: translate3d(-20vw, -9vh, 0) scale(1.07);
    }}
    100% {{
        opacity: 0.52;
        transform: translate3d(-8vw, -28vh, 0) scale(1.04);
    }}
}}

@keyframes estatAmbientTextureDrift {{
    0% {{
        background-position:
            0 0,
            0 0,
            0 0,
            0 0,
            0 0;
    }}
    100% {{
        background-position:
            148px -104px,
            -112px 86px,
            126px 96px,
            0 -124px,
            0 0;
    }}
}}

@keyframes estatAmbientDotDrift {{
    0% {{
        transform: translate3d(-1.5vw, -0.8vh, 0);
        background-position: 0 0, 14px 18px;
    }}
    50% {{
        transform: translate3d(2vw, 1.3vh, 0);
        background-position: 64px 44px, -28px 64px;
    }}
    100% {{
        transform: translate3d(-1vw, 1.8vh, 0);
        background-position: 128px 88px, -72px 110px;
    }}
}}

@media (max-width: 760px) {{
    .stApp,
    [data-testid="stAppViewContainer"] {{
        background-size:
            165% 165%,
            150% 150%,
            155% 155%,
            145% 145%,
            100% 100% !important;
    }}

    [data-testid="stAppViewContainer"]::before,
    [data-testid="stAppViewContainer"]::after,
    .estat-ambient-dots,
    .estat-ambient-glow {{
        filter: blur(28px);
        opacity: 0.42;
    }}
}}

@media (prefers-reduced-motion: reduce) {{
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"]::before,
    [data-testid="stAppViewContainer"]::after,
    .estat-ambient-dots,
    .estat-ambient-glow {{
        animation: none !important;
        transform: none !important;
    }}
}}
</style>
<div class="estat-ambient-motion" aria-hidden="true">
    <span class="estat-ambient-dots"></span>
    <span class="estat-ambient-glow estat-ambient-glow-primary"></span>
    <span class="estat-ambient-glow estat-ambient-glow-secondary"></span>
</div>
""",
        unsafe_allow_html=True,
    )


def render_top_nav(active_page, primary=DEFAULT_PRIMARY, secondary=DEFAULT_SECONDARY):
    safe_primary = _safe_hex_color(primary, DEFAULT_PRIMARY)
    safe_secondary = _safe_hex_color(secondary, DEFAULT_SECONDARY)

    links_html = "\n".join(
        f'<a class="site-nav-link{" active" if label == active_page else ""}" href="{href}" target="_self">{label}</a>'
        for label, href in NAV_ITEMS
    )

    render_global_background(safe_primary, safe_secondary)

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
    max-width: 100%;
    overflow-x: hidden;
}}

img,
svg,
canvas {{
    max-width: 100%;
}}

div[data-testid="stPlotlyChart"] {{
    max-width: 100%;
    overflow: hidden;
}}

.stTabs {{
    margin-top: 0.25rem;
}}

.stTabs [data-baseweb="tab-list"] {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.42rem;
    row-gap: 0.42rem;
    align-items: center;
    width: 100%;
    padding: 0.34rem;
    border-radius: 16px;
    border: 1px solid rgba(15, 23, 42, 0.12);
    background:
        linear-gradient(135deg, rgba(255,255,255,0.94), rgba(255,255,255,0.76));
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.72),
        0 12px 26px rgba(15, 23, 42, 0.075);
    backdrop-filter: blur(14px) saturate(1.12);
    -webkit-backdrop-filter: blur(14px) saturate(1.12);
}}

.stTabs [data-baseweb="tab"] {{
    min-height: 42px;
    flex: 1 1 160px;
    justify-content: center;
    white-space: normal;
    text-align: center;
    border-radius: 12px;
    border: 1px solid rgba(15, 23, 42, 0.10);
    background: rgba(255,255,255,0.76);
    color: #0F172A !important;
    font-size: 0.92rem;
    font-weight: 900;
    line-height: 1.15;
    padding: 0.62rem 0.82rem;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.70);
    transition:
        background 0.18s ease,
        border-color 0.18s ease,
        box-shadow 0.18s ease,
        color 0.18s ease,
        transform 0.18s ease;
}}

.stTabs [data-baseweb="tab"] p,
.stTabs [data-baseweb="tab"] span {{
    color: inherit !important;
    font-weight: inherit !important;
}}

.stTabs [data-baseweb="tab"]:hover {{
    border-color: {primary}88;
    background: rgba(255,255,255,0.96);
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.86),
        0 8px 18px rgba(15, 23, 42, 0.08);
    transform: translateY(-1px);
}}

.stTabs [aria-selected="true"] {{
    border-color: transparent !important;
    background: linear-gradient(135deg, {primary}, {secondary}) !important;
    color: #FFFFFF !important;
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.24),
        0 12px 24px {primary}36 !important;
}}

.stTabs [aria-selected="true"] p,
.stTabs [aria-selected="true"] span {{
    color: #FFFFFF !important;
}}

.stTabs [data-baseweb="tab-highlight"] {{
    height: 0 !important;
    background: transparent !important;
}}

@media (max-width: 700px) {{
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.34rem;
        padding: 0.28rem;
        border-radius: 14px;
    }}

    .stTabs [data-baseweb="tab"] {{
        min-height: 38px;
        flex-basis: calc(50% - 0.34rem);
        padding: 0.52rem 0.56rem;
        font-size: 0.82rem;
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
    overflow: hidden !important;
    border: 1px solid rgba(255, 255, 255, 0.58) !important;
    border-radius: 16px !important;
    background:
        linear-gradient(135deg, rgba(255,255,255,0.58), rgba(255,255,255,0.36)) !important;
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.72),
        inset 0 -1px 0 rgba(15,23,42,0.04),
        0 12px 32px rgba(15, 23, 42, 0.085) !important;
    backdrop-filter: blur(22px) saturate(1.28) !important;
    -webkit-backdrop-filter: blur(22px) saturate(1.28) !important;
    transform: translateY(0);
    opacity: 1;
    transition:
        transform 0.24s ease,
        opacity 0.18s ease,
        box-shadow 0.18s ease;
    will-change: transform;
}}

body .site-top-nav::before {{
    content: "";
    position: absolute;
    inset: 0;
    z-index: 0;
    pointer-events: none;
    background:
        linear-gradient(120deg, rgba(255,255,255,0.42), rgba(255,255,255,0.06) 44%, rgba(255,255,255,0.24)),
        radial-gradient(circle at 10% 0%, rgba(255,255,255,0.46), transparent 38%);
    opacity: 0.68;
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
    position: relative;
    z-index: 1;
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
    content: "";
    position: absolute;
    left: 0.9rem;
    right: 0.9rem;
    bottom: 0.22rem;
    height: 2px;
    border-radius: 999px;
    background: linear-gradient(90deg, {safe_primary}, {safe_secondary});
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
        border-radius: 14px !important;
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
