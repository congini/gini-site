import html
import streamlit as st
import streamlit.components.v1 as components


NAV_ITEMS = [
    ("About Me", "/"),
    ("Gini Dashboard", "/gini-dashboard"),
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
        align-items: center;
        left: 0.35rem !important;
        right: 0.35rem !important;
        min-height: 58px;
    }}

    body .site-top-nav-spacer {{
        height: 0.85rem;
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
