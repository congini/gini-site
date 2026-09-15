from datetime import datetime, timedelta

import streamlit as st
import streamlit.components.v1 as components

from refresh_current_season import refresh_current_season_if_due

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None

st.set_page_config(
    page_title="Gini Metric",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def schedule_daily_app_reload(hour=23, minute=59):
    """Reload any open page when the metadata-driven refresh becomes due."""
    eastern = ZoneInfo("America/New_York") if ZoneInfo is not None else None
    moment = datetime.now(eastern)
    next_reload = moment.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if moment >= next_reload:
        next_reload += timedelta(days=1)
    delay_ms = max(1000, int((next_reload - moment).total_seconds() * 1000))
    components.html(
        f"""
<script>
(function() {{
  window.setTimeout(function() {{
    try {{
      window.parent.location.reload();
    }} catch (error) {{
      window.location.reload();
    }}
  }}, {delay_ms});
}})();
</script>
""",
        height=0,
        scrolling=False,
    )


schedule_daily_app_reload()

auto_refresh_ok, auto_refresh_message, auto_refresh_status = refresh_current_season_if_due()
if auto_refresh_ok:
    st.cache_data.clear()
elif not auto_refresh_status.get("skipped", False):
    st.warning(auto_refresh_message)

pages = [
    st.Page(
        "1_About_Me.py",
        title="About Me",
        icon="🏈",
        default=True,
    ),
    st.Page(
        "pages/Gini_Dashboard.py",
        title="Gini Dashboard",
        icon="📊",
        url_path="gini-dashboard",
    ),
    st.Page(
        "pages/Live_Leaderboard.py",
        title="Live Leaderboard",
        icon="📈",
        url_path="live-leaderboard",
    ),
    st.Page(
        "pages/Super_Square.py",
        title="Super Square",
        icon="🏆",
        url_path="super-square",
    ),
    st.Page(
        "pages/Predictive_Model.py",
        title="Predictive Model",
        icon="🔮",
        url_path="predictive-model",
    ),
    st.Page(
        "pages/Model_History.py",
        title="Model History",
        icon="📚",
        url_path="model-history",
    ),
    st.Page(
        "pages/Terminology.py",
        title="Terminology",
        icon="📖",
        url_path="terminology",
    ),
]

pg = st.navigation(pages, position="hidden")
pg.run()
