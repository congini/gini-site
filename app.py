import streamlit as st

st.set_page_config(
    page_title="Gini Metric",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

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
