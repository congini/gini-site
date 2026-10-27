"""Smoke-test production UI calls with the installed Streamlit runtime."""

import ast
from pathlib import Path

from streamlit.testing.v1 import AppTest


PAGE = Path(__file__).resolve().parents[1] / "pages" / "Live_Leaderboard.py"


def leaderboard_ui_app():
    # Load the real renderers without triggering source refreshes or snapshot writes.
    tree = ast.parse(PAGE.read_text(encoding="utf-8"))
    functions = {
        "escape", "number", "pct", "ordinal", "current_record_text",
        "logo_source", "logo_html", "team_history_html", "render_leaderboard",
        "roster_strengths", "render_team_detail", "render_css",
        "weekly_movement_html", "movement_arrow_svg",
    }
    colors = {"PRIMARY", "SECONDARY", "NAVY", "TEXT", "MUTED"}
    definitions = [
        node for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in functions)
        or (isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in colors
            for target in node.targets
        ))
    ]
    source = "\n".join([
        "import html\nimport pandas as pd\nimport streamlit as st",
        "from live_leaderboard_history import week_label",
        ast.unparse(ast.Module(body=definitions, type_ignores=[])),
        """
leaderboard = pd.DataFrame([{
    'team': 'DEN', 'team_name': 'Denver Broncos', 'team_logo': '',
    'season': 2026, 'live_rank': 1, 'live_market_score': 90.0,
    'current_gini_score': 110.0, 'roster_score': 85.0, 'projected_wins': 10.5,
    'projected_wins_low': 8, 'projected_wins_high': 13,
    'current_wins': 5, 'current_losses': 2, 'current_ties': 0,
    'status_label': 'High Upside', 'most_likely_quadrant': 'Q3 - Playoff Contenders',
    'highest_quadrant_probability': .6,
    'q1_probability': .1, 'q2_probability': .1,
    'q3_probability': .6, 'q4_probability': .2,
    'weekly_rank_change': 3, 'weekly_movement_available': True,
    'weekly_comparison_key': '2026-Week-06',
}])
period = st.query_params.get('leaderboard_week', '2026-Week-07')
render_css()
render_leaderboard(leaderboard, pd.DataFrame(columns=['team', 'week']), 'Current',
                   period, ['2026-Week-07', '2026-Week-06'])
render_team_detail(leaderboard, production_found=True)
""",
    ])
    return AppTest.from_string(source, default_timeout=10)


def test_header_and_team_picker_render_and_navigate_without_runtime_errors():
    app = leaderboard_ui_app().run()
    assert not app.exception
    assert app.selectbox(key="live_leaderboard_team_detail").value == "#1 Denver Broncos (DEN)"
    assert app.button(key="leaderboard-week-forward").disabled
    cards = next(item.value for item in app.markdown if 'class="leaderboard-wrap"' in item.value)
    assert 'weekly-up' in cards and 'High Upside' not in cards
    assert 'Moved up 3 places from finalized Week 6' in cards

    app.button(key="leaderboard-week-back").click().run()
    assert not app.exception
    assert app.query_params["leaderboard_week"] == ["2026-Week-06"]
    assert app.button(key="leaderboard-week-back").disabled

    app.button(key="leaderboard-week-forward").click().run()
    assert not app.exception
    assert app.query_params["leaderboard_week"] == ["2026-Week-07"]
