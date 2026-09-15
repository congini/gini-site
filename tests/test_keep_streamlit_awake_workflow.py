from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_keep_awake_workflow_visits_the_deployed_app_within_11_hours():
    workflow = (
        PROJECT_ROOT / ".github" / "workflows" / "keep-streamlit-awake.yml"
    ).read_text(encoding="utf-8")
    visitor = (
        PROJECT_ROOT / ".github" / "scripts" / "keep_streamlit_awake.mjs"
    ).read_text(encoding="utf-8")

    assert 'cron: "17 0,11,22 * * *"' in workflow
    assert "https://gini-site.streamlit.app/" in workflow
    assert "contents: read" in workflow
    assert "playwright-core" in workflow
    assert "get this app back up" in visitor
    assert "page.waitForTimeout(30_000)" in visitor
