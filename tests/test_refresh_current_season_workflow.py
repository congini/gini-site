from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "refresh-current-season.yml"


def test_refresh_is_staged_before_six_and_waits_for_target_time():
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert 'cron: "47 5 * * 2"' in workflow
    assert 'timezone: "America/New_York"' in workflow
    assert "Wait until 6:00 AM Eastern" in workflow
    assert "06:00:00" in workflow


def test_refresh_remains_tuesday_only():
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "workflow_dispatch" not in workflow
    assert workflow.count("cron:") == 1
