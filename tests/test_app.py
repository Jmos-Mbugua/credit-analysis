"""Smoke tests for the Streamlit app: every page renders without errors."""

from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def _run(page: str | None = None) -> "AppTest":
    app = AppTest.from_file(APP, default_timeout=60).run()
    if page is not None:
        app.radio(key="page").set_value(page).run()
    assert not app.exception, app.exception
    return app


def test_score_a_customer_page_shows_score_pd_and_decision() -> None:
    app = _run()
    labels = [m.label for m in app.metric]
    assert labels[:2] == ["Score", "Probability of default"]
    assert any("Approve" in m.value or "Refer" in m.value or "Decline" in m.value for m in app.markdown)


def test_manual_entry_scores_without_errors() -> None:
    app = _run()
    app.radio(key="customer_mode").set_value("Enter values manually").run()
    assert not app.exception
    app.selectbox(key="pay_1").set_value(2).run()
    assert not app.exception
    assert int(app.metric[0].value) < 520  # 2 months late now: decline territory for the sample's first customer


def test_batch_page_renders_upload_and_template() -> None:
    app = _run("Score a batch")
    assert any("Upload a CSV" in m.value for m in app.markdown)


def test_performance_page_shows_test_metrics() -> None:
    app = _run("Model performance")
    assert [m.label for m in app.metric] == ["Gini (test)", "KS (test)", "AUC (test)"]
    assert app.metric[0].value == "0.517"
