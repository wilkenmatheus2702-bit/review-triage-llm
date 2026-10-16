import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from review_triage import cli
from review_triage.classifiers import ReplayClassifier
from review_triage.evaluate import evaluate, to_markdown
from review_triage.pipeline import run
from review_triage.report import build_report, topic_table

from .conftest import label

ROOT = Path(__file__).parents[1]


def gold_frame():
    g = pd.DataFrame(
        {
            "review_id": ["a", "b", "c", "d"],
            "sentiment": ["negative", "positive", "negative", "neutral"],
            "topics": ["not_received", "praise", "wrong_or_missing_item;seller_service", "other"],
            "needs_action": [True, False, True, False],
            "verified": [True, True, False, True],
        }
    )
    g["topic_set"] = g["topics"].str.split(";").map(set)
    return g


def labels_frame(preds: dict):
    return pd.DataFrame(
        {
            "review_id": list(preds),
            "label_json": [p.model_dump_json() if p else None for p in preds.values()],
        }
    )


def test_perfect_predictions_score_100():
    g = gold_frame()
    preds = {
        "a": label(),
        "b": label("positive", ("praise",), False),
        "c": label(topics=("wrong_or_missing_item", "seller_service")),
        "d": label("neutral", ("other",), False),
    }
    m = evaluate(g, labels_frame(preds))
    for key in ("sentiment_accuracy", "topics_micro_f1", "topics_exact_match", "needs_action_recall"):
        assert m[key] == 1.0


def test_metrics_by_hand():
    g = gold_frame()
    preds = {
        "a": label(),  # all correct
        "b": label("negative", ("praise", "late_delivery"), True),  # sentiment, extra topic, action FP
        "c": label(topics=("wrong_or_missing_item",)),  # misses seller_service
        "d": None,  # classifier failed
    }
    m = evaluate(g, labels_frame(preds))
    assert m["sentiment_accuracy"] == pytest.approx(2 / 4)
    # topics: TP = not_received, praise, wrong_or_missing = 3; FP = late_delivery; FN = seller_service, other
    assert m["topics_micro_precision"] == pytest.approx(3 / 4)
    assert m["topics_micro_recall"] == pytest.approx(3 / 5)
    assert m["topics_exact_match"] == pytest.approx(1 / 4)
    # needs_action: TP a, c ; FP b ; FN none
    assert m["needs_action_precision"] == pytest.approx(2 / 3)
    assert m["needs_action_recall"] == 1.0
    assert m["coverage"] == 1.0


def test_markdown_has_one_column_per_classifier():
    g = gold_frame()
    m = evaluate(g, labels_frame({"a": label()}))
    md = to_markdown({"rules": m, "claude": m}, "note")
    assert "| Metric | rules | claude |" in md and "## Per topic: claude" in md


def test_report(reviews, store, answers, tmp_path):
    run(reviews, ReplayClassifier(answers), store)
    labels = store.labels("replay")
    path = build_report(reviews, labels, "replay", tmp_path)
    text = path.read_text()
    assert "Need action from the company" in text and (tmp_path / "topics_replay.png").exists()
    table = topic_table(
        pd.DataFrame(
            {
                "sentiment": ["negative", "negative", "positive"],
                "topics": [["not_received"], ["praise"], ["praise"]],
                "needs_action": [True, False, False],
                "stars": [1, 2, 5],
            }
        )
    ).set_index("topic")
    assert table.loc["not_received", "share_of_negative"] == 0.5


@pytest.fixture
def project(tmp_path, monkeypatch):
    """Run CLI commands inside a copy of the project's data folder.

    The copy's gold labels are reset to unverified, so these tests behave the
    same before and after the real gold set has been verified.
    """
    shutil.copytree(ROOT / "data", tmp_path / "data")
    gold = pd.read_csv(tmp_path / "data/gold.csv")
    gold["verified"] = False
    gold.to_csv(tmp_path / "data/gold.csv", index=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_cli_rules_flow(project, capsys):
    assert cli.main(["classify", "--classifier", "rules", "--gold-only"]) == 0
    assert cli.main(["evaluate"]) == 0
    assert "Sentiment accuracy" in capsys.readouterr().out
    assert json.loads((project / "output/evaluation.json").read_text())["rules"]["reviews"] == 200
    assert cli.main(["report", "--source", "rules"]) == 0
    assert cli.main(["estimate", "--limit", "10"]) == 0


def test_cli_llm_without_key_fails_cleanly(project, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert cli.main(["classify", "--limit", "1"]) == 2


def test_cli_errors(project):
    assert cli.main(["evaluate"]) == 2  # nothing classified yet
    assert cli.main(["evaluate", "--verified-only"]) == 2  # draft gold labels not verified
    assert cli.main(["report", "--source", "rules"]) == 2
    assert cli.main(["revert", "42"]) == 2


def _app(project, monkeypatch):
    monkeypatch.setenv("TRIAGE_DB", str(project / "output/triage.db"))
    monkeypatch.setenv("TRIAGE_GOLD", str(project / "data/gold.csv"))
    monkeypatch.setenv("TRIAGE_REVIEWS", str(project / "data/reviews.csv"))
    return AppTest.from_file(str(ROOT / "review_app.py"), default_timeout=60)


def test_review_app_queue_approve_and_revert(project, monkeypatch):
    cli.main(["classify", "--classifier", "rules", "--limit", "30"])
    at = _app(project, monkeypatch).run()
    assert not at.exception
    queued = int(at.metric[0].value)
    assert queued > 0
    next(b for b in at.button if b.label == "Approve suggestion").click().run()
    assert not at.exception and int(at.metric[0].value) == queued - 1
    next(b for b in at.button if b.label == "Revert this decision").click().run()
    assert not at.exception and int(at.metric[0].value) == queued


def test_review_app_gold_verification(project, monkeypatch):
    at = _app(project, monkeypatch).run()
    assert not at.exception
    next(b for b in at.button if b.label == "Confirm / save").click().run()
    assert not at.exception
    assert pd.read_csv(project / "data/gold.csv")["verified"].sum() == 1
