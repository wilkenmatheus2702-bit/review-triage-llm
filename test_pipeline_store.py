import pandas as pd
import pytest

from review_triage.classifiers import ReplayClassifier, RuleClassifier
from review_triage.pipeline import estimate_cost, route, run

from .conftest import label


def statuses(store, source):
    return store.labels(source).set_index("review_id")["status"].to_dict()


def test_routing_by_confidence(reviews, store, answers):
    summary = run(reviews, ReplayClassifier(answers), store, threshold=0.7)
    s = statuses(store, "replay")
    assert s["r1"] == s["r2"] == s["r3"] == "auto_accepted"
    assert s["r4"] == "needs_review"  # confidence 0.4
    assert summary.auto_accepted == 4 and summary.needs_review == 1  # r5 is a duplicate of r2


def test_duplicate_text_is_served_from_cache(reviews, store, answers):
    clf = ReplayClassifier(answers)
    summary = run(reviews, clf, store)
    assert clf.calls == 4  # 5 reviews, one duplicate text
    assert summary.cache_hits == 1


def test_rerun_costs_nothing(reviews, store, answers):
    first = run(reviews, ReplayClassifier(answers), store)
    clf = ReplayClassifier(answers)
    second = run(reviews, clf, store)
    assert first.cost_usd == 0 or first.cost_usd > 0
    assert clf.calls == 0 and second.cache_hits == 5 and second.cost_usd == 0


def test_failed_predictions_go_to_queue(reviews, store, answers):
    answers.pop(reviews.loc[2, "text"])  # r3 now has no answer
    summary = run(reviews, ReplayClassifier(answers), store)
    labels = store.labels("replay").set_index("review_id")
    assert labels.loc["r3", "status"] == "failed" and pd.isna(labels.loc["r3", "label_json"])
    assert summary.failed == 1


def test_budget_stops_the_run(reviews, store, answers):
    clf = ReplayClassifier(answers, name="claude-haiku-5-5")  # priced model: 500 in / 60 out per call
    summary = run(reviews, clf, store, budget_usd=0.0001)
    assert summary.stopped_by_budget
    assert summary.processed < len(reviews)


def test_human_decisions_are_never_overwritten(reviews, store, answers):
    run(reviews, ReplayClassifier(answers), store)
    store.review("r4", "replay", label("neutral", ("other",), False, 1.0, "Human label"), actor="ana")
    summary = run(reviews, ReplayClassifier(answers), store)
    assert summary.skipped_reviewed == 1
    assert "Human label" in store.labels("replay").set_index("review_id").loc["r4", "label_json"]


def test_approve_correct_and_audit(store, reviews, answers):
    run(reviews, ReplayClassifier(answers), store)
    a1 = store.review("r4", "replay", None, actor="ana")
    a2 = store.review("r1", "replay", label(topics=("wrong_or_missing_item",), confidence=1.0), actor="bo")
    log = store.audit_log().set_index("audit_id")
    assert log.loc[a1, "action"] == "approve" and log.loc[a1, "after_status"] == "approved"
    assert log.loc[a2, "action"] == "correct" and log.loc[a2, "before_status"] == "auto_accepted"
    assert statuses(store, "replay")["r1"] == "corrected"


def test_revert_restores_previous_label(store, reviews, answers):
    run(reviews, ReplayClassifier(answers), store)
    before = store.labels("replay").set_index("review_id").loc["r1", "label_json"]
    audit = store.review("r1", "replay", label(topics=("other",), confidence=1.0), actor="bo")
    store.revert(audit, actor="lead")
    row = store.labels("replay").set_index("review_id").loc["r1"]
    assert row["label_json"] == before and row["status"] == "auto_accepted"
    log = store.audit_log()
    assert log.iloc[0]["action"] == "revert" and log.iloc[0]["reverts_id"] == audit
    with pytest.raises(ValueError):
        store.revert(int(log.iloc[0]["audit_id"]), actor="lead")


def test_review_errors(store, reviews, answers):
    answers.pop(reviews.loc[2, "text"])
    run(reviews, ReplayClassifier(answers), store)
    with pytest.raises(ValueError, match="failed prediction"):
        store.review("r3", "replay", None, actor="ana")
    with pytest.raises(KeyError):
        store.review("nope", "replay", None, actor="ana")
    with pytest.raises(KeyError):
        store.revert(999, actor="ana")


def test_rules_run_on_real_sample(store):
    df = pd.read_csv("data/reviews.csv").head(50)
    summary = run(df, RuleClassifier(), store)
    assert summary.processed == 50 and summary.failed == 0 and summary.cost_usd == 0


def test_estimate_cost():
    est = estimate_cost(pd.Series(["a" * 320, "b" * 320, "a" * 320]), "claude-haiku-5-5")
    assert est["reviews"] == 3 and est["unique_texts"] == 2
    assert 0 < est["cost_usd"] < 0.01


@pytest.mark.parametrize("conf, expected", [(None, "failed"), (0.69, "needs_review"), (0.7, "auto_accepted")])
def test_route(conf, expected):
    assert route(conf, 0.7) == expected
