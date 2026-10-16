import pandas as pd
import pytest
from pydantic import ValidationError

from review_triage.classifiers import LLMClassifier, RuleClassifier, cost_usd, normalise
from review_triage.prompt import FEW_SHOT, system_prompt
from review_triage.schema import TOPICS, ReviewLabel, tool_schema

from .conftest import api_response, fake_anthropic

GOOD = {
    "sentiment": "negative",
    "topics": ["not_received"],
    "needs_action": True,
    "summary_en": "Customer has not received the order.",
    "confidence": 0.92,
}


# Schema -------------------------------------------------------------------
def test_schema_rejects_unknown_topic():
    with pytest.raises(ValidationError, match="unknown topics"):
        ReviewLabel(**{**GOOD, "topics": ["shipping"]})


@pytest.mark.parametrize("field, value", [("topics", []), ("confidence", 1.5), ("sentiment", "angry")])
def test_schema_rejects_invalid_values(field, value):
    with pytest.raises(ValidationError):
        ReviewLabel(**{**GOOD, field: value})


def test_schema_dedupes_and_sorts_topics():
    assert ReviewLabel(**{**GOOD, "topics": ["praise", "late_delivery", "praise"]}).topics == [
        "late_delivery",
        "praise",
    ]


def test_tool_schema_matches_taxonomy():
    schema = tool_schema()
    assert schema["properties"]["topics"]["items"]["enum"] == list(TOPICS)
    assert set(schema["required"]) == set(ReviewLabel.model_fields)


def test_prompt_examples_are_valid_and_not_from_the_gold_set():
    gold_texts = set(pd.read_csv("data/gold.csv")["text"])
    for text, example in FEW_SHOT:
        ReviewLabel.model_validate(example)
        assert text not in gold_texts  # no evaluation leakage
    for topic in TOPICS:
        assert topic in system_prompt()


# LLM classifier (real SDK, fake HTTP) ---------------------------------------
def test_llm_parses_forced_tool_call():
    seen = []
    clf = LLMClassifier(client=fake_anthropic([api_response(GOOD)], seen))
    pred = clf.classify("Ainda não recebi")
    assert pred.label.topics == ["not_received"] and pred.error is None
    assert (pred.input_tokens, pred.output_tokens) == (1200, 80)
    request = seen[0]
    assert request["model"] == "claude-haiku-5-5"
    assert request["tool_choice"] == {"type": "tool", "name": "record_triage"}
    assert request["tools"][0]["input_schema"] == tool_schema()
    assert "Ainda não recebi" in request["messages"][0]["content"]


def test_llm_retries_once_with_the_validation_error():
    seen = []
    bad = {**GOOD, "topics": ["shipping_problem"]}
    clf = LLMClassifier(client=fake_anthropic([api_response(bad), api_response(GOOD)], seen))
    pred = clf.classify("Ainda não recebi")
    assert pred.label is not None
    assert pred.input_tokens == 2400  # both calls are counted
    retry = seen[1]["messages"]
    assert retry[1]["role"] == "assistant"
    assert retry[2]["content"][0]["type"] == "tool_result" and retry[2]["content"][0]["is_error"]


def test_llm_gives_up_after_retry_and_reports_error():
    bad = {**GOOD, "confidence": 3}
    clf = LLMClassifier(client=fake_anthropic([api_response(bad), api_response(bad)], []))
    pred = clf.classify("x")
    assert pred.label is None and pred.error.startswith("invalid_output")


def test_llm_handles_missing_tool_call():
    clf = LLMClassifier(client=fake_anthropic([api_response(None, text="I cannot help")], []))
    pred = clf.classify("x")
    assert pred.label is None and "did not call the tool" in pred.error


def test_llm_api_errors_do_not_crash():
    error = {"_status": 401, "type": "error", "error": {"type": "authentication_error", "message": "bad key"}}
    pred = LLMClassifier(client=fake_anthropic([error], [])).classify("x")
    assert pred.label is None and pred.error.startswith("api_error")


def test_cost():
    assert cost_usd("claude-haiku-5-5", 1_000_000, 1_000_000) == pytest.approx(0.60)
    assert cost_usd("rules", 10**9, 10**9) == 0
    assert cost_usd("unknown-model", 100, 100) == 0


# Rules baseline ---------------------------------------------------------------
def test_normalise_strips_accents():
    assert normalise("NÃO Recebi") == "nao recebi"


@pytest.mark.parametrize(
    "text, topics, sentiment, action",
    [
        ("Ainda não recebi o produto", {"not_received"}, "negative", True),
        ("Comprei 2 relógios mas só veio 1", {"wrong_or_missing_item"}, "negative", True),
        ("Nao veio o espeto", {"wrong_or_missing_item"}, "negative", True),
        ("Produto ótimo, chegou antes do prazo", {"praise"}, "positive", False),
        ("Não recomendo, veio quebrado", {"damaged_or_defective"}, "negative", True),
        ("Produto bom mas a entrega demorou", {"late_delivery", "praise"}, "neutral", False),
        ("Quero meu dinheiro de volta", {"return_or_refund"}, "negative", True),
        ("hmm", {"other"}, "neutral", False),
    ],
)
def test_rules(text, topics, sentiment, action):
    label = RuleClassifier().classify(text).label
    assert set(label.topics) == topics
    assert label.sentiment == sentiment
    assert label.needs_action is action
