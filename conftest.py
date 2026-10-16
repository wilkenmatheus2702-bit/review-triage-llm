from __future__ import annotations

import json

import pandas as pd
import pytest

from review_triage.data import text_hash
from review_triage.schema import ReviewLabel
from review_triage.store import Store

TEXTS = {
    "r1": "Comprei 2 cadeiras e só recebi 1. Ninguém responde meus emails.",
    "r2": "Produto excelente, chegou antes do prazo. Recomendo!",
    "r3": "Ainda não recebi o produto.",
    "r4": "ok",
    "r5": "Produto excelente, chegou antes do prazo. Recomendo!",  # duplicate text of r2
}


def label(
    sentiment="negative",
    topics=("not_received",),
    needs_action=True,
    confidence=0.9,
    summary="Customer has not received the order.",
):
    return ReviewLabel(
        sentiment=sentiment,
        topics=list(topics),
        needs_action=needs_action,
        summary_en=summary,
        confidence=confidence,
    )


@pytest.fixture
def reviews() -> pd.DataFrame:
    df = pd.DataFrame({"review_id": list(TEXTS), "text": list(TEXTS.values())})
    df["text_hash"] = df["text"].map(text_hash)
    df["stars"] = [1, 5, 1, 3, 5]
    df["delivered_late"] = [False, False, None, False, False]
    return df


@pytest.fixture
def store(tmp_path) -> Store:
    return Store(tmp_path / "triage.db")


@pytest.fixture
def answers() -> dict:
    return {
        TEXTS["r1"]: label(topics=("wrong_or_missing_item", "seller_service"), confidence=0.95),
        TEXTS["r2"]: label("positive", ("praise",), False, 0.97, "Great product, arrived early."),
        TEXTS["r3"]: label(confidence=0.92),
        TEXTS["r4"]: label("neutral", ("other",), False, 0.4, "Very short, unclear comment."),
    }


def fake_anthropic(responses: list[dict], seen: list[dict]):
    """A real `anthropic.Anthropic` client whose HTTP layer returns canned API responses.

    Exercises the actual SDK request building and response parsing, with no network.
    """
    import anthropic

    try:  # recent SDKs ship their own fork of httpx
        import httpx2 as httpx
    except ImportError:  # pragma: no cover
        import httpx

    queue = list(responses)

    def handler(request):
        seen.append(json.loads(request.content))
        body = queue.pop(0)
        status = body.pop("_status", 200)
        return httpx.Response(status, json=body)

    return anthropic.Anthropic(
        api_key="test-key", max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )


def api_response(tool_input: dict | None, input_tokens=1200, output_tokens=80, text=None) -> dict:
    content = []
    if text:
        content.append({"type": "text", "text": text})
    if tool_input is not None:
        content.append({"type": "tool_use", "id": "toolu_01", "name": "record_triage", "input": tool_input})
    return {
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-5-5",
        "content": content,
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }
