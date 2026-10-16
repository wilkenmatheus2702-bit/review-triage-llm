"""Prompt for the LLM classifier.

PROMPT_VERSION is part of the cache key: changing the prompt invalidates cached
answers, so results always match the prompt that produced them.
"""

from __future__ import annotations

from .schema import TOPICS

PROMPT_VERSION = "v1"

_TOPIC_LINES = "\n".join(f"- {name}: {desc}" for name, desc in TOPICS.items())

SYSTEM_PROMPT = f"""You triage customer reviews for a Brazilian online marketplace.
Reviews are in Portuguese, often informal, with typos and no punctuation.
For each review, call the `record_triage` tool exactly once.

Topics (choose 1-4 that apply):
{_TOPIC_LINES}

Rules:
- If the order has NOT arrived yet, use not_received (not late_delivery).
- Mixed reviews can have both a problem topic and praise.
- needs_action is true when the customer still has a problem the company must fix:
  item not received, wrong/missing/damaged item, a pending return or refund,
  or an unanswered contact. Late-but-received orders with no other issue: false.
- sentiment reflects the text, not the star rating.
- summary_en: one neutral sentence in English describing what the customer said.
- confidence: your probability (0-1) that the topics and needs_action are correct.
  Use below 0.7 when the text is ambiguous, very short, or sarcastic."""

# Invented examples (not taken from the dataset), so the evaluation set never
# leaks into the prompt. A test enforces this.
FEW_SHOT = [
    (
        "Pedi uma cafeteira 110v e veio 220v. Mandei email e ninguém respondeu ainda.",
        {
            "sentiment": "negative",
            "topics": ["seller_service", "wrong_or_missing_item"],
            "needs_action": True,
            "summary_en": "Customer got a 220V coffee maker instead of 110V and no reply to an email.",
            "confidence": 0.95,
        },
    ),
    (
        "A luminária é linda, mas a entrega atrasou quase uma semana.",
        {
            "sentiment": "neutral",
            "topics": ["late_delivery", "praise"],
            "needs_action": False,
            "summary_en": "Customer likes the lamp but delivery was almost a week late.",
            "confidence": 0.9,
        },
    ),
]


def system_prompt() -> str:
    """System prompt plus the worked examples."""
    import json

    examples = "\n\n".join(
        f"Example review: {text}\nExpected record_triage input: {json.dumps(label, ensure_ascii=False)}"
        for text, label in FEW_SHOT
    )
    return f"{SYSTEM_PROMPT}\n\n{examples}"


def user_message(text: str) -> str:
    return f"Review:\n<review>\n{text}\n</review>"
