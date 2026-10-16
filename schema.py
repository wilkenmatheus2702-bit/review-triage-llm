"""The output contract: every label, from the LLM or a human, must fit this schema."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator

# Topic taxonomy. Definitions are shown to the model in the prompt and to
# reviewers in the review app, so both apply the same rules.
TOPICS: dict[str, str] = {
    "late_delivery": "The order arrived, but after the promised date or slower than expected.",
    "not_received": "The customer says the order (or the whole order) has not arrived yet.",
    "wrong_or_missing_item": "A different product, colour, size or voltage was sent, or items are missing.",
    "damaged_or_defective": "The product arrived broken, dirty, scratched, or does not work.",
    "quality_or_description": (
        "Works, but quality is poor or it differs from the listing (size, material, look)."
    ),
    "seller_service": "Problems with contact, answers, information or support from the seller/store.",
    "return_or_refund": (
        "The customer asks for, or is waiting on, a return, exchange, cancellation or refund."
    ),
    "praise": "Positive feedback about the product, delivery or store.",
    "other": "Anything else: price, packaging only, unclear or generic comments.",
}


class Sentiment(str, Enum):
    positive = "positive"
    neutral = "neutral"
    negative = "negative"


class ReviewLabel(BaseModel):
    """Structured triage of one customer review."""

    sentiment: Sentiment
    topics: list[str] = Field(min_length=1, max_length=4)
    needs_action: bool = Field(
        description="True if the customer has an unresolved problem the company must act on."
    )
    summary_en: str = Field(min_length=3, max_length=200, description="One-sentence summary in English.")
    confidence: float = Field(ge=0, le=1)

    @field_validator("topics")
    @classmethod
    def topics_in_taxonomy(cls, value: list[str]) -> list[str]:
        unknown = sorted(set(value) - set(TOPICS))
        if unknown:
            raise ValueError(f"unknown topics {unknown}; allowed: {sorted(TOPICS)}")
        return sorted(set(value))


def tool_schema() -> dict:
    """JSON schema handed to the LLM as a tool, so its answer is forced into this shape."""
    return {
        "type": "object",
        "properties": {
            "sentiment": {"type": "string", "enum": [s.value for s in Sentiment]},
            "topics": {
                "type": "array",
                "items": {"type": "string", "enum": list(TOPICS)},
                "minItems": 1,
                "maxItems": 4,
            },
            "needs_action": {"type": "boolean"},
            "summary_en": {"type": "string", "description": "One sentence in English, max 25 words."},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["sentiment", "topics", "needs_action", "summary_en", "confidence"],
    }
