"""Run a classifier over reviews: cache, budget guard, and routing to human review."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import pandas as pd

from .classifiers import Classifier, cost_usd
from .prompt import PROMPT_VERSION, system_prompt
from .store import Store

log = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.7
OUTPUT_TOKENS_ESTIMATE = 90
CHARS_PER_TOKEN = 3.2  # Portuguese text, conservative


@dataclass
class RunSummary:
    source: str
    processed: int = 0
    cache_hits: int = 0
    skipped_reviewed: int = 0
    auto_accepted: int = 0
    needs_review: int = 0
    failed: int = 0
    cost_usd: float = 0.0
    stopped_by_budget: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


def estimate_cost(texts: pd.Series, model: str) -> dict:
    """Upper-bound cost estimate before spending anything."""
    prompt_tokens = len(system_prompt()) / CHARS_PER_TOKEN + 600  # + tool schema
    input_tokens = int(len(texts) * prompt_tokens + texts.str.len().sum() / CHARS_PER_TOKEN)
    output_tokens = len(texts) * OUTPUT_TOKENS_ESTIMATE
    return {
        "reviews": len(texts),
        "unique_texts": int(texts.nunique()),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost_usd(model, input_tokens, output_tokens), 4),
    }


def route(confidence: float | None, threshold: float) -> str:
    if confidence is None:
        return "failed"
    return "auto_accepted" if confidence >= threshold else "needs_review"


def run(
    reviews: pd.DataFrame,
    classifier: Classifier,
    store: Store,
    threshold: float = DEFAULT_THRESHOLD,
    budget_usd: float | None = None,
) -> RunSummary:
    """Classify each review and store the result.

    - Reviews already approved/corrected by a human are skipped.
    - LLM answers are cached by (text, model, prompt version): re-runs and
      duplicate comments cost nothing.
    - Low-confidence and failed predictions go to the human review queue.
    - The run stops before exceeding ``budget_usd``.
    """
    summary = RunSummary(source=classifier.name)
    for row in reviews.itertuples(index=False):
        if store.status(row.review_id, classifier.name) in ("approved", "corrected"):
            summary.skipped_reviewed += 1
            continue
        if budget_usd is not None and summary.cost_usd >= budget_usd:
            summary.stopped_by_budget = True
            log.warning("Budget of US$%.2f reached; stopping.", budget_usd)
            break

        label = (
            store.cached(row.text_hash, classifier.name, PROMPT_VERSION) if classifier.uses_cache else None
        )
        if label is not None:
            summary.cache_hits += 1
            tokens_in = tokens_out = 0
            cost = 0.0
            error = None
        else:
            pred = classifier.classify(row.text)
            label, error = pred.label, pred.error
            tokens_in, tokens_out, cost = pred.input_tokens, pred.output_tokens, pred.cost_usd
            summary.cost_usd += cost
            if label is not None and classifier.uses_cache:
                store.put_cache(row.text_hash, classifier.name, PROMPT_VERSION, label)

        status = route(label.confidence if label else None, threshold)
        store.save_prediction(
            review_id=row.review_id,
            source=classifier.name,
            text_hash=row.text_hash,
            prompt_version=PROMPT_VERSION,
            label=label,
            status=status,
            error=error,
            input_tokens=tokens_in,
            output_tokens=tokens_out,
            cost_usd=cost,
        )
        summary.processed += 1
        setattr(summary, status, getattr(summary, status) + 1)
        if summary.processed % 100 == 0:
            log.info("%d reviews processed (US$%.4f so far)", summary.processed, summary.cost_usd)

    summary.cost_usd = round(summary.cost_usd, 6)
    return summary
