"""Compare classifier labels against the hand-labelled gold set."""

from __future__ import annotations

import json

import pandas as pd

from .schema import TOPICS


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def load_gold(path, verified_only: bool = False) -> pd.DataFrame:
    gold = pd.read_csv(path)
    if verified_only:
        gold = gold[gold["verified"].astype(bool)]
    gold["topic_set"] = gold["topics"].str.split(";").map(set)
    return gold


def evaluate(gold: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """Metrics for one classifier. ``labels`` is a Store.labels() frame.

    Reviews the classifier failed on count as wrong (no silent exclusion).
    """
    preds = {}
    for row in labels.itertuples(index=False):
        preds[row.review_id] = json.loads(row.label_json) if isinstance(row.label_json, str) else None

    n = len(gold)
    covered = sum(1 for r in gold["review_id"] if r in preds)
    sentiment_ok = exact = 0
    act = {"tp": 0, "fp": 0, "fn": 0}
    per_topic = {t: {"tp": 0, "fp": 0, "fn": 0} for t in TOPICS}

    for row in gold.itertuples(index=False):
        p = preds.get(row.review_id)
        p_topics = set(p["topics"]) if p else set()
        p_sent = p["sentiment"] if p else None
        p_act = bool(p["needs_action"]) if p else None
        sentiment_ok += p_sent == row.sentiment
        exact += p_topics == row.topic_set
        for t in TOPICS:
            if t in p_topics and t in row.topic_set:
                per_topic[t]["tp"] += 1
            elif t in p_topics:
                per_topic[t]["fp"] += 1
            elif t in row.topic_set:
                per_topic[t]["fn"] += 1
        g_act = bool(row.needs_action)
        if p_act and g_act:
            act["tp"] += 1
        elif p_act and not g_act:
            act["fp"] += 1
        elif g_act and not p_act:
            act["fn"] += 1

    micro = _prf(*(sum(v[k] for v in per_topic.values()) for k in ("tp", "fp", "fn")))
    topic_rows = []
    for t, c in per_topic.items():
        p, r, f = _prf(c["tp"], c["fp"], c["fn"])
        topic_rows.append({"topic": t, "support": c["tp"] + c["fn"], "precision": p, "recall": r, "f1": f})
    supported = [row["f1"] for row in topic_rows if row["support"] > 0]
    ap, ar, af = _prf(act["tp"], act["fp"], act["fn"])
    return {
        "reviews": n,
        "coverage": covered / n if n else 0.0,
        "sentiment_accuracy": sentiment_ok / n if n else 0.0,
        "topics_micro_precision": micro[0],
        "topics_micro_recall": micro[1],
        "topics_micro_f1": micro[2],
        "topics_macro_f1": sum(supported) / len(supported) if supported else 0.0,
        "topics_exact_match": exact / n if n else 0.0,
        "needs_action_precision": ap,
        "needs_action_recall": ar,
        "needs_action_f1": af,
        "per_topic": topic_rows,
    }


SUMMARY_ROWS = [
    ("sentiment_accuracy", "Sentiment accuracy"),
    ("topics_micro_f1", "Topics micro-F1"),
    ("topics_macro_f1", "Topics macro-F1"),
    ("topics_exact_match", "Topics exact match"),
    ("needs_action_precision", "Needs-action precision"),
    ("needs_action_recall", "Needs-action recall"),
]


def to_markdown(results: dict[str, dict], gold_note: str) -> str:
    """Side-by-side comparison of several classifiers."""
    names = list(results)
    lines = [
        "# Evaluation against the gold set",
        "",
        gold_note,
        "",
        "| Metric | " + " | ".join(names) + " |",
        "|---|" + "---|" * len(names),
    ]
    for key, label in SUMMARY_ROWS:
        lines.append(f"| {label} | " + " | ".join(f"{results[n][key]:.1%}" for n in names) + " |")
    for name in names:
        lines += [
            "",
            f"## Per topic: {name}",
            "",
            "| Topic | Support | Precision | Recall | F1 |",
            "|---|---|---|---|---|",
        ]
        for row in sorted(results[name]["per_topic"], key=lambda r: -r["support"]):
            lines.append(
                f"| {row['topic']} | {row['support']} | {row['precision']:.0%} | "
                f"{row['recall']:.0%} | {row['f1']:.0%} |"
            )
    return "\n".join(lines) + "\n"
