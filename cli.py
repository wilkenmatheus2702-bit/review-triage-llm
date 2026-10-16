"""Command line: ``python -m review_triage <command>``.

Commands
  prepare   build data/reviews.csv from the raw Olist files
  estimate  show the expected LLM cost before running
  classify  label reviews with the LLM or the rules baseline
  evaluate  compare every classifier against the gold set
  report    business summary + chart for one classifier
  revert    undo a human review decision from the audit log
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd

from . import data
from .classifiers import DEFAULT_MODEL, PRICES, LLMClassifier, RuleClassifier
from .evaluate import evaluate, load_gold, to_markdown
from .pipeline import DEFAULT_THRESHOLD, estimate_cost, run
from .report import build_report
from .store import Store

REVIEWS = Path("data/reviews.csv")
GOLD = Path("data/gold.csv")
DB = Path("output/triage.db")
OUT = Path("output")

log = logging.getLogger("review_triage")


def _reviews(gold_only: bool, limit: int | None) -> pd.DataFrame:
    df = pd.read_csv(REVIEWS)
    if gold_only:
        df = df[df["review_id"].isin(pd.read_csv(GOLD)["review_id"])]
    return df.head(limit) if limit else df


def cmd_estimate(args) -> int:
    est = estimate_cost(_reviews(args.gold_only, args.limit)["text"], args.model)
    print(json.dumps({"model": args.model, **est}, indent=2))
    return 0


def cmd_classify(args) -> int:
    reviews = _reviews(args.gold_only, args.limit)
    if args.classifier == "rules":
        classifier = RuleClassifier()
    else:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            log.error("Set ANTHROPIC_API_KEY first (see README). Try `--classifier rules` without a key.")
            return 2
        est = estimate_cost(reviews["text"], args.model)
        log.info(
            "Estimated cost: up to US$%.4f for %d reviews (budget US$%.2f)",
            est["cost_usd"],
            len(reviews),
            args.budget,
        )
        classifier = LLMClassifier(model=args.model)
    summary = run(reviews, classifier, Store(args.db), threshold=args.threshold, budget_usd=args.budget)
    print(json.dumps(summary.as_dict(), indent=2))
    return 0


def cmd_evaluate(args) -> int:
    gold = load_gold(GOLD, verified_only=args.verified_only)
    if gold.empty:
        log.error("No gold labels to evaluate%s.", " (none verified yet)" if args.verified_only else "")
        return 2
    store = Store(args.db)
    sources = args.sources or store.sources()
    results = {}
    for source in sources:
        labels = store.labels(source)
        if labels["review_id"].isin(gold["review_id"]).any():
            results[source] = evaluate(gold, labels)
    if not results:
        log.error("No classifier has labelled the gold reviews yet. Run `classify --gold-only` first.")
        return 2
    verified = int(gold["verified"].astype(bool).sum())
    note = f"Gold set: {len(gold)} reviews ({verified} verified by a human reviewer)."
    OUT.mkdir(exist_ok=True)
    (OUT / "evaluation.md").write_text(to_markdown(results, note), encoding="utf-8")
    (OUT / "evaluation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(to_markdown(results, note).split("## Per topic")[0])
    return 0


def cmd_report(args) -> int:
    labels = Store(args.db).labels(args.source)
    if labels.empty:
        log.error("No labels from %s. Run `classify` first.", args.source)
        return 2
    path = build_report(pd.read_csv(REVIEWS), labels, args.source, OUT)
    print(path.read_text(encoding="utf-8"))
    return 0


def cmd_revert(args) -> int:
    new_id = Store(args.db).revert(args.audit_id, actor=args.actor)
    print(f"Reverted audit entry {args.audit_id} (new audit entry {new_id}).")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="review_triage", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--db", type=Path, default=DB, help=f"SQLite database (default {DB})")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("prepare", help="build data/reviews.csv from raw Olist files")
    s.add_argument("--raw", required=True)

    for name in ("estimate", "classify"):
        s = sub.add_parser(name)
        s.add_argument("--model", default=DEFAULT_MODEL, choices=[m for m in PRICES if m != "rules"])
        s.add_argument("--limit", type=int, help="only the first N reviews")
        s.add_argument("--gold-only", action="store_true", help="only the 200 gold-set reviews")
    sub.choices["classify"].add_argument("--classifier", choices=["llm", "rules"], default="llm")
    sub.choices["classify"].add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="confidence below this goes to human review",
    )
    sub.choices["classify"].add_argument(
        "--budget", type=float, default=1.0, help="stop before spending more than this many US$"
    )

    s = sub.add_parser("evaluate")
    s.add_argument("--sources", nargs="*", help="classifiers to compare (default: all)")
    s.add_argument("--verified-only", action="store_true", help="only human-verified gold labels")

    s = sub.add_parser("report")
    s.add_argument("--source", default=DEFAULT_MODEL)

    s = sub.add_parser("revert")
    s.add_argument("audit_id", type=int)
    s.add_argument("--actor", default=os.environ.get("USER", "cli"))
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        return data.main(["--raw", args.raw, "--out", str(REVIEWS)])
    handlers = {
        "estimate": cmd_estimate,
        "classify": cmd_classify,
        "evaluate": cmd_evaluate,
        "report": cmd_report,
        "revert": cmd_revert,
    }
    try:
        return handlers[args.command](args)
    except (KeyError, ValueError) as exc:
        log.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
