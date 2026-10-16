"""Prepare the review dataset from the raw Olist files.

Keeps reviews that have a written comment, cleans the text, attaches the
delivery outcome of the order (late or not) and writes a reproducible sample.

Usage::

    python -m review_triage.data --raw path/to/olist/datasets --out data/reviews.csv
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import duckdb
import pandas as pd

# Sample sizes per star rating. Negative reviews are over-represented on purpose:
# they are where the triage work is.
DEFAULT_SAMPLE = {1: 600, 2: 250, 3: 300, 4: 300, 5: 550}
SEED = 42

_WS = re.compile(r"\s+")


def clean_text(text: str | None) -> str:
    """Normalise whitespace (the source mixes \\r\\n, tabs and double spaces)."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    return _WS.sub(" ", str(text)).strip()


def text_hash(text: str) -> str:
    """Stable ID for a piece of text: identical comments share one LLM call."""
    return hashlib.sha256(clean_text(text).lower().encode("utf-8")).hexdigest()[:16]


def _sql_path(path: Path) -> str:
    return "'" + Path(path).as_posix().replace("'", "''") + "'"


def load_reviews(raw_dir: Path) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    reviews = _sql_path(raw_dir / "olist_order_reviews_dataset.csv")
    orders = _sql_path(raw_dir / "olist_orders_dataset.csv")
    df = duckdb.sql(
        f"""
        WITH r AS (
            SELECT * FROM read_csv_auto({reviews}, header = true)
            WHERE review_comment_message IS NOT NULL
            QUALIFY ROW_NUMBER() OVER (PARTITION BY review_id ORDER BY review_answer_timestamp DESC) = 1
        ),
        o AS (SELECT * FROM read_csv_auto({orders}, header = true))
        SELECT
            r.review_id,
            r.order_id,
            r.review_score AS stars,
            r.review_comment_title AS title,
            r.review_comment_message AS comment,
            CAST(r.review_creation_date AS DATE) AS review_date,
            o.order_status,
            CASE WHEN o.order_delivered_customer_date IS NULL THEN NULL
                 ELSE CAST(o.order_delivered_customer_date AS DATE)
                      > CAST(o.order_estimated_delivery_date AS DATE) END AS delivered_late
        FROM r LEFT JOIN o ON o.order_id = r.order_id
        """
    ).df()
    df["text"] = [
        clean_text(f"{t}. {c}" if isinstance(t, str) and t.strip() else c)
        for t, c in zip(df["title"], df["comment"], strict=True)
    ]
    df = df[df["text"].str.len() >= 3].drop(columns=["title", "comment"])
    df["text_hash"] = df["text"].map(text_hash)
    return df.reset_index(drop=True)


def stratified_sample(df: pd.DataFrame, sizes: dict[int, int] = DEFAULT_SAMPLE, seed: int = SEED):
    parts = []
    for stars, n in sizes.items():
        pool = df[df["stars"] == stars]
        parts.append(pool.sample(n=min(n, len(pool)), random_state=seed))
    return pd.concat(parts).sort_values("review_id").reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the review sample from raw Olist files")
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--out", default=Path("data/reviews.csv"), type=Path)
    args = parser.parse_args(argv)
    df = stratified_sample(load_reviews(args.raw))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} reviews to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
