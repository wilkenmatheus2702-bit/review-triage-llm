"""Business report: what are customers complaining about, and what needs action?"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

PROBLEM_TOPICS = [
    "not_received",
    "wrong_or_missing_item",
    "damaged_or_defective",
    "quality_or_description",
    "late_delivery",
    "seller_service",
    "return_or_refund",
]
BLUE, INK, INK_2, GRID, SURFACE = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def explode(reviews: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """One row per review with parsed label fields, joined to review metadata."""
    parsed = labels.dropna(subset=["label_json"]).copy()
    fields = parsed["label_json"].map(json.loads)
    parsed["sentiment"] = fields.map(lambda f: f["sentiment"])
    parsed["topics"] = fields.map(lambda f: f["topics"])
    parsed["needs_action"] = fields.map(lambda f: f["needs_action"])
    parsed["summary_en"] = fields.map(lambda f: f["summary_en"])
    cols = ["review_id", "status", "sentiment", "topics", "needs_action", "summary_en"]
    return reviews.merge(parsed[cols], on="review_id", how="inner")


def topic_table(df: pd.DataFrame) -> pd.DataFrame:
    """Share of negative reviews mentioning each problem topic."""
    neg = df[df["sentiment"] == "negative"]
    rows = []
    for t in PROBLEM_TOPICS:
        has = neg["topics"].map(lambda ts, t=t: t in ts)
        rows.append(
            {
                "topic": t,
                "reviews": int(has.sum()),
                "share_of_negative": has.mean() if len(neg) else 0.0,
                "needs_action": int((has & neg["needs_action"]).sum()),
                "avg_stars": neg.loc[has, "stars"].mean() if has.any() else None,
            }
        )
    return pd.DataFrame(rows).sort_values("reviews", ascending=False).reset_index(drop=True)


def chart(table: pd.DataFrame, out: Path, source: str) -> Path:
    d = table.sort_values("share_of_negative")
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    fig.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    labels = d["topic"].str.replace("_", " ").str.capitalize()
    ax.barh(labels, d["share_of_negative"], color=BLUE, height=0.6)
    for y, (v, n) in enumerate(zip(d["share_of_negative"], d["reviews"], strict=True)):
        ax.text(v + 0.005, y, f"{v:.0%}  ({n})", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, max(d["share_of_negative"].max() * 1.25, 0.1))
    ax.xaxis.set_visible(False)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0, labelsize=10)
    fig.text(0.02, 0.95, "What negative reviews are about", fontsize=13, fontweight="bold", color=INK)
    fig.text(
        0.02,
        0.89,
        f"Share of negative reviews mentioning each problem · classified by {source}",
        fontsize=9.5,
        color=INK_2,
    )
    fig.subplots_adjust(top=0.82, left=0.25, right=0.97, bottom=0.04)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def build_report(reviews: pd.DataFrame, labels: pd.DataFrame, source: str, out_dir: Path) -> Path:
    df = explode(reviews, labels)
    table = topic_table(df)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / f"topics_{source}.csv", index=False)
    chart(table, out_dir / f"topics_{source}.png", source)

    late = df.dropna(subset=["delivered_late"])
    late_neg = late[late["sentiment"] == "negative"].groupby("delivered_late").size()
    status = labels["status"].value_counts().to_dict()
    action = df[df["needs_action"]]
    lines = [
        f"# Review triage report ({source})",
        "",
        f"- Reviews classified: {len(df):,} | failed: {status.get('failed', 0)}",
        f"- Auto-accepted: {status.get('auto_accepted', 0):,} | sent to human review: "
        f"{status.get('needs_review', 0):,} | reviewed by a human: "
        f"{status.get('approved', 0) + status.get('corrected', 0):,}",
        f"- Need action from the company: {len(action):,} ({len(action) / max(len(df), 1):.1%})",
        f"- Total LLM cost: US${labels['cost_usd'].sum():.4f}",
        "",
        "## Problem topics in negative reviews",
        "",
        "| Topic | Reviews | Share of negative | Need action | Avg stars |",
        "|---|---|---|---|---|",
    ]
    for r in table.itertuples(index=False):
        stars = "–" if pd.isna(r.avg_stars) else f"{r.avg_stars:.1f}"
        lines.append(f"| {r.topic} | {r.reviews} | {r.share_of_negative:.0%} | {r.needs_action} | {stars} |")
    if len(late_neg) == 2:
        lines += [
            "",
            f"Negative reviews on late orders: {late_neg.get(True, 0)}; "
            f"on on-time orders: {late_neg.get(False, 0)}.",
        ]
    sample = action[action["summary_en"].str.startswith("(rules") == False].head(5)  # noqa: E712
    if len(sample):
        lines += ["", "## Example action items (English summaries)", ""]
        lines += [f"- [{', '.join(r.topics)}] {r.summary_en}" for r in sample.itertuples()]
    path = out_dir / f"report_{source}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
