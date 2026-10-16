"""Human review app: work the low-confidence queue, audit and revert decisions,
and verify the gold labels used for evaluation.

Run with:  streamlit run review_app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from review_triage.schema import TOPICS, ReviewLabel, Sentiment  # noqa: E402
from review_triage.store import Store  # noqa: E402

DB = Path(os.environ.get("TRIAGE_DB", ROOT / "output" / "triage.db"))
GOLD = Path(os.environ.get("TRIAGE_GOLD", ROOT / "data" / "gold.csv"))
REVIEWS = Path(os.environ.get("TRIAGE_REVIEWS", ROOT / "data" / "reviews.csv"))
SENTIMENTS = [s.value for s in Sentiment]

st.set_page_config(page_title="Review Triage", page_icon="🗂️", layout="wide")


def topic_label(t: str) -> str:
    return t.replace("_", " ").capitalize()


def label_form(key: str, current: dict | None, submit_labels: tuple[str, ...]) -> tuple[str | None, dict]:
    """Editable label form. Returns (button pressed, field values)."""
    current = current or {}
    with st.form(key):
        sentiment = st.radio(
            "Sentiment",
            SENTIMENTS,
            horizontal=True,
            index=SENTIMENTS.index(current.get("sentiment", "neutral")),
        )
        topics = st.multiselect(
            "Topics",
            list(TOPICS),
            default=current.get("topics", []),
            format_func=topic_label,
            max_selections=4,
        )
        needs_action = st.checkbox("Needs action from the company", value=current.get("needs_action", False))
        summary = st.text_input("English summary", value=current.get("summary_en", ""))
        cols = st.columns(len(submit_labels))
        pressed = None
        for col, text in zip(cols, submit_labels, strict=True):
            if col.form_submit_button(text, type="primary" if text == submit_labels[0] else "secondary"):
                pressed = text
    return pressed, {
        "sentiment": sentiment,
        "topics": topics,
        "needs_action": needs_action,
        "summary_en": summary or "(reviewed by a human)",
    }


with st.expander("Topic definitions", expanded=False):
    st.table(pd.DataFrame({"Topic": [topic_label(t) for t in TOPICS], "Definition": TOPICS.values()}))

st.sidebar.header("Reviewer")
actor = st.sidebar.text_input("Your name", value=os.environ.get("USER", "reviewer"), key="actor")
store = Store(DB)
reviews = pd.read_csv(REVIEWS).set_index("review_id") if REVIEWS.exists() else pd.DataFrame()

st.title("Review triage")
tab_queue, tab_audit, tab_gold = st.tabs(["Review queue", "Audit log", "Verify gold labels"])

# --------------------------------------------------------------------------- queue
with tab_queue:
    sources = store.sources()
    if not sources:
        st.info("No predictions yet. Run `python -m review_triage classify` first.")
    else:
        source = st.selectbox("Classifier", sources, key="source")
        labels = store.labels(source)
        queue = labels[labels["status"].isin(["needs_review", "failed"])].sort_values("confidence")
        done = labels["status"].isin(["approved", "corrected"]).sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("In queue", len(queue))
        c2.metric("Reviewed by humans", int(done))
        c3.metric("Auto-accepted", int((labels["status"] == "auto_accepted").sum()))

        if queue.empty:
            st.success("Queue is empty. Nothing left to review.")
        else:
            item = queue.iloc[0]
            meta = reviews.loc[item["review_id"]] if item["review_id"] in reviews.index else None
            left, right = st.columns([3, 2])
            with left:
                st.markdown("##### Customer review (Portuguese)")
                if meta is not None:
                    late = {True: "late", False: "on time"}.get(meta.get("delivered_late"), "not delivered")
                    st.caption(f"{'★' * int(meta['stars'])}{'☆' * (5 - int(meta['stars']))} · order {late}")
                    st.markdown(f"> {meta['text']}")
                st.caption(f"Review ID {item['review_id']}")
            with right:
                st.markdown("##### Model suggestion")
                if isinstance(item["label_json"], str):  # NULL arrives as NaN, which is truthy
                    suggestion = ReviewLabel.model_validate_json(item["label_json"]).model_dump(mode="json")
                    st.write(f"**Confidence:** {suggestion['confidence']:.0%}")
                    st.write(f"**Summary:** {suggestion['summary_en']}")
                else:
                    suggestion = None
                    st.error(f"The classifier failed: {item['error']}")
            pressed, values = label_form(
                "queue_form",
                suggestion,
                ("Save correction", "Approve suggestion") if suggestion else ("Save correction",),
            )
            if pressed == "Approve suggestion":
                store.review(item["review_id"], source, None, actor=actor)
                st.toast("Approved")
                st.rerun()
            elif pressed == "Save correction":
                if not values["topics"]:
                    st.error("Pick at least one topic.")
                else:
                    label = ReviewLabel(**values, confidence=1.0)
                    store.review(item["review_id"], source, label, actor=actor)
                    st.toast("Correction saved")
                    st.rerun()

# --------------------------------------------------------------------------- audit
with tab_audit:
    log = store.audit_log(limit=200)
    if log.empty:
        st.info("No human decisions yet.")
    else:
        st.dataframe(
            log[
                [
                    "audit_id",
                    "at_utc",
                    "actor",
                    "action",
                    "source",
                    "review_id",
                    "before_status",
                    "after_status",
                    "reverts_id",
                ]
            ],
            hide_index=True,
            width="stretch",
        )
        with st.form("revert_form"):
            audit_id = st.number_input(
                "Audit ID to revert", min_value=1, step=1, value=int(log["audit_id"].iloc[0])
            )
            if st.form_submit_button("Revert this decision"):
                try:
                    store.revert(int(audit_id), actor=actor)
                    st.toast(f"Reverted #{audit_id}")
                    st.rerun()
                except (KeyError, ValueError) as exc:
                    st.error(str(exc))

# --------------------------------------------------------------------------- gold
with tab_gold:
    st.caption(
        "The gold set is the answer key used to measure accuracy. Each draft label must be "
        "checked by a person before the evaluation results are published."
    )
    gold = pd.read_csv(GOLD)
    gold["verified"] = gold["verified"].astype(bool)
    verified = int(gold["verified"].sum())
    st.progress(verified / len(gold), text=f"{verified} of {len(gold)} verified")
    pending = gold[~gold["verified"]]
    if pending.empty:
        st.success("All gold labels are verified.")
    else:
        idx = pending.index[0]
        row = gold.loc[idx]
        st.caption(f"{'★' * int(row['stars'])}{'☆' * (5 - int(row['stars']))}")
        st.markdown(f"> {row['text']}")
        current = {
            "sentiment": row["sentiment"],
            "topics": row["topics"].split(";"),
            "needs_action": bool(row["needs_action"]),
            "summary_en": "(gold label)",
        }
        pressed, values = label_form("gold_form", current, ("Confirm / save", "Skip for now"))
        if pressed == "Confirm / save":
            if not values["topics"]:
                st.error("Pick at least one topic.")
            else:
                gold.loc[idx, ["sentiment", "topics", "needs_action", "verified"]] = [
                    values["sentiment"],
                    ";".join(sorted(values["topics"])),
                    values["needs_action"],
                    True,
                ]
                gold.to_csv(GOLD, index=False)
                st.rerun()
        elif pressed == "Skip for now":
            gold = pd.concat([gold.drop(idx), gold.loc[[idx]]])
            gold.to_csv(GOLD, index=False)
            st.rerun()
