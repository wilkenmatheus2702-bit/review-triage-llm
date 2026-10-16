"""SQLite storage: labels, an LLM response cache, and an audit log with revert.

Nothing a human approves or corrects is ever overwritten silently: every
change writes the before/after state to ``audit_log``, and any change can be
reverted from there.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .schema import ReviewLabel

DDL = """
CREATE TABLE IF NOT EXISTS labels (
    review_id      TEXT NOT NULL,
    source         TEXT NOT NULL,     -- classifier name, e.g. claude-haiku-5-5 or rules
    text_hash      TEXT NOT NULL,
    prompt_version TEXT,
    label_json     TEXT,              -- NULL when the classifier failed
    confidence     REAL,
    status         TEXT NOT NULL,     -- auto_accepted | needs_review | approved | corrected | failed
    error          TEXT,
    input_tokens   INTEGER DEFAULT 0,
    output_tokens  INTEGER DEFAULT 0,
    cost_usd       REAL DEFAULT 0,
    updated_at     TEXT NOT NULL,
    PRIMARY KEY (review_id, source)
);
CREATE TABLE IF NOT EXISTS llm_cache (
    text_hash      TEXT NOT NULL,
    model          TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    label_json     TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    PRIMARY KEY (text_hash, model, prompt_version)
);
CREATE TABLE IF NOT EXISTS audit_log (
    audit_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    review_id      TEXT NOT NULL,
    source         TEXT NOT NULL,
    action         TEXT NOT NULL,     -- approve | correct | revert
    before_status  TEXT,
    after_status   TEXT,
    before_json    TEXT,
    after_json     TEXT,
    actor          TEXT NOT NULL,
    at_utc         TEXT NOT NULL,
    reverts_id     INTEGER
);
"""

REVIEWED = ("approved", "corrected")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._con() as con:
            con.executescript(DDL)

    def _con(self):
        con = sqlite3.connect(self.path)
        con.row_factory = sqlite3.Row
        return closing(con)

    # Cache -------------------------------------------------------------------
    def cached(self, text_hash: str, model: str, prompt_version: str) -> ReviewLabel | None:
        with self._con() as con:
            row = con.execute(
                "SELECT label_json FROM llm_cache WHERE text_hash=? AND model=? AND prompt_version=?",
                (text_hash, model, prompt_version),
            ).fetchone()
        return ReviewLabel.model_validate_json(row["label_json"]) if row else None

    def put_cache(self, text_hash: str, model: str, prompt_version: str, label: ReviewLabel) -> None:
        with self._con() as con, con:
            con.execute(
                "INSERT OR REPLACE INTO llm_cache VALUES (?, ?, ?, ?, ?)",
                (text_hash, model, prompt_version, label.model_dump_json(), _now()),
            )

    # Labels ------------------------------------------------------------------
    def status(self, review_id: str, source: str) -> str | None:
        with self._con() as con:
            row = con.execute(
                "SELECT status FROM labels WHERE review_id=? AND source=?", (review_id, source)
            ).fetchone()
        return row["status"] if row else None

    def save_prediction(
        self,
        *,
        review_id,
        source,
        text_hash,
        prompt_version,
        label,
        status,
        error,
        input_tokens=0,
        output_tokens=0,
        cost_usd=0.0,
    ) -> None:
        """Insert or refresh a machine label. Human-reviewed labels are never overwritten."""
        if self.status(review_id, source) in REVIEWED:
            return
        with self._con() as con, con:
            con.execute(
                "INSERT OR REPLACE INTO labels VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    review_id,
                    source,
                    text_hash,
                    prompt_version,
                    label.model_dump_json() if label else None,
                    label.confidence if label else None,
                    status,
                    error,
                    input_tokens,
                    output_tokens,
                    cost_usd,
                    _now(),
                ),
            )

    def labels(self, source: str) -> pd.DataFrame:
        with self._con() as con:
            return pd.read_sql("SELECT * FROM labels WHERE source = ?", con, params=(source,))

    def sources(self) -> list[str]:
        with self._con() as con:
            return [r[0] for r in con.execute("SELECT DISTINCT source FROM labels ORDER BY 1")]

    # Human review ------------------------------------------------------------
    def review(self, review_id: str, source: str, label: ReviewLabel | None, actor: str) -> int:
        """Approve the current label (label=None) or replace it with a correction.

        Returns the audit_id of the change.
        """
        with self._con() as con, con:
            row = con.execute(
                "SELECT label_json, status FROM labels WHERE review_id=? AND source=?", (review_id, source)
            ).fetchone()
            if row is None:
                raise KeyError(f"no label for review {review_id} from {source}")
            if label is None:
                if row["label_json"] is None:
                    raise ValueError("cannot approve a failed prediction; submit a correction")
                action, new_status, new_json = "approve", "approved", row["label_json"]
            else:
                action, new_status, new_json = "correct", "corrected", label.model_dump_json()
            con.execute(
                "UPDATE labels SET label_json=?, status=?, confidence=?, updated_at=? "
                "WHERE review_id=? AND source=?",
                (new_json, new_status, json.loads(new_json)["confidence"], _now(), review_id, source),
            )
            cur = con.execute(
                "INSERT INTO audit_log (review_id, source, action, before_status, after_status, "
                "before_json, after_json, actor, at_utc) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    review_id,
                    source,
                    action,
                    row["status"],
                    new_status,
                    row["label_json"],
                    new_json,
                    actor,
                    _now(),
                ),
            )
            return cur.lastrowid

    def revert(self, audit_id: int, actor: str) -> int:
        """Undo one audited change, restoring the label and status it replaced."""
        with self._con() as con, con:
            entry = con.execute("SELECT * FROM audit_log WHERE audit_id=?", (audit_id,)).fetchone()
            if entry is None:
                raise KeyError(f"audit entry {audit_id} not found")
            if entry["action"] == "revert":
                raise ValueError("a revert cannot itself be reverted; review the label again instead")
            current = con.execute(
                "SELECT label_json, status FROM labels WHERE review_id=? AND source=?",
                (entry["review_id"], entry["source"]),
            ).fetchone()
            restored = entry["before_json"]
            con.execute(
                "UPDATE labels SET label_json=?, status=?, confidence=?, updated_at=? "
                "WHERE review_id=? AND source=?",
                (
                    restored,
                    entry["before_status"],
                    json.loads(restored)["confidence"] if restored else None,
                    _now(),
                    entry["review_id"],
                    entry["source"],
                ),
            )
            cur = con.execute(
                "INSERT INTO audit_log (review_id, source, action, before_status, after_status, "
                "before_json, after_json, actor, at_utc, reverts_id) "
                "VALUES (?, ?, 'revert', ?, ?, ?, ?, ?, ?, ?)",
                (
                    entry["review_id"],
                    entry["source"],
                    current["status"],
                    entry["before_status"],
                    current["label_json"],
                    restored,
                    actor,
                    _now(),
                    audit_id,
                ),
            )
            return cur.lastrowid

    def audit_log(self, limit: int = 50) -> pd.DataFrame:
        with self._con() as con:
            return pd.read_sql("SELECT * FROM audit_log ORDER BY audit_id DESC LIMIT ?", con, params=(limit,))
