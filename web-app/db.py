"""
SQLite-backed review storage for the Grain Review app.
Thread-safe via WAL mode and thread-local connections.

Tables:
    reviews     – one row per (team, datapoint_id)
    issue_types – shared global list of issue-type labels
"""
import json
import sqlite3
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "reviews.db"
_local = threading.local()


def _get_conn():
    """Return a thread-local SQLite connection."""
    if not hasattr(_local, "conn") or _local.conn is None:
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        _local.conn = conn
    return _local.conn


# ------------------------------------------------------------------
# Schema
# ------------------------------------------------------------------
def init_db():
    """Create tables if they don't exist."""
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS reviews (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            team            TEXT    NOT NULL,
            sample_number   INTEGER NOT NULL,
            datapoint_id    TEXT    NOT NULL,
            form_type       TEXT    NOT NULL,
            sample_category TEXT,
            verdict         TEXT    NOT NULL CHECK(verdict IN ('accept', 'flag')),
            issue_types     TEXT,
            remark          TEXT,
            created_at      TEXT    NOT NULL,
            updated_at      TEXT    NOT NULL,
            UNIQUE(team, datapoint_id)
        );

        CREATE TABLE IF NOT EXISTS issue_types (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            label           TEXT    NOT NULL UNIQUE COLLATE NOCASE,
            created_by_team TEXT    NOT NULL,
            created_at      TEXT    NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_reviews_sample
            ON reviews(sample_number);
        CREATE INDEX IF NOT EXISTS idx_reviews_team
            ON reviews(team);
    """)
    conn.commit()
    _ensure_optional_review_columns()


def _ensure_optional_review_columns():
    """ADD COLUMN migrations for existing SQLite DBs (idempotent)."""
    conn = _get_conn()
    try:
        conn.execute(
            "ALTER TABLE reviews ADD COLUMN reviewer_username TEXT"
        )
        conn.commit()
    except sqlite3.OperationalError as e:
        if "duplicate column name" not in str(e).lower():
            raise


# ------------------------------------------------------------------
# Reviews
# ------------------------------------------------------------------
def save_review(team, sample_number, datapoint_id, form_type,
                sample_category, verdict, issue_types=None, remark=None,
                reviewer_username=None):
    """UPSERT a review. issue_types is a list of strings, stored as JSON."""
    conn = _get_conn()
    now = datetime.now(timezone.utc).isoformat()
    issue_types_json = json.dumps(issue_types) if issue_types else None
    conn.execute("""
        INSERT INTO reviews
            (team, sample_number, datapoint_id, form_type,
             sample_category, verdict, issue_types, remark, reviewer_username,
             created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(team, datapoint_id) DO UPDATE SET
            verdict     = excluded.verdict,
            issue_types = excluded.issue_types,
            remark      = excluded.remark,
            reviewer_username = excluded.reviewer_username,
            updated_at  = excluded.updated_at
    """, (team, int(sample_number), str(datapoint_id), form_type,
          sample_category, verdict, issue_types_json, remark, reviewer_username,
          now, now))
    conn.commit()
    return {
        "team": team,
        "datapoint_id": str(datapoint_id),
        "verdict": verdict,
        "issue_types": issue_types or [],
        "remark": remark,
        "updated_at": now,
    }


def delete_review(team, datapoint_id):
    """Delete a review. Returns True if a row was deleted."""
    conn = _get_conn()
    cursor = conn.execute(
        "DELETE FROM reviews WHERE team = ? AND datapoint_id = ?",
        (team, str(datapoint_id)),
    )
    conn.commit()
    return cursor.rowcount > 0


def _parse_issue_types(raw):
    """Parse the issue_types JSON column, handling legacy scalar values."""
    if raw is None:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
        return [parsed]
    except (json.JSONDecodeError, TypeError):
        return [raw] if raw else []


def get_reviews_for_sample(team, sample_number):
    """Return {datapoint_id: {verdict, issue_types, remark, updated_at}} for one team + sample."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT datapoint_id, verdict, issue_types, remark, updated_at, "
        "       reviewer_username "
        "FROM reviews WHERE team = ? AND sample_number = ?",
        (team, int(sample_number)),
    ).fetchall()
    result = {}
    for row in rows:
        d = dict(row)
        d["issue_types"] = _parse_issue_types(d["issue_types"])
        result[d["datapoint_id"]] = d
    return result


def get_reviews_all_teams_for_sample(sample_number):
    """
    Return { datapoint_id: [ { team, verdict, issue_types, remark, reviewer_username,
                               form_type, sample_category, updated_at, ... }, ... ] }
    for every team's review on this sample.
    """
    conn = _get_conn()
    rows = conn.execute(
        "SELECT team, datapoint_id, form_type, sample_category, verdict, "
        "       issue_types, remark, reviewer_username, updated_at "
        "FROM reviews WHERE sample_number = ? "
        "ORDER BY datapoint_id, team",
        (int(sample_number),),
    ).fetchall()
    by_dp: dict[str, list] = defaultdict(list)
    for row in rows:
        d = dict(row)
        d["issue_types"] = _parse_issue_types(d["issue_types"])
        by_dp[str(d["datapoint_id"])].append(d)
    return dict(by_dp)


def get_all_reviews():
    """Return all reviews as a list of dicts (for dashboard)."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT team, sample_number, datapoint_id, verdict, "
        "       issue_types, remark, created_at, updated_at "
        "FROM reviews"
    ).fetchall()
    results = []
    for r in rows:
        d = dict(r)
        d["issue_types"] = _parse_issue_types(d["issue_types"])
        results.append(d)
    return results


def get_review_counts_by_sample_and_team():
    """
    Aggregated review counts per (sample_number, team).
    Returns list of dicts: sample_number, team, n_reviewed, n_accepted, n_flagged.
    """
    conn = _get_conn()
    rows = conn.execute(
        """
        SELECT sample_number, team,
               COUNT(*) AS n_reviewed,
               SUM(CASE WHEN verdict = 'accept' THEN 1 ELSE 0 END) AS n_accepted,
               SUM(CASE WHEN verdict = 'flag' THEN 1 ELSE 0 END) AS n_flagged
        FROM reviews
        GROUP BY sample_number, team
        """
    ).fetchall()
    return [dict(r) for r in rows]


# ------------------------------------------------------------------
# Issue Types
# ------------------------------------------------------------------
def get_issue_types():
    """Return sorted list of issue-type labels."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT label FROM issue_types ORDER BY label"
    ).fetchall()
    return [r["label"] for r in rows]


def add_issue_type(label, created_by_team):
    """Add a new issue type. Returns True if added, False if duplicate."""
    conn = _get_conn()
    now = datetime.now(timezone.utc).isoformat()
    try:
        conn.execute(
            "INSERT INTO issue_types (label, created_by_team, created_at) "
            "VALUES (?, ?, ?)",
            (label.strip(), created_by_team, now),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
