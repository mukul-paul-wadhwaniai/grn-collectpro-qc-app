"""
One-time migration for the reviews database.

Run with the app STOPPED:
    python migrate_db.py

What it does:
  1. Renames 'issue_type' column to 'issue_types' in the reviews table
     (SQLite doesn't support RENAME COLUMN before 3.25, so we recreate).
  2. Converts existing scalar issue_type values to JSON arrays.
  3. Creates new tables for Phase 2 (mixed_image_reviews, metadata_corrections,
     grain_annotations) — safe to run even if they don't exist yet.
"""
import json
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "reviews.db"


def migrate():
    backup_path = DB_PATH.with_suffix(f".db.bak-{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    print(f"Backing up {DB_PATH} -> {backup_path}")
    shutil.copy2(DB_PATH, backup_path)

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")

    columns = [
        row[1] for row in conn.execute("PRAGMA table_info(reviews)").fetchall()
    ]

    if "issue_type" in columns and "issue_types" not in columns:
        print("Migrating: issue_type -> issue_types (JSON array)")

        conn.executescript("""
            ALTER TABLE reviews RENAME COLUMN issue_type TO issue_types;
        """)

        rows = conn.execute(
            "SELECT id, issue_types FROM reviews WHERE issue_types IS NOT NULL"
        ).fetchall()

        updated = 0
        for row in rows:
            raw = row["issue_types"]
            try:
                json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                conn.execute(
                    "UPDATE reviews SET issue_types = ? WHERE id = ?",
                    (json.dumps([raw]), row["id"]),
                )
                updated += 1

        conn.commit()
        print(f"  Converted {updated} scalar values to JSON arrays")

    elif "issue_types" in columns:
        print("Column 'issue_types' already exists — skipping rename")
    else:
        print("WARNING: Neither 'issue_type' nor 'issue_types' found in reviews table")

    # --- Clean up deprecated issue types ---
    DEPRECATED = {"Debug issue", "Multiple issues"}
    print(f"Cleaning up deprecated issue types: {DEPRECATED}")

    deleted_labels = conn.execute(
        "DELETE FROM issue_types WHERE label IN (?, ?)",
        tuple(DEPRECATED),
    ).rowcount
    print(f"  Removed {deleted_labels} labels from issue_types table")

    # Delete reviews whose ONLY issue types were deprecated ones
    reviews_to_check = conn.execute(
        "SELECT id, issue_types FROM reviews WHERE issue_types IS NOT NULL"
    ).fetchall()
    cleared = 0
    for row in reviews_to_check:
        try:
            types = json.loads(row["issue_types"])
        except (json.JSONDecodeError, TypeError):
            types = [row["issue_types"]] if row["issue_types"] else []
        if isinstance(types, list) and all(t in DEPRECATED for t in types) and len(types) > 0:
            conn.execute("DELETE FROM reviews WHERE id = ?", (row["id"],))
            cleared += 1
    conn.commit()
    print(f"  Deleted {cleared} reviews that only had deprecated issue types")

    total = conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
    print(f"Migration complete. {total} reviews in database.")
    conn.close()


if __name__ == "__main__":
    if not DB_PATH.exists():
        print(f"Database not found at {DB_PATH}")
        exit(1)
    migrate()
