"""
queue_manager.py — Disk-backed upload queue using SQLite.

Why SQLite?
───────────
  - Survives crashes and reboots (data is written to disk, not held in RAM).
  - WAL (Write-Ahead Logging) mode allows concurrent reads and writes
    from different threads without blocking.
  - Zero setup — it's a single file, no separate server process.
  - Ships with Python (sqlite3 module) — no extra dependency.

How the queue works
───────────────────
  1. When a plate is ready for upload, we INSERT a row with status='pending'.
  2. The upload thread polls for pending rows and processes them.
  3. On success, status becomes 'completed'.
  4. On failure, retries is incremented and retry_after is set to a
     future timestamp (exponential backoff: 60 s, 120 s, 240 s, …).
  5. After max_retries the status becomes 'failed' (permanent).
  6. Old completed rows are periodically cleaned up to save disk space.

Table schema
────────────
  id            Auto-incrementing primary key
  plate         Normalised plate string (e.g. "AB12CDE")
  dateless      1 if dateless, 0 otherwise
  ts            Unix timestamp of the sighting
  image_path    Path to the saved evidence JPEG
  meta          JSON string with extra metadata (confidence, format, etc.)
  status        'pending' | 'uploading' | 'completed' | 'failed'
  retries       Number of failed upload attempts so far
  retry_after   Unix timestamp — don't retry before this time
  created       When this row was inserted
  updated       When this row was last changed
"""

import json
import logging
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class UploadQueue:
    """Persistent upload queue backed by a SQLite database file.

    Args:
        db_path: Path to the SQLite database file (created if missing).
    """

    def __init__(self, db_path: str = "./upload_queue.db"):
        self.db_path = db_path
        # A threading lock to serialise write operations
        self._lock = threading.Lock()
        self._init()

    # ------------------------------------------------------------------ #
    #  Database setup
    # ------------------------------------------------------------------ #

    def _conn(self) -> sqlite3.Connection:
        """Open a new connection to the database.

        We create a fresh connection each time because sqlite3 connections
        are NOT safe to share across threads.  Each method opens, uses,
        and closes its own connection.

        Row factory makes rows behave like dicts (row["plate"]) instead
        of tuples (row[1]).
        """
        c = sqlite3.connect(self.db_path, timeout=10)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")  # enable WAL for concurrency
        return c

    def _init(self):
        """Create the queue table and index if they don't exist yet."""
        with self._conn() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS queue (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    plate         TEXT    NOT NULL,
                    dateless      INTEGER NOT NULL DEFAULT 0,
                    ts            REAL    NOT NULL,
                    image_path    TEXT,
                    meta          TEXT,
                    status        TEXT    NOT NULL DEFAULT 'pending',
                    retries       INTEGER NOT NULL DEFAULT 0,
                    retry_after   REAL    NOT NULL DEFAULT 0,
                    created       REAL    NOT NULL,
                    updated       REAL    NOT NULL
                )
            """)
            # Index speeds up the "get pending items" query
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_q_status ON queue(status, retry_after)"
            )
            c.commit()
        logger.info("Upload queue ready: %s", self.db_path)

    # ------------------------------------------------------------------ #
    #  Enqueue / dequeue
    # ------------------------------------------------------------------ #

    def enqueue(
        self,
        plate: str,
        dateless: bool,
        ts: float,
        image_path: str,
        meta: Optional[dict] = None,
    ) -> int:
        """Add a new item to the upload queue.

        Args:
            plate:      Normalised plate string.
            dateless:   True if the plate is dateless.
            ts:         Unix timestamp of the sighting.
            image_path: Path to the saved vehicle/plate image.
            meta:       Optional extra metadata (stored as JSON).

        Returns:
            The database row ID of the new item.
        """
        now = time.time()
        with self._lock, self._conn() as c:
            cur = c.execute(
                """INSERT INTO queue
                   (plate, dateless, ts, image_path, meta, status, created, updated)
                   VALUES (?,?,?,?,?,  'pending',?,?)""",
                (plate, int(dateless), ts, image_path,
                 json.dumps(meta or {}), now, now),
            )
            c.commit()
            rid = cur.lastrowid
        logger.info("Enqueued plate=%s  id=%d", plate, rid)
        return rid

    def pending(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get pending items that are ready for upload.

        Only returns items whose retry_after timestamp has passed.

        Args:
            limit: Maximum number of items to return.

        Returns:
            List of dicts, each representing one queue row.
        """
        now = time.time()
        with self._conn() as c:
            rows = c.execute(
                """SELECT * FROM queue
                   WHERE status='pending' AND retry_after <= ?
                   ORDER BY created LIMIT ?""",
                (now, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ #
    #  Status updates
    # ------------------------------------------------------------------ #

    def set_status(self, item_id: int, status: str):
        """Change the status of a queue item.

        Args:
            item_id: Database row ID.
            status:  New status string ('uploading', 'completed', etc.).
        """
        now = time.time()
        with self._lock, self._conn() as c:
            c.execute(
                "UPDATE queue SET status=?, updated=? WHERE id=?",
                (status, now, item_id),
            )
            c.commit()

    def fail(self, item_id: int, max_retries: int = 5, delay: float = 60.0):
        """Record a failed upload attempt and schedule a retry.

        Uses exponential backoff: delay × 2^(retries-1).
        After max_retries, the item is permanently marked as 'failed'.

        Args:
            item_id:     Database row ID.
            max_retries: Give up after this many attempts.
            delay:       Base delay in seconds before first retry.
        """
        now = time.time()
        with self._lock, self._conn() as c:
            row = c.execute(
                "SELECT retries FROM queue WHERE id=?", (item_id,)
            ).fetchone()
            if row is None:
                return

            retries = row["retries"] + 1

            if retries >= max_retries:
                # Give up — mark as permanently failed
                st, ra = "failed", 0
                logger.warning(
                    "Permanently failed id=%d after %d retries", item_id, retries
                )
            else:
                # Schedule retry with exponential backoff
                st = "pending"
                ra = now + delay * (2 ** (retries - 1))
                logger.info(
                    "Retry id=%d  attempt=%d  in=%.0fs",
                    item_id, retries, ra - now,
                )

            c.execute(
                "UPDATE queue SET status=?, retries=?, retry_after=?, updated=? WHERE id=?",
                (st, retries, ra, now, item_id),
            )
            c.commit()

    # ------------------------------------------------------------------ #
    #  Maintenance
    # ------------------------------------------------------------------ #

    def cleanup(self, max_age_h: int = 24):
        """Delete completed items older than *max_age_h* hours."""
        cutoff = time.time() - max_age_h * 3600
        with self._lock, self._conn() as c:
            c.execute(
                "DELETE FROM queue WHERE status='completed' AND updated<?",
                (cutoff,),
            )
            c.commit()

    def stats(self) -> Dict[str, int]:
        """Return a {status: count} summary of the queue."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT status, COUNT(*) n FROM queue GROUP BY status"
            ).fetchall()
        return {r["status"]: r["n"] for r in rows}
