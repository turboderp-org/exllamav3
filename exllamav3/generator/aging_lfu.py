from __future__ import annotations
import os
import sqlite3
import threading
import time
from typing import Iterable


class PrefixWeightStore:
    """
    Persistent sidecar for the page table's aging-LFU weights.

    The page table keys prompt-cache pages by a chained content hash, so a weight can outlive the process that
    credited it: this store keeps (phash, weight, last_hit) in SQLite and reloads it when the model is loaded
    again. Prefix token IDs are kept separately, so a restart can re-prefill the same token sequence and attach
    the old weights to the pages it recreates. They come from two sources: prefixes named in the server config,
    and hot chains the page table captured on a cache hit (see PageTable._capture_prefix).

    Writes are batched per allocation, not per token. The connection is opened once and closed by
    PageTable.close(); WAL mode keeps the hot path from serializing on the journal.
    """

    def __init__(self, directory: str):
        self.directory = os.path.expanduser(directory)
        os.makedirs(self.directory, exist_ok = True)
        self.path = os.path.join(self.directory, "prefix_weights.sqlite3")
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread = False, timeout = 5.0)
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = NORMAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS page_weights ("
            "phash BLOB PRIMARY KEY, weight REAL NOT NULL, last_hit REAL NOT NULL)"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS prefixes ("
            "label TEXT PRIMARY KEY, token_ids BLOB NOT NULL, updated REAL NOT NULL)"
        )
        self._conn.commit()

    def load_weights(self) -> dict[bytes, tuple[float, float]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT phash, weight, last_hit FROM page_weights"
            ).fetchall()
        return {bytes(h): (float(w), float(t)) for h, w, t in rows}

    def put_weights(self, items: Iterable[tuple[bytes, float, float]]):
        with self._lock:
            self._conn.executemany(
                "INSERT INTO page_weights (phash, weight, last_hit) VALUES (?, ?, ?) "
                "ON CONFLICT(phash) DO UPDATE SET "
                "weight = excluded.weight, last_hit = excluded.last_hit",
                items,
            )
            self._conn.commit()

    def put_prefix(self, label: str, token_ids: bytes):
        with self._lock:
            self._conn.execute(
                "INSERT INTO prefixes (label, token_ids, updated) VALUES (?, ?, ?) "
                "ON CONFLICT(label) DO UPDATE SET "
                "token_ids = excluded.token_ids, updated = excluded.updated",
                (label, token_ids, time.time()),
            )
            self._conn.commit()

    def load_prefixes(self) -> list[tuple[str, bytes]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT label, token_ids FROM prefixes ORDER BY updated DESC"
            ).fetchall()
        return [(label, bytes(ids)) for label, ids in rows]

    def remove_prefix(self, label: str):
        with self._lock:
            self._conn.execute("DELETE FROM prefixes WHERE label = ?", (label,))
            self._conn.commit()

    def prune_prefixes(self, max_entries: int):
        """
        Keep the max_entries most recently written prefixes, dropping the rest. Bounds the number of chains a
        restart has to replay, and hence how long the warm-up prefill can take.
        """
        if max_entries < 0:
            return
        with self._lock:
            self._conn.execute(
                "DELETE FROM prefixes WHERE label NOT IN "
                "(SELECT label FROM prefixes ORDER BY updated DESC LIMIT ?)",
                (max_entries,),
            )
            self._conn.commit()

    def close(self):
        with self._lock:
            self._conn.close()
