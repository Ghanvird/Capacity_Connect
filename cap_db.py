# cap_db.py — single SQLite backend for the whole app
from __future__ import annotations
import os, sqlite3, json, datetime as dt
import pandas as pd
from typing import Optional

DB_PATH = os.path.join(os.path.dirname(__file__), "capability.sqlite3")

def _conn():
    cx = sqlite3.connect(DB_PATH, check_same_thread=False)
    cx.row_factory = sqlite3.Row
    return cx

def init_db(path: str | None = None):
    global DB_PATH
    if path:
        DB_PATH = path
    with _conn() as cx:
        cx.executescript("""
        PRAGMA journal_mode=WAL;

        CREATE TABLE IF NOT EXISTS datasets (
            name TEXT PRIMARY KEY,
            csv  TEXT,
            updated_at TEXT
        );

        CREATE TABLE IF NOT EXISTS kv (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TEXT
        );

        -- per-scope settings (location or BA|SubBA|LOB)
        CREATE TABLE IF NOT EXISTS settings_scoped (
            scope_type TEXT NOT NULL,        -- 'location' | 'hier'
            scope_key  TEXT NOT NULL,        -- e.g. 'India' or 'BA|SubBA|LOB'
            value      TEXT NOT NULL,        -- JSON
            updated_at TEXT NOT NULL,
            PRIMARY KEY(scope_type, scope_key)
        );

        -- optional: planning clients (used for hierarchy source)
        CREATE TABLE IF NOT EXISTS clients (
            business_area  TEXT PRIMARY KEY,
            hierarchy_json TEXT               -- JSON with sub_business_areas/channels/etc
        );
        """)
        cx.commit()

def save_df(name: str, df: pd.DataFrame):
    csv = df.to_csv(index=False)
    ts  = dt.datetime.utcnow().isoformat()
    with _conn() as cx:
        cx.execute(
            "REPLACE INTO datasets(name,csv,updated_at) VALUES(?,?,?)",
            (name, csv, ts)
        )
        cx.commit()

def load_df(name: str) -> Optional[pd.DataFrame]:
    with _conn() as cx:
        row = cx.execute("SELECT csv FROM datasets WHERE name=?", (name,)).fetchone()
    if not row:
        return pd.DataFrame()
    import io
    return pd.read_csv(io.StringIO(row["csv"]))

def save_kv(key: str, obj):
    ts  = dt.datetime.utcnow().isoformat()
    js  = json.dumps(obj)
    with _conn() as cx:
        cx.execute("REPLACE INTO kv(key,value,updated_at) VALUES(?,?,?)", (key, js, ts))
        cx.commit()

def load_kv(key: str):
    with _conn() as cx:
        row = cx.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
    return (json.loads(row["value"]) if row else None)
