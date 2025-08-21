# # cap_store.py — convenience API over cap_db.py
# from __future__ import annotations
# import os, json, sqlite3
# from typing import List, Tuple, Dict
# import pandas as pd
# import re
# from contextlib import contextmanager
# import hashlib
# from datetime import datetime


# from cap_db import (
#     init_db as _init,
#     _conn, save_df, load_df, save_kv, load_kv, DB_PATH
# )

# def init_db():
#     _init(DB_PATH)

# def _conn():
#     os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
#     cx = sqlite3.connect(DB_PATH, check_same_thread=False)
#     cx.row_factory = sqlite3.Row
#     return cx

# # ─── bootstrap ───────────────────────────────────────────────



# def ensure_indexes() -> None:
#     """Create any missing DB indexes (safe to run repeatedly)."""
#     with _conn() as cx:
#         cx.execute("""
#             CREATE INDEX IF NOT EXISTS idx_cap_plans_vertical_subba_current
#             ON capacity_plans(vertical, sub_ba, is_current);
#         """)
#         cx.commit()

# # ─── defaults ────────────────────────────────────────────────
# def load_defaults() -> dict | None:
#     return load_kv("defaults")

# def save_defaults(cfg: dict):
#     save_kv("defaults", cfg)

# # ─── datasets save/load ──────────────────────────────────────
# def _ensure_df(x) -> pd.DataFrame:
#     return x if isinstance(x, pd.DataFrame) else pd.DataFrame()

# def load_roster() -> pd.DataFrame:
#     return _ensure_df(load_df("roster"))

# def save_roster(df: pd.DataFrame):
#     """
#     Saves roster with safe de-duplication:
#     - If a 'date' column exists (long format), drop duplicates by (BRID, date).
#     - If wide format (many YYYY-MM-DD columns), melt to long as 'roster_long'
#       and drop duplicates by (BRID, date). Also keep the wide file under 'roster'
#       for convenience.
#     - If neither is present (legacy schema), de-dupe by BRID only.
#     """
#     if df is None or df.empty:
#         save_df("roster", pd.DataFrame())
#         return

#     # tolerant column lookup
#     L = {c.lower(): c for c in df.columns}
#     brid_col = L.get("brid") or L.get("employee_id") or "BRID"

#     # --- CASE A: long schema already has a date column ---
#     if "date" in L:
#         date_col = L["date"]
#         # normalize types
#         df = df.copy()
#         df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.date.astype(str)
#         df = df.dropna(subset=[brid_col, date_col])
#         df = df.drop_duplicates(subset=[brid_col, date_col], keep="last")
#         save_df("roster", df)
#         return

#     # --- CASE B: wide schema: detect date-like columns and melt ---
#     date_like_cols = [c for c in df.columns if re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(c))]
#     if date_like_cols:
#         static_cols = [c for c in df.columns if c not in date_like_cols]
#         long = df.melt(
#             id_vars=static_cols,
#             value_vars=date_like_cols,
#             var_name="date",
#             value_name="shift"
#         )
#         # keep only non-empty shift rows
#         long = long[long["shift"].notna() & (long["shift"].astype(str).str.strip() != "")]
#         long["date"] = pd.to_datetime(long["date"], errors="coerce").dt.date.astype(str)
#         long = long.dropna(subset=[brid_col, "date"])
#         long = long.drop_duplicates(subset=[brid_col, "date"], keep="last")

#         # Store both versions: wide for user reference, long for engines
#         save_df("roster", df)            # wide
#         save_df("roster_long", long)     # normalized (BRID+date unique)
#         return

#     # --- CASE C: legacy schema (no explicit per-day dates yet) ---
#     # Fall back to unique by BRID so repeated rows don’t inflate HC.
#     save_df("roster", df.drop_duplicates(subset=[brid_col], keep="last"))

# def load_hiring() -> pd.DataFrame:
#     return _ensure_df(load_df("hiring"))

# def save_hiring(df: pd.DataFrame):
#     save_df("hiring", df if isinstance(df, pd.DataFrame) else pd.DataFrame())

# def load_shrinkage() -> pd.DataFrame:
#     return _ensure_df(load_df("shrinkage"))

# def save_shrinkage(df: pd.DataFrame):
#     save_df("shrinkage", df if isinstance(df, pd.DataFrame) else pd.DataFrame())

# def load_attrition() -> pd.DataFrame:
#     return _ensure_df(load_df("attrition"))

# def save_attrition(df: pd.DataFrame):
#     save_df("attrition", df if isinstance(df, pd.DataFrame) else pd.DataFrame())

# def load_attrition_raw() -> pd.DataFrame | None:
#     return load_df("attrition_raw")

# def save_attrition_raw(df: pd.DataFrame):
#     save_df("attrition_raw", df)

# # ─── sample templates (for download) ─────────────────────────
# def roster_template_df() -> pd.DataFrame:
#     return pd.DataFrame([
#         dict(employee_id="UK0001", name="Alex Doe", status="Active", employment_type="FT",
#              fte=1.0, contract_hours_per_week=37.5, country="UK", site="Glasgow", timezone="Europe/London",
#              program="WFM", sub_business_area="Retail", lob="Cards", channel="Back Office",
#              skill_voice=False, skill_bo=True, skill_ob=False,
#              start_date="2025-07-01", end_date=""),
#         dict(employee_id="IN0002", name="Priya Singh", status="Active", employment_type="PT",
#              fte=0.5, contract_hours_per_week=20, country="India", site="Chennai", timezone="Asia/Kolkata",
#              program="WFM", sub_business_area="Retail", lob="Cards", channel="Voice",
#              skill_voice=True, skill_bo=False, skill_ob=False,
#              start_date="2025-07-08", end_date=""),
#     ])

# def hiring_template_df() -> pd.DataFrame:
#     return pd.DataFrame([
#         dict(start_week="2025-07-07", fte=3, program="WFM", country="UK", site="Glasgow"),
#         dict(start_week="2025-07-14", fte=5, program="WFM", country="India", site="Chennai"),
#     ])

# def shrinkage_bo_template_df() -> pd.DataFrame:
#     return pd.DataFrame([
#         dict(week="2025-07-07", program="WFM", sub_business_area="Retail", lob="Cards", site="Glasgow",
#              shrinkage_pct=11.0),
#         dict(week="2025-07-14", program="WFM", sub_business_area="Retail", lob="Cards", site="Glasgow",
#              shrinkage_pct=10.7),
#     ])

# def shrinkage_voice_template_df() -> pd.DataFrame:
#     return pd.DataFrame([
#         dict(week="2025-07-07", program="WFM", queue="Inbound", site="Chennai", shrinkage_pct=12.5),
#         dict(week="2025-07-14", program="WFM", queue="Inbound", site="Chennai", shrinkage_pct=11.9),
#     ])

# def attrition_template_df() -> pd.DataFrame:
#     return pd.DataFrame([
#         dict(week="2025-07-07", program="WFM", site="Glasgow", attrition_pct=0.8),
#         dict(week="2025-07-14", program="WFM", site="Chennai", attrition_pct=1.1),
#     ])

# # ─── auto-detection from roster ──────────────────────────────
# def auto_locations_from_roster(roster: pd.DataFrame) -> List[Tuple[str, str]]:
#     if roster is None or roster.empty:
#         return []
#     cols = [c.lower() for c in roster.columns]
#     ctry_col = roster.columns[cols.index("country")] if "country" in cols else None
#     site_col = roster.columns[cols.index("site")] if "site" in cols else None
#     if not ctry_col or not site_col:
#         return []
#     pairs = roster[[ctry_col, site_col]].dropna().drop_duplicates()
#     return list(map(tuple, pairs.values.tolist()))

# def auto_hierarchy_from_roster(roster: pd.DataFrame) -> Dict[str, dict]:
#     if roster is None or roster.empty:
#         return {}
#     L = {c.lower(): c for c in roster.columns}
#     keys = dict(
#         program=L.get("program"),
#         sba=L.get("sub_business_area"),
#         lob=L.get("lob"),
#         channel=L.get("channel"),
#     )
#     if not all(keys.values()):
#         return {}
#     out: Dict[str, dict] = {}
#     for _, r in roster.dropna(subset=[keys["program"]]).iterrows():
#         p = str(r[keys["program"]])
#         s = str(r.get(keys["sba"], ""))
#         l = str(r.get(keys["lob"], ""))
#         ch = str(r.get(keys["channel"], ""))
#         out.setdefault(p, {}).setdefault(s, {}).setdefault(l, set()).add(ch)
#     for p in list(out.keys()):
#         for s in list(out[p].keys()):
#             for l in list(out[p][s].keys()):
#                 out[p][s][l] = sorted(list(out[p][s][l]))
#     return out

# def save_scoped_settings(scope_type: str, scope_key: str, d: dict):
#     assert scope_type in ("location","hier")
#     blob = json.dumps(d)
#     with _conn() as cx:
#         cx.execute("""INSERT INTO settings_scoped(scope_type, scope_key, value)
#                       VALUES (?,?,?)
#                       ON CONFLICT(scope_type,scope_key) DO UPDATE SET value=excluded.value""",
#                    (scope_type, scope_key, blob))

# def load_scoped_settings(scope_type: str, scope_key: str) -> dict:
#     with _conn() as cx:
#         row = cx.execute("""SELECT value FROM settings_scoped
#                             WHERE scope_type=? AND scope_key=?""", (scope_type, scope_key)).fetchone()
#         return json.loads(row["value"]) if row else {}

# def resolve_settings(location: str | None = None,
#                      ba: str | None = None,
#                      subba: str | None = None,
#                      lob: str | None = None) -> dict:
#     if ba and subba and lob:
#         key = f"{ba}|{subba}|{lob}"
#         s = load_scoped_settings("hier", key)
#         if s: return s
#     if location:
#         s = load_scoped_settings("location", location)
#         if s: return s
#     return load_defaults() or {}

# def get_roster_locations() -> list[str]:
#     df = load_roster()
#     if df.empty: return []
#     vals: set[str] = set()
#     for c in ["location","country","site","region"]:
#         if c in df.columns:
#             vals |= set(df[c].dropna().astype(str).str.strip().replace({"": None}).dropna().tolist())
#     return sorted(vals)

# def _hier_from_rows(rows) -> dict:
#     out: dict[str, dict] = {}
#     for r in rows:
#         ba = (r["business_area"] if isinstance(r, sqlite3.Row) else r[0]) or "Unknown"
#         hj = (r["hierarchy_json"] if isinstance(r, sqlite3.Row) else r[1]) or "{}"
#         try:
#             h = json.loads(hj)
#         except Exception:
#             h = {}
#         subs = h.get("sub_business_areas") or ["Default"]
#         lobs = h.get("channels") or ["Voice","Back Office","Outbound"]
#         out.setdefault(ba, {})
#         for s in subs:
#             out[ba][s] = list(lobs)
#     return out

# def get_clients_hierarchy() -> dict:
#     merged: dict[str, dict] = {}
#     with _conn() as cx:
#         try:
#             merged = _hier_from_rows(cx.execute("SELECT business_area, hierarchy_json FROM clients").fetchall())
#         except Exception:
#             merged = {}
#     capdb_path = DB_PATH  # use the same DB now
#     if os.path.exists(capdb_path):
#         try:
#             c2 = sqlite3.connect(capdb_path)
#             c2.row_factory = sqlite3.Row
#             rows2 = c2.execute("SELECT business_area, hierarchy_json FROM clients").fetchall()
#             other = _hier_from_rows(rows2)
#             for ba, subs in other.items():
#                 merged.setdefault(ba, {})
#                 for s, lobs in subs.items():
#                     merged[ba].setdefault(s, list(lobs))
#             c2.close()
#         except Exception:
#             pass
#     return merged

# def load_roster_wide() -> pd.DataFrame:
#     df = load_df("roster")         # wide view key we save
#     return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

# def save_roster_wide(df: pd.DataFrame):
#     save_df("roster_wide", df)

# def load_roster_long() -> pd.DataFrame:
#     df = load_df("roster_long")
#     return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

# def save_roster_long(df: pd.DataFrame):
#     save_df("roster_long", df)

# # ─── mappings & raw attrition ─────────────────────────────────────────────
# def save_mapping_sheet1(df: pd.DataFrame):
#     # Columns expected (free text ok): Business Area, Sub Business Area, LOB, Team, Site
#     save_df("mapping_sheet1", df)

# def load_mapping_sheet1() -> pd.DataFrame | None:
#     return load_df("mapping_sheet1")

# def save_mapping_sheet2(df: pd.DataFrame):
#     # Columns expected: IMH 06 (or IMH L06), Business Area Nomenclature
#     save_df("mapping_sheet2", df)

# def load_mapping_sheet2() -> pd.DataFrame | None:
#     return load_df("mapping_sheet2")

# def save_attrition_raw(df: pd.DataFrame):
#     # Persist the raw upload exactly as given
#     save_df("attrition_raw", df)

# def load_attrition_raw() -> pd.DataFrame | None:
#     return load_df("attrition_raw")

# # ─── tiny template makers to help users download a sample ─────────────────
# def mapping_sheet1_template_df() -> pd.DataFrame:
#     return pd.DataFrame([
#         {
#             "Business Area": "BFA",
#             "Sub Business Area": "BBFA",
#             "LOB": "Business",
#             "Team": "BFA - Business Complaints Team",
#             "Site": "UK",
#         },
#         {
#             "Business Area": "Collections",
#             "Sub Business Area": "Retail",
#             "LOB": "Cards",
#             "Team": "Collections - Early Stage",
#             "Site": "IN",
#         },
#     ])

# def mapping_sheet2_template_df() -> pd.DataFrame:
#     return pd.DataFrame([
#         {"IMH 06": "BFA", "Business Area Nomenclature": "BFA"},
#         {"IMH 06": "Collections", "Business Area Nomenclature": "Collections"},
#     ])



# cap_store.py — convenience API over cap_db.py
from __future__ import annotations
import os, json, sqlite3
from typing import List, Tuple, Dict
import pandas as pd
import re
from contextlib import contextmanager
import hashlib
from datetime import datetime

from cap_db import (
    init_db as _init,
    save_df, load_df, save_kv, load_kv, DB_PATH
)

# ─── connection ──────────────────────────────────────────────
def _conn():
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    cx = sqlite3.connect(DB_PATH, check_same_thread=False)
    cx.row_factory = sqlite3.Row
    return cx

# ─── bootstrap (project indexes) ─────────────────────────────
def ensure_indexes() -> None:
    """Create any missing DB indexes (safe to run repeatedly)."""
    with _conn() as cx:
        cx.execute("""
            CREATE INDEX IF NOT EXISTS idx_cap_plans_vertical_subba_current
            ON capacity_plans(vertical, sub_ba, is_current);
        """)
        cx.commit()


def migrate_capacity_plans_location_site() -> None:
    """One-time, safe migration to add location/site (and is_deleted) to capacity_plans."""
    with _conn() as cx:
        cols = {r["name"] for r in cx.execute("PRAGMA table_info(capacity_plans)")}
        if "location" not in cols:
            cx.execute("ALTER TABLE capacity_plans ADD COLUMN location TEXT")
        if "site" not in cols:
            cx.execute("ALTER TABLE capacity_plans ADD COLUMN site TEXT")
        if "is_deleted" not in cols:
            cx.execute("ALTER TABLE capacity_plans ADD COLUMN is_deleted INTEGER DEFAULT 0")
        cx.commit()

def init_db():
    _init(DB_PATH)
    migrate_capacity_plans_location_site()   # <-- add
    ensure_indexes()
                       # keep your indexes call


# ─── helpers ────────────────────────────────────────────────

def _ensure_df(x) -> pd.DataFrame:
    return x if isinstance(x, pd.DataFrame) else pd.DataFrame()

# ─── datasets save/load ──────────────────────────────────────
def load_roster() -> pd.DataFrame:
    return _ensure_df(load_df("roster"))

def save_roster(df: pd.DataFrame):
    """
    Saves roster with safe de-duplication:
    - If a 'date' column exists (long format), drop duplicates by (BRID, date).
    - If wide format (many YYYY-MM-DD columns), melt to long as 'roster_long'
      and drop duplicates by (BRID, date). Also keep the wide file under 'roster'
      for convenience.
    - If neither is present (legacy schema), de-dupe by BRID only.
    """
    if df is None or df.empty:
        save_df("roster", pd.DataFrame())
        return

    # tolerant column lookup
    L = {c.lower(): c for c in df.columns}
    brid_col = L.get("brid") or L.get("employee_id") or "BRID"

    # --- CASE A: long schema already has a date column ---
    if "date" in L:
        date_col = L["date"]
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.date.astype(str)
        df = df.dropna(subset=[brid_col, date_col])
        df = df.drop_duplicates(subset=[brid_col, date_col], keep="last")
        save_df("roster", df)
        return

    # --- CASE B: wide schema: detect date-like columns and melt ---
    date_like_cols = [c for c in df.columns if re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(c))]
    if date_like_cols:
        static_cols = [c for c in df.columns if c not in date_like_cols]
        long = df.melt(
            id_vars=static_cols,
            value_vars=date_like_cols,
            var_name="date",
            value_name="shift"
        )
        long = long[long["shift"].notna() & (long["shift"].astype(str).str.strip() != "")]
        long["date"] = pd.to_datetime(long["date"], errors="coerce").dt.date.astype(str)
        long = long.dropna(subset=[brid_col, "date"])
        long = long.drop_duplicates(subset=[brid_col, "date"], keep="last")

        # Store both versions: wide for user reference, long for engines
        save_df("roster", df)            # wide
        save_df("roster_long", long)     # normalized (BRID+date unique)
        return

    # --- CASE C: legacy schema (no explicit per-day dates yet) ---
    save_df("roster", df.drop_duplicates(subset=[brid_col], keep="last"))

def load_hiring() -> pd.DataFrame:
    return _ensure_df(load_df("hiring"))

def save_hiring(df: pd.DataFrame):
    save_df("hiring", df if isinstance(df, pd.DataFrame) else pd.DataFrame())

def load_shrinkage() -> pd.DataFrame:
    return _ensure_df(load_df("shrinkage"))

def save_shrinkage(df: pd.DataFrame):
    save_df("shrinkage", df if isinstance(df, pd.DataFrame) else pd.DataFrame())

def load_attrition() -> pd.DataFrame:
    return _ensure_df(load_df("attrition"))

def save_attrition(df: pd.DataFrame):
    save_df("attrition", df if isinstance(df, pd.DataFrame) else pd.DataFrame())

def load_attrition_raw() -> pd.DataFrame | None:
    return load_df("attrition_raw")

def save_attrition_raw(df: pd.DataFrame):
    save_df("attrition_raw", df)

# ─── sample templates (for download) ─────────────────────────
def roster_template_df() -> pd.DataFrame:
    return pd.DataFrame([
        dict(employee_id="UK0001", name="Alex Doe", status="Active", employment_type="FT",
             fte=1.0, contract_hours_per_week=37.5, country="UK", site="Glasgow", timezone="Europe/London",
             program="WFM", sub_business_area="Retail", lob="Cards", channel="Back Office",
             skill_voice=False, skill_bo=True, skill_ob=False,
             start_date="2025-07-01", end_date=""),
        dict(employee_id="IN0002", name="Priya Singh", status="Active", employment_type="PT",
             fte=0.5, contract_hours_per_week=20, country="India", site="Chennai", timezone="Asia/Kolkata",
             program="WFM", sub_business_area="Retail", lob="Cards", channel="Voice",
             skill_voice=True, skill_bo=False, skill_ob=False,
             start_date="2025-07-08", end_date=""),
    ])

def hiring_template_df() -> pd.DataFrame:
    return pd.DataFrame([
        dict(start_week="2025-07-07", fte=3, program="WFM", country="UK", site="Glasgow"),
        dict(start_week="2025-07-14", fte=5, program="WFM", country="India", site="Chennai"),
    ])

def shrinkage_bo_template_df() -> pd.DataFrame:
    return pd.DataFrame([
        dict(week="2025-07-07", program="WFM", sub_business_area="Retail", lob="Cards", site="Glasgow",
             shrinkage_pct=11.0),
        dict(week="2025-07-14", program="WFM", sub_business_area="Retail", lob="Cards", site="Glasgow",
             shrinkage_pct=10.7),
    ])

def shrinkage_voice_template_df() -> pd.DataFrame:
    return pd.DataFrame([
        dict(week="2025-07-07", program="WFM", queue="Inbound", site="Chennai", shrinkage_pct=12.5),
        dict(week="2025-07-14", program="WFM", queue="Inbound", site="Chennai", shrinkage_pct=11.9),
    ])

def attrition_template_df() -> pd.DataFrame:
    return pd.DataFrame([
        dict(week="2025-07-07", program="WFM", site="Glasgow", attrition_pct=0.8),
        dict(week="2025-07-14", program="WFM", site="Chennai", attrition_pct=1.1),
    ])

# ─── auto-detection from roster ──────────────────────────────
def auto_locations_from_roster(roster: pd.DataFrame) -> List[Tuple[str, str]]:
    if roster is None or roster.empty:
        return []
    cols = [c.lower() for c in roster.columns]
    ctry_col = roster.columns[cols.index("country")] if "country" in cols else None
    site_col = roster.columns[cols.index("site")] if "site" in cols else None
    if not ctry_col or not site_col:
        return []
    pairs = roster[[ctry_col, site_col]].dropna().drop_duplicates()
    return list(map(tuple, pairs.values.tolist()))

def auto_hierarchy_from_roster(roster: pd.DataFrame) -> Dict[str, dict]:
    if roster is None or roster.empty:
        return {}
    L = {c.lower(): c for c in roster.columns}
    keys = dict(
        program=L.get("program"),
        sba=L.get("sub_business_area"),
        lob=L.get("lob"),
        channel=L.get("channel"),
    )
    if not all(keys.values()):
        return {}
    out: Dict[str, dict] = {}
    for _, r in roster.dropna(subset=[keys["program"]]).iterrows():
        p = str(r[keys["program"]])
        s = str(r.get(keys["sba"], ""))
        l = str(r.get(keys["lob"], ""))
        ch = str(r.get(keys["channel"], ""))
        out.setdefault(p, {}).setdefault(s, {}).setdefault(l, set()).add(ch)
    for p in list(out.keys()):
        for s in list(out[p].keys()):
            for l in list(out[p][s].keys()):
                out[p][s][l] = sorted(list(out[p][s][l]))
    return out

def save_scoped_settings(scope_type: str, scope_key: str, d: dict):
    assert scope_type in ("location","hier")
    blob = json.dumps(d)
    with _conn() as cx:
        cx.execute("""INSERT INTO settings_scoped(scope_type, scope_key, value)
                      VALUES (?,?,?)
                      ON CONFLICT(scope_type,scope_key) DO UPDATE SET value=excluded.value""",
                   (scope_type, scope_key, blob))

def load_scoped_settings(scope_type: str, scope_key: str) -> dict:
    with _conn() as cx:
        row = cx.execute("""SELECT value FROM settings_scoped
                            WHERE scope_type=? AND scope_key=?""", (scope_type, scope_key)).fetchone()
        return json.loads(row["value"]) if row else {}

def resolve_settings(location: str | None = None,
                     ba: str | None = None,
                     subba: str | None = None,
                     lob: str | None = None) -> dict:
    if ba and subba and lob:
        key = f"{ba}|{subba}|{lob}"
        s = load_scoped_settings("hier", key)
        if s: return s
    if location:
        s = load_scoped_settings("location", location)
        if s: return s
    return load_defaults() or {}

def get_roster_locations() -> list[str]:
    df = load_roster()
    if df.empty: return []
    vals: set[str] = set()
    for c in ["location","country","site","region"]:
        if c in df.columns:
            vals |= set(df[c].dropna().astype(str).str.strip().replace({"": None}).dropna().tolist())
    return sorted(vals)

def _hier_from_rows(rows) -> dict:
    out: dict[str, dict] = {}
    for r in rows:
        ba = (r["business_area"] if isinstance(r, sqlite3.Row) else r[0]) or "Unknown"
        hj = (r["hierarchy_json"] if isinstance(r, sqlite3.Row) else r[1]) or "{}"
        try:
            h = json.loads(hj)
        except Exception:
            h = {}
        subs = h.get("sub_business_areas") or ["Default"]
        lobs = h.get("channels") or ["Voice","Back Office","Outbound"]
        out.setdefault(ba, {})
        for s in subs:
            out[ba][s] = list(lobs)
    return out

def get_clients_hierarchy() -> dict:
    merged: dict[str, dict] = {}
    with _conn() as cx:
        try:
            merged = _hier_from_rows(cx.execute("SELECT business_area, hierarchy_json FROM clients").fetchall())
        except Exception:
            merged = {}
    capdb_path = DB_PATH  # use the same DB now
    if os.path.exists(capdb_path):
        try:
            c2 = sqlite3.connect(capdb_path)
            c2.row_factory = sqlite3.Row
            rows2 = c2.execute("SELECT business_area, hierarchy_json FROM clients").fetchall()
            other = _hier_from_rows(rows2)
            for ba, subs in other.items():
                merged.setdefault(ba, {})
                for s, lobs in subs.items():
                    merged[ba].setdefault(s, list(lobs))
            c2.close()
        except Exception:
            pass
    return merged

def load_roster_wide() -> pd.DataFrame:
    df = load_df("roster")         # wide view key we save
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

def save_roster_wide(df: pd.DataFrame):
    save_df("roster_wide", df)

def load_roster_long() -> pd.DataFrame:
    df = load_df("roster_long")
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

def save_roster_long(df: pd.DataFrame):
    save_df("roster_long", df)

# ─────────────────────────────────────────────────────────────
#                    M A P P I N G   S T O R E
#       (append-only, deduped rows + file ledger for uploads)
# ─────────────────────────────────────────────────────────────

def _norm(s: str | None) -> str:
    return (s or "").strip().lower()

def _colpick(df, *names):
    low = {c.lower(): c for c in df.columns}
    for n in names:
        c = low.get(n.lower())
        if c: return c
    return None

def _sha256_of_df(df: pd.DataFrame) -> str:
    csv_bytes = df.to_csv(index=False).encode("utf-8", errors="ignore")
    return hashlib.sha256(csv_bytes).hexdigest()

def ensure_mapping_tables():
    with _conn() as cx:
        # File ledger
        cx.execute("""
            CREATE TABLE IF NOT EXISTS mapping_files (
              id INTEGER PRIMARY KEY,
              kind TEXT CHECK (kind IN ('map1','map2')) NOT NULL,
              filename TEXT,
              sha256 TEXT UNIQUE,
              uploaded_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Rows for Mapping 1 (dedupe by BA,SBA,Channel,Location,Site)
        cx.execute("""
            CREATE TABLE IF NOT EXISTS map1_rows (
              id INTEGER PRIMARY KEY,
              file_id INTEGER NOT NULL REFERENCES mapping_files(id) ON DELETE CASCADE,
              business_area TEXT,
              sub_business_area TEXT,
              channel TEXT,
              location TEXT,
              site TEXT,
              ba_norm TEXT,
              sba_norm TEXT,
              ch_norm TEXT,
              loc_norm TEXT,
              site_norm TEXT,
              UNIQUE (ba_norm, sba_norm, ch_norm, loc_norm, site_norm) ON CONFLICT IGNORE
            );
        """)

        # Rows for Mapping 2 (dedupe by BA)
        cx.execute("""
            CREATE TABLE IF NOT EXISTS map2_rows (
              id INTEGER PRIMARY KEY,
              file_id INTEGER NOT NULL REFERENCES mapping_files(id) ON DELETE CASCADE,
              business_area TEXT,
              ba_norm TEXT UNIQUE ON CONFLICT IGNORE
            );
        """)

        cx.execute("CREATE INDEX IF NOT EXISTS idx_map1_ba_sba ON map1_rows(ba_norm, sba_norm);")
        cx.execute("CREATE INDEX IF NOT EXISTS idx_map1_site ON map1_rows(site_norm);")
        cx.execute("CREATE INDEX IF NOT EXISTS idx_map2_ba ON map2_rows(ba_norm);")
        cx.commit()

# Ensure tables exist as soon as module is imported
ensure_mapping_tables()

# ---- SAVE: Mapping 1 (append + dedupe)
def save_mapping_sheet1(df: pd.DataFrame, filename: str | None = None) -> dict:
    """
    Append rows from Mapping Sheet 1 (union across uploads).
    Dedupe key: (BA, SBA, Channel, Location, Site) normalized.
    Accepts varying column names.
    """
    if df is None or df.empty:
        # keep the original API shape
        save_df("mapping_sheet1", pd.DataFrame())
        return {"inserted": 0, "skipped": 0}

    ba_c  = _colpick(df, "Business Area", "Program", "BA")
    sba_c = _colpick(df, "Sub Business Area", "LOB", "Sub BA")
    ch_c  = _colpick(df, "Channel", "Channels")
    loc_c = _colpick(df, "Location", "Country")
    site_c= _colpick(df, "Site", "Location Site", "Site/Location")

    sha = _sha256_of_df(df)
    with _conn() as cx:
        cx.execute("""
            INSERT OR IGNORE INTO mapping_files(kind, filename, sha256, uploaded_at)
            VALUES('map1', ?, ?, ?)
        """, (filename or "mapping1.csv", sha, datetime.utcnow().isoformat(timespec="seconds")))
        file_id = cx.execute("SELECT id FROM mapping_files WHERE sha256=?", (sha,)).fetchone()["id"]

        inserted = skipped = 0
        for _, r in df.iterrows():
            BA   = str(r.get(ba_c,  "") or "").strip()
            SBA  = str(r.get(sba_c, "") or "").strip()
            CH   = str(r.get(ch_c,  "") or "").strip()
            LOC  = str(r.get(loc_c, "") or "").strip()
            SITE = str(r.get(site_c,"") or "").strip()
            try:
                cx.execute("""
                    INSERT INTO map1_rows(file_id, business_area, sub_business_area, channel, location, site,
                                          ba_norm, sba_norm, ch_norm, loc_norm, site_norm)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?)
                """, (file_id, BA, SBA, CH, LOC, SITE,
                      _norm(BA), _norm(SBA), _norm(CH), _norm(LOC), _norm(SITE)))
                inserted += 1
            except Exception:
                skipped += 1
        cx.commit()

    # keep a snapshot for any legacy readers (won't be used by our new loaders)
    save_df("mapping_sheet1", df)
    return {"inserted": inserted, "skipped": skipped, "file_id": file_id}

# ---- LOAD: Mapping 1 (union)
def load_mapping_sheet1() -> pd.DataFrame | None:
    with _conn() as cx:
        cur = cx.execute("""
            SELECT DISTINCT
                business_area   AS "Business Area",
                sub_business_area AS "Sub Business Area",
                channel         AS "Channel",
                location        AS "Location",
                site            AS "Site"
            FROM map1_rows
            WHERE business_area IS NOT NULL AND business_area <> ''
        """)
        rows = [dict(r) for r in cur.fetchall()]
    if not rows:
        return pd.DataFrame(columns=["Business Area","Sub Business Area","Channel","Location","Site"])
    return pd.DataFrame(rows)

# ---- SAVE: Mapping 2 (append + dedupe)
def save_mapping_sheet2(df: pd.DataFrame, filename: str | None = None) -> dict:
    """
    Append rows from Mapping Sheet 2 (union across uploads).
    Dedupe key: Business Area (normalized).
    """
    if df is None or df.empty:
        save_df("mapping_sheet2", pd.DataFrame())
        return {"inserted": 0, "skipped": 0}

    ba_c = _colpick(df, "Business Area Nomenclature", "BA Nomenclature", "Business Area", "Program", "BA")
    sha = _sha256_of_df(df)

    with _conn() as cx:
        cx.execute("""
            INSERT OR IGNORE INTO mapping_files(kind, filename, sha256, uploaded_at)
            VALUES('map2', ?, ?, ?)
        """, (filename or "mapping2.csv", sha, datetime.utcnow().isoformat(timespec="seconds")))
        file_id = cx.execute("SELECT id FROM mapping_files WHERE sha256=?", (sha,)).fetchone()["id"]

        inserted = skipped = 0
        for _, r in df.iterrows():
            BA = str(r.get(ba_c, "") or "").strip()
            try:
                cx.execute("""
                    INSERT INTO map2_rows(file_id, business_area, ba_norm)
                    VALUES(?,?,?)
                """, (file_id, BA, _norm(BA)))
                inserted += 1
            except Exception:
                skipped += 1
        cx.commit()

    save_df("mapping_sheet2", df)
    return {"inserted": inserted, "skipped": skipped, "file_id": file_id}

# ---- LOAD: Mapping 2 (union)
def load_mapping_sheet2() -> pd.DataFrame | None:
    with _conn() as cx:
        cur = cx.execute("""
            SELECT DISTINCT business_area AS "Business Area"
            FROM map2_rows
            WHERE business_area IS NOT NULL AND business_area <> ''
            ORDER BY 1
        """)
        rows = [dict(r) for r in cur.fetchall()]
    if not rows:
        return pd.DataFrame(columns=["Business Area"])
    return pd.DataFrame(rows)

# ─── defaults & misc KV ──────────────────────────────────────
def load_defaults() -> dict | None:
    return load_kv("defaults")

def save_defaults(cfg: dict):
    save_kv("defaults", cfg)

# ─── tiny template makers (unchanged) ────────────────────────
def mapping_sheet1_template_df() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Business Area": "BFA",
            "Sub Business Area": "BBFA",
            "LOB": "Business",
            "Team": "BFA - Business Complaints Team",
            "Site": "UK",
        },
        {
            "Business Area": "Collections",
            "Sub Business Area": "Retail",
            "LOB": "Cards",
            "Team": "Collections - Early Stage",
            "Site": "IN",
        },
    ])

def mapping_sheet2_template_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"IMH 06": "BFA", "Business Area Nomenclature": "BFA"},
        {"IMH 06": "Collections", "Business Area Nomenclature": "Collections"},
    ])

import json

def _load_clients_hier(ba: str):
    with _conn() as cx:
        row = cx.execute("SELECT hierarchy_json FROM clients WHERE business_area=?", (ba,)).fetchone()
        if not row:
            return None, None
        try:
            h = json.loads(row["hierarchy_json"] or "{}")
        except Exception:
            h = {}
        return h, cx

def rename_site_for_ba(ba: str, old_site: str, new_site: str) -> tuple[bool, str]:
    """
    Rename a site inside clients.hierarchy_json for a BA and update any capacity_plans rows.
    Returns (ok, message).
    """
    h, cx = _load_clients_hier(ba)
    if h is None:
        return False, f"Business Area '{ba}' not found in clients."

    # fix sites in clients.hierarchy_json
    sites = [s.strip() for s in (h.get("sites") or []) if str(s).strip()]
    if old_site not in sites and new_site not in sites:
        # if neither exists, just add the new one
        sites.append(new_site)
    else:
        sites = [new_site if s == old_site else s for s in sites]
        if new_site not in sites:
            sites.append(new_site)
        # optionally drop duplicate if both were present
        sites = sorted(set(sites))

    if sites:
        h["sites"] = sites
    else:
        h.pop("sites", None)

    cx.execute("UPDATE clients SET hierarchy_json=? WHERE business_area=?",
               (json.dumps(h), ba))

    # update any existing plans that used the old site
    cx.execute("""
        UPDATE capacity_plans
           SET site=?
         WHERE vertical=? AND COALESCE(site,'')=?
    """, (new_site, ba, old_site))

    cx.commit()
    return True, f"Renamed site for BA '{ba}': '{old_site}' → '{new_site}'."

def remove_site_for_ba(ba: str, site: str) -> tuple[bool, str]:
    """
    Remove a site from the BA's clients.hierarchy_json (does NOT touch plans).
    """
    h, cx = _load_clients_hier(ba)
    if h is None:
        return False, f"Business Area '{ba}' not found in clients."

    sites = [s for s in (h.get("sites") or []) if str(s).strip() and s != site]
    if sites:
        h["sites"] = sorted(set(sites))
    else:
        h.pop("sites", None)

    cx.execute("UPDATE clients SET hierarchy_json=? WHERE business_area=?",
               (json.dumps(h), ba))
    cx.commit()
    return True, f"Removed site '{site}' from BA '{ba}'."
