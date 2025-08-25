# cap_store.py — convenience API over cap_db.py
from __future__ import annotations

import os
import re
import json
import hashlib
import sqlite3
from typing import List, Tuple, Dict
from datetime import datetime, timezone

import pandas as pd

from cap_db import (
    init_db as _init,
    save_df, load_df, save_kv, load_kv, DB_PATH
)


# ─────────────────────────────────────────────────────────────
# Init / Connection
# ─────────────────────────────────────────────────────────────

def _conn():
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    cx = sqlite3.connect(DB_PATH, check_same_thread=False)
    cx.row_factory = sqlite3.Row
    return cx


def ensure_indexes() -> None:
    """Create any missing DB indexes (safe to run repeatedly)."""
    with _conn() as cx:
        cx.execute("""
            CREATE INDEX IF NOT EXISTS idx_cap_plans_vertical_subba_current
            ON capacity_plans(vertical, sub_ba, is_current);
        """)
        cx.commit()


def migrate_capacity_plans_location_site() -> None:
    """Safe migration to add location/site/is_deleted to capacity_plans."""
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
    ensure_headcount_table()
    ensure_mapping_tables()
    migrate_capacity_plans_location_site()
    ensure_indexes()


# ─────────────────────────────────────────────────────────────
# Headcount (BRID ⇄ Manager) store
# ─────────────────────────────────────────────────────────────

# --- add these helpers near other utils in cap_store.py ---
def _canon(s: str | None) -> str:
    return (s or "").strip().lower()

def _canon_hier_key(scope_key: str) -> str:
    parts = (scope_key or "").split("|")
    ba   = _canon(parts[0] if len(parts) > 0 else "")
    sub  = _canon(parts[1] if len(parts) > 1 else "")
    lob  = _canon(parts[2] if len(parts) > 2 else "")
    return f"{ba}|{sub}|{lob}"

def _normalize_settings_dict(d: dict | None) -> dict:
    """
    Make settings robust to different field labels:
    - Map many possible keys to canonical: occupancy, target_aht, budgeted_aht, target_sut, budgeted_sut
    - Coerce numbers; for occupancy accept 80, '80', '80%', 0.8, '0.8'
    """
    if not isinstance(d, dict):
        return {}
    low = {str(k).strip().lower(): v for k, v in d.items()}

    def pick(*names):
        for n in names:
            if n in low and low[n] not in (None, ""):
                return low[n]
        return None

    def num(x):
        try:
            s = str(x).replace(",", "").strip()
            if s.endswith("%"):
                return float(s[:-1].strip())
            return float(s)
        except Exception:
            return None

    def pct_to_fraction(x):
        """Return fraction in 0..1 if we can; else None."""
        v = num(x)
        if v is None:
            return None
        if v > 1.0:   # 80 -> 0.8
            return v / 100.0
        return v      # already fractional like 0.8

    out = dict(low)

    # --- Occupancy
    occ_raw = pick("occupancy", "occupancy_pct", "occupancy percent", "occupancy%", "occupancy (%)",
                   "occ", "target_occupancy", "target occupancy", "budgeted_occupancy", "budgeted occupancy")
    occ_frac = pct_to_fraction(occ_raw) if occ_raw is not None else None
    if occ_frac is not None:
        out["occupancy"] = occ_frac                 # fraction 0..1
        out["occupancy_pct"] = round(occ_frac*100)  # percent 0..100

    # --- AHT/SUT canonicals
    canon_map = {
        "target_aht":   ["target_aht", "target aht", "voice_target_aht", "aht_target"],
        "budgeted_aht": ["budgeted_aht", "budgeted aht", "voice_budgeted_aht", "aht_budgeted"],
        "target_sut":   ["target_sut", "target sut", "bo_target_sut", "target_sut_sec", "sut_target"],
        "budgeted_sut": ["budgeted_sut", "budgeted sut", "bo_budgeted_sut", "budgeted_sut_sec", "sut_budgeted"],
    }
    for canon, alts in canon_map.items():
        v = pick(*alts)
        v = num(v) if v is not None else None
        if v is not None:
            out[canon] = v

    return out


HC_CANON_COLS = [
    "level_0", "level_1", "level_2", "level_3", "level_4", "level_5", "level_6",
    "brid", "full_name", "position_description", "hc_operational_status",
    "employee_group_description", "corporate_grade_description",
    "line_manager_brid", "line_manager_full_name",
    "current_org_unit", "current_org_unit_description",
    "position_location_country", "position_location_city", "position_location_building_description",
    "ccid", "cc_name", "journey", "position_group"
]


def ensure_headcount_table():
    with _conn() as cx:
        cx.execute("""
        CREATE TABLE IF NOT EXISTS headcount (
            level_0 TEXT, level_1 TEXT, level_2 TEXT, level_3 TEXT, level_4 TEXT, level_5 TEXT, level_6 TEXT,
            brid TEXT PRIMARY KEY,
            full_name TEXT,
            position_description TEXT,
            hc_operational_status TEXT,
            employee_group_description TEXT,
            corporate_grade_description TEXT,
            line_manager_brid TEXT,
            line_manager_full_name TEXT,
            current_org_unit TEXT,
            current_org_unit_description TEXT,
            position_location_country TEXT,
            position_location_city TEXT,
            position_location_building_description TEXT,
            ccid TEXT,
            cc_name TEXT,
            journey TEXT,
            position_group TEXT,
            updated_at TEXT
        )
        """)
        cx.execute("CREATE INDEX IF NOT EXISTS idx_headcount_lmbrid ON headcount(line_manager_brid)")
        cx.execute("CREATE INDEX IF NOT EXISTS idx_headcount_org ON headcount(current_org_unit)")
        cx.commit()


def _normalize_headcount_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame(columns=HC_CANON_COLS)

    L = {str(c).strip().lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            c = L.get(n.lower())
            if c:
                return c
        return None

    rename = {}
    mapping = {
        "level_0": ["level 0"],
        "level_1": ["level 1"],
        "level_2": ["level 2"],
        "level_3": ["level 3"],
        "level_4": ["level 4"],
        "level_5": ["level 5"],
        "level_6": ["level 6"],
        "brid": ["brid", "employee id", "employee number"],
        "full_name": ["full name", "employee name", "name"],
        "position_description": ["position description", "position"],
        "hc_operational_status": ["headcount operational status description", "operational status"],
        "employee_group_description": ["employee group description", "employee group"],
        "corporate_grade_description": ["corporate grade description", "grade"],
        "line_manager_brid": ["line manager brid", "manager brid", "tl brid"],
        "line_manager_full_name": ["line manager full name", "manager name", "tl name", "team manager"],
        "current_org_unit": ["current organisation unit", "current organization unit", "org unit"],
        "current_org_unit_description": ["current organisation unit description", "current organization unit description", "org unit description"],
        "position_location_country": ["position location country", "country"],
        "position_location_city": ["position location city", "city"],
        "position_location_building_description": ["position location building description", "building"],
        "ccid": ["ccid"],
        "cc_name": ["cc name"],
        "journey": ["journey"],
        "position_group": ["position group", "group"],
    }
    for canon, names in mapping.items():
        src = pick(*names)
        if src:
            rename[src] = canon

    dff = df.rename(columns=rename)
    keep = [c for c in HC_CANON_COLS if c in dff.columns]
    dff = dff[keep].copy()

    for c in dff.columns:
        if dff[c].dtype == object:
            dff[c] = dff[c].astype(str).str.strip()

    if "brid" in dff.columns:
        dff["brid"] = dff["brid"].astype(str).str.strip()
        dff = dff.drop_duplicates(subset=["brid"], keep="last")

    return dff


# keep ensure_headcount_table() as-is (it creates headcount with PRIMARY KEY (brid))

def save_headcount_df(df: pd.DataFrame) -> int:
    """
    Append/Upsert headcount:
      - Does NOT truncate the table anymore.
      - Upserts by PRIMARY KEY (brid). New BRIDs are inserted; existing BRIDs are updated.
      - Dedup inside the upload by BRID (last row wins).
    """
    dff = _normalize_headcount_df(df)

    # 1) Dedup within this upload by BRID (last wins)
    if "brid" in dff.columns:
        dff["brid"] = dff["brid"].astype(str).str.strip()
        dff = dff.drop_duplicates(subset=["brid"], keep="last")

    ensure_headcount_table()
    ts = datetime.now(timezone.utc).isoformat()

    # Make sure all expected columns exist in the frame
    all_cols = HC_CANON_COLS + ["updated_at"]
    for c in HC_CANON_COLS:
        if c not in dff.columns:
            dff[c] = None
    dff["updated_at"] = ts

    # 2) Upsert row-by-row (keeps prior uploads; only updates duplicates by BRID)
    col_list = HC_CANON_COLS + ["updated_at"]
    placeholders = ",".join(["?"] * len(col_list))
    assign_sql = ",".join([f"{c}=excluded.{c}" for c in col_list if c != "brid"])

    with _conn() as cx:
        # helpful indexes (idempotent)
        cx.execute("CREATE INDEX IF NOT EXISTS idx_headcount_lmbrid ON headcount(line_manager_brid)")
        cx.execute("CREATE INDEX IF NOT EXISTS idx_headcount_org ON headcount(current_org_unit)")

        stmt = f"""
        INSERT INTO headcount({",".join(col_list)})
        VALUES ({placeholders})
        ON CONFLICT(brid) DO UPDATE SET {assign_sql}
        """
        vals = []
        for _, r in dff.iterrows():
            row_vals = [r.get(c, None) for c in HC_CANON_COLS] + [ts]
            vals.append(row_vals)

        cx.executemany(stmt, vals)
        cx.commit()

    # Optional snapshot (last upload only)
    save_df("headcount_raw", dff[all_cols])
    return int(len(dff))



def load_headcount(limit: int | None = None) -> pd.DataFrame:
    with _conn() as cx:
        q = "SELECT * FROM headcount"
        return pd.read_sql_query(q + (" LIMIT ?" if limit else ""), cx, params=[int(limit)] if limit else None)


def brid_manager_map() -> pd.DataFrame:
    with _conn() as cx:
        try:
            return pd.read_sql_query(
                "SELECT brid, line_manager_brid, line_manager_full_name FROM headcount", cx
            )
        except Exception:
            return pd.DataFrame(columns=["brid", "line_manager_brid", "line_manager_full_name"])

# ─── Journey lookups from Headcount (Level 2 → Journey; dependent Journey → Sites) ───

def level2_to_journey_map(pretty: bool = False) -> pd.DataFrame:
    """
    Returns a mapping between Level 2 and Journey derived from headcount.
    - If pretty=True, columns are ['Level 2','Journey'] (for UI tables).
    - Else columns are ['level_2','journey'].
    Dedupes on Level 2 (keeps first non-null Journey seen).
    """
    with _conn() as cx:
        try:
            df = pd.read_sql_query(
                """
                SELECT DISTINCT
                    COALESCE(level_2,'')       AS level_2,
                    COALESCE(journey,'')       AS journey
                FROM headcount
                WHERE COALESCE(level_2,'') <> ''
                """,
                cx,
            )
        except Exception:
            df = pd.DataFrame(columns=["level_2", "journey"])

    if not df.empty:
        df["level_2"] = df["level_2"].astype(str).str.strip()
        df["journey"] = df["journey"].astype(str).str.strip()
        df = df[df["level_2"] != ""]
        # one row per Level 2
        df = df.sort_values(["level_2", "journey"]).drop_duplicates(subset=["level_2"], keep="first")

    if pretty:
        return df.rename(columns={"level_2": "Level 2", "journey": "Journey"})
    return df


def unique_journeys() -> list[str]:
    """Distinct Journey values from headcount, sorted (empty removed)."""
    with _conn() as cx:
        try:
            df = pd.read_sql_query(
                "SELECT DISTINCT COALESCE(journey,'') AS journey FROM headcount", cx
            )
        except Exception:
            return []
    if df.empty:
        return []
    s = df["journey"].astype(str).str.strip()
    return sorted([x for x in s.unique().tolist() if x])


def journeys_sites_from_headcount() -> dict[str, list[str]]:
    """
    Returns { Journey: [Sites...] } where Sites come from
    'position_location_building_description' in headcount.
    """
    with _conn() as cx:
        try:
            df = pd.read_sql_query(
                """
                SELECT
                    COALESCE(journey,'') AS journey,
                    COALESCE(position_location_building_description,'') AS site
                FROM headcount
                """,
                cx,
            )
        except Exception:
            return {}

    if df.empty:
        return {}

    df["journey"] = df["journey"].astype(str).str.strip()
    df["site"] = df["site"].astype(str).str.strip()
    df = df[(df["journey"] != "") & (df["site"] != "")]
    out: dict[str, list[str]] = {}
    for j, grp in df.groupby("journey"):
        sites = sorted(grp["site"].dropna().astype(str).str.strip().unique().tolist())
        out[j] = sites
    return out


def sites_for_journey(journey: str) -> list[str]:
    """Convenience: list of sites for a given Journey (case-insensitive match)."""
    if not journey:
        return []
    jnorm = str(journey).strip().lower()
    with _conn() as cx:
        try:
            df = pd.read_sql_query(
                """
                SELECT
                    COALESCE(journey,'') AS journey,
                    COALESCE(position_location_building_description,'') AS site
                FROM headcount
                WHERE COALESCE(journey,'') <> ''
                """,
                cx,
            )
        except Exception:
            return []
    if df.empty:
        return []
    df["journey"] = df["journey"].astype(str).str.strip()
    df["site"] = df["site"].astype(str).str.strip()
    df = df[(df["journey"].str.lower() == jnorm) & (df["site"] != "")]
    return sorted(df["site"].unique().tolist())


# Optional helpers to tweak BA sites in clients.hierarchy_json + plans
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
    h, cx = _load_clients_hier(ba)
    if h is None:
        return False, f"Business Area '{ba}' not found in clients."

    sites = [s.strip() for s in (h.get("sites") or []) if str(s).strip()]
    if old_site not in sites and new_site not in sites:
        sites.append(new_site)
    else:
        sites = [new_site if s == old_site else s for s in sites]
        if new_site not in sites:
            sites.append(new_site)
        sites = sorted(set(sites))

    if sites:
        h["sites"] = sites
    else:
        h.pop("sites", None)

    cx.execute("UPDATE clients SET hierarchy_json=? WHERE business_area=?", (json.dumps(h), ba))
    cx.execute("""
        UPDATE capacity_plans SET site=? WHERE vertical=? AND COALESCE(site,'')=?
    """, (new_site, ba, old_site))
    cx.commit()
    return True, f"Renamed site for BA '{ba}': '{old_site}' → '{new_site}'."


def remove_site_for_ba(ba: str, site: str) -> tuple[bool, str]:
    h, cx = _load_clients_hier(ba)
    if h is None:
        return False, f"Business Area '{ba}' not found in clients."

    sites = [s for s in (h.get("sites") or []) if str(s).strip() and s != site]
    if sites:
        h["sites"] = sorted(set(sites))
    else:
        h.pop("sites", None)

    cx.execute("UPDATE clients SET hierarchy_json=? WHERE business_area=?", (json.dumps(h), ba))
    cx.commit()
    return True, f"Removed site '{site}' from BA '{ba}'."


# ─────────────────────────────────────────────────────────────
# Timeseries store by scope (BA|SBA|LOB)
# ─────────────────────────────────────────────────────────────

def _ensure_df(x) -> pd.DataFrame:
    return x if isinstance(x, pd.DataFrame) else pd.DataFrame()

def _canon_scope_key(sk: str) -> str:
    """
    Normalize scope keys ('BA|SubBA|Channel') for storage and lookup.
    We lower-case and strip whitespace to avoid case/space mismatches.
    """
    return str(sk or "").strip().lower()
    


def save_timeseries(kind: str, scope_key: str, df: pd.DataFrame):
    """
    kind ∈ {
      'voice_forecast_volume','voice_actual_volume','voice_forecast_aht','voice_actual_aht',
      'bo_forecast_volume','bo_actual_volume','bo_forecast_sut','bo_actual_sut',
      'voice_tactical_volume','voice_tactical_aht','bo_tactical_volume','bo_tactical_sut'
    }
    """
    sk = _canon_scope_key(scope_key)
    save_df(f"{kind}::{sk}", df if isinstance(df, pd.DataFrame) else pd.DataFrame())


def load_timeseries(kind: str, scope_key: str) -> pd.DataFrame:
    """
    Load by canonical scope key; if not found, attempt case-insensitive recovery
    and migrate to canonical name.
    """
    sk = _canon_scope_key(scope_key)
    df = _ensure_df(load_df(f"{kind}::{sk}"))
    if not isinstance(df, pd.DataFrame) or df.empty:
        # Try to find any saved dataset whose normalized scope matches
        with _conn() as cx:
            rows = cx.execute("SELECT name FROM datasets WHERE name LIKE ?", (f"{kind}::%",)).fetchall()
        for r in rows:
            name = (r["name"] if isinstance(r, dict) else r[0]) if r else ""
            if not name or "::" not in name:
                continue
            _, raw_sk = name.split("::", 1)
            if _canon_scope_key(raw_sk) == sk:
                # migrate to canonical key
                tmp = _ensure_df(load_df(name))
                if not tmp.empty:
                    save_df(f"{kind}::{sk}", tmp)
                    return tmp
    return df


def load_timeseries_any(kind: str, scopes: list[str]) -> pd.DataFrame:
    frames = []
    for sk in scopes or []:
        d = load_timeseries(kind, _canon_scope_key(sk))
        if isinstance(d, pd.DataFrame) and not d.empty:
            d = d.copy()
            d["scope_key"] = _canon_scope_key(sk)
            frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ─────────────────────────────────────────────────────────────
# Mapping stores (append-only + dedupe)
# ─────────────────────────────────────────────────────────────

def _norm(s: str | None) -> str:
    return (s or "").strip().lower()


def _colpick(df, *names):
    low = {c.lower(): c for c in df.columns}
    for n in names:
        c = low.get(n.lower())
        if c:
            return c
    return None


def _sha256_of_df(df: pd.DataFrame) -> str:
    csv_bytes = df.to_csv(index=False).encode("utf-8", errors="ignore")
    return hashlib.sha256(csv_bytes).hexdigest()


def ensure_mapping_tables():
    with _conn() as cx:
        cx.execute("""
            CREATE TABLE IF NOT EXISTS mapping_files (
              id INTEGER PRIMARY KEY,
              kind TEXT CHECK (kind IN ('map1','map2')) NOT NULL,
              filename TEXT,
              sha256 TEXT UNIQUE,
              uploaded_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """)
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


# Ensure mapping tables exist at import
ensure_mapping_tables()


def save_mapping_sheet1(df: pd.DataFrame, filename: str | None = None) -> dict:
    """
    Append rows from Mapping Sheet 1 (union across uploads).
    Dedupe key: (BA, SBA, Channel, Location, Site) normalized.
    Accepts varying column names.
    """
    if df is None or df.empty:
        save_df("mapping_sheet1", pd.DataFrame())
        return {"inserted": 0, "skipped": 0}

    ba_c = _colpick(df, "Business Area", "Program", "BA")
    sba_c = _colpick(df, "Sub Business Area", "Sub BA")
    ch_c = _colpick(df, "LOB", "Channel", "Channels")
    loc_c = _colpick(df, "Location", "Country")
    site_c = _colpick(df, "Site", "Location Site", "Site/Location")

    sha = _sha256_of_df(df)
    with _conn() as cx:
        cx.execute("""
            INSERT OR IGNORE INTO mapping_files(kind, filename, sha256, uploaded_at)
            VALUES('map1', ?, ?, ?)
        """, (filename or "mapping1.csv", sha, datetime.now(timezone.utc).isoformat(timespec="seconds")))
        file_id = cx.execute("SELECT id FROM mapping_files WHERE sha256=?", (sha,)).fetchone()["id"]

        inserted = skipped = 0
        for _, r in df.iterrows():
            BA = str(r.get(ba_c, "") or "").strip()
            SBA = str(r.get(sba_c, "") or "").strip()
            CH = str(r.get(ch_c, "") or "").strip()
            LOC = str(r.get(loc_c, "") or "").strip()
            SITE = str(r.get(site_c, "") or "").strip()
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

    # Legacy snapshot for any old readers
    save_df("mapping_sheet1", df)
    return {"inserted": inserted, "skipped": skipped, "file_id": file_id}


def load_mapping_sheet1() -> pd.DataFrame | None:
    with _conn() as cx:
        cur = cx.execute("""
            SELECT DISTINCT
                business_area     AS "Business Area",
                sub_business_area AS "Sub Business Area",
                channel           AS "Channel",
                location          AS "Location",
                site              AS "Site"
            FROM map1_rows
            WHERE business_area IS NOT NULL AND business_area <> ''
        """)
        rows = [dict(r) for r in cur.fetchall()]
    if not rows:
        return pd.DataFrame(columns=["Business Area", "Sub Business Area", "Channel", "Location", "Site"])
    return pd.DataFrame(rows)


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
        """, (filename or "mapping2.csv", sha, datetime.now(timezone.utc).isoformat(timespec="seconds")))
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


def _hierarchy_from_map1() -> dict[str, dict[str, list[str]]]:
    """
    Returns: { Business Area -> { Sub Business Area -> [LOBs/channels...] } }
    Uses Mapping Sheet 1 columns (case-insensitive): BA, Sub BA, LOB/Channel.
    """
    try:
        m1 = load_mapping_sheet1()
    except Exception:
        m1 = None
    if not isinstance(m1, pd.DataFrame) or m1.empty:
        return {}

    L = {str(c).strip().lower(): c for c in m1.columns}
    ba_col = L.get("business area") or L.get("ba")
    sub_col = L.get("sub business area") or L.get("sub_ba") or L.get("sub-business area")
    lob_col = L.get("lob") or L.get("channel")

    if not ba_col:
        return {}

    out: dict[str, dict[str, set[str]]] = {}
    for _, r in m1.iterrows():
        ba = str(r.get(ba_col, "")).strip()
        if not ba:
            continue
        sub = str(r.get(sub_col, "")).strip() if sub_col else ""
        lob = str(r.get(lob_col, "")).strip() if lob_col else ""
        out.setdefault(ba, {})
        if sub:
            out[ba].setdefault(sub, set())
            if lob:
                out[ba][sub].add(lob)

    channel_defaults = ["Voice", "Back Office", "Outbound", "Blended"]
    hmap: dict[str, dict[str, list[str]]] = {}
    for ba, subs in out.items():
        hmap[ba] = {}
        for sub, lobset in subs.items():
            hmap[ba][sub] = sorted(list(lobset)) if lobset else channel_defaults
    return hmap


def sites_from_map1() -> list[str]:
    m1 = load_mapping_sheet1()
    if m1 is None or m1.empty:
        return []
    L = {c.lower(): c for c in m1.columns}
    site_c = L.get("site")
    if not site_c:
        return []
    vals = m1[site_c].dropna().astype(str).str.strip()
    return sorted([v for v in vals.unique() if v])


# ─────────────────────────────────────────────────────────────
# Defaults & Scoped Settings
# ─────────────────────────────────────────────────────────────

def load_defaults() -> dict | None:
    return load_kv("defaults")


def save_defaults(cfg: dict):
    save_kv("defaults", _normalize_settings_dict(cfg or {}))


# def save_scoped_settings(scope_type: str, scope_key: str, d: dict):
#     assert scope_type in ("location", "hier")
#     blob = json.dumps(d)
#     with _conn() as cx:
#         cx.execute("""
#             INSERT INTO settings_scoped(scope_type, scope_key, value)
#             VALUES (?,?,?)
#             ON CONFLICT(scope_type,scope_key) DO UPDATE SET value=excluded.value
#         """, (scope_type, scope_key, blob))

def save_scoped_settings(scope_type: str, scope_key: str, d: dict):
    assert scope_type in ("location", "hier")
    if scope_type == "hier":
        scope_key = _canon_hier_key(scope_key)
    else:
        scope_key = _canon(scope_key)

    blob = json.dumps(_normalize_settings_dict(d or {}))
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as cx:
        cx.execute("""
            INSERT INTO settings_scoped(scope_type, scope_key, value, updated_at)
            VALUES (?,?,?,?)
            ON CONFLICT(scope_type, scope_key)
            DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
        """, (scope_type, scope_key, blob, now))
        cx.commit()

def load_scoped_settings(scope_type: str, scope_key: str) -> dict:
    with _conn() as cx:
        row = None
        if scope_type == "hier":
            canon = _canon_hier_key(scope_key)
            row = cx.execute("""
                SELECT value FROM settings_scoped
                WHERE scope_type=? AND scope_key=?
            """, ("hier", canon)).fetchone()
            if not row:
                row = cx.execute("""
                    SELECT value FROM settings_scoped
                    WHERE scope_type=? AND LOWER(scope_key)=LOWER(?)
                """, ("hier", scope_key or "")).fetchone()
        else:
            canon = _canon(scope_key)
            row = cx.execute("""
                SELECT value FROM settings_scoped
                WHERE scope_type=? AND scope_key=?
            """, ("location", canon)).fetchone()
            if not row:
                row = cx.execute("""
                    SELECT value FROM settings_scoped
                    WHERE scope_type=? AND LOWER(scope_key)=LOWER(?)
                """, ("location", scope_key or "")).fetchone()

    return _normalize_settings_dict(json.loads(row["value"])) if row else {}

def resolve_settings(location: str | None = None,
                     ba: str | None = None,
                     subba: str | None = None,
                     lob: str | None = None) -> dict:
    # exact hierarchical scope first
    if ba and subba and lob:
        s = load_scoped_settings("hier", f"{ba}|{subba}|{lob}")
        if s:
            return s
    # location scope next
    if location:
        s = load_scoped_settings("location", location)
        if s:
            return s
    # global defaults last
    d = load_kv("defaults") or {}
    return _normalize_settings_dict(d)



# ─────────────────────────────────────────────────────────────
# Roster / Hiring / Shrinkage / Attrition datasets
# ─────────────────────────────────────────────────────────────

def load_roster() -> pd.DataFrame:
    return _ensure_df(load_df("roster"))


def save_roster(df: pd.DataFrame):
    """
    Saves roster with safe de-duplication:
    - If a 'date' column exists (long format), drop duplicates by (BRID, date).
    - If wide format (YYYY-MM-DD columns), melt to 'roster_long' by (BRID, date).
    - Else, de-dupe by BRID only.
    """
    if df is None or df.empty:
        save_df("roster", pd.DataFrame())
        return

    L = {c.lower(): c for c in df.columns}
    brid_col = L.get("brid") or L.get("employee_id") or "BRID"

    # Long
    if "date" in L:
        date_col = L["date"]
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.date.astype(str)
        df = df.dropna(subset=[brid_col, date_col])
        df = df.drop_duplicates(subset=[brid_col, date_col], keep="last")
        save_df("roster", df)
        return

    # Wide → Long
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
        long = long.dropna(subset=[brid_col, "date"]).drop_duplicates(subset=[brid_col, "date"], keep="last")

        save_df("roster", df)         # keep the wide view here for back-compat
        save_df("roster_long", long)  # normalized
        return

    # Legacy
    save_df("roster", df.drop_duplicates(subset=[brid_col], keep="last"))


def load_roster_wide() -> pd.DataFrame:
    """
    Prefer the dedicated 'roster_wide' key (new), fallback to legacy 'roster'.
    """
    df = load_df("roster_wide")
    if isinstance(df, pd.DataFrame) and not df.empty:
        return df
    df2 = load_df("roster")
    return df2 if isinstance(df2, pd.DataFrame) else pd.DataFrame()


def save_roster_wide(df: pd.DataFrame):
    save_df("roster_wide", df if isinstance(df, pd.DataFrame) else pd.DataFrame())


def load_roster_long() -> pd.DataFrame:
    df = load_df("roster_long")
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


def save_roster_long(df: pd.DataFrame):
    save_df("roster_long", df if isinstance(df, pd.DataFrame) else pd.DataFrame())


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
    save_df("attrition_raw", df if isinstance(df, pd.DataFrame) else pd.DataFrame())


# ─────────────────────────────────────────────────────────────
# Helpers for UI sources (locations / hierarchy)
# ─────────────────────────────────────────────────────────────

def get_roster_locations() -> list[str]:
    df = load_roster()
    if df.empty:
        return []
    vals: set[str] = set()
    for c in ["location", "country", "site", "region"]:
        if c in df.columns:
            vals |= set(
                df[c].dropna().astype(str).str.strip().replace({"": None}).dropna().tolist()
            )
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
        lobs = h.get("channels") or ["Voice", "Back Office", "Outbound"]
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
    capdb_path = DB_PATH  # same DB now; keep fallback scaffold
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


# ─────────────────────────────────────────────────────────────
# Sample template makers (for downloads)
# ─────────────────────────────────────────────────────────────

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


# One-time helper: migrate datasets saved with mixed-case scope keys to canonical lower-case.
def migrate_timeseries_scope_keys_to_lower() -> int:
    moved = 0
    with _conn() as cx:
        rows = cx.execute("SELECT name FROM datasets WHERE name LIKE '%::%'").fetchall()
    for r in rows:
        name = r["name"] if isinstance(r, sqlite3.Row) else r[0]
        if "::" not in name:
            continue
        kind, raw_sk = name.split("::", 1)
        canon = _canon_scope_key(raw_sk)
        if canon != raw_sk:
            df = load_df(name)
            if isinstance(df, pd.DataFrame) and not df.empty:
                save_df(f"{kind}::{canon}", df)
                moved += 1
    return moved
