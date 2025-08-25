from __future__ import annotations
import json
import datetime as dt
from datetime import datetime
from typing import List, Dict, Any, Optional

# Reuse the same DB file/connection as the rest of the app
from cap_db import _conn

# ──────────────────────────────────────────────────────────────────────────────
# Bootstrap
# ──────────────────────────────────────────────────────────────────────────────
def _init():
    with _conn() as cx:
        cx.execute("""
        CREATE TABLE IF NOT EXISTS capacity_plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            org TEXT,
            business_entity TEXT,
            vertical TEXT,            -- Business Area
            sub_ba TEXT,
            channel TEXT,             -- CSV (normalized)
            location TEXT,            -- NEW: location/country/city
            site TEXT,                -- campus/site
            plan_name TEXT NOT NULL,
            plan_type TEXT,
            start_week TEXT,          -- YYYY-MM-DD (Monday)
            end_week TEXT,            -- YYYY-MM-DD
            ft_weekly_hours REAL,
            pt_weekly_hours REAL,
            tags TEXT,                -- JSON list or ""
            is_current INTEGER DEFAULT 0,
            status TEXT DEFAULT 'draft',  -- 'current' | 'history' | 'draft'
            hierarchy_json TEXT,      -- optional BA/SubBA/Channels/Site bundle
            owner TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """)
        cx.commit()

_init()

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _now_iso():
    return dt.datetime.utcnow().isoformat()

def _norm_text(s) -> str:
    return ("" if s is None else str(s)).strip().lower()

def _norm_channel_csv(x) -> str:
    """
    Return a normalized, sorted CSV for a channel field that may be a list or CSV.
    All lower-case, spaces trimmed, sorted and deduped.
    """
    if x is None:
        return ""
    if isinstance(x, (list, tuple)):
        parts = [str(v).strip().lower() for v in x if str(v).strip()]
    else:
        parts = [p.strip().lower() for p in str(x).split(",") if p.strip()]
    parts = sorted(set(parts))
    return ", ".join(parts)

# ──────────────────────────────────────────────────────────────────────────────
# Create / Update
# ──────────────────────────────────────────────────────────────────────────────
def create_plan(payload: dict) -> int:
    """
    Enforces:
      1) Exact duplicate guard → error when
         (BA, SubBA, Channel-set, Location, Site, PlanName) match.
      2) If is_current is requested and there already is a CURRENT plan with the
         same (BA, SubBA, Channel-set, Location, Site) but a different PlanName,
         that existing current plan is demoted to history.
    """
    with _conn() as cx:
        vertical   = (payload.get("vertical") or "").strip()
        sub_ba     = (payload.get("sub_ba") or "").strip()
        name       = (payload.get("plan_name") or "").strip()
        location   = (payload.get("location") or "").strip()
        site       = (payload.get("site") or "").strip()
        chan_norm  = _norm_channel_csv(payload.get("channel"))
        is_current = 1 if payload.get("is_current") else 0
        status     = payload.get("status") or ("current" if is_current else "draft")

        # --- Rule #1: Exact-duplicate guard ----------------------------------
        # Match BA+SBA+PlanName+Location+Site (case-insensitive) and then confirm
        # the normalized Channel-set matches.
        dup_rows = cx.execute(
            """
            SELECT id, channel, location, site, plan_name
              FROM capacity_plans
             WHERE LOWER(vertical) = LOWER(?)
               AND COALESCE(sub_ba,'') = COALESCE(?, '')
               AND LOWER(TRIM(plan_name)) = LOWER(?)
               AND LOWER(COALESCE(TRIM(location),'')) = LOWER(COALESCE(?, ''))
               AND LOWER(COALESCE(TRIM(site),''))     = LOWER(COALESCE(?, ''))
            """,
            (vertical, sub_ba, name, location, site)
        ).fetchall()

        for r in dup_rows:
            if _norm_channel_csv(r["channel"]) == chan_norm:
                # same Channel-set + same Location/Site + same Name → duplicate
                raise ValueError("Duplicate: that plan already exists for this Business Area & Sub Business Area with the same channels, location/site and name.")

        # --- Rule #2: Demote other current plans in the *same scope* ----------
        # Scope for demotion is: BA + SubBA + Channel-set + Location + Site.
        if is_current:
            # Fetch all CURRENT plans in BA/SBA/Location/Site; compare channels in Python.
            cand = cx.execute(
                """
                SELECT id, channel
                  FROM capacity_plans
                 WHERE LOWER(vertical) = LOWER(?)
                   AND COALESCE(sub_ba,'') = COALESCE(?, '')
                   AND LOWER(COALESCE(TRIM(location),'')) = LOWER(COALESCE(?, ''))
                   AND LOWER(COALESCE(TRIM(site),''))     = LOWER(COALESCE(?, ''))
                   AND is_current = 1
                """,
                (vertical, sub_ba, location, site)
            ).fetchall()

            to_demote = [r["id"] for r in cand if _norm_channel_csv(r["channel"]) == chan_norm]
            if to_demote:
                placeholders = ",".join("?" for _ in to_demote)
                ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                cx.execute(
                    f"UPDATE capacity_plans "
                    f"   SET is_current=0, status='history', updated_at=? "
                    f" WHERE id IN ({placeholders})",
                    [ts] + to_demote
                )

        # --- Insert new plan ---------------------------------------------------
        p = payload.copy()

        # normalize/serialize
        p["channel"] = chan_norm
        if isinstance(p.get("tags"), (list, dict)):
            p["tags"] = json.dumps(p["tags"])
        elif p.get("tags") is None:
            p["tags"] = ""

        p["is_current"] = is_current
        p["status"] = status

        # timestamps
        cols = [r["name"] for r in cx.execute("PRAGMA table_info(capacity_plans)").fetchall()]
        now  = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        if "created_at" in cols and "created_at" not in p:
            p["created_at"] = now
        if "updated_at" in cols and "updated_at" not in p:
            p["updated_at"] = now

        fields_all = [
            "org","business_entity","vertical","sub_ba","channel","location","site",
            "plan_name","plan_type","start_week","end_week",
            "ft_weekly_hours","pt_weekly_hours","tags","is_current","status","hierarchy_json",
            "created_at","updated_at"
        ]
        fields = [f for f in fields_all if f in cols]
        for f in fields:
            p.setdefault(f, None)
        sql = f"INSERT INTO capacity_plans ({', '.join(fields)}) VALUES ({', '.join(':'+f for f in fields)})"

        cur = cx.execute(sql, p)
        pid = cur.lastrowid
        cx.commit()
        return pid

# ──────────────────────────────────────────────────────────────────────────────
# Status helpers
# ──────────────────────────────────────────────────────────────────────────────
def set_plan_status(plan_id: int, status: str):
    assert status in ("current", "history", "draft")
    # ts = _now_iso()
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S") 
    with _conn() as cx:
        if status == "current":
            row = cx.execute(
                "SELECT vertical, sub_ba, channel, location, site FROM capacity_plans WHERE id=?",
                (plan_id,)
            ).fetchone()
            if row:
                chan_norm = _norm_channel_csv(row["channel"])
                # demote any other current plan with SAME BA/SBA/Channel-set/Location/Site
                cand = cx.execute(
                    """
                    SELECT id, channel
                      FROM capacity_plans
                     WHERE LOWER(vertical)=LOWER(?)
                       AND COALESCE(sub_ba,'')=COALESCE(?, '')
                       AND LOWER(COALESCE(TRIM(location),''))=LOWER(COALESCE(?, ''))
                       AND LOWER(COALESCE(TRIM(site),''))    =LOWER(COALESCE(?, ''))
                       AND is_current=1
                       AND id <> ?
                    """,
                    (row["vertical"], row["sub_ba"], row["location"], row["site"], plan_id)
                ).fetchall()
                to_demote = [r["id"] for r in cand if _norm_channel_csv(r["channel"]) == chan_norm]
                if to_demote:
                    placeholders = ",".join("?" for _ in to_demote)
                    cx.execute(
                        f"UPDATE capacity_plans "
                        f"   SET is_current=0, status='history', updated_at=? "
                        f" WHERE id IN ({placeholders})",
                        [ts] + to_demote
                    )
        cx.execute(
            "UPDATE capacity_plans SET status=?, is_current=?, updated_at=? WHERE id=?",
            (status, 1 if status == "current" else 0, ts, plan_id)
        )
        cx.commit()

# ──────────────────────────────────────────────────────────────────────────────
# Reads / Deletes
# ──────────────────────────────────────────────────────────────────────────────
def list_business_areas(status_filter: Optional[str] = "current") -> List[str]:
    q = "SELECT DISTINCT vertical FROM capacity_plans WHERE 1=1"
    args: list = []
    if status_filter:
        q += " AND status=?"; args.append(status_filter)
    with _conn() as cx:
        cols = {r["name"] for r in cx.execute("PRAGMA table_info(capacity_plans)")}
        if "is_deleted" in cols:
            q += " AND COALESCE(is_deleted,0)=0"
        elif "deleted_at" in cols:
            q += " AND deleted_at IS NULL"
        rows = cx.execute(q, args).fetchall()
        return sorted([r["vertical"] for r in rows if r["vertical"]])

def list_plans(vertical: Optional[str] = None,
               status_filter: Optional[str] = None,
               include_deleted: bool = False) -> List[Dict]:
    sql = "SELECT * FROM capacity_plans WHERE 1=1"
    params: list = []
    if vertical:
        sql += " AND vertical=?"; params.append(vertical)
    if status_filter == "current":
        sql += " AND is_current=1"
    elif status_filter == "history":
        sql += " AND status='history'"
    elif status_filter == "draft":
        sql += " AND status='draft'"

    with _conn() as cx:
        cols = {r["name"] for r in cx.execute("PRAGMA table_info(capacity_plans)")}

        if not include_deleted:
            if "is_deleted" in cols:
                sql += " AND COALESCE(is_deleted,0)=0"
            elif "deleted_at" in cols:
                sql += " AND deleted_at IS NULL"

        rows = cx.execute(sql + " ORDER BY created_at DESC", params).fetchall()
        return [dict(r) for r in rows]

def mark_history(plan_id: int):
    set_plan_status(plan_id, "history")

def get_plan(plan_id: int) -> Optional[Dict[str, Any]]:
    with _conn() as cx:
        row = cx.execute("SELECT * FROM capacity_plans WHERE id=?", (plan_id,)).fetchone()
        return dict(row) if row else None

def delete_plan(plan_id: int, hard_if_missing: bool = True) -> None:
    with _conn() as cx:
        cols = {r["name"] for r in cx.execute("PRAGMA table_info(capacity_plans)")}
        try:
            if "is_deleted" in cols:
                cx.execute("UPDATE capacity_plans SET is_deleted=1, updated_at=datetime('now') WHERE id=?", (plan_id,))
            elif "deleted_at" in cols:
                cx.execute("UPDATE capacity_plans SET deleted_at=datetime('now') WHERE id=?", (plan_id,))
            elif hard_if_missing:
                cx.execute("DELETE FROM capacity_plans WHERE id=?", (plan_id,))
        finally:
            cx.commit()
