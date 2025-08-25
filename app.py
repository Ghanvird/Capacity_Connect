# file: app.py
from __future__ import annotations
import os, platform, getpass, base64, io, datetime as dt
import pandas as pd
import numpy as np
from typing import List
from planning_workspace import planning_layout, register_planning_ws
from dash import Dash, html, dcc, dash_table, Output, Input, State, callback
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import plotly.express as px
import dash
from datetime import date, timedelta
from plan_store import get_plan
from cap_db import save_df, load_df
from plan_detail import layout_for_plan, plan_detail_validation_layout, register_plan_detail

# ---- Core math & demo (replace with real when ready) ----
from capacity_core import (
    required_fte_daily, supply_fte_daily, understaffed_accounts_next_4w,
    kpi_hiring, kpi_shrinkage,
    make_projects_sample, make_voice_sample, make_backoffice_sample,
    make_outbound_sample, make_roster_sample, make_hiring_sample,
    make_shrinkage_sample, make_attrition_sample, _last_next_4, min_agents
)

# ---- SQLite persistence & dynamic sources (NO Mapping 1 anywhere) ----
from cap_store import (
    init_db, load_defaults, save_defaults, save_roster_long, save_roster_wide,
    load_roster, save_roster, load_roster_long, load_roster_wide,
    load_hiring, save_hiring, load_attrition_raw, save_attrition_raw,
    load_shrinkage, save_shrinkage,
    load_attrition, save_attrition,
    save_scoped_settings, resolve_settings,
    ensure_indexes, save_timeseries, brid_manager_map,
    load_headcount, level2_to_journey_map, load_timeseries, get_clients_hierarchy
)

# Initialize DB file
init_db()
ensure_indexes()

SYSTEM_NAME = (os.environ.get("HOSTNAME") or getpass.getuser() or platform.node())

# ---------------------- Dash App ----------------------
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="CAPACITY CONNECT"
)
server = app.server

def _save_budget_hc_timeseries(key: str, dff: pd.DataFrame):
    """Also persist weekly headcount for the HC tab.
       Planned HC = Budget HC (per your requirement)."""
    if dff is None or dff.empty:
        return
    if not {"week","budget_headcount"}.issubset(dff.columns):
        return
    hc = dff[["week","budget_headcount"]].copy()
    hc["week"] = pd.to_datetime(hc["week"], errors="coerce").dt.date.astype(str)
    hc.rename(columns={"budget_headcount":"headcount"}, inplace=True)

    # store both budget and planned (planned = budget)
    save_timeseries("hc_budget",  key, hc)   # week, headcount
    save_timeseries("hc_planned", key, hc)   # week, headcount


def _first_non_empty_ts(scope_key: str, keys: list[str]) -> pd.DataFrame:
    """Return the first non-empty timeseries DF for any of the given keys."""
    for k in keys:
        try:
            df = load_timeseries(k, scope_key)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df.copy()
        except Exception:
            pass
    return pd.DataFrame()

def _canon_scope(ba, sba, ch):
    canon = lambda x: (x or '').strip().lower()
    return f"{canon(ba)}|{canon(sba)}|{canon(ch)}"

def _coerce_time(s):
    s = str(s).strip()
    # Accept "09:00", "9:00", "900" → "09:00"
    if ":" in s:
        h, m = s.split(":")[0:2]
        return f"{int(h):02d}:{int(m):02d}"
    if s.isdigit():
        s = s.zfill(4)
        return f"{int(s[:2]):02d}:{int(s[2:]):02d}"
    return ""

def _minutes_to_seconds(x):
    # Accept "HH:MM" or number of minutes; return seconds
    s = str(x).strip()
    if ":" in s:
        h, m = s.split(":")[0:2]
        return (int(h)*60 + int(m)) * 60
    try:
        return float(s) * 60.0
    except:
        return None

def _week_monday(x):
    d = pd.to_datetime(x, errors="coerce")
    if pd.isna(d): return None
    d = d.normalize()
    return (d - pd.Timedelta(days=int(d.weekday()))).date()

def _scope_key(ba, subba, channel):
    return f"{(ba or '').strip()}|{(subba or '').strip()}|{(channel or '').strip()}"

def _budget_voice_template(start_week=None, weeks=8):
    start = _week_monday(start_week or pd.Timestamp.today())
    rows = []
    for w in range(weeks):
        wk = (pd.Timestamp(start) + pd.Timedelta(weeks=w)).date()
        rows.append(dict(week=wk.isoformat(), budget_headcount=25 + (w%3)*5, budget_aht_sec=300))
    return pd.DataFrame(rows, columns=["week","budget_headcount","budget_aht_sec"])

def _budget_bo_template(start_week=None, weeks=8):
    start = _week_monday(start_week or pd.Timestamp.today())
    rows = []
    for w in range(weeks):
        wk = (pd.Timestamp(start) + pd.Timedelta(weeks=w)).date()
        rows.append(dict(week=wk.isoformat(), budget_headcount=30 + (w%2)*3, budget_sut_sec=600))
    return pd.DataFrame(rows, columns=["week","budget_headcount","budget_sut_sec"])

def _budget_normalize_voice(df):
    if df is None or df.empty: return pd.DataFrame(columns=["week","budget_headcount","budget_aht_sec"])
    L = {c.lower(): c for c in df.columns}
    wk = L.get("week") or L.get("start_week") or L.get("monday") or list(df.columns)[0]
    hc = L.get("budget_headcount") or "budget_headcount"
    aht= L.get("budget_aht_sec") or "budget_aht_sec"
    dff = df.rename(columns={wk:"week"}).copy()
    if hc not in dff: dff[hc] = None
    if aht not in dff: dff[aht] = None
    dff["week"] = dff["week"].map(_week_monday).astype(str)
    dff["budget_headcount"] = pd.to_numeric(dff[hc], errors="coerce")
    dff["budget_aht_sec"]   = pd.to_numeric(dff[aht], errors="coerce")
    dff = dff.dropna(subset=["week"]).drop_duplicates(subset=["week"], keep="last")
    return dff[["week","budget_headcount","budget_aht_sec"]]

def _budget_normalize_bo(df):
    if df is None or df.empty: return pd.DataFrame(columns=["week","budget_headcount","budget_sut_sec"])
    L = {c.lower(): c for c in df.columns}
    wk = L.get("week") or L.get("start_week") or L.get("monday") or list(df.columns)[0]
    hc = L.get("budget_headcount") or "budget_headcount"
    sut= L.get("budget_sut_sec") or "budget_sut_sec"
    dff = df.rename(columns={wk:"week"}).copy()
    if hc not in dff: dff[hc] = None
    if sut not in dff: dff[sut] = None
    dff["week"] = dff["week"].map(_week_monday).astype(str)
    dff["budget_headcount"] = pd.to_numeric(dff[hc], errors="coerce")
    dff["budget_sut_sec"]   = pd.to_numeric(dff[sut], errors="coerce")
    dff = dff.dropna(subset=["week"]).drop_duplicates(subset=["week"], keep="last")
    return dff[["week","budget_headcount","budget_sut_sec"]]


# === SHRINKAGE (RAW) — helpers & templates ====================================

def _hhmm_to_minutes(x) -> float:
    if pd.isna(x): return 0.0
    s = str(x).strip()
    if not s: return 0.0
    # allow "HH:MM", "H:MM", "MM", "H.MM" etc.
    m = None
    if ":" in s:
        parts = s.split(":")
        if len(parts) >= 2:
            try:
                h = int(parts[0]); mm = int(parts[1])
                return float(h * 60 + mm)
            except Exception:
                pass
    try:
        # fallback: numeric minutes
        return float(s)
    except Exception:
        return 0.0

def _hc_lookup():
    """Return simple dict lookups from headcount: BRID→{lm_name, site, city, country, journey, level_3}"""
    try:
        hc = load_headcount()
    except Exception:
        hc = pd.DataFrame()
    if not isinstance(hc, pd.DataFrame) or hc.empty:
        return {}
    L = {c.lower(): c for c in hc.columns}
    def col(name):
        return L.get(name, name)
    out = {}
    for _, r in hc.iterrows():
        brid = str(r.get(col("brid"), "")).strip()
        if not brid: 
            continue
        out[brid] = dict(
            lm_name = r.get(col("line_manager_full_name")),
            site    = r.get(col("position_location_building_description")),
            city    = r.get(col("position_location_city")),
            country = r.get(col("position_location_country")),
            journey = r.get(col("journey")),
            level_3 = r.get(col("level_3")),
        )
    return out

# ---- Back Office RAW template (seconds) ----
def shrinkage_bo_raw_template_df(rows: int = 16) -> pd.DataFrame:
    today = pd.Timestamp.today().normalize().date()
    cats = [
        "Staff Complement","Flextime","Borrowed Staff","Lend Staff",
        "Overtime","Core Time","Diverted","Downtime","Time Worked","Work out"
    ]
    demo = []
    for i in range(rows):
        cat = cats[i % len(cats)]
        dur = 1800 if cat in ("Diverted","Downtime") else (3_600 if cat in ("Core Time","Time Worked") else 1200)
        brid = f"IN{1000+i}"
        demo.append({
            "Category":"Shrinkage", "StartDate": today.isoformat(), "EndDate": today.isoformat(),
            "DateId": int(pd.Timestamp(today).strftime("%Y%m%d")),
            "Date": today.isoformat(),
            "GroupId": "BO1", "WorkgroupId": "WG1", "WorkgroupName": "BO Cases",
            "Activity": cat,
            "SaffMemberId": brid, "StaffLastName": "Doe", "SatffFirstName": "Alex",
            "StaffReferenceId": brid, "TaskId": "T-001", "Units": 10 if cat=="Work out" else 0,
            "DurationSeconds": dur, "EmploymentType": "FT",
            "AgentID(BRID)": brid, "Agent Name": "Alex Doe",
            "TL Name": "",  # will be filled from Headcount on upload
            "Time": round(dur/3600,2),
            "Sub Business Area": ""  # will be filled from Headcount (Level 3)
        })
    return pd.DataFrame(demo)

# ---- Voice RAW template (HH:MM) ----
def shrinkage_voice_raw_template_df(rows: int = 18) -> pd.DataFrame:
    today = pd.Timestamp.today().normalize().date()
    superstates = [
        "SC_INCLUDED_TIME","SC_ABSENCE_TOTAL","SC_A_Sick_Long_Term",
        "SC_HOLIDAY","SC_TRAINING_TOTAL","SC_BREAKS","SC_SYSTEM_EXCEPTION"
    ]
    demo = []
    for i in range(rows):
        ss = superstates[i % len(superstates)]
        hhmm = f"{(i%3)+1:02d}:{(i*10)%60:02d}"  # 01:00, 02:10, 03:20...
        brid = f"UK{2000+i}"
        demo.append({
            "Employee": f"User {i+1}",
            "BRID": brid, "First Name": "Sam", "Last Name": "Patel",
            "Superstate": ss, "Date": today.isoformat(), "Day of Week": "Mon",
            "Day": int(pd.Timestamp(today).day), "Month": int(pd.Timestamp(today).month),
            "Year": int(pd.Timestamp(today).year), "Week Number": int(pd.Timestamp(today).week),
            "Week of": (pd.Timestamp(today) - pd.Timedelta(days=pd.Timestamp(today).weekday())).date().isoformat(),
            "Hours": hhmm, "Management_Line": "", "Location": "", "CSM": "",
            "Monthly":"", "Weekly":"", "Business Area":"", "Sub Business Area":"", "Channel":"Voice"
        })
    return pd.DataFrame(demo)

# ---- Back Office RAW normalize + summary ----
def normalize_shrinkage_bo(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    L = {c.lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n.lower() in L: return L[n.lower()]
        return None
    # columns (case-insensitive)
    col_act  = pick("Activity")
    col_sec  = pick("DurationSeconds","Duration (sec)","duration_seconds")
    col_date = pick("Date")
    col_units = pick("Units")
    col_brid = pick("AgentID(BRID)","StaffReferenceId","SaffMemberId","StaffMemberId","BRID")
    col_fname = pick("SatffFirstName","StaffFirstName","FirstName")
    col_lname = pick("StaffLastName","LastName")
    if not (col_act and col_sec and col_date and col_brid):
        return pd.DataFrame()

    dff = df.copy()
    dff.rename(columns={
        col_act:"activity", col_sec:"duration_seconds", col_date:"date",
        col_units: "units" if col_units else "units",
        col_brid: "brid",
        col_fname or "": "first_name", col_lname or "": "last_name"
    }, inplace=True, errors="ignore")

    dff["date"] = pd.to_datetime(dff["date"], errors="coerce").dt.date
    dff["duration_seconds"] = pd.to_numeric(dff["duration_seconds"], errors="coerce").fillna(0).astype(float)
    if "units" in dff.columns:
        dff["units"] = pd.to_numeric(dff["units"], errors="coerce").fillna(0).astype(float)
    else:
        dff["units"] = 0.0
    dff["brid"] = dff["brid"].astype(str).str.strip()

    # enrich from headcount
    hc = _hc_lookup()
    dff["tl_name"] = dff["brid"].map(lambda x: (hc.get(x) or {}).get("lm_name"))
    dff["journey"] = dff["brid"].map(lambda x: (hc.get(x) or {}).get("journey"))
    dff["sub_business_area"] = dff["brid"].map(lambda x: (hc.get(x) or {}).get("level_3"))
    dff["time_hours"] = dff["duration_seconds"] / 3600.0
    dff["channel"] = "Back Office"
    return dff

def _bo_bucket(activity: str) -> str:
    s = (activity or "").strip().lower()
    # flexible matching
    if "divert" in s: return "diverted"
    if "down" in s: return "downtime"
    if "staff complement" in s or s == "staff complement": return "staff_complement"
    if "flex" in s: return "flextime"
    if "lend" in s: return "lend_staff"
    if "borrow" in s: return "borrowed_staff"
    if "overtime" in s or s=="ot": return "overtime"
    if "core time" in s or s=="core": return "core_time"
    if "time worked" in s: return "time_worked"
    if "work out" in s or "workout" in s: return "work_out"
    return "other"

def summarize_shrinkage_bo(dff: pd.DataFrame) -> pd.DataFrame:
    if dff is None or dff.empty: return pd.DataFrame()
    d = dff.copy()
    d["bucket"] = d["activity"].map(_bo_bucket)

    keys = ["date","journey","sub_business_area","channel"]
    # seconds per bucket
    sec = d.pivot_table(index=keys, columns="bucket", values="duration_seconds", aggfunc="sum", fill_value=0).reset_index()
    if "work_out" not in sec.columns: sec["work_out"] = 0.0
    if "time_worked" not in sec.columns: sec["time_worked"] = 0.0
    if "core_time" not in sec.columns: sec["core_time"] = 0.0
    if "overtime" not in sec.columns: sec["overtime"] = 0.0
    if "diverted" not in sec.columns: sec["diverted"] = 0.0
    if "downtime" not in sec.columns: sec["downtime"] = 0.0
    if "staff_complement" not in sec.columns: sec["staff_complement"] = 0.0
    if "flextime" not in sec.columns: sec["flextime"] = 0.0
    if "lend_staff" not in sec.columns: sec["lend_staff"] = 0.0
    if "borrowed_staff" not in sec.columns: sec["borrowed_staff"] = 0.0

    # denominator for shrinkage (all seconds)
    sec["shr_num"]  = sec["diverted"] + sec["downtime"]
    sec["shr_denom"] = (sec["staff_complement"] + sec["flextime"] + sec["overtime"] - sec["lend_staff"] + sec["borrowed_staff"]).clip(lower=0)

    # metrics
    sec["shrinkage_pct"] = np.where(sec["shr_denom"]>0, (sec["shr_num"]/sec["shr_denom"])*100.0, np.nan)
    # productivity = Work out / Core Time (units per hour)
    sec["productivity_u_per_hr"] = np.where(sec["core_time"]>0, (sec.get("work_out",0.0) / (sec["core_time"]/3600.0)), np.nan)
    # utilisation = Core Time / Time Worked
    denom_tw = np.where(sec["time_worked"]>0, sec["time_worked"], sec["staff_complement"] + sec["flextime"] + sec["overtime"])  # fallback
    sec["utilisation_pct"] = np.where(denom_tw>0, (sec["core_time"]/denom_tw)*100.0, np.nan)
    # OT hours
    sec["ot_hours"] = sec["overtime"] / 3600.0

    keep = keys + ["shrinkage_pct","productivity_u_per_hr","utilisation_pct","ot_hours","shr_num","shr_denom"]
    sec = sec[keep].sort_values(keys)
    # nice names
    sec = sec.rename(columns={"journey":"Business Area","sub_business_area":"Sub Business Area"})
    return sec

def weekly_shrinkage_from_bo_summary(daily: pd.DataFrame) -> pd.DataFrame:
    if daily is None or daily.empty: return pd.DataFrame(columns=["week","attrition_pct","program"])
    df = daily.copy()
    df["week"] = pd.to_datetime(df["date"]).dt.date.apply(lambda x: _week_floor(x,"Monday"))
    # program = Business Area
    df["program"] = df["Business Area"].fillna("All").astype(str)
    grp = df.groupby(["week","program"], as_index=False).agg({"shr_num":"sum","shr_denom":"sum"})
    grp["shrinkage_pct"] = np.where(grp["shr_denom"]>0, (grp["shr_num"]/grp["shr_denom"])*100.0, np.nan)
    return grp[["week","shrinkage_pct","program"]]

# ---- Voice RAW normalize + summary ----
def normalize_shrinkage_voice(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    L = {c.lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n.lower() in L: return L[n.lower()]
        return None
    col_date = pick("Date")
    col_state = pick("Superstate")
    col_hours = pick("Hours")
    col_brid  = pick("BRID","AgentID(BRID)","Employee Id","EmployeeID")
    if not (col_date and col_state and col_hours and col_brid):
        return pd.DataFrame()

    dff = df.copy()
    dff.rename(columns={col_date:"date", col_state:"superstate", col_hours:"hours_raw", col_brid:"brid"}, inplace=True)
    dff["date"] = pd.to_datetime(dff["date"], errors="coerce").dt.date
    dff["brid"] = dff["brid"].astype(str).str.strip()
    # convert HH:MM -> minutes, then to hours (as per spec they divide by 60)
    mins = dff["hours_raw"].map(_hhmm_to_minutes).fillna(0.0)
    dff["hours"] = mins/60.0

    # enrich from headcount
    hc = _hc_lookup()
    dff["TL Name"] = dff["brid"].map(lambda x: (hc.get(x) or {}).get("lm_name"))
    dff["Site"]    = dff["brid"].map(lambda x: (hc.get(x) or {}).get("site"))
    dff["City"]    = dff["brid"].map(lambda x: (hc.get(x) or {}).get("city"))
    dff["Country"] = dff["brid"].map(lambda x: (hc.get(x) or {}).get("country"))
    dff["Business Area"] = dff.get("Business Area", pd.Series(index=dff.index)).fillna(dff["brid"].map(lambda x: (hc.get(x) or {}).get("journey")))
    dff["Sub Business Area"] = dff.get("Sub Business Area", pd.Series(index=dff.index)).fillna(dff["brid"].map(lambda x: (hc.get(x) or {}).get("level_3")))
    if "Channel" not in dff.columns:
        dff["Channel"] = "Voice"

    # NEW: defaults so the pivot in summarize_shrinkage_voice never drops rows
    for col, default in [("Business Area", "All"), ("Sub Business Area", "All"), ("Country", "All")]:
        if col not in dff.columns:
            dff[col] = default
        else:
            dff[col] = dff[col].replace("", np.nan).fillna(default)
    dff["Channel"] = dff["Channel"].replace("", np.nan).fillna("Voice")
    return dff

def summarize_shrinkage_voice(dff: pd.DataFrame) -> pd.DataFrame:
    if dff is None or dff.empty: return pd.DataFrame()
    d = dff.copy()

    # Use Country only if present and not entirely NaN
    keys = ["date","Business Area","Sub Business Area","Channel"]
    if "Country" in d.columns and d["Country"].notna().any():
        keys.append("Country")

    piv = d.pivot_table(index=keys, columns="superstate", values="hours", aggfunc="sum", fill_value=0.0).reset_index()
    inc = piv.get("SC_INCLUDED_TIME", pd.Series([0.0]*len(piv)))

    def part(code): return piv[code] if code in piv.columns else 0.0
    abs_h  = part("SC_ABSENCE_TOTAL")
    lts_h  = part("SC_A_Sick_Long_Term")
    hol_h  = part("SC_HOLIDAY")
    trn_h  = part("SC_TRAINING_TOTAL")
    brk_h  = part("SC_BREAKS")
    sys_h  = part("SC_SYSTEM_EXCEPTION")

    def ratio(x): return np.where(inc>0, (x / inc) * 100.0, np.nan)
    piv["Absence %"]        = ratio(abs_h)
    piv["LTS %"]            = ratio(lts_h)
    piv["Holiday %"]        = ratio(hol_h)
    piv["Training %"]       = ratio(trn_h)
    piv["Breaks %"]         = ratio(brk_h)
    piv["Total Shrink %"]   = piv[["Absence %","Holiday %","Training %","Breaks %"]].sum(axis=1, min_count=1)
    piv["System Downtime %"]= ratio(sys_h)
    piv["Paid Hours"]       = inc

    keep = keys + ["Absence %","LTS %","Holiday %","Training %","Breaks %","Total Shrink %","System Downtime %","Paid Hours"]
    return piv[keep].sort_values(keys)


def weekly_shrinkage_from_voice_summary(daily: pd.DataFrame) -> pd.DataFrame:
    if daily is None or daily.empty: return pd.DataFrame(columns=["week","shrinkage_pct","program"])
    df = daily.copy()
    df["week"] = pd.to_datetime(df["date"]).dt.date.apply(lambda x: _week_floor(x,"Monday"))
    df["program"] = df["Business Area"].fillna("All").astype(str)
    # compute weekly Total Shrink weighted by included time (Paid Hours)
    grp = df.groupby(["week","program"], as_index=False).agg({
        "Total Shrink %":"mean",         # simple mean across BA/program/day
        "Paid Hours":"sum"
    })
    # If you prefer time-weighted shrink %, replace with:
    # grp["Total Shrink %"] = (weight sum of component hours) / (weight sum of inc) * 100
    grp = grp.rename(columns={"Total Shrink %":"shrinkage_pct"})
    return grp[["week","shrinkage_pct","program"]]


# ---------- BRID enrichment using headcount ----------
def enrich_with_manager(df: pd.DataFrame) -> pd.DataFrame:
    """Add Team Manager/Manager BRID to a wide or long roster using BRID mapping."""
    if df is None or df.empty:
        return df
    try:
        mgr = brid_manager_map()  # columns: brid, line_manager_brid, line_manager_full_name
    except Exception:
        return df
    if mgr is None or mgr.empty:
        return df

    out = df.copy()
    L = {str(c).lower(): c for c in out.columns}
    brid_col = L.get("brid") or L.get("employee id") or L.get("employee_id") or ("BRID" if "BRID" in out.columns else None)
    if not brid_col:
        return out

    out = out.merge(mgr, left_on=brid_col, right_on="brid", how="left")
    if "Team Manager" not in out.columns:
        out["Team Manager"] = out["line_manager_full_name"]
    else:
        out["Team Manager"] = out["Team Manager"].fillna(out["line_manager_full_name"])
    if "Manager BRID" not in out.columns:
        out["Manager BRID"] = out["line_manager_brid"]
    return out.drop(columns=["brid", "line_manager_full_name", "line_manager_brid"], errors="ignore")

# ===== Headcount-only helpers (Scope) =====
def _hcu_df() -> pd.DataFrame:
    try:
        df = load_headcount()
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def _hcu_cols(df: pd.DataFrame) -> dict:
    """
    Column resolver (case-insensitive) for the Headcount Update file.
      - ba: Journey / Business Area (a.k.a. Vertical)
      - sba: Level 3 (Sub Business Area)
      - loc: Country / Location
      - site: Building / Site
      - lob: Channel / Program if present (fallback handled later)
    """
    L = {str(c).strip().lower(): c for c in df.columns}

    ba = (
        L.get("journey")
        or L.get("business area")
        or L.get("vertical")
        or L.get("current org unit description")
        or L.get("current_org_unit_description")
        or L.get("level 0")
        or L.get("level_0")
    )
    sba = (
        L.get("level 3")
        or L.get("level_3")
        or L.get("sub business area")
        or L.get("sub_business_area")
    )
    loc = (
        L.get("position_location_country")
        or L.get("location country")
        or L.get("location_country")
        or L.get("country")
        or L.get("location")
    )
    site = (
        L.get("position_location_building_description")
        or L.get("building description")
        or L.get("building")
        or L.get("site")
    )
    lob = (
        L.get("lob")
        or L.get("channel")
        or L.get("program")
        or L.get("position group")
        or L.get("position_group")
    )
    return {"ba": ba, "sba": sba, "loc": loc, "site": site, "lob": lob}

CHANNEL_LIST = ["Voice", "Back Office", "Outbound", "Blended", "Chat", "MessageUs"]

def _all_locations() -> List[str]:
    df = _hcu_df()
    if df.empty: return []
    C = _hcu_cols(df)
    c = C["loc"]
    if not c: return []
    vals = (
        df[c].dropna().astype(str).str.strip().replace({"": np.nan})
        .dropna().drop_duplicates().sort_values()
    )
    return vals.tolist()

def _bas_from_headcount() -> List[str]:
    df = _hcu_df()
    if df.empty:
        return []
    C = _hcu_cols(df)
    if not C["ba"]:
        return []
    vals = (
        df[C["ba"]]
        .dropna().astype(str).str.strip().replace({"": np.nan})
        .dropna().drop_duplicates().sort_values()
    )
    return vals.tolist()

def _sbas_from_headcount(ba: str) -> List[str]:
    if not ba:
        return []
    df = _hcu_df()
    if df.empty:
        return []
    C = _hcu_cols(df)
    if not (C["ba"] and C["sba"]):
        return []
    dff = df[[C["ba"], C["sba"]]].dropna()
    dff = dff[dff[C["ba"]].astype(str).str.strip().str.lower() == str(ba).strip().lower()]
    vals = (
        dff[C["sba"]]
        .astype(str).str.strip().replace({"": np.nan})
        .dropna().drop_duplicates().sort_values()
    )
    return vals.tolist()

def _lobs_for_ba_sba(ba: str, sba: str) -> List[str]:
    """If headcount has a LOB/Channel column, use it; else fall back to fixed list."""
    if not (ba and sba):
        return []
    df = _hcu_df()
    if df.empty: return CHANNEL_LIST
    C = _hcu_cols(df)
    if not (C["ba"] and C["sba"] and C["lob"]):
        return CHANNEL_LIST
    dff = df[[C["ba"], C["sba"], C["lob"]]].dropna()
    mask = (
        dff[C["ba"]].astype(str).str.strip().str.lower().eq(str(ba).strip().lower()) &
        dff[C["sba"]].astype(str).str.strip().str.lower().eq(str(sba).strip().lower())
    )
    vals = (
        dff.loc[mask, C["lob"]]
        .astype(str).str.strip().replace({"": np.nan})
        .dropna().drop_duplicates().sort_values()
    )
    # return vals.tolist() or CHANNEL_LIST
    return CHANNEL_LIST

def _locations_for_ba(ba: str) -> List[str]:
    if not ba:
        return []
    df = _hcu_df()
    if df.empty:
        return []
    C = _hcu_cols(df)
    if not (C["ba"] and C["loc"]):
        return []
    dff = df[[C["ba"], C["loc"]]].dropna()
    dff = dff[dff[C["ba"]].astype(str).str.strip().str.lower() == str(ba).strip().lower()]
    vals = (
        dff[C["loc"]]
        .astype(str).str.strip().replace({"": np.nan})
        .dropna().drop_duplicates().sort_values()
    )
    return vals.tolist()

def _sites_for_ba_location(ba: str, location: str | None) -> List[str]:
    if not ba:
        return []
    df = _hcu_df()
    if df.empty:
        return []
    C = _hcu_cols(df)
    if not (C["ba"] and C["site"]):
        return []
    dff = df[df[C["ba"]].astype(str).str.strip().str.lower() == str(ba).strip().lower()]
    if C["loc"] and location:
        dff = dff[dff[C["loc"]].astype(str).str.strip().str.lower() == str(location).strip().lower()]
    vals = (
        dff[C["site"]]
        .dropna().astype(str).str.strip().replace({"": np.nan})
        .dropna().drop_duplicates().sort_values()
    )
    return vals.tolist()

# ---------------------- MAIN LAYOUT (unchanged shell) ----------------------
def _planning_ids_skeleton():
    return html.Div([
        dcc.Store(id="ws-status"),
        dcc.Store(id="ws-selected-ba"),
        dcc.Store(id="ws-refresh"),
    ], style={"display": "none"})

app.layout = html.Div([
    dcc.Location(id="url-router"),
    dcc.Store(id="sidebar_collapsed", data=True, storage_type="session"),
    _planning_ids_skeleton(),
    html.Div(id="app-wrapper", className="sidebar-collapsed", children=[
        html.Div(id="sidebar"),
        html.Div(id="root")
    ]),
    dcc.Interval(id="cap-plans-refresh", interval=5000, n_intervals=0)
])

# ---------------------- Demo data for Home ----------------------
DEFAULT_SETTINGS = dict(
    interval_minutes=30, hours_per_fte=8.0, shrinkage_pct=0.30, target_sl=0.80,
    sl_seconds=20, occupancy_cap_voice=0.85, util_bo=0.85, util_ob=0.85,
)

if not load_defaults():
    save_defaults(DEFAULT_SETTINGS)

projects_df  = make_projects_sample()
voice_df     = make_voice_sample(DEFAULT_SETTINGS["interval_minutes"], days=7)
bo_df        = make_backoffice_sample(days=7)
ob_df        = make_outbound_sample(days=7)
roster_demo  = make_roster_sample()
hiring_demo  = make_hiring_sample()
shrink_demo  = make_shrinkage_sample()
attr_demo    = make_attrition_sample()

req_df = required_fte_daily(voice_df, bo_df, ob_df, DEFAULT_SETTINGS)
sup_df = supply_fte_daily(roster_demo, hiring_demo)
understaffed = understaffed_accounts_next_4w(req_df, sup_df)
hire_lw, hire_tw, hire_nw = kpi_hiring(hiring_demo)
shr_last4, shr_next4 = kpi_shrinkage(shrink_demo)
attr_last4, attr_next4 = _last_next_4(attr_demo, "week", "attrition_pct")

# ---------------------- Helpers (templates & normalizers) ----------------------
ATTRITION_RAW_COLUMNS = [
    "Reporting Full Date","BRID","Employee Name","Operational Status",
    "Corporate Grade Description","Employee Email Address","Employee Position",
    "Position Description","Employee Line Manager Indicator","Length of Service Date",
    "Cost Centre","Line Manager BRID","Line Manager Name","IMH L05","IMH L06","IMH L07",
    "Org Unit","Org Unit ID","Employee Line Manager lvl 07","Employee Line Manager lvl 08",
    "Employee Line Manager lvl 09","City","Building","Gender Description",
    "Voluntary Involuntary Exit Description","Resignation Date","Employee Contract HC",
    "HC","FTE"
]

HC_COLUMNS = [
    "Level 0","Level 1","Level 2","Level 3","Level 4","Level 5","Level 6",
    "BRID","Full Name","Position Description","Headcount Operational Status Description",
    "Employee Group Description","Corporate Grade Description","Line Manager BRID","Line Manager Full Name",
    "Current Organisation Unit","Current Organisation Unit Description",
    "Position Location Country","Position Location City","Position Location Building Description",
    "CCID","CC Name","Journey","Position Group"
]
def headcount_template_df(rows: int = 5) -> pd.DataFrame:
    sample = [
        ["BUK","COO","Business Services","BFA","Refers","","","IN0001","Asha Rao","Agent","Active","FT","BA4","IN9999","Priyanka Menon","Ops|BFA|Refers","Ops BFA Refers","India","Chennai","DLF IT Park","12345","Complaints","Onboarding","Agent"],
        ["BUK","COO","Business Services","BFA","Appeals","","","IN0002","Rahul Jain","Team Leader","Active","FT","BA5","IN8888","Arjun Mehta","Ops|BFA|Appeals","Ops BFA Appeals","India","Pune","EON Cluster C","12345","Complaints","Onboarding","Team Leader"],
    ]
    df = pd.DataFrame(sample[:rows], columns=HC_COLUMNS)
    if rows > len(sample):
        df = pd.concat([df, pd.DataFrame(columns=HC_COLUMNS)], ignore_index=True)
    return df

# === New combined templates (as requested) ===
def voice_forecast_template_df():
    # Same columns as schema
    return pd.DataFrame([{
        "Date": date.today().isoformat(),
        "Interval": "09:00",
        "Forecast Volume": 120,
        "Forecast AHT": 300,
        "Business Area": "Retail Banking",
        "Sub Business Area": "Cards",
        "Channel": "Voice",
    }])

def voice_actual_template_df():
    return pd.DataFrame([{
        "Date": date.today().isoformat(),
        "Interval": "09:00",
        "Actual Volume": 115,
        "Actual AHT": 310,
        "Business Area": "Retail Banking",
        "Sub Business Area": "Cards",
        "Channel": "Voice",
    }])

def bo_forecast_template_df():
    return pd.DataFrame([{
        "Date": date.today().isoformat(),
        "Forecast Volume": 550,
        "Forecast SUT": 600,
        "Business Area": "Retail Banking",
        "Sub Business Area": "Cards",
        "Channel": "Back Office",
    }])

def bo_actual_template_df():
    return pd.DataFrame([{
        "Date": date.today().isoformat(),
        "Actual Volume": 520,
        "Actual SUT": 610,
        "Business Area": "Retail Banking",
        "Sub Business Area": "Cards",
        "Channel": "Back Office",
    }])

# === Normalizers for new combined sheets ===
def _norm_voice_combo(df: pd.DataFrame, kind: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (vol_df, aht_df) in internal normalized shapes for saving:
      vol_df: date, interval_start, volume
      aht_df: date, interval_start, aht_sec
    Accepts either Forecast or Actual columns depending on 'kind'.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["date","interval_start","volume"]), pd.DataFrame(columns=["date","interval_start","aht_sec"])
    L = {str(c).lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            c = L.get(n.lower())
            if c: return c
        return None

    date_col = pick("date")
    intv_col = pick("interval","interval start","intervalstart","time","slot")
    if kind == "forecast":
        vol_col = pick("forecast volume","volume")
        aht_col = pick("forecast aht","aht","aht (sec)")
    else:
        vol_col = pick("actual volume","volume")
        aht_col = pick("actual aht","aht","aht (sec)")

    if not (date_col and intv_col and vol_col and aht_col):
        return pd.DataFrame(), pd.DataFrame()

    df2 = df[[date_col, intv_col, vol_col, aht_col]].copy()
    df2.columns = ["date","interval_start","volume","aht_sec"]
    df2["date"] = pd.to_datetime(df2["date"], errors="coerce").dt.date.astype(str)
    df2["interval_start"] = df2["interval_start"].astype(str).str.extract(r"(\d{1,2}:\d{2})")[0]
    df2["volume"] = pd.to_numeric(df2["volume"], errors="coerce").fillna(0)
    df2["aht_sec"] = pd.to_numeric(df2["aht_sec"], errors="coerce").fillna(0)
    df2 = df2.dropna(subset=["date","interval_start"])
    df2 = df2.drop_duplicates(["date","interval_start"], keep="last").sort_values(["date","interval_start"])

    vol_df = df2[["date","interval_start","volume"]].copy()
    aht_df = df2[["date","interval_start","aht_sec"]].copy()
    return vol_df, aht_df

def _norm_bo_combo(df: pd.DataFrame, kind: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (vol_df, sut_df) in internal normalized shapes for saving:
      vol_df: date, volume
      sut_df: date, sut_sec
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["date","volume"]), pd.DataFrame(columns=["date","sut_sec"])
    L = {str(c).lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            c = L.get(n.lower())
            if c: return c
        return None

    date_col = pick("date")
    if kind == "forecast":
        vol_col = pick("forecast volume","volume")
        sut_col = pick("forecast sut","sut","sut (sec)")
    else:
        vol_col = pick("actual volume","volume")
        sut_col = pick("actual sut","sut","sut (sec)")

    if not (date_col and vol_col and sut_col):
        return pd.DataFrame(), pd.DataFrame()

    df2 = df[[date_col, vol_col, sut_col]].copy()
    df2.columns = ["date","volume","sut_sec"]
    df2["date"] = pd.to_datetime(df2["date"], errors="coerce").dt.date.astype(str)
    df2["volume"] = pd.to_numeric(df2["volume"], errors="coerce").fillna(0)
    df2["sut_sec"] = pd.to_numeric(df2["sut_sec"], errors="coerce").fillna(0)
    df2 = df2.dropna(subset=["date"]).drop_duplicates(["date"], keep="last").sort_values(["date"])

    vol_df = df2[["date","volume"]].copy()
    sut_df = df2[["date","sut_sec"]].copy()
    return vol_df, sut_df

# ---------- Parsing helpers ----------
def pretty_columns(df_or_cols) -> list[dict]:
    cols = list(df_or_cols.columns) if hasattr(df_or_cols, "columns") else list(df_or_cols)
    return [{"name": c, "id": c} for c in cols]

def lock_variance_cols(cols):
    """
    Return a copy of DataTable column defs with any Variance columns set read-only.
    Matches on id or header text containing 'variance' (case-insensitive).
    """
    out = []
    for col in cols:
        c = dict(col)  # copy
        name_txt = c.get("name", "")
        if isinstance(name_txt, list):  # sometimes headers are multi-line arrays
            name_txt = " ".join(map(str, name_txt))
        id_txt = str(c.get("id", ""))
        if "variance" in str(name_txt).lower() or "variance" in id_txt.lower():
            c["editable"] = False
        else:
            # make other columns explicitly editable unless already set
            c.setdefault("editable", True)
        out.append(c)
    return out


def parse_upload(contents, filename) -> pd.DataFrame:
    if not contents:
        return pd.DataFrame()
    try:
        _, content_string = contents.split(',', 1)
        data = base64.b64decode(content_string)
        if filename and filename.lower().endswith(".csv"):
            return pd.read_csv(io.StringIO(data.decode("utf-8")))
        if filename and filename.lower().endswith((".xls", ".xlsx", ".xlsm")):
            return pd.read_excel(io.BytesIO(data))
    except Exception:
        pass
    return pd.DataFrame()

# ---------------------- UI Fragments (left intact) ----------------------
def header_bar():
    return dbc.Navbar(
        dbc.Container([
            dbc.Button("☰", id="btn-burger-top", color="link", className="me-3", n_clicks=0,
                       style={"fontSize":"24px","textDecoration":"none"}),
            html.Span(style={"fontSize":"28px","fontWeight":800}),
            dbc.Breadcrumb(
                id="ws-breadcrumb",
                items=[{"label": "Home", "href": "/", "active": True}],
                className="mb-0 ms-3 flex-grow-1"
            ),
            dbc.Nav([ dcc.Link(SYSTEM_NAME, href="/", className="nav-link") ], className="ms-auto"),
        ], fluid=True),
        className="mb-0", sticky="top", style={"backgroundColor":"white"}
    )

def tile(label: str, emoji: str, href: str):
    return dcc.Link(
        html.Div([html.Span(emoji, className="circle"), html.Div(label, className="label")], className="cap-tile"),
        href=href, style={"textDecoration":"none","color":"inherit"}
    )

def left_capability_panel():
    return html.Div([
        html.H5("CAPACITY CONNECT"),
        dbc.Row([
            dbc.Col(tile("Planning Workspace","📅","/planning"), width=6),
            dbc.Col(tile("Budget","💰","/budget"), width=6),
            dbc.Col(tile("Operational Dashboard","📊","/ops"), width=6),
            dbc.Col(tile("New Hire Summary","🧑‍🎓","/newhire"), width=6),
        ], className="ghani"),
        dbc.Row([
            dbc.Col(tile("Employee Roster","🗂️","/roster"), width=6),
            dbc.Col(tile("Planner Dataset","🧮","/dataset"), width=6),
            dbc.Col(tile("Default Settings","⚙️","/settings"), width=6),
            dbc.Col(tile("Upload Shrinkage & Attrition","📤","/shrink"), width=6),
        ], className="ghani"),
    ], style={"padding":"12px","borderRadius":"12px","background":"#fff","boxShadow":"0 2px 8px rgba(0,0,0,.06)", "height": "100%"})

def center_projects_table():
    return html.Div([
        html.H5("Capacity Plans"),
        dash_table.DataTable(
            id="tbl-projects",
            data=projects_df.to_dict("records"),
            columns=pretty_columns(projects_df),
            style_as_list_view=True, page_size=10,
            style_table={"overflowX":"auto","maxWidth":"100%"},
            style_header={"textTransform": "none"},
        )
    ], style={"padding":"12px","borderRadius":"12px","background":"#fff","boxShadow":"0 2px 8px rgba(0,0,0,.06)", "height": "100%"})

def right_kpi_cards():
    return html.Div(id="right-kpis", children=[
        html.Div([
            html.Div("Staffing", className="kpi-head"),
            html.Div(className="kpi-topdash"),
            html.Div([
                html.Div([html.Div("0", className="num"), html.Div("Last Week", className="lbl")], className="cell"),
                html.Div([html.Div("0", className="num"), html.Div("This Week", className="lbl")], className="cell"),
                html.Div([html.Div(str(understaffed), className="num"), html.Div("Next Week", className="lbl")], className="cell"),
            ], className="kpi3"),
        ], className="kpi-card mb-3 edge-teal"),
        html.Div([
            html.Div("Hiring", className="kpi-head"),
            html.Div(className="kpi-topdash"),
            html.Div([
                html.Div([html.Div(str(int(hire_lw)), className="num"), html.Div("Last Week", className="lbl")], className="cell"),
                html.Div([html.Div(str(int(hire_tw)), className="num"), html.Div("This Week", className="lbl")], className="cell"),
                html.Div([html.Div(str(int(hire_nw)), className="num"), html.Div("Next Week", className="lbl")], className="cell"),
            ], className="kpi3"),
        ], className="kpi-card mb-3 edge-blue"),
        html.Div([
            html.Div("Shrinkage", className="kpi-head"),
            html.Div(className="kpi-topdash"),
            html.Div([
                html.Div([html.Div(f"{shr_last4:.2f}%", className="num"), html.Div("Last 4 Weeks", className="lbl")], className="cell"),
                html.Div([html.Div(f"{shr_next4:.2f}%", className="num"), html.Div("This Week", className="lbl")], className="cell"),
                html.Div([html.Div(f"{shr_next4:.2f}%", className="num"), html.Div("Next 4 Weeks", className="lbl")], className="cell"),
            ], className="kpi3"),
        ], className="kpi-card edge-orange"),
        html.Div([
            html.Div("Attrition", className="kpi-head"),
            html.Div(className="kpi-topdash"),
            html.Div([
                html.Div([html.Div(f"{attr_last4:.2f}%", className="num"), html.Div("Last 4 Weeks", className="lbl")], className="cell"),
                html.Div([html.Div(f"{attr_next4:.2f}%", className="num"), html.Div("This Week", className="lbl")], className="cell"),
                html.Div([html.Div(f"{attr_next4:.2f}%", className="num"), html.Div("Next 4 Weeks", className="lbl")], className="cell"),
            ], className="kpi3"),
        ], className="kpi-card edge-red"),
    ])

def sidebar_component(collapsed: bool) -> html.Div:
    items = [
        ("📅","Planning Workspace","/planning","sb-planning"),
        ("💰","Budget","/budget","sb-budget"),
        ("📊","Operational Dashboard","/ops","sb-ops"),
        ("🧑‍🎓","New Hire Summary","/newhire","sb-newhire"),
        ("🗂️","Employee Roster","/roster","sb-roster"),
        ("🧮","Planner Dataset","/dataset","sb-dataset"),
        ("⚙️","Default Settings","/settings","sb-settings"),
        ("📤","Upload Shrinkage & Attrition","/shrink","sb-shrink"),
    ]
    nav, tooltips = [], []
    for ico,lbl,href,anchor in items:
        nav.append(dcc.Link(
            html.Div([html.Div(ico, className="nav-ico"), html.Div(lbl, className="nav-label")],
                     className="nav-item", id=f"{anchor}-item"),
            href=href, id=anchor, refresh=False
        ))
        tooltips.append(dbc.Tooltip(lbl, target=f"{anchor}-item", placement="right", style={"fontSize":"0.85rem"}))
    return html.Div([
        html.Div([
            html.Div([ html.Img(src="/assets/barclays-wordmark.svg", alt="Barclays") ], className="logo-full"),
            html.Img(src="/assets/barclays-eagle.svg", alt="Barclays Eagle", className="logo-eagle"),
        ], className="brand"),
        html.Div(nav, className="nav"),
        *tooltips
    ], id="sidebar")

# ---------------------- PAGES ----------------------
def page_default_settings():
    base = load_defaults() or DEFAULT_SETTINGS
    return dbc.Container([
        header_bar(),

        # (Removed Mapping 1 stores/UI completely)

        dbc.Card(dbc.CardBody([
            html.H5("Default Settings — Scope"),
            dbc.Row([
                dbc.Col(dbc.RadioItems(
                    id="set-scope",
                    options=[
                        {"label":" Global","value":"global"},
                        {"label":" Location","value":"location"},
                        {"label":" Business Area ▶ Sub Business Area ▶ Channel","value":"hier"},
                    ],
                    value="global", style={"display": "flex", "gap": "10px"}
                ), md=12),
            ], className="mb-2"),
            dbc.Row([
                dbc.Col(dbc.Label("Location"), md=2),
                dbc.Col(dcc.Dropdown(id="set-location", placeholder="Select location (India, UK, ...)"), md=4),
                dbc.Col(html.Div(id="set-location-hint", className="text-muted small"), md=6),
            ], id="row-location", className="mb-1", style={"display":"none"}),
            dbc.Row([
                dbc.Col(dbc.Label("Business Area"), md=2),
                dbc.Col(dcc.Dropdown(id="set-ba", placeholder="Business Area"), md=3),
                dbc.Col(dcc.Dropdown(id="set-subba", placeholder="Sub Business Area"), md=3),
                dbc.Col(dcc.Dropdown(id="set-lob", placeholder="Channel"), md=3),
            ], id="row-hier", className="mb-1", style={"display":"none"}),
        ]), className="mb-3"),

        dbc.Card(dbc.CardBody([
            html.H5("Parameters"),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Interval Minutes (Voice)"),
                    dbc.Input(id="set-interval", type="number", min=5, max=120, step=5, value=int(base["interval_minutes"])),
                    dbc.Label("Work Hours per FTE / Day", className="mt-3"),
                    dbc.Input(id="set-hours", type="number", min=1, max=12, step=0.25, value=float(base["hours_per_fte"])),
                    dbc.Label("Shrinkage % (0–100)", className="mt-3"),
                    dbc.Input(id="set-shrink", type="number", min=0, max=100, step=0.5, value=float(base["shrinkage_pct"]*100)),
                ], md=4),
                dbc.Col([
                    html.Strong("Voice Targets – "),
                    dbc.Label("Service Level % (0–100)", className="mt-2"),
                    dbc.Input(id="set-sl", type="number", min=50, max=99, step=1, value=int(base["target_sl"]*100)),
                    dbc.Label("Service Level T (seconds)", className="mt-3"),
                    dbc.Input(id="set-slsec", type="number", min=1, max=120, step=1, value=int(base["sl_seconds"])),
                    dbc.Label("Max Occupancy % (Voice)", className="mt-3"),
                    dbc.Input(id="set-occ", type="number", min=60, max=100, step=1, value=int(base["occupancy_cap_voice"]*100)),
                ], md=4),
                dbc.Col([
                    html.Strong("Productivity – "),
                    dbc.Label("Utilization % (Back Office)", className="mt-2"),
                    dbc.Input(id="set-utilbo", type="number", min=50, max=100, step=1, value=int(base["util_bo"]*100)),
                    dbc.Label("Utilization % (Outbound)", className="mt-3"),
                    dbc.Input(id="set-utilob", type="number", min=50, max=100, step=1, value=int(base["util_ob"]*100)),
                    html.Div(id="settings-scope-note", className="text-muted small mt-3"),
                ], md=4),
            ], className="g-3"),
            html.Div([
                dbc.Button("Save Settings", id="btn-save-settings", color="primary", className="me-2"),
                html.Span(id="settings-save-msg", className="text-success ms-2")
            ], className="mt-3"),
        ])),

        # ====== CLUBBED TABS: Voice & Back Office only ======
        dbc.Card(dbc.CardBody([
            html.H5("Upload Volume & AHT/SUT (by scope)"),
            html.Div("Voice uses 30-min intervals; Back Office uses daily totals.", className="text-muted small mb-2"),
            dbc.Alert("Alert:- Uploads are saved to the selected scope. Even if your file includes Business Area, Sub Business Area, and Channel, please choose the scope above first.",color="light"),
            dbc.Tabs(id="vol-tabs", children=[

                # ---------- Voice (Forecast + Actual) ----------
                dcc.Tab(label="Voice", value="tab-voice", children=[
                    html.H6("Forecast"),
                    dbc.Row([
                        dbc.Col(dcc.Upload(
                            id="up-voice-forecast",
                            children=html.Div(["⬆️ Upload Voice Forecast (XLSX/CSV)"]),
                            multiple=False, className="upload-box"
                        ), md=6),
                        dbc.Col(dbc.Button("Save Voice Forecast", id="btn-save-voice-forecast", color="primary", className="w-100"), md=3),
                        dbc.Col(dbc.Button("Download Template", id="btn-dl-voice-forecast-tmpl", outline=True, color="secondary", className="w-100"), md=3),
                    ], className="my-2"),
                    dcc.Download(id="dl-voice-forecast-tmpl"),
                    dash_table.DataTable(
                        id="tbl-voice-forecast", page_size=10,
                        style_table={"overflowX":"auto"},
                        style_as_list_view=True, style_header={"textTransform":"none"}
                    ),
                    html.Div(id="voice-forecast-msg", className="text-success mt-1"),

                    html.Hr(),
                    html.H6("Actual"),
                    dbc.Row([
                        dbc.Col(dcc.Upload(
                            id="up-voice-actual",
                            children=html.Div(["⬆️ Upload Voice Actual (XLSX/CSV)"]),
                            multiple=False, className="upload-box"
                        ), md=6),
                        dbc.Col(dbc.Button("Save Voice Actual", id="btn-save-voice-actual", color="primary", className="w-100"), md=3),
                        dbc.Col(dbc.Button("Download Template", id="btn-dl-voice-actual-tmpl", outline=True, color="secondary", className="w-100"), md=3),
                    ], className="my-2"),
                    dcc.Download(id="dl-voice-actual-tmpl"),
                    dash_table.DataTable(
                        id="tbl-voice-actual", page_size=10,
                        style_table={"overflowX":"auto"},
                        style_as_list_view=True, style_header={"textTransform":"none"}
                    ),
                    html.Div(id="voice-actual-msg", className="text-success mt-1"),
                ]),

                # ---------- Back Office (Forecast + Actual) ----------
                dcc.Tab(label="Back Office", value="tab-bo", children=[
                    html.H6("Forecast"),
                    dbc.Row([
                        dbc.Col(dcc.Upload(
                            id="up-bo-forecast",
                            children=html.Div(["⬆️ Upload Back Office Forecast (XLSX/CSV)"]),
                            multiple=False, className="upload-box"
                        ), md=6),
                        dbc.Col(dbc.Button("Save BO Forecast", id="btn-save-bo-forecast", color="primary", className="w-100"), md=3),
                        dbc.Col(dbc.Button("Download Template", id="btn-dl-bo-forecast-tmpl", outline=True, color="secondary", className="w-100"), md=3),
                    ], className="my-2"),
                    dcc.Download(id="dl-bo-forecast-tmpl"),
                    dash_table.DataTable(
                        id="tbl-bo-forecast", page_size=10,
                        style_table={"overflowX":"auto"},
                        style_as_list_view=True, style_header={"textTransform":"none"}
                    ),
                    html.Div(id="bo-forecast-msg", className="text-success mt-1"),

                    html.Hr(),
                    html.H6("Actual"),
                    dbc.Row([
                        dbc.Col(dcc.Upload(
                            id="up-bo-actual",
                            children=html.Div(["⬆️ Upload Back Office Actual (XLSX/CSV)"]),
                            multiple=False, className="upload-box"
                        ), md=6),
                        dbc.Col(dbc.Button("Save BO Actual", id="btn-save-bo-actual", color="primary", className="w-100"), md=3),
                        dbc.Col(dbc.Button("Download Template", id="btn-dl-bo-actual-tmpl", outline=True, color="secondary", className="w-100"), md=3),
                    ], className="my-2"),
                    dcc.Download(id="dl-bo-actual-tmpl"),
                    dash_table.DataTable(
                        id="tbl-bo-actual", page_size=10,
                        style_table={"overflowX":"auto"},
                        style_as_list_view=True, style_header={"textTransform":"none"}
                    ),
                    html.Div(id="bo-actual-msg", className="text-success mt-1"),
                ]),
            ])
        ]), className="mb-3"),

        # ===== Headcount upload remains =====
        dbc.Card(dbc.CardBody([
            html.H5("Headcount Update — BRID Mapping"),
            html.Div("Upload the latest headcount file to keep BRID ⇄ Manager/Hierarchy mappings in sync.", className="text-muted small mb-2"),

            dbc.Row([
                dbc.Col(dcc.Upload(
                    id="up-headcount",
                    children=html.Div(["⬆️ Upload Headcount XLSX"]),
                    multiple=False, className="upload-box"
                ), md=6),
                dbc.Col(dbc.Button("Save Headcount", id="btn-save-headcount", color="primary", className="w-100"), md=3),
                dbc.Col(dbc.Button("Download Template", id="btn-dl-hc-template", outline=True, color="secondary", className="w-100"), md=3),
            ], className="my-2"),
            dcc.Download(id="dl-hc-template"),
            dash_table.DataTable(
                id="tbl-headcount-preview", page_size=8,
                style_table={"overflowX":"auto"}, style_as_list_view=True,
                style_header={"textTransform":"none"}
            ),
            html.Div(id="hc-msg", className="text-success mt-1"),
        ]), className="mb-3"),
    ], fluid=True)

def page_roster():
    df_wide_db = load_roster_wide()
    if df_wide_db is None or df_wide_db.empty:
        df_wide_db = pd.DataFrame()
    df_long_db = load_roster_long()
    if df_long_db is None or df_long_db.empty:
        df_long_db = pd.DataFrame()

    return dbc.Container([
        header_bar(),

        dcc.Store(id="roster_wide_store", data=(df_wide_db.to_dict("records") if not df_wide_db.empty else [])),
        dcc.Store(id="roster_long_store", data=(df_long_db.to_dict("records") if not df_long_db.empty else [])),

        dbc.Card(dbc.CardBody([
            html.H5("Download Roster Template"),
            dbc.Row([
                dbc.Col(dcc.DatePickerRange(
                    id="roster-template-dates",
                    className="date-compact",
                    display_format="YYYY-MM-DD"
                ), md=4),
                dbc.Col(dbc.Button("Download Empty Template (CSV)", id="btn-dl-roster-template", color="secondary", className="w-100"), md=4),
                dbc.Col(dbc.Button("Download Sample", id="btn-dl-roster-sample", outline=True, color="secondary", className="w-100"), md=4),
            ], className="my-2"),
            dcc.Download(id="dl-roster-template"),
            dcc.Download(id="dl-roster-sample"),
        ]), className="mb-3"),

        dbc.Card(dbc.CardBody([
            html.H5("Upload Filled Roster"),
            dbc.Row([
                dbc.Col(dcc.Upload(
                    id="up-roster-wide",
                    children=html.Div(["⬆️ Drag & drop or click to upload CSV"]),
                    multiple=False, className="upload-box"
                ), md=6),
                dbc.Col(dbc.Button("Save", id="btn-save-roster-wide", color="primary"), md=2),
                dbc.Col(html.Span(id="roster-save-msg", className="text-success"), md=4),
            ], className="my-2"),
            dash_table.DataTable(
                id="tbl-roster-wide",
                data=df_wide_db.to_dict("records"),
                columns=pretty_columns(df_wide_db if not df_wide_db.empty else
                    ["BRID","Name","Team Manager","Business Area","Sub Business Area","LOB","Site","Location","Country"]),
                editable=True, row_deletable=True, page_size=10,
                style_table={"overflowX":"auto"},
                style_as_list_view=True,
                style_header={"textTransform": "none"},
            ),
            html.Div(id="roster-wide-msg", className="text-muted mt-1"),
        ]), className="mb-3"),

        dbc.Card(dbc.CardBody([
            html.H5("Normalized Schedule Preview"),
            dbc.Row([
                dbc.Col(dcc.DatePickerRange(
                    id="roster-preview-dates",
                    className="date-compact",
                    display_format="YYYY-MM-DD"
                ), md=4),
            ], className="my-2"),
            dash_table.DataTable(
                id="tbl-roster-long",
                data=(df_long_db.to_dict("records") if not df_long_db.empty else []),
                columns=pretty_columns(df_long_db if not df_long_db.empty else
                    ["BRID","Name","Team Manager","Business Area","Sub Business Area","LOB","Site","Location","Country","date","entry","is_leave"]),
                page_size=12,
                style_table={"overflowX":"auto"},
                style_as_list_view=True,
                style_header={"textTransform": "none"},
            ),
        ]), className="mb-3"),

        dbc.Card(dbc.CardBody([
            html.H6("Bulk edit helpers"),
            dbc.Row([
                dbc.Col(dcc.DatePickerRange(
                    id="clear-range",
                    className="date-compact",
                    display_format="YYYY-MM-DD"
                ), md=5),
                dbc.Col(dcc.Dropdown(
                    id="clear-brids",
                    multi=True,
                    placeholder="Limit to BRIDs (optional)"
                ), md=4),
                dbc.Col(dcc.RadioItems(
                    id="clear-action",
                    options=[{"label": " Clear", "value": "blank"},
                             {"label": " Leave", "value": "Leave"},
                             {"label": " OFF", "value": "OFF"}],
                    value="blank",
                ), md=3),
            ], className="g-2"),
            dbc.Button("Apply to range", id="btn-apply-clear", color="danger", outline=True, className="mt-2"),
            html.Div(id="bulk-clear-msg", className="text-muted small mt-2")
        ]), className="mb-3"),

    ], fluid=True)

def page_new_hire():
    df = load_hiring()
    return dbc.Container([
        header_bar(),
        html.H4("New Hire Summary", className="ghanii"),
        dbc.Card(dbc.CardBody([
            dbc.Row([
                dbc.Col(dcc.Upload(id="up-hire", children=html.Div(["⬆️ Upload XLSX"]), multiple=False, className="upload-box"), md=4),
                dbc.Col(dbc.Button("Save", id="btn-save-hire", color="primary"), md=2),
                dbc.Col(html.Span(id="hire-save-msg", className="text-success"), md=3),
                dbc.Col(dbc.Button("Download Sample", id="btn-dl-hire", outline=True, color="secondary"), md=3),
            ], className="my-2"),
            dcc.Download(id="dl-hire-sample"),
            dash_table.DataTable(
                id="tbl-hire",
                columns=pretty_columns(df if not df.empty else ["start_week","fte","program"]),
                data=df.to_dict("records"),
                editable=True, row_deletable=True, page_size=10,
                style_table={"overflowX":"auto"}, 
                style_as_list_view=True,
                style_header={"textTransform":"none"},
            ),
            dcc.Graph(id="fig-hire", style={"height":"280px"}, config={"displayModeBar": False}),
        ])),
        
    ], fluid=True)


def page_shrink_attr():
    shr = load_shrinkage()
    att = load_attrition()
    return dbc.Container([
        header_bar(),
        dbc.Tabs(id="shr-tabs", active_tab="tab-shrink", children=[
            dbc.Tab(label="Shrinkage", tab_id="tab-shrink", children=[
                # inner tabs: Manual (existing), Back Office (Raw), Voice (Raw)
                dbc.Tabs(id="shr-inner-tabs", active_tab="inner-manual", children=[
                    dbc.Tab(label="Weekly Shrink %", tab_id="inner-manual", children=[
                        dbc.Row([
                            dbc.Col(dcc.Upload(id="up-shrink", children=html.Div(["⬆️ Drag & drop or click to upload XLSX"]), multiple=False, className="upload-box"), md=4),
                            dbc.Col(dbc.Button("Save", id="btn-save-shrink", color="primary"), md=2),
                            dbc.Col(html.Span(id="shr-save-msg", className="text-success"), md=3),
                            dbc.Col(dbc.Button("Download Sample", id="btn-dl-shrink", outline=True, color="secondary"), md=3),
                        ], className="my-2"),
                        dcc.Download(id="dl-shrink-sample"),
                        dash_table.DataTable(
                            id="tbl-shrink",
                            columns=pretty_columns(shr if not shr.empty else ["week","shrinkage_pct","program"]),
                            data=shr.to_dict("records"),
                            editable=True, row_deletable=True, page_size=10,
                            style_table={"overflowX":"auto"}, style_as_list_view=True,
                            style_header={"textTransform":"none"},
                        ),
                        dcc.Graph(id="fig-shrink", style={"height":"280px"}, config={"displayModeBar": False}),
                    ]),

                    dbc.Tab(label="Back Office — Shrinkage (Raw)", tab_id="inner-bo", children=[
                        dcc.Store(id="bo-shr-raw-store"),
                        dbc.Row([
                            dbc.Col(dcc.Upload(id="up-shr-bo-raw",
                                               children=html.Div(["⬆️ Upload Back Office Shrinkage (seconds) XLSX/CSV"]),
                                               multiple=False, className="upload-box"), md=6),
                            dbc.Col(dbc.Button("Save Back Office Shrinkage", id="btn-save-shr-bo-raw", color="primary", className="w-100"), md=3),
                            dbc.Col(dbc.Button("Download BO Template", id="btn-dl-shr-bo-template", outline=True, color="secondary", className="w-100"), md=3),
                        ], className="my-2"),
                        dcc.Download(id="dl-shr-bo-template"),
                        html.H6("Uploaded (normalized)"),
                        dash_table.DataTable(id="tbl-shr-bo-raw", page_size=8, style_table={"overflowX":"auto"},
                                             style_as_list_view=True, style_header={"textTransform":"none"}),
                        html.H6("Daily Summary (derived)"),
                        dash_table.DataTable(id="tbl-shr-bo-sum", page_size=8, style_table={"overflowX":"auto"},
                                             style_as_list_view=True, style_header={"textTransform":"none"}),
                        html.Div(id="bo-shr-save-msg", className="text-success mt-2"),
                    ]),

                    dbc.Tab(label="Voice — Shrinkage (Raw)", tab_id="inner-voice", children=[
                        dcc.Store(id="voice-shr-raw-store"),
                        dbc.Row([
                            dbc.Col(dcc.Upload(id="up-shr-voice-raw",
                                               children=html.Div(["⬆️ Upload Voice Shrinkage (HH:MM) XLSX/CSV"]),
                                               multiple=False, className="upload-box"), md=6),
                            dbc.Col(dbc.Button("Save Voice Shrinkage", id="btn-save-shr-voice-raw", color="primary", className="w-100"), md=3),
                            dbc.Col(dbc.Button("Download Voice Template", id="btn-dl-shr-voice-template", outline=True, color="secondary", className="w-100"), md=3),
                        ], className="my-2"),
                        dcc.Download(id="dl-shr-voice-template"),
                        html.H6("Uploaded (normalized)"),
                        dash_table.DataTable(id="tbl-shr-voice-raw", page_size=8, style_table={"overflowX":"auto"},
                                             style_as_list_view=True, style_header={"textTransform":"none"}),
                        html.H6("Daily Summary (derived)"),
                        dash_table.DataTable(id="tbl-shr-voice-sum", page_size=8, style_table={"overflowX":"auto"},
                                             style_as_list_view=True, style_header={"textTransform":"none"}),
                        html.Div(id="voice-shr-save-msg", className="text-success mt-2"),
                    ]),
                ])
            ]),

            dbc.Tab(label="Attrition", tab_id="tab-attr", children=[
                dbc.Row([
                    dbc.Col(dcc.Upload(id="up-attr", children=html.Div(["⬆️ Drag & drop or click to upload XLSX"]), multiple=False, className="upload-box"), md=4),
                    dbc.Col(dbc.Button("Save", id="btn-save-attr", color="primary"), md=2),
                    dbc.Col(html.Span(id="attr-save-msg", className="text-success"), md=3),
                    dbc.Col(dbc.Button("Download Leavers Sample", id="btn-dl-attr", outline=True, color="secondary"), md=3),
                ], className="my-2"),
                dcc.Download(id="dl-attr-sample"),
                dcc.Store(id="attr_raw_store"),
                dash_table.DataTable(
                    id="tbl-attr",
                    columns=pretty_columns(att if not att.empty else ["week","attrition_pct","program"]),
                    data=att.to_dict("records"),
                    editable=True, row_deletable=True, page_size=10,
                    style_table={"overflowX":"auto"}, style_as_list_view=True,
                    style_header={"textTransform":"none"},
                ),
                dcc.Graph(id="fig-attr", style={"height":"280px"}, config={"displayModeBar": False}),
            ]),
        ])
    ], fluid=True)

def page_budget():
    return dbc.Container([
        header_bar(),
        html.H4("Budgets", className="ghanii"),
        dbc.Card(dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Business Area"),
                    dcc.Dropdown(id="bud-ba", options=[], placeholder="Select Business Area"),
                ], md=4),
                dbc.Col([
                    html.Label("Sub Business Area"),
                    dcc.Dropdown(id="bud-subba", options=[], placeholder="Select Sub Business Area"),
                ], md=4),
                dbc.Col([
                    html.Label("Channel"),
                    dcc.Dropdown(
                        id="bud-channel",
                        options=[{"label": c, "value": c} for c in CHANNEL_LIST],
                        value="Voice",
                        placeholder="Select Channel"
                    ),
                ], md=4),
            ], className="gy-2 my-1"),

            dbc.Tabs(id="bud-tabs", active_tab="bud-voice", children=[

                # VOICE
                dbc.Tab(label="Voice budget", tab_id="bud-voice", children=[
                    dcc.Store(id="store-bud-voice"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Start week (Monday)"),
                            dcc.DatePickerSingle(id="bud-voice-start", display_format="YYYY-MM-DD")
                        ], md=3),
                        dbc.Col([
                            html.Label("Weeks"),
                            dbc.Input(id="bud-voice-weeks", type="number", value=8, min=1, step=1)
                        ], md=2),
                        dbc.Col(dbc.Button("Download Template", id="btn-bud-voice-tmpl", outline=True, className="mt-4"), md=3),
                        dbc.Col(dcc.Upload(id="up-bud-voice", children=html.Div(["⬆️ Upload CSV/XLSX"]), multiple=False,
                                        className="upload-box mt-4"), md=4),
                    ], className="my-2"),
                    dcc.Download(id="dl-bud-voice"),
                    dash_table.DataTable(
                        id="tbl-bud-voice",
                        page_size=10, editable=True, style_table={"overflowX":"auto"},
                        style_as_list_view=True, style_header={"textTransform":"none"}
                    ),
                    dbc.Row([
                        dbc.Col(dbc.Button("Save Voice Budget", id="btn-save-bud-voice", color="primary"), md=3),
                        dbc.Col(html.Span(id="msg-save-bud-voice", className="text-success"), md=9),
                    ], className="mt-2"),
                ]),

                # BACK OFFICE
                dbc.Tab(label="Back Office budget", tab_id="bud-bo", children=[
                    dcc.Store(id="store-bud-bo"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Start week (Monday)"),
                            dcc.DatePickerSingle(id="bud-bo-start", display_format="YYYY-MM-DD")
                        ], md=3),
                        dbc.Col([
                            html.Label("Weeks"),
                            dbc.Input(id="bud-bo-weeks", type="number", value=8, min=1, step=1)
                        ], md=2),
                        dbc.Col(dbc.Button("Download Template", id="btn-bud-bo-tmpl", outline=True, className="mt-4"), md=3),
                        dbc.Col(dcc.Upload(id="up-bud-bo", children=html.Div(["⬆️ Upload CSV/XLSX"]), multiple=False,
                                        className="upload-box mt-4"), md=4),
                    ], className="my-2"),
                    dcc.Download(id="dl-bud-bo"),
                    dash_table.DataTable(
                        id="tbl-bud-bo",
                        page_size=10, editable=True, style_table={"overflowX":"auto"},
                        style_as_list_view=True, style_header={"textTransform":"none"}
                    ),
                    dbc.Row([
                        dbc.Col(dbc.Button("Save Back Office Budget", id="btn-save-bud-bo", color="primary"), md=3),
                        dbc.Col(html.Span(id="msg-save-bud-bo", className="text-success"), md=9),
                    ], className="mt-2"),
                ]),
            ])
        ])),        
    ], fluid=True)


def page_dataset():
    roster = load_roster()
    hire   = load_hiring()
    sup = supply_fte_daily(roster if not roster.empty else roster_demo,
                           hire if not hire.empty else hiring_demo)
    df = pd.merge(req_df, sup, on=["date","program"], how="left").fillna({"supply_fte":0})
    df["staffing_pct"] = np.where(df["total_req_fte"]>0, (df["supply_fte"]/df["total_req_fte"])*100, np.nan)
    return dbc.Container([
        header_bar(),
        html.H5("Planner Dataset — Inputs Snapshot", className="mkc"),
        dash_table.DataTable(
            id="dataset-table",
            data=df.to_dict("records"),
            columns=pretty_columns(df),
            page_size=12, style_table={"overflowX":"auto"}, style_as_list_view=True,
            style_header={"textTransform": "none"},
        ),
        dcc.Graph(figure=px.line(df.groupby("date",as_index=False)[["total_req_fte","supply_fte"]].sum(),
                                 x="date", y=["total_req_fte","supply_fte"], markers=True), className="mt-3"),
    ], fluid=True)

# ---- Scope option chaining ------------------------------------------------------------------------
# Fill BA options from Headcount when visiting /budget (and pick the first by default)
@callback(
    Output("bud-ba", "options"),
    Output("bud-ba", "value"),
    Input("url-router", "pathname"),
    prevent_initial_call=False
)
def _bud_fill_ba(path):
    bas = _bas_from_headcount()  # from Headcount Update
    opts = [{"label": b, "value": b} for b in bas]
    return opts, (bas[0] if bas else None)

# Fill SubBA options when BA changes (from Headcount) & default first
@callback(
    Output("bud-subba", "options"),
    Output("bud-subba", "value"),
    Input("bud-ba", "value"),
    prevent_initial_call=False
)
def _bud_fill_subba(ba):
    if not ba:
        return [], None
    subs = _sbas_from_headcount(ba)
    opts = [{"label": s, "value": s} for s in subs]
    return opts, (subs[0] if subs else None)

# Channel is hardcoded; if you want a default other than Voice, set it here.
@callback(
    Output("bud-channel", "options"),
    Output("bud-channel", "value"),
    Input("bud-subba", "value"),
    State("bud-channel", "value"),
    prevent_initial_call=False
)
def _bud_fill_channels(_subba, current):
    opts = [{"label": c, "value": c} for c in CHANNEL_LIST]
    # keep existing selection if still valid, else default to "Voice"
    val = current if current in CHANNEL_LIST else "Voice"
    return opts, val


# ---- Load existing budgets when scope changes ----
@callback(
    Output("tbl-bud-voice","data", allow_duplicate=True),
    Output("tbl-bud-voice","columns", allow_duplicate=True),
    Output("store-bud-voice","data", allow_duplicate=True),
    Input("bud-ba","value"), Input("bud-subba","value"), Input("bud-channel","value"),
    prevent_initial_call=True
)
def load_voice_budget(ba, subba, channel):
    if not (ba and subba and channel): return [], [], None
    if (channel or "").lower() != "voice":  # still allow loading if user picks voice later
        return [], [], None
    key = _scope_key(ba, subba, "Voice")
    df = load_timeseries("voice_budget", key)
    if df is None or df.empty:
        return [], [], None
    return df.to_dict("records"), pretty_columns(df), df.to_dict("records")

@callback(
    Output("tbl-bud-bo","data", allow_duplicate=True),
    Output("tbl-bud-bo","columns", allow_duplicate=True),
    Output("store-bud-bo","data", allow_duplicate=True),
    Input("bud-ba","value"), Input("bud-subba","value"), Input("bud-channel","value"),
    prevent_initial_call=True
)
def load_bo_budget(ba, subba, channel):
    if not (ba and subba and channel): return [], [], None
    if (channel or "").lower() not in ("back office", "bo"):
        return [], [], None
    key = _scope_key(ba, subba, "Back Office")
    df = load_timeseries("bo_budget", key)
    if df is None or df.empty:
        return [], [], None
    return df.to_dict("records"), pretty_columns(df), df.to_dict("records")

# ---- Download templates ----
@callback(Output("dl-bud-voice","data"),
          Input("btn-bud-voice-tmpl","n_clicks"),
          State("bud-voice-start","date"), State("bud-voice-weeks","value"),
          prevent_initial_call=True)
def dl_voice_tmpl(_n, start_date, weeks):
    df = _budget_voice_template(start_date, int(weeks or 8))
    return dcc.send_data_frame(df.to_csv, "voice_budget_template.csv", index=False)

@callback(Output("dl-bud-bo","data"),
          Input("btn-bud-bo-tmpl","n_clicks"),
          State("bud-bo-start","date"), State("bud-bo-weeks","value"),
          prevent_initial_call=True)
def dl_bo_tmpl(_n, start_date, weeks):
    df = _budget_bo_template(start_date, int(weeks or 8))
    return dcc.send_data_frame(df.to_csv, "bo_budget_template.csv", index=False)

# ---- Upload / normalize ----
@callback(
    Output("tbl-bud-voice","data"),
    Output("tbl-bud-voice","columns"),
    Output("store-bud-voice","data"),
    Input("up-bud-voice","contents"),
    State("up-bud-voice","filename"),
    prevent_initial_call=False
)
def up_voice(contents, filename):
    df = parse_upload(contents, filename)
    dff = _budget_normalize_voice(df)
    if dff.empty: return [], [], None
    return dff.to_dict("records"), lock_variance_cols(pretty_columns(dff)), dff.to_dict("records")

@callback(
    Output("tbl-bud-bo","data"),
    Output("tbl-bud-bo","columns"),
    Output("store-bud-bo","data"),
    Input("up-bud-bo","contents"),
    State("up-bud-bo","filename"),
    prevent_initial_call=False
)
def up_bo(contents, filename):
    df = parse_upload(contents, filename)
    dff = _budget_normalize_bo(df)
    if dff.empty: return [], [], None
    return dff.to_dict("records"), lock_variance_cols(pretty_columns(dff)), dff.to_dict("records")

# ---- Save budgets ----
@callback(
    Output("msg-save-bud-voice","children"),
    Input("btn-save-bud-voice","n_clicks"),
    State("bud-ba","value"), State("bud-subba","value"), State("bud-channel","value"),
    State("store-bud-voice","data"),
    prevent_initial_call=True
)
def save_voice_budget(_n, ba, subba, channel, store):
    if not (ba and subba): return "Pick BA & Sub BA."
    dff = pd.DataFrame(store or [])
    if dff.empty: return "Nothing to save."
    key = _canon_scope(ba, subba, "Voice")   # ⬅️ canonical key
    save_timeseries("voice_budget", key, dff)
    _save_budget_hc_timeseries(key, dff)     # ⬅️ canonical key in HC mirror
    aht = dff[["week", "budget_aht_sec"]].rename(columns={"budget_aht_sec":"aht_sec"})
    save_timeseries("voice_planned_aht", key, aht)
    return f"Saved Voice budget for {key} ✓  ({len(dff)} rows)."

@callback(
    Output("msg-save-bud-bo","children"),
    Input("btn-save-bud-bo","n_clicks"),
    State("bud-ba","value"), State("bud-subba","value"), State("bud-channel","value"),
    State("store-bud-bo","data"),
    prevent_initial_call=True
)
def save_bo_budget(_n, ba, subba, channel, store):
    if not (ba and subba): return "Pick BA & Sub BA."
    dff = pd.DataFrame(store or [])
    if dff.empty: return "Nothing to save."
    key = _canon_scope(ba, subba, "Back Office")  # ⬅️ canonical key
    save_timeseries("bo_budget", key, dff)
    _save_budget_hc_timeseries(key, dff)          # ⬅️ canonical key in HC mirror
    sut = dff[["week", "budget_sut_sec"]].rename(columns={"budget_sut_sec":"sut_sec"})
    save_timeseries("bo_planned_sut", key, sut)
    return f"Saved Back Office budget for {key} ✓  ({len(dff)} rows)."

# ---------------------- Download templates (new clubbed) ----------------------
@callback(Output("dl-voice-forecast-tmpl","data"), Input("btn-dl-voice-forecast-tmpl","n_clicks"), prevent_initial_call=True)
def dl_voice_fc_tmpl(_n): return dcc.send_data_frame(voice_forecast_template_df().to_csv, "voice_forecast_template.csv", index=False)

@callback(Output("dl-voice-actual-tmpl","data"), Input("btn-dl-voice-actual-tmpl","n_clicks"), prevent_initial_call=True)
def dl_voice_ac_tmpl(_n): return dcc.send_data_frame(voice_actual_template_df().to_csv, "voice_actual_template.csv", index=False)

@callback(Output("dl-bo-forecast-tmpl","data"), Input("btn-dl-bo-forecast-tmpl","n_clicks"), prevent_initial_call=True)
def dl_bo_fc_tmpl(_n): return dcc.send_data_frame(bo_forecast_template_df().to_csv, "backoffice_forecast_template.csv", index=False)

@callback(Output("dl-bo-actual-tmpl","data"), Input("btn-dl-bo-actual-tmpl","n_clicks"), prevent_initial_call=True)
def dl_bo_ac_tmpl(_n): return dcc.send_data_frame(bo_actual_template_df().to_csv, "backoffice_actual_template.csv", index=False)

# ---------------------- Upload previews (clubbed) ----------------------
@callback(Output("tbl-voice-forecast","data"), Output("tbl-voice-forecast","columns"), Output("voice-forecast-msg","children", allow_duplicate=True),
          Input("up-voice-forecast","contents"), State("up-voice-forecast","filename"), prevent_initial_call=True)
def up_voice_forecast(contents, filename):
    df = parse_upload(contents, filename)
    if df.empty: return [], [], "Could not read file"

    L = {c.lower(): c for c in df.columns}
    # Minimal: Date, Interval Start, Volume ; Optional: AHT
    date_c     = L.get("date")
    ivl_c      = L.get("interval") or L.get("interval start") or L.get("interval_start")
    vol_c      = L.get("forecast volume") or L.get("volume")
    aht_c      = L.get("forecast aht") or L.get("aht")

    if not (date_c and ivl_c and vol_c):
        return [], [], "Need at least Date, Interval (or Interval Start), and Volume"

    dff = pd.DataFrame({
        "date": pd.to_datetime(df[date_c], errors="coerce").dt.date.astype(str),
        "interval": df[ivl_c].map(_coerce_time),
        "volume": pd.to_numeric(df[vol_c], errors="coerce").fillna(0.0),
    })
    if aht_c:
        dff["aht_sec"] = pd.to_numeric(df[aht_c], errors="coerce").apply(_minutes_to_seconds)
    else:
        dff["aht_sec"] = 300.0  # sensible default for missing AHT

    # keep BA/SBA/Channel columns for preview only (saving uses scope)
    for extra in ["Business Area","Sub Business Area","Channel"]:
        if extra in df.columns:
            dff[extra] = df[extra]

    cols = ["date","interval","volume","aht_sec","Business Area","Sub Business Area","Channel"]
    dff = dff[[c for c in cols if c in dff.columns]]

    return dff.to_dict("records"), pretty_columns(dff), f"Loaded {len(dff)} rows"

@callback(Output("tbl-voice-actual","data"), Output("tbl-voice-actual","columns"), Output("voice-actual-msg","children", allow_duplicate=True),
          Input("up-voice-actual","contents"), State("up-voice-actual","filename"), prevent_initial_call=True)
def up_voice_actual(contents, filename):
    df = parse_upload(contents, filename)
    if df.empty: return [], [], "Could not read file"

    L = {c.lower(): c for c in df.columns}
    date_c     = L.get("date")
    ivl_c      = L.get("interval") or L.get("interval start") or L.get("interval_start")
    vol_c      = L.get("actual volume") or L.get("volume")
    aht_c      = L.get("actual aht") or L.get("aht")

    if not (date_c and ivl_c and vol_c):
        return [], [], "Need at least Date, Interval (or Interval Start), and Volume"

    dff = pd.DataFrame({
        "date": pd.to_datetime(df[date_c], errors="coerce").dt.date.astype(str),
        "interval": df[ivl_c].map(_coerce_time),
        "volume": pd.to_numeric(df[vol_c], errors="coerce").fillna(0.0),
    })
    dff["aht_sec"] = pd.to_numeric(df[aht_c], errors="coerce").apply(_minutes_to_seconds) if aht_c else 300.0

    for extra in ["Business Area","Sub Business Area","Channel"]:
        if extra in df.columns:
            dff[extra] = df[extra]

    cols = ["date","interval","volume","aht_sec","Business Area","Sub Business Area","Channel"]
    dff = dff[[c for c in cols if c in dff.columns]]

    return dff.to_dict("records"), pretty_columns(dff), f"Loaded {len(dff)} rows"

@callback(Output("tbl-bo-forecast","data"), Output("tbl-bo-forecast","columns"), Output("bo-forecast-msg","children", allow_duplicate=True),
          Input("up-bo-forecast","contents"), State("up-bo-forecast","filename"), prevent_initial_call=True)
def up_bo_forecast(contents, filename):
    df = parse_upload(contents, filename)
    if df.empty: return [], [], "Could not read file"

    L = {c.lower(): c for c in df.columns}
    date_c = L.get("date")
    vol_c  = L.get("forecast volume") or L.get("volume") or L.get("items")
    sut_c  = L.get("forecast sut") or L.get("sut") or L.get("durationseconds")

    if not (date_c and vol_c and sut_c):
        return [], [], "Need Date, Volume/Items and SUT/DurationSeconds"

    dff = pd.DataFrame({
        "date": pd.to_datetime(df[date_c], errors="coerce").dt.date.astype(str),
        "items": pd.to_numeric(df[vol_c], errors="coerce").fillna(0.0),
        "sut_sec": pd.to_numeric(df[sut_c], errors="coerce").fillna(0.0),
    })

    for extra in ["Business Area","Sub Business Area","Channel"]:
        if extra in df.columns:
            dff[extra] = df[extra]

    cols = ["date","items","sut_sec","Business Area","Sub Business Area","Channel"]
    dff = dff[[c for c in cols if c in dff.columns]]

    return dff.to_dict("records"), pretty_columns(dff), f"Loaded {len(dff)} rows"

@callback(Output("tbl-bo-actual","data"), Output("tbl-bo-actual","columns"), Output("bo-actual-msg","children", allow_duplicate=True),
          Input("up-bo-actual","contents"), State("up-bo-actual","filename"), prevent_initial_call=True)
def up_bo_actual(contents, filename):
    df = parse_upload(contents, filename)
    if df.empty: return [], [], "Could not read file"

    L = {c.lower(): c for c in df.columns}
    date_c = L.get("date")
    vol_c  = L.get("actual volume") or L.get("volume") or L.get("items")
    sut_c  = L.get("actual sut") or L.get("sut") or L.get("durationseconds")

    if not (date_c and vol_c and sut_c):
        return [], [], "Need Date, Volume/Items and SUT/DurationSeconds"

    dff = pd.DataFrame({
        "date": pd.to_datetime(df[date_c], errors="coerce").dt.date.astype(str),
        "items": pd.to_numeric(df[vol_c], errors="coerce").fillna(0.0),
        "sut_sec": pd.to_numeric(df[sut_c], errors="coerce").fillna(0.0),
    })

    for extra in ["Business Area","Sub Business Area","Channel"]:
        if extra in df.columns:
            dff[extra] = df[extra]

    cols = ["date","items","sut_sec","Business Area","Sub Business Area","Channel"]
    dff = dff[[c for c in cols if c in dff.columns]]

    return dff.to_dict("records"), pretty_columns(dff), f"Loaded {len(dff)} rows"

# ---------------------- Save (per-scope, clubbed) ----------------------
def _scope_guard(scope, ba, sba, lob):
    if scope != "hier": return False, "Switch scope to Business Area ▶ Sub Business Area ▶ Channel."
    if not (ba and sba and lob): return False, "Pick BA, Sub BA and Channel first."
    return True, ""

@callback(Output("voice-forecast-msg","children", allow_duplicate=True),
          Input("btn-save-voice-forecast","n_clicks"),
          State("tbl-voice-forecast","data"),
          State("set-scope","value"), State("set-ba","value"), State("set-subba","value"), State("set-lob","value"),
          prevent_initial_call=True)
def save_voice_forecast(_n, preview_rows, scope, ba, sba, ch):
    ok, msg = _scope_guard(scope, ba, sba, ch)
    if not ok: return msg
    sk = _canon_scope(ba, sba, ch)
    df = pd.DataFrame(preview_rows or [])
    if df.empty:
        return 'No rows to save'
    # Voice forecast/actual: interval-level
    req_cols = {'date','interval','volume','aht_sec'}
    if not req_cols.issubset(df.columns):
        return 'Missing required columns for Voice (date/interval/volume/aht_sec)'
    vf = df[['date','interval','volume']].copy()
    af = df[['date','interval','aht_sec']].copy()
    from cap_store import save_timeseries
    save_timeseries('voice_forecast_volume', sk, vf)
    save_timeseries('voice_forecast_aht',    sk, af)
    return f"Saved VOICE forecast ({len(vf)} intervals) for {sk}"

@callback(Output("voice-actual-msg","children", allow_duplicate=True),
          Input("btn-save-voice-actual","n_clicks"),
          State("tbl-voice-actual","data"),
          State("set-scope","value"), State("set-ba","value"), State("set-subba","value"), State("set-lob","value"),
          prevent_initial_call=True)
def save_voice_actual(_n, preview_rows, scope, ba, sba, ch):
    ok, msg = _scope_guard(scope, ba, sba, ch)
    if not ok: return msg
    sk = _canon_scope(ba, sba, ch)
    df = pd.DataFrame(preview_rows or [])
    if df.empty:
        return 'No rows to save'
    # Voice forecast/actual: interval-level
    req_cols = {'date','interval','volume','aht_sec'}
    if not req_cols.issubset(df.columns):
        return 'Missing required columns for Voice (date/interval/volume/aht_sec)'
    vf = df[['date','interval','volume']].copy()
    af = df[['date','interval','aht_sec']].copy()
    from cap_store import save_timeseries
    save_timeseries('voice_actual_volume', sk, vf)
    save_timeseries('voice_actual_aht',    sk, af)
    return f"Saved VOICE actual ({len(vf)} intervals) for {sk}"

@callback(Output("bo-forecast-msg","children", allow_duplicate=True),
          Input("btn-save-bo-forecast","n_clicks"),
          State("tbl-bo-forecast","data"),
          State("set-scope","value"), State("set-ba","value"), State("set-subba","value"), State("set-lob","value"),
          prevent_initial_call=True)
def save_bo_forecast(_n, preview_rows, scope, ba, sba, ch):
    ok, msg = _scope_guard(scope, ba, sba, ch)
    if not ok: return msg
    sk = _canon_scope(ba, sba, ch)
    df = pd.DataFrame(preview_rows or [])
    if df.empty:
        return 'No rows to save'
    # BO forecast/actual: date-level
    if 'date' not in df or 'items' not in df or 'sut_sec' not in df:
        return 'Missing required columns (date/items/sut_sec)'
    vf = df[['date','items']].rename(columns={'items':'volume'})
    af = df[['date','sut_sec']].copy()
    from cap_store import save_timeseries
    save_timeseries('bo_forecast_volume', sk, vf)
    save_timeseries('bo_forecast_sut',    sk, af)
    return f"Saved BO forecast ({len(vf)} days) for {sk}"

@callback(Output("bo-actual-msg","children", allow_duplicate=True),
          Input("btn-save-bo-actual","n_clicks"),
          State("tbl-bo-actual","data"),
          State("set-scope","value"), State("set-ba","value"), State("set-subba","value"), State("set-lob","value"),
          prevent_initial_call=True)
def save_bo_actual(_n, preview_rows, scope, ba, sba, ch):
    ok, msg = _scope_guard(scope, ba, sba, ch)
    if not ok: return msg
    sk = _canon_scope(ba, sba, ch)
    df = pd.DataFrame(preview_rows or [])
    if df.empty:
        return 'No rows to save'
    # BO forecast/actual: date-level
    if 'date' not in df or 'items' not in df or 'sut_sec' not in df:
        return 'Missing required columns (date/items/sut_sec)'
    vf = df[['date','items']].rename(columns={'items':'volume'})
    af = df[['date','sut_sec']].copy()
    from cap_store import save_timeseries
    save_timeseries('bo_actual_volume', sk, vf)
    save_timeseries('bo_actual_sut',    sk, af)
    return f"Saved BO actual ({len(vf)} days) for {sk}"

@callback(
    Output("sidebar_collapsed","data"),
    Input("btn-burger-top","n_clicks"),
    State("sidebar_collapsed","data"),
    prevent_initial_call=True
)
def toggle_sidebar(n, collapsed):
    if not n: raise PreventUpdate
    return not bool(collapsed)

@callback(Output("app-wrapper","className"), Input("sidebar_collapsed","data"))
def set_wrapper_class(collapsed):
    return "sidebar-collapsed" if collapsed else "sidebar-expanded"

@callback(Output("sidebar","children"), Input("sidebar_collapsed","data"))
def render_sidebar(collapsed):
    return sidebar_component(bool(collapsed)).children

# ---------------------- Settings: Headcount-only dynamic sources ----------------------
# Populate BA + Location on entering settings
@callback(
    Output("set-ba","options"),
    Output("set-ba","value"),
    Output("set-location","options"),
    Output("set-location","value"),
    Input("url-router","pathname"),
    prevent_initial_call=False
)
def settings_enter(path):
    if (path or "").rstrip("/") != "/settings":
        raise PreventUpdate
    bas = _bas_from_headcount()
    locs = _all_locations()
    ba_val = bas[0] if bas else None
    loc_val = locs[0] if locs else None
    return (
        [{"label": b, "value": b} for b in bas], ba_val,
        [{"label": l, "value": l} for l in locs], loc_val
    )

# BA → Level 3/Sub BA
@callback(
    Output("set-subba","options"),
    Output("set-subba","value"),
    Input("set-ba","value"),
)
def settings_fill_sba(ba):
    sbas = _sbas_from_headcount(ba) if ba else []
    return [{"label": s, "value": s} for s in sbas], (sbas[0] if sbas else None)

# BA+SubBA → LOB/Channel
@callback(
    Output("set-lob","options"),
    Output("set-lob","value"),
    Input("set-ba","value"), Input("set-subba","value")
)
def settings_fill_lob(ba, sba):
    lobs = _lobs_for_ba_sba(ba, sba) if (ba and sba) else []
    return [{"label": l, "value": l} for l in lobs], (lobs[0] if lobs else None)

# Show/hide scope rows
@callback(Output("row-location","style"), Output("row-hier","style"), Input("set-scope","value"))
def scope_vis(scope):
    return ({"display":"flex"} if scope=="location" else {"display":"none"},
            {"display":"flex"} if scope=="hier" else {"display":"none"})

# Load settings for the chosen scope/key (unchanged)
@callback(
    Output("set-interval","value"),
    Output("set-hours","value"),
    Output("set-shrink","value"),
    Output("set-sl","value"),
    Output("set-slsec","value"),
    Output("set-occ","value"),
    Output("set-utilbo","value"),
    Output("set-utilob","value"),
    Output("settings-scope-note","children"),
    Input("set-scope","value"), Input("set-location","value"),
    Input("set-ba","value"), Input("set-subba","value"), Input("set-lob","value"),
    prevent_initial_call=False
)
def load_for_scope(scope, loc, ba, subba, lob):
    s = None
    note = ""
    if scope == "hier" and ba and subba and lob:
        s = resolve_settings(ba=ba, subba=subba, lob=lob)
        note = f"Scope: {ba} › {subba} › {lob}"
    elif scope == "location" and loc:
        s = resolve_settings(location=loc)
        note = f"Scope: Location = {loc}"
    else:
        s = load_defaults()
        note = "Scope: Global defaults"
    s = (s or DEFAULT_SETTINGS)

    return (
        int(s.get("interval_minutes", 30)),
        float(s.get("hours_per_fte", 8.0)),
        float(s.get("shrinkage_pct", 0.30)) * 100.0,
        float(s.get("target_sl", 0.80)) * 100.0,
        int(s.get("sl_seconds", 20)),
        float(s.get("occupancy_cap_voice", 0.85)) * 100.0,
        float(s.get("util_bo", 0.85)) * 100.0,
        float(s.get("util_ob", 0.85)) * 100.0,
        note,
    )

# Save per-scope (unchanged)
@callback(
    Output("settings-save-msg","children"),
    Input("btn-save-settings","n_clicks"),
    State("set-scope","value"), State("set-location","value"),
    State("set-ba","value"), State("set-subba","value"), State("set-lob","value"),
    State("set-interval","value"), State("set-hours","value"),
    State("set-shrink","value"), State("set-sl","value"), State("set-slsec","value"),
    State("set-occ","value"), State("set-utilbo","value"), State("set-utilob","value"),
    prevent_initial_call=True
)
def save_scoped(n, scope, loc, ba, subba, lob, ivl, hrs, shr, sl, slsec, occ, utilbo, utilob):
    if not n: raise PreventUpdate
    payload = dict(
        interval_minutes=int(ivl or 30),
        hours_per_fte=float(hrs or 8.0),
        shrinkage_pct=float(shr or 0)/100.0,
        target_sl=float(sl or 80)/100.0,
        sl_seconds=int(slsec or 20),
        occupancy_cap_voice=float(occ or 85)/100.0,
        util_bo=float(utilbo or 85)/100.0,
        util_ob=float(utilob or 85)/100.0,
    )
    if scope == "global":
        save_defaults(payload); return "Saved global defaults ✓"
    if scope == "location":
        if not loc: return "Select a location to save."
        save_scoped_settings("location", loc, payload); return f"Saved for location: {loc} ✓"
    if scope == "hier":
        if not (ba and subba and lob): return "Pick BA/SubBA/LOB to save."
        key = f"{ba}|{subba}|{lob}"
        save_scoped_settings("hier", key, payload); return f"Saved for {ba} › {subba} › {lob} ✓"
    return ""

# ---------------------- Roster (unchanged) ----------------------
def build_roster_template_wide(start_date: dt.date, end_date: dt.date, include_sample: bool = False) -> pd.DataFrame:
    base_cols = [
        "BRID", "Name", "Team Manager",
        "Business Area", "Sub Business Area", "LOB",
        "Site", "Location", "Country"
    ]
    if not isinstance(start_date, dt.date):
        start_date = pd.to_datetime(start_date).date()
    if not isinstance(end_date, dt.date):
        end_date = pd.to_datetime(end_date).date()
    if start_date > end_date:
        start_date, end_date = end_date, start_date

    date_cols = [(start_date + dt.timedelta(days=i)).isoformat()
                 for i in range((end_date - start_date).days + 1)]
    cols = base_cols + date_cols
    df = pd.DataFrame(columns=cols)

    if include_sample and date_cols:
        r1 = {c: "" for c in cols}
        r1.update({
            "BRID": "IN0001", "Name": "Asha Rao", "Team Manager": "Priyanka Menon",
            "Business Area": "Retail", "Sub Business Area": "Cards", "LOB": "Back Office",
            "Site": "Chennai", "Location": "IN-Chennai", "Country": "India",
            date_cols[0]: "09:00-17:30"
        })
        r2 = {c: "" for c in cols}
        r2.update({
            "BRID": "UK0002", "Name": "Alex Doe", "Team Manager": "Chris Lee",
            "Business Area": "Retail", "Sub Business Area": "Cards", "LOB": "Voice",
            "Site": "Glasgow", "Location": "UK-Glasgow", "Country": "UK",
            date_cols[0]: "Leave"
        })
        if len(date_cols) > 1:
            r1[date_cols[1]] = "10:00-18:00"
        df = pd.DataFrame([r1, r2])[cols]
    return df

def normalize_roster_wide(df_wide: pd.DataFrame) -> pd.DataFrame:
    if df_wide is None or df_wide.empty:
        return pd.DataFrame(columns=[
            "BRID","Name","Team Manager","Business Area","Sub Business Area",
            "LOB","Site","Location","Country","date","entry"
        ])
    id_cols = ["BRID","Name","Team Manager","Business Area","Sub Business Area","LOB","Site","Location","Country"]
    id_cols = [c for c in id_cols if c in df_wide.columns]
    date_cols = [c for c in df_wide.columns if c not in id_cols]
    long = df_wide.melt(id_vars=id_cols, value_vars=date_cols, var_name="date", value_name="entry")
    long["entry"] = long["entry"].fillna("").astype(str).str.strip()
    long = long[long["entry"] != ""]
    long["date"] = pd.to_datetime(long["date"], errors="coerce", dayfirst=True).dt.date
    long = long[pd.notna(long["date"])]
    long["is_leave"] = long["entry"].str.lower().isin({"leave","l","off","pto"})
    return long

@callback(
    Output("dl-roster-template","data"),
    Input("btn-dl-roster-template","n_clicks"),
    State("roster-template-dates","start_date"),
    State("roster-template-dates","end_date"),
    prevent_initial_call=True
)
def dl_roster_tmpl(n, sd, ed):
    df = build_roster_template_wide(sd, ed, include_sample=False)
    s, e = pd.to_datetime(sd).date(), pd.to_datetime(ed).date()
    return dcc.send_data_frame(df.to_csv, f"roster_template_{s}_{e}.csv", index=False)

@callback(
    Output("dl-roster-sample","data"),
    Input("btn-dl-roster-sample","n_clicks"),
    State("roster-template-dates","start_date"),
    State("roster-template-dates","end_date"),
    prevent_initial_call=True
)
def dl_roster_sample(n, sd, ed):
    df = build_roster_template_wide(sd, ed, include_sample=True)
    s, e = pd.to_datetime(sd).date(), pd.to_datetime(ed).date()
    return dcc.send_data_frame(df.to_csv, f"roster_sample_{s}_{e}.csv", index=False)

@callback(
    Output("tbl-roster-wide","data"),
    Output("tbl-roster-wide","columns"),
    Output("roster_wide_store","data"),
    Output("tbl-roster-long","data"),
    Output("tbl-roster-long","columns"),
    Output("roster_long_store","data"),
    Output("roster-wide-msg","children"),
    Input("up-roster-wide","contents"),
    State("up-roster-wide","filename"),
    prevent_initial_call=True
)
def on_upload_roster(contents, filename):
    df = parse_upload(contents, filename)
    if df.empty:
        raise PreventUpdate
    df = enrich_with_manager(df)
    long = normalize_roster_wide(df)
    msg = f"Loaded {len(df)} rows. Normalized rows: {len(long)}."
    return (
        df.to_dict("records"), pretty_columns(df), df.to_dict("records"),
        long.to_dict("records"), pretty_columns(long), long.to_dict("records"),
        msg
    )

@callback(
    Output("tbl-roster-long", "data", allow_duplicate=True),
    Input("roster-preview-dates","start_date"),
    Input("roster-preview-dates","end_date"),
    State("roster_long_store","data"),
    prevent_initial_call=True
)
def filter_long_for_preview(sd, ed, store):
    base = pd.DataFrame(store or [])
    if base.empty or not sd or not ed:
        raise PreventUpdate
    sd = pd.to_datetime(sd).date()
    ed = pd.to_datetime(ed).date()
    df = base.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df = df[(df["date"] >= sd) & (df["date"] <= ed)]
    df["is_leave"] = df.get("entry","").astype(str).str.lower().isin({"leave","l","off","pto"})
    return df.to_dict("records")

@callback(
    Output("roster-save-msg","children"),
    Input("btn-save-roster-wide","n_clicks"),
    State("tbl-roster-wide","data"),
    State("tbl-roster-long","data"),
    prevent_initial_call=True
)
def save_roster_wide_and_long(n, rows_wide, rows_long):
    dfw = pd.DataFrame(rows_wide or [])
    dfl = pd.DataFrame(rows_long or [])
    dfw = enrich_with_manager(dfw)
    dfl = enrich_with_manager(dfl)
    save_roster_wide(dfw)
    save_roster_long(dfl)
    return "Saved ✓ (wide + normalized)"

@callback(
    Output("clear-brids", "options"),
    Input("tbl-roster-long", "data"),
    prevent_initial_call=False
)
def _fill_brid_options(rows):
    df = pd.DataFrame(rows or [])
    if df.empty:
        return []
    brid_col = "BRID" if "BRID" in df.columns else ("brid" if "brid" in df.columns else None)
    if not brid_col:
        return []
    vals = sorted(df[brid_col].dropna().astype(str).unique().tolist())
    return [{"label": v, "value": v} for v in vals]

@callback(
    Output("tbl-roster-long", "data", allow_duplicate=True),
    Output("roster_long_store", "data", allow_duplicate=True),
    Output("bulk-clear-msg", "children"),
    Input("btn-apply-clear", "n_clicks"),
    State("tbl-roster-long", "data"),
    State("clear-range", "start_date"),
    State("clear-range", "end_date"),
    State("clear-brids", "value"),
    State("clear-action", "value"),
    prevent_initial_call=True
)
def apply_bulk_clear(n, rows, start, end, brids, action):
    if not n or not start or not end:
        raise PreventUpdate
    df = pd.DataFrame(rows or [])
    if df.empty or "date" not in df.columns:
        raise PreventUpdate

    date_ser = pd.to_datetime(df["date"], errors="coerce").dt.date
    target_col = "entry" if "entry" in df.columns else ("value" if "value" in df.columns else None)
    if target_col is None:
        raise PreventUpdate

    mask = (date_ser >= pd.to_datetime(start).date()) & (date_ser <= pd.to_datetime(end).date())
    brid_col = "BRID" if "BRID" in df.columns else ("brid" if "brid" in df.columns else None)
    if brids and brid_col:
        mask &= df[brid_col].astype(str).isin([str(b) for b in brids])

    edits = int(mask.sum())
    if edits == 0:
        return rows, rows, "No matching rows in range."

    if action == "blank":
        df.loc[mask, target_col] = ""
        msg = f"Cleared {edits} cells."
    else:
        df.loc[mask, target_col] = action
        msg = f"Set '{action}' on {edits} cells."

    df["is_leave"] = df[target_col].astype(str).str.lower().isin({"leave","l","off","pto"})
    updated = df.to_dict("records")
    return updated, updated, msg

# ---------------------- New Hire / Shrinkage / Attrition (unchanged) ----------------------
@callback(
    Output("tbl-hire","data"), Output("tbl-hire","columns"),
    Output("hire-save-msg","children"), Output("fig-hire","figure"),
    Input("up-hire","contents"), State("up-hire","filename"),
    Input("btn-save-hire","n_clicks"),
    State("tbl-hire","data"),
    prevent_initial_call=True
)
def hire_upload_save(contents, filename, n, rows):
    ctx = dash.callback_context  # type: ignore
    trigger = ctx.triggered[0]["prop_id"] if ctx.triggered else ""
    if "up-hire" in trigger:
        df = parse_upload(contents, filename)
        if df.empty: raise PreventUpdate
        return df.to_dict("records"), pretty_columns(df), "", {}
    df = pd.DataFrame(rows or [])
    save_hiring(df)
    fig = px.bar(df, x="start_week", y="fte", color=("program" if "program" in df.columns else None), title="Hiring by Week") if not df.empty else {}
    return rows, pretty_columns(df if not df.empty else []), "Saved ✓", fig

@callback(
    Output("tbl-shrink","data"), Output("tbl-shrink","columns"),
    Input("up-shrink","contents"), State("up-shrink","filename"),
    prevent_initial_call=True
)
def shrink_upload(contents, filename):
    df = parse_upload(contents, filename)
    if df.empty: raise PreventUpdate
    return df.to_dict("records"), pretty_columns(df)

@callback(
    Output("shr-save-msg","children"), Output("fig-shrink","figure"),
    Input("btn-save-shrink","n_clicks"), State("tbl-shrink","data"),
    prevent_initial_call=True
)
def shrink_save(n, rows):
    df = pd.DataFrame(rows or [])
    if not df.empty:
        save_shrinkage(df[["week","shrinkage_pct","program"]] if set(["week","shrinkage_pct","program"]).issubset(df.columns) else df)
        fig = px.line(df, x="week", y="shrinkage_pct", color=("program" if "program" in df.columns else None), markers=True, title="Shrinkage %")
        return "Saved ✓", fig
    return "Saved ✓ (empty)", {}

def _week_floor(d: pd.Timestamp | str | dt.date, week_start: str = "Monday") -> dt.date:
    d = pd.to_datetime(d).date()
    wd = d.weekday()
    if (week_start or "Monday").lower().startswith("sun"):
        return d - dt.timedelta(days=(wd + 1) % 7)
    return d - dt.timedelta(days=wd)

def weekly_avg_active_fte_from_roster(week_start: str = "Monday") -> pd.DataFrame:
    roster = load_roster()
    if isinstance(roster, pd.DataFrame) and (not roster.empty) and {"start_date","fte"}.issubset(roster.columns):
        def _to_date(x):
            try: return pd.to_datetime(x).date()
            except Exception: return None
        r = roster.copy()
        r["sd"] = r["start_date"].apply(_to_date)
        r["ed"] = (r["end_date"] if "end_date" in r.columns else pd.Series([None]*len(r))).apply(_to_date)
        sd_min = min([d for d in r["sd"].dropna()] or [dt.date.today()])
        ed_max = max([d for d in r["ed"].dropna()] or [dt.date.today() + dt.timedelta(days=180)])
        if ed_max < sd_min: ed_max = sd_min + dt.timedelta(days=180)
        days = pd.date_range(sd_min, ed_max, freq="D").date
        rows = []
        for _, row in r.iterrows():
            sd = row["sd"] or sd_min
            ed = row["ed"] or ed_max
            fte = float(row.get("fte", 0) or 0)
            if fte <= 0: continue
            start, end = max(sd, sd_min), min(ed, ed_max)
            for d in days:
                if start <= d <= end:
                    rows.append({"date": d, "fte": fte})
        if not rows:
            return pd.DataFrame(columns=["week","avg_active_fte"])
        daily = pd.DataFrame(rows).groupby("date", as_index=False)["fte"].sum()
        daily["week"] = daily["date"].apply(lambda x: _week_floor(x, week_start))
        weekly = daily.groupby("week", as_index=False)["fte"].mean().rename(columns={"fte":"avg_active_fte"})
        return weekly.sort_values("week")

    try:
        long = load_roster_long()
    except Exception:
        long = None
    if long is None or long.empty or "date" not in long.columns:
        return pd.DataFrame(columns=["week","avg_active_fte"])
    df = long.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=["date"])
    id_col = "BRID" if "BRID" in df.columns else ("employee_id" if "employee_id" in df.columns else None)
    if not id_col:
        return pd.DataFrame(columns=["week","avg_active_fte"])
    daily = df.groupby(["date"], as_index=False)[id_col].nunique().rename(columns={id_col:"fte"})
    daily["week"] = daily["date"].apply(lambda x: _week_floor(x, week_start))
    weekly = daily.groupby("week", as_index=False)["fte"].mean().rename(columns={"fte":"avg_active_fte"})
    return weekly.sort_values("week")

def attrition_weekly_from_raw(df_raw: pd.DataFrame, week_start: str = "Monday") -> pd.DataFrame:
    if df_raw is None or df_raw.empty:
        return pd.DataFrame(columns=["week","leavers_fte","avg_active_fte","attrition_pct","program"])
    df = df_raw.copy()
    if "Resignation Date" not in df.columns:
        if "Reporting Full Date" in df.columns:
            df["Resignation Date"] = df["Reporting Full Date"]
        else:
            return pd.DataFrame(columns=["week","leavers_fte","avg_active_fte","attrition_pct","program"])
    df = df[~df["Resignation Date"].isna()].copy()
    if "FTE" not in df.columns:
        df["FTE"] = 1.0

    program_series = None
    try:
        hc = load_headcount()
    except Exception:
        hc = pd.DataFrame()

    if "BRID" in df.columns and isinstance(hc, pd.DataFrame) and not hc.empty and "journey" in hc.columns:
        map_brid_to_j = dict(zip(hc["brid"].astype(str), hc["journey"].astype(str)))
        program_series = df["BRID"].astype(str).map(lambda x: map_brid_to_j.get(x))

    if program_series is None or program_series.isna().all():
        raw_l2_col = None
        for c in df.columns:
            cl = str(c).strip().lower()
            if cl in ("imh 06","imh l06","imh l 06","imh06","level 2","level_2"):
                raw_l2_col = c
                break
        if raw_l2_col is not None:
            try:
                l2_map = level2_to_journey_map()
            except Exception:
                l2_map = {}
            if l2_map:
                program_series = df[raw_l2_col].astype(str).map(lambda x: l2_map.get(str(x).strip()))

    if program_series is None:
        lower = df.columns.str.lower()
        if any(lower.isin(["org unit","business area","journey"])):
            pick_col = df.columns[lower.isin(["org unit","business area","journey"])][0]
            program_series = df[pick_col]
        else:
            program_series = pd.Series(["All"] * len(df))

    df["program"] = program_series.fillna("All").astype(str)
    df["week"] = df["Resignation Date"].apply(lambda x: _week_floor(x, week_start))
    wk = df.groupby(["week","program"], as_index=False)["FTE"].sum().rename(columns={"FTE":"leavers_fte"})
    s = load_defaults() or {}
    wkstart = s.get("week_start","Monday") or week_start
    den = weekly_avg_active_fte_from_roster(week_start=wkstart)
    out = wk.merge(den, on="week", how="left")
    out["attrition_pct"] = (out["leavers_fte"] / out["avg_active_fte"].replace({0:np.nan})) * 100.0
    out["attrition_pct"] = out["attrition_pct"].round(2)
    keep = ["week","leavers_fte","avg_active_fte","attrition_pct","program"]
    for k in keep:
        if k not in out.columns: out[k] = np.nan if k!="program" else "All"
    return out[keep].sort_values(["week","program"])

@callback(
    Output("tbl-attr","data", allow_duplicate=True),
    Output("tbl-attr","columns", allow_duplicate=True),
    Output("attr_raw_store","data"),
    Input("up-attr","contents"),
    State("up-attr","filename"),
    prevent_initial_call=True
)
def attr_upload(contents, filename):
    df = parse_upload(contents, filename)
    if df.empty: raise PreventUpdate
    looks_raw = ("Resignation Date" in df.columns) or ("Reporting Full Date" in df.columns)
    if looks_raw:
        s = load_defaults() or {}
        wkstart = s.get("week_start","Monday")
        wk = attrition_weekly_from_raw(df, week_start=wkstart)
        return wk.to_dict("records"), pretty_columns(wk), df.to_dict("records")
    return df.to_dict("records"), pretty_columns(df), None

@callback(
    Output("attr-save-msg","children"),
    Output("fig-attr","figure"),
    Input("btn-save-attr","n_clicks"),
    State("tbl-attr","data"),
    State("attr_raw_store","data"),
    prevent_initial_call=True
)
def attr_save(n, rows, raw_store):
    df = pd.DataFrame(rows or [])
    if df.empty: 
        if raw_store:
            save_attrition_raw(pd.DataFrame(raw_store))
            return "Saved ✓ (weekly empty) + raw ✓", {}
        return "Saved ✓ (empty)", {}

    if "attrition_pct" not in df.columns and {"leavers_fte","avg_active_fte"}.issubset(df.columns):
        df = df.copy()
        df["attrition_pct"] = (df["leavers_fte"] / df["avg_active_fte"].replace({0:np.nan})) * 100.0
    if "program" not in df.columns:
        df["program"] = "All"
    if "week" in df.columns:
        df["week"] = pd.to_datetime(df["week"]).dt.date.astype(str)

    keep = [c for c in ["week","attrition_pct","program"] if c in df.columns]
    save_attrition(df[keep])

    raw_msg = ""
    if raw_store:
        raw_df = pd.DataFrame(raw_store or [])
        if not raw_df.empty:
            save_attrition_raw(raw_df)
            raw_msg = " + raw ✓"

    fig = px.line(df, x="week", y="attrition_pct",
                  color=("program" if "program" in df.columns else None),
                  markers=True, title="Attrition %")
    return f"Saved ✓{raw_msg}", fig

# 1) Sample template (static columns you expect from leavers file)
@app.callback(
    Output("dl-attr-sample", "data"),
    Input("btn-dl-attr", "n_clicks"),
    prevent_initial_call=True
)
def download_leavers_sample(n):
    if not n:
        raise PreventUpdate

    cols = [
        "Reporting Full Date","BRID","Employee Name","Operational Status",
        "Corporate Grade Description","Employee Email Address","Employee Position",
        "Position Description","Employee Line Manager Indicator","Length of Service Date",
        "Cost Centre","Line Manager BRID","Line Manager Name","IMH L05","IMH L06",
        "Employee Line Manager lvl 01","Employee Line Manager lvl 02","Employee Line Manager lvl 03",
        "Employee Line Manager lvl 04","Employee Line Manager lvl 05","Employee Line Manager lvl 06",
        "Employee Line Manager lvl 07","Employee Line Manager lvl 08","Employee Line Manager lvl 09",
        "City","Building","Gender Description","Voluntary Involuntary Exit Description",
        "Resignation Date","Employee Contract","HC","HC FTE"
    ]

    df = pd.DataFrame(columns=cols)
    return dcc.send_data_frame(df.to_csv, "leavers_sample.csv", index=False)

# 2) Raw (whatever the user uploaded and you stored in attr_raw_store)
@app.callback(
    Output("dl-attr-raw", "data"),
    Input("btn-dl-attr-raw", "n_clicks"),
    State("attr_raw_store", "data"),
    prevent_initial_call=True
)
def dl_attr_raw(_n, raw):
    df = pd.DataFrame(raw or [])
    if df.empty: raise dash.exceptions.PreventUpdate
    return dcc.send_data_frame(df.to_csv, "attrition_raw.csv", index=False)


# ---------------------- Headcount upload/preview/save ----------------------
@callback(
    Output("tbl-headcount-preview","data"),
    Output("tbl-headcount-preview","columns"),
    Output("hc-msg","children"),
    Input("up-headcount","contents"),
    State("up-headcount","filename"),
    prevent_initial_call=False
)
def hc_preview(contents, filename):
    df = parse_upload(contents, filename)
    if df is None or df.empty:
        return [], [], ""
    wanted = ["BRID","Full Name","Line Manager BRID","Line Manager Full Name",
              "Business Area","Sub Business Area","LOB","Site","Location"]
    cols = [c for c in df.columns if str(c).strip() in wanted] or list(df.columns)[:12]
    dff = df[cols].copy()
    return dff.to_dict("records"), pretty_columns(dff), f"Preview loaded: {len(df)} rows"

@callback(
    Output("hc-msg","children", allow_duplicate=True),
    Input("btn-save-headcount","n_clicks"),
    State("up-headcount","contents"),
    State("up-headcount","filename"),
    prevent_initial_call=True
)
def hc_save(n, contents, filename):
    if not n:
        raise PreventUpdate
    df = parse_upload(contents, filename)
    if df is None or df.empty:
        return "No data to save."
    try:
        from cap_store import save_headcount_df
        count = save_headcount_df(df)
        return f"Saved headcount: {count} rows."
    except Exception as e:
        return f"Error saving headcount: {e}"

@callback(Output("dl-hc-template","data"),
          Input("btn-dl-hc-template","n_clicks"),
          prevent_initial_call=True)
def dl_hc_tmpl(_n):
    return dcc.send_data_frame(headcount_template_df().to_csv, "headcount_template.csv", index=False)

# === Shrinkage RAW: Download templates ===
@callback(Output("dl-shr-bo-template","data"),
          Input("btn-dl-shr-bo-template","n_clicks"), prevent_initial_call=True)
def dl_shr_bo_tmpl(_n):
    return dcc.send_data_frame(shrinkage_bo_raw_template_df().to_csv, "shrinkage_backoffice_raw_template.csv", index=False)

@callback(Output("dl-shr-voice-template","data"),
          Input("btn-dl-shr-voice-template","n_clicks"), prevent_initial_call=True)
def dl_shr_voice_tmpl(_n):
    return dcc.send_data_frame(shrinkage_voice_raw_template_df().to_csv, "shrinkage_voice_raw_template.csv", index=False)

# === Shrinkage RAW: Upload/preview/summary (Back Office) ===
@callback(
    Output("tbl-shr-bo-raw","data"),
    Output("tbl-shr-bo-raw","columns"),
    Output("tbl-shr-bo-sum","data"),
    Output("tbl-shr-bo-sum","columns"),
    Output("bo-shr-raw-store","data"),
    Input("up-shr-bo-raw","contents"),
    State("up-shr-bo-raw","filename"),
    prevent_initial_call=True
)
def up_shr_bo(contents, filename):
    df = parse_upload(contents, filename)
    dff = normalize_shrinkage_bo(df)
    if dff.empty:
        return [], [], [], [], None
    summ = summarize_shrinkage_bo(dff)
    return (
        dff.to_dict("records"), lock_variance_cols(pretty_columns(dff)),
        summ.to_dict("records"), lock_variance_cols(pretty_columns(summ)),
        dff.to_dict("records")
    )

@callback(
    Output("bo-shr-save-msg","children"),
    Input("btn-save-shr-bo-raw","n_clicks"),
    State("bo-shr-raw-store","data"),
    prevent_initial_call=True
)
def save_shr_bo(_n, store):
    dff = pd.DataFrame(store or [])
    if dff.empty: return "Nothing to save."
    # Save raw
    save_df("shrinkage_raw_backoffice", dff)
    # Also compute + save weekly % (to keep your existing shrinkage chart working)
    daily = summarize_shrinkage_bo(dff)
    weekly = weekly_shrinkage_from_bo_summary(daily)
    base = load_shrinkage()
    merged = pd.concat([base, weekly], ignore_index=True) if isinstance(base, pd.DataFrame) and not base.empty else weekly
    save_shrinkage(merged)
    return f"Saved Back Office shrinkage ✓  (raw rows: {len(dff)}, weekly points: {len(weekly)})"

# === Shrinkage RAW: Upload/preview/summary (Voice) ===
@callback(
    Output("tbl-shr-voice-raw","data"),
    Output("tbl-shr-voice-raw","columns"),
    Output("tbl-shr-voice-sum","data"),
    Output("tbl-shr-voice-sum","columns"),
    Output("voice-shr-raw-store","data"),
    Input("up-shr-voice-raw","contents"),
    State("up-shr-voice-raw","filename"),
    prevent_initial_call=True
)
def up_shr_voice(contents, filename):
    df = parse_upload(contents, filename)
    dff = normalize_shrinkage_voice(df)
    if dff.empty:
        return [], [], [], [], None
    summ = summarize_shrinkage_voice(dff)
    return (
        dff.to_dict("records"), lock_variance_cols(pretty_columns(dff)),
        summ.to_dict("records"), lock_variance_cols(pretty_columns(summ)),
        dff.to_dict("records")
    )

@callback(
    Output("voice-shr-save-msg","children"),
    Input("btn-save-shr-voice-raw","n_clicks"),
    State("voice-shr-raw-store","data"),
    prevent_initial_call=True
)
def save_shr_voice(_n, store):
    dff = pd.DataFrame(store or [])
    if dff.empty: return "Nothing to save."
    # Save raw
    save_df("shrinkage_raw_voice", dff)
    # Weekly % into main shrinkage series (program = Business Area)
    daily = summarize_shrinkage_voice(dff)
    weekly = weekly_shrinkage_from_voice_summary(daily)
    base = load_shrinkage()
    merged = pd.concat([base, weekly], ignore_index=True) if isinstance(base, pd.DataFrame) and not base.empty else weekly
    save_shrinkage(merged)
    return f"Saved Voice shrinkage ✓  (raw rows: {len(dff)}, weekly points: {len(weekly)})"

# ---------------------- Router + Home ----------------------
@callback(
    Output("tbl-projects", "data"),
    Input("cap-plans-refresh", "n_intervals"),
    State("url-router", "pathname"),
    prevent_initial_call=False
)
def _refresh_projects_table(_n, pathname):
    path = (pathname or "").rstrip("/")
    if path not in ("", "/"):
        raise PreventUpdate
    df = make_projects_sample()
    return df.to_dict("records")

def home_layout():
    return dbc.Container([
        header_bar(),
        dbc.Row([
            dbc.Col(left_capability_panel(), width=5),
            dbc.Col(center_projects_table(), width=4),
            dbc.Col(right_kpi_cards(), width=3, className="col-kpis"),
        ], className="g-3"),
    ], fluid=True)

def not_found_layout():
    return dbc.Container([header_bar(), dbc.Alert("Page not found.", color="warning"), dcc.Link("← Home", href="/")], fluid=True)

@app.callback(Output("root","children"), Input("url-router","pathname"))
def route(pathname: str):
    path = (pathname or "").rstrip("/")

    if path.startswith("/plan/"):
        try:
            pid = int(path.rsplit("/", 1)[-1])
        except Exception:
            return not_found_layout()
        return dbc.Container([header_bar(), layout_for_plan(pid)], fluid=True)

    if path in ("/", None, ""):
        return home_layout()

    pages = {
        "/settings": page_default_settings,
        "/roster":   page_roster,
        "/newhire":  page_new_hire,
        "/shrink":   page_shrink_attr,
        "/dataset":  page_dataset,
        "/planning": lambda: dbc.Container([header_bar(), planning_layout()], fluid=True),
        "/ops":      lambda: dbc.Container([header_bar(), dbc.Alert("Operational Dashboard — todo", color="info")], fluid=True),
        "/budget":   page_budget,
    }
    fn = pages.get(path)
    return fn() if fn else not_found_layout()

# ✅ VALIDATION LAYOUT — include plan-detail skeleton
app.validation_layout = html.Div([
    dcc.Location(id="url-router"),
    dcc.Store(id="sidebar_collapsed"),
    dcc.Store(id="ws-status"),
    dcc.Store(id="ws-selected-ba"),
    dcc.Store(id="ws-refresh"),
    header_bar(),
    planning_layout(),
    plan_detail_validation_layout(),
    dash_table.DataTable(id="tbl-projects"),
])

# Register callbacks (planning + plan-detail)
register_planning_ws(app)
register_plan_detail(app)

# ---------------------- Main ----------------------
if __name__ == "__main__":
    app.run(debug=True)
