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
import dash_svg as svg
from datetime import date, timedelta
from plan_store import get_plan
from cap_db import save_df, load_df, _conn
from plan_detail import layout_for_plan, plan_detail_validation_layout, register_plan_detail

# ---- Core math & demo (replace with real when ready) ----
from capacity_core import (
    required_fte_daily, supply_fte_daily, understaffed_accounts_next_4w,
    kpi_hiring, kpi_shrinkage,
    make_projects_sample, make_voice_sample, make_backoffice_sample,
    make_outbound_sample, make_roster_sample, make_hiring_sample,
    make_shrinkage_sample, make_attrition_sample, _last_next_4, min_agents,
    voice_requirements_interval,
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
# (app init moved to app_instance.py)


# ====================== helpers ======================

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

def _week_floor(d: pd.Timestamp | str | dt.date, week_start: str = "Monday") -> dt.date:
    d = pd.to_datetime(d).date()
    wd = d.weekday()
    if (week_start or "Monday").lower().startswith("sun"):
        return d - dt.timedelta(days=(wd + 1) % 7)
    return d - dt.timedelta(days=wd)

def _all_locations() -> list[str]:
    """Unique Position Location Country values (used for the Site-only scope = Country)."""
    df = _hcu_df()
    if df.empty:
        return []
    C = _hcu_cols(df)
    loc_col = C["loc"]
    if not loc_col:
        return []
    vals = (
        df[loc_col]
        .dropna()
        .astype(str)
        .str.strip()
        .replace({"": np.nan, "na": np.nan, "none": np.nan, "all": np.nan})
        .dropna()
        .drop_duplicates()
        .sort_values()
    )
    return vals.tolist()

def _read_upload_to_df(contents, filename):
    if not contents:
        return pd.DataFrame()
    _type, b64 = contents.split(",", 1)
    buf = base64.b64decode(b64)
    if (filename or "").lower().endswith(".csv"):
        return pd.read_csv(io.StringIO(buf.decode("utf-8")), dtype=str)
    return pd.read_excel(io.BytesIO(buf))

def _preview_cols_data(df):
    if df is None or df.empty:
        return [], []
    return [{"name": c, "id": c} for c in df.columns], df.to_dict("records")

def _voice_tactical_canon(df_in: pd.DataFrame):
    """Return (vol_df, aht_df, debug) for voice tactical uploads."""
    if df_in is None or df_in.empty:
        return pd.DataFrame(), pd.DataFrame(), "no rows"

    df = df_in.copy()
    L = {str(c).strip().lower(): c for c in df.columns}

    # date
    date_col = L.get("date") or L.get("week")
    if not date_col:
        return pd.DataFrame(), pd.DataFrame(), "missing date/week"
    df["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()

    # interval (optional but recommended for voice)
    ivl_col = L.get("interval") or L.get("interval_start")
    df["interval"] = df[ivl_col] if ivl_col else pd.NaT

    # volume
    vol_col = L.get("volume") or L.get("vol") or L.get("calls")
    if vol_col:
        df["volume"] = pd.to_numeric(df[vol_col], errors="coerce").fillna(0.0)
    else:
        df["volume"] = 0.0

    # aht
    aht_col = L.get("aht_sec") or L.get("aht") or L.get("avg_aht") or L.get("aht_seconds")
    vol_df = df[["date","interval","volume"]]
    if aht_col:
        aht_df = df[["date","interval", aht_col]].rename(columns={aht_col:"aht_sec"})
        aht_df["aht_sec"] = pd.to_numeric(aht_df["aht_sec"], errors="coerce")
        aht_df = aht_df.dropna(subset=["aht_sec"])
    else:
        aht_df = pd.DataFrame(columns=["date","interval","aht_sec"])

    dbg = f"mapped: date={date_col!r}, interval={ivl_col!r}, volume={vol_col!r}, aht={aht_col!r}; rows: vol={len(vol_df)}, aht={len(aht_df)}"
    return vol_df, aht_df, dbg

def _bo_tactical_canon(df_in: pd.DataFrame):
    """Return (vol_df, sut_df, debug) for back-office tactical uploads."""
    if df_in is None or df_in.empty:
        return pd.DataFrame(), pd.DataFrame(), "no rows"

    df = df_in.copy()
    L = {str(c).strip().lower(): c for c in df.columns}

    # date
    date_col = L.get("date") or L.get("week")
    if not date_col:
        return pd.DataFrame(), pd.DataFrame(), "missing date/week"
    df["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()

    # items
    items_col = L.get("items") or L.get("volume") or L.get("txns") or L.get("transactions")
    df["items"] = pd.to_numeric(df[items_col], errors="coerce").fillna(0.0) if items_col else 0.0

    # sut/aht
    sut_col = L.get("sut_sec") or L.get("sut") or L.get("avg_sut") or L.get("aht_sec") or L.get("aht")
    vol_df = df[["date","items"]]
    if sut_col:
        sut_df = df[["date", sut_col]].rename(columns={sut_col:"sut_sec"})
        sut_df["sut_sec"] = pd.to_numeric(sut_df["sut_sec"], errors="coerce")
        sut_df = sut_df.dropna(subset=["sut_sec"])
    else:
        sut_df = pd.DataFrame(columns=["date","sut_sec"])

    dbg = f"mapped: date={date_col!r}, items={items_col!r}, sut={sut_col!r}; rows: items={len(vol_df)}, sut={len(sut_df)}"
    return vol_df, sut_df, dbg


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


def _canon_scope(ba, sba, ch, site=None):
    canon = lambda x: (x or '').strip()
    if site:
        return f"{canon(ba)}|{canon(sba)}|{canon(ch)}|{canon(site)}"
    return f"{canon(ba)}|{canon(sba)}|{canon(ch)}|{canon(site)}"  # legacy 3-part key still valid

def _all_sites() -> list[str]:
    df = _hcu_df()  # uses load_headcount() safely
    if df.empty:
        return []
    C = _hcu_cols(df)
    site_col = C["site"]
    if not site_col:
        return []
    vals = (
        df[site_col]
        .dropna()
        .astype(str)
        .str.strip()
        .replace({"": np.nan, "na": np.nan, "none": np.nan, "all": np.nan})
        .dropna()
        .drop_duplicates()
        .sort_values()
    )
    return vals.tolist()

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
            "Year": int(pd.Timestamp(today).year), "Week Number": int(pd.Timestamp(today).isocalendar().week),
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

    # defaults so the pivot in summarize_shrinkage_voice never drops rows
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
    grp = df.groupby(["week","program"], as_index=False).agg({
        "Total Shrink %":"mean",
        "Paid Hours":"sum"
    })
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
        ["BUK","COO","Business Services","BFA","Refers","","","IN0001","Asha Rao","Agent","Active","FT","BA4","IN9999","Priyanka Menon","Ops|BFA|Refers","Ops BFA Refers","India","Chennai","DLF IT Park","12345","Complaints","Onboarding","Onboarding","Agent"],
        ["BUK","COO","Business Services","BFA","Appeals","","","IN0002","Rahul Jain","Team Leader","Active","FT","BA5","IN8888","Arjun Mehta","Ops|BFA|Appeals","Ops BFA Appeals","India","Pune","EON Cluster C","12345","Complaints","Onboarding","Onboarding","Team Leader"],
    ]
    df = pd.DataFrame(sample[:rows], columns=HC_COLUMNS)
    if rows > len(sample):
        df = pd.concat([df, pd.DataFrame(columns=HC_COLUMNS)], ignore_index=True)
    return df

# === New combined templates (as requested) ===
def voice_forecast_template_df():
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
        if isinstance(name_txt, list):
            name_txt = " ".join(map(str, name_txt))
        id_txt = str(c.get("id", ""))
        if "variance" in str(name_txt).lower() or "variance" in id_txt.lower():
            c["editable"] = False
        else:
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
            dbc.Nav([dcc.Link(SYSTEM_NAME, href="/", className="nav-link") ], className="ms-auto"),
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
        dbc.Row([
            dbc.Col(tile("Help & Docs","ℹ️","/help"), width=6),
        ], className="ghani"),
    ], style={"padding":"12px","borderRadius":"12px","background":"#fff","boxShadow":"0 2px 8px rgba(0,0,0,.06)", "minHeight": "100%"})

def center_projects_table():
    """Render the Home table with empty data; a callback fills the data.
    This avoids shipping a stale module-level snapshot at import time.
    """
    return html.Div([
        html.H5("Capacity Plans"),
        dash_table.DataTable(
            id="tbl-projects",
            data=[], # filled by router._refresh_projects_table
            columns=[
                {"name": "Business Area", "id": "Business Area"},
                {"name": "Active Plans", "id": "Active Plans"},
            ],
            style_as_list_view=True,
            page_size=10,
            style_table={"overflowX": "auto", "maxWidth": "100%"},
            style_header={"textTransform": "none"},
        ),
    ], style={
        "padding": "12px",
        "borderRadius": "12px",
        "background": "#fff",
        "boxShadow": "0 2px 8px rgba(0,0,0,.06)",
        "minHeight": "90vh",
    })

def _filter_by_ba(df: pd.DataFrame, ba: str | None) -> pd.DataFrame:
    if df is None or df.empty or not ba:
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    d = df.copy()
    L = {str(c).strip().lower(): c for c in d.columns}
    for col in (L.get("program"), L.get("business area"), L.get("journey")):
        if col:
            return d[d[col].astype(str).str.strip().str.lower() == str(ba).strip().lower()]
    return d

def _home_kpis_for_ba(ba: str | None = None) -> dict:
    # ---- Requirements: prefer saved Voice/BO timeseries aggregated across scopes; fallback to samples
    def _all_names(prefix: str) -> list[str]:
        try:
            with _conn() as cx:
                rows = cx.execute("SELECT name FROM datasets WHERE name LIKE ?", (f"{prefix}::%",)).fetchall()
            return [r["name"] if isinstance(r, dict) else r[0] for r in rows]
        except Exception:
            return []

    def _scope_matches(name: str, ba: str | None) -> bool:
        if not ba:
            return True
        try:
            sk = name.split("::",1)[1]
            first = (sk.split("|")[0] or "").strip()
            return first.lower() == str(ba).strip().lower()
        except Exception:
            return True

    # Collect Voice req staff_seconds
    voice_ss = pd.DataFrame(columns=["date","staff_seconds"])  # aggregated
    v_names = [n for n in _all_names("voice_forecast_volume") if _scope_matches(n, ba)]
    if v_names:
        acc = []
        for n in v_names:
            try:
                vol = load_df(n)
                sk = n.split("::",1)[1]
                aht = load_df(f"voice_forecast_aht::{sk}")
                L = {str(c).strip().lower(): c for c in vol.columns}
                dc = L.get("date"); ic = L.get("interval"); vc = L.get("volume")
                if not (dc and ic and vc):
                    continue
                dff = vol[[dc,ic,vc]].rename(columns={dc:"date", ic:"interval", vc:"volume"})
                if isinstance(aht, pd.DataFrame) and not aht.empty:
                    LA = {str(c).strip().lower(): c for c in aht.columns}
                    if {LA.get("date"), LA.get("interval"), LA.get("aht_sec")} <= set(aht.columns):
                        a = aht[[LA.get("date"), LA.get("interval"), LA.get("aht_sec")]].rename(columns={LA.get("date"):"date", LA.get("interval"):"interval", LA.get("aht_sec"):"aht_sec"})
                        dff = dff.merge(a, on=["date","interval"], how="left")
                dff["aht_sec"] = pd.to_numeric(dff.get("aht_sec"), errors="coerce").fillna(300.0)
                vr = voice_requirements_interval(dff, DEFAULT_SETTINGS)
                if isinstance(vr, pd.DataFrame) and not vr.empty and "staff_seconds" in vr.columns:
                    s = vr.groupby("date", as_index=False)["staff_seconds"].sum()
                    acc.append(s)
            except Exception:
                continue
        if acc:
            voice_ss = pd.concat(acc, ignore_index=True).groupby("date", as_index=False)["staff_seconds"].sum()

    # Collect BO staff_seconds = items * sut_sec
    bo_ss = pd.DataFrame(columns=["date","staff_seconds"])
    b_names = [n for n in _all_names("bo_forecast_volume") if _scope_matches(n, ba)]
    if b_names:
        acc = []
        for n in b_names:
            try:
                vol = load_df(n)
                sk = n.split("::",1)[1]
                sut = load_df(f"bo_forecast_sut::{sk}")
                if not isinstance(vol, pd.DataFrame) or vol.empty or not isinstance(sut, pd.DataFrame) or sut.empty:
                    continue
                LV = {str(c).strip().lower(): c for c in vol.columns}
                LS = {str(c).strip().lower(): c for c in sut.columns}
                if LV.get("date") and LV.get("volume") and LS.get("date") and LS.get("sut_sec"):
                    v = vol[[LV.get("date"), LV.get("volume")]].rename(columns={LV.get("date"):"date", LV.get("volume"):"volume"})
                    s = sut[[LS.get("date"), LS.get("sut_sec")]].rename(columns={LS.get("date"):"date", LS.get("sut_sec"):"sut_sec"})
                    m = v.merge(s, on="date", how="inner").dropna()
                    if not m.empty:
                        m["staff_seconds"] = pd.to_numeric(m["volume"], errors="coerce").fillna(0) * pd.to_numeric(m["sut_sec"], errors="coerce").fillna(0)
                        acc.append(m.groupby("date", as_index=False)["staff_seconds"].sum())
            except Exception:
                continue
        if acc:
            bo_ss = pd.concat(acc, ignore_index=True).groupby("date", as_index=False)["staff_seconds"].sum()

    req_store = pd.DataFrame()
    if not voice_ss.empty or not bo_ss.empty:
        denom_voice = float(DEFAULT_SETTINGS.get("hours_per_fte", 8.0)) * 3600 * (1.0 - float(DEFAULT_SETTINGS.get("shrinkage_pct", 0.30)))
        bo_hours = float(DEFAULT_SETTINGS.get("bo_hours_per_day", DEFAULT_SETTINGS.get("hours_per_fte", 8.0)))
        bo_shr = float(DEFAULT_SETTINGS.get("bo_shrinkage_pct", DEFAULT_SETTINGS.get("shrinkage_pct", 0.0)))
        denom_bo = bo_hours * 3600 * (1.0 - bo_shr)
        vv = voice_ss.copy(); vv["fte"] = vv["staff_seconds"] / max(denom_voice, 1e-6)
        bb = bo_ss.copy();    bb["fte"] = bb["staff_seconds"]   / max(denom_bo, 1e-6)
        dfm = pd.merge(vv[["date","fte"]].rename(columns={"fte":"voice_fte"}),
                       bb[["date","fte"]].rename(columns={"fte":"bo_fte"}), on="date", how="outer").fillna(0)
        dfm["total_req_fte"] = dfm.get("voice_fte",0) + dfm.get("bo_fte",0)
        dfm["program"] = (ba or "All")
        req_store = dfm[["date","program","total_req_fte"]].copy()

    if not req_store.empty:
        req = req_store
    else:
        v = _filter_by_ba(voice_df, ba)
        b = _filter_by_ba(bo_df, ba)
        o = _filter_by_ba(ob_df, ba)
        req = required_fte_daily(v if isinstance(v, pd.DataFrame) else voice_df,
                                 b if isinstance(b, pd.DataFrame) else bo_df,
                                 o if isinstance(o, pd.DataFrame) else ob_df,
                                 DEFAULT_SETTINGS)
    # Prefer stored roster + hiring for supply; fallback to demos
    try:
        r_store = load_roster()
    except Exception:
        r_store = pd.DataFrame()
    try:
        h_store = load_hiring()
    except Exception:
        h_store = pd.DataFrame()

    r_use = _filter_by_ba(r_store, ba) if isinstance(r_store, pd.DataFrame) and not r_store.empty else _filter_by_ba(roster_demo, ba)
    h_use = _filter_by_ba(h_store, ba) if isinstance(h_store, pd.DataFrame) and not h_store.empty else _filter_by_ba(hiring_demo, ba)
    sup = supply_fte_daily(r_use if isinstance(r_use, pd.DataFrame) else roster_demo,
                           h_use if isinstance(h_use, pd.DataFrame) else hiring_demo)
    underst = understaffed_accounts_next_4w(req, sup)

    # Hiring: prefer stored hiring
    if isinstance(h_store, pd.DataFrame) and not h_store.empty:
        hk = _filter_by_ba(h_store, ba)
        lw, tw, nw = kpi_hiring(hk if isinstance(hk, pd.DataFrame) and not hk.empty else h_store)
    else:
        hk = _filter_by_ba(hiring_demo, ba)
        lw, tw, nw = kpi_hiring(hk if isinstance(hk, pd.DataFrame) else hiring_demo)

    # Shrinkage (prefer saved store; fallback to demo)
    try:
        shr_store = load_shrinkage()
    except Exception:
        shr_store = pd.DataFrame()
    if isinstance(shr_store, pd.DataFrame) and not shr_store.empty:
        sdemo = _filter_by_ba(shr_store, ba)
        s_last4, s_next4 = kpi_shrinkage(sdemo if not sdemo.empty else shr_store)
    else:
        sdemo = _filter_by_ba(shrink_demo, ba)
        s_last4, s_next4 = kpi_shrinkage(sdemo if isinstance(sdemo, pd.DataFrame) and not sdemo.empty else shrink_demo)

    # Attrition (prefer saved store; fallback to demo)
    try:
        attr_store = load_attrition()
    except Exception:
        attr_store = pd.DataFrame()
    if isinstance(attr_store, pd.DataFrame) and not attr_store.empty:
        A = _filter_by_ba(attr_store, ba)
        alast, anext = _last_next_4(A if not A.empty else attr_store, "week", "attrition_pct")
    else:
        alast, anext = _last_next_4(attr_demo, "week", "attrition_pct")
        if ba and isinstance(attr_demo, pd.DataFrame) and not attr_demo.empty and ("program" in attr_demo.columns or "Business Area" in attr_demo.columns):
            A = _filter_by_ba(attr_demo, ba)
            if isinstance(A, pd.DataFrame) and not A.empty:
                alast, anext = _last_next_4(A, "week", "attrition_pct")

    return dict(
        underst=underst,
        hire_lw=float(lw or 0), hire_tw=float(tw or 0), hire_nw=float(nw or 0),
        shr_last4=float(s_last4 or 0), shr_next4=float(s_next4 or 0),
        attr_last4=float(alast or 0), attr_next4=float(anext or 0),
    )

def _kpi_cards_children(k: dict) -> list:
    return [
        html.Div([
            html.Div("👥 Staffing", className="kpi-head"),
            html.Div(className="kpi-topdash"),
            html.Div([
                html.Div([html.Div("0", className="num"), html.Div("Last Week", className="lbl")], className="cell"),
                html.Div([html.Div("0", className="num"), html.Div("This Week", className="lbl")], className="cell"),
                html.Div([html.Div(str(int(k.get("underst", k.get("underst", 0)))), className="num"), html.Div("Next Week", className="lbl")], className="cell"),
            ], className="kpi3"),
        ], className="kpi-card mb-3 edge-teal"),
        html.Div([
            html.Div("🎯 Hiring", className="kpi-head"),
            html.Div(className="kpi-topdash"),
            html.Div([
                html.Div([html.Div(str(int(k.get("hire_lw", 0))), className="num"), html.Div("Last Week", className="lbl")], className="cell"),
                html.Div([html.Div(str(int(k.get("hire_tw", 0))), className="num"), html.Div("This Week", className="lbl")], className="cell"),
                html.Div([html.Div(str(int(k.get("hire_nw", 0))), className="num"), html.Div("Next Week", className="lbl")], className="cell"),
            ], className="kpi3"),
        ], className="kpi-card mb-3 edge-blue"),
        html.Div([
            html.Div("📉 Shrinkage", className="kpi-head"),
            html.Div(className="kpi-topdash"),
            html.Div([
                html.Div([html.Div(f"{k.get('shr_last4',0):.2f}%", className="num"), html.Div("Last 4 Weeks", className="lbl")], className="cell"),
                html.Div([html.Div(f"{k.get('shr_next4',0):.2f}%", className="num"), html.Div("This Week", className="lbl")], className="cell"),
                html.Div([html.Div(f"{k.get('shr_next4',0):.2f}%", className="num"), html.Div("Next 4 Weeks", className="lbl")], className="cell"),
            ], className="kpi3"),
        ], className="kpi-card edge-orange"),
        html.Div([
            html.Div("🔄 Attrition", className="kpi-head"),
            html.Div(className="kpi-topdash"),
            html.Div([
                html.Div([html.Div(f"{k.get('attr_last4',0):.2f}%", className="num"), html.Div("Last 4 Weeks", className="lbl")], className="cell"),
                html.Div([html.Div(f"{k.get('attr_next4',0):.2f}%", className="num"), html.Div("This Week", className="lbl")], className="cell"),
                html.Div([html.Div(f"{k.get('attr_next4',0):.2f}%", className="num"), html.Div("Next 4 Weeks", className="lbl")], className="cell"),
            ], className="kpi3"),
        ], className="kpi-card edge-red"),
    ]

def right_kpi_cards():
    # Default view = All business areas
    return html.Div(id="right-kpis", children=_kpi_cards_children(_home_kpis_for_ba(None)), style={"minHeight":"100%"})

# ---- Activity log helpers ----
def log_activity(user: str, action: str, path: str = "", meta: dict | None = None) -> None:
    try:
        df = load_df("activity_log")
    except Exception:
        df = None
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(columns=["ts","user","action","path","meta"])    
    rec = {
        "ts": pd.Timestamp.utcnow().isoformat(timespec="seconds"),
        "user": user,
        "action": action,
        "path": path,
        "meta": (meta or {}),
    }
    try:
        df = pd.concat([df, pd.DataFrame([rec])], ignore_index=True)
    except Exception:
        pass
    try:
        save_df("activity_log", df)
    except Exception:
        pass

def timeline_card() -> html.Div:
    def svg_down():
        return svg.Svg([
            svg.Path(
                d="m19 9-7 7-7-7",
                stroke="currentColor",
                strokeLinecap="round",
                strokeLinejoin="round",
                strokeWidth="2"
            )
        ], xmlns="http://www.w3.org/2000/svg", width=24, height=24, fill="none", viewBox="0 0 24 24"),
    down_svg = svg_down()
    return dbc.Card(
        dbc.CardBody([
            html.Div([
                html.Span("User Timeline", className="fw-bold"),
                html.Button(down_svg, id="timeline-toggle", title="Collapse/Expand",
                            style={"marginLeft":"auto","border":"none","background":"transparent","cursor":"pointer"})
            ], className="d-flex align-items-center mb-2 ritu"),
            dbc.Collapse(
                html.Div(id="timeline-body", className="timeline"),
                id="timeline-collapse", is_open=False
            )
        ], class_name="cgdivu"),
        className="mb-3 cg"
    )

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
        ("ℹ️","Help & Docs","/help","sb-help"),
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
