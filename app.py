# file: app.py
from __future__ import annotations
import os, platform, getpass, base64, io, datetime as dt
import pandas as pd
import numpy as np
from planning_workspace import planning_layout, register_planning_ws
from dash import Dash, html, dcc, dash_table, Output, Input, State, callback
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import plotly.express as px
import dash
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

# ---- SQLite persistence & dynamic sources ----
from cap_store import (
    init_db, load_defaults, save_defaults, save_roster_long, save_roster_wide,
    load_roster, save_roster, load_roster_long, load_roster_wide,
    load_hiring, save_hiring, load_attrition_raw, save_attrition_raw,
    load_shrinkage, save_shrinkage,
    load_attrition, save_attrition,
    save_scoped_settings, resolve_settings,
    get_roster_locations, get_clients_hierarchy, save_mapping_sheet1, load_mapping_sheet1,
    save_mapping_sheet2, load_mapping_sheet2,
    mapping_sheet1_template_df, mapping_sheet2_template_df,ensure_indexes
)

ensure_indexes()

# Initialize DB file
init_db()

SYSTEM_NAME = (os.environ.get("HOSTNAME") or getpass.getuser() or platform.node())

# ---------------------- Dash App ----------------------
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="CAPACITY CONNECT"
)
server = app.server

def _planning_ids_skeleton():
    # Only cross-page Stores. Do NOT mount Planning controls here.
    return html.Div([
        dcc.Store(id="ws-status"),
        dcc.Store(id="ws-selected-ba"),
        dcc.Store(id="ws-refresh"),
        # removed: planning-mounted (unused)
    ], style={"display": "none"})

# ✅ MAIN LAYOUT — do NOT include planning_layout() here
app.layout = html.Div([
    dcc.Location(id="url-router"),
    dcc.Store(id="sidebar_collapsed", data=True, storage_type="session"),
    _planning_ids_skeleton(),
    html.Div(id="app-wrapper", className="sidebar-collapsed", children=[
        html.Div(id="sidebar"),
        html.Div(id="root")
    ])
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

# ---------------------- Helpers ----------------------

# ===== Attrition (RAW) schema + template =====
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

def build_attrition_raw_template(rows: int = 3) -> pd.DataFrame:
    """Create a CSV-ready template with your exact 29 columns, with a few example rows."""
    today = dt.date.today()
    ex = []
    for i in range(rows):
        ex.append({
            "Reporting Full Date": (today - dt.timedelta(days=7-i)).isoformat(),
            "BRID": f"IN{i+1:04d}",
            "Employee Name": ["Asha Rao","Alex Doe","Priya Singh"][i % 3],
            "Operational Status": "Exited",
            "Corporate Grade Description": "G6",
            "Employee Email Address": f"user{i+1}@example.com",
            "Employee Position": "Analyst",
            "Position Description": "Ops Analyst",
            "Employee Line Manager Indicator": "No",
            "Length of Service Date": (today - dt.timedelta(days=365*(i+1))).isoformat(),
            "Cost Centre": "COST-1001",
            "Line Manager BRID": f"LM{i+1:03d}",
            "Line Manager Name": ["Chris Lee","Priyanka Menon","Sam Patel"][i % 3],
            "IMH L05": "IMH5",
            "IMH L06": "IMH6",
            "IMH L07": "IMH7",
            "Org Unit": "Operations",
            "Org Unit ID": "OU-01",
            "Employee Line Manager lvl 07": "Mgr7",
            "Employee Line Manager lvl 08": "Mgr8",
            "Employee Line Manager lvl 09": "Mgr9",
            "City": ["Chennai","Glasgow","Pune"][i % 3],
            "Building": "HQ",
            "Gender Description": ["Female","Male","Female"][i % 3],
            "Voluntary Involuntary Exit Description": "Voluntary",
            "Resignation Date": (today - dt.timedelta(days=5-i)).isoformat(),
            "Employee Contract HC": 1,
            "HC": 1,
            "FTE": 1.0
        })
    df = pd.DataFrame(ex, columns=ATTRITION_RAW_COLUMNS)
    return df

# ---------- Pretty column names (render-only) ----------
PRETTY_LABELS = {
    # common time keys
    "date": "Date",
    "week": "Week",
    "start_week": "Start Week",
    "interval": "Interval",
    # metrics
    "aht_sec": "AHT (sec)",
    "asa_sec": "ASA (sec)",
    "volume": "Volume",
    "items": "Items",
    "calls": "Calls",
    "agents_req": "Agents Required",
    "fte_req": "Billable FTE Required",
    "supply_fte": "Supply FTE",
    "total_req_fte": "Total Required FTE",
    "staffing_pct": "Staffing %",
    "shrinkage_pct": "Shrinkage %",
    "attrition_pct": "Attrition %",
    "occupancy": "Occupancy",
    "service_level": "Service Level",
    # entities
    "program": "Program",
    "campaign": "Campaign",
    "sub_task": "Sub Task",
    "tranche": "Tranche",
    # roster columns
    "employee_id": "Employee ID",
    "name": "Name",
    "status": "Status",
    "fte": "FTE",
    "skill_voice": "Skill: Voice",
    "skill_bo": "Skill: Back Office",
    "skill_ob": "Skill: Outbound",
    "start_date": "Start Date",
    "end_date": "End Date",
    "location": "Location",
    "country": "Country",
    "site": "Site",
    "region": "Region",
    # generic
    "leavers_fte": "Leavers FTE",
    "avg_active_fte": "Avg Active FTE",
}

# ==== Roster wide template helpers (Dash) ====
def build_roster_template_wide(start_date: dt.date, end_date: dt.date, include_sample: bool = False) -> pd.DataFrame:
    """
    Headers:
      BRID, Name, Team Manager, Business Area, Sub Business Area, LOB, Site, Location, Country, <date columns...>
    """
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
    """
    Wide → long:
      id cols: BRID, Name, Team Manager, Business Area, Sub Business Area, LOB, Site, Location, Country
      result: those id cols + date + entry (free text like '09:00-17:30' or 'Leave')
    """
    if df_wide is None or df_wide.empty:
        return pd.DataFrame(columns=[
            "BRID","Name","Team Manager","Business Area","Sub Business Area",
            "LOB","Site","Location","Country","date","entry"
        ])
    id_cols = ["BRID","Name","Team Manager","Business Area","Sub Business Area","LOB","Site","Location","Country"]
    id_cols = [c for c in id_cols if c in df_wide.columns]  # tolerate partial
    date_cols = [c for c in df_wide.columns if c not in id_cols]
    long = df_wide.melt(id_vars=id_cols, value_vars=date_cols, var_name="date", value_name="entry")
    long["entry"] = long["entry"].fillna("").astype(str).str.strip()
    long = long[long["entry"] != ""]
    long["date"] = pd.to_datetime(long["date"], errors="coerce", dayfirst=True).dt.date
    long = long[pd.notna(long["date"])]
    long["is_leave"] = long["entry"].str.lower().isin(["leave", "l", "off", "pto"])
    return long


def pretty_col(col: str) -> str:
    if not isinstance(col, str):
        return str(col)
    return PRETTY_LABELS.get(col, col.replace("_", " ").title())

def pretty_columns(df_or_cols) -> list[dict]:
    if hasattr(df_or_cols, "columns"):
        cols = list(df_or_cols.columns)
    else:
        cols = list(df_or_cols)
    return [{"name": pretty_col(c), "id": c} for c in cols]

def parse_upload(contents, filename) -> pd.DataFrame:
    """Parse CSV/XLSX/XLSM from dcc.Upload."""
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

def _week_floor(d: pd.Timestamp | str | dt.date, week_start: str = "Monday") -> dt.date:
    d = pd.to_datetime(d).date()
    wd = d.weekday()  # Mon=0..Sun=6
    if (week_start or "Monday").lower().startswith("sun"):
        return d - dt.timedelta(days=(wd + 1) % 7)
    return d - dt.timedelta(days=wd)

def weekly_avg_active_fte_from_roster(week_start: str = "Monday") -> pd.DataFrame:
    """
    Returns weekly average 'active FTE' for denominator.
    Prefers classic roster (start_date/end_date/fte). If that’s empty, falls back to roster_long.
    """
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

    # 2) Fallback: use normalized roster_long (distinct BRIDs per day -> FTE=1)
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
    """
    Leavers list (raw) -> weekly % using roster denominator, and 'program' = Business Area
    via Mapping Sheet 2 (IMH 06 -> Business Area Nomenclature). Falls back gracefully.
    """
    if df_raw is None or df_raw.empty:
        return pd.DataFrame(columns=["week","leavers_fte","avg_active_fte","attrition_pct","program"])

    df = df_raw.copy()

    # Normalize resignation date column
    if "Resignation Date" not in df.columns:
        if "Reporting Full Date" in df.columns:
            df["Resignation Date"] = df["Reporting Full Date"]
        else:
            return pd.DataFrame(columns=["week","leavers_fte","avg_active_fte","attrition_pct","program"])
    df = df[~df["Resignation Date"].isna()].copy()

    # FTE default
    if "FTE" not in df.columns:
        df["FTE"] = 1.0

    # ---- Mapping: IMH 06 -> Business Area ----
    m2 = load_mapping_sheet2()
    def _norm(s): 
        return str(s).strip().lower()
    imh_candidates = [c for c in df.columns if _norm(c) in ("imh 06","imh l06","imh l 06","imh06")]
    raw_key_col = imh_candidates[0] if imh_candidates else None

    program_series = None
    if isinstance(m2, pd.DataFrame) and not m2.empty and raw_key_col:
        m2_cols = { _norm(c): c for c in m2.columns }
        key_col = m2_cols.get("imh 06") or m2_cols.get("imh l06") or m2_cols.get("imh06") or list(m2.columns)[0]
        val_col = (m2_cols.get("business area nomenclature") 
                   or m2_cols.get("ba nomenclature")
                   or m2_cols.get("business area")
                   or list(m2.columns)[-1])
        look = dict(zip(m2[key_col].astype(str).map(_norm), m2[val_col].astype(str)))
        program_series = df[raw_key_col].astype(str).map(_norm).map(look)

    if program_series is None:
        # Fallback to Org Unit / Business Area / constant "All"
        lower = df.columns.str.lower()
        if raw_key_col:
            fallback_series = df[raw_key_col]
        elif any(lower.isin(["org unit","business area"])):
            pick_col = df.columns[lower.isin(["org unit","business area"])][0]
            fallback_series = df[pick_col]
        else:
            fallback_series = pd.Series(["All"] * len(df))
        program_series = fallback_series

    df["program"] = program_series.fillna("All").astype(str)

    # weekly aggregation of leavers FTE by BA/program
    df["week"] = df["Resignation Date"].apply(lambda x: _week_floor(x, week_start))
    wk = df.groupby(["week","program"], as_index=False)["FTE"].sum().rename(columns={"FTE":"leavers_fte"})

    # denominator from roster (existing logic)
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


# Treat any of these values as "leave"
LEAVE_TOKENS = {"leave","l","off","pto","absent","holiday","al","sl","el"}

def with_is_leave(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure is_leave reflects the current entry/value column."""
    if df is None or df.empty:
        return df
    df = df.copy()
    valcol = "entry" if "entry" in df.columns else ("value" if "value" in df.columns else None)
    if valcol is None:
        return df
    s = df[valcol].fillna("").astype(str).str.strip().str.lower()
    df["is_leave"] = s.isin(LEAVE_TOKENS)
    return df

# ---------------------- UI Fragments ----------------------
def header_bar():
    return dbc.Navbar(
        dbc.Container([
            dbc.Button("☰", id="btn-burger-top", color="link", className="me-3", n_clicks=0,
                       style={"fontSize":"24px","textDecoration":"none"}),

            html.Span(style={"fontSize":"28px","fontWeight":800}),

            # sticky, dynamic breadcrumb for Planning
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

        # dynamic option stores
        dcc.Store(id="settings-locations", data=[]),
        dcc.Store(id="settings-hier-map", data={}),

        dbc.Card(dbc.CardBody([
            html.H5("Default Settings — Scope"),
            dbc.Row([
                dbc.Col(dcc.RadioItems(
                    id="set-scope",
                    options=[
                        {"label":" Global","value":"global"},
                        {"label":" Location","value":"location"},
                        {"label":" Business Area ▶ Sub Business Area ▶ LOB","value":"hier"},
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
                dbc.Col(dcc.Dropdown(id="set-lob", placeholder="LOB / Channel"), md=3),
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
        dbc.Card(dbc.CardBody([
            html.H5("Mappings"),
            html.P(
                "Upload business mappings so reports can show Business Area instead of 'All'. "
                "Sheet 2 maps Attrition ‘IMH 06’ → Business Area; Sheet 1 is a general catalog "
                "you may use elsewhere (BA, Sub BA, LOB, Team, Site).",
                className="text-muted"
            ),

            html.H6("Mapping Sheet 1 — Catalog"),
            dbc.Row([
                dbc.Col(dcc.Upload(
                    id="up-map1",
                    children=html.Div(["⬆️ Upload Mapping Sheet 1 (CSV/XLSX)"]),
                    multiple=False, className="upload-box"
                ), md=6),
                dbc.Col(dbc.Button("Save Mapping 1", id="btn-save-map1", color="primary", className="w-100"), md=3),
                dbc.Col(dbc.Button("Download Sample", id="btn-dl-map1", outline=True, color="secondary", className="w-100"), md=3),
            ], className="my-2"),
            dcc.Download(id="dl-map1"),
            dash_table.DataTable(id="tbl-map1", page_size=6, style_table={"overflowX":"auto"},
                                 style_as_list_view=True, style_header={"textTransform":"none"}),
            html.Div(id="map1-save-msg", className="text-success mt-1"),

            html.Hr(),

            html.H6("Mapping Sheet 2 — IMH 06 → Business Area"),
            dbc.Row([
                dbc.Col(dcc.Upload(
                    id="up-map2",
                    children=html.Div(["⬆️ Upload Mapping Sheet 2 (CSV/XLSX)"]),
                    multiple=False, className="upload-box"
                ), md=6),
                dbc.Col(dbc.Button("Save Mapping 2", id="btn-save-map2", color="primary", className="w-100"), md=3),
                dbc.Col(dbc.Button("Download Sample", id="btn-dl-map2", outline=True, color="secondary", className="w-100"), md=3),
            ], className="my-2"),
            dcc.Download(id="dl-map2"),
            dash_table.DataTable(id="tbl-map2", page_size=6, style_table={"overflowX":"auto"},
                                 style_as_list_view=True, style_header={"textTransform":"none"}),
            html.Div(id="map2-save-msg", className="text-success mt-1"),
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

        # Stores for keeping originals and filtered long
        dcc.Store(id="roster_wide_store", data=(df_wide_db.to_dict("records") if not df_wide_db.empty else [])),
        dcc.Store(id="roster_long_store", data=(df_long_db.to_dict("records") if not df_long_db.empty else [])),

        # ---------------- Template builder + downloads ----------------
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

        # ---------------- Upload + Save ----------------
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
            html.Div(id="roster-wide-msg", className="text-muted mt-1"),  # <-- target for upload status
        ]), className="mb-3"),

        # ---------------- Normalized view + date filter ----------------
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

        # ---------------- Bulk edit helpers (NEW) ----------------
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
        dbc.Row([
            dbc.Col(dcc.Upload(id="up-hire", children=html.Div(["⬆️ Upload CSV/XLSX"]), multiple=False, className="upload-box"), md=4),
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
            style_header={"textTransform": "none"},
        ),
        dcc.Graph(id="fig-hire", style={"height":"280px"}, config={"displayModeBar": False}),
    ], fluid=True)

def page_shrink_attr():
    shr = load_shrinkage()
    att = load_attrition()
    return dbc.Container([
        header_bar(),
        dbc.Tabs(id="shr-tabs", active_tab="tab-shrink", children=[
            dbc.Tab(label="Shrinkage", tab_id="tab-shrink", children=[
                dbc.Row([
                    dbc.Col(dcc.Upload(id="up-shrink", children=html.Div(["⬆️ Drag & drop or click to upload CSV/XLSX"]), multiple=False, className="upload-box"), md=4),
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
                    style_header={"textTransform": "none"},
                ),
                dcc.Graph(id="fig-shrink", style={"height":"280px"}, config={"displayModeBar": False}),
            ]),
            dbc.Tab(label="Attrition", tab_id="tab-attr", children=[
                dbc.Row([
                    dbc.Col(dcc.Upload(id="up-attr", children=html.Div(["⬆️ Drag & drop or click to upload CSV/XLSX"]), multiple=False, className="upload-box"), md=4),
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
                    style_header={"textTransform": "none"},
                ),
                dcc.Graph(id="fig-attr", style={"height":"280px"}, config={"displayModeBar": False}),
            ]),
        ])
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

# ---------------------- Sidebar interactions ----------------------
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

# ---------------------- Settings: dynamic sources ----------------------
@callback(
    Output("settings-locations","data"),
    Output("settings-hier-map","data"),
    Input("url-router","pathname"),
    prevent_initial_call=False
)
def refresh_scope_sources(_path):
    return get_roster_locations(), get_clients_hierarchy()

@callback(Output("set-location","options"), Input("settings-locations","data"))
def fill_locations(locs): return [{"label":x,"value":x} for x in (locs or [])]

@callback(
    Output("set-ba","options"), Output("set-ba","value"),
    Input("settings-hier-map","data"), prevent_initial_call=False
)
def fill_ba(hmap):
    if not hmap: return [], None
    bas = sorted(hmap.keys())
    return [{"label":b,"value":b} for b in bas], (bas[0] if bas else None)

@callback(
    Output("set-subba","options"), Output("set-subba","value"),
    Input("set-ba","value"), State("settings-hier-map","data")
)
def fill_subba(ba, hmap):
    if not (ba and hmap and ba in hmap): return [], None
    subs = sorted((hmap[ba] or {}).keys())
    return [{"label":s,"value":s} for s in subs], (subs[0] if subs else None)

@callback(
    Output("set-lob","options"), Output("set-lob","value"),
    Input("set-ba","value"), Input("set-subba","value"), State("settings-hier-map","data")
)
def fill_lob(ba, sub, hmap):
    if not (ba and sub and hmap and ba in hmap and sub in hmap[ba]): return [], None
    lobs = hmap[ba][sub]
    return [{"label":l,"value":l} for l in lobs], (lobs[0] if lobs else None)

# Show/hide scope rows
@callback(Output("row-location","style"), Output("row-hier","style"), Input("set-scope","value"))
def scope_vis(scope):
    return ({"display":"flex"} if scope=="location" else {"display":"none"},
            {"display":"flex"} if scope=="hier" else {"display":"none"})

# Load settings for the chosen scope/key
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

# Save per-scope
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

# ---------------------- Roster ----------------------
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
    long = with_is_leave(normalize_roster_wide(df))
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
    df = with_is_leave(df)
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
    dfl = with_is_leave(dfl)
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

    df = with_is_leave(df)
    updated = df.to_dict("records")
    return updated, updated, msg

# ---------------------- New Hire ----------------------
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

# ---------------------- Shrinkage ----------------------
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

# ---------------------- Attrition (raw → weekly %) ----------------------
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

# ---------------------- Mapping Callbacks ----------------

# ---------- Mapping Sheet 1 ----------
@callback(
    Output("tbl-map1","data"), Output("tbl-map1","columns"),
    Input("up-map1","contents"), State("up-map1","filename"),
    prevent_initial_call=True
)
def map1_upload(contents, filename):
    df = parse_upload(contents, filename)
    if df.empty: raise PreventUpdate
    return df.to_dict("records"), pretty_columns(df)

@callback(
    Output("map1-save-msg","children"),
    Input("btn-save-map1","n_clicks"),
    State("tbl-map1","data"),
    prevent_initial_call=True
)
def map1_save(n, rows):
    df = pd.DataFrame(rows or [])
    if df.empty: return "Nothing to save."
    save_mapping_sheet1(df)
    return "Mapping Sheet 1 saved ✓"

@callback(Output("dl-map1","data"), Input("btn-dl-map1","n_clicks"), prevent_initial_call=True)
def dl_map1(n):
    df = mapping_sheet1_template_df()
    return dcc.send_data_frame(df.to_csv, "mapping_sheet1_template.csv", index=False)

# ---------- Mapping Sheet 2 ----------
@callback(
    Output("tbl-map2","data"), Output("tbl-map2","columns"),
    Input("up-map2","contents"), State("up-map2","filename"),
    prevent_initial_call=True
)
def map2_upload(contents, filename):
    df = parse_upload(contents, filename)
    if df.empty: raise PreventUpdate
    return df.to_dict("records"), pretty_columns(df)

@callback(
    Output("map2-save-msg","children"),
    Input("btn-save-map2","n_clicks"),
    State("tbl-map2","data"),
    prevent_initial_call=True
)
def map2_save(n, rows):
    df = pd.DataFrame(rows or [])
    if df.empty: return "Nothing to save."
    save_mapping_sheet2(df)
    return "Mapping Sheet 2 saved ✓"

@callback(Output("dl-map2","data"), Input("btn-dl-map2","n_clicks"), prevent_initial_call=True)
def dl_map2(n):
    df = mapping_sheet2_template_df()
    return dcc.send_data_frame(df.to_csv, "mapping_sheet2_template.csv", index=False)

# ---------------------- Router + Home ----------------------
app.layout.children.append(dcc.Interval(id="cap-plans-refresh", interval=5000, n_intervals=0))

@callback(
    Output("tbl-projects", "data"),
    Input("cap-plans-refresh", "n_intervals"),
    State("url-router", "pathname"),
    prevent_initial_call=False
)
def _refresh_projects_table(_n, pathname):
    path = (pathname or "").rstrip("/")
    if path not in ("", "/"):  # only refresh on Home
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

    # --- NEW: Plan detail route (must come before pages dict lookup) ---
    if path.startswith("/plan/"):
        try:
            pid = int(path.rsplit("/", 1)[-1])
        except Exception:
            return not_found_layout()
        # wrap with your shared header
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
        "/budget":   lambda: dbc.Container([header_bar(), dbc.Alert("Budget — todo", color="info")], fluid=True),
    }
    fn = pages.get(path)
    return fn() if fn else not_found_layout()

# ✅ VALIDATION LAYOUT — add the plan-detail skeleton so Dash “knows” the IDs
app.validation_layout = html.Div([
    dcc.Location(id="url-router"),
    dcc.Store(id="sidebar_collapsed"),
    dcc.Store(id="ws-status"),
    dcc.Store(id="ws-selected-ba"),
    dcc.Store(id="ws-refresh"),
    header_bar(),
    planning_layout(),                  # existing planning skeleton
    plan_detail_validation_layout(),    # ← NEW: tiny, hidden layout for /plan/<id>
    dash_table.DataTable(id="tbl-projects"),
])

# Register callbacks (planning + plan-detail)
register_planning_ws(app)
register_plan_detail(app)  # ← keep this after app.validation_layout is set
# ---------------------- Main ----------------------
if __name__ == "__main__":
    app.run(debug=True)
