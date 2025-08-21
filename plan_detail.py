from __future__ import annotations
import datetime as dt
from typing import List, Tuple, Dict
import base64, io
import pandas as pd
import numpy as np
import dash
from dash import html, dcc, dash_table, Input, Output, State, no_update, ctx
import dash_bootstrap_components as dbc
from plan_store import get_plan
from cap_db import save_df, load_df
from capacity_core import min_agents, offered_load_erlangs  # ready for future formulas
from cap_db import save_df, load_df
import pandas as pd

pd.set_option('future.no_silent_downcasting', True)

# ───────────────────────── Baseline Marker ─────────────────────────
__CAPCONNECT_PLAN_DETAIL_BASELINE__ = "plan_detail.py · Baseline v1 · 2025-08-21"
# (bump this when we make major structural changes)
# ───────────────────────────────────────────────────────────────────



# ──────────────────────────────────────────────────────────────────────────────
# helpers

def _load_master_roster_wide() -> pd.DataFrame:
    df = load_df("roster_wide")
    if isinstance(df, pd.DataFrame) and not df.empty:
        return df
    df = load_df("roster")  # fallback
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

def _prefill_roster_from_master(pid: int, persist: bool = False) -> pd.DataFrame:
    """
    Seed plan roster from the master roster filtered by the plan crumb.
    De-duplicates by BRID (case/space tolerant) and keeps last occurrence.
    """
    p = get_plan(pid) or {}
    ba   = (p.get("business_area") or "").strip()
    sub  = (p.get("sub_business_area") or "").strip()
    lob  = (p.get("lob") or "").strip()
    site = (p.get("site") or "").strip()

    master = _load_master_roster_wide()
    if master.empty:
        return pd.DataFrame(columns=[c["id"] for c in _roster_columns()])

    L = {c.lower(): c for c in master.columns}
    c_brid = L.get("brid") or L.get("employee_id") or L.get("id") or "BRID"
    c_name = L.get("name") or "Name"
    c_mgr  = L.get("team manager") or L.get("manager") or "Team Manager"
    c_ba   = L.get("business area") or "Business Area"
    c_sub  = L.get("sub business area") or "Sub Business Area"
    c_lob  = L.get("lob") or "LOB"
    c_site = L.get("site") or "Site"

    df = master.copy()

    # Filter by crumb (tolerant—only apply when column exists and value provided)
    for col_key, value in [(c_ba, ba), (c_sub, sub), (c_lob, lob), (c_site, site)]:
        if value and col_key in df.columns:
            df = df[df[col_key].astype(str).str.strip() == value]

    if df.empty:
        return pd.DataFrame(columns=[c["id"] for c in _roster_columns()])

    # --- De-dup by BRID, keep last ---
    if c_brid not in df.columns:
        return pd.DataFrame(columns=[c["id"] for c in _roster_columns()])
    df = df.dropna(subset=[c_brid])
    df["_brid_norm"] = df[c_brid].astype(str).str.strip()
    df = df[df["_brid_norm"] != ""]
    df = df.drop_duplicates(subset=["_brid_norm"], keep="last")

    rows = []
    for _, r in df.iterrows():
        row = {cid: "" for cid in _ROSTER_REQUIRED_IDS}
        row.update({
            "brid":         str(r.get(c_brid, "")).strip(),
            "name":         str(r.get(c_name, "")).strip(),
            "team_leader":  str(r.get(c_mgr, "")).strip(),
            "biz_area":     str(r.get(c_ba, ba)).strip() if c_ba in df.columns else ba,
            "sub_biz_area": str(r.get(c_sub, sub)).strip() if c_sub in df.columns else sub,
            "lob":          str(r.get(c_lob, lob)).strip() if c_lob in df.columns else lob,
            "site":         str(r.get(c_site, site)).strip() if c_site in df.columns else site,
        })
        rows.append(row)

    out = pd.DataFrame(rows, columns=[c["id"] for c in _roster_columns() if c["id"] != "_select"])
    if persist:
        save_df(f"plan_{pid}_emp", out)
    return out


def _extract_ba_parts(pid: int) -> tuple[str, str, str, str]:
    """
    Return (Business Area, Sub Business Area, Channel/LOB, Site) from the plan.
    Tries many key aliases, then falls back to parsing a breadcrumb/path string.
    If BA is still missing, we default it to 'BDA' so the crumb is complete.
    """
    p = get_plan(pid) or {}

    def _nice(val: str) -> str:
        s = str(val or "").strip()
        if not s:
            return ""
        # keep short acronyms uppercase (e.g., BDA); otherwise Title Case
        return s if (s.isupper() and len(s) <= 4) else s.title()

    def _first_alias(aliases: list[str], *, fuzzy_any: list[str] | None = None,
                     exclude: list[str] | None = None) -> str:
        for k in aliases:
            if k in p and str(p[k]).strip():
                return _nice(p[k])
        if fuzzy_any:
            for k, v in p.items():
                key = str(k).lower()
                if all(term in key for term in (t.lower() for t in fuzzy_any)):
                    if exclude and any(ex.lower() in key for ex in exclude):
                        continue
                    if v is not None and str(v).strip():
                        return _nice(v)
        return ""

    # --- Business Area
    bda = _first_alias(
        aliases=[
            "bda","ba","biz_area","business_area","business_area_name","ba_name",
            "businessDeliveryArea","businessDeliveryAreaName","business_area_code",
            "business_area_desc","business_area_l1","ba_code","ba_short","bda_code"
        ],
        fuzzy_any=["business","area"], exclude=["sub"]
    )
    # --- Sub Business Area
    sba = _first_alias(
        aliases=[
            "sba","sub_ba","sub_ba_name","sub_biz_area","sub_business_area",
            "sub_business_area_name","business_area_l2","work_type","workstream",
            "sub_area","subarea","subarea_name","subBusinessArea","subBusinessAreaName"
        ],
        fuzzy_any=["sub","business","area"]
    )
    # --- Channel / LOB
    chan = _first_alias(
        aliases=[
            "channel","channel_name","channelName","lob","lob_name",
            "line_of_business","lineOfBusiness"
        ],
        fuzzy_any=["channel"]
    ) or _first_alias(["lob"], fuzzy_any=["lob"])

    # --- Site / Location
    site = _first_alias(
        aliases=[
            "site","site_name","siteName","site_location","geo_site",
            "location","location_name","siteLocation"
        ],
        fuzzy_any=["site"]
    ) or _first_alias([], fuzzy_any=["location"])

    # --- Fallback: parse breadcrumb/path if provided
    if not (bda and sba) or not (chan and site):
        crumb_val = ""
        for ck in ("breadcrumb","breadcrumbs","crumb","org_path","hierarchy","path"):
            if ck in p and str(p[ck]).strip():
                crumb_val = str(p[ck])
                break
        if crumb_val:
            import re
            parts = [_nice(x) for x in re.split(r">\s*|/\s*|»\s*|\|\s*", crumb_val) if str(x).strip()]
            if len(parts) >= 2 and not site: site = parts[-1]
            if len(parts) >= 2 and not chan: chan = parts[-2]
            if len(parts) >= 3 and not sba:  sba  = parts[-3]
            if len(parts) >= 4 and not bda:  bda  = parts[-4]

    # Final safety: ensure BA is never blank in the crumb
    if not bda:
        bda = "BDA"

    return bda, sba, chan, site


def _roster_crumb_from_plan(p: dict) -> str:
    """
    Build 'BDA > Sub BA > LOB > Site' crumb from plan meta.
    Accepts either 'business_area'/'sub_business_area' or 'biz_area'/'sub_biz_area'.
    """
    bda = p.get("business_area") or p.get("biz_area")
    sba = p.get("sub_business_area") or p.get("sub_biz_area")
    lob = p.get("lob")
    site = p.get("site")
    parts = [x for x in ("BDA", bda, sba, lob, site) if x]   # include literal 'BDA' prefix
    return " > ".join(parts[1:]) if len(parts) > 1 else ""   # show from Business Area onwards

def _load_or_empty_roster(pid: int) -> pd.DataFrame:
    """Load roster DF for a plan, or return an empty DF with the right columns."""
    cols = [c["id"] for c in _roster_columns()]   # assumes you already have _roster_columns()
    df = load_df(f"plan_{pid}_emp")
    if isinstance(df, pd.DataFrame) and not df.empty:
        # ensure all expected columns exist and in the right order
        for col in cols:
            if col not in df.columns:
                df[col] = ""
        return df[cols]
    return pd.DataFrame(columns=cols)

def _load_or_empty_bulk_files(pid: int) -> pd.DataFrame:
    """Bulk Upload table: always return DF with expected columns/order."""
    cols = [c["id"] for c in _bulkfile_columns()]  # e.g. ["file_name","ext","size_kb","is_valid","status"]
    df = load_df(f"plan_{pid}_bulk_files")
    if isinstance(df, pd.DataFrame) and not df.empty:
        # ensure all expected columns exist
        for col in cols:
            if col not in df.columns:
                df[col] = ""  # or 0 for numeric columns if you prefer
        return df[cols].copy()
    return pd.DataFrame(columns=cols)

def _load_or_empty_notes(pid: int) -> pd.DataFrame:
    """Notes table: keep only when/user/note columns and in that order."""
    cols = ["when", "user", "note"]
    df = load_df(f"plan_{pid}_notes")
    if isinstance(df, pd.DataFrame) and not df.empty:
        for col in cols:
            if col not in df.columns:
                df[col] = ""
        return df[cols].copy()
    return pd.DataFrame(columns=cols)

def _monday(d: dt.date) -> dt.date:
    return d - dt.timedelta(days=d.weekday())

def _week_span(start_week: str | None, end_week: str | None) -> List[dt.date]:
    """Inclusive Monday→Monday list from plan start to end. Defaults to 12 weeks from today."""
    today = _monday(dt.date.today())
    try:
        sw = _monday(pd.to_datetime(start_week).date()) if start_week else today
    except Exception:
        sw = today
    try:
        ew = _monday(pd.to_datetime(end_week).date()) if end_week else (sw + dt.timedelta(weeks=11))
    except Exception:
        ew = sw + dt.timedelta(weeks=11)
    if ew < sw:
        sw, ew = ew, sw
    out: List[dt.date] = []
    cur = sw
    while cur <= ew:
        out.append(cur)
        cur += dt.timedelta(weeks=1)
    return out

def _week_cols(weeks: List[dt.date]) -> Tuple[List[dict], List[str]]:
    """Create DataTable column specs with Actual/Plan prefixes."""
    today = _monday(dt.date.today())
    ids: List[str] = []
    cols: List[dict] = [{"name": "Metric", "id": "metric", "editable": False}]
    for w in weeks:
        tag = "Actual" if w <= today else "Plan"
        wid = w.isoformat()
        ids.append(wid)
        cols.append({"name": f"{tag}\n{w.strftime('%m/%d/%y')}", "id": wid, "type": "numeric"})
    return cols, ids

def _blank_grid(metrics: List[str], week_ids: List[str]) -> pd.DataFrame:
    rows = []
    for m in metrics:
        r = {"metric": m}
        for wid in week_ids:
            r[wid] = 0.0
        rows.append(r)
    return pd.DataFrame(rows)

def _load_or_blank(key: str, metrics: List[str], week_ids: List[str]) -> pd.DataFrame:
    df = load_df(key)
    if isinstance(df, pd.DataFrame) and not df.empty:
        # ensure all columns exist
        for wid in week_ids:
            if wid not in df.columns:
                df[wid] = 0.0
        if "metric" not in df.columns:
            df.insert(0, "metric", metrics[: len(df)])
        return df[["metric"] + week_ids].copy()
    return _blank_grid(metrics, week_ids)

def _save_table(pid: int, tab_key: str, df: pd.DataFrame):
    save_df(f"plan_{pid}_{tab_key}", df if isinstance(df, pd.DataFrame) else pd.DataFrame())

# ──────────────────────────────────────────────────────────────────────────────
# roster / bulk columns

def _roster_columns() -> List[dict]:
    # Fixed schema (no weeks)
    names = [
        ("BRID", "brid"), ("Name", "name"), ("Class Reference", "class_ref"),
        ("Work Status", "work_status"), ("Role", "role"),
        ("FT/PT Status", "ftpt_status"), ("FT/PT Hours", "ftpt_hours"),
        ("Current Status", "current_status"),
        ("Training Start", "training_start"), ("Training End", "training_end"),
        ("Nesting Start", "nesting_start"), ("Nesting End", "nesting_end"),
        ("Production Start", "production_start"), ("Terminate Date", "terminate_date"),
        ("Team Leader", "team_leader"), ("AVP", "avp"),
        ("Business Area", "biz_area"), ("Sub Business Area", "sub_biz_area"),
        ("LOB", "lob"), ("LOA Date", "loa_date"), ("Back from LOA Date", "back_from_loa_date"), ("Site", "site"),
    ]
    cols = [{"name": " ", "id": "_select", "presentation": "markdown"}]  # spacer; checkboxes appear automatically
    for n, cid in names:
        # keep text input for all; you can switch date-like ones to 'type': 'datetime'
        cols.append({"name": n, "id": cid, "presentation": "input"})
    return cols

def _bulkfile_columns() -> List[dict]:
    return [
        {"name": "File Name", "id": "file_name"},
        {"name": "Extension", "id": "ext"},
        {"name": "File Size (in KB)", "id": "size_kb", "type": "numeric"},
        {"name": "Is Valid?", "id": "is_valid"},
        {"name": "File Status", "id": "status"},
    ]

# Required roster fields for uploads
_ROSTER_REQUIRED_IDS = [c["id"] for c in _roster_columns() if c["id"] not in ("_select",)]


# ──────────────────────────────────────────────────────────────────────────────
# layout builders
def _upper_summary_header_card() -> html.Div:
    return dbc.Card(
        dbc.CardBody([
            html.Div([
                # left: back and name
                html.Div([
                    dcc.Link(dbc.Button("🢀", id="plan-hdr-back", color="light", title="Back"),
                             href="/planning", className="me-2"),
                    html.Span(id="plan-hdr-name", className="fw-bold")
                ], className="d-flex align-items-center"),
                # right: icons
                html.Div([
                    dbc.Button("💾", id="btn-plan-save", color="light", title="Save", className="me-1"),
                    dbc.Button("⟳", id="btn-plan-refresh", color="light", title="Refresh", className="me-1"),
                    html.Div(id="plan-msg", className="text-success mt-2"),
                ], style={"display":"flex"}),
                
                html.Div([
                    dbc.Button("▼", id="plan-hdr-collapse", color="light", title="Collapse/Expand")
                ]),
            ], className="d-flex justify-content-between align-items-center mb-2 hhh"),
        ],style={"padding": "3px"}),
        className="mb-3 sandy"
    )

def _upper_summary_body_card() -> html.Div:
    return dbc.Card(
        dbc.CardBody([
            html.Div(id="plan-upper", className="cp-grid")  # filled by callback; read-only table
        ], class_name="gaurav"),
        className="mb-3 avyaan"
    )

def _lower_tabs() -> dbc.Card:
    return dbc.Card(
        dbc.CardBody([
            dbc.Tabs(id="plan-tabs", style={"display": "flex", "justifyContent": "space-between"} ,active_tab="tab-fw", children=[
                dbc.Tab(label="Forecast & Workload", tab_id="tab-fw",
                        children=[dash_table.DataTable(id="tbl-fw", editable=True,
                                                       style_as_list_view=True,
                                                       style_table={"overflowX":"auto"},
                                                       style_header={"whiteSpace":"pre"},
                                                       page_size=12)]),
                dbc.Tab(label="Headcount", tab_id="tab-hc",
                        children=[dash_table.DataTable(id="tbl-hc", editable=True,
                                                       style_as_list_view=True,
                                                       style_table={"overflowX":"auto"},
                                                       style_header={"whiteSpace":"pre"},
                                                       page_size=12)]),
                dbc.Tab(label="Attrition", tab_id="tab-attr",
                        children=[dash_table.DataTable(id="tbl-attr", editable=True,
                                                       style_as_list_view=True,
                                                       style_table={"overflowX":"auto"},
                                                       style_header={"whiteSpace":"pre"},
                                                       page_size=12)]),
                dbc.Tab(label="Shrinkage", tab_id="tab-shr",
                        children=[dash_table.DataTable(id="tbl-shr", editable=True,
                                                       style_as_list_view=True,
                                                       style_table={"overflowX":"auto"},
                                                       style_header={"whiteSpace":"pre"},
                                                       page_size=12)]),
                dbc.Tab(label="Training Lifecycle", tab_id="tab-train",
                        children=[dash_table.DataTable(id="tbl-train", editable=True,
                                                       style_as_list_view=True,
                                                       style_table={"overflowX":"auto"},
                                                       style_header={"whiteSpace":"pre"},
                                                       page_size=12)]),
                dbc.Tab(label="Ratios", tab_id="tab-ratio",
                        children=[dash_table.DataTable(id="tbl-ratio", editable=True,
                                                       style_as_list_view=True,
                                                       style_table={"overflowX":"auto"},
                                                       style_header={"whiteSpace":"pre"},
                                                       page_size=12)]),
                dbc.Tab(label="Seat Utilization", tab_id="tab-seat",
                        children=[dash_table.DataTable(id="tbl-seat", editable=True,
                                                       style_as_list_view=True,
                                                       style_table={"overflowX":"auto"},
                                                       style_header={"whiteSpace":"pre"},
                                                       page_size=12)]),
                dbc.Tab(label="Budget vs Actual", tab_id="tab-bva",
                        children=[dash_table.DataTable(id="tbl-bva", editable=True,
                                                       style_as_list_view=True,
                                                       style_table={"overflowX":"auto"},
                                                       style_header={"whiteSpace":"pre"},
                                                       page_size=12)]),
                dbc.Tab(label="New Hire", tab_id="tab-nh",
                        children=[dash_table.DataTable(id="tbl-nh", editable=True,
                                                       style_as_list_view=True,
                                                       style_table={"overflowX":"auto"},
                                                       style_header={"whiteSpace":"pre"},
                                                       page_size=12)]),
                dbc.Tab(label="Employee Roster", tab_id="tab-roster", children=[
                    dbc.Tabs([
                        dbc.Tab(label="Roster", tab_id="tab-roster-main", children=[
                            # keeps an audit trail so Undo can work
                            dcc.Store(id="emp-audit-log", data=[]),

                            html.Div([
                                dbc.Button("+ Add new", id="btn-emp-add", className="me-1", color="secondary"),
                                dbc.Button("Transfer & Promotion", id="btn-emp-tp", className="me-1", color="secondary", disabled=True),
                                dbc.Button("Move to LOA", id="btn-emp-loa", className="me-1", color="secondary", disabled=True),
                                dbc.Button("Back from LOA", id="btn-emp-back", className="me-1", color="secondary", disabled=True),
                                dbc.Button("Terminate", id="btn-emp-term", className="me-1", color="secondary", disabled=True),
                                dbc.Button("FT/PT Conversion", id="btn-emp-ftp", className="me-1", color="secondary", disabled=True),
                                dbc.Button("Undo", id="btn-emp-undo", className="me-1", color="secondary", disabled=True),
                                dbc.Button("Change Class", id="btn-emp-class", className="me-1", color="secondary", disabled=True),
                                dbc.Button("Remove", id="btn-emp-remove", className="me-2", color="secondary", disabled=True),
                                html.Span("Total: 00 Records", id="lbl-emp-total", className="me-2"),
                                dbc.Button("Workstatus Dataset", id="btn-emp-dl", color="warning", outline=True),
                                dcc.Download(id="dl-workstatus"),
                            ], className="mb-2 ashwini"),

                            dash_table.DataTable(
                                id="tbl-emp-roster",
                                editable=True,
                                row_selectable=False,      # checkboxes only appear when there is at least one row
                                selected_rows=[],
                                style_as_list_view=True,
                                style_table={"overflowX": "auto"},
                                page_size=10,
                            ),
                            
                            dbc.Modal(
                                [
                                    dbc.ModalHeader(dbc.ModalTitle("Please confirm")),
                                    dbc.ModalBody([
                                        html.Div("Are you sure?"),
                                        html.Div("Deleting this record will remove employee from database and will impact headcount projections.",
                                                className="text-muted small mt-1"),
                                    ]),
                                    dbc.ModalFooter([
                                        dbc.Button("Yes", id="btn-remove-ok", color="danger", className="me-2"),
                                        dbc.Button("No", id="btn-remove-cancel", color="secondary"),
                                    ])
                                ],
                                id="modal-remove", is_open=False, backdrop="static"
                            ),

                            # Change Class
                            dbc.Modal(
                                [
                                    dbc.ModalHeader(dbc.ModalTitle("Change Class Reference")),
                                    dbc.ModalBody([
                                        dbc.Row([
                                            dbc.Col(dbc.Label("Class Reference"), md=6),
                                            dbc.Col(dcc.Dropdown(id="inp-class-ref", placeholder="Select class reference…"), md=12),
                                        ]),
                                        html.Div(id="class-change-hint", className="text-muted small mt-2"),
                                    ]),
                                    dbc.ModalFooter([
                                        dbc.Button("Save", id="btn-class-save", color="primary", className="me-2"),
                                        dbc.Button("Cancel", id="btn-class-cancel", color="secondary"),
                                    ]),
                                ],
                                id="modal-class", is_open=False, backdrop="static"
                            ),

                            # FT/PT Conversion
                            dbc.Modal(
                                [
                                    dbc.ModalHeader(dbc.ModalTitle("FT/PT Conversion")),
                                    dbc.ModalBody([
                                        dbc.Row([
                                            dbc.Col(dbc.Label("Effective Date"), md=6),
                                            dbc.Col(dcc.DatePickerSingle(id="inp-ftp-date"), md=6),
                                        ], className="mb-2"),
                                        dbc.Row([
                                            dbc.Col(dbc.Label("Hours (weekly)"), md=6),
                                            dbc.Col(dbc.Input(id="inp-ftp-hours", type="number", min=1, step=0.5, placeholder="e.g. 20"), md=6),
                                        ], className="mb-2"),
                                        html.Div(id="ftp-who", className="text-muted small"),
                                    ]),
                                    dbc.ModalFooter([
                                        dbc.Button("Save", id="btn-ftp-save", color="primary", className="me-2"),
                                        dbc.Button("Cancel", id="btn-ftp-cancel", color="secondary"),
                                    ])
                                ],
                                id="modal-ftp", is_open=False, backdrop="static"
                            ),

                            # Move to LOA
                            dbc.Modal(
                                [
                                    dbc.ModalHeader(dbc.ModalTitle("Move to LOA")),
                                    dbc.ModalBody([
                                        dbc.Label("Effective Date (snapped to Monday)"),
                                        dcc.DatePickerSingle(id="inp-loa-date"),
                                        html.Div(id="loa-who", className="text-muted small mt-2"),
                                    ]),
                                    dbc.ModalFooter([
                                        dbc.Button("Save", id="btn-loa-save", color="primary", className="me-2"),
                                        dbc.Button("Cancel", id="btn-loa-cancel", color="secondary"),
                                    ])
                                ],
                                id="modal-loa", is_open=False, backdrop="static"
                            ),

                            # Back from LOA
                            dbc.Modal(
                                [
                                    dbc.ModalHeader(dbc.ModalTitle("Back from LOA")),
                                    dbc.ModalBody([
                                        dbc.Label("Effective Date (snapped to Monday)"),
                                        dcc.DatePickerSingle(id="inp-back-date"),
                                        html.Div(id="back-who", className="text-muted small mt-2"),
                                    ]),
                                    dbc.ModalFooter([
                                        dbc.Button("Save", id="btn-back-save", color="primary", className="me-2"),
                                        dbc.Button("Cancel", id="btn-back-cancel", color="secondary"),
                                    ])
                                ],
                                id="modal-back", is_open=False, backdrop="static"
                            ),

                            # Terminate
                            dbc.Modal(
                                [
                                    dbc.ModalHeader(dbc.ModalTitle("Terminate")),
                                    dbc.ModalBody([
                                        dbc.Label("Termination Date"),
                                        dcc.DatePickerSingle(id="inp-term-date"),
                                        html.Div(id="term-who", className="text-muted small mt-2"),
                                    ]),
                                    dbc.ModalFooter([
                                        dbc.Button("Save", id="btn-term-save", color="danger", className="me-2"),
                                        dbc.Button("Cancel", id="btn-term-cancel", color="secondary"),
                                    ])
                                ],
                                id="modal-term", is_open=False, backdrop="static"
                            ),

                            # Transfer & Promotion (combined modal with Tabs)
                            dbc.Modal(
                                [
                                    dbc.ModalHeader(dbc.ModalTitle("Transfer & Promotion")),
                                    dbc.ModalBody([
                                        dcc.Tabs(id="tp-active-tab", value="tp-transfer", children=[
                                            dcc.Tab(label="Transfer", value="tp-transfer", children=[
                                                dbc.Row([
                                                    dbc.Col(dcc.Dropdown(id="tp-ba", placeholder="Business Area"), md=3),
                                                    dbc.Col(dcc.Dropdown(id="tp-subba", placeholder="Sub Business Area"), md=3),
                                                    dbc.Col(dcc.Dropdown(id="tp-lob", placeholder="Channel"), md=3),
                                                    dbc.Col(dcc.Dropdown(id="tp-site", placeholder="Site"), md=3),
                                                ], className="mb-2"),
                                                dbc.Row([
                                                    dbc.Col(dbc.RadioItems(
                                                        id="tp-transfer-type",
                                                        options=[{"label":" Permanent", "value":"perm"},
                                                                {"label":" Interim", "value":"interim"}],
                                                        value="perm", inline=True
                                                    ), md=6),
                                                    dbc.Col(dbc.Checklist(
                                                        id="tp-new-class",
                                                        options=[{"label":" Transfer with new class", "value":"yes"}],
                                                        value=[]
                                                    ), md=6),
                                                ], className="mb-2"),
                                                dbc.Row([
                                                    dbc.Col(dbc.Label("Effective Date"), md=6),
                                                    dbc.Col(dcc.DatePickerSingle(id="tp-date-from"), md=6),
                                                ], className="mb-2"),
                                                dbc.Row([
                                                    dbc.Col(dbc.Label("Return Date (Interim only)"), md=6),
                                                    dbc.Col(dcc.DatePickerSingle(id="tp-date-to"), md=6),
                                                ], className="mb-2"),
                                                dbc.Row([
                                                    dbc.Col(dcc.Dropdown(id="tp-class-ref", placeholder="Class Reference (if new class)"), md=6),
                                                ]),
                                            ]),
                                            dcc.Tab(label="Promotion", value="tp-promo", children=[
                                                dbc.Row([
                                                    dbc.Col(dbc.RadioItems(
                                                        id="promo-type",
                                                        options=[{"label":" Permanent", "value":"perm"},
                                                                {"label":" Temporary", "value":"interim"}],
                                                        value="perm", inline=True
                                                    ), md=6),
                                                ], className="mb-2"),
                                                dbc.Row([
                                                    dbc.Col(dbc.Label("Effective Date"), md=6),
                                                    dbc.Col(dcc.DatePickerSingle(id="promo-date-from"), md=6),
                                                ], className="mb-2"),
                                                dbc.Row([
                                                    dbc.Col(dbc.Label("Stop Date (Temporary only)"), md=6),
                                                    dbc.Col(dcc.DatePickerSingle(id="promo-date-to"), md=6),
                                                ], className="mb-2"),
                                                dbc.Row([
                                                    dbc.Col(dcc.Dropdown(
                                                        id="promo-role",
                                                        placeholder="Role (e.g., Team Leader, Trainer, SME, QA …)"
                                                    ), md=6),
                                                ]),
                                            ]),
                                            dcc.Tab(label="Transfer with Promotion", value="tp-both", children=[
                                                dbc.Row([
                                                    dbc.Col(dcc.Dropdown(id="twp-ba", placeholder="Business Area"), md=3),
                                                    dbc.Col(dcc.Dropdown(id="twp-subba", placeholder="Sub Business Area"), md=3),
                                                    dbc.Col(dcc.Dropdown(id="twp-lob", placeholder="Channel"), md=3),
                                                    dbc.Col(dcc.Dropdown(id="twp-site", placeholder="Site"), md=3),
                                                ], className="mb-2"),
                                                dbc.Row([
                                                    dbc.Col(dbc.RadioItems(
                                                        id="twp-type",
                                                        options=[{"label":" Permanent", "value":"perm"},
                                                                {"label":" Temporary", "value":"interim"}],
                                                        value="perm", inline=True
                                                    ), md=6),
                                                    dbc.Col(dbc.Checklist(
                                                        id="twp-new-class",
                                                        options=[{"label":" Transfer with new class", "value":"yes"}],
                                                        value=[]
                                                    ), md=6),
                                                ], className="mb-2"),
                                                dbc.Row([
                                                    dbc.Col(dbc.Label("Effective Date"), md=6),
                                                    dbc.Col(dcc.DatePickerSingle(id="twp-date-from"), md=6),
                                                ], className="mb-2"),
                                                dbc.Row([
                                                    dbc.Col(dbc.Label("Stop Date (Temporary only)"), md=6),
                                                    dbc.Col(dcc.DatePickerSingle(id="twp-date-to"), md=6),
                                                ], className="mb-2"),
                                                dbc.Row([
                                                    dbc.Col(dcc.Dropdown(id="twp-class-ref", placeholder="Class Reference (if new class)"), md=6),
                                                    dbc.Col(dcc.Dropdown(id="twp-role", placeholder="Role"), md=6),
                                                ])
                                            ]),
                                        ]),
                                        html.Div(id="tp-who", className="text-muted small mt-2"),
                                    ]),
                                    dbc.ModalFooter([
                                        dbc.Button("Save", id="btn-tp-save", color="primary", className="me-2"),
                                        dbc.Button("Cancel", id="btn-tp-cancel", color="secondary"),
                                    ])
                                ],
                                id="modal-tp", is_open=False, size="xl", backdrop="static"
                            ),

                        ]),
                        dbc.Tab(label="Bulk Upload", tab_id="tab-roster-bulk", children=[
                            dbc.Tab(label="Upload File", tab_id="tab-emp-upload", children=[
                                html.Div([
                                    dcc.Upload(id="up-roster-bulk", children=html.Div(["⬆️ Upload CSV/XLSX"]),
                                                multiple=False, className="upload-box"),
                                    dbc.Button("Download Template", id="btn-template-dl", color="secondary"),
                                    dcc.Download(id="dl-template")
                                ], className="mb-2 batra", style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"}),
                                dash_table.DataTable(
                                    id="tbl-bulk-files", editable=False, style_as_list_view=True,
                                    filter_action="native", sort_action="native",
                                    style_table={"overflowX":"auto"}, page_size=10
                                ),
                            ]),
                        ])
                    ], style={"marginBottom": "1rem", "marginTop": "1rem"})
                ]),
                dbc.Tab(label="Notes", tab_id="tab-notes", children=[
                    dbc.Row([
                        dbc.Col(dcc.Textarea(id="notes-input", style={"width":"100%","height":"120px"},
                                             placeholder="Write a note and click Save…"), md=9, class_name="panwar"),
                        dbc.Col(dbc.Button("Save Note", id="btn-note-save", color="primary", className="mt-2"), md=3, class_name="aggarwal"),
                    ], className="mb-2"),
                    dash_table.DataTable(
                        id="tbl-notes",
                        columns=[
                            {"name":"Date","id":"when"},
                            {"name":"User","id":"user"},
                            {"name":"Note","id":"note"},
                        ],
                        data=[], editable=False, style_as_list_view=True, page_size=10,
                        style_table={"overflowX":"auto"}
                    )
                ]),
            ]),
        ], class_name="ankit"),
        className="mb-3"
    )

def _add_employee_modal() -> dbc.Modal:
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Add New Employee"), style={"background":"#2f3747", "color": "white"}),
            dbc.ModalBody([
                dbc.CardBody([
                    html.Div(id="modal-roster-crumb", className="text-muted small mb-2", style={"font-size": "medium"}),
                ], class_name="abc"),
                dbc.Row([
                    dbc.Col(dbc.Input(id="inp-brid", placeholder="BRID"), md=3, class_name="sandeep"),
                    dbc.Col(dbc.Input(id="inp-name", placeholder="Employee Name"), md=3, class_name="sandeep"),
                    dbc.Col(dbc.Input(id="inp-tl", placeholder="Team Leader"), md=3, class_name="sandeep"),
                    dbc.Col(dbc.Input(id="inp-avp", placeholder="AVP"), md=3, class_name="sandeep"),
                ], className="mb-2 negi"),
                dbc.Row([
                    dbc.Col(dbc.RadioItems(
                        id="inp-ftpt", value="Full-time",
                        options=[{"label":"Full-time","value":"Full-time"},
                                 {"label":"Part-time","value":"Part-time"}],
                        inline=True
                    ), md=4),
                    dbc.Col(dbc.Input(id="inp-role", placeholder="Role"), md=4),
                                        dbc.Col(dcc.DatePickerSingle(id="inp-prod-date", placeholder="Production Date"), md=4),
                ], className="mb-2 govinda"),
            ], style={"display": "flex", "flexDirection": "column"}),
            dbc.ModalFooter([
                dbc.Button("Save", id="btn-emp-modal-save", color="primary", className="me-2"),
                dbc.Button("Cancel", id="btn-emp-modal-cancel", color="secondary"),
            ]),
        ],
        id="modal-emp-add", is_open=False, size="lg", backdrop="static"
    )

def layout_for_plan(pid: int) -> html.Div:
    """Main page UI; data comes from callbacks."""
    return dbc.Container([
        dcc.Store(id="plan-detail-id", data=pid),
        dcc.Store(id="plan-upper-collapsed", data=False),
        dcc.Store(id="plan-type"),         # "Volume Based", "Billable Hours Based", etc.
        dcc.Store(id="plan-weeks"),        # list[str] ISO Mondays
        # _upper_summary_card(),
        dcc.Interval(id="plan-msg-timer", interval=5000, n_intervals=0, disabled=True),
        _upper_summary_header_card(),
        _upper_summary_body_card(),
        _lower_tabs(),
        _add_employee_modal(),
    ], fluid=True)

# Tiny, hidden skeleton so Dash knows all IDs on initial load
def plan_detail_validation_layout() -> html.Div:
    dummy_cols = [{"name": "Metric", "id": "metric"}] + [{"name": "Plan\n01/01/70", "id": "1970-01-01"}]
    return html.Div(
        [
            dcc.Store(id="plan-detail-id"),
            dcc.Store(id="plan-upper-collapsed"),
            dcc.Store(id="plan-type"),
            dcc.Store(id="plan-weeks"),
            dcc.Interval(id="plan-msg-timer"),

            html.Div(id="plan-hdr-name"),
            html.Div(id="plan-upper"),
            html.Div(id="plan-msg"),

            dash_table.DataTable(id="tbl-fw", columns=dummy_cols, data=[]),
            dash_table.DataTable(id="tbl-hc", columns=dummy_cols, data=[]),
            dash_table.DataTable(id="tbl-attr", columns=dummy_cols, data=[]),
            dash_table.DataTable(id="tbl-shr", columns=dummy_cols, data=[]),
            dash_table.DataTable(id="tbl-train", columns=dummy_cols, data=[]),
            dash_table.DataTable(id="tbl-ratio", columns=dummy_cols, data=[]),
            dash_table.DataTable(id="tbl-seat", columns=dummy_cols, data=[]),
            dash_table.DataTable(id="tbl-bva", columns=dummy_cols, data=[]),
            dash_table.DataTable(id="tbl-nh", columns=dummy_cols, data=[]),

            dash_table.DataTable(id="tbl-emp-roster", columns=[{"name":"BRID","id":"brid"}], data=[]),

            # Use the new bulk file grid, NOT tbl-roster-bulk
            dash_table.DataTable(
                id="tbl-bulk-files",
                columns=[{"name":"File Name","id":"file_name"},{"name":"Extension","id":"ext"},
                         {"name":"File Size (in KB)","id":"size_kb"},{"name":"Is Valid?","id":"is_valid"},
                         {"name":"File Status","id":"status"}],
                data=[]
            ),

            dash_table.DataTable(id="tbl-notes", columns=[{"name":"Date","id":"when"},
                                                          {"name":"User","id":"user"},
                                                          {"name":"Note","id":"note"}], data=[]),

            # ONE instance only
            dcc.Upload(id="up-roster-bulk"),
            dcc.Download(id="dl-template"),
            dcc.Download(id="dl-workstatus"),

            dbc.Button(id="btn-plan-save"),
            dbc.Button(id="btn-plan-refresh"),
            dbc.Button(id="plan-hdr-collapse"),
            dbc.Button(id="btn-template-dl"),
            dbc.Button(id="btn-emp-modal-save"),
            dbc.Button(id="btn-emp-modal-cancel"),
            dbc.Modal(id="modal-emp-add"),
            dcc.DatePickerSingle(id="inp-prod-date"),
            dbc.RadioItems(id="inp-ftpt"),
            dbc.Input(id="inp-brid"), dbc.Input(id="inp-name"),
            dbc.Input(id="inp-role"), dbc.Input(id="inp-tl"), dbc.Input(id="inp-avp"),
            html.Div(id="modal-roster-crumb"),
            dbc.Tabs(id="plan-tabs"),
        ],
        style={"display": "none"}
    )


def _parse_upload(contents: str, filename: str) -> Tuple[pd.DataFrame, dict]:
    """Return (records_df, ledger_row). Valid if required headers present."""
    if not contents or not filename:
        return pd.DataFrame(), {}
    header, b64 = contents.split(",", 1)
    raw = base64.b64decode(b64)
    ext = filename.split(".")[-1].lower()
    try:
        if ext in ("csv",):
            df = pd.read_csv(io.BytesIO(raw))
        elif ext in ("xlsx","xls"):
            df = pd.read_excel(io.BytesIO(raw))
        else:
            return pd.DataFrame(), {"file_name": filename, "ext": ext, "size_kb": round(len(raw)/1024,1),
                                    "is_valid": "No", "status": "Unsupported"}
    except Exception:
        return pd.DataFrame(), {"file_name": filename, "ext": ext, "size_kb": round(len(raw)/1024,1),
                                "is_valid": "No", "status": "Read Error"}

    # normalize columns (accept header names as shown in UI)
    rename_map = {c["name"]: c["id"] for c in _roster_columns() if c["id"] not in ("_select",)}
    # Use case-insensitive match
    lower_map = {k.lower(): v for k,v in rename_map.items()}
    df = df.rename(columns={col: lower_map.get(str(col).lower(), col) for col in df.columns})

    missing = [cid for cid in _ROSTER_REQUIRED_IDS if cid not in df.columns]
    valid = len(missing) == 0
    ledger = {"file_name": filename, "ext": ext, "size_kb": round(len(raw)/1024,1),
              "is_valid": "Yes" if valid else "No", "status": "Loaded" if valid else f"Missing: {', '.join(missing[:3])}"}
    if not valid:
        return pd.DataFrame(), ledger

    return df[_ROSTER_REQUIRED_IDS].copy(), ledger


# ──────────────────────────────────────────────────────────────────────────────
# callbacks

def register_plan_detail(app: dash.Dash):

    @app.callback(
        Output("plan-detail-id", "data"),
        Input("url-router", "pathname"),
        prevent_initial_call=False,
    )
    def _capture_pid(pathname):
        path = (pathname or "").rstrip("/")
        if not path.startswith("/plan/"):
            raise dash.exceptions.PreventUpdate
        try:
            return int(path.rsplit("/", 1)[-1])
        except Exception:
            return no_update

    @app.callback(
        Output("plan-hdr-name", "children"),
        Output("plan-type", "data"),
        Output("plan-weeks", "data"),
        Output("tbl-fw", "columns"),
        Output("tbl-hc", "columns"),
        Output("tbl-attr", "columns"),
        Output("tbl-shr", "columns"),
        Output("tbl-train", "columns"),
        Output("tbl-ratio", "columns"),
        Output("tbl-seat", "columns"),
        Output("tbl-bva", "columns"),
        Output("tbl-nh", "columns"),
        Output("tbl-emp-roster", "columns"),
        Output("tbl-bulk-files", "columns"),        # <- changed id here
        Output("tbl-notes", "columns"),
        Input("plan-detail-id", "data"),
        State("url-router", "pathname"),
        prevent_initial_call=False,
    )
    def _init_cols(pid, pathname):
        path = (pathname or "").rstrip("/")
        if not (isinstance(pid, int) and path.startswith("/plan/")):
            raise dash.exceptions.PreventUpdate

        p = get_plan(pid) or {}
        name = p.get("plan_name") or f"Plan {pid}"
        ptype = p.get("plan_type") or "Volume Based"
        weeks = _week_span(p.get("start_week"), p.get("end_week"))
        cols, _ = _week_cols(weeks)

        notes_cols = [{"name":"Date","id":"when"},{"name":"User","id":"user"},{"name":"Note","id":"note"}]

        return (
            name, ptype, [w.isoformat() for w in weeks],
            cols, cols, cols, cols, cols, cols, cols, cols, cols,           # weekly tables
            _roster_columns(),                                               # roster: fixed schema
            _bulkfile_columns(),                                             # bulk files table: fixed schema
            notes_cols,
        )

    @app.callback(
        Output("plan-upper", "children"),
        Output("tbl-fw", "data"), Output("tbl-hc", "data"),
        Output("tbl-attr", "data"), Output("tbl-shr", "data"),
        Output("tbl-train", "data"), Output("tbl-ratio", "data"),
        Output("tbl-seat", "data"), Output("tbl-bva", "data"),
        Output("tbl-nh", "data"),
        Output("tbl-emp-roster", "data"),
        Output("tbl-bulk-files", "data"),
        Output("tbl-notes", "data"),
        Input("plan-type", "data"),
        State("plan-detail-id", "data"),
        State("tbl-fw", "columns"),
        prevent_initial_call=False,
    )
    def _fill_tables(ptype, pid, fw_cols):
        if not (pid and fw_cols):
            raise dash.exceptions.PreventUpdate

        week_ids = [c["id"] for c in fw_cols if c["id"] != "metric"]

        # (same plan-type spec as your current version) -------------->
        def _plan_specs(k: str) -> Dict[str, List[str]]:
            k = (k or "").strip().lower()
            if k.startswith("volume"):
                return {"fw": ["Forecast","Tactical Forecast","Actual Volume","Budgeted AHT/SUT","Target AHT/SUT","Actual AHT/SUT","Occupancy","Shift Inflex"],
                        "upper": ["FTE Required @ Actual Volume","FTE Over/Under MTP Vs Actual","FTE Over/Under Tactical Vs Actual","FTE Over/Under Budgeted Vs Actual","Projected Supply HC","Projected Handling Capacity (#)"]}
            if k.startswith("billable hours"):
                return {"fw": ["Billable Hours","AHT/SUT","Shrinkage","Training"],
                        "upper": ["Billable FTE Required (#)","Headcount Required With Shrinkage (#)","FTE Over/Under (#)"]}
            if k.startswith("fte based billable"):
                return {"fw": ["Billable Txns","AHT/SUT","Efficiency","Shrinkage"],
                        "upper": ["Billable Transactions","FTE Required (#)","FTE Over/Under (#)"]}
            return {"fw": ["Billable FTE Required","Shrinkage","Training"],
                    "upper": ["FTE Required (#)","FTE Over/Under (#)"]}
        spec = _plan_specs(ptype)
        # <----------------------------------------------------------

        # Weekly tabs from disk or blanks
        fw   = _load_or_blank(f"plan_{pid}_fw",   spec["fw"], week_ids)
        hc   = _load_or_blank(f"plan_{pid}_hc",   ["Budgeted HC (#)","Planned/Tactical HC (#)","Actual Agent HC (#)","SME Billable HC (#)"], week_ids)
        att  = _load_or_blank(f"plan_{pid}_attr", ["Planned Attrition HC (#)","Actual Attrition HC (#)","Attrition %"], week_ids)
        shr  = _load_or_blank(f"plan_{pid}_shr",  ["OOO Shrink Hours (#)","Inoffice Shrink Hours (#)","OOO Shrinkage %","Inoffice Shrinkage %","Overall Shrinkage %"], week_ids)
        trn  = _load_or_blank(f"plan_{pid}_train",["Training Start (#)","Training End (#)","Nesting Start (#)","Nesting End (#)"], week_ids)
        rat  = _load_or_blank(f"plan_{pid}_ratio",["AHT (sec)","ASA (sec)","SL %","Abandon %"], week_ids)
        seat = _load_or_blank(f"plan_{pid}_seat", ["Seats Required (#)","Seats Available (#)","Utilization %"], week_ids)
        bva  = _load_or_blank(f"plan_{pid}_bva",  ["Budgeted FTE (#)","Actual FTE (#)","Variance (#)"], week_ids)
        nh   = _load_or_blank(f"plan_{pid}_nh",   ["Planned New Hire HC (#)","Actual New Hire HC (#)"], week_ids)

        roster_df = _load_or_empty_roster(pid)
        if roster_df.empty:
            roster_df = _prefill_roster_from_master(pid, persist=True)
        bulk_df   = _load_or_empty_bulk_files(pid)
        notes_df  = _load_or_empty_notes(pid)


        # Upper (read-only) table
        upper_df = _blank_grid(spec["upper"], week_ids)
        upper = dash_table.DataTable(
            id="tbl-upper",
            data=upper_df.to_dict("records"),
            columns=[{"name":"Metric","id":"metric","editable":False}] + [{"name":c["name"],"id":c["id"]} for c in fw_cols if c["id"]!="metric"],
            editable=False, style_as_list_view=True, style_table={"overflowX":"auto"}, style_header={"whiteSpace":"pre"},
        )

        return (
            upper,
            fw.to_dict("records"), hc.to_dict("records"),
            att.to_dict("records"), shr.to_dict("records"),
            trn.to_dict("records"), rat.to_dict("records"),
            seat.to_dict("records"), bva.to_dict("records"),
            nh.to_dict("records"),
            roster_df.to_dict("records"),
            bulk_df.to_dict("records"),
            notes_df.to_dict("records"),
        )


    @app.callback(
        Output("plan-msg", "children", allow_duplicate=True),
        Output("plan-msg-timer", "disabled", allow_duplicate=True),
        Input("btn-plan-save", "n_clicks"),
        State("plan-detail-id", "data"),
        State("tbl-fw", "data"), State("tbl-hc", "data"),
        State("tbl-attr", "data"), State("tbl-shr", "data"),
        State("tbl-train", "data"), State("tbl-ratio", "data"),
        State("tbl-seat", "data"), State("tbl-bva", "data"),
        State("tbl-nh", "data"), State("tbl-emp-roster", "data"),
        State("tbl-notes", "data"),
        prevent_initial_call=True
    )
    def _save(_n, pid, fw, hc, attr, shr, trn, rat, seat, bva, nh, emp, bulk_files, notes):
        if not pid:
            raise dash.exceptions.PreventUpdate
        _save_table(pid, "fw",         pd.DataFrame(fw or []))
        _save_table(pid, "hc",         pd.DataFrame(hc or []))
        _save_table(pid, "attr",       pd.DataFrame(attr or []))
        _save_table(pid, "shr",        pd.DataFrame(shr or []))
        _save_table(pid, "train",      pd.DataFrame(trn or []))
        _save_table(pid, "ratio",      pd.DataFrame(rat or []))
        _save_table(pid, "seat",       pd.DataFrame(seat or []))
        _save_table(pid, "bva",        pd.DataFrame(bva or []))
        _save_table(pid, "nh",         pd.DataFrame(nh or []))
        _save_table(pid, "emp",        pd.DataFrame(emp or []))
        _save_table(pid, "notes",      pd.DataFrame(notes or []))
        return "Saved ✓", False


    @app.callback(
        Output("plan-upper-collapsed", "data"),
        Output("plan-upper", "style"),
        Output("plan-hdr-collapse", "children"),
        Input("plan-hdr-collapse", "n_clicks"),
        State("plan-upper-collapsed", "data"),
        prevent_initial_call=False
    )
    def _toggle_upper(n_clicks, collapsed):
        # collapsed is False on first load
        collapsed = bool(collapsed)
        if n_clicks:  # toggle on click
            collapsed = not collapsed
        style = {"display": "none"} if collapsed else {"display": "block"}
        # icon: up arrow when visible (you can switch to ▸ if you prefer)
        icon = "▾" if collapsed else "▴"
        return collapsed, style, icon


    @app.callback(
        Output("plan-msg", "children", allow_duplicate=True),
        Output("plan-msg-timer", "disabled", allow_duplicate=True),
        Input("btn-plan-refresh", "n_clicks"),
        prevent_initial_call=True
    )
    def _refresh_msg(_n):
        return "Refreshed ✓", False
    
    @app.callback(
        Output("plan-msg", "children"),
        Output("plan-msg-timer", "disabled"),
        Input("plan-msg-timer", "n_intervals"),
        prevent_initial_call=True
    )
    def _clear_msg(_ticks):
        return "", True  # clear message and stop timer

    @app.callback(
        Output("btn-emp-tp",    "disabled", allow_duplicate=True),
        Output("btn-emp-loa",   "disabled", allow_duplicate=True),
        Output("btn-emp-back",  "disabled", allow_duplicate=True),
        Output("btn-emp-term",  "disabled", allow_duplicate=True),
        Output("btn-emp-ftp",   "disabled", allow_duplicate=True),
        Output("btn-emp-undo",  "disabled", allow_duplicate=True),
        Output("btn-emp-class", "disabled", allow_duplicate=True),
        Output("btn-emp-remove","disabled", allow_duplicate=True),
        Input("tbl-emp-roster", "data"),
        Input("tbl-emp-roster", "selected_rows"),
        prevent_initial_call=True
    )
    def _toggle_roster_buttons(data, selected_rows):
        has_rows = bool(data) and len(data) > 0
        has_sel  = has_rows and bool(selected_rows)
        disabled = not has_sel
        return (disabled,)*8

    @app.callback(
        Output("modal-emp-add", "is_open"),
        Output("modal-roster-crumb", "children"),
        Input("btn-emp-add", "n_clicks"),
        Input("btn-emp-modal-cancel", "n_clicks"),
        State("modal-emp-add", "is_open"),
        State("plan-detail-id", "data"),
        prevent_initial_call=True
    )
    def _modal_toggle(n_add, n_cancel, is_open, pid):
        trigger = ctx.triggered_id
        if trigger == "btn-emp-add":
            bda, sba, chan, site = _extract_ba_parts(pid)
            crumb = " > ".join([x for x in (bda, sba, chan, site) if x])
            return True, crumb
        return False, ""


    @app.callback(
        Output("tbl-emp-roster", "data", allow_duplicate=True),
        Output("lbl-emp-total", "children", allow_duplicate=True),
        Output("modal-emp-add", "is_open", allow_duplicate=True),
        Output("plan-msg", "children", allow_duplicate=True),
        Output("plan-msg-timer", "disabled", allow_duplicate=True),
        Input("btn-emp-modal-save", "n_clicks"),
        State("tbl-emp-roster", "data"),
        State("inp-brid", "value"), State("inp-name", "value"),
        State("inp-ftpt", "value"), State("inp-role", "value"),
        State("inp-prod-date", "date"), State("inp-tl", "value"), State("inp-avp", "value"),
        State("plan-detail-id", "data"),
        prevent_initial_call=True
    )
    def _add_emp(_n, data, brid, name, ftpt, role, prod_date, tl, avp, pid):
        data = data or []
        # enforce unique BRID
        if not brid or any(str(r.get("brid","")).strip()==str(brid).strip() for r in data):
            return data, f"Total: {len(data):02d} Records", False, "BRID exists or missing ✗", False

        bda, sba, chan, site = _extract_ba_parts(pid)

        row = {cid: "" for cid in _ROSTER_REQUIRED_IDS}
        row.update({
            "brid": brid,
            "name": name or "",
            "ftpt_status": ftpt or "",
            "role": role or "",
            "production_start": prod_date or "",
            "team_leader": tl or "",
            "avp": avp or "",
            "biz_area": bda,
            "sub_biz_area": sba,
            "lob": chan,
            # harmless extra key (will persist even if not shown as a column)
            "site": site,
        })


        new = data + [row]
        save_df(f"plan_{pid}_emp", pd.DataFrame(new))
        return new, f"Total: {len(new):02d} Records", False, "Employee added ✓", False


    @app.callback(
        Output("lbl-emp-total", "children"),
        Input("tbl-emp-roster", "data"),
        prevent_initial_call=False
    )
    def _update_emp_total(data):
        df = pd.DataFrame(data or [])
        if "brid" in df.columns:
            s = df["brid"].astype(str).str.strip()
            n = s.replace({"": np.nan, "nan": np.nan}).nunique(dropna=True)
        else:
            n = len(df)
        return f"Total: {int(n):02d} Records"


    @app.callback(
        Output("dl-workstatus", "data"),
        Input("btn-emp-dl", "n_clicks"),
        State("tbl-emp-roster", "data"),
        prevent_initial_call=True
    )
    def _download_workstatus(_n, data):
        df = pd.DataFrame(data or [])
        return dcc.send_data_frame(df.to_csv, "workstatus_dataset.csv", index=False)

    @app.callback(
        Output("tbl-bulk-files", "data", allow_duplicate=True),
        Output("tbl-emp-roster", "data", allow_duplicate=True),
        Output("lbl-emp-total", "children", allow_duplicate=True),
        Output("plan-msg", "children", allow_duplicate=True),
        Output("plan-msg-timer", "disabled", allow_duplicate=True),
        Input("up-roster-bulk", "contents"),
        State("up-roster-bulk", "filename"),
        State("tbl-bulk-files", "data"),
        State("tbl-emp-roster", "data"),
        State("plan-detail-id", "data"),
        prevent_initial_call=True
    )
    def _ingest_bulk(contents, filename, files_data, roster_data, pid):
        if not contents:
            raise dash.exceptions.PreventUpdate
        files_data = files_data or []
        roster_data = roster_data or []

        recs_df, ledger = _parse_upload(contents, filename)
        files_data.append(ledger or {"file_name": filename, "ext":"", "size_kb":0, "is_valid":"No", "status":"Invalid"})

        # merge unique by BRID if valid
        if not recs_df.empty and "brid" in recs_df.columns:
            existing = {str(r.get("brid","")).strip(): i for i, r in enumerate(roster_data)}
            for _, row in recs_df.iterrows():
                key = str(row.get("brid","")).strip()
                if not key:
                    continue
                if key in existing:
                    # update existing row (simple upsert)
                    roster_data[existing[key]].update({k: row.get(k, roster_data[existing[key]].get(k)) for k in _ROSTER_REQUIRED_IDS})
                else:
                    # ensure all keys exist
                    new_row = {cid: row.get(cid, "") for cid in _ROSTER_REQUIRED_IDS}
                    roster_data.append(new_row)

            # persist both
            save_df(f"plan_{pid}_emp", pd.DataFrame(roster_data))
            save_df(f"plan_{pid}_bulk_files", pd.DataFrame(files_data))

            return (files_data, roster_data, f"Total: {len(roster_data):02d} Records", "Bulk file loaded ✓", False)

        # persist file ledger even if invalid
        save_df(f"plan_{pid}_bulk_files", pd.DataFrame(files_data))
        return (files_data, roster_data, f"Total: {len(roster_data):02d} Records", "Bulk file invalid ✗", False)

    @app.callback(
        Output("dl-template", "data"),
        Input("btn-template-dl", "n_clicks"),
        prevent_initial_call=True
    )
    def _download_template(_n):
        cols = [c["name"] for c in _roster_columns() if c["id"] not in ("_select",)]
        df = pd.DataFrame(columns=cols)
        return dcc.send_data_frame(df.to_csv, "employee_roster_template.csv", index=False)

    @app.callback(
        Output("tbl-notes", "data", allow_duplicate=True),
        Output("plan-msg", "children", allow_duplicate=True),
        Output("plan-msg-timer", "disabled", allow_duplicate=True),
        Input("btn-note-save", "n_clicks"),
        State("notes-input", "value"),
        State("tbl-notes", "data"),
        State("plan-detail-id", "data"),
        prevent_initial_call=True
    )
    def _save_note(_n, text, data, pid):
        if not (text and text.strip()):
            raise dash.exceptions.PreventUpdate
        data = data or []
        stamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
        # If you have a real user, replace "User"
        row = {"when": stamp, "user": "User", "note": text.strip()}
        data = [row] + data
        save_df(f"plan_{pid}_notes", pd.DataFrame(data))
        return data, "Note saved ✓", False

    @app.callback(
        Output("tbl-emp-roster", "row_selectable"),
        Output("tbl-emp-roster", "selected_rows"),
        Input("tbl-emp-roster", "data"),
        prevent_initial_call=False
    )
    def _roster_selectability(data):
        has_rows = bool(data) and len(data) > 0
        # If no rows: remove the checkbox column and clear selection
        return ("multi" if has_rows else False, [] if not has_rows else no_update)
    
    # ------------- helpers used by the action callbacks -------------
    def _monday_iso(d) -> str:
        if not d: return ""
        try:
            dd = pd.to_datetime(d).date()
            dd = dd - dt.timedelta(days=dd.weekday())
            return dd.isoformat()
        except Exception:
            return ""

    def _selected_brids(data: list[dict], sel_rows: list[int]) -> list[str]:
        data = data or []
        sel_rows = sel_rows or []
        out = []
        for i in sel_rows:
            try:
                b = str((data[i] or {}).get("brid", "")).strip()
                if b:
                    out.append(b)
            except Exception:
                pass
        return out

    def _audit_append(audit: list[dict], action: str, rows_before: list[dict], brids: list[str]) -> list[dict]:
        stamp = pd.Timestamp.now().isoformat(timespec="seconds")
        audit = list(audit or [])
        for r in rows_before:
            if str(r.get("brid","")).strip() in brids:
                audit.append({"when": stamp, "action": action, "brid": r.get("brid"), "snapshot": r})
        # Cap history (optional): keep last 500 edits
        return audit[-500:]

    def _roster_defaults_from_crumb(pid: int) -> dict:
        # same crumb you already build for Add New Employee
        p = get_plan(pid) or {}
        return dict(
            biz_area=p.get("business_area",""),
            sub_biz_area=p.get("sub_business_area",""),
            lob=p.get("lob",""),
            site=p.get("site",""),
        )

    # Fill class/role/BA dropdowns dynamically when modal opens
    @app.callback(
        Output("inp-class-ref","options"),
        Output("tp-class-ref","options"),
        Output("twp-class-ref","options"),
        Output("promo-role","options"),
        Output("twp-role","options"),
        Output("tp-ba","options"), Output("tp-subba","options"), Output("tp-lob","options"), Output("tp-site","options"),
        Output("twp-ba","options"), Output("twp-subba","options"), Output("twp-lob","options"), Output("twp-site","options"),
        Input("modal-class","is_open"),
        Input("modal-tp","is_open"),
        State("tbl-emp-roster","data"),
        State("plan-detail-id","data"),
        prevent_initial_call=False
    )
    def _fill_modal_options(is_open_class, is_open_tp, data, pid):
        df = pd.DataFrame(data or [])
        # Class refs from current roster known values
        class_opts = sorted(df.get("class_ref", pd.Series([], dtype=str)).dropna().astype(str).unique().tolist())
        class_opt_dicts = [{"label": c, "value": c} for c in class_opts]

        # Roles from roster values + a few common ones
        existing_roles = set(df.get("role", pd.Series([], dtype=str)).dropna().astype(str).unique().tolist())
        base_roles = {"Agent","Team Leader","Trainer","HR","SME","QA","Supervisor","Manager"}
        role_opts = [{"label": r, "value": r} for r in sorted(existing_roles | base_roles)]

        # BA/SubBA/LOB/Site from roster (fallback to plan crumb if empty)
        def _opts(col):
            vals = sorted(df.get(col, pd.Series([], dtype=str)).dropna().astype(str).unique().tolist())
            if not vals:
                v = _roster_defaults_from_crumb(pid).get(col, "")
                if v: vals = [v]
            return [{"label": v, "value": v} for v in vals]

        return (
            class_opt_dicts, class_opt_dicts, class_opt_dicts,     # inp-class-ref, tp-class-ref, twp-class-ref
            role_opts, role_opts,                                  # promo-role,  twp-role
            _opts("biz_area"), _opts("sub_biz_area"), _opts("lob"), _opts("site"),
            _opts("biz_area"), _opts("sub_biz_area"), _opts("lob"), _opts("site"),
        )

    # Enable/disable action buttons (reuses your version)
    @app.callback(
        Output("btn-emp-tp",    "disabled"),
        Output("btn-emp-loa",   "disabled"),
        Output("btn-emp-back",  "disabled"),
        Output("btn-emp-term",  "disabled"),
        Output("btn-emp-ftp",   "disabled"),
        Output("btn-emp-undo",  "disabled"),
        Output("btn-emp-class", "disabled"),
        Output("btn-emp-remove","disabled"),
        Input("tbl-emp-roster", "data"),
        Input("tbl-emp-roster", "selected_rows"),
        prevent_initial_call=False
    )
    def _toggle_roster_buttons2(data, selected_rows):
        has_rows = bool(data) and len(data) > 0
        has_sel  = has_rows and bool(selected_rows)
        disabled = not has_sel
        return (disabled,)*8

    # Open modals
    @app.callback(
        Output("modal-remove","is_open"),
        Output("modal-class","is_open"),
        Output("modal-ftp","is_open"),
        Output("modal-loa","is_open"),
        Output("modal-back","is_open"),
        Output("modal-term","is_open"),
        Output("modal-tp","is_open"),
        Output("class-change-hint","children"),
        Output("ftp-who","children"),
        Output("loa-who","children"),
        Output("back-who","children"),
        Output("term-who","children"),
        Output("tp-who","children"),
        Input("btn-emp-remove","n_clicks"),
        Input("btn-emp-class","n_clicks"),
        Input("btn-emp-ftp","n_clicks"),
        Input("btn-emp-loa","n_clicks"),
        Input("btn-emp-back","n_clicks"),
        Input("btn-emp-term","n_clicks"),
        Input("btn-emp-tp","n_clicks"),
        Input("btn-remove-cancel","n_clicks"),
        Input("btn-class-cancel","n_clicks"),
        Input("btn-ftp-cancel","n_clicks"),
        Input("btn-loa-cancel","n_clicks"),
        Input("btn-back-cancel","n_clicks"),
        Input("btn-term-cancel","n_clicks"),
        Input("btn-tp-cancel","n_clicks"),
        State("tbl-emp-roster","selected_rows"),
        State("tbl-emp-roster","data"),
        prevent_initial_call=True
    )
    def _open_any_modal(*args):
        # unpack context
        selected_rows = args[-2]
        data = args[-1]
        trig = ctx.triggered_id

        # Who is selected (for small hints)
        brids = _selected_brids(data, selected_rows)
        who = f"Selected: {', '.join(brids[:5])}" + ("" if len(brids) <= 5 else f" (+{len(brids)-5} more)")
        hint = "Choose a new class reference for selected employee(s)."

        open_map = dict(modal_remove=False, modal_class=False, modal_ftp=False,
                        modal_loa=False, modal_back=False, modal_term=False, modal_tp=False)

        if trig == "btn-emp-remove": open_map["modal_remove"] = True
        elif trig == "btn-emp-class": open_map["modal_class"]  = True
        elif trig == "btn-emp-ftp":   open_map["modal_ftp"]    = True
        elif trig == "btn-emp-loa":   open_map["modal_loa"]    = True
        elif trig == "btn-emp-back":  open_map["modal_back"]   = True
        elif trig == "btn-emp-term":  open_map["modal_term"]   = True
        elif trig == "btn-emp-tp":    open_map["modal_tp"]     = True
        else:
            # any cancel click -> close all
            pass

        return (open_map["modal_remove"], open_map["modal_class"], open_map["modal_ftp"],
                open_map["modal_loa"], open_map["modal_back"], open_map["modal_term"], open_map["modal_tp"],
                hint, who, who, who, who, who)

    # REMOVE (confirm)
    @app.callback(
        Output("tbl-emp-roster","data", allow_duplicate=True),
        Output("lbl-emp-total","children", allow_duplicate=True),
        Output("modal-remove","is_open", allow_duplicate=True),
        Output("emp-audit-log","data", allow_duplicate=True),
        Output("plan-msg","children", allow_duplicate=True),
        Output("plan-msg-timer","disabled", allow_duplicate=True),
        Input("btn-remove-ok","n_clicks"),
        State("tbl-emp-roster","data"),
        State("tbl-emp-roster","selected_rows"),
        State("emp-audit-log","data"),
        State("plan-detail-id","data"),
        prevent_initial_call=True
    )
    def _remove_selected(_n, data, sel, audit, pid):
        if not _n: raise dash.exceptions.PreventUpdate
        data = data or []
        sel = sel or []
        if not sel: raise dash.exceptions.PreventUpdate

        brids = _selected_brids(data, sel)
        before = [data[i] for i in sel]
        # filter out selected
        keep = [r for i, r in enumerate(data) if i not in sel]
        save_df(f"plan_{pid}_emp", pd.DataFrame(keep))
        audit = _audit_append(audit, "remove", before, brids)
        return (keep, f"Total: {len(keep):02d} Records", False, audit, "Removed ✓", False)

    # CHANGE CLASS
    @app.callback(
        Output("tbl-emp-roster","data", allow_duplicate=True),
        Output("modal-class","is_open", allow_duplicate=True),
        Output("emp-audit-log","data", allow_duplicate=True),
        Output("plan-msg","children", allow_duplicate=True),
        Output("plan-msg-timer","disabled", allow_duplicate=True),
        Input("btn-class-save","n_clicks"),
        State("inp-class-ref","value"),
        State("tbl-emp-roster","data"),
        State("tbl-emp-roster","selected_rows"),
        State("emp-audit-log","data"),
        State("plan-detail-id","data"),
        prevent_initial_call=True
    )
    def _change_class(_n, class_ref, data, sel, audit, pid):
        if not _n or not class_ref: raise dash.exceptions.PreventUpdate
        data = data or []; sel = sel or []
        if not sel: raise dash.exceptions.PreventUpdate
        brids = _selected_brids(data, sel)
        before = [data[i].copy() for i in sel]
        for i in sel:
            data[i]["class_ref"] = class_ref
        save_df(f"plan_{pid}_emp", pd.DataFrame(data))
        audit = _audit_append(audit, "change_class", before, brids)
        return (data, False, audit, "Class changed ✓", False)

    # FT/PT conversion
    @app.callback(
        Output("tbl-emp-roster","data", allow_duplicate=True),
        Output("modal-ftp","is_open", allow_duplicate=True),
        Output("emp-audit-log","data", allow_duplicate=True),
        Output("plan-msg","children", allow_duplicate=True),
        Output("plan-msg-timer","disabled", allow_duplicate=True),
        Input("btn-ftp-save","n_clicks"),
        State("inp-ftp-date","date"),
        State("inp-ftp-hours","value"),
        State("tbl-emp-roster","data"),
        State("tbl-emp-roster","selected_rows"),
        State("emp-audit-log","data"),
        State("plan-detail-id","data"),
        prevent_initial_call=True
    )
    def _ftpt_save(_n, eff_date, hours, data, sel, audit, pid):
        if not _n: raise dash.exceptions.PreventUpdate
        data = data or []; sel = sel or []
        if not sel: raise dash.exceptions.PreventUpdate
        brids = _selected_brids(data, sel)
        before = [data[i].copy() for i in sel]
        for i in sel:
            cur = (data[i].get("ftpt_status") or "").strip() or "Full-time"
            if cur.lower().startswith("full"):
                data[i]["ftpt_status"] = "Part-time"
                if hours: data[i]["ftpt_hours"] = hours
            else:
                data[i]["ftpt_status"] = "Full-time"
                if hours: data[i]["ftpt_hours"] = hours
            # no status change; this is an employment type change
        save_df(f"plan_{pid}_emp", pd.DataFrame(data))
        audit = _audit_append(audit, "ftpt", before, brids)
        return (data, False, audit, "FT/PT conversion saved ✓", False)

    # LOA
    @app.callback(
        Output("tbl-emp-roster","data", allow_duplicate=True),
        Output("modal-loa","is_open", allow_duplicate=True),
        Output("emp-audit-log","data", allow_duplicate=True),
        Output("plan-msg","children", allow_duplicate=True),
        Output("plan-msg-timer","disabled", allow_duplicate=True),
        Input("btn-loa-save","n_clicks"),
        State("inp-loa-date","date"),
        State("tbl-emp-roster","data"),
        State("tbl-emp-roster","selected_rows"),
        State("emp-audit-log","data"),
        State("plan-detail-id","data"),
        prevent_initial_call=True
    )
    def _move_to_loa(_n, d, data, sel, audit, pid):
        if not _n or not d: raise dash.exceptions.PreventUpdate
        data = data or []; sel = sel or []
        brids = _selected_brids(data, sel)
        before = [data[i].copy() for i in sel]
        snap = _monday_iso(d)
        for i in sel:
            data[i]["loa_date"] = snap
            data[i]["current_status"] = "Moved to LOA"
            data[i]["work_status"] = "LOA"
        save_df(f"plan_{pid}_emp", pd.DataFrame(data))
        audit = _audit_append(audit, "move_loa", before, brids)
        return (data, False, audit, "Moved to LOA ✓", False)

    # BACK from LOA
    @app.callback(
        Output("tbl-emp-roster","data", allow_duplicate=True),
        Output("modal-back","is_open", allow_duplicate=True),
        Output("emp-audit-log","data", allow_duplicate=True),
        Output("plan-msg","children", allow_duplicate=True),
        Output("plan-msg-timer","disabled", allow_duplicate=True),
        Input("btn-back-save","n_clicks"),
        State("inp-back-date","date"),
        State("tbl-emp-roster","data"),
        State("tbl-emp-roster","selected_rows"),
        State("emp-audit-log","data"),
        State("plan-detail-id","data"),
        prevent_initial_call=True
    )
    def _back_from_loa(_n, d, data, sel, audit, pid):
        if not _n or not d: raise dash.exceptions.PreventUpdate
        data = data or []; sel = sel or []
        brids = _selected_brids(data, sel)
        before = [data[i].copy() for i in sel]
        snap = _monday_iso(d)
        for i in sel:
            data[i]["back_from_loa_date"] = snap
            data[i]["current_status"] = "Production"
            data[i]["work_status"] = "Production"
        save_df(f"plan_{pid}_emp", pd.DataFrame(data))
        audit = _audit_append(audit, "back_loa", before, brids)
        return (data, False, audit, "Back from LOA ✓", False)

    # TERMINATE
    @app.callback(
        Output("tbl-emp-roster","data", allow_duplicate=True),
        Output("modal-term","is_open", allow_duplicate=True),
        Output("emp-audit-log","data", allow_duplicate=True),
        Output("plan-msg","children", allow_duplicate=True),
        Output("plan-msg-timer","disabled", allow_duplicate=True),
        Input("btn-term-save","n_clicks"),
        State("inp-term-date","date"),
        State("tbl-emp-roster","data"),
        State("tbl-emp-roster","selected_rows"),
        State("emp-audit-log","data"),
        State("plan-detail-id","data"),
        prevent_initial_call=True
    )
    def _terminate(_n, d, data, sel, audit, pid):
        if not _n or not d: raise dash.exceptions.PreventUpdate
        data = data or []; sel = sel or []
        brids = _selected_brids(data, sel)
        before = [data[i].copy() for i in sel]
        iso = pd.to_datetime(d).date().isoformat()
        for i in sel:
            data[i]["terminate_date"] = iso
            data[i]["work_status"] = "Terminated"
            data[i]["current_status"] = "Terminated"
        save_df(f"plan_{pid}_emp", pd.DataFrame(data))
        audit = _audit_append(audit, "terminate", before, brids)
        return (data, False, audit, "Terminated ✓", False)

    # TRANSFER / PROMOTION / BOTH
    @app.callback(
        Output("tbl-emp-roster","data", allow_duplicate=True),
        Output("modal-tp","is_open", allow_duplicate=True),
        Output("emp-audit-log","data", allow_duplicate=True),
        Output("plan-msg","children", allow_duplicate=True),
        Output("plan-msg-timer","disabled", allow_duplicate=True),
        Input("btn-tp-save","n_clicks"),
        State("tp-active-tab","value"),
        # Transfer
        State("tp-ba","value"), State("tp-subba","value"), State("tp-lob","value"), State("tp-site","value"),
        State("tp-transfer-type","value"),
        State("tp-new-class","value"), State("tp-class-ref","value"),
        State("tp-date-from","date"), State("tp-date-to","date"),
        # Promotion
        State("promo-type","value"), State("promo-role","value"),
        State("promo-date-from","date"), State("promo-date-to","date"),
        # Both
        State("twp-ba","value"), State("twp-subba","value"), State("twp-lob","value"), State("twp-site","value"),
        State("twp-type","value"), State("twp-new-class","value"),
        State("twp-class-ref","value"), State("twp-role","value"),
        State("twp-date-from","date"), State("twp-date-to","date"),
        State("tbl-emp-roster","data"),
        State("tbl-emp-roster","selected_rows"),
        State("emp-audit-log","data"),
        State("plan-detail-id","data"),
        prevent_initial_call=True
    )
    def _tp_save(_n, tab,
                t_ba, t_sba, t_lob, t_site, t_type, t_newcls, t_clsref, t_from, t_to,
                p_type, p_role, p_from, p_to,
                b_ba, b_sba, b_lob, b_site, b_type, b_newcls, b_clsref, b_role, b_from, b_to,
                data, sel, audit, pid):
        if not _n: raise dash.exceptions.PreventUpdate
        data = data or []; sel = sel or []
        if not sel: raise dash.exceptions.PreventUpdate
        brids = _selected_brids(data, sel)
        before = [data[i].copy() for i in sel]

        if tab == "tp-transfer":
            for i in sel:
                if t_ba:   data[i]["biz_area"] = t_ba
                if t_sba:  data[i]["sub_biz_area"] = t_sba
                if t_lob:  data[i]["lob"] = t_lob
                if t_site: data[i]["site"] = t_site
                if "yes" in (t_newcls or []) and t_clsref:
                    data[i]["class_ref"] = t_clsref
                data[i]["current_status"] = "Interim Transfer" if t_type == "interim" else "Transferred"
                data[i]["work_status"] = "Production"
                # You can store the date range in notes columns later if needed

        elif tab == "tp-promo":
            for i in sel:
                if p_role: data[i]["role"] = p_role
                data[i]["current_status"] = "Temporary Promotion" if p_type == "interim" else "Promoted"
                data[i]["work_status"] = "Production"

        else:  # tp-both
            for i in sel:
                if b_ba:   data[i]["biz_area"] = b_ba
                if b_sba:  data[i]["sub_biz_area"] = b_sba
                if b_lob:  data[i]["lob"] = b_lob
                if b_site: data[i]["site"] = b_site
                if "yes" in (b_newcls or []) and b_clsref:
                    data[i]["class_ref"] = b_clsref
                if b_role: data[i]["role"] = b_role
                data[i]["current_status"] = "Transfer with Promotion (Temp)" if b_type == "interim" else "Transfer with Promotion"
                data[i]["work_status"] = "Production"

        save_df(f"plan_{pid}_emp", pd.DataFrame(data))
        audit = _audit_append(audit, tab, before, brids)
        return (data, False, audit, "Saved ✓", False)

    # UNDO: revert the last change for the selected BRIDs (simple & safe)
    @app.callback(
        Output("tbl-emp-roster","data", allow_duplicate=True),
        Output("emp-audit-log","data", allow_duplicate=True),
        Output("plan-msg","children", allow_duplicate=True),
        Output("plan-msg-timer","disabled", allow_duplicate=True),
        Input("btn-emp-undo","n_clicks"),
        State("tbl-emp-roster","data"),
        State("tbl-emp-roster","selected_rows"),
        State("emp-audit-log","data"),
        State("plan-detail-id","data"),
        prevent_initial_call=True
    )
    def _undo_last(n, data, sel, audit, pid):
        if not n: raise dash.exceptions.PreventUpdate
        data = data or []; sel = sel or []
        if not sel: raise dash.exceptions.PreventUpdate
        df = pd.DataFrame(data)
        brids = _selected_brids(data, sel)

        undone = 0
        audit = list(audit or [])
        for b in brids:
            # find last snapshot for this BRID
            idxs = [i for i in range(len(audit)-1, -1, -1) if audit[i].get("brid")==b]
            if not idxs: continue
            snap = audit[idxs[0]].get("snapshot") or {}
            # replace the row in data
            where = df.index[df["brid"].astype(str)==str(b)].tolist()
            if not where: continue
            i = where[0]
            data[i] = {**data[i], **snap}
            # remove that audit entry
            audit.pop(idxs[0])
            undone += 1

        if undone:
            save_df(f"plan_{pid}_emp", pd.DataFrame(data))
        return (data, audit, (f"Undid {undone} change(s) ✓" if undone else "Nothing to undo"), False)
