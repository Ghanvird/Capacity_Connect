# file: plan_detail.py
from __future__ import annotations
import datetime as dt
from typing import List, Tuple, Dict
import base64, io, re

import pandas as pd
import numpy as np

import dash
from dash import html, dcc, dash_table, Input, Output, State, no_update
import dash_bootstrap_components as dbc

from plan_store import get_plan
from cap_db import save_df, load_df
from cap_store import load_headcount, get_clients_hierarchy  # <-- single source for BA/Level 3/Site
from cap_store import load_timeseries, resolve_settings, load_roster, load_hiring, load_roster_long, load_defaults, load_roster_wide
from capacity_core import voice_requirements_interval, voice_rollups, bo_rollups, required_fte_daily, supply_fte_daily

# Optional capacity_core
try:
    from capacity_core import min_agents, offered_load_erlangs  # (placeholder for future calcs)
except Exception:
    def min_agents(*args, **kwargs): return None
    def offered_load_erlangs(*args, **kwargs): return None

# Dash ctx fallback
try:
    from dash import ctx
except Exception:
    from dash import callback_context as ctx  # type: ignore

pd.set_option('future.no_silent_downcasting', True)

CHANNEL_DEFAULTS = ["Voice", "Back Office", "Outbound", "Blended", "Chat", "MessageUs"]
# ──────────────────────────────────────────────────────────────────────────────
def _settings_volume_aht_overrides(sk, which: str):
    """
    Read 'Settings' uploads where AHT/SUT is uploaded together with volume.
    Returns dicts:
      - vol_w: week -> total volume/items
      - aht_w (voice) or sut_w (bo): week -> weighted AHT/SUT
    `which` is 'voice' or 'bo'.
    """
    # Try a handful of tolerant keys; keep/add keys that match your storage
    keys_voice = [
        "settings_voice_volume_aht", "voice_settings_upload",
        "settings_volume_aht_voice", "voice_volume_aht", "voice_forecast_settings"
    ]
    keys_bo = [
        "settings_bo_volume_sut", "settings_backoffice_volume_sut",
        "bo_settings_upload", "backoffice_volume_sut", "bo_forecast_settings"
    ]
    df = _first_non_empty_ts(sk, keys_voice if which.lower()=="voice" else keys_bo)
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {"vol_w": {}, "aht_or_sut_w": {}}

    d = df.copy()
    # Column detection
    L = {str(c).strip().lower(): c for c in d.columns}
    c_date = L.get("date") or L.get("week") or L.get("start_date")
    c_vol  = L.get("vol") or L.get("volume") or L.get("calls") or L.get("items") or L.get("txns") or L.get("transactions")
    if which.lower()=="voice":
        c_time = L.get("aht_sec") or L.get("aht") or L.get("avg_aht")
    else:
        c_time = L.get("sut_sec") or L.get("sut") or L.get("avg_sut") or L.get("aht_sec")  # tolerate 'aht_sec' for BO

    if not c_date or not c_vol or not c_time:
        return {"vol_w": {}, "aht_or_sut_w": {}}

    # Normalize & weekly group
    d[c_date] = pd.to_datetime(d[c_date], errors="coerce")
    d = d.dropna(subset=[c_date])
    d["week"] = (d[c_date] - pd.to_timedelta(d[c_date].dt.weekday, unit="D")).dt.date.astype(str)

    d[c_vol]  = pd.to_numeric(d[c_vol],  errors="coerce").fillna(0.0)
    d[c_time] = pd.to_numeric(d[c_time], errors="coerce").fillna(0.0)

    g = d.groupby("week", as_index=False)[[c_vol]].sum().rename(columns={c_vol: "_vol_"})
    # weighted time per week
    d["_num_"] = d[c_time] * d[c_vol]
    w = d.groupby("week", as_index=False)[["_num_", c_vol]].sum()
    w["_wt_"] = np.where(w[c_vol] > 0, w["_num_"] / w[c_vol], np.nan)

    vol_w = dict(zip(g["week"], g["_vol_"]))
    aht_or_sut_w = dict(zip(w["week"], w["_wt_"]))  # may be NaN if no volume

    # Clean NaNs
    aht_or_sut_w = {k: float(v) for k, v in aht_or_sut_w.items() if pd.notna(v) and v > 0}
    vol_w        = {k: float(v) for k, v in vol_w.items()        if pd.notna(v) and v > 0}
    return {"vol_w": vol_w, "aht_or_sut_w": aht_or_sut_w}

# Build settings overrides (if any)
voice_ovr = _settings_volume_aht_overrides(sk, "voice")
bo_ovr    = _settings_volume_aht_overrides(sk, "bo")

# _____________________ erlangs _____________________

import math

def _metric_week_dict(df, row_name, week_ids):
    """Read a single 'metric' row into {week -> float}."""
    if not isinstance(df, pd.DataFrame) or df.empty or "metric" not in df.columns:
        return {}
    m = df["metric"].astype(str).str.strip().eq(row_name)
    if not m.any():
        return {}
    ser = pd.to_numeric(df.loc[m, week_ids].iloc[0], errors="coerce").fillna(0.0)
    return {str(k): float(v) for k, v in ser.to_dict().items()}

def _prev_week_id(w, week_ids):
    try:
        i = week_ids.index(w)
        return week_ids[i-1] if i > 0 else None
    except Exception:
        return None

def _erlang_c(traffic_erlangs: float, agents: float) -> float:
    """Classic Erlang-C waiting probability; agents may be float, we floor to int."""
    a = max(0, int(math.floor(agents)))
    A = max(0.0, float(traffic_erlangs))
    if a <= 0:
        return 1.0
    if A >= a:
        return 1.0
    # p0
    s = sum((A**k)/math.factorial(k) for k in range(a))
    last = (A**a)/math.factorial(a) * (a/(a - A))
    p0 = 1.0 / (s + last)
    pw = last * p0
    return min(1.0, max(0.0, pw))

def _erlang_service_level(offered_calls: float, aht_sec: float, agents: float,
                          interval_sec: int, asa_sec: int) -> float:
    """Return service level in [0..1]."""
    if aht_sec <= 0 or interval_sec <= 0 or agents <= 0 or offered_calls <= 0:
        return 0.0
    # Traffic in Erlangs in this interval
    A = (offered_calls * aht_sec) / float(interval_sec)
    pw = _erlang_c(A, agents)
    # P(wait <= T) = 1 - Pw * exp(-(a - A) * T / AHT)
    tail = math.exp(-max(0.0, (agents - A)) * (asa_sec / max(1.0, aht_sec)))
    sl = 1.0 - pw * tail
    return min(1.0, max(0.0, sl))
# ──────────────────────────────────────────────────────────────────────────────
def _build_global_hierarchy() -> dict:
    try:
        hmap, sites, _locs = get_clients_hierarchy()   # {BA:{SubBA:[LOBs...]}}
    except Exception:
        hmap, sites = {}, []

    ba_list = sorted(hmap.keys())
    sub_map = {ba: sorted((hmap.get(ba) or {}).keys()) for ba in ba_list}

    lob_map = {}
    for ba, subs in (hmap or {}).items():
        for sba, channels in (subs or {}).items():
            lob_map[f"{ba}|{sba}"] = list(channels or CHANNEL_DEFAULTS)

    return {
        "ba": ba_list,
        "subba": sub_map,
        "lob": lob_map,
        "site": sites or []
    }


def _lower_map(df: pd.DataFrame) -> dict[str, str]:
    return {str(c).strip().lower(): c for c in df.columns}

# ---------- Parsing helpers ----------
def pretty_columns(df_or_cols) -> list[dict]:
    cols = list(df_or_cols.columns) if hasattr(df_or_cols, "columns") else list(df_or_cols)
    return [{"name": c, "id": c} for c in cols]

def lock_variance_cols(cols):
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

def _scope_key(ba, subba, channel):
    return f"{(ba or '').strip()}|{(subba or '').strip()}|{(channel or '').strip()}"

# ──────────────────────────────────────────────────────────────────────────────
# helpers shared

def _week_monday(d):  # returns Monday date
    d = pd.to_datetime(d, errors="coerce").date()
    return d - pd.Timedelta(days=d.weekday())

# Fullscreen overlay style (centered)
_OVERLAY_STYLE = {
    "position": "fixed",
    "inset": "0",
    "background": "rgba(0,0,0,0.6)",
    "display": "none",                   # toggled by callback
    "alignItems": "center",
    "justifyContent": "center",
    "flexDirection": "column",
    "zIndex": 9999
}

def _loading_overlay() -> html.Div:
    return html.Div(
        id="plan-loading-overlay",
        children=[
            html.Img(src="/assets/Infinity.svg", style={"width": "96px", "height": "96px"}, className="avy"),
            html.Div("Preparing your plan…", style={"color": "white", "marginLeft": "10px"})
        ],
        style=_OVERLAY_STYLE
    )

def _settings_for_scope_key(sk: str) -> dict:
    try:
        ba, sba, ch = (sk.split("|", 2) + ["",""])[:3]
    except Exception:
        ba, sba, ch = "", "", ""
    return resolve_settings(ba=ba, subba=sba, lob=ch)

def _canon_scope(ba, sba, ch):
    canon = lambda x: (x or "").strip().lower()
    return f"{canon(ba)}|{canon(sba)}|{canon(ch)}"

def _monday(x):
    s = pd.to_datetime(x, errors="coerce")
    if isinstance(s, (pd.Series, pd.DatetimeIndex)):
        if isinstance(s, pd.DatetimeIndex):
            return (s - pd.to_timedelta(s.weekday, unit="D")).date
        else:
            return (s - pd.to_timedelta(s.dt.weekday, unit="D")).dt.date
    if pd.isna(s):
        s = pd.Timestamp(dt.date.today())
    return (s - pd.Timedelta(days=int(s.weekday()))).date()

def _weekly_voice(df):
    if df is None or df.empty:
        return pd.DataFrame(columns=["week", "vol", "aht"])

    x = df.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x = x.dropna(subset=["date"])

    # Week bucket (Monday)
    x["week"] = _monday(x["date"])

    # Safe numerics
    x["w"] = pd.to_numeric(x.get("volume"),  errors="coerce").fillna(0.0)
    x["a"] = pd.to_numeric(x.get("aht_sec"), errors="coerce").fillna(0.0)

    # Weighted-average pieces
    x["num"] = x["w"] * x["a"]

    g = (
        x.groupby("week", as_index=False)
         .agg(vol=("w", "sum"), num=("num", "sum"))
    )
    g["aht"] = np.where(g["vol"] > 0, g["num"] / g["vol"], np.nan)
    g = g.drop(columns=["num"])
    g["week"] = g["week"].astype(str)

    return g[["week", "vol", "aht"]]


def _weekly_bo(df):
    if df is None or df.empty:
        return pd.DataFrame(columns=["week", "items", "sut"])

    x = df.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x = x.dropna(subset=["date"])

    # Week bucket (Monday)
    x["week"] = _monday(x["date"])

    # Safe numerics (for BO, aht_sec column carries SUT)
    x["i"] = pd.to_numeric(x.get("items"),   errors="coerce").fillna(0.0)
    x["s"] = pd.to_numeric(x.get("aht_sec"), errors="coerce").fillna(0.0)

    # Weighted-average pieces
    x["num"] = x["i"] * x["s"]

    g = (
        x.groupby("week", as_index=False)
         .agg(items=("i", "sum"), num=("num", "sum"))
    )
    g["sut"] = np.where(g["items"] > 0, g["num"] / g["items"], np.nan)
    g = g.drop(columns=["num"])
    g["week"] = g["week"].astype(str)

    return g[["week", "items", "sut"]]


def _assemble_voice(scope_key, which):
    vol = load_timeseries(f"voice_{which}_volume", scope_key)
    aht = load_timeseries(f"voice_{which}_aht",    scope_key)
    if vol is None or vol.empty:
        return pd.DataFrame(columns=["date","interval","volume","aht_sec","program"])
    df = vol.copy()
    if isinstance(aht, pd.DataFrame) and not aht.empty:
        df = df.merge(aht, on=["date","interval"], how="left")
    if "aht_sec" not in df.columns or df["aht_sec"].isna().all():
        s = _settings_for_scope_key(scope_key)
        df["aht_sec"] = float(s.get("target_aht", s.get("budgeted_aht", 300)) or 300)
    df["program"] = "WFM"
    return df[["date","interval","volume","aht_sec","program"]]

def _assemble_bo(scope_key, which):
    vol = load_timeseries(f"bo_{which}_volume", scope_key)
    sut = load_timeseries(f"bo_{which}_sut",    scope_key)
    if vol is None or vol.empty:
        return pd.DataFrame(columns=["date","items","aht_sec","program"])
    df = vol.rename(columns={"volume":"items"}).copy()
    if isinstance(sut, pd.DataFrame) and not sut.empty:
        df = df.merge(sut, on=["date"], how="left")
        if "aht_sec" not in df.columns and "sut_sec" in df.columns:
            df = df.rename(columns={"sut_sec":"aht_sec"})
    if "aht_sec" not in df.columns or df["aht_sec"].isna().all():
        s = _settings_for_scope_key(scope_key)
        df["aht_sec"] = float(s.get("target_sut", s.get("budgeted_sut", 600)) or 600)
    df["program"] = "WFM"
    return df[["date","items","aht_sec","program"]]

def _snap_to_monday(value: str | dt.date | None) -> str:
    if not value:
        return ""
    return _monday(value).isoformat()

def _week_span(start_week: str | None, end_week: str | None, fallback_weeks: int = 12) -> list[str]:
    start = _monday(start_week) if start_week else _monday(dt.date.today())
    end   = _monday(end_week)   if end_week   else (pd.Timestamp(start) + pd.Timedelta(weeks=fallback_weeks-1)).date()
    if end < start:
        start, end = end, start
    weeks = []
    cur = pd.Timestamp(start)
    stop = pd.Timestamp(end)
    while cur <= stop:
        weeks.append(cur.date().isoformat())
        cur += pd.Timedelta(weeks=1)
    return weeks

def _week_cols(weeks: list[str]):
    today = dt.date.today()
    cols = [{"name": "Metric", "id": "metric", "editable": False}]
    week_ids: list[str] = []
    for w in weeks:
        wd = pd.to_datetime(w, errors="coerce")
        if pd.isna(wd):
            continue
        if not isinstance(wd, pd.Timestamp):
            wd = pd.Timestamp(wd)
        d = wd.date()
        tag = "Actual" if d <= today else "Plan"
        cols.append({"name": f"{tag}\n{d.strftime('%m/%d/%y')}", "id": d.isoformat()})
        week_ids.append(d.isoformat())
    return cols, week_ids

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
        for wid in week_ids:
            if wid not in df.columns:
                df[wid] = 0.0
        if "metric" not in df.columns:
            df.insert(0, "metric", metrics[: len(df)])
        return df[["metric"] + week_ids].copy()
    return _blank_grid(metrics, week_ids)

def _round_week_cols_int(df: pd.DataFrame, week_ids: list[str]) -> pd.DataFrame:
    """Round all weekly numeric columns to integers (no decimals) for display."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    out = df.copy()
    for wid in week_ids:
        if wid in out.columns:
            out[wid] = pd.to_numeric(out[wid], errors="coerce").fillna(0).round(0).astype(int)
    return out


def _save_table(pid: int, tab_key: str, df: pd.DataFrame):
    save_df(f"plan_{pid}_{tab_key}", df if isinstance(df, pd.DataFrame) else pd.DataFrame())

# ──────────────────────────────────────────────────────────────────────────────
# roster / bulk columns

def _roster_columns() -> List[dict]:
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
        ("LOB", "lob"), ("LOA Date", "loa_date"), ("Back from LOA Date", "back_from_loa_date"),
        ("Site", "site"),
    ]
    cols = []
    for n, cid in names:
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

_ROSTER_REQUIRED_IDS = [c["id"] for c in _roster_columns()]

def _load_or_empty_roster(pid: int) -> pd.DataFrame:
    cols = [c["id"] for c in _roster_columns()]
    df = load_df(f"plan_{pid}_emp")
    if isinstance(df, pd.DataFrame) and not df.empty:
        for col in cols:
            if col not in df.columns:
                df[col] = ""
        return df[cols].copy()
    return pd.DataFrame(columns=cols)

def _load_or_empty_bulk_files(pid: int) -> pd.DataFrame:
    cols = [c["id"] for c in _bulkfile_columns()]

    # sensible defaults for an empty grid
    empty = pd.DataFrame({c: pd.Series(dtype="object") for c in cols})
    if "size_kb" in empty.columns:
        empty["size_kb"] = empty["size_kb"].astype("float64")

    try:
        df = load_df(f"plan_{pid}_bulk_files")
    except Exception:
        # covers pandas.errors.EmptyDataError and any bad/corrupt payloads
        return empty

    if isinstance(df, pd.DataFrame) and not df.empty:
        # ensure required columns exist and types align
        for c in cols:
            if c not in df.columns:
                df[c] = "" if c != "size_kb" else 0.0
        # coerce size_kb numeric
        if "size_kb" in df.columns:
            df["size_kb"] = pd.to_numeric(df["size_kb"], errors="coerce").fillna(0.0)
        return df[cols].copy()

    return empty


def _load_or_empty_notes(pid: int) -> pd.DataFrame:
    """Notes table: always return a DF with ['when','user','note'] and handle empty payloads."""
    cols = ["when", "user", "note"]

    # canonical empty frame (preserve dtypes)
    empty = pd.DataFrame({c: pd.Series(dtype="object") for c in cols})

    try:
        df = load_df(f"plan_{pid}_notes")
    except Exception:
        # covers pandas.errors.EmptyDataError and other corrupt/empty payloads
        return empty

    if isinstance(df, pd.DataFrame) and not df.empty:
        # ensure required columns exist
        for c in cols:
            if c not in df.columns:
                df[c] = ""
        # keep only expected columns / order
        return df[cols].copy()

    return empty


def _parse_upload(contents: str, filename: str) -> Tuple[pd.DataFrame, dict]:
    if not contents or not filename:
        return pd.DataFrame(), {}
    try:
        header, b64 = contents.split(",", 1)
    except ValueError:
        return pd.DataFrame(), {"file_name": filename, "ext": "", "size_kb": 0, "is_valid": "No", "status": "Invalid format"}

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

    rename_map = {c["name"]: c["id"] for c in _roster_columns()}
    lower_map = {k.lower(): v for k,v in rename_map.items()}
    df = df.rename(columns={col: lower_map.get(str(col).lower(), col) for col in df.columns})

    missing = [cid for cid in _ROSTER_REQUIRED_IDS if cid not in df.columns]
    valid = len(missing) == 0
    ledger = {"file_name": filename, "ext": ext, "size_kb": round(len(raw)/1024,1),
              "is_valid": "Yes" if valid else "No", "status": "Loaded" if valid else f"Missing: {', '.join(missing[:3])}"}
    if not valid:
        return pd.DataFrame(), ledger

    df = df[_ROSTER_REQUIRED_IDS].copy()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
    return df, ledger

def _format_crumb(p: dict) -> str:
    ba  = (p.get("business_area") or p.get("plan_ba") or p.get("ba") or p.get("vertical") or "").strip()
    sba = (p.get("sub_business_area") or p.get("plan_sub_ba") or p.get("sub_ba") or p.get("subba") or "").strip()
    lob = (p.get("lob") or p.get("channel") or "").strip()
    if lob:
        lob = lob.title()
    site = (p.get("site") or "").strip()
    parts = [x for x in [ba, sba, lob, site] if x]
    return " > ".join(parts)

# ──────────────────────────────────────────────────────────────────────────────
# NEW: Build BA -> Sub-BA (Level 3) and Sites from Headcount Update only

def _load_hcu_df() -> pd.DataFrame:
    try:
        df = load_headcount()
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def _build_hierarchy_sites_from_headcount() -> tuple[dict[str, list[str]], list[str]]:
    df = _load_hcu_df()
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {}, []

    L = {str(c).strip().lower(): c for c in df.columns}

    ba_col   = L.get("journey") or L.get("business area") or L.get("vertical") \
               or L.get("current_org_unit_description") or L.get("current org unit description") \
               or L.get("current_org_unit") or L.get("current org unit") \
               or L.get("level_0") or L.get("level 0")
    sba_col  = L.get("level_3") or L.get("level 3") or L.get("sub business area") or L.get("sub_business_area")
    site_col = L.get("position_location_building_description") or L.get("building") \
               or L.get("building description") or L.get("site")

    hmap: dict[str, set[str]] = {}
    if ba_col:
        for _, r in df.iterrows():
            ba = str(r.get(ba_col, "")).strip()
            if not ba:
                continue
            sba_val = str(r.get(sba_col, "")).strip() if sba_col else ""
            hmap.setdefault(ba, set())
            if sba_val:
                hmap[ba].add(sba_val)

    out_hmap = {ba: sorted(list(subs)) for ba, subs in hmap.items()}
    sites: list[str] = []
    if site_col:
        s = (
            df[site_col].dropna().astype(str).str.strip().replace({"": np.nan}).dropna().unique().tolist()
        )
        sites = sorted(set(s))
    return out_hmap, sites

def _hier_from_hcu() -> dict:
    hmap, sites = _build_hierarchy_sites_from_headcount()  # {BA: [SubBAs]}, [sites]
    bas = sorted(hmap.keys())
    sub_map = {ba: sorted(list(subs or [])) for ba, subs in (hmap or {}).items()}

    lob_map = {}
    for ba, subs in sub_map.items():
        for sba in subs:
            lob_map[f"{ba}|{sba}"] = list(CHANNEL_DEFAULTS)

    return {"ba": bas, "subba": sub_map, "lob": lob_map, "site": sorted(sites or [])}

# ──────────────────────────────────────────────────────────────────────────────
# NEW: generic weekly loaders & small parsers

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

def _weekly_reduce(df: pd.DataFrame, value_candidates=("value","hc","headcount","hours","items","volume","count","amt","amount"),
                   how: str = "sum") -> dict:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {}
    date_col = None
    for c in ("date","dt","day","when"):
        if c in df.columns:
            date_col = c
            break
    if not date_col:
        # try to cast an index to dates
        if isinstance(df.index, (pd.DatetimeIndex, pd.Index)):
            df = df.reset_index().rename(columns={df.columns[0]:"date"})
            date_col = "date"
        else:
            return {}
    val_col = None
    for c in value_candidates:
        if c in df.columns:
            val_col = c
            break
    if not val_col:
        # single numeric column fallback
        nums = [c for c in df.columns if c != date_col and np.issubdtype(df[c].dtype, np.number)]
        if nums:
            val_col = nums[0]
        else:
            return {}

    d = df[[date_col, val_col]].copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col])
    d["week"] = (d[date_col] - pd.to_timedelta(d[date_col].dt.weekday, unit="D")).dt.date.astype(str)
    d[val_col] = pd.to_numeric(d[val_col], errors="coerce").fillna(0.0)

    if how == "mean":
        s = d.groupby("week", as_index=False)[val_col].mean().set_index("week")[val_col]
    else:
        s = d.groupby("week", as_index=False)[val_col].sum().set_index("week")[val_col]
    return s.to_dict()

def _parse_ratio_setting(v) -> float:
    try:
        if isinstance(v, str) and ":" in v:
            a, b = v.split(":", 1)
            a = float(str(a).strip()); b = float(str(b).strip())
            return (a / b) if b else 0.0
        return float(v)
    except Exception:
        return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# layout builders

def _upper_summary_header_card() -> html.Div:
    return dbc.Card(
        dbc.CardBody([
            html.Div([
                html.Div([
                    dcc.Link(dbc.Button("🢀", id="plan-hdr-back", color="light", title="Back"),
                             href="/planning", className="me-2"),
                    html.Span(id="plan-hdr-name", className="fw-bold")
                ], className="d-flex align-items-center"),
                html.Div([
                    dbc.Button("💾", id="btn-plan-save", color="light", title="Save", className="me-1"),
                    dbc.Button("⟳", id="btn-plan-refresh", color="light", title="Refresh", className="me-1"),
                    html.Div(id="plan-msg", className="text-success mt-2"),
                ], style={"display":"flex"}),
                html.Div([
                    dbc.Button("▼", id="plan-hdr-collapse", color="light", title="Collapse/Expand")
                ]),
            ], className="d-flex justify-content-between align-items-center mb-2 hhh"),
        ], style={"padding": "3px"}),
        className="mb-3"
    )

def _upper_summary_body_card() -> html.Div:
    return dbc.Card(
        dbc.CardBody([
            html.Div(id="plan-upper", className="cp-grid")
        ], class_name="gaurav"),
        className="mb-3"
    )

def _lower_tabs() -> dbc.Card:
    return dbc.Card(
        dbc.CardBody([
            dbc.Tabs(id="plan-tabs", active_tab="tab-fw", children=[
                # Auto-computed tabs → editable=False
                dbc.Tab(label="Forecast & Workload", tab_id="tab-fw",
                        children=[dash_table.DataTable(id="tbl-fw", editable=False,
                                                       style_as_list_view=True,
                                                       style_table={"overflowX":"auto"},
                                                       style_header={"whiteSpace":"pre"},
                                                       page_size=12)], class_name="ddd"),
                dbc.Tab(label="Headcount", tab_id="tab-hc",
                        children=[dash_table.DataTable(id="tbl-hc", editable=False,
                                                       style_as_list_view=True,
                                                       style_table={"overflowX":"auto"},
                                                       style_header={"whiteSpace":"pre"},
                                                       page_size=12)], class_name="ddd"),
                dbc.Tab(label="Attrition", tab_id="tab-attr",
                        children=[dash_table.DataTable(id="tbl-attr", editable=False,
                                                       style_as_list_view=True,
                                                       style_table={"overflowX":"auto"},
                                                       style_header={"whiteSpace":"pre"},
                                                       page_size=12)], class_name="ddd"),
                dbc.Tab(label="Shrinkage", tab_id="tab-shr",
                        children=[dash_table.DataTable(id="tbl-shr", editable=False,
                                                       style_as_list_view=True,
                                                       style_table={"overflowX":"auto"},
                                                       style_header={"WhiteSpace":"pre"},
                                                       page_size=12)], class_name="ddd"),

                # User-editable tabs → editable=True (keep as-is)
                dbc.Tab(label="Training Lifecycle", tab_id="tab-train",
                        children=[dash_table.DataTable(id="tbl-train", editable=True,
                                                       style_as_list_view=True,
                                                       style_table={"overflowX":"auto"},
                                                       style_header={"whiteSpace":"pre"},
                                                       page_size=12)], class_name="ddd"),
                dbc.Tab(label="Ratios", tab_id="tab-ratio",
                        children=[dash_table.DataTable(id="tbl-ratio", editable=True,
                                                       style_as_list_view=True,
                                                       style_table={"overflowX":"auto"},
                                                       style_header={"whiteSpace":"pre"},
                                                       page_size=12)], class_name="ddd"),
                dbc.Tab(label="Seat Utilization", tab_id="tab-seat",
                        children=[dash_table.DataTable(id="tbl-seat", editable=True,
                                                       style_as_list_view=True,
                                                       style_table={"overflowX":"auto"},
                                                       style_header={"whiteSpace":"pre"},
                                                       page_size=12)], class_name="ddd"),
                dbc.Tab(label="Budget vs Actual", tab_id="tab-bva",
                        children=[dash_table.DataTable(id="tbl-bva", editable=False,
                                                       style_as_list_view=True,
                                                       style_table={"overflowX":"auto"},
                                                       style_header={"whiteSpace":"pre"},
                                                       page_size=12)], class_name="ddd"),
                dbc.Tab(label="New Hire", tab_id="tab-nh",
                        children=[dash_table.DataTable(id="tbl-nh", editable=True,
                                                       style_as_list_view=True,
                                                       style_table={"overflowX":"auto"},
                                                       style_header={"whiteSpace":"pre"},
                                                       page_size=12)], class_name="ddd"),

                # Employee roster & Notes (unchanged)
                dbc.Tab(label="Employee Roster", tab_id="tab-roster", children=[
                    dbc.Tabs([
                        dbc.Tab(label="Roster", tab_id="tab-roster-main", children=[
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
                                row_selectable=False,
                                selected_rows=[],
                                style_as_list_view=True,
                                style_table={"overflowX": "auto"},
                                page_size=10,
                            ),
                        ]),
                        dbc.Tab(label="Bulk Upload", tab_id="tab-roster-bulk", children=[
                            html.Div([
                                dcc.Upload(id="up-roster-bulk", children=html.Div(["⬆️ Upload CSV/XLSX"]),
                                           multiple=False, className="upload-box"),
                                dbc.Button("Download Template", id="btn-template-dl", color="secondary"),
                                dcc.Download(id="dl-template")
                            ], className="mb-2", style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"}),
                            dash_table.DataTable(
                                id="tbl-bulk-files", editable=False, style_as_list_view=True,
                                filter_action="native", sort_action="native",
                                style_table={"overflowX":"auto"}, page_size=10
                            ),
                        ])
                    ], style={"marginBottom": "1rem", "marginTop": "1rem"})
                ]),

                # Notes
                dbc.Tab(label="Notes", tab_id="tab-notes", children=[
                    dbc.Row([
                        dbc.Col(dcc.Textarea(id="notes-input", style={"width":"100%","height":"120px"},
                                             placeholder="Write a note and click Save…"), md=9, class_name="panwar"),
                        dbc.Col(dbc.Button("Save Note", id="btn-note-save", color="primary", className="mt-2"), md=3, class_name="aggarwal"),
                    ], className="mb-2"),
                    dash_table.DataTable(
                        id="tbl-notes",
                        columns=[{"name":"Date","id":"when"},{"name":"User","id":"user"},{"name":"Note","id":"note"}],
                        data=[], editable=False, style_as_list_view=True, page_size=10,
                        style_table={"overflowX":"auto"}
                    )
                ], class_name="akl"),
            ]),
        ], class_name="ankit"),
        className="mb-3"
    )

def _add_employee_modal() -> dbc.Modal:
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Add New Employee"), style={"background": "#2f3747", "color": "white"}),
            dbc.ModalBody([
                html.Div(id="modal-roster-crumb", className="text-muted small mb-2"),
                dbc.Row([
                    dbc.Col(dbc.Input(id="inp-brid", placeholder="BRID"), md=6),
                    dbc.Col(dbc.Input(id="inp-name", placeholder="Employee Name"), md=6),
                ], className="mb-2"),
                dbc.Row([
                    dbc.Col(dbc.RadioItems(
                        id="inp-ftpt", value="Full-time",
                        options=[{"label":" Full-time","value":"Full-time"},
                                 {"label":" Part-time","value":"Part-time"}],
                        inline=True
                    ), md=6),
                    dbc.Col(dcc.Dropdown(
                        id="inp-role", placeholder="Role",
                        options=[{"label":r,"value":r} for r in ["Agent","SME","Trainer","Team Leader","QA","HR","WFM","AVP","VP"]],
                        value="Agent", clearable=False
                    ), md=6),
                ], className="mb-2"),
                dbc.Row([
                    dbc.Col(dcc.DatePickerSingle(id="inp-prod-date", placeholder="Production Date"), md=6),
                    dbc.Col(dbc.Input(id="inp-tl", placeholder="Team Leader"), md=6),
                ], className="mb-2"),
                dbc.Row([
                    dbc.Col(dbc.Input(id="inp-avp", placeholder="AVP"), md=6),
                ]),
            ]),
            dbc.ModalFooter([
                dbc.Button("Save", id="btn-emp-modal-save", color="primary", className="me-2"),
                dbc.Button("Cancel", id="btn-emp-modal-cancel", color="secondary"),
            ]),
        ],
        id="modal-emp-add", is_open=False, size="lg", backdrop="static"
    )

def _actions_modals() -> list:
    return [
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Please confirm"), style={"background": "#2f3747", "color": "white"}),
                dbc.ModalBody([
                    html.Div("Are you sure?"),
                    html.Div(
                        "Deleting this record will remove employee from database and will impact headcount projections.",
                        className="text-muted small mt-1"
                    ),
                ]),
                dbc.ModalFooter([
                    dbc.Button("Yes", id="btn-remove-ok", color="danger", className="me-2"),
                    dbc.Button("No", id="btn-remove-cancel", color="secondary"),
                ])
            ],
            id="modal-remove", is_open=False, backdrop="static"
        ),
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Change Class Reference"), style={"background": "#2f3747", "color": "white"}),
                dbc.ModalBody([
                    dbc.Row([
                        dbc.Col(dbc.Label("Class Reference"), md=12),
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
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("FT/PT Conversion"), style={"background": "#2f3747", "color": "white"}),
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
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Move to LOA"), style={"background": "#2f3747", "color": "white"}),
                dbc.ModalBody([
                    dbc.Label("Effective Date"),
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
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Back from LOA"), style={"background": "#2f3747", "color": "white"}),
                dbc.ModalBody([
                    dbc.Label("Effective Date"),
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
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Terminate"), style={"background": "#2f3747", "color": "white"}),
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
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Transfer & Promotion"), style={"background": "#2f3747", "color": "white"}),
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
                                    options=[{"label":" Permanent","value":"perm"},
                                             {"label":" Interim","value":"interim"}],
                                    value="perm", inline=True
                                ), md=6),
                                dbc.Col(dbc.Checklist(
                                    id="tp-new-class",
                                    options=[{"label":" Transfer with new class","value":"yes"}],
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
                                    options=[{"label":" Permanent","value":"perm"},
                                             {"label":" Temporary","value":"interim"}],
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
                                    placeholder="Role (e.g., Team Leader, Trainer, SME, QA …)",
                                    options=[{"label":r,"value":r} for r in ["Agent","SME","Trainer","Team Leader","QA","HR","WFM","AVP","VP"]],
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
                                    options=[{"label":" Permanent","value":"perm"},
                                             {"label":" Temporary","value":"interim"}],
                                    value="perm", inline=True
                                ), md=6),
                                dbc.Col(dbc.Checklist(
                                    id="twp-new-class",
                                    options=[{"label":" Transfer with new class","value":"yes"}],
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
                                dbc.Col(dcc.Dropdown(
                                    id="twp-role", placeholder="Role",
                                    options=[{"label":r,"value":r} for r in ["Agent","SME","Trainer","Team Leader","QA","HR","WFM","AVP","VP"]]
                                ), md=6),
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
    ]

# ──────────────────────────────────────────────────────────────────────────────

def layout_for_plan(pid: int) -> html.Div:
    """Main page UI; data comes from callbacks."""
    return dbc.Container([
        dcc.Store(id="plan-detail-id", data=pid),
        dcc.Store(id="plan-upper-collapsed", data=False),
        dcc.Store(id="plan-type"),
        dcc.Store(id="plan-weeks"),
        dcc.Store(id="tp-hier-map"),
        dcc.Store(id="tp-sites-map"),
        dcc.Store(id="plan-loading", data=True), 
        dcc.Store(id="plan-refresh-tick", data=0),  # NEW: refresh trigger store
        dcc.Store(id="tp-current", data={}),
        dcc.Interval(id="plan-msg-timer", interval=5000, n_intervals=0, disabled=True),

        _upper_summary_header_card(),
        _upper_summary_body_card(),
        _lower_tabs(),
        _loading_overlay(),
        _add_employee_modal(),
        *_actions_modals(),
    ], fluid=True)

def plan_detail_validation_layout() -> html.Div:
    dummy_cols = [{"name": "Metric", "id": "metric"}] + [{"name": "Plan\\n01/01/70", "id": "1970-01-01"}]
    return html.Div(
        [
            dcc.Store(id="plan-detail-id"),
            dcc.Store(id="plan-upper-collapsed"),
            dcc.Store(id="plan-type"),
            dcc.Store(id="plan-loading"),
            dcc.Store(id="plan-weeks"),
            dcc.Store(id="plan-refresh-tick"),  # ensure present in validation layout too
            dcc.Interval(id="plan-msg-timer"),

            html.Div(id="plan-loading-overlay"),
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
            dcc.Store(id="tp-hier-map"), dcc.Store(id="tp-current"), dcc.Tabs(id="tp-active-tab"),
            dcc.Dropdown(id="tp-ba"), dcc.Dropdown(id="tp-subba"), dcc.Dropdown(id="tp-lob"), dcc.Dropdown(id="tp-site"),
            dcc.Dropdown(id="tp-class-ref"), dcc.RadioItems(id="tp-transfer-type"),
            dcc.Checklist(id="tp-new-class"), dcc.DatePickerSingle(id="tp-date-from"), dcc.DatePickerSingle(id="tp-date-to"),
            dcc.Dropdown(id="promo-role"), dcc.RadioItems(id="promo-type"), dcc.DatePickerSingle(id="promo-date-from"), dcc.DatePickerSingle(id="promo-date-to"),
            dcc.Dropdown(id="twp-ba"), dcc.Dropdown(id="twp-subba"), dcc.Dropdown(id="twp-lob"), dcc.Dropdown(id="twp-site"),
            dcc.Dropdown(id="twp-class-ref"), dcc.Dropdown(id="twp-role"), dcc.RadioItems(id="twp-type"),
            dcc.DatePickerSingle(id="twp-date-from"), dcc.DatePickerSingle(id="twp-date-to"),
            html.Div(id="tp-who"),
            html.Div(id="lbl-emp-total"),
            dbc.Tabs(id="plan-tabs"),
            # 👇 Add ALL the action buttons used as Inputs in callbacks
            dbc.Button(id="btn-emp-tp"),       dbc.Button(id="btn-tp-save"),       dbc.Button(id="btn-tp-cancel"),
            dbc.Button(id="btn-emp-loa"),      dbc.Button(id="btn-loa-save"),      dbc.Button(id="btn-loa-cancel"),
            dbc.Button(id="btn-emp-back"),     dbc.Button(id="btn-back-save"),     dbc.Button(id="btn-back-cancel"),
            dbc.Button(id="btn-emp-term"),     dbc.Button(id="btn-term-save"),     dbc.Button(id="btn-term-cancel"),
            dbc.Button(id="btn-emp-ftp"),      dbc.Button(id="btn-ftp-save"),      dbc.Button(id="btn-ftp-cancel"),
            dbc.Button(id="btn-emp-class"),    dbc.Button(id="btn-class-save"),    dbc.Button(id="btn-class-cancel"),
            dbc.Button(id="btn-emp-remove"),   dbc.Button(id="btn-remove-ok"),     dbc.Button(id="btn-remove-cancel"),
            # (optional) placeholder for the “Undo” button even though it has no callback yet
            dbc.Button(id="btn-emp-undo"),

            # 👇 Add Inputs/State targets referenced by callbacks (if any were missing)
            dcc.DatePickerSingle(id="inp-loa-date"),
            dcc.DatePickerSingle(id="inp-back-date"),   # you already had this — keep it
            dcc.DatePickerSingle(id="inp-term-date"),
            dcc.DatePickerSingle(id="inp-ftp-date"),
            dbc.Input(id="inp-ftp-hours", type="number"),
            dcc.Dropdown(id="inp-class-ref"),           # you already had this — keep it

            # 👇 Add **all** the modal shells used as Outputs
            dbc.Modal(id="modal-emp-add"),  # you already had this — keep it
            dbc.Modal(id="modal-remove"),
            dbc.Modal(id="modal-class"),
            dbc.Modal(id="modal-ftp"),
            dbc.Modal(id="modal-loa"),
            dbc.Modal(id="modal-back"),
            dbc.Modal(id="modal-term"),
            dbc.Modal(id="modal-tp"),
        ],
        style={"display": "none"}
    )

# ──────────────────────────────────────────────────────────────────────────────
# callbacks

def register_plan_detail(app: dash.Dash):
    #________________________________ Employee Roster Modals & Actions _________________________________
    def _selected_rows(data, selected_rows):
        df = pd.DataFrame(data or [])
        if df.empty or not selected_rows:
            return df, []
        idx = [i for i in selected_rows if 0 <= i < len(df)]
        return df, idx
    # Remove

    @app.callback(
        Output("modal-remove", "is_open"),
        Input("btn-emp-remove", "n_clicks"),
        Input("btn-remove-cancel", "n_clicks"),
        prevent_initial_call=True
    )
    def _open_remove(n_open, n_cancel):
        t = ctx.triggered_id
        return True if t == "btn-emp-remove" else False

    @app.callback(
        Output("tbl-emp-roster", "data", allow_duplicate=True),
        Output("lbl-emp-total", "children", allow_duplicate=True),
        Output("modal-remove", "is_open", allow_duplicate=True),
        Output("plan-msg", "children", allow_duplicate=True),
        Output("plan-msg-timer", "disabled", allow_duplicate=True),
        Input("btn-remove-ok", "n_clicks"),
        State("tbl-emp-roster", "data"),
        State("tbl-emp-roster", "selected_rows"),
        State("plan-detail-id", "data"),
        prevent_initial_call=True
    )
    def _do_remove(n, data, selected_rows, pid):
        df, idx = _selected_rows(data, selected_rows)
        if df.empty or not idx:
            raise dash.exceptions.PreventUpdate
        keep = df.drop(index=idx).reset_index(drop=True)
        save_df(f"plan_{pid}_emp", keep)
        return keep.to_dict("records"), f"Total: {len(keep):02d} Records", False, "Removed ✓", False

    # Change Class
    @app.callback(
        Output("modal-class", "is_open"),
        Output("class-change-hint", "children"),
        Input("btn-emp-class", "n_clicks"),
        Input("btn-class-cancel", "n_clicks"),
        State("tbl-emp-roster", "data"),
        State("tbl-emp-roster", "selected_rows"),
        prevent_initial_call=True
    )
    def _open_class(n_open, n_cancel, data, sel):
        t = ctx.triggered_id
        if t == "btn-emp-class":
            df, idx = _selected_rows(data, sel)
            who = ", ".join(df.loc[idx, "name"].astype(str).tolist()) if not df.empty and idx else ""
            return True, f"Selected: {who}"
        return False, ""

    @app.callback(
        Output("tbl-emp-roster", "data", allow_duplicate=True),
        Output("modal-class", "is_open", allow_duplicate=True),
        Output("plan-msg", "children", allow_duplicate=True),
        Output("plan-msg-timer", "disabled", allow_duplicate=True),
        Input("btn-class-save", "n_clicks"),
        State("inp-class-ref", "value"),
        State("tbl-emp-roster", "data"),
        State("tbl-emp-roster", "selected_rows"),
        State("plan-detail-id", "data"),
        prevent_initial_call=True
    )
    def _do_class(n, cref, data, sel, pid):
        if not n or not cref:
            raise dash.exceptions.PreventUpdate
        df, idx = _selected_rows(data, sel)
        if df.empty or not idx:
            raise dash.exceptions.PreventUpdate
        df.loc[idx, "class_ref"] = cref
        save_df(f"plan_{pid}_emp", df)
        return df.to_dict("records"), False, "Class updated ✓", False

    # FT/PT
    @app.callback(
        Output("modal-ftp", "is_open"),
        Output("ftp-who", "children"),
        Input("btn-emp-ftp", "n_clicks"),
        Input("btn-ftp-cancel", "n_clicks"),
        State("tbl-emp-roster", "data"),
        State("tbl-emp-roster", "selected_rows"),
        prevent_initial_call=True
    )
    def _open_ftp(n_open, n_cancel, data, sel):
        t = ctx.triggered_id
        if t == "btn-emp-ftp":
            df, idx = _selected_rows(data, sel)
            who = ", ".join(df.loc[idx, "name"].astype(str).tolist()) if not df.empty and idx else ""
            return True, f"Selected: {who}"
        return False, ""

    @app.callback(
        Output("tbl-emp-roster", "data", allow_duplicate=True),
        Output("modal-ftp", "is_open", allow_duplicate=True),
        Output("plan-msg", "children", allow_duplicate=True),
        Output("plan-msg-timer", "disabled", allow_duplicate=True),
        Input("btn-ftp-save", "n_clicks"),
        State("inp-ftp-date", "date"),
        State("inp-ftp-hours", "value"),
        State("tbl-emp-roster", "data"),
        State("tbl-emp-roster", "selected_rows"),
        State("plan-detail-id", "data"),
        prevent_initial_call=True
    )
    def _do_ftp(n, date, hours, data, sel, pid):
        if not n:
            raise dash.exceptions.PreventUpdate
        df, idx = _selected_rows(data, sel)
        if df.empty or not idx:
            raise dash.exceptions.PreventUpdate
        for i in idx:
            cur = str(df.at[i, "ftpt_status"] or "")
            if cur.lower().startswith("full"):
                df.at[i, "ftpt_status"] = "Part-time"
                if hours: df.at[i, "ftpt_hours"] = hours
            else:
                df.at[i, "ftpt_status"] = "Full-time"
                df.at[i, "ftpt_hours"] = hours or ""
        save_df(f"plan_{pid}_emp", df)
        return df.to_dict("records"), False, "FT/PT updated ✓", False

    # Move to LOA
    @app.callback(
        Output("modal-loa", "is_open"),
        Output("loa-who", "children"),
        Input("btn-emp-loa", "n_clicks"),
        Input("btn-loa-cancel", "n_clicks"),
        State("tbl-emp-roster", "data"),
        State("tbl-emp-roster", "selected_rows"),
        prevent_initial_call=True
    )
    def _open_loa(n_open, n_cancel, data, sel):
        t = ctx.triggered_id
        if t == "btn-emp-loa":
            df, idx = _selected_rows(data, sel)
            who = ", ".join(df.loc[idx, "name"].astype(str).tolist()) if not df.empty and idx else ""
            return True, f"Selected: {who}"
        return False, ""

    @app.callback(
        Output("tbl-emp-roster", "data", allow_duplicate=True),
        Output("modal-loa", "is_open", allow_duplicate=True),
        Output("plan-msg", "children", allow_duplicate=True),
        Output("plan-msg-timer", "disabled", allow_duplicate=True),
        Input("btn-loa-save", "n_clicks"),
        State("inp-loa-date", "date"),
        State("tbl-emp-roster", "data"),
        State("tbl-emp-roster", "selected_rows"),
        State("plan-detail-id", "data"),
        prevent_initial_call=True
    )
    def _do_loa(n, date, data, sel, pid):
        if not n or not date:
            raise dash.exceptions.PreventUpdate
        df, idx = _selected_rows(data, sel)
        if df.empty or not idx:
            raise dash.exceptions.PreventUpdate
        monday = _monday(date).isoformat()
        for i in idx:
            df.at[i, "loa_date"] = monday
            df.at[i, "current_status"] = "Moved to LOA"
            df.at[i, "work_status"] = "Moved to LOA"
        save_df(f"plan_{pid}_emp", df)
        return df.to_dict("records"), False, "Moved to LOA ✓", False

    # Back from LOA
    @app.callback(
        Output("modal-back", "is_open"),
        Output("back-who", "children"),
        Input("btn-emp-back", "n_clicks"),
        Input("btn-back-cancel", "n_clicks"),
        State("tbl-emp-roster", "data"),
        State("tbl-emp-roster", "selected_rows"),
        prevent_initial_call=True
    )
    def _open_back(n_open, n_cancel, data, sel):
        t = ctx.triggered_id
        if t == "btn-emp-back":
            df, idx = _selected_rows(data, sel)
            who = ", ".join(df.loc[idx, "name"].astype(str).tolist()) if not df.empty and idx else ""
            return True, f"Selected: {who}"
        return False, ""

    @app.callback(
        Output("tbl-emp-roster", "data", allow_duplicate=True),
        Output("modal-back", "is_open", allow_duplicate=True),
        Output("plan-msg", "children", allow_duplicate=True),
        Output("plan-msg-timer", "disabled", allow_duplicate=True),
        Input("btn-back-save", "n_clicks"),
        State("inp-back-date", "date"),
        State("tbl-emp-roster", "data"),
        State("tbl-emp-roster", "selected_rows"),
        State("plan-detail-id", "data"),
        prevent_initial_call=True
    )
    def _do_back(n, date, data, sel, pid):
        if not n or not date:
            raise dash.exceptions.PreventUpdate
        df, idx = _selected_rows(data, sel)
        if df.empty or not idx:
            raise dash.exceptions.PreventUpdate
        monday = _monday(date).isoformat()
        for i in idx:
            df.at[i, "back_from_loa_date"] = monday
            df.at[i, "current_status"] = "Production"
            df.at[i, "work_status"] = "Production"
        save_df(f"plan_{pid}_emp", df)
        return df.to_dict("records"), False, "Back from LOA ✓", False

    # Terminate
    @app.callback(
        Output("modal-term", "is_open"),
        Output("term-who", "children"),
        Input("btn-emp-term", "n_clicks"),
        Input("btn-term-cancel", "n_clicks"),
        State("tbl-emp-roster", "data"),
        State("tbl-emp-roster", "selected_rows"),
        prevent_initial_call=True
    )
    def _open_term(n_open, n_cancel, data, sel):
        t = ctx.triggered_id
        if t == "btn-emp-term":
            df, idx = _selected_rows(data, sel)
            who = ", ".join(df.loc[idx, "name"].astype(str).tolist()) if not df.empty and idx else ""
            return True, f"Selected: {who}"
        return False, ""

    @app.callback(
        Output("tbl-emp-roster", "data", allow_duplicate=True),
        Output("modal-term", "is_open", allow_duplicate=True),
        Output("plan-msg", "children", allow_duplicate=True),
        Output("plan-msg-timer", "disabled", allow_duplicate=True),
        Input("btn-term-save", "n_clicks"),
        State("inp-term-date", "date"),
        State("tbl-emp-roster", "data"),
        State("tbl-emp-roster", "selected_rows"),
        State("plan-detail-id", "data"),
        prevent_initial_call=True
    )
    def _do_term(n, date, data, sel, pid):
        if not n or not date:
            raise dash.exceptions.PreventUpdate
        df, idx = _selected_rows(data, sel)
        if df.empty or not idx:
            raise dash.exceptions.PreventUpdate
        for i in idx:
            df.at[i, "terminate_date"] = pd.to_datetime(date).date().isoformat()
            df.at[i, "current_status"] = "Terminated"
            df.at[i, "work_status"] = "Terminated"
        save_df(f"plan_{pid}_emp", df)
        return df.to_dict("records"), False, "Terminated ✓", False

    # Transfer & Promotion — open
    @app.callback(
        Output("modal-tp", "is_open"),
        Output("tp-hier-map", "data"),
        Output("tp-current", "data"),
        Output("tp-who", "children"),
        Input("btn-emp-tp", "n_clicks"),
        Input("btn-tp-cancel", "n_clicks"),
        State("plan-detail-id", "data"),
        State("tbl-emp-roster", "data"),
        State("tbl-emp-roster", "selected_rows"),
        prevent_initial_call=True
    )
    def _open_tp(n_open, n_cancel, pid, data, sel):
        t = ctx.triggered_id
        if t == "btn-emp-tp":
            # 🔁 Prefer Headcount Update (Journey / Level 3 / Position Location Building Description)
            hier = _hier_from_hcu()
            if not hier.get("ba"):  # fallback if HCU missing/empty
                hier = _build_global_hierarchy()

            p = get_plan(pid) or {}
            cur = dict(
                ba=p.get("business_area","") or p.get("plan_ba","") or p.get("vertical","") or p.get("ba",""),
                subba=p.get("sub_business_area","") or p.get("plan_sub_ba","") or p.get("sub_ba","") or p.get("subba",""),
                lob=p.get("lob","") or p.get("channel",""),
                site=p.get("site",""),
            )

            df, idx = _selected_rows(data, sel)
            who = ", ".join(df.loc[idx, "name"].astype(str).tolist()) if not df.empty and idx else ""
            return True, hier, cur, f"Selected: {who}"
        return False, {}, {}, ""

    @app.callback(
        Output("tp-ba", "options"),
        Output("tp-ba", "value"),
        Input("tp-hier-map", "data"),
        State("tp-current", "data"),
        prevent_initial_call=False
    )
    def _tp_fill_ba(hmap, cur):
        bas = list((hmap or {}).get("ba") or [])
        opts = [{"label": b, "value": b} for b in bas]
        cur_ba = (cur or {}).get("ba")
        val = cur_ba if cur_ba in bas else (bas[0] if bas else None)
        return opts, val

    @app.callback(
        Output("tp-subba", "options"),
        Output("tp-subba", "value"),
        Input("tp-ba", "value"),
        State("tp-hier-map", "data"),
        State("tp-current", "data"),
        prevent_initial_call=False
    )
    def _tp_fill_subba(ba_val, hmap, cur):
        sub_map = dict((hmap or {}).get("subba") or {})
        # tolerant key match for BA
        key = None
        if ba_val:
            for k in sub_map.keys():
                if str(k).strip().lower() == str(ba_val).strip().lower():
                    key = k; break
        subs = list(sub_map.get(key, [])) if key else []
        opts = [{"label": s, "value": s} for s in subs]
        cur_sub = (cur or {}).get("subba")
        val = cur_sub if cur_sub in subs else (subs[0] if subs else None)
        return opts, val

    @app.callback(
        Output("tp-lob", "options"),
        Output("tp-lob", "value"),
        Input("tp-ba", "value"),
        Input("tp-subba", "value"),
        State("tp-hier-map", "data"),
        State("tp-current", "data"),
        prevent_initial_call=False
    )
    def _tp_fill_lob(ba_val, sub_val, hmap, cur):
        lob_map = dict((hmap or {}).get("lob") or {})
        # tolerant composite key
        key = None
        if ba_val and sub_val:
            target = f"{str(ba_val).strip().lower()}|{str(sub_val).strip().lower()}"
            for k in lob_map.keys():
                if str(k).strip().lower() == target:
                    key = k; break
        lobs = list(lob_map.get(key, CHANNEL_DEFAULTS))
        opts = [{"label": l, "value": l} for l in lobs]
        cur_lob = (cur or {}).get("lob")
        val = cur_lob if cur_lob in lobs else (lobs[0] if lobs else None)
        return opts, val


    # TP: mirror values to twp-* for the "Transfer with Promotion" tab on open/change

    @app.callback(
        Output("twp-ba", "options"), Output("twp-ba", "value"),
        Output("twp-subba", "options"), Output("twp-subba", "value"),
        Output("twp-lob", "options"), Output("twp-lob", "value"),
        Input("tp-ba", "options"), Input("tp-ba", "value"),
        Input("tp-subba", "options"), Input("tp-subba", "value"),
        Input("tp-lob", "options"), Input("tp-lob", "value"),
        prevent_initial_call=False

    )
    def _mirror_tp_to_twp(ba_opts, ba_val, sub_opts, sub_val, lob_opts, lob_val):
        return ba_opts or [], ba_val, sub_opts or [], sub_val, lob_opts or [], lob_val

    # TP: simple Site picker – gather unique sites we know (from roster wide/long + plan site)
    @app.callback(
        Output("tp-site", "options"), Output("tp-site", "value"),
        Output("twp-site", "options"), Output("twp-site", "value"),
        Input("tp-current", "data"),
        prevent_initial_call=False
    )
    def _fill_sites(cur):
        # Collect sites from the Headcount Update upload only
        sites: set[str] = set()
        try:
            hcu = _load_hcu_df()  # → load_headcount() under the hood
            if isinstance(hcu, pd.DataFrame) and not hcu.empty:
                L = _lower_map(hcu)
                # primary column, plus a few tolerant aliases
                site_col = (
                    L.get("position location building description")
                    or L.get("position_location_building_description")
                    or L.get("building description")
                    or L.get("site")
                )
                if site_col:
                    s = (
                        hcu[site_col]
                        .dropna()
                        .astype(str)
                        .str.strip()
                        .replace({"": np.nan})
                        .dropna()
                        .unique()
                        .tolist()
                    )
                    sites |= set(s)
        except Exception:
            pass

        # Always include current plan site (so user sees what's already set)
        cur_site = (cur or {}).get("site")
        if cur_site:
            sites.add(str(cur_site).strip())

        opts = [{"label": s, "value": s} for s in sorted(sites)]
        # Prefer current plan site if it exists in options; else first option (if any)
        val = cur_site if (cur_site in sites) else (sorted(sites)[0] if sites else None)
        return opts, val, opts, val



    # TP: Save (applies to selected rows)

    @app.callback(
        Output("tbl-emp-roster", "data", allow_duplicate=True),
        Output("modal-tp", "is_open", allow_duplicate=True),
        Output("plan-msg", "children", allow_duplicate=True),
        Output("plan-msg-timer", "disabled", allow_duplicate=True),
        Input("btn-tp-save", "n_clicks"),
        State("tp-active-tab", "value"),
        # transfer
        State("tp-ba", "value"), State("tp-subba", "value"), State("tp-lob", "value"), State("tp-site", "value"),
        State("tp-transfer-type", "value"),
        State("tp-new-class", "value"),
        State("tp-class-ref", "value"),
        State("tp-date-from", "date"), State("tp-date-to", "date"),
        # promotion
        State("promo-type", "value"), State("promo-role", "value"),
        State("promo-date-from", "date"), State("promo-date-to", "date"),
        # both
        State("twp-ba", "value"), State("twp-subba", "value"),
        State("twp-lob", "value"), State("twp-site", "value"),
        State("twp-type", "value"), State("twp-new-class", "value"),
        State("twp-class-ref", "value"), State("twp-role", "value"),
        State("twp-date-from", "date"), State("twp-date-to", "date"),
        # selection + data
        State("tbl-emp-roster", "data"),
        State("tbl-emp-roster", "selected_rows"),
        State("plan-detail-id", "data"),
        prevent_initial_call=True
    )
    def _tp_save(n, tab,
                 t_ba, t_sub, t_lob, t_site, t_type, t_newcls, t_clref, t_from, t_to,
                 p_type, p_role, p_from, p_to,
                 b_ba, b_sub, b_lob, b_site, b_type, b_newcls, b_clref, b_role, b_from, b_to,
                 data, sel, pid):
        if not n:
            raise dash.exceptions.PreventUpdate
        df, idx = _selected_rows(data, sel)
        if df.empty or not idx:
            raise dash.exceptions.PreventUpdate

        def _apply_transfer(I, ba, sub, lob, site, typ, newclass, cref, dfrom, dto):
            if ba:   df.at[I, "biz_area"] = ba
            if sub:  df.at[I, "sub_biz_area"] = sub
            if lob:  df.at[I, "lob"] = lob
            if site: df.at[I, "site"] = site
            if newclass and cref:
                df.at[I, "class_ref"] = cref
                if dfrom: df.at[I, "training_start"] = pd.to_datetime(dfrom).date().isoformat()
            df.at[I, "current_status"] = "Interim Transfer" if typ == "interim" else "Transferred"
            df.at[I, "work_status"] = df.at[I, "current_status"]

        def _apply_promo(I, typ, role, dfrom, dto):
            if role: df.at[I, "role"] = role
            df.at[I, "current_status"] = "Promotion (Temp)" if typ == "interim" else "Promotion"
            df.at[I, "work_status"] = "Production"

        if tab == "tp-transfer":
            for I in idx:
                _apply_transfer(I, t_ba, t_sub, t_lob, t_site, (t_type or "perm"),
                                bool(t_newcls), t_clref, t_from, t_to)
        elif tab == "tp-promo":
            for I in idx:
                _apply_promo(I, (p_type or "perm"), p_role, p_from, p_to)
        else:  # both
            for I in idx:
                _apply_transfer(I, b_ba, b_sub, b_lob, b_site, (b_type or "perm"),
                                bool(b_newcls), b_clref, b_from, b_to)
                _apply_promo(I, (b_type or "perm"), b_role, b_from, b_to)
        save_df(f"plan_{pid}_emp", df)
        return df.to_dict("records"), False, "Transfer / Promotion saved ✓", False

    #_________________________________End Employee Roster Modals & Actions __________________________________

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
        Output("tbl-hc", "columns", allow_duplicate=True),
        Output("tbl-attr", "columns"),
        Output("tbl-shr", "columns"),
        Output("tbl-train", "columns"),
        Output("tbl-ratio", "columns"),
        Output("tbl-seat", "columns"),
        Output("tbl-bva", "columns"),
        Output("tbl-nh", "columns"),
        Output("tbl-emp-roster", "columns"),
        Output("tbl-bulk-files", "columns"),
        Output("tbl-notes", "columns"),
        Input("plan-detail-id", "data"),
        State("url-router", "pathname"),
        prevent_initial_call=True,
    )
    def _init_cols(pid, pathname):
        path = (pathname or "").rstrip("/")
        if not (isinstance(pid, int) and path.startswith("/plan/")):
            raise dash.exceptions.PreventUpdate

        p = get_plan(pid) or {}
        name  = p.get("plan_name")  or f"Plan {pid}"
        ptype = p.get("plan_type")  or "Volume Based"

        weeks = _week_span(p.get("start_week"), p.get("end_week"))
        cols, week_ids = _week_cols(weeks)

        notes_cols = [{"name": "Date", "id": "when"},{"name": "User", "id": "user"},{"name": "Note", "id": "note"}]

        return (
            name, ptype, week_ids,
            cols, cols, cols, cols, cols, cols, cols, cols, cols,
            _roster_columns(), _bulkfile_columns(), notes_cols
        )

    @app.callback(
        Output("plan-upper", "children"),
        Output("tbl-fw", "data"), Output("tbl-hc", "data", allow_duplicate=True),
        Output("tbl-attr", "data"), Output("tbl-shr", "data"),
        Output("tbl-train", "data"), Output("tbl-ratio", "data"),
        Output("tbl-seat", "data"), Output("tbl-bva", "data"),
        Output("tbl-nh", "data"),
        Output("tbl-emp-roster", "data"),
        Output("tbl-bulk-files", "data"),
        Output("tbl-notes", "data"),
        Output("plan-loading", "data", allow_duplicate=True),   # ← NEW
        Input("plan-type", "data"),
        State("plan-detail-id", "data"),
        State("tbl-fw", "columns"),
        Input("plan-refresh-tick", "data"),
        prevent_initial_call=True,
    )
    def _fill_tables(ptype, pid, fw_cols, _tick):
        results = _fill_tables_fixed(ptype, pid, fw_cols, _tick)
        return (*results, False)

    # Save all tabs
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
        State("tbl-bulk-files", "data"),
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
        _save_table(pid, "bulk_files", pd.DataFrame(bulk_files or []))
        _save_table(pid, "notes",      pd.DataFrame(notes or []))
        return "Saved ✓", False

    # Upper collapse
    @app.callback(
        Output("plan-upper-collapsed", "data"),
        Output("plan-upper", "style"),
        Output("plan-hdr-collapse", "children"),
        Input("plan-hdr-collapse", "n_clicks"),
        State("plan-upper-collapsed", "data"),
        prevent_initial_call=False
    )
    def _toggle_upper(n_clicks, collapsed):
        collapsed = bool(collapsed)
        if n_clicks:
            collapsed = not collapsed
        style = {"display": "none"} if collapsed else {"display": "block"}
        icon = "▾" if collapsed else "▴"
        return collapsed, style, icon

    # Refresh trigger
    @app.callback(
        Output("plan-msg", "children", allow_duplicate=True),
        Output("plan-msg-timer", "disabled", allow_duplicate=True),
        Output("plan-refresh-tick", "data", allow_duplicate=True),
        Output("plan-loading", "data", allow_duplicate=True), 
        Input("btn-plan-refresh", "n_clicks"),
        State("plan-refresh-tick", "data"),
        prevent_initial_call=True
    )
    def _refresh_msg(_n, tick):
        tick = int(tick or 0) + 1
        return "Refreshed ✓", False, tick, True

    # Clear banner after 5s
    @app.callback(
        Output("plan-msg", "children"),
        Output("plan-msg-timer", "disabled"),
        Input("plan-msg-timer", "n_intervals"),
        prevent_initial_call=True
    )
    def _clear_msg(_ticks):
        return "", True

    # Enable action buttons only when rows selected
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
    def _toggle_roster_buttons(data, selected_rows):
        has_rows = bool(data) and len(data) > 0
        has_sel  = has_rows and bool(selected_rows)
        disabled = not has_sel
        return (disabled,)*8

    @app.callback(
        Output("tbl-emp-roster", "row_selectable"),
        Output("tbl-emp-roster", "selected_rows"),
        Input("tbl-emp-roster", "data"),
        prevent_initial_call=False
    )
    def _roster_selectability(data):
        has_rows = bool(data) and len(data) > 0
        return ("multi" if has_rows else False, [] if not has_rows else no_update)

    # "+ Add New" modal open/crumb
    @app.callback(
        Output("modal-emp-add", "is_open"),
        Output("modal-roster-crumb", "children"),
        Input("btn-emp-add", "n_clicks"),
        Input("btn-emp-modal-cancel", "n_clicks"),
        State("plan-detail-id", "data"),
        prevent_initial_call=True
    )
    def _modal_toggle(n_add, n_cancel, pid):
        trigger = ctx.triggered_id
        if trigger == "btn-emp-add":
            p = get_plan(pid) or {}
            crumb = _format_crumb(p)
            return True, crumb
        return False, ""

    # Add employee Save
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
        if not brid or any(str(r.get("brid","")).strip()==str(brid).strip() for r in data):
            return data, f"Total: {len(data):02d} Records", False, "BRID exists or missing ✗", False

        p = get_plan(pid) or {}
        r = {cid: "" for cid in _ROSTER_REQUIRED_IDS}
        r.update({
            "brid": brid,
            "name": name or "",
            "ftpt_status": ftpt or "",
            "ftpt_hours": "",
            "role": role or "Agent",
            "production_start": prod_date or "",
            "team_leader": tl or "",
            "avp": avp or "",
            "work_status": "Production",
            "current_status": "Production",
            "biz_area": (p.get("business_area") or p.get("plan_ba") or "").strip(),
            "sub_biz_area": (p.get("sub_business_area") or p.get("plan_sub_ba") or "").strip(),
            "lob": ((p.get("lob") or p.get("channel") or "").strip().title()),
            "site": (p.get("site") or "").strip(),
        })
        new = data + [r]
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

    # Workstatus dataset download
    @app.callback(
        Output("dl-workstatus", "data"),
        Input("btn-emp-dl", "n_clicks"),
        State("tbl-emp-roster", "data"),
        prevent_initial_call=True
    )
    def _download_workstatus(_n, data):
        df = pd.DataFrame(data or [])
        return dcc.send_data_frame(df.to_csv, "workstatus_dataset.csv", index=False)

    # Bulk upload ingest
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

        if not recs_df.empty and "brid" in recs_df.columns:
            existing = {str(r.get("brid","")).strip(): i for i, r in enumerate(roster_data)}
            for _, row in recs_df.iterrows():
                key = str(row.get("brid","")).strip()
                if not key:
                    continue
                if key in existing:
                    roster_data[existing[key]].update({k: row.get(k, roster_data[existing[key]].get(k)) for k in _ROSTER_REQUIRED_IDS})
                else:
                    new_row = {cid: row.get(cid, "") for cid in _ROSTER_REQUIRED_IDS}
                    roster_data.append(new_row)

            save_df(f"plan_{pid}_emp", pd.DataFrame(roster_data))
            save_df(f"plan_{pid}_bulk_files", pd.DataFrame(files_data))

            return (files_data, roster_data, f"Total: {len(roster_data):02d} Records", "Bulk file loaded ✓", False)

        save_df(f"plan_{pid}_bulk_files", pd.DataFrame(files_data))
        return (files_data, roster_data, f"Total: {len(roster_data):02d} Records", "Bulk file invalid ✗", False)

    @app.callback(
        Output("dl-template", "data"),
        Input("btn-template-dl", "n_clicks"),
        prevent_initial_call=True
    )
    def _download_template(_n):
        cols = [c["name"] for c in _roster_columns()]
        df = pd.DataFrame(columns=cols)
        return dcc.send_data_frame(df.to_csv, "employee_roster_template.csv", index=False)

    # Notes save
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
        row = {"when": stamp, "user": "User", "note": text.strip()}
        data = [row] + data
        save_df(f"plan_{pid}_notes", pd.DataFrame(data))
        return data, "Note saved ✓", False
    
    @app.callback(
        Output("plan-loading", "data"),
        Input("plan-detail-id", "data"),
        prevent_initial_call=False
    )
    def _start_page_loading(_pid):
        # Show overlay as soon as we land on /plan/<id>
        return True
    
    @app.callback(
        Output("plan-loading-overlay", "style"),
        Input("plan-loading", "data"),
        prevent_initial_call=False
    )
    def _toggle_overlay(is_loading):
        style = _OVERLAY_STYLE.copy()
        style["display"] = "flex" if is_loading else "none"
        return style

    @app.callback(
        Output("tbl-hc","data", allow_duplicate=True),
        Output("tbl-hc","columns", allow_duplicate=True),
        Input("set-scope","value"),
        Input("set-ba","value"), Input("set-subba","value"), Input("set-lob","value"),
        State("plan-weeks","data"),
        prevent_initial_call=True
    )
    def hc_tab_data(scope, ba, sba, lob, week_ids, roster_rows):
        from dash import no_update
        import pandas as pd

        if scope != "hier" or not (ba and sba and lob):
            return [], no_update

        week_ids = list(week_ids or [])
        metrics = ["Budget HC (#)","Planned HC (#)","Actual HC (#)","SME Billable HC (#)","Variance (#)"]
        df = _blank_grid(metrics, week_ids)  # your existing helper

        # tiny local helpers
        def _set(metric, w, val): df.loc[df["metric"]==metric, w] = float(val or 0)
        def _get(metric, w):
            ser = df.loc[df["metric"]==metric, w]
            return float(ser.iloc[0]) if len(ser) and pd.notna(ser.iloc[0]) else 0.0

        # ---- Budget / Planned from saved timeseries (unchanged) ----
        key = _scope_key(ba, sba, lob)
        bud = load_timeseries("hc_budget",  key) or pd.DataFrame(columns=["week","headcount"])
        pla = load_timeseries("hc_planned", key) or pd.DataFrame(columns=["week","headcount"])

        if isinstance(bud, pd.DataFrame) and not bud.empty:
            for _, r in bud.iterrows():
                w = str(r.get("week",""))
                if w in week_ids:
                    _set("Budget HC (#)", w, r.get("headcount", 0))

        if isinstance(pla, pd.DataFrame) and not pla.empty:
            for _, r in pla.iterrows():
                w = str(r.get("week",""))
                if w in week_ids:
                    _set("Planned HC (#)", w, r.get("headcount", 0))

        for w in week_ids:  # Planned fallback to Budget
            if _get("Planned HC (#)", w) == 0:
                _set("Planned HC (#)", w, _get("Budget HC (#)", w))

        # ---- Actuals from Employee Roster sub-tab (what you asked) ----
        r = pd.DataFrame(roster_rows or [])
        if not r.empty:
            L = {str(c).strip().lower(): c for c in r.columns}
            # core columns (be liberal with names)
            brid_c = L.get("brid") or L.get("employee id") or L.get("employee_id")
            role_c = L.get("role") or L.get("position group") or L.get("position description")
            ba_c   = L.get("business area") or L.get("ba")
            sba_c  = L.get("sub business area") or L.get("level 3") or L.get("level_3")
            lob_c  = L.get("lob") or L.get("channel") or L.get("program")
            cur_c  = L.get("current status") or L.get("current_status") or L.get("status")
            work_c = L.get("work status") or L.get("work_status")

            if brid_c and role_c:
                r[brid_c] = r[brid_c].astype(str).str.strip()

                # scope filter (only apply if the columns exist)
                def _match(col, val):
                    if not col or col not in r.columns: return True
                    return r[col].astype(str).str.strip().str.lower() == (val or "").strip().lower()
                r = r[_match(ba_c, ba) & _match(sba_c, sba) & _match(lob_c, lob)]

                # effective status: Current Status else Work Status
                if cur_c and cur_c in r.columns:
                    eff = r[cur_c].astype(str)
                    if work_c and work_c in r.columns:
                        eff = eff.where(eff.str.strip()!="", r[work_c].astype(str))
                else:
                    eff = r[work_c].astype(str) if work_c and work_c in r.columns else ""

                eff = eff.str.strip().str.lower()
                is_prod = eff.eq("production")

                role_txt = r[role_c].astype(str).str.strip().str.lower()
                is_agent = role_txt.str.contains(r"\bagent\b")
                is_sme   = role_txt.str.contains(r"\bsme\b")

                # distinct BRIDs
                agent_cnt = (
                    r.loc[is_prod & is_agent, brid_c]
                    .dropna().astype(str).str.strip().nunique()
                )
                sme_cnt = (
                    r.loc[is_prod & is_sme, brid_c]
                    .dropna().astype(str).str.strip().nunique()
                )

                for w in week_ids:
                    _set("Actual HC (#)", w, agent_cnt)
                    _set("SME Billable HC (#)", w, sme_cnt)

        # ---- Variance = Actual Agent − Budget (leave this as-is unless you want to include SMEs) ----
        for w in week_ids:
            _set("Variance (#)", w, _get("Actual HC (#)", w) - _get("Budget HC (#)", w))

        # finalize types/columns
        for w in week_ids:
            df[w] = pd.to_numeric(df[w], errors="coerce").fillna(0).round(0).astype(int)
        cols = [{"name":"Metric","id":"metric","editable":False}] + [{"name": w, "id": w} for w in week_ids]
        return df.to_dict("records"), cols

def _fill_tables_fixed(ptype, pid, fw_cols, _tick):
    if not (pid and fw_cols):
        raise dash.exceptions.PreventUpdate

    week_ids = [c["id"] for c in fw_cols if c["id"] != "metric"]

    def _plan_specs(k: str) -> Dict[str, List[str]]:
        k = (k or "").strip().lower()
        if k.startswith("volume"):
            return {"fw": ["Forecast","Tactical Forecast","Actual Volume","Budgeted AHT/SUT","Target AHT/SUT","Actual AHT/SUT","Occupancy"],
                    "upper": ["FTE Required @ Forecast Volume","FTE Required @ Actual Volume","FTE Over/Under MTP Vs Actual","FTE Over/Under Tactical Vs Actual","FTE Over/Under Budgeted Vs Actual","Projected Supply HC","Projected Handling Capacity (#)","Projected Service Level"]}
        if k.startswith("billable hours"):
            return {"fw": ["Billable Hours","AHT/SUT","Shrinkage","Training"],
                    "upper": ["Billable FTE Required (#)","Headcount Required With Shrinkage (#)","FTE Over/Under (#)"]}
        if k.startswith("fte based billable"):
            return {"fw": ["Billable Txns","AHT/SUT","Efficiency","Shrinkage"],
                    "upper": ["Billable Transactions","FTE Required (#)","FTE Over/Under (#)"]}
        return {"fw": ["Billable FTE Required","Shrinkage","Training"],
                "upper": ["FTE Required (#)","FTE Over/Under (#)"]}

    spec = _plan_specs(ptype)

    # --- Scope and settings ---
    p = get_plan(pid) or {}
    ch_first = (p.get("channel") or "").split(",")[0].strip()
    sk = _canon_scope(p.get("vertical"), p.get("sub_ba"), ch_first)
    loc_first = (p.get("location") or p.get("country") or p.get("site") or "").strip()

    settings = resolve_settings(ba=p.get("vertical"), subba=p.get("sub_ba"), lob=ch_first)
    s_target_aht = float(settings.get("target_aht", settings.get("budgeted_aht", 300)) or 300)
    s_budget_aht = float(settings.get("budgeted_aht", settings.get("target_aht", s_target_aht)) or s_target_aht)
    s_target_sut = float(settings.get("target_sut", settings.get("budgeted_sut", 600)) or 600)
    s_budget_sut = float(settings.get("budgeted_sut", settings.get("target_sut", s_target_sut)) or s_target_sut)
    sl_seconds   = int(settings.get("sl_seconds", 20) or 20)

    # SLA target (%) if provided (default 80)
    sl_target_pct = None
    for k in ("sl_target_pct","service_level_target","sl_target","sla_target_pct","sla_target"):
        v = settings.get(k)
        if v not in (None, ""):
            try:
                sl_target_pct = float(str(v).replace("%",""))
            except Exception:
                pass
            break
    if sl_target_pct is None:
        sl_target_pct = 80.0

    # --- Time series ---
    vF = _assemble_voice(sk, "forecast"); vA = _assemble_voice(sk, "actual");  vT = _assemble_voice(sk, "tactical")
    bF = _assemble_bo(sk,   "forecast"); bA = _assemble_bo(sk,   "actual");    bT = _assemble_bo(sk,   "tactical")

    # Fallback for requirements if actuals missing
    use_voice_for_req = vA if isinstance(vA, pd.DataFrame) and not vA.empty else vF
    use_bo_for_req    = bA if isinstance(bA, pd.DataFrame) and not bA.empty else bF

    # Weekly rollups
    vF_w = _weekly_voice(vF); vA_w = _weekly_voice(vA); vT_w = _weekly_voice(vT)
    bF_w = _weekly_bo(bF);   bA_w = _weekly_bo(bA);   bT_w = _weekly_bo(bT)
    vF_w = vF_w.set_index("week") if not vF_w.empty else pd.DataFrame()
    vA_w = vA_w.set_index("week") if not vA_w.empty else pd.DataFrame()
    vT_w = vT_w.set_index("week") if not vT_w.empty else pd.DataFrame()
    bF_w = bF_w.set_index("week") if not bF_w.empty else pd.DataFrame()
    bA_w = bA_w.set_index("week") if not bA_w.empty else pd.DataFrame()
    bT_w = bT_w.set_index("week") if not bT_w.empty else pd.DataFrame()

    # Helpers for column picking / safe gets
    def _pick(df, names):
        if not isinstance(df, pd.DataFrame) or df.empty: return None
        for n in names:
            if n in df.columns: return n
        return None
    def _get(df, idx, col, default=0.0):
        try:
            if isinstance(df, pd.DataFrame) and (col in df.columns) and (idx in df.index):
                val = df.loc[idx, col]
                return float(val) if pd.notna(val) else default
        except Exception:
            return default
        return default
    def _first_positive(*vals, default=None):
        for v in vals:
            try:
                x = float(v)
                if x > 0:
                    return x
            except Exception:
                pass
        return default

    # choose voice columns per frame (lets us mix forecast/actual safely)
    v_vol_col_F = _pick(vF_w, ["vol","calls","volume"]) or "vol"
    v_vol_col_A = _pick(vA_w, ["vol","calls","volume"]) or v_vol_col_F
    v_vol_col_T = _pick(vT_w, ["vol","calls","volume"]) or v_vol_col_F

    b_itm_col   = _pick(bF_w, ["items","txns","transactions","volume"]) or "items"
    v_aht_col_F = _pick(vF_w, ["aht","aht_sec","avg_aht"])
    v_aht_col_A = _pick(vA_w, ["aht","aht_sec","avg_aht"])
    b_sut_col_F = _pick(bF_w, ["sut","sut_sec","aht_sec","avg_sut"])
    b_sut_col_A = _pick(bA_w, ["sut","sut_sec","aht_sec","avg_sut"])

    # --- Build Forecast & Workload grid ---
    fw_rows = spec["fw"]
    fw = pd.DataFrame({"metric": fw_rows})
    for w in week_ids: fw[w] = 0.0

    # Track weekly AHT/SUT used later
    wk_aht_sut_actual, wk_aht_sut_forecast = {}, {}

    # For SL – count intervals if present, else fallback to 24x7
    ivl_sec = 1800  # 30-minute intervals
    weekly_voice_intervals = {}
    try:
        if isinstance(vF, pd.DataFrame) and not vF.empty and {"date","interval_start"}.issubset(vF.columns):
            tmp = vF.copy()
            dts = pd.to_datetime(tmp["date"], errors="coerce")
            tmp["week"] = (dts - pd.to_timedelta(dts.dt.weekday, unit="D")).dt.date.astype(str)
            weekly_voice_intervals = tmp.groupby("week", as_index=False)["interval_start"].count().set_index("week")["interval_start"].to_dict()
    except Exception:
        weekly_voice_intervals = {}
    intervals_per_week_default = 7 * (24 * 3600 // ivl_sec)  # 336 for 24x7

    # demand used for SL
    weekly_demand_voice, weekly_demand_bo = {}, {}
    # Build settings overrides (if any)
    voice_ovr = _settings_volume_aht_overrides(sk, "voice")
    bo_ovr    = _settings_volume_aht_overrides(sk, "bo")
    for w in week_ids:
        f_voice = _get(vF_w, w, v_vol_col_F, 0.0) if v_vol_col_F else 0.0
        f_bo    = _get(bF_w, w, b_itm_col,   0.0) if b_itm_col   else 0.0
        a_voice = _get(vA_w, w, v_vol_col_A, 0.0) if v_vol_col_A else 0.0
        a_bo    = _get(bA_w, w, b_itm_col,   0.0) if b_itm_col   else 0.0
        t_voice = _get(vT_w, w, v_vol_col_T, 0.0) if v_vol_col_T else 0.0
        t_bo    = _get(bT_w, w, b_itm_col,   0.0) if b_itm_col   else 0.0

        if w in voice_ovr["vol_w"]:
            f_voice = voice_ovr["vol_w"][w]
        if w in bo_ovr["vol_w"]:
            f_bo = bo_ovr["vol_w"][w]
        
        # Use overridden AHT/SUT in the forecast weighting when available
        ovr_aht_voice = voice_ovr["aht_or_sut_w"].get(w, None)
        ovr_sut_bo    = bo_ovr["aht_or_sut_w"].get(w, None)

        weekly_demand_voice[w] = (f_voice if f_voice > 0 else (a_voice if a_voice > 0 else t_voice))
        weekly_demand_bo[w]    = (f_bo    if f_bo    > 0 else (a_bo    if a_bo    > 0 else t_bo))

        if "Forecast" in fw_rows:
            fw.loc[fw["metric"]=="Forecast", w] = f_voice + f_bo
        if "Tactical Forecast" in fw_rows:
            fw.loc[fw["metric"]=="Tactical Forecast", w] = t_voice + t_bo
        if "Actual Volume" in fw_rows:
            fw.loc[fw["metric"]=="Actual Volume", w] = a_voice + a_bo

        # Weighted Actual AHT/SUT (voice+bo)
        a_num = a_den = 0.0
        if v_aht_col_A: a_num += _get(vA_w, w, v_aht_col_A, 0.0) * _get(vA_w, w, v_vol_col_A, 0.0); a_den += _get(vA_w, w, v_vol_col_A, 0.0)
        if b_sut_col_A: a_num += _get(bA_w, w, b_sut_col_A, 0.0) * _get(bA_w, w, b_itm_col,   0.0); a_den += _get(bA_w, w, b_itm_col,   0.0)
        actual_aht_sut = (a_num / a_den) if a_den > 0 else 0.0
        actual_aht_sut = _first_positive(actual_aht_sut, s_target_aht, default=s_target_aht)  # reject 0
        wk_aht_sut_actual[w] = actual_aht_sut
        if "Actual AHT/SUT" in fw_rows:
            fw.loc[fw["metric"]=="Actual AHT/SUT", w] = actual_aht_sut

       # --- Weighted Forecast AHT/SUT (override-aware) ---
        f_num = f_den = 0.0
        # voice contribution
        if ovr_aht_voice is not None and f_voice > 0:
            f_num += ovr_aht_voice * f_voice
            f_den += f_voice
        elif v_aht_col_F:
            f_num += _get(vF_w, w, v_aht_col_F, 0.0) * _get(vF_w, w, v_vol_col_F, 0.0)
            f_den += _get(vF_w, w, v_vol_col_F, 0.0)
        
        # back-office contribution
        if ovr_sut_bo is not None and f_bo > 0:
            f_num += ovr_sut_bo * f_bo
            f_den += f_bo
        elif b_sut_col_F:
            f_num += _get(bF_w, w, b_sut_col_F, 0.0) * _get(bF_w, w, b_itm_col,   0.0)
            f_den += _get(bF_w, w, b_itm_col,   0.0)
        
        forecast_aht_sut = (f_num / f_den) if f_den > 0 else 0.0
        forecast_aht_sut = _first_positive(forecast_aht_sut, s_target_aht, default=s_target_aht)
        wk_aht_sut_forecast[w] = forecast_aht_sut
        denom = float(f_voice + f_bo)

        # if no forecast volume at all this week, fall back to a sensible scalar
        if "Budgeted AHT/SUT" in fw_rows:
            if denom > 0:
                bud_val = ((f_voice * s_budget_aht) + (f_bo * s_budget_sut)) / denom
            else:
                # fallback: prefer voice budget AHT, else BO budget SUT
                bud_val = _first_positive(s_budget_aht, s_budget_sut, default=s_budget_aht)
            fw.loc[fw["metric"] == "Budgeted AHT/SUT", w] = float(bud_val)

        if "Target AHT/SUT" in fw_rows:
            if denom > 0:
                tgt_val = ((f_voice * s_target_aht) + (f_bo * s_target_sut)) / denom
            else:
                # fallback: prefer voice target AHT, else BO target SUT
                tgt_val = _first_positive(s_target_aht, s_target_sut, default=s_target_aht)
            fw.loc[fw["metric"] == "Target AHT/SUT", w] = float(tgt_val)
        if "Forecast AHT/SUT" in fw_rows:
            fw.loc[fw["metric"]=="Forecast AHT/SUT", w] = forecast_aht_sut
        
        # also ensure the "Forecast" row uses the possibly overridden volumes
        if "Forecast" in fw_rows:
            fw.loc[fw["metric"]=="Forecast", w] = f_voice + f_bo
        
        # demand used for SL should see the overrides too
        weekly_demand_voice[w] = f_voice if f_voice > 0 else (a_voice if a_voice > 0 else t_voice)
        weekly_demand_bo[w]    = f_bo    if f_bo    > 0 else (a_bo    if a_bo    > 0 else t_bo)


    # Occupancy value (and fraction for capacity)
    if "Occupancy" in fw_rows:
        def _setting(d, keys, default=None):
            if not isinstance(d, dict): return default
            for k in keys:
                if d.get(k) not in (None,""): return d.get(k)
            low = {str(k).strip().lower(): v for k,v in d.items()}
            for k in keys:
                kk = str(k).strip().lower()
                if low.get(kk) not in (None,""): return low.get(kk)
            return default
        occ_raw = _setting(
            settings,
            ["occupancy","occupancy_pct","occupancy percent","occupancy%","occupancy (%)","occ","target_occupancy",
             "target occupancy","budgeted_occupancy","budgeted occupancy","occupancy_cap_voice"],
            0.85
        )
        try:
            if isinstance(occ_raw, str) and occ_raw.strip().endswith("%"):
                occ = float(occ_raw.strip()[:-1])
            else:
                occ = float(occ_raw)
                if occ <= 1.0: occ *= 100.0
        except Exception:
            occ = 85.0
        occ = int(round(occ))
        for w in week_ids:
            fw.loc[fw["metric"] == "Occupancy", w] = occ
    else:
        occ = 85
    occ_frac = min(0.99, max(0.01, float(occ)/100.0))

    # Requirements (daily→weekly sums)
    req_daily_actual   = required_fte_daily(use_voice_for_req, use_bo_for_req, pd.DataFrame(), settings)
    req_daily_forecast = required_fte_daily(vF, bF, pd.DataFrame(), settings)
    req_daily_tactical = required_fte_daily(vT, bT, pd.DataFrame(), settings) if (isinstance(vT, pd.DataFrame) and not vT.empty) or (isinstance(bT, pd.DataFrame) and not bT.empty) else pd.DataFrame()
    vB = vF.copy(); bB = bF.copy()
    if isinstance(vB, pd.DataFrame) and not vB.empty: vB["aht_sec"] = float(s_budget_aht)
    if isinstance(bB, pd.DataFrame) and not bB.empty: bB["aht_sec"] = float(s_budget_sut)
    req_daily_budgeted = required_fte_daily(vB, bB, pd.DataFrame(), settings)

    def _daily_to_weekly(df):
        if not isinstance(df, pd.DataFrame) or df.empty or "date" not in df.columns or "total_req_fte" not in df.columns:
            return {}
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d = d.dropna(subset=["date"])
        d["week"] = (d["date"] - pd.to_timedelta(d["date"].dt.weekday, unit="D")).dt.date.astype(str)
        w = d.groupby("week", as_index=False)["total_req_fte"].sum().set_index("week")["total_req_fte"]
        return w.to_dict()

    req_w_actual   = _daily_to_weekly(req_daily_actual)
    req_w_forecast = _daily_to_weekly(req_daily_forecast)
    req_w_tactical = _daily_to_weekly(req_daily_tactical)
    req_w_budgeted = _daily_to_weekly(req_daily_budgeted)

    # ---------- FW overlay: user-entered AHT/SUT rows override computed ----------
    fw_saved = _load_or_blank(f"plan_{pid}_fw", spec["fw"], week_ids)

    def _merge_fw_user_overrides(fw_calc: pd.DataFrame, fw_user: pd.DataFrame, week_ids: list) -> pd.DataFrame:
        """
        Overlay user-entered FW rows onto the computed FW rows.
        We override only Budgeted/Target (aka Planned) AHT/SUT rows (with tolerant matching).
        """
        if not isinstance(fw_calc, pd.DataFrame) or fw_calc.empty:
            return fw_user if isinstance(fw_user, pd.DataFrame) else pd.DataFrame()

        calc = fw_calc.copy()
        if not isinstance(fw_user, pd.DataFrame) or fw_user.empty:
            return calc

        c = calc.set_index("metric")
        u = fw_user.set_index("metric")

        # make week columns numeric for safe coercion
        for w in week_ids:
            if w in c.columns: c[w] = pd.to_numeric(c[w], errors="coerce")
            if w in u.columns: u[w] = pd.to_numeric(u[w], errors="coerce")

        def _find_row(idx_like, *alts):
            low = {str(k).strip().lower(): k for k in idx_like}
            for a in alts:
                k = str(a).strip().lower()
                if k in low: return low[k]
            # fallback: substring contains
            for key, orig in low.items():
                for a in alts:
                    if str(a).strip().lower() in key:
                        return orig
            return None

        budget_label_calc = _find_row(c.index, "Budgeted AHT/SUT", "Budget AHT/SUT", "Budget AHT", "Budget SUT")
        target_label_calc = _find_row(c.index, "Target AHT/SUT",   "Planned AHT/SUT", "Planned AHT", "Planned SUT", "Target AHT", "Target SUT")

        budget_label_user = _find_row(u.index, "Budgeted AHT/SUT", "Budget AHT/SUT", "Budget AHT", "Budget SUT")
        target_label_user = _find_row(u.index, "Target AHT/SUT",   "Planned AHT/SUT", "Planned AHT", "Planned SUT", "Target AHT", "Target SUT")

        def _apply(canon_label, user_label):
            if not canon_label or not user_label:
                return
            for w in week_ids:
                if w in u.columns and w in c.columns:
                    val = u.at[user_label, w]
                    if pd.notna(val):
                        c.at[canon_label, w] = float(val)

        _apply(budget_label_calc, budget_label_user)
        _apply(target_label_calc, target_label_user)

        return c.reset_index()

    fw_to_use = _merge_fw_user_overrides(fw, fw_saved, week_ids)
    # ---------------------------------------------------------------------------

    # Lower grids (start from what's saved, then overwrite with computed/uploaded)
    hc   = _load_or_blank(f"plan_{pid}_hc",   ["Budgeted HC (#)","Planned/Tactical HC (#)","Actual Agent HC (#)","SME Billable HC (#)"], week_ids)
    att  = _load_or_blank(f"plan_{pid}_attr", ["Planned Attrition HC (#)","Actual Attrition HC (#)","Attrition %"], week_ids)
    shr  = _load_or_blank(f"plan_{pid}_shr",  ["OOO Shrink Hours (#)","Inoffice Shrink Hours (#)","OOO Shrinkage %","Inoffice Shrinkage %","Overall Shrinkage %"], week_ids)
    trn  = _load_or_blank(f"plan_{pid}_train",["Training Start (#)","Training End (#)","Nesting Start (#)","Nesting End (#)"], week_ids)
    rat  = _load_or_blank(f"plan_{pid}_ratio",["Planned TL/Agent Ratio","Actual TL/Agent Ratio","Variance"], week_ids)
    seat = _load_or_blank(f"plan_{pid}_seat", ["Seats Required (#)","Seats Available (#)","Seat Utilization %"], week_ids)
    bva  = _load_or_blank(f"plan_{pid}_bva",  ["Budgeted FTE (#)","Actual FTE (#)","Variance (#)"], week_ids)
    nh   = _load_or_blank(f"plan_{pid}_nh",   ["Planned New Hire HC (#)","Actual New Hire HC (#)","Recruitment Achievement"], week_ids)

    # Roster / bulk / notes
    roster_df = _load_or_empty_roster(pid)
    bulk_df   = _load_or_empty_bulk_files(pid)
    notes_df  = _load_or_empty_notes(pid)

    # Actual Agent HC from roster
    actual_agent_hc = 0
    sme_billable_hc = 0
    try:
        r = roster_df.copy()
        if isinstance(r, pd.DataFrame) and not r.empty:
            L = {str(c).strip().lower(): c for c in r.columns}

            brid_c = L.get("brid") or L.get("employee id") or L.get("employee_id")
            role_c = L.get("role") or L.get("position group") or L.get("position description")
            ba_c   = L.get("business area") or L.get("ba")
            sba_c  = L.get("sub business area") or L.get("level 3") or L.get("level_3")
            lob_c  = L.get("lob") or L.get("channel") or L.get("program")
            cur_c  = L.get("current status") or L.get("current_status") or L.get("status")
            work_c = L.get("work status")   or L.get("work_status")

            if brid_c and role_c:
                def _eq(col, val):
                    if not col or col not in r.columns or val in (None, ""):
                        return pd.Series(True, index=r.index)
                    return r[col].astype(str).str.strip().str.lower().eq(str(val).strip().lower())

                r = r[_eq(ba_c,  p.get("vertical")) &
                      _eq(sba_c, p.get("sub_ba")) &
                      _eq(lob_c, ch_first)]

                cur  = r[cur_c].astype(str)  if cur_c  in r.columns else pd.Series("", index=r.index)
                work = r[work_c].astype(str) if work_c in r.columns else pd.Series("", index=r.index)
                eff  = cur.where(cur.str.strip()!="", work).str.strip().str.lower()
                is_prod = eff.eq("production")

                role = r[role_c].astype(str).str.strip().str.lower()
                is_agent = role.str.contains(r"\bagent\b", regex=True)
                is_sme   = role.str.contains(r"\bsme\b",   regex=True)

                brid = r[brid_c].astype(str).str.strip()

                actual_agent_hc = int(brid[is_prod & is_agent].nunique())
                sme_billable_hc = int(brid[is_prod & is_sme].nunique())
    except Exception:
        pass

    # --- Headcount (Budget/Planned from Budget upload) ---
    budget_df = _first_non_empty_ts(sk, ["budget_headcount","budget_hc","headcount_budget","hc_budget"])
    budget_w  = _weekly_reduce(budget_df, value_candidates=("hc","headcount","value","count"), how="sum")
    for w in week_ids:
        b = float(budget_w.get(w, 0.0))
        if "metric" in hc.columns:
            hc.loc[hc["metric"]=="Budgeted HC (#)",            w] = b
            hc.loc[hc["metric"]=="Planned/Tactical HC (#)",    w] = b
            hc.loc[hc["metric"]=="Actual Agent HC (#)",        w] = actual_agent_hc
            hc.loc[hc["metric"]=="SME Billable HC (#)",        w] = sme_billable_hc

    # --- Attrition (from Upload) ---
    att_plan_w = _weekly_reduce(_first_non_empty_ts(sk, ["attrition_planned_hc","attrition_plan_hc","planned_attrition_hc"]),
                                value_candidates=("hc","headcount","value","count"), how="sum")
    att_act_w  = _weekly_reduce(_first_non_empty_ts(sk, ["attrition_actual_hc","attrition_actual","actual_attrition_hc"]),
                                value_candidates=("hc","headcount","value","count"), how="sum")
    att_pct_w  = _weekly_reduce(_first_non_empty_ts(sk, ["attrition_pct","attrition_percent","attrition%","attrition_rate"]),
                                value_candidates=("pct","percent","value"), how="mean")

    for w in week_ids:
        plan_hc = float(att_plan_w.get(w, 0.0))
        act_hc  = float(att_act_w.get(w, 0.0))
        pct     = att_pct_w.get(w, None)
        if pct is None:
            pct = 100.0 * (act_hc / actual_agent_hc) if actual_agent_hc > 0 else 0.0
        if "metric" in att.columns:
            att.loc[att["metric"]=="Planned Attrition HC (#)", w] = plan_hc
            att.loc[att["metric"]=="Actual Attrition HC (#)",  w] = act_hc
            att.loc[att["metric"]=="Attrition %",              w] = pct

    # -------- Shrinkage (RAW → plan) -----------------------------------------
    ooo_hours_w, io_hours_w, base_hours_w = {}, {}, {}

    def _week_key(s):
        ds = pd.to_datetime(s, errors="coerce")
        if isinstance(ds, pd.Series):
            monday = ds.dt.normalize() - pd.to_timedelta(ds.dt.weekday, unit="D")
            return monday.dt.date.astype(str)
        else:
            ds = pd.DatetimeIndex(ds)
            monday = ds.normalize() - pd.to_timedelta(ds.weekday, unit="D")
            return pd.Index(monday.date.astype(str))

    def _agg_weekly(date_idx, ooo_series, ino_series, base_series):
        wk = _week_key(date_idx)
        g = pd.DataFrame({"week": wk, "ooo": ooo_series, "ino": ino_series, "base": base_series}) \
                .groupby("week", as_index=False).sum()
        for _, r in g.iterrows():
            k = str(r["week"])
            ooo_hours_w[k]  = ooo_hours_w.get(k, 0.0)  + float(r["ooo"])
            io_hours_w[k]   = io_hours_w.get(k, 0.0)   + float(r["ino"])
            base_hours_w[k] = base_hours_w.get(k, 0.0) + float(r["base"])

    # ---- VOICE (this plan channel only) ----
    if ch_first.lower() == "voice":
        try:
            vraw = load_df("shrinkage_raw_voice")
        except Exception:
            vraw = None

        if isinstance(vraw, pd.DataFrame) and not vraw.empty:
            v = vraw.copy()
            L = {str(c).strip().lower(): c for c in v.columns}
            c_date = L.get("date")
            c_hours= L.get("hours") or L.get("duration_hours") or L.get("duration")
            c_state= L.get("superstate") or L.get("state")
            c_ba   = L.get("business area") or L.get("ba")
            c_sba  = L.get("sub business area") or L.get("sub_ba")
            c_ch   = L.get("channel")
            c_loc  = L.get("country") or L.get("location") or L.get("site") or L.get("city")

            mask = pd.Series(True, index=v.index)
            if c_ba  and p.get("vertical"): mask &= v[c_ba ].astype(str).str.strip().str.lower().eq(p["vertical"].strip().lower())
            if c_sba and p.get("sub_ba"):  mask &= v[c_sba].astype(str).str.strip().str.lower().eq(p["sub_ba"].strip().lower())
            if c_ch:                        mask &= v[c_ch ].astype(str).str.strip().str.lower().eq("voice")

            # tolerant location filter
            if c_loc and loc_first:
                loc_series = v[c_loc].astype(str).str.strip()
                loc_l = loc_series.str.lower()
                target = loc_first.strip().lower()
                has_target = loc_l.eq(target).any()
                has_real_locations = loc_l.ne("").any() and loc_l.ne("all").any()
                if has_target and has_real_locations:
                    mask &= loc_l.eq(target)

            v = v.loc[mask]

            if c_date and c_state and c_hours and not v.empty:
                pv = v.pivot_table(index=c_date, columns=c_state, values=c_hours, aggfunc="sum", fill_value=0.0)

                def col(name): return pv[name] if name in pv.columns else 0.0
                base = col("SC_INCLUDED_TIME")  # paid/included hours
                ooo  = col("SC_ABSENCE_TOTAL") + col("SC_HOLIDAY") + col("SC_A_Sick_Long_Term")
                ino  = col("SC_TRAINING_TOTAL") + col("SC_BREAKS") + col("SC_SYSTEM_EXCEPTION")

                idx_dates = pd.to_datetime(pv.index, errors="coerce")
                _agg_weekly(idx_dates, ooo, ino, base)

    # ---- BACK OFFICE (this plan channel only) ----
    if ch_first.lower() in ("back office", "bo"):
        try:
            braw = load_df("shrinkage_raw_backoffice")
        except Exception:
            braw = None

        if isinstance(braw, pd.DataFrame) and not braw.empty:
            b = braw.copy()
            L = {str(c).strip().lower(): c for c in b.columns}
            c_date = L.get("date")
            c_act  = L.get("activity")
            c_sec  = L.get("duration_seconds") or L.get("seconds") or L.get("duration")
            c_ba   = L.get("journey") or L.get("business area") or L.get("ba")
            c_sba  = L.get("sub_business_area") or L.get("sub business area") or L.get("sub_ba")
            c_brid = L.get("brid") or L.get("employee id") or L.get("employee_id")

            mask = pd.Series(True, index=b.index)
            if c_ba  and p.get("vertical"): mask &= b[c_ba ].astype(str).str.strip().str.lower().eq(p["vertical"].strip().lower())
            if c_sba and p.get("sub_ba"):  mask &= b[c_sba].astype(str).str.strip().str.lower().eq(p["sub_ba"].strip().lower())

            # Location via roster mapping (BRID → Country/Location)
            if loc_first and c_brid and isinstance(roster_df, pd.DataFrame) and not roster_df.empty:
                RL = {str(c).strip().lower(): c for c in roster_df.columns}
                rc_brid = RL.get("brid") or RL.get("employee id") or RL.get("employee_id")
                rc_loc  = RL.get("position location country") or RL.get("country") or RL.get("location") or RL.get("site")
                if rc_brid and rc_loc:
                    m = roster_df[[rc_brid, rc_loc]].copy()
                    m[rc_brid] = m[rc_brid].astype(str).str.strip()
                    m[rc_loc]  = m[rc_loc].astype(str).str.strip().str.lower()
                    mp = dict(zip(m[rc_brid], m[rc_loc]))
                    b["_loc_"] = b[c_brid].astype(str).str.strip().map(mp)
                    mask &= b["_loc_"].eq(loc_first.strip().lower())

            b = b.loc[mask]

            if c_date and c_act and c_sec and not b.empty:
                d = b[[c_date, c_act, c_sec]].copy()
                d[c_act] = d[c_act].astype(str).str.strip().str.lower()
                d[c_sec] = pd.to_numeric(d[c_sec], errors="coerce").fillna(0.0)
                d[c_date] = pd.to_datetime(d[c_date], errors="coerce").dt.date  # normalize to date

                def has(s): return d[c_act].str.contains(s, na=False)
                sec_div = d.loc[has("divert"), c_sec].groupby(d[c_date]).sum()
                sec_dow = d.loc[has("down"),   c_sec].groupby(d[c_date]).sum()
                sec_sc  = d.loc[has("staff complement"), c_sec].groupby(d[c_date]).sum()
                sec_fx  = d.loc[has("flex"),            c_sec].groupby(d[c_date]).sum()
                sec_ot  = d.loc[has("overtime") | d[c_act].eq("ot"), c_sec].groupby(d[c_date]).sum()
                sec_lend= d.loc[has("lend"),            c_sec].groupby(d[c_date]).sum()
                sec_borr= d.loc[has("borrow"),          c_sec].groupby(d[c_date]).sum()

                idx = pd.to_datetime(pd.Index(
                    set(sec_div.index) | set(sec_dow.index) | set(sec_sc.index) |
                    set(sec_fx.index)  | set(sec_ot.index)  | set(sec_lend.index) | set(sec_borr.index)
                ), errors="coerce").sort_values()

                def get(s): return s.reindex(idx, fill_value=0.0)
                num_sec = get(sec_div) + get(sec_dow)
                den_sec = (get(sec_sc) + get(sec_fx) + get(sec_ot) - get(sec_lend) + get(sec_borr)).clip(lower=0)

                # BO raw: all shrinkage treated as in-office in this dataset
                ooo = (0.0 * den_sec).astype(float) / 3600.0
                ino = num_sec.astype(float)         / 3600.0
                base= den_sec.astype(float)         / 3600.0

                _agg_weekly(idx, ooo, ino, base)

    # ---- Write weekly values into the Shrinkage grid for THIS plan only ----------
    for w in week_ids:
        if w not in shr.columns:
            shr[w] = np.nan
        shr[w] = pd.to_numeric(shr[w], errors="coerce").astype("float64")

    for w in week_ids:
        base = float(base_hours_w.get(w, 0.0))
        ooo  = float(ooo_hours_w.get(w, 0.0))
        ino  = float(io_hours_w.get(w, 0.0))
        ooo_pct = (100.0 * ooo / base) if base > 0 else 0.0
        ino_pct = (100.0 * ino / base) if base > 0 else 0.0
        ov_pct  = (100.0 * (ooo + ino) / base) if base > 0 else 0.0
        if "metric" in shr.columns:
            shr.loc[shr["metric"]=="OOO Shrink Hours (#)",       w] = ooo
            shr.loc[shr["metric"]=="Inoffice Shrink Hours (#)",  w] = ino
            shr.loc[shr["metric"]=="OOO Shrinkage %",            w] = ooo_pct
            shr.loc[shr["metric"]=="Inoffice Shrinkage %",       w] = ino_pct
            shr.loc[shr["metric"]=="Overall Shrinkage %",        w] = ov_pct

    # --- Budget vs Actual ---
    for w in week_ids:
        if w not in bva.columns:
            bva[w] = pd.Series(np.nan, index=bva.index, dtype="float64")
        elif not pd.api.types.is_float_dtype(bva[w].dtype):
            bva[w] = pd.to_numeric(bva[w], errors="coerce").astype("float64")
    for w in week_ids:
        bud = float(req_w_budgeted.get(w, 0.0))
        act = float(req_w_actual.get(w,   0.0))
        if "metric" in bva.columns:
            bva.loc[bva["metric"]=="Budgeted FTE (#)", w] = bud
            bva.loc[bva["metric"]=="Actual FTE (#)",   w] = act
            bva.loc[bva["metric"]=="Variance (#)",     w] = act - bud

    # --- Ratios ---
    planned_ratio = _parse_ratio_setting(
        settings.get("planned_tl_agent_ratio") or settings.get("tl_agent_ratio") or settings.get("tl_per_agent")
    )
    actual_ratio = 0.0
    try:
        if isinstance(roster_df, pd.DataFrame) and not roster_df.empty and "role" in roster_df.columns:
            r = roster_df.copy()
            r["role"] = r["role"].astype(str).str.strip().str.lower()
            tl = (r["role"] == "team leader").sum()
            ag = (r["role"] == "agent").sum()
            actual_ratio = (float(tl)/float(ag)) if ag > 0 else 0.0
    except Exception:
        pass
    for w in week_ids:
        if "metric" in rat.columns:
            rat.loc[rat["metric"]=="Planned TL/Agent Ratio", w] = planned_ratio
            rat.loc[rat["metric"]=="Actual TL/Agent Ratio",  w] = actual_ratio
            rat.loc[rat["metric"]=="Variance",               w] = actual_ratio - planned_ratio

    # ---------------- Projected Supply / Capacity / SL -------------------------
    def _row_as_dict(df, metric_name):
        if not isinstance(df, pd.DataFrame) or df.empty or "metric" not in df.columns:
            return {}
        m = df["metric"].astype(str).str.strip()
        if metric_name not in m.values:
            return {}
        row = df.loc[m==metric_name].iloc[0]
        return {w: float(pd.to_numeric(row.get(w), errors="coerce")) for w in week_ids}

    hc_plan   = _row_as_dict(hc,  "Planned/Tactical HC (#)")
    hc_actual = _row_as_dict(hc,  "Actual Agent HC (#)")
    att_act   = _row_as_dict(att, "Actual Attrition HC (#)")
    nh_act    = _row_as_dict(nh,  "Actual New Hire HC (#)") or _row_as_dict(nh, "Planned New Hire HC (#)")

    # weekly iterative: prev + NH - Attrition (start from Actual or Planned)
    projected_supply = {}
    prev = None
    for w in week_ids:
        start = prev
        if start is None:
            start = float(hc_actual.get(w, 0.0)) or float(hc_plan.get(w, 0.0))
        projected = max(start - float(att_act.get(w, 0.0)) + float(nh_act.get(w, 0.0)), 0.0)
        projected_supply[w] = projected
        prev = projected

    # ---- Erlang helpers ----
    import math
    def _erlang_c(A: float, N: int) -> float:
        if N <= 0: return 1.0
        if A <= 0: return 0.0
        if A >= N: return 1.0
        s = 0.0
        for k in range(N):
            s += (A**k) / math.factorial(k)
        last = (A**N)/math.factorial(N) * (N/(N-A))
        p0 = 1.0 / (s + last)
        return last * p0

    def _erlang_sl(calls_per_ivl: float, aht_sec: float, agents: float, asa_sec: int, ivl_sec: int) -> float:
        if aht_sec <= 0 or ivl_sec <= 0 or agents <= 0 or calls_per_ivl <= 0:
            return 0.0
        A = (calls_per_ivl * aht_sec) / ivl_sec
        pw = _erlang_c(A, int(math.floor(agents)))
        return max(0.0, min(1.0, 1.0 - pw * math.exp(-max(0.0, (agents - A)) * (asa_sec / max(1.0, aht_sec)))))

    def _erlang_calls_capacity(agents: float, aht_sec: float, asa_sec: int, ivl_sec: int, target_pct: float) -> float:
        """Max calls (or items) per interval meeting target SL via binary search."""
        if agents <= 0 or aht_sec <= 0 or ivl_sec <= 0:
            return 0.0
        target = float(target_pct)/100.0
        hi = max(1, int((agents * ivl_sec) / aht_sec))
        def sl_for(x): return _erlang_sl(x, aht_sec, agents, asa_sec, ivl_sec)
        lo = 0
        while sl_for(hi) >= target and hi < 10_000_000:
            lo = hi
            hi *= 2
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if sl_for(mid) >= target:
                lo = mid
            else:
                hi = mid - 1
        return float(lo)

    # --- Handling capacity (weekly) ---
    handling_capacity = {}
    for w in week_ids:
        agents_eff_raw = projected_supply.get(w, 0.0) * occ_frac
        agents_eff = max(1.0, float(agents_eff_raw))  # guard: Erlang needs >=1 agent

        if ch_first.lower() == "voice":
            aht = _first_positive(wk_aht_sut_actual.get(w), wk_aht_sut_forecast.get(w), s_target_aht, default=s_target_aht)
            aht = max(1.0, float(aht))
            n = weekly_voice_intervals.get(w)
            intervals = int(n) if isinstance(n, (int, np.integer)) and n > 0 else intervals_per_week_default
            calls_per_ivl = _erlang_calls_capacity(agents_eff, aht, sl_seconds, ivl_sec, sl_target_pct)
            handling_capacity[w] = calls_per_ivl * intervals
        else:
            sut = _first_positive(wk_aht_sut_actual.get(w), wk_aht_sut_forecast.get(w), s_target_sut, default=s_target_sut)
            sut = max(1.0, float(sut))
            items_per_ivl = _erlang_calls_capacity(agents_eff, sut, sl_seconds, ivl_sec, sl_target_pct)
            handling_capacity[w] = items_per_ivl * intervals_per_week_default

    # --- Projected Service Level (voice + non-voice) ---
    proj_sl = {}
    for w in week_ids:
        ch_l = ch_first.lower()
        if ch_l == "voice":
            weekly_load = float(weekly_demand_voice.get(w, 0.0))
            aht_sut = _first_positive(wk_aht_sut_actual.get(w), wk_aht_sut_forecast.get(w), s_target_aht, default=s_target_aht)
            n = weekly_voice_intervals.get(w)
            intervals = int(n) if isinstance(n, (int, np.integer)) and n > 0 else intervals_per_week_default
        else:
            weekly_load = float(weekly_demand_bo.get(w, 0.0))
            aht_sut = _first_positive(wk_aht_sut_actual.get(w), wk_aht_sut_forecast.get(w), s_target_sut, default=s_target_sut)
            intervals = intervals_per_week_default

        if weekly_load <= 0:
            proj_sl[w] = 0.0
            continue

        calls_per_ivl = weekly_load / float(max(1, intervals))
        agents_eff_raw = projected_supply.get(w, 0.0) * occ_frac
        agents_eff = max(1.0, float(agents_eff_raw))
        sl_frac = _erlang_sl(calls_per_ivl, max(1.0, float(aht_sut)), agents_eff, sl_seconds, ivl_sec)
        proj_sl[w] = 100.0 * sl_frac

    # ---------------------- Upper summary table -------------------------------
    upper_df = _blank_grid(spec["upper"], week_ids)
    if "FTE Required @ Forecast Volume" in spec["upper"]:
        for w in week_ids:
            upper_df.loc[upper_df["metric"]=="FTE Required @ Forecast Volume", w] = float(req_w_forecast.get(w, 0.0))
    if "FTE Required @ Actual Volume" in spec["upper"]:
        for w in week_ids:
            upper_df.loc[upper_df["metric"]=="FTE Required @ Actual Volume", w] = float(req_w_actual.get(w, 0.0))
    if "FTE Over/Under MTP Vs Actual" in spec["upper"]:
        for w in week_ids:
            mtp = float(req_w_forecast.get(w, 0.0)); act = float(req_w_actual.get(w, 0.0))
            upper_df.loc[upper_df["metric"]=="FTE Over/Under MTP Vs Actual", w] = mtp - act
    if "FTE Over/Under Tactical Vs Actual" in spec["upper"]:
        for w in week_ids:
            tac = float(req_w_tactical.get(w, 0.0)); act = float(req_w_actual.get(w, 0.0))
            upper_df.loc[upper_df["metric"]=="FTE Over/Under Tactical Vs Actual", w] = tac - act
    if "FTE Over/Under Budgeted Vs Actual" in spec["upper"]:
        for w in week_ids:
            bud = float(req_w_budgeted.get(w, 0.0)); act = float(req_w_actual.get(w, 0.0))
            upper_df.loc[upper_df["metric"]=="FTE Over/Under Budgeted Vs Actual", w] = bud - act
    if "Projected Supply HC" in spec["upper"]:
        for w in week_ids:
            upper_df.loc[upper_df["metric"]=="Projected Supply HC", w] = projected_supply.get(w, 0.0)
    if "Projected Handling Capacity (#)" in spec["upper"]:
        for w in week_ids:
            upper_df.loc[upper_df["metric"]=="Projected Handling Capacity (#)", w] = handling_capacity.get(w, 0.0)
    if "Projected Service Level" in spec["upper"]:
        for w in week_ids:
            upper_df.loc[upper_df["metric"]=="Projected Service Level", w] = proj_sl.get(w, 0.0)

    # ------------ Display rounding / formatting --------------------------------
    def _round_all(df):
        return _round_week_cols_int(df, week_ids)

    fw_to_use = _round_all(fw_to_use)
    hc        = _round_all(hc)
    att       = _round_all(att)
    trn       = _round_all(trn)
    rat       = _round_all(rat)
    seat      = _round_all(seat)
    bva       = _round_all(bva)
    nh        = _round_all(nh)

    # Shrinkage grid: hours as ints, percent rows as "NN%"
    def _format_shrinkage(df):
        if not isinstance(df, pd.DataFrame) or df.empty:
            return df
        out = df.copy()
        pct_rows = out["metric"].astype(str).str.contains("Shrinkage %", regex=False)
        hr_rows  = out["metric"].astype(str).str.contains("Hours (#)", regex=False)
        for w in week_ids:
            if w in out.columns:
                out[w] = out[w].astype(object)
        for w in week_ids:
            if w not in out.columns:
                continue
            out.loc[hr_rows, w] = (
                pd.to_numeric(out.loc[hr_rows, w], errors="coerce")
                .fillna(0)
                .round(0)
                .astype(int)
            )
            vals = pd.to_numeric(out.loc[pct_rows, w], errors="coerce").fillna(0)
            out.loc[pct_rows, w] = vals.round(0).astype(int).astype(str) + "%"
        return out

    shr_display = _format_shrinkage(shr)

    # Round the "upper" values: keep 1 decimal for SL, else int
    if isinstance(upper_df, pd.DataFrame) and not upper_df.empty:
        for w in week_ids:
            if w not in upper_df.columns:
                continue
            mask_sl = upper_df["metric"].eq("Projected Service Level")
            mask_not_sl = ~mask_sl
            upper_df.loc[mask_sl, w] = pd.to_numeric(upper_df.loc[mask_sl, w], errors="coerce").fillna(0.0).round(1)
            upper_df.loc[mask_not_sl, w] = pd.to_numeric(upper_df.loc[mask_not_sl, w], errors="coerce").fillna(0.0).round(0).astype(int)

    upper = dash_table.DataTable(
        id="tbl-upper",
        data=upper_df.to_dict("records"),
        columns=[{"name":"Metric","id":"metric","editable":False}] + [{"name":c["name"],"id":c["id"]} for c in fw_cols if c["id"]!="metric"],
        editable=False, style_as_list_view=True, style_table={"overflowX":"auto"}, style_header={"whiteSpace":"pre"},
    )

    return (
        upper,
        fw_to_use.to_dict("records"),
        hc.to_dict("records"),
        att.to_dict("records"), shr_display.to_dict("records"),
        trn.to_dict("records"), rat.to_dict("records"),
        seat.to_dict("records"), bva.to_dict("records"),
        nh.to_dict("records"),
        roster_df.to_dict("records"),
        bulk_df.to_dict("records"),
        notes_df.to_dict("records"),
    )
