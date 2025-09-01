from __future__ import annotations
from dash import html, dcc, dash_table, Output, Input, State, callback
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import dash
from app_instance import app
import datetime as dt
from common import *  # noqa

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
        return "Saved", fig
    return "Saved (empty)", {}

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
    Output("attr-save-msg","children", allow_duplicate=True),
    Output("fig-attr","figure"),
    Output("shr-msg-timer","disabled", allow_duplicate=True),
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
            return "Saved (weekly empty) + raw", {}, False
            return "Saved (weekly empty) + raw", {}, False
        return "Saved (empty)", {}, False

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
            raw_msg = " + raw"

    fig = px.line(df, x="week", y="attrition_pct",
                  color=("program" if "program" in df.columns else None),
                  markers=True, title="Attrition %")
    return f"Saved{raw_msg}", fig, False
    return f"Saved{raw_msg}", fig, False

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
    Output("bo-shr-save-msg","children", allow_duplicate=True),
    Output("shr-msg-timer","disabled", allow_duplicate=True),
    Input("btn-save-shr-bo-raw","n_clicks"),
    State("bo-shr-raw-store","data"),
    prevent_initial_call=True
)
def save_shr_bo(_n, store):
    dff = pd.DataFrame(store or [])
    if dff.empty: return "Nothing to save.", False
    # Save raw
    save_df("shrinkage_raw_backoffice", dff)
    # Also compute + save weekly % (to keep your existing shrinkage chart working)
    daily = summarize_shrinkage_bo(dff)
    weekly = weekly_shrinkage_from_bo_summary(daily)
    base = load_shrinkage()
    merged = pd.concat([base, weekly], ignore_index=True) if isinstance(base, pd.DataFrame) and not base.empty else weekly
    save_shrinkage(merged)
    save_shrinkage(merged)
    return f"Saved Back Office shrinkage (raw rows: {len(dff)}, weekly points: {len(weekly)})", False
    return f"Saved Back Office shrinkage (raw rows: {len(dff)}, weekly points: {len(weekly)})", False
    return f"Saved Back Office shrinkage (raw rows: {len(dff)}, weekly points: {len(weekly)})", False
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
    Output("voice-shr-save-msg","children", allow_duplicate=True),
    Output("shr-msg-timer","disabled", allow_duplicate=True),
    Input("btn-save-shr-voice-raw","n_clicks"),
    State("voice-shr-raw-store","data"),
    prevent_initial_call=True
)
def save_shr_voice(_n, store):
    dff = pd.DataFrame(store or [])
    if dff.empty: return "Nothing to save.", False
    # Save raw
    save_df("shrinkage_raw_voice", dff)
    # Weekly % into main shrinkage series (program = Business Area)
    daily = summarize_shrinkage_voice(dff)
    weekly = weekly_shrinkage_from_voice_summary(daily)
    base = load_shrinkage()
    merged = pd.concat([base, weekly], ignore_index=True) if isinstance(base, pd.DataFrame) and not base.empty else weekly
    save_shrinkage(merged)
    save_shrinkage(merged)
    return f"Saved Voice shrinkage (raw rows: {len(dff)}, weekly points: {len(weekly)})", False
# --- Auto-dismiss shrink page messages ---
@callback(
    Output("shr-msg-timer", "disabled", allow_duplicate=True),
    Input("bo-shr-save-msg", "children"),
    Input("voice-shr-save-msg", "children"),
    Input("attr-save-msg", "children"),
    prevent_initial_call=True
)
def _arm_shrink_timer(m1, m2, m3):
    msg = "".join([str(m or "") for m in (m1, m2, m3)]).strip()
    return False if msg else dash.no_update

@callback(
    Output("bo-shr-save-msg", "children", allow_duplicate=True),
    Output("voice-shr-save-msg", "children", allow_duplicate=True),
    Output("attr-save-msg", "children", allow_duplicate=True),
    Output("shr-msg-timer", "disabled", allow_duplicate=True),
    Input("shr-msg-timer", "n_intervals"),
    prevent_initial_call=True
)
def _clear_shrink_msgs(_n):
    return "", "", "", True
