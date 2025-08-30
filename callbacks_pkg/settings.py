from __future__ import annotations
from dash import html, dcc, dash_table, Output, Input, State, callback
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import datetime as dt
from app_instance import app
from dash.exceptions import PreventUpdate
from cap_store import save_defaults, save_scoped_settings, save_timeseries
from common import _bas_from_headcount, _bo_tactical_canon, _read_upload_to_df, _preview_cols_data, _coerce_time, _minutes_to_seconds, _sbas_from_headcount, _sites_for_ba_location, _voice_tactical_canon, headcount_template_df, parse_upload, pretty_columns, voice_forecast_template_df, voice_actual_template_df, bo_forecast_template_df, bo_actual_template_df, sidebar_component, _canon_scope, _all_locations, _lobs_for_ba_sba, resolve_settings, load_defaults, DEFAULT_SETTINGS

@app.callback(
    Output("dl-voice-forecast-tmpl1", "data"),
    Input("btn-dl-voice-forecast-tmpl1", "n_clicks"),
    prevent_initial_call=True
)
def dl_voice_tactical_template(n):
    if not n: raise PreventUpdate
    df = pd.DataFrame({
        "date": ["2025-07-07","2025-07-07"],
        "interval": ["09:00","09:30"],  # 30-min
        "volume": [120, 115],
        "aht_sec": [360, 355],
    })
    return dcc.send_data_frame(df.to_csv, "voice_tactical_template.csv", index=False)

@app.callback(
    Output("dl-bo-forecast-tmpl1", "data"),
    Input("btn-dl-bo-forecast-tmpl1", "n_clicks"),
    prevent_initial_call=True
)
def dl_bo_tactical_template(n):
    if not n: raise PreventUpdate
    df = pd.DataFrame({
        "date": ["2025-07-07","2025-07-08"],
        "items": [5000, 5200],
        "sut_sec": [540, 540],
    })
    return dcc.send_data_frame(df.to_csv, "bo_tactical_template.csv", index=False)

# ====================== PREVIEW (on upload) ======================

@app.callback(
    Output("tbl-voice-forecast1", "columns"),
    Output("tbl-voice-forecast1", "data"),
    Output("voice-forecast-msg1", "children", allow_duplicate=True),
    Input("up-voice-forecast1", "contents"),
    State("up-voice-forecast1", "filename"),
    prevent_initial_call=True
)
def preview_voice_tactical(contents, filename):
    if not contents: raise PreventUpdate
    df = _read_upload_to_df(contents, filename)
    cols, data = _preview_cols_data(df)
    return cols, data, f"Loaded {len(df)} rows from {filename or 'upload'}."

@app.callback(
    Output("tbl-bo-forecast1", "columns"),
    Output("tbl-bo-forecast1", "data"),
    Output("bo-forecast-msg1", "children", allow_duplicate=True),
    Input("up-bo-forecast1", "contents"),
    State("up-bo-forecast1", "filename"),
    prevent_initial_call=True
)
def preview_bo_tactical(contents, filename):
    if not contents: raise PreventUpdate
    df = _read_upload_to_df(contents, filename)
    cols, data = _preview_cols_data(df)
    return cols, data, f"Loaded {len(df)} rows from {filename or 'upload'}."

# ====================== SAVE ======================

@app.callback(
    Output("voice-forecast-msg1", "children", allow_duplicate=True),
    Input("btn-save-voice-forecast1", "n_clicks"),
    State("up-voice-forecast1", "contents"),
    State("up-voice-forecast1", "filename"),
    State("set-ba", "value"),
    State("set-subba", "value"),
    State("set-channel", "value"),
    State("set-site-hier", "value"),
    prevent_initial_call=True,
)
def save_voice_tactical(n, contents, filename, ba, subba, channel, site):
    if not n: raise PreventUpdate
    if not contents: return "No file uploaded."
    if not (ba and subba and channel and site):
        return "Pick BA, Sub BA, Channel and Site first."
    scope_key = _canon_scope(ba, subba, channel, site)
    raw = _read_upload_to_df(contents, filename)
    vol_df, aht_df, dbg = _voice_tactical_canon(raw)
    save_timeseries("voice_tactical_volume", scope_key, vol_df)
    if not aht_df.empty:
        save_timeseries("voice_tactical_aht", scope_key, aht_df)
    return f"Saved voice_tactical_volume ({len(vol_df)}) and voice_tactical_aht ({len(aht_df)}) for scope {scope_key}. {dbg}"

@callback(
    Output("bo-forecast-msg1", "children", allow_duplicate=True),
    Input("btn-save-bo-forecast1", "n_clicks"),
    State("up-bo-forecast1", "contents"),
    State("up-bo-forecast1", "filename"),
    State("set-ba", "value"),
    State("set-subba", "value"),
    State("set-channel", "value"),
    State("set-site-hier", "value"),
    prevent_initial_call=True,
)
def save_bo_tactical(n, contents, filename, ba, subba, channel, site):
    if not n: raise PreventUpdate
    if not contents: return "No file uploaded."
    if not (ba and subba and channel and site):
        return "Pick BA, Sub BA, Channel and Site first."
    scope_key = _canon_scope(ba, subba, channel, site)
    raw = _read_upload_to_df(contents, filename)
    vol_df, sut_df, dbg = _bo_tactical_canon(raw)
    save_timeseries("bo_tactical_volume", scope_key, vol_df)
    if not sut_df.empty:
        save_timeseries("bo_tactical_sut", scope_key, sut_df)
    return f"Saved bo_tactical_volume ({len(vol_df)}) and bo_tactical_sut ({len(sut_df)}) for scope {scope_key}. {dbg}"


# ---- Scope option chaining ------------------------------------------------------------------------

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
        dff["aht_sec"] = 300.0  # sensible default

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
def _scope_guard(scope, ba, sba, ch, site):
    if scope != "hier":
        return False, "Switch scope to Business Area ▶ Sub Business Area ▶ Channel ▶ Site."
    if not (ba and sba and ch and site):
        return False, "Pick BA, Sub BA, Channel and Site first."
    return True, ""

@callback(Output("voice-forecast-msg","children", allow_duplicate=True),
          Input("btn-save-voice-forecast","n_clicks"),
          State("tbl-voice-forecast","data"),
          State("set-scope","value"), State("set-ba","value"), State("set-subba","value"),
          State("set-channel","value"), State("set-site-hier","value"),
          prevent_initial_call=True)
def save_voice_forecast(_n, preview_rows, scope, ba, sba, ch, site):
    ok, msg = _scope_guard(scope, ba, sba, ch, site)
    if not ok: return msg
    sk = _canon_scope(ba, sba, ch, site)
    df = pd.DataFrame(preview_rows or [])
    if df.empty or not {'date','interval','volume','aht_sec'}.issubset(df.columns):
        return 'Missing required columns for Voice (date/interval/volume/aht_sec)'
    save_timeseries('voice_forecast_volume', sk, df[['date','interval','volume']].copy())
    save_timeseries('voice_forecast_aht',    sk, df[['date','interval','aht_sec']].copy())
    return f"Saved VOICE forecast ({len(df)}) for {sk}"

@callback(Output("voice-actual-msg","children", allow_duplicate=True),
          Input("btn-save-voice-actual","n_clicks"),
          State("tbl-voice-actual","data"),
          State("set-scope","value"), State("set-ba","value"), State("set-subba","value"),
          State("set-channel","value"), State("set-site-hier","value"),
          prevent_initial_call=True)
def save_voice_actual(_n, preview_rows, scope, ba, sba, ch, site):
    ok, msg = _scope_guard(scope, ba, sba, ch, site)
    if not ok: return msg
    sk = _canon_scope(ba, sba, ch, site)
    df = pd.DataFrame(preview_rows or [])
    if df.empty or not {'date','interval','volume','aht_sec'}.issubset(df.columns):
        return 'Missing required columns for Voice (date/interval/volume/aht_sec)'
    save_timeseries('voice_actual_volume', sk, df[['date','interval','volume']].copy())
    save_timeseries('voice_actual_aht',    sk, df[['date','interval','aht_sec']].copy())
    return f"Saved VOICE actual ({len(df)}) for {sk}"

@callback(Output("bo-forecast-msg","children", allow_duplicate=True),
          Input("btn-save-bo-forecast","n_clicks"),
          State("tbl-bo-forecast","data"),
          State("set-scope","value"), State("set-ba","value"), State("set-subba","value"),
          State("set-channel","value"), State("set-site-hier","value"),
          prevent_initial_call=True)
def save_bo_forecast(_n, preview_rows, scope, ba, sba, ch, site):
    ok, msg = _scope_guard(scope, ba, sba, ch, site)
    if not ok: return msg
    sk = _canon_scope(ba, sba, ch, site)
    df = pd.DataFrame(preview_rows or [])
    if df.empty or not {'date','items','sut_sec'}.issubset(df.columns):
        return 'Missing required columns (date/items/sut_sec)'
    save_timeseries('bo_forecast_volume', sk, df[['date','items']].rename(columns={'items':'volume'}))
    save_timeseries('bo_forecast_sut',    sk, df[['date','sut_sec']].copy())
    return f"Saved BO forecast ({len(df)}) for {sk}"

@callback(Output("bo-actual-msg","children", allow_duplicate=True),
          Input("btn-save-bo-actual","n_clicks"),
          State("tbl-bo-actual","data"),
          State("set-scope","value"), State("set-ba","value"), State("set-subba","value"),
          State("set-channel","value"), State("set-site-hier","value"),
          prevent_initial_call=True)
def save_bo_actual(_n, preview_rows, scope, ba, sba, ch, site):
    ok, msg = _scope_guard(scope, ba, sba, ch, site)
    if not ok: return msg
    sk = _canon_scope(ba, sba, ch, site)
    df = pd.DataFrame(preview_rows or [])
    if df.empty or not {'date','items','sut_sec'}.issubset(df.columns):
        return 'Missing required columns (date/items/sut_sec)'
    save_timeseries('bo_actual_volume', sk, df[['date','items']].rename(columns={'items':'volume'}))
    save_timeseries('bo_actual_sut',    sk, df[['date','sut_sec']].copy())
    return f"Saved BO actual ({len(df)}) for {sk}"

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

@callback(
    Output("row-location", "style"),
    Output("row-hier", "style"),
    Input("set-scope", "value"),
)
def _toggle_scope_rows(scope):
    if scope == "location":
        return ({"display":"flex"}, {"display":"none"})
    if scope == "hier":
        return ({"display":"none"}, {"display":"flex"})
    return ({"display":"none"}, {"display":"none"})

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
    bas  = _bas_from_headcount()
    locs = _all_locations()  # from Position Location Country
    ba_val  = bas[0]  if bas  else None
    loc_val = locs[0] if locs else None
    return (
        [{"label": b, "value": b} for b in bas], ba_val,
        [{"label": l, "value": l} for l in locs], loc_val
    )

@callback(
    Output("set-subba","options", allow_duplicate=True),
    Output("set-subba","value", allow_duplicate=True),
    Input("set-ba","value"),
    prevent_initial_call=True,
)
def settings_fill_sba(ba):
    sbas = _sbas_from_headcount(ba) if ba else []
    return [{"label": s, "value": s} for s in sbas], (sbas[0] if sbas else None)

@callback(
    Output("set-site-hier","options"),
    Output("set-site-hier","value"),
    Input("set-ba","value"),
    Input("set-location","value"),
    prevent_initial_call=False
)
def settings_fill_site_hier(ba, location):
    sites = _sites_for_ba_location(ba, location) if ba else []
    return [{"label": s, "value": s} for s in sites], (sites[0] if sites else None)

@callback(
    Output("set-lob","options"),
    Output("set-lob","value"),
    Input("set-ba","value"), Input("set-subba","value")
)
def settings_fill_lob(ba, sba):
    lobs = _lobs_for_ba_sba(ba, sba) if (ba and sba) else []
    return [{"label": l, "value": l} for l in lobs], (lobs[0] if lobs else None)

@callback(
    Output("set-interval","value"),
    Output("set-hours","value"),
    Output("set-shrink","value"),
    Output("set-sl","value"),
    Output("set-slsec","value"),
    Output("set-occ","value"),
    Output("set-utilbo","value"),
    Output("set-utilob","value"),
    Output("set-bo-model","value"),
    Output("set-bo-tat","value"),
    Output("set-bo-wd","value"),
    Output("set-bo-hpd","value"),
    Output("set-bo-shrink","value"),
    Output("settings-scope-note","children"),
    Input("set-scope","value"),
    Input("set-ba","value"),
    Input("set-subba","value"),
    Input("set-lob","value"),
    Input("set-location","value"),
    Input("set-site-hier","value"),
    prevent_initial_call=False
)
def load_for_scope(scope, ba, subba, channel, location_only, site_hier):
    # most specific first
    if scope == "hier" and ba and subba and channel and site_hier:
        s = resolve_settings(ba=ba, subba=subba, lob=channel, location=site_hier)
        note = f"Scope: {ba} › {subba} › {channel} › {site_hier}"
    elif scope == "location" and location_only:
        s = resolve_settings(location=location_only)
        note = f"Scope: Location = {location_only}"
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
        (s.get("bo_capacity_model","tat")),
        float(s.get("bo_tat_days", 5)),
        int(s.get("bo_workdays_per_week", 5)),
        float(s.get("bo_hours_per_day", s.get("hours_per_fte", 8))),
        float(s.get("bo_shrinkage_pct", s.get("shrinkage_pct", 0))) * 100.0,
        note,
    )

@callback(
    Output("settings-save-msg","children"),
    Input("btn-save-settings","n_clicks"),
    State("set-scope","value"),
    State("set-ba","value"), State("set-subba","value"),
    State("set-lob","value"),
    State("set-location","value"), State("set-site-hier","value"),
    State("set-interval","value"), State("set-hours","value"),
    State("set-shrink","value"), State("set-sl","value"), State("set-slsec","value"),
    State("set-occ","value"), State("set-utilbo","value"), State("set-utilob","value"),
    State("set-bo-model","value"), State("set-bo-tat","value"),
    State("set-bo-wd","value"), State("set-bo-hpd","value"), State("set-bo-shrink","value"),
    prevent_initial_call=True
)
def save_scoped(n, scope, ba, subba, channel, location_only, site_hier,
                ivl, hrs, shr, sl, slsec, occ, utilbo, utilob,
                bo_model, bo_tat, bo_wd, bo_hpd, bo_shrink):
    if not n:
        raise PreventUpdate

    payload = {
        "interval_minutes":     int(ivl or 30),
        "interval_sec":         int(ivl or 30) * 60,
        "hours_per_fte":        float(hrs or 8.0),
        "shrinkage_pct":        float(shr or 0) / 100.0,
        "target_sl":            float(sl or 80) / 100.0,
        "sl_seconds":           int(slsec or 20),
        "occupancy_cap_voice":  float(occ or 85) / 100.0,
        "util_bo":              float(utilbo or 85) / 100.0,
        "util_ob":              float(utilob or 85) / 100.0,
        "bo_capacity_model":    (bo_model or "tat").lower(),
        "bo_tat_days":          float(bo_tat or 5),
        "bo_workdays_per_week": int(bo_wd or 5),
        "bo_hours_per_day":     float(bo_hpd or (hrs or 8.0)),
        "bo_shrinkage_pct":     float(bo_shrink or 0) / 100.0,
    }

    if scope == "global":
        save_defaults(payload)
        return "Saved global defaults ✓"

    if scope == "location":
        if not location_only:
            return "Select a location to save."
        save_scoped_settings("location", location_only, payload)
        return f"Saved for location: {location_only} ✓"

    if scope == "hier":
        if not (ba and subba and channel and site_hier):
            return "Pick BA/SubBA/Channel/Site to save."
        key = f"{ba}|{subba}|{channel}|{site_hier}"
        save_scoped_settings("hier", key, payload)
        return f"Saved for {ba} › {subba} › {channel} › {site_hier} ✓"

    return ""

@callback(Output("set-location-hint","children"), Input("set-location","value"))
def _loc_hint(val):
    return "" if not val else f"Using Position Location Country = {val}"


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

@app.callback(
    Output("tbl-headcount-preview","data"),
    Output("tbl-headcount-preview","columns"),
    Output("hc-msg", "children", allow_duplicate=True),
    Input("up-headcount","contents"),
    State("up-headcount","filename"),
    prevent_initial_call=True
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

@app.callback(
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

@app.callback(Output("dl-hc-template","data"),
          Input("btn-dl-hc-template","n_clicks"),
          prevent_initial_call=True)
def dl_hc_tmpl(_n):
    return dcc.send_data_frame(headcount_template_df().to_csv, "headcount_template.csv", index=False)

# === Shrinkage RAW: Download templates ===
