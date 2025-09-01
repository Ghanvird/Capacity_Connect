from __future__ import annotations
from dash import html, dcc, dash_table, Output, Input, State, callback
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import dash
from app_instance import app
from common import _sbas_from_headcount, _bas_from_headcount, parse_upload, pretty_columns, lock_variance_cols, _all_sites, _sites_for_ba_location, _canon_scope, CHANNEL_LIST, load_timeseries, save_timeseries, _save_budget_hc_timeseries, _budget_voice_template, _budget_bo_template, _budget_normalize_voice, _budget_normalize_bo

@app.callback(
    Output("bud-site", "options"),
    Output("bud-site", "value"),
    Input("bud-ba", "value"),
    prevent_initial_call=False,
    )
def _bud_fill_site(ba):
    sites = _sites_for_ba_location(ba, None) if ba else _all_sites()
    opts = [{"label": s, "value": s} for s in (sites or [])]
    return opts, (sites[0] if sites else None)
#============= Tactical Upload =============#
# ====================== TEMPLATE DOWNLOADS ======================

@app.callback(
    Output("bud-ba", "options"),
    Output("bud-ba", "value"),
    Input("url-router", "pathname"),
    prevent_initial_call=False
)
def _bud_fill_ba(path):
    bas = _bas_from_headcount()  # from Headcount Update
    opts = [{"label": b, "value": b} for b in bas]
    return opts, (bas[0] if bas else None)

@app.callback(
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

@app.callback(
    Output("bud-channel", "options"),
    Output("bud-channel", "value"),
    Input("bud-subba", "value"),
    State("bud-channel", "value"),
    prevent_initial_call=False
)
def _bud_fill_channels(_subba, current):
    opts = [{"label": c, "value": c} for c in CHANNEL_LIST]
    val = current if current in CHANNEL_LIST else "Voice"
    return opts, val


# ---- Load existing budgets when scope changes ----

@app.callback(
    Output("tbl-bud-voice", "data", allow_duplicate=True),
    Output("tbl-bud-voice", "columns", allow_duplicate=True),
    Output("store-bud-voice", "data", allow_duplicate=True),
    Input("bud-ba", "value"), Input("bud-subba", "value"), Input("bud-channel", "value"), Input("bud-site", "value"),
    prevent_initial_call=True,
)
def load_voice_budget(ba, subba, channel, site):
    if not (ba and subba and channel and site):
        return [], [], None
    if (channel or "").strip().lower() != "voice":
        return [], [], None
    key4 = _canon_scope(ba, subba, "Voice", site)
    df = load_timeseries("voice_budget", key4)
    if df is None or df.empty:
        key3 = _canon_scope(ba, subba, "Voice") # legacy
        df = load_timeseries("voice_budget", key3)
    if df is None or df.empty:
        return [], [], None
    return df.to_dict("records"), pretty_columns(df), df.to_dict("records")

@app.callback(
    Output("tbl-bud-bo", "data", allow_duplicate=True),
    Output("tbl-bud-bo", "columns", allow_duplicate=True),
    Output("store-bud-bo", "data", allow_duplicate=True),
    Input("bud-ba", "value"), Input("bud-subba", "value"), Input("bud-channel", "value"), Input("bud-site", "value"),
    prevent_initial_call=True,
)
def load_bo_budget(ba, subba, channel, site):
    if not (ba and subba and channel and site):
        return [], [], None
    ch = (channel or "").strip().lower()
    if ch not in ("back office", "bo"):
        return [], [], None
    key4 = _canon_scope(ba, subba, "Back Office", site)
    df = load_timeseries("bo_budget", key4)
    if df is None or df.empty:
        key3 = _canon_scope(ba, subba, "Back Office") # legacy
        df = load_timeseries("bo_budget", key3)
    if df is None or df.empty:
        return [], [], None
    return df.to_dict("records"), pretty_columns(df), df.to_dict("records")

# ---- Download templates ----

@app.callback(Output("dl-bud-voice","data"),
          Input("btn-bud-voice-tmpl","n_clicks"),
          State("bud-voice-start","date"), State("bud-voice-weeks","value"),
          prevent_initial_call=True)
def dl_voice_tmpl(_n, start_date, weeks):
    df = _budget_voice_template(start_date, int(weeks or 8))
    return dcc.send_data_frame(df.to_csv, "voice_budget_template.csv", index=False)

@app.callback(Output("dl-bud-bo","data"),
          Input("btn-bud-bo-tmpl","n_clicks"),
          State("bud-bo-start","date"), State("bud-bo-weeks","value"),
          prevent_initial_call=True)
def dl_bo_tmpl(_n, start_date, weeks):
    df = _budget_bo_template(start_date, int(weeks or 8))
    return dcc.send_data_frame(df.to_csv, "bo_budget_template.csv", index=False)

# ---- Upload / normalize ----

@app.callback(
    Output("tbl-bud-voice", "data", allow_duplicate=True),
    Output("tbl-bud-voice", "columns", allow_duplicate=True),
    Output("store-bud-voice", "data", allow_duplicate=True),
    Input("up-bud-voice","contents"),
    State("up-bud-voice","filename"),
    prevent_initial_call=True
)
def up_voice(contents, filename):
    df = parse_upload(contents, filename)
    dff = _budget_normalize_voice(df)
    if dff.empty: return [], [], None
    return dff.to_dict("records"), lock_variance_cols(pretty_columns(dff)), dff.to_dict("records")

@app.callback(
    Output("tbl-bud-bo", "data", allow_duplicate=True),
    Output("tbl-bud-bo", "columns", allow_duplicate=True),
    Output("store-bud-bo", "data", allow_duplicate=True),
    Input("up-bud-bo","contents"),
    State("up-bud-bo","filename"),
    prevent_initial_call=True
)
def up_bo(contents, filename):
    df = parse_upload(contents, filename)
    dff = _budget_normalize_bo(df)
    if dff.empty: return [], [], None
    return dff.to_dict("records"), lock_variance_cols(pretty_columns(dff)), dff.to_dict("records")

# ---- Save budgets ----

@app.callback(
    Output("msg-save-bud-voice", "children"),
    Input("btn-save-bud-voice", "n_clicks"),
    State("bud-ba", "value"), State("bud-subba", "value"), State("bud-channel", "value"), State("bud-site", "value"),
    State("store-bud-voice", "data"),
    prevent_initial_call=True,
)
def save_voice_budget(_n, ba, subba, channel, site, store):
    if not (ba and subba and channel and site):
        return "Pick BA, Sub BA, Channel and Site first."
    dff = pd.DataFrame(store or [])
    if dff.empty:
        return "Nothing to save."
    key = _canon_scope(ba, subba, "Voice", site)
    save_timeseries("voice_budget", key, dff)
    _save_budget_hc_timeseries(key, dff)
    aht = dff[["week", "budget_aht_sec"]].rename(columns={"budget_aht_sec": "aht_sec"})
    save_timeseries("voice_planned_aht", key, aht)
    return f"Saved Voice budget for {key} ✓ ({len(dff)} rows)."

@app.callback(
    Output("msg-save-bud-bo", "children"),
    Input("btn-save-bud-bo", "n_clicks"),
    State("bud-ba", "value"), State("bud-subba", "value"), State("bud-channel", "value"), State("bud-site", "value"),
    State("store-bud-bo", "data"),
    prevent_initial_call=True,
)
def save_bo_budget(_n, ba, subba, channel, site, store):
    if not (ba and subba and channel and site):
        return "Pick BA, Sub BA, Channel and Site first."
    dff = pd.DataFrame(store or [])
    if dff.empty:
        return "Nothing to save."
    key = _canon_scope(ba, subba, "Back Office", site)
    save_timeseries("bo_budget", key, dff)
    _save_budget_hc_timeseries(key, dff)
    sut = dff[["week", "budget_sut_sec"]].rename(columns={"budget_sut_sec": "sut_sec"})
    save_timeseries("bo_planned_sut", key, sut)
    return f"Saved Back Office budget for {key} ✓ ({len(dff)} rows)."

# ---------------------- Download templates (new clubbed) ----------------------
