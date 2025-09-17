import ast
import calendar
import json
import math
import re
import dash
import numpy as np
import pandas as pd
import datetime as dt
from cap_db import load_df
from cap_store import load_roster_long, resolve_settings
from capacity_core import required_fte_daily
from plan_detail._calc import _load_roster_normalized, _nh_effective_count 
from plan_detail._common import _assemble_bo, _assemble_chat, _assemble_ob, _assemble_voice, _blank_grid, _canon_scope, _first_non_empty_ts, _learning_curve_for_week, _load_or_blank, _load_or_empty_bulk_files, _load_or_empty_notes, _load_ts_with_fallback, _parse_ratio_setting
from plan_store import get_plan
from dash import dash_table

def _get_fw_value(fw_df, metric, col_id, default=0.0):
    """
    Safely get a single numeric value from the FW grid:
    returns float or `default` if missing/NaN.
    """
    try:
        if not isinstance(fw_df, pd.DataFrame) or fw_df.empty or col_id not in fw_df.columns:
            return default
        mask = fw_df["metric"].astype(str).str.strip().eq(metric)
        if not mask.any():
            return default
        ser = pd.to_numeric(fw_df.loc[mask, col_id], errors="coerce").dropna()
        return float(ser.iloc[0]) if not ser.empty else default
    except Exception:
        return default


def _fill_tables_fixed_monthly(ptype, pid, fw_cols, _tick, whatif=None):
    # ---- guards ----
    if not (pid and fw_cols):
        raise dash.exceptions.PreventUpdate

    # calendar columns (YYYY-MM-01 month ids coming from UI)
    month_ids = [c["id"] for c in fw_cols if c.get("id") != "metric"]

    # ---- read persisted What-If ----
    wf_start = ""
    wf_end   = ""
    wf_ovr   = {}
    try:
        wf_df = load_df(f"plan_{pid}_whatif")
        if isinstance(wf_df, pd.DataFrame) and not wf_df.empty:
            last = wf_df.tail(1).iloc[0]
            wf_start = str(last.get("start_week") or "").strip()  # keep window semantics
            wf_end   = str(last.get("end_week")   or "").strip()
            raw = last.get("overrides")
            if isinstance(raw, dict):
                wf_ovr = raw
            elif isinstance(raw, str) and raw.strip():
                try:
                    wf_ovr = json.loads(raw)
                except Exception:
                    try:
                        wf_ovr = ast.literal_eval(raw)
                    except Exception:
                        wf_ovr = {}
    except Exception:
        wf_start, wf_end, wf_ovr = "", "", {}

    # merge persisted overrides into live param
    whatif = dict(whatif or {})
    if isinstance(wf_ovr, dict):
        whatif.update(wf_ovr)

    # extract simple dials (with safe defaults)
    def _f(x, d=0.0):
        try: return float(x)
        except Exception: return d
    aht_delta    = _f(whatif.get("aht_delta"),    0.0)   # %
    shrink_delta = _f(whatif.get("shrink_delta"), 0.0)   # %
    attr_delta   = _f(whatif.get("attr_delta"),   0.0)   # HC
    vol_delta    = _f(whatif.get("vol_delta"),    0.0)   # %
    occ_override = whatif.get("occupancy_pct", None)
    backlog_carryover = bool(whatif.get("backlog_carryover", True))

    # per-week Nest/SDA dials (windowed) — we apply to months by date window as well
    _nest_login_w = dict((whatif.get("nesting_login_pct") or {}))
    _nest_ahtm_w  = dict((whatif.get("nesting_aht_multiplier") or {}))

    # helper: use the persisted window as an “active future” flag using month ids too
    def _wf_active_month(m):
        # treat wf_start/wf_end as weeks; if not provided, all months are active
        return True if (not wf_start and not wf_end) else True  # keep permissive for month view

    # helpers for nest overrides (applied when month is in active window)
    def _ovr_login_frac_m(m):
        # allow monthly override via the same mapping if keys match, else None
        v = _nest_login_w.get(m)
        if v in (None, "") or not _wf_active_month(m): return None
        try:
            x = float(v);  x = x/100.0 if x > 1.0 else x
            return max(0.0, min(1.0, x))
        except Exception:
            return None

    def _ovr_aht_mult_m(m):
        v = _nest_ahtm_w.get(m)
        if v in (None, "") or not _wf_active_month(m): return None
        try:
            mlt = float(v)
            return max(0.1, mlt)
        except Exception:
            return None

    # ---- scope, plan, settings ----
    p = get_plan(pid) or {}
    ch_first = (p.get("channel") or "").split(",")[0].strip()
    sk = _canon_scope(
        p.get("vertical"),
        p.get("sub_ba"),
        ch_first,
        (p.get("site") or p.get("location") or p.get("country") or "").strip(),
    )
    loc_first = (p.get("location") or p.get("country") or p.get("site") or "").strip()
    settings = resolve_settings(ba=p.get("vertical"), subba=p.get("sub_ba"), lob=ch_first)
    try:
        lc_ovr_df = load_df(f"plan_{pid}_lc_overrides")
    except Exception:
        lc_ovr_df = None

    def _lc_with_wf_m(lc_dict, m_id):
        out = dict(lc_dict or {})
        p_ = _ovr_login_frac_m(m_id)
        m_ = _ovr_aht_mult_m(m_id)
        if p_ is not None:
            L = out.get("nesting_prod_pct") or [50,60,70,80]
            out["nesting_prod_pct"] = [float(p_ * 100.0)] * len(L)
        if m_ is not None:
            uplift = (float(m_) - 1.0) * 100.0
            L = out.get("nesting_aht_uplift_pct") or [100,90,80,70]
            out["nesting_aht_uplift_pct"] = [float(uplift)] * len(L)
        return out

    # ---- SLA/AHT/SUT defaults ----
    s_target_aht = float(settings.get("target_aht", settings.get("budgeted_aht", 300)) or 300)
    s_budget_aht = float(settings.get("budgeted_aht", settings.get("target_aht", s_target_aht)) or s_target_aht)
    s_target_sut = float(settings.get("target_sut", settings.get("budgeted_sut", 600)) or 600)
    s_budget_sut = float(settings.get("budgeted_sut", settings.get("target_sut", s_target_sut)) or s_target_sut)
    sl_seconds   = int(settings.get("sl_seconds", 20) or 20)

    planned_aht_df = _load_ts_with_fallback("voice_planned_aht", sk)
    planned_sut_df = _load_ts_with_fallback("bo_planned_sut",   sk)

    # util for month id
    def _mid(s):
        return pd.to_datetime(s, errors="coerce").dt.to_period("M").dt.to_timestamp().dt.normalize().dt.date.astype(str)

    # monthly dict from ts (accepts either week or date columns)
    def _ts_month_dict(df: pd.DataFrame, val_candidates: list[str]) -> dict:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return {}
        d = df.copy()
        if "week" in d.columns:
            d["month"] = _mid(d["week"])
        elif "date" in d.columns:
            d["month"] = _mid(d["date"])
        elif "month" in d.columns:
            d["month"] = _mid(d["month"])
        else:
            return {}
        low = {c.lower(): c for c in d.columns}
        vcol = None
        for c in val_candidates:
            vcol = low.get(c.lower())
            if vcol:
                break
        if not vcol:
            return {}
        d[vcol] = pd.to_numeric(d[vcol], errors="coerce")
        g = d.dropna(subset=["month", vcol]).groupby("month", as_index=True)[vcol].mean()
        return g.astype(float).to_dict()

    planned_aht_m = _ts_month_dict(planned_aht_df, ["aht_sec","sut_sec","aht","avg_aht"])
    planned_sut_m = _ts_month_dict(planned_sut_df, ["sut_sec","aht_sec","sut","avg_sut"])

    sl_target_pct = None
    for k in ("sl_target_pct","service_level_target","sl_target","sla_target_pct","sla_target","target_sl"):
        v = settings.get(k)
        if v not in (None, ""):
            try:
                x = float(str(v).replace("%",""))
                sl_target_pct = x * 100.0 if x <= 1.0 else x
            except Exception:
                pass
            break
    if sl_target_pct is None:
        sl_target_pct = 80.0

    # ---- assemble time series ----
    vF = _assemble_voice(sk, "forecast"); vA = _assemble_voice(sk, "actual");  vT = _assemble_voice(sk, "tactical")
    bF = _assemble_bo(sk,   "forecast");  bA = _assemble_bo(sk,   "actual");   bT = _assemble_bo(sk,   "tactical")
    oF = _assemble_ob(sk,   "forecast");  oA = _assemble_ob(sk,   "actual");   oT = _assemble_ob(sk,   "tactical")
    cF = _assemble_chat(sk, "forecast");  cA = _assemble_chat(sk, "actual");   cT = _assemble_chat(sk, "tactical")

    use_voice_for_req = vA if isinstance(vA, pd.DataFrame) and not vA.empty else vF
    use_bo_for_req    = bA if isinstance(bA, pd.DataFrame) and not bA.empty else bF

    # ---- FW grid shell (same spec as weekly) ----
    spec = (lambda k: {
        "fw": (["Forecast","Tactical Forecast","Actual Volume","Budgeted AHT/SUT","Forecast AHT/SUT","Actual AHT/SUT","Occupancy","Overtime Hours (#)","Backlog (Items)"]
               if (k or "").strip().lower().startswith("volume") else
               ["Billable Hours","AHT/SUT","Shrinkage","Training"] if (k or "").strip().lower().startswith("billable hours") else ["Billable Txns","AHT/SUT","Efficiency","Shrinkage"] if (k or "").strip().lower().startswith("fte based billable") else
               ["Billable FTE Required","Shrinkage","Training"]),
        "upper": (["FTE Required @ Forecast Volume","FTE Required @ Actual Volume","FTE Over/Under MTP Vs Actual","FTE Over/Under Tactical Vs Actual","FTE Over/Under Budgeted Vs Actual","Projected Supply HC","Projected Handling Capacity (#)","Projected Service Level"]
                  if (k or "").strip().lower().startswith("volume") else
                  ["Billable FTE Required (#)","Headcount Required With Shrinkage (#)","FTE Over/Under (#)"] if (k or "").strip().lower().startswith("billable hours") else
                  ["Billable Transactions","FTE Required (#)","FTE Over/Under (#)"] if (k or "").strip().lower().startswith("fte based billable") else
                  ["FTE Required (#)","FTE Over/Under (#)"])})(ptype)

    fw_rows = spec["fw"]
    fw = pd.DataFrame({"metric": fw_rows})
    for m in month_ids:
        fw[m] = 0.0

    # ---- monthly demand + AHT/SUT actual/forecast (voice+bo) ----
    def _norm_voice(df):
        if not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame(columns=["date","interval","volume","aht_sec","interval_start"])
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.normalize()
        if "interval" in d.columns:
            d["interval"] = d["interval"].astype(str)
        if "interval_start" in d.columns:
            d["interval_start"] = d["interval_start"].astype(str)
        d["volume"]  = pd.to_numeric(d.get("volume"),  errors="coerce").fillna(0.0)
        if "aht_sec" not in d.columns and "aht" in d.columns:
            d["aht_sec"] = pd.to_numeric(d["aht"], errors="coerce")
        d["aht_sec"] = pd.to_numeric(d.get("aht_sec"), errors="coerce").fillna(0.0)
        return d.dropna(subset=["date"])

    def _norm_bo(df):
        if not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame(columns=["date","items","aht_sec"])
        d = df.copy()
        d["date"]   = pd.to_datetime(d["date"], errors="coerce").dt.normalize()
        d["items"]  = pd.to_numeric(d.get("items", d.get("volume")), errors="coerce").fillna(0.0)
        d["aht_sec"]= pd.to_numeric(d.get("aht_sec", d.get("sut_sec", d.get("sut"))), errors="coerce").fillna(0.0)
        return d.dropna(subset=["date"])

    vF = _norm_voice(vF); vA = _norm_voice(vA); vT = _norm_voice(vT)
    bF = _norm_bo(bF);   bA = _norm_bo(bA);   bT = _norm_bo(bT)

    # monthly weighted figures
    def _voice_monthly(df):
        if not isinstance(df, pd.DataFrame) or df.empty:
            return {}, {}
        d = df.copy()
        d["_m"] = _mid(d["date"])
        d["_num"] = d["volume"] * d["aht_sec"]
        vol = d.groupby("_m", as_index=True)["volume"].sum()
        num = d.groupby("_m", as_index=True)["_num"].sum()
        aht = (num / vol.replace({0: np.nan})).fillna(np.nan)
        return vol.to_dict(), aht.to_dict()

    def _bo_monthly(df):
        if not isinstance(df, pd.DataFrame) or df.empty:
            return {}, {}
        d = df.copy()
        d["_m"] = _mid(d["date"])
        d["_num"] = d["items"] * d["aht_sec"]
        itm = d.groupby("_m", as_index=True)["items"].sum()
        num = d.groupby("_m", as_index=True)["_num"].sum()
        sut = (num / itm.replace({0: np.nan})).fillna(np.nan)
        return itm.to_dict(), sut.to_dict()

    vF_vol, vF_aht = _voice_monthly(vF); vA_vol, vA_aht = _voice_monthly(vA); vT_vol, vT_aht = _voice_monthly(vT)
    bF_itm, bF_sut = _bo_monthly(bF);   bA_itm, bA_sut = _bo_monthly(bA);   bT_itm, bT_sut = _bo_monthly(bT)

    # Write rows in FW table
    for m in month_ids:
        f_voice = float(vF_vol.get(m, 0.0)); f_bo = float(bF_itm.get(m, 0.0))
        a_voice = float(vA_vol.get(m, 0.0)); a_bo = float(bA_itm.get(m, 0.0))
        t_voice = float(vT_vol.get(m, 0.0)); t_bo = float(bT_itm.get(m, 0.0))
        if _wf_active_month(m) and vol_delta:
            f_voice *= (1.0 + vol_delta / 100.0)
            f_bo    *= (1.0 + vol_delta / 100.0)
        if "Forecast" in fw_rows:          fw.loc[fw["metric"] == "Forecast",          m] = f_voice + f_bo
        if "Tactical Forecast" in fw_rows: fw.loc[fw["metric"] == "Tactical Forecast", m] = t_voice + t_bo
        if "Actual Volume" in fw_rows:     fw.loc[fw["metric"] == "Actual Volume",     m] = a_voice + a_bo

        # Actual AHT/SUT (weighted across voice+bo)
        a_num = a_den = 0.0
        if m in vA_aht and m in vA_vol and vA_vol[m] > 0: a_num += float(vA_aht[m]) * float(vA_vol[m]); a_den += float(vA_vol[m])
        if m in bA_sut and m in bA_itm and bA_itm[m] > 0: a_num += float(bA_sut[m]) * float(bA_itm[m]); a_den += float(bA_itm[m])
        actual_aht_sut = (a_num / a_den) if a_den > 0 else 0.0
        actual_aht_sut = (actual_aht_sut if actual_aht_sut > 0 else s_target_aht)
        if "Actual AHT/SUT" in fw_rows:
            fw.loc[fw["metric"] == "Actual AHT/SUT", m] = actual_aht_sut

        # Forecast AHT/SUT (planned overrides aware)
        f_num = f_den = 0.0
        ovr_aht_voice = planned_aht_m.get(m)  # treat planned as forecast override when present
        ovr_sut_bo    = planned_sut_m.get(m)
        if f_voice > 0:
            f_num += (ovr_aht_voice if ovr_aht_voice not in (None, 0, np.nan) else float(vF_aht.get(m, s_target_aht))) * f_voice; f_den += f_voice
        if f_bo > 0:
            f_num += (ovr_sut_bo if ovr_sut_bo not in (None, 0, np.nan) else float(bF_sut.get(m, s_target_sut))) * f_bo; f_den += f_bo
        forecast_aht_sut = (f_num / f_den) if f_den > 0 else s_target_aht
        if "Forecast AHT/SUT" in fw_rows:
            fw.loc[fw["metric"] == "Forecast AHT/SUT", m] = forecast_aht_sut

        # Budgeted AHT/SUT
        if "Budgeted AHT/SUT" in fw_rows:
            b_num = b_den = 0.0
            if f_voice > 0: b_num += float(planned_aht_m.get(m, s_budget_aht)) * f_voice; b_den += f_voice
            if f_bo    > 0: b_num += float(planned_sut_m.get(m, s_budget_sut)) * f_bo;  b_den += f_bo
            budget_aht_sut = (b_num / b_den) if b_den > 0 else s_budget_aht
            fw.loc[fw["metric"] == "Budgeted AHT/SUT", m] = budget_aht_sut

    # Save current FW to support Backlog/Overtime readback
    fw_saved = load_df(f"plan_{pid}_fw")

    def _row_to_month_dict(df: pd.DataFrame, metric_name: str) -> dict:
        try:
            if not isinstance(df, pd.DataFrame) or df.empty:
                return {}
            d = df.copy()
            # harmonize column ids to month ids
            for c in list(d.columns):
                if c == "metric": continue
                try:
                    mid = _mid(pd.Series([c]))[0]
                    if mid != c:
                        d.rename(columns={c: mid}, inplace=True)
                except Exception:
                    pass
            m = d["metric"].astype(str).str.strip()
            if metric_name not in m.values:
                return {}
            row = d.loc[m == metric_name].iloc[0]
            out = {}
            for mid in month_ids:
                try:
                    out[mid] = float(pd.to_numeric(row.get(mid), errors="coerce"))
                except Exception:
                    out[mid] = 0.0
            return out
        except Exception:
            return {}

    overtime_m = _row_to_month_dict(fw_saved, "Overtime Hours (#)")
    backlog_m  = _row_to_month_dict(fw_saved, "Backlog (Items)")

    # ---- Apply Backlog carryover (Back Office only): add previous month's backlog to next month's BO forecast ----
    if backlog_carryover and str(ch_first).strip().lower() in ("back office", "bo") and backlog_m:
        for i in range(len(month_ids) - 1):
            cur_m = month_ids[i]; nxt_m = month_ids[i+1]
            add = float(backlog_m.get(cur_m, 0.0) or 0.0)
            if add:
                fw.loc[fw["metric"] == "Forecast", nxt_m] = float(fw.loc[fw["metric"] == "Forecast", nxt_m]) + add

    # ---- Occupancy/Utilization by channel (% in FW grid) ----
    ch_key = str(ch_first or "").strip().lower()
    if ch_key in ("voice",):
        occ_base_raw = settings.get("occupancy_cap_voice", settings.get("occupancy", 0.85))
    elif ch_key in ("back office", "bo"):
        occ_base_raw = settings.get("util_bo", 0.85)
    elif ch_key in ("outbound",):
        occ_base_raw = settings.get("util_ob", 0.85)
    elif ch_key in ("chat",):
        occ_base_raw = settings.get("util_chat", settings.get("util_bo", 0.85))
    else:
        occ_base_raw = (settings.get("occupancy") or settings.get("occupancy_pct") or settings.get("target_occupancy") or
                        settings.get("budgeted_occupancy") or settings.get("occ") or settings.get("occupancy_cap_voice") or 0.85)
    try:
        if isinstance(occ_base_raw, str) and occ_base_raw.strip().endswith("%"):
            occ_base = float(occ_base_raw.strip()[:-1])
        else:
            occ_base = float(occ_base_raw)
            if occ_base <= 1.0:
                occ_base *= 100.0
    except Exception:
        occ_base = 85.0

    occ_m = {m: int(round(occ_base)) for m in month_ids}
    if occ_override not in (None, ""):
        try:
            ov = float(occ_override); ov = ov*100.0 if ov <= 1.0 else ov
            ov = int(round(ov))
            for m in month_ids:
                if _wf_active_month(m):
                    occ_m[m] = ov
        except Exception:
            pass

    if "Occupancy" in fw_rows:
        for m in month_ids:
            fw.loc[fw["metric"] == "Occupancy", m] = occ_m[m]

    occ_frac_m = {m: min(0.99, max(0.01, float(occ_m[m]) / 100.0)) for m in month_ids}

    # ---- requirements: Interval/Daily → Monthly (true monthly, no weekly roll-up) ----
    req_daily_actual   = required_fte_daily(use_voice_for_req, use_bo_for_req, oA, settings)
    req_daily_forecast = required_fte_daily(vF, bF, oF, settings)
    req_daily_tactical = required_fte_daily(vT, bT, oT, settings) if (isinstance(vT, pd.DataFrame) and not vT.empty) or (isinstance(bT, pd.DataFrame) and not bT.empty) or (isinstance(oT, pd.DataFrame) and not oT.empty) else pd.DataFrame()
    # add Chat to daily totals
    from capacity_core import chat_fte_daily as _chat_fd
    for _df, chat_df in ((req_daily_actual, cA), (req_daily_forecast, cF), (req_daily_tactical, cT)):
        if isinstance(_df, pd.DataFrame) and not _df.empty and isinstance(chat_df, pd.DataFrame) and not chat_df.empty:
            try:
                ch = _chat_fd(chat_df, settings)
                m = _df.merge(ch, on=["date","program"], how="left")
                m["chat_fte"] = pd.to_numeric(m.get("chat_fte"), errors="coerce").fillna(0.0)
                m["total_req_fte"] = pd.to_numeric(m.get("total_req_fte"), errors="coerce").fillna(0.0) + m["chat_fte"]
                _df.drop(_df.index, inplace=True)
                _df[list(m.columns)] = m
            except Exception:
                pass

    # For budgeted, inject budgeted AHT/SUT (month plans) into copies then run daily engine
    vB = vF.copy(); bB = bF.copy(); oB = oF.copy(); cB = cF.copy()
    if isinstance(vB, pd.DataFrame) and not vB.empty:
        vB["month_id"] = _mid(vB["date"])
        vB["aht_sec"]  = vB["month_id"].map(planned_aht_m).fillna(float(s_budget_aht))
        vB.drop(columns=["month_id"], inplace=True)
    if isinstance(bB, pd.DataFrame) and not bB.empty:
        bB["month_id"] = _mid(bB["date"])
        bB["aht_sec"]  = bB["month_id"].map(planned_sut_m).fillna(float(s_budget_sut))
        bB.drop(columns=["month_id"], inplace=True)
    if isinstance(oB, pd.DataFrame) and not oB.empty:
        oB["month_id"] = _mid(oB["date"])
        oB["aht_sec"]  = oB["month_id"].map(lambda m: planned_aht_m.get(m, None)).fillna(float(s_budget_aht))
        oB.drop(columns=["month_id"], inplace=True)
    req_daily_budgeted = required_fte_daily(vB, bB, oB, settings)
    if isinstance(req_daily_budgeted, pd.DataFrame) and not req_daily_budgeted.empty and isinstance(cB, pd.DataFrame) and not cB.empty:
        try:
            chb = _chat_fd(cB, settings)
            m = req_daily_budgeted.merge(chb, on=["date","program"], how="left")
            m["chat_fte"] = pd.to_numeric(m.get("chat_fte"), errors="coerce").fillna(0.0)
            m["total_req_fte"] = pd.to_numeric(m.get("total_req_fte"), errors="coerce").fillna(0.0) + m["chat_fte"]
            req_daily_budgeted.drop(req_daily_budgeted.index, inplace=True)
            req_daily_budgeted[list(m.columns)] = m
        except Exception:
            pass

    def _workdays_in_month(mid: str, is_bo: bool) -> int:
        t = pd.to_datetime(mid, errors="coerce")
        if pd.isna(t): return 22 if is_bo else 30
        y, m = int(t.year), int(t.month)
        days = calendar.monthrange(y, m)[1]
        if not is_bo:
            return days
        return sum(1 for d in range(1, days + 1) if pd.Timestamp(year=y, month=m, day=d).weekday() < 5)

    def _daily_to_monthly(df, is_bo: bool):
        if not isinstance(df, pd.DataFrame) or df.empty or "date" not in df.columns or "total_req_fte" not in df.columns:
            return {}
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d = d.dropna(subset=["date"])
        d["month"] = _mid(d["date"])
        g = d.groupby("month", as_index=False)["total_req_fte"].sum()
        req_m = {}
        for _, r in g.iterrows():
            mid = str(r["month"])
            wd = _workdays_in_month(mid, is_bo=is_bo)
            req_m[mid] = float(r["total_req_fte"]) / max(1, wd)  # average daily FTE for month
        return req_m

    is_bo_ch = str(ch_first).strip().lower() in ("back office", "bo")
    req_m_actual   = _daily_to_monthly(req_daily_actual,   is_bo=is_bo_ch)
    req_m_forecast = _daily_to_monthly(req_daily_forecast, is_bo=is_bo_ch)
    req_m_tactical = _daily_to_monthly(req_daily_tactical, is_bo=is_bo_ch)
    req_m_budgeted = _daily_to_monthly(req_daily_budgeted, is_bo=is_bo_ch)

    # What-If: adjust forecast requirements by volume and shrink deltas
    if vol_delta or shrink_delta:
        for mid in list(req_m_forecast.keys()):
            v = float(req_m_forecast[mid])
            if vol_delta:
                v *= (1.0 + vol_delta / 100.0)
            if shrink_delta:
                denom = max(0.1, 1.0 - (shrink_delta / 100.0))
                v /= denom
            req_m_forecast[mid] = v

    # ---- Interval supply from global roster_long (monthly avg per interval) ----
    ivl_min = int(float(settings.get("interval_minutes", 30)) or 30)
    ivl_sec = 60 * ivl_min
    monthly_voice_intervals = {m: (calendar.monthrange(int(m[:4]), int(m[5:7]))[1] * (24 * 3600 // ivl_sec)) for m in month_ids}
    schedule_supply_avg_m = {}
    try:
        rl = load_roster_long()
    except Exception:
        rl = None
    if isinstance(rl, pd.DataFrame) and not rl.empty:
        df = rl.copy()

        def _col(d, opts):
            for c in opts:
                if c in d.columns:
                    return c
            return None

        c_ba  = _col(df, ["Business Area","business area","vertical"])
        c_sba = _col(df, ["Sub Business Area","sub business area","sub_ba"])
        c_lob = _col(df, ["LOB","lob","Channel","channel"])
        c_site= _col(df, ["Site","site","Location","location","Country","country"])

        BA  = p.get("vertical"); SBA = p.get("sub_ba"); LOB = ch_first
        SITE= p.get("site") or p.get("location") or p.get("country")

        def _match(series, val):
            if not val or not isinstance(series, pd.Series):
                return pd.Series([True]*len(series))
            s = series.astype(str).str.strip().str.lower()
            return s.eq(str(val).strip().lower())

        m = pd.Series([True]*len(df))
        if c_ba:  m &= _match(df[c_ba], BA)
        if c_sba and (SBA not in (None, "")): m &= _match(df[c_sba], SBA)
        if c_lob: m &= _match(df[c_lob], LOB)
        if c_site and (SITE not in (None, "")): m &= _match(df[c_site], SITE)
        df = df[m]

        if "is_leave" in df.columns:
            df = df[~df["is_leave"].astype(bool)]
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])

        import re as _re
        def _shift_len_ivl(s: str) -> int:
            try:
                s = str(s or "").strip()
                mm = _re.match(r"^(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})$", s)
                if not mm:
                    return 0
                sh, sm, eh, em = map(int, mm.groups())
                sh = min(23, max(0, sh)); eh = min(24, max(0, eh))
                start = sh*60 + sm; end = eh*60 + em
                if end < start:
                    end += 24*60
                return max(0, int((end - start + (ivl_min-1)) // ivl_min))
            except Exception:
                return 0

        if "entry" in df.columns and "date" in df.columns:
            df["ivl_count"] = df["entry"].apply(_shift_len_ivl)
            df["month"] = _mid(df["date"])
            agg = df.groupby("month", as_index=False)["ivl_count"].sum()
            monthly_agent_ivls = dict(zip(agg["month"], agg["ivl_count"]))
            for m_id in month_ids:
                denom = monthly_voice_intervals.get(m_id, 0)
                if denom <= 0:
                    denom = monthly_voice_intervals.get(m_id, 1)
                schedule_supply_avg_m[m_id] = float(monthly_agent_ivls.get(m_id, 0.0)) / float(denom or 1)

    # ---- lower shells (same metrics) ----
    hc   = _load_or_blank(f"plan_{pid}_hc",   ["Budgeted HC (#)","Planned/Tactical HC (#)","Actual Agent HC (#)","SME Billable HC (#)"], month_ids)
    att  = _load_or_blank(f"plan_{pid}_attr", ["Planned Attrition HC (#)","Actual Attrition HC (#)","Attrition %"], month_ids)
    shr  = _load_or_blank(f"plan_{pid}_shr",  ["OOO Shrink Hours (#)","Inoffice Shrink Hours (#)","OOO Shrinkage %","Inoffice Shrinkage %","Overall Shrinkage %","Planned Shrinkage %","Variance vs Planned (pp)"], month_ids)
    trn  = _load_or_blank(f"plan_{pid}_train",["Training Start (#)","Training End (#)","Nesting Start (#)","Nesting End (#)"], month_ids)
    rat  = _load_or_blank(f"plan_{pid}_ratio",["Planned TL/Agent Ratio","Actual TL/Agent Ratio","Variance"], month_ids)
    seat = _load_or_blank(f"plan_{pid}_seat", ["Seats Required (#)","Seats Available (#)","Seat Utilization %"], month_ids)
    bva  = _load_or_blank(f"plan_{pid}_bva",  ["Budgeted FTE (#)","Actual FTE (#)","Variance (#)"], month_ids)
    nh   = _load_or_blank(f"plan_{pid}_nh",   ["Planned New Hire HC (#)","Actual New Hire HC (#)","Recruitment Achievement"], month_ids)

    # ── New Hire overlay (classes + roster) ──
    today_m = pd.to_datetime(dt.date.today()).to_period("M").to_timestamp().date().isoformat()

    # Planned NH from classes: count production_start in month (throughput via LC)
    def _monthly_planned_nh_from_classes(pid, mids):
        classes_df = load_df(f"plan_{pid}_nh_classes")
        out = {m: 0 for m in mids}
        if not isinstance(classes_df, pd.DataFrame) or classes_df.empty:
            # ALWAYS return a tuple
            return out, pd.DataFrame()

        c = classes_df.copy()
        # production_start column normalization
        for cand in ("production_start", "prod_start", "to_production", "go_live"):
            if cand in c.columns:
                c["production_start"] = pd.to_datetime(c[cand], errors="coerce")
                break
        if "production_start" not in c.columns:
            return out, c  # still return tuple

        c = c.dropna(subset=["production_start"]).copy()
        c["month"] = pd.to_datetime(c["production_start"]).dt.to_period("M").dt.to_timestamp().dt.normalize().dt.date.astype(str)

        # Count planned joiners by production month
        g = c.groupby("month")["production_start"].count().to_dict()
        for m in mids:
            out[m] = int(g.get(m, 0))

        return out, c


    planned_nh_m_raw, classes_df = _monthly_planned_nh_from_classes(pid, month_ids)

    # Apply learning-curve throughput to planned NH (month-level)
    def _learning_curve_for_month(settings, lc_ovr_df, mid):
        # reuse weekly curve but just return dict of lists
        return _lc_with_wf_m(_learning_curve_for_week(settings, lc_ovr_df, mid), mid)

    planned_nh_m = {}
    for m in month_ids:
        lc = _learning_curve_for_month(settings, lc_ovr_df, m)
        tp = (lc.get("throughput_train_pct", 100.0) / 100.0) * (lc.get("throughput_nest_pct", 100.0) / 100.0)
        planned_nh_m[m] = int(round(planned_nh_m_raw.get(m, 0) * max(0.0, min(1.0, tp))))

    # Buckets from classes for month (in phase counts)
    def _monthly_buckets_from_classes(df, mids):
        from collections import defaultdict
        nest = {m: defaultdict(int) for m in mids}
        sda  = {m: defaultdict(int) for m in mids}
        if not isinstance(df, pd.DataFrame) or df.empty: return nest, sda

        def _m(d):
            t = pd.to_datetime(d, errors="coerce")
            if pd.isna(t): return None
            return pd.Timestamp(t).to_period("M").to_timestamp().date().isoformat()

        for _, r in df.iterrows():
            n = _nh_effective_count(r)
            if n <= 0: continue
            ns = _m(r.get("nesting_start")); ne = _m(r.get("nesting_end")); ps = _m(r.get("production_start"))
            if ns and ne:
                mlist = [mm for mm in mids if (mm >= ns and mm <= ne)]
                for i, mm in enumerate(mlist, start=1): nest[mm][i] += n
            if ps:
                sda_weeks = int(float(settings.get("sda_weeks", settings.get("default_sda_weeks", 0)) or 0))
                if sda_weeks > 0:
                    mlist = [mm for mm in mids if mm >= ps][:max(1, (sda_weeks+3)//4)]  # approx weeks→months
                    for i, mm in enumerate(mlist, start=1): sda[mm][i] += n
        return nest, sda

    nest_buckets, sda_buckets = _monthly_buckets_from_classes(classes_df, month_ids)

    # in-phase counters (peak within month)
    m_train_in_phase = {m: sum(nest_buckets[m].values()) for m in month_ids}  # training ~= nesting buckets prior to prod
    m_nest_in_phase  = {m: sum(nest_buckets[m].values()) for m in month_ids}
    for m in month_ids:
        trn.loc[trn["metric"] == "Training Start (#)", m] = m_train_in_phase.get(m, 0)
        trn.loc[trn["metric"] == "Training End (#)",   m] = m_train_in_phase.get(m, 0)
        trn.loc[trn["metric"] == "Nesting Start (#)",  m] = m_nest_in_phase.get(m, 0)
        trn.loc[trn["metric"] == "Nesting End (#)",    m] = m_nest_in_phase.get(m, 0)

    # Actual joiners by production month (from roster)
    roster_df = _load_roster_normalized(pid)
    def _monthly_actual_nh_from_roster(roster_df, mids):
        if not isinstance(roster_df, pd.DataFrame) or roster_df.empty:
            return {m:0 for m in mids}
        r = roster_df.copy()
        c = None
        for k in ("production_start","prod_start","to_production","go_live"):
            if k in r.columns:
                c = k; break
        if not c:
            return {m:0 for m in mids}
        r[c] = pd.to_datetime(r[c], errors="coerce")
        r = r.dropna(subset=[c])
        r["month"] = _mid(r[c])
        g = r.groupby("month")[c].count().to_dict()
        return {m: int(g.get(m, 0)) for m in mids}

    actual_nh_m = _monthly_actual_nh_from_roster(roster_df, month_ids)

    for m in month_ids:
        nh.loc[nh["metric"] == "Planned New Hire HC (#)", m] = int(planned_nh_m.get(m, 0))
        nh.loc[nh["metric"] == "Actual New Hire HC (#)",  m] = int(actual_nh_m.get(m, 0) if m <= today_m else 0)
        plan = float(planned_nh_m.get(m, 0))
        act  = float(actual_nh_m.get(m, 0) if m <= today_m else 0)
        nh.loc[nh["metric"] == "Recruitment Achievement", m] = (0.0 if plan <= 0 else 100.0 * act / plan)

    # ---- Actual HC snapshots from roster → monthly average ----
    def _monthly_hc_average_from_roster(roster_df, mids, role_regex=r"\bagent\b"):
        if not isinstance(roster_df, pd.DataFrame) or roster_df.empty:
            return {m:0 for m in mids}
        R = roster_df.copy()
        # normalize timestamps
        for c in ("production_start","prod_start","terminate_date","term_date"):
            if c in R.columns:
                R[c] = pd.to_datetime(R[c], errors="coerce").dt.normalize()
        if "production_start" not in R.columns and "prod_start" in R.columns:
            R["production_start"] = R["prod_start"]
        if "terminate_date" not in R.columns and "term_date" in R.columns:
            R["terminate_date"] = R["term_date"]
        # role filter
        try:
            R["role"] = R["role"].astype(str)
            mask_role = R["role"].str.contains(role_regex, flags=re.IGNORECASE, regex=True)
            R = R[mask_role]
        except Exception:
            pass
        out = {}
        for mid in mids:
            base = pd.to_datetime(mid, errors="coerce").normalize()
            if pd.isna(base):
                out[mid] = 0; continue
            y, m = int(base.year), int(base.month)
            days = calendar.monthrange(y, m)[1]
            dates = pd.date_range(base, periods=days, freq="D")
            counts = []
            for d in dates:
                active = ((R["production_start"].isna()) | (R["production_start"] <= d)) & ((R["terminate_date"].isna()) | (R["terminate_date"] > d))
                counts.append(int(active.sum()))
            out[mid] = int(round(float(np.mean(counts)))) if counts else 0
        return out

    hc_actual_m    = _monthly_hc_average_from_roster(roster_df, month_ids, r"\bagent\b")
    sme_billable_m = _monthly_hc_average_from_roster(roster_df, month_ids, r"\bsme\b")

    for m in month_ids:
        hc.loc[hc["metric"] == "Actual Agent HC (#)", m] = hc_actual_m.get(m, 0)
        hc.loc[hc["metric"] == "SME Billable HC (#)", m] = sme_billable_m.get(m, 0)

    # ---- Budget vs simple Planned HC (monthly reduce) ----
    def _monthly_reduce(df: pd.DataFrame, value_candidates=("hc","headcount","value","count"), how="sum"):
        if not isinstance(df, pd.DataFrame) or df.empty:
            return {}
        d = df.copy()
        c_date = None
        for c in ("date","week","month","start_date"):
            if c in d.columns: c_date = c; break
        if not c_date:
            return {}
        d[c_date] = pd.to_datetime(d[c_date], errors="coerce")
        d["month"] = _mid(d[c_date])
        low = {c.lower(): c for c in d.columns}
        vcol = None
        for c in value_candidates:
            vcol = low.get(c.lower()); 
            if vcol: break
        if not vcol: return {}
        d[vcol] = pd.to_numeric(d[vcol], errors="coerce").fillna(0.0)
        g = d.groupby("month", as_index=True)[vcol].agg(how)
        return g.astype(float).to_dict()

    budget_df = _first_non_empty_ts(sk, ["budget_headcount","budget_hc","headcount_budget","hc_budget"])
    budget_m  = _monthly_reduce(budget_df, value_candidates=("hc","headcount","value","count"), how="sum")
    for m in month_ids:
        hc.loc[hc["metric"] == "Budgeted HC (#)",         m] = float(budget_m.get(m, 0.0))
        hc.loc[hc["metric"] == "Planned/Tactical HC (#)", m] = float(budget_m.get(m, 0.0))

    # ---- Attrition (planned/actual/pct) monthly ----
    att_plan_m = _monthly_reduce(_first_non_empty_ts(sk, ["attrition_planned_hc","attrition_plan_hc","planned_attrition_hc"]),
                                 value_candidates=("hc","headcount","value","count"), how="sum")
    att_act_m  = _monthly_reduce(_first_non_empty_ts(sk, ["attrition_actual_hc","attrition_actual","actual_attrition_hc"]),
                                 value_candidates=("hc","headcount","value","count"), how="sum")
    att_pct_m  = _monthly_reduce(_first_non_empty_ts(sk, ["attrition_pct","attrition_percent","attrition%","attrition_rate"]),
                                 value_candidates=("pct","percent","value"), how="mean")

    att_plan_saved, att_actual_saved = {}, {}
    try:
        att_saved_df = load_df(f"plan_{pid}_attr")
    except Exception:
        att_saved_df = None
    if isinstance(att_saved_df, pd.DataFrame) and not att_saved_df.empty:
        metrics = att_saved_df["metric"].astype(str).str.strip()
        if "Planned Attrition HC (#)" in metrics.values:
            row = att_saved_df.loc[metrics == "Planned Attrition HC (#)"].iloc[0]
            for m in month_ids:
                if m in row:
                    val = pd.to_numeric(row.get(m), errors="coerce")
                    if pd.notna(val):
                        att_plan_saved[m] = float(val)
        if "Actual Attrition HC (#)" in metrics.values:
            row = att_saved_df.loc[metrics == "Actual Attrition HC (#)"].iloc[0]
            for m in month_ids:
                if m in row:
                    val = pd.to_numeric(row.get(m), errors="coerce")
                    if pd.notna(val):
                        att_actual_saved[m] = float(val)
    attr_roster_m = {m: 0 for m in month_ids}
    if isinstance(roster_df, pd.DataFrame) and not roster_df.empty:
        R = roster_df.copy()
        L = {str(c).strip().lower(): c for c in R.columns}
        c_role = L.get("role") or L.get("position group") or L.get("position description")
        c_term = L.get("terminate date") or L.get("terminate_date") or L.get("termination date")
        if c_role and c_term:
            role = R[c_role].astype(str).str.strip().str.lower()
            mask = role.str.contains(r"agent", na=False, regex=True)
            term_series = pd.to_datetime(R.loc[mask, c_term], errors="coerce").dropna()
            if not term_series.empty:
                months = _mid(pd.Series(term_series))
                for tm in months:
                    tm = str(tm)
                    if tm in attr_roster_m:
                        attr_roster_m[tm] += 1

    try:
        _hc_plan_row = hc.loc[hc["metric"].astype(str).str.strip() == "Planned/Tactical HC (#)"].iloc[0]
    except Exception:
        _hc_plan_row = None

    for m in month_ids:
        plan_ts = float(att_plan_m.get(m, 0.0))
        plan_manual = att_plan_saved.get(m)
        roster_term = float(attr_roster_m.get(m, 0.0))
        if plan_manual is not None:
            plan_hc = plan_manual
        else:
            plan_hc = plan_ts
            if roster_term > 0 and m > today_m and plan_hc < roster_term:
                plan_hc = roster_term
        if _wf_active_month(m) and attr_delta and m > today_m:
            plan_hc += float(attr_delta)
        plan_hc = max(0.0, plan_hc)

        act_manual = att_actual_saved.get(m)
        act_ts = float(att_act_m.get(m, 0.0))
        if act_manual is not None:
            act_hc = act_manual
        else:
            act_hc = act_ts
            if roster_term > 0 and m <= today_m:
                act_hc = max(act_hc, roster_term)
        act_hc = max(0.0, act_hc)

        pct = att_pct_m.get(m, None)
        if pct is None:
            base_actual = float(hc_actual_m.get(m, 0))
            pct = 100.0 * (act_hc / base_actual) if base_actual > 0 else 0.0
        if _wf_active_month(m) and attr_delta and m > today_m:
            try:
                denom = float(pd.to_numeric((_hc_plan_row or {}).get(m), errors="coerce")) if hasattr(_hc_plan_row, 'get') else float(pd.to_numeric(_hc_plan_row[m], errors="coerce"))
            except Exception:
                denom = 0.0
            if denom > 0:
                pct = 100.0 * (plan_hc / denom)
        att.loc[att["metric"] == "Planned Attrition HC (#)", m] = plan_hc
        att.loc[att["metric"] == "Actual Attrition HC (#)",  m] = act_hc
        att.loc[att["metric"] == "Attrition %",              m] = pct
    # ---- Shrinkage raw → monthly ----
    ch_key = str(ch_first or '').strip().lower()
    def _planned_shr(val, fallback):
        try:
            if val in (None, '', 'nan'):
                raise ValueError
            x = float(val)
            if x > 1.0:
                x /= 100.0
            return max(0.0, x)
        except Exception:
            try:
                return max(0.0, float(fallback))
            except Exception:
                return max(0.0, 0.0)
    planned_shrink_fraction = _planned_shr(settings.get('shrinkage_pct'), 0.0)
    if ch_key == 'voice':
        planned_shrink_fraction = _planned_shr(settings.get('voice_shrinkage_pct'), planned_shrink_fraction)
    elif ch_key in ('back office', 'bo'):
        planned_shrink_fraction = _planned_shr(settings.get('bo_shrinkage_pct'), planned_shrink_fraction)
    elif 'chat' in ch_key:
        planned_shrink_fraction = _planned_shr(settings.get('chat_shrinkage_pct'), planned_shrink_fraction)
    elif ch_key in ('outbound', 'out bound', 'ob'):
        planned_shrink_fraction = _planned_shr(settings.get('ob_shrinkage_pct'), planned_shrink_fraction)

    ooo_hours_m, io_hours_m, base_hours_m = {}, {}, {}

    def _month_key(s):
        ds = pd.to_datetime(s, errors="coerce")
        if isinstance(ds, pd.Series):
            return _mid(ds)
        else:
            return _mid(pd.Series(ds))

    def _agg_monthly(date_idx, ooo_series, ino_series, base_series):
        mk = _month_key(date_idx)
        g = pd.DataFrame({"month": mk, "ooo": ooo_series, "ino": ino_series, "base": base_series}).groupby("month", as_index=False).sum()
        for _, r in g.iterrows():
            k = str(r["month"])
            ooo_hours_m[k]  = ooo_hours_m.get(k, 0.0)  + float(r["ooo"])
            io_hours_m[k]   = io_hours_m.get(k, 0.0)   + float(r["ino"])
            base_hours_m[k] = base_hours_m.get(k, 0.0) + float(r["base"])

    if ch_first.lower() == "voice":
        try:
            vraw = load_df("shrinkage_raw_voice")
        except Exception:
            vraw = None
        if isinstance(vraw, pd.DataFrame) and not vraw.empty:
            v = vraw.copy()
            L = {str(c).strip().lower(): c for c in v.columns}
            c_date = L.get("date"); c_hours = L.get("hours") or L.get("duration_hours") or L.get("duration")
            c_state= L.get("superstate") or L.get("state")
            c_ba   = L.get("business area") or L.get("ba")
            c_sba  = L.get("sub business area") or L.get("sub_ba")
            c_ch   = L.get("channel")
            c_loc  = L.get("country") or L.get("location") or L.get("site") or L.get("city")

            mask = pd.Series(True, index=v.index)
            if c_ba and p.get("vertical"): mask &= v[c_ba].astype(str).str.strip().str.lower().eq(p["vertical"].strip().lower())
            if c_sba and p.get("sub_ba"): mask &= v[c_sba].astype(str).str.strip().str.lower().eq(p["sub_ba"].strip().lower())
            if c_ch: mask &= v[c_ch].astype(str).str.strip().str.lower().eq("voice")
            if c_loc and loc_first:
                loc_series = v[c_loc].astype(str).str.strip()
                loc_l = loc_series.str.lower()
                target = loc_first.strip().lower()
                if loc_l.ne("").any() and loc_l.ne("all").any() and loc_l.eq(target).any():
                    mask &= loc_l.eq(target)

            v = v.loc[mask]
            if c_date and c_state and c_hours and not v.empty:
                pv = v.pivot_table(index=c_date, columns=c_state, values=c_hours, aggfunc="sum", fill_value=0.0)
                def col(name): return pv[name] if name in pv.columns else 0.0
                base = col("SC_INCLUDED_TIME")
                ooo  = col("SC_ABSENCE_TOTAL") + col("SC_HOLIDAY") + col("SC_A_Sick_Long_Term")
                ino  = col("SC_TRAINING_TOTAL") + col("SC_BREAKS") + col("SC_SYSTEM_EXCEPTION")
                idx_dates = pd.to_datetime(pv.index, errors="coerce")
                _agg_monthly(idx_dates, ooo, ino, base)

    if ch_first.lower() in ("back office", "bo"):
        try:
            braw = load_df("shrinkage_raw_backoffice")
        except Exception:
            braw = None
        if isinstance(braw, pd.DataFrame) and not braw.empty:
            b = braw.copy()
            L = {str(c).strip().lower(): c for c in b.columns}
            c_date = L.get("date"); c_act = L.get("activity")
            c_sec  = L.get("duration_seconds") or L.get("seconds") or L.get("duration")
            c_ba   = L.get("journey") or L.get("business area") or L.get("ba")
            c_sba  = L.get("sub_business_area") or L.get("sub business area") or L.get("sub_ba")

            mask = pd.Series(True, index=b.index)
            if c_ba and p.get("vertical"): mask &= b[c_ba].astype(str).str.strip().str.lower().eq(p["vertical"].strip().lower())
            if c_sba and p.get("sub_ba"): mask &= b[c_sba].astype(str).str.strip().str.lower().eq(p["sub_ba"].strip().lower())

            if c_date and c_act and c_sec and not b.empty:
                d = b[[c_date, c_act, c_sec]].copy()
                d[c_act] = d[c_act].astype(str).str.strip().str.lower()
                d[c_sec] = pd.to_numeric(d[c_sec], errors="coerce").fillna(0.0)
                d[c_date] = pd.to_datetime(d[c_date], errors="coerce").dt.date
                def has(s): return d[c_act].str.contains(s, na=False)
                sec_div = d.loc[has("divert"), c_sec].groupby(d[c_date]).sum()
                sec_dow = d.loc[has("down"),   c_sec].groupby(d[c_date]).sum()
                sec_sc  = d.loc[has("staff complement"), c_sec].groupby(d[c_date]).sum()
                sec_fx  = d.loc[has("flex"),            c_sec].groupby(d[c_date]).sum()
                sec_ot  = d.loc[has("overtime") | d[c_act].eq("ot"), c_sec].groupby(d[c_date]).sum()
                sec_lend= d.loc[has("lend"),            c_sec].groupby(d[c_date]).sum()
                sec_borr= d.loc[has("borrow"),          c_sec].groupby(d[c_date]).sum()

                idx = pd.to_datetime(pd.Index(
                    set(sec_div.index) | set(sec_dow.index) | set(sec_sc.index) | set(sec_fx.index) |
                    set(sec_ot.index)  | set(sec_lend.index)| set(sec_borr.index)
                ), errors="coerce").sort_values()

                def get(s): return s.reindex(idx, fill_value=0.0)
                num_sec = get(sec_div) + get(sec_dow)
                den_sec = (get(sec_sc) + get(sec_fx) + get(sec_ot) - get(sec_lend) + get(sec_borr)).clip(lower=0)

                ooo = (0.0 * den_sec).astype(float) / 3600.0
                ino = num_sec.astype(float)         / 3600.0
                base= den_sec.astype(float)         / 3600.0
                _agg_monthly(idx, ooo, ino, base)

    # Build shrink table (+ What-If Δ onto Overall % display)
    for m in month_ids:
        if m not in shr.columns:
            shr[m] = np.nan
        shr[m] = pd.to_numeric(shr[m], errors="coerce").astype("float64")
        base = float(base_hours_m.get(m, 0.0))
        ooo  = float(ooo_hours_m.get(m, 0.0))
        ino  = float(io_hours_m.get(m, 0.0))
        ooo_pct = (100.0 * ooo / base) if base > 0 else 0.0
        ino_pct = (100.0 * ino / base) if base > 0 else 0.0
        ov_pct  = (100.0 * (ooo + ino) / base) if base > 0 else 0.0
        if _wf_active_month(m) and shrink_delta:
            ov_pct = min(100.0, max(0.0, ov_pct + shrink_delta))

        planned_pct = 100.0 * planned_shrink_fraction
        variance_pp = ov_pct - planned_pct

        shr.loc[shr["metric"] == "OOO Shrink Hours (#)",      m] = ooo
        shr.loc[shr["metric"] == "Inoffice Shrink Hours (#)", m] = ino
        shr.loc[shr["metric"] == "OOO Shrinkage %",           m] = ooo_pct
        shr.loc[shr["metric"] == "Inoffice Shrinkage %",      m] = ino_pct
        shr.loc[shr["metric"] == "Overall Shrinkage %",       m] = ov_pct
        shr.loc[shr["metric"] == "Planned Shrinkage %",       m] = planned_pct
        shr.loc[shr["metric"] == "Variance vs Planned (pp)",  m] = variance_pp

    # ---- BvA ----
    for m in month_ids:
        if m not in bva.columns:
            bva[m] = pd.Series(np.nan, index=bva.index, dtype="float64")
        elif not pd.api.types.is_float_dtype(bva[m].dtype):
            bva[m] = pd.to_numeric(bva[m], errors="coerce").astype("float64")
        bud = float(req_m_budgeted.get(m, 0.0))
        act = float(req_m_actual.get(m,   0.0))
        bva.loc[bva["metric"] == "Budgeted FTE (#)", m] = bud
        bva.loc[bva["metric"] == "Actual FTE (#)",   m] = act
        bva.loc[bva["metric"] == "Variance (#)",     m] = act - bud

    # ---- TL/Agent ratios ----
    planned_ratio = _parse_ratio_setting(settings.get("planned_tl_agent_ratio") or settings.get("tl_agent_ratio") or settings.get("tl_per_agent"))
    actual_ratio = 0.0
    try:
        if isinstance(roster_df, pd.DataFrame) and not roster_df.empty and "role" in roster_df.columns:
            r = roster_df.copy()
            r["role"] = r["role"].astype(str).str.strip().str.lower()
            tl = (r["role"] == "team leader").sum()
            ag = (r["role"] == "agent").sum()
            actual_ratio = (float(tl) / float(ag)) if ag > 0 else 0.0
    except Exception:
        pass
    for m in month_ids:
        rat.loc[rat["metric"] == "Planned TL/Agent Ratio", m] = planned_ratio
        rat.loc[rat["metric"] == "Actual TL/Agent Ratio",  m] = actual_ratio
        rat.loc[rat["metric"] == "Variance",               m] = actual_ratio - planned_ratio

    # ---- Projected supply (actuals to date; planned + attr/nH forward) ----
    def _row_as_dict(df, metric_name):
        if not isinstance(df, pd.DataFrame) or df.empty or "metric" not in df.columns:
            return {}
        d = df.copy()
        for c in list(d.columns):
            if c == "metric": continue
            try:
                mid = _mid(pd.Series([c]))[0]
                if mid != c: d.rename(columns={c:mid}, inplace=True)
            except Exception:
                pass
        m = d["metric"].astype(str).str.strip()
        if metric_name not in m.values:
            return {}
        row = d.loc[m == metric_name].iloc[0]
        return {mm: float(pd.to_numeric(row.get(mm), errors="coerce")) for mm in month_ids}

    hc_plan_row   = _row_as_dict(hc,  "Planned/Tactical HC (#)")
    hc_actual_row = {m: float(hc_actual_m.get(m, 0)) for m in month_ids}
    att_plan_row  = _row_as_dict(att, "Planned Attrition HC (#)")
    att_act_row   = _row_as_dict(att, "Actual Attrition HC (#)")

    # attrition to use (add attr_delta within active window)
    att_use_row = {}
    for m in month_ids:
        base = float(att_act_row.get(m, 0)) if m <= today_m else float(att_plan_row.get(m, 0))
        if _wf_active_month(m):
            base += attr_delta
        att_use_row[m] = base

    # NH additions to use: actual for past/current, planned for future
    nh_add_row = {}
    for m in month_ids:
        nh_add_row[m] = float(actual_nh_m.get(m, 0) if m <= today_m else planned_nh_m.get(m, 0))

    projected_supply = {}
    prev = None
    for m in month_ids:
        if m <= today_m and hc_actual_row.get(m, 0) > 0:
            projected_supply[m] = hc_actual_row.get(m, 0)
            prev = projected_supply[m]
        else:
            if prev is None:
                prev = float(hc_plan_row.get(m, 0) or 0.0)
            next_val = max(prev - float(att_use_row.get(m, 0)) + float(nh_add_row.get(m, 0)), 0.0)
            projected_supply[m] = next_val
            prev = next_val

    # ---- Handling capacity & Projected SL (monthly) ----
    def _erlang_c(A: float, N: int) -> float:
        if N <= 0: return 1.0
        if A <= 0: return 0.0
        if A >= N: return 1.0
        s = 0.0
        for k in range(N):
            s += (A**k) / math.factorial(k)
        last = (A**N) / math.factorial(N) * (N / (N - A))
        p0 = 1.0 / (s + last)
        return last * p0

    def _erlang_sl(calls_per_ivl: float, aht_sec: float, agents: float, asa_sec: int, ivl_sec: int) -> float:
        if aht_sec <= 0 or ivl_sec <= 0 or agents <= 0 or calls_per_ivl <= 0:
            return 0.0
        A = (calls_per_ivl * aht_sec) / ivl_sec
        pw = _erlang_c(A, int(math.floor(agents)))
        return max(0.0, min(1.0, 1.0 - pw * math.exp(-max(0.0, (agents - A)) * (asa_sec / max(1.0, aht_sec)))))

    def _erlang_calls_capacity(agents: float, aht_sec: float, asa_sec: int, ivl_sec: int, target_pct: float) -> float:
        if agents <= 0 or aht_sec <= 0 or ivl_sec <= 0:
            return 0.0
        target = float(target_pct) / 100.0
        hi = max(1, int((agents * ivl_sec) / aht_sec))
        def sl_for(x): return _erlang_sl(x, aht_sec, agents, asa_sec, ivl_sec)
        lo = 0
        while sl_for(hi) >= target and hi < 10_000_000:
            lo = hi; hi *= 2
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if sl_for(mid) >= target: lo = mid
            else: hi = mid - 1
        return float(lo)

    handling_capacity = {}
    bo_model   = (settings.get("bo_capacity_model") or "tat").lower()
    bo_hpd     = float(settings.get("bo_hours_per_day", settings.get("hours_per_fte", 8.0)))
    util_bo    = float(settings.get("util_bo", 0.85))

    # base shrink (fraction)
    bo_shr_base = float(settings.get("bo_shrinkage_pct", settings.get("shrinkage_pct", 0.0)) or 0.0)
    if bo_shr_base > 1.0: bo_shr_base /= 100.0
    voice_shr_base = float(settings.get("voice_shrinkage_pct", settings.get("shrinkage_pct", 0.0)) or 0.0)
    if voice_shr_base > 1.0: voice_shr_base /= 100.0

    for m in month_ids:
        if ch_first.lower() == "voice":
            agents_prod = float(schedule_supply_avg_m.get(m, projected_supply.get(m, 0.0)))
            lc = _learning_curve_for_month(settings, lc_ovr_df, m)

            def eff_from_buckets(buckets, prod_pct_list, uplift_pct_list):
                total = 0.0
                login_f = _ovr_login_frac_m(m)
                aht_m   = _ovr_aht_mult_m(m)
                for age, cnt in (buckets.get(m, {}) or {}).items():
                    p = (prod_pct_list[age-1] / 100.0) if age-1 < len(prod_pct_list) else 0.0
                    u = (uplift_pct_list[age-1] / 100.0) if age-1 < len(uplift_pct_list) else 0.0
                    if login_f is not None: p *= login_f
                    denom = (1.0 + u)
                    if aht_m is not None:   denom *= aht_m
                    total += float(cnt) * (p / max(1.0, denom))
                return total

            nest_eff = eff_from_buckets(nest_buckets, lc["nesting_prod_pct"], lc["nesting_aht_uplift_pct"])
            sda_eff  = eff_from_buckets(sda_buckets,  lc["sda_prod_pct"],     lc["sda_aht_uplift_pct"])

            v_shr_add = (shrink_delta / 100.0) if (_wf_active_month(m) and shrink_delta) else 0.0
            v_eff_shr = min(0.99, max(0.0, voice_shr_base + v_shr_add))
            agents_eff = max(1.0, (agents_prod + nest_eff + sda_eff) * occ_frac_m[m] * (1.0 - v_eff_shr))

            aht = (vA_aht.get(m) if not np.isnan(vA_aht.get(m, np.nan)) else vF_aht.get(m, s_target_aht))
            aht = max(1.0, float(aht) * (1.0 + aht_delta / 100.0))
            intervals = monthly_voice_intervals.get(m, 0)
            calls_per_ivl = _erlang_calls_capacity(agents_eff, aht, sl_seconds, ivl_sec, sl_target_pct)
            handling_capacity[m] = calls_per_ivl * intervals
        else:
            # Backoffice
            sut = (bA_sut.get(m) if not np.isnan(bA_sut.get(m, np.nan)) else bF_sut.get(m, s_target_sut))
            sut = max(1.0, float(sut) * (1.0 + aht_delta / 100.0))
            lc = _learning_curve_for_month(settings, lc_ovr_df, m)
            def eff(buckets, prod_pct_list, uplift_pct_list):
                total = 0.0
                login_f = _ovr_login_frac_m(m)
                aht_m   = _ovr_aht_mult_m(m)
                for age, cnt in (buckets.get(m, {}) or {}).items():
                    p = (prod_pct_list[age-1] / 100.0) if age-1 < len(prod_pct_list) else 0.0
                    u = (uplift_pct_list[age-1] / 100.0) if age-1 < len(uplift_pct_list) else 0.0
                    if login_f is not None: p *= login_f
                    denom = (1.0 + u)
                    if aht_m is not None:   denom *= aht_m
                    total += float(cnt) * (p / max(1.0, denom))
                return total
            agents_eff = max(1.0, float(projected_supply.get(m, 0.0)) + eff(nest_buckets, lc["nesting_prod_pct"], lc["nesting_aht_uplift_pct"]) + eff(sda_buckets, lc["sda_prod_pct"], lc["sda_aht_uplift_pct"]))
            wd = _workdays_in_month(m, is_bo=True)
            if bo_model == "tat":
                shr_add = (shrink_delta / 100.0) if (_wf_active_month(m) and shrink_delta) else 0.0
                eff_shr = min(0.99, max(0.0, bo_shr_base + shr_add))
                base_prod_hours = wd * bo_hpd * (1.0 - eff_shr) * util_bo
                ot = float(overtime_m.get(m, 0.0) or 0.0)
                total_prod_hours = (float(agents_eff) * base_prod_hours) + (max(0.0, ot) * (1.0 - eff_shr) * util_bo)
                handling_capacity[m] = total_prod_hours * (3600.0 / sut)
            else:
                ivl_per_month = int(round(wd * bo_hpd / (ivl_sec / 3600.0)))
                items_per_ivl = _erlang_calls_capacity(max(1.0, float(agents_eff) * util_bo), sut, sl_seconds, ivl_sec, sl_target_pct)
                handling_capacity[m] = items_per_ivl * max(1, ivl_per_month)

    # projected service level
    proj_sl = {}
    for m in month_ids:
        if ch_first.lower() == "voice":
            if "Forecast" in fw_rows:
                try:
                    ser = pd.to_numeric(fw.loc[fw["metric"] == "Forecast", m], errors="coerce")
                    monthly_load = float(ser.iloc[0]) if not ser.empty and pd.notna(ser.iloc[0]) else 0.0
                except Exception:
                    monthly_load = float(vF_vol.get(m, 0.0))
            else:
                monthly_load = float(vF_vol.get(m, 0.0))
            aht_sut = (vA_aht.get(m) if not np.isnan(vA_aht.get(m, np.nan)) else vF_aht.get(m, s_target_aht))
            aht_sut = max(1.0, float(aht_sut) * (1.0 + aht_delta / 100.0))
            intervals = monthly_voice_intervals.get(m, 1)
            calls_per_ivl = monthly_load / float(max(1, intervals))
            lc = _learning_curve_for_month(settings, lc_ovr_df, m)
            def eff_from_buckets(buckets, prod_pct_list, uplift_pct_list):
                total = 0.0
                login_f = _ovr_login_frac_m(m)
                aht_m   = _ovr_aht_mult_m(m)
                for age, cnt in (buckets.get(m, {}) or {}).items():
                    p = (prod_pct_list[age-1] / 100.0) if age-1 < len(prod_pct_list) else 0.0
                    u = (uplift_pct_list[age-1] / 100.0) if age-1 < len(uplift_pct_list) else 0.0
                    if login_f is not None: p *= login_f
                    denom = (1.0 + u)
                    if aht_m is not None:   denom *= aht_m
                    total += float(cnt) * (p / max(1.0, denom))
                return total
            nest_eff = eff_from_buckets(nest_buckets, lc["nesting_prod_pct"], lc["nesting_aht_uplift_pct"])
            sda_eff  = eff_from_buckets(sda_buckets,  lc["sda_prod_pct"],     lc["sda_aht_uplift_pct"])
            v_shr_add = (shrink_delta / 100.0) if (_wf_active_month(m) and shrink_delta) else 0.0
            v_eff_shr = min(0.99, max(0.0, voice_shr_base + v_shr_add))
            agents_eff = max(1.0, (float(projected_supply.get(m, 0.0)) + nest_eff + sda_eff) * occ_frac_m[m] * (1.0 - v_eff_shr))
            sl_frac = _erlang_sl(calls_per_ivl, max(1.0, float(aht_sut)), agents_eff, sl_seconds, ivl_sec)
            proj_sl[m] = 100.0 * sl_frac
        else:
            monthly_load = (
                _get_fw_value(fw, "Forecast", m, 0.0)
                if "Forecast" in fw_rows else float(bF_itm.get(m, 0.0) or 0.0)
            )
            cap = float(handling_capacity.get(m, 0.0))
            proj_sl[m] = 0.0 if monthly_load <= 0 else min(100.0, 100.0 * cap / monthly_load)

    # ---- Upper summary table (same rows) ----
    upper_df = _blank_grid(spec["upper"], month_ids)
    if "FTE Required @ Forecast Volume" in spec["upper"]:
        for mm in month_ids:
            upper_df.loc[upper_df["metric"] == "FTE Required @ Forecast Volume", mm] = float(req_m_forecast.get(mm, 0.0))
    if "FTE Required @ Actual Volume" in spec["upper"]:
        for mm in month_ids:
            upper_df.loc[upper_df["metric"] == "FTE Required @ Actual Volume", mm] = float(req_m_actual.get(mm, 0.0))
    if "FTE Over/Under MTP Vs Actual" in spec["upper"]:
        for mm in month_ids:
            upper_df.loc[upper_df["metric"] == "FTE Over/Under MTP Vs Actual", mm] = float(req_m_forecast.get(mm, 0.0)) - float(req_m_actual.get(mm, 0.0))
    if "FTE Over/Under Tactical Vs Actual" in spec["upper"]:
        for mm in month_ids:
            upper_df.loc[upper_df["metric"] == "FTE Over/Under Tactical Vs Actual", mm] = float(req_m_tactical.get(mm, 0.0)) - float(req_m_actual.get(mm, 0.0))
    if "FTE Over/Under Budgeted Vs Actual" in spec["upper"]:
        for mm in month_ids:
            upper_df.loc[upper_df["metric"] == "FTE Over/Under Budgeted Vs Actual", mm] = float(req_m_budgeted.get(mm, 0.0)) - float(req_m_actual.get(mm, 0.0))
    if "Projected Supply HC" in spec["upper"]:
        for mm in month_ids:
            upper_df.loc[upper_df["metric"] == "Projected Supply HC", mm] = projected_supply.get(mm, 0.0)
    if "Projected Handling Capacity (#)" in spec["upper"]:
        for mm in month_ids:
            upper_df.loc[upper_df["metric"] == "Projected Handling Capacity (#)", mm] = handling_capacity.get(mm, 0.0)
    if "Projected Service Level" in spec["upper"]:
        for mm in month_ids:
            upper_df.loc[upper_df["metric"] == "Projected Service Level", mm] = proj_sl.get(mm, 0.0)

    # ---- rounding & display formatting ----
    def _round_cols_int(df, col_ids):  # month friendly
        if not isinstance(df, pd.DataFrame) or df.empty: return df
        out = df.copy()
        for c in col_ids:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).round(0).astype(int)
        return out

    fw_to_use = _round_cols_int(fw, month_ids)
    hc        = _round_cols_int(hc, month_ids)
    att       = _round_cols_int(att, month_ids)
    trn       = _round_cols_int(trn, month_ids)
    rat       = _round_cols_int(rat, month_ids)
    seat      = _round_cols_int(seat, month_ids)
    bva       = _round_cols_int(bva, month_ids)
    nh        = _round_cols_int(nh, month_ids)

    def _format_shrinkage_month(df):
        if not isinstance(df, pd.DataFrame) or df.empty: return df
        out = df.copy()
        pct_rows = out["metric"].astype(str).str.contains("Shrinkage %", regex=False)
        hr_rows  = out["metric"].astype(str).str.contains("Hours (#)",   regex=False)
        for c in month_ids:
            if c in out.columns: out[c] = out[c].astype(object)
        for c in month_ids:
            if c not in out.columns: continue
            out.loc[hr_rows,  c] = pd.to_numeric(out.loc[hr_rows,  c], errors="coerce").fillna(0).round(0).astype(int)
            vals = pd.to_numeric(out.loc[pct_rows, c], errors="coerce").fillna(0)
            out.loc[pct_rows, c] = vals.round(0).astype(int).astype(str) + "%"
        return out

    shr_display = _format_shrinkage_month(shr)

    # format upper: SL one decimal, others int
    if isinstance(upper_df, pd.DataFrame) and not upper_df.empty:
        for c in month_ids:
            if c not in upper_df.columns: continue
            mask_sl = upper_df["metric"].eq("Projected Service Level")
            mask_not_sl = ~mask_sl
            upper_df.loc[mask_sl, c] = pd.to_numeric(upper_df.loc[mask_sl, c], errors="coerce").fillna(0.0).round(1)
            upper_df.loc[mask_not_sl, c] = pd.to_numeric(upper_df.loc[mask_not_sl, c], errors="coerce").fillna(0.0).round(0).astype(int)

    upper = dash_table.DataTable(
        id="tbl-upper",
        data=upper_df.to_dict("records"),
        columns=[{"name": "Metric", "id": "metric", "editable": False}] +
                [{"name": c["name"], "id": c["id"]} for c in fw_cols if c["id"] != "metric"],
        editable=False,
        style_as_list_view=True,
        style_table={"overflowX": "auto"},
        style_header={"whiteSpace": "pre"},
    )

    # ---- Roster/Bulk/Notes passthroughs ----
    bulk_df  = _load_or_empty_bulk_files(pid)
    notes_df = _load_or_empty_notes(pid)

    return (
        upper,
        fw_to_use.to_dict("records"),
        hc.to_dict("records"),
        att.to_dict("records"),
        shr_display.to_dict("records"),
        trn.to_dict("records"),
        rat.to_dict("records"),
        seat.to_dict("records"),
        bva.to_dict("records"),
        nh.to_dict("records"),
        (roster_df.to_dict("records") if isinstance(roster_df, pd.DataFrame) else []),
        bulk_df.to_dict("records")  if isinstance(bulk_df,  pd.DataFrame) else [],
        notes_df.to_dict("records") if isinstance(notes_df, pd.DataFrame) else [],
    )
