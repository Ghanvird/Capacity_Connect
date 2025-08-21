# capacity_core.py — core math + samples + KPIs
from __future__ import annotations
import math, datetime as dt
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from itertools import cycle

# ─── Erlang / Queueing ───────────────────────────────────────
def erlang_b(A: float, N: int) -> float:
    if N <= 0: return 1.0
    B = 1.0
    for n in range(1, N+1):
        B = (A * B) / (n + A * B)
    return B

def erlang_c(A: float, N: int) -> float:
    if N <= 0: return 1.0
    if A <= 0: return 0.0
    if N <= A: return 1.0
    rho = A / N
    B = erlang_b(A, N)
    denom = 1 - rho + rho * B
    if denom <= 0: return 1.0
    return B / denom

def service_level(A: float, N: int, aht_sec: float, T_sec: float) -> float:
    if N <= 0: return 0.0
    if A <= 0: return 1.0
    pw = erlang_c(A, N)
    gap = N - A
    if gap <= 0: return 0.0
    return 1.0 - pw * math.exp(-gap * (T_sec / max(aht_sec, 1e-9)))

def asa(A: float, N: int, aht_sec: float) -> float:
    if N <= 0 or A <= 0 or N <= A: return float("inf")
    pw = erlang_c(A, N)
    return (pw * aht_sec) / (N - A)

def offered_load_erlangs(calls: float, aht_sec: float, interval_minutes: int) -> float:
    interval_minutes = max(5, int(interval_minutes or 30))
    return (calls * aht_sec) / (interval_minutes * 60.0)

def min_agents(calls: float, aht_sec: float, ivl_min: int, target_sl: float, T: float,
               occ_cap: Optional[float] = None, asa_cap: Optional[float] = None, Ncap: int = 2000) -> Tuple[int,float,float,float]:
    A = offered_load_erlangs(calls, aht_sec, ivl_min)
    if A <= 0: return 0, 1.0, 0.0, 0.0
    start = max(1, math.ceil(A))
    for N in range(start, min(start+1000, Ncap)):
        sl = service_level(A, N, aht_sec, T)
        occ = A / N
        _asa = asa(A, N, aht_sec)
        ok = True
        if occ_cap is not None and occ > occ_cap: ok = False
        if target_sl is not None and sl < target_sl: ok = False
        if asa_cap is not None and _asa > asa_cap: ok = False
        if ok: return N, sl, occ, _asa
    N = min(start+1000, Ncap) - 1
    return N, service_level(A, N, aht_sec, T), A/max(N,1), asa(A, N, aht_sec)

# ─── Samples ─────────────────────────────────────────────────
def make_projects_sample() -> pd.DataFrame:
    # ---- Preferred: query the DB directly (fast) ----
    try:
        from cap_store import _conn  # your existing DB connector

        with _conn() as cx:
            rows = cx.execute(
                """
                SELECT COALESCE(vertical, '') AS vertical, COUNT(*) AS cnt
                FROM capacity_plans
                WHERE (is_current = 1)
                   OR (LOWER(COALESCE(status, '')) = 'current')
                GROUP BY vertical
                ORDER BY vertical
                """
            ).fetchall()

        data = [{"Business Area": (r["vertical"] or "—"), "Active Plans": int(r["cnt"])} for r in rows]
        return pd.DataFrame(data, columns=["Business Area", "Active Plans"])

    except Exception:
        # ---- Fallback: use the plan_store API if direct SQL isn't available ----
        try:
            from plan_store import list_plans

            rows = list_plans(status_filter="current") or []
            if not rows:
                return pd.DataFrame(columns=["Business Area", "Active Plans"])

            df = pd.DataFrame(rows)

            # Keep only current/active
            if "is_current" in df.columns:
                df = df[df["is_current"] == 1]
            elif "status" in df.columns:
                df = df[df["status"].astype(str).str.lower() == "current"]

            if df.empty:
                return pd.DataFrame(columns=["Business Area", "Active Plans"])

            # Find the BA column name
            ba_col = (
                "vertical" if "vertical" in df.columns else
                "business_area" if "business_area" in df.columns else
                None
            )
            if not ba_col:
                return pd.DataFrame(columns=["Business Area", "Active Plans"])

            grp = (
                df.groupby(ba_col)
                  .size()
                  .reset_index(name="Active Plans")
                  .rename(columns={ba_col: "Business Area"})
                  .sort_values("Business Area", kind="stable")
                  .reset_index(drop=True)
            )
            grp["Active Plans"] = grp["Active Plans"].astype(int)
            return grp[["Business Area", "Active Plans"]]

        except Exception:
            # Nothing we can pull — return the expected empty structure
            return pd.DataFrame(columns=["Business Area", "Active Plans"])

def make_voice_sample(interval_minutes: int = 30, days: int = 5) -> pd.DataFrame:
    today = dt.date.today()
    rows: List[dict] = []
    rng = np.random.default_rng(42)
    for d in range(days):
        date = today + dt.timedelta(days=d)
        start = dt.datetime.combine(date, dt.time(9,0))
        end   = dt.datetime.combine(date, dt.time(18,0))
        t = start
        while t < end:
            rows.append({"date": date, "interval": t.time().strftime("%H:%M"),
                         "volume": int(rng.integers(20,80)), "aht_sec": 300, "program":"WFM"})
            t += dt.timedelta(minutes=interval_minutes)
    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"]).dt.date
    return out

def make_backoffice_sample(days: int = 5) -> pd.DataFrame:
    """Backoffice demo data; all columns length == days."""
    today = dt.date.today()
    dates = [(today + dt.timedelta(days=i)).isoformat() for i in range(days)]
    rng = np.random.default_rng(7)

    sub_tasks_cycle = cycle(["Case Review", "KYB", "Doc Check", "QA", "QC"])

    return pd.DataFrame({
        "date": dates,
        "items": rng.integers(200, 600, size=days),
        "aht_sec": np.full(days, 600),
        "sub_task": [next(sub_tasks_cycle) for _ in range(days)],
        "program": ["WFM"] * days,
    })

def make_outbound_sample(days: int = 5) -> pd.DataFrame:
    """Outbound demo data; all columns length == days."""
    today = dt.date.today()
    dates = [(today + dt.timedelta(days=i)).isoformat() for i in range(days)]
    rng = np.random.default_rng(11)

    campaigns_cycle = cycle(["Retention", "Welcome", "NPS", "Upsell", "Collections"])

    return pd.DataFrame({
        "date": dates,
        "calls": rng.integers(300, 700, size=days),
        "aht_sec": np.full(days, 240),
        "campaign": [next(campaigns_cycle) for _ in range(days)],
        "program": ["WFM"] * days,
    })

def make_roster_sample() -> pd.DataFrame:
    today = dt.date.today()
    return pd.DataFrame({
        "employee_id":["101","102","103","104"],
        "name":["Asha","Bala","Chan","Drew"],
        "status":["Active"]*4,
        "employment_type":["FT","FT","FT","PT"],
        "fte":[1.0,1.0,1.0,0.5],
        "contract_hours_per_week":[40,40,40,20],
        "country":["India","India","India","India"],
        "site":["Chennai"]*4, "timezone":["Asia/Kolkata"]*4,
        "program":["WFM"]*4, "sub_business_area":["Retail"]*4, "lob":["Cards"]*4, "channel":["Voice"]*4,
        "skill_voice":[True,True,False,True], "skill_bo":[True,False,True,True], "skill_ob":[False,True,True,False],
        "start_date":[today.isoformat()]*4, "end_date":[""]*4
    })

def make_hiring_sample() -> pd.DataFrame:
    today = dt.date.today()
    sow = today - dt.timedelta(days=today.weekday())  # Monday
    return pd.DataFrame({
        "start_week":[(sow - dt.timedelta(weeks=1)).isoformat(), sow.isoformat(), (sow + dt.timedelta(weeks=1)).isoformat()],
        "fte":[2,0,5], "program":["WFM","WFM","WFM"], "country":["India"]*3, "site":["Chennai"]*3
    })

def make_shrinkage_sample() -> pd.DataFrame:
    today = dt.date.today()
    sow = today - dt.timedelta(days=today.weekday())
    return pd.DataFrame({"week":[(sow - dt.timedelta(weeks=i)).isoformat() for i in [3,2,1,0]],
                         "shrinkage_pct":[10.0, 11.2, 10.8, 11.0], "program":["WFM"]*4})

def make_attrition_sample() -> pd.DataFrame:
    today = dt.date.today()
    sow = today - dt.timedelta(days=today.weekday())
    return pd.DataFrame({"week":[(sow - dt.timedelta(weeks=i)).isoformat() for i in [3,2,1,0]],
                         "attrition_pct":[0.7, 0.8, 0.9, 0.85], "program":["WFM"]*4})

# ─── Core daily requirement/supply ───────────────────────────
def required_fte_daily(voice_df: pd.DataFrame, bo_df: pd.DataFrame, ob_df: pd.DataFrame, settings: dict) -> pd.DataFrame:
    frames = []
    # Voice per-interval → daily
    if isinstance(voice_df, pd.DataFrame) and not voice_df.empty:
        vrows = []
        for _, r in voice_df.iterrows():
            calls = float(r.get("volume", 0) or 0)
            aht   = float(r.get("aht_sec", 0) or 0)
            A     = offered_load_erlangs(calls, aht, int(settings["interval_minutes"]))
            # use min agents; convert to FTE via shrinkage
            N, sl, occ, asa_val = min_agents(calls, aht, int(settings["interval_minutes"]),
                                             float(settings["target_sl"]), float(settings["sl_seconds"]),
                                             float(settings["occupancy_cap_voice"]))
            fte = N / max(1e-6, (1 - float(settings["shrinkage_pct"])))
            vrows.append({"date": pd.to_datetime(r["date"]).date(), "program": r.get("program","WFM"),
                          "fte_req": fte})
        v = pd.DataFrame(vrows).groupby(["date","program"], as_index=False)["fte_req"].sum().rename(columns={"fte_req":"voice_fte"})
        frames.append(v)

    # Backoffice (daily)
    if isinstance(bo_df, pd.DataFrame) and not bo_df.empty:
        denom = float(settings["hours_per_fte"]) * 3600.0 * (1 - float(settings["shrinkage_pct"])) * float(settings["util_bo"])
        b = bo_df.copy()
        b["date"] = pd.to_datetime(b["date"]).dt.date
        b["bo_fte"] = b.apply(lambda r: (float(r["items"]) * float(r["aht_sec"])) / max(denom, 1e-6), axis=1)
        b = b.groupby(["date","program"], as_index=False)["bo_fte"].sum()
        frames.append(b)

    # Outbound (daily)
    if isinstance(ob_df, pd.DataFrame) and not ob_df.empty:
        denom = float(settings["hours_per_fte"]) * 3600.0 * (1 - float(settings["shrinkage_pct"])) * float(settings["util_ob"])
        o = ob_df.copy()
        o["date"] = pd.to_datetime(o["date"]).dt.date
        o["ob_fte"] = o.apply(lambda r: (float(r["calls"]) * float(r["aht_sec"])) / max(denom, 1e-6), axis=1)
        o = o.groupby(["date","program"], as_index=False)["ob_fte"].sum()
        frames.append(o)

    if not frames:
        return pd.DataFrame(columns=["date","program","voice_fte","bo_fte","ob_fte","total_req_fte"])
    out = frames[0]
    for f in frames[1:]:
        out = pd.merge(out, f, on=["date","program"], how="outer")
    for c in ["voice_fte","bo_fte","ob_fte"]:
        if c not in out: out[c] = 0.0
    out["total_req_fte"] = out[["voice_fte","bo_fte","ob_fte"]].fillna(0).sum(axis=1)
    return out.fillna(0)

def supply_fte_daily(roster: pd.DataFrame, hiring: pd.DataFrame) -> pd.DataFrame:
    if roster is None or roster.empty:
        return pd.DataFrame(columns=["date","program","supply_fte"])
    # expand roster by date
    rows: List[dict] = []
    today = dt.date.today()
    horizon = today + dt.timedelta(days=28)
    date_list = [today + dt.timedelta(days=i) for i in range((horizon - today).days + 1)]
    for _, r in roster.iterrows():
        try:
            sd = pd.to_datetime(r.get("start_date")).date()
        except Exception:
            sd = today
        ed_val = r.get("end_date", "")
        ed = pd.to_datetime(ed_val).date() if str(ed_val).strip() else horizon
        fte = float(r.get("fte", 1.0) or 0.0)
        prog = r.get("program", "WFM")
        stat = str(r.get("status","Active")).strip().lower()
        if stat and stat != "active":
            continue
        for d in date_list:
            if sd <= d <= ed:
                rows.append({"date": d, "program": prog, "fte": fte})
    sup = pd.DataFrame(rows)
    if sup.empty:
        return pd.DataFrame(columns=["date","program","supply_fte"])
    sup = sup.groupby(["date","program"], as_index=False)["fte"].sum().rename(columns={"fte":"supply_fte"})

    # add hiring by week
    if isinstance(hiring, pd.DataFrame) and not hiring.empty and "start_week" in hiring.columns:
        add_rows: List[dict] = []
        for _, r in hiring.iterrows():
            try:
                ws = pd.to_datetime(r["start_week"]).date()
            except Exception:
                continue
            prog = r.get("program","WFM")
            f = float(r.get("fte",0) or 0)
            for i in range(7):
                add_rows.append({"date": ws + dt.timedelta(days=i), "program": prog, "supply_fte": f})
        add = pd.DataFrame(add_rows)
        if not add.empty:
            sup = pd.concat([sup, add]).groupby(["date","program"], as_index=False)["supply_fte"].sum()
    sup["date"] = pd.to_datetime(sup["date"]).dt.date
    return sup

# ─── KPIs ────────────────────────────────────────────────────
def kpi_hiring(hiring_df: pd.DataFrame) -> Tuple[float,float,float]:
    if hiring_df is None or hiring_df.empty: return 0,0,0
    today = dt.date.today()
    sow = today - dt.timedelta(days=today.weekday())
    lw, tw, nw = sow - dt.timedelta(weeks=1), sow, sow + dt.timedelta(weeks=1)
    def wk(x): 
        try: return pd.to_datetime(x).date()
        except: return None
    ser = hiring_df["start_week"].map(wk)
    fte = hiring_df["fte"].astype(float)
    last = fte[ser==lw].sum(); this = fte[ser==tw].sum(); nxt = fte[ser==nw].sum()
    return float(last), float(this), float(nxt)

def kpi_shrinkage(shrink_df: pd.DataFrame) -> Tuple[float,float]:
    if shrink_df is None or shrink_df.empty: return 0.0, 0.0
    df = shrink_df.copy()
    df["week"] = pd.to_datetime(df["week"], errors="coerce")
    df = df.dropna(subset=["week"])
    df = df.sort_values("week")
    last4 = df.tail(4)["shrinkage_pct"].astype(float).mean()
    # "next4" placeholder same as last value for now (until you project)
    next4 = df.tail(1)["shrinkage_pct"].astype(float).mean()
    return float(last4 or 0.0), float(next4 or 0.0)

def understaffed_accounts_next_4w(req: pd.DataFrame, sup: pd.DataFrame) -> int:
    if req is None or req.empty: return 0
    if sup is None or sup.empty: return 0
    df = pd.merge(req.copy(), sup.copy(), on=["date","program"], how="left").fillna({"supply_fte":0})
    df["gap"] = df["supply_fte"] - df["total_req_fte"]
    by_prog = df.groupby("program")["gap"].min()
    return int((by_prog < 0).sum())


def _last_next_4(df: pd.DataFrame, week_col: str, value_col: str):
    """Return (avg of last 4 weeks, avg of next 4 weeks) based on current Monday."""
    if df is None or df.empty:
        return 0.0, 0.0
    tmp = df.copy()
    tmp[week_col] = pd.to_datetime(tmp[week_col]).dt.date
    today = dt.date.today()
    sow = today - dt.timedelta(days=today.weekday())  # Monday
    past   = tmp[tmp[week_col] <= sow].sort_values(week_col).tail(4)
    future = tmp[tmp[week_col] >  sow].sort_values(week_col).head(4)
    last4 = float(past[value_col].mean()) if not past.empty else 0.0
    next4 = float(future[value_col].mean()) if not future.empty else last4
    return last4, next4

