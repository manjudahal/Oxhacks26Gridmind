# GridMind: decides every 30 minutes when a grid battery should charge or discharge,
# using real GB electricity market data from Elexon.  Run locally with:  streamlit run streamlit_app.py
#
# Tabs:   Live decision  -> what the battery should do in the next half-hour, and why
#         Gemini Copilot -> Google Gemini explains decisions and runs GridMind's own optimiser via function calling
#         Replay a day   -> what GridMind did on any past test day (works offline)
#         Decision Lab   -> re-run any test day with your own battery settings and risk preference
#         Results        -> backtest, where the value comes from, is it luck, robustness
#         Explainability -> what drives the forecasts
#         How it works   -> the method in plain English
import json
import os
import threading
from datetime import datetime, timedelta, timezone

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import gridmind_core as core   # the notebook's own functions: data download, cleaning, features, models, optimiser

try:                              # Google Gemini API (google-genai SDK); the app works without it
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

st.set_page_config(page_title="GridMind", page_icon="⚡", layout="wide")
HERE = os.path.dirname(os.path.abspath(__file__))
GREEN, BLUE, CORAL, PURPLE, GRAY, AMBER = "#1D9E75", "#378ADD", "#D85A30", "#534AB7", "#888780", "#BA7517"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
UK = "Europe/London"


# ------------------------------------------------------------------ load saved files once (cached across visitors)
@st.cache_resource
def load_all():
    def js(name):
        p = os.path.join(HERE, name)
        return json.load(open(p)) if os.path.exists(p) else {}
    bundle = joblib.load(os.path.join(HERE, "gridmind_model.joblib"))   # trained models + settings chosen on validation
    rp_path = os.path.join(HERE, "gridmind_replay.csv")
    rp = pd.read_csv(rp_path, parse_dates=["startTime"]) if os.path.exists(rp_path) else pd.DataFrame()
    if not rp.empty:
        rp["day"] = rp.startTime.dt.tz_convert(UK).dt.date
    pp = os.path.join(HERE, "gridmind_paths.npz")       # the 44 expected prices GridMind planned with, per test half-hour
    paths = dict(np.load(pp)) if os.path.exists(pp) else {}
    return bundle, js("gridmind_results.json"), js("gridmind_snapshot.json"), rp, paths

B, R, SNAP, RP, PATHS = load_all()
core.FEATURES = B["features"]


# ------------------------------------------------------------------ live decision
@st.cache_data(ttl=900, show_spinner=False)     # download at most every 15 minutes, shared by all visitors
def live_raw():
    now = datetime.now(timezone.utc)
    prices, fuel, mid, windf, demf = core.fetch_all((now - timedelta(days=31)).date(), now.date())
    if prices.empty:
        raise RuntimeError("Elexon returned no price data. Please try again in a minute.")
    return core.combine(prices, fuel, mid, windf, demf, extend_periods=52)

def recommend(raw, now, soc):
    f = core.build_features(core.clean_data(raw, verbose=False))
    return core.live_plan(f, now, B["kind"], B["model"], B["res_q"], B.get("mh_model"), B["alpha"],
                          B.get("beta", 0.0), B["kappa"], soc, B.get("explain_kind"), B.get("explain_model"))

def show_decision(rec, soc, note):
    icon = {"DISCHARGE (sell)": "🟢", "CHARGE (buy)": "🔵", "HOLD": "⚪"}[rec["action"]]
    st.subheader(f"{icon} {rec['action']} at {rec['mw']:.0f} MW")
    st.caption(f"Half-hour starting {rec['time'].tz_convert(UK):%a %d %b, %H:%M} (UK time) · battery at {soc:.0f} of 100 MWh")
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted imbalance price", f"£{rec['q50']:,.0f}/MWh")
    c2.metric("80% range", f"£{rec['q10']:,.0f} – £{rec['q90']:,.0f}")
    c3.metric("Typical at this time of day", f"£{rec['typical']:,.0f}/MWh")
    st.markdown(f"**Why:** {rec['why']}")
    if rec.get("drivers"):
        if rec.get("drivers_note"):
            st.caption(f"Forecast method: {B.get('best', 'chosen model')} (picked on validation). The drivers below come "
                       "from our LightGBM model and explain what is moving the price right now.")
        st.markdown(f"**Main drivers of the forecast{rec.get('drivers_note', '')}:**\n" + "\n".join(
            f"- {'🔺' if v >= 0 else '🔻'} {n}: {'+' if v >= 0 else '−'}£{abs(v):,.0f}/MWh" for n, v in rec["drivers"]))
    st.pyplot(live_figure(rec))
    st.caption(note)

def live_figure(rec):
    fig, ax = plt.subplots(1, 2, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1, 1.4]})
    r = rec["recent"]
    ax[0].plot(r.startTime.dt.tz_convert(UK), r.price, color=CORAL, lw=1.8, label="Real imbalance price")
    ax[0].errorbar([rec["time"].tz_convert(UK)], [rec["q50"]], yerr=[[rec["q50"] - rec["q10"]], [rec["q90"] - rec["q50"]]],
                   fmt="o", color=PURPLE, capsize=6, label="Our forecast (80% range)")
    ax[0].set_title("Last 48 hours + next half-hour"); ax[0].set_ylabel("£/MWh")
    ax[0].legend(fontsize=8); ax[0].tick_params(axis="x", rotation=30)
    pt = rec["plan_times"].tz_convert(UK)
    ax[1].bar(pt, rec["plan_mw"], width=0.018, color=np.where(rec["plan_mw"] > 0, GREEN, BLUE))
    ax[1].axhline(0, color="black", lw=0.6); ax[1].set_ylabel("Planned MW (+ sell, − buy)")
    tw = ax[1].twinx(); tw.plot(pt, rec["plan_price"], color=GRAY, lw=1.2); tw.set_ylabel("Expected £/MWh")
    tw.spines["right"].set_visible(True)
    ax[1].set_title("22-hour plan (re-planned every half-hour)"); ax[1].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig

def snapshot_rec():
    rec = dict(SNAP)
    rec["time"] = pd.Timestamp(SNAP["time"])
    rec["plan_times"] = pd.DatetimeIndex(pd.to_datetime(SNAP["plan_times"], utc=True))
    rec["plan_mw"], rec["plan_price"] = np.array(SNAP["plan_mw"]), np.array(SNAP["plan_price"])
    rec["recent"] = pd.DataFrame({"startTime": pd.to_datetime(SNAP["recent_times"], utc=True), "price": SNAP["recent_price"]})
    return rec


# ------------------------------------------------------------------ replay a past day (no internet needed)
def replay_figure(d):
    t = d.startTime.dt.tz_convert(UK)
    fig, ax = plt.subplots(2, 1, figsize=(13, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax[0].fill_between(t, d.q10, d.q90, color=PURPLE, alpha=0.15, label="Forecast 80% range")
    ax[0].plot(t, d.q50, color=PURPLE, lw=1.2, label="Forecast")
    ax[0].plot(t, d.price, color=CORAL, lw=2, label="Actual price")
    ax[0].set_ylabel("£/MWh"); ax[0].legend(fontsize=8, loc="upper left")
    mw = d.gm_action * core.P_MAX_MW
    ax[1].bar(t, mw, width=0.018, color=np.where(mw > 0, GREEN, BLUE), label="GridMind (+ sell, − buy)")
    ax[1].step(t, d.fixed_action * core.P_MAX_MW, where="mid", color=GRAY, lw=1, label="Fixed schedule")
    ax[1].set_ylabel("MW"); ax[1].legend(fontsize=8, loc="upper left")
    tw = ax[1].twinx(); tw.plot(t, d.gm_soc, color="black", lw=1, ls=":"); tw.set_ylabel("Battery MWh")
    tw.spines["right"].set_visible(True); ax[1].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ Decision Lab: re-run a test day with other settings
_LAB_LOCK = threading.Lock()
MODES = ["Balanced (chosen settings)", "Conservative (bigger hurdle when the forecast is uncertain)",
         "Aggressive (no hurdle)"]

@st.cache_data(show_spinner=False)
def lab_run(day, soc0, rte, deg, mode):
    d = RP[RP.day == day]
    paths, hw = PATHS["paths"][d.index], PATHS["half_width"][d.index]
    price = d.price.values
    with _LAB_LOCK:                                   # battery settings live in the core module: change, run, restore
        saved = (core.RTE, core.ETA, core.DEG_COST)
        core.RTE, core.ETA, core.DEG_COST = rte, rte ** 0.5, deg
        try:
            soc, acts, socs, pnls = soc0, [], [], []
            for i in range(len(d)):
                k = {MODES[0]: B["kappa"], MODES[1]: B["kappa"] + 0.25 * float(hw[i]), MODES[2]: 0.0}[mode]
                ch, dis = core.lp_schedule(soc, paths[i] + k, paths[i] - k)
                a = 0.0 if ch is None else (dis[0] - ch[0]) / core.E_STEP
                soc, _, _, pnl = core.step(soc, a, price[i])
                acts.append(a); socs.append(soc); pnls.append(pnl)
            loc = d.startTime.dt.tz_convert(UK)
            fx = core.run_fixed(pd.DataFrame({"startTime": d.startTime, "hour": loc.dt.hour + loc.dt.minute / 60,
                                              "price": price}), soc=soc0)
        finally:
            core.RTE, core.ETA, core.DEG_COST = saved
    return np.array(acts), np.array(socs), np.array(pnls), fx.action.values, fx.pnl.values

def lab_figure(d, acts, socs):
    t = d.startTime.dt.tz_convert(UK)
    fig, ax = plt.subplots(2, 1, figsize=(13, 5.5), sharex=True, gridspec_kw={"height_ratios": [1.4, 1]})
    ax[0].plot(t, d.price, color=CORAL, lw=2, label="Actual price")
    ax[0].plot(t, d.q50, color=PURPLE, lw=1.2, label="Forecast at the time")
    ax[0].set_ylabel("£/MWh"); ax[0].legend(fontsize=8, loc="upper left")
    mw = acts * core.P_MAX_MW
    ax[1].bar(t, mw, width=0.018, color=np.where(mw > 0, GREEN, BLUE), label="Decision (+ sell, − buy)")
    ax[1].set_ylabel("MW"); ax[1].legend(fontsize=8, loc="upper left")
    tw = ax[1].twinx(); tw.plot(t, socs, color="black", lw=1, ls=":"); tw.set_ylabel("Battery MWh")
    tw.spines["right"].set_visible(True); ax[1].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig

# ------------------------------------------------------------------ Gemini Copilot
# Roles are kept separate: GridMind's Python models make every forecast and the optimiser makes every decision.
# Gemini explains them in plain English and, through FUNCTION CALLING, can run GridMind's own optimiser and
# backtest tools to answer what-if questions with real computed numbers instead of guesses.
def secret(name, default=None):
    # st.secrets raises when no secrets file exists (e.g. a local run), so fall back to environment variables
    try:
        v = st.secrets.get(name)
        if v:
            return v
    except Exception:
        pass
    return os.environ.get(name, default)

GEMINI_KEY = secret("GEMINI_API_KEY")
GEMINI_MODELS = list(dict.fromkeys([secret("GEMINI_MODEL", "gemini-2.5-flash"), "gemini-2.5-flash", "gemini-flash-latest"]))
RISK = {"balanced": MODES[0], "conservative": MODES[1], "aggressive": MODES[2]}

SYSTEM_PROMPT = """You are GridMind Copilot, an assistant for an energy-trading desk and for hackathon judges.
GridMind decides every 30 minutes whether a simulated 50 MW / 100 MWh grid battery in Great Britain should charge,
discharge or hold. Its Python models forecast the Elexon imbalance price; a mixed-integer optimiser plans the next
22 hours (88% round-trip efficiency, wear cost, 5-95 MWh limits, never charging and discharging at once), acts on the
first half-hour and re-plans. You EXPLAIN GridMind; you do not make the forecasts or decisions yourself.
Rules:
- Use only the facts provided or numbers returned by your tools. Never invent prices, dates or results.
- For any what-if or historical question, CALL A TOOL and quote its numbers.
- All money figures are SIMULATED margins at Elexon imbalance (settlement) prices, not achievable revenue.
- Be concise and concrete: under 150 words unless asked for more, GBP, UK times, no bullet lists longer than 4 items.
"""

@st.cache_resource
def gemini_client(key):
    return genai.Client(api_key=key)

def _local_times(idx):
    t = pd.DatetimeIndex(idx)
    return (t if t.tz is not None else t.tz_localize("UTC")).tz_convert(UK)

def _plan_summary(times, path, plan_mw):
    sell, buy = plan_mw > 0.5, plan_mw < -0.5
    return {"planned_charge_times_uk": [f"{t:%a %H:%M}" for t in times[buy]][:10],
            "planned_discharge_times_uk": [f"{t:%a %H:%M}" for t in times[sell]][:10],
            "avg_expected_price_when_charging": round(float(path[buy].mean()), 1) if buy.any() else None,
            "avg_expected_price_when_discharging": round(float(path[sell].mean()), 1) if sell.any() else None}

def decision_facts(rec, soc):
    path, mw = np.asarray(rec["plan_price"], float), np.asarray(rec["plan_mw"], float)
    facts = {"half_hour_starting_uk": f"{rec['time'].tz_convert(UK):%a %d %b %Y %H:%M}",
             "decision": rec["action"], "power_mw": round(float(rec["mw"]), 1), "battery_now_mwh": soc,
             "forecast_price_gbp_mwh": round(float(rec["q50"]), 1),
             "forecast_80pct_range": [round(float(rec["q10"]), 1), round(float(rec["q90"]), 1)],
             "typical_price_at_this_time": round(float(rec["typical"]), 1),
             "gridmind_reason": rec["why"], "forecast_method": B.get("best"),
             "main_forecast_drivers_gbp_mwh": {n: round(float(v), 1) for n, v in rec.get("drivers", [])},
             "drivers_source": "LightGBM model (TreeSHAP)" + (" used to interpret the price" if rec.get("drivers_note") else ""),
             "trading_hurdle_gbp_mwh": B["kappa"]}
    facts.update(_plan_summary(_local_times(rec["plan_times"]), path, mw))
    return facts

def _clamp(x, lo, hi, default):
    try:
        return float(min(max(float(x), lo), hi))
    except Exception:
        return default

# ---- tools Gemini can call (plain Python functions: the SDK reads their names, types and docstrings)
def what_if_now(battery_mwh: float, efficiency_percent: float = 88.0, wear_cost_gbp_per_mwh: float = 5.0,
                risk_mode: str = "balanced") -> dict:
    """Re-run GridMind's optimiser for the CURRENT decision (the half-hour on the Live decision tab) with a different
    battery charge (5-95 MWh), round-trip efficiency (%), wear cost (GBP per MWh discharged) or risk mode
    ('balanced', 'conservative' or 'aggressive'). Returns the new action, power and when the 22-hour plan charges and discharges."""
    rec = st.session_state.get("rec")
    if rec is None:
        return {"error": "no current decision available"}
    soc = _clamp(battery_mwh, core.SOC_MIN, core.SOC_MAX, 50.0)
    rte = _clamp(efficiency_percent, 50, 100, 88.0) / 100
    deg = _clamp(wear_cost_gbp_per_mwh, 0, 100, 5.0)
    mode = str(risk_mode).lower() if str(risk_mode).lower() in RISK else "balanced"
    path = np.asarray(rec["plan_price"], float)
    k = {"balanced": B["kappa"], "conservative": B["kappa"] + 0.25 * (rec["q90"] - rec["q10"]) / 2, "aggressive": 0.0}[mode]
    with _LAB_LOCK:
        saved = (core.RTE, core.ETA, core.DEG_COST)
        core.RTE, core.ETA, core.DEG_COST = rte, rte ** 0.5, deg
        try:
            ch, dis = core.lp_schedule(soc, path + k, path - k)
        finally:
            core.RTE, core.ETA, core.DEG_COST = saved
    a = 0.0 if ch is None else (dis[0] - ch[0]) / core.E_STEP
    out = {"half_hour_starting_uk": f"{rec['time'].tz_convert(UK):%a %d %b %H:%M}",
           "decision": "DISCHARGE (sell)" if a > 1e-6 else "CHARGE (buy)" if a < -1e-6 else "HOLD",
           "power_mw": round(abs(a) * core.P_MAX_MW, 1), "battery_mwh": soc, "efficiency_percent": rte * 100,
           "wear_cost_gbp_per_mwh": deg, "risk_mode": mode, "hurdle_gbp_mwh": round(float(k), 1),
           "expected_price_now": round(float(path[0]), 1)}
    if ch is not None:
        out.update(_plan_summary(_local_times(rec["plan_times"]), path, (dis - ch) / core.DT))
    return out

def replay_test_day(date: str, battery_mwh: float = 50.0, efficiency_percent: float = 88.0,
                    wear_cost_gbp_per_mwh: float = 5.0, risk_mode: str = "balanced") -> dict:
    """Re-run GridMind on a historical TEST day (date as YYYY-MM-DD; test days were never used for training or tuning)
    with a chosen starting battery charge, efficiency (%), wear cost and risk mode ('balanced', 'conservative',
    'aggressive'), using exactly the forecasts it had at the time. Returns GridMind's simulated margin, the fixed
    timetable's margin with the same battery, the number of half-hours traded and the day's price range."""
    if RP.empty or not PATHS:
        return {"error": "replay data not available"}
    try:
        day = pd.Timestamp(date).date()
    except Exception:
        return {"error": f"could not read the date {date!r}; use YYYY-MM-DD"}
    days = sorted(RP.day.unique())
    if day not in days:
        return {"error": f"{day} is not a test day; test days run from {days[0]} to {days[-1]}"}
    mode = str(risk_mode).lower() if str(risk_mode).lower() in RISK else "balanced"
    acts, socs, pnls, fx_a, fx_p = lab_run(day, _clamp(battery_mwh, 5, 95, 50.0),
                                          _clamp(efficiency_percent, 50, 100, 88.0) / 100,
                                          _clamp(wear_cost_gbp_per_mwh, 0, 100, 5.0), RISK[mode])
    d = RP[RP.day == day]
    i_best = int(np.argmax(pnls))
    return {"date": str(day), "risk_mode": mode, "gridmind_simulated_margin_gbp": round(float(pnls.sum())),
            "fixed_timetable_simulated_margin_gbp": round(float(fx_p.sum())),
            "difference_gbp": round(float(pnls.sum() - fx_p.sum())),
            "half_hours_traded": int((np.abs(acts) > 0.01).sum()), "half_hours_in_day": int(len(d)),
            "price_min_gbp_mwh": round(float(d.price.min()), 1), "price_max_gbp_mwh": round(float(d.price.max()), 1),
            "best_half_hour_uk": f"{d.startTime.iloc[i_best].tz_convert(UK):%H:%M}",
            "best_half_hour_margin_gbp": round(float(pnls[i_best])), "battery_at_end_mwh": round(float(socs[-1]), 1)}

def backtest_results() -> dict:
    """Headline results of GridMind's backtest on the unseen test months: simulated margins of every strategy,
    the block-bootstrap significance test, forecast-range coverage, the ablation (what each component adds),
    balanced vs conservative risk modes, and which forecasting model was chosen on validation."""
    keep = ("Strategy", "Profit (GBP)", "GBP/MW/year", "Worst day (GBP)", "Losing days", "% of max", "vs fixed")
    return {"chosen_next_half_hour_model": meta.get("best"), "test_days": meta.get("test_days"),
            "forecast_80pct_range_coverage": meta.get("coverage_test"),
            "strategies": [{k: r.get(k) for k in keep} for r in R.get("backtest", [])],
            "significance": R.get("significance", []), "ablation": R.get("ablation", []),
            "risk_modes": R.get("risk", []), "note": "all figures are simulated margins at Elexon imbalance prices"}

def test_days_overview() -> dict:
    """The test days available for replay: first and last date, plus the five most volatile and five calmest days
    with GridMind's and the fixed timetable's simulated margins on each."""
    if RP.empty:
        return {"error": "replay data not available"}
    g = RP.groupby("day").agg(price_std=("price", "std"), gridmind=("gm_pnl", "sum"), fixed=("fixed_pnl", "sum"))
    fmt = lambda df: [{"date": str(i), "price_volatility": round(float(r.price_std), 1),
                       "gridmind_margin_gbp": round(float(r.gridmind)), "fixed_margin_gbp": round(float(r.fixed))}
                      for i, r in df.iterrows()]
    return {"first_day": str(g.index.min()), "last_day": str(g.index.max()), "n_days": int(len(g)),
            "most_volatile": fmt(g.sort_values("price_std", ascending=False).head(5)),
            "calmest": fmt(g.sort_values("price_std").head(5))}

TOOLS = [what_if_now, replay_test_day, backtest_results, test_days_overview]

def ask_gemini(prompt, use_tools=False):
    """One Gemini request; returns (answer text, list of GridMind tools Gemini called)."""
    client = gemini_client(GEMINI_KEY)
    cfg = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.3,
                                      tools=TOOLS if use_tools else None)
    last = None
    for model in GEMINI_MODELS:              # fall back to another model name if one is unavailable
        try:
            r = client.models.generate_content(model=model, contents=prompt, config=cfg)
            calls = []
            for content in (getattr(r, "automatic_function_calling_history", None) or []):
                for part in (getattr(content, "parts", None) or []):
                    fc = getattr(part, "function_call", None)
                    if fc is not None and getattr(fc, "name", None):
                        args = ", ".join(f"{k}={v}" for k, v in (fc.args or {}).items())
                        calls.append(f"{fc.name}({args})")
            return (r.text or "").strip() or "(Gemini returned no text.)", calls
        except Exception as e:
            last = e
    raise RuntimeError(last)

def gemini_box(key):
    # show a stored Gemini answer (kept in session state so reruns don't call the API again)
    ans = st.session_state.get(key)
    if ans:
        st.markdown(ans["text"])
        if ans.get("calls"):
            st.caption("🔧 Gemini ran GridMind's tools: " + "; ".join(ans["calls"]))


# ------------------------------------------------------------------ result charts
def cumulative_figure():
    cum = R.get("cumulative")
    if not cum: return None
    fig, ax = plt.subplots(figsize=(12, 4))
    dates = pd.to_datetime(cum["dates"]); colours = [GRAY, AMBER, PURPLE, GREEN, BLUE]
    for i, (name, vals) in enumerate(cum["series"].items()):
        hind = "hindsight" in name.lower()
        ax.plot(dates, np.array(vals) / 1000, label=name, color="black" if hind else colours[i % 5],
                lw=1 if hind else 2, ls="--" if hind else "-")
    ax.set_ylabel("Cumulative profit (£ thousands)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("50 MW / 100 MWh battery on unseen test months")
    fig.tight_layout(); return fig

def ablation_figure():
    ab = R.get("ablation")
    if not ab: return None
    fig, ax = plt.subplots(figsize=(12, 3.8)); prev = 0.0
    for k, row in enumerate(ab):
        tot = row["profit"]; inc = tot if k == 0 else tot - prev
        ax.bar(k, inc / 1000, bottom=0 if k == 0 else prev / 1000, color=GRAY if k == 0 else (GREEN if inc >= 0 else CORAL))
        ax.text(k, max(tot, prev) / 1000, f"{'' if k == 0 else ('+' if inc >= 0 else '')}{inc / 1000:,.0f}k",
                ha="center", va="bottom", fontsize=9)
        prev = tot
    ax.set_xticks(range(len(ab))); ax.set_xticklabels([r["step"] for r in ab], fontsize=8)
    ax.set_ylabel("Profit, £ thousands"); ax.set_title("Where the value comes from (test months)")
    fig.tight_layout(); return fig

def importance_figure():
    imp, imp2 = R.get("importance"), R.get("importance_mh")
    if not imp: return None
    fig, ax = plt.subplots(1, 2 if imp2 else 1, figsize=(13, 4.2), squeeze=False); ax = ax[0]
    ax[0].barh(imp["names"], imp["values"], color=GREEN)
    ax[0].set_title("Next half-hour forecast:\naverage impact, £/MWh", fontsize=10)
    if imp2:
        ax[1].barh(imp2["names"], imp2["values"], color=PURPLE)
        ax[1].set_title("22-hour plan forecast:\naverage impact, £/MWh", fontsize=10)
    for a in ax: a.tick_params(labelsize=8)
    fig.tight_layout(); return fig

def shape_figure():
    p = R.get("daily_shape")
    if not p: return None
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(p["hour"], p["weekday"], color=GREEN, lw=2, label="Weekday")
    ax.plot(p["hour"], p["weekend"], color=AMBER, lw=2, label="Weekend")
    ax.set_xlabel("Hour (UK time)"); ax.set_ylabel("Median £/MWh"); ax.grid(alpha=0.3); ax.legend()
    ax.set_title("When electricity is cheap and expensive (real data)")
    fig.tight_layout(); return fig

def show_table(title, key):
    rows = R.get(key)
    if rows:
        st.markdown(f"#### {title}")
        df = pd.DataFrame(rows)
        st.dataframe(df.astype(object).where(df.ne(""), None), hide_index=True)   # blank cells shown as empty


# ------------------------------------------------------------------ page
meta = R.get("meta", {})
st.title("⚡ GridMind")
st.markdown("**Decides every 30 minutes when a grid battery should charge or discharge, "
            "from live GB electricity market data, and explains why.**")
tab_live, tab_copilot, tab_replay, tab_lab, tab_results, tab_explain, tab_how = st.tabs(
    ["Live decision", "🤖 Gemini Copilot", "Replay a day", "Decision Lab", "Results", "Explainability", "How it works"])

# The decision shown on the Live tab is kept across reruns (sliders, buttons, questions) so the Copilot
# always talks about the same decision and battery level that are on screen.
if "rec" not in st.session_state:
    st.session_state.rec = snapshot_rec() if SNAP else None
    st.session_state.rec_soc = float(SNAP.get("soc", 50)) if SNAP else 50.0
    st.session_state.rec_live = False

with tab_live:
    c1, c2 = st.columns([2, 1])
    soc = c1.slider("Battery charge now (MWh of 100)", 5, 95, int(st.session_state.rec_soc))
    fresh = c2.button("Get a fresh decision from live data", type="primary")
    if fresh:
        try:
            with st.spinner("Downloading the latest Elexon data (up to a minute the first time)..."):
                raw = live_raw()
            rec = recommend(raw, datetime.now(timezone.utc), float(soc))
            if rec is None:
                st.warning("No upcoming half-hour in the latest data yet. Try again shortly.")
            else:
                st.session_state.rec, st.session_state.rec_soc, st.session_state.rec_live = rec, float(soc), True
                for k in ("brief", "whatif"):
                    st.session_state.pop(k, None)          # old Gemini answers belong to the old decision
        except Exception as e:
            st.error(f"Live data unavailable right now: {e}. Try the Replay a day tab meanwhile.")
    rec = st.session_state.rec
    if rec is None:
        st.info("Press the button to get a live decision.")
    elif st.session_state.rec_live:
        show_decision(rec, st.session_state.rec_soc, "Live from Elexon data, refreshed every 15 minutes.")
    else:
        created = pd.Timestamp(SNAP["created"]).tz_convert(UK)
        st.info(f"📌 Showing a SAVED decision from {created:%a %d %b %Y, %H:%M} (offline snapshot, battery at "
                f"{st.session_state.rec_soc:.0f} MWh). Press the button for a fresh decision from live Elexon data.")
        show_decision(rec, st.session_state.rec_soc, "Saved snapshot made when the notebook was run.")

with tab_copilot:
    st.markdown("### 🤖 GridMind Copilot, powered by Google Gemini")
    st.markdown("GridMind's Python models make the forecasts and the optimiser makes the decisions. **Gemini explains "
                "them in plain English and can run GridMind's own optimiser** (function calling) to answer what-if "
                "and historical questions with real computed numbers.")
    rec, rec_soc = st.session_state.rec, st.session_state.rec_soc
    if not GENAI_AVAILABLE:
        st.error("The `google-genai` library is not installed (it is listed in requirements.txt).")
    elif not GEMINI_KEY:
        st.warning("Gemini is not configured. Add `GEMINI_API_KEY` in Streamlit Cloud → your app → Settings → Secrets "
                   "(locally: `.streamlit/secrets.toml` or an environment variable). Everything else in GridMind works without it.")
    elif rec is None:
        st.info("No decision yet: press 'Get a fresh decision from live data' on the Live decision tab first.")
    else:
        st.caption(f"Talking about: {rec['action']} at {rec['mw']:.0f} MW for the half-hour starting "
                   f"{rec['time'].tz_convert(UK):%a %d %b %H:%M} UK, battery at {rec_soc:.0f} MWh "
                   f"({'live decision' if st.session_state.rec_live else 'saved snapshot'}).")
        facts = json.dumps(decision_facts(rec, rec_soc), default=str)
        left, right = st.columns(2)
        with left:
            st.markdown("#### 1. Explain this decision")
            if st.button("Analyse the current decision", type="primary"):
                with st.spinner("Gemini is reviewing GridMind's plan..."):
                    try:
                        text, calls = ask_gemini(
                            "Brief the trading desk on this GridMind decision in 3 short paragraphs: (1) the economic "
                            "logic: price now vs the prices the 22-hour plan charges and discharges at, after 12% losses "
                            "and wear; (2) how the battery level and limits shape it; (3) what the main forecast drivers "
                            "say about the grid, and one risk to watch.\n\nFACTS:\n" + facts)
                        st.session_state.brief = {"text": text, "calls": calls}
                    except Exception as err:
                        st.error(f"Gemini request failed: {err}")
            gemini_box("brief")
        with right:
            st.markdown("#### 2. What-if: a different battery level")
            st.caption("GridMind's optimiser recomputes the decision; Gemini explains why it changed.")
            alt = st.slider("Battery charge for the what-if (MWh)", 5, 95, 15 if rec_soc > 50 else 85)
            if st.button("Run what-if"):
                res = what_if_now(float(alt))
                with st.spinner("Gemini is comparing the two decisions..."):
                    try:
                        text, calls = ask_gemini(
                            "GridMind's optimiser re-planned the same half-hour with a different battery level. In 2 short "
                            "paragraphs explain why the decision changed or stayed the same (headroom to the 5 and 95 MWh "
                            "limits, keeping energy for later peaks, losses and wear).\n\nORIGINAL:\n" + facts
                            + "\n\nWHAT-IF RESULT (computed by GridMind's optimiser):\n" + json.dumps(res, default=str))
                    except Exception as err:
                        text, calls = f"Gemini request failed: {err}", []
                st.session_state.whatif = {"text": f"**Optimiser result at {alt} MWh: {res['decision']} at "
                                                   f"{res['power_mw']:.0f} MW** (original: {rec['action']} at "
                                                   f"{rec['mw']:.0f} MW at {rec_soc:.0f} MWh)\n\n" + text, "calls": calls}
            gemini_box("whatif")

        st.markdown("---")
        st.markdown("#### 3. Ask the Copilot")
        st.caption("Gemini can call four GridMind tools: what-if on the current decision, replay any test day with your "
                   "own settings, the backtest results, and an overview of the test days.")
        if "chat" not in st.session_state:
            st.session_state.chat = []
        for turn in st.session_state.chat:
            with st.chat_message(turn["role"]):
                st.markdown(turn["text"])
                if turn.get("calls"):
                    st.caption("🔧 Gemini ran GridMind's tools: " + "; ".join(turn["calls"]))
        examples = ["What are the main risks in following this recommendation?",
                    "On the most volatile test day, compare balanced and conservative mode.",
                    "Is GridMind's advantage over the market-price baseline statistically convincing?"]
        ex_cols = st.columns(3)
        picked = None
        for i, (col, q) in enumerate(zip(ex_cols, examples)):
            if col.button(q, key=f"example_{i}"):
                picked = q
        with st.form("gemini_question_form", clear_on_submit=True):
            typed = st.text_input("Your question", placeholder="e.g. What if the battery were 92% efficient on 2026-09-05?")
            submitted = st.form_submit_button("Ask Gemini")
        question = picked or (typed.strip() if submitted and typed and typed.strip() else None)
        if question:                                   # only runs on a click, never on an ordinary rerun
            history = "\n".join(f"{t['role'].upper()}: {t['text']}" for t in st.session_state.chat[-6:])
            with st.spinner("Gemini is thinking (and may run GridMind's tools)..."):
                try:
                    text, calls = ask_gemini("CURRENT DECISION FACTS:\n" + facts
                                             + "\n\nBACKTEST HEADLINES:\n" + json.dumps(R.get("pitch", []))
                                             + ("\n\nCONVERSATION SO FAR:\n" + history if history else "")
                                             + "\n\nQUESTION: " + question, use_tools=True)
                except Exception as err:
                    text, calls = f"Gemini request failed: {err}", []
            st.session_state.chat += [{"role": "user", "text": question}, {"role": "assistant", "text": text, "calls": calls}]
            st.rerun()
        if st.session_state.chat and st.button("Clear conversation"):
            st.session_state.chat = []
            st.rerun()
        st.caption("Gemini explains and explores; it does not make GridMind's forecasts or decisions. All £ figures are "
                   "simulated margins at Elexon imbalance prices.")

with tab_replay:
    if RP.empty:
        st.info("Replay data not available.")
    else:
        days = sorted(RP.day.unique())
        default = RP.groupby("day").price.std().idxmax()          # the most volatile day: the best story
        day = st.selectbox("Pick a day from the test months (never used to train or tune anything)", days,
                           index=days.index(default), format_func=lambda d: f"{pd.Timestamp(d):%a %d %b %Y}")
        d = RP[RP.day == day]
        gm, fx = d.gm_pnl.sum(), d.fixed_pnl.sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("GridMind simulated margin", f"£{gm:,.0f}", f"{gm - fx:+,.0f} vs fixed")
        c2.metric("Fixed schedule simulated margin", f"£{fx:,.0f}")
        c3.metric("Half-hours traded", f"{(d.gm_action.abs() > 0.01).sum()} of {len(d)}")
        st.pyplot(replay_figure(d))
        best = d.loc[d.gm_pnl.idxmax()]
        st.caption(f"Prices ranged from £{d.price.min():,.0f} to £{d.price.max():,.0f}/MWh. Best moment: "
                   f"{pd.Timestamp(best.startTime).tz_convert(UK):%H:%M}, "
                   f"{'sold' if best.gm_action > 0 else 'bought'} at £{best.price:,.0f}/MWh, earning £{best.gm_pnl:,.0f}.")

with tab_lab:
    if RP.empty or not PATHS:
        st.info("Decision Lab data not available (run the notebook's export cell).")
    else:
        st.markdown("Change the battery and the operator's risk preference: GridMind re-plans **every half-hour of the "
                    "chosen day** with exactly the forecasts it had at the time, so you can see how each setting changes "
                    "the decisions.")
        days_l = sorted(RP.day.unique())
        default_l = RP.groupby("day").price.std().idxmax()
        c1, c2 = st.columns(2)
        day_l = c1.selectbox("Test day", days_l, index=days_l.index(default_l), key="lab_day",
                             format_func=lambda d: f"{pd.Timestamp(d):%a %d %b %Y}")
        mode = c2.radio("Operating preference", MODES, index=0)
        c3, c4, c5 = st.columns(3)
        soc0 = c3.slider("Starting charge (MWh)", 5, 95, 50, key="lab_soc")
        rte = c4.slider("Round-trip efficiency (%)", 80, 95, int(round(R.get("meta", {}).get("rte", 0.88) * 100))) / 100
        deg = c5.slider("Wear cost (£ per MWh discharged)", 0, 15, int(R.get("meta", {}).get("deg_cost", 5)))
        with st.spinner("Re-planning every half-hour of the day..."):
            acts, socs, pnls, fx_a, fx_p = lab_run(day_l, float(soc0), float(rte), float(deg), mode)
        d = RP[RP.day == day_l]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("GridMind simulated margin", f"£{pnls.sum():,.0f}", f"{pnls.sum() - fx_p.sum():+,.0f} vs fixed")
        m2.metric("Fixed schedule (same battery)", f"£{fx_p.sum():,.0f}")
        m3.metric("Half-hours traded", f"{int((np.abs(acts) > 0.01).sum())} of {len(d)}")
        m4.metric("Battery at end of day", f"{socs[-1]:.0f} MWh")
        st.pyplot(lab_figure(d, acts, socs))
        st.caption("Margins are simulated at Elexon imbalance prices. Conservative mode raises the hurdle by 25% of the "
                   "forecast's 80% half-range at each half-hour, so it trades less when the forecast is unsure.")

with tab_results:
    st.markdown(f"### Backtest on {meta.get('test_days', '?')} unseen days")
    st.caption("All £ figures are SIMULATED margins: energy valued at Elexon imbalance (settlement) prices minus charging "
               "cost and wear. They measure decision quality; a real battery would need market access to realise them.")
    if "coverage_test" in meta:
        st.metric("80% forecast range: share of actual test prices inside it", f"{meta['coverage_test']:.0%}", "target 80%",
                  delta_color="off")
    st.markdown("\n".join(f"- {line}" for line in R.get("pitch", [])) or "_Run the notebook's export cell._")
    for fig in (cumulative_figure(), ablation_figure()):
        if fig is not None: st.pyplot(fig)
    show_table("Battery strategies compared", "backtest")
    show_table("Is the advantage real? (block bootstrap over days)", "significance")
    show_table("Robustness to battery assumptions (simulated margin on test months, £)", "robustness")
    show_table("Risk preference: balanced vs conservative (uses the forecast's uncertainty)", "risk")
    show_table("Forecasting models compared (validation months, chosen by profit)", "comparison")

with tab_explain:
    st.markdown("### What drives GridMind's forecasts\nEach forecast is split exactly into contributions from its inputs "
                "(TreeSHAP, built into LightGBM). Bars show the average size of each input's effect.")
    for fig in (importance_figure(), shape_figure()):
        if fig is not None: st.pyplot(fig)

with tab_how:
    st.markdown(f"""
### The problem
Britain's grid must balance supply and demand every second. When wind under-delivers or demand surprises, the grid
runs **short** and the half-hourly **imbalance price** spikes; with surplus power it falls, sometimes below zero.
Britain has over 8 GW of grid batteries that must decide **every 30 minutes** whether to charge, discharge or hold,
**before** that price is known. Many smaller sites still follow fixed timetables.

### How GridMind works
1. **Data:** real, public Elexon data: imbalance prices, generation by fuel, market prices, wind and demand forecasts.
   Only information published before each decision is used.
2. **Forecast:** {meta.get('best', 'a model')} (chosen on validation) predicts the next half-hour's price with an 80% range, and a multi-horizon
   LightGBM model predicts the following 22 hours from the official day-ahead wind and demand forecasts.
   Models are retrained regularly and chosen on validation months by the profit they produce.
3. **Decide:** a mixed-integer optimiser plans the next 22 hours for a 50 MW / 100 MWh battery (88% efficiency, wear
   cost, charge limits, never charging and discharging at once), acts on the first half-hour, then re-plans with fresh data.
   The Decision Lab lets you change the battery and the risk preference and see the decisions change.
4. **Prove:** tested on months never used for any choice, against a fixed schedule, a market-price forecast with the
   same optimiser, and perfect hindsight; plus an ablation, a significance test and robustness checks.
5. **Explain:** every forecast is split into plain-English drivers (TreeSHAP), shown with each decision.
6. **Gemini Copilot:** Google Gemini turns each decision into a plain-English briefing and, through function calling,
   runs GridMind's own optimiser and backtest tools to answer what-if and historical questions with computed numbers.

### Honest limitations
GridMind is a research prototype. Margins are simulated at Elexon imbalance (settlement) prices, which a battery cannot
freely trade at; a real battery would also trade in other markets and needs market access, dispatch and settlement
integration. The battery is a price taker (50 MW vs ~30,000 MW national demand).

*Data: Elexon Insights Solution (BMRS), used under the BMRS Data Licence.*
""")
