import io
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

# =========================================================
# i18n (English only) — centralize UI strings
# =========================================================
STR = {
    "en": {
        "app_title": "Kinetic Impact Calculator",
        "app_subtitle": "Decision-support MVP: net energy, practical uses, costs, uncertainty scenarios, and a lightweight forecast module.",
        "reset_defaults": "🔄 Reset to defaults",
        "defaults_loaded": "Defaults reloaded.",
        "load_example": "Load example scenario",
        "load_example_btn": "📌 Load example",
        "loaded_scenario": "Example loaded:",
        "glossary_btn": "📘 Glossary",
        "glossary_title": "📘 Glossary",
        "glossary_search": "Search the glossary",
        "glossary_none": "No results.",
        "see_source": "See source",
        "mode": "Mode",
        "beginner": "Beginner",
        "advanced": "Advanced",
        "quick_start": "Quick start (5 steps)",
        "quick_start_caption": "Steps: 1) place → 2) visitors/day → 3) % on zone → 4) useful steps → 5) results",
        "prev": "← Previous",
        "next": "Next →",
        "step": "Step",
        "inputs_tab": "Inputs",
        "results_tab": "Results",
        "method_tab": "Methodology / Limits",
        "context": "Context",
        "preset_caption": "Preset:",
        "preset_uncertainty": "Uncertainty:",
        "apply_preset": "Apply place preset",
        "preset_applied": "Preset applied for:",
        "tour_step_1": "✅ Step 1: choose a place and apply a preset. Then click Next.",
        "tour_step_2": "✅ Step 2: adjust visitors/day (+ peak multiplier if needed). Then click Next.",
        "tour_step_3": "✅ Step 3: adjust % on zone. Then click Next.",
        "tour_step_4": "✅ Step 4: adjust useful steps. Then click Next.",
        "tour_step_5": "✅ Step 5: you are on Results. You can now refine % on zone / useful steps / J_net/step.",
        "key_drivers": "Key drivers (what changes the result the most)",
        "visitors_day": "Visitors / day (average)",
        "data_quality": "Data quality",
        "measured": "Measured",
        "estimated": "Estimated",
        "very_uncertain": "Very uncertain",
        "confidence": "Confidence",
        "peak_multiplier": "Peak multiplier",
        "flow_zone": "Flow on equipped zone",
        "pct_on_zone": "% visitors on equipped zone",
        "useful_steps": "Useful steps / visitor",
        "examples_pct": "Example: 2–10% (small zone) / 10–30% (central zone)",
        "examples_steps": "Example: 20–60 (short corridor) / 80–200 (long passage)",
        "tech_assumptions": "Technical assumptions",
        "jnet": "J_net per step",
        "dq_jnet": "Data quality (J_net/step)",
        "auto_consumption": "System self-consumption (Wh/day)",
        "sizing": "Installation sizing",
        "area_ft2": "Equipped area (ft²)",
        "tile_area_ft2": "Tile area (ft²)",
        "tiles_est": "Estimate:",
        "tiles_for": "tiles for",
        "if_tile": "if 1 tile ≈",
        "warning_steps": "⚠️ Useful steps per visitor look high vs zone geometry — risk of overestimation.",
        "adjust_typical": "Adjust to a typical value",
        "why": "Why?",
        "why_text": (
            "This is a rough sanity check using typical walking speed (~1.34 m/s) and step frequency (~2 Hz) "
            "to detect very unlikely inputs. It’s a guardrail, not a truth."
        ),
        "costs": "Costs",
        "capex_per_ft2": "Installed CAPEX ($/ft²)",
        "fixed_cost": "Fixed cost (work/electrical/signage) $",
        "maintenance_pct": "Annual maintenance (% of CAPEX)",
        "amort_years": "Amortization (years)",
        "sai": "Sustainable AI (lightweight)",
        "use_demo": "Use demo dataset",
        "upload_csv": "Upload CSV (columns: date, visitors)",
        "horizon": "Horizon (days)",
        "csv_loaded": "CSV loaded:",
        "csv_error": "Could not read CSV:",
        "dataset_preview": "Dataset preview",
        "forecast_preview": "Forecast (lightweight)",
        "results_title": "Results",
        "model_not_do": "What this model does NOT do",
        "not_quote": "Not a quote: costs vary by vendor and project.",
        "not_building": "Not powering a building.",
        "not_climate": "Not a climate solution alone (main value is pedagogy + micro-local loads).",
        "exec_summary": "Executive summary (actionable)",
        "net_energy": "Net energy",
        "phrase_modest": "Key message: outputs are usually modest — strongest value is engagement + micro-local uses.",
        "total_cost": "Total cost",
        "cost_kwh_rough": "Cost/kWh (rough)",
        "verdicts": "Verdicts (separated)",
        "drivers_box": "What drives your result the most",
        "wh_day": "Wh / day (primary)",
        "wh_month": "Wh / month (~30d)",
        "kwh_year": "kWh / year (~365d)",
        "uncertainty": "Uncertainty (scenarios)",
        "uncertainty_caption": "Reality depends mostly on % on zone and useful steps (placement + path).",
        "what_power": "What can it power (per day)",
        "led10": "10W LED (hours)",
        "sensor2": "Low-power sensor 2Wh/day (days)",
        "screen15": "Small screen ~15W (minutes)",
        "phone12": "Phone charges (~12Wh)",
        "reminder": "Reminder: harvesting energy is often modest. Best use: make energy visible + power micro-local loads.",
        "costs_block": "Costs",
        "capex": "CAPEX ($)",
        "opex_year": "OPEX/year ($)",
        "explain_costkwh": "ⓘ Explain cost/kWh (rough)",
        "explain_costkwh_text": "Approx indicator to compare scenarios, not a quote. It explodes when production is tiny (normal for human harvesting).",
        "export": "Export",
        "download_csv": "Download results (CSV)",
        "method_title": "Methodology / Limits",
        "sources_jnet": "🔎 Source — J_net/step",
        "sources_motion": "🔎 Source — speed/step rate",
        "sources_units": "🔎 Source — units",
        "core_formula": "Core formula (transparent)",
        "math_expand": "Math (expand)",
        "what_not": "What this is NOT",
        "limits": "Limits (anti-greenwashing)",
        "cost_note": "Cost note",
        "cost_note_text": "Costs (CAPEX/OPEX) are not scientific constants: they depend on quotes and site constraints.",
        "footer": "© Camille Chichet — Kinetic Impact Calculator",
        "ok": "OK",
        "go": "GO",
        "no_go": "NO-GO",
        "mixed": "MIXED",
        "go_msg": "✅ GO —",
        "no_go_msg": "⛔ NO-GO —",
        "mixed_msg": "⚠️ MIXED —",
        "plausible": "✅ plausible",
        "optimistic": "⚠️ optimistic",
        "improbable": "🚩 very unlikely",
    }
}
LANG = "en"


def t(k: str) -> str:
    return STR[LANG].get(k, k)


# =========================================================
# Page config
# =========================================================
st.set_page_config(page_title=t("app_title"), page_icon="⚡", layout="wide")


# =========================================================
# Glossary + Sources
# =========================================================
GLOSSARY = {
    "CAPEX": "Capital Expenditure: upfront cost (equipment + install + fixed works).",
    "OPEX": "Operational Expenditure: yearly operating cost (maintenance, etc.).",
    "Amortization": "Years used to spread CAPEX for rough annual/total costing.",
    "Cost/kWh (rough)": "Approx indicator to compare scenarios (NOT a quote). Highly sensitive to assumptions.",
    "J_net/step": "Net electrical energy delivered per step (already net; avoids double-counting efficiencies).",
    "% on zone": "Share of visitors who actually walk on the equipped zone (placement is key).",
    "Useful steps": "Captured steps per visitor on the zone (depends on zone length/path design/density).",
    "Uncertainty": "We show low/mid/high because % on zone, useful steps, and J_net vary a lot.",
    "Dataset": "CSV history (date, visitors) used by the lightweight forecast model.",
    "Horizon": "Number of future days predicted by the forecast module.",
    "Self-consumption": "System electronics consumption in Wh/day; can wipe out small gains.",
    "Units (J, Wh, kWh)": "1 Wh = 3600 J. Convert J → Wh by dividing by 3600.",
}

SOURCES = {
    "J_net/step": [
        ("Asadi et al. (2023) ~511 mJ/step", "https://doi.org/10.1016/j.seta.2023.103571"),
        ("Jintanawan et al. (2020) up to ~702 mJ/step", "https://www.mdpi.com/1996-1073/13/20/5419"),
        ("Thainiramit et al. (2022) tribo ~mJ range", "https://www.mdpi.com/1996-1944/15/24/8853"),
    ],
    "speed/step rate": [
        ("Weidmann (1993) free speed ~1.34 m/s", "https://www.ped-net.org/uploads/media/weidmann-1993_01.pdf"),
        ("Pachi & Ji (2005) step frequency ~2 Hz (observations)", "https://trid.trb.org/View/750847"),
    ],
    "units": [
        ("BIPM SI Brochure (2019)", "https://www.bipm.org/en/publications/si-brochure"),
    ]
}


def glossary_ui():
    st.markdown(f"### {t('glossary_title')}")
    q = st.text_input(t("glossary_search"), placeholder="e.g., CAPEX, J_net/step, uncertainty…", key="glossary_search_input")
    items = list(GLOSSARY.items())
    if q:
        ql = q.lower()
        items = [(k, v) for k, v in items if ql in k.lower() or ql in v.lower()]
    if not items:
        st.info(t("glossary_none"))
        return
    for term, definition in items:
        with st.expander(term):
            st.write(definition)
            if term in SOURCES:
                st.markdown(f"**{t('see_source')}**")
                for label, url in SOURCES[term]:
                    st.link_button(label, url)


def try_popover(label: str):
    if hasattr(st, "popover"):
        return st.popover(label)
    return st.expander(label)


def badge_realism(value: float, ok_range: tuple[float, float], warn_range: tuple[float, float]) -> str:
    if ok_range[0] <= value <= ok_range[1]:
        return t("plausible")
    if warn_range[0] <= value <= warn_range[1]:
        return t("optimistic")
    return t("improbable")


def safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else float("nan")


def ft2_to_m2(x_ft2: float) -> float:
    return x_ft2 * 0.092903


def fmt_money(x: float) -> str:
    return f"{x:,.0f}".replace(",", " ")


# =========================================================
# Lightweight forecast (no sklearn)
# =========================================================
def make_demo_visitors(n_days: int = 60, start: date | None = None) -> pd.DataFrame:
    if start is None:
        start = date.today() - timedelta(days=n_days)
    dates = pd.date_range(start=start, periods=n_days, freq="D")
    base = 1200
    trend = np.linspace(0, 250, n_days)
    weekday = np.array([1.0, 1.0, 1.05, 1.05, 1.1, 1.3, 1.25])  # Mon..Sun
    w = np.array([weekday[d.weekday()] for d in dates])
    noise = np.random.normal(0, 60, n_days)
    visitors = np.maximum(0, (base + trend) * w + noise).round().astype(int)
    return pd.DataFrame({"date": dates.date, "visitors": visitors})


def load_csv_visitors(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    if "date" not in df.columns or "visitors" not in df.columns:
        raise ValueError("CSV must contain columns: date, visitors")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["visitors"] = pd.to_numeric(df["visitors"], errors="coerce").fillna(0).astype(int)
    df = df.sort_values("date").drop_duplicates("date")
    return df


def lightweight_forecast(df: pd.DataFrame, horizon_days: int = 14) -> pd.DataFrame:
    d = df.copy().dropna()
    if len(d) < 7:
        last = int(d["visitors"].tail(7).mean()) if len(d) > 0 else 0
        start_date = (d["date"].max() + timedelta(days=1)) if len(d) > 0 else date.today()
        future_dates = [start_date + timedelta(days=i) for i in range(horizon_days)]
        return pd.DataFrame({"date": future_dates, "visitors_pred": [max(0, last)] * horizon_days})

    x = np.array([pd.Timestamp(dt).toordinal() for dt in d["date"]], dtype=float)
    y = d["visitors"].to_numpy(dtype=float)

    coeff = np.polyfit(x, y, 1)
    trend_fn = np.poly1d(coeff)
    y_trend = trend_fn(x)
    resid = y - y_trend

    wday = np.array([pd.Timestamp(dt).weekday() for dt in d["date"]], dtype=int)
    resid_by_wday = {wd: 0.0 for wd in range(7)}
    for wd in range(7):
        mask = (wday == wd)
        resid_by_wday[wd] = float(np.mean(resid[mask])) if np.any(mask) else 0.0

    start_date = d["date"].max() + timedelta(days=1)
    future_dates = [start_date + timedelta(days=i) for i in range(horizon_days)]
    xf = np.array([pd.Timestamp(dt).toordinal() for dt in future_dates], dtype=float)
    yf = trend_fn(xf)

    adj = np.array([resid_by_wday[pd.Timestamp(dt).weekday()] for dt in future_dates], dtype=float)
    pred = np.maximum(0, yf + adj)

    return pd.DataFrame({"date": future_dates, "visitors_pred": pred})


# =========================================================
# Presets + example scenarios
# =========================================================
PRESETS = {
    "Museum": {
        "desc": "Moderate flow, slower paths → medium % on zone, medium steps.",
        "uncertainty": "Medium",
        "values": {"pct_on_zone": 8.0, "useful_steps": 60.0, "peak_multiplier": 1.2, "area_ft2": 120.0, "visitors_per_day": 1200,
                   "installed_cost_per_ft2": 140.0, "fixed_cost": 12000.0, "J_net_per_step": 0.5, "auto_consumption_wh_day": 0.0},
    },
    "Train station": {
        "desc": "High flow, repeated crossings → higher % on zone, higher steps.",
        "uncertainty": "High",
        "values": {"pct_on_zone": 18.0, "useful_steps": 120.0, "peak_multiplier": 1.5, "area_ft2": 220.0, "visitors_per_day": 8000,
                   "installed_cost_per_ft2": 160.0, "fixed_cost": 25000.0, "J_net_per_step": 0.5, "auto_consumption_wh_day": 5.0},
    },
    "Stadium": {
        "desc": "Big event peaks → high peak multiplier.",
        "uncertainty": "High",
        "values": {"pct_on_zone": 12.0, "useful_steps": 80.0, "peak_multiplier": 2.5, "area_ft2": 300.0, "visitors_per_day": 25000,
                   "installed_cost_per_ft2": 180.0, "fixed_cost": 40000.0, "J_net_per_step": 0.5, "auto_consumption_wh_day": 10.0},
    },
    "Mall": {
        "desc": "Continuous flow, weekend variations → medium peak, medium/high steps.",
        "uncertainty": "Medium",
        "values": {"pct_on_zone": 10.0, "useful_steps": 90.0, "peak_multiplier": 1.4, "area_ft2": 180.0, "visitors_per_day": 6000,
                   "installed_cost_per_ft2": 150.0, "fixed_cost": 20000.0, "J_net_per_step": 0.5, "auto_consumption_wh_day": 5.0},
    },
}

EXAMPLE_SCENARIOS = {
    "Museum (realistic)": {"place_type": "Museum", "visitors_per_day": 1500, "pct_on_zone": 8.0, "useful_steps": 60.0, "J_net_per_step": 0.5,
                           "auto_consumption_wh_day": 0.0, "area_ft2": 120.0, "installed_cost_per_ft2": 140.0, "fixed_cost": 12000.0,
                           "maintenance_pct": 6.0, "amort_years": 8, "peak_multiplier": 1.2},
    "Train station (realistic)": {"place_type": "Train station", "visitors_per_day": 12000, "pct_on_zone": 18.0, "useful_steps": 120.0, "J_net_per_step": 0.5,
                                  "auto_consumption_wh_day": 5.0, "area_ft2": 220.0, "installed_cost_per_ft2": 160.0, "fixed_cost": 25000.0,
                                  "maintenance_pct": 8.0, "amort_years": 10, "peak_multiplier": 1.5},
    "Stadium (event)": {"place_type": "Stadium", "visitors_per_day": 35000, "pct_on_zone": 12.0, "useful_steps": 80.0, "J_net_per_step": 0.5,
                        "auto_consumption_wh_day": 10.0, "area_ft2": 300.0, "installed_cost_per_ft2": 180.0, "fixed_cost": 40000.0,
                        "maintenance_pct": 8.0, "amort_years": 10, "peak_multiplier": 2.5},
}


# =========================================================
# Session defaults
# =========================================================
DEFAULTS = {
    "mode": t("beginner"),
    "place_type": "Museum",
    "visitors_per_day": 3300,
    "peak_multiplier": 1.0,
    "pct_on_zone": 12.0,
    "useful_steps": 115.0,
    "J_net_per_step": 0.5,
    "auto_consumption_wh_day": 0.0,
    "dq_visitors": t("estimated"),
    "dq_pct_on_zone": t("very_uncertain"),
    "dq_useful_steps": t("very_uncertain"),
    "dq_J_net": t("very_uncertain"),
    "area_ft2": 190.0,
    "tile_area_ft2": 1.10,
    "installed_cost_per_ft2": 175.0,
    "fixed_cost": 20000.0,
    "maintenance_pct": 8.0,
    "amort_years": 9,
    "use_demo_dataset": True,
    "forecast_horizon_days": 14,
    "tour_step": 1,
    "tour_on": False,
}

if "inputs" not in st.session_state:
    st.session_state.inputs = DEFAULTS.copy()
else:
    for k, v in DEFAULTS.items():
        st.session_state.inputs.setdefault(k, v)

inp = st.session_state.inputs


# =========================================================
# Header
# =========================================================
st.title(t("app_title"))
st.caption(t("app_subtitle"))

top_l, top_m, top_r = st.columns([1.2, 1.2, 1.6])

with top_l:
    if st.button(t("reset_defaults"), key="btn_reset_defaults"):
        st.session_state.inputs = DEFAULTS.copy()
        st.success(t("defaults_loaded"))
        st.rerun()

with top_m:
    scenario_name = st.selectbox(t("load_example"), list(EXAMPLE_SCENARIOS.keys()), key="select_example_scenario")
    if st.button(t("load_example_btn"), key="btn_load_example"):
        ex = EXAMPLE_SCENARIOS[scenario_name]
        for k, v in ex.items():
            st.session_state.inputs[k] = v
        st.success(f"{t('loaded_scenario')} {scenario_name}")
        st.rerun()

with top_r:
    with try_popover(t("glossary_btn")):
        glossary_ui()

mode_col, tour_col, _ = st.columns([1.2, 1.2, 1.6])
with mode_col:
    mode = st.radio(
        t("mode"),
        [t("beginner"), t("advanced")],
        horizontal=True,
        index=0 if inp["mode"] == t("beginner") else 1,
        key="radio_mode",
    )
    inp["mode"] = mode

with tour_col:
    # ✅ KEY ADDED (required)
    inp["tour_on"] = st.toggle(t("quick_start"), value=bool(inp["tour_on"]), key="tour_on")
    if inp["tour_on"]:
        st.caption(t("quick_start_caption"))


# =========================================================
# Tabs
# =========================================================
tab_inputs, tab_results, tab_methods = st.tabs([t("inputs_tab"), t("results_tab"), t("method_tab")])


# =========================================================
# TOUR CONTROLS — RENDERED ONLY ONCE (Inputs tab only)
# =========================================================
def tour_controls_once():
    """Render tour controls exactly once (Inputs tab only)."""
    if not inp["tour_on"]:
        return

    step = int(inp.get("tour_step", 1))

    b1, b2, b3 = st.columns([1, 1, 2])
    with b1:
        # ✅ KEY ADDED (required)
        if st.button(t("prev"), disabled=(step <= 1), key="tour_prev_btn"):
            inp["tour_step"] = max(1, step - 1)
            st.rerun()
    with b2:
        # ✅ KEY ADDED (required)
        if st.button(t("next"), disabled=(step >= 5), key="tour_next_btn"):
            inp["tour_step"] = min(5, step + 1)
            st.rerun()
    with b3:
        st.progress(step / 5)
        st.write(f"{t('step')} {step}/5")


# =========================================================
# Inputs tab
# =========================================================
with tab_inputs:
    # ✅ Tour controls shown ONCE here
    tour_controls_once()

    tour_step = int(inp.get("tour_step", 1)) if inp["tour_on"] else 999

    c1, c2, c3 = st.columns([1.1, 1.0, 1.0], gap="large")

    # ---------------- Column 1: Context + Flow ----------------
    with c1:
        st.subheader(t("context"))

        place_type = st.selectbox(
            "Place type",
            options=list(PRESETS.keys()),
            index=list(PRESETS.keys()).index(inp["place_type"]) if inp["place_type"] in PRESETS else 0,
            key="select_place_type",
        )
        inp["place_type"] = place_type
        st.caption(f"{t('preset_caption')} {PRESETS[place_type]['desc']} • {t('preset_uncertainty')} **{PRESETS[place_type]['uncertainty']}**")

        if st.button(t("apply_preset"), key="btn_apply_preset"):
            p = PRESETS.get(place_type, {}).get("values", {})
            for k, v in p.items():
                inp[k] = v
            st.success(f"{t('preset_applied')} {place_type}")
            st.rerun()

        if inp["tour_on"] and tour_step == 1:
            st.info(t("tour_step_1"))
            st.stop()

        st.markdown("---")
        st.subheader(t("key_drivers"))

        visitors_per_day = st.number_input(
            t("visitors_day"),
            min_value=0,
            value=int(inp["visitors_per_day"]),
            step=50,
            key="num_visitors_day",
        )
        dq_visitors = st.selectbox(
            f"{t('data_quality')} ({t('visitors_day')})",
            [t("measured"), t("estimated"), t("very_uncertain")],
            index=[t("measured"), t("estimated"), t("very_uncertain")].index(inp["dq_visitors"]),
            key="dq_visitors",
        )
        inp["dq_visitors"] = dq_visitors
        st.write(f"🔎 {t('confidence')}: **{dq_visitors}** • {badge_realism(float(visitors_per_day), (300, 50000), (50, 120000))}")

        peak_multiplier = st.slider(
            t("peak_multiplier"),
            1.0, 5.0, float(inp["peak_multiplier"]), 0.05,
            key="slider_peak_multiplier",
        )

        st.markdown("---")
        st.subheader(t("flow_zone"))

        pct_on_zone = st.slider(
            t("pct_on_zone"),
            0.0, 100.0, float(inp["pct_on_zone"]), 0.5,
            key="slider_pct_on_zone",
        )
        dq_pct = st.selectbox(
            f"{t('data_quality')} ({t('pct_on_zone')})",
            [t("measured"), t("estimated"), t("very_uncertain")],
            index=[t("measured"), t("estimated"), t("very_uncertain")].index(inp["dq_pct_on_zone"]),
            key="dq_pct_on_zone",
        )
        inp["dq_pct_on_zone"] = dq_pct
        st.caption(t("examples_pct"))
        st.write(f"🔎 {t('confidence')}: **{dq_pct}** • {badge_realism(pct_on_zone, (2, 30), (0.5, 60))}")

        useful_steps = st.slider(
            t("useful_steps"),
            0.0, 300.0, float(inp["useful_steps"]), 5.0,
            key="slider_useful_steps",
        )
        dq_steps = st.selectbox(
            f"{t('data_quality')} ({t('useful_steps')})",
            [t("measured"), t("estimated"), t("very_uncertain")],
            index=[t("measured"), t("estimated"), t("very_uncertain")].index(inp["dq_useful_steps"]),
            key="dq_useful_steps",
        )
        inp["dq_useful_steps"] = dq_steps
        st.caption(t("examples_steps"))
        st.write(f"🔎 {t('confidence')}: **{dq_steps}** • {badge_realism(useful_steps, (20, 200), (5, 300))}")

        inp["visitors_per_day"] = int(visitors_per_day)
        inp["peak_multiplier"] = float(peak_multiplier)
        inp["pct_on_zone"] = float(pct_on_zone)
        inp["useful_steps"] = float(useful_steps)

        if inp["tour_on"] and tour_step == 2:
            st.info(t("tour_step_2"))
            st.stop()
        if inp["tour_on"] and tour_step == 3:
            st.info(t("tour_step_3"))
            st.stop()
        if inp["tour_on"] and tour_step == 4:
            st.info(t("tour_step_4"))
            st.stop()

    # ---------------- Column 2: Technical + sizing ----------------
    with c2:
        st.subheader(t("tech_assumptions"))

        jnet = st.slider(
            t("jnet"),
            0.005, 1.0, float(inp["J_net_per_step"]), 0.005,
            key="slider_jnet",
        )
        dq_jnet = st.selectbox(
            t("dq_jnet"),
            [t("measured"), t("estimated"), t("very_uncertain")],
            index=[t("measured"), t("estimated"), t("very_uncertain")].index(inp["dq_J_net"]),
            key="dq_jnet",
        )
        inp["dq_J_net"] = dq_jnet

        with try_popover(f"ⓘ {t('see_source')} (J_net/step)"):
            for label, url in SOURCES["J_net/step"]:
                st.link_button(label, url)

        auto_c = st.number_input(
            t("auto_consumption"),
            min_value=0.0,
            value=float(inp["auto_consumption_wh_day"]),
            step=1.0,
            key="num_auto_consumption",
        )

        inp["J_net_per_step"] = float(jnet)
        inp["auto_consumption_wh_day"] = float(auto_c)

        st.write(f"🔎 {t('confidence')}: **{dq_jnet}** • {badge_realism(jnet, (0.05, 0.8), (0.01, 1.0))}")

        st.markdown("---")
        st.subheader(t("sizing"))

        area_ft2 = st.number_input(t("area_ft2"), min_value=1.0, value=float(inp["area_ft2"]), step=10.0, key="num_area_ft2")
        tile_area_ft2 = st.number_input(t("tile_area_ft2"), min_value=0.2, value=float(inp["tile_area_ft2"]), step=0.05, key="num_tile_area_ft2")

        inp["area_ft2"] = float(area_ft2)
        inp["tile_area_ft2"] = float(tile_area_ft2)

        est_tiles = int(round(area_ft2 / tile_area_ft2))
        st.info(f"{t('tiles_est')} ~ **{est_tiles}** {t('tiles_for')} {area_ft2:.0f} ft² ({t('if_tile')} {tile_area_ft2:.2f} ft²).")

        # sanity check warning
        area_m2 = ft2_to_m2(area_ft2)
        approx_length_m = max(0.5, float(np.sqrt(area_m2)))
        v_free = 1.34
        f_step = 2.0
        step_len = v_free / f_step
        plausible_upper_steps = 2.5 * (approx_length_m / step_len)

        if inp["useful_steps"] > plausible_upper_steps and inp["useful_steps"] > 30:
            st.warning(f"{t('warning_steps')} (typical length ~{approx_length_m:.1f} m)")
            a1, a2 = st.columns([1, 1])
            with a1:
                if st.button(t("adjust_typical"), key="btn_adjust_typical_steps"):
                    typical = {
                        "Museum": 60.0,
                        "Train station": 120.0,
                        "Stadium": 80.0,
                        "Mall": 90.0
                    }.get(inp["place_type"], 80.0)
                    inp["useful_steps"] = typical
                    st.rerun()
            with a2:
                with st.expander(t("why"), expanded=False):
                    st.write(t("why_text"))
                    for label, url in SOURCES["speed/step rate"]:
                        st.link_button(label, url)

    # ---------------- Column 3: Costs + (advanced) forecast/export ----------------
    with c3:
        st.subheader(t("costs"))

        capex_ft2 = st.slider(t("capex_per_ft2"), 50.0, 900.0, float(inp["installed_cost_per_ft2"]), 5.0, key="slider_capex_ft2")
        fixed_cost = st.number_input(t("fixed_cost"), min_value=0.0, value=float(inp["fixed_cost"]), step=1000.0, key="num_fixed_cost")
        maint = st.slider(t("maintenance_pct"), 0.0, 20.0, float(inp["maintenance_pct"]), 0.5, key="slider_maint_pct")
        amort = st.slider(t("amort_years"), 1, 20, int(inp["amort_years"]), 1, key="slider_amort")

        inp["installed_cost_per_ft2"] = float(capex_ft2)
        inp["fixed_cost"] = float(fixed_cost)
        inp["maintenance_pct"] = float(maint)
        inp["amort_years"] = int(amort)

        if inp["mode"] == t("advanced"):
            st.markdown("---")
            st.subheader(t("sai"))

            use_demo = st.checkbox(t("use_demo"), value=bool(inp["use_demo_dataset"]), key="chk_use_demo")
            uploaded = st.file_uploader(t("upload_csv"), type=["csv"], key="uploader_csv")
            horizon = st.slider(t("horizon"), 7, 60, int(inp["forecast_horizon_days"]), 1, key="slider_horizon")

            inp["use_demo_dataset"] = bool(use_demo)
            inp["forecast_horizon_days"] = int(horizon)

            df_hist = None
            if uploaded is not None:
                try:
                    df_hist = load_csv_visitors(uploaded)
                    st.success(f"{t('csv_loaded')} {len(df_hist)} rows")
                except Exception as e:
                    st.error(f"{t('csv_error')} {e}")
            elif use_demo:
                df_hist = make_demo_visitors(n_days=60)

            if df_hist is not None:
                st.write(t("dataset_preview"))
                st.dataframe(df_hist.tail(10), use_container_width=True)

                df_fc = lightweight_forecast(df_hist, horizon_days=horizon)
                st.write(t("forecast_preview"))
                st.dataframe(df_fc.head(10), use_container_width=True)

                chart_df = pd.concat(
                    [
                        df_hist.rename(columns={"visitors": "value"}).assign(kind="history")[["date", "value", "kind"]],
                        df_fc.rename(columns={"visitors_pred": "value"}).assign(kind="forecast")[["date", "value", "kind"]],
                    ],
                    ignore_index=True,
                )
                chart_df["date"] = pd.to_datetime(chart_df["date"])
                st.line_chart(chart_df.set_index("date")[["value"]])


# =========================================================
# Shared computations
# =========================================================
steps_captured = (
    inp["visitors_per_day"]
    * inp["peak_multiplier"]
    * (inp["pct_on_zone"] / 100.0)
    * inp["useful_steps"]
)

gross_wh_day = steps_captured * inp["J_net_per_step"] / 3600.0
net_wh_day = max(0.0, gross_wh_day - inp["auto_consumption_wh_day"])

wh_month = net_wh_day * 30.0
kwh_year = (net_wh_day / 1000.0) * 365.0

capex = inp["area_ft2"] * inp["installed_cost_per_ft2"] + inp["fixed_cost"]
opex_year = (inp["maintenance_pct"] / 100.0) * capex
N = inp["amort_years"]
total_cost_N = capex + opex_year * N
cost_per_kwh = safe_div(total_cost_N, kwh_year * N) if kwh_year > 0 else float("inf")

df_scen = pd.DataFrame(
    {"scenario": ["low", "mid", "high"], "Wh/day": [net_wh_day * 0.6, net_wh_day * 1.0, net_wh_day * 1.4]}
).set_index("scenario")

led10_hours = safe_div(net_wh_day, 10.0)
sensor_days = safe_div(net_wh_day, 2.0)
screen_minutes = safe_div(net_wh_day, 15.0) * 60.0
phone_charges = safe_div(net_wh_day, 12.0)


def verdict_energy_roi():
    if kwh_year <= 0:
        return t("no_go"), "net energy is ~0 after self-consumption."
    return t("no_go"), "cost/kWh is typically extremely high vs output (human harvesting is usually modest)."


def verdict_pedagogy():
    if inp["pct_on_zone"] < 1.0 or inp["useful_steps"] < 10:
        return t("mixed"), "zone is rarely crossed — consider placement/path redesign."
    return t("go"), "good for engagement: making energy tangible + powering micro-local loads."


def show_verdict(kind: str, reason: str):
    if kind == t("go"):
        st.success(f"{t('go_msg')} {reason}")
    elif kind == t("mixed"):
        st.warning(f"{t('mixed_msg')} {reason}")
    else:
        st.error(f"{t('no_go_msg')} {reason}")


roi_kind, roi_reason = verdict_energy_roi()
ped_kind, ped_reason = verdict_pedagogy()


# =========================================================
# Results tab
# =========================================================
with tab_results:
    st.subheader(t("results_title"))

    with st.container(border=True):
        st.markdown(f"**{t('model_not_do')}**")
        st.markdown(f"- ❌ {t('not_quote')}\n- ❌ {t('not_building')}\n- ❌ {t('not_climate')}")

    with st.container(border=True):
        st.markdown(f"### {t('exec_summary')}")
        st.write(f"**{t('net_energy')}**: **{net_wh_day:.2f} Wh/day** • {wh_month:.1f} Wh/month • {kwh_year:.2f} kWh/year")
        st.caption(t("phrase_modest"))
        st.write(f"**{t('total_cost')}**: {fmt_money(capex)}$ CAPEX + {fmt_money(opex_year)}$/year OPEX → **{cost_per_kwh:,.2f} $/kWh**".replace(",", " "))
        st.caption(t("cost_kwh_rough"))
        st.markdown(f"**{t('verdicts')}**")
        show_verdict(roi_kind, roi_reason)
        show_verdict(ped_kind, ped_reason)

    with st.container(border=True):
        st.markdown(f"### {t('drivers_box')}")
        st.write("👉 1) % on zone  2) useful steps  3) J_net/step (then visitors/day).")

    m1, m2, m3 = st.columns(3)
    m1.metric(t("wh_day"), f"{net_wh_day:.2f}")
    m2.metric(t("wh_month"), f"{wh_month:.1f}")
    m3.metric(t("kwh_year"), f"{kwh_year:.2f}")

    st.markdown(f"### {t('uncertainty')}")
    st.bar_chart(df_scen[["Wh/day"]])
    st.caption(t("uncertainty_caption"))

    st.markdown(f"### {t('what_power')}")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric(t("led10"), f"{led10_hours:.2f}")
    e2.metric(t("sensor2"), f"{sensor_days:.2f}")
    e3.metric(t("screen15"), f"{screen_minutes:.1f}")
    e4.metric(t("phone12"), f"{phone_charges:.2f}")

    st.info(t("reminder"))

    st.markdown(f"### {t('costs_block')}")
    c1, c2, c3 = st.columns(3)
    c1.metric(t("capex"), fmt_money(capex))
    c2.metric(t("opex_year"), fmt_money(opex_year))
    c3.metric(t("cost_kwh_rough"), f"{cost_per_kwh:.2f}" if np.isfinite(cost_per_kwh) else "∞")

    with try_popover(t("explain_costkwh")):
        st.write(t("explain_costkwh_text"))

    if inp["mode"] == t("advanced"):
        st.markdown(f"### {t('export')}")
        export = {
            "place_type": inp["place_type"],
            "visitors_per_day": inp["visitors_per_day"],
            "peak_multiplier": inp["peak_multiplier"],
            "pct_on_zone": inp["pct_on_zone"],
            "useful_steps": inp["useful_steps"],
            "J_net_per_step": inp["J_net_per_step"],
            "auto_consumption_Wh_day": inp["auto_consumption_wh_day"],
            "area_ft2": inp["area_ft2"],
            "tile_area_ft2": inp["tile_area_ft2"],
            "installed_cost_per_ft2": inp["installed_cost_per_ft2"],
            "fixed_cost": inp["fixed_cost"],
            "maintenance_pct": inp["maintenance_pct"],
            "amort_years": inp["amort_years"],
            "steps_captured_per_day": steps_captured,
            "gross_Wh_day": gross_wh_day,
            "net_Wh_day": net_wh_day,
            "kWh_year": kwh_year,
            "CAPEX_$": capex,
            "OPEX_year_$": opex_year,
            "cost_per_kWh_$": cost_per_kwh,
            "energy_ROI_verdict": f"{roi_kind}: {roi_reason}",
            "pedagogy_verdict": f"{ped_kind}: {ped_reason}",
        }
        out_df = pd.DataFrame([export])
        buf = io.StringIO()
        out_df.to_csv(buf, index=False)
        st.download_button(t("download_csv"), data=buf.getvalue().encode("utf-8"),
                           file_name="kinetic_impact_results.csv", mime="text/csv", key="btn_download_csv")

    if inp["tour_on"] and int(inp.get("tour_step", 1)) == 5:
        st.success(t("tour_step_5"))


# =========================================================
# Methodology tab
# =========================================================
with tab_methods:
    st.subheader(t("method_title"))

    cols = st.columns(3)
    with cols[0]:
        with try_popover(t("sources_jnet")):
            for label, url in SOURCES["J_net/step"]:
                st.link_button(label, url)
    with cols[1]:
        with try_popover(t("sources_motion")):
            for label, url in SOURCES["speed/step rate"]:
                st.link_button(label, url)
    with cols[2]:
        with try_popover(t("sources_units")):
            for label, url in SOURCES["units"]:
                st.link_button(label, url)

    st.markdown(f"### {t('core_formula')}")
    with st.expander(t("math_expand")):
        st.code(
            "Net Energy (Wh/day) = visitors/day × peak_multiplier × (%on_zone/100) × useful_steps × J_net_per_step ÷ 3600  −  self_consumption_Wh_day",
            language="text",
        )
        st.caption("1 Wh = 3600 J → divide by 3600 to convert J → Wh.")

    st.markdown(f"### {t('what_not')}")
    st.markdown(
        "- ❌ Not powering a building (outputs are usually modest)\n"
        "- ❌ Not a climate solution alone (main value is educational + micro-local loads)\n"
        "- ❌ Not a quote: CAPEX/OPEX vary by project\n"
    )

    st.markdown(f"### {t('limits')}")
    st.markdown(
        "- Outputs depend mostly on **% on zone** and **useful steps** (placement + path)\n"
        "- **J_net/step** varies by technology, load, frequency, and test conditions\n"
        "- At low energy, **self-consumption** can wipe out gains → explicit in the model\n"
        "- No personal data: only aggregate visitor counts\n"
    )

    st.markdown(f"### {t('cost_note')}")
    st.info(t("cost_note_text"))


# =========================================================
# Global footer (visible everywhere)
# =========================================================
st.markdown("---")
st.caption(t("footer"))
