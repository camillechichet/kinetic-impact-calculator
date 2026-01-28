import math
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# Optional: CodeCarbon (for "Sustainable AI")
try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except Exception:
    CODECARBON_AVAILABLE = False


# ----------------------------
# Helpers
# ----------------------------
def energy_wh_per_day(visitors_per_day: float,
                      peak_multiplier: float,
                      pct_on_zone: float,
                      steps_useful: float,
                      j_per_step: float,
                      efficiency: float,
                      storage_losses: float) -> float:
    """
    Energy (Wh/day) = visitors/day * peak_multiplier * (% on zone) * steps_useful * J/step * efficiency * (1-losses) / 3600
    pct_on_zone in [0..100]
    efficiency in [0..1]
    storage_losses in [0..1]
    """
    visitors = visitors_per_day * peak_multiplier
    frac = pct_on_zone / 100.0
    wh = visitors * frac * steps_useful * j_per_step * efficiency * (1.0 - storage_losses) / 3600.0
    return max(0.0, float(wh))


def fmt_kwh(wh: float) -> float:
    return wh / 1000.0


def dollars(x: float) -> str:
    return f"${x:,.0f}"


def dollars2(x: float) -> str:
    return f"${x:,.2f}"


def recommend_size(area_ft2: float) -> tuple[str, str]:
    # Simple buckets
    if area_ft2 < 80:
        return ("Small", "Best for pedagogy / engagement (LEDs, small display).")
    if area_ft2 < 200:
        return ("Medium", "Good for local loads (LEDs + sensors) + strong educational impact.")
    return ("Large", "Likely needs a strong engagement goal; check cost/kWh to avoid over-installation.")


def go_no_go(kwh_year: float, cost_per_kwh: float, target_met: bool) -> tuple[str, str]:
    """
    Intentionally conservative: kinetic floors are usually NOT about cheap kWh.
    We frame Go/No-Go as decision support:
    - GO if target is met AND economics not absurd OR clear engagement/pedagogy.
    - NO-GO if extremely low energy and very high cost per kWh and target not met.
    """
    if target_met and kwh_year >= 50:
        return ("GO", "Meets your chosen objective and produces a meaningful local amount of energy over the year.")
    if (kwh_year < 20) and (cost_per_kwh > 500) and (not target_met):
        return ("NO-GO", "Energy is very modest vs. cost. Better as a small pedagogical installation, or reconsider assumptions.")
    return ("GO (pedagogy)", "Treat as an engagement/education device (like Coldplay) rather than an energy generator for a building.")


def equivalents(kwh_per_day: float) -> dict:
    """
    Convert kWh/day to very concrete local use cases.
    """
    wh = kwh_per_day * 1000
    # Typical small loads
    led_10w_hours = wh / 10.0
    screen_25w_hours = wh / 25.0
    sensor_1w_days = wh / 24.0  # 1W for 24h = 24Wh/day
    phone_charges = wh / 12.0   # ~12Wh per phone charge (rough)

    return {
        "10W LEDs (hours/day)": max(0.0, led_10w_hours),
        "25W small screen (hours/day)": max(0.0, screen_25w_hours),
        "1W sensors (sensor-days/day)": max(0.0, sensor_1w_days),
        "Phone charges (approx/day)": max(0.0, phone_charges),
    }


def load_visitors_csv(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    # normalize columns
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" not in df.columns or "visitors" not in df.columns:
        raise ValueError("CSV must contain columns: date, visitors")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["visitors"] = pd.to_numeric(df["visitors"], errors="coerce")
    df = df.dropna(subset=["visitors"])
    return df


def simple_forecast(df: pd.DataFrame, horizon_days: int = 14) -> pd.DataFrame:
    """
    Very lightweight "AI": trend + weekly seasonality baseline.
    No sklearn: uses numpy only.

    Steps:
    - Compute day index t
    - Fit linear trend visitors ~ a*t + b
    - Add weekly seasonal adjustment from average residual per weekday
    - Produce forecast + simple uncertainty band (std of residuals)
    """
    d = df.copy()
    d = d.sort_values("date")
    d["t"] = np.arange(len(d), dtype=float)
    y = d["visitors"].to_numpy(dtype=float)

    # Linear trend fit
    a, b = np.polyfit(d["t"].to_numpy(), y, 1)

    # Residuals and weekday seasonality
    d["weekday"] = d["date"].dt.weekday
    trend = a * d["t"] + b
    resid = y - trend
    wd_adj = pd.DataFrame({"weekday": d["weekday"], "resid": resid}).groupby("weekday")["resid"].mean()
    resid_std = float(np.std(resid)) if len(resid) > 2 else 0.0

    last_date = d["date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon_days, freq="D")
    t_future = np.arange(len(d), len(d) + horizon_days, dtype=float)
    base_future = a * t_future + b
    future_weekday = pd.Series(future_dates).dt.weekday
    adj_future = future_weekday.map(wd_adj).fillna(0.0).to_numpy()

    pred = base_future + adj_future
    pred = np.maximum(pred, 0.0)

    out = pd.DataFrame({
        "date": future_dates,
        "visitors_pred": pred,
        "low": np.maximum(pred - 1.0 * resid_std, 0.0),
        "high": pred + 1.0 * resid_std,
    })
    return out


def build_result_export(assumptions: dict, results: dict) -> pd.DataFrame:
    rows = []
    for k, v in assumptions.items():
        rows.append({"type": "assumption", "key": k, "value": v})
    for k, v in results.items():
        rows.append({"type": "result", "key": k, "value": v})
    return pd.DataFrame(rows)


# ----------------------------
# App config
# ----------------------------
st.set_page_config(
    page_title="Kinetic Impact Calculator (inspired by Coldplay)",
    layout="wide"
)

st.title("Kinetic Impact Calculator (inspired by Coldplay)")
st.caption(
    "Decision-support tool: estimate kinetic floor energy, size a minimal installation, compare costs vs local use cases, "
    "and keep it honest (anti-greenwashing)."
)

tabs = st.tabs(["Inputs", "Results", "Methodology"])

# ----------------------------
# Inputs tab
# ----------------------------
with tabs[0]:
    # --- Top: "Goal" makes it decision-support
    st.subheader("1) Decision goal (what do you want to power?)")

    col_goal1, col_goal2, col_goal3 = st.columns([1.2, 1.2, 1.6])

    with col_goal1:
        goal_type = st.selectbox(
            "Objective (local use case)",
            ["10W LEDs", "25W small screen", "1W sensors (24/7)", "Custom"],
            index=0
        )

    with col_goal2:
        if goal_type == "10W LEDs":
            power_w = 10
        elif goal_type == "25W small screen":
            power_w = 25
        elif goal_type == "1W sensors (24/7)":
            power_w = 1
        else:
            power_w = st.number_input("Power (W)", min_value=0.1, value=10.0, step=0.5)

        hours_per_day = st.slider("Hours per day (for the objective)", 0.0, 24.0, 6.0, 0.5)
        if goal_type == "1W sensors (24/7)":
            hours_per_day = 24.0

    with col_goal3:
        target_wh_day = power_w * hours_per_day
        st.metric("Target energy (Wh/day)", f"{target_wh_day:,.0f}")
        st.info(
            "This turns the app into a sizing tool: the app will recommend the **minimum area** "
            "to reach this objective (when possible)."
        )

    st.divider()

    # --- Layout columns like your UI
    colA, colB, colC = st.columns([1.1, 1.1, 1.0])

    with colA:
        st.subheader("Context")

        place_type = st.selectbox("Type de lieu", ["Musée", "Gare", "Stade", "Centre commercial", "Autre"], index=0)

        visitors_per_day = st.number_input("Visiteurs / jour (moyenne)", min_value=0.0, value=2050.0, step=50.0)
        peak_multiplier = st.slider("Multiplicateur pic (week-end / événement)", 0.5, 5.0, 1.0, 0.05)

        duration_h = st.slider("Durée moyenne de présence (heures)", 0.25, 8.0, 2.5, 0.25)

        st.markdown("### Flow on equipped zone")
        pct_on_zone = st.slider("% visiteurs passant sur la zone équipée", 0.1, 100.0, 5.0, 0.1)

        auto_steps = st.toggle("Auto-calcule les pas utiles à partir de la durée (plus pédagogique)", value=True)
        if auto_steps:
            # simple proxy: steps/min while on the equipped area
            steps_per_min = st.slider("Hypothèse: pas/minute sur la zone", 20, 160, 80, 5)
            minutes_on_zone = st.slider("Temps moyen passé sur la zone (minutes)", 0.5, 20.0, 3.0, 0.5)
            steps_useful = float(steps_per_min) * float(minutes_on_zone)
            st.caption(f"→ Pas utiles estimés ≈ {steps_useful:,.0f} pas/visiteur sur zone")
        else:
            steps_useful = st.slider("Pas utiles / visiteur sur zone", 1.0, 400.0, 80.0, 1.0)

        st.success("Astuce capstone : l’objectif est d’éviter la sur-installation (matériaux/maintenance) via scénarios + prévision IA.")

    with colB:
        st.subheader("Technical assumptions")

        j_per_step = st.slider("Énergie par pas (J)", 0.5, 6.0, 3.0, 0.1,
                               help="Ordre de grandeur souvent cité pour certaines dalles: ~2–4 J/pas (variable).")

        efficiency = st.slider("Rendement global", 0.05, 0.9, 0.5, 0.01,
                               help="Inclut conversion mécanique→électrique + électronique. Typiquement 30–60%.")

        storage_losses = st.slider("Pertes stockage / conversion", 0.0, 0.5, 0.10, 0.01)

        st.markdown("### Installation sizing (simple)")
        equipped_area_ft2 = st.number_input("Surface équipée (ft²)", min_value=1.0, value=190.0, step=5.0)
        tile_area_ft2 = st.number_input("Surface d’une dalle (ft²)", min_value=0.2, value=1.10, step=0.05)

        num_tiles = equipped_area_ft2 / tile_area_ft2
        st.metric("Nombre de dalles (est.)", f"{num_tiles:,.0f}")

        size_label, size_note = recommend_size(equipped_area_ft2)
        st.info(f"**Recommended label:** {size_label} — {size_note}")

    with colC:
        st.subheader("Costs")

        cost_installed_per_ft2 = st.slider("Coût installé ($/ft²)", 50.0, 900.0, 175.0, 5.0,
                                           help="Ordres de grandeur: 75–160$/ft² (typique), projets vitrines peuvent être bien plus élevés.")
        fixed_cost = st.number_input("Coût fixe (travaux/élec/signalétique) $", min_value=0.0, value=10000.0, step=1000.0)

        maint_pct = st.slider("Maintenance annuelle (% du CAPEX)", 0.0, 20.0, 8.0, 0.5)
        amort_years = st.slider("Amortissement (années)", 1, 20, 7, 1)

        capex = equipped_area_ft2 * cost_installed_per_ft2 + fixed_cost
        opex_year = capex * (maint_pct / 100.0)

        st.metric("CAPEX (est.)", dollars(capex))
        st.metric("OPEX / an (est.)", dollars(opex_year))

    st.divider()

    # --- Scenarios controls (pedagogic)
    st.subheader("2) Uncertainty (scenarios)")
    scen_col1, scen_col2 = st.columns([1.2, 1.8])
    with scen_col1:
        scen_spread = st.slider("Scenario spread (± % on key inputs)", 0, 60, 25, 5,
                                help="Creates low/medium/high scenarios by varying visitors, % on zone and steps.")
    with scen_col2:
        st.caption(
            "Instead of a single number, we show a **range**: low / mid / high. "
            "That’s more honest and better for decision-making."
        )

    # Compute mid energy
    wh_day_mid = energy_wh_per_day(
        visitors_per_day=visitors_per_day,
        peak_multiplier=peak_multiplier,
        pct_on_zone=pct_on_zone,
        steps_useful=steps_useful,
        j_per_step=j_per_step,
        efficiency=efficiency,
        storage_losses=storage_losses
    )

    # Scenarios: vary visitors, pct_on_zone, steps
    spread = scen_spread / 100.0
    def clamp_pct(x): return float(np.clip(x, 0.0, 100.0))
    visitors_low = visitors_per_day * (1 - spread)
    visitors_high = visitors_per_day * (1 + spread)
    pct_low = clamp_pct(pct_on_zone * (1 - spread))
    pct_high = clamp_pct(pct_on_zone * (1 + spread))
    steps_low = steps_useful * (1 - spread)
    steps_high = steps_useful * (1 + spread)

    wh_day_low = energy_wh_per_day(visitors_low, peak_multiplier, pct_low, steps_low, j_per_step, efficiency, storage_losses)
    wh_day_high = energy_wh_per_day(visitors_high, peak_multiplier, pct_high, steps_high, j_per_step, efficiency, storage_losses)

    # --- Objective sizing (minimum area estimate)
    # assumption: energy scales ~linearly with equipped area (same flow density)
    # required_area = current_area * target_wh / current_wh
    if wh_day_mid > 0:
        required_area_ft2 = equipped_area_ft2 * (target_wh_day / wh_day_mid)
    else:
        required_area_ft2 = float("inf")

    required_area_ft2 = max(0.0, required_area_ft2)
    required_tiles = required_area_ft2 / tile_area_ft2 if math.isfinite(required_area_ft2) else float("inf")

    target_met = wh_day_mid >= target_wh_day

    # --- Quick range preview (pedagogic, directly on Inputs)
    kwh_day_mid = fmt_kwh(wh_day_mid)
    kwh_day_low = fmt_kwh(wh_day_low)
    kwh_day_high = fmt_kwh(wh_day_high)

    st.subheader("3) Quick preview (range)")

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("kWh/day (mid)", f"{kwh_day_mid:,.3f}")
    p2.metric("Range (low → high)", f"{kwh_day_low:,.3f} → {kwh_day_high:,.3f}")
    p3.metric("Target met?", "Yes ✅" if target_met else "Not yet ❌")
    if math.isfinite(required_area_ft2):
        p4.metric("Min area to hit target (ft²)", f"{required_area_ft2:,.0f}")
    else:
        p4.metric("Min area to hit target (ft²)", "—")

    # Anti-greenwashing box (super important for capstone)
    st.warning(
        "⚠️ **Reality check (anti-greenwashing):** Kinetic floors usually produce **modest energy**. "
        "They are best for **local loads** (LEDs, sensors, small display) + **public engagement** (making energy tangible), "
        "not powering a building."
    )

    # Store state for Results tab
    st.session_state["inputs"] = {
        "place_type": place_type,
        "visitors_per_day": visitors_per_day,
        "peak_multiplier": peak_multiplier,
        "duration_h": duration_h,
        "pct_on_zone": pct_on_zone,
        "steps_useful": steps_useful,
        "j_per_step": j_per_step,
        "efficiency": efficiency,
        "storage_losses": storage_losses,
        "equipped_area_ft2": equipped_area_ft2,
        "tile_area_ft2": tile_area_ft2,
        "cost_installed_per_ft2": cost_installed_per_ft2,
        "fixed_cost": fixed_cost,
        "maint_pct": maint_pct,
        "amort_years": amort_years,
        "goal_type": goal_type,
        "power_w": power_w,
        "hours_per_day": hours_per_day,
        "target_wh_day": target_wh_day,
        "scenario_spread_pct": scen_spread,
    }
    st.session_state["computed"] = {
        "wh_day_low": wh_day_low,
        "wh_day_mid": wh_day_mid,
        "wh_day_high": wh_day_high,
        "required_area_ft2": required_area_ft2,
        "required_tiles": required_tiles,
        "target_met": target_met,
        "capex": capex,
        "opex_year": opex_year,
    }

# ----------------------------
# Results tab
# ----------------------------
with tabs[1]:
    st.subheader("Results (decision support)")

    if "inputs" not in st.session_state:
        st.info("Go to the Inputs tab first.")
        st.stop()

    inp = st.session_state["inputs"]
    comp = st.session_state["computed"]

    wh_low, wh_mid, wh_high = comp["wh_day_low"], comp["wh_day_mid"], comp["wh_day_high"]
    kwh_day_low, kwh_day_mid, kwh_day_high = fmt_kwh(wh_low), fmt_kwh(wh_mid), fmt_kwh(wh_high)

    kwh_month_mid = kwh_day_mid * 30.0
    kwh_year_mid = kwh_day_mid * 365.0

    # Economics
    capex = comp["capex"]
    opex_year = comp["opex_year"]
    amort_years = inp["amort_years"]
    total_cost_over_amort = capex + opex_year * amort_years
    kwh_over_amort = kwh_year_mid * amort_years
    cost_per_kwh = (total_cost_over_amort / kwh_over_amort) if kwh_over_amort > 0 else float("inf")

    # Cost per Wh/day useful (more intuitive)
    cost_per_wh_day = (capex / inp["target_wh_day"]) if inp["target_wh_day"] > 0 else float("inf")

    # Go / No-go
    verdict, verdict_reason = go_no_go(kwh_year_mid, cost_per_kwh, comp["target_met"])

    # Top summary
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("kWh/day (mid)", f"{kwh_day_mid:,.3f}")
    c2.metric("kWh/month (mid)", f"{kwh_month_mid:,.2f}")
    c3.metric("kWh/year (mid)", f"{kwh_year_mid:,.1f}")
    c4.metric("Range (low → high)", f"{kwh_day_low:,.3f} → {kwh_day_high:,.3f}")

    st.markdown("### What can it power (concrete)")
    eq = equivalents(kwh_day_mid)
    eq_df = pd.DataFrame({"Use case": list(eq.keys()), "Equivalent": [eq[k] for k in eq]})
    st.dataframe(eq_df, use_container_width=True, hide_index=True)

    st.markdown("### Installation sizing vs your objective")
    req_area = comp["required_area_ft2"]
    req_tiles = comp["required_tiles"]

    sc1, sc2, sc3 = st.columns([1.1, 1.1, 1.2])
    with sc1:
        st.metric("Current equipped area (ft²)", f"{inp['equipped_area_ft2']:,.0f}")
        st.metric("Estimated tiles", f"{(inp['equipped_area_ft2']/inp['tile_area_ft2']):,.0f}")
    with sc2:
        if math.isfinite(req_area):
            st.metric("Min area to hit objective (ft²)", f"{req_area:,.0f}")
            st.metric("Min tiles to hit objective", f"{req_tiles:,.0f}")
        else:
            st.metric("Min area to hit objective (ft²)", "—")
            st.metric("Min tiles to hit objective", "—")
    with sc3:
        st.metric("Target (Wh/day)", f"{inp['target_wh_day']:,.0f}")
        st.metric("Target met?", "Yes ✅" if comp["target_met"] else "Not yet ❌")
        st.caption("Assumption: energy scales ~linearly with equipped area (same flow density).")

    st.markdown("### Costs & decision metrics")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("CAPEX", dollars(capex))
    d2.metric("OPEX/year", dollars(opex_year))
    d3.metric("Cost per kWh (amort.)", "∞" if not math.isfinite(cost_per_kwh) else dollars2(cost_per_kwh))
    d4.metric("CAPEX per target Wh/day", "∞" if not math.isfinite(cost_per_wh_day) else dollars2(cost_per_wh_day))

    st.markdown("### Verdict")
    if verdict.startswith("GO"):
        st.success(f"**{verdict}** — {verdict_reason}")
    else:
        st.error(f"**{verdict}** — {verdict_reason}")

    st.divider()

    # ----------------------------
    # Sustainable AI block: forecast + scenarios + CodeCarbon
    # ----------------------------
    st.subheader("Sustainable AI module: forecast visitors (lightweight) + avoid over-installation")

    ai_col1, ai_col2 = st.columns([1.2, 1.0])

    with ai_col1:
        use_demo = st.toggle("Use demo dataset", value=True)
        uploaded = st.file_uploader("Upload CSV (columns: date, visitors)", type=["csv"])

        df_hist = None
        if use_demo:
            # Small synthetic demo
            demo_dates = pd.date_range("2025-11-01", periods=10, freq="D")
            demo_vis = [1200, 1500, 900, 950, 980, 1100, 1300, 1600, 1800, 1000]
            df_hist = pd.DataFrame({"date": demo_dates, "visitors": demo_vis})
        elif uploaded is not None:
            try:
                df_hist = load_visitors_csv(uploaded)
            except Exception as e:
                st.error(str(e))

        if df_hist is not None:
            st.dataframe(df_hist, use_container_width=True, hide_index=True)

    with ai_col2:
        horizon = st.slider("Forecast horizon (days)", 7, 60, 14, 1)

        measure_ai = st.toggle("Measure AI footprint with CodeCarbon", value=True) if CODECARBON_AVAILABLE else False
        if not CODECARBON_AVAILABLE:
            st.caption("CodeCarbon not installed (optional). Add it to requirements.txt to enable footprint tracking.")

        run_forecast = st.button("Run forecast + scenarios")

        emissions = None
        if run_forecast and df_hist is not None:
            tracker = None
            if measure_ai and CODECARBON_AVAILABLE:
                tracker = EmissionsTracker(project_name="kinetic-impact-forecast", output_dir=".codecarbon", log_level="error")
                tracker.start()

            fc = simple_forecast(df_hist, horizon_days=horizon)

            if tracker is not None:
                emissions = tracker.stop()

            st.success("Forecast generated.")

            st.markdown("**Forecast (visitors/day)**")
            st.dataframe(fc, use_container_width=True, hide_index=True)

            # Energy forecast using median assumptions (mid)
            fc_mid_wh = []
            for v in fc["visitors_pred"].to_numpy():
                fc_mid_wh.append(energy_wh_per_day(
                    visitors_per_day=float(v),
                    peak_multiplier=inp["peak_multiplier"],
                    pct_on_zone=inp["pct_on_zone"],
                    steps_useful=inp["steps_useful"],
                    j_per_step=inp["j_per_step"],
                    efficiency=inp["efficiency"],
                    storage_losses=inp["storage_losses"]
                ))
            fc["kwh_pred"] = np.array(fc_mid_wh) / 1000.0

            st.markdown("**Forecasted energy (kWh/day)**")
            st.line_chart(fc.set_index("date")["kwh_pred"])

            if emissions is not None:
                st.info(f"CodeCarbon estimate for this run: **{emissions:.6f} kgCO₂e**")

    st.divider()

    # Export results
    st.subheader("Export (assumptions + results)")

    export_assumptions = {
        "place_type": inp["place_type"],
        "visitors_per_day": inp["visitors_per_day"],
        "peak_multiplier": inp["peak_multiplier"],
        "pct_on_zone": inp["pct_on_zone"],
        "steps_useful": inp["steps_useful"],
        "j_per_step": inp["j_per_step"],
        "efficiency": inp["efficiency"],
        "storage_losses": inp["storage_losses"],
        "equipped_area_ft2": inp["equipped_area_ft2"],
        "tile_area_ft2": inp["tile_area_ft2"],
        "cost_installed_per_ft2": inp["cost_installed_per_ft2"],
        "fixed_cost": inp["fixed_cost"],
        "maint_pct": inp["maint_pct"],
        "amort_years": inp["amort_years"],
        "goal_type": inp["goal_type"],
        "power_w": inp["power_w"],
        "hours_per_day": inp["hours_per_day"],
        "target_wh_day": inp["target_wh_day"],
        "scenario_spread_pct": inp["scenario_spread_pct"],
    }

    export_results = {
        "kwh_day_low": kwh_day_low,
        "kwh_day_mid": kwh_day_mid,
        "kwh_day_high": kwh_day_high,
        "kwh_year_mid": kwh_year_mid,
        "capex": capex,
        "opex_year": opex_year,
        "cost_per_kwh_amort": cost_per_kwh if math.isfinite(cost_per_kwh) else None,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "min_area_ft2_for_target": comp["required_area_ft2"] if math.isfinite(comp["required_area_ft2"]) else None,
        "min_tiles_for_target": comp["required_tiles"] if math.isfinite(comp["required_tiles"]) else None,
    }

    export_df = build_result_export(export_assumptions, export_results)
    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download results (CSV)", data=csv_bytes, file_name="kinetic_impact_results.csv", mime="text/csv")

# ----------------------------
# Methodology tab
# ----------------------------
with tabs[2]:
    st.subheader("Methodology & transparency")

    st.markdown(
        """
### Energy model (simple, defensable)
We compute energy as:

**Energy (Wh/day) = visitors/day × peak_multiplier × (% on zone) × useful_steps × J/step × efficiency × (1 − losses) ÷ 3600**

This is intentionally transparent: you can defend it in a capstone.

### Why “Sustainable AI” here?
The AI module forecasts visitor flow with a lightweight model (trend + weekly seasonality).
It supports **minimal installation sizing** (avoid over-installing tiles → less material/maintenance).
Optionally, CodeCarbon can measure the footprint of the forecast run.

### Anti-greenwashing note
Kinetic floors typically produce **modest energy**. The best value is often:
- **local loads** (LEDs, sensors, small display)
- **public engagement** (making energy tangible — like Coldplay’s narrative)
- **decision support** (size minimal installation and be honest about costs)
        """
    )
