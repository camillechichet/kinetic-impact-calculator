import io
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd
import streamlit as st

# CodeCarbon (mesure CO2 du calcul IA)
try:
    from codecarbon import EmissionsTracker
    CODECARBON_OK = True
except Exception:
    CODECARBON_OK = False


st.set_page_config(page_title="Kinetic Impact Calculator (inspired by Coldplay)", layout="wide")

# -----------------------------
# Helpers / core calculations
# -----------------------------
@dataclass
class Inputs:
    place_type: str
    visitors_per_day: float
    peak_multiplier: float
    avg_duration_hours: float
    pct_on_zone: float
    useful_steps_per_person: float
    joules_per_step: float
    efficiency: float
    storage_losses: float

    # Costs
    installed_cost_per_ft2: float
    fixed_cost: float
    maintenance_pct_per_year: float
    amort_years: int

    # Installation
    surface_ft2: float
    tile_area_ft2: float

def energy_wh_per_day(visitors_per_day, peak_multiplier, pct_on_zone, useful_steps, j_per_step, eff, losses):
    # Wh/day = visitors * peak * (%passage) * steps * J/step * efficiency * (1-losses) / 3600
    visitors_effective = visitors_per_day * peak_multiplier
    steps_captured = visitors_effective * (pct_on_zone / 100.0) * useful_steps
    wh = steps_captured * j_per_step * eff * (1.0 - losses) / 3600.0
    return wh

def wh_to_kwh(wh): 
    return wh / 1000.0

def kwh_periods(kwh_per_day):
    return {
        "kWh/day": kwh_per_day,
        "kWh/month (~30d)": kwh_per_day * 30.0,
        "kWh/year (~365d)": kwh_per_day * 365.0,
    }

def capex_opex_costs(surface_ft2, cost_per_ft2, fixed_cost, maint_pct, amort_years):
    capex = surface_ft2 * cost_per_ft2 + fixed_cost
    opex_per_year = (maint_pct / 100.0) * capex
    return capex, opex_per_year, amort_years

def cost_per_kwh(capex, opex_per_year, years, kwh_per_year):
    if kwh_per_year <= 0:
        return np.inf
    total_cost = capex + opex_per_year * years
    total_kwh = kwh_per_year * years
    return total_cost / total_kwh

def go_nogo(kwh_per_day, cost_kwh, led10w_hours, screen25w_hours):
    # Règles simples (transparentes)
    # - "Go pédagogique" si tu peux alimenter une petite charge quelques heures
    # - "Go énergétique" rare, si coût/kWh pas délirant (seuil arbitraire mais utile décision)
    go_pedago = (led10w_hours >= 2.0) or (screen25w_hours >= 1.0)
    go_energy = (cost_kwh <= 1.0) and (kwh_per_day >= 0.5)  # seuils simples, modifiables si besoin

    if go_energy:
        return "GO ✅ (énergie + pédagogie)", "Energy is meaningful vs cost (by our simple thresholds)."
    if go_pedago:
        return "GO (pédagogie/local) ✅", "Energy is modest but enough for local devices + engagement."
    return "NO-GO ❌", "Energy is too low for meaningful use; consider alternative or smaller pilot."

def scenario_pack(inp: Inputs):
    # 3 scénarios: bas / moyen / haut (incertitude)
    scenarios = [
        ("Low", 0.8, 0.8, max(0.30, inp.efficiency - 0.10)),
        ("Mid", 1.0, 1.0, inp.efficiency),
        ("High", 1.2, 1.2, min(0.80, inp.efficiency + 0.10)),
    ]
    rows = []
    for name, v_mult, s_mult, eff in scenarios:
        wh = energy_wh_per_day(
            inp.visitors_per_day * v_mult,
            inp.peak_multiplier,
            inp.pct_on_zone,
            inp.useful_steps_per_person * s_mult,
            inp.joules_per_step,
            eff,
            inp.storage_losses,
        )
        kwhd = wh_to_kwh(wh)
        rows.append({"Scenario": name, "kWh/day": kwhd, "kWh/month": kwhd*30, "kWh/year": kwhd*365, "efficiency": eff})
    return pd.DataFrame(rows)

# IA légère : prévision visiteurs (sans sklearn)
def forecast_visitors_linear(df, horizon_days=14):
    """
    df must have columns: date, visitors
    returns forecast dataframe with date + visitors_pred
    """
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").dropna(subset=["date", "visitors"])
    d["visitors"] = pd.to_numeric(d["visitors"], errors="coerce")
    d = d.dropna(subset=["visitors"])

    if len(d) < 5:
        raise ValueError("Need at least 5 rows to forecast.")

    # Convert dates to day index
    x = (d["date"] - d["date"].min()).dt.days.values.astype(float)
    y = d["visitors"].values.astype(float)

    # Fit simple trend line
    coeff = np.polyfit(x, y, deg=1)  # slope, intercept
    slope, intercept = coeff[0], coeff[1]

    last_day = int(x.max())
    future_x = np.arange(last_day + 1, last_day + 1 + horizon_days).astype(float)
    future_dates = [d["date"].min() + timedelta(days=int(i)) for i in future_x]
    y_pred = slope * future_x + intercept
    y_pred = np.clip(y_pred, 0, None)  # no negative visitors

    out = pd.DataFrame({"date": future_dates, "visitors_pred": y_pred})
    return out

def run_codecarbon():
    if not CODECARBON_OK:
        return None, None
    try:
        tracker = EmissionsTracker(save_to_file=False, log_level="error")
        tracker.start()
        return tracker, None
    except Exception as e:
        return None, str(e)


# -----------------------------
# UI
# -----------------------------
st.title("Kinetic Impact Calculator (inspired by Coldplay)")
st.caption("MVP : énergie (kWh), usages locaux (LED/écran), coûts (CAPEX/OPEX), Go/No-Go, + IA légère (prévision) + CodeCarbon.")

tabs = st.tabs(["Inputs", "Results", "Methodology"])

# -----------------------------
# Inputs tab
# -----------------------------
with tabs[0]:
    c1, c2, c3 = st.columns([1.2, 1.2, 1.0], gap="large")

    with c1:
        st.subheader("Context")
        place_type = st.selectbox("Type de lieu", ["Musée", "Gare", "Stade", "Centre commercial", "Autre"])
        visitors_per_day = st.number_input("Visiteurs / jour (moyenne)", min_value=0.0, value=2000.0, step=50.0)
        peak_multiplier = st.slider("Multiplicateur pic (week-end / événement)", 1.0, 5.0, 1.5, 0.1)
        avg_duration_hours = st.slider("Durée moyenne de présence (heures)", 0.5, 8.0, 2.0, 0.5)

        st.subheader("Flow on equipped zone")
        pct_on_zone = st.slider("% visiteurs passant sur la zone équipée", 0.0, 30.0, 5.0, 0.5)
        useful_steps = st.slider("Pas utiles / visiteur sur zone", 0.0, 300.0, 80.0, 5.0)

    with c2:
        st.subheader("Technical assumptions")
        joules_per_step = st.slider("Énergie par pas (J)", 1.0, 6.0, 3.0, 0.1)
        efficiency = st.slider("Rendement global", 0.1, 0.9, 0.5, 0.05)
        storage_losses = st.slider("Pertes stockage / conversion", 0.0, 0.5, 0.1, 0.05)

        st.subheader("Installation sizing (simple)")
        surface_ft2 = st.number_input("Surface équipée (ft²)", min_value=1.0, value=120.0, step=10.0)
        tile_area_ft2 = st.number_input("Surface d’une dalle (ft²)", min_value=0.1, value=1.0, step=0.1)

    with c3:
        st.subheader("Costs")
        installed_cost_per_ft2 = st.slider("Coût installé ($/ft²)", 50.0, 900.0, 120.0, 5.0)
        fixed_cost = st.number_input("Coût fixe (travaux/élec/signalétique) $", min_value=0.0, value=10000.0, step=500.0)
        maintenance_pct = st.slider("Maintenance annuelle (% du CAPEX)", 0.0, 20.0, 8.0, 0.5)
        amort_years = st.slider("Amortissement (années)", 1, 20, 7, 1)

    inp = Inputs(
        place_type=place_type,
        visitors_per_day=visitors_per_day,
        peak_multiplier=peak_multiplier,
        avg_duration_hours=avg_duration_hours,
        pct_on_zone=pct_on_zone,
        useful_steps_per_person=useful_steps,
        joules_per_step=joules_per_step,
        efficiency=efficiency,
        storage_losses=storage_losses,
        installed_cost_per_ft2=installed_cost_per_ft2,
        fixed_cost=fixed_cost,
        maintenance_pct_per_year=maintenance_pct,
        amort_years=amort_years,
        surface_ft2=surface_ft2,
        tile_area_ft2=tile_area_ft2,
    )

    st.info("✅ Astuce capstone : tu peux dire que l’objectif est *d’éviter la sur-installation* (matériaux/maintenance) via scénarios + prévision IA.")

# -----------------------------
# Results tab
# -----------------------------
with tabs[1]:
    # Base energy
    wh_day = energy_wh_per_day(
        inp.visitors_per_day,
        inp.peak_multiplier,
        inp.pct_on_zone,
        inp.useful_steps_per_person,
        inp.joules_per_step,
        inp.efficiency,
        inp.storage_losses,
    )
    kwh_day = wh_to_kwh(wh_day)
    periods = kwh_periods(kwh_day)

    # Equivalences (local use)
    led10w_hours = (kwh_day * 1000.0) / 10.0 if kwh_day > 0 else 0.0
    screen25w_hours = (kwh_day * 1000.0) / 25.0 if kwh_day > 0 else 0.0
    sensor1w_hours = (kwh_day * 1000.0) / 1.0 if kwh_day > 0 else 0.0

    # Costs
    capex, opex_per_year, years = capex_opex_costs(
        inp.surface_ft2, inp.installed_cost_per_ft2, inp.fixed_cost, inp.maintenance_pct_per_year, inp.amort_years
    )
    cost_kwh = cost_per_kwh(capex, opex_per_year, years, periods["kWh/year (~365d)"])

    # Go/No-Go
    verdict, reason = go_nogo(kwh_day, cost_kwh, led10w_hours, screen25w_hours)

    # Tiles
    num_tiles = int(np.ceil(inp.surface_ft2 / inp.tile_area_ft2))

    st.subheader("Key outputs")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("kWh / day", f"{periods['kWh/day']:.3f}")
    k2.metric("kWh / month", f"{periods['kWh/month (~30d)']:.2f}")
    k3.metric("kWh / year", f"{periods['kWh/year (~365d)']:.1f}")
    k4.metric("Estimated tiles", f"{num_tiles}")

    st.subheader("What can it power (local + pedagogy)")
    st.write(
        f"- **LED strip (10W)**: ~**{led10w_hours:.1f} h/day**\n"
        f"- **Small screen (25W)**: ~**{screen25w_hours:.1f} h/day**\n"
        f"- **Sensor (1W)**: ~**{sensor1w_hours:.0f} h/day**"
    )

    st.subheader("Business / decision")
    b1, b2, b3 = st.columns(3)
    b1.metric("CAPEX (est.)", f"${capex:,.0f}")
    b2.metric("OPEX / year (est.)", f"${opex_per_year:,.0f}")
    b3.metric("Cost per kWh (est.)", "∞" if not np.isfinite(cost_kwh) else f"${cost_kwh:,.2f}")

    st.markdown(f"### Verdict: **{verdict}**")
    st.caption(reason)
    st.warning("⚠️ Note anti-greenwashing: l’énergie est souvent **modeste** — l’intérêt est surtout **local + engagement/pédagogie** (ex: Coldplay).")

    st.subheader("Uncertainty scenarios (Low / Mid / High)")
    df_s = scenario_pack(inp)
    st.dataframe(df_s, use_container_width=True)

    # Export results
    st.subheader("Export")
    export = {
        **periods,
        "LED_10W_hours_per_day": led10w_hours,
        "Screen_25W_hours_per_day": screen25w_hours,
        "CAPEX": capex,
        "OPEX_per_year": opex_per_year,
        "Cost_per_kWh": cost_kwh,
        "Verdict": verdict,
        "Reason": reason,
        "Surface_ft2": inp.surface_ft2,
        "Tiles": num_tiles,
    }
    df_export = pd.DataFrame([export])
    csv_bytes = df_export.to_csv(index=False).encode("utf-8")
    st.download_button("Download results (CSV)", data=csv_bytes, file_name="kinetic_impact_results.csv", mime="text/csv")

    st.divider()

    # -----------------------------
    # IA Section: Forecast visitors + CodeCarbon
    # -----------------------------
    st.subheader("Sustainable AI (lightweight): visitor forecast → better sizing (avoid over-installation)")
    st.caption("Upload a CSV with columns: date, visitors (or use demo). Then we predict the next 10 days with a tiny linear model.")

    use_demo = st.checkbox("Use demo dataset", value=True)
    uploaded = st.file_uploader("Upload CSV", type=["csv"], disabled=use_demo)

    if use_demo:
        demo = pd.DataFrame({
            "date": pd.date_range("2025-11-01", periods=10, freq="D"),
            "visitors": [1200, 1500, 900, 950, 980, 1100, 1300, 1600, 1800, 1000],
        })
        df_in = demo
    else:
        df_in = None
        if uploaded is not None:
            df_in = pd.read_csv(uploaded)

    if df_in is not None:
        st.dataframe(df_in, use_container_width=True)

        horizon = st.slider("Forecast horizon (days)", 5, 60, 10, 1)
        run = st.button("Run forecast + measure footprint (CodeCarbon)")

        if run:
            tracker, cc_err = run_codecarbon()

            try:
                # Forecast
                df_in2 = df_in.copy()
                # Normalize column names
                df_in2.columns = [c.strip().lower() for c in df_in2.columns]
                if "date" not in df_in2.columns or "visitors" not in df_in2.columns:
                    st.error("CSV must contain columns: date, visitors")
                else:
                    forecast = forecast_visitors_linear(df_in2[["date", "visitors"]], horizon_days=horizon)
                    st.success("Forecast generated.")

                    st.dataframe(forecast, use_container_width=True)

                    # Convert forecast to energy range using current assumptions (mid scenario)
                    # Use predicted visitors as visitors/day (average) and compute kWh/day
                    tmp = forecast.copy()
                    tmp["kWh_day_pred"] = tmp["visitors_pred"].apply(
                        lambda v: wh_to_kwh(
                            energy_wh_per_day(
                                v,
                                inp.peak_multiplier,
                                inp.pct_on_zone,
                                inp.useful_steps_per_person,
                                inp.joules_per_step,
                                inp.efficiency,
                                inp.storage_losses,
                            )
                        )
                    )
                    st.line_chart(tmp.set_index("date")[["kWh_day_pred"]])

                    # CodeCarbon results
                    emissions = None
                    if tracker is not None:
                        try:
                            emissions = tracker.stop()  # kgCO2e
                        except Exception:
                            emissions = None

                    if CODECARBON_OK and emissions is not None:
                        st.info(f"CodeCarbon estimate (kgCO₂e): **{emissions:.6f}**")
                    elif CODECARBON_OK and cc_err:
                        st.warning(f"CodeCarbon could not start: {cc_err}")
                    elif not CODECARBON_OK:
                        st.warning("CodeCarbon not available (missing dependency).")

            except Exception as e:
                if tracker is not None:
                    try:
                        tracker.stop()
                    except Exception:
                        pass
                st.error(f"Forecast error: {e}")

# -----------------------------
# Methodology tab
# -----------------------------
with tabs[2]:
    st.subheader("Formula (transparent)")
    st.markdown(
        """
**Steps captured/day** = visitors/day × peak_multiplier × (% on zone) × useful_steps/person

**Energy (Wh/day)** = steps_captured × J_per_step × efficiency × (1 - losses) ÷ 3600

This is an **order-of-magnitude** model to support decisions (not a guarantee).
"""
    )

    st.subheader("Interpretation (anti-greenwashing)")
    st.markdown(
        """
- Output energy is often **modest** → best for **local loads** (LEDs, sensors, small screen).
- The main value can be **engagement/pedagogy** (making energy tangible), plus **better sizing** (avoid wasting materials).
"""
    )

    st.subheader("What makes it 'Sustainable AI'")
    st.markdown(
        """
- The forecast model is **tiny** (simple trend line) → cheap in compute.
- Goal: **avoid over-installation** (materials, maintenance, CAPEX) by sizing based on predicted demand.
- CodeCarbon provides a **footprint estimate** for the AI run.
"""
    )
