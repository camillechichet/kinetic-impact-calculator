import math
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# Optional (won't crash if not installed)
try:
    from codecarbon import EmissionsTracker
except Exception:
    EmissionsTracker = None


# -----------------------------
# Helpers
# -----------------------------
def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def energy_wh_per_day(visitors_per_day, peak_multiplier, pct_on_zone, useful_steps,
                      j_per_step, efficiency, storage_loss):
    """
    Energy (Wh/day) = visitors * peak_multiplier * (pct_on_zone/100) * useful_steps * J/step
                      * efficiency * (1 - storage_loss) / 3600
    """
    visitors = visitors_per_day * peak_multiplier
    steps_captured = visitors * (pct_on_zone / 100.0) * useful_steps
    wh = steps_captured * j_per_step * efficiency * (1.0 - storage_loss) / 3600.0
    return max(0.0, wh)


def wh_to_kwh(wh):
    return wh / 1000.0


def kwh_period(kwh_day, days):
    return kwh_day * days


def equivalences(kwh):
    # Very rough, but pedagogical equivalences (avoid claiming powering buildings)
    # 1 LED bulb 10W for 1 hour = 0.01 kWh
    led_10w_hours = kwh / 0.01 if kwh > 0 else 0
    # phone charge ~ 12 Wh = 0.012 kWh
    phone_charges = kwh / 0.012 if kwh > 0 else 0
    # small sensor ~ 1W continuous = 0.024 kWh/day
    sensor_days = kwh / 0.024 if kwh > 0 else 0
    return led_10w_hours, phone_charges, sensor_days


def go_nogo(cost_per_kwh, kwh_year, pedagogical_weight=True):
    """
    Simple decision logic:
    - If kWh/year is tiny, it's mainly pedagogical.
    - If cost/kWh is huge, No-Go for "energy ROI" but can still be Go for engagement.
    """
    if kwh_year < 20:
        return "NO-GO (energy ROI)", "Energy output is very small; consider it primarily an engagement / educational installation."
    if cost_per_kwh > 1.0:
        if pedagogical_weight:
            return "GO (pedagogical/local use)", "Cost per kWh is high for pure energy ROI, but it can still make sense for education + local low-power use."
        return "NO-GO", "Cost per kWh is too high compared to typical electricity prices."
    return "GO", "Energy and cost look reasonable for a small local use case + engagement."


def simple_forecast(df, horizon_days=14):
    """
    Lightweight forecast WITHOUT sklearn.
    - expects columns: date, visitors
    - converts date to ordinal and fits a simple linear trend using numpy polyfit
    - outputs forecast for next horizon_days
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df = df.dropna(subset=["date", "visitors"])

    if len(df) < 3:
        # fallback: constant forecast
        last = float(df["visitors"].iloc[-1]) if len(df) else 0.0
        future_dates = pd.date_range(df["date"].max() + pd.Timedelta(days=1), periods=horizon_days, freq="D")
        preds = np.full(horizon_days, last)
        return pd.DataFrame({"date": future_dates, "visitors_pred": preds})

    x = df["date"].map(datetime.toordinal).to_numpy(dtype=float)
    y = df["visitors"].to_numpy(dtype=float)

    # linear trend
    coeff = np.polyfit(x, y, deg=1)  # y = a*x + b
    a, b = coeff[0], coeff[1]

    last_date = df["date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon_days, freq="D")
    x_future = future_dates.map(datetime.toordinal).to_numpy(dtype=float)
    preds = a * x_future + b
    preds = np.maximum(preds, 0.0)  # no negative visitors
    return pd.DataFrame({"date": future_dates, "visitors_pred": preds})


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Kinetic Impact Calculator (inspired by Coldplay)", layout="wide")

st.title("Kinetic Impact Calculator (inspired by Coldplay)")
st.caption(
    "Decision-support MVP: estimate kinetic floor energy, practical uses, costs (CAPEX/OPEX), uncertainty scenarios, "
    "and a lightweight forecast module (Sustainable AI)."
)

tabs = st.tabs(["Inputs", "Results", "Methodology / Limits"])

with tabs[0]:
    colA, colB, colC = st.columns([1.1, 1.1, 1.1])

    with colA:
        st.subheader("Context")
        place_type = st.selectbox("Type de lieu", ["Musée", "Gare", "Stade", "Centre commercial", "Autre"], index=0)

        visitors_day = st.number_input("Visiteurs / jour (moyenne)", min_value=0.0, value=3300.0, step=50.0, format="%.2f")
        peak_mult = st.slider("Multiplicateur pic (week-end / événement)", min_value=0.5, max_value=5.0, value=1.0, step=0.05)

        dwell_hours = st.slider("Durée moyenne de présence (heures)", 0.25, 8.0, 2.5, 0.25)

        st.subheader("Flow on equipped zone")
        pct_zone = st.slider("% visiteurs passant sur la zone équipée", 0.5, 30.0, 12.0, 0.5)
        useful_steps = st.slider("Pas utiles / visiteur sur zone", 5, 300, 115, 5)

    with colB:
        st.subheader("Technical assumptions")
        j_per_step = st.slider("Énergie par pas (J)", 1.0, 5.0, 3.0, 0.1)
        efficiency = st.slider("Rendement global", 0.1, 0.9, 0.5, 0.05)
        storage_loss = st.slider("Pertes stockage / conversion", 0.0, 0.5, 0.10, 0.02)

        st.subheader("Installation sizing (simple)")
        equipped_area_ft2 = st.number_input("Surface équipée (ft²)", min_value=1.0, value=190.0, step=5.0, format="%.2f")
        tile_area_ft2 = st.number_input("Surface d’une dalle (ft²)", min_value=0.1, value=1.10, step=0.05, format="%.2f")

        # Simple recommendation based on area
        tiles = equipped_area_ft2 / tile_area_ft2
        st.info(f"Estimation: ~ **{int(round(tiles))} dalles** pour {equipped_area_ft2:.0f} ft² (si 1 dalle ≈ {tile_area_ft2:.2f} ft²).")

    with colC:
        st.subheader("Costs")
        cost_per_ft2 = st.slider("Coût installé ($/ft²)", 50.0, 900.0, 175.0, 5.0)
        fixed_cost = st.number_input("Coût fixe (travaux/élec/signalétique) $", min_value=0.0, value=20000.0, step=1000.0, format="%.2f")
        maint_pct = st.slider("Maintenance annuelle (% du CAPEX)", 0.0, 20.0, 8.0, 0.5)
        amort_years = st.slider("Amortissement (années)", 1, 20, 9, 1)

        st.subheader("Sustainable AI (lightweight)")
        st.write("Option: importer un CSV (colonnes `date`, `visitors`) pour prévoir la fréquentation.")
        use_demo = st.checkbox("Utiliser dataset démo", value=True)

        demo_path = "data/sample_visitors.csv"
        uploaded = st.file_uploader("Upload CSV", type=["csv"])

        df_vis = None
        if uploaded is not None:
            try:
                df_vis = pd.read_csv(uploaded)
            except Exception as e:
                st.error(f"Impossible de lire le CSV: {e}")

        if df_vis is None and use_demo:
            try:
                df_vis = pd.read_csv(demo_path)
            except Exception:
                df_vis = None

        if df_vis is not None:
            st.dataframe(df_vis.head(10), use_container_width=True)

            horizon = st.slider("Horizon de prévision (jours)", 7, 60, 14, 1)
            run_forecast = st.button("Lancer prévision (IA légère) + mesurer empreinte (CodeCarbon)")

            if run_forecast:
                tracker = None
                if EmissionsTracker is not None:
                    try:
                        tracker = EmissionsTracker(log_level="error")
                        tracker.start()
                    except Exception:
                        tracker = None

                try:
                    # validate columns
                    if not {"date", "visitors"}.issubset(set(df_vis.columns)):
                        st.error("Le CSV doit contenir les colonnes: `date`, `visitors`.")
                    else:
                        df_vis2 = df_vis[["date", "visitors"]].copy()
                        df_fore = simple_forecast(df_vis2, horizon_days=horizon)
                        st.success("Prévision générée.")
                        st.dataframe(df_fore, use_container_width=True)

                        # simple visualization
                        df_plot = df_vis2.copy()
                        df_plot["date"] = pd.to_datetime(df_plot["date"])
                        df_plot = df_plot.sort_values("date")

                        df_fore_plot = df_fore.copy()
                        df_fore_plot["date"] = pd.to_datetime(df_fore_plot["date"])

                        st.line_chart(
                            pd.concat([
                                df_plot.rename(columns={"visitors": "value"}).assign(series="history")[["date", "value", "series"]],
                                df_fore_plot.rename(columns={"visitors_pred": "value"}).assign(series="forecast")[["date", "value", "series"]],
                            ]).pivot(index="date", columns="series", values="value"),
                            use_container_width=True,
                        )

                        # Store in session for Results tab
                        st.session_state["forecast_df"] = df_fore_plot

                finally:
                    if tracker is not None:
                        try:
                            emissions = tracker.stop()
                            st.info(f"CodeCarbon (kgCO₂e): **{emissions:.6f}**")
                            st.session_state["last_emissions"] = emissions
                        except Exception:
                            pass

with tabs[1]:
    st.subheader("Results (energy + uncertainty + costs + Go/No-Go)")

    # Baseline energy
    wh_day = energy_wh_per_day(
        visitors_per_day=visitors_day,
        peak_multiplier=peak_mult,
        pct_on_zone=pct_zone,
        useful_steps=useful_steps,
        j_per_step=j_per_step,
        efficiency=efficiency,
        storage_loss=storage_loss,
    )
    kwh_day = wh_to_kwh(wh_day)
    kwh_month = kwh_period(kwh_day, 30)
    kwh_year = kwh_period(kwh_day, 365)

    # Uncertainty scenarios (simple ranges)
    low = energy_wh_per_day(visitors_day, peak_mult, pct_zone * 0.7, int(useful_steps * 0.7), j_per_step * 0.9, efficiency * 0.9, storage_loss)
    mid = wh_day
    high = energy_wh_per_day(visitors_day, peak_mult, pct_zone * 1.3, int(useful_steps * 1.3), j_per_step * 1.1, clamp(efficiency * 1.05, 0.1, 0.9), storage_loss)

    col1, col2, col3 = st.columns(3)
    col1.metric("kWh / jour", f"{kwh_day:.3f}")
    col2.metric("kWh / mois (~30j)", f"{kwh_month:.1f}")
    col3.metric("kWh / an (~365j)", f"{kwh_year:.1f}")

    st.write("### Uncertainty scenarios (Wh/day)")
    scen_df = pd.DataFrame({
        "scenario": ["low", "mid", "high"],
        "Wh/day": [low, mid, high],
        "kWh/day": [wh_to_kwh(low), wh_to_kwh(mid), wh_to_kwh(high)],
    })
    st.dataframe(scen_df, use_container_width=True)

    # Equivalences
    st.write("### What can it power (pedagogical, local uses)")
    led_hours, phone_charges, sensor_days = equivalences(kwh_day)
    cA, cB, cC = st.columns(3)
    cA.metric("LED 10W (heures)", f"{led_hours:.0f}")
    cB.metric("Charges téléphone (~12Wh)", f"{phone_charges:.0f}")
    cC.metric("Capteur 1W (jours)", f"{sensor_days:.1f}")

    st.info(
        "Important: kinetic floors usually generate **modest energy**. The main value is often **engagement + pedagogy** "
        "(making energy visible), plus powering small local loads (LEDs, sensors, small displays)."
    )

    # Costs
    st.write("### Costs (CAPEX/OPEX) + cost per kWh (rough)")
    capex = equipped_area_ft2 * cost_per_ft2 + fixed_cost
    opex_year = (maint_pct / 100.0) * capex
    total_cost_over_N = capex + opex_year * amort_years

    # avoid divide by zero
    kwh_total_over_N = max(1e-9, kwh_year * amort_years)
    cost_per_kwh = total_cost_over_N / kwh_total_over_N

    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("CAPEX ($)", f"{capex:,.0f}")
    cc2.metric("OPEX / an ($)", f"{opex_year:,.0f}")
    cc3.metric("Coût approx ($/kWh)", f"{cost_per_kwh:,.2f}")

    verdict, reason = go_nogo(cost_per_kwh, kwh_year, pedagogical_weight=True)
    st.write("### Verdict")
    if verdict.startswith("GO"):
        st.success(f"**{verdict}** — {reason}")
    else:
        st.warning(f"**{verdict}** — {reason}")

    # If forecast exists, show forecasted energy
    if "forecast_df" in st.session_state:
        st.write("### Forecast-based energy (Sustainable AI)")
        df_fore = st.session_state["forecast_df"].copy()
        df_fore["kWh/day_pred"] = df_fore["forecast"] = df_fore["forecast"] if "forecast" in df_fore.columns else df_fore["visitors_pred"]
        # ensure correct column
        if "visitors_pred" in df_fore.columns:
            v = df_fore["visitors_pred"].to_numpy(dtype=float)
            wh_pred = [
                energy_wh_per_day(float(x), peak_mult, pct_zone, useful_steps, j_per_step, efficiency, storage_loss)
                for x in v
            ]
            df_fore["kWh/day_pred"] = np.array(wh_pred) / 1000.0
            st.dataframe(df_fore[["date", "visitors_pred", "kWh/day_pred"]], use_container_width=True)
            st.line_chart(df_fore.set_index("date")[["kWh/day_pred"]], use_container_width=True)

        if "last_emissions" in st.session_state:
            st.info(f"Last CodeCarbon run (kgCO₂e): **{st.session_state['last_emissions']:.6f}**")

with tabs[2]:
    st.subheader("Methodology / Limits (anti-greenwashing)")
    st.markdown(
        """
**Core formula (transparent):**  
Energy (Wh/day) = visitors/day × peak_multiplier × (%on_zone/100) × useful_steps × J_per_step × efficiency × (1 - storage_loss) ÷ 3600

**Why it’s “Sustainable AI”:**  
The forecast is lightweight (no large models). It helps avoid over-installation (materials, costs, maintenance) by sizing to realistic demand.

**Limits:**  
- Energy outputs are usually modest; the strongest benefit is often engagement/pedagogy + powering small local loads.  
- Costs vary by vendor, site constraints, and “showcase” projects. Treat cost outputs as ranges, not quotes.  
- No personal data: use aggregated visitor counts only.
        """
    )
