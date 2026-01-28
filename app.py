import io
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

# Optional: CodeCarbon can fail on some cloud environments; we handle gracefully.
try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except Exception:
    CODECARBON_AVAILABLE = False


# ----------------------------
# App config
# ----------------------------
st.set_page_config(
    page_title="Kinetic Impact Calculator (inspired by Coldplay)",
    layout="wide",
)

st.title("Kinetic Impact Calculator (inspired by Coldplay)")
st.caption(
    "Decision-support MVP: estimate kinetic floor energy, practical uses, costs (CAPEX/OPEX), "
    "uncertainty scenarios, and a lightweight forecast module (Sustainable AI)."
)

tabs = st.tabs(["Inputs", "Results", "Methodology / Limits"])


# ----------------------------
# Helpers
# ----------------------------
@dataclass
class Inputs:
    place_type: str
    visitors_per_day: float
    peak_multiplier: float
    avg_dwell_hours: float

    pct_on_zone: float
    useful_steps_per_visitor: float

    j_per_step: float
    efficiency: float
    storage_loss: float

    equipped_area_ft2: float
    tile_area_ft2: float

    installed_cost_per_ft2: float
    fixed_cost: float
    maintenance_pct_capex: float
    amort_years: int


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def energy_wh_per_day(visitors_day, peak_mult, pct_on_zone, useful_steps, j_per_step, eff, storage_loss):
    # visitors_day * peak_mult => effective visitors
    # pct_on_zone is percent (0-100)
    # Wh/day = visitors * pct * steps * J/step * eff * (1-loss) / 3600
    return (
        visitors_day
        * peak_mult
        * (pct_on_zone / 100.0)
        * useful_steps
        * j_per_step
        * eff
        * (1.0 - storage_loss)
        / 3600.0
    )


def kwh_from_wh(wh):
    return wh / 1000.0


def capex_total(area_ft2, installed_cost_per_ft2, fixed_cost):
    return area_ft2 * installed_cost_per_ft2 + fixed_cost


def opex_year(capex, maintenance_pct):
    return capex * (maintenance_pct / 100.0)


def cost_per_kwh(capex, opex_y, amort_years, kwh_year):
    # Rough: total cost over N years / total energy over N years
    if kwh_year <= 0:
        return np.inf
    total_cost = capex + opex_y * amort_years
    total_kwh = kwh_year * amort_years
    return total_cost / total_kwh


def sizing_bucket(area_ft2):
    # Simple “small/medium/large” heuristic (adjust as you like)
    if area_ft2 < 50:
        return "Small (pilot / educational corner)"
    if area_ft2 < 200:
        return "Medium (visible demo zone)"
    return "Large (high-traffic showcase)"


def uses_equivalences(kwh_day):
    # Very simple local uses (educational)
    # 10W LED: hours = (kWh*1000 Wh) / 10W
    wh_day = kwh_day * 1000.0
    led_10w_hours = wh_day / 10.0

    # Phone charge ~12Wh
    phone_charges = wh_day / 12.0

    # 1W sensor: days = Wh / (1W*24h)
    sensor_1w_days = wh_day / 24.0

    return led_10w_hours, phone_charges, sensor_1w_days


def make_demo_visitors():
    dates = pd.date_range("2025-11-01", periods=30, freq="D")
    base = 1200 + 300*np.sin(np.linspace(0, 3*np.pi, len(dates)))
    noise = np.random.RandomState(42).normal(0, 80, len(dates))
    visitors = np.maximum(200, base + noise).round().astype(int)
    return pd.DataFrame({"date": dates.date.astype(str), "visitors": visitors})


def lightweight_forecast(df, horizon_days=10):
    """
    Lightweight forecast:
    - rolling mean of last 7 days
    - plus a simple weekday adjustment if enough data
    No sklearn. No heavy dependencies.
    """
    df2 = df.copy()
    df2["date"] = pd.to_datetime(df2["date"], errors="coerce")
    df2 = df2.dropna(subset=["date"])
    df2 = df2.sort_values("date")
    df2["visitors"] = pd.to_numeric(df2["visitors"], errors="coerce")
    df2 = df2.dropna(subset=["visitors"])
    if df2.empty:
        return pd.DataFrame()

    df2["dow"] = df2["date"].dt.dayofweek
    last_date = df2["date"].max()

    # baseline = rolling mean
    baseline = df2["visitors"].tail(7).mean()

    # weekday factor (if enough data)
    dow_means = df2.groupby("dow")["visitors"].mean()
    overall_mean = df2["visitors"].mean()
    dow_factor = (dow_means / overall_mean) if overall_mean > 0 else None

    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon_days, freq="D")
    preds = []
    for d in future_dates:
        pred = baseline
        if dow_factor is not None and len(dow_factor) >= 3:
            pred *= float(dow_factor.get(d.dayofweek, 1.0))
        preds.append(pred)

    out = pd.DataFrame({"date": future_dates.date.astype(str), "visitors_pred": np.round(preds, 1)})
    return out


# ----------------------------
# Inputs tab
# ----------------------------
with tabs[0]:
    colA, colB, colC = st.columns(3)

    with colA:
        st.subheader("Context")
        place_type = st.selectbox("Place type", ["Museum", "Train station", "Stadium", "Mall", "Other"])
        visitors_per_day = st.number_input("Visitors / day (average)", min_value=0.0, value=3300.0, step=50.0)
        peak_multiplier = st.slider("Peak multiplier (weekend / event)", 1.0, 5.0, 1.0, 0.1)
        avg_dwell_hours = st.slider("Average dwell time (hours)", 0.5, 6.0, 2.5, 0.1)

        st.subheader("Flow on equipped zone")
        pct_on_zone = st.slider("% visitors passing on the equipped zone", 0.0, 50.0, 12.0, 0.5)
        useful_steps_per_visitor = st.slider("Useful steps / visitor on zone", 0.0, 300.0, 115.0, 1.0)

    with colB:
        st.subheader("Technical assumptions")
        j_per_step = st.slider("Energy per step (J)", 1.0, 5.0, 3.0, 0.1)
        efficiency = st.slider("Overall efficiency", 0.1, 0.9, 0.5, 0.05)
        storage_loss = st.slider("Storage / conversion losses", 0.0, 0.5, 0.1, 0.01)

        st.subheader("Installation sizing (simple)")
        equipped_area_ft2 = st.number_input("Equipped area (ft²)", min_value=1.0, value=190.0, step=5.0)
        tile_area_ft2 = st.number_input("Area per tile (ft²)", min_value=0.1, value=1.10, step=0.05)
        n_tiles = int(round(equipped_area_ft2 / tile_area_ft2))
        st.info(f"Estimation: ~ **{n_tiles} tiles** for {equipped_area_ft2:.0f} ft² (1 tile ≈ {tile_area_ft2:.2f} ft²).")
        st.caption(f"Suggested size: **{sizing_bucket(equipped_area_ft2)}**")

    with colC:
        st.subheader("Costs")
        installed_cost_per_ft2 = st.slider("Installed cost ($/ft²)", 50.0, 900.0, 175.0, 5.0)
        fixed_cost = st.number_input("Fixed cost (work/electrical/signage) $", min_value=0.0, value=20000.0, step=1000.0)
        maintenance_pct_capex = st.slider("Annual maintenance (% of CAPEX)", 0.0, 15.0, 8.0, 0.5)
        amort_years = st.slider("Amortization (years)", 1, 15, 9, 1)

        st.subheader("Sustainable AI (lightweight)")
        use_demo = st.checkbox("Use demo dataset", value=True)
        uploaded = st.file_uploader("Upload CSV (columns: date, visitors)", type=["csv"])

        df_vis = None
        if use_demo:
            df_vis = make_demo_visitors()
        elif uploaded is not None:
            try:
                df_vis = pd.read_csv(uploaded)
            except Exception:
                st.error("Could not read the CSV. Make sure it has columns: date, visitors.")
                df_vis = None

        if df_vis is not None:
            st.dataframe(df_vis.head(10), use_container_width=True)

        run_forecast = st.button("Run lightweight forecast + measure compute footprint (CodeCarbon)")

    inputs = Inputs(
        place_type=place_type,
        visitors_per_day=visitors_per_day,
        peak_multiplier=peak_multiplier,
        avg_dwell_hours=avg_dwell_hours,
        pct_on_zone=pct_on_zone,
        useful_steps_per_visitor=useful_steps_per_visitor,
        j_per_step=j_per_step,
        efficiency=efficiency,
        storage_loss=storage_loss,
        equipped_area_ft2=equipped_area_ft2,
        tile_area_ft2=tile_area_ft2,
        installed_cost_per_ft2=installed_cost_per_ft2,
        fixed_cost=fixed_cost,
        maintenance_pct_capex=maintenance_pct_capex,
        amort_years=amort_years,
    )

    # Forecast section (runs only when clicked)
    if run_forecast:
        if df_vis is None:
            st.warning("Please upload a CSV (date, visitors) or enable the demo dataset.")
        else:
            tracker = None
            emissions = None

            if CODECARBON_AVAILABLE:
                try:
                    tracker = EmissionsTracker(project_name="kinetic-impact-forecast", log_level="error")
                    tracker.start()
                except Exception:
                    tracker = None

            fc = lightweight_forecast(df_vis, horizon_days=10)

            if tracker is not None:
                try:
                    emissions = tracker.stop()
                except Exception:
                    emissions = None

            if fc.empty:
                st.error("Forecast could not be generated (check your CSV).")
            else:
                st.success("Forecast generated.")
                st.dataframe(fc, use_container_width=True)

                if emissions is not None:
                    st.info(f"CodeCarbon estimate (kgCO₂e): **{emissions:.6f}**")
                else:
                    st.caption("CodeCarbon footprint measurement not available on this environment (optional).")


# ----------------------------
# Results tab
# ----------------------------
with tabs[1]:
    st.subheader("Results (energy + uncertainty + costs + Go/No-Go)")

    # Mid estimate (base)
    wh_day_mid = energy_wh_per_day(
        inputs.visitors_per_day,
        inputs.peak_multiplier,
        inputs.pct_on_zone,
        inputs.useful_steps_per_visitor,
        inputs.j_per_step,
        inputs.efficiency,
        inputs.storage_loss,
    )

    # Scenarios: low / mid / high multipliers
    wh_day_low = wh_day_mid * 0.4
    wh_day_high = wh_day_mid * 1.95

    kwh_day_mid = kwh_from_wh(wh_day_mid)
    kwh_month_mid = kwh_day_mid * 30.0
    kwh_year_mid = kwh_day_mid * 365.0

    m1, m2, m3 = st.columns(3)
    m1.metric("kWh / day", f"{kwh_day_mid:.3f}")
    m2.metric("kWh / month (~30d)", f"{kwh_month_mid:.1f}")
    m3.metric("kWh / year (~365d)", f"{kwh_year_mid:.1f}")

    st.markdown("### Uncertainty scenarios (Wh/day)")
    scen = pd.DataFrame(
        {
            "scenario": ["low", "mid", "high"],
            "Wh/day": [wh_day_low, wh_day_mid, wh_day_high],
        }
    )
    scen["kWh/day"] = scen["Wh/day"] / 1000.0
    st.dataframe(scen, use_container_width=True)

    st.markdown("### What can it power (pedagogical, local uses)")
    led_hours, phone_charges, sensor_days = uses_equivalences(kwh_day_mid)
    c1, c2, c3 = st.columns(3)
    c1.metric("10W LED (hours)", f"{led_hours:.1f}")
    c2.metric("Phone charges (~12Wh)", f"{phone_charges:.1f}")
    c3.metric("1W sensor (days)", f"{sensor_days:.2f}")

    st.info(
        "Important: kinetic floors usually generate modest energy. The main value is often **engagement + pedagogy** "
        "(making energy visible), plus powering small local loads (LEDs, sensors, small displays)."
    )

    st.markdown("### Costs (CAPEX/OPEX) + cost per kWh (rough)")
    capex = capex_total(inputs.equipped_area_ft2, inputs.installed_cost_per_ft2, inputs.fixed_cost)
    opex = opex_year(capex, inputs.maintenance_pct_capex)
    cpk = cost_per_kwh(capex, opex, inputs.amort_years, kwh_year_mid)

    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("CAPEX ($)", f"{capex:,.0f}")
    cc2.metric("OPEX / year ($)", f"{opex:,.0f}")
    cc3.metric("Approx cost ($/kWh)", "∞" if np.isinf(cpk) else f"{cpk:,.2f}")

    st.markdown("### Verdict")
    # Simple go/no-go rule of thumb
    # If annual energy is extremely low AND cost/kWh is huge -> No-Go (energy ROI)
    if kwh_year_mid < 50 or (not np.isinf(cpk) and cpk > 2.0):
        st.warning(
            "**NO-GO (energy ROI)** — Energy output is very small; consider it primarily an engagement / educational installation."
        )
    else:
        st.success(
            "**GO (for local uses + visibility)** — Output can support small local loads; still treat as educational + local energy."
        )


# ----------------------------
# Methodology / Limits tab
# ----------------------------
with tabs[2]:
    st.header("Methodology / Limits (anti-greenwashing)")

    st.markdown("### Core formula (transparent)")
    st.code(
        "Energy (Wh/day) = visitors/day × peak_multiplier × (%on_zone/100) × useful_steps × J_per_step × efficiency × (1 - storage_loss) ÷ 3600",
        language="text",
    )

    st.markdown("### Why this is “Sustainable AI”")
    st.write(
        "The forecast is lightweight (no large models). It helps avoid over-installation (materials, costs, maintenance) "
        "by sizing to realistic demand."
    )

    st.markdown("### Limits")
    st.markdown(
        "- Energy outputs are usually modest; the strongest benefit is often engagement/pedagogy + powering small local loads.\n"
        "- Costs vary by vendor, site constraints, and showcase projects. Treat cost outputs as ranges, not quotes.\n"
        "- No personal data: use aggregated visitor counts only.\n"
        "- This MVP is a decision-support tool; real projects require vendor specs + pilot measurements."
    )
