# app.py
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

# Optional (CodeCarbon). If not installed, the app still works.
try:
    from codecarbon import EmissionsTracker
except Exception:
    EmissionsTracker = None


# ----------------------------
# Helpers / business logic
# ----------------------------

@dataclass
class Inputs:
    place_type: str
    visitors_per_day: float
    peak_multiplier: float
    avg_presence_hours: float  # not directly used in energy formula but kept for context
    pct_on_zone: float
    useful_steps: float
    j_per_step: float
    efficiency: float
    storage_loss: float
    area_ft2: float
    tile_ft2: float
    cost_per_ft2: float
    fixed_cost: float
    maint_pct: float
    amort_years: int


def energy_wh_per_day(inp: Inputs) -> float:
    """
    Core formula (transparent):
    Energy (Wh/day) = visitors/day * peak_multiplier * (%on_zone/100) * useful_steps * J_per_step
                      * efficiency * (1 - storage_loss) / 3600
    """
    steps_captured = inp.visitors_per_day * inp.peak_multiplier * (inp.pct_on_zone / 100.0) * inp.useful_steps
    wh_day = steps_captured * inp.j_per_step * inp.efficiency * (1.0 - inp.storage_loss) / 3600.0
    return max(0.0, float(wh_day))


def scenarios_wh_per_day(mid_wh: float) -> pd.DataFrame:
    # Simple uncertainty: +/- 35% around mid.
    low = mid_wh * 0.65
    high = mid_wh * 1.95  # make "high" a bit wider to illustrate uncertainty
    # keep consistent ordering
    df = pd.DataFrame(
        {
            "scenario": ["low", "mid", "high"],
            "Wh/day": [low, mid_wh, high],
        }
    )
    df["kWh/day"] = df["Wh/day"] / 1000.0
    return df


def sizing_tiles(area_ft2: float, tile_ft2: float) -> int:
    if tile_ft2 <= 0:
        return 0
    return int(math.ceil(max(0.0, area_ft2) / tile_ft2))


def costs_summary(inp: Inputs, kwh_year: float) -> dict:
    tiles = sizing_tiles(inp.area_ft2, inp.tile_ft2)
    capex = inp.area_ft2 * inp.cost_per_ft2 + inp.fixed_cost
    opex_year = (inp.maint_pct / 100.0) * capex

    # "Rough" levelized cost per kWh over amortization period
    # Total cost over N years / total kWh over N years
    n = max(1, int(inp.amort_years))
    total_cost = capex + opex_year * n
    total_kwh = max(1e-9, kwh_year * n)  # avoid divide by zero
    cost_per_kwh = total_cost / total_kwh

    return {
        "tiles": tiles,
        "capex": capex,
        "opex_year": opex_year,
        "cost_per_kwh": cost_per_kwh,
    }


def equivalences(kwh_day: float) -> dict:
    """
    Pedagogical local uses.
    """
    wh_day = kwh_day * 1000.0

    # 10W LED hours
    led10_h = wh_day / 10.0 if wh_day > 0 else 0.0

    # Smartphone charge ~12 Wh
    phone_charges = wh_day / 12.0 if wh_day > 0 else 0.0

    # 1W sensor days
    sensor1_days = (wh_day / 1.0) / 24.0 if wh_day > 0 else 0.0

    return {
        "led10_h": led10_h,
        "phone_charges": phone_charges,
        "sensor1_days": sensor1_days,
    }


def verdict_text(kwh_year: float, cost_per_kwh: float) -> tuple[str, str]:
    """
    Keep it realistic: typically kinetic floors produce modest energy.
    We'll make "GO" only when the use-case is explicitly local/pedagogical and costs are not absurd.
    """
    # Heuristic thresholds (you can tweak):
    if kwh_year < 50:
        return ("NO-GO (energy ROI)", "Energy output is very small; consider it primarily an engagement / educational installation.")
    if cost_per_kwh > 5.0:  # $/kWh
        return ("NO-GO (cost)", "Energy is not cost-effective vs grid electricity; consider smaller scope or a pure pedagogy deployment.")
    return ("GO (local + pedagogy)", "Reasonable for small local loads (LEDs, sensors, small displays) + engagement value.")


def make_demo_visitors() -> pd.DataFrame:
    start = date(2025, 11, 1)
    days = 20
    rng = np.random.default_rng(42)
    base = 1200
    trend = 15
    data = []
    for i in range(days):
        d = start + timedelta(days=i)
        seasonal = 150 * math.sin(2 * math.pi * i / 7.0)  # weekly pattern
        noise = rng.normal(0, 60)
        visitors = max(0, base + trend * i + seasonal + noise)
        data.append((d.isoformat(), int(round(visitors))))
    return pd.DataFrame(data, columns=["date", "visitors"])


def forecast_visitors(df: pd.DataFrame, horizon_days: int = 10) -> pd.DataFrame:
    """
    Lightweight forecast WITHOUT sklearn:
    - convert date -> ordinal integer
    - simple linear fit with numpy.polyfit
    """
    dfx = df.copy()
    dfx["date"] = pd.to_datetime(dfx["date"], errors="coerce")
    dfx = dfx.dropna(subset=["date", "visitors"]).sort_values("date")
    if len(dfx) < 3:
        return pd.DataFrame(columns=["date", "visitors_pred"])

    x = dfx["date"].map(pd.Timestamp.toordinal).to_numpy(dtype=float)
    y = dfx["visitors"].to_numpy(dtype=float)

    # linear trend
    slope, intercept = np.polyfit(x, y, 1)

    last_date = dfx["date"].iloc[-1].date()
    future_dates = [last_date + timedelta(days=i + 1) for i in range(horizon_days)]
    xf = np.array([pd.Timestamp(d).toordinal() for d in future_dates], dtype=float)
    y_pred = slope * xf + intercept
    y_pred = np.clip(y_pred, 0, None)

    out = pd.DataFrame({"date": [d.isoformat() for d in future_dates], "visitors_pred": y_pred})
    return out


# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="Kinetic Impact Calculator", layout="wide")

st.title("Kinetic Impact Calculator (inspired by Coldplay)")
st.caption(
    "Decision-support MVP: estimate kinetic floor energy, practical uses, costs (CAPEX/OPEX), uncertainty scenarios, "
    "and a lightweight forecast module (Sustainable AI)."
)

# Sidebar: mode + quick guidance
st.sidebar.title("Settings")
ui_mode = st.sidebar.radio("Mode", ["Simple", "Expert"], index=0)
UI_EXPERT = (ui_mode == "Expert")

st.sidebar.markdown("---")
st.sidebar.markdown("**Capstone tip:** Your goal is to avoid **over-installation** (materials/maintenance) using scenarios + lightweight forecasting.")

tab_inputs, tab_results, tab_method = st.tabs(["Inputs", "Results", "Methodology / Limits"])

# Default values
default_place = "Musée"
default_visitors = 3300
default_peak = 1.0
default_presence = 2.5
default_pct_zone = 12.0
default_steps = 115.0
default_j = 3.0
default_eff = 0.5
default_loss = 0.1
default_area = 190.0
default_tile = 1.10
default_cost_ft2 = 175.0
default_fixed = 20000.0
default_maint = 8.0
default_amort = 9

# Keep state stable
if "inputs" not in st.session_state:
    st.session_state.inputs = {
        "place_type": default_place,
        "visitors_per_day": default_visitors,
        "peak_multiplier": default_peak,
        "avg_presence_hours": default_presence,
        "pct_on_zone": default_pct_zone,
        "useful_steps": default_steps,
        "j_per_step": default_j,
        "efficiency": default_eff,
        "storage_loss": default_loss,
        "area_ft2": default_area,
        "tile_ft2": default_tile,
        "cost_per_ft2": default_cost_ft2,
        "fixed_cost": default_fixed,
        "maint_pct": default_maint,
        "amort_years": default_amort,
    }


with tab_inputs:
    # Layout like your screenshot: 3 columns
    c1, c2, c3 = st.columns(3, gap="large")

    with c1:
        st.subheader("Context")

        place_type = st.selectbox(
            "Type de lieu",
            ["Musée", "Gare", "Stade", "Centre commercial"],
            index=["Musée", "Gare", "Stade", "Centre commercial"].index(st.session_state.inputs["place_type"]),
        )
        visitors_per_day = st.number_input(
            "Visiteurs / jour (moyenne)",
            min_value=0.0,
            value=float(st.session_state.inputs["visitors_per_day"]),
            step=50.0,
        )
        peak_multiplier = st.slider(
            "Multiplicateur pic (week-end / événement)",
            0.5, 5.0, float(st.session_state.inputs["peak_multiplier"]), 0.05
        )
        avg_presence_hours = st.slider(
            "Durée moyenne de présence (heures)",
            0.25, 12.0, float(st.session_state.inputs["avg_presence_hours"]), 0.25
        )

        st.subheader("Flow on equipped zone")
        pct_on_zone = st.slider(
            "% visiteurs passant sur la zone équipée",
            0.0, 100.0, float(st.session_state.inputs["pct_on_zone"]), 0.5
        )
        useful_steps = st.slider(
            "Pas utiles / visiteur sur zone",
            0.0, 300.0, float(st.session_state.inputs["useful_steps"]), 5.0
        )

        st.caption("Repères: 2–10% si zone petite; 10–30% si zone centrale. 20–60 pas (court), 80–200 (long).")

    with c2:
        st.subheader("Technical assumptions")

        if UI_EXPERT:
            j_per_step = st.slider("Énergie par pas (J)", 0.5, 10.0, float(st.session_state.inputs["j_per_step"]), 0.1)
            efficiency = st.slider("Rendement global", 0.0, 1.0, float(st.session_state.inputs["efficiency"]), 0.05)
            storage_loss = st.slider("Pertes stockage / conversion", 0.0, 0.8, float(st.session_state.inputs["storage_loss"]), 0.05)
        else:
            # fixed defaults in simple mode
            j_per_step = default_j
            efficiency = default_eff
            storage_loss = default_loss

            st.slider("Énergie par pas (J)", 0.5, 10.0, float(j_per_step), 0.1, disabled=True)
            st.slider("Rendement global", 0.0, 1.0, float(efficiency), 0.05, disabled=True)
            st.slider("Pertes stockage / conversion", 0.0, 0.8, float(storage_loss), 0.05, disabled=True)
            st.caption("Mode Simple: valeurs par défaut (3 J/pas, 50% rendement, 10% pertes).")

        st.subheader("Installation sizing (simple)")
        area_ft2 = st.number_input("Surface équipée (ft²)", min_value=0.0, value=float(st.session_state.inputs["area_ft2"]), step=5.0)
        tile_ft2 = st.number_input("Surface d’une dalle (ft²)", min_value=0.1, value=float(st.session_state.inputs["tile_ft2"]), step=0.05)
        tiles = sizing_tiles(area_ft2, tile_ft2)
        st.info(f"Estimation: ~ **{tiles} dalles** pour **{area_ft2:.0f} ft²** (si 1 dalle ≈ {tile_ft2:.2f} ft²).")

    with c3:
        st.subheader("Costs")
        cost_per_ft2 = st.slider("Coût installé ($/ft²)", 50.0, 900.0, float(st.session_state.inputs["cost_per_ft2"]), 5.0)
        fixed_cost = st.number_input("Coût fixe (travaux/élec/signalétique) $", min_value=0.0, value=float(st.session_state.inputs["fixed_cost"]), step=1000.0)
        maint_pct = st.slider("Maintenance annuelle (% du CAPEX)", 0.0, 20.0, float(st.session_state.inputs["maint_pct"]), 0.5)
        amort_years = st.slider("Amortissement (années)", 1, 25, int(st.session_state.inputs["amort_years"]), 1)

        st.subheader("Sustainable AI (lightweight)")
        use_demo = st.checkbox("Utiliser dataset démo", value=True)
        uploaded = st.file_uploader("Upload CSV (colonnes: date, visitors)", type=["csv"])

        # Build history dataframe
        if uploaded is not None:
            try:
                hist = pd.read_csv(uploaded)
            except Exception:
                hist = None
                st.error("Impossible de lire le CSV. Vérifie l'encodage/format.")
        elif use_demo:
            hist = make_demo_visitors()
        else:
            hist = None

        if hist is not None:
            st.dataframe(hist.head(10), use_container_width=True)
            horizon = st.slider("Horizon prévision (jours)", 7, 60, 10, 1)

            emissions_kg = None
            if EmissionsTracker is not None:
                # Track only the "forecast" part
                tracker = EmissionsTracker(project_name="kinetic-impact-forecast", log_level="error")
                try:
                    tracker.start()
                    pred = forecast_visitors(hist, horizon_days=horizon)
                    emissions_kg = tracker.stop()
                except Exception:
                    pred = forecast_visitors(hist, horizon_days=horizon)
                    try:
                        tracker.stop()
                    except Exception:
                        pass
            else:
                pred = forecast_visitors(hist, horizon_days=horizon)

            st.write("Prévision générée.")
            st.dataframe(pred, use_container_width=True)

            if emissions_kg is None:
                st.caption("CodeCarbon non activé (package non installé).")
            else:
                st.success(f"CodeCarbon (kgCO₂e): {emissions_kg:.6f}")

    # Save inputs in session
    st.session_state.inputs.update(
        {
            "place_type": place_type,
            "visitors_per_day": visitors_per_day,
            "peak_multiplier": peak_multiplier,
            "avg_presence_hours": avg_presence_hours,
            "pct_on_zone": pct_on_zone,
            "useful_steps": useful_steps,
            "j_per_step": j_per_step,
            "efficiency": efficiency,
            "storage_loss": storage_loss,
            "area_ft2": area_ft2,
            "tile_ft2": tile_ft2,
            "cost_per_ft2": cost_per_ft2,
            "fixed_cost": fixed_cost,
            "maint_pct": maint_pct,
            "amort_years": amort_years,
        }
    )


# Build Inputs object from session
inp = Inputs(
    place_type=st.session_state.inputs["place_type"],
    visitors_per_day=float(st.session_state.inputs["visitors_per_day"]),
    peak_multiplier=float(st.session_state.inputs["peak_multiplier"]),
    avg_presence_hours=float(st.session_state.inputs["avg_presence_hours"]),
    pct_on_zone=float(st.session_state.inputs["pct_on_zone"]),
    useful_steps=float(st.session_state.inputs["useful_steps"]),
    j_per_step=float(st.session_state.inputs["j_per_step"]),
    efficiency=float(st.session_state.inputs["efficiency"]),
    storage_loss=float(st.session_state.inputs["storage_loss"]),
    area_ft2=float(st.session_state.inputs["area_ft2"]),
    tile_ft2=float(st.session_state.inputs["tile_ft2"]),
    cost_per_ft2=float(st.session_state.inputs["cost_per_ft2"]),
    fixed_cost=float(st.session_state.inputs["fixed_cost"]),
    maint_pct=float(st.session_state.inputs["maint_pct"]),
    amort_years=int(st.session_state.inputs["amort_years"]),
)

mid_wh = energy_wh_per_day(inp)
mid_kwh_day = mid_wh / 1000.0
kwh_month = mid_kwh_day * 30.0
kwh_year = mid_kwh_day * 365.0

sc_df = scenarios_wh_per_day(mid_wh)
costs = costs_summary(inp, kwh_year=kwh_year)
eq = equivalences(mid_kwh_day)
verdict_title, verdict_msg = verdict_text(kwh_year=kwh_year, cost_per_kwh=costs["cost_per_kwh"])

with tab_results:
    st.header("Results (energy + uncertainty + costs + Go/No-Go)")

    m1, m2, m3 = st.columns(3)
    m1.metric("kWh / jour", f"{mid_kwh_day:.3f}")
    m2.metric("kWh / mois (~30j)", f"{kwh_month:.1f}")
    m3.metric("kWh / an (~365j)", f"{kwh_year:.1f}")

    st.subheader("Uncertainty scenarios (Wh/day)")
    st.dataframe(sc_df, use_container_width=True)

    st.subheader("What can it power (pedagogical, local uses)")
    e1, e2, e3 = st.columns(3)
    e1.metric("LED 10W (heures)", f"{eq['led10_h']:.1f}")
    e2.metric("Charges téléphone (~12Wh)", f"{eq['phone_charges']:.1f}")
    e3.metric("Capteur 1W (jours)", f"{eq['sensor1_days']:.1f}")

    st.info(
        "Important: kinetic floors usually generate **modest** energy. "
        "The main value is often **engagement + pedagogy (making energy visible)**, "
        "plus powering small local loads (LEDs, sensors, small displays)."
    )

    st.subheader("Costs (CAPEX/OPEX) + cost per kWh (rough)")
    c1, c2, c3 = st.columns(3)
    c1.metric("CAPEX ($)", f"{costs['capex']:,.0f}")
    c2.metric("OPEX / an ($)", f"{costs['opex_year']:,.0f}")
    c3.metric("Coût approx ($/kWh)", f"{costs['cost_per_kwh']:,.2f}")

    st.subheader("Verdict")
    if verdict_title.startswith("GO"):
        st.success(f"**{verdict_title}** — {verdict_msg}")
    else:
        st.warning(f"**{verdict_title}** — {verdict_msg}")

    # Export
    st.subheader("Export (inputs + outputs)")
    export = {
        **st.session_state.inputs,
        "mid_Wh_day": mid_wh,
        "mid_kWh_day": mid_kwh_day,
        "kWh_month": kwh_month,
        "kWh_year": kwh_year,
        "tiles_estimate": costs["tiles"],
        "capex": costs["capex"],
        "opex_year": costs["opex_year"],
        "cost_per_kwh": costs["cost_per_kwh"],
        "verdict": verdict_title,
    }
    export_df = pd.DataFrame([export])
    st.download_button(
        "Download CSV report",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name="kinetic_impact_report.csv",
        mime="text/csv",
    )

with tab_method:
    st.header("Methodology / Limits (anti-greenwashing)")

    st.subheader("Core formula (transparent)")
    st.code(
        "Energy (Wh/day) = visitors/day × peak_multiplier × (%on_zone/100) × useful_steps × J_per_step × "
        "efficiency × (1 - storage_loss) ÷ 3600",
        language="text",
    )

    st.subheader('Why it’s “Sustainable AI”')
    st.write(
        "The forecast is lightweight (no large models). It helps avoid **over-installation** (materials, costs, maintenance) "
        "by sizing to realistic demand. If CodeCarbon is installed, we also measure the footprint of the forecasting step."
    )

    st.subheader("Limits (what you should say in your capstone)")
    st.markdown(
        """
- Energy outputs are usually modest; the strongest benefit is often **engagement/pedagogy** + powering small local loads.
- Costs vary by vendor and site constraints; treat outputs as **ranges**, not quotes.
- No personal data: this tool uses **aggregate visitor counts** only.
"""
    )
