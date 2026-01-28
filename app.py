import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Optional / safe imports (won't crash the app if unavailable)
SKLEARN_AVAILABLE = True
try:
    from sklearn.linear_model import LinearRegression
except Exception:
    SKLEARN_AVAILABLE = False

CODECARBON_AVAILABLE = True
try:
    from codecarbon import EmissionsTracker
except Exception:
    CODECARBON_AVAILABLE = False


# -----------------------------
# Helpers
# -----------------------------
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
DEMO_CSV = DATA_DIR / "sample_visitors.csv"


def wh_per_day(visitors_per_day: float,
               peak_multiplier: float,
               pct_pass: float,
               useful_steps: float,
               joules_per_step: float,
               efficiency: float,
               storage_losses: float) -> float:
    """
    Energy (Wh/day) = visitors/day * peak * (pct_pass) * (steps) * J/step * efficiency * (1-losses) / 3600
    """
    pct = pct_pass / 100.0
    losses = 1.0 - (storage_losses / 100.0)
    wh = visitors_per_day * peak_multiplier * pct * useful_steps * joules_per_step * efficiency * losses / 3600.0
    return max(0.0, wh)


def kwh(wh: float) -> float:
    return wh / 1000.0


def fmt(n: float, digits: int = 2) -> str:
    return f"{n:,.{digits}f}".replace(",", " ").replace(".", ",")


def scenario_triplet(base: float, low_pct: float = 0.8, high_pct: float = 1.2):
    return base * low_pct, base, base * high_pct


def go_no_go(cost_per_kwh: float, kwh_per_day: float) -> tuple[str, str]:
    """
    Simple rule:
    - If kWh/day is tiny (<0.5) -> usually only pedagogical value
    - If cost/kWh is extremely high -> No-Go for energy purpose
    """
    if kwh_per_day < 0.5:
        return "NO-GO (energy)", "Energy is modest. This makes sense mainly for pedagogy/engagement (LEDs, display, sensors)."
    if cost_per_kwh > 5.0:  # $5/kWh is already very high vs grid
        return "NO-GO (economics)", "Cost per kWh is very high. Consider reducing surface or targeting purely pedagogical use."
    return "GO (for local use)", "Economically reasonable for small local loads + strong engagement value."


def equivalents(kwh_per_day: float) -> dict:
    """
    Intuitive equivalents for pedagogy.
    Assumptions:
    - LED bulb: 10W used 5h/day => 0.05 kWh/day
    - Sensor node: 1W 24h => 0.024 kWh/day
    - Small display: 30W 8h/day => 0.24 kWh/day
    """
    led_kwh = 0.05
    sensor_kwh = 0.024
    display_kwh = 0.24

    return {
        "10W LED bulb (5h/day)": int(kwh_per_day / led_kwh) if led_kwh > 0 else 0,
        "1W sensor node (24h/day)": int(kwh_per_day / sensor_kwh) if sensor_kwh > 0 else 0,
        "30W small display (8h/day)": int(kwh_per_day / display_kwh) if display_kwh > 0 else 0,
    }


def load_demo_csv() -> pd.DataFrame:
    if DEMO_CSV.exists():
        df = pd.read_csv(DEMO_CSV)
        return df
    # fallback demo if file missing
    dates = pd.date_range(datetime.today().date() - timedelta(days=20), periods=10, freq="D")
    visitors = [1200, 1500, 900, 950, 980, 1100, 1300, 1600, 1800, 1000]
    return pd.DataFrame({"date": dates.astype(str), "visitors": visitors})


def prepare_visitors_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # normalize columns
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" not in df.columns or "visitors" not in df.columns:
        raise ValueError("CSV must contain columns: date, visitors")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["visitors"] = pd.to_numeric(df["visitors"], errors="coerce")
    df = df.dropna(subset=["visitors"])
    df = df.sort_values("date")
    return df


def forecast_visitors(df: pd.DataFrame, horizon_days: int = 10) -> pd.DataFrame:
    """
    Lightweight forecasting:
    - If sklearn available: LinearRegression on day index
    - Else: simple mean baseline
    """
    df = df.copy()
    df["t"] = (df["date"] - df["date"].min()).dt.days.astype(int)

    future_dates = pd.date_range(df["date"].max() + timedelta(days=1), periods=horizon_days, freq="D")
    future_t = np.array([(d - df["date"].min()).days for d in future_dates]).reshape(-1, 1)

    if SKLEARN_AVAILABLE and len(df) >= 3:
        X = df[["t"]].values
        y = df["visitors"].values
        model = LinearRegression()
        model.fit(X, y)
        preds = model.predict(future_t)
    else:
        preds = np.repeat(df["visitors"].mean(), horizon_days)

    preds = np.maximum(0, preds)
    out = pd.DataFrame({"date": future_dates, "visitors_pred": preds})
    return out


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Kinetic Impact Calculator", layout="wide")

st.title("Kinetic Impact Calculator (inspired by Coldplay)")
st.caption(
    "Decision-support MVP: estimate kinetic floor energy, realistic local uses, costs (CAPEX/OPEX), scenarios, "
    "and a lightweight visitor forecast + optional CodeCarbon."
)

tabs = st.tabs(["Inputs", "Results", "Methodology"])

# -----------------------------
# Inputs Tab
# -----------------------------
with tabs[0]:
    colA, colB, colC = st.columns([1, 1, 1], gap="large")

    with colA:
        st.header("Context")
        place_type = st.selectbox("Type de lieu", ["Musée", "Gare", "Stade", "Centre commercial", "Autre"])

        visitors_day = st.number_input("Visiteurs / jour (moyenne)", min_value=0.0, value=2050.0, step=50.0)
        peak_mult = st.slider("Multiplicateur pic (week-end / événement)", 1.0, 5.0, 1.0, 0.05)
        avg_duration_h = st.slider("Durée moyenne de présence (heures)", 0.5, 6.0, 2.5, 0.1)

        st.subheader("Flow on equipped zone")
        pct_pass = st.slider("% visiteurs passant sur la zone équipée", 0.0, 30.0, 5.0, 0.5)
        useful_steps = st.slider("Pas utiles / visiteur sur zone", 0.0, 300.0, 80.0, 1.0)

    with colB:
        st.header("Technical assumptions")
        joules_per_step = st.slider("Énergie par pas (J)", 1.0, 6.0, 3.0, 0.1)
        efficiency = st.slider("Rendement global", 0.10, 0.80, 0.50, 0.01)
        storage_losses = st.slider("Pertes stockage / conversion (%)", 0.0, 40.0, 10.0, 0.5)

        st.subheader("Installation sizing (simple)")
        surface_ft2 = st.number_input("Surface équipée (ft²)", min_value=1.0, value=190.0, step=10.0)
        tile_ft2 = st.number_input("Surface d’une dalle (ft²)", min_value=0.2, value=1.10, step=0.05)
        tiles_count = int(np.ceil(surface_ft2 / tile_ft2))
        st.info(f"≈ Nombre de dalles estimé : **{tiles_count}**")

    with colC:
        st.header("Costs")
        cost_per_ft2 = st.slider("Coût installé ($/ft²)", 50.0, 900.0, 175.0, 5.0)
        fixed_cost = st.number_input("Coût fixe (travaux/élec/signalétique) $", min_value=0.0, value=10000.0, step=1000.0)
        maint_pct = st.slider("Maintenance annuelle (% du CAPEX)", 0.0, 15.0, 8.0, 0.5)
        amort_years = st.slider("Amortissement (années)", 1, 20, 9, 1)

        st.markdown("---")
        st.success("✅ Astuce capstone : objectif = éviter la sur-installation (matériaux/maintenance) via scénarios + prévision IA.")

    # IA block input
    st.markdown("---")
    st.subheader("Data (bonus) + IA (prévision) + CodeCarbon")

    use_demo = st.checkbox("Utiliser dataset démo", value=True)
    uploaded = st.file_uploader("Upload CSV (colonnes: date, visitors)", type=["csv"])

    df_hist = None
    try:
        if uploaded is not None:
            df_hist = pd.read_csv(uploaded)
        elif use_demo:
            df_hist = load_demo_csv()

        if df_hist is not None:
            df_hist = prepare_visitors_df(df_hist)
            st.dataframe(df_hist.tail(10), use_container_width=True)
    except Exception as e:
        st.error(f"CSV invalide : {e}")
        df_hist = None

    forecast_horizon = st.slider("Horizon prévision (jours)", 5, 60, 10, 1)
    run_ai = st.button("Lancer prévision (IA légère) + mesurer empreinte (CodeCarbon)")

    if run_ai:
        if df_hist is None:
            st.warning("Ajoute un CSV valide ou coche le dataset démo.")
        else:
            emissions = None
            tracker = None
            if CODECARBON_AVAILABLE:
                try:
                    tracker = EmissionsTracker(project_name="kinetic-impact-calculator", log_level="error")
                    tracker.start()
                except Exception:
                    tracker = None

            fc = forecast_visitors(df_hist, horizon_days=forecast_horizon)

            if tracker is not None:
                try:
                    emissions = tracker.stop()
                except Exception:
                    emissions = None

            st.success("Prévision générée.")
            st.dataframe(fc, use_container_width=True)

            if emissions is not None:
                st.info(f"CodeCarbon (kgCO₂e) : {emissions:.6f}")
            else:
                st.caption("CodeCarbon non disponible / ignoré sur cet environnement (l’app continue quand même).")


# -----------------------------
# Results Tab
# -----------------------------
with tabs[1]:
    st.header("Results (decision-support)")

    # Base energy
    wh_day_base = wh_per_day(
        visitors_day, peak_mult, pct_pass, useful_steps,
        joules_per_step, efficiency, storage_losses
    )
    kwh_day_base = kwh(wh_day_base)
    kwh_month = kwh_day_base * 30
    kwh_year = kwh_day_base * 365

    # Scenarios
    kwh_day_low, kwh_day_mid, kwh_day_high = scenario_triplet(kwh_day_base, 0.7, 1.3)

    # Costs
    capex = surface_ft2 * cost_per_ft2 + fixed_cost
    opex_year = (maint_pct / 100.0) * capex
    total_cost_over_n = capex + opex_year * amort_years

    # Cost per kWh (over amort period)
    total_kwh_over_n = (kwh_year * amort_years) if kwh_year > 0 else np.nan
    cost_per_kwh_val = (total_cost_over_n / total_kwh_over_n) if total_kwh_over_n and not np.isnan(total_kwh_over_n) else np.nan

    # Verdict
    verdict, reason = go_no_go(cost_per_kwh_val if not np.isnan(cost_per_kwh_val) else 9999.0, kwh_day_mid)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Energy")
        st.metric("kWh / day", f"{fmt(kwh_day_mid, 3)}")
        st.metric("kWh / month", f"{fmt(kwh_month, 2)}")
        st.metric("kWh / year", f"{fmt(kwh_year, 2)}")

    with col2:
        st.subheader("Scenarios (uncertainty)")
        st.write(f"Low: **{fmt(kwh_day_low,3)} kWh/day**")
        st.write(f"Mid: **{fmt(kwh_day_mid,3)} kWh/day**")
        st.write(f"High: **{fmt(kwh_day_high,3)} kWh/day**")

        # Simple bar chart
        chart_df = pd.DataFrame({
            "scenario": ["low", "mid", "high"],
            "kWh/day": [kwh_day_low, kwh_day_mid, kwh_day_high]
        })
        st.bar_chart(chart_df.set_index("scenario"))

    with col3:
        st.subheader("Costs")
        st.metric("CAPEX ($)", f"{fmt(capex, 0)}")
        st.metric("OPEX / year ($)", f"{fmt(opex_year, 0)}")
        if not np.isnan(cost_per_kwh_val):
            st.metric("Cost per kWh ($/kWh)", f"{fmt(cost_per_kwh_val, 2)}")
        else:
            st.metric("Cost per kWh ($/kWh)", "N/A")

    st.markdown("---")
    st.subheader("What can it power? (realistic local uses)")
    eq = equivalents(kwh_day_mid)
    st.write("These equivalents are meant to be *pedagogical* (local loads), not “power a building”.")
    st.json(eq)

    st.markdown("---")
    st.subheader("Go / No-Go")
    if "NO-GO" in verdict:
        st.error(f"**{verdict}** — {reason}")
    else:
        st.success(f"**{verdict}** — {reason}")

    st.markdown("---")
    st.subheader("Export")
    export = {
        "place_type": place_type,
        "visitors_day": visitors_day,
        "peak_multiplier": peak_mult,
        "pct_pass": pct_pass,
        "useful_steps": useful_steps,
        "joules_per_step": joules_per_step,
        "efficiency": efficiency,
        "storage_losses_pct": storage_losses,
        "surface_ft2": surface_ft2,
        "tile_ft2": tile_ft2,
        "tiles_estimated": tiles_count,
        "capex_usd": capex,
        "opex_year_usd": opex_year,
        "amort_years": amort_years,
        "kwh_day_mid": kwh_day_mid,
        "kwh_year": kwh_year,
        "cost_per_kwh": None if np.isnan(cost_per_kwh_val) else float(cost_per_kwh_val),
        "verdict": verdict,
    }
    export_df = pd.DataFrame([export])
    st.download_button(
        "Download results (CSV)",
        export_df.to_csv(index=False).encode("utf-8"),
        file_name="kinetic_impact_results.csv",
        mime="text/csv"
    )


# -----------------------------
# Methodology Tab
# -----------------------------
with tabs[2]:
    st.header("Methodology & limits (anti-greenwashing)")

    st.markdown("""
**Core formula (transparent):**  
Energy (Wh/day) = visitors/day × peak × (%pass) × (useful steps) × (J per step) × (efficiency) × (1 − losses) ÷ 3600

**Key message:** kinetic floors usually generate **modest energy**, best suited for:
- LEDs, sensors, small educational displays
- engagement / “making energy tangible” (like Coldplay)

**Why the lightweight AI is “Sustainable AI”:**
- it helps avoid over-installation (materials + maintenance)
- simple regression / baseline instead of heavy models
- optional CodeCarbon to measure compute footprint
""")

    st.caption("Tip: In your pitch, emphasize decision-support + avoiding material waste, not “free energy”.")
