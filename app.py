import io
import math
import time
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# CodeCarbon is optional at runtime (we'll try to import)
try:
    from codecarbon import EmissionsTracker
    CODECARBON_OK = True
except Exception:
    CODECARBON_OK = False


# -----------------------------
# App config + style
# -----------------------------
st.set_page_config(
    page_title="Kinetic Impact Calculator (inspired by Coldplay)",
    layout="wide",
)

st.markdown(
    """
    <style>
      .small-muted { color: #6b7280; font-size: 0.9rem; }
      .badge { display: inline-block; padding: 0.18rem 0.55rem; border-radius: 999px; font-size: 0.85rem; font-weight: 600; }
      .badge-ok { background: #e7f7ee; color: #166534; border: 1px solid #86efac; }
      .badge-warn { background: #fff7ed; color: #9a3412; border: 1px solid #fdba74; }
      .badge-bad { background: #fef2f2; color: #991b1b; border: 1px solid #fca5a5; }
      .card { padding: 1rem; border-radius: 14px; border: 1px solid #e5e7eb; background: white; }
      .hr { border-top: 1px solid #e5e7eb; margin: 0.75rem 0; }
      .big { font-size: 2.0rem; font-weight: 800; }
      .kpi { font-size: 1.35rem; font-weight: 800; }
      .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Kinetic Impact Calculator (inspired by Coldplay)")
st.caption(
    "Decision-support MVP: estimate kinetic floor energy, realistic uses, costs (CAPEX/OPEX), uncertainty scenarios, and a lightweight forecast module (Sustainable AI)."
)

tabs = st.tabs(["Inputs", "Results", "Methodology / Limits"])


# -----------------------------
# Presets / defaults
# -----------------------------
@dataclass
class VenuePreset:
    visitors_day: int
    peak_multiplier: float
    dwell_hours: float
    pct_on_zone: float
    useful_steps: int
    area_ft2: float


VENUE_PRESETS: Dict[str, VenuePreset] = {
    "Museum": VenuePreset(visitors_day=1800, peak_multiplier=1.2, dwell_hours=2.0, pct_on_zone=8.0, useful_steps=60, area_ft2=120),
    "Train station": VenuePreset(visitors_day=25000, peak_multiplier=1.4, dwell_hours=0.4, pct_on_zone=12.0, useful_steps=80, area_ft2=250),
    "Stadium (event)": VenuePreset(visitors_day=45000, peak_multiplier=1.0, dwell_hours=3.0, pct_on_zone=6.0, useful_steps=40, area_ft2=200),
    "Mall": VenuePreset(visitors_day=12000, peak_multiplier=1.3, dwell_hours=1.5, pct_on_zone=10.0, useful_steps=70, area_ft2=180),
}

DEFAULTS_TECH = {
    "J_per_step": 3.0,
    "efficiency": 0.50,
    "storage_loss": 0.10,
}

DEFAULTS_COSTS = {
    "installed_cost_per_ft2": 125.0,  # typical rough range 75-160
    "fixed_cost": 15000.0,
    "maintenance_pct": 6.0,           # 2-10% of CAPEX
    "amort_years": 8,
}

DEFAULTS_SIZING = {
    "tile_ft2": 1.10,
}

GOAL_PRESETS_WH_PER_DAY = {
    "LED signage (10W for 4h/day)": 10 * 4,   # 40 Wh/day
    "Small sensor kit (5W continuous)": 5 * 24,  # 120 Wh/day
    "Small info display (8W for 8h/day)": 8 * 8,  # 64 Wh/day
    "Phone charging station (10 charges/day @ ~12Wh)": 10 * 12,  # 120 Wh/day
}


# -----------------------------
# Helpers
# -----------------------------
def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))

def energy_wh_per_day(
    visitors_day: float,
    peak_multiplier: float,
    pct_on_zone: float,
    useful_steps: float,
    J_per_step: float,
    efficiency: float,
    storage_loss: float,
) -> float:
    # Captured steps/day
    steps = visitors_day * peak_multiplier * (pct_on_zone / 100.0) * useful_steps
    wh = steps * J_per_step * efficiency * (1.0 - storage_loss) / 3600.0
    return float(max(0.0, wh))

def uncertainty_scenarios(base_inputs: dict) -> pd.DataFrame:
    """
    low/mid/high scenario multipliers focused on the two most uncertain parameters:
    pct_on_zone and useful_steps.
    """
    pct = base_inputs["pct_on_zone"]
    steps = base_inputs["useful_steps"]

    scenarios = [
        ("low", clamp(pct * 0.6, 0.1, 100.0), max(1, int(steps * 0.7))),
        ("mid", pct, steps),
        ("high", clamp(pct * 1.4, 0.1, 100.0), max(1, int(steps * 1.3))),
    ]

    rows = []
    for name, pct_s, steps_s in scenarios:
        wh = energy_wh_per_day(
            visitors_day=base_inputs["visitors_day"],
            peak_multiplier=base_inputs["peak_multiplier"],
            pct_on_zone=pct_s,
            useful_steps=steps_s,
            J_per_step=base_inputs["J_per_step"],
            efficiency=base_inputs["efficiency"],
            storage_loss=base_inputs["storage_loss"],
        )
        rows.append({"scenario": name, "pct_on_zone": pct_s, "useful_steps": steps_s, "Wh/day": wh, "kWh/day": wh / 1000.0})
    return pd.DataFrame(rows)

def realism_badge_pct_on_zone(pct: float) -> str:
    # heuristic
    if pct <= 15:
        return '<span class="badge badge-ok">Plausible</span>'
    if pct <= 30:
        return '<span class="badge badge-warn">Optimistic</span>'
    return '<span class="badge badge-bad">Unlikely</span>'

def realism_badge_steps(steps: float) -> str:
    if steps <= 120:
        return '<span class="badge badge-ok">Plausible</span>'
    if steps <= 180:
        return '<span class="badge badge-warn">Optimistic</span>'
    return '<span class="badge badge-bad">Unlikely</span>'

def what_can_it_power(wh_per_day: float) -> Dict[str, float]:
    # Very simple, pedagogical equivalents
    led10_h = wh_per_day / 10.0
    phone_charges = wh_per_day / 12.0
    sensor1w_days = wh_per_day / 24.0  # 1W for 24h = 24Wh/day
    eink_hours = wh_per_day / 3.0      # e-ink / low-power display ~3W
    return {
        "LED 10W (hours/day)": led10_h,
        "Phone charges (~12Wh each)": phone_charges,
        "1W sensor (days)": sensor1w_days,
        "Small e-ink/info display 3W (hours/day)": eink_hours,
    }

def costs_and_cost_per_kwh(
    area_ft2: float,
    installed_cost_per_ft2: float,
    fixed_cost: float,
    maintenance_pct: float,
    amort_years: int,
    kwh_per_year: float,
) -> Tuple[float, float, float]:
    capex = area_ft2 * installed_cost_per_ft2 + fixed_cost
    opex_year = (maintenance_pct / 100.0) * capex
    total_cost_over_life = capex + opex_year * amort_years

    if kwh_per_year <= 0:
        cost_per_kwh = float("inf")
    else:
        cost_per_kwh = total_cost_over_life / (kwh_per_year * amort_years)

    return capex, opex_year, cost_per_kwh

def verdicts(cost_per_kwh: float, wh_per_day: float) -> Tuple[str, str]:
    # Energy ROI verdict: basically "is cost/kWh remotely reasonable?"
    # Pedagogy verdict: based on whether it can power small local loads
    if math.isinf(cost_per_kwh) or cost_per_kwh > 0.5:
        energy_verdict = "NO-GO (Energy ROI) — Energy output is too small relative to costs."
    else:
        energy_verdict = "GO (Energy ROI) — Costs per kWh look reasonable."

    if wh_per_day >= 40:
        pedagogy_verdict = "GO (Pedagogy/Engagement) — Enough for meaningful small local uses (LEDs/sensors/displays)."
    else:
        pedagogy_verdict = "MAYBE (Pedagogy/Engagement) — Very modest energy, but still useful as an educational installation."

    return energy_verdict, pedagogy_verdict

def min_area_for_goal(
    target_wh_per_day: float,
    base_inputs: dict,
    area_ft2_current: float,
) -> float:
    # We assume energy scales linearly with area only indirectly via % on zone / steps.
    # Here we implement a *simple* sizing heuristic: required area scales with target energy
    # relative to current estimated energy, capped to avoid nonsense.
    current_wh = energy_wh_per_day(**base_inputs)
    if current_wh <= 0:
        return float("inf")
    factor = target_wh_per_day / current_wh
    return area_ft2_current * factor

def lightweight_forecast(df: pd.DataFrame, horizon_days: int = 30) -> pd.DataFrame:
    """
    Lightweight forecast without sklearn:
    - Convert date -> ordinal index
    - Fit simple linear regression using numpy polyfit (degree 1)
    - Adds a weekday effect (mean residual by weekday)
    This is intentionally small + explainable.
    """
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date")
    data["t"] = (data["date"] - data["date"].min()).dt.days.astype(int)

    y = data["visitors"].astype(float).to_numpy()
    t = data["t"].astype(float).to_numpy()

    # Linear trend
    if len(data) >= 2:
        slope, intercept = np.polyfit(t, y, 1)
    else:
        slope, intercept = 0.0, float(y[0]) if len(y) else (0.0, 0.0)

    data["trend_pred"] = intercept + slope * t
    data["resid"] = y - data["trend_pred"]

    # Weekday adjustment
    data["weekday"] = data["date"].dt.weekday
    weekday_effect = data.groupby("weekday")["resid"].mean().to_dict()

    # Forecast dates
    last_date = data["date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon_days, freq="D")
    future_t = (future_dates - data["date"].min()).days.astype(int).to_numpy()

    trend = intercept + slope * future_t
    wday = pd.Series(future_dates).dt.weekday
    adj = np.array([weekday_effect.get(int(d), 0.0) for d in wday])

    pred = np.maximum(0, trend + adj)

    out = pd.DataFrame({"date": future_dates, "visitors_pred": pred})
    return out

def run_codecarbon_tracker() -> Optional[float]:
    if not CODECARBON_OK:
        return None
    try:
        tracker = EmissionsTracker(project_name="kinetic-impact-forecast", output_dir=".", save_to_file=False, log_level="error")
        tracker.start()
        # Tiny "workload"
        _ = sum(i*i for i in range(20000))
        emissions = tracker.stop()
        return emissions  # kgCO2eq
    except Exception:
        return None


# -----------------------------
# State initialization
# -----------------------------
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.venue = "Museum"
    preset = VENUE_PRESETS[st.session_state.venue]

    st.session_state.visitors_day = preset.visitors_day
    st.session_state.peak_multiplier = preset.peak_multiplier
    st.session_state.dwell_hours = preset.dwell_hours
    st.session_state.pct_on_zone = preset.pct_on_zone
    st.session_state.useful_steps = preset.useful_steps
    st.session_state.area_ft2 = preset.area_ft2

    st.session_state.J_per_step = DEFAULTS_TECH["J_per_step"]
    st.session_state.efficiency = DEFAULTS_TECH["efficiency"]
    st.session_state.storage_loss = DEFAULTS_TECH["storage_loss"]

    st.session_state.installed_cost_per_ft2 = DEFAULTS_COSTS["installed_cost_per_ft2"]
    st.session_state.fixed_cost = DEFAULTS_COSTS["fixed_cost"]
    st.session_state.maintenance_pct = DEFAULTS_COSTS["maintenance_pct"]
    st.session_state.amort_years = DEFAULTS_COSTS["amort_years"]

    st.session_state.tile_ft2 = DEFAULTS_SIZING["tile_ft2"]

    st.session_state.mode = "Simple"
    st.session_state.goal_name = "LED signage (10W for 4h/day)"


def apply_preset(venue_name: str):
    st.session_state.venue = venue_name
    p = VENUE_PRESETS[venue_name]
    st.session_state.visitors_day = p.visitors_day
    st.session_state.peak_multiplier = p.peak_multiplier
    st.session_state.dwell_hours = p.dwell_hours
    st.session_state.pct_on_zone = p.pct_on_zone
    st.session_state.useful_steps = p.useful_steps
    st.session_state.area_ft2 = p.area_ft2


# -----------------------------
# INPUTS TAB
# -----------------------------
with tabs[0]:
    st.subheader("Guided inputs (Step 1 → Step 3)")

    colA, colB = st.columns([2, 1])
    with colB:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Mode**")
        st.session_state.mode = st.radio(
            "Choose input complexity",
            ["Simple", "Expert"],
            index=0 if st.session_state.mode == "Simple" else 1,
            label_visibility="collapsed",
        )
        if st.button("Reset to defaults"):
            for k in list(st.session_state.keys()):
                pass
            # re-init quickly
            st.session_state.initialized = False
            st.rerun()

        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
        st.markdown("**Load an example**")
        venue_choice = st.selectbox(
           venue_choice = st.selectbox(
    "Venue type",
    list(VENUE_PRESETS.keys()),
    index=list(VENUE_PRESETS.keys()).index(st.session_state.venue),
    key="venue_select",
)

if venue_choice != st.session_state.venue:
    apply_preset(venue_choice)
    st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    with colA:
        # Step 1
        with st.expander("Step 1 — Context", expanded=True):
            c1, c2, c3 = st.columns(3)

            with c1:
                venue = st.selectbox(
                    "Venue type",
                    list(VENUE_PRESETS.keys()),
                    index=list(VENUE_PRESETS.keys()).index(st.session_state.venue),
                )
                if venue != st.session_state.venue:
                    apply_preset(venue)

            with c2:
                st.session_state.visitors_day = st.number_input(
                    "Visitors per day (average)",
                    min_value=0,
                    value=int(st.session_state.visitors_day),
                    step=100,
                    help="Average visitors/day. For event venues, use visitors per event.",
                )

            with c3:
                st.session_state.peak_multiplier = st.slider(
                    "Peak multiplier (weekend/event)",
                    min_value=1.0,
                    max_value=3.0,
                    value=float(st.session_state.peak_multiplier),
                    step=0.05,
                    help="1.0 = typical day. 1.2–1.6 for busy weekends. Event venues can stay at 1.0.",
                )

            st.session_state.dwell_hours = st.slider(
                "Average dwell time (hours)",
                min_value=0.1,
                max_value=8.0,
                value=float(st.session_state.dwell_hours),
                step=0.1,
                help="Used for user intuition (not directly in the base energy formula).",
            )

        # Step 2
        with st.expander("Step 2 — Flow on equipped zone", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                st.session_state.pct_on_zone = st.slider(
                    "% of visitors crossing the equipped zone",
                    min_value=0.1,
                    max_value=60.0,
                    value=float(st.session_state.pct_on_zone),
                    step=0.1,
                    help="Typical: 2–10% for a small zone; 10–30% for a central passage.",
                )
                st.markdown(realism_badge_pct_on_zone(st.session_state.pct_on_zone), unsafe_allow_html=True)

            with c2:
                st.session_state.useful_steps = st.slider(
                    "Useful steps per visitor on the zone",
                    min_value=1,
                    max_value=250,
                    value=int(st.session_state.useful_steps),
                    step=1,
                    help="Typical: 20–60 (short zone) • 80–200 (long passage).",
                )
                st.markdown(realism_badge_steps(st.session_state.useful_steps), unsafe_allow_html=True)

        # Step 3
        with st.expander("Step 3 — Installation sizing + Costs + Sustainable AI", expanded=True):
            left, mid, right = st.columns([1.2, 1.0, 1.2])

            with left:
                st.markdown("### Installation sizing")
                st.session_state.area_ft2 = st.number_input(
                    "Equipped area (ft²)",
                    min_value=1.0,
                    value=float(st.session_state.area_ft2),
                    step=5.0,
                    help="Total surface area covered by tiles.",
                )
                st.session_state.tile_ft2 = st.number_input(
                    "Tile area (ft²)",
                    min_value=0.1,
                    value=float(st.session_state.tile_ft2),
                    step=0.05,
                    help="If unknown, 1.10 ft² is a reasonable placeholder.",
                )
                tiles = max(1, int(round(st.session_state.area_ft2 / st.session_state.tile_ft2)))
                st.info(f"Estimated tile count: **~{tiles} tiles** (area {st.session_state.area_ft2:.1f} ft²).")

                st.markdown("### Goal-based sizing (optional)")
                st.session_state.goal_name = st.selectbox("Pick a target use", list(GOAL_PRESETS_WH_PER_DAY.keys()))
                st.caption("This estimates the minimum area needed to reach a daily energy target (very rough, linear scaling).")

            with mid:
                st.markdown("### Technical assumptions")
                if st.session_state.mode == "Expert":
                    st.session_state.J_per_step = st.slider(
                        "Energy per step (J)",
                        min_value=1.0,
                        max_value=6.0,
                        value=float(st.session_state.J_per_step),
                        step=0.1,
                        help="Often cited order of magnitude: ~2–4 J/step for some kinetic tiles.",
                    )
                    st.session_state.efficiency = st.slider(
                        "Overall efficiency",
                        min_value=0.10,
                        max_value=0.80,
                        value=float(st.session_state.efficiency),
                        step=0.01,
                    )
                    st.session_state.storage_loss = st.slider(
                        "Storage / conversion loss",
                        min_value=0.0,
                        max_value=0.50,
                        value=float(st.session_state.storage_loss),
                        step=0.01,
                    )
                else:
                    st.caption("Switch to **Expert** mode to edit technical assumptions (J/step, efficiency, losses).")
                    st.markdown('<p class="small-muted">Using defaults: 3 J/step, 50% efficiency, 10% conversion loss.</p>', unsafe_allow_html=True)
                    st.session_state.J_per_step = DEFAULTS_TECH["J_per_step"]
                    st.session_state.efficiency = DEFAULTS_TECH["efficiency"]
                    st.session_state.storage_loss = DEFAULTS_TECH["storage_loss"]

            with right:
                st.markdown("### Costs (rough)")
                if st.session_state.mode == "Expert":
                    st.session_state.installed_cost_per_ft2 = st.slider(
                        "Installed cost ($/ft²)",
                        min_value=25.0,
                        max_value=900.0,
                        value=float(st.session_state.installed_cost_per_ft2),
                        step=5.0,
                        help="Typical rough range: 75–160 $/ft². Some showcase projects can be much higher.",
                    )
                    st.session_state.fixed_cost = st.number_input(
                        "Fixed costs (electrical/signage/works) $",
                        min_value=0.0,
                        value=float(st.session_state.fixed_cost),
                        step=1000.0,
                    )
                    st.session_state.maintenance_pct = st.slider(
                        "Annual maintenance (% of CAPEX)",
                        min_value=0.0,
                        max_value=15.0,
                        value=float(st.session_state.maintenance_pct),
                        step=0.5,
                    )
                    st.session_state.amort_years = st.slider(
                        "Amortization (years)",
                        min_value=1,
                        max_value=15,
                        value=int(st.session_state.amort_years),
                        step=1,
                    )
                else:
                    st.caption("Switch to **Expert** mode to edit cost assumptions.")
                    st.markdown('<p class="small-muted">Using defaults: $125/ft², $15k fixed, 6% maintenance, 8 years amortization.</p>', unsafe_allow_html=True)
                    st.session_state.installed_cost_per_ft2 = DEFAULTS_COSTS["installed_cost_per_ft2"]
                    st.session_state.fixed_cost = DEFAULTS_COSTS["fixed_cost"]
                    st.session_state.maintenance_pct = DEFAULTS_COSTS["maintenance_pct"]
                    st.session_state.amort_years = DEFAULTS_COSTS["amort_years"]

                st.markdown("### Sustainable AI (lightweight forecast)")
                st.caption("Upload a CSV with columns: `date, visitors` (or use the demo dataset).")

                use_demo = st.checkbox("Use demo dataset", value=True)
                uploaded = st.file_uploader("Upload CSV", type=["csv"])

                df_hist = None
                if use_demo:
                    demo = pd.DataFrame({
                        "date": pd.date_range("2025-11-01", periods=10, freq="D"),
                        "visitors": [1200,1500,900,950,980,1100,1300,1600,1800,1000],
                    })
                    df_hist = demo
                elif uploaded is not None:
                    try:
                        df = pd.read_csv(uploaded)
                        df_hist = df.copy()
                    except Exception as e:
                        st.error(f"Could not read CSV: {e}")

                if df_hist is not None:
                    st.dataframe(df_hist.head(15), use_container_width=True)


# -----------------------------
# RESULTS TAB
# -----------------------------
with tabs[1]:
    st.subheader("Results (energy + uncertainty + costs + Go/No-Go)")

    base_inputs = {
        "visitors_day": float(st.session_state.visitors_day),
        "peak_multiplier": float(st.session_state.peak_multiplier),
        "pct_on_zone": float(st.session_state.pct_on_zone),
        "useful_steps": float(st.session_state.useful_steps),
        "J_per_step": float(st.session_state.J_per_step),
        "efficiency": float(st.session_state.efficiency),
        "storage_loss": float(st.session_state.storage_loss),
    }

    wh_day = energy_wh_per_day(**base_inputs)
    kwh_day = wh_day / 1000.0
    kwh_month = kwh_day * 30.0
    kwh_year = kwh_day * 365.0

    # Costs
    capex, opex_year, cost_per_kwh = costs_and_cost_per_kwh(
        area_ft2=float(st.session_state.area_ft2),
        installed_cost_per_ft2=float(st.session_state.installed_cost_per_ft2),
        fixed_cost=float(st.session_state.fixed_cost),
        maintenance_pct=float(st.session_state.maintenance_pct),
        amort_years=int(st.session_state.amort_years),
        kwh_per_year=float(kwh_year),
    )

    # Verdicts
    energy_v, pedagogy_v = verdicts(cost_per_kwh, wh_day)

    # Executive summary
    st.markdown('<div class="card">', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("kWh / day", f"{kwh_day:.4f}")
    c2.metric("kWh / month (~30d)", f"{kwh_month:.2f}")
    c3.metric("kWh / year (~365d)", f"{kwh_year:.1f}")
    c4.metric("Approx. cost ($/kWh)", "∞" if math.isinf(cost_per_kwh) else f"{cost_per_kwh:,.2f}")
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    st.markdown("**Decision summary**")
    st.write(f"- **Energy ROI**: {energy_v}")
    st.write(f"- **Pedagogy/Engagement**: {pedagogy_v}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("")

    # Uncertainty
    df_unc = uncertainty_scenarios(base_inputs)
    st.markdown("### Uncertainty scenarios (Wh/day)")
    fig = plt.figure()
    plt.bar(df_unc["scenario"], df_unc["Wh/day"])
    plt.ylabel("Wh/day")
    plt.title("Low / Mid / High scenarios (focus on % crossing + useful steps)")
    st.pyplot(fig, clear_figure=True)

    st.dataframe(df_unc[["scenario", "pct_on_zone", "useful_steps", "Wh/day", "kWh/day"]], use_container_width=True)

    # What can it power
    st.markdown("### What can it power (pedagogical, local uses)")
    eq = what_can_it_power(wh_day)
    a, b, c, d = st.columns(4)
    a.metric("LED 10W (hours/day)", f"{eq['LED 10W (hours/day)']:.1f}")
    b.metric("Phone charges (~12Wh)", f"{eq['Phone charges (~12Wh each)']:.1f}")
    c.metric("1W sensor (days)", f"{eq['1W sensor (days)']:.2f}")
    d.metric("3W info display (hours/day)", f"{eq['Small e-ink/info display 3W (hours/day)']:.1f}")

    st.info(
        "Important: kinetic floors usually generate **modest energy**. The main value is often **engagement + pedagogy** (making energy tangible), "
        "plus powering small local loads (LEDs, sensors, small displays)."
    )

    # Costs block
    st.markdown("### Costs (CAPEX/OPEX) + cost per kWh (rough)")
    c1, c2, c3 = st.columns(3)
    c1.metric("CAPEX ($)", f"{capex:,.0f}")
    c2.metric("OPEX / year ($)", f"{opex_year:,.0f}")
    c3.metric("Cost approx. ($/kWh)", "∞" if math.isinf(cost_per_kwh) else f"{cost_per_kwh:,.2f}")

    # Goal-based sizing
    target_wh = GOAL_PRESETS_WH_PER_DAY.get(st.session_state.goal_name, 40)
    needed_area = min_area_for_goal(target_wh, base_inputs, float(st.session_state.area_ft2))
    st.markdown("### Goal-based sizing (minimum material)")
    if math.isinf(needed_area):
        st.warning("Current energy estimate is ~0, so goal-based sizing can't be computed. Increase % crossing / steps / visitors.")
    else:
        st.write(f"Target: **{st.session_state.goal_name}** → **{target_wh:.0f} Wh/day**")
        st.write(f"Estimated minimum area (very rough): **{needed_area:,.1f} ft²**")
        needed_tiles = int(round(needed_area / float(st.session_state.tile_ft2)))
        st.write(f"Estimated tiles: **~{needed_tiles:,}** (tile size {float(st.session_state.tile_ft2):.2f} ft²).")
        st.caption("This is a *linear scaling heuristic* for quick decision-support. Real deployments require site-specific testing.")

    # Forecast module
    st.markdown("### Sustainable AI (lightweight forecast)")
    st.caption("This forecast is intentionally lightweight: a simple trend + weekday pattern (no large model).")

    # Use same df_hist logic (rebuild quickly)
    df_hist = None
    # Try to reconstruct from session UI: user can be in Inputs tab
    # We'll provide a demo dataset if nothing exists
    demo = pd.DataFrame({
        "date": pd.date_range("2025-11-01", periods=10, freq="D"),
        "visitors": [1200,1500,900,950,980,1100,1300,1600,1800,1000],
    })
    df_hist = demo

    horizon = st.slider("Forecast horizon (days)", 7, 90, 30, 1)
    fc = lightweight_forecast(df_hist, horizon_days=int(horizon))

    fig2 = plt.figure()
    plt.plot(pd.to_datetime(df_hist["date"]), df_hist["visitors"], marker="o")
    plt.plot(pd.to_datetime(fc["date"]), fc["visitors_pred"], marker="o")
    plt.title("Visitors: history and lightweight forecast")
    plt.ylabel("Visitors")
    plt.xlabel("Date")
    st.pyplot(fig2, clear_figure=True)

    st.dataframe(fc.head(30), use_container_width=True)

    # CodeCarbon
    st.markdown("### CodeCarbon (compute footprint)")
    if CODECARBON_OK:
        if st.button("Measure compute emissions (tiny run)"):
            em = run_codecarbon_tracker()
            if em is None:
                st.warning("Could not measure emissions in this environment.")
            else:
                st.success(f"Estimated emissions for a tiny workload: **{em:.6f} kgCO₂e**")
                st.caption("In your pitch: 'We measure compute footprint and use lightweight forecasting to avoid over-installation.'")
    else:
        st.warning("CodeCarbon is not available in this runtime. If this persists, check requirements.txt and redeploy.")


# -----------------------------
# METHODOLOGY TAB
# -----------------------------
with tabs[2]:
    st.subheader("Methodology / Limits (anti-greenwashing)")

    st.markdown("### Transparent core formula")
    st.markdown(
        """
        <div class="card mono">
        Energy (Wh/day) = visitors/day × peak_multiplier × (%crossing/100) × useful_steps × J_per_step × efficiency × (1 - storage_loss) ÷ 3600
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### What this is NOT")
    st.markdown(
        """
        - Not a way to power a building.
        - Not a standalone climate solution.
        - Not a quote: costs are **rough ranges** and can vary widely by vendor/site constraints.
        """
    )

    st.markdown("### Why this qualifies as “Sustainable AI”")
    st.markdown(
        """
        - The forecast is **lightweight** (trend + weekday pattern, no large model).
        - The purpose is to **avoid over-installation** (materials, maintenance, costs) by sizing to realistic demand.
        - Compute footprint can be measured (CodeCarbon) to keep the AI component accountable.
        """
    )

    st.markdown("### Key limits / realism notes")
    st.markdown(
        """
        - Real output is usually modest. The strongest value is often **engagement + pedagogy** plus powering small local loads.
        - The two most uncertain inputs are typically:
          - % of visitors crossing the equipped zone
          - useful steps per visitor on that zone
        - Always validate assumptions with small pilots / sensors / short-term tests before full deployment.
        - No personal data: use aggregated counts only.
        """
    )

    st.markdown("### How to present it (capstone-friendly)")
    st.markdown(
        """
        Frame it as a **decision-support tool**, not a "magic energy generator":
        - "Should we install? Where? How big — and what's the minimum installation to achieve a small local target?"
        - Show uncertainty (low/mid/high) and emphasize the educational value.
        """
    )
