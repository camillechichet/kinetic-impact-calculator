import io
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st


# =========================
# Helpers (UX + Pedagogy)
# =========================

def realism_badge(label: str, value: float, ok_range: tuple[float, float], warn_range: tuple[float, float]) -> str:
    """Return a small badge to signal plausibility."""
    if ok_range[0] <= value <= ok_range[1]:
        return f"✅ {label}: plausible"
    if warn_range[0] <= value <= warn_range[1]:
        return f"⚠️ {label}: optimiste"
    return f"🚩 {label}: très improbable"


def fmt_money(x: float) -> str:
    return f"{x:,.0f}".replace(",", " ")


def fmt_num(x: float, nd: int = 2) -> str:
    return f"{x:.{nd}f}"


def safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else float("nan")


def make_demo_visitors(n_days: int = 60, start: date | None = None) -> pd.DataFrame:
    """Synthetic demo series with weekday pattern + trend + noise."""
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
    """
    Lightweight forecast WITHOUT sklearn:
    - trend: linear fit (numpy polyfit)
    - seasonality: weekday mean residual adjustment
    """
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
    resid_by_wday = {}
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


# =========================
# Presets
# =========================

PRESETS = {
    "Musée": {
        "pct_on_zone": 8.0,
        "useful_steps": 60.0,
        "peak_multiplier": 1.2,
        "avg_presence_hours": 2.0,
        "area_ft2": 120.0,
        "visitors_per_day": 1200,
    },
    "Gare": {
        "pct_on_zone": 18.0,
        "useful_steps": 120.0,
        "peak_multiplier": 1.5,
        "avg_presence_hours": 0.75,
        "area_ft2": 220.0,
        "visitors_per_day": 8000,
    },
    "Stade": {
        "pct_on_zone": 12.0,
        "useful_steps": 80.0,
        "peak_multiplier": 2.5,
        "avg_presence_hours": 3.0,
        "area_ft2": 300.0,
        "visitors_per_day": 25000,
    },
    "Centre commercial": {
        "pct_on_zone": 10.0,
        "useful_steps": 90.0,
        "peak_multiplier": 1.4,
        "avg_presence_hours": 1.5,
        "area_ft2": 180.0,
        "visitors_per_day": 6000,
    },
}


# =========================
# Streamlit config
# =========================

st.set_page_config(page_title="Kinetic Impact Calculator", page_icon="⚡", layout="wide")

st.title("Kinetic Impact Calculator (inspired by Coldplay)")
st.caption(
    "Decision-support MVP: estimate kinetic floor energy, practical uses, costs (CAPEX/OPEX), "
    "uncertainty scenarios, and a lightweight forecast module (Sustainable AI)."
)

with st.expander("Quick start (1 minute)"):
    st.markdown(
        """
1) Choisis un **type de lieu** puis clique **Appliquer preset**.  
2) Ajuste **% sur zone** et **pas utiles** (badges ✅⚠️🚩).  
3) Va sur **Results** pour voir **Executive summary + 2 verdicts**.  
4) Option : CSV (**date, visitors**) pour la **prévision** (IA légère).
"""
    )

# =========================
# Session state defaults
# =========================

DEFAULTS = {
    "place_type": "Musée",
    "visitors_per_day": 3300,
    "peak_multiplier": 1.0,
    "avg_presence_hours": 2.5,
    "pct_on_zone": 12.0,
    "useful_steps": 115.0,
    "J_per_step": 3.0,
    "efficiency": 0.50,
    "storage_loss": 0.10,
    "area_ft2": 190.0,
    "tile_area_ft2": 1.10,
    "installed_cost_per_ft2": 175.0,
    "fixed_cost": 20000.0,
    "maintenance_pct": 8.0,
    "amort_years": 9,
    "forecast_horizon_days": 14,
    "use_demo_dataset": True,
}

if "inputs" not in st.session_state:
    st.session_state.inputs = DEFAULTS.copy()
else:
    for k, v in DEFAULTS.items():
        st.session_state.inputs.setdefault(k, v)

tab_inputs, tab_results, tab_methods = st.tabs(["Inputs", "Results", "Methodology / Limits"])


# =========================
# Inputs
# =========================
with tab_inputs:
    c1, c2, c3 = st.columns([1.1, 1.0, 1.0], gap="large")

    with c1:
        st.subheader("Context")

        place_type = st.selectbox(
            "Type de lieu",
            options=list(PRESETS.keys()),
            index=list(PRESETS.keys()).index(st.session_state.inputs["place_type"])
            if st.session_state.inputs["place_type"] in PRESETS else 0
        )

        if st.button("Appliquer preset du lieu"):
            p = PRESETS.get(place_type, {})
            for k, v in p.items():
                st.session_state.inputs[k] = v
            st.session_state.inputs["place_type"] = place_type
            st.success(f"Preset appliqué pour: {place_type}")
            st.rerun()

        st.session_state.inputs["place_type"] = place_type

        visitors_per_day = st.number_input(
            "Visiteurs / jour (moyenne)",
            min_value=0,
            value=int(st.session_state.inputs["visitors_per_day"]),
            step=50
        )
        st.caption("Ex: musée 500–5 000 ; gare 2 000–50 000 ; stade (événement) 10 000–80 000")
        st.write(realism_badge("Visiteurs/jour", float(visitors_per_day), ok_range=(300, 50000), warn_range=(50, 120000)))

        peak_multiplier = st.slider(
            "Multiplicateur pic (week-end / événement)",
            1.0, 5.0, float(st.session_state.inputs["peak_multiplier"]), 0.05
        )
        st.caption("Ex: 1.0 (normal), 1.2–1.8 (week-end), 2–3 (événement)")
        st.write(realism_badge("Pic", peak_multiplier, ok_range=(1.0, 2.5), warn_range=(1.0, 4.0)))

        avg_presence_hours = st.slider(
            "Durée moyenne de présence (heures)",
            0.25, 6.0, float(st.session_state.inputs["avg_presence_hours"]), 0.25
        )
        st.caption("Ex: gare 0.25–1h ; musée 1–3h ; stade 2–5h")
        st.write(realism_badge("Présence", avg_presence_hours, ok_range=(0.5, 4.0), warn_range=(0.25, 6.0)))

        st.markdown("---")
        st.subheader("Flow on equipped zone")

        pct_on_zone = st.slider(
            "% visiteurs passant sur la zone équipée",
            0.0, 100.0, float(st.session_state.inputs["pct_on_zone"]), 0.5
        )
        st.caption("Ex: 2–10% (zone petite) / 10–30% (zone centrale)")
        st.write(realism_badge("% sur zone", pct_on_zone, ok_range=(2, 30), warn_range=(0.5, 60)))

        useful_steps = st.slider(
            "Pas utiles / visiteur sur zone",
            0.0, 300.0, float(st.session_state.inputs["useful_steps"]), 5.0
        )
        st.caption("Ex: 20–60 (petit couloir) / 80–200 (long passage)")
        st.write(realism_badge("Pas utiles", useful_steps, ok_range=(20, 200), warn_range=(5, 300)))

    with c2:
        st.subheader("Technical assumptions")

        J_per_step = st.slider(
            "Énergie par pas (J)",
            1.0, 6.0, float(st.session_state.inputs["J_per_step"]), 0.1
        )
        st.caption("Ordre de grandeur: 2–4 J/pas (souvent cité).")
        st.write(realism_badge("J/pas", J_per_step, ok_range=(2.0, 4.0), warn_range=(1.0, 6.0)))

        efficiency = st.slider(
            "Rendement global",
            0.10, 0.90, float(st.session_state.inputs["efficiency"]), 0.01
        )
        st.caption("Inclut mécanique + conversion: souvent 0.3–0.6.")
        st.write(realism_badge("Rendement", efficiency, ok_range=(0.30, 0.60), warn_range=(0.15, 0.80)))

        storage_loss = st.slider(
            "Pertes stockage / conversion",
            0.0, 0.50, float(st.session_state.inputs["storage_loss"]), 0.01
        )
        st.caption("Ex: 5–20% selon stockage/régulation.")
        st.write(realism_badge("Pertes", storage_loss, ok_range=(0.05, 0.20), warn_range=(0.0, 0.35)))

        st.markdown("---")
        st.subheader("Installation sizing (simple)")

        area_ft2 = st.number_input(
            "Surface équipée (ft²)",
            min_value=1.0,
            value=float(st.session_state.inputs["area_ft2"]),
            step=10.0
        )
        st.caption("Ex: 50–400 ft² selon projet (petite/moyenne/grande zone).")

        tile_area_ft2 = st.number_input(
            "Surface d’une dalle (ft²)",
            min_value=0.2,
            value=float(st.session_state.inputs["tile_area_ft2"]),
            step=0.05
        )
        st.caption("Ex: ~1 ft² pour une dalle ~30x30 cm (ordre de grandeur).")

        est_tiles = int(round(area_ft2 / tile_area_ft2))
        st.info(f"Estimation: ~ **{est_tiles} dalles** pour {fmt_num(area_ft2,0)} ft² (si 1 dalle ≈ {fmt_num(tile_area_ft2,2)} ft²).")

    with c3:
        st.subheader("Costs")

        installed_cost_per_ft2 = st.slider(
            "Coût installé ($/ft²)",
            50.0, 900.0, float(st.session_state.inputs["installed_cost_per_ft2"]), 5.0
        )
        st.caption("Ordre de grandeur: 75–160 $/ft² (souvent cité). Projets “vitrine” peuvent être bien plus élevés.")
        st.write(realism_badge("$/ft²", installed_cost_per_ft2, ok_range=(75, 200), warn_range=(50, 600)))

        fixed_cost = st.number_input(
            "Coût fixe (travaux/élec/signalétique) $",
            min_value=0.0,
            value=float(st.session_state.inputs["fixed_cost"]),
            step=1000.0
        )

        maintenance_pct = st.slider(
            "Maintenance annuelle (% du CAPEX)",
            0.0, 20.0, float(st.session_state.inputs["maintenance_pct"]), 0.5
        )
        st.caption("Ex: 2–10% du CAPEX (paramètre libre).")

        amort_years = st.slider(
            "Amortissement (années)",
            1, 20, int(st.session_state.inputs["amort_years"]), 1
        )

        st.markdown("---")
        st.subheader("Sustainable AI (lightweight)")

        use_demo = st.checkbox("Utiliser dataset démo", value=bool(st.session_state.inputs["use_demo_dataset"]))
        uploaded = st.file_uploader("Upload CSV (colonnes: date, visitors)", type=["csv"])
        horizon = st.slider("Horizon de prévision (jours)", 7, 60, int(st.session_state.inputs["forecast_horizon_days"]), 1)

        st.caption("IA frugale : trend + saisonnalité (jours de semaine).")

        # Store all inputs
        st.session_state.inputs.update({
            "visitors_per_day": int(visitors_per_day),
            "peak_multiplier": float(peak_multiplier),
            "avg_presence_hours": float(avg_presence_hours),
            "pct_on_zone": float(pct_on_zone),
            "useful_steps": float(useful_steps),
            "J_per_step": float(J_per_step),
            "efficiency": float(efficiency),
            "storage_loss": float(storage_loss),
            "area_ft2": float(area_ft2),
            "tile_area_ft2": float(tile_area_ft2),
            "installed_cost_per_ft2": float(installed_cost_per_ft2),
            "fixed_cost": float(fixed_cost),
            "maintenance_pct": float(maintenance_pct),
            "amort_years": int(amort_years),
            "use_demo_dataset": bool(use_demo),
            "forecast_horizon_days": int(horizon),
        })

        # Optional forecast preview
        df_hist = None
        if uploaded is not None:
            try:
                df_hist = load_csv_visitors(uploaded)
                st.success(f"CSV chargé: {len(df_hist)} lignes.")
            except Exception as e:
                st.error(f"Impossible de lire le CSV: {e}")
        elif use_demo:
            df_hist = make_demo_visitors(n_days=60)

        if df_hist is not None:
            st.write("Aperçu dataset:")
            st.dataframe(df_hist.tail(10), use_container_width=True)

            df_fc = lightweight_forecast(df_hist, horizon_days=horizon)
            st.write("Prévision (IA légère):")
            st.dataframe(df_fc.head(10), use_container_width=True)

            chart_df = pd.concat(
                [
                    df_hist.rename(columns={"visitors": "value"}).assign(kind="history")[["date", "value", "kind"]],
                    df_fc.rename(columns={"visitors_pred": "value"}).assign(kind="forecast")[["date", "value", "kind"]],
                ],
                ignore_index=True
            )
            chart_df["date"] = pd.to_datetime(chart_df["date"])
            st.line_chart(chart_df.set_index("date")[["value"]])

            st.info("Astuce capstone: l’objectif = éviter la sur-installation (matériaux/maintenance) via scénarios + prévision IA.")


# =========================
# Compute shared results
# =========================
inp = st.session_state.inputs

steps_captured = (
    inp["visitors_per_day"]
    * inp["peak_multiplier"]
    * (inp["pct_on_zone"] / 100.0)
    * inp["useful_steps"]
)

energy_wh_day = steps_captured * inp["J_per_step"] * inp["efficiency"] * (1.0 - inp["storage_loss"]) / 3600.0
energy_kwh_day = energy_wh_day / 1000.0
energy_kwh_month = energy_kwh_day * 30.0
energy_kwh_year = energy_kwh_day * 365.0

# Costs
capex = inp["area_ft2"] * inp["installed_cost_per_ft2"] + inp["fixed_cost"]
opex_year = (inp["maintenance_pct"] / 100.0) * capex
N = inp["amort_years"]
total_cost_N = capex + opex_year * N
cost_per_kwh = safe_div(total_cost_N, energy_kwh_year * N) if energy_kwh_year > 0 else float("inf")

# Uncertainty scenarios
scenarios = {"low": 0.6, "mid": 1.0, "high": 1.4}
rows = []
for name, mult in scenarios.items():
    wh = energy_wh_day * mult
    rows.append({"scenario": name, "Wh/day": wh, "kWh/day": wh / 1000.0})
df_scen = pd.DataFrame(rows)

# =========================
# Equivalences (MORE PEDAGOGICAL, per day)
# =========================

# LED 10W
led10w_hours_per_day = safe_div(energy_wh_day, 10.0)

# Phone charges (12Wh)
phone_charges_per_day = safe_div(energy_wh_day, 12.0)

# Sensor 1W continuous => 24 Wh/day
sensor1w_days_per_day = safe_div(energy_wh_day, 24.0)  # days of sensor powered by 1 day harvest
sensor1w_hours_per_day = sensor1w_days_per_day * 24.0

# E-ink small sign (example): assume 2Wh/day
eink_sign_days_per_day = safe_div(energy_wh_day, 2.0)

# CO2 sensor node (example): assume 5Wh/day
co2_sensor_days_per_day = safe_div(energy_wh_day, 5.0)

# Projector: assume 200W => Wh / 200W = hours => minutes
projector_minutes_per_day = safe_div(energy_wh_day, 200.0) * 60.0

# =========================
# Verdicts (split into 2)
# =========================

# 1) Energy ROI verdict
roi_verdict = "NO-GO (ROI énergie) — coût/kWh très élevé vs production."
if energy_kwh_year >= 200 and np.isfinite(cost_per_kwh) and cost_per_kwh <= 0.5:
    roi_verdict = "GO (ROI énergie) — rare, mais ici ça devient comparativement intéressant."
elif energy_kwh_year >= 50 and np.isfinite(cost_per_kwh) and cost_per_kwh <= 2.0:
    roi_verdict = "MIXTE (ROI énergie) — jouable pour micro-usages, mais coût/kWh reste élevé."

# 2) Pedagogy / engagement verdict
pedago_verdict = "GO (pédagogie/engagement) — impact principal = rendre l’énergie tangible + vitrine durable."
if inp["pct_on_zone"] < 1.0 or inp["useful_steps"] < 10:
    pedago_verdict = "MIXTE (pédagogie/engagement) — zone trop peu traversée : revoir l’emplacement / surface."

# One-liner summary for actionability
key_driver = "La réalité dépend surtout de **% sur zone** et **pas utiles** (emplacement + design du parcours)."


# =========================
# Results tab (VISUAL + ACTIONABLE)
# =========================
with tab_results:
    st.subheader("Results (visual + actionable)")

    # --- Executive summary (top)
    with st.container(border=True):
        st.markdown("### Executive summary (résumé décisionnel)")

        # Energy label
        energy_label = f"{fmt_num(energy_kwh_year, 1)} kWh/an"
        if energy_kwh_year < 10:
            energy_label += " (très faible)"
        elif energy_kwh_year < 100:
            energy_label += " (modeste)"

        usage_realiste = "LEDs / capteurs / écran pédagogique (micro-usages locaux)"

        st.write(f"**Énergie estimée** : {energy_label}")
        st.write(f"**Usage réaliste** : {usage_realiste}")
        st.write(f"**Coût total** : {fmt_money(capex)}$ CAPEX + {fmt_money(opex_year)}$/an OPEX → ~ **{fmt_num(cost_per_kwh, 2)} $/kWh**")
        st.write(f"**Verdict ROI énergie** : {roi_verdict}")
        st.write(f"**Verdict pédagogie / vitrine** : {pedago_verdict}")
        st.info(key_driver)

    st.markdown("---")

    # --- Metrics row
    m1, m2, m3 = st.columns(3)
    m1.metric("kWh / jour", fmt_num(energy_kwh_day, 3))
    m2.metric("kWh / mois (~30j)", fmt_num(energy_kwh_month, 2))
    m3.metric("kWh / an (~365j)", fmt_num(energy_kwh_year, 1))

    # --- Scenarios: GRAPH instead of table
    st.markdown("### Uncertainty scenarios (visual)")
    scen_plot = df_scen.copy()
    scen_plot["scenario"] = pd.Categorical(scen_plot["scenario"], categories=["low", "mid", "high"], ordered=True)
    scen_plot = scen_plot.sort_values("scenario").set_index("scenario")[["Wh/day"]]
    st.bar_chart(scen_plot)

    st.caption("🧠 Interprétation : l’incertitude vient surtout de **% sur zone** et **pas utiles** (où tu places la zone + design du passage).")

    st.markdown("---")

    # --- Equivalences (more "speaking", per day)
    st.markdown("### What can it power (more intuitive, per day)")

    e1, e2, e3 = st.columns(3)
    e1.metric("LED 10W (heures / jour)", fmt_num(led10w_hours_per_day, 2))
    e2.metric("Charges téléphone (~12Wh) / jour", fmt_num(phone_charges_per_day, 2))
    e3.metric("Capteur 1W (heures / jour)", fmt_num(sensor1w_hours_per_day, 2))

    e4, e5, e6 = st.columns(3)
    e4.metric("Petit panneau e-ink (2Wh/j) : jours", fmt_num(eink_sign_days_per_day, 2))
    e5.metric("Capteur CO₂ (5Wh/j) : jours", fmt_num(co2_sensor_days_per_day, 2))
    e6.metric("Projecteur (200W) : minutes / jour", fmt_num(projector_minutes_per_day, 2))

    st.info(
        "Important : l’énergie récupérée est généralement **modeste**. "
        "La valeur forte est souvent **pédagogique/engagement** (comme Coldplay : rendre l’énergie tangible), "
        "plus des **micro-usages locaux** (LEDs, capteurs, petit affichage)."
    )

    st.markdown("---")

    # --- Costs
    st.markdown("### Costs (CAPEX/OPEX) + cost per kWh (rough)")
    c1, c2, c3 = st.columns(3)
    c1.metric("CAPEX ($)", fmt_money(capex))
    c2.metric("OPEX / an ($)", fmt_money(opex_year))
    c3.metric("Coût approx ($/kWh)", fmt_num(cost_per_kwh, 2) if np.isfinite(cost_per_kwh) else "∞")

    st.markdown("---")

    # --- Verdict section (split)
    st.markdown("### Verdicts (split: ROI vs pedagogy)")
    v1, v2 = st.columns(2)
    with v1:
        st.warning(roi_verdict)
        st.caption("Conseil : si tu veux améliorer le ROI, le levier #1 = **augmenter l’usage sur zone** (emplacement / parcours) sans exploser la surface.")
    with v2:
        st.success(pedago_verdict)
        st.caption("Conseil : pour maximiser la valeur pédagogique, ajoute un **affichage** “Tu viens de produire X Wh” + une jauge “objectif LED/capteur/écran”.")

    # --- Export
    st.markdown("### Export")
    export = {
        "place_type": inp["place_type"],
        "visitors_per_day": inp["visitors_per_day"],
        "peak_multiplier": inp["peak_multiplier"],
        "avg_presence_hours": inp["avg_presence_hours"],
        "pct_on_zone": inp["pct_on_zone"],
        "useful_steps": inp["useful_steps"],
        "J_per_step": inp["J_per_step"],
        "efficiency": inp["efficiency"],
        "storage_loss": inp["storage_loss"],
        "area_ft2": inp["area_ft2"],
        "tile_area_ft2": inp["tile_area_ft2"],
        "installed_cost_per_ft2": inp["installed_cost_per_ft2"],
        "fixed_cost": inp["fixed_cost"],
        "maintenance_pct": inp["maintenance_pct"],
        "amort_years": inp["amort_years"],
        "steps_captured_per_day": steps_captured,
        "energy_Wh_day": energy_wh_day,
        "energy_kWh_day": energy_kwh_day,
        "energy_kWh_year": energy_kwh_year,
        "capex_$": capex,
        "opex_year_$": opex_year,
        "cost_per_kWh_$": cost_per_kwh,
        "roi_verdict": roi_verdict,
        "pedago_verdict": pedago_verdict,
    }
    out_df = pd.DataFrame([export])
    buf = io.StringIO()
    out_df.to_csv(buf, index=False)
    st.download_button(
        "Télécharger résultats (CSV)",
        data=buf.getvalue().encode("utf-8"),
        file_name="kinetic_impact_results.csv",
        mime="text/csv",
    )


# =========================
# Methodology tab
# =========================
with tab_methods:
    st.subheader("Methodology / Limits (anti-greenwashing)")

    st.markdown("**Core formula (transparent):**")
    st.code(
        "Energy (Wh/day) = visitors/day × peak_multiplier × (%on_zone/100) × useful_steps × J_per_step × efficiency × (1 - storage_loss) ÷ 3600",
        language="text"
    )

    st.markdown("**Why it’s “Sustainable AI”:**")
    st.write(
        "The forecast is lightweight (no large models). It helps avoid over-installation (materials, costs, maintenance) "
        "by sizing to realistic demand."
    )

    st.markdown("**Limits:**")
    st.markdown(
        """
- Energy outputs are usually modest; strongest benefit is often engagement/pedagogy + powering small local loads.
- Costs vary by vendor and site constraints. Treat cost outputs as ranges, not quotes.
- No personal data: use aggregated visitor counts only.
"""
    )
