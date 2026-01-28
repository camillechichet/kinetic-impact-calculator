import io
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st


# =========================
# Helpers (UX + Pedagogy)
# =========================

def realism_badge(label: str, value: float, ok_range: tuple[float, float], warn_range: tuple[float, float]) -> str:
    if ok_range[0] <= value <= ok_range[1]:
        return f"✅ {label}: plausible"
    if warn_range[0] <= value <= warn_range[1]:
        return f"⚠️ {label}: optimiste"
    return f"🚩 {label}: très improbable"


def verdict_badge(kind: str, reason: str) -> tuple[str, str]:
    k = kind.upper().strip()
    if k == "GO":
        return (f"✅ GO — {reason}", "success")
    if k == "MIXED":
        return (f"⚠️ MIXTE — {reason}", "warning")
    if k == "NO-GO":
        return (f"⛔ NO-GO — {reason}", "error")
    return (f"ℹ️ {kind} — {reason}", "info")


def fmt_money(x: float) -> str:
    return f"{x:,.0f}".replace(",", " ")


def fmt_num(x: float, nd: int = 2) -> str:
    return f"{x:.{nd}f}"


def safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else float("nan")


def ft2_to_m2(x_ft2: float) -> float:
    return x_ft2 * 0.092903


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
# Presets (and example scenarios)
# =========================

PRESETS = {
    "Musée": {
        "pct_on_zone": 8.0,
        "useful_steps": 60.0,
        "peak_multiplier": 1.2,
        "area_ft2": 120.0,
        "visitors_per_day": 1200,
        "installed_cost_per_ft2": 140.0,
        "fixed_cost": 12000.0,
        "J_net_per_step": 0.5,
        "auto_consumption_wh_day": 0.0,
    },
    "Gare": {
        "pct_on_zone": 18.0,
        "useful_steps": 120.0,
        "peak_multiplier": 1.5,
        "area_ft2": 220.0,
        "visitors_per_day": 8000,
        "installed_cost_per_ft2": 160.0,
        "fixed_cost": 25000.0,
        "J_net_per_step": 0.5,
        "auto_consumption_wh_day": 5.0,
    },
    "Stade": {
        "pct_on_zone": 12.0,
        "useful_steps": 80.0,
        "peak_multiplier": 2.5,
        "area_ft2": 300.0,
        "visitors_per_day": 25000,
        "installed_cost_per_ft2": 180.0,
        "fixed_cost": 40000.0,
        "J_net_per_step": 0.5,
        "auto_consumption_wh_day": 10.0,
    },
    "Centre commercial": {
        "pct_on_zone": 10.0,
        "useful_steps": 90.0,
        "peak_multiplier": 1.4,
        "area_ft2": 180.0,
        "visitors_per_day": 6000,
        "installed_cost_per_ft2": 150.0,
        "fixed_cost": 20000.0,
        "J_net_per_step": 0.5,
        "auto_consumption_wh_day": 5.0,
    },
}

EXAMPLE_SCENARIOS = {
    "Musée (réaliste)": {
        "place_type": "Musée",
        "visitors_per_day": 1500,
        "pct_on_zone": 8.0,
        "useful_steps": 60.0,
        "J_net_per_step": 0.5,
        "auto_consumption_wh_day": 0.0,
        "area_ft2": 120.0,
        "installed_cost_per_ft2": 140.0,
        "fixed_cost": 12000.0,
        "maintenance_pct": 6.0,
        "amort_years": 8,
        "peak_multiplier": 1.2,
    },
    "Gare (réaliste)": {
        "place_type": "Gare",
        "visitors_per_day": 12000,
        "pct_on_zone": 18.0,
        "useful_steps": 120.0,
        "J_net_per_step": 0.5,
        "auto_consumption_wh_day": 5.0,
        "area_ft2": 220.0,
        "installed_cost_per_ft2": 160.0,
        "fixed_cost": 25000.0,
        "maintenance_pct": 8.0,
        "amort_years": 10,
        "peak_multiplier": 1.5,
    },
    "Stade (événement)": {
        "place_type": "Stade",
        "visitors_per_day": 35000,
        "pct_on_zone": 12.0,
        "useful_steps": 80.0,
        "J_net_per_step": 0.5,
        "auto_consumption_wh_day": 10.0,
        "area_ft2": 300.0,
        "installed_cost_per_ft2": 180.0,
        "fixed_cost": 40000.0,
        "maintenance_pct": 8.0,
        "amort_years": 10,
        "peak_multiplier": 2.5,
    },
}


# =========================
# Streamlit config
# =========================

st.set_page_config(page_title="Kinetic Impact Calculator", page_icon="⚡", layout="wide")

st.title("Kinetic Impact Calculator")
st.caption(
    "Decision-support MVP: estimate kinetic floor energy, practical uses, costs (CAPEX/OPEX), "
    "uncertainty scenarios, and a lightweight forecast module (Sustainable AI)."
)

# =========================
# Session state defaults
# =========================

DEFAULTS = {
    "place_type": "Musée",
    "visitors_per_day": 3300,
    "peak_multiplier": 1.0,
    "pct_on_zone": 12.0,
    "useful_steps": 115.0,

    # NEW: single net electrical energy per step (no double counting)
    "J_net_per_step": 0.5,  # J_net/step

    # NEW: auto-consumption
    "auto_consumption_wh_day": 0.0,

    # sizing + costs
    "area_ft2": 190.0,
    "tile_area_ft2": 1.10,
    "installed_cost_per_ft2": 175.0,
    "fixed_cost": 20000.0,
    "maintenance_pct": 8.0,
    "amort_years": 9,

    # forecast
    "forecast_horizon_days": 14,
    "use_demo_dataset": True,
}

if "inputs" not in st.session_state:
    st.session_state.inputs = DEFAULTS.copy()
else:
    for k, v in DEFAULTS.items():
        st.session_state.inputs.setdefault(k, v)

# =========================
# Top UI buttons (small details that help)
# =========================
top_l, top_m, top_r = st.columns([1.2, 1.2, 1.6])
with top_l:
    if st.button("🔄 Reset to defaults"):
        st.session_state.inputs = DEFAULTS.copy()
        st.success("Defaults rechargés.")
        st.rerun()

with top_m:
    scenario_name = st.selectbox("Load example scenario", list(EXAMPLE_SCENARIOS.keys()))
    if st.button("📌 Charger scénario exemple"):
        ex = EXAMPLE_SCENARIOS[scenario_name]
        for k, v in ex.items():
            st.session_state.inputs[k] = v
        st.success(f"Scénario chargé: {scenario_name}")
        st.rerun()

with top_r:
    with st.expander("Quick start (1 minute)"):
        st.markdown(
            """
1) Choisis un **type de lieu** puis clique **Appliquer preset** (ou charge un scénario).  
2) Ajuste **% sur zone** et **pas utiles** (badges ✅⚠️🚩 + warning cohérence).  
3) Ajuste **J_net/pas** (énergie électrique nette récupérée par pas) + **auto-consommation**.  
4) Va sur **Results** pour voir **Executive summary + 2 verdicts**.  
5) Option : CSV (**date, visitors**) pour la **prévision** (IA légère).
"""
        )

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

        # A + B: replace J + efficiency + losses by ONE parameter: J_net per step
        J_net_per_step = st.slider(
            "Énergie électrique nette récupérée par pas (J_net/pas)",
            0.005, 1.0, float(st.session_state.inputs["J_net_per_step"]), 0.005
        )
        st.caption(
            "Définition : énergie électrique réellement délivrée à une charge/stockage, déjà 'net'. "
            "Ex: tribo ~mJ → EM ~centaines de mJ."
        )
        st.write(realism_badge("J_net/pas", J_net_per_step, ok_range=(0.05, 0.8), warn_range=(0.01, 1.0)))

        # C: auto-consumption
        auto_consumption_wh_day = st.number_input(
            "Auto-consommation système (Wh/jour)",
            min_value=0.0,
            value=float(st.session_state.inputs["auto_consumption_wh_day"]),
            step=1.0
        )
        st.caption("Ex: contrôleur/communication/LED 'always-on' peut consommer quelques Wh/j et annuler le gain à faible énergie.")

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

        # E: coherence warning (rough geometry sanity check)
        # Assume zone length ~ sqrt(area_m2) and step length ~ v/f with v=1.34 m/s, f=2 Hz => 0.67 m/step
        area_m2 = ft2_to_m2(area_ft2)
        approx_length_m = max(0.5, float(np.sqrt(area_m2)))  # avoid 0 for tiny areas
        v_free = 1.34
        f_step = 2.0
        step_len = v_free / f_step  # ~0.67 m
        max_steps_single_cross = approx_length_m / step_len
        # allow some back-and-forth / meandering: ×2.5
        plausible_upper_steps = 2.5 * max_steps_single_cross

        if useful_steps > plausible_upper_steps and useful_steps > 30:
            st.warning(
                f"⚠️ Contrôle cohérence : **{fmt_num(useful_steps,0)} pas/visiteur** paraît élevé "
                f"vs une zone de ~{fmt_num(area_ft2,0)} ft² (longueur typique ~{fmt_num(approx_length_m,1)} m). "
                f"Risque de **surestimation**. (Check surtout l'emplacement + la longueur réellement traversée.)"
            )

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
            "pct_on_zone": float(pct_on_zone),
            "useful_steps": float(useful_steps),
            "J_net_per_step": float(J_net_per_step),
            "auto_consumption_wh_day": float(auto_consumption_wh_day),
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

            st.info("Astuce capstone : objectif = éviter la sur-installation (matériaux/maintenance) via scénarios + prévision IA.")


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

# Net electrical energy harvested (no efficiency/loss terms anymore)
gross_energy_wh_day = steps_captured * inp["J_net_per_step"] / 3600.0

# Subtract auto-consumption explicitly, clamp at 0
net_energy_wh_day = max(0.0, gross_energy_wh_day - inp["auto_consumption_wh_day"])

net_energy_kwh_day = net_energy_wh_day / 1000.0
net_energy_kwh_month = net_energy_kwh_day * 30.0
net_energy_kwh_year = net_energy_kwh_day * 365.0

# Costs
capex = inp["area_ft2"] * inp["installed_cost_per_ft2"] + inp["fixed_cost"]
opex_year = (inp["maintenance_pct"] / 100.0) * capex
N = inp["amort_years"]
total_cost_N = capex + opex_year * N
cost_per_kwh = safe_div(total_cost_N, net_energy_kwh_year * N) if net_energy_kwh_year > 0 else float("inf")

# Uncertainty scenarios (keep for communication; based mainly on % on zone + steps variability)
scenarios = {"low": 0.6, "mid": 1.0, "high": 1.4}
rows = []
for name, mult in scenarios.items():
    wh = net_energy_wh_day * mult
    rows.append({"scenario": name, "Wh/day": wh, "kWh/day": wh / 1000.0})
df_scen = pd.DataFrame(rows)

# Equivalences (per day)
led10w_hours_per_day = safe_div(net_energy_wh_day, 10.0)
phone_charges_per_day = safe_div(net_energy_wh_day, 12.0)
sensor1w_hours_per_day = safe_div(net_energy_wh_day, 1.0)  # 1W for 1 hour = 1Wh
eink_sign_days_per_day = safe_div(net_energy_wh_day, 2.0)   # 2Wh/day
co2_sensor_days_per_day = safe_div(net_energy_wh_day, 5.0)  # 5Wh/day
projector_minutes_per_day = safe_div(net_energy_wh_day, 200.0) * 60.0  # 200W projector

# Verdicts (split in 2) + reason
# ROI energy
if net_energy_kwh_year <= 0:
    roi_kind, roi_reason = "NO-GO", "énergie nette ~0 après auto-consommation (vérifie paramètres + auto-conso)."
elif np.isfinite(cost_per_kwh) and cost_per_kwh < 5 and net_energy_kwh_year > 300:
    roi_kind, roi_reason = "MIXED", "ROI devient moins extrême, mais reste rarement compétitif vs réseau."
else:
    roi_kind, roi_reason = "NO-GO", "coût/kWh très élevé vs production (energy harvesting généralement modeste)."

# Pedagogy / engagement
if inp["pct_on_zone"] < 1.0 or inp["useful_steps"] < 10:
    ped_kind, ped_reason = "MIXED", "zone trop peu traversée → revoir emplacement ou surface."
else:
    ped_kind, ped_reason = "GO", "valeur forte d’engagement : rendre l’énergie tangible + micro-usages locaux."

key_driver = "La réalité dépend surtout de **% sur zone** et **pas utiles** (emplacement + design du parcours)."


# =========================
# Results tab
# =========================
with tab_results:
    st.subheader("Results (visual + actionable)")

    # Executive summary (top)
    with st.container(border=True):
        st.markdown("### Executive summary (résumé décisionnel)")

        energy_label = f"{fmt_num(net_energy_kwh_year, 2)} kWh/an"
        if net_energy_kwh_year < 10:
            energy_label += " (très faible)"
        elif net_energy_kwh_year < 100:
            energy_label += " (modeste)"

        usage_realiste = "LEDs / capteurs / petit affichage (micro-usages locaux)"

        st.write(f"**Énergie nette estimée** : {energy_label}")
        st.write(f"**Usage réaliste** : {usage_realiste}")
        st.write(
            f"**Coût total** : {fmt_money(capex)}$ CAPEX + {fmt_money(opex_year)}$/an OPEX → "
            f"~ **{fmt_num(cost_per_kwh, 2)} $/kWh**"
        )
        st.caption(f"Note: énergie brute ~ {fmt_num(gross_energy_wh_day,2)} Wh/j — auto-conso {fmt_num(inp['auto_consumption_wh_day'],2)} Wh/j → nette {fmt_num(net_energy_wh_day,2)} Wh/j.")

        roi_text, roi_style = verdict_badge(roi_kind, roi_reason)
        ped_text, ped_style = verdict_badge(ped_kind, ped_reason)

        if roi_style == "success":
            st.success(roi_text)
        elif roi_style == "warning":
            st.warning(roi_text)
        else:
            st.error(roi_text)

        if ped_style == "success":
            st.success(ped_text)
        elif ped_style == "warning":
            st.warning(ped_text)
        else:
            st.info(ped_text)

        st.info(key_driver)

    st.markdown("---")

    # Metrics: consistent units order
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Wh / jour (net)", fmt_num(net_energy_wh_day, 2))
    m2.metric("kWh / jour (net)", fmt_num(net_energy_kwh_day, 5))
    m3.metric("kWh / mois (~30j)", fmt_num(net_energy_kwh_month, 3))
    m4.metric("kWh / an (~365j)", fmt_num(net_energy_kwh_year, 2))

    # Scenarios chart
    st.markdown("### Uncertainty scenarios (visual)")
    scen_plot = df_scen.copy()
    scen_plot["scenario"] = pd.Categorical(scen_plot["scenario"], categories=["low", "mid", "high"], ordered=True)
    scen_plot = scen_plot.sort_values("scenario").set_index("scenario")[["Wh/day"]]
    st.bar_chart(scen_plot)
    st.caption("🧠 Interprétation : l’incertitude vient surtout de **% sur zone** et **pas utiles** (emplacement + parcours).")

    st.markdown("---")

    # Equivalences (per day)
    st.markdown("### What can it power (per day)")

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
        "La valeur forte est souvent **pédagogique/engagement** (rendre l’énergie tangible), "
        "plus des **micro-usages locaux** (LEDs, capteurs, petit affichage)."
    )

    st.markdown("---")

    # Costs
    st.markdown("### Costs (CAPEX/OPEX) + cost per kWh (rough)")
    c1, c2, c3 = st.columns(3)
    c1.metric("CAPEX ($)", fmt_money(capex))
    c2.metric("OPEX / an ($)", fmt_money(opex_year))
    c3.metric("Coût approx ($/kWh)", fmt_num(cost_per_kwh, 2) if np.isfinite(cost_per_kwh) else "∞")

    st.caption("⚠️ Les coûts (CAPEX/OPEX) ne sont pas des constantes scientifiques : ils dépendent des devis/projets.")

    st.markdown("---")

    # Export
    st.markdown("### Export")
    export = {
        "place_type": inp["place_type"],
        "visitors_per_day": inp["visitors_per_day"],
        "peak_multiplier": inp["peak_multiplier"],
        "pct_on_zone": inp["pct_on_zone"],
        "useful_steps": inp["useful_steps"],
        "J_net_per_step": inp["J_net_per_step"],
        "auto_consumption_wh_day": inp["auto_consumption_wh_day"],
        "area_ft2": inp["area_ft2"],
        "tile_area_ft2": inp["tile_area_ft2"],
        "installed_cost_per_ft2": inp["installed_cost_per_ft2"],
        "fixed_cost": inp["fixed_cost"],
        "maintenance_pct": inp["maintenance_pct"],
        "amort_years": inp["amort_years"],
        "steps_captured_per_day": steps_captured,
        "gross_energy_Wh_day": gross_energy_wh_day,
        "net_energy_Wh_day": net_energy_wh_day,
        "net_energy_kWh_day": net_energy_kwh_day,
        "net_energy_kWh_year": net_energy_kwh_year,
        "capex_$": capex,
        "opex_year_$": opex_year,
        "cost_per_kWh_$": cost_per_kwh,
        "roi_verdict": f"{roi_kind}: {roi_reason}",
        "pedago_verdict": f"{ped_kind}: {ped_reason}",
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
# Methodology tab (anti-greenwashing + references)
# =========================
with tab_methods:
    st.subheader("Methodology / Limits (anti-greenwashing)")

    st.markdown("### Core formula (math-consistent)")
    st.code(
        "Net Energy (Wh/day) = visitors/day × peak_multiplier × (%on_zone/100) × useful_steps × J_net_per_step ÷ 3600  −  auto_consumption_Wh_day",
        language="text"
    )

    st.markdown("### What this is NOT")
    st.markdown(
        """
- ❌ **Not powering a building** (outputs are usually modest).
- ❌ **Not a climate solution alone** (main value is often educational + micro-local loads).
- ❌ **Not a quote**: CAPEX/OPEX numbers are **project-dependent** (vendor quotes/site constraints).
- ✅ **What it is**: a **decision-support** tool to avoid over-installation and size for realistic micro-uses.
"""
    )

    st.markdown("### Why outputs are usually modest (anti-greenwashing framing)")
    st.markdown(
        """
Even with optimistic footstep energy, the number of *captured useful steps* is the main limiter (placement + flow).
At low daily energy, **auto-consumption** can dominate — which is why we expose it explicitly.
"""
    )

    st.markdown("### References (academic) & how they are used")
    st.markdown(
        """
**Unit definitions & math consistency (J, W, Wh)**  
- **BIPM – SI Brochure**: SI unit definitions (J, W, dimensional relations). Used to justify conversions and the division by **3600** (1 h = 3600 s).

**Energy per footstep for kinetic floor tiles (core input: J_net/pas)**  
- **Asadi et al. (2023)**: reports ~**511 mJ/step** (electromagnetic tile). Used as a realistic default anchor for **J_net/step**.  
- **Jintanawan et al. (2020)**: prototype up to ~**702 mJ/step** (“Genpath”). Used as a realistic upper bound for strong prototypes.  
- **Thainiramit et al. (2022)**: triboelectric tile with energy on the order of **mJ**. Used as a realistic lower bound and reminder that tech/load/frequency matter.

**Why outputs are usually modest (anti-greenwashing framing)**  
- **Paradiso & Starner (2005)**: classic energy scavenging review. Used to frame practical limitations and typical orders of magnitude.  
- **Mitcheson et al. (2008)**: Proceedings of the IEEE review on motion energy harvesting. Used to contextualize variability and key factors (architecture, frequency, load).

**Sanity checks for pedestrian motion (coherence warnings)**  
- **Weidmann (1993)**: typical free walking speed (~**1.34 m/s**). Used for plausibility checks.  
- **Pachi & Ji (2005)**: observed step frequency around ~**2 Hz** (in real environments). Used to derive typical step length (v/f) for warnings.

**Power electronics (why we expose auto-consumption explicitly)**  
- Academic MPPT/rectification examples for piezo/harvesting show conditioning is possible, but **system power budget** must be explicit — hence the auto-consumption field.
"""
    )

    st.markdown("### Bibliographie (liens copiable)")
    st.code(
        """Asadi, M., Ahmadi, R., & Abazari, A. M. (2023). Footstep-powered floor tile: Design and evaluation of an electromagnetic frequency up-converted energy harvesting system enhanced by a cylindrical Halbach array. Sustainable Energy Technologies and Assessments, 60, 103571. https://doi.org/10.1016/j.seta.2023.103571

Jintanawan, T., et al. (2020). Design of Kinetic-Energy Harvesting Floors. Energies, 13(20), 5419. https://www.mdpi.com/1996-1073/13/20/5419

Thainiramit, P., et al. (2022). Triboelectric Energy-Harvesting Floor Tile. Materials, 15(24), 8853. https://www.mdpi.com/1996-1944/15/24/8853

Paradiso, J. A., & Starner, T. (2005). Energy Scavenging for Mobile and Wireless Electronics. IEEE Pervasive Computing, 4(1), 18–27. https://doi.org/10.1109/MPRV.2005.9

Mitcheson, P. D., Yeatman, E. M., Rao, G. K., Holmes, A. S., & Green, T. C. (2008). Energy Harvesting From Human and Machine Motion for Wireless Electronic Devices. Proceedings of the IEEE, 96(9), 1457–1486. https://doi.org/10.1109/JPROC.2008.927494

Weidmann, U. (1993). Transporttechnik der Fussgänger. Institut für Verkehrsplanung, Transporttechnik, Strassen- und Eisenbahnbau (IVT), ETH Zürich. https://www.ped-net.org/uploads/media/weidmann-1993_01.pdf

Pachi, A., & Ji, T. (2005). Frequency and velocity of people walking. The Structural Engineer, 83(3). https://trid.trb.org/View/750847

BIPM. (2019). The International System of Units (SI Brochure, 9th ed.). https://www.bipm.org/en/publications/si-brochure""",
        language="text"
    )

    st.markdown("### Limits / caveats")
    st.markdown(
        """
- Results are sensitive to **% on zone** and **useful steps**: location + pathway design matter most.
- **J_net/step** depends on technology, mechanical design, frequency, electrical load, and test conditions.
- CAPEX/OPEX are **project-dependent** (quotes/site constraints), not “scientific constants”.
- No personal data: use aggregated visitor counts only.
"""
    )
