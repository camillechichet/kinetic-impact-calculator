import io
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st


# =========================================================
# Page config
# =========================================================
st.set_page_config(page_title="Kinetic Impact Calculator", page_icon="⚡", layout="wide")


# =========================================================
# Glossary + help system
# =========================================================
GLOSSARY = {
    "CAPEX": "Capital Expenditure: coût initial (matériel + installation + travaux fixes).",
    "OPEX": "Operational Expenditure: coût annuel d’exploitation (maintenance, etc.).",
    "Amortissement": "Période (années) sur laquelle on répartit le CAPEX pour estimer un coût annuel/total.",
    "Coût/kWh (rough)": "Indicateur approximatif: compare un projet à un autre (pas un devis). Très sensible aux hypothèses.",
    "J_net/pas": (
        "Énergie électrique nette récupérée par pas (déjà 'net', mesurée en sortie électrique d’un système). "
        "Cela évite de compter deux fois des rendements/pertes."
    ),
    "% sur zone": "Pourcentage des visiteurs qui passent réellement sur la zone équipée (placement = clé).",
    "Pas utiles": (
        "Nombre de pas 'captés' sur la zone par visiteur (dépend de la longueur du passage, du design, de la densité)."
    ),
    "Incertitude": "On affiche une plage (bas/moyen/haut) car % sur zone, pas utiles, et J_net varient beaucoup.",
    "Dataset": "Historique (CSV: date, visitors) utilisé pour faire une prévision légère (trend + saisonnalité).",
    "Horizon": "Nombre de jours prévus dans le futur par le module de prévision.",
    "Auto-consommation": (
        "Consommation propre du système (électronique, communication, LED témoin, etc.) en Wh/jour. "
        "À faible énergie, elle peut annuler le gain."
    ),
    "Zone équipée": "Surface totale couverte par des dalles (ft²).",
    "Vitesse / cadence": "Ordres de grandeur utiles pour vérifier la cohérence des pas vs la géométrie (sanity check).",
    "Unités (J, Wh, kWh)": "1 Wh = 3600 J. On convertit J→Wh en divisant par 3600.",
}

SOURCES = {
    "J_net/pas": {
        "title": "Sources pour J_net/pas (exemples académiques)",
        "links": [
            ("Asadi et al. (2023) ~511 mJ/step", "https://doi.org/10.1016/j.seta.2023.103571"),
            ("Jintanawan et al. (2020) jusqu’à ~702 mJ/step", "https://www.mdpi.com/1996-1073/13/20/5419"),
            ("Thainiramit et al. (2022) tribo ~mJ", "https://www.mdpi.com/1996-1944/15/24/8853"),
        ],
        "note": "Les valeurs dépendent de la techno, de la charge électrique, de la fréquence et des conditions de test."
    },
    "Vitesse / cadence": {
        "title": "Sources vitesse / cadence (sanity checks)",
        "links": [
            ("Weidmann (1993) vitesse libre ~1.34 m/s", "https://www.ped-net.org/uploads/media/weidmann-1993_01.pdf"),
            ("Pachi & Ji (2005) cadence ~2 Hz (observations)", "https://trid.trb.org/View/750847"),
        ],
        "note": "On utilise ces ordres de grandeur uniquement pour détecter des saisies très improbables."
    },
    "Unités (J, Wh, kWh)": {
        "title": "Source unités SI",
        "links": [
            ("BIPM SI Brochure (2019)", "https://www.bipm.org/en/publications/si-brochure"),
        ],
        "note": "Justifie la cohérence dimensionnelle et la conversion 1 h = 3600 s."
    },
}


def glossary_ui():
    """Global glossary with search."""
    st.markdown("### 📘 Glossaire")
    q = st.text_input("Rechercher dans le glossaire", placeholder="Ex: CAPEX, J_net/pas, incertitude…")
    items = list(GLOSSARY.items())
    if q:
        ql = q.lower()
        items = [(k, v) for k, v in items if ql in k.lower() or ql in v.lower()]

    if not items:
        st.info("Aucun résultat.")
        return

    for term, definition in items:
        with st.expander(term):
            st.write(definition)
            if term in SOURCES:
                st.markdown("**Voir la source**")
                for label, url in SOURCES[term]["links"]:
                    st.link_button(label, url)
                st.caption(SOURCES[term].get("note", ""))


def try_popover(label: str):
    """
    Streamlit a st.popover sur des versions récentes.
    Si indisponible, on fallback sur un expander.
    """
    if hasattr(st, "popover"):
        return st.popover(label)
    return st.expander(label)


def help_tag(term: str) -> str:
    """Help text includes pointer to glossary."""
    base = GLOSSARY.get(term, "")
    if base:
        return f"{base}\n\n📘 Voir dans le glossaire: {term}"
    return f"📘 Voir dans le glossaire: {term}"


def badge_realism(value: float, ok_range: tuple[float, float], warn_range: tuple[float, float]) -> str:
    if ok_range[0] <= value <= ok_range[1]:
        return "✅ plausible"
    if warn_range[0] <= value <= warn_range[1]:
        return "⚠️ optimiste"
    return "🚩 très improbable"


def fmt_money(x: float) -> str:
    return f"{x:,.0f}".replace(",", " ")


def safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else float("nan")


def ft2_to_m2(x_ft2: float) -> float:
    return x_ft2 * 0.092903


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
    """
    IA frugale:
    - trend: fit linéaire numpy.polyfit
    - saisonnalité: correction par moyenne des résidus par jour de semaine
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
# Presets + example scenarios (explicit)
# =========================================================
PRESETS = {
    "Musée": {
        "desc": "Flux modéré, parcours plus lent → % sur zone moyen, pas utiles moyens.",
        "uncertainty": "Moyenne",
        "values": {
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
    },
    "Gare": {
        "desc": "Flux fort, passages répétitifs → % sur zone plus élevé, pas utiles élevés.",
        "uncertainty": "Élevée",
        "values": {
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
    },
    "Stade": {
        "desc": "Très gros pics (événements) → multiplier pic important.",
        "uncertainty": "Élevée",
        "values": {
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
    },
    "Centre commercial": {
        "desc": "Flux continu, variations week-end → pic modéré, pas utiles moyens/élevés.",
        "uncertainty": "Moyenne",
        "values": {
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


# =========================================================
# Session defaults
# =========================================================
DEFAULTS = {
    "mode": "Débutant",
    "place_type": "Musée",

    # Key drivers
    "visitors_per_day": 3300,
    "peak_multiplier": 1.0,
    "pct_on_zone": 12.0,
    "useful_steps": 115.0,
    "J_net_per_step": 0.5,
    "auto_consumption_wh_day": 0.0,

    # Data quality tags
    "dq_visitors": "Estimé",
    "dq_pct_on_zone": "Très incertain",
    "dq_useful_steps": "Très incertain",
    "dq_J_net": "Très incertain",

    # sizing + costs
    "area_ft2": 190.0,
    "tile_area_ft2": 1.10,
    "installed_cost_per_ft2": 175.0,
    "fixed_cost": 20000.0,
    "maintenance_pct": 8.0,
    "amort_years": 9,

    # forecast (advanced)
    "use_demo_dataset": True,
    "forecast_horizon_days": 14,

    # guided tour
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
# Header + global controls
# =========================================================
st.title("Kinetic Impact Calculator")
st.caption("Decision-support MVP: énergie (net), usages concrets, coûts, scénarios d’incertitude, et prévision frugale.")

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
    # Glossary button + search
    with try_popover("📘 Glossaire"):
        glossary_ui()


# Mode switch + Quick start guided tour
mode_col, tour_col, _ = st.columns([1.2, 1.2, 1.6])
with mode_col:
    mode = st.radio("Mode", ["Débutant", "Avancé"], horizontal=True, index=0 if inp["mode"] == "Débutant" else 1)
    inp["mode"] = mode

with tour_col:
    inp["tour_on"] = st.toggle("Quick start (tour 5 étapes)", value=bool(inp["tour_on"]))
    if inp["tour_on"]:
        st.caption("Étapes: 1) lieu → 2) visiteurs → 3) % zone → 4) pas utiles → 5) résultats")


tab_inputs, tab_results, tab_methods = st.tabs(["Inputs", "Results", "Methodology / Limits"])


# =========================================================
# Inputs tab
# =========================================================
with tab_inputs:
    c1, c2, c3 = st.columns([1.1, 1.0, 1.0], gap="large")

    # ---------- Guided tour step gating ----------
    tour_step = int(inp.get("tour_step", 1))
    if not inp["tour_on"]:
        tour_step = 999  # show all

    def tour_controls():
        if not inp["tour_on"]:
            return
        b1, b2, b3 = st.columns([1, 1, 2])
        with b1:
            if st.button("⬅️ Précédent", disabled=(tour_step <= 1)):
                inp["tour_step"] = max(1, tour_step - 1)
                st.rerun()
        with b2:
            if st.button("Suivant ➡️", disabled=(tour_step >= 5)):
                inp["tour_step"] = min(5, tour_step + 1)
                st.rerun()
        with b3:
            st.progress(tour_step / 5)
            st.write(f"Étape {tour_step}/5")

    tour_controls()

    # ---------- Column 1: Context + Flow ----------
    with c1:
        st.subheader("Context")

        # Preset selection with description + uncertainty
        place_type = st.selectbox(
            "Type de lieu",
            options=list(PRESETS.keys()),
            index=list(PRESETS.keys()).index(inp["place_type"]) if inp["place_type"] in PRESETS else 0,
            help=help_tag("Dataset") + "\n\n(Le preset ne charge pas un dataset; il pré-remplit des valeurs typiques.)",
        )
        inp["place_type"] = place_type

        preset_desc = PRESETS[place_type]["desc"]
        preset_unc = PRESETS[place_type]["uncertainty"]
        st.caption(f"Preset: {preset_desc}  •  Incertitude: **{preset_unc}**")

        if (tour_step >= 1) and st.button("Appliquer preset du lieu"):
            p = PRESETS.get(place_type, {}).get("values", {})
            for k, v in p.items():
                inp[k] = v
            st.success(f"Preset appliqué pour: {place_type}")
            st.rerun()

        # Step 1 in tour: stop here
        if inp["tour_on"] and tour_step == 1:
            st.info("✅ Étape 1 : choisis un lieu + applique un preset. Puis clique 'Suivant'.")
            tour_controls()
            st.stop()

        st.markdown("---")
        st.subheader("Key drivers (ce qui change le plus le résultat)")

        visitors_per_day = st.number_input(
            "Visiteurs / jour (moyenne)",
            min_value=0,
            value=int(inp["visitors_per_day"]),
            step=50,
            help=help_tag("Dataset"),
        )
        dq_visitors = st.selectbox("Qualité donnée (visiteurs)", ["Mesuré", "Estimé", "Très incertain"],
                                  index=["Mesuré", "Estimé", "Très incertain"].index(inp["dq_visitors"]),
                                  help="Badge confiance pour expliquer d’où vient la valeur.")
        inp["dq_visitors"] = dq_visitors
        st.write(f"🔎 Confiance: **{dq_visitors}**  •  {badge_realism(float(visitors_per_day), (300, 50000), (50, 120000))}")

        peak_multiplier = st.slider(
            "Multiplicateur pic",
            1.0, 5.0, float(inp["peak_multiplier"]), 0.05,
            help="Ex: 1.0 (normal), 1.2–1.8 (week-end), 2–3 (événement).",
        )

        st.markdown("---")
        st.subheader("Flow on equipped zone")

        pct_on_zone = st.slider(
            "% visiteurs sur zone",
            0.0, 100.0, float(inp["pct_on_zone"]), 0.5,
            help=help_tag("% sur zone"),
        )
        dq_pct = st.selectbox("Qualité donnée (% sur zone)", ["Mesuré", "Estimé", "Très incertain"],
                              index=["Mesuré", "Estimé", "Très incertain"].index(inp["dq_pct_on_zone"]))
        inp["dq_pct_on_zone"] = dq_pct
        st.caption("Ex: 2–10% (zone petite) / 10–30% (zone centrale)")
        st.write(f"🔎 Confiance: **{dq_pct}**  •  {badge_realism(pct_on_zone, (2, 30), (0.5, 60))}")

        useful_steps = st.slider(
            "Pas utiles / visiteur",
            0.0, 300.0, float(inp["useful_steps"]), 5.0,
            help=help_tag("Pas utiles"),
        )
        dq_steps = st.selectbox("Qualité donnée (pas utiles)", ["Mesuré", "Estimé", "Très incertain"],
                                index=["Mesuré", "Estimé", "Très incertain"].index(inp["dq_useful_steps"]))
        inp["dq_useful_steps"] = dq_steps
        st.caption("Ex: 20–60 (petit couloir) / 80–200 (long passage)")
        st.write(f"🔎 Confiance: **{dq_steps}**  •  {badge_realism(useful_steps, (20, 200), (5, 300))}")

        # Step gating for tour: visitors then pct then steps
        if inp["tour_on"] and tour_step in (2, 3, 4):
            if tour_step == 2:
                st.info("✅ Étape 2 : ajuste visiteurs/jour (+ pic si besoin). Puis 'Suivant'.")
                tour_controls()
                st.stop()
            if tour_step == 3:
                st.info("✅ Étape 3 : ajuste % sur zone. Puis 'Suivant'.")
                tour_controls()
                st.stop()
            if tour_step == 4:
                st.info("✅ Étape 4 : ajuste pas utiles. Puis 'Suivant'.")
                tour_controls()
                st.stop()

        inp["visitors_per_day"] = int(visitors_per_day)
        inp["peak_multiplier"] = float(peak_multiplier)
        inp["pct_on_zone"] = float(pct_on_zone)
        inp["useful_steps"] = float(useful_steps)

    # ---------- Column 2: Technical + sizing ----------
    with c2:
        st.subheader("Technical assumptions")

        # Beginner shows only key drivers. Advanced can see sizing too (still useful in beginner though).
        J_net_per_step = st.slider(
            "J_net/pas",
            0.005, 1.0, float(inp["J_net_per_step"]), 0.005,
            help=help_tag("J_net/pas"),
        )
        dq_jnet = st.selectbox("Qualité donnée (J_net/pas)", ["Mesuré", "Estimé", "Très incertain"],
                               index=["Mesuré", "Estimé", "Très incertain"].index(inp["dq_J_net"]))
        inp["dq_J_net"] = dq_jnet

        # quick access to sources (trust & transparency)
        with try_popover("ⓘ Voir la source (J_net/pas)"):
            st.write(SOURCES["J_net/pas"]["title"])
            for label, url in SOURCES["J_net/pas"]["links"]:
                st.link_button(label, url)
            st.caption(SOURCES["J_net/pas"]["note"])

        auto_consumption_wh_day = st.number_input(
            "Auto-consommation (Wh/jour)",
            min_value=0.0,
            value=float(inp["auto_consumption_wh_day"]),
            step=1.0,
            help=help_tag("Auto-consommation"),
        )

        inp["J_net_per_step"] = float(J_net_per_step)
        inp["auto_consumption_wh_day"] = float(auto_consumption_wh_day)

        st.caption(f"🔎 Confiance: **{dq_jnet}**  •  {badge_realism(J_net_per_step, (0.05, 0.8), (0.01, 1.0))}")

        st.markdown("---")
        st.subheader("Installation sizing")

        area_ft2 = st.number_input(
            "Zone équipée (ft²)",
            min_value=1.0,
            value=float(inp["area_ft2"]),
            step=10.0,
            help=help_tag("Zone équipée"),
        )
        tile_area_ft2 = st.number_input(
            "Surface d’une dalle (ft²)",
            min_value=0.2,
            value=float(inp["tile_area_ft2"]),
            step=0.05,
        )
        est_tiles = int(round(area_ft2 / tile_area_ft2))
        st.info(f"≈ **{est_tiles} dalles** pour {area_ft2:.0f} ft² (si 1 dalle ≈ {tile_area_ft2:.2f} ft²)")

        inp["area_ft2"] = float(area_ft2)
        inp["tile_area_ft2"] = float(tile_area_ft2)

        # Intelligent warning + action
        area_m2 = ft2_to_m2(area_ft2)
        approx_length_m = max(0.5, float(np.sqrt(area_m2)))
        v_free = 1.34
        f_step = 2.0
        step_len = v_free / f_step
        plausible_upper_steps = 2.5 * (approx_length_m / step_len)

        if useful_steps > plausible_upper_steps and useful_steps > 30:
            st.warning(
                f"⚠️ Pas/visiteur élevé vs zone (~{area_ft2:.0f} ft²). "
                f"Risque de surestimation. (Longueur typique ~{approx_length_m:.1f} m)"
            )
            a1, a2 = st.columns([1, 1])
            with a1:
                if st.button("Ajuster à une valeur typique"):
                    # Typical fallback by place
                    typical = {
                        "Musée": 60.0,
                        "Gare": 120.0,
                        "Stade": 80.0,
                        "Centre commercial": 90.0
                    }.get(inp["place_type"], 80.0)
                    inp["useful_steps"] = typical
                    st.success(f"Pas utiles réglés à {typical:.0f}.")
                    st.rerun()
            with a2:
                with st.expander("Pourquoi ?"):
                    st.write(
                        "On compare grossièrement tes pas utiles à ce qu’une traversée plausible de la zone "
                        "permettrait (ordre de grandeur via vitesse libre ~1.34 m/s et cadence ~2 Hz). "
                        "Ce n’est pas une vérité, juste un garde-fou anti-surestimation."
                    )
                    st.markdown("**Voir la source**")
                    st.link_button("Weidmann (1993)", SOURCES["Vitesse / cadence"]["links"][0][1])
                    st.link_button("Pachi & Ji (2005)", SOURCES["Vitesse / cadence"]["links"][1][1])

    # ---------- Column 3: Costs + (advanced) forecast/export ----------
    with c3:
        st.subheader("Costs")

        installed_cost_per_ft2 = st.slider(
            "CAPEX $/ft²",
            50.0, 900.0, float(inp["installed_cost_per_ft2"]), 5.0,
            help=help_tag("CAPEX"),
        )
        fixed_cost = st.number_input(
            "Coût fixe (travaux/élec/signalétique) $",
            min_value=0.0,
            value=float(inp["fixed_cost"]),
            step=1000.0,
            help=help_tag("CAPEX"),
        )
        maintenance_pct = st.slider(
            "OPEX maintenance (% du CAPEX)",
            0.0, 20.0, float(inp["maintenance_pct"]), 0.5,
            help=help_tag("OPEX"),
        )
        amort_years = st.slider(
            "Amortissement (années)",
            1, 20, int(inp["amort_years"]), 1,
            help=help_tag("Amortissement"),
        )

        inp["installed_cost_per_ft2"] = float(installed_cost_per_ft2)
        inp["fixed_cost"] = float(fixed_cost)
        inp["maintenance_pct"] = float(maintenance_pct)
        inp["amort_years"] = int(amort_years)

        if inp["mode"] == "Avancé":
            st.markdown("---")
            st.subheader("Sustainable AI (lightweight)")

            use_demo = st.checkbox("Utiliser dataset démo", value=bool(inp["use_demo_dataset"]), help=help_tag("Dataset"))
            uploaded = st.file_uploader("Upload CSV (date, visitors)", type=["csv"], help=help_tag("Dataset"))
            horizon = st.slider("Horizon (jours)", 7, 60, int(inp["forecast_horizon_days"]), 1, help=help_tag("Horizon"))
            inp["use_demo_dataset"] = bool(use_demo)
            inp["forecast_horizon_days"] = int(horizon)

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


# =========================================================
# Compute results (shared)
# =========================================================
steps_captured = (
    inp["visitors_per_day"]
    * inp["peak_multiplier"]
    * (inp["pct_on_zone"] / 100.0)
    * inp["useful_steps"]
)

gross_energy_wh_day = steps_captured * inp["J_net_per_step"] / 3600.0
net_energy_wh_day = max(0.0, gross_energy_wh_day - inp["auto_consumption_wh_day"])

net_kwh_day = net_energy_wh_day / 1000.0
net_wh_month = net_energy_wh_day * 30.0
net_kwh_year = net_kwh_day * 365.0

capex = inp["area_ft2"] * inp["installed_cost_per_ft2"] + inp["fixed_cost"]
opex_year = (inp["maintenance_pct"] / 100.0) * capex
N = inp["amort_years"]
total_cost_N = capex + opex_year * N
cost_per_kwh = safe_div(total_cost_N, net_kwh_year * N) if net_kwh_year > 0 else float("inf")

# Uncertainty scenarios (simple multipliers)
scenarios = {"low": 0.6, "mid": 1.0, "high": 1.4}
df_scen = pd.DataFrame([{"scenario": k, "Wh/day": net_energy_wh_day * v} for k, v in scenarios.items()])
df_scen["scenario"] = pd.Categorical(df_scen["scenario"], categories=["low", "mid", "high"], ordered=True)
df_scen = df_scen.sort_values("scenario").set_index("scenario")

# Dominant parameters box
dominants = [
    ("1) % sur zone", inp["pct_on_zone"]),
    ("2) Pas utiles", inp["useful_steps"]),
    ("3) J_net/pas", inp["J_net_per_step"]),
]

# Equivalences per day (very simple)
led10w_hours = safe_div(net_energy_wh_day, 10.0)
lowpower_sensor_days = safe_div(net_energy_wh_day, 2.0)      # 2Wh/day device budget
small_screen_minutes = safe_div(net_energy_wh_day, 15.0) * 60 # 15W small screen
phone_charges = safe_div(net_energy_wh_day, 12.0)


# Verdicts split
def verdict_energy_roi():
    if net_kwh_year <= 0:
        return "NO-GO", "énergie nette ~0 après auto-consommation."
    if np.isfinite(cost_per_kwh) and cost_per_kwh < 5 and net_kwh_year > 300:
        return "MIXED", "moins extrême, mais rarement compétitif vs réseau."
    return "NO-GO", "coût/kWh très élevé vs production (harvesting généralement modeste)."


def verdict_pedagogy():
    if inp["pct_on_zone"] < 1.0 or inp["useful_steps"] < 10:
        return "MIXED", "zone trop peu traversée → revoir emplacement/surface."
    return "GO", "bon pour engagement: rendre l’énergie tangible + micro-usages locaux."


roi_kind, roi_reason = verdict_energy_roi()
ped_kind, ped_reason = verdict_pedagogy()


def show_verdict(kind: str, reason: str):
    if kind == "GO":
        st.success(f"✅ GO — {reason}")
    elif kind == "MIXED":
        st.warning(f"⚠️ MIXTE — {reason}")
    else:
        st.error(f"⛔ NO-GO — {reason}")


# =========================================================
# Results tab
# =========================================================
with tab_results:
    st.subheader("Results")

    # Mini 'what model is NOT' visible (trust)
    with st.container(border=True):
        st.markdown("**Ce que ce modèle ne fait pas**")
        st.markdown("- ❌ Pas un devis (CAPEX/OPEX varient selon projets)\n- ❌ Pas 'alimenter un bâtiment'\n- ❌ Pas une solution climat seule (valeur surtout pédagogique)")

    # Executive summary
    with st.container(border=True):
        st.markdown("### Executive summary (actionnable)")
        st.write(f"**Énergie nette**: **{net_energy_wh_day:.2f} Wh/jour**  •  {net_wh_month:.1f} Wh/mois  •  {net_kwh_year:.2f} kWh/an")
        st.caption("Phrase clé: c’est généralement modeste — l’intérêt principal est souvent l’engagement + micro-usages.")
        st.write(f"**Coût total**: {fmt_money(capex)}$ CAPEX + {fmt_money(opex_year)}$/an OPEX → **{cost_per_kwh:,.2f} $/kWh**".replace(",", " "))
        st.caption("Coût/kWh (rough): à utiliser pour comparer des scénarios, pas comme un devis.")
        st.markdown("**Verdicts (séparés)**")
        show_verdict(roi_kind, roi_reason)
        show_verdict(ped_kind, ped_reason)

    # Dominant parameters
    with st.container(border=True):
        st.markdown("### Ce qui change le plus ton résultat")
        st.write("👉 **1) % sur zone  2) pas utiles  3) J_net/pas** (et ensuite visiteurs/jour).")

    # Energy views (consistent)
    m1, m2, m3 = st.columns(3)
    m1.metric("Wh / jour (principal)", f"{net_energy_wh_day:.2f}")
    m2.metric("Wh / mois (~30j)", f"{net_wh_month:.1f}")
    m3.metric("kWh / an (~365j)", f"{net_kwh_year:.2f}")

    # Scenarios chart
    st.markdown("### Incertitude (scénarios)")
    st.bar_chart(df_scen[["Wh/day"]])
    st.caption("La réalité dépend surtout de **% sur zone** et **pas utiles** (placement + parcours).")

    # What can it power (more concrete)
    st.markdown("### What can it power (par jour)")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("LED 10W (heures)", f"{led10w_hours:.2f}")
    e2.metric("Capteur low-power 2Wh/j (jours)", f"{lowpower_sensor_days:.2f}")
    e3.metric("Petit écran ~15W (minutes)", f"{small_screen_minutes:.1f}")
    e4.metric("Charges téléphone (~12Wh)", f"{phone_charges:.2f}")

    st.info("Rappel: l’énergie est souvent **modeste**. Valeur forte: rendre l’énergie visible + alimenter des micro-usages locaux.")

    # Costs block + explainers
    st.markdown("### Costs")
    c1, c2, c3 = st.columns(3)
    c1.metric("CAPEX ($)", fmt_money(capex))
    c2.metric("OPEX/an ($)", fmt_money(opex_year))
    c3.metric("Coût/kWh (rough)", f"{cost_per_kwh:.2f}" if np.isfinite(cost_per_kwh) else "∞")

    with try_popover("ⓘ Expliquer coût/kWh (rough)"):
        st.write(GLOSSARY["Coût/kWh (rough)"])
        st.caption("Il explose si la production est très faible — c’est normal sur du harvesting piéton.")
        st.caption("Astuce: utilise-le pour comparer des scénarios (emplacement A vs B) plutôt que comme 'prix absolu'.")

    # Export
    if inp["mode"] == "Avancé":
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
            "net_energy_kWh_year": net_kwh_year,
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

    # Tour step 5
    if inp["tour_on"] and inp["tour_step"] == 5:
        st.success("✅ Étape 5 : tu es sur Results. Tu peux maintenant affiner % zone / pas utiles / J_net/pas.")
        st.stop()


# =========================================================
# Methodology tab
# =========================================================
with tab_methods:
    st.subheader("Methodology / Limits")

    # Direct "sources" buttons for trust
    cols = st.columns(3)
    with cols[0]:
        with try_popover("🔎 Voir la source — J_net/pas"):
            st.write(SOURCES["J_net/pas"]["title"])
            for label, url in SOURCES["J_net/pas"]["links"]:
                st.link_button(label, url)
            st.caption(SOURCES["J_net/pas"]["note"])

    with cols[1]:
        with try_popover("🔎 Voir la source — Vitesse/cadence"):
            st.write(SOURCES["Vitesse / cadence"]["title"])
            for label, url in SOURCES["Vitesse / cadence"]["links"]:
                st.link_button(label, url)
            st.caption(SOURCES["Vitesse / cadence"]["note"])

    with cols[2]:
        with try_popover("🔎 Voir la source — Unités"):
            st.write(SOURCES["Unités (J, Wh, kWh)"]["title"])
            for label, url in SOURCES["Unités (J, Wh, kWh)"]["links"]:
                st.link_button(label, url)
            st.caption(SOURCES["Unités (J, Wh, kWh)"]["note"])

    st.markdown("### Core formula (transparent)")
    with st.expander("Math (expand)"):
        st.code(
            "Net Energy (Wh/day) = visitors/day × peak_multiplier × (%on_zone/100) × useful_steps × J_net_per_step ÷ 3600  −  auto_consumption_Wh_day",
            language="text",
        )
        st.caption("1 Wh = 3600 J → division par 3600 pour convertir J → Wh.")

    st.markdown("### What this is NOT")
    st.markdown(
        """
- ❌ Not powering a building (outputs are usually modest).
- ❌ Not a climate solution alone (main value is educational + micro-local loads).
- ❌ Not a quote: CAPEX/OPEX are project-dependent.
"""
    )

    st.markdown("### Limits (anti-greenwashing)")
    st.markdown(
        """
- Les sorties dépendent surtout de **% sur zone** et **pas utiles** (placement + parcours).
- **J_net/pas** varie énormément selon techno, charge, fréquence, et conditions de test.
- À faible énergie, l’**auto-consommation** peut annuler le gain → d’où le champ explicite.
- Aucun tracking perso: on utilise des volumes agrégés.
"""
    )

    st.markdown("### Note sur les coûts")
    st.info("Les coûts (CAPEX/OPEX) ne sont pas des constantes scientifiques : ils dépendent des devis/projets.")


# =========================================================
# Beginner-mode hiding (soft)
# =========================================================
# (No extra needed: we already hid forecast/export in beginner.)
