import io
import math
import pandas as pd
import streamlit as st

from sklearn.linear_model import LinearRegression
from codecarbon import EmissionsTracker

st.set_page_config(page_title="Kinetic Impact Calculator", layout="wide")
st.title("Kinetic Impact Calculator (inspired by Coldplay)")
st.caption("MVP : énergie (kWh), usages locaux, coûts (CAPEX/OPEX), coût/kWh, GO/NO-GO, + CodeCarbon (empreinte du calcul IA).")

tabs = st.tabs(["Inputs", "Results", "Methodology"])

def energy_wh_per_day(visitors_per_day, pass_share, useful_steps, joules_per_step, efficiency, storage_loss):
    steps = visitors_per_day * pass_share * useful_steps
    return (steps * joules_per_step * efficiency * (1 - storage_loss)) / 3600

def m2_to_ft2(x):
    return x * 10.7639104167

def cost_model(area_m2, cost_per_ft2, fixed_cost, maint_rate, years, annual_kwh):
    area_ft2 = m2_to_ft2(area_m2)
    capex = area_ft2 * cost_per_ft2 + fixed_cost
    opex_year = maint_rate * capex
    total_cost = capex + opex_year * years
    total_kwh = max(1e-9, annual_kwh * years)
    return capex, opex_year, total_cost, (total_cost / total_kwh)

def forecast_visitors(df, horizon_days=30):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df["t"] = (df["date"] - df["date"].min()).dt.days.astype(int)

    X = df[["t"]].values
    y = df["visitors"].values
    model = LinearRegression()
    model.fit(X, y)

    last = df["date"].max()
    future = []
    for i in range(1, horizon_days + 1):
        d = last + pd.Timedelta(days=i)
        t = int((d - df["date"].min()).days)
        pred = max(0.0, float(model.predict([[t]])[0]))
        future.append({"date": d.date(), "visitors_pred": pred})
    return pd.DataFrame(future)

with tabs[0]:
    col1, col2, col3 = st.columns(3)

    with col1:
        place_type = st.selectbox("Type de lieu", ["Musée", "Gare", "Stade/Arena", "Centre commercial", "Campus/Université", "Autre"])
        visitors_per_day = st.number_input("Visiteurs / jour (moyenne)", min_value=0, value=1000, step=50)
        peak_multiplier = st.slider("Multiplicateur pic (weekend/événement)", 1.0, 5.0, 1.5, 0.1)

    with col2:
        pass_share = st.slider("% visiteurs passant sur la zone équipée", 0.0, 30.0, 5.0, 0.5) / 100.0
        useful_steps = st.slider("Pas utiles / visiteur sur zone", 0, 400, 80, 10)
        joules_per_step = st.slider("Énergie par pas (J)", 1.0, 6.0, 3.0, 0.1)
        efficiency = st.slider("Rendement global", 0.0, 1.0, 0.5, 0.05)
        storage_loss = st.slider("Pertes stockage/conversion", 0.0, 0.5, 0.1, 0.05)

    with col3:
        st.markdown("### Coûts")
        installed_cost_per_ft2 = st.slider("Coût installé ($/ft²)", 50.0, 1000.0, 120.0, 5.0)
        fixed_install_cost = st.number_input("Coût fixe (travaux/élec/signalétique) $", min_value=0.0, value=10000.0, step=1000.0)
        annual_maintenance_rate = st.slider("Maintenance annuelle (% CAPEX)", 0.0, 20.0, 5.0, 0.5) / 100.0
        amort_years = st.slider("Amortissement (années)", 1, 15, 7, 1)

    st.divider()
    st.subheader("Données (bonus) + IA (prévision) + CodeCarbon")
    st.write("Upload CSV avec colonnes: `date, visitors`. Sinon, tu peux utiliser le dataset démo.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    use_sample = st.checkbox("Utiliser dataset démo", value=True)

    df = None
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    elif use_sample:
        df = pd.read_csv("data/sample_visitors.csv")

    if df is not None:
        st.dataframe(df.head(10), use_container_width=True)

    run_forecast = st.button("Lancer prévision (IA légère) + mesurer empreinte (CodeCarbon)")
    if run_forecast and df is not None:
        tracker = EmissionsTracker(project_name="kinetic-impact-calculator")
        tracker.start()
        fc = forecast_visitors(df, horizon_days=30)
        kg = tracker.stop()
        st.session_state["forecast_df"] = fc
        st.session_state["codecarbon_kg"] = kg
        st.success("Prévision générée.")
        st.dataframe(fc.head(10), use_container_width=True)
        st.info(f"CodeCarbon (kgCO₂e) : {kg:.6f}" if kg is not None else "CodeCarbon non disponible ici, mais l’instrumentation est en place.")

with tabs[1]:
    st.subheader("Résultats")

    visitors_mid = float(visitors_per_day)
    if "forecast_df" in st.session_state:
        visitors_mid = float(st.session_state["forecast_df"]["visitors_pred"].mean())

    wh_day_mid = energy_wh_per_day(visitors_mid, pass_share, useful_steps, joules_per_step, efficiency, storage_loss)
    wh_day_peak = energy_wh_per_day(visitors_mid * peak_multiplier, pass_share, useful_steps, joules_per_step, efficiency, storage_loss)

    kwh_year = (wh_day_mid * 365) / 1000

    st.metric("Wh/jour (moyen)", f"{wh_day_mid:.1f}")
    st.metric("kWh/an (moyen)", f"{kwh_year:.2f}")
    st.metric("Wh/jour (pic)", f"{wh_day_peak:.1f}")

    st.divider()
    st.subheader("Équivalences (indicatives)")
    st.write(f"LED 10W : ~{(wh_day_mid/10):.1f} heures/jour")
    st.write(f"Écran 20W : ~{(wh_day_mid/20):.1f} heures/jour")

    st.divider()
    st.subheader("Matériel & coûts")
    walk_zone_area_m2 = st.number_input("Surface potentielle de zone piétonne (m²)", min_value=0.0, value=20.0, step=1.0)
    tile_area_m2 = st.number_input("Surface d’une dalle (m²)", min_value=0.01, value=0.25, step=0.01)

    area_equipped_m2 = walk_zone_area_m2 * pass_share
    tiles = int(math.ceil(area_equipped_m2 / tile_area_m2))

    capex, opex_year, total_cost, cost_per_kwh = cost_model(
        area_equipped_m2, installed_cost_per_ft2, fixed_install_cost,
        annual_maintenance_rate, amort_years, kwh_year
    )

    st.write(f"Surface équipée estimée : {area_equipped_m2:.2f} m² (~{m2_to_ft2(area_equipped_m2):.1f} ft²)")
    st.write(f"Nombre de dalles (est.) : {tiles}")
    st.write(f"CAPEX : ${capex:,.0f}")
    st.write(f"OPEX/an : ${opex_year:,.0f}")
    st.write(f"Coût/kWh (sur {amort_years} ans) : ${cost_per_kwh:,.0f} / kWh")

    st.divider()
    st.subheader("Verdict GO / NO-GO (anti-gadget)")
    goal_wh = st.number_input("Objectif Wh/jour (usage local)", min_value=0.0, value=100.0, step=10.0)
    max_cost_kwh = st.number_input("Seuil coût/kWh acceptable", min_value=1.0, value=500.0, step=50.0)

    reasons = []
    if wh_day_mid < goal_wh:
        reasons.append("Ne couvre pas l’objectif énergétique journalier (scénario moyen).")
    if kwh_year < 10:
        reasons.append("Énergie annuelle très faible (<10 kWh/an) → intérêt surtout pédagogique.")
    if cost_per_kwh > max_cost_kwh:
        reasons.append("Coût/kWh élevé vs seuil choisi.")

    if len(reasons) == 0:
        st.success("✅ GO — dimensionnement cohérent avec l’objectif et les seuils.")
    else:
        st.warning("⚠️ NO-GO — " + " ".join(reasons))

    st.divider()
    st.subheader("Export (CSV)")
    export = pd.DataFrame({
        "metric": ["visitors_mid", "wh_day_mid", "kwh_year", "capex", "opex_year", "cost_per_kwh"],
        "value": [visitors_mid, wh_day_mid, kwh_year, capex, opex_year, cost_per_kwh]
    })
    bio = io.StringIO()
    export.to_csv(bio, index=False)
    st.download_button("Télécharger résumé (CSV)", bio.getvalue(), file_name="kinetic_summary.csv", mime="text/csv")

with tabs[2]:
    st.subheader("Méthodologie & transparence")
    st.markdown("""
- **Pas captés/jour** = visiteurs × %passage × pas utiles  
- **Wh/jour** = pas captés × J/pas × rendement × (1 - pertes) / 3600  
- **CAPEX** = surface(ft²) × coût($/ft²) + coût fixe  
- **OPEX** = maintenance × CAPEX  
- **IA** : prévision simple de fréquentation à partir d’un CSV (régression linéaire légère)  
- **CodeCarbon** : mesure l’empreinte (kgCO₂e) de l’étape prévision (calcul IA)  
- **Données** : agrégées (pas de tracking individuel)  
- **Limite** : énergie souvent modeste → usages locaux + valeur pédagogique/engagement
""")
