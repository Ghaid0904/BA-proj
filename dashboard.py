
import os
import sys
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

st.set_page_config(
    page_title="Dynamic Pricing Dashboard",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>
    .metric-card {
        background: #f0f4ff;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 6px 0;
        border-left: 5px solid #2196F3;
    }
    .metric-card.green { border-left-color: #4CAF50; background: #f0fff4; }
    .metric-card.orange { border-left-color: #FF9800; background: #fff8f0; }
    .big-number { font-size: 2rem; font-weight: bold; color: #1a237e; }
    .label { font-size: 0.85rem; color: #555; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    price_csv = "outputs/preisempfehlungen.csv"
    sim_csv   = "outputs/simulation_ergebnisse.csv"

    price_df = pd.read_csv(price_csv) if os.path.exists(price_csv) else None
    sim_df   = pd.read_csv(sim_csv)   if os.path.exists(sim_csv)   else None
    return price_df, sim_df


@st.cache_resource
def load_model(model_name):
    path = f"models/{model_name}"
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


price_df, sim_df = load_data()
#model = load_model()

st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg",
                 width=120)
st.sidebar.title("")
modell_wahl = st.sidebar.selectbox(
    "🤖 Modell auswählen",
    ["Random Forest", "XGBoost"]
)
model_file = "rf_model.pkl" if modell_wahl == "Random Forest" else "xgb_model.pkl"
model = load_model(model_file)

page = st.sidebar.radio("", [
    "🏠 Übersicht",
    "💰 Preisempfehlungen",
    "📈 Gewinnprognose",
    "🔬 Modell-Details",
])
st.sidebar.markdown("---")



# ═══════════════════════════════════════════════════════════════════════════════
# SEITE 1: ÜBERSICHT
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Übersicht":
    st.title("📊 Dynamic Pricing Dashboard")
    st.markdown("**Amazon E-Commerce · ML-basierte Preisoptimierung · Gattas, 2025**")
    st.markdown("---")

    # KPI-Karten
    col1, col2, col3, col4 = st.columns(4)

    if sim_df is not None:
        rev_static  = sim_df["Umsatz_Statisch (₹)"].sum()
        rev_dynamic = sim_df["Umsatz_Dynamisch (₹)"].sum()
        mar_static  = sim_df["Marge_Statisch (₹)"].sum()
        mar_dynamic = sim_df["Marge_Dynamisch (₹)"].sum()
        rev_uplift  = (rev_dynamic - rev_static) / rev_static * 100
        mar_uplift  = (mar_dynamic - mar_static) / mar_static * 100

        with col1:
            st.metric("📦 Gesamtumsatz (Dynamisch)",
                      f"₹{rev_dynamic/1e6:.1f} Mio.",
                      f"+{rev_uplift:.1f}% vs. statisch")
        with col2:
            st.metric("💹 Gesamtmarge (Dynamisch)",
                      f"₹{mar_dynamic/1e6:.1f} Mio.",
                      f"+{mar_uplift:.1f}% vs. statisch")
        with col3:
            st.metric("📉 Umsatz (Statisch)",
                      f"₹{rev_static/1e6:.1f} Mio.", "Fixpreis-Strategie")
        with col4:
            st.metric("📉 Marge (Statisch)",
                      f"₹{mar_static/1e6:.1f} Mio.", "Fixpreis-Strategie")

    st.markdown("---")

    # Modellkennzahlen (hardcoded aus main.py Output — nach dem Run aktualisieren)
    st.subheader("🤖 Modell-Kennzahlen")
    col1, col2, col3 = st.columns(3)

    plots = {
        "plot1_modellvergleich.png":    "Modellvergleich",
        "plot2_actual_vs_predicted.png": "Actual vs. Predicted",
        "plot3_feature_importance.png":  "Feature Importance",
    }
    for fname, title in plots.items():
        path = os.path.join("outputs", fname)
        if os.path.exists(path):
            st.image(path, caption=title, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SEITE 2: PREISEMPFEHLUNGEN
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💰 Preisempfehlungen":
    st.title("💰 Preisempfehlungen")
    st.markdown("Vergleich: ursprünglicher Preis vs. ML-empfohlener Preis")
    st.markdown("---")

    if price_df is None:
        st.warning("⚠️ Zuerst `python main.py` ausführen um die Daten zu generieren.")
    else:
        # Filter
        col1, col2 = st.columns(2)
        with col1:
            search = st.text_input("🔍 Produkt suchen", "")
        with col2:
            direction = st.selectbox("Filter", ["Alle", "Preiserhöhung", "Preissenkung"])

        df_show = price_df.copy()
        if search:
            df_show = df_show[df_show["Produktname"].str.contains(
                search, case=False, na=False)]
        if direction == "Preiserhöhung":
            df_show = df_show[df_show["Differenz (₹)"] > 0]
        elif direction == "Preissenkung":
            df_show = df_show[df_show["Differenz (₹)"] < 0]

        # Tabelle farbig
        def color_diff(val):
            if val > 0:
                return "color: green; font-weight: bold"
            elif val < 0:
                return "color: red; font-weight: bold"
            return ""

        styled = df_show.style.applymap(color_diff, subset=["Differenz (₹)", "Änderung (%)"])
        st.dataframe(styled, use_container_width=True, height=500)

        # Download
        csv = df_show.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ CSV herunterladen", csv,
                           "preisempfehlungen.csv", "text/csv")

        # Mini-Chart
        st.markdown("---")
        st.subheader("📊 Preis-Verteilung")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(price_df["Ursprünglicher Preis (₹)"], bins=30,
                alpha=0.6, color="#1E88E5", label="Ursprünglicher Preis")
        ax.hist(price_df["Empfohlener Preis (₹)"], bins=30,
                alpha=0.6, color="#E53935", label="Empfohlener Preis")
        ax.set_xlabel("Preis (₹)"); ax.set_ylabel("Anzahl Produkte")
        ax.set_title("Verteilung: Ursprünglicher vs. Empfohlener Preis")
        ax.legend()
        st.pyplot(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# SEITE 3: GEWINNPROGNOSE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Gewinnprognose":
    st.title("📈 Gewinnprognose – 30-Tage-Simulation")
    st.markdown("Dynamic Pricing vs. Statisches Pricing über 30 simulierte Tage")
    st.markdown("---")

    if sim_df is None:
        st.warning("⚠️ Zuerst `python main.py` ausführen.")
    else:
        rev_static  = sim_df["Umsatz_Statisch (₹)"].sum()
        rev_dynamic = sim_df["Umsatz_Dynamisch (₹)"].sum()
        mar_static  = sim_df["Marge_Statisch (₹)"].sum()
        mar_dynamic = sim_df["Marge_Dynamisch (₹)"].sum()
        rev_uplift  = (rev_dynamic - rev_static) / rev_static * 100
        mar_uplift  = (mar_dynamic - mar_static) / mar_static * 100

        # KPIs
        col1, col2 = st.columns(2)
        col1.metric("Umsatz-Uplift", f"+{rev_uplift:.1f}%",
                    f"₹{(rev_dynamic-rev_static)/1e6:.1f} Mio. mehr")
        col2.metric("Margin-Uplift", f"+{mar_uplift:.1f}%",
                    f"₹{(mar_dynamic-mar_static)/1e6:.1f} Mio. mehr Gewinn")

        st.markdown("---")

        tab1, tab2, tab3 = st.tabs(["Umsatz", "Marge/Gewinn", "Rohdaten"])

        with tab1:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle("Umsatzentwicklung – Dynamic vs. Statisch", fontweight="bold")
            axes[0].plot(sim_df["Tag"], sim_df["Umsatz_Dynamisch (₹)"] / 1e6,
                         color="#E53935", lw=2, label="Dynamic")
            axes[0].plot(sim_df["Tag"], sim_df["Umsatz_Statisch (₹)"]  / 1e6,
                         color="#1E88E5", lw=2, ls="--", label="Statisch")
            axes[0].fill_between(sim_df["Tag"],
                                 sim_df["Umsatz_Statisch (₹)"]  / 1e6,
                                 sim_df["Umsatz_Dynamisch (₹)"] / 1e6,
                                 alpha=0.15, color="#E53935")
            axes[0].set_xlabel("Tag"); axes[0].set_ylabel("₹ Mio.")
            axes[0].set_title("Täglicher Umsatz"); axes[0].legend()

            cum_dyn = sim_df["Umsatz_Dynamisch (₹)"].cumsum() / 1e6
            cum_sta = sim_df["Umsatz_Statisch (₹)"].cumsum()  / 1e6
            axes[1].plot(sim_df["Tag"], cum_dyn, color="#E53935", lw=2, label="Dynamic")
            axes[1].plot(sim_df["Tag"], cum_sta, color="#1E88E5", lw=2,
                         ls="--", label="Statisch")
            axes[1].set_xlabel("Tag"); axes[1].set_ylabel("₹ Mio.")
            axes[1].set_title(f"Kumulierter Umsatz (+{rev_uplift:.1f}%)"); axes[1].legend()
            st.pyplot(fig)

        with tab2:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle("Gewinnentwicklung – Dynamic vs. Statisch", fontweight="bold")
            axes[0].plot(sim_df["Tag"], sim_df["Marge_Dynamisch (₹)"] / 1e6,
                         color="#E53935", lw=2, label="Dynamic")
            axes[0].plot(sim_df["Tag"], sim_df["Marge_Statisch (₹)"]  / 1e6,
                         color="#1E88E5", lw=2, ls="--", label="Statisch")
            axes[0].fill_between(sim_df["Tag"],
                                 sim_df["Marge_Statisch (₹)"]  / 1e6,
                                 sim_df["Marge_Dynamisch (₹)"] / 1e6,
                                 alpha=0.15, color="#E53935")
            axes[0].set_xlabel("Tag"); axes[0].set_ylabel("₹ Mio.")
            axes[0].set_title("Täglicher Gewinn"); axes[0].legend()

            cum_dyn = sim_df["Marge_Dynamisch (₹)"].cumsum() / 1e6
            cum_sta = sim_df["Marge_Statisch (₹)"].cumsum()  / 1e6
            axes[1].plot(sim_df["Tag"], cum_dyn, color="#E53935", lw=2, label="Dynamic")
            axes[1].plot(sim_df["Tag"], cum_sta, color="#1E88E5", lw=2,
                         ls="--", label="Statisch")
            axes[1].set_xlabel("Tag"); axes[1].set_ylabel("₹ Mio.")
            axes[1].set_title(f"Kumulierter Gewinn (+{mar_uplift:.1f}%)"); axes[1].legend()
            st.pyplot(fig)

        with tab3:
            st.dataframe(sim_df, use_container_width=True)
            csv = sim_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ CSV herunterladen", csv,
                               "simulation_ergebnisse.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════════════════
# SEITE 4: MODELL-DETAILS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Modell-Details":
    st.title("🔬 Modell-Details & Erklärbarkeit")
    st.markdown("---")

    plots = {
        "plot1_modellvergleich.png":     "Abbildung 1: Modellvergleich (MAE / RMSE / R²)",
        "plot2_actual_vs_predicted.png": "Abbildung 2: Actual vs. Predicted",
        "plot3_feature_importance.png":  "Abbildung 3: Feature Importance",
        "plot4_shap.png":                "Abbildung 4: SHAP-Werte (Erklärbarkeit)",
        "plot5_umsatz.png":              "Abbildung 5: Umsatzentwicklung",
        "plot6_gewinn.png":              "Abbildung 6: Gewinnprognose",
        "plot7_stabilitaet.png":         "Abbildung 7: Preisvolatilität",
    }

    for fname, caption in plots.items():
        path = os.path.join("outputs", fname)
        if os.path.exists(path):
            st.image(path, caption=caption, use_container_width=True)
            st.markdown("---")
        else:
            st.info(f"📁 {fname} noch nicht generiert → `python main.py` ausführen")
