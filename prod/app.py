from pathlib import Path
import sys
from typing import Optional
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="Footy Predictor", layout="wide", page_icon="⚽")

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from dev.preprocess import preprocessCSV
from dev.utils import LegacySelectPreprocessPCA

# ====== CSS ======
st.markdown("""
<style>
html, body, [class*="css"] {
  font-family: 'Inter', system-ui, sans-serif;
}

/* ===== Header ===== */
.header { text-align: center; margin-top: -30px; }
.header h1 { font-size: 42px; font-weight: 900; color: #2563eb; }
.header hr { border-top: 1px solid #334155; width: 90%; margin: 4px auto 20px auto; }

/* ===== Sidebar ===== */
[data-testid="stSidebar"] {
  background-color: #12275a;
  padding: 25px 18px;
  color: white;
}
[data-testid="stSidebar"] h3 {
  color: #fff;
  font-weight: 800;
  margin-bottom: 14px;
  font-size: 20px;
}
div[role="radiogroup"] > label {
  background-color: rgba(255,255,255,0.05);
  padding: 12px;
  width: 100%;
  margin-bottom: 8px;
  border-radius: 8px;
  font-size: 17px !important;
  font-weight: 600;
  color: white !important;
  cursor: pointer;
  border: 1px solid transparent;
}
div[role="radiogroup"] > label:hover {
  background-color: rgba(255,255,255,0.12);
  border: 1px solid #1d4ed8;
}
input[type="radio"] { accent-color: #2563eb !important; }

/* ===== Tabs ===== */
.stTabs { width: 100%; margin-top: 20px; margin-bottom: 25px; }
.stTabs [data-baseweb="tab-list"] {
  border-bottom: none !important;
  display: flex;
  justify-content: center;
  gap: 16px;
}
.stTabs [data-baseweb="tab"] {
  background-color: #1e3a8a;
  color: #e2e8f0;
  font-size: 18px;
  font-weight: 700;
  padding: 12px 28px;
  border-radius: 10px;
  transition: all 0.25s ease;
  border: 2px solid transparent;
  box-shadow: 0 4px 8px rgba(0,0,0,0.15);
  min-width: 140px;
  text-align: center;
}
.stTabs [data-baseweb="tab"]:hover {
  background-color: #2563eb;
  color: white;
  transform: translateY(-2px);
}
.stTabs [aria-selected="true"] {
  background-color: #2563eb !important;
  color: white !important;
  border: 2px solid #60a5fa !important;
  box-shadow: 0 6px 14px rgba(0,0,0,0.25);
}

/* ===== Buttons in tables ===== */
.predict-btn {
  background-color: #2563eb;
  color: white;
  border: none;
  padding: 6px 12px;
  border-radius: 6px;
  font-weight: 600;
  cursor: pointer;
  transition: 0.25s ease;
}
.predict-btn:hover {
  background-color: #1d4ed8;
  transform: translateY(-1px);
}

/* ===== Cards About ===== */
.card-about {
  background: linear-gradient(145deg, #1e3a8a, #1d4ed8);
  padding: 24px;
  border-radius: 14px;
  box-shadow: 0 4px 16px rgba(0,0,0,0.2);
  margin-bottom: 20px;
  color: #f8fafc;
  border-left: 5px solid #60a5fa;
}
.card-about b { color: #fff; }
.card-about ul { margin-top: 10px; margin-bottom: 0; }
.card-about li { color: #e2e8f0; font-weight: 500; }

/* ===== Section titles ===== */
.section-title {
  font-size: 24px;
  font-weight: 800;
  color: #2563eb;
  margin-bottom: 15px;
}

/* ===== Footer ===== */
.footer {
  font-size: 13px;
  text-align: center;
  color: #cbd5e1;
  margin-top: 25px;
}
</style>
""", unsafe_allow_html=True)

# ===== Estado =====
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "selected_page" not in st.session_state:
    st.session_state.selected_page = "⚽ Ligas"
if "selected_league" not in st.session_state:
    st.session_state.selected_league = "Premier League"

LEAGUES = ["Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1"]

# ===== Funciones auxiliares =====
def build_prob_table(probas: np.ndarray, classes: np.ndarray) -> pd.DataFrame:
    df_probs = pd.DataFrame(probas, columns=classes)
    for col in ["H","D","A"]:
        if col not in df_probs.columns:
            df_probs[col] = np.nan
    ordered = [c for c in ["H","D","A"] if c in df_probs.columns]
    return df_probs[ordered] if ordered else df_probs


def render_manual_form(pipeline):
    st.subheader("📝 Cargar un partido manualmente")
    prep = pipeline.named_steps.get("legacy_prep", None)
    if prep is None:
        st.error("No se encontró el paso 'legacy_prep' en el pipeline.")
        return
    cat_feats = list(getattr(prep, "categorical_features_", []))
    num_feats = list(getattr(prep, "numeric_features_", []))
    with st.form("manual_form"):
        cat_inputs = {c: st.text_input(c, "") for c in cat_feats}
        num_inputs = {c: st.number_input(c, 0.0) for c in num_feats}
        submitted = st.form_submit_button("🔮 Predecir partido")
    if submitted:
        df_one = pd.DataFrame([{**num_inputs, **cat_inputs}])
        probas = pipeline.predict_proba(df_one)
        classes = getattr(pipeline.named_steps["model"], "classes_", [])
        df_probs = build_prob_table(probas, classes)
        st.dataframe(df_probs.style.format("{:.2%}"))
        st.bar_chart(df_probs.iloc[0])

# ===== Sección Ligas =====
def render_ligas():
    st.markdown('<div class="section-title">⚽ Seleccioná la liga</div>', unsafe_allow_html=True)
    league = st.selectbox("Liga", LEAGUES, index=LEAGUES.index(st.session_state.selected_league))
    st.session_state.selected_league = league

    tabs = st.tabs(["📅 Fecha actual", "🏆 Tabla", "📜 Fixture", "📈 Visualizaciones"])

    # --- Fecha actual ---
    with tabs[0]:
        st.subheader(f"{league} — Fecha actual")

        df_matches = pd.DataFrame({
            "Local": ["Arsenal", "Chelsea", "Liverpool", "Man City", "Tottenham"],
            "Visitante": ["Brighton", "Brentford", "Newcastle", "Aston Villa", "Fulham"],
            "Predicción de resultado (H/D/A)": ["—", "—", "—", "—", "—"]
        })

        # Mostrar tabla con botones de predicción
        for i, row in df_matches.iterrows():
            col1, col2, col3, col4 = st.columns([3, 3, 2.5, 1])
            with col1:
                st.write(f"**{row['Local']}**")
            with col2:
                st.write(f"{row['Visitante']}")
            with col3:
                st.write(f"Predicción: {row['Predicción de resultado (H/D/A)']}")
            with col4:
                st.markdown(f"<button class='predict-btn'>Predecir</button>", unsafe_allow_html=True)

    # --- Tabla ---
    with tabs[1]:
        st.subheader(f"{league} — Tabla de posiciones")
        df_table = pd.DataFrame({
            "Pos": range(1,6),
            "Equipo": ["Man City", "Arsenal", "Liverpool", "Tottenham", "Newcastle"],
            "PJ": [20,20,20,20,20],
            "G": [15,14,13,12,10],
            "E": [4,5,4,5,6],
            "P": [1,1,3,3,4],
            "GF": [45,40,42,38,36],
            "GC": [18,20,25,28,30],
            "Pts": [49,47,43,41,36]
        })
        st.dataframe(df_table, use_container_width=True)

    # --- Fixture ---
    with tabs[2]:
        st.subheader(f"{league} — Fixture completo (simulado)")
        jornadas = []
        for j in range(1,4):
            jornadas.append(pd.DataFrame({
                "Jornada": [j]*5,
                "Local": ["Arsenal","Chelsea","Liverpool","Man City","Tottenham"],
                "Visitante": ["Brighton","Brentford","Newcastle","Aston Villa","Fulham"]
            }))
        for jornada in jornadas:
            st.markdown(f"### Jornada {int(jornada['Jornada'].iloc[0])}")
            st.table(jornada.drop(columns="Jornada"))

    # --- Visualizaciones ---
    with tabs[3]:
        st.subheader(f"{league} — Visualizaciones")
        df_viz = pd.DataFrame({
            "Equipo": ["Man City","Arsenal","Liverpool","Tottenham","Newcastle"],
            "Rendimiento": [85,82,79,75,70],
            "Goles_Promedio": [2.4,2.1,2.2,2.0,1.8]
        })
        chart1 = alt.Chart(df_viz).mark_bar().encode(
            x="Equipo", y="Rendimiento", color=alt.Color("Equipo", legend=None)
        ).properties(title="Rendimiento (%)", width=500, height=300)
        chart2 = alt.Chart(df_viz).mark_line(point=True).encode(
            x="Equipo", y="Goles_Promedio", color="Equipo"
        ).properties(title="Goles promedio por partido", width=500, height=300)
        st.altair_chart(chart1 | chart2, use_container_width=True)

# ===== Sección Predicción Manual =====
def render_prediccion_manual():
    st.markdown('<div class="section-title">🔮 Predicción manual</div>', unsafe_allow_html=True)
    if st.session_state.pipeline is None:
        try:
            st.session_state.pipeline = joblib.load("dev/pipeline_logreg_pca_09.joblib")
            st.success("Modelo cargado automáticamente ✅")
        except Exception as e:
            st.error(f"No se pudo cargar el modelo: {e}")
            return
    pipeline = st.session_state.pipeline
    tabs = st.tabs(["🧾 CSV", "📝 Manual"])

    with tabs[0]:
        st.subheader("📂 Cargar CSV con columnas crudas")
        data_file = st.file_uploader("Subí tu archivo CSV", type=["csv"])
        if data_file:
            df_pred = preprocessCSV(pd.read_csv(data_file))
            st.dataframe(df_pred.head())
            if st.button("🔮 Predecir desde CSV"):
                probas = pipeline.predict_proba(df_pred)
                classes = pipeline.named_steps["model"].classes_
                prob_table = build_prob_table(probas, classes)
                st.dataframe(prob_table.style.format("{:.2%}"))

    with tabs[1]:
        render_manual_form(pipeline)

    st.markdown("---")
    st.warning("❗ Esta herramienta tiene fines educativos. No promueve apuestas deportivas.")

# ===== Sección About =====
def render_about():
    st.markdown('<div class="section-title">ℹ️ Acerca del proyecto</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="card-about">
<b>Footy Predictor</b> es una aplicación educativa para el análisis y predicción de partidos de fútbol en las 5 grandes ligas europeas.  
Utiliza Machine Learning (Logistic Regression + PCA) y visualizaciones interactivas para explorar patrones en los resultados.
</div>

<div class="card-about">
<b>🎯 Objetivo</b><br>
Analizar datos reales del fútbol europeo y generar predicciones H/D/A (local, empate, visitante) de manera automatizada.
</div>

<div class="card-about">
<b>🧠 Tecnologías</b>
<ul>
<li>Python 3.11</li>
<li>Streamlit</li>
<li>Scikit-learn</li>
<li>Pandas / Numpy</li>
<li>Altair</li>
</ul>
</div>

<div class="card-about">
<b>👨‍💻 Autores</b><br>
Diego Páez y Nicolás Carcaño  
UTN — Facultad Regional Mendoza (2025)
</div>
""", unsafe_allow_html=True)
    st.markdown('<div class="footer">© 2025 Footy Predictor — Proyecto académico</div>', unsafe_allow_html=True)

# ===== Layout =====
st.markdown('<div class="header"><h1>Footy Predictor</h1><hr></div>', unsafe_allow_html=True)
with st.sidebar:
    st.markdown("### ⚙️ Navegación")
    page = st.radio("", ["⚽ Ligas", "🧮 Predicción manual", "ℹ️ Acerca"])
    st.session_state.selected_page = page

if st.session_state.selected_page == "⚽ Ligas":
    render_ligas()
elif st.session_state.selected_page == "🧮 Predicción manual":
    render_prediccion_manual()
elif st.session_state.selected_page == "ℹ️ Acerca":
    render_about()
