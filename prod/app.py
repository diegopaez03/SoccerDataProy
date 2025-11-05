from pathlib import Path
import sys
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Ensure the project root is importable when running from subdirectories
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from dev.utils import LegacySelectPreprocessPCA
from dev.preprocess import preprocessCSV

# -------------------------------------------------------
# Config
# -------------------------------------------------------
st.set_page_config(page_title="Soccer Predictor", layout="wide")

# -------------------------------------------------------
# Utils
# -------------------------------------------------------

def build_prob_table(probas: np.ndarray, classes: np.ndarray) -> pd.DataFrame:
    """
    Convierte las probabilidades (n x k) a un DataFrame con columnas por clase
    y agrega P_H (Home), P_D (Draw) y P_A (Away).
    """
    df_probs = pd.DataFrame(probas, columns=classes)
    # Normalizamos nombres canónicos por si las clases vienen en otro orden
    # Esperamos clases en {'H','D','A'} (Home, Draw, Away). Si vinieran como otras,
    # se respetan tal cual, pero calculamos combinadas solo si están.
    for col in ["H", "D", "A"]:
        if col not in df_probs.columns:
            df_probs[col] = np.nan

    # Orden amigable si existen
    ordered = [c for c in ["H", "D", "A"] if c in df_probs.columns]
    return df_probs[ordered] if ordered else df_probs

def render_manual_form(pipeline):
    """
    Renderiza un formulario para capturar una FILA con los nombres de columnas
    que el pipeline espera (los features ya seleccionados en entrenamiento).
    Extrae las listas de features del paso 'legacy_prep' del pipeline.
    """
    st.subheader("📝 Cargar un partido manualmente")
    st.markdown(
        "Completá los campos y obtené las probabilidades de resultado para **ese** partido."
    )

    # 1) Obtenemos las listas de features que el pipeline espera
    prep = pipeline.named_steps.get("legacy_prep", None)
    if prep is None:
        st.error("No se encontró el paso 'legacy_prep' en el pipeline.")
        return

    cat_feats = list(getattr(prep, "categorical_features_", []))
    num_feats = list(getattr(prep, "numeric_features_", []))

    if not cat_feats and not num_feats:
        st.warning(
            "No hay listas de features en 'legacy_prep'. "
            "¿Seguro que el pipeline fue entrenado con LegacySelectPreprocessPCA y ya está ajustado?"
        )
        return

    with st.expander("📋 Campos requeridos (features esperados)", expanded=False):
        st.write("**Categóricos**:", cat_feats if cat_feats else "(ninguno)")
        st.write("**Numéricos**:", num_feats if num_feats else "(ninguno)")

    # 2) Formulario de entrada
    with st.form("single_match_form"):
        st.markdown("**Completá los valores de entrada**")
        inputs = {}

        # Campos categóricos: texto
        if cat_feats:
            st.markdown("**Categóricos**")
            cols = st.columns(min(3, len(cat_feats)))
            for i, col_name in enumerate(cat_feats):
                with cols[i % len(cols)]:
                    val = st.text_input(col_name, value="")
                    inputs[col_name] = val

        # Campos numéricos: number_input
        if num_feats:
            st.markdown("**Numéricos**")
            cols = st.columns(min(3, len(num_feats)))
            for i, col_name in enumerate(num_feats):
                with cols[i % len(cols)]:
                    # number_input devuelve float, podés ajustar step si querés enteros
                    val = st.number_input(col_name, value=0.0, step=1.0, format="%.6f")
                    inputs[col_name] = float(val)

        submitted = st.form_submit_button("🔮 Predecir un partido")
    
    if not submitted:
        return

    # 3) Construimos el DataFrame de UNA fila en el orden correcto
    #    (si faltan columnas, lo avisamos; si sobran, las ignoramos)
    input_cols = num_feats + cat_feats
    provided_cols = list(inputs.keys())

    missing = [c for c in input_cols if c not in provided_cols]
    if missing:
        st.error(f"Faltan columnas requeridas: {missing}")
        return

    # Armamos el DF con una sola fila
    row = {c: inputs[c] for c in input_cols}
    df_one = pd.DataFrame([row])

    # 4) Predicción de probabilidades
    try:
        probas = pipeline.predict_proba(df_one)
        classes = getattr(pipeline.named_steps["model"], "classes_", None)
        if classes is None:
            st.error("El estimador final no posee `classes_`. ¿Es un clasificador?")
            return

        df_probs = pd.DataFrame(probas, columns=classes)

        # Normalizamos nombres esperados H/D/A si existen
        for col in ["H", "D", "A"]:
            if col not in df_probs.columns:
                df_probs[col] = np.nan

        st.success("✅ Predicción realizada para el partido ingresado.")
        st.write("**Probabilidades** (una sola fila):")
        st.dataframe(df_probs[["H","D","A"]].style.format("{:.2%}"))

        # Gráfico
        st.bar_chart(df_probs[["H","D","A"]].iloc[0])

    except Exception as e:
        st.error(f"Ocurrió un error en la predicción: {e}")


def main():
    st.title("Pronóstico de partidos de fútbol en ligas")

    # =========================
    # Intro
    # =========================
    with st.container():
        st.subheader("🔍 ¿Qué hace esta herramienta?")
        st.markdown("""
        Esta aplicación permite **estimar las probabilidades de resultado** de partidos de fútbol
        (Local **H**, Empate **D**, Visitante **A**) usando un **modelo entrenado** (Logistic Regression) encapsulado en un **pipeline de scikit-learn**.

        > **Importante:** para predecir, cargá un **CSV con las columnas crudas** que usaste al entrenar.
        El pipeline se encarga de seleccionar features, imputar, escalar, codificar, aplicar **PCA** y predecir.
        """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("⚙️ **Pipeline end-to-end**\n\n`select_features` → preprocesamiento → PCA → modelo.")
    with col2:
        st.success("📈 **Probabilidades**\n\nObtendrás **P(H), P(D) y P(A)**.")
    with col3:
        st.warning("🧪 **Uso educativo**\n\nNo promueve apuestas. El objetivo es didáctico e investigativo.")

    st.subheader("⚙️ ¿Cómo usar la aplicación?")
    with st.expander("➡️ Ver instrucciones"):
        st.markdown("""
        1. **Seleccioná el modelo** (cargá el archivo `.joblib` de tu pipeline).
        2. **Cargá un CSV** con filas a predecir (mismas columnas crudas del entrenamiento).
        3. **Presioná “Predecir”** para ver las probabilidades por partido.
        4. (Opcional) Seleccioná una fila para ver un gráfico rápido.
        """)

    st.markdown("---")
    st.header("📌 ¡Predecí tu partido ahora mismo!")

    pipeline = joblib.load("dev\pipeline_logreg_pca_09.joblib")

    render_manual_form(pipeline)
    
    # =========================
    # Carga de datos a predecir
    # =========================
    st.subheader("2) Cargar datos (CSV con columnas crudas)")
    data_file = st.file_uploader("Subí tu archivo CSV", type=["csv"])
    df_pred: Optional[pd.DataFrame] = None
    if data_file is not None:
        try:
            df_pred = preprocessCSV(pd.read_csv(data_file))
            st.write("Vista previa de los datos cargados:")
            st.dataframe(df_pred.head(5))
        except Exception as e:
            st.error(f"No pude leer el CSV. Detalle: {e}")

    st.markdown("---")

    # =========================
    # Predicción
    # =========================
    st.subheader("3) Predicción de probabilidades")
    colp1, colp2 = st.columns([1,2])

    if pipeline is None:
        colp1.warning("⛔ Primero cargá un **modelo**.")
    elif df_pred is None:
        colp1.warning("⛔ Falta cargar el **CSV** con datos.")
    else:
        if st.button("🔮 Predecir", type="primary"):
            try:
                # predict_proba y clases
                probas = pipeline.predict_proba(df_pred)
                # clases vienen del estimador final dentro del pipeline
                classes = getattr(pipeline.named_steps["model"], "classes_", None)
                if classes is None:
                    raise RuntimeError("El estimador final no tiene atributo `classes_`. ¿Es un clasificador?")

                prob_table = build_prob_table(probas, classes)
                st.success("Predicción realizada.")
                st.write("**Probabilidades por fila:**")

                prob_display = prob_table.reset_index(drop=True)
                meta_df = pd.DataFrame(index=prob_display.index)
                meta_columns = []

                if "HomeTeam" in df_pred.columns:
                    meta_df["Equipo local"] = df_pred["HomeTeam"].reset_index(drop=True)
                    meta_columns.append("Equipo local")

                if "AwayTeam" in df_pred.columns:
                    meta_df["Equipo visitante"] = df_pred["AwayTeam"].reset_index(drop=True)
                    meta_columns.append("Equipo visitante")

                if "matchday" in df_pred.columns:
                    meta_df["Jornada"] = df_pred["matchday"].reset_index(drop=True)
                    meta_columns.append("Jornada")

                if meta_columns:
                    meta_display = meta_df[meta_columns]
                    display_table = pd.concat([meta_display, prob_display], axis=1)
                else:
                    display_table = prob_display

                format_dict = {col: "{:.2%}" for col in prob_table.columns}
                st.dataframe(display_table.style.format(format_dict))

                with st.expander("Descargar resultados (CSV)"):
                    out = pd.concat([df_pred.reset_index(drop=True), prob_table.reset_index(drop=True)], axis=1)
                    csv = out.to_csv(index=False).encode("utf-8")
                    st.download_button("Descargar CSV con probabilidades", csv, file_name="predicciones_con_probabilidades.csv", mime="text/csv")


            except Exception as e:
                st.error(f"Ocurrió un error durante la predicción: {e}")

    # =========================
    # Advertencia + créditos
    # =========================
    st.markdown("---")
    st.subheader("⚠️ A tener en cuenta")
    st.warning("""
    ❗​ Esta herramienta no promueve su uso para apuestas deportivas ni de ningún tipo.
    Su uso es **puramente educativo** y busca apoyar el aprendizaje de **machine learning**.
    """)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Hecho por Diego Paez y Nicolas Carcaño</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()

