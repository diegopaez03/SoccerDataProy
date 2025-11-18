from pathlib import Path
import sys
from typing import Optional
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

load_dotenv(ROOT_DIR / ".env")

from dev.preprocess import preprocessCSV
from dev.utils import LegacySelectPreprocessPCA, build_match_row_from_api
from prod.football_data_client import (
    FootballDataAPIError,
    FootballDataClient,
    latest_matchday_df,
    standings_to_dataframe,
)

st.set_page_config(page_title="Footy Predictor", layout="wide", page_icon="â½")

LEAGUES = ["Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1"]
API_LEAGUES = [
    {"label": "Premier League", "code": "PL"},
    {"label": "La Liga", "code": "PD"},
    {"label": "Serie A", "code": "SA"},
    {"label": "Bundesliga", "code": "BL1"},
    {"label": "Ligue 1", "code": "FL1"},
]
API_LEAGUE_CODE_INDEX = {opt["code"]: idx for idx, opt in enumerate(API_LEAGUES)}
HISTORY_CSV_PATH = ROOT_DIR / 'data' / 'ALL_top5_20seasons_consolidated.csv'
PIPELINE_PATH = ROOT_DIR / 'dev' / 'pipeline_logreg_pca_09.joblib'

DIVISION_CODES = {
    'PL': 'E0',
    'PD': 'SP1',
    'SA': 'I1',
    'BL1': 'D1',
    'FL1': 'F1',
}

LEAGUE_TO_CSV = {
    'Premier League': 'England',
    'La Liga': 'Spain',
    'Serie A': 'Italy',
    'Bundesliga': 'Germany',
    'Ligue 1': 'France',
}

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
  padding: 25px 0px;
  color: white;
}
[data-testid="stSidebar"] h3 {
  color: #fff;
  font-weight: 800;
  margin-bottom: 14px;
  font-size: 20px;
}
div[role="radiogroup"] {
  display: flex;
  justify-content: center;
  flex-wrap: wrap;
  gap: 10px;
  padding: 8px 5px;
  margin: -5px auto 24px auto;
  backdrop-filter: blur(12px);
  max-width: 700px;
}
div[role="radiogroup"] > label[data-baseweb="radio"] {
  background: transparent;
  color: #e2e8f0 !important;
  padding: 10px 15px;
  border-radius: 999px;
  border: 1px solid rgba(148, 163, 184, 0.45);
  cursor: pointer;
  transition: all .2s ease;
  font-size: 15px !important;
  font-weight: 700;
  letter-spacing: 0.4px;
}
div[role="radiogroup"] > label[data-baseweb="radio"] > div:first-child {
  display: none !important;
}
div[role="radiogroup"] > label[data-baseweb="radio"]:hover {
  border-color: rgba(37, 99, 235, 0.9);
  background: rgba(37, 99, 235, 0.12);
  color: #f8fafc !important;
}
div[role="radiogroup"] > label[data-baseweb="radio"][aria-checked="true"] {
  background: #2563eb;
  color: #0f172a !important;
  border-color: #2563eb;
  box-shadow: 0 8px 20px rgba(37, 99, 235, 0.35);
}
div[role="radiogroup"] > label[data-baseweb="radio"][aria-checked="true"]:hover {
  background: #1d4ed8;
  border-color: #1d4ed8;
}
input[type="radio"] {
  position: absolute;
  opacity: 0;
  pointer-events: none;
}

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

# ===== Estado ======
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "selected_page" not in st.session_state:
    st.session_state.selected_page = "â½ Ligas"
if "selected_league" not in st.session_state:
    st.session_state.selected_league = "Premier League"
if "selected_api_league" not in st.session_state:
    st.session_state.selected_api_league = API_LEAGUES[0]["code"]

if "pipeline_error" not in st.session_state:
    st.session_state.pipeline_error = None
if "last_api_prediction" not in st.session_state:
    st.session_state.last_api_prediction = None

@st.cache_data(show_spinner=False)
def load_history_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_resource(show_spinner=False)
def get_football_data_client(token: Optional[str] = None) -> FootballDataClient:
    return FootballDataClient(token=token)

# ===== Funciones auxiliares =====
def build_prob_table(probas: np.ndarray, classes: np.ndarray) -> pd.DataFrame:
    df_probs = pd.DataFrame(probas, columns=classes)
    for col in ["H","D","A"]:
        if col not in df_probs.columns:
            df_probs[col] = np.nan
    ordered = [c for c in ["H","D","A"] if c in df_probs.columns]
    return df_probs[ordered] if ordered else df_probs


def get_prediction_pipeline():
    if st.session_state.pipeline is None:
        try:
            st.session_state.pipeline = joblib.load(PIPELINE_PATH)
            st.session_state.pipeline_error = None
        except Exception as exc:
            st.session_state.pipeline_error = str(exc)
            return None
    return st.session_state.pipeline


def _parse_year(date_str):
    if not date_str:
        return None
    try:
        return pd.to_datetime(date_str).year
    except Exception:
        return None


def _season_code(start_year, end_year):
    if start_year and end_year:
        return int(f"{start_year % 100:02d}{end_year % 100:02d}")
    return start_year or end_year


def _build_meta_from_match(row, league_label, competition_code, season_info):
    start_year = _parse_year(season_info.get('startDate'))
    end_year = _parse_year(season_info.get('endDate'))
    if start_year is None:
        start_year = _parse_year(row.get('season'))
    if end_year is None and start_year is not None:
        end_year = start_year + 1
    match_dt = None
    if row.get('utc_date') is not None:
        try:
            match_dt = pd.to_datetime(row.get('utc_date'))
        except Exception:
            match_dt = None
    match_date_str = match_dt.strftime('%Y-%m-%d') if match_dt is not None else row.get('utc_date')
    season_value = _season_code(start_year, end_year)
    league_csv = LEAGUE_TO_CSV.get(league_label, league_label)
    meta = {
        'Div': DIVISION_CODES.get(competition_code, competition_code),
        'HomeTeam': row.get('home_team'),
        'AwayTeam': row.get('away_team'),
        'match_date': match_date_str,
        'league': league_csv,
        'season': season_value,
        'Date_raw': match_date_str,
        'season_start': start_year,
        'season_end': end_year,
        'matchday': int(row['matchday']) if pd.notna(row.get('matchday')) else None,
        'match_id': int(row['match_id']) if pd.notna(row.get('match_id')) else None,
    }
    return meta


def predict_match_from_api(row, standings_payload, competition_code, league_label):
    pipeline = get_prediction_pipeline()
    if pipeline is None:
        raise RuntimeError(st.session_state.pipeline_error or 'No se pudo cargar el modelo.')
    home_id = row.get('home_team_id')
    away_id = row.get('away_team_id')
    if pd.isna(home_id) or pd.isna(away_id):
        raise ValueError('No se encontraron los IDs de equipos para el partido seleccionado.')
    meta = _build_meta_from_match(row, league_label, competition_code, standings_payload.get('season', {}))
    features_df = build_match_row_from_api(standings_payload, int(home_id), int(away_id), meta=meta)
    probas = pipeline.predict_proba(features_df)
    classes = getattr(pipeline.named_steps['model'], 'classes_', [])
    prob_table = build_prob_table(probas, classes)
    predicted_label = prob_table.iloc[0].idxmax() if not prob_table.empty else None
    return {
        'home': row.get('home_team'),
        'away': row.get('away_team'),
        'matchday': row.get('matchday'),
        'probabilities': prob_table,
        'label': predicted_label,
    }


def render_manual_form(pipeline):
    st.subheader("ð Cargar un partido manualmente")
    prep = pipeline.named_steps.get("legacy_prep", None)
    if prep is None:
        st.error("No se encontró el paso 'legacy_prep' en el pipeline.")
        return
    cat_feats = list(getattr(prep, "categorical_features_", []))
    num_feats = list(getattr(prep, "numeric_features_", []))
    with st.form("manual_form"):
        cat_inputs = {c: st.text_input(c, "") for c in cat_feats}
        num_inputs = {c: st.number_input(c, 0.0) for c in num_feats}
        submitted = st.form_submit_button("ð® Predecir partido")
    if submitted:
        df_one = pd.DataFrame([{**num_inputs, **cat_inputs}])
        probas = pipeline.predict_proba(df_one)
        classes = getattr(pipeline.named_steps["model"], "classes_", [])
        df_probs = build_prob_table(probas, classes)
        st.dataframe(df_probs.style.format("{:.2%}"))
        st.bar_chart(df_probs.iloc[0])


# ===== Seccion Informe Modelos =====
def render_model_report():
    st.markdown('<div class="section-title">Informe comparativo de modelos</div>', unsafe_allow_html=True)
    st.write("Usa esta vista para documentar el rendimiento de los distintos modelos entrenados. Los datos mostrados son de ejemplo para que completes el informe mas adelante.")

    models_df = pd.DataFrame([
        {'Modelo': 'LogReg + PCA', 'Accuracy': 0.61, 'F1_macro': 0.58, 'ROC_AUC': 0.66},
        {'Modelo': 'Random Forest', 'Accuracy': 0.64, 'F1_macro': 0.60, 'ROC_AUC': 0.69},
        {'Modelo': 'XGBoost', 'Accuracy': 0.67, 'F1_macro': 0.63, 'ROC_AUC': 0.72},
    ])
    st.markdown('#### Tabla resumen (demo)')
    st.dataframe(
        models_df.style.format({'Accuracy': '{:.2%}', 'F1_macro': '{:.2%}', 'ROC_AUC': '{:.2f}'}),
        hide_index=True,
    )

    st.markdown('#### Visualizacion de metricas (demo)')
    metrics_long = models_df.melt(id_vars='Modelo', var_name='Metrica', value_name='Score')
    chart = (
        alt.Chart(metrics_long)
        .mark_bar()
        .encode(
            x=alt.X('Modelo:N', title='Modelo'),
            y=alt.Y('Score:Q', title='Valor'),
            color='Metrica:N',
            column=alt.Column('Metrica:N', title=None),
        )
        .properties(height=240)
    )
    st.altair_chart(chart, use_container_width=True)

    st.markdown('#### Proximos pasos')
    st.info("Reemplaza la tabla y el grafico con metricas reales (por ejemplo, importando un CSV con resultados de experimentos). Podes sumar graficas adicionales como matrices de confusion, curvas ROC y analisis de error.")


# ===== Sección Visualizaciones =====
def render_ligas():
    st.markdown('<div class="section-title">Visualizaciones</div>', unsafe_allow_html=True)

    # Selector global de liga
    league = st.selectbox("Liga", LEAGUES, index=LEAGUES.index(st.session_state.selected_league))
    st.session_state.selected_league = league

    st.subheader('Visualizaciones con datos desde la temporada 06/07 a la actualidad')

    if not HISTORY_CSV_PATH.exists():
        st.warning(f"No se encontro el archivo {HISTORY_CSV_PATH}.")
        return

    try:
        history_df = load_history_data(HISTORY_CSV_PATH)
    except Exception as exc:
        st.error(f'No se pudo leer el CSV: {exc}')
        return

    # Filtrar una sola vez segun la liga elegida
    csv_league_name = LEAGUE_TO_CSV.get(league, league)
    league_df_global = history_df[history_df['league'] == csv_league_name].copy()

    # ===== Vista previa (ahora filtrada por liga seleccionada) =====
    st.markdown('#### Vista previa')
    st.dataframe(league_df_global.head(25), use_container_width=True)

    # ===== Resumen rapido (también sobre la liga filtrada) =====
    st.markdown('#### Resumen rapido')
    info_cols = st.columns(3)
    info_cols[0].metric('Filas', f"{len(league_df_global):,}")
    info_cols[1].metric('Columnas', f"{len(league_df_global.columns):,}")
    info_cols[2].metric('Ultima carga', pd.Timestamp.utcnow().strftime('%d/%m/%Y %H:%M UTC'))

    # ===== Evolución de puntos por temporada =====
    st.markdown('---')
    st.markdown('#### Evolución de puntos por jornada')
    st.markdown('**Interactivo**')
    required_cols = {
        "season_start", "season_end", "matchday", "FTR",
        "HomeTeam", "AwayTeam", "points_home_pre", "points_away_pre"
    }
    missing_cols = [col for col in required_cols if col not in league_df_global.columns]

    if missing_cols:
        st.info(f"Faltan columnas para construir la evolución de puntos: {', '.join(missing_cols)}.")
    else:
        season_catalog = (
            league_df_global[["season_start", "season_end"]]
            .dropna()
            .drop_duplicates()
            .sort_values("season_start")
            .copy()
        )

        if season_catalog.empty:
            st.info("No hay temporadas disponibles para esta liga.")
        else:
            def _format_season(row):
                try:
                    start = int(row["season_start"])
                    end = int(row["season_end"])
                    return f"{start}/{end % 100:02d}"
                except (TypeError, ValueError):
                    return f"{row['season_start']} - {row['season_end']}"

            season_catalog["season_label"] = season_catalog.apply(_format_season, axis=1)
            season_labels = season_catalog["season_label"].tolist()
            default_idx = len(season_labels) - 1 if season_labels else 0
            selected_label = st.selectbox("Temporada", season_labels, index=default_idx)

            season_row = season_catalog[season_catalog["season_label"] == selected_label].iloc[0]
            season_mask = (
                (league_df_global["season_start"] == season_row["season_start"]) &
                (league_df_global["season_end"] == season_row["season_end"])
            )
            season_df = league_df_global.loc[season_mask].copy()

            if season_df.empty:
                st.info("No hay partidos para la temporada seleccionada.")
            else:
                season_df = season_df.sort_values(["matchday", "Date"])
                result_points = {"H": (3, 0), "D": (1, 1), "A": (0, 3)}
                point_records = []

                for _, row in season_df.iterrows():
                    matchday = row.get("matchday")
                    if pd.isna(matchday):
                        continue
                    try:
                        matchday = int(matchday)
                    except (TypeError, ValueError):
                        continue

                    ftr = row.get("FTR")
                    home_gain, away_gain = result_points.get(ftr, (0, 0))

                    for team_col, points_col, gain in [
                        ("HomeTeam", "points_home_pre", home_gain),
                        ("AwayTeam", "points_away_pre", away_gain),
                    ]:
                        team = row.get(team_col)
                        if pd.isna(team) or team == "null":
                            continue
                        pre_points = row.get(points_col)
                        if pd.isna(pre_points):
                            continue

                        point_records.append({
                            "Team": team,
                            "Matchday": matchday,
                            "Points": float(pre_points + gain),
                            "Season": selected_label,
                        })

                if not point_records:
                    st.info("No se pudieron calcular puntos acumulados para esta temporada.")
                else:
                    points_df = pd.DataFrame(point_records)
                    ranking = (
                        points_df.groupby("Team")["Points"]
                        .max()
                        .sort_values(ascending=False)
                    )
                    team_options = ranking.index.tolist()
                    default_selection = team_options[:5] if len(team_options) >= 5 else team_options
                    selected_teams = st.multiselect(
                        "Equipos a mostrar",
                        team_options,
                        default=default_selection,
                        help="ElegÃ­ qué equipos comparar en la progresión de puntos."
                    )

                    plot_df = points_df if not selected_teams else points_df[points_df["Team"].isin(selected_teams)]

                    if plot_df.empty:
                        st.info("Seleccioná al menos un equipo para visualizar la curva de puntos.")
                    else:
                        chart_points = (
                            alt.Chart(plot_df)
                            .mark_line(point=alt.OverlayMarkDef(size=35))
                            .encode(
                                x=alt.X("Matchday:Q", title="Jornada"),
                                y=alt.Y("Points:Q", title="Puntos acumulados"),
                                color=alt.Color("Team:N", title="Equipo"),
                                tooltip=[
                                    alt.Tooltip("Season:N", title="Temporada"),
                                    alt.Tooltip("Team:N", title="Equipo"),
                                    alt.Tooltip("Matchday:Q", title="Jornada"),
                                    alt.Tooltip("Points:Q", title="Puntos", format="d"),
                                ]
                            )
                            .properties(width=950, height=450)
                            .interactive()
                        )
                        st.altair_chart(chart_points, use_container_width=True)

    # ===== Distribución de resultados por equipo local =====
    st.markdown('#### Resultados por equipo jugando como local')

    league_df_local = league_df_global[
        league_df_global["HomeTeam"].notna() & (league_df_global["HomeTeam"] != "null")
    ].copy()

    if "HomeTeam" in league_df_local.columns and "FTR" in league_df_local.columns and not league_df_local.empty:
        def resultado_local(row):
            if row["FTR"] == "H":
                return "Victoria"
            elif row["FTR"] == "D":
                return "Empate"
            else:
                return "Derrota"

        league_df_local["Resultado"] = league_df_local.apply(resultado_local, axis=1)

        win_rates_local = (
            league_df_local[league_df_local["Resultado"] == "Victoria"]
            .groupby("HomeTeam")
            .size()
            .div(league_df_local.groupby("HomeTeam").size())
            .sort_values(ascending=False)
            .index.tolist()
        )

        chart_local = (
            alt.Chart(league_df_local)
            .transform_aggregate(
                count="count()",
                groupby=["HomeTeam", "Resultado"]
            )
            .transform_joinaggregate(
                total_per_team="sum(count)",
                groupby=["HomeTeam"]
            )
            .transform_calculate(
                porcentaje="datum.count / datum.total_per_team"
            )
            .mark_bar()
            .encode(
                x=alt.X(
                    "HomeTeam:N",
                    title="Equipo local",
                    sort=win_rates_local,
                    axis=alt.Axis(labelAngle=270, labelFontSize=11)
                ),
                y=alt.Y(
                    "porcentaje:Q",
                    title="Porcentaje de resultados",
                    axis=alt.Axis(format='%')
                ),
                color=alt.Color(
                    "Resultado:N",
                    title="Resultado",
                    scale=alt.Scale(
                        domain=["Victoria", "Empate", "Derrota"],
                        range=["#1b9e77", "#c7b800", "#ff3737"]
                    )
                ),
                tooltip=[
                    alt.Tooltip("HomeTeam:N", title="Equipo local"),
                    alt.Tooltip("Resultado:N", title="Resultado"),
                    alt.Tooltip("count:Q", title="Cantidad", format="d"),
                    alt.Tooltip("porcentaje:Q", title="Porcentaje", format=".1%")
                ]
            )
            .properties(width=950, height=450)
        )
        st.altair_chart(chart_local, use_container_width=True)
    else:
        st.info("El dataset no contiene datos suficientes para equipos locales en esta liga.")

    # ===== Distribución de resultados por equipo visitante =====
    st.markdown('#### Resultados por equipo jugando como visitante')

    league_df_visit = league_df_global[
        league_df_global["AwayTeam"].notna() & (league_df_global["AwayTeam"] != "null")
    ].copy()

    if "AwayTeam" in league_df_visit.columns and "FTR" in league_df_visit.columns and not league_df_visit.empty:
        def resultado_visitante(row):
            if row["FTR"] == "A":
                return "Victoria"
            elif row["FTR"] == "D":
                return "Empate"
            else:
                return "Derrota"

        league_df_visit["Resultado"] = league_df_visit.apply(resultado_visitante, axis=1)

        win_rates_visit = (
            league_df_visit[league_df_visit["Resultado"] == "Victoria"]
            .groupby("AwayTeam")
            .size()
            .div(league_df_visit.groupby("AwayTeam").size())
            .sort_values(ascending=False)
            .index.tolist()
        )

        chart_visit = (
            alt.Chart(league_df_visit)
            .transform_aggregate(
                count="count()",
                groupby=["AwayTeam", "Resultado"]
            )
            .transform_joinaggregate(
                total_per_team="sum(count)",
                groupby=["AwayTeam"]
            )
            .transform_calculate(
                porcentaje="datum.count / datum.total_per_team"
            )
            .mark_bar()
            .encode(
                x=alt.X(
                    "AwayTeam:N",
                    title="Equipo visitante",
                    sort=win_rates_visit,
                    axis=alt.Axis(labelAngle=270, labelFontSize=11)
                ),
                y=alt.Y(
                    "porcentaje:Q",
                    title="Porcentaje de resultados",
                    axis=alt.Axis(format='%')
                ),
                color=alt.Color(
                    "Resultado:N",
                    title="Resultado",
                    scale=alt.Scale(
                        domain=["Victoria", "Empate", "Derrota"],
                        range=["#1b9e77", "#c7b800", "#ff3737"]
                    )
                ),
                tooltip=[
                    alt.Tooltip("AwayTeam:N", title="Equipo visitante"),
                    alt.Tooltip("Resultado:N", title="Resultado"),
                    alt.Tooltip("count:Q", title="Cantidad", format="d"),
                    alt.Tooltip("porcentaje:Q", title="Porcentaje", format=".1%")
                ]
            )
            .properties(width=950, height=450)
        )
        st.altair_chart(chart_visit, use_container_width=True)
    else:
        st.info("El dataset no contiene datos suficientes para equipos visitantes en esta liga.")

    # ===== Historial de enfrentamientos por equipo =====
    st.markdown('#### Historial de enfrentamientos por equipo')

    league_df = league_df_global[
        league_df_global["HomeTeam"].notna() &
        league_df_global["AwayTeam"].notna()
    ].copy()

    if league_df.empty:
        st.info("No hay datos de enfrentamientos para esta liga.")
        return

    equipos = sorted(
        pd.unique(
            league_df["HomeTeam"].dropna().tolist() +
            league_df["AwayTeam"].dropna().tolist()
        )
    )
    equipo_sel = st.selectbox("Seleccioná el equipo", equipos)

    team_matches = league_df[
        (league_df["HomeTeam"] == equipo_sel) |
        (league_df["AwayTeam"] == equipo_sel)
    ].copy()

    if not team_matches.empty:

        def resultado_vs(row):
            if row["HomeTeam"] == equipo_sel:
                if row["FTR"] == "H":
                    return "Victoria"
                elif row["FTR"] == "D":
                    return "Empate"
                else:
                    return "Derrota"
            else:
                if row["FTR"] == "A":
                    return "Victoria"
                elif row["FTR"] == "D":
                    return "Empate"
                else:
                    return "Derrota"

        def rival_de(row):
            return row["AwayTeam"] if row["HomeTeam"] == equipo_sel else row["HomeTeam"]

        team_matches["Resultado"] = team_matches.apply(resultado_vs, axis=1)
        team_matches["Rival"] = team_matches.apply(rival_de, axis=1)

        resumen = (
            team_matches.groupby(["Rival", "Resultado"])
            .size()
            .reset_index(name="Cantidad")
        )

        total_jugados = (
            resumen.groupby("Rival")["Cantidad"]
            .sum()
            .reset_index(name="Total")
        )

        resumen = resumen.merge(total_jugados, on="Rival")
        resumen["Porcentaje"] = resumen["Cantidad"] / resumen["Total"]

        porcentaje_victorias = (
            resumen[resumen["Resultado"] == "Victoria"]
            .set_index("Rival")["Porcentaje"]
            .reindex(resumen["Rival"].unique(), fill_value=0)
        )

        rivales_ordenados = porcentaje_victorias.sort_values(ascending=False).index.tolist()

        chart_historial = (
            alt.Chart(resumen)
            .mark_bar()
            .encode(
                x=alt.X(
                    "Rival:N",
                    sort=rivales_ordenados,
                    title="Rival",
                    axis=alt.Axis(labelAngle=270, labelFontSize=11)
                ),
                y=alt.Y(
                    "Porcentaje:Q",
                    title="Porcentaje de resultados",
                    axis=alt.Axis(format='%')
                ),
                color=alt.Color(
                    "Resultado:N",
                    title="Resultado",
                    scale=alt.Scale(
                        domain=["Victoria", "Empate", "Derrota"],
                        range=["#1b9e77", "#c7b800", "#ff3737"]
                    )
                ),
                tooltip=[
                    alt.Tooltip("Rival:N", title="Rival"),
                    alt.Tooltip("Resultado:N", title="Resultado"),
                    alt.Tooltip("Cantidad:Q", title="Cantidad", format="d"),
                    alt.Tooltip("Porcentaje:Q", title="Porcentaje", format=".1%"),
                    alt.Tooltip("Total:Q", title="Total jugados", format="d")
                ]
            )
            .properties(width=950, height=450)
        )

        st.altair_chart(chart_historial, use_container_width=True)

    else:
        st.info(f"No se encontraron partidos históricos del equipo {equipo_sel} en {league}.")

    

    # ===== Comparación entre dos equipos (Radar Chart general) =====
    st.markdown("### Comparador general entre equipos")

    # Usamos directamente el dataframe ya filtrado por liga
    league_df = league_df_global.copy()

    equipos = sorted(
        pd.unique(
            league_df["HomeTeam"].dropna().tolist() +
            league_df["AwayTeam"].dropna().tolist()
        )
    )

    colA, colB = st.columns(2)
    with colA:
        equipo_a = st.selectbox("Equipo A", equipos, key="radar_team_a")
    with colB:
        equipo_b = st.selectbox("Equipo B", equipos, key="radar_team_b")

    if equipo_a != equipo_b:

        # Función para extraer estadÃ­sticas del equipo (local/visitante)
        def stats_por_equipo(df, equipo):
            registros = []

            for _, row in df.iterrows():
                if row["HomeTeam"] == equipo:
                    registros.append({
                        "Team": equipo,
                        "GF": row["FTHG"],
                        "GA": row["FTAG"],
                        "Shots": row["HS"],
                        "ShotsOnTarget": row["HST"],
                        "Fouls": row["HF"],
                        "Corners": row["HC"],
                        "Yellow": row["HY"],
                        "Red": row["HR"]
                    })
                elif row["AwayTeam"] == equipo:
                    registros.append({
                        "Team": equipo,
                        "GF": row["FTAG"],
                        "GA": row["FTHG"],
                        "Shots": row["AS"],
                        "ShotsOnTarget": row["AST"],
                        "Fouls": row["AF"],
                        "Corners": row["AC"],
                        "Yellow": row["AY"],
                        "Red": row["AR"]
                    })

            df_stats = pd.DataFrame(registros)
            if df_stats.empty:
                return None

            return (
                df_stats.mean(numeric_only=True)
                .reset_index()
                .rename(columns={0: "Value"})
            )

        df_a = stats_por_equipo(league_df, equipo_a)
        df_b = stats_por_equipo(league_df, equipo_b)

        if df_a is None or df_b is None:
            st.info("No hay datos suficientes para ambos equipos en esta liga.")
        else:
            # Construimos tabla para radar
            radar_df = pd.concat([
                df_a.assign(Team=equipo_a),
                df_b.assign(Team=equipo_b)
            ])

            # Reemplazar NaN â fundamental para que el radar no colapse
            radar_df = radar_df.fillna(0)

            metric_labels = {
                "GF": "Goles a favor",
                "GA": "Goles en contra",
                "Shots": "Tiros",
                "ShotsOnTarget": "Tiros al arco",
                "Fouls": "Faltas",
                "Corners": "Córners",
                "Yellow": "Amarillas",
                "Red": "Rojas"
            }

            radar_df["MetricLabel"] = radar_df["index"].map(metric_labels)

            # Escala radial dinámica (visible en TODOS los casos)
            max_val = radar_df["Value"].max()
            if max_val < 5:
                max_val = 5
            elif max_val < 10:
                max_val = 10
            else:
                max_val = round(max_val + 2)

            # ===== Radar grande, visible y relleno =====
            radar_chart = (
                alt.Chart(radar_df, title="Comparativa promedio por partido")
                .mark_area(opacity=0.25)
                .encode(
                    theta=alt.Theta(
                        "MetricLabel:N",
                        sort=list(metric_labels.values())
                    ),
                    radius=alt.Radius(
                        "Value:Q",
                        scale=alt.Scale(domain=[0, max_val], nice=False)
                    ),
                    color=alt.Color("Team:N", title="Equipo"),
                    tooltip=[
                        alt.Tooltip("Team:N", title="Equipo"),
                        alt.Tooltip("MetricLabel:N", title="Métrica"),
                        alt.Tooltip("Value:Q", title="Prom. por partido", format=".2f"),
                    ]
                )
                .properties(
                    width=650,
                    height=650,
                    background="#f2f2f2",
                    padding={"left": 40, "right": 40, "top": 40, "bottom": 40}
                )
                .configure_view(stroke=None)
                .configure_title(
                    fontSize=22,
                    color="black",
                    anchor="start"
                )
            )

            st.altair_chart(radar_chart)

    else:
        st.info("ElegÃ­ dos equipos distintos para comparar.")





# ===== Seccion Datos API =====
def render_api_data():
    st.markdown('<div class="section-title">Tablas y predicción</div>', unsafe_allow_html=True)

    league_labels = [opt["label"] for opt in API_LEAGUES]
    label_to_code = {opt["label"]: opt["code"] for opt in API_LEAGUES}
    default_index = API_LEAGUE_CODE_INDEX.get(st.session_state.selected_api_league, 0)
    selected_label = st.selectbox(
        "Liga",
        league_labels,
        index=default_index,
        help="Datos provistos por el endpoint /competitions/{code}/standings",
    )
    selected_code = label_to_code[selected_label]
    st.session_state.selected_api_league = selected_code

    try:
        client = get_football_data_client()
    except FootballDataAPIError as exc:
        st.error(f"No se pudo inicializar el cliente del API: {exc}")
        st.info("Crea un token gratuito y configuralo como FOOTBALL_DATA_API_TOKEN.")
        return
    except Exception as exc:
        st.error(f"Error inesperado al inicializar el cliente: {exc}")
        return

    st.subheader("Tabla de posiciones")
    standings_df = pd.DataFrame()
    try:
        with st.spinner("Descargando posiciones desde Football-Data.org..."):
            payload = client.get_standings(selected_code)
        standings_df = standings_to_dataframe(payload)
    except FootballDataAPIError as exc:
        st.error(f"No se pudo obtener la tabla de posiciones: {exc}")
    except Exception as exc:
        st.error(f"Error inesperado al consultar standings: {exc}")

    standings_payload = locals().get('payload') if 'payload' in locals() else None
    if standings_payload is None:
        st.error("No se pudieron recuperar los standings para esta liga.")
        return

    if standings_df.empty:
        st.info("No hay datos de posiciones disponibles para esta competencia.")
    else:
        standings_display = standings_df.rename(
            columns={
                "position": "Pos",
                "team": "Equipo",
                "played": "PJ",
                "won": "G",
                "draw": "E",
                "lost": "P",
                "goals_for": "GF",
                "goals_against": "GC",
                "goal_diff": "DG",
                "points": "Pts",
            }
        )[["Pos", "Equipo", "PJ", "G", "E", "P", "GF", "GC", "DG", "Pts"]]
        st.dataframe(standings_display, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Predecir próxima fecha")
    matches_df = pd.DataFrame()
    try:
        with st.spinner("Cargando la ultima jornada disputada..."):
            matches_df = latest_matchday_df(client, selected_code)
    except FootballDataAPIError as exc:
        st.error(f"No se pudieron obtener los partidos: {exc}")
        return
    except Exception as exc:
        st.error(f"Error inesperado al consultar partidos: {exc}")
        return

    if matches_df.empty:
        st.info("Aun no hay partidos finalizados para mostrar.")
        return
    matches_df = matches_df.copy()
    matches_df['utc_date'] = pd.to_datetime(matches_df['utc_date'], errors='coerce')
    matchday = matches_df['matchday'].dropna().max()
    if pd.notna(matchday):
        st.caption(f"Jornada {int(matchday)} - Fuente: Football-Data.org")

    def _format_date(dt_value):
        if pd.isna(dt_value):
            return "-"
        return dt_value.strftime("%d/%m/%Y %H:%M")

    def _render_logo(column, url: Optional[str]):
        if url:
            column.image(url, width=38)
        else:
            column.write("-")

    header_cols = st.columns([1.6, 0.7, 0.7, 2.2, 0.7, 2.2, 0.6])
    header_cols[0].markdown("**Fecha**")
    header_cols[1].markdown("**Jornada**")
    header_cols[2].write("")
    header_cols[3].markdown("**Local**")
    header_cols[4].write("")
    header_cols[5].markdown("**Visitante**")
    header_cols[6].markdown("**Predecir**")

    for _, row in matches_df.sort_values('utc_date').iterrows():
        col_fecha, col_matchday, col_home_logo, col_home_name, col_away_logo, col_away_name, col_btn = st.columns(
            [1.6, 0.7, 0.7, 2.2, 0.7, 2.2, 0.6]
        )
        col_fecha.markdown(f"**{_format_date(row.get('utc_date'))}**")
        md_label = f"{int(row['matchday'])}" if pd.notna(row.get('matchday')) else "MD -"
        col_matchday.markdown(md_label)
        _render_logo(col_home_logo, row.get('home_team_logo'))
        col_home_name.markdown(f"**{row.get('home_team', 'N/D')}**")
        _render_logo(col_away_logo, row.get('away_team_logo'))
        col_away_name.markdown(row.get('away_team', 'N/D'))
        if col_btn.button('✳️', key=f"predict_{row.get('match_id')}", help='Calcular prediccion para este partido'):
            try:
                prediction = predict_match_from_api(row, standings_payload, selected_code, selected_label)
                st.session_state.last_api_prediction = prediction
                st.success(f"Prediccion lista para {row.get('home_team')} vs {row.get('away_team')}")
            except Exception as exc:
                st.error(f"No se pudo generar la prediccion: {exc}")

    last_prediction = st.session_state.get('last_api_prediction')
    if last_prediction:
        st.markdown("#### Ultima prediccion generada")
        st.write(f"{last_prediction.get('home', '-')} vs {last_prediction.get('away', '-')} (Jornada {last_prediction.get('matchday', '-')})")
        st.dataframe(last_prediction['probabilities'], use_container_width=True, hide_index=True)
        if last_prediction.get('label'):
            st.caption(f"Resultado estimado: {last_prediction.get('label')}")

def render_prediccion_manual():
    st.markdown('<div class="section-title">Prediccion manual</div>', unsafe_allow_html=True)
    pipeline = get_prediction_pipeline()
    if pipeline is None:
        st.error(st.session_state.pipeline_error or 'No se pudo cargar el modelo para predecir.')
        return
    tabs = st.tabs(['CSV', 'Manual'])

    with tabs[0]:
        st.subheader('Cargar CSV con columnas crudas')
        data_file = st.file_uploader('Subí tu archivo CSV', type=['csv'])
        if data_file:
            df_pred = preprocessCSV(pd.read_csv(data_file))
            st.dataframe(df_pred.head())
            if st.button('Predecir desde CSV'):
                probas = pipeline.predict_proba(df_pred)
                classes = pipeline.named_steps['model'].classes_
                prob_table = build_prob_table(probas, classes)
                st.dataframe(prob_table.style.format('{:.2%}'))

    with tabs[1]:
        render_manual_form(pipeline)

    st.markdown('---')
    st.warning('Esta herramienta tiene fines educativos. No promueve apuestas deportivas.')

# ===== Sección About =====
def render_about():
    st.markdown('<div class="section-title">Acerca del proyecto</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="card-about">
<b>Footy Predictor</b> es una aplicación educativa para el análisis y predicción de partidos de fútbol en las 5 grandes ligas europeas.  
Utiliza Machine Learning (Logistic Regression + PCA) y visualizaciones interactivas para explorar patrones en los resultados.
</div>

<div class="card-about">
<b> Objetivo</b><br>
Analizar datos reales del fútbol europeo y generar predicciones H/D/A (local, empate, visitante) de manera automatizada.
</div>

<div class="card-about">
<b> Tecnologí­as</b>
<ul>
<li>Python 3.11</li>
<li>Streamlit</li>
<li>Scikit-learn</li>
<li>Pandas / Numpy</li>
<li>Altair</li>
</ul>
</div>

<div class="card-about">
<b> Autores</b><br>
Diego Páez y Nicolás Carcaño  
UTN - Facultad Regional Mendoza (2025)
</div>
""", unsafe_allow_html=True)
    st.markdown('<div class="footer">© 2025 Footy Predictor - Proyecto académico</div>', unsafe_allow_html=True)

# ===== Layout =====
st.markdown('<div class="header"><h1>Footy Predictor</h1><hr></div>', unsafe_allow_html=True)
with st.sidebar:
    st.markdown("### Navegación")
    page = st.radio("", [ "Tablas y predicción", "Visualizaciones", "Informe de modelos", "Predicción manual", "Acerca"])
    st.session_state.selected_page = page

if st.session_state.selected_page == "Visualizaciones":
    render_ligas()
elif st.session_state.selected_page == "Tablas y predicción":
    render_api_data()
elif st.session_state.selected_page == "Informe de modelos":
    render_model_report()
elif st.session_state.selected_page == "Predicción manual":
    render_prediccion_manual()
elif st.session_state.selected_page == "Acerca":
    render_about()
