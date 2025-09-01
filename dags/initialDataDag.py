from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import json
import time
import io
import requests
import pandas as pd

# ---------- Config ----------
SOURCE_PATH = 'https://www.football-data.co.uk/mmz4281'
leaguesPath = {
    'England' : '/E0',   # Premier League
    'France'  : '/F1',   # Ligue 1
    'Spain'   : '/SP1',  # LaLiga
    'Italy'   : '/I1',   # Serie A
    'Germany' : '/D1',   # Bundesliga
}

# Temporadas (22/23, 23/24, 24/25)
SEASONS = ['2223', '2324', '2425']

# Dónde guardar
DEFAULT_OUTPUT_DIR = "/tmp/football_data"


# ---------- Utilidades ----------
def _safe_read_csv(content_bytes):
    """
    Lee CSV en pandas desde bytes probando encodings y separadores típicos.
    """
    for enc in ("latin1", "utf-8", "cp1252"):
        try:
            # football-data suele usar coma como separador
            df = pd.read_csv(io.BytesIO(content_bytes), encoding=enc)
            return df
        except Exception:
            continue
    # último intento: especificar sep="," explícito
    df = pd.read_csv(io.BytesIO(content_bytes), encoding="latin1", sep=",", engine="python")
    return df


def _clean_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Normaliza columnas (strip)
    - Intenta parsear 'Date' a datetime (dayfirst=True)
    - Agrega 'match_date' en ISO (YYYY-MM-DD)
    - Ordena por fecha, HomeTeam, AwayTeam (si existen)
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Parseo robusto de la fecha
    if "Date" in df.columns:
        # algunos CSV tienen dd/mm/yy o dd/mm/yyyy
        df["__dt"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    else:
        df["__dt"] = pd.NaT

    # match_date ISO para downstream
    df["match_date"] = df["__dt"].dt.strftime("%Y-%m-%d")

    # Orden seguro (si no existen columnas, no rompe)
    sort_cols = [c for c in ["__dt", "HomeTeam", "AwayTeam"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, na_position="last").reset_index(drop=True)

    # limpiar helper
    df = df.drop(columns=["__dt"], errors="ignore")
    return df


def _download_with_retries(url: str, retries: int = 3, delay: float = 1.5) -> bytes:
    """
    Descarga con pequeños reintentos.
    """
    last_exc = None
    for i in range(retries):
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            return r.content
        except Exception as e:
            last_exc = e
            time.sleep(delay)
    raise last_exc


# ---------- Tarea principal ----------
def fetch_eu_top5_last3_seasons(output_dir: str = DEFAULT_OUTPUT_DIR, **kwargs):
    """
    Descarga CSVs de 5 grandes ligas (E0, F1, SP1, I1, D1) para temporadas 2223, 2324, 2425.
    Guarda:
      - raw: output_dir/<Liga>/<Temporada>/<code>.csv
      - clean: output_dir/<Liga>/<Temporada>/<code>_sorted.csv
    Crea:
      - manifest.json con detalle de archivos
      - consolidated por liga y total (opcional, si las columnas permiten concatenar)
    """
    os.makedirs(output_dir, exist_ok=True)
    manifest = {"downloaded": [], "errors": []}

    per_league_frames = {lg: [] for lg in leaguesPath.keys()}
    all_frames = []

    for league_name, league_code in leaguesPath.items():
        for season in SEASONS:
            code = league_code.strip("/").upper()  # E0, F1, SP1, I1, D1
            url = f"{SOURCE_PATH}/{season}/{code}.csv"
            league_dir = os.path.join(output_dir, league_name, season)
            os.makedirs(league_dir, exist_ok=True)

            raw_path = os.path.join(league_dir, f"{code}.csv")
            sorted_path = os.path.join(league_dir, f"{code}_sorted.csv")

            try:
                content = _download_with_retries(url)
                # Guardar raw
                with open(raw_path, "wb") as f:
                    f.write(content)

                # Leer y ordenar
                df = _safe_read_csv(content)
                df = _clean_and_sort(df)

                # Guardar ordenado
                df.to_csv(sorted_path, index=False)

                # Para consolidado por liga y global (si tiene columnas mínimas razonables)
                if not df.empty:
                    df["league"] = league_name
                    df["season"] = season
                    per_league_frames[league_name].append(df)
                    all_frames.append(df)

                manifest["downloaded"].append({
                    "league": league_name,
                    "season": season,
                    "code": code,
                    "url": url,
                    "raw_path": raw_path,
                    "sorted_path": sorted_path,
                    "rows": int(df.shape[0]),
                    "cols": int(df.shape[1]),
                })

                print(f"[OK] {league_name} {season} -> {sorted_path}")

            except Exception as e:
                manifest["errors"].append({
                    "league": league_name,
                    "season": season,
                    "code": code,
                    "url": url,
                    "error": str(e),
                })
                print(f"[ERR] {league_name} {season}: {e}")

    # Guardar manifest
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Manifest -> {manifest_path}")

    # Consolidado por liga
    for lg, frames in per_league_frames.items():
        if frames:
            try:
                df_ = pd.concat(frames, ignore_index=True, sort=False)
                outp = os.path.join(output_dir, lg, f"{lg}_consolidated_5leagues_3seasons.csv")
                df_.to_csv(outp, index=False)
                print(f"[INFO] Consolidado {lg} -> {outp}")
            except Exception as e:
                print(f"[WARN] No se pudo consolidar {lg}: {e}")

    # Consolidado total
    if all_frames:
        try:
            df_all = pd.concat(all_frames, ignore_index=True, sort=False)
            outp_all = os.path.join(output_dir, "ALL_top5_3seasons_consolidated.csv")
            df_all.to_csv(outp_all, index=False)
            print(f"[INFO] Consolidado TOTAL -> {outp_all}")
        except Exception as e:
            print(f"[WARN] No se pudo consolidar TOTAL: {e}")


# ---------- DAG ----------
default_args = {
    'owner': 'Diego',
    'start_date': datetime.today() - timedelta(days=1),
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
    'email_on_failure': False,
    'depends_on_past': False,
}

with DAG(
    dag_id="initial_data_dag",
    description="DAG para la carga inicial de datos",
    default_args=default_args,
    schedule=None,
    catchup=False,
    tags=['Inicio', 'Carga Inicial']
) as dag:

    fetch_top5_data = PythonOperator(
        task_id="fetch_eu_top5_last3_seasons",
        python_callable=fetch_eu_top5_last3_seasons,
        op_kwargs={
            "output_dir": "/tmp/football_data",   # podés templatar si querés
        },
    )
