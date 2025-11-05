import pandas as pd
import numpy as np

# @title
def limpiar_csv(df: pd.DataFrame):
  # 1. Eliminar columnas completamente vacías
  df = df.dropna(axis=1, how='all')

  # 2. Eliminar filas duplicadas
  df = df.drop_duplicates()

  # 3. Corregir fechas
  # ✅ Guardamos la fecha original por seguridad
  df["Date_raw"] = df["Date"]

  # ✅ Intentamos primero con formato dd/mm/yyyy
  df["Date"] = pd.to_datetime(df["Date_raw"], format="%d/%m/%Y", errors="coerce")

  # ✅ Las que fallaron (formato dd/mm/yy) las parseamos acá:
  mask = df["Date"].isna()
  df.loc[mask, "Date"] = pd.to_datetime(df.loc[mask, "Date_raw"], format="%d/%m/%y", errors="coerce")

  # ✅ Devolvemos todo a formato string dd/mm/yyyy
  df["Date"] = df["Date"].dt.strftime("%d/%m/%Y")

  return df

# @title
def eliminar_cols_muchos_nulos(df: pd.DataFrame, umbral=0.40, columnas=None) -> pd.DataFrame:
    """
    Elimina columnas con proporción de nulos > umbral.
    - umbral: 0.40 => 40%
    - columnas: si pasás una lista, solo evalúa esas columnas (p. ej., las de apuestas).
    """
    cols = columnas if columnas is not None else df.columns
    frac_nulos = df[cols].isna().mean()  # proporción de nulos por columna (0..1)
    a_borrar = frac_nulos[frac_nulos > umbral].index.tolist()

    if a_borrar:
        print(f"Columnas con >{int(umbral*100)}% nulos ({len(a_borrar)}):")
        print(a_borrar)
        df = df.drop(columns=a_borrar)
    else:
        print(f"No hay columnas con más de {int(umbral*100)}% de nulos en el conjunto evaluado.")

    return df

# @title
def OrganizarTemporadas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea dos columnas season_start y season_end a partir de la columna Date.
    La temporada va de julio (inclusive) a junio (exclusive) del año siguiente.
    """
    if "Date" not in df.columns:
        print("⚠️ No se encontró la columna 'Date'. No se generaron las temporadas.")
        return df

    # Asegurarse de que Date sea datetime
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

    # Crear columnas de inicio y fin de temporada
    df["season_start"] = df["Date"].apply(
        lambda d: d.year if pd.notna(d) and d.month >= 7 else (d.year - 1 if pd.notna(d) else pd.NA)
    )
    df["season_end"] = df["season_start"] + 1

    return df

# @title
def ordenarDataFrame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ordena el DataFrame por:
      1. División (columna 'Div') — ascendente
      2. Temporada (columna 'season_start') — descendente
      3. Fecha (columna 'Date') — ascendente
    """
    # Validar columnas
    columnas_necesarias = ["Div", "season_start", "Date"]
    faltantes = [c for c in columnas_necesarias if c not in df.columns]
    if faltantes:
        print(f"⚠️ Faltan columnas para ordenar: {faltantes}")
        return df

    # Asegurar tipo datetime
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

    # Ordenar según criterios
    df = df.sort_values(
        by=["Div", "season_start", "Date", "HomeTeam"],
        ascending=[True, False, True, True] 
    ).reset_index(drop=True)

    return df

# @title
def agregar_matchday(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea:
      - team_match_no: # acumulado de partidos por equipo (local+visita) en cada (Div, season_start)
      - matchday: jornada del encuentro = max(team_match_no_home, team_match_no_away)
    Requiere columnas: Div, season_start, Date, HomeTeam, AwayTeam.
    """
    req = ["Div", "season_start", "Date", "HomeTeam", "AwayTeam"]
    faltantes = [c for c in req if c not in df.columns]
    if faltantes:
        print(f"⚠️ Faltan columnas necesarias: {faltantes}")
        return df

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.sort_values(["Div", "season_start", "Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)

    # Identificador estable del partido para poder volver a unir
    df["match_id"] = df.index

    # Formato largo: una fila por (partido, equipo)
    home_long = df[["match_id", "Div", "season_start", "Date", "HomeTeam"]].rename(columns={"HomeTeam":"Team"})
    away_long = df[["match_id", "Div", "season_start", "Date", "AwayTeam"]].rename(columns={"AwayTeam":"Team"})
    long = pd.concat([home_long.assign(side="H"), away_long.assign(side="A")], ignore_index=True)

    # Conteo acumulado de partidos por equipo (local+visita)
    long = long.sort_values(["Div", "season_start", "Team", "Date", "match_id"])
    long["team_match_no"] = long.groupby(["Div", "season_start", "Team"]).cumcount() + 1

    # Volver a la tabla de partidos: obtener el nro de partido de cada equipo en ese encuentro
    home_counts = long[long["side"]=="H"][["match_id", "team_match_no"]].rename(columns={"team_match_no":"team_match_no_home"})
    away_counts = long[long["side"]=="A"][["match_id", "team_match_no"]].rename(columns={"team_match_no":"team_match_no_away"})
    df = df.merge(home_counts, on="match_id", how="left").merge(away_counts, on="match_id", how="left")

    # Jornada = max de los # acumulados de ambos equipos (respeta reprogramaciones)
    df["matchday"] = df[["team_match_no_home", "team_match_no_away"]].max(axis=1).astype("Int64")

    # (Opcional) chequeo de consistencia por liga-temporada
    def _check(group):
        teams = pd.unique(pd.concat([group["HomeTeam"], group["AwayTeam"]], ignore_index=True))
        n = len(teams)
        expected = 2 * (n - 1)
        real_max = int(group["matchday"].max()) if not group["matchday"].isna().all() else None
        if real_max is not None and real_max != expected:
            print(f"⚠️ {group.name}: equipos={n}, esperado={expected}, max(matchday)={real_max}")
        return group

    df = df.groupby(["Div", "season_start"], group_keys=False).apply(_check)

    # Limpieza opcional
    df = df.drop(columns=["match_id"])
    return df

# @title
def _pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

# --------------------------
# Tabla por equipo (Home/Away)
# --------------------------
def _to_team_long(df: pd.DataFrame):
    """
    Devuelve tabla 'long' por equipo con:
      Div, season_start, Date, match_id, Team, side(H/A), GF, GA, result(W/D/L)
    Soporta FTHG/FTAG ~ HG/AG y FTR ~ Res.
    """
    df = df.copy()
    df = df.sort_values(["Div", "season_start", "Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)
    df["match_id"] = df.index

    # Detectar columnas de goles / resultado
    col_hg  = _pick_col(df, ["FTHG", "HG"])
    col_ag  = _pick_col(df, ["FTAG", "AG"])
    col_res = _pick_col(df, ["FTR", "Res"])
    req = ["Div", "season_start", "Date", "HomeTeam", "AwayTeam", col_hg, col_ag, col_res]
    if any(c is None for c in [col_hg, col_ag, col_res]):
        faltantes = [("FTHG/HG" if col_hg is None else None),
                     ("FTAG/AG" if col_ag is None else None),
                     ("FTR/Res" if col_res is None else None)]
        raise ValueError(f"Faltan columnas de goles/resultado: {list(filter(None, faltantes))}")

    # Partimos en dos filas (local, visita)
    base_cols = ["match_id", "Div", "season_start", "Date", col_hg, col_ag, col_res]
    home = (
        df[["HomeTeam"] + base_cols]
          .rename(columns={"HomeTeam":"Team"})
          .assign(side="H",
                  GF=lambda x: x[col_hg],
                  GA=lambda x: x[col_ag])
    )
    away = (
        df[["AwayTeam"] + base_cols]
          .rename(columns={"AwayTeam":"Team"})
          .assign(side="A",
                  GF=lambda x: x[col_ag],
                  GA=lambda x: x[col_hg])
    )

    long = pd.concat([home, away], ignore_index=True)
    long = long.rename(columns={col_res:"Res"})
    long["Res"] = long["Res"].astype(str).str.upper().str.strip()

    # Resultado desde la perspectiva del equipo
    # Si Res == 'H' => gana el local -> team gana si side=='H'
    # Si Res == 'A' => gana el visitante -> team gana si side=='A'
    cond_win  = ((long["Res"] == "H") & (long["side"] == "H")) | ((long["Res"] == "A") & (long["side"] == "A"))
    cond_draw = (long["Res"] == "D")
    long["result_team"] = np.select([cond_win, cond_draw], ["W", "D"], default="L")

    # Orden estable por equipo-temporada
    long = long.sort_values(["Div", "season_start", "Team", "Date", "match_id"]).reset_index(drop=True)
    return long

# @title
def _cumulative_features(long: pd.DataFrame, include_current: bool = False):
    """
    Corrige los acumulados para que los valores *_pre representen
    el estado ANTES del partido actual.
    """
    long = long.copy()
    grp = long.groupby(["Div", "season_start", "Team"], sort=False)

    long["is_win"]  = (long["result_team"] == "W").astype(int)
    long["is_draw"] = (long["result_team"] == "D").astype(int)
    long["is_loss"] = (long["result_team"] == "L").astype(int)
    long["is_home"] = (long["side"] == "H").astype(int)
    long["is_away"] = (long["side"] == "A").astype(int)

    suf = "post" if include_current else "pre"

    def _cum_and_shift(series):
        csum = series.cumsum()
        return csum if include_current else csum.shift(1, fill_value=0)

    # Totales
    for name, expr in {
        "wins": "is_win",
        "draws": "is_draw",
        "losses": "is_loss",
        "goals_for": "GF",
        "goals_against": "GA"
    }.items():
        long[f"{name}_{suf}"] = grp[expr].transform(_cum_and_shift)

    # Condicionales (local/visitante)
    specs = {
        "wins_home":     lambda g: g["is_win"]  * g["is_home"],
        "wins_away":     lambda g: g["is_win"]  * g["is_away"],
        "draws_home":    lambda g: g["is_draw"] * g["is_home"],
        "draws_away":    lambda g: g["is_draw"] * g["is_away"],
        "losses_home":   lambda g: g["is_loss"] * g["is_home"],
        "losses_away":   lambda g: g["is_loss"] * g["is_away"],
        "goals_for_home":     lambda g: g["GF"] * g["is_home"],
        "goals_for_away":     lambda g: g["GF"] * g["is_away"],
        "goals_against_home": lambda g: g["GA"] * g["is_home"],
        "goals_against_away": lambda g: g["GA"] * g["is_away"],
    }

    for name, fn in specs.items():
        long[f"{name}_{suf}"] = grp.apply(lambda g: _cum_and_shift(fn(g))).reset_index(level=[0,1,2], drop=True)

    return long

# @title
def _streaks(long: pd.DataFrame, include_current: bool = False):
    """
    Rachas consecutivas por equipo (Div, season_start, Team) con shift(1) *por grupo*.
    """
    long = long.copy()
    grp = long.groupby(["Div", "season_start", "Team"], sort=False)
    suf = "post" if include_current else "pre"

    def _streak_for_code(series, code):
        out = np.zeros(len(series), dtype=int)
        c = 0
        for i, v in enumerate(series):
            if v == code:
                c += 1
            else:
                c = 0
            out[i] = c
        return pd.Series(out, index=series.index)

    for code in ["W", "D", "L"]:
        st = grp["result_team"].apply(lambda s: _streak_for_code(s, code))
        st.index = st.index.droplevel([0,1,2])
        if include_current:
            long[f"streak_{code}_{suf}"] = st
        else:
            st_pre = grp["result_team"].apply(lambda s: _streak_for_code(s, code).shift(1).fillna(0))
            st_pre.index = st_pre.index.droplevel([0,1,2])
            long[f"streak_{code}_{suf}"] = st_pre

    return long

# @title
# --------------------------
# VOLVER a la tabla de partidos
# --------------------------
def _merge_back(df_matches: pd.DataFrame, long_feats: pd.DataFrame, include_current: bool = False):
    """
    Fusiona las features de 'long' a la tabla de partidos, creando columnas para HOME y AWAY.
    Sufijos: _home/_away y *_pre (o *_post).
    """
    df = df_matches.copy()
    suf = "post" if include_current else "pre"

    # Subconjuntos home/away con columnas de interés
    cols_keep = ["match_id", "Team", "side"] + [c for c in long_feats.columns if c.endswith(f"_{suf}")]
    feats = long_feats[cols_keep].copy()

    # HOME
    home_feats = feats[feats["side"]=="H"].drop(columns=["side"]).rename(columns=lambda c: c if c in ["match_id","Team"] else c.replace(f"_{suf}", f"_home_{suf}"))
    home_feats = home_feats.rename(columns={"Team":"HomeTeam"})
    df = df.merge(home_feats, on=["match_id", "HomeTeam"], how="left")

    # AWAY
    away_feats = feats[feats["side"]=="A"].drop(columns=["side"]).rename(columns=lambda c: c if c in ["match_id","Team"] else c.replace(f"_{suf}", f"_away_{suf}"))
    away_feats = away_feats.rename(columns={"Team":"AwayTeam"})
    df = df.merge(away_feats, on=["match_id", "AwayTeam"], how="left")

    return df

# @title
# --------------------------
# FUNCIÓN COMBINADA PARA EL PIPELINE
# --------------------------
def agregar_estadisticas_equipo_temporada(df: pd.DataFrame, include_current: bool = False) -> pd.DataFrame:
    """
    Agrega al DataFrame de partidos columnas con:
      - victorias/empates/derrotas acumuladas (totales y por local/visitante)
      - goles a favor/contra acumulados (totales y por local/visitante)
      - rachas de V/E/D
    Todas las métricas se calculan por (Div, season_start, Team), orden cronológico.
    Por defecto devuelve valores 'pre' (antes del partido actual).
    Cambiar include_current=True para valores 'post' (incluyendo el partido actual).
    """
    req = ["Div", "season_start", "Date", "HomeTeam", "AwayTeam"]
    faltantes = [c for c in req if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {faltantes}")

    df = df.sort_values(["Div", "season_start", "Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)
    df["match_id"] = df.index

    long = _to_team_long(df)
    long = _cumulative_features(long, include_current=include_current)
    long = _streaks(long, include_current=include_current)

    out = _merge_back(df, long, include_current=include_current)
    # If you don't want to keep match_id, drop it here or later
    # out = out.drop(columns=["match_id"])

    return out

# @title
def agregar_puntos_y_partidos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Usa columnas *_pre ya creadas (wins/draws/losses split home/away)
    para derivar partidos jugados y puntos acumulados (previos al partido actual).
    Requiere: wins_home_pre, draws_home_pre, losses_home_pre,
              wins_away_pre, draws_away_pre, losses_away_pre.
    """
    req = [
        "wins_home_pre","draws_home_pre","losses_home_pre",
        "wins_away_pre","draws_away_pre","losses_away_pre",
    ]
    faltan = [c for c in req if c not in df.columns]
    if faltan:
        raise ValueError(f"Faltan columnas acumuladas para calcular puntos/PJ: {faltan}")

    out = df.copy()

    # Partidos jugados (pre) = W + D + L
    out["matches_home_pre"] = (
        out["wins_home_pre"] + out["draws_home_pre"] + out["losses_home_pre"]
    ).astype("Int64")
    out["matches_away_pre"] = (
        out["wins_away_pre"] + out["draws_away_pre"] + out["losses_away_pre"]
    ).astype("Int64")
    out["matches_pre"] = (out["matches_home_pre"] + out["matches_away_pre"]).astype("Int64")

    # Puntos (pre) = 3*W + 1*D
    out["points_home_pre"] = (3*out["wins_home_pre"] + out["draws_home_pre"]).astype("Int64")
    out["points_away_pre"] = (3*out["wins_away_pre"] + out["draws_away_pre"]).astype("Int64")
    out["points_pre"] = (out["points_home_pre"] + out["points_away_pre"]).astype("Int64")

    return out

def select_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Selecciona las variables predictoras (features) y las clasifica en categóricas y numéricas.
    Elimina columnas "posteriores" al partido, de cuotas, y otras no predictivas directas.

    Args:
        df: DataFrame de entrada.

    Returns:
        Una tupla conteniendo:
        - DataFrame con las columnas seleccionadas.
        - Lista de nombres de columnas categóricas.
        - Lista de nombres de columnas numéricas.
    """
    df = df.copy()

    # 1. Columnas que nunca deben usarse porque son "posteriores" al partido
    cols_to_drop = [
        "FTR", "HTR",  # resultados finales y al medio tiempo
        "FTHG", "FTAG", "HTHG", "HTAG",  # goles (información del futuro)
        "Date", "Time",  # fechas/hora no predictivas directas
        "Referee",  # árbitro: podría influir, pero no lo usamos por simplicidad
        "HS", "AST", "HST", "HHW", "AHW", "HC", "AC", "HF", "AF", "HFKC", "AFKC",
        "HO", "AO", "HY", "AY", "AS", "HR", "AR", "HBP", "ABP"  # Stats del partido
    ]

    # 2. Detectar columnas relacionadas con cuotas o probabilidades de casas de apuestas
    odds_keywords = ["B365", "BWD", "BWA", "WH", "VC", "PS", "Bb", "Max", "Avg", "GB", "IWH", "IWD", "IWA", "LB", "SB", "Pinnacle", "BFDH", "BFDD", "BFDA", "BMGMH", "BMGMD", "BMGMA", "BVH", "BVD", "BVA", "CLH", "CLD", "CLA", "MaxH", "MaxD", "MaxA", "AvgH", "AvgD", "AvgA", "BFEH", "BFED", "BFEA", "B365>2.5", "B365<2.5", "P>2.5", "P<2.5", "Max>2.5", "Max<2.5", "Avg>2.5", "Avg<2.5", "BFE>2.5", "BFE<2.5", "AHh", "B365AHH", "B365AHA", "PAHH", "PAHA", "MaxAHH", "MaxAHA", "AvgAHH", "AvgAHA", "BFEAHH", "BFEAHA", "B365CH", "B365CD", "B365CA", "BFDCH", "BFDCD", "BFDCA", "BMGMCH", "BMGMCD", "BMGMCA", "BVCH", "BVCD", "BVCA", "BWCH", "BWCD", "BWCA", "CLCH", "CLCD", "CLCA", "LBCH", "LBCD", "LBCA", "MaxCH", "MaxCD", "MaxCA", "AvgCH", "AvgCD", "AvgCA", "BFECH", "BFECD", "BFECA", "B365C>2.5", "B365C<2.5", "PC>2.5", "PC<2.5", "MaxC>2.5", "MaxC<2.5", "AvgC>2.5", "AvgC<2.5", "BFEC>2.5", "BFEC<2.5", "AHCh", "B365CAHH", "B365CAHA", "PCAHH", "PCAHA", "MaxCAHH", "MaxCAHA", "AvgCAHH", "AvgCAHA", "BFECAHH", "BFECAHA", "BFH", "BFD", "BFA", "1XBH", "1XBD", "1XBA", "BFCH", "BFCD", "BFCA", "WHCH", "WHCD", "WHCA", "1XBCH", "1XBCD", "1XBCA", "IWCH", "IWCD", "IWCA", "VCCH", "VCCD", "VCCA", "SJH", "SJD", "SJA", "GBH", "GBD", "GBA", "BSH", "BSD", "BSA", "SBH", "SBD", "SBA"]
    odds_cols = [c for c in df.columns if any(k in c for k in odds_keywords)]

    # Agregamos a la lista general de exclusión
    cols_to_drop.extend(odds_cols)

    # Eliminamos columnas del conjunto de entrenamiento y test
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

    return df


def preprocessCSV(df: pd.DataFrame):

    df = (
        df
        .pipe(limpiar_csv)
        .pipe(eliminar_cols_muchos_nulos)
        .pipe(OrganizarTemporadas)
        .pipe(ordenarDataFrame)
        .pipe(agregar_matchday)
        .pipe(agregar_estadisticas_equipo_temporada)
        .pipe(agregar_puntos_y_partidos)
        .pipe(select_features)
    )

    return df