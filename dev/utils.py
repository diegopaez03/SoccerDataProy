from typing import List, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
import pandas as pd

class LegacySelectPreprocessPCA(BaseEstimator, TransformerMixin):
    """
    Reusa TUS funciones para que el preprocesamiento viva dentro de un Pipeline sklearn.
    - fit(X): usa select_features(X) y luego preprocess_features(X, X, ...) para
      AJUSTAR (imputers, scaler, OHE) y guardar los objetos ajustados.
      Si use_pca=True, llama apply_pca(X_base, X_base) para ajustar PCA y guardar el objeto.
    - transform(X): aplica los objetos guardados para transformar NUEVOS datos,
      y si corresponde, aplica PCA.transform.
    """
    def __init__(self, use_pca: bool = False, pca_n_components: float | int = 0.90):
        self.use_pca = use_pca
        self.pca_n_components = pca_n_components

        # atributos poblados en fit
        self.categorical_features_: List[str] = []
        self.numeric_features_: List[str] = []
        self.feature_names_base_: np.ndarray | None = None
        self.ohe_ = None
        self.num_imputer_ = None
        self.num_scaler_ = None
        self.cat_imputer_ = None
        self.pca_ = None
        self._fitted_ = False

    def fit(self, X: pd.DataFrame, y=None):
        # 1) seleccionar features con tu función
        X_sel, cats, nums = select_features(X)
        self.categorical_features_ = list(cats)
        self.numeric_features_ = list(nums)

        # 2) reusar tu preprocess_features para AJUSTAR y obtener objetos
        #    (le pasamos X como "train" y también como "test" solo para que te devuelva
        #     los objetos ya ajustados; ignoramos la matriz "test" que retorna)
        Xtr_base, _, feature_names_base, ohe, num_imputer, num_scaler, cat_imputer = preprocess_features(
            X_sel, X_sel, self.categorical_features_, self.numeric_features_
        )
        self.feature_names_base_ = feature_names_base
        self.ohe_ = ohe
        self.num_imputer_ = num_imputer
        self.num_scaler_ = num_scaler
        self.cat_imputer_ = cat_imputer

        # 3) opcional PCA usando tu apply_pca para AJUSTAR el objeto PCA
        if self.use_pca:
            Xtr_pca, _, pca = apply_pca(Xtr_base, Xtr_base, n_components=self.pca_n_components)
            self.pca_ = pca

        self._fitted_ = True
        return self

    def transform(self, X: pd.DataFrame):
        if not self._fitted_:
            raise RuntimeError("Debe llamarse fit antes de transform.")
        # Seleccionar mismas columnas en el MISMO orden
        X_sel = X[self.numeric_features_ + self.categorical_features_].copy()

        # --- Numéricas: imputer -> (scaler opcional si fue ajustado)
        X_num = None
        if len(self.numeric_features_) > 0:
            X_num = self.num_imputer_.transform(X_sel[self.numeric_features_])
            if self.num_scaler_ is not None:
                X_num = self.num_scaler_.transform(X_num)
        else:
            X_num = np.empty((len(X_sel), 0))

        # --- Categóricas: imputer -> OHE
        X_cat = None
        if len(self.categorical_features_) > 0:
            X_cat_imp = self.cat_imputer_.transform(X_sel[self.categorical_features_])
            X_cat = self.ohe_.transform(X_cat_imp)
        else:
            X_cat = np.empty((len(X_sel), 0))

        X_base = np.hstack([X_num, X_cat]) if X_num.size or X_cat.size else np.empty((len(X_sel), 0))

        # --- PCA opcional
        if self.use_pca and self.pca_ is not None:
            return self.pca_.transform(X_base)
        return X_base

    def get_feature_names_out(self):
        if self.use_pca and self.pca_ is not None:
            n = self.pca_.n_components_
            return np.array([f"pca_{i}" for i in range(n)])
        return self.feature_names_base_ if self.feature_names_base_ is not None else None


# ==========================
# Helper para armar el Pipeline con transformer y el modelo final
# ==========================
def make_legacy_pipeline(model, use_pca: bool = False, pca_n_components: float | int = 0.90):
    return Pipeline(steps=[
        ("legacy_prep", LegacySelectPreprocessPCA(use_pca=use_pca, pca_n_components=pca_n_components)),
        ("model", model),
    ])


def apply_pca(X_train_base: np.ndarray, X_test_base: np.ndarray, n_components: float = 0.90):
    """
    Aplica PCA a los datos preprocesados.

    Args:
        X_train_base: Matriz NumPy preprocesada para entrenamiento (sin PCA).
        X_test_base: Matriz NumPy preprocesada para test (sin PCA).
        n_components: Número de componentes a mantener (entero) o
                      la proporción de varianza a explicar (float entre 0 y 1).

    Returns:
        Una tupla conteniendo:
        - Matriz NumPy de entrenamiento con PCA aplicado.
        - Matriz NumPy de test con PCA aplicado.
        - Objeto PCA ajustado.
    """
    pca = PCA(n_components=n_components, svd_solver='full', random_state=42)
    Xtr = pca.fit_transform(X_train_base)
    Xte = pca.transform(X_test_base)

    return Xtr, Xte, pca

def preprocess_features(X_train_selected: pd.DataFrame, X_test_selected: pd.DataFrame,
                        categorical_features: list[str], numeric_features: list[str]):
    """
    Preprocesa las variables numéricas y categóricas.

    - Imputa numéricas con mediana y categóricas con la moda.
    - Escala numéricas (StandardScaler).
    - Hace One-Hot Encoding para categóricas.
    - Ajusta (fit) solo con TRAIN y transforma TRAIN y TEST → evita fuga de información.

    Args:
        X_train_selected: DataFrame de entrenamiento con features seleccionadas.
        X_test_selected: DataFrame de test con features seleccionadas.
        categorical_features: Lista de nombres de columnas categóricas.
        numeric_features: Lista de nombres de columnas numéricas.

    Returns:
        Una tupla conteniendo:
        - Matriz NumPy preprocesada para entrenamiento.
        - Matriz NumPy preprocesada para test.
        - Nombres de las features resultantes del preprocesamiento.
        - OneHotEncoder ajustado (para posibles usos futuros).
        - SimpleImputer ajustado para numéricas (para posibles usos futuros).
        - StandardScaler ajustado (para posibles usos futuros).
        - SimpleImputer ajustado para categóricas (para posibles usos futuros).
    """

    # --- Numéricas
    num_imputer = SimpleImputer(strategy='median')
    num_scaler  = StandardScaler()

    Xtr_num = num_imputer.fit_transform(X_train_selected[numeric_features])
    Xtr_num = num_scaler.fit_transform(Xtr_num)

    Xte_num = num_imputer.transform(X_test_selected[numeric_features])
    Xte_num = num_scaler.transform(Xte_num)

    # --- Categóricas
    cat_imputer = SimpleImputer(strategy='most_frequent')
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)  # si scikit-learn <1.2 usar sparse=False

    Xtr_cat_raw = cat_imputer.fit_transform(X_train_selected[categorical_features])
    Xte_cat_raw = cat_imputer.transform(X_test_selected[categorical_features])


    Xtr_cat = ohe.fit_transform(Xtr_cat_raw)
    Xte_cat = ohe.transform(Xte_cat_raw)

    # --- Concatenar
    Xtr_base = np.hstack([Xtr_num, Xtr_cat])
    Xte_base = np.hstack([Xte_num, Xte_cat])


    # --- Nombres de columnas (útil para inspección / importancia)
    ohe_names = ohe.get_feature_names_out(categorical_features)
    feature_names_base = np.r_[numeric_features, ohe_names]

    return Xtr_base, Xte_base, feature_names_base, ohe, num_imputer, num_scaler, cat_imputer

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
        "HS", "AST", "HHW", "AHW", "HC", "AC", "HF", "AF", "HFKC", "AFKC",
        "HO", "AO", "HY", "AY", "HR", "AR", "HBP", "ABP"  # Stats del partido
    ]

    # 2. Detectar columnas relacionadas con cuotas o probabilidades de casas de apuestas
    odds_keywords = ["B365", "WH", "VC", "PS", "Bb", "Max", "Avg", "GB", "IWH", "LB", "SB", "Pinnacle", "BFDH", "BFDD", "BFDA", "BMGMH", "BMGMD", "BMGMA", "BVH", "BVD", "BVA", "CLH", "CLD", "CLA", "MaxH", "MaxD", "MaxA", "AvgH", "AvgD", "AvgA", "BFEH", "BFED", "BFEA", "B365>2.5", "B365<2.5", "P>2.5", "P<2.5", "Max>2.5", "Max<2.5", "Avg>2.5", "Avg<2.5", "BFE>2.5", "BFE<2.5", "AHh", "B365AHH", "B365AHA", "PAHH", "PAHA", "MaxAHH", "MaxAHA", "AvgAHH", "AvgAHA", "BFEAHH", "BFEAHA", "B365CH", "B365CD", "B365CA", "BFDCH", "BFDCD", "BFDCA", "BMGMCH", "BMGMCD", "BMGMCA", "BVCH", "BVCD", "BVCA", "BWCH", "BWCD", "BWCA", "CLCH", "CLCD", "CLCA", "LBCH", "LBCD", "LBCA", "MaxCH", "MaxCD", "MaxCA", "AvgCH", "AvgCD", "AvgCA", "BFECH", "BFECD", "BFECA", "B365C>2.5", "B365C<2.5", "PC>2.5", "PC<2.5", "MaxC>2.5", "MaxC<2.5", "AvgC>2.5", "AvgC<2.5", "BFEC>2.5", "BFEC<2.5", "AHCh", "B365CAHH", "B365CAHA", "PCAHH", "PCAHA", "MaxCAHH", "MaxCAHA", "AvgCAHH", "AvgCAHA", "BFECAHH", "BFECAHA", "BFH", "BFD", "BFA", "1XBH", "1XBD", "1XBA", "BFCH", "BFCD", "BFCA", "WHCH", "WHCD", "WHCA", "1XBCH", "1XBCD", "1XBCA", "IWCH", "IWCD", "IWCA", "VCCH", "VCCD", "VCCA", "SJH", "SJD", "SJA", "GBH", "GBD", "GBA", "BSH", "BSD", "BSA", "SBH", "SBD", "SBA"]
    odds_cols = [c for c in df.columns if any(k in c for k in odds_keywords)]

    # Agregamos a la lista general de exclusión
    cols_to_drop.extend(odds_cols)

    # Eliminamos columnas del conjunto de entrenamiento y test
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")


    # 3. Categóricas principales
    categorical_features = ["Div", "HomeTeam", "AwayTeam"]

    # 4. Variables numéricas: todas las demás que sean de tipo numérico
    numeric_features = df.select_dtypes(include=["number"]).columns.tolist()

    return df, categorical_features, numeric_features
