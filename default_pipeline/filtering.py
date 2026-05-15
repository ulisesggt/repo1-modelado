"""default_pipeline.filtering

Clase de filtrado de features de la Práctica 1 (Practica1Filtering).
Se incluye aquí para que joblib.load("filter.pkl") pueda reconstruir
el objeto ya fitteado sin necesidad de tener el repo de P1.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, VarianceThreshold, f_classif


class Practica1Filtering:
    def __init__(self, max_features=60, random_state=42):
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y):
        datos = self._preparar_X(X)
        y_bin = self._preparar_y(y)

        self.columnas_entrada_ = datos.columns.tolist()

        self.filtro_varianza_ = VarianceThreshold(threshold=0.01)
        matriz_var = self.filtro_varianza_.fit_transform(datos)
        self.columnas_varianza_ = datos.columns[self.filtro_varianza_.get_support()].tolist()

        if matriz_var.shape[1] == 0:
            self.columnas_salida_ = []
            self.selector_kbest_ = None
            self.selector_modelo_ = None
            return self

        k = min(self.max_features, matriz_var.shape[1])
        self.selector_kbest_ = SelectKBest(score_func=f_classif, k=k)
        matriz_kbest = self.selector_kbest_.fit_transform(matriz_var, y_bin)
        self.columnas_kbest_ = np.array(self.columnas_varianza_)[self.selector_kbest_.get_support()].tolist()

        modelo = RandomForestClassifier(
            n_estimators=100,
            max_depth=7,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=-1
        )
        self.selector_modelo_ = SelectFromModel(modelo, threshold="median")
        self.selector_modelo_.fit(matriz_kbest, y_bin)
        self.columnas_salida_ = np.array(self.columnas_kbest_)[self.selector_modelo_.get_support()].tolist()

        if len(self.columnas_salida_) == 0:
            self.columnas_salida_ = self.columnas_kbest_
            self.selector_modelo_ = None

        return self

    def transform(self, X):
        datos = self._preparar_X(X)

        for columna in self.columnas_entrada_:
            if columna not in datos.columns:
                datos[columna] = 0
        datos = datos[self.columnas_entrada_]

        if len(getattr(self, "columnas_salida_", [])) == 0:
            return pd.DataFrame(index=datos.index)

        matriz_var = self.filtro_varianza_.transform(datos)
        matriz_kbest = self.selector_kbest_.transform(matriz_var)

        if self.selector_modelo_ is not None:
            matriz_final = self.selector_modelo_.transform(matriz_kbest)
        else:
            matriz_final = matriz_kbest

        return pd.DataFrame(matriz_final, columns=self.columnas_salida_, index=datos.index)

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    def _preparar_X(self, X):
        datos = X.copy()
        datos = datos.replace([np.inf, -np.inf], np.nan)
        datos = datos.fillna(0)
        return datos

    def _preparar_y(self, y):
        serie = pd.Series(y).copy()
        if serie.dtype == "object":
            serie = (serie != "Fully Paid").astype(int)
        return serie.astype(int).values
