"""default_pipeline.preprocessing
Clase de preprocesamiento de la Práctica 1 (Practica1Preprocess).
Se incluye aquí para que joblib.load("preprocessor.pkl") pueda
reconstruir el objeto ya fitteado sin necesidad de tener el repo de P1.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, RobustScaler


class Practica1Preprocess:
    def __init__(self, variables_path="data/variables_withExperts.xlsx", target_col="loan_status"):
        self.variables_path = variables_path
        self.target_col = target_col

    def fit(self, X, y=None):
        datos = X.copy()
        if self.target_col in datos.columns:
            datos = datos.drop(columns=[self.target_col])

        self.variables_excel_ = self._leer_variables(datos.columns)
        datos = datos[self.variables_excel_].copy()
        datos = self._limpiar_columnas(datos)
        datos = self._crear_features(datos)

        self.columnas_entrada_ = list(datos.columns)
        self.columnas_numericas_ = datos.select_dtypes(include=[np.number]).columns.tolist()
        self.columnas_categoricas_ = [c for c in datos.columns if c not in self.columnas_numericas_]

        self.columnas_numericas_ = [c for c in self.columnas_numericas_ if datos[c].notna().sum() > 0]
        self.columnas_categoricas_ = [c for c in self.columnas_categoricas_ if datos[c].notna().sum() > 0]

        if self.columnas_numericas_:
            self.imputador_numerico_ = SimpleImputer(strategy="mean")
            matriz_num = self.imputador_numerico_.fit_transform(datos[self.columnas_numericas_])
            self.escalador_ = RobustScaler()
            self.escalador_.fit(matriz_num)
        else:
            self.imputador_numerico_ = None
            self.escalador_ = None

        if self.columnas_categoricas_:
            self.imputador_categorico_ = SimpleImputer(strategy="constant", fill_value="Desconocido")
            matriz_cat = self.imputador_categorico_.fit_transform(datos[self.columnas_categoricas_].astype("object"))
            self.codificador_ = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            self.codificador_.fit(matriz_cat)
        else:
            self.imputador_categorico_ = None
            self.codificador_ = None

        self.columnas_salida_ = self.columnas_numericas_ + self.columnas_categoricas_
        return self

    def transform(self, X):
        datos = X.copy()
        if self.target_col in datos.columns:
            datos = datos.drop(columns=[self.target_col])

        for columna in self.variables_excel_:
            if columna not in datos.columns:
                datos[columna] = np.nan

        datos = datos[self.variables_excel_].copy()
        datos = self._limpiar_columnas(datos)
        datos = self._crear_features(datos)

        for columna in self.columnas_entrada_:
            if columna not in datos.columns:
                datos[columna] = np.nan

        datos = datos[self.columnas_entrada_].copy()
        partes = []

        if self.columnas_numericas_:
            matriz_num = self.imputador_numerico_.transform(datos[self.columnas_numericas_])
            matriz_num = self.escalador_.transform(matriz_num)
            partes.append(pd.DataFrame(matriz_num, columns=self.columnas_numericas_, index=datos.index))

        if self.columnas_categoricas_:
            matriz_cat = self.imputador_categorico_.transform(datos[self.columnas_categoricas_].astype("object"))
            matriz_cat = self.codificador_.transform(matriz_cat)
            partes.append(pd.DataFrame(matriz_cat, columns=self.columnas_categoricas_, index=datos.index))

        salida = pd.concat(partes, axis=1)
        salida = salida.replace([np.inf, -np.inf], np.nan).fillna(0)
        return salida[self.columnas_salida_]

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def _leer_variables(self, columnas_disponibles):
        try:
            variables = pd.read_excel(self.variables_path)
            cols = {c.lower(): c for c in variables.columns}
            col_variable = cols.get("variable", variables.columns[0])
            col_predictora = cols.get("posible_predictora")

            if col_predictora is not None:
                mascara = variables[col_predictora].astype(str).str.lower().str.strip().isin(["si", "sí", "yes", "1", "true"])
                lista = variables.loc[mascara, col_variable].dropna().astype(str).tolist()
            else:
                lista = variables[col_variable].dropna().astype(str).tolist()
        except Exception:
            lista = list(columnas_disponibles)

        columnas = [c for c in lista if c in columnas_disponibles and c != self.target_col]
        if len(columnas) == 0:
            columnas = [c for c in columnas_disponibles if c != self.target_col]
        return columnas

    def _limpiar_columnas(self, datos):
        datos = datos.copy()

        for columna in ["int_rate", "revol_util", "sec_app_revol_util"]:
            if columna in datos.columns:
                datos[columna] = datos[columna].astype(str).str.replace("%", "", regex=False)
                datos[columna] = pd.to_numeric(datos[columna], errors="coerce")

        if "term" in datos.columns:
            datos["term_meses"] = datos["term"].astype(str).str.extract(r"(\d+)").astype(float)

        if "emp_length" in datos.columns:
            emp = datos["emp_length"].astype(str).str.lower()
            emp = emp.str.replace("10+ years", "10", regex=False)
            emp = emp.str.replace("< 1 year", "0", regex=False)
            emp = emp.str.extract(r"(\d+)")
            datos["emp_length_num"] = pd.to_numeric(emp[0], errors="coerce")

        for columna in ["earliest_cr_line", "sec_app_earliest_cr_line"]:
            if columna in datos.columns:
                fecha = pd.to_datetime(datos[columna], format="%b-%Y", errors="coerce")
                datos[columna + "_anio"] = fecha.dt.year
                datos[columna + "_mes"] = fecha.dt.month

        for columna in datos.columns:
            if datos[columna].dtype == "object":
                posible_num = pd.to_numeric(datos[columna], errors="coerce")
                if posible_num.notna().mean() > 0.9:
                    datos[columna] = posible_num

        datos = datos.replace([np.inf, -np.inf], np.nan)
        return datos

    def _crear_features(self, datos):
        datos = datos.copy()

        datos["fico_medio"] = self._media_columnas(datos, "fico_range_low", "fico_range_high")
        datos["sec_app_fico_medio"] = self._media_columnas(datos, "sec_app_fico_range_low", "sec_app_fico_range_high")
        datos["ingreso_mensual"] = self._dividir(datos, "annual_inc", 12)
        datos["cuota_sobre_ingreso_mensual"] = self._ratio(datos, "installment", "ingreso_mensual")
        datos["prestamo_sobre_ingreso"] = self._ratio(datos, "loan_amnt", "annual_inc")
        datos["revol_sobre_limite"] = self._ratio(datos, "revol_bal", "total_rev_hi_lim")
        datos["deuda_total_sobre_ingreso"] = self._ratio(datos, "total_bal_ex_mort", "annual_inc")
        datos["credito_usado_sobre_total"] = self._ratio(datos, "total_bal_ex_mort", "tot_hi_cred_lim")
        datos["limite_bc_sobre_ingreso"] = self._ratio(datos, "total_bc_limit", "annual_inc")

        if "annual_inc" in datos.columns:
            cortes = [-np.inf, 30000, 60000, 100000, np.inf]
            etiquetas = ["bajo", "medio", "alto", "muy_alto"]
            datos["tramo_ingreso"] = pd.cut(
                pd.to_numeric(datos["annual_inc"], errors="coerce"),
                bins=cortes, labels=etiquetas
            ).astype("object")

        if "dti" in datos.columns and "installment" in datos.columns:
            datos["riesgo_cuota_dti"] = (
                pd.to_numeric(datos["dti"], errors="coerce")
                * pd.to_numeric(datos["installment"], errors="coerce")
            )

        datos = datos.replace([np.inf, -np.inf], np.nan)
        return datos

    def _ratio(self, datos, numerador, denominador):
        if numerador not in datos.columns or denominador not in datos.columns:
            return np.nan
        num = pd.to_numeric(datos[numerador], errors="coerce")
        den = pd.to_numeric(datos[denominador], errors="coerce").replace(0, np.nan)
        return num / den

    def _dividir(self, datos, columna, divisor):
        if columna not in datos.columns:
            return np.nan
        return pd.to_numeric(datos[columna], errors="coerce") / divisor

    def _media_columnas(self, datos, col1, col2):
        if col1 not in datos.columns or col2 not in datos.columns:
            return np.nan
        valores = pd.concat([
            pd.to_numeric(datos[col1], errors="coerce"),
            pd.to_numeric(datos[col2], errors="coerce")
        ], axis=1)
        return valores.mean(axis=1)
