"""default_pipeline.model
=========================

Clases del artefacto final de la Práctica 2 (`practica2_model.pkl`).

Contiene:

* :class:`VennAbersInterval` — predictor de intervalos de probabilidad
  ``[p_low, p_high]`` mediante *Inductive Venn-Abers Predictors* (IVAP), que
  pertenece a la familia de la *Conformal Prediction* (Vovk et al.). Para cada
  predicción devuelve un intervalo de probabilidad con garantía de validez y,
  como subproducto, una probabilidad puntual perfectamente calibrada.

* :class:`Practica2Model` — pipeline final = clasificador base
  (LightGBM / XGBoost) + calibrador opcional + objeto de intervalo
  Venn-Abers. Expone ``predict``, ``predict_proba``, ``predict_interval`` y
  ``predict_with_decision`` (esta última aplica la política de derivación a un
  agente humano).

Este módulo debe ser importable **tanto en el Repo 1 (notebook) como en el
Repo 2 (API)** para poder cargar ``practica2_model.pkl`` con ``joblib.load``.
Por eso el paquete ``default_pipeline`` se incluye, idéntico, en los dos repos.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

import numpy as np

try:  # sklearn siempre está disponible en ambos repos
    from sklearn.isotonic import IsotonicRegression
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "scikit-learn es necesario para default_pipeline.model"
    ) from exc


__all__ = ["VennAbersInterval", "Practica2Model"]


# ---------------------------------------------------------------------------
# Venn-Abers (Inductive Venn-Abers Predictor)
# ---------------------------------------------------------------------------
class VennAbersInterval:
    """Inductive Venn-Abers Predictor (IVAP).

    A partir de un conjunto de calibración ``(scores, labels)`` —donde
    ``scores`` son las puntuaciones del clasificador base para la clase
    positiva— produce, para cada nuevo ``score`` ``s``:

    * ``p0`` : regresión isotónica ajustada sobre el conjunto de calibración
      MÁS el par ``(s, 0)``, evaluada en ``s``.
    * ``p1`` : regresión isotónica ajustada sobre el conjunto de calibración
      MÁS el par ``(s, 1)``, evaluada en ``s``.

    Se cumple siempre ``p0 <= p1``. El intervalo ``[p0, p1]`` es la medida de
    incertidumbre de la probabilidad: cuanto más ancho, menos seguro está el
    modelo de su propia probabilidad. La probabilidad puntual "fusionada"
    ``p = p1 / (1 - p0 + p1)`` está calibrada por construcción.

    Para acelerar el cálculo sobre lotes grandes, las puntuaciones se redondean
    a ``round_to`` decimales y se cachea el resultado por valor único (el
    isotónico es monótono, así que dos scores iguales dan el mismo intervalo).
    """

    def __init__(self, round_to: int = 3) -> None:
        self.round_to = round_to
        self.cal_scores_: np.ndarray | None = None
        self.cal_labels_: np.ndarray | None = None

    # -- ajuste -------------------------------------------------------------
    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "VennAbersInterval":
        scores = np.asarray(scores, dtype=float).ravel()
        labels = np.asarray(labels, dtype=float).ravel()
        if scores.shape != labels.shape:
            raise ValueError("scores y labels deben tener la misma longitud")
        order = np.argsort(scores, kind="mergesort")
        self.cal_scores_ = scores[order]
        self.cal_labels_ = labels[order]
        return self

    # -- núcleo IVAP --------------------------------------------------------
    def _p0_p1(self, s: float) -> tuple[float, float]:
        """Calcula (p0, p1) para una única puntuación ``s``."""
        x = np.append(self.cal_scores_, s)
        # p0 -> el punto de test se etiqueta como 0
        iso0 = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso0.fit(x, np.append(self.cal_labels_, 0.0))
        p0 = float(iso0.predict([s])[0])
        # p1 -> el punto de test se etiqueta como 1
        iso1 = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso1.fit(x, np.append(self.cal_labels_, 1.0))
        p1 = float(iso1.predict([s])[0])
        # garantía teórica: p0 <= p1 (se fuerza por robustez numérica)
        if p0 > p1:
            p0, p1 = p1, p0
        return p0, p1

    # -- API pública --------------------------------------------------------
    def predict_interval(self, scores: np.ndarray) -> np.ndarray:
        """Devuelve un array ``(n, 2)`` con columnas ``[p_low, p_high]``."""
        if self.cal_scores_ is None:
            raise RuntimeError("VennAbersInterval no está ajustado (fit).")
        scores = np.asarray(scores, dtype=float).ravel()
        rounded = np.round(scores, self.round_to)
        cache: dict[float, tuple[float, float]] = {}
        for u in np.unique(rounded):
            cache[float(u)] = self._p0_p1(float(u))
        p0 = np.array([cache[float(r)][0] for r in rounded])
        p1 = np.array([cache[float(r)][1] for r in rounded])
        return np.column_stack([p0, p1])

    def predict_proba_point(self, scores: np.ndarray) -> np.ndarray:
        """Probabilidad puntual fusionada de Venn-Abers ``p1 / (1 - p0 + p1)``."""
        iv = self.predict_interval(scores)
        p0, p1 = iv[:, 0], iv[:, 1]
        denom = 1.0 - p0 + p1
        denom[denom == 0.0] = 1e-12
        return p1 / denom


# ---------------------------------------------------------------------------
# Artefacto final de la Práctica 2
# ---------------------------------------------------------------------------
class Practica2Model:
    """Pipeline final servido por la API.

    Encadena (sobre features YA preprocesadas y filtradas):

    1. ``base_model`` — clasificador ganador (LightGBM o XGBoost) optimizado
       con Optuna.
    2. ``calibrator`` — calibrador opcional (``CalibratedClassifierCV`` con
       sigmoid o isotónica). Puede ser ``None`` si en la sección 1.2 se decide
       no calibrar puntualmente.
    3. ``va`` — :class:`VennAbersInterval` que aporta el intervalo de
       incertidumbre ``[p_low, p_high]`` y, si se desea, la probabilidad
       puntual calibrada.

    ``point_proba_source`` controla de dónde sale la probabilidad puntual que
    devuelve :meth:`predict_proba`:

    * ``"base"``        -> probabilidad cruda del modelo base.
    * ``"calibrator"``  -> probabilidad del calibrador sigmoid/isotónico.
    * ``"venn_abers"``  -> probabilidad fusionada de Venn-Abers (calibrada por
      construcción; es la opción coherente con la reflexión de la sección 1.3).
    """

    def __init__(
        self,
        base_model: Any,
        va: VennAbersInterval,
        calibrator: Any | None = None,
        point_proba_source: str = "base",
        width_threshold: float = 0.2,
        feature_names: list[str] | None = None,
        metadata: dict | None = None,
    ) -> None:
        if point_proba_source not in {"base", "calibrator", "venn_abers"}:
            raise ValueError(f"point_proba_source inválido: {point_proba_source}")
        if point_proba_source == "calibrator" and calibrator is None:
            raise ValueError("point_proba_source='calibrator' pero calibrator es None")
        self.base_model = base_model
        self.va = va
        self.calibrator = calibrator
        self.point_proba_source = point_proba_source
        self.width_threshold = float(width_threshold)
        self.feature_names = list(feature_names) if feature_names is not None else None
        self.metadata = dict(metadata or {})
        self.metadata.setdefault(
            "created_at", _dt.datetime.now().isoformat(timespec="seconds")
        )
        self.version = str(self.metadata.get("version", "practica2-model-v1"))

    # -- utilidades internas ------------------------------------------------
    def _as_frame(self, X):
        """Reordena/valida columnas si tenemos ``feature_names`` y X es DataFrame."""
        if self.feature_names is not None:
            try:
                import pandas as pd  # import perezoso

                if isinstance(X, pd.DataFrame):
                    missing = set(self.feature_names) - set(X.columns)
                    if missing:
                        raise ValueError(
                            f"Faltan columnas para el modelo: {sorted(missing)}"
                        )
                    return X[self.feature_names]
            except ImportError:  # pragma: no cover
                pass
        return X

    def _base_score(self, X) -> np.ndarray:
        """Puntuación del modelo base para la clase positiva (impago)."""
        X = self._as_frame(X)
        return np.asarray(self.base_model.predict_proba(X))[:, 1]

    def _compute(self, X) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Devuelve ``(p_point, p_low, p_high)`` calculando los scores una vez."""
        X = self._as_frame(X)
        base_scores = np.asarray(self.base_model.predict_proba(X))[:, 1]
        interval = self.va.predict_interval(base_scores)
        p_low, p_high = interval[:, 0], interval[:, 1]

        if self.point_proba_source == "venn_abers":
            denom = 1.0 - p_low + p_high
            denom = np.where(denom == 0.0, 1e-12, denom)
            p_point = p_high / denom
        elif self.point_proba_source == "calibrator":
            p_point = np.asarray(self.calibrator.predict_proba(X))[:, 1]
        else:  # "base"
            p_point = base_scores
        return p_point, p_low, p_high

    # -- API estilo scikit-learn -------------------------------------------
    def predict_proba(self, X) -> np.ndarray:
        """Matriz ``(n, 2)`` con ``[1 - p_default, p_default]``."""
        p_point, _, _ = self._compute(X)
        p_point = np.clip(p_point, 0.0, 1.0)
        return np.column_stack([1.0 - p_point, p_point])

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        """Clase predicha (1 = impago) usando el umbral indicado."""
        p_point, _, _ = self._compute(X)
        return (p_point >= threshold).astype(int)

    def predict_interval(self, X) -> np.ndarray:
        """Array ``(n, 2)`` con columnas ``[p_low, p_high]`` (Venn-Abers)."""
        _, p_low, p_high = self._compute(X)
        return np.column_stack([p_low, p_high])

    # -- política de derivación a un agente --------------------------------
    def predict_with_decision(self, X) -> list[dict]:
        """Para cada fila devuelve un dict con probabilidad, intervalo y decisión.

        Política: ``decision = "agent"`` si ``p_high - p_low > width_threshold``
        (por defecto 0.2); en caso contrario ``decision = "auto"``.
        """
        p_point, p_low, p_high = self._compute(X)
        out: list[dict] = []
        thr = self.width_threshold
        for pp, lo, hi in zip(p_point, p_low, p_high):
            width = float(hi) - float(lo)
            if width > thr:
                decision = "agent"
                reason = f"p_high - p_low = {width:.3f} > {thr}"
            else:
                decision = "auto"
                reason = f"p_high - p_low = {width:.3f} <= {thr}"
            out.append(
                {
                    "p_default": round(float(np.clip(pp, 0.0, 1.0)), 6),
                    "p_low": round(float(lo), 6),
                    "p_high": round(float(hi), 6),
                    "decision": decision,
                    "reason": reason,
                }
            )
        return out

    # -- introspección ------------------------------------------------------
    def describe(self) -> dict:
        """Resumen ligero del artefacto (sin exponer datos sensibles)."""
        return {
            "version": self.version,
            "point_proba_source": self.point_proba_source,
            "width_threshold": self.width_threshold,
            "has_calibrator": self.calibrator is not None,
            "n_features": len(self.feature_names) if self.feature_names else None,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Practica2Model(version={self.version!r}, "
            f"source={self.point_proba_source!r}, "
            f"calibrator={'sí' if self.calibrator is not None else 'no'})"
        )
