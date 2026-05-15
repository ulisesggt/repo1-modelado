# Práctica 2 — Repo 1: Modelado y Calibración

Repositorio del pipeline de modelado para la detección de impago en préstamos (Lending Club).

## Contenido

```
repo1-modelado/
├── practica2_notebook.ipynb   # Notebook principal (ejecutado)
├── default_pipeline/          # Paquete de compatibilidad con Práctica 1
│   ├── __init__.py
│   ├── preprocessing.py       # Practica1Preprocess (igual que P1)
│   ├── filtering.py           # Practica1Filtering (igual que P1)
│   ├── model.py               # VennAbersInterval + Practica2Model
│   └── compat.py              # Alias de módulos para joblib.load
├── artifacts/
│   ├── preprocessor.pkl       # Preprocesador ajustado (Práctica 1)
│   ├── filter.pkl             # Selector de features ajustado (Práctica 1)
│   ├── practica2_model.pkl    # Modelo final P2 (LightGBM + Venn-Abers)
│   └── feature_schema.json    # Esquema de features y metadatos del modelo
├── data/
│   └── df_train_small.csv     # Muestra de entrenamiento (20 000 filas)
├── pyproject.toml
└── README.md
```

## Qué hace el notebook

El notebook `practica2_notebook.ipynb` cubre cuatro secciones:

**1.1 — Búsqueda de hiperparámetros con Optuna**  
Cuatro estudios (LightGBM ± balanced, XGBoost ± balanced) optimizando Log Loss con TPESampler(multivariate=True) + HyperbandPruner. Se reporta tabla completa de métricas (Accuracy, Precision, Recall, F1, MCC, ROC-AUC, PR-AUC, Log Loss, Brier, ECE) y se elige el modelo ganador: **LightGBM sin balanceo** (Log Loss = 0.4524, ECE = 0.0037).

**1.2 — Decisión de calibración**  
Diagnóstico con reliability diagram y ECE. El modelo ya está bien calibrado (ECE = 0.0037); aplicar calibración sigmoidal empeora el ECE ×4.6. Decisión: **no calibrar**.

**1.3 — Intervalos de incertidumbre con Venn-Abers**  
Implementación del predictor IVAP (Inductive Venn-Abers). WIDTH_THRESHOLD fijo = 0.2; decisión automática si p_high − p_low ≤ 0.2, y derivación al agente humano en caso contrario. Respuesta argumentada a por qué sigmoid no es equivalente a cuantificación de incertidumbre.

**1.4 — Persistencia**  
Guardado de `practica2_model.pkl` y `feature_schema.json` en `artifacts/`.

## Requisitos

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) — gestor de entornos y dependencias

## Instalación

```bash
# Desde la raíz del repositorio
uv sync
```

## Ejecutar el notebook

```bash
uv run jupyter notebook practica2_notebook.ipynb
```

O en JupyterLab:

```bash
uv run jupyter lab practica2_notebook.ipynb
```

## Artefactos generados

| Archivo | Descripción |
|---|---|
| `artifacts/preprocessor.pkl` | `Practica1Preprocess` ajustado sobre datos de entrenamiento |
| `artifacts/filter.pkl` | `Practica1Filtering` ajustado (30 features seleccionadas) |
| `artifacts/practica2_model.pkl` | `Practica2Model`: LightGBM + VennAbersInterval, sin calibrador |
| `artifacts/feature_schema.json` | Nombres de features, encoding del target, WIDTH_THRESHOLD |

## Modelo final

- **Algoritmo**: LightGBM (sin class_weight='balanced')
- **Métrica de optimización**: Log Loss (scoring rule propio)
- **Calibración**: Ninguna (modelo ya calibrado, ECE = 0.0037)
- **Incertidumbre**: Inductive Venn-Abers Predictor (IVAP)
- **WIDTH_THRESHOLD**: 0.2 (fijo)
