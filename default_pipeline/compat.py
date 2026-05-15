"""default_pipeline.compat
===========================

Compatibilidad de deserialización para joblib.load.

Los pickles de la Práctica 1 (preprocessor.pkl, filter.pkl) se guardaron
con las clases Practica1Preprocess y Practica1Filtering bajo la ruta de
módulo src.preprocessing.practica1_preprocessing / src.filtering.practica1_filtering.

Este módulo registra alias en sys.modules para que joblib pueda encontrar
las clases sin importar con qué ruta de módulo se guardó el pickle.
"""

from __future__ import annotations

import sys
import types

from . import preprocessing as _preprocessing
from . import filtering as _filtering
from .preprocessing import Practica1Preprocess
from .filtering import Practica1Filtering

_CLASS_REGISTRY = {
    "Practica1Preprocess": Practica1Preprocess,
    "Practica1Filtering": Practica1Filtering,
}

# Todos los alias de ruta de módulo que pueden aparecer dentro de los pkl
_MODULE_ALIASES: dict[str, types.ModuleType] = {
    # preprocesamiento — variantes vistas en la naturaleza
    "practica1_preprocessing":                       _preprocessing,
    "preprocessing":                                 _preprocessing,
    "src.preprocessing.practica1_preprocessing":     _preprocessing,
    "src.prepocessing.practica1_preprocessing":      _preprocessing,  # typo real P1
    "src.practica1.preprocessing":                   _preprocessing,
    "src.preprocessing":                             _preprocessing,
    "base_pre":                                      _preprocessing,
    "src.preprocessing.base_preprocessing":          _preprocessing,
    # filtrado — variantes vistas en la naturaleza
    "practica1_filtering":                           _filtering,
    "filtering":                                     _filtering,
    "src.filtering.practica1_filtering":             _filtering,
    "src.filtering":                                 _filtering,
    "src.practica1.filtering":                       _filtering,
    "base_filter":                                   _filtering,
    "src.filtering.base_filtering":                  _filtering,
}


def _ensure_parent_packages(dotted: str) -> None:
    parts = dotted.split(".")
    for i in range(1, len(parts)):
        parent = ".".join(parts[:i])
        if parent not in sys.modules:
            mod = types.ModuleType(parent)
            mod.__path__ = []
            sys.modules[parent] = mod


def register_aliases() -> None:
    """Registra todos los alias (idempotente)."""
    for alias, target in _MODULE_ALIASES.items():
        _ensure_parent_packages(alias)
        if alias not in sys.modules:
            sys.modules[alias] = target
        # También añadir las clases al módulo aliasado por si el pkl busca
        # el atributo por nombre de clase directamente.
        for cls_name, cls in _CLASS_REGISTRY.items():
            if not hasattr(sys.modules[alias], cls_name):
                setattr(sys.modules[alias], cls_name, cls)

    # Inyectar en __main__ (pkl guardado desde notebook)
    main_mod = sys.modules.get("__main__")
    if main_mod is not None:
        for name, cls in _CLASS_REGISTRY.items():
            if not hasattr(main_mod, name):
                setattr(main_mod, name, cls)


def safe_load(path):
    """joblib.load tolerante: si falta un módulo/clase, lo aliasa y reintenta."""
    import joblib

    register_aliases()
    last_err: Exception | None = None
    for _ in range(8):
        try:
            return joblib.load(path)
        except ModuleNotFoundError as err:
            missing = err.name or ""
            target = _filtering if "filt" in missing.lower() else _preprocessing
            _ensure_parent_packages(missing)
            if missing not in sys.modules:
                sys.modules[missing] = target
            for cls_name, cls in _CLASS_REGISTRY.items():
                setattr(sys.modules[missing], cls_name, cls)
            last_err = err
        except AttributeError as err:
            # La clase no estaba en el módulo: la añadimos a todos los alias
            for target in (_preprocessing, _filtering):
                for name, cls in _CLASS_REGISTRY.items():
                    setattr(target, name, cls)
            last_err = err
    raise RuntimeError(
        f"No se pudo cargar {path} con alias de compatibilidad"
    ) from last_err


# Ejecutar el registro al importar el paquete
register_aliases()
