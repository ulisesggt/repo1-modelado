"""Paquete ``default_pipeline``.

Reúne las clases necesarias para (de)serializar los artefactos del pipeline
de detección de impago:

* ``preprocessing`` / ``filtering`` — clases heredadas de la Práctica 1, que
  permiten cargar ``preprocessor.pkl`` y ``filter.pkl``.
* ``model`` — :class:`Practica2Model` y :class:`VennAbersInterval`, el
  artefacto generado en la Práctica 2 (``practica2_model.pkl``).
* ``compat`` — registra alias en ``sys.modules`` para que ``joblib.load``
  encuentre las clases sea cual sea la ruta de módulo con la que se guardaron
  los pickles heredados.

El paquete se incluye, idéntico, en el Repo 1 (modelado) y en el Repo 2 (API).
"""

from . import compat  # noqa: F401  (efecto secundario: registra los alias)

__all__ = ["compat"]
