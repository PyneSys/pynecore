"""
The generated lib registry (``lib_types.json``), read once per process.

Its own module, apart from the Pine type pass that interprets most of it, so a
step that only needs a raw section (the import normalizer, function isolation)
does not depend on the type pass. ``pine_type_infer`` builds its typed views
(``lib_types``, ``lib_classes``, ...) on :func:`lib_registry`.
"""
import json
from pathlib import Path
from typing import Any

__all__ = ['lib_registry', 'lib_routes', 'lib_star_exports']

_LIB_TYPES_PATH = Path(__file__).parent / 'lib_types.json'
_LIB_REGISTRY: dict[str, Any] = {}


def lib_registry() -> dict[str, Any]:
    """
    The generated lib registry file, parsed once for every view of it.

    :return: Its top-level sections
    """
    if not _LIB_REGISTRY:
        _LIB_REGISTRY.update(json.loads(_LIB_TYPES_PATH.read_text()))
    return _LIB_REGISTRY


def lib_routes() -> dict[str, str]:
    """
    The call-site route of every lib callee path that does not route uniform.

    ``FunctionIsolationTransformer`` reads a ``pynecore.lib`` callee's route
    here instead of importing and inspecting the object; the generator computed
    each entry with the transformer's own ``classify_callee``.

    :return: Callee path as a script spells it (``lib.ta.sma``) -> route
    """
    return lib_registry()['routes']


def lib_star_exports() -> dict[str, list[str]]:
    """
    The ``__all__`` of every lib module that defines one.

    ``ImportNormalizerTransformer`` expands ``from pynecore.lib... import *``
    from here instead of importing the module.

    :return: Module name (``pynecore.lib.ta``) -> its ``__all__``, sorted
    """
    return lib_registry()['star_exports']
