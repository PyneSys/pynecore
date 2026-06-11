"""
@pyne

Stateful implementation of ``lib.fixnan``. It lives in its own small module
because the ``@pyne`` marker is module-level and the host module
(``lib/__init__.py``) must stay untransformed; the host re-exports the
function, and the layout travels on the function object.
"""
# Absolute imports on purpose: the call-site classifier resolves absolute
# imports at transform time, so NA() calls stay direct instead of anchored
from typing import Any

from pynecore.types import NA, Persistent

__all__ = ['fixnan']


def fixnan(source: Any) -> Any:
    """
    Fix NA values by replacing them with the last non-NA value

    :param source: The source value
    :return: The source value if it is not NA, otherwise the last non-NA value
    """
    last_not_nan: Persistent[Any] = NA(None)
    last_not_nan = source if not isinstance(source, NA) else last_not_nan
    return last_not_nan
