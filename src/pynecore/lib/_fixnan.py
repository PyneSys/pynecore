"""
@pyne lib

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

    Hole detection is nan-only (``x != x``), deliberately NARROWER than the
    ``na()`` predicate: inf passes through as a regular value even though
    ``na(inf)`` is true. NA objects (non-float na) are holes too.

    :param source: The source value
    :return: The source value if it is not a nan-hole, otherwise the last such value
    """
    last_not_nan: Persistent[Any] = NA(None)
    last_not_nan = last_not_nan if not (source == source) else source  # is_na_arg
    return last_not_nan
