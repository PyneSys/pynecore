import types
from typing import Any

__all__ = ['shadowed_namespace']


def shadowed_namespace(library: types.ModuleType, builtin_ns: Any) -> types.ModuleType:
    """
    Merge an imported Pine library over the builtin namespace its alias shadows.

    MEASURED on TradingView (BINANCE:BTCUSDT 30m): an import alias that spells a
    builtin namespace resolves MEMBER BY MEMBER, not wholesale. The library's
    exports win — ``import TradingView/ta/8`` + ``ta.supertrend(3.0, 10, true)``
    compiles, and the builtin ``ta.supertrend`` takes only two arguments — while
    every other member falls back to the builtin namespace (``ta.atr(14)`` in the
    same script). The rule keys on the ALIAS, not on the library: the same import
    spelled ``as tvta`` rejects ``tvta.atr`` (CE10271), and an unrelated library
    imported ``as ta`` answers ``ta.atr(14)`` from the builtin.

    The shim is a real module object so the import machinery, the isolation
    transformer and ``repr`` all keep seeing a namespace, and the fallback rides
    PEP 562's module ``__getattr__``: it fires only for names the merged namespace
    does not define itself, and resolves through to the builtin on every access,
    so a builtin module property still reads its current bar's value.

    :param library: The compiled Pine library module the alias imports
    :param builtin_ns: The builtin namespace the alias shadows
    :return: A namespace answering the library's exports first, the builtin after
    """
    shim = types.ModuleType(library.__name__)
    exported: Any = getattr(library, '__all__', None)
    if exported is None:
        # A library with no export surface (all-private helper module) — Pine has
        # no such thing, but a hand-written Pyne library need not spell __all__.
        exported = [name for name in vars(library) if not name.startswith('_')]
    for name in exported:
        setattr(shim, name, getattr(library, name))
    shim.__dict__['__getattr__'] = lambda name: getattr(builtin_ns, name)
    return shim
