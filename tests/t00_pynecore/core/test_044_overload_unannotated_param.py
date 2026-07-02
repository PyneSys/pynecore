"""
@pyne
"""
from typing import Any

# noinspection PyProtectedMember
from pynecore.core.overload import _check_type, overload


def main():
    """Dummy main to satisfy the @pyne script loader."""
    pass


#
# ``Any`` must act as a wildcard in overload dispatch. Parameters without a type
# hint default to ``Any`` in ``param_types`` (PyneComp threads closure variables
# into nested functions as leading, unannotated parameters), and Python's
# ``isinstance`` raises ``TypeError: typing.Any cannot be used with isinstance()``
# on it -- the dispatcher's quick path crashed on every such call.
#

def __test_check_type_any_matches_everything__():
    """_check_type(value, Any) is True for any value instead of raising"""
    for value in (1, 1.5, "s", None, object(), [1], {"k": "v"}):
        assert _check_type(value, Any) is True


class _Dot:
    pass


class _Candle:
    pass


# Mirrors the compiled shape of an overloaded Pine method using a closure
# variable: the compiler passes the closure value as a leading parameter that
# carries no annotation.
# noinspection PyShadowingNames,PyUnusedLocal
@overload
def _create(closure_color, this: _Dot):  # type: ignore[no-redef]
    return 'dot'


# noinspection PyShadowingNames,PyUnusedLocal,PyRedeclaration
@overload
def _create(closure_color, this: _Candle):  # type: ignore[no-redef]
    return 'candle'


def __test_dispatch_with_unannotated_leading_param__():
    """positional dispatch over an unannotated (closure) param picks by the typed one"""
    marker = object()
    assert _create(marker, _Dot()) == 'dot'
    assert _create(marker, _Candle()) == 'candle'


#
# An untyped call site emits keyword arguments under their original Pine
# spelling, while a compiled library declares a trigger-named parameter under
# its canonical image (``position`` -> ``position__ren__``). The dispatcher
# must adapt such keywords the same way pine_method._adapt_exported_kwargs
# does for plain exports (regression: string_utils ``announce(position=...)``
# died with ``No matching implementation``).
#

# noinspection PyUnusedLocal
@overload
def _announce(this: str, height: float = 0.0, position__ren__: str = 'top'):  # type: ignore[no-redef]
    return f'str:{position__ren__}'


# noinspection PyUnusedLocal,PyRedeclaration
@overload
def _announce(this: int, height: float = 0.0, position__ren__: str = 'top'):  # type: ignore[no-redef]
    return f'int:{position__ren__}'


def __test_dispatch_adapts_canonical_kwargs__():
    """a bare Pine keyword targets the canonically renamed parameter"""
    assert _announce('x', position='mid') == 'str:mid'
    assert _announce(3, position='mid') == 'int:mid'
    # the canonical spelling keeps working, and a correct call is unaltered
    assert _announce('x', position__ren__='low') == 'str:low'
    assert _announce('x', height=1.0) == 'str:top'
