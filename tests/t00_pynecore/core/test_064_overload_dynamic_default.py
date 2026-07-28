"""
@pyne
"""
# noinspection PyProtectedMember
from pynecore.core.instance_state import __dyn_default__
# noinspection PyProtectedMember
from pynecore.core.overload import overload


def main():
    """Dummy main to satisfy the @pyne script loader."""
    pass


#
# A Pine parameter whose default references per-bar runtime state -- ``= na``
# above all -- is taken over by DynamicDefaultTransformer: the declared default
# becomes the ``__dyn_default__`` sentinel and the real expression is evaluated
# in the body when the argument is omitted. Overload dispatch used to type-check
# the applied defaults, and the sentinel is a bare ``object()`` that satisfies no
# annotation, so every overload with an omitted optional parameter was rejected.
# Whole libraries died on it (chrono_utils' ``SessionTimeRange.init()``).
#

class _Marker:
    pass


class _Holder:
    pass


# noinspection PyUnusedLocal
@overload
def _init(this: _Marker, value: int = 0):  # type: ignore[no-redef]
    return 'marker'


# The compiled shape of ``method init(Holder this, Marker a = na, Marker b = na)``
# noinspection PyUnusedLocal,PyRedeclaration
@overload
def _init(this: _Holder,  # type: ignore[no-redef]
          a: _Marker = __dyn_default__, b: _Marker = __dyn_default__):
    return f'holder:{a is __dyn_default__}:{b is __dyn_default__}'


def __test_omitted_dynamic_default_dispatches__():
    """the sentinel standing in for an omitted default does not veto the match"""
    assert _init(_Holder()) == 'holder:True:True'


def __test_supplied_argument_still_type_checked__():
    """a passed argument is matched normally; only the sentinel is exempt"""
    m = _Marker()
    assert _init(_Holder(), m) == 'holder:False:True'
    assert _init(_Holder(), m, m) == 'holder:False:False'


def __test_sibling_overload_unaffected__():
    """the exempt sentinel does not let a wrong-typed first argument through"""
    assert _init(_Marker()) == 'marker'
