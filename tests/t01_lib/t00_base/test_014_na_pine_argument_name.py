"""
@pyne

Pine accepts ``na(x = close)`` as a named argument, and a compiled script emits Pine's
own keyword verbatim. If PyneCore's predicate names its parameter anything else, the
call is a TypeError that halts the script at runtime.
"""
from pynecore.types.na import NA
from pynecore.lib import na, nz


def main():
    """Dummy main to satisfy the @pyne script loader."""
    pass


def __test_na_predicate_argument_is_named_x__():
    """na() takes ``x`` -- the name Pine uses and the compiler emits."""
    assert na(x=1.0) is False
    assert na(x=float('nan')) is True
    assert na(x=float('inf')) is True
    assert na(x=NA()) is True
    # The keyword form must agree with the positional form on every face
    assert na(x=1.0) == na(1.0)


def __test_na_type_constructor_and_bare_value_still_work__():
    """Naming the parameter must not break na's other two faces."""
    assert isinstance(na(), NA)
    assert isinstance(na(x=str), NA)
    # The type-constructor face must behave identically through the keyword
    assert repr(na(x=float)) == repr(na(float))


def __test_nz_argument_names__():
    """nz() takes ``source`` and ``replacement``, both confirmed against Pine."""
    assert nz(source=1.0, replacement=-5.0) == 1.0
    assert nz(source=float('nan'), replacement=-5.0) == -5.0
