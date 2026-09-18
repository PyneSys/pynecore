"""
The language rule that an object created outside a function is read-only inside one.

Module-level storage survives every rollback the runtime performs — a
``request.security`` child's discarded developing round, a ``calc_on_order_fills``
re-execution, a live intrabar tick — while the script's own slots do not. The
pass rejects the DIRECT writes; these tests pin every category it names, and the
shapes it must leave alone: a local of the same spelling, a read, and a
module-level binding whose value is not a literal.
"""
import ast
from pathlib import Path

import pytest

from pynecore.core.import_hook import _analyse_tree

#: A path outside the pynecore package, so the pass is applied (its own lib is
#: exempt, like every other user-script-only pass of the pipeline).
_PATH = Path('/tmp/outer-write-test.py')


def __test_helper_analyse(body: str) -> None:
    """Run the analysing half of the pipeline over a script snippet."""
    source = body.lstrip('\n')
    _analyse_tree(ast.parse(source), source, _PATH, None)


def __test_helper_rejects(body: str, name: str) -> SyntaxError:
    """Assert the snippet is rejected and the error names the binding."""
    with pytest.raises(SyntaxError) as info:
        __test_helper_analyse(body)
    error = info.value
    assert f"'{name}'" in str(error), str(error)
    assert 'cannot be modified inside one' in str(error), str(error)
    return error


def __test_a_subscript_write_is_rejected__(log):
    """``BUF[0] = x`` writes the module-level list"""
    __test_helper_rejects("""
BUF = [0.0]


@lib.script.indicator("T")
def main():
    BUF[0] = lib.close
""", 'BUF')


def __test_an_attribute_write_is_rejected__(log):
    """``OBJ.field = x`` writes the module-level object"""
    __test_helper_rejects("""
OBJ = lib.array.new_float(1)


@lib.script.indicator("T")
def main():
    OBJ.n = 1.0
""", 'OBJ')


def __test_an_augmented_write_is_rejected__(log):
    """``BUF[0] += 1`` is a write like any other"""
    __test_helper_rejects("""
BUF = [0.0]


@lib.script.indicator("T")
def main():
    BUF[0] += 1.0
""", 'BUF')


def __test_a_delete_is_rejected__(log):
    """``del BUF[0]`` removes an element of the module-level list"""
    __test_helper_rejects("""
BUF = [0.0]


@lib.script.indicator("T")
def main():
    del BUF[0]
""", 'BUF')


def __test_a_mutating_method_call_is_rejected__(log):
    """``BUF.append(...)`` is the plain Python spelling of the same write"""
    __test_helper_rejects("""
BUF = [0.0]


@lib.script.indicator("T")
def main():
    BUF.append(lib.close)
""", 'BUF')


def __test_a_mutating_array_builtin_is_rejected__(log):
    """``array.push(STORE, ...)`` mutates its first argument"""
    __test_helper_rejects("""
STORE = lib.array.new_float(0)


@lib.script.indicator("T")
def main():
    lib.array.push(STORE, lib.close)
""", 'STORE')


def __test_a_mutating_matrix_builtin_is_rejected__(log):
    """``matrix.set(M, ...)`` mutates its first argument"""
    __test_helper_rejects("""
M = lib.matrix.new(1, 1, 0.0)


@lib.script.indicator("T")
def main():
    lib.matrix.set(M, 0, 0, lib.close)
""", 'M')


def __test_a_mutating_map_builtin_is_rejected__(log):
    """``map.put(M, ...)`` mutates its first argument"""
    __test_helper_rejects("""
M = lib.map.new()


@lib.script.indicator("T")
def main():
    lib.map.put(M, "k", lib.close)
""", 'M')


def __test_a_keyword_collection_argument_is_rejected__(log):
    """The collection may be named by its ``id`` keyword as well"""
    __test_helper_rejects("""
STORE = lib.array.new_float(0)


@lib.script.indicator("T")
def main():
    lib.array.clear(id=STORE)
""", 'STORE')


def __test_a_global_statement_is_rejected__(log):
    """Rebinding the module-level name is the same escape"""
    __test_helper_rejects("""
COUNT = 0


@lib.script.indicator("T")
def main():
    global COUNT
    COUNT = COUNT + 1
""", 'COUNT')


def __test_a_global_the_module_does_not_bind_is_rejected__(log):
    """``global`` creates module-level storage even where the top level binds nothing"""
    __test_helper_rejects("""
@lib.script.indicator("T")
def main():
    global TICKS
    TICKS = 1
""", 'TICKS')
    log.info("a global with no module-level binding was rejected")


def __test_a_nonlocal_is_allowed__(log):
    """``nonlocal`` names the enclosing function's variable, never the module's"""
    __test_helper_analyse("""
COUNT = 0


@lib.script.indicator("T")
def main():
    total = 0

    def add(value):
        nonlocal total
        total = total + value

    add(1)
""")
    log.info("a nonlocal write of an enclosing local was accepted")


def __test_a_write_in_a_helper_is_rejected__(log):
    """The rule is about functions, not about ``main`` alone"""
    __test_helper_rejects("""
STORE = lib.array.new_float(0)


def record(value):
    lib.array.push(STORE, value)


@lib.script.indicator("T")
def main():
    record(lib.close)
""", 'STORE')


def __test_the_error_carries_the_file_and_the_line__(log):
    """The diagnostic is located, like every other pipeline error"""
    error = __test_helper_rejects("""
BUF = [0.0]


@lib.script.indicator("T")
def main():
    BUF[0] = lib.close
""", 'BUF')
    assert error.filename == str(_PATH.resolve())
    assert error.lineno == 6


def __test_a_local_of_the_same_name_is_not_the_outer_object__(log):
    """A local binding shadows the module-level one for the whole call

    The shape is the one the ZigZag library writes: an exported function and a
    local variable share the spelling, and the local is what the body writes.
    """
    __test_helper_analyse("""
lastPivot = lib.array.new_float(0)


@lib.script.indicator("T")
def main():
    lastPivot = lib.array.new_float(1)
    lib.array.set(lastPivot, 0, lib.close)
    lib.plot(lib.array.get(lastPivot, 0))
""")


def __test_a_parameter_of_the_same_name_is_not_the_outer_object__(log):
    """A parameter shadows the module-level binding too"""
    __test_helper_analyse("""
STORE = lib.array.new_float(0)


def record(STORE, value):
    lib.array.push(STORE, value)


@lib.script.indicator("T")
def main():
    own = lib.array.new_float(0)
    record(own, lib.close)
""")


def __test_reading_the_outer_object_is_allowed__(log):
    """Only writing is rejected; every read stays as it was"""
    __test_helper_analyse("""
STORE = lib.array.new_float(1, 0.0)
LIMIT = 3.0


@lib.script.indicator("T")
def main():
    lib.plot(lib.array.get(STORE, 0) + LIMIT + lib.array.size(STORE))
""")


def __test_a_non_literal_outer_definition_is_allowed__(log):
    """Defining is free — ``color.new(...)``, ``strategy.fixed``, an object"""
    __test_helper_analyse("""
COL = lib.color.new(lib.color.red, 50)
QTY_TYPE = lib.strategy.fixed


@lib.script.strategy("T", default_qty_type=QTY_TYPE)
def main():
    lib.plot(lib.close, color=COL)
""")


def __test_a_test_prefixed_function_is_exempt__(log):
    """Harness code is not script code — the AOT exporter skips it as well"""
    __test_helper_analyse("""
STORE = lib.array.new_float(0)


def __test_helper_record(value):
    lib.array.push(STORE, value)


@lib.script.indicator("T")
def main():
    lib.plot(lib.array.size(STORE))
""")


def __test_a_non_mutating_builtin_is_allowed__(log):
    """``array.slice`` / ``array.copy`` hand back a value, they write nothing"""
    __test_helper_analyse("""
STORE = lib.array.new_float(2, 0.0)


@lib.script.indicator("T")
def main():
    own = lib.array.copy(STORE)
    lib.array.set(own, 0, lib.close)
    lib.plot(lib.array.get(own, 0))
""")
