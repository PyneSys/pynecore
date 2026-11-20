"""
The typing coverage the Pine inference owes the corpus.

The inference is complete on the Pine-expressible subset or it is nothing: an
UNKNOWN in a value position is what the ``@pyne edge`` gate rejects, and it is
also what drops the overload pin of every expression built on top of it. The
gaps closed here are the ones a census over the live compiled corpus MEASURED,
grouped the way they were found:

* the lib CONSTANTS a namespace publishes as bare constructor calls,
* the STATEMENT calls (``plot()``, ``strategy.entry()``) whose result nobody
  reads and which are VOID rather than untypable,
* the type-preserving families -- generic ``input``, ``ta.change``, ``nz``,
* the typed ``na`` constructors,
* the PLUMBING earlier passes emit around a script, which is not something the
  script wrote and must not read as a typing failure,
* the CONSTRUCTORS, which name a class rather than a function,
* and the string-dispatched ``method_call``.

The type expectations that are not mechanical are MEASURED on TradingView; the
measurement is named where it is used.
"""
import ast
from pathlib import Path

import pytest

from ..pine_analysis import analyse_module
from pynecore.errorcodes import CALL_ARGUMENT_TWICE, CALL_UNKNOWN_PARAMETER
from pynecore.transformers.lib_registry import lib_types
from pynecore.transformers.pine_type_infer import infer_module
from pynecore.transformers.pine_type_rules import (
    BOOL, FLOAT, INT, OBJECT, STR, UNKNOWN, annotation_type, builtin_class_id, get_ty, object_ty,
    tuple_of,
)


def _types(source: str, scope: str = '') -> dict[str, str]:
    """Infer a snippet and return one scope's bindings as name -> type."""
    tree = ast.parse(source)
    table = infer_module(tree, 'test')
    return {name: binding.ty for name, binding in table.bindings.get(scope, {}).items()}


def _expr_type(expression: str, preamble: str = '') -> str:
    """Infer one expression in the standard measured setting."""
    source = (
        'from pynecore import lib\n'
        'R = lib.input.int(14)\n'
        'd = R / 8\n'
        f'{preamble}'
        f'value = {expression}\n'
    )
    return _types(source)['value']


# --- 1. the lib registry's constants ---------------------------------------

@pytest.mark.parametrize("name,expected", [
    # An object-annotated namespace constant: ``data_window: Display = Display()``
    ('display.data_window', 'o:lib#Display'),
    ('display.none', 'o:lib#Display'),
    # An unannotated constructor call: ``white = Color('#FFFFFF')``
    ('color.white', 'c'),
    ('color.green', 'c'),
    # ... and the enum namespaces built the same way
    ('barmerge.gaps_off', 'o:lib#BarMerge'),
    ('hline.style_dashed', 'o:lib#HLineEnum'),
    ('xloc.bar_time', 'o:lib#XLoc'),
    ('line.style_dashed', 'o:lib#LineEnum'),
    ('location.absolute', 'o:lib#Location'),
    ('position.top_right', 'o:lib#Position'),
    ('plot.style_columns', 'o:lib#PlotEnum'),
    ('extend.none', 'o:lib#Extend'),
    ('format.mintick', 'o:lib#Format'),
    # A cross-namespace alias: ``strategy/__init__.py`` spells
    # ``long = direction.long``
    ('strategy.long', 'o:lib#Direction'),
    ('strategy.short', 'o:lib#Direction'),
    # A bool constant, which is a plain literal assignment
    ('barstate.islast', 'b'),
    ('barstate.isfirst', 'b'),
    # An annotated primitive, which the collector always had
    ('syminfo.mintick', 'f'),
    ('syminfo.tickerid', 's'),
])
def __test_the_registry_types_every_constant_the_corpus_reads__(name: str, expected: str):
    """A namespace constant is a value with a type, not an untypable name"""
    assert _expr_type(f'lib.{name}') == expected, name


def __test_a_namespace_reference_is_an_object__():
    """``lib.ta`` is the module itself, which is known and non-scalar"""
    assert _expr_type('lib.ta') == 'o'
    # ``chart.point`` is a namespace that is an INSTANCE, so it carries the
    # class it is an instance of
    assert _expr_type('lib.chart.point') == 'o:lib#_ChartPoint'


def __test_a_misspelled_lib_name_is_still_unknown__():
    """The constant rule must not turn every dotted lib path into a type"""
    assert _expr_type('lib.color.chartreuse') == UNKNOWN
    assert _expr_type('lib.display.nowhere') == UNKNOWN


# --- 2. statement calls -----------------------------------------------------

@pytest.mark.parametrize("call", [
    'lib.strategy.entry("id", lib.strategy.long)',
    'lib.strategy.cancel("id")',
    'lib.strategy.close_all()',
    'lib.alertcondition(True)',
    'lib.runtime.error("boom")',
    'lib.line.delete(lib.na(lib.Line))',
])
def __test_a_call_that_returns_nothing_is_void__(call: str):
    """A body with no ``return <value>`` is a statement, not an unknown"""
    assert _expr_type(call) == 'v', call


def __test_plot_returns_the_plot_object__():
    """``plot()`` hands back a Plot, which ``fill()`` takes -- so it is an object"""
    assert _expr_type('lib.plot(lib.close)') == 'o:lib#Plot'
    assert _expr_type('lib.plot.plot(lib.close)') == 'o:lib#Plot'


def __test_a_container_read_stays_unknown__():
    """
    ``array.get`` needs the ELEMENT type, which this pass does not carry.

    Guessing one here would be worse than saying nothing: the guess reaches
    the enclosing overload pin and selects an implementation the runtime would
    not.
    """
    for entry in ('array.get', 'array.pop', 'array.last', 'array.shift', 'array.remove'):
        assert lib_types()[entry]['ret'] == UNKNOWN, entry


# --- 3. the generic input() -------------------------------------------------

@pytest.mark.parametrize("call,expected", [
    # MEASURED (FX:EURUSD@60): ``int g = input(14)``, ``float h = input(1.5)``,
    # ``bool i = input(true)``, ``string j = input("x")`` and
    # ``color k = input(color.red)`` all compile
    ('lib.input(14)', 'i'),
    ('lib.input(1.5)', 'f'),
    ('lib.input(True)', 'b'),
    ('lib.input("x")', 's'),
    ('lib.input(lib.color.red)', 'c'),
    ('lib.input(lib.color.new(lib.color.red, 50))', 'c'),
    # The default is spellable by keyword, and the type must not change with it
    ('lib.input(title="Length", defval=10)', 'i'),
])
def __test_the_generic_input_is_typed_by_its_default__(call: str, expected: str):
    """``input(defval)`` evaluates to the default's own type"""
    assert _expr_type(call) == expected, call


def __test_an_input_with_no_typeable_default_is_unknown__():
    """The rule copies a type, it does not invent one"""
    assert _expr_type('lib.input(unknown_thing)') == UNKNOWN


# --- 4. the type-preserving families ---------------------------------------

@pytest.mark.parametrize("call,expected", [
    # MEASURED: ``int a = ta.change(bar_index)``, ``float b = ta.change(close)``
    # and ``bool c = ta.change(close > open)`` all compile
    ('lib.ta.change(lib.bar_index)', 'i'),
    ('lib.ta.change(lib.close)', 'f'),
    ('lib.ta.change(lib.close > lib.open)', 'b'),
    ('lib.ta.change(d, 2)', 'i'),
    # MEASURED: ``int e = nz(R, 1.0)`` is REJECTED with CE10173 ("simple
    # float"), while ``nz(R, 2)`` and ``nz(R)`` are both int -- so nz joins
    ('lib.nz(d)', 'i'),
    ('lib.nz(d, 2)', 'i'),
    ('lib.nz(d, 1.0)', 'f'),
    ('lib.nz(lib.close, 0)', 'f'),
    # fixnan takes one argument, so joining and copying agree
    ('lib.fixnan(d)', 'i'),
    ('lib.fixnan(lib.close)', 'f'),
])
def __test_the_source_type_travels_through__(call: str, expected: str):
    """``ta.change``, ``nz`` and ``fixnan`` carry their source's type"""
    assert _expr_type(call) == expected, call


@pytest.mark.parametrize("call,expected", [
    # Read bare it is the variable form, the call without arguments
    ('lib.ta.vwap', 'f'),
    ('lib.ta.vwap()', 'f'),
    ('lib.ta.vwap(lib.hlc3)', 'f'),
    ('lib.ta.vwap(lib.high, lib.high > lib.open)', 'f'),
    ('lib.ta.vwap(source=lib.close, anchor=lib.close > lib.open)', 'f'),
    # The band multiplier turns the value into the value with its two bands
    ('lib.ta.vwap(lib.hlc3, lib.close > lib.open, 1)', tuple_of([FLOAT, FLOAT, FLOAT])),
    ('lib.ta.vwap(lib.hlc3, stdev_mult=2.0)', tuple_of([FLOAT, FLOAT, FLOAT])),
    # A ``None`` spelled out is the default, no bands
    ('lib.ta.vwap(stdev_mult=None)', 'f'),
    ('lib.ta.vwap(lib.hlc3, lib.close > lib.open, None)', 'f'),
    # An unpacking hides whether the multiplier is passed
    ('lib.ta.vwap(*unknown_thing)', UNKNOWN),
])
def __test_vwap_is_a_value_or_the_value_with_its_bands__(call: str, expected: str):
    assert _expr_type(call) == expected


def __test_a_vwap_multiplier_bound_to_none_is_the_default__():
    # A name bound to ``None`` passes the default as well as the literal does,
    # and a na multiplier is still a multiplier
    assert _expr_type('lib.ta.vwap(stdev_mult=mult)', preamble='mult = None\n') == FLOAT
    assert _expr_type('lib.ta.vwap(lib.hlc3, lib.close > lib.open, mult)',
                      preamble='mult = None\n') == FLOAT
    assert _expr_type('lib.ta.vwap(stdev_mult=mult)', preamble='mult = lib.na\n') \
        == tuple_of([FLOAT, FLOAT, FLOAT])


@pytest.mark.parametrize("expression,expected", [
    # A set of flags combines with another into one of the same class
    ('lib.display.all - lib.display.status_line', 'o:lib#Display'),
    ('lib.display.pane + lib.display.data_window', 'o:lib#Display'),
    ('lib.text.format_bold + lib.text.format_italic', 'o:lib#FormatEnum'),
    # ... and with nothing else, nor through any other operator
    ('lib.display.all - lib.text.format_bold', UNKNOWN),
    ('lib.display.all - 1', UNKNOWN),
    ('lib.display.all * lib.display.pane', UNKNOWN),
    # An object that is no set of flags has no arithmetic
    ('lib.xloc.bar_index + lib.xloc.bar_time', UNKNOWN),
])
def __test_two_flag_sets_combine_into_one_of_their_class__(expression: str, expected: str):
    assert _expr_type(expression) == expected


def __test_a_flag_set_is_updated_in_place__():
    types = _types('from pynecore import lib\n'
                   'shown = lib.display.all\n'
                   'shown -= lib.display.status_line\n')
    assert types['shown'] == 'o:lib#Display'


@pytest.mark.parametrize("expression,expected", [
    # A lib constant Pine spells as a string is one type with a string
    ("lib.format.volume if lib.close > 0 else '#.##'", STR),
    ("'#.##' if lib.close > 0 else lib.format.volume", STR),
    # ... and with the constant of another such class
    ('lib.extend.none if lib.close > 0 else lib.line.style_solid', STR),
    # Two of one class keep it
    ('lib.size.small if lib.close > 0 else lib.size.large', 'o:lib#Size'),
    # A set of flags is a type of its own, and so is an object
    ('lib.display.all if lib.close > 0 else 1', UNKNOWN),
    ("lib.format.volume if lib.close > 0 else lib.line.new(0, 0, 1, 1)", UNKNOWN),
])
def __test_a_string_constant_of_the_lib_joins_with_a_string__(expression: str, expected: str):
    assert _expr_type(expression) == expected


def __test_a_string_parameter_takes_a_lib_constant_as_its_default__():
    table = infer_module(ast.parse(
        'from pynecore import lib\n\n\n'
        'def cell(alignment: str = lib.text.align_right):\n    return alignment\n\n\n'
        'def main():\n    return cell()\n'), 'test')

    assert not table.diags
    assert table.bindings['cell']['alignment'].ty == STR


def __test_an_untypable_source_does_not_become_a_type__():
    """A family rule copies what it is given, including nothing"""
    assert _expr_type('lib.ta.change(unknown_thing)') == UNKNOWN
    assert _expr_type('lib.nz(unknown_thing, 1)') == UNKNOWN


# --- 5. the typed na --------------------------------------------------------

@pytest.mark.parametrize("call,expected", [
    ('lib.na(int)', 'i'),
    ('lib.na(float)', 'f'),
    ('lib.na(bool)', 'b'),
    ('lib.na(str)', 's'),
    ('lib.na(lib.Color)', 'c'),
    ('lib.na(lib.Line)', 'o:lib#Line'),
    ('lib.na(lib.ChartPoint)', 'o:lib#ChartPoint'),
])
def __test_a_typed_na_is_of_the_type_it_names__(call: str, expected: str):
    """``na(int)`` builds an na OF int, it does not test one"""
    assert _expr_type(call) == expected, call


def __test_the_na_predicate_is_still_a_bool__():
    """``na(x)`` on a VALUE is the predicate, whatever the constructor face does"""
    assert _expr_type('lib.na(d)') == 'b'
    assert _expr_type('lib.na(lib.close)') == 'b'
    # A name the script binds is a value even when it is spelled like a type
    assert _expr_type('lib.na(float)', preamble='float = 3\n') == 'b'


def __test_the_bare_NA_constructor_is_typed_too__():
    """The compiled form spells a declared na ``NA(Line)``"""
    source = (
        'from pynecore import lib\n'
        'from pynecore.types import NA, Color, Line\n'
        'a = NA(int)\n'
        'b = NA(Color)\n'
        'c = NA(Line)\n'
        'd = NA(lib.close)\n'
    )
    types = _types(source)
    assert (types['a'], types['b'], types['c']) == ('i', 'c', 'o:lib#Line')
    # Not a type name: the na object itself, which is Pine's TYPELESS marker
    # -- it carries no type of its own and takes the one of whatever it meets
    assert types['d'] == '*'


# --- 6. the plumbing earlier passes emit ------------------------------------

def __test_a_walrus_binds_the_type_it_captures__():
    """``PineTruthinessTransformer``'s temporary reads back as what it bound"""
    types = _types(
        'from pynecore import lib\n'
        'value = (__bool1__ := lib.close) > 0\n'
        'echo = __bool1__\n'
    )
    assert types['__bool1__'] == 'f'
    assert types['echo'] == 'f'


def __test_the_class_test_is_typed_end_to_end__():
    """``x.__class__ is float`` carries no untyped node anywhere in it"""
    tree = ast.parse(
        'from pynecore import lib\n'
        'value = (__bool1__ := lib.close).__class__ is float\n'
    )
    infer_module(tree, 'test')
    untyped = [ast.unparse(node) for node in ast.walk(tree)
               if isinstance(node, ast.expr) and get_ty(node) == UNKNOWN]
    assert not untyped, untyped


def __test_the_module_tail_is_typed__():
    """``if __name__ == '__main__': run(__file__)`` closes every compiled script"""
    types = _types(
        'from pynecore.standalone import run\n'
        'name = __name__\n'
        'here = __file__\n'
        'started = run(__file__)\n'
    )
    assert types['name'] == 's'
    assert types['here'] == 's'
    assert types['started'] == 'v'


def __test_a_security_read_is_typed_from_its_write__():
    """The transform splits one expression in two; the read is the write's type"""
    types = _types(
        'from pynecore import lib\n'
        "def main():\n"
        "    if __active_security__ is not None "
        "and 'sec-1' in __active_security__:\n"
        "        __sec_write__('sec-1', lib.close * 2)\n"
        "    value = __sec_read__('sec-1', lib._na_none)\n"
        "    written = __sec_write__('sec-4', lib.close)\n"
        "    signalled = __sec_signal__('sec-1', 'AAPL', '60')\n"
        "    waited = __sec_wait__('sec-1')\n",
        scope='main')
    assert types['value'] == 'f'
    assert types['written'] == 'v'
    assert types['signalled'] == 'v'
    assert types['waited'] == 'v'


def __test_a_security_read_joins_a_typed_default__():
    """A default that is not the typeless ``na`` is a second possible value"""
    types = _types(
        'from pynecore import lib\n'
        "def main():\n"
        "    __sec_write__('sec-2', lib.bar_index)\n"
        "    same = __sec_read__('sec-2', 0)\n"
        "    widened = __sec_read__('sec-2', 0.0)\n",
        scope='main')
    assert types['same'] == 'i'
    assert types['widened'] == 'f'


def __test_a_security_read_with_no_write_is_unknown__():
    """Nothing is claimed for an id this module never publishes"""
    types = _types(
        'from pynecore import lib\n'
        "def main():\n"
        "    value = __sec_read__('sec-3', lib._na_none)\n",
        scope='main')
    assert types['value'] == UNKNOWN


def __test_the_process_identity_globals_are_known__():
    """The security guards are comparisons, and their operands are not values"""
    types = _types(
        "def main():\n"
        "    here = __active_security__\n"
        "    shared = __same_context__\n"
        "    is_chart = __active_security__ is None\n"
        "    is_active = 'sec-1' in __active_security__\n"
        "    is_same = 'sec-1' in __same_context__\n",
        scope='main')
    assert types['here'] == 'o'
    assert types['shared'] == 'o'
    assert types['is_chart'] == 'b'
    assert types['is_active'] == 'b'
    assert types['is_same'] == 'b'


# --- 7. constructors --------------------------------------------------------

def __test_an_in_module_class_is_constructed_to_an_object__():
    """A UDT compiles to a dataclass, and constructing one yields the object"""
    types = _types(
        'class SessionInfo:\n'
        '    start: int\n'
        '    end: int\n'
        'built = SessionInfo(1, 2)\n'
        'made = SessionInfo.new(1, 2)\n'
        'missing = NotAClass(1)\n'
    )
    assert types['built'] == 'o:test#SessionInfo'
    assert types['made'] == 'o:test#SessionInfo'
    assert types['missing'] == UNKNOWN


def __test_pine_range_is_an_object_and_its_counter_keeps_the_bounds__():
    """
    MEASURED: a Pine ``for`` does not truncate, and the counter has the type of ``from``.

    The ITERABLE is an ordinary known non-scalar; only the counter carries the
    algebra. A float ``to`` leaves an int ``from``'s counter an int.
    """
    types = _types(
        'from pynecore import lib, pine_range\n'
        'R = lib.input.int(14)\n'
        'span = pine_range(0, R, 1)\n'
        'for i in pine_range(0, R, 1):\n'
        '    counter = i\n'
        'for j in pine_range(0.0, R, 1):\n'
        '    widened = j\n'
        'for k in pine_range(1, lib.input.float(3.5)):\n'
        '    kept = k\n'
    )
    assert types['span'] == 'o'
    assert types['counter'] == 'i'
    assert types['widened'] == 'f'
    assert types['kept'] == 'i'


# --- 8. method_call ---------------------------------------------------------

@pytest.mark.parametrize("call,expected", [
    # Only ``box`` publishes these, so there is one implementation to reach
    ('method_call("get_top", drawing)', 'f'),
    ('method_call("set_right", drawing, 1)', 'v'),
    # Every namespace that publishes ``delete`` returns nothing
    ('method_call("delete", drawing)', 'v'),
])
def __test_a_builtin_method_call_resolves_where_the_candidates_agree__(
        call: str, expected: str):
    """The name selects the namespace family; agreement is the answer"""
    assert _expr_type(call, preamble='drawing = lib.box.new(0, 0.0, 1, 1.0)\n') == expected, call


def __test_a_method_call_stays_unknown_where_it_cannot_be_settled__():
    """A container read, a user method and a dynamic selector all decline"""
    preamble = ('drawing = lib.box.new(0, 0.0, 1, 1.0)\n'
                'opaque = lib.map.new()\n')
    # ``array.get``/``matrix.get``/``map.get`` all need the element type
    assert _expr_type('method_call("get", drawing, 0)', preamble) == UNKNOWN
    # No namespace publishes it, and this pass does not carry the receiver's class
    assert _expr_type('method_call("lastPivot", drawing)', preamble) == UNKNOWN
    # A callable selector on a receiver of NO known class: the runtime would
    # try the builtin namespaces by the receiver's class first and only then
    # call what it was handed, and neither half is answerable here
    assert _expr_type('method_call(delete, opaque)', preamble) == UNKNOWN


def __test_a_user_function_shadows_the_builtin_namespaces__():
    """
    A user method of the same name only wins where the receiver's class does not.

    ``core.pine_method.method_call`` tries ``_get_builtin_method`` FIRST, by
    the receiver's runtime class, and only falls through to the user-method
    dispatch when that answers nothing -- so a ``Box`` receiver reaches
    ``box.delete`` whatever else is in scope, and the static answer follows.
    Where the receiver's class is NOT known the ambiguity is real again, and
    the user definition takes the answer away.
    """
    source = (
        'from pynecore import lib\n'
        'from pynecore.core.pine_method import method_call\n'
        'def delete(this) -> int:\n'
        '    return 1\n'
        'drawing = lib.box.new(0, 0.0, 1, 1.0)\n'
        'value = method_call("delete", drawing)\n'
    )
    assert _types(source)['value'] == 'v'
    assert _types(source.replace('drawing = lib.box.new(0, 0.0, 1, 1.0)',
                                 'drawing = unknown_thing'))['value'] == UNKNOWN


# --- the whole analysed pipeline -------------------------------------------

#: A script that reaches BOTH plumbing families at once: a non-bool ``if`` test
#: is what ``PineTruthinessTransformer`` rewrites into a ``__class__`` compare
#: over a bound temporary, and ``request.security`` is what
#: ``SecurityTransformer`` splits into a write, a signal, a wait and a read.
_PLUMBED = '''"""
@pyne
"""
from pynecore.lib import close, high, low, open, plot, request, script, ta, timeframe


@script.indicator("plumbing")
def main():
    hot = close > open
    if hot:
        upper = high
    else:
        upper = low
    if close - open:
        lower = low
    else:
        lower = high
    plot(upper)
    plot(lower)
    daily = request.security(timeframe.period, "D", ta.sma(close, 5))
    plot(daily + low)


if __name__ == "__main__":
    from pynecore.standalone import run
    run(__file__)
'''


def _value_positions(tree: ast.Module) -> list[ast.expr]:
    """
    Every expression whose TYPE a consumer reads.

    What is left out is structural rather than valued: a callee chain, an
    annotation, a decorator, a store target, a docstring, and the base of an
    attribute chain -- the ``lib`` of ``lib.volume`` is a namespace reference,
    not a value.

    :param tree: The analysed module
    :return: The value-position expressions, in walk order
    """
    skip: set[int] = set()

    def bury(node: ast.AST) -> None:
        for child in ast.walk(node):
            skip.add(id(child))

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            bury(node.func)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for decorator in node.decorator_list:
                bury(decorator)
            for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
                if arg.annotation is not None:
                    bury(arg.annotation)
            if node.returns is not None:
                bury(node.returns)
        elif isinstance(node, ast.AnnAssign):
            bury(node.annotation)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            skip.add(id(node.value))
        elif isinstance(node, ast.Attribute):
            skip.add(id(node.value))
        elif not isinstance(getattr(node, 'ctx', ast.Load()), ast.Load):
            bury(node)
    return [node for node in ast.walk(tree)
            if isinstance(node, ast.expr) and id(node) not in skip]


def __test_the_analysed_pipeline_leaves_no_untyped_value__(tmp_path: Path):
    """
    The plumbing the earlier passes emit is typed along with the script.

    Run over the REAL pipeline prefix, the same one an import takes: what a
    hand-built snippet cannot show is that the truthiness rewrite and the
    security split leave value positions behind, and each of those is an
    expression the ``@pyne edge`` gate would reject.
    """
    script = tmp_path / 'plumbed.py'
    script.write_text(_PLUMBED)
    analysed = analyse_module(str(script))
    assert analysed is not None, "the analysis did not recognize the script"
    tree = analysed[0]

    emitted = {node.func.id for node in ast.walk(tree)
               if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert '__sec_read__' in emitted, "the security split did not run"
    assert any(isinstance(node, ast.Attribute) and node.attr == '__class__'
               for node in ast.walk(tree)), "the truthiness rewrite did not run"

    untyped = [ast.unparse(node) for node in _value_positions(tree)
               if get_ty(node) == UNKNOWN]
    assert not untyped, f'{len(untyped)} untyped value position(s): {untyped}'


def __test_a_bare_wrapper_declaration_takes_the_type_of_its_value__():
    """``x: Series = expr`` says how x lives; what it holds is what expr is"""
    types = _types('''
from pynecore import lib
from pynecore.types import Series, Persistent


def main(length: int):
    ema: Series = lib.ta.ema(lib.close, length)
    flag: Persistent = length != 3
    ln: Persistent = lib.line.new(1, 1.0, 2, 2.0)
    later: Series
    later = ema * 2
    declared: Series[int] = 1
    return ema + later
''', 'main')
    assert types['ema'] == FLOAT
    assert types['flag'] == BOOL
    assert types['ln'] == object_ty(builtin_class_id('Line'))
    assert types['later'] == FLOAT
    assert types['declared'] == INT


def __test_a_declared_series_rejects_a_value_it_does_not_hold__():
    """``Series[int] = 1.5`` is Pine's ``int x = 1.5``: a declaration the value contradicts"""
    source = '''
from pynecore.types import Series


def main():
    declared: Series[int] = 1.5
    return declared
'''
    table = infer_module(ast.parse(source), 'test')
    binding = table.binding('main', 'declared')
    assert binding is not None and binding.ty == UNKNOWN and binding.series
    assert binding.unknown is not None and binding.unknown.reason == 'type-mismatch'
    assert [(d.origin.reason, d.line) for d in table.diags if d.origin is not None] \
        == [('type-mismatch', 6)]


def __test_a_dynamic_default_is_absence_until_the_prologue_assigns__():
    """
    ``DynamicDefaultTransformer`` moves a ``lib``-reading default behind a
    sentinel and a prologue; a caller that omits the argument passes the
    sentinel, and the parameter's type is what the prologue assigns.
    """
    types = _types('''
from pynecore import lib
from pynecore.core.instance_state import __dyn_default__


def calc(price, compared=__dyn_default__):
    if compared is __dyn_default__:
        compared = lib.close
    return price >= compared


def main():
    one = calc(lib.high)
    two = calc(lib.high, lib.low)
    return one and two
''', 'calc')
    assert types['compared'] == FLOAT
    assert types['price'] == FLOAT


# --- a function nothing calls ------------------------------------------------


NEVER_CALLED = '''
from pynecore import lib


def scale(value):
    return value * 2.0


def main():
    start = lib.input.int(10)
{body}
    return scale(lib.close)
'''


@pytest.mark.parametrize('body', [
    # Its parameter has no type, and neither has anything built on it
    '    def mid(length):\n        return (lib.ta.highest(length) + lib.ta.lowest(length)) / 2\n',
    # A helper it calls is only typed by the calls that are made
    '    def twice(value):\n        return scale(value)\n',
    # One it would have called is never called either
    '    def outer(value):\n        return inner(value)\n\n'
    '    def inner(value):\n        return value + start\n',
])
def __test_a_nested_function_nothing_calls_has_no_instance__(body: str):
    """Pine types a function per call, so an untyped parameter of one never called is no gap"""
    table = infer_module(ast.parse(NEVER_CALLED.format(body=body)), 'test')

    assert not table.diags
    assert table.bindings['scale']['value'].ty == FLOAT
    assert not any(scope.startswith('main·') for scope in table.bindings)


@pytest.mark.parametrize('body, scope', [
    # Every parameter states its type: the definition describes its one instance
    ('    def shifted(length: int):\n        return missing(length)\n', 'main·shifted'),
    # A parameterless one is its own instance too
    ('    def level():\n        return missing(start)\n', 'main·level'),
])
def __test_a_nested_function_that_states_its_types_is_still_typed__(body: str, scope: str):
    table = infer_module(ast.parse(NEVER_CALLED.format(body=body)), 'test')

    assert scope in table.bindings
    assert [diag.origin.reason for diag in table.diags if diag.origin] == ['unknown-call']


def __test_a_module_level_function_nothing_calls_is_still_typed__():
    """An importer may call it, so the context its definition describes stands"""
    table = infer_module(ast.parse(
        'def spread(high, low):\n    return high - low\n\n\ndef main():\n    return 1\n'), 'test')

    assert {diag.origin.detail for diag in table.diags if diag.origin} == {'high', 'low'}


# --- a lib parameter filled twice --------------------------------------------


def _codes(statement: str) -> list[str]:
    """The diagnostic codes one statement over the lib reports."""
    table = infer_module(ast.parse(f'from pynecore import lib\n{statement}\n'), 'test')
    return [diag.code for diag in table.diags]


@pytest.mark.parametrize('statement', [
    # A function
    'x = lib.ta.sma(lib.close, source=lib.close)',
    'x = lib.ta.ema(lib.close, source=lib.close)',
    # A group, each of whose implementations starts with the same parameter
    'x = lib.ta.highest(lib.close, 5, source=lib.high)',
    'x = lib.math.round(1.5, number=2.5)',
])
def __test_a_keyword_naming_a_filled_lib_parameter_is_reported__(statement: str):
    """Python raises on a parameter given a positional and a keyword argument"""
    assert _codes(statement) == [CALL_ARGUMENT_TWICE.code]


@pytest.mark.parametrize('statement', [
    'x = lib.ta.sma(lib.close, length=5)',
    'x = lib.ta.sma(source=lib.close, length=5)',
    'x = lib.math.round(1.5, precision=2)',
    # The one-argument implementation names ``length`` first; the two-argument
    # one takes ``close`` for its source and the keyword for its length
    'x = lib.ta.highest(lib.close, length=5)',
])
def __test_a_keyword_naming_a_free_lib_parameter_is_taken__(statement: str):
    """A group is called wrongly only where no implementation binds the call"""
    assert not _codes(statement)


# --- the variables one local name spells --------------------------------------


def _main_types(tmp_path: Path, body: str) -> tuple[dict[str, str], list[str]]:
    """Analyse a script whose ``main`` holds a body: its bindings and its diagnostic codes."""
    lines = ''.join(f'    {line}\n' for line in body.splitlines())
    script = tmp_path / 'segmented.py'
    script.write_text('"""@pyne"""\nfrom pynecore import lib\nfrom pynecore.lib import script\n\n\n'
                      f'@script.indicator("segmented")\ndef main():\n{lines}')
    analysed = analyse_module(str(script))
    assert analysed is not None, "the analysis did not recognize the script"
    table = analysed[1]
    return ({name: binding.ty for name, binding in table.bindings['main'].items()},
            [diag.code for diag in table.diags])


@pytest.mark.parametrize('body,expected', [
    # Two blocks each declaring their own: Pine scopes a variable to its block
    ('if lib.bar_index > 0:\n    v = 1\n    lib.math.abs(v)\n'
     'if lib.bar_index > 1:\n    v = "s"\n    w = v + "x"',
     {'v': INT, 'v@2': STR}),
    # ... and so does a loop, for the variable it steps
    ('for v in lib.array.new_int(1, 0):\n    lib.math.abs(v)\n'
     'for v in lib.array.new_string(1, "s"):\n    w = v + "x"',
     {'v': INT, 'v@2': STR}),
    # A statement of the same block storing a type the variable cannot hold
    # starts another variable: nothing read above it is stored below
    ('v = 1\nlib.math.abs(v)\nv = "s"\nw = v + "x"\nv = 2\nlib.math.abs(v)',
     {'v': INT, 'v@2': STR, 'v@3': INT}),
    # Two blocks may hold a number each without the float reaching the int
    ('if lib.bar_index > 0:\n    v = 1\n    lib.math.abs(v)\n'
     'if lib.bar_index > 1:\n    v = 2.5\n    lib.math.abs(v)',
     {'v': INT, 'v@2': FLOAT}),
])
def __test_a_local_name_spelling_several_variables_is_bound_once_per_variable__(
        tmp_path: Path, body: str, expected: dict[str, str]):
    types, codes = _main_types(tmp_path, body)
    assert {name: ty for name, ty in types.items() if name != 'w'} == expected
    assert not codes


@pytest.mark.parametrize('body,expected', [
    # Pine's ``:=`` on a variable declared above: one variable, a float from the start
    ('v = 1\nlib.math.abs(v)\nv = 2.5\nlib.math.abs(v)', {'v': FLOAT}),
    # A value read where no variable of the name was declared in the block
    ('if lib.bar_index > 0:\n    v = 1\nelse:\n    v = 2.5\nlib.math.abs(v)', {'v': FLOAT}),
    # A loop reads above the store what the round before stored
    ('n = 0\nwhile n < 3:\n    n = n + 1\n    v = n\n    lib.math.abs(v)', {'n': INT, 'v': INT}),
])
def __test_a_local_name_spelling_one_variable_keeps_one_binding__(
        tmp_path: Path, body: str, expected: dict[str, str]):
    types, codes = _main_types(tmp_path, body)
    assert types == expected
    assert not codes


@pytest.mark.parametrize('body', [
    # A branch that assigns leaves the variable holding either value
    'v = 1.0\nif lib.bar_index > 0:\n    v = "s"',
    # Two branches read after them are one variable
    'if lib.bar_index > 0:\n    v = 1\nelse:\n    v = "s"\nw = v + "x"',
    # A loop body storing what the next round reads above it
    'v = 1\nwhile lib.bar_index > 0:\n    lib.math.abs(v)\n    v = "s"',
])
def __test_a_variable_holding_two_types_is_still_reported__(tmp_path: Path, body: str):
    """Pine rejects a variable whose assignments disagree"""
    _, codes = _main_types(tmp_path, body)
    assert codes[:1] == ['E0023']


def __test_a_name_a_nested_function_reads_is_one_variable__():
    """The nested body reads whichever store ran last, so the stores are not kept apart"""
    table = infer_module(ast.parse(
        'def main():\n    v = 1\n    def read():\n        return v\n    v = "s"\n    read()\n'
        'main()\n'), 'test')
    assert set(table.bindings['main']) == {'v'}
    assert table.diags[0].code == 'E0023'


def __test_a_nested_function_splits_its_own_locals__(tmp_path: Path):
    """A segment belongs to its function: the enclosing one's same-named segments say nothing"""
    types, codes = _main_types(
        tmp_path,
        'v = "a"\nw = v + "x"\nv = "b"\nw = v + "x"\n'
        'def inner():\n    v = "a"\n    w = v + "x"\n    v = 2\n    return v + 1\ninner()')
    assert not codes
    assert types == {'v': STR, 'w': STR}


# --- a body typed differently per call site, scaffolding, a loop over a map ---

@pytest.mark.parametrize('body', [
    # A scalar of another type at each call site
    'def pick(value, other):\n    return other if lib.bar_index > 0 else value\n'
    'a = pick(1, 2)\nb = pick("x", "y")',
    # A container of another element type at each call site
    'def push(values, value):\n    lib.array.remove(values, 0)\n'
    '    return lib.array.unshift(values, value)\n'
    'push(lib.array.new_int(1, 0), 1)\npush(lib.array.new_float(1, 0.0), 1.5)',
])
def __test_a_body_typed_differently_per_call_site_is_typed__(tmp_path: Path, body: str):
    """Pine instantiates a function per call: what one stamp cannot spell is still no gap"""
    _, codes = _main_types(tmp_path, body)
    assert not codes


def __test_a_value_one_call_site_leaves_untyped_is_still_reported__():
    """A call site passing a value of no known type is a gap, whatever the others pass"""
    table = infer_module(ast.parse(
        'def main(untyped):\n    def pick(value):\n        return value\n'
        '    a = pick(1)\n    b = pick("x")\n    c = pick(untyped)\n'), 'test')
    # The read inside the helper, and the argument that brought the gap
    assert [(diag.code, diag.line) for diag in table.diags] == [('E0012', 3), ('E0012', 6)]


SHIM_WITH_A_REQUIRED_TAIL = '''"""
@pyne edge
"""
from pynecore.core.pine_export import Exported, export
from pynecore.lib import script
from typing import Protocol, Any


__all__ = ['scaled']


class _ProtocolScaled(Protocol):
    def __call__(self, factor: int = 2, length: int = ...) -> Any: ...


scaled: _ProtocolScaled = Exported()


@script.library('probe')
def main():
    @export
    def scaled(factor: int = 2, length: int = 1):
        return length * factor
'''


def __test_an_export_shim_holds_no_value__():
    """A parameter required after a defaulted one stands in the shim as ``= ...``"""
    table = infer_module(ast.parse(SHIM_WITH_A_REQUIRED_TAIL), 'test')
    assert not [diag.code for diag in table.diags]


def __test_a_loop_over_a_map_types_the_key_and_the_value__(tmp_path: Path):
    """``for [key, value] in m`` compiles to ``for key, value in m.items()``"""
    types, codes = _main_types(
        tmp_path,
        'm: dict[str, float] = lib.map.new()\n'
        'for key, value in m.items():\n    w = key + "x"\n    lib.math.abs(value)')
    assert not codes
    assert (types['key'], types['value']) == (STR, FLOAT)


OWN_EXPORT_BY_NAME = '''"""
@pyne edge
"""
from pynecore.core.pine_export import Exported, export
from pynecore.core.pine_method import method, method_call
from pynecore.core.pine_udt import udt
from pynecore.lib import script
from typing import Protocol, Any


__all__ = ['Tile', 'area', 'double']


@udt
class Tile:
    side: float = 1.0


class _ProtocolArea(Protocol):
    def __call__(self, this: Tile) -> Any: ...


class _ProtocolDouble(Protocol):
    def __call__(self, tile: Tile) -> Any: ...


area: _ProtocolArea = Exported()
double: _ProtocolDouble = Exported()


@script.library('probe')
def main():
    @export
    @method
    def area(this: Tile):
        return this.side * this.side
DECORATOR
    def double(tile: Tile):
        return method_call('area', tile) * 2.0
'''


def __test_a_library_reaches_its_own_export_by_name__():
    """The module declaring the receiver's class is asked first, and holds the proxy"""
    table = infer_module(
        ast.parse(OWN_EXPORT_BY_NAME.replace('DECORATOR', '\n    @export')), 'test')
    assert not [diag.code for diag in table.diags]
    assert table.funcs['main·double'].ret == FLOAT


def __test_a_method_no_proxy_holds_is_not_reached_by_name__():
    """Only an ``Exported`` proxy answers a name"""
    source = OWN_EXPORT_BY_NAME.replace('DECORATOR', '\n    @export') \
        .replace('area: _ProtocolArea = Exported()\n', '') \
        .replace("    @export\n    @method\n", "    @method\n")
    table = infer_module(ast.parse(source), 'test')
    assert [diag.code for diag in table.diags][:1] == ['E0054']


# --- names a library module reaches through its imports and its own proxies --


def _analysed_codes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, modules: dict[str, str],
                    script: str) -> tuple[dict[str, str], list[str]]:
    """Analyse a script next to the modules it imports: its module bindings and diagnostic codes."""
    monkeypatch.syspath_prepend(str(tmp_path))
    for name, source in modules.items():
        (tmp_path / f'{name}.py').write_text(source, encoding='utf-8')
    path = tmp_path / 'importer.py'
    path.write_text(script, encoding='utf-8')
    analysed = analyse_module(str(path))
    assert analysed is not None, "the analysis did not recognize the script"
    table = analysed[1]
    return ({name: binding.ty for name, binding in table.bindings[''].items()},
            [diag.code for diag in table.diags])


CLASS_LIB = '''"""
@pyne edge
"""
from pynecore.core.pine_export import export
from pynecore.core.pine_udt import udt
from pynecore.lib import script


__all__ = ['Fluent', 'one']


@udt
class Fluent:
    x: int = 0


@script.library('fluent')
def main():
    @export
    def one() -> int:
        return 1
'''


def __test_an_import_named_in_a_module_level_class_body_is_the_import__(tmp_path, monkeypatch):
    """The class body stands in the module's own scope, where the import is no rebinding"""
    _, codes = _analysed_codes(tmp_path, monkeypatch, {'t113_class_body_lib': CLASS_LIB}, '''"""
@pyne
"""
from pynecore.core.pine_udt import udt
from pynecore.lib import na, script
import t113_class_body_lib as fl


@udt
class Holder:
    bar: fl.Fluent = na(fl.Fluent)


@script.indicator('holder')
def main():
    h: Holder = Holder.new()
''')
    assert not codes


def __test_an_import_a_function_rebinds_is_not_the_import_in_its_class_body__(
        tmp_path, monkeypatch):
    """A parameter of the enclosing function holds whatever was passed, not the module"""
    _, codes = _analysed_codes(tmp_path, monkeypatch, {'t113_class_rebound_lib': CLASS_LIB}, '''"""
@pyne
"""
from pynecore.core.pine_udt import udt
from pynecore.lib import na, script
import t113_class_rebound_lib as fl


@script.indicator('holder')
def main():
    def build(fl):
        @udt
        class Holder:
            bar: fl.Fluent = na(fl.Fluent)
        return Holder.new()
    build(1)
''')
    # The int passed for ``fl`` has no field ``Fluent``
    assert codes == ['E0019']


SHADOWING_LIB = '''"""
@pyne edge
"""
from pynecore.core.pine_export import Exported, export
from pynecore.lib import bar_index, script
from typing import Protocol, Any


__all__ = ['leftBarIndex']


class _ProtocolLeftbarindex(Protocol):
    def __call__(self, ) -> Any: ...


leftBarIndex: _ProtocolLeftbarindex = Exported()


@script.library('shadowing')
def main():
    @export
    def leftBarIndex() -> int:
        return bar_index
'''

SHADOWED_READ = '''"""
@pyne
"""
from pynecore.core.pine_import import shadowed_namespace
from pynecore.lib import chart, script
import t113_shadowing_lib as chart__ren____lib


chart__ren__ = shadowed_namespace(chart__ren____lib, chart)
first = chart__ren__.MEMBER


@script.indicator('shadowed')
def main():
    a: int = chart__ren__.leftBarIndex()
'''


def __test_a_value_the_shadowed_namespace_keeps_is_read_through_the_alias__(
        tmp_path, monkeypatch):
    """A member the library does not publish is the builtin namespace's"""
    types, codes = _analysed_codes(
        tmp_path, monkeypatch, {'t113_shadowing_lib': SHADOWING_LIB},
        SHADOWED_READ.replace('MEMBER', 'left_visible_bar_time'))
    assert not codes
    assert types['first'] == INT


def __test_a_member_neither_half_of_a_shadowed_namespace_has_is_still_reported__(
        tmp_path, monkeypatch):
    """The builtin half answers only the names the builtin namespace has"""
    _, codes = _analysed_codes(
        tmp_path, monkeypatch, {'t113_shadowing_lib': SHADOWING_LIB},
        SHADOWED_READ.replace('MEMBER', 'no_such_member'))
    assert codes == ['E0019']


RENAMED_KEYWORD_LIB = '''"""
@pyne edge
"""
from pynecore.core.pine_export import Exported, export
from pynecore.core.pine_method import method
from pynecore.lib import script
from typing import Protocol, Any


__all__ = ['shout']


class _ProtocolShout(Protocol):
    def __call__(self, this: str, position__ren__: str = 'top') -> Any: ...


shout: _ProtocolShout = Exported()


@script.library('renamed')
def main():
    @export
    @method
    def shout(this: str, position__ren__: str = 'top') -> str:
        return this + position__ren__
'''

RENAMED_KEYWORD_CALL = '''"""
@pyne
"""
from pynecore.core.pine_method import method_call
from pynecore.lib import script
from t113_renamed_keyword_lib import shout


@script.indicator('renamed')
def main():
    a = CALL
'''


def __test_a_keyword_reaches_its_renamed_parameter_through_a_call_by_name__(
        tmp_path, monkeypatch):
    """``method_call`` hands a keyword no parameter takes to its ``__ren__`` image"""
    _, codes = _analysed_codes(
        tmp_path, monkeypatch, {'t113_renamed_keyword_lib': RENAMED_KEYWORD_LIB},
        RENAMED_KEYWORD_CALL.replace('CALL', "method_call('shout', 'x', position='bottom')"))
    assert not codes


def __test_a_direct_call_does_not_rename_its_keyword__(tmp_path, monkeypatch):
    """A plain proxy called directly passes the keyword on as written"""
    _, codes = _analysed_codes(
        tmp_path, monkeypatch, {'t113_renamed_keyword_lib': RENAMED_KEYWORD_LIB},
        RENAMED_KEYWORD_CALL.replace('CALL', "shout('x', position='bottom')"))
    assert codes == [CALL_UNKNOWN_PARAMETER.code]


def __test_a_bare_map_constructor_takes_the_other_ternary_arm_s_map__(tmp_path: Path):
    """``map.new()`` lost its template, which the map in the other arm spells"""
    types, codes = _main_types(
        tmp_path,
        'm: dict[str, int] = lib.map.new()\n'
        'n = m if lib.bar_index > 0 else lib.map.new()\n'
        'k = lib.map.new() if lib.bar_index > 0 else m')
    assert not codes
    assert types['n'] == types['k'] == types['m']


def __test_a_bare_map_constructor_next_to_another_shape_is_still_reported__(tmp_path: Path):
    """Only a map gives the constructor its template"""
    _, codes = _main_types(
        tmp_path, 'n = lib.array.new_int(1, 0) if lib.bar_index > 0 else lib.map.new()')
    assert codes[:1] == ['E0023']


@pytest.mark.parametrize('namespace,cls', [('box', 'Box'), ('line', 'Line'), ('label', 'Label')])
def __test_a_copied_drawing_is_of_its_class__(tmp_path: Path, namespace: str, cls: str):
    """``box.copy``, ``line.copy`` and ``label.copy`` return the class they take"""
    types, codes = _main_types(
        tmp_path, f'a = lib.{namespace}.new(0, 1.0, 1, 0.0)\nb = lib.{namespace}.copy(a)')
    assert not codes
    assert types['b'] == object_ty(builtin_class_id(cls))


OWN_CLASS_NAMED_MAP = '''
from pynecore import lib
from pynecore.core.pine_udt import udt
from pynecore.types import Box


@udt
class Map:
    strong: Box = lib.na(Box)


def main():
    def retire(m: Map):
        return lib.box.set_bgcolor(m.strong, lib.color.red)
    retire(Map.new())
'''


def __test_a_module_s_own_class_named_like_a_container_is_that_class__():
    """A script's own ``Map`` is the class it wrote, not the container"""
    table = infer_module(ast.parse(OWN_CLASS_NAMED_MAP), 'test')
    assert not [diag.code for diag in table.diags]


@pytest.mark.parametrize('classes', [{}, {'Map': builtin_class_id('Map')}])
def __test_a_container_name_no_own_class_takes_is_the_container__(classes: dict[str, str]):
    """The lib's own container classes still spell the container"""
    assert annotation_type(ast.parse('Map', mode='eval').body, classes) == OBJECT


OWN_PROXY_REFERENCE = '''"""
@pyne edge
"""
from pynecore.core.pine_export import Exported, export
from pynecore.core.pine_udt import udt
from pynecore.lib import script
from typing import Protocol, Any

__all__ = ['disp', 'disp__ren__']


@udt
class disp__ren__:
    hi: float = 0.0


class _ProtocolDisp(Protocol):
    def __call__(self, x: float) -> Any: ...


disp: _ProtocolDisp = PROXY


disp.new = disp__ren__


@script.library('proxy')
def main():
    @export
    def disp(x: float):
        return x * 2
'''


def __test_a_module_s_own_proxy_is_a_reference__():
    """``disp.new = disp__ren__`` sets an attribute of the export's proxy, no Pine object"""
    table = infer_module(ast.parse(OWN_PROXY_REFERENCE.replace('PROXY', 'Exported()')), 'test')
    assert not [diag.code for diag in table.diags]


def __test_an_own_export_read_inside_a_function_is_no_reference__():
    """Inside the library's body the name is the definition, whose attributes are no fields"""
    source = OWN_PROXY_REFERENCE.replace('PROXY', 'Exported()') \
        .replace('\n\ndisp.new = disp__ren__\n', '') \
        .replace('        return x * 2\n', '        return x * 2\n\n    disp.new = disp__ren__\n')
    table = infer_module(ast.parse(source), 'test')
    assert [diag.code for diag in table.diags]


OWN_METHOD_ON_AN_ARRAY = '''"""
@pyne edge
"""
from pynecore.core.overload import overload
from pynecore.core.pine_export import Exported, export
from pynecore.core.pine_method import method, method_call
from pynecore import lib
from pynecore.lib import script
from typing import Protocol, Any


class _ProtocolShiftr(Protocol):
    def __call__(self, arr: list[int]) -> Any: ...


shiftr: _ProtocolShiftr = Exported()
shiftr: _ProtocolShiftr = Exported()


@script.library('own')
def main():
DECORATOR
    @overload
    @method
    def shiftr(arr: list[int]):
        return lib.array.shift(arr)

DECORATOR
    @overload
    @method
    def shiftr(arr: list[float]):
        return lib.array.shift(arr)

    def trim(arr, n):
        while method_call('size', arr) > n:
            method_call('shiftr', arr)
        return arr

    @export
    def keep(arr: list[int], n: int):
        return trim(arr, n)
'''


def __test_a_library_reaches_its_own_export_by_name_on_an_array__():
    """The caller's own proxy answers before the libraries it imports"""
    table = infer_module(
        ast.parse(OWN_METHOD_ON_AN_ARRAY.replace('DECORATOR', '    @export')), 'test')
    assert not [diag.code for diag in table.diags]


def __test_a_method_no_proxy_holds_is_not_reached_by_name_on_an_array__():
    """Only an ``Exported`` proxy answers a name, whatever the receiver"""
    source = OWN_METHOD_ON_AN_ARRAY.replace('DECORATOR\n', '') \
        .replace('shiftr: _ProtocolShiftr = Exported()\n', '')
    table = infer_module(ast.parse(source), 'test')
    assert [diag.code for diag in table.diags]
