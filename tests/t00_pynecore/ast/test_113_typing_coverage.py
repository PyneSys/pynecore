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

from pynecore.core.import_hook import analyse_source
from pynecore.transformers.pine_type_infer import infer_module, lib_types
from pynecore.transformers.pine_type_rules import (
    BOOL, FLOAT, INT, UNKNOWN, builtin_class_id, get_ty, object_ty,
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
        "    if __active_security__ == 'sec-1':\n"
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
        "    is_same = 'sec-1' in __same_context__\n",
        scope='main')
    assert types['here'] == 'o'
    assert types['shared'] == 'o'
    assert types['is_chart'] == 'b'
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
    MEASURED: a Pine ``for`` does not truncate, so the counter joins the bounds.

    The ITERABLE is an ordinary known non-scalar; only the counter carries the
    algebra.
    """
    types = _types(
        'from pynecore import lib, pine_range\n'
        'R = lib.input.int(14)\n'
        'span = pine_range(0, R, 1)\n'
        'for i in pine_range(0, R, 1):\n'
        '    counter = i\n'
        'for j in pine_range(0.0, R, 1):\n'
        '    widened = j\n'
    )
    assert types['span'] == 'o'
    assert types['counter'] == 'i'
    assert types['widened'] == 'f'


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
        plot(high)
    if close - open:
        plot(low)
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
    analysed = analyse_source(str(script))
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
