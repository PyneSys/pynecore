"""
@pyne
"""
import ast
import importlib
import struct
import builtins as py_builtins
from pathlib import Path
from typing import Any

from pynecore.core import inline_support
from pynecore.lib import array as lib_array, math as lib_math
from pynecore.transformers.call_inline import (
    CallInlineTransformer,
    NotDerivable,
    INLINABLE_FUNCTIONS,
    SUPPORT_MODULE,
    SUPPORT_NAMES,
    derive_template,
    varargs_expansion_arities,
)
from pynecore.transformers.float_tolerance import FloatToleranceTransformer
from pynecore.types.na import NA, na_float, na_int


def main():
    """Dummy main so this file is a valid Pyne script."""
    pass


__test_helper_PROLOGUE = (
    "from pynecore import lib\n"
    "import pynecore.lib.array\n"
    "import pynecore.lib.math\n"
)


def __test_helper_inline(body: str, prologue: str = __test_helper_PROLOGUE) -> str:
    """Run the pass over a module source and return the emission.

    :param body: Source of the module below the import prologue.
    :param prologue: The import prologue the callee paths resolve through.
    :return: The unparsed emission.
    """
    tree = ast.parse(prologue + body)
    return ast.unparse(CallInlineTransformer().visit(tree))


def __test_helper_count(body: str, prologue: str = __test_helper_PROLOGUE) -> int:
    """How many sites the pass rewrote in a module source.

    :param body: Source of the module below the import prologue.
    :param prologue: The import prologue.
    :return: The number of inlined sites.
    """
    transformer = CallInlineTransformer()
    transformer.visit(ast.parse(prologue + body))
    return transformer.inlined


def __test_helper_bits(value: Any) -> Any:
    """A comparison key that separates values a plain ``==`` cannot.

    Floats are compared by their bit pattern, so ``-0.0`` differs from ``0.0``
    and a nan never equals itself by accident; an ``NA`` is identified by its
    exact type argument, and every other value by its own type and value.

    :param value: The value to key.
    :return: A hashable key.
    """
    if isinstance(value, NA):
        return 'NA', value.type
    if type(value) is float:
        return 'f', struct.pack('<d', value)
    return type(value).__name__, value


__test_helper_PROBES: dict[tuple[str, int, bool], Any] = {}


def __test_helper_probe(expression: str, args: tuple[Any, ...], inline: bool) -> Any:
    """Evaluate one call expression with and without the pass.

    The compiled probe is memoized: the differential grids call the same few
    expressions tens of thousands of times.

    :param expression: The call, written over ``a0``, ``a1``, ``a2``.
    :param args: The actual arguments.
    :param inline: Whether to run the pass over the module first.
    :return: The result, or the raised exception's type.
    """
    key = (expression, len(args), inline)
    probe = __test_helper_PROBES.get(key)
    if probe is None:
        params = ', '.join(f'a{i}' for i in range(len(args)))
        tree = ast.parse(__test_helper_PROLOGUE
                         + f"def probe({params}):\n    return {expression}\n")
        if inline:
            transformer = CallInlineTransformer()
            tree = transformer.visit(tree)
            assert transformer.inlined == 1, f'{expression} was not inlined'
        ast.fix_missing_locations(tree)
        namespace: dict[str, Any] = {}
        exec(compile(tree, '<probe>', 'exec'), namespace)  # noqa: the test's own source
        probe = namespace['probe']
        __test_helper_PROBES[key] = probe
    try:
        return probe(*args)
    except Exception as exc:  # noqa: the exception TYPE is the compared result
        return type(exc)


__test_helper_VALUES = (
    0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, 3.0, -3.0, 0.1, 1e-300, 1e300,
    -1e300, 1.5e-8, 123456.789, -123456.789,
    float('inf'), float('-inf'), float('nan'),
    0, 1, -1, 2, 7, -7,
    na_float, na_int, NA(None), NA(float), NA(int), NA(bool),
)

__test_helper_EXPONENTS = (
    2, 2.0, 1, 1.0, 0, 0.0, 0.5, -1, -1.0, 3, 3.0, 0.25, -0.5, 2.5,
    float('inf'), float('nan'), na_float, NA(None),
)

__test_helper_INDEXES = (
    0, 1, 2, 3, -1, -3, -4, 4, 10, 0.0, 1.0, 1.9, -1.5, 2.999,
    na_int, na_float, NA(None), float('nan'),
)


def __test_call_inline_emission__():
    """The emitted expression is the wrapper body, with one binding per
    non-trivial argument."""
    emitted = __test_helper_inline(
        "def f(v):\n    return lib.math.abs(v)\n")
    assert 'lib.math.abs(' not in emitted
    assert '__inl·na_float__ if not v == v else __inl·py_builtins__.abs(v)' in emitted
    assert f'from {SUPPORT_MODULE} import' in emitted

    # A non-trivial argument is bound once, at its first read
    emitted = __test_helper_inline(
        "def f(v):\n    return lib.math.abs(v * 2.0)\n")
    assert emitted.count('v * 2.0') == 1
    assert ':=' in emitted

    # A literal exponent decides the measured shortcuts at transform time
    emitted = __test_helper_inline(
        "def f(v):\n    return lib.math.pow(v, 2)\n")
    assert 'v * v' in emitted
    assert '** ' not in emitted

    # array.get keeps the float truncation and the raw subscript
    emitted = __test_helper_inline(
        "def f(a, i):\n    return lib.array.get(a, i)\n")
    assert 'a[__inl·py_int__(i)]' in emitted
    assert '__inl·array_na_element__(a)' in emitted

    # array.size is the float of the length
    emitted = __test_helper_inline(
        "def f(a):\n    return lib.array.size(a)\n")
    assert '__inl·py_float__(__inl·py_len__(a))' in emitted


def __test_call_inline_evaluation_order__():
    """Arguments the body would read out of order, or not at all, are bound in
    front of the expression, in source order."""
    # array.get reads the index first; a side-effecting array argument must
    # still be evaluated before it
    emitted = __test_helper_inline(
        "def f(g, h):\n    return lib.array.get(g(), h())\n")
    first = emitted.index('g()')
    second = emitted.index('h()')
    assert first < second
    assert emitted.count('g()') == 1 and emitted.count('h()') == 1

    # math.max short-circuits its na scan, so both operands are bound up front
    emitted = __test_helper_inline(
        "def f(g, h):\n    return lib.math.max(g(), h())\n")
    assert emitted.index('g()') < emitted.index('h()')
    assert emitted.count('g()') == 1 and emitted.count('h()') == 1

    # A folded-away guard must not swallow an argument's side effect
    emitted = __test_helper_inline(
        "def f(g):\n    return lib.math.pow(g(), 0)\n")
    assert 'g()' in emitted

    # The order really holds at runtime
    order: list[str] = []

    def g() -> float:
        order.append('g')
        return 1.0

    def h() -> float:
        order.append('h')
        return 2.0

    tree = CallInlineTransformer().visit(ast.parse(
        __test_helper_PROLOGUE + "def probe(g, h):\n    return lib.math.max(g(), h())\n"))
    ast.fix_missing_locations(tree)
    namespace: dict[str, Any] = {}
    exec(compile(tree, '<probe>', 'exec'), namespace)  # noqa: the test's own source
    assert namespace['probe'](g, h) == 2.0
    assert order == ['g', 'h']


def __test_call_inline_skipped_sites__():
    """Every site the pass cannot prove safe keeps its call."""
    # Module level: a temporary would become a global
    assert __test_helper_count("x = lib.math.abs(close - 1.0)\n") == 0
    # Class body: a temporary would become a class attribute
    assert __test_helper_count(
        "class C:\n    x = lib.math.abs(close - 1.0)\n") == 0
    # Comprehension and lambda: the walrus would bind in the wrong scope
    assert __test_helper_count(
        "def f(xs):\n    return [lib.math.abs(x + 1) for x in xs]\n") == 0
    assert __test_helper_count(
        "def f():\n    return lambda v: lib.math.abs(v + 1)\n") == 0
    # A generator expression is a scope of its own too
    assert __test_helper_count(
        "def f(xs):\n    return sum(lib.math.abs(x + 1) for x in xs)\n") == 0
    # Decorators and defaults are evaluated in the enclosing scope
    assert __test_helper_count(
        "def f(v=lib.math.abs(1.5 + 1)):\n    return v\n") == 0
    # Keywords, starred arguments and a wrong arity
    assert __test_helper_count(
        "def f(v):\n    return lib.math.abs(number=v)\n") == 0
    assert __test_helper_count(
        "def f(xs):\n    return lib.math.max(*xs)\n") == 0
    assert __test_helper_count(
        "def f(a, b, c, d):\n    return lib.math.max(a, b, c, d)\n") == 0
    # A name the module binds anywhere takes the whole base out
    # A same-named function elsewhere does not touch the ``lib.`` path
    assert __test_helper_count(
        "def math():\n    return 1.0\n\ndef f(v):\n    return lib.math.abs(v)\n") == 1
    assert __test_helper_count(
        "def f(v):\n    return math.abs(v)\n",
        prologue="from pynecore.lib import math\n") == 1
    assert __test_helper_count(
        "def math(v):\n    return v\n\ndef f(v):\n    return math.abs(v)\n",
        prologue="from pynecore.lib import math\n") == 0
    assert __test_helper_count(
        "def f(v):\n    return abs(v)\n",
        prologue="def abs(v):\n    return v\n") == 0
    # The function used as a VALUE, not called
    emitted = __test_helper_inline("def f(xs):\n    return list(map(lib.math.abs, xs))\n")
    assert 'lib.math.abs' in emitted


def __test_call_inline_marks_comparisons_exact__():
    """Every comparison the pass emits is marked ``pine_exact`` so the float
    tolerance rewrite leaves the copied raw na tests alone."""
    tree = CallInlineTransformer().visit(ast.parse(
        __test_helper_PROLOGUE
        + "def f(v, w):\n    return lib.math.max(v * 2, lib.math.pow(w, 2))\n"))
    compares = [node for node in ast.walk(tree) if isinstance(node, ast.Compare)]
    assert compares
    assert all(getattr(node, 'pine_exact', False) for node in compares)


def __test_call_inline_support_anchors__():
    """Every anchor is the very object the wrapper's own module resolves the
    name to -- a module global first, a builtin otherwise."""
    for module_name, names in SUPPORT_NAMES.items():
        module = importlib.import_module(module_name)
        for name, anchor in names.items():
            assert hasattr(inline_support, anchor), f'{anchor} missing'
            expected = vars(module).get(name, getattr(py_builtins, name, None))
            assert expected is not None, f'{module_name} does not resolve {name!r}'
            assert getattr(inline_support, anchor) is expected, \
                f'{module_name}.{name} is not anchored by {anchor}'


def __test_call_inline_every_entry_is_derivable__():
    """Every allow-listed function still has the body shape the pass copies.

    A body edit that breaks the shape must fail here: the site would silently
    fall back to a plain call, and the measured behaviour the pass copies
    would drift away from what it was verified against.
    """
    assert INLINABLE_FUNCTIONS
    for module_name, func_name in INLINABLE_FUNCTIONS:
        template = derive_template(module_name, func_name)
        assert template.params


def __test_call_inline_differential_math__():
    """Every one-argument math wrapper answers bit-identically inlined."""
    for module_name, func_name in INLINABLE_FUNCTIONS:
        if module_name != 'pynecore.lib.math' \
                or INLINABLE_FUNCTIONS[(module_name, func_name)] != 'derive' \
                or func_name == 'pow':
            continue
        expression = f'lib.math.{func_name}(a0)'
        for value in __test_helper_VALUES:
            plain = __test_helper_probe(expression, (value,), inline=False)
            inlined = __test_helper_probe(expression, (value,), inline=True)
            assert __test_helper_bits(plain) == __test_helper_bits(inlined), \
                f'{func_name}({value!r}): {plain!r} != {inlined!r}'


def __test_call_inline_differential_pow__():
    """``math.pow`` answers bit-identically inlined, with the exponent as a
    variable and as the literal that folds the measured shortcuts."""
    for base in __test_helper_VALUES:
        for exponent in __test_helper_EXPONENTS:
            plain = __test_helper_probe('lib.math.pow(a0, a1)', (base, exponent),
                                        inline=False)
            inlined = __test_helper_probe('lib.math.pow(a0, a1)', (base, exponent),
                                          inline=True)
            assert __test_helper_bits(plain) == __test_helper_bits(inlined), \
                f'pow({base!r}, {exponent!r}): {plain!r} != {inlined!r}'
    # The literal-exponent form is the one the guard folding rewrites
    for base in __test_helper_VALUES:
        for literal in ('2', '2.0', '1', '1.0', '0', '0.0', '-0.0', '0.5',
                        '3', '-1', '-1.0', '2.5', 'True', 'False'):
            expression = f'lib.math.pow(a0, {literal})'
            plain = __test_helper_probe(expression, (base,), inline=False)
            inlined = __test_helper_probe(expression, (base,), inline=True)
            assert __test_helper_bits(plain) == __test_helper_bits(inlined), \
                f'pow({base!r}, {literal}): {plain!r} != {inlined!r}'


def __test_call_inline_differential_minmax__():
    """``math.max`` / ``math.min`` answer bit-identically at every expanded
    arity, including which ``na`` object they hand back."""
    values = __test_helper_VALUES
    for func_name in ('max', 'min'):
        for left in values:
            for right in values:
                expression = f'lib.math.{func_name}(a0, a1)'
                plain = __test_helper_probe(expression, (left, right), inline=False)
                inlined = __test_helper_probe(expression, (left, right), inline=True)
                assert __test_helper_bits(plain) == __test_helper_bits(inlined), \
                    f'{func_name}({left!r}, {right!r}): {plain!r} != {inlined!r}'
        # A literal operand is what the constant fold rewrites, so it gets
        # its own pass over the grid
        for literal in ('0', '1', '1.0', '-0.0', '0.0', '2', '-1', '0.5', 'True'):
            expression = f'lib.math.{func_name}(a0, {literal})'
            for left in values:
                plain = __test_helper_probe(expression, (left,), inline=False)
                inlined = __test_helper_probe(expression, (left,), inline=True)
                assert __test_helper_bits(plain) == __test_helper_bits(inlined), \
                    f'{func_name}({left!r}, {literal}): {plain!r} != {inlined!r}'
        if 3 in varargs_expansion_arities():
            for third in values:
                expression = f'lib.math.{func_name}(a0, a1, a2)'
                args = (1.0, -2.5, third)
                plain = __test_helper_probe(expression, args, inline=False)
                inlined = __test_helper_probe(expression, args, inline=True)
                assert __test_helper_bits(plain) == __test_helper_bits(inlined), \
                    f'{func_name}{args!r}: {plain!r} != {inlined!r}'


def __test_call_inline_differential_array__():
    """``array.get`` and ``array.size`` answer bit-identically inlined, down to
    the raised exception type and which ``na`` an empty or na-headed array
    yields for an na index."""
    arrays: tuple[list[Any], ...] = (
        [],
        [10.0, 20.0, 30.0, 40.0],
        [1, 2, 3],
        [na_float, 2.0],
        [NA(None), 5.0],
    )
    for source in arrays:
        for index in __test_helper_INDEXES:
            plain = __test_helper_probe('lib.array.get(a0, a1)', (list(source), index),
                                        inline=False)
            inlined = __test_helper_probe('lib.array.get(a0, a1)', (list(source), index),
                                          inline=True)
            assert __test_helper_bits(plain) == __test_helper_bits(inlined), \
                f'get({source!r}, {index!r}): {plain!r} != {inlined!r}'
        # A literal index is folded at transform time: the truncation, the
        # bounds behaviour and the na guard must all survive it
        for literal in ('0', '1', '2', '3', '4', '-1', '-3', '-4', '10',
                        '0.0', '1.0', '1.9', '-1.5', '2.999', '-0.0', 'True'):
            expression = f'lib.array.get(a0, {literal})'
            plain = __test_helper_probe(expression, (list(source),), inline=False)
            inlined = __test_helper_probe(expression, (list(source),), inline=True)
            assert __test_helper_bits(plain) == __test_helper_bits(inlined), \
                f'get({source!r}, {literal}): {plain!r} != {inlined!r}'
        plain = __test_helper_probe('lib.array.size(a0)', (list(source),), inline=False)
        inlined = __test_helper_probe('lib.array.size(a0)', (list(source),), inline=True)
        assert __test_helper_bits(plain) == __test_helper_bits(inlined)


def __test_call_inline_rejected_functions__():
    """The functions left off the allow-list are left off on purpose, and the
    reason still holds: their bodies are not expressions."""
    for func_name in ('avg', 'round', 'round_to_mintick'):
        assert ('pynecore.lib.math', func_name) not in INLINABLE_FUNCTIONS
        assert hasattr(lib_math, func_name)
    assert ('pynecore.lib.array', 'set') not in INLINABLE_FUNCTIONS
    assert hasattr(lib_array, 'set')


def __test_helper_pipeline(source: str, inline: bool) -> dict[str, Any]:
    """Run the real lowering order -- call inlining, then the float tolerance
    rewrite -- over a module source and execute it.

    :param source: Source below the import prologue.
    :param inline: Whether the inlining pass runs.
    :return: The executed module namespace, plus the emission under ``_src``.
    """
    tree = ast.parse(__test_helper_PROLOGUE + source)
    if inline:
        tree = CallInlineTransformer().visit(tree)
    tree = FloatToleranceTransformer().visit(tree)
    ast.fix_missing_locations(tree)
    namespace: dict[str, Any] = {'_src': ast.unparse(tree)}
    exec(compile(tree, '<pipeline>', 'exec'), namespace)  # noqa: the test's own source
    return namespace


def __test_call_inline_float_tolerance_only_touches_body__():
    """The pass marks its OWN comparisons exact, never the user's.

    A comparison inside an argument is Pine code and must still get the
    tolerant rewrite; the copied body's raw ``x == x`` na test must not.
    """
    source = "def f(a, b):\n    return lib.math.abs(1.0 if a == b else 2.0)\n"
    inlined = __test_helper_pipeline(source, inline=True)
    plain = __test_helper_pipeline(source, inline=False)
    emission = inlined['_src']
    # The user's comparison became the tolerant form
    assert '1e-10' in emission
    # The body's na test stayed raw: the temporary is compared to itself with
    # a plain ``==``, not through the tolerant form
    assert ') == __inl1·__ else __inl·py_builtins__.abs(__inl1·__)' in emission
    # ... and the pass added no tolerance of its own and removed none either
    assert emission.count('1e-10') == plain['_src'].count('1e-10')

    for args in ((0.1 + 0.2, 0.3), (1.0, 2.0), (0.3, 0.1 + 0.2)):
        assert __test_helper_bits(inlined['f'](*args)) \
            == __test_helper_bits(plain['f'](*args)), args
    # The concrete regression: a tolerant compare that the marker used to eat
    assert inlined['f'](0.1 + 0.2, 0.3) == 1.0


def __test_call_inline_float_tolerance_nested__():
    """A nested inline arrives as an argument subtree: its own body keeps the
    exact marker, the user's comparison inside it still gets the rewrite."""
    source = ("def f(a, b, c, x, y):\n"
              "    return lib.math.abs(lib.math.max(a if x == y else b, c))\n")
    inlined = __test_helper_pipeline(source, inline=True)
    plain = __test_helper_pipeline(source, inline=False)
    assert '1e-10' in inlined['_src']
    for args in ((1.0, -2.0, 3.0, 0.1 + 0.2, 0.3),
                 (1.0, -2.0, 3.0, 1.0, 2.0),
                 (na_float, -2.0, 3.0, 1.0, 1.0)):
        assert __test_helper_bits(inlined['f'](*args)) \
            == __test_helper_bits(plain['f'](*args)), args
    assert inlined['f'](1.0, -2.0, 3.0, 0.1 + 0.2, 0.3) == 3.0


def __test_call_inline_missing_source__():
    """A wrapper module without readable source makes the site keep its call
    instead of failing the whole script import."""
    module = importlib.import_module('pynecore.lib.math')
    original = module.__file__
    try:
        module.__file__ = None  # type: ignore[assignment]
        __test_helper_clear_caches()
        try:
            derive_template('pynecore.lib.math', 'abs')
            raise AssertionError('expected NotDerivable')
        except NotDerivable:
            pass
        assert __test_helper_count("def f(v):\n    return lib.math.abs(v)\n") == 0

        module.__file__ = '/nonexistent/pynecore/lib/math.py'
        __test_helper_clear_caches()
        assert __test_helper_count("def f(v):\n    return lib.math.abs(v)\n") == 0
    finally:
        module.__file__ = original
        __test_helper_clear_caches()
    # Back to normal once the source is readable again
    assert __test_helper_count("def f(v):\n    return lib.math.abs(v)\n") == 1


def __test_call_inline_argument_order_with_names__():
    """A plain name preceding a side-effecting argument is still read first:
    the name's own error, not the later argument's."""
    emitted = __test_helper_inline(
        "def f(boom):\n    return lib.math.max(missing, boom())\n")
    assert emitted.index('missing is missing') < emitted.index('boom()')

    tree = CallInlineTransformer().visit(ast.parse(
        __test_helper_PROLOGUE
        + "def probe(boom):\n    return lib.math.max(missing, boom())\n"))
    ast.fix_missing_locations(tree)
    namespace: dict[str, Any] = {}
    exec(compile(tree, '<probe>', 'exec'), namespace)  # noqa: the test's own source

    def boom() -> float:
        raise ValueError('must not be reached')

    try:
        namespace['probe'](boom)
        raise AssertionError('expected NameError')
    except NameError:
        pass


def __test_call_inline_branch_dependent_import__():
    """An import that is not a direct statement of the module body binds its
    name only on some paths, so the pass must not trust it."""
    conditional = ("import os\n"
                   "if os.name:\n"
                   "    from pynecore.lib import math\n"
                   "else:\n"
                   "    import math\n")
    assert __test_helper_count("def f(v):\n    return math.abs(v)\n",
                               prologue=conditional) == 0
    tried = ("try:\n"
             "    from pynecore.lib import math\n"
             "except ImportError:\n"
             "    math = None\n")
    assert __test_helper_count("def f(v):\n    return math.abs(v)\n",
                               prologue=tried) == 0
    # A straight-line re-import is still last-write-wins
    straight = ("from pynecore.lib import ta as m\n"
                "from pynecore.lib import math as m\n")
    assert __test_helper_count("def f(v):\n    return m.abs(v)\n",
                               prologue=straight) == 1


def __test_call_inline_literal_folding__():
    """A literal argument decides the guards it controls at transform time."""
    # isinstance(2, NA) and 2 != 2 are gone; the shortcut is all that is left
    emitted = __test_helper_inline("def f(v):\n    return lib.math.pow(v, 2)\n")
    assert '__inl·py_isinstance__(2' not in emitted
    assert '2 != 2' not in emitted
    assert 'v * v' in emitted

    # not 1.0 == 1.0 is gone from the na scan
    emitted = __test_helper_inline("def f(v):\n    return lib.math.max(v, 1.0)\n")
    assert '1.0 == 1.0' not in emitted
    assert 'not v == v' in emitted

    # int(0) is evaluated, and the na guard with it
    emitted = __test_helper_inline("def f(a):\n    return lib.array.get(a, 0)\n")
    assert emitted.strip().endswith('return a[0]')

    # A bool is NOT a numeric literal, so nothing folds on it
    emitted = __test_helper_inline("def f(v):\n    return lib.math.pow(v, True)\n")
    assert '__inl·py_isinstance__(True' in emitted


def __test_call_inline_slot_arguments__():
    """A slot read is free of side effects: it needs no forcing, and no
    temporary when the body reads it once."""
    emitted = __test_helper_inline(
        "def f(i):\n    return lib.array.size(__state__[11])\n")
    assert ':=' not in emitted
    assert '__inl·py_float__(__inl·py_len__(__state__[11]))' in emitted

    emitted = __test_helper_inline(
        "def f(i):\n    return lib.array.get(__state·main__[7], i)\n")
    assert ' is ' not in emitted
    assert '__inl·array_na_element__(__state·main__[7])' in emitted
    assert '__state·main__[7][__inl·py_int__(i)]' in emitted


def __test_call_inline_pipeline_digest_covers_sources__():
    """Editing an inlined wrapper body, or the anchor module, must invalidate
    cached script bytecode."""
    from pynecore.core import import_hook
    source = Path(import_hook.__file__).read_text(encoding='utf-8')
    hashed = source[source.index('files = [Path(__file__)'):
                    source.index('digest = hashlib.sha256()')]
    assert '"math.py"' in hashed
    assert '"array.py"' in hashed
    assert '"inline_support.py"' in hashed


def __test_helper_clear_caches() -> None:
    """Drop the pass's process-global memos so a source change is seen."""
    call_inline_module = importlib.import_module('pynecore.transformers.call_inline')
    getattr(call_inline_module, '_module_asts').clear()
