"""
Transformed code runs at the source lines of the code it serves.

The transform emits code the user never wrote: slot plumbing, inlined builtin
bodies, ``request.security`` protocol blocks, parameter prologues. Each such
node needs a location, and the location decides the line events debuggers
step on, the line a traceback names and the line a profiler charges. A wrong
one never breaks a result, so nothing else notices it:

- a body copied from ``lib/math.py`` kept that file's line numbers, landing
  on whatever the script has on those lines, or past its end;
- a statement rebuilt without ``copy_location`` fell back to the ``def`` line,
  between two body lines;
- code prepended above a docstring turned it into a statement evaluated on
  every call and dropped ``__doc__``.

The invariants below run the REAL pipeline over every ``@pyne`` module of the
test suite and the library, so a pass nobody thought about is caught too.
"""
import ast
import types
from pathlib import Path

import pytest

import pynecore
from pynecore.core.import_hook import PyneLoader, _PYNE_HEAD_RE

_TESTS = Path(__file__).parents[2]
_LIB = Path(pynecore.__file__).parent / 'lib'


def _corpus() -> list[Path]:
    """Every ``@pyne`` module of the test suite and of ``pynecore.lib``.

    The ``*_ast_modified.py`` files are expected transform OUTPUTS: they
    contain the reserved names the transform generates and are rejected as
    input by design.
    """
    return sorted(
        path for root in (_TESTS, _LIB) for path in root.rglob('*.py')
        if not path.name.endswith('_ast_modified.py')
        and _PYNE_HEAD_RE.search(path.read_bytes()[:4096]))


def _first_line(node: ast.stmt) -> int:
    """The first source line of a statement, its decorators included."""
    return min([d.lineno for d in getattr(node, 'decorator_list', [])] + [node.lineno])


def _code_objects(code: types.CodeType):
    """``code`` and every code object nested in it."""
    stack = [code]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(const for const in current.co_consts if isinstance(const, types.CodeType))


def _lines(code: types.CodeType) -> list[int]:
    """The line of each bytecode range in order, the module's line-0 RESUME aside."""
    return [line for start, _, line in code.co_lines()
            if line is not None and not (start == 0 and line == 0)]


class _Transformed:
    """One corpus module: its source, the final tree and the compiled code."""

    def __init__(self, path: Path, source: str, tree: ast.Module, code: types.CodeType):
        self.path = path
        self.source = source
        self.tree = tree
        self.code = code

    def functions(self) -> dict[tuple[str, int], ast.FunctionDef | ast.AsyncFunctionDef]:
        """The source's function definitions keyed the way their code objects
        name them: ``(co_name, co_firstlineno)``."""
        return {(node.name, _first_line(node)): node
                for node in ast.walk(ast.parse(self.source))
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}

    def source_functions(self) -> dict[types.CodeType, ast.FunctionDef | ast.AsyncFunctionDef]:
        """The source function each code object runs the code of.

        A function the transform generated from one of the source (a
        ``request.security`` slice of ``main``, a per-call-site instance) has a name
        of its own but starts at the source function's ``def`` line.
        """
        functions = self.functions()
        by_def_line = {node.lineno: node for node in functions.values()}
        found = {}
        for code in _code_objects(self.code):
            function = (functions.get((code.co_name, code.co_firstlineno))
                        or by_def_line.get(code.co_firstlineno))
            if function is not None:
                found[code] = function
        return found


def _transform(path: Path, source: str, monkeypatch: pytest.MonkeyPatch | None = None
               ) -> tuple[ast.Module, types.CodeType]:
    """Run the real pipeline; return the tree handed to the compiler and the code.

    ``fix_locations`` is the last pass of the transform, so intercepting it
    captures the finished tree without the loader having to give one up.
    """
    import pynecore.transformers.locations as locations

    captured: list[ast.Module] = []
    original = locations.fix_locations

    def capture(tree, *args, **kwargs):
        result = original(tree, *args, **kwargs)
        if isinstance(tree, ast.Module):
            captured.append(tree)
        return result

    patcher = monkeypatch or pytest.MonkeyPatch()
    patcher.setattr(locations, 'fix_locations', capture)
    try:
        code = PyneLoader(path.stem, str(path)).source_to_code(source, str(path))
    finally:
        if monkeypatch is None:
            patcher.undo()
    assert captured, "the pipeline never reached fix_locations"
    return captured[-1], code


@pytest.fixture(scope='module')
def corpus() -> list[_Transformed]:
    modules: list[_Transformed] = []
    for path in _corpus():
        source = path.read_text()
        try:
            tree, code = _transform(path, source)
        except SyntaxError:
            # A module the transform rejects on purpose has no code to check
            continue
        modules.append(_Transformed(path, source, tree, code))
    assert len(modules) > 400, "the corpus lost its modules"
    return modules


def _report(violations: list[str]) -> str:
    return f"{len(violations)} violation(s):\n" + "\n".join(violations[:40])


def __test_every_line_is_in_the_file__(corpus):
    """No code runs at a line the source file does not have."""
    violations = []
    for module in corpus:
        last = len(module.source.splitlines())
        for code in _code_objects(module.code):
            bad = [line for line in _lines(code) if not 1 <= line <= last]
            if bad:
                violations.append(f"{module.path}: {code.co_name} at line {bad[0]} "
                                  f"(the file has {last})")
    assert not violations, _report(violations)


def __test_function_code_stays_in_its_function__(corpus):
    """A function's code runs at lines of that function's own source span."""
    violations = []
    for module in corpus:
        source_functions = module.source_functions()
        for code, function in source_functions.items():
            first, last = _first_line(function), function.end_lineno
            # Defining a nested function runs at that function's header: a
            # ``request.security`` slice defines copies of module-level helpers
            defined = {line for nested in code.co_consts if isinstance(nested, types.CodeType)
                       and nested in source_functions
                       for line in range(_first_line(source_functions[nested]),
                                         source_functions[nested].lineno + 1)}
            bad = [line for line in _lines(code)
                   if not first <= line <= last and line not in defined]
            if bad:
                violations.append(f"{module.path}: {code.co_name} ({first}-{last}) "
                                  f"runs at line {bad[0]}")
    assert not violations, _report(violations)


def __test_prologue_never_returns_after_the_body__(corpus):
    """The header lines (decorators, ``def``, signature) only run before the body.

    Function-entry code is anchored there; a statement that lost its own
    location falls back to it too, and then shows up between body lines.
    """
    violations = []
    for module in corpus:
        for code, function in module.source_functions().items():
            first, header_end = _first_line(function), _first_line(function.body[0]) - 1
            in_body = False
            for line in _lines(code):
                if not first <= line <= header_end:
                    in_body = True
                elif in_body:
                    violations.append(f"{module.path}: {code.co_name} runs header line "
                                      f"{line} after its body started")
                    break
    assert not violations, _report(violations)


def __test_docstrings_survive__(corpus):
    """Every docstring of the source is still a docstring after the transform."""
    violations = []
    for module in corpus:
        if ast.get_docstring(ast.parse(module.source), clean=False) \
                != ast.get_docstring(module.tree, clean=False):
            violations.append(f"{module.path}: module docstring")
        transformed = {(node.name, _first_line(node)): node
                       for node in ast.walk(module.tree)
                       if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        for key, function in module.functions().items():
            expected = ast.get_docstring(function, clean=False)
            if expected is None or key not in transformed:
                continue
            if ast.get_docstring(transformed[key], clean=False) != expected:
                violations.append(f"{module.path}: {key[0]} at line {key[1]}")
    assert not violations, _report(violations)


SCRIPT = '''"""
@pyne
"""
from pynecore import Series, Persistent, PersistentSeries
from pynecore.lib import array, close, high, input, low, math, plot, script


@script.indicator("Locations")
def main(src: Series[float] = input.source(close, "Source")):
    """Straight-line body: one statement per line, no branch, no loop."""
    values = array.from_items(1.0, 2.0, 3.0)
    first = array.get(values, 0)
    distance = math.abs(src - first)
    smooth: Series[float] = math.max(distance, 0.0)
    smooth = smooth + 1.0
    smooth += 1.0
    held: PersistentSeries[float] = 0.0
    held = held + smooth[1]
    total: Persistent[float] = 0.0
    total += math.abs(high - low)
    def scaled(value: float) -> float:
        return value * 2.0
    hi, lo = high, low
    plot(scaled(smooth) + held + total + hi - lo + math.sign(src - first))
'''


def __test_straight_line_body_runs_in_source_order__(tmp_path, monkeypatch):
    """With no control flow, the lines a body runs at only ever grow.

    Every construct here makes the transform emit code of its own: an input
    source, inlined ``array.get`` / ``math.abs`` / ``math.sign`` bodies, Series
    and PersistentSeries writes, a Persistent accumulator, a nested function
    with a layout attach, a tuple unpack. A body copied from another file or
    a statement anchored at the ``def`` line breaks the order.
    """
    path = tmp_path / 'locations_script.py'
    tree, code = _transform(path, SCRIPT, monkeypatch)
    main = next(c for c in _code_objects(code) if c.co_name == 'main')
    source_main = next(node for node in ast.parse(SCRIPT).body
                       if isinstance(node, ast.FunctionDef) and node.name == 'main')
    header_end = source_main.body[0].lineno - 1
    # A Persistent with a literal initializer is a layout entry, no per-call code
    silent = {node.lineno for node in source_main.body
              if isinstance(node, ast.AnnAssign) and isinstance(node.value, ast.Constant)
              and isinstance(node.annotation, ast.Subscript)
              and isinstance(node.annotation.value, ast.Name)
              and node.annotation.value.id == 'Persistent'}
    statement_lines = {node.lineno for node in source_main.body[1:]} - silent

    body_lines = [line for line in _lines(main) if line > header_end]
    assert body_lines == sorted(body_lines), f"body runs out of order: {body_lines}"
    assert set(body_lines) == statement_lines, (
        f"lines without a statement: {sorted(set(body_lines) - statement_lines)}, "
        f"statements without code: {sorted(statement_lines - set(body_lines))}")

    transformed_main = next(node for node in tree.body
                            if isinstance(node, ast.FunctionDef) and node.name == 'main')
    assert ast.get_docstring(transformed_main) == ast.get_docstring(source_main)


FUTURE_SCRIPTS = {
    'security': '''
@script.indicator("Future security")
def main(src: Series[float] = input.source(close, "Source")):
    plot(request.security(syminfo.tickerid, "1D", src))
''',
    'security_lower_tf': '''
@script.indicator("Future lower tf")
def main():
    a, b = request.security_lower_tf(syminfo.tickerid, "1", (close, high))
    plot(close)
''',
    'strategy': '''
@script.strategy("Future strategy")
def main():
    if close > open:
        strategy.entry("L", strategy.long)
''',
}


@pytest.mark.parametrize('name', sorted(FUTURE_SCRIPTS))
def __test_generated_imports_follow_future_imports__(name, tmp_path, monkeypatch):
    """A generated module-level import goes below the ``__future__`` block.

    ``from __future__`` must lead the module: an import prepended above it
    makes a valid script fail to compile.
    """
    source = ('"""\n@pyne\n"""\nfrom __future__ import annotations\n'
              'from pynecore import Series\n'
              'from pynecore.lib import (close, high, input, open, plot, request, script,\n'
              '                          strategy, syminfo)\n\n'
              + FUTURE_SCRIPTS[name])
    path = tmp_path / f'future_{name}.py'
    tree, _ = _transform(path, source, monkeypatch)
    future = tree.body[1]
    assert isinstance(future, ast.ImportFrom) and future.module == '__future__'
