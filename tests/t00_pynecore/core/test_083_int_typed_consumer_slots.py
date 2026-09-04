"""
Static guard: no int-annotated parameter reaches an integer slot bare.

Pine's ``int`` is a static type only. An int-TYPED expression can carry a
fractional value (``14 / 8`` is int-typed and 1.75), so every slot in ``lib/``
that genuinely needs an integer -- a subscript, a slice bound, a ``range()``
argument, a bit operand -- must truncate what it consumes. Seven such slots
raised at runtime before this guard existed (``array.new_int``,
``str.substring``, ``str.replace``, the ``math.random`` seed, the
``strategy.*trades`` accessors and ``table.clear``/``merge_cells``).

The sanctioned forms are an inline ``int(x)`` at the point of use, or rebinding
the parameter once at the top of the body (``x = int(x)``,
``trade_num = _trade_index(trade_num)``, ``size = _na_size(size)``).

``types/`` is deliberately NOT scanned: the normalization boundary is the
``lib/`` façade, which truncates every coordinate before it reaches
``types/matrix.py`` or ``types/table.py``.
"""
import ast
from pathlib import Path

import pynecore

LIB_ROOT = Path(pynecore.__file__).parent / "lib"

INT_ANNOTATIONS = {"int", "PyneInt"}
NORMALIZERS = {"int", "_trade_index", "_na_size"}

# Verified exceptions: internal bookkeeping that never carries a Pine value.
# ``closed_before`` is a list length captured by the caller inside the same
# module, not an argument any script can reach.
ALLOWED = {
    ("strategy/__init__.py", "_settle_close_pass_trades", "closed_before"),
}


def _is_int_annotation(node: ast.expr | None) -> bool:
    if node is None:
        return False
    if isinstance(node, ast.Name):
        return node.id in INT_ANNOTATIONS
    if isinstance(node, ast.Attribute):
        return node.attr in INT_ANNOTATIONS
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value in INT_ANNOTATIONS
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        # ``int | NA``, ``int | None``
        return _is_int_annotation(node.left) or _is_int_annotation(node.right)
    if isinstance(node, ast.Subscript):
        # ``NA[int]``, ``Series[int]``, ``Optional[int]``
        return _is_int_annotation(node.slice)
    return False


def _int_parameters(func: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    args = func.args
    every = list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)
    if args.vararg:
        every.append(args.vararg)
    if args.kwarg:
        every.append(args.kwarg)
    names = {arg.arg for arg in every if _is_int_annotation(arg.annotation)}

    # A parameter rebound to a normalized value is no longer a bare slot
    for stmt in ast.walk(func):
        if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1):
            continue
        target = stmt.targets[0]
        if not (isinstance(target, ast.Name) and target.id in names):
            continue
        for call in ast.walk(stmt.value):
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) \
                    and call.func.id in NORMALIZERS:
                names.discard(target.id)
                break
    return names


def _bare_uses(func: ast.FunctionDef | ast.AsyncFunctionDef,
               params: set[str]) -> list[tuple[int, str, str]]:
    """Collect ``(lineno, parameter, slot kind)`` for every bare consumption."""
    found: list[tuple[int, str, str]] = []

    def named(node: ast.expr | None) -> str | None:
        return node.id if isinstance(node, ast.Name) and node.id in params else None

    for node in ast.walk(func):
        if isinstance(node, ast.Subscript):
            name = named(node.slice)
            if name:
                found.append((node.lineno, name, "index"))
            elif isinstance(node.slice, ast.Slice):
                for part in (node.slice.lower, node.slice.upper, node.slice.step):
                    name = named(part)
                    if name:
                        found.append((node.lineno, name, "slice"))
        elif isinstance(node, ast.Call):
            callee = node.func
            fname = callee.id if isinstance(callee, ast.Name) else \
                (callee.attr if isinstance(callee, ast.Attribute) else "")
            if fname == "range":
                for arg in node.args:
                    name = named(arg)
                    if name:
                        found.append((node.lineno, name, "range()"))
        elif isinstance(node, ast.BinOp) and isinstance(
                node.op, (ast.LShift, ast.RShift, ast.BitAnd, ast.BitOr, ast.BitXor)):
            for side in (node.left, node.right):
                name = named(side)
                if name:
                    found.append((node.lineno, name, "bit operand"))
    return found


def __test_no_bare_int_typed_consumer_slot__():
    """Every integer-consuming slot under lib/ truncates what it consumes"""
    offenders = []
    for path in sorted(LIB_ROOT.rglob("*.py")):
        rel = path.relative_to(LIB_ROOT).as_posix()
        tree = ast.parse(path.read_text())
        for func in ast.walk(tree):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            params = _int_parameters(func)
            if not params:
                continue
            for lineno, name, kind in _bare_uses(func, params):
                if (rel, func.name, name) in ALLOWED:
                    continue
                offenders.append(f"lib/{rel}:{lineno} {func.name}({name}) -> {kind}")

    assert not offenders, \
        "int-typed parameters reaching an integer slot bare:\n  " + "\n  ".join(offenders)


def __test_allowlist_entries_still_exist__():
    """The allowlist does not outlive the sites it excuses"""
    for rel, func_name, param in ALLOWED:
        tree = ast.parse((LIB_ROOT / rel).read_text())
        matches = [f for f in ast.walk(tree)
                   if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and f.name == func_name and param in _int_parameters(f)]
        assert matches, f"stale allowlist entry: lib/{rel} {func_name}({param})"
