import ast
import builtins
import math
from decimal import Decimal, ROUND_HALF_EVEN, localcontext

from pynecore.core import fdlibm
from pynecore.transformers.pine_type_rules import (
    INT, FLOAT, BOOL, STR, UNKNOWN, NUMERIC, join, binop_type, unaryop_type,
    annotation_type, set_ty, LIB_TYPE_OVERRIDES,
)

__all__ = ['ConstFoldTransformer', 'quantize_embed']

# Sentinel for "not a compile-time constant"
_BAIL = object()

# lib.math functions folded with the fdlibm (StrictMath) implementations.
# TradingView's parser folds constant expressions with StrictMath; at runtime
# sin/cos/exp go through the JIT's Intel-LIBM intrinsics instead (see
# ``core.pine_math``), which disagree with fdlibm in the last ulp, so their
# fold must NOT go through the runtime implementations. asin/acos have no
# intrinsic -- their runtime is fdlibm too -- but they fold all the same.
_FOLD_FDLIBM = {
    'sin': fdlibm.sin,
    'cos': fdlibm.cos,
    'exp': fdlibm.exp,
    'asin': fdlibm.asin,
    'acos': fdlibm.acos,
}

# lib.math functions with no fold/runtime split: they are plain IEEE-754
# arithmetic on both sides (TV's JVM has no JIT intrinsic for them), so the
# fold calls the runtime implementations themselves and cannot diverge from
# them. Functions with an Intel-LIBM runtime intrinsic but no ported fdlibm
# fold port (pow, log, log10, tan) and the stateful/instrument-dependent ones
# (random, sum, round_to_mintick) are deliberately absent: their constant
# calls stay in the code and evaluate at runtime.
#
# Filled on first use: this module is imported by the import hook while it
# transforms pynecore.lib's own @pyne modules, so a top-level lib import
# would re-enter the partially initialized lib package.
_FOLD_EXACT: dict = {}
_FOLD_EXACT_NAMES = ('sqrt', 'abs', 'floor', 'ceil', 'min', 'max', 'avg', 'sign',
                     'round', 'todegrees', 'toradians')


def _fold_exact() -> dict:
    if not _FOLD_EXACT:
        from pynecore.lib import math as lib_math
        _FOLD_EXACT.update({name: getattr(lib_math, name) for name in _FOLD_EXACT_NAMES})
    return _FOLD_EXACT

# lib.math module constants, read from the values the runtime module defines
# so the fold and a residual runtime read cannot drift apart
_MATH_CONSTANTS = {
    'pi': math.pi,
    'e': math.e,
    'phi': (1 + math.sqrt(5)) / 2,
    'rphi': 1 / ((1 + math.sqrt(5)) / 2),
}

# Binary operators folded. Mod/FloorDiv/Pow are absent: Pine's runtime forms
# of those do not compile to the plain Python operators, so a fold here could
# disagree with what the emitted code computes.
_BIN_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
}


def quantize_embed(v: int | float) -> int | float:
    """
    TradingView's parse-time embedding quantization.

    When the parser embeds a folded constant (or a source literal) into the
    runtime program it caps the value at 16 decimal *places*, rounding
    half-even on the shortest decimal representation -- but only while
    ``|v| >= 1e-3``; below that the exact double is embedded. Constant
    chains keep exact doubles internally: this cap applies exactly once,
    at the runtime embedding site. Measured over 60+ anchored delta probes
    (probe_fold_* series, 2026-08).
    """
    if type(v) is int:
        return v
    if builtins.abs(v) < 1e-3:
        return v
    # A double can carry ~309 integer digits in front of the 16 capped
    # decimal places; the default 28-digit context would overflow on them
    with localcontext() as ctx:
        ctx.prec = 340
        return float(Decimal(repr(v)).quantize(Decimal('1e-16'), rounding=ROUND_HALF_EVEN))


def _is_lib_math_attr(node: ast.expr) -> str | None:
    """Return the attribute name for a ``lib.math.<name>`` chain, else None."""
    if (isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == 'math'
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == 'lib'):
        return node.attr
    return None


def _is_stateful_annotation(node: ast.expr) -> bool:
    """Whether an annotation names a state-carrying type: ``Persistent``,
    ``PersistentSeries``, ``IBPersistent`` or ``IBPersistentSeries``."""
    if isinstance(node, ast.Subscript):
        node = node.value
    name = node.attr if isinstance(node, ast.Attribute) else (
        node.id if isinstance(node, ast.Name) else '')
    return name.startswith('Persistent') or name.startswith('IBPersistent')


def _mutated_stateful_names(body: list[ast.stmt]) -> frozenset[str]:
    """
    Names of a scope that carry state across bars AND are stored more than
    once: a later mutation of a ``Persistent`` variable makes every read of
    it depend on the previous bar, so straight-line constant propagation
    (which only kills names at/after the mutating line) must never track
    them. A Persistent assigned only by its initializer stays a constant on
    every bar and folds like TradingView folds a ``var`` chain.
    """
    stateful: set[str] = set()
    stores: dict[str, int] = {}
    for stmt in body:
        for n in ast.walk(stmt):
            if (isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
                    and _is_stateful_annotation(n.annotation)):
                stateful.add(n.target.id)
            elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                stores[n.id] = stores.get(n.id, 0) + 1
    # The AnnAssign target itself is one store: only additional ones mutate
    return frozenset(name for name in stateful if stores.get(name, 0) > 1)


def _assigned_names(node: ast.AST) -> set[str]:
    """Every name a statement (sub)tree can (re)bind."""
    names: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            names.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(n.name)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            names.update(n.names)
        elif isinstance(n, ast.NamedExpr) and isinstance(n.target, ast.Name):
            names.add(n.target.id)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            names.add(n.name)
    return names


class _ExprFolder(ast.NodeTransformer):
    """
    Replace every maximal constant subtree of one expression with the
    quantized literal. Non-constant nodes are recursed into; a successful
    fold is terminal (the cap is applied once, on the maximal subtree).
    """

    def __init__(self, env: dict[str, int | float],
                 types: dict[str, str] | None = None) -> None:
        self.env = env
        # Pine types of the folded names, so the emitted literal can keep the
        # TYPE the folded expression had (see ``const_type``)
        self.types = types if types is not None else {}

    def visit(self, node: ast.AST) -> ast.AST:
        if isinstance(node, ast.expr):
            v = _try_eval(node, self.env)
            if v is not _BAIL:
                q = quantize_embed(v)  # type: ignore[arg-type]
                ty = const_type(node, self.env, self.types)
                if isinstance(node, ast.Constant) and node.value == q and type(node.value) is type(q):
                    return _stamp(node, ty)
                return _stamp(ast.copy_location(ast.Constant(value=q), node), ty)
        return super().visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        # ``x[1]`` on a constant-valued series variable is NOT the constant:
        # bar 0 reads na from the history. Keep a bare subscripted name; only
        # the index and any composite value expression are folded.
        if not isinstance(node.value, ast.Name):
            node.value = self.visit(node.value)  # type: ignore[assignment]
        node.slice = self.visit(node.slice)  # type: ignore[assignment]
        return node

    def _visit_shadowing(self, node: ast.expr, bound: set[str]) -> ast.AST:
        inner = {k: v for k, v in self.env.items() if k not in bound}
        folder = _ExprFolder(inner, self.types)
        for field, value in ast.iter_fields(node):
            if isinstance(value, ast.expr):
                setattr(node, field, folder.visit(value))
            elif isinstance(value, list):
                setattr(node, field, [folder.visit(v) if isinstance(v, ast.AST) else v
                                      for v in value])
        return node

    def visit_Lambda(self, node: ast.Lambda) -> ast.AST:
        bound = {a.arg for a in (node.args.args + node.args.posonlyargs + node.args.kwonlyargs)}
        if node.args.vararg:
            bound.add(node.args.vararg.arg)
        if node.args.kwarg:
            bound.add(node.args.kwarg.arg)
        node.body = _ExprFolder({k: v for k, v in self.env.items() if k not in bound},
                                self.types).visit(node.body)  # type: ignore[assignment]
        return node

    def _visit_comprehension(self, node: ast.expr) -> ast.AST:
        bound = _assigned_names(node)
        return self._visit_shadowing(node, bound)

    visit_ListComp = _visit_comprehension
    visit_SetComp = _visit_comprehension
    visit_DictComp = _visit_comprehension
    visit_GeneratorExp = _visit_comprehension


def _stamp(node: ast.expr, ty: str) -> ast.expr:
    """Give an emitted literal its Pine type, when there is one to give."""
    if ty != UNKNOWN:
        set_ty(node, ty)
    return node


def const_type(node: ast.expr, env: dict[str, int | float],
               types: dict[str, str]) -> str:
    """
    Pine type of a subtree that folded to a constant.

    The fold is where the int TYPE would otherwise be lost for good: Pine's
    ``14 / 8`` is int-typed with the value 1.75, and the emitted ``1.75``
    literal is indistinguishable from a float one -- the later inference pass
    would call it a float and pick the wrong overload. TradingView folds the
    same expression and keeps its type, so the literal has to carry it.

    Only names the value environment resolved are looked up here: a name
    absent from ``env`` is not constant, so nothing folds and nothing is
    asked of it. ``types`` is scoped exactly like ``env`` -- see
    :class:`ConstFoldTransformer` for why a shared one would be wrong.

    :param node: The subtree that folded
    :param env: The constant value environment
    :param types: Pine types of the names in ``env``
    :return: The type character, UNKNOWN when it cannot be established
    """
    match node:
        case ast.Constant(value=bool()):
            return BOOL
        case ast.Constant(value=int()):
            return INT
        case ast.Constant(value=float()):
            return FLOAT
        case ast.Constant(value=str()):
            return STR
        case ast.Name(id=name):
            return types.get(name, UNKNOWN) if name in env else UNKNOWN
        case ast.UnaryOp():
            return unaryop_type(node.op, const_type(node.operand, env, types))
        case ast.BinOp():
            return binop_type(node.op, const_type(node.left, env, types),
                              const_type(node.right, env, types))
        case ast.IfExp():
            return join(const_type(node.body, env, types),
                        const_type(node.orelse, env, types))
        case ast.Call():
            return _const_call_type(node, env, types)
    return UNKNOWN


def _const_call_type(node: ast.Call, env: dict[str, int | float],
                     types: dict[str, str]) -> str:
    """
    Pine type of a folded ``lib.math.*`` call.

    Only the measured overrides are consulted: the fold surface is exactly the
    math functions, and their TradingView types are the ones in that table
    (``math.floor`` is int, ``math.sqrt`` is float, ``math.max`` is int only
    when every argument is).
    """
    parts: list[str] = []
    current: ast.expr = node.func
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return UNKNOWN
    parts.append(current.id)
    dotted = '.'.join(reversed(parts))
    # ``lib.math.floor`` -> ``math.floor``
    _, _, key = dotted.partition('.')
    override = LIB_TYPE_OVERRIDES.get(key or dotted)
    if isinstance(override, dict):
        override = override.get(len(node.args))
    if not isinstance(override, str):
        return UNKNOWN
    if override == 'all_int':
        argument_types = [const_type(a, env, types) for a in node.args]
        if not argument_types or any(t not in NUMERIC for t in argument_types):
            return UNKNOWN
        return INT if all(t == INT for t in argument_types) else FLOAT
    if override.startswith('arg') and override[3:].isdigit():
        index = int(override[3:])
        return const_type(node.args[index], env, types) if index < len(node.args) else UNKNOWN
    return override


def _try_eval(node: ast.expr, env: dict[str, int | float]):
    """
    Evaluate a subtree as a parse-time constant with exact doubles.

    Returns the exact value, or ``_BAIL`` when the subtree is not constant,
    leaves the verified fold surface, or hits a domain edge (nan/inf, zero
    divisor): those stay in the code and keep their runtime behavior.
    """
    if isinstance(node, ast.Constant):
        if type(node.value) in (int, float):
            return node.value
        return _BAIL
    if isinstance(node, ast.Name):
        if isinstance(node.ctx, ast.Load) and node.id in env:
            return env[node.id]
        return _BAIL
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _try_eval(node.operand, env)
        if not isinstance(v, (int, float)):
            return _BAIL
        return -v if isinstance(node.op, ast.USub) else +v
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            return _BAIL
        left = _try_eval(node.left, env)
        right = _try_eval(node.right, env)
        if left is _BAIL or right is _BAIL:
            return _BAIL
        if isinstance(node.op, ast.Div) and right == 0:
            return _BAIL
        v = op(left, right)
        return _BAIL if isinstance(v, float) and not math.isfinite(v) else v
    if isinstance(node, ast.Attribute):
        name = _is_lib_math_attr(node)
        if name is not None and name in _MATH_CONSTANTS:
            return _MATH_CONSTANTS[name]
        return _BAIL
    if isinstance(node, ast.Call):
        if node.keywords:
            return _BAIL
        name = _is_lib_math_attr(node.func)
        if name is None:
            return _BAIL
        args = []
        for arg in node.args:
            if isinstance(arg, ast.Starred):
                return _BAIL
            v = _try_eval(arg, env)
            if v is _BAIL:
                return _BAIL
            args.append(v)
        if name in _FOLD_FDLIBM:
            if len(args) != 1:
                return _BAIL
            v = _FOLD_FDLIBM[name](float(args[0]))
        elif name in _FOLD_EXACT_NAMES:
            if not args:
                return _BAIL
            try:
                v = _fold_exact()[name](*args)
            except (ValueError, OverflowError, TypeError, ZeroDivisionError):
                return _BAIL
        else:
            return _BAIL
        # Also rejects NA results (domain edges) -- those stay runtime
        if type(v) not in (int, float) or (type(v) is float and not math.isfinite(v)):
            return _BAIL
        return v
    return _BAIL


class ConstFoldTransformer:
    """
    TradingView parse-time constant folding, emulated at import time.

    TradingView evaluates every maximal constant subtree once, at parse time,
    with StrictMath (fdlibm) transcendentals, and embeds the result into the
    runtime program through the 16-decimal-place half-even cap of
    :func:`quantize_embed`. Constant chains -- plain declarations,
    ``var``-declared variables and ``:=`` lines whose right side is constant
    -- carry exact doubles between each other; only the embedding into a
    runtime expression is capped. Runtime (series/input-fed) calls of the
    same functions go through the Intel-LIBM intrinsics instead, which
    ``lib.math`` reproduces via ``core.pine_math``.

    This transformer replays that split: it propagates constants through
    single names in straight-line order (control flow conservatively kills
    the names it assigns), evaluates the constant subtrees with exact
    doubles on the verified fold surface (fdlibm sin/cos/exp/asin/acos and
    the correctly-rounded sqrt/abs/floor/ceil/min/max), and replaces each
    maximal constant subtree with its quantized literal. Anything outside
    the verified surface is left in place and keeps its runtime behavior.

    The Pine type of each folded name travels beside its value, in an
    environment scoped the same way: a nested function or a branch that
    rebinds a name must not move the type the enclosing scope still reads
    for it.

    Runs right after import normalization (the folder matches the
    ``lib.math.*`` chains that pass emits) and only over user/compiled
    scripts -- pynecore's own lib modules must keep their raw expressions.
    """

    def __init__(self) -> None:
        # Per-scope set of Persistent names a later store mutates; their
        # reads are never constant (see _mutated_stateful_names)
        self._blocked: frozenset[str] = frozenset()

    def visit(self, tree: ast.Module) -> ast.Module:
        self._blocked = _mutated_stateful_names(tree.body)
        self._process_body(tree.body, {}, {})
        return tree

    def _process_body(self, body: list[ast.stmt], env: dict[str, int | float],
                      types: dict[str, str]) -> None:
        for stmt in body:
            self._process_stmt(stmt, env, types)

    def _fold(self, node: ast.expr, env: dict[str, int | float],
              types: dict[str, str]) -> ast.expr:
        return _ExprFolder(env, types).visit(node)  # type: ignore[return-value]

    def _process_stmt(self, stmt: ast.stmt, env: dict[str, int | float],
                      types: dict[str, str]) -> None:
        # A walrus rebinding anywhere in the statement makes that name
        # untrackable from here on (evaluation order inside one statement
        # is not modeled)
        for n in ast.walk(stmt):
            if isinstance(n, ast.NamedExpr) and isinstance(n.target, ast.Name):
                env.pop(n.target.id, None)

        match stmt:
            case ast.FunctionDef() | ast.AsyncFunctionDef():
                for i, default in enumerate(stmt.args.defaults):
                    stmt.args.defaults[i] = self._fold(default, env, types)
                for i, kw_default in enumerate(stmt.args.kw_defaults):
                    if kw_default is not None:
                        stmt.args.kw_defaults[i] = self._fold(kw_default, env, types)
                env.pop(stmt.name, None)
                # Free variables of a nested scope are not tracked: the body
                # starts from an empty environment
                outer_blocked = self._blocked
                self._blocked = _mutated_stateful_names(stmt.body)
                self._process_body(stmt.body, {}, {})
                self._blocked = outer_blocked
            case ast.ClassDef():
                env.pop(stmt.name, None)
                self._process_body(stmt.body, {}, {})
            case ast.Assign():
                v = _try_eval(stmt.value, env)
                targets = [t.id for t in stmt.targets if isinstance(t, ast.Name)]
                simple = len(targets) == len(stmt.targets)
                if v is not _BAIL and simple:
                    # The emitted right side is the embedding-quantized
                    # literal (a residual runtime read -- series history,
                    # nested scope -- must see the embedded value); the
                    # environment keeps the exact double for const chains
                    q = quantize_embed(v)
                    ty = const_type(stmt.value, env, types)
                    if not (isinstance(stmt.value, ast.Constant) and stmt.value.value == q
                            and type(stmt.value.value) is type(q)):
                        stmt.value = ast.copy_location(ast.Constant(value=q), stmt.value)
                    _stamp(stmt.value, ty)
                    for name in targets:
                        types[name] = ty
                        if name in self._blocked:
                            env.pop(name, None)
                        else:
                            env[name] = v
                else:
                    stmt.value = self._fold(stmt.value, env, types)
                    for name in _assigned_names(stmt):
                        env.pop(name, None)
            case ast.AnnAssign():
                if stmt.value is None:
                    if isinstance(stmt.target, ast.Name):
                        env.pop(stmt.target.id, None)
                    return
                v = _try_eval(stmt.value, env)
                if v is not _BAIL and isinstance(stmt.target, ast.Name):
                    q = quantize_embed(v)
                    # An explicit annotation is a declaration, so it outranks
                    # what the folded initializer happens to be
                    declared = annotation_type(stmt.annotation)
                    ty = declared if declared != UNKNOWN else const_type(stmt.value, env, types)
                    if not (isinstance(stmt.value, ast.Constant) and stmt.value.value == q
                            and type(stmt.value.value) is type(q)):
                        stmt.value = ast.copy_location(ast.Constant(value=q), stmt.value)
                    _stamp(stmt.value, ty)
                    types[stmt.target.id] = ty
                    if stmt.target.id in self._blocked:
                        env.pop(stmt.target.id, None)
                    else:
                        env[stmt.target.id] = v
                else:
                    stmt.value = self._fold(stmt.value, env, types)
                    if isinstance(stmt.target, ast.Name):
                        env.pop(stmt.target.id, None)
            case ast.AugAssign():
                stmt.value = self._fold(stmt.value, env, types)
                if isinstance(stmt.target, ast.Name):
                    env.pop(stmt.target.id, None)
            case ast.If():
                stmt.test = self._fold(stmt.test, env, types)
                self._process_body(stmt.body, dict(env), dict(types))
                self._process_body(stmt.orelse, dict(env), dict(types))
                for name in _assigned_names(stmt):
                    env.pop(name, None)
            case ast.For() | ast.AsyncFor():
                stmt.iter = self._fold(stmt.iter, env, types)
                # Names the loop assigns are unknown both inside (previous
                # iteration) and after it; a constant assigned inside the
                # body re-enters the environment past its own line
                for name in _assigned_names(stmt):
                    env.pop(name, None)
                self._process_body(stmt.body, dict(env), dict(types))
                self._process_body(stmt.orelse, dict(env), dict(types))
                for name in _assigned_names(stmt):
                    env.pop(name, None)
            case ast.While():
                for name in _assigned_names(stmt):
                    env.pop(name, None)
                stmt.test = self._fold(stmt.test, env, types)
                self._process_body(stmt.body, dict(env), dict(types))
                self._process_body(stmt.orelse, dict(env), dict(types))
                for name in _assigned_names(stmt):
                    env.pop(name, None)
            case ast.With() | ast.AsyncWith():
                for item in stmt.items:
                    item.context_expr = self._fold(item.context_expr, env, types)
                for name in _assigned_names(stmt):
                    env.pop(name, None)
                self._process_body(stmt.body, dict(env), dict(types))
                for name in _assigned_names(stmt):
                    env.pop(name, None)
            case ast.Try():
                for name in _assigned_names(stmt):
                    env.pop(name, None)
                self._process_body(stmt.body, dict(env), dict(types))
                for handler in stmt.handlers:
                    self._process_body(handler.body, dict(env), dict(types))
                self._process_body(stmt.orelse, dict(env), dict(types))
                self._process_body(stmt.finalbody, dict(env), dict(types))
                for name in _assigned_names(stmt):
                    env.pop(name, None)
            case ast.Return() | ast.Expr():
                if stmt.value is not None:
                    stmt.value = self._fold(stmt.value, env, types)
            case ast.Assert():
                stmt.test = self._fold(stmt.test, env, types)
                if stmt.msg is not None:
                    stmt.msg = self._fold(stmt.msg, env, types)
            case ast.Raise():
                if stmt.exc is not None:
                    stmt.exc = self._fold(stmt.exc, env, types)
                if stmt.cause is not None:
                    stmt.cause = self._fold(stmt.cause, env, types)
            case ast.Global() | ast.Nonlocal():
                for name in stmt.names:
                    env.pop(name, None)
            case ast.Import() | ast.ImportFrom() | ast.Pass() | ast.Break() | ast.Continue() \
                    | ast.Delete():
                pass
            case _:
                # Unmodeled statement kind: fold nothing inside, kill what
                # it can rebind
                for name in _assigned_names(stmt):
                    env.pop(name, None)
