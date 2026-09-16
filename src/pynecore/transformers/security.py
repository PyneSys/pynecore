import ast
import copy
import hashlib
from collections.abc import Container

from ..core.import_hook import PYNE_RESERVED_NAME_CHAR
from .dynamic_default import is_script_entry
from .pine_type_rules import FactoryFields


# Strategy state accessors are meaningful only in the chart context — the
# security child process has no strategy state of its own, so referencing
# these in a request.security() / request.security_lower_tf() expression is
# always a programmer error. We raise SyntaxError when the offending attribute
# access is direct (i.e. not bound through a local alias). Transitive flow
# analysis is intentionally skipped — at runtime the security child returns
# inert defaults via the `lib._script is None` guard in the strategy module,
# so escaped cases fail safe rather than crash.
_FORBIDDEN_STRATEGY_STATE_ATTRS = frozenset({
    "equity", "eventrades", "grossloss", "grossprofit", "initial_capital",
    "losstrades", "max_drawdown", "max_runup", "netprofit", "openprofit",
    "position_avg_price", "position_size", "wintrades",
})

# Raw price series the security child can serve WITHOUT running the script's
# ``main()``: every one is set straight from the bar by ``_set_lib_properties``
# (the derived sources hl2/hlc3/ohlc4/hlcc4 are pure functions of that bar's
# OHLC, also set there), so reading ``lib.<field>`` after the per-bar property
# set is byte-identical to what ``main()`` would have produced. A
# ``request.security[_lower_tf]`` whose expression is only these can take the
# fast path (see ``security_process.security_process_main``).
_OHLCV_PASSTHROUGH_FIELDS = frozenset({
    "open", "high", "low", "close", "volume",
    "hl2", "hlc3", "ohlc4", "hlcc4",
})


# --- Cross-function signal lift ---

# Statement forms that always run once when their body is entered, so a
# ``request.security()`` reached through one of them runs exactly as often as
# the body itself. ``return`` belongs here: everything ahead of it is what
# decides whether it is reached, and ``_EXIT_STMTS`` already stops the scan at
# it. Everything else (``if`` / loops / ``with`` / ``try`` / ``match``) can
# skip or repeat the call and blocks the lift.
_UNCONDITIONAL_STMTS: tuple[type[ast.AST], ...] = (
    ast.Expr, ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Return,
)

# Expression forms that may leave a subexpression unevaluated (``a if c else
# b``, ``and`` / ``or`` short-circuit) or defer it to another invocation
# (lambda, comprehension). A call under one of them is not unconditional.
_CONDITIONAL_EXPRS: tuple[type[ast.AST], ...] = (
    ast.IfExp, ast.BoolOp, ast.Lambda, ast.ListComp, ast.SetComp,
    ast.DictComp, ast.GeneratorExp,
)

# Statements that can end the caller's body before a later statement is
# reached. One of them ahead of the call site means the call may not run on
# every bar, so its signal must not move above it.
_EXIT_STMTS: tuple[type[ast.AST], ...] = (
    ast.Return, ast.Raise, ast.Break, ast.Continue,
)

# Lift rounds: each round moves a signal one call level up, so the cap only
# has to exceed the deepest unconditional helper chain a script can have.
_MAX_LIFT_ROUNDS = 16


class _SignalArgSubstituter(ast.NodeTransformer):
    """Rewrite one lifted ``__sec_signal__`` argument into the caller's scope.

    A helper's signal argument is written in the helper's own names: its
    parameters and the simple bindings standing above its signal block. Both
    have an exact equivalent at the call site — the argument expression the
    caller passed, and the binding's own value expression — so substituting
    them yields the very expression the helper would have evaluated.

    :ivar failed: set when a name has no such equivalent; the caller then
        leaves the signal where it is
    """

    def __init__(self, mapping: dict[str, ast.expr],
                 bindings: dict[str, ast.expr]):
        self.mapping = mapping
        self.bindings = bindings
        self.failed = False
        # Guards a binding that (illegally) reads itself: without it the
        # substitution would recurse forever instead of bailing out.
        self._active: set[str] = set()

    def visit_Name(self, node: ast.Name) -> ast.expr:
        if node.id == 'lib':
            return node
        # Bindings come first: one of them may shadow a parameter, and then the
        # signal read the binding's value, not the argument the caller passed.
        if node.id in self.bindings and node.id not in self._active:
            self._active.add(node.id)
            try:
                return self.visit(copy.deepcopy(self.bindings[node.id]))
            finally:
                self._active.discard(node.id)
        if node.id in self.mapping:
            # The caller's expression is already in the target scope — return
            # it unvisited so its own names are left alone.
            return copy.deepcopy(self.mapping[node.id])
        self.failed = True
        return node


class SecurityTransformer(ast.NodeTransformer):
    """
    Transform request.security() calls into multiprocessing signal/write/read pattern.

    Transforms each lib.request.security(symbol, timeframe, expression, ...) call:

    1. Function start (chart context only):
       if __active_security__ is None:
           __sec_signal__("sec_id", symbol_expr, timeframe_expr)

    2. Original call position:
       if __active_security__ == "sec_id":
           __sec_write__("sec_id", expression)
       var = __sec_read__("sec_id", lib.na)

    3. Function end (chart context only):
       if __active_security__ is None:
           __sec_wait__("sec_id")

    Also creates module-level __security_contexts__ dict with metadata for each context.
    Non-constant symbol/timeframe/lookahead values (e.g., function parameters or
    input-derived expressions) are stored as None in __security_contexts__ and
    resolved at runtime via __sec_signal__ arguments.

    Must be applied after ImportNormalizerTransformer, before PersistentSeriesTransformer.
    """

    def __init__(self):
        self._counter = 0
        self._all_contexts: dict[str, dict[str, ast.expr]] = {}
        self._signal_args: dict[
            str, tuple[ast.expr | None, ast.expr | None, ast.expr | None]
        ] = {}
        self._needs_barmerge = False
        self._needs_ltf_unzip = False
        self._ltf_sec_ids: set[str] = set()
        self._module_file: str = '<script>'
        # Sids whose __sec_signal__ is emitted in the function's top block
        # (module-level or hoistable arguments). Only these may be depended on.
        self._top_sec_ids: set[str] = set()
        self._sid_lineno: dict[str, int] = {}
        # Runtime-resolved sids kept out of the top block because their call is
        # not reached on every run, and the ones that must stay there anyway
        # because another context reads them (see ``visit_Module``).
        self._deferred_by_reach: set[str] = set()
        self._keep_top: frozenset[str] = frozenset()

    def _gen_id(self) -> str:
        # The module hash keeps sec ids unique across modules: the main script and
        # any imported library may each have their own security calls, and their
        # contexts are merged into one registry by the runner
        module_hash = hashlib.sha1(self._module_file.encode()).hexdigest()[:8]
        sec_id = f"sec\xb7{module_hash}\xb7{self._counter}"
        self._counter += 1
        return sec_id

    @staticmethod
    def _is_security_call(node: ast.Call) -> bool:
        """Check if node is lib.request.security(...)."""
        return (isinstance(node.func, ast.Attribute)
                and node.func.attr == 'security'
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == 'request'
                and isinstance(node.func.value.value, ast.Name)
                and node.func.value.value.id == 'lib')

    @staticmethod
    def _is_security_lower_tf_call(node: ast.Call) -> bool:
        """Check if node is lib.request.security_lower_tf(...)."""
        return (isinstance(node.func, ast.Attribute)
                and node.func.attr == 'security_lower_tf'
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == 'request'
                and isinstance(node.func.value.value, ast.Name)
                and node.func.value.value.id == 'lib')

    @staticmethod
    def _extract_ltf_args(call: ast.Call) -> tuple[
        ast.expr | None, ast.expr | None, ast.expr | None, ast.expr | None
    ]:
        """Extract (symbol, timeframe, expression, ignore_invalid_symbol)
        from request.security_lower_tf() call.

        Note: no gaps parameter (LTF has no gaps/lookahead).
        """
        args = list(call.args)
        kwargs = {kw.arg: kw.value for kw in call.keywords if kw.arg is not None}
        return (
            kwargs.get('symbol', args[0] if len(args) > 0 else None),
            kwargs.get('timeframe', args[1] if len(args) > 1 else None),
            kwargs.get('expression', args[2] if len(args) > 2 else None),
            kwargs.get('ignore_invalid_symbol', args[3] if len(args) > 3 else None),
        )

    @staticmethod
    def _extract_args(call: ast.Call) -> tuple[
        ast.expr | None, ast.expr | None, ast.expr | None, ast.expr | None,
        ast.expr | None, ast.expr | None, ast.expr | None
    ]:
        """Extract (symbol, timeframe, expression, gaps, lookahead,
        ignore_invalid_symbol, currency) from request.security() call.

        Positional order matches Pine v6:
        ``security(symbol, timeframe, expression, gaps, lookahead,
        ignore_invalid_symbol, currency, ...)``.
        """
        args = list(call.args)
        kwargs = {kw.arg: kw.value for kw in call.keywords if kw.arg is not None}
        return (
            kwargs.get('symbol', args[0] if len(args) > 0 else None),
            kwargs.get('timeframe', args[1] if len(args) > 1 else None),
            kwargs.get('expression', args[2] if len(args) > 2 else None),
            kwargs.get('gaps', args[3] if len(args) > 3 else None),
            kwargs.get('lookahead', args[4] if len(args) > 4 else None),
            kwargs.get('ignore_invalid_symbol', args[5] if len(args) > 5 else None),
            kwargs.get('currency', args[6] if len(args) > 6 else None),
        )

    @staticmethod
    def _walk_skip_funcs(node: ast.AST):
        """Walk AST nodes, skipping nested function definitions."""
        yield node
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            yield from SecurityTransformer._walk_skip_funcs(child)

    @staticmethod
    def _find_forbidden_strategy_state(expr: ast.expr) -> ast.Attribute | None:
        """Return the first `lib.strategy.<state_attr>` Attribute node found in
        ``expr``, or None if there is none.

        Only direct attribute chains (`lib.strategy.position_size` form,
        produced by ImportNormalizerTransformer for any of `strategy.x`,
        `from pynecore.lib import strategy; strategy.x`, or
        `from pynecore.lib.strategy import x`) are detected. Local aliases
        (`ps = strategy.position_size; security(..., ps)`) are not —
        catching those would need a def-use chain analyzer; instead the
        runtime fallback in the strategy module returns inert defaults.
        """
        for sub in ast.walk(expr):
            if not isinstance(sub, ast.Attribute):
                continue
            if sub.attr not in _FORBIDDEN_STRATEGY_STATE_ATTRS:
                continue
            parent = sub.value
            if (isinstance(parent, ast.Attribute) and parent.attr == 'strategy'
                    and isinstance(parent.value, ast.Name)
                    and parent.value.id == 'lib'):
                return sub
        return None

    @staticmethod
    def _is_module_level_expr(node: ast.expr) -> bool:
        """Check if an expression can be evaluated at module level.

        Constants and lib.* attribute chains are safe. Function parameters
        and other local variables are not.
        """
        if isinstance(node, ast.Constant):
            return True
        if isinstance(node, ast.Attribute):
            return SecurityTransformer._is_module_level_expr(node.value)
        if isinstance(node, ast.Name):
            return node.id == 'lib'
        if isinstance(node, ast.Call):
            # A call is only safe if its callee AND every argument are —
            # ``ticker.heikinashi(inputSymbol)`` has a lib.* callee but a local
            # argument, so it must not be emitted at module level.
            return (SecurityTransformer._is_module_level_expr(node.func)
                    and all(SecurityTransformer._is_module_level_expr(a) for a in node.args)
                    and all(SecurityTransformer._is_module_level_expr(kw.value)
                            for kw in node.keywords))
        return False

    @classmethod
    def _is_simple_chain(cls, node: ast.expr,
                         hoistable: "Container[str]") -> bool:
        """Whether ``node`` is a Pine "simple" expression chain.

        Simple means: constants, ``lib.*`` attribute chains (``syminfo.*``,
        ``timeframe.*``, ...), calls over such chains (``input.*``,
        ``ticker.heikinashi(...)``), plain operators over them, and names bound
        by a hoistable binding or by a never-rebound parameter (see
        :meth:`_hoistable_bindings`). Such an expression can be evaluated at the
        very start of the function, which is what lets its ``__sec_signal__``
        move into the top block.

        :param node: expression to classify
        :param hoistable: names that already hold their final value at function
            entry — hoistable top-level bindings and stable parameters
        :return: True if the whole chain is simple
        """
        if isinstance(node, ast.Constant):
            return True
        if isinstance(node, ast.Name):
            return node.id == 'lib' or node.id in hoistable
        if isinstance(node, ast.Attribute):
            return cls._is_simple_chain(node.value, hoistable)
        if isinstance(node, ast.Call):
            return (cls._is_simple_chain(node.func, hoistable)
                    and all(cls._is_simple_chain(a, hoistable) for a in node.args)
                    and all(cls._is_simple_chain(kw.value, hoistable)
                            for kw in node.keywords))
        if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
                             ast.IfExp, ast.Tuple, ast.List)):
            return all(cls._is_simple_chain(child, hoistable)
                       for child in ast.iter_child_nodes(node)
                       if isinstance(child, ast.expr))
        return False

    @classmethod
    def _hoistable_bindings(
            cls, func: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> tuple[dict[str, ast.Assign], set[str]]:
        """Top-level bindings of ``func`` that may be moved to its first
        statements, together with its stable parameters.

        A parameter is stable when the function body never assigns or deletes
        it and does not declare it ``global`` / ``nonlocal``: it keeps the value
        bound at function entry for the whole call, so reading it in the first
        statement is exactly what the call site passed. Stable parameters are
        not hoisted (there is nothing to move), they only count as already
        available names for :meth:`_is_simple_chain`.

        A binding qualifies when it is a plain ``name = <simple chain>``
        assignment standing directly in the function body, the name is assigned
        exactly once in the whole function, and it is not declared ``global`` /
        ``nonlocal``. A parameter counts too, but only while every preceding
        statement is itself a hoistable binding — that is the form the
        instantiation pass produces when it pins a call site's constant
        argument into the function it instantiates. Such a value depends on nothing that
        runs inside the function, so evaluating it first is equivalent —
        and it makes a ``request.security()`` whose symbol/timeframe is built
        from it startable at the top of the function.

        Annotated assignments are excluded: those carry ``var`` / ``Persistent``
        semantics that later passes rewrite.

        :param func: the function being transformed
        :return: mapping of binding name to its assignment statement, and the
            set of stable parameter names
        """
        counts: dict[str, int] = {}
        declared: set[str] = set()
        for sub in ast.walk(func):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, (ast.Store, ast.Del)):
                counts[sub.id] = counts.get(sub.id, 0) + 1
            elif isinstance(sub, (ast.Global, ast.Nonlocal)):
                declared.update(sub.names)
        args = func.args
        params = {a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)}
        if args.vararg is not None:
            params.add(args.vararg.arg)
        if args.kwarg is not None:
            params.add(args.kwarg.arg)

        stable_params = {p for p in params
                         if p not in counts and p not in declared}

        hoistable: dict[str, ast.Assign] = {}
        # Names already holding their final value at function entry: the
        # bindings accepted so far plus every stable parameter.
        available: set[str] = set(stable_params)
        # A parameter may only be rebound while the whole prefix of the body
        # consists of hoistable bindings — otherwise an earlier statement could
        # still read the value passed by the caller.
        prefix = True
        for stmt in func.body:
            ok = False
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                target = stmt.targets[0]
                if isinstance(target, ast.Name):
                    name = target.id
                    if (name not in declared and counts.get(name) == 1
                            and (prefix or name not in params)
                            and cls._is_simple_chain(stmt.value, available)):
                        hoistable[name] = stmt
                        available.add(name)
                        ok = True
            prefix = prefix and ok
        return hoistable, stable_params

    @staticmethod
    def _referenced_names(node: ast.expr) -> set[str]:
        """Names read by ``node`` (``lib`` excluded)."""
        return {n.id for n in ast.walk(node)
                if isinstance(n, ast.Name) and n.id != 'lib'}

    @classmethod
    def _collect_hoisted(
            cls, hoistable: dict[str, ast.Assign], needed: set[str]
    ) -> list[ast.Assign]:
        """Transitive closure of the hoistable bindings ``needed`` requires,
        in the original source order."""
        selected: set[str] = set()
        pending = [n for n in needed if n in hoistable]
        while pending:
            name = pending.pop()
            if name in selected:
                continue
            selected.add(name)
            for ref in cls._referenced_names(hoistable[name].value):
                if ref in hoistable and ref not in selected:
                    pending.append(ref)
        return [stmt for name, stmt in hoistable.items() if name in selected]

    @staticmethod
    def _ohlcv_field(node: ast.expr) -> str | None:
        """Return the raw-OHLCV field name if ``node`` is a bare reference to one
        (``close`` or ``lib.close``), else None. Conservative: only a plain
        Name/``lib.<attr>`` matches — any computation falls through to the full
        ``main()`` path."""
        if isinstance(node, ast.Name) and node.id in _OHLCV_PASSTHROUGH_FIELDS:
            return node.id
        if (isinstance(node, ast.Attribute) and node.attr in _OHLCV_PASSTHROUGH_FIELDS
                and isinstance(node.value, ast.Name) and node.value.id == 'lib'):
            return node.attr
        return None

    @classmethod
    def _ohlcv_passthrough(cls, expression: ast.expr | None) -> tuple[list[str], bool] | None:
        """Detect a plain-OHLCV ``request.security[_lower_tf]`` expression.

        Returns ``(field_names, is_tuple)`` when every element is a raw price
        series (so the security child can serve it without running ``main()``),
        or ``None`` otherwise. ``is_tuple`` is True for a tuple/list expression
        (column-major arrays) and False for a scalar.
        """
        if expression is None:
            return None
        if isinstance(expression, (ast.Tuple, ast.List)):
            if not expression.elts:
                return None
            fields = [cls._ohlcv_field(e) for e in expression.elts]
            if all(f is not None for f in fields):
                return [f for f in fields if f is not None], True
            return None
        field = cls._ohlcv_field(expression)
        if field is not None:
            return [field], False
        return None

    # --- AST node builders ---

    @staticmethod
    def _lib_na() -> ast.Attribute:
        """Build: lib.na"""
        return ast.Attribute(
            value=ast.Name(id='lib', ctx=ast.Load()),
            attr='na', ctx=ast.Load()
        )

    @staticmethod
    def _default_gaps() -> ast.Attribute:
        """Build: lib.barmerge.gaps_off"""
        return ast.Attribute(
            value=ast.Attribute(
                value=ast.Name(id='lib', ctx=ast.Load()),
                attr='barmerge', ctx=ast.Load()
            ),
            attr='gaps_off', ctx=ast.Load()
        )

    @staticmethod
    def _func_call(name: str, *args: ast.expr) -> ast.Call:
        return ast.Call(
            func=ast.Name(id=name, ctx=ast.Load()),
            args=list(args), keywords=[]
        )

    @staticmethod
    def _is_none_check() -> ast.Compare:
        """Build: __active_security__ is None"""
        return ast.Compare(
            left=ast.Name(id='__active_security__', ctx=ast.Load()),
            ops=[ast.Is()], comparators=[ast.Constant(value=None)]
        )

    @staticmethod
    def _eq_check(sec_id: str) -> ast.Compare:
        """Build: __active_security__ == sec_id"""
        return ast.Compare(
            left=ast.Name(id='__active_security__', ctx=ast.Load()),
            ops=[ast.Eq()], comparators=[ast.Constant(value=sec_id)]
        )

    def _signal_block(self, sec_ids: list[str]) -> ast.If:
        """Build chart-context signal block for function start.

        Passes actual symbol and timeframe expressions to __sec_signal__
        so that runtime values (e.g., function parameters) are available.

        Only includes sec_ids whose symbol/timeframe are module-level
        expressions (constants or lib.* refs). Runtime-dependent signals
        are emitted inline before their write blocks by _transform_body.
        """
        body = []
        for s in sec_ids:
            args: list[ast.expr] = [ast.Constant(value=s)]
            sym_expr, tf_expr, la_expr = self._signal_args[s]
            args.append(copy.deepcopy(sym_expr) if sym_expr is not None
                        else ast.Constant(value=None))
            args.append(copy.deepcopy(tf_expr) if tf_expr is not None
                        else ast.Constant(value=None))
            # A lookahead that is not module-level evaluable (an input-derived
            # Pine "simple" value) is still passed here when its chain is
            # hoistable — the binding it reads is moved above this block.
            if la_expr is not None:
                args.append(copy.deepcopy(la_expr))
            body.append(ast.Expr(value=self._func_call('__sec_signal__', *args)))
        return ast.If(
            test=self._is_none_check(),
            body=body,
            orelse=[]
        )

    def _inline_signal(self, sec_id: str) -> ast.If:
        """Build a single inline signal for runtime-dependent
        symbol/timeframe/lookahead."""
        args: list[ast.expr] = [ast.Constant(value=sec_id)]
        sym_expr, tf_expr, la_expr = self._signal_args[sec_id]
        args.append(copy.deepcopy(sym_expr) if sym_expr is not None
                    else ast.Constant(value=None))
        args.append(copy.deepcopy(tf_expr) if tf_expr is not None
                    else ast.Constant(value=None))
        if la_expr is not None:
            args.append(copy.deepcopy(la_expr))
        return ast.If(
            test=self._is_none_check(),
            body=[ast.Expr(value=self._func_call('__sec_signal__', *args))],
            orelse=[]
        )

    def _wait_block(self, sec_ids: list[str]) -> ast.If:
        """Build chart-context wait block for function end."""
        return ast.If(
            test=self._is_none_check(),
            body=[
                ast.Expr(value=self._func_call('__sec_wait__', ast.Constant(value=s)))
                for s in sec_ids
            ],
            orelse=[]
        )

    @staticmethod
    def _in_same_context(sec_id: str) -> ast.Compare:
        """Build: sec_id in __same_context__"""
        return ast.Compare(
            left=ast.Constant(value=sec_id),
            ops=[ast.In()],
            comparators=[ast.Name(id='__same_context__', ctx=ast.Load())]
        )

    def _write_block(self, sec_id: str, expression: ast.expr) -> ast.If:
        """Build security-context write block.

        The condition fires in two cases:
        1. This IS the security process for sec_id (__active_security__ == sec_id)
        2. This is the chart process and sec_id is same-context (sec_id in __same_context__)
        """
        return ast.If(
            test=ast.BoolOp(
                op=ast.Or(),
                values=[self._eq_check(sec_id), self._in_same_context(sec_id)]
            ),
            body=[
                ast.Expr(value=self._func_call(
                    '__sec_write__', ast.Constant(value=sec_id), expression
                ))
            ],
            orelse=[]
        )

    def _sec_read_call(self, sec_id: str, tuple_len: int | None = None) -> ast.Call:
        """Build: __sec_read__("sec_id", <default>)

        Default is ``lib.na`` for scalar reads, or an N-tuple of ``lib.na``
        when the LHS unpacks the result. Pine `request.security()` returns a
        tuple of `na` (one per element) on no-data bars — emitting a single
        scalar would crash tuple-unpack with ``TypeError: cannot unpack
        non-iterable NA object`` in `gaps_on` between-period reads or after a
        `write_na` (session gap).
        """
        if tuple_len is None:
            default: ast.expr = self._lib_na()
        else:
            default = ast.Tuple(
                elts=[self._lib_na() for _ in range(tuple_len)],
                ctx=ast.Load()
            )
        return self._func_call(
            '__sec_read__', ast.Constant(value=sec_id), default
        )

    def _sec_read_call_ltf(self, sec_id: str, arity: int | None = None) -> ast.Call:
        """Build the LTF read expression.

        Scalar expression: ``__sec_read__("sec_id", [])`` returns the array
        directly. Tuple expression (``arity`` set): wrap in
        ``__ltf_unzip__(__sec_read__("sec_id", []), arity)`` to transpose the
        row-major intrabar buffer into ``arity`` column arrays.
        """
        read = self._func_call(
            '__sec_read__', ast.Constant(value=sec_id), ast.List(elts=[], ctx=ast.Load())
        )
        if arity is None:
            return read
        return self._func_call('__ltf_unzip__', read, ast.Constant(value=arity))

    @staticmethod
    def _detect_tuple_arity(stmt: ast.stmt, call: ast.Call) -> int | None:
        """If ``stmt`` is a tuple-unpack assignment whose RHS is exactly
        ``call``, return the LHS arity. Otherwise None.

        Pine statically enforces LHS-arity == RHS-arity for tuple-returning
        ``request.security()`` (compile errors CE10239 / CE10172), so the
        unpack target's arity is the authoritative source.

        Skipped: star-unpack (``*rest``), multi-target (``a = b = sec(...)``),
        annotated/augmented assigns, calls wrapped in a larger expression.
        """
        if not isinstance(stmt, ast.Assign):
            return None
        if len(stmt.targets) != 1:
            return None
        target = stmt.targets[0]
        if not isinstance(target, (ast.Tuple, ast.List)):
            return None
        if any(isinstance(e, ast.Starred) for e in target.elts):
            return None
        if stmt.value is not call:
            return None
        return len(target.elts)

    # --- Collection ---

    def _collect_calls(
            self, body: list[ast.stmt]
    ) -> list[tuple[ast.Call, str, bool]]:
        """
        Find all request.security() and request.security_lower_tf() calls in
        function body, skipping nested functions. Marks each call node with
        _sec_id attribute.

        :return: List of (call_node, sec_id, is_ltf) tuples
        """
        calls = []
        for stmt in body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in self._walk_skip_funcs(stmt):
                if isinstance(node, ast.Call):
                    if self._is_security_call(node):
                        sec_id = self._gen_id()
                        node._sec_id = sec_id  # type: ignore[attr-defined]
                        calls.append((node, sec_id, False))
                    elif self._is_security_lower_tf_call(node):
                        sec_id = self._gen_id()
                        node._sec_id = sec_id  # type: ignore[attr-defined]
                        self._ltf_sec_ids.add(sec_id)
                        calls.append((node, sec_id, True))
        return calls

    # --- Body transformation ---

    def _transform_body(
            self, body: list[ast.stmt], call_exprs: dict[str, ast.expr],
            runtime_sec_ids: set[str]
    ) -> list[ast.stmt]:
        """
        Recursively transform a body list: insert write blocks before statements
        containing security calls, and replace calls with __sec_read__.

        For runtime-dependent sec_ids (those in ``runtime_sec_ids``), also
        emits an inline signal just before the write block, so the signal
        runs after the symbol/timeframe variables are defined.

        Algorithm:
        1. For each statement, first recurse into compound sub-bodies
           (this replaces calls there and removes their _sec_id markers)
        2. Then walk the full statement — only expression-level calls remain
        3. Insert inline signals (if runtime) + write blocks, replace calls
        """
        new_body: list[ast.stmt] = []
        replacer = _CallReplacer(self)

        for stmt in body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                new_body.append(stmt)
                continue

            self._recurse_subbodies(stmt, call_exprs, runtime_sec_ids)

            call_nodes_here = [
                n for n in self._walk_skip_funcs(stmt)
                if isinstance(n, ast.Call) and hasattr(n, '_sec_id')
            ]

            if call_nodes_here:
                for call_node in call_nodes_here:
                    arity = self._detect_tuple_arity(stmt, call_node)
                    if arity is not None:
                        if not hasattr(call_node, '_tuple_len'):
                            call_node._tuple_len = arity  # type: ignore[attr-defined]
                        # LTF call with an opaque tuple expression (a function
                        # call, not a literal): the expression's own arity is
                        # unknowable here, but Pine enforces LHS-arity ==
                        # RHS-arity, so the unpack target is just as
                        # authoritative — wire the __ltf_unzip__ wrap from it.
                        if (getattr(call_node, '_sec_id') in self._ltf_sec_ids
                                and not hasattr(call_node, '_ltf_arity')):
                            call_node._ltf_arity = arity  # type: ignore[attr-defined]
                            self._needs_ltf_unzip = True

                for call_node in call_nodes_here:
                    sid = getattr(call_node, '_sec_id')
                    if sid in runtime_sec_ids:
                        new_body.append(self._inline_signal(sid))
                    expr = copy.deepcopy(call_exprs[sid])
                    expr = replacer.visit(expr)
                    new_body.append(self._write_block(sid, expr))
                new_body.append(replacer.visit(stmt))
            else:
                new_body.append(stmt)

        return new_body

    def _recurse_subbodies(
            self, stmt: ast.stmt, call_exprs: dict[str, ast.expr],
            runtime_sec_ids: set[str]
    ):
        """Recurse into sub-bodies of compound statements."""
        if isinstance(stmt, ast.If):
            stmt.body = self._transform_body(stmt.body, call_exprs, runtime_sec_ids)
            stmt.orelse = self._transform_body(stmt.orelse, call_exprs, runtime_sec_ids)
        elif isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
            stmt.body = self._transform_body(stmt.body, call_exprs, runtime_sec_ids)
            stmt.orelse = self._transform_body(stmt.orelse, call_exprs, runtime_sec_ids)
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            stmt.body = self._transform_body(stmt.body, call_exprs, runtime_sec_ids)
        elif isinstance(stmt, ast.Try):
            stmt.body = self._transform_body(stmt.body, call_exprs, runtime_sec_ids)
            for handler in stmt.handlers:
                handler.body = self._transform_body(handler.body, call_exprs, runtime_sec_ids)
            stmt.orelse = self._transform_body(stmt.orelse, call_exprs, runtime_sec_ids)
            stmt.finalbody = self._transform_body(stmt.finalbody, call_exprs, runtime_sec_ids)
        elif hasattr(ast, 'TryStar') and isinstance(stmt, ast.TryStar):
            stmt.body = self._transform_body(stmt.body, call_exprs, runtime_sec_ids)
            for handler in stmt.handlers:
                handler.body = self._transform_body(handler.body, call_exprs, runtime_sec_ids)
            stmt.orelse = self._transform_body(stmt.orelse, call_exprs, runtime_sec_ids)
            stmt.finalbody = self._transform_body(stmt.finalbody, call_exprs, runtime_sec_ids)
        elif hasattr(ast, 'Match') and isinstance(stmt, ast.Match):
            for case in stmt.cases:
                case.body = self._transform_body(case.body, call_exprs, runtime_sec_ids)

    # --- Function & module visitors ---

    def _process_func(self, node: ast.FunctionDef | ast.AsyncFunctionDef):
        """Transform a function containing request.security() / security_lower_tf() calls."""
        calls = self._collect_calls(node.body)

        if not calls:
            return self.generic_visit(node)

        call_exprs: dict[str, ast.expr] = {}
        sec_ids: list[str] = []

        for call, sec_id, is_ltf in calls:
            currency = None
            lookahead = None
            if is_ltf:
                symbol, timeframe, expression, ignore_invalid = (
                    self._extract_ltf_args(call)
                )
                gaps = None
                # A tuple/list expression makes security_lower_tf() return one
                # array per element (column-major). The arity is authoritative
                # from the expression itself, independent of how the result is
                # unpacked. Mark the call so the read is wrapped in __ltf_unzip__.
                if isinstance(expression, (ast.Tuple, ast.List)):
                    call._ltf_arity = len(expression.elts)  # type: ignore[attr-defined]
                    self._needs_ltf_unzip = True
            else:
                symbol, timeframe, expression, gaps, lookahead, ignore_invalid, currency = (
                    self._extract_args(call)
                )
                # A tuple/list expression pins the arity at the call itself,
                # which the unpack target below cannot see when the call is
                # not the direct RHS of an assignment (a nested function's
                # ``return``, say) — the security child would then read a
                # scalar ``na`` and crash the unpack at the call site.
                if isinstance(expression, (ast.Tuple, ast.List)):
                    call._tuple_len = len(expression.elts)  # type: ignore[attr-defined]

            if expression is not None:
                bad = self._find_forbidden_strategy_state(expression)
                if bad is not None:
                    fn_name = 'request.security_lower_tf' if is_ltf else 'request.security'
                    raise SyntaxError(
                        f"'strategy.{bad.attr}' cannot be used as the expression "
                        f"argument of {fn_name}() — strategy state is only "
                        f"available in the chart context, not in a security "
                        f"context.",
                        (self._module_file, getattr(bad, 'lineno', 0),
                         getattr(bad, 'col_offset', 0) + 1, None)
                    )

            call_exprs[sec_id] = expression if expression is not None else self._lib_na()

            # Input-derived (Pine "simple") lookahead — e.g. the standard TV
            # non-repaint HTF pattern ``repaint ? lookahead_off : lookahead_on``
            # — cannot be evaluated at module level, so it is resolved at
            # runtime through __sec_signal__ like a deferred symbol/timeframe.
            lookahead_rt: ast.expr | None = None
            if lookahead is not None and not self._is_module_level_expr(lookahead):
                lookahead_rt = copy.deepcopy(lookahead)

            # Store actual expressions for __sec_signal__ args (always passed at runtime)
            self._signal_args[sec_id] = (
                copy.deepcopy(symbol) if symbol is not None else None,
                copy.deepcopy(timeframe) if timeframe is not None else None,
                lookahead_rt,
            )

            # For __security_contexts__ at module level: only use values that are
            # evaluable at module scope. Function parameters etc. become None.
            ctx: dict[str, ast.expr] = {}
            if symbol is not None:
                if self._is_module_level_expr(symbol):
                    ctx['symbol'] = copy.deepcopy(symbol)
                else:
                    ctx['symbol'] = ast.Constant(value=None)
            if timeframe is not None:
                if self._is_module_level_expr(timeframe):
                    ctx['timeframe'] = copy.deepcopy(timeframe)
                else:
                    ctx['timeframe'] = ast.Constant(value=None)

            if is_ltf:
                ctx['is_ltf'] = ast.Constant(value=True)
            else:
                gaps_expr = (
                    copy.deepcopy(gaps) if gaps is not None else self._default_gaps()
                )
                ctx['gaps'] = gaps_expr
                if lookahead is not None:
                    # A module-level lookahead (``barmerge.lookahead_*``) is
                    # consumed at module load to wire the HTF transport. An
                    # input-derived expression would NameError at import time,
                    # so store None as a placeholder — the runtime value passed
                    # to ``__sec_signal__`` resolves the mode on the first bar.
                    ctx['lookahead'] = (copy.deepcopy(lookahead) if lookahead_rt is None
                                        else ast.Constant(value=None))

            if ignore_invalid is not None:
                ctx['ignore_invalid_symbol'] = copy.deepcopy(ignore_invalid)

            if not is_ltf and currency is not None:
                ctx['currency'] = copy.deepcopy(currency)

            # Plain-OHLCV fast path: when the expression is only raw price
            # series, the security child can serve it straight from each bar
            # without running main() (see security_process.security_process_main).
            passthrough = self._ohlcv_passthrough(expression)
            if passthrough is not None:
                fields, is_tuple = passthrough
                ctx['ohlcv_fields'] = ast.List(
                    elts=[ast.Constant(value=f) for f in fields], ctx=ast.Load()
                )
                ctx['ohlcv_tuple'] = ast.Constant(value=is_tuple)

            # Track if barmerge is used (only for non-LTF)
            if not is_ltf:
                if gaps is None:
                    self._needs_barmerge = True
                elif isinstance(gaps, ast.Attribute) and hasattr(gaps, 'value'):
                    v = gaps.value
                    if isinstance(v, ast.Attribute) and v.attr == 'barmerge':
                        self._needs_barmerge = True
                if (lookahead is not None
                        and isinstance(lookahead, ast.Attribute)
                        and hasattr(lookahead, 'value')):
                    v = lookahead.value
                    if isinstance(v, ast.Attribute) and v.attr == 'barmerge':
                        self._needs_barmerge = True

            self._all_contexts[sec_id] = ctx
            self._sid_lineno[sec_id] = getattr(call, 'lineno', 0)
            sec_ids.append(sec_id)

        # Separate top-block signals from runtime-dependent ones. A signal can
        # start at function entry when every argument is a Pine "simple" chain:
        # constants, lib.* values, a parameter the body never rebinds, or a
        # local bound once from such a chain. In the last case the binding
        # itself is hoisted above the signal block, so the value is already
        # there. Everything else (rebound parameters, series-dependent
        # expressions) must be signalled inline, after the variables it reads
        # have been assigned.
        #
        # A context whose symbol or timeframe is only known at runtime is
        # resolved -- its data located and loaded -- at its first signal. From
        # the top block that would happen on every run, even one that never
        # takes the branch the call stands in, and a feed nobody reads would
        # have to exist. Such a signal moves up only when its call is reached
        # every time the function runs -- or when another context depends on it:
        # that consumer's child waits for the producer's record, which the chart
        # only sends at the producer's signal, so a producer signalled behind a
        # branch the chart skipped would stall the consumer for good.
        hoistable, stable_params = self._hoistable_bindings(node)
        available = set(hoistable) | stable_params
        reached: set[str] = set()
        for stmt in self._unconditional_stmts(node.body):
            for call, sid, _is_ltf in calls:
                if sid not in reached and self._reached_unconditionally(stmt, call):
                    reached.add(sid)
        top_sec_ids = []
        runtime_sec_ids: set[str] = set()
        needed_names: set[str] = set()
        for sid in sec_ids:
            sym_expr, tf_expr, _la_expr = self._signal_args[sid]
            runtime_resolved = any(
                e is not None and not self._is_module_level_expr(e)
                for e in (sym_expr, tf_expr))
            if runtime_resolved and sid not in reached and sid not in self._keep_top:
                self._deferred_by_reach.add(sid)
                runtime_sec_ids.add(sid)
                continue
            exprs = [e for e in self._signal_args[sid] if e is not None]
            if all(self._is_simple_chain(e, available) for e in exprs):
                top_sec_ids.append(sid)
                for expr in exprs:
                    needed_names |= self._referenced_names(expr)
            else:
                runtime_sec_ids.add(sid)
        self._top_sec_ids.update(top_sec_ids)

        hoisted = self._collect_hoisted(hoistable, needed_names)
        hoisted_ids = {id(stmt) for stmt in hoisted}
        original_body = [stmt for stmt in node.body if id(stmt) not in hoisted_ids]
        top_block = [self._signal_block(top_sec_ids)] if top_sec_ids else []
        node.body = (
                list(hoisted)
                + top_block
                + self._transform_body(original_body, call_exprs, runtime_sec_ids)
                + [self._wait_block(sec_ids)]
        )

        return self.generic_visit(node)

    def _analyze_dependencies(self, node: ast.Module) -> None:
        """Run the taint analysis on the lowered module and record its result.

        Writes two keys into every context: ``depends`` (the sorted sids whose
        results flow into this sid's ``__sec_write__`` expression) and, when the
        write can run more than once per bar, ``in_loop``.

        A dependency on a context that is only resolved while the bar runs is
        legal — the runtime hands the resolved contexts to the children over the
        registry pipe — so nothing is rejected here.

        :param node: the lowered module
        """
        analyzer = _DependencyAnalyzer(node, list(self._all_contexts))
        analyzer.run()

        known = set(self._all_contexts)
        for sid, ctx in self._all_contexts.items():
            deps = sorted(analyzer.depends.get(sid, set()) & known)
            ctx['depends'] = ast.List(
                elts=[ast.Constant(value=d) for d in deps], ctx=ast.Load()
            )
            if sid in analyzer.in_loop:
                ctx['in_loop'] = ast.Constant(value=True)

    def _mark_always_read(self, module: ast.Module) -> None:
        """Flag every context whose ``__sec_read__`` runs on every chart bar.

        A read standing unconditionally in the script entry's body — directly,
        or in a helper that body reaches unconditionally — is taken on every
        bar the script runs. The runtime starts such contexts as a group when
        it first has to wait for one of them, instead of one cold start after
        the other (see ``_start`` in :mod:`pynecore.core.security`). A read
        behind a branch, a loop or a short-circuit gets no flag: that run may
        never take it, and a context nobody reads must not get a child process.

        :param module: the transformed module, with the reads already emitted
        """
        funcs: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
        ambiguous: set[str] = set()
        entry: ast.FunctionDef | ast.AsyncFunctionDef | None = None
        for node in ast.walk(module):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name in funcs:
                # Two definitions of one name: a call by that name cannot be
                # attributed to either, so neither is followed.
                ambiguous.add(node.name)
            funcs[node.name] = node
            if entry is None and is_script_entry(node):
                entry = node
        if entry is None:
            return

        pending = [entry]
        seen = {id(entry)}
        while pending:
            func = pending.pop()
            for stmt in self._unconditional_stmts(func.body):
                for sub in self._walk_skip_funcs(stmt):
                    if not (isinstance(sub, ast.Call)
                            and isinstance(sub.func, ast.Name)
                            and self._reached_unconditionally(stmt, sub)):
                        continue
                    if sub.func.id == '__sec_read__':
                        sid = self._call_sid(sub)
                        ctx = self._all_contexts.get(sid) if sid else None
                        if ctx is not None:
                            ctx['always_read'] = ast.Constant(value=True)
                        continue
                    callee = funcs.get(sub.func.id)
                    if (callee is not None and sub.func.id not in ambiguous
                            and id(callee) not in seen):
                        seen.add(id(callee))
                        pending.append(callee)

    def _mark_signal_per_bar(self, module: ast.Module) -> None:
        """Flag every context signalled exactly once per script entry run.

        The developing batch plans one round per chart bar and every
        ``__sec_signal__`` consumes the next one, so the runtime may only batch
        a context whose signal is emitted once per ``main()`` call. That is
        provable for exactly one shape: the sid's only ``__sec_signal__`` in the
        whole module stands in a chart-guard block directly in the script
        entry's body, ahead of anything that can end that body early. A signal
        left in a helper (conditionally called, called twice, or called in a
        loop) gets no flag and keeps the per-bar transport.

        :param module: the transformed module, after the cross-function lift
        """
        counts: dict[str, int] = {}
        for node in ast.walk(module):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == '__sec_signal__'):
                sid = self._call_sid(node)
                if sid is not None:
                    counts[sid] = counts.get(sid, 0) + 1

        entry: ast.FunctionDef | ast.AsyncFunctionDef | None = None
        for node in ast.walk(module):
            if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and is_script_entry(node)):
                entry = node
                break
        if entry is None:
            return

        for stmt in entry.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                continue
            calls = self._guard_block_calls(stmt, '__sec_signal__')
            if calls is not None:
                for call in calls:
                    sid = self._call_sid(call)
                    if sid is None or counts.get(sid) != 1:
                        continue
                    ctx = self._all_contexts.get(sid)
                    if ctx is not None:
                        ctx['signal_per_bar'] = ast.Constant(value=True)
            if any(isinstance(sub, _EXIT_STMTS)
                   for sub in self._walk_skip_funcs(stmt)):
                return

    @classmethod
    def _unconditional_stmts(cls, body: list[ast.stmt]):
        """The statements of ``body`` that run every time it is entered.

        Stops after the first statement that can end the body early, and skips
        nested definitions — what their own body does belongs to their scope.

        :param body: the statement list to scan
        """
        for stmt in body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                continue
            if isinstance(stmt, _UNCONDITIONAL_STMTS):
                yield stmt
            if any(isinstance(sub, _EXIT_STMTS)
                   for sub in cls._walk_skip_funcs(stmt)):
                return

    # --- Cross-function signal lift ---

    @classmethod
    def _guard_block_calls(cls, stmt: ast.stmt, name: str) -> list[ast.Call] | None:
        """Return the calls of ``name`` in a chart-guard block, else None.

        Recognises exactly the shape this transformer emits: an
        ``if __active_security__ is None:`` whose body is nothing but
        ``name(...)`` expression statements.

        :param stmt: candidate statement
        :param name: ``__sec_signal__`` or ``__sec_wait__``
        :return: the calls in source order, or None if ``stmt`` is not such a
            block
        """
        if not isinstance(stmt, ast.If) or stmt.orelse or not stmt.body:
            return None
        test = stmt.test
        if not (isinstance(test, ast.Compare) and len(test.ops) == 1
                and isinstance(test.ops[0], ast.Is)
                and isinstance(test.left, ast.Name)
                and test.left.id == '__active_security__'):
            return None
        comparator = test.comparators[0]
        if not (isinstance(comparator, ast.Constant) and comparator.value is None):
            return None
        calls = []
        for sub in stmt.body:
            if not (isinstance(sub, ast.Expr) and isinstance(sub.value, ast.Call)
                    and isinstance(sub.value.func, ast.Name)
                    and sub.value.func.id == name):
                return None
            calls.append(sub.value)
        return calls

    @classmethod
    def _find_guard_block(cls, body: list[ast.stmt],
                          name: str) -> tuple[int, ast.If] | None:
        """First chart-guard block of ``name`` in ``body`` with its index."""
        for idx, stmt in enumerate(body):
            if (isinstance(stmt, ast.If)
                    and cls._guard_block_calls(stmt, name) is not None):
                return idx, stmt
        return None

    @staticmethod
    def _call_sid(call: ast.Call) -> str | None:
        """The sid a ``__sec_signal__`` / ``__sec_wait__`` call carries."""
        if not call.args:
            return None
        first = call.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            return first.value
        return None

    @classmethod
    def _reached_unconditionally(cls, node: ast.AST, target: ast.Call) -> bool:
        """Whether ``target`` is evaluated every time ``node`` is evaluated.

        Walks only the expression children, stopping at any form that may skip
        its operands (see ``_CONDITIONAL_EXPRS``). Call arguments count as
        unconditional: Python evaluates every one of them before the call.
        """
        if node is target:
            return True
        if isinstance(node, _CONDITIONAL_EXPRS):
            return False
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.expr, ast.keyword)):
                if cls._reached_unconditionally(child, target):
                    return True
        return False

    @classmethod
    def _leading_bindings(cls, body: list[ast.stmt], limit: int,
                          params: set[str]) -> dict[str, ast.expr] | None:
        """``name = <expr>`` bindings standing in ``body[:limit]``.

        These are the statements the signal block was placed after — the
        bindings hoisted for its arguments and the call-site constants the
        instantiation pass pinned — so a signal argument reading one of them
        reads exactly the value expression recorded here.

        Returns None when the region is not that shape, which blocks the lift:

        - a statement that is not a plain single-name assignment could have
          changed anything the signal reads,
        - a name bound twice has no single value expression to substitute,
        - a parameter rebound from something other than a static expression is
          the case requirement (2) rules out: the value the helper signals is
          no longer the argument the call site passed. A rebinding to a
          constant or a ``lib.*`` chain (what the instantiation pass pins) is
          exact, so it stays allowed.

        :param body: the helper's statement list
        :param limit: index of the signal block
        :param params: the helper's parameter names
        :return: the bindings, or None if the region blocks the lift
        """
        assigned: list[tuple[str, ast.expr]] = []
        for stmt in body[:limit]:
            if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1):
                return None
            target = stmt.targets[0]
            if not isinstance(target, ast.Name):
                return None
            if target.id in params and cls._referenced_names(stmt.value):
                return None
            assigned.append((target.id, stmt.value))

        bindings: dict[str, ast.expr] = {}
        for idx, (name, value) in enumerate(assigned):
            if name in bindings:
                return None
            # Names bound BELOW this statement. A value reading one of them
            # would be substituted with that later value — the substituter
            # resolves a binding against the whole region — while the helper
            # evaluated what stood here at the assignment point.
            later = {n for n, _ in assigned[idx + 1:]}
            if cls._referenced_names(value) & later:
                return None
            bindings[name] = value
        return bindings

    @staticmethod
    def _param_mapping(func: ast.FunctionDef | ast.AsyncFunctionDef,
                       call: ast.Call) -> dict[str, ast.expr] | None:
        """Map ``func``'s parameters to the expressions ``call`` passes.

        Returns None for any shape whose binding is not statically decidable
        (``*args`` / ``**kwargs`` on either side, an unknown keyword, too many
        positional arguments). Parameters the call leaves to their default are
        simply absent: substituting a default expression could duplicate an
        ``input.*`` registration, so a signal argument reading one blocks the
        lift.
        """
        args = func.args
        if args.vararg is not None or args.kwarg is not None:
            return None
        if any(isinstance(a, ast.Starred) for a in call.args):
            return None
        if any(kw.arg is None for kw in call.keywords):
            return None
        positional = [a.arg for a in (*args.posonlyargs, *args.args)]
        known = set(positional) | {a.arg for a in args.kwonlyargs}
        if len(call.args) > len(positional):
            return None
        mapping: dict[str, ast.expr] = {}
        for idx, value in enumerate(call.args):
            mapping[positional[idx]] = value
        for kw in call.keywords:
            if kw.arg not in known or kw.arg in mapping:
                return None
            mapping[kw.arg] = kw.value
        return mapping

    def _lift_candidates(
            self, module: ast.Module
    ) -> list[tuple[ast.FunctionDef | ast.AsyncFunctionDef,
                    ast.FunctionDef | ast.AsyncFunctionDef, ast.Call, int]]:
        """Helper functions whose signal block may move up one call level.

        A candidate is a function with a chart-guard signal block whose name is
        read exactly once in the whole module, at a direct-``Name`` call site
        that stands in an enclosing function's own statement list, is evaluated
        unconditionally there, and has no statement ahead of it that can end
        that body early.

        :param module: the lowered module
        :return: ``(helper, caller, call, stmt_index)`` tuples, ordered by call
            site so lifted signals keep their source order
        """
        funcs: list[ast.FunctionDef | ast.AsyncFunctionDef] = [
            n for n in ast.walk(module)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        # A name read anywhere else (an alias, a callback, a second call) means
        # the helper may run a different number of times than the single site
        # below suggests.
        load_counts: dict[str, int] = {}
        for sub in ast.walk(module):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                load_counts[sub.id] = load_counts.get(sub.id, 0) + 1

        # Every unconditional direct-call site, keyed by callee name.
        sites: dict[str, tuple[ast.FunctionDef | ast.AsyncFunctionDef,
                               ast.Call, int]] = {}
        for caller in funcs:
            blocked = False
            for idx, stmt in enumerate(caller.body):
                if blocked:
                    break
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef,
                                     ast.ClassDef)):
                    # A definition neither calls nor exits this body; what its
                    # own body does belongs to its scope.
                    continue
                if isinstance(stmt, _UNCONDITIONAL_STMTS):
                    for sub in self._walk_skip_funcs(stmt):
                        if (isinstance(sub, ast.Call)
                                and isinstance(sub.func, ast.Name)
                                and self._reached_unconditionally(stmt, sub)):
                            sites[sub.func.id] = (caller, sub, idx)
                # Anything beyond this statement may never be reached. A
                # nested def's own ``return`` belongs to that function, not to
                # this body, so nested scopes are skipped.
                blocked = any(isinstance(sub, _EXIT_STMTS)
                              for sub in self._walk_skip_funcs(stmt))

        candidates = []
        for helper in funcs:
            if self._find_guard_block(helper.body, '__sec_signal__') is None:
                continue
            if load_counts.get(helper.name, 0) != 1:
                continue
            site = sites.get(helper.name)
            if site is None:
                continue
            caller, call, idx = site
            if caller is helper:
                continue
            candidates.append((helper, caller, call, idx))
        candidates.sort(key=lambda c: c[3])
        return candidates

    def _lift_signal(self, helper: ast.FunctionDef | ast.AsyncFunctionDef,
                     caller: ast.FunctionDef | ast.AsyncFunctionDef,
                     call: ast.Call) -> bool:
        """Move every provably liftable signal of ``helper`` into ``caller``.

        :return: True if at least one signal moved
        """
        found = self._find_guard_block(helper.body, '__sec_signal__')
        if found is None:
            return False
        sig_idx, sig_block = found
        mapping = self._param_mapping(helper, call)
        if mapping is None:
            return False
        args = helper.args
        params = {a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)}
        bindings = self._leading_bindings(helper.body, sig_idx, params)
        if bindings is None:
            return False

        hoistable, stable_params = self._hoistable_bindings(caller)
        available = set(hoistable) | stable_params

        lifted: list[tuple[ast.Expr, str, list[ast.expr]]] = []
        for stmt in list(sig_block.body):
            if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)):
                continue
            sid = self._call_sid(stmt.value)
            if sid is None:
                continue
            new_args: list[ast.expr] = []
            ok = True
            for arg in stmt.value.args[1:]:
                substituter = _SignalArgSubstituter(mapping, bindings)
                new_arg = substituter.visit(copy.deepcopy(arg))
                if substituter.failed or not self._is_simple_chain(new_arg, available):
                    ok = False
                    break
                new_args.append(new_arg)
            if ok:
                lifted.append((stmt, sid, new_args))

        if not lifted:
            return False

        lifted_sids = {sid for _, sid, _ in lifted}
        needed: set[str] = set()
        for _, _, new_args in lifted:
            for arg in new_args:
                needed |= self._referenced_names(arg)

        # The caller's signal block: reuse the one it already has, or start one
        # at the top of its body (a caller without security calls of its own
        # has none yet).
        found_caller = self._find_guard_block(caller.body, '__sec_signal__')
        if found_caller is None:
            caller_block = ast.If(test=self._is_none_check(), body=[], orelse=[])
            caller.body.insert(0, caller_block)
        else:
            caller_block = found_caller[1]

        for stmt, sid, new_args in lifted:
            sig_block.body.remove(stmt)
            new_call = self._func_call(
                '__sec_signal__', ast.Constant(value=sid), *new_args
            )
            new_stmt = ast.Expr(value=new_call)
            ast.copy_location(new_stmt, call)
            ast.fix_missing_locations(new_stmt)
            caller_block.body.append(new_stmt)
        if not sig_block.body:
            helper.body.remove(sig_block)

        # Every binding the moved arguments read has to stand above the block.
        block_pos = caller.body.index(caller_block)
        for stmt in self._collect_hoisted(hoistable, needed):
            pos = caller.body.index(stmt)
            if pos > block_pos:
                caller.body.pop(pos)
                caller.body.insert(block_pos, stmt)
                block_pos += 1

        self._move_waits(helper, caller, lifted_sids, call)
        return True

    def _move_waits(self, helper: ast.FunctionDef | ast.AsyncFunctionDef,
                    caller: ast.FunctionDef | ast.AsyncFunctionDef,
                    sids: set[str], call: ast.Call) -> None:
        """Follow the lifted signals with their ``__sec_wait__`` calls.

        ``__sec_wait__`` settles the round the signal launched, so it belongs to
        the end of whatever scope signalled — leave it in the helper and the
        helper would settle a round the caller starts on a later statement,
        serialising exactly what the lift set out to overlap. The wait is
        idempotent (``_settle_round`` clears ``needs_wait``), the next bar's
        signal settles first anyway, and ``end_bar()`` catches whatever
        ``main()`` skipped, so moving it can strand nothing.
        """
        found = self._find_guard_block(helper.body, '__sec_wait__')
        if found is not None:
            wait_block = found[1]
            for stmt in list(wait_block.body):
                if (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
                        and self._call_sid(stmt.value) in sids):
                    wait_block.body.remove(stmt)
            if not wait_block.body:
                helper.body.remove(wait_block)

        found_caller = self._find_guard_block(caller.body, '__sec_wait__')
        if found_caller is None:
            trailing = caller.body[-1] if caller.body else None
            if isinstance(trailing, ast.Return):
                if trailing.value is not None and any(
                        node is call for node in ast.walk(trailing.value)):
                    # The ``return`` itself runs the helper, and with it the
                    # reads. A wait in front of it would settle the round
                    # before the value is read: the chart would sit out the
                    # child's whole ``main()`` instead of being released at the
                    # write, one context after the other. This scope has no
                    # statement after the reads, so the next signal and
                    # ``end_bar()`` settle the round.
                    return
                caller_block = ast.If(test=self._is_none_check(), body=[], orelse=[])
                ast.copy_location(caller_block, call)
                # Appended after the ``return``, the block would be dead code.
                caller.body.insert(len(caller.body) - 1, caller_block)
            else:
                caller_block = ast.If(test=self._is_none_check(), body=[], orelse=[])
                ast.copy_location(caller_block, call)
                caller.body.append(caller_block)
        else:
            caller_block = found_caller[1]
        present = {self._call_sid(c)
                   for c in self._guard_block_calls(caller_block, '__sec_wait__') or []}
        for sid in sorted(sids):
            if sid in present:
                continue
            new_stmt = ast.Expr(value=self._func_call(
                '__sec_wait__', ast.Constant(value=sid)
            ))
            ast.copy_location(new_stmt, call)
            ast.fix_missing_locations(new_stmt)
            caller_block.body.append(new_stmt)

    def _lift_helper_signals(self, module: ast.Module) -> None:
        """Move helper-scoped ``__sec_signal__`` calls up to their caller.

        A signal emitted at a helper's top runs where the helper is CALLED, so
        N helper calls standing side by side in ``main()`` launch their rounds
        one after another, each behind the previous one's read. Lifting the
        signal to the caller's top block starts every round before the first
        read, which is the shape a script with its ``request.security()`` calls
        written directly in ``main()`` already gets.

        The move is only made where it provably changes nothing: the helper's
        single call site must run exactly once per invocation of the caller's
        body, and every signal argument must translate into an expression the
        caller can evaluate at its own top (see :meth:`_lift_signal`). Each
        round moves one call level, so a chain of unconditional helpers ends up
        signalling from the outermost caller.

        :param module: the lowered module
        """
        for _ in range(_MAX_LIFT_ROUNDS):
            moved = False
            for helper, caller, call, _idx in self._lift_candidates(module):
                if self._lift_signal(helper, caller, call):
                    moved = True
            if not moved:
                break

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        return self._process_func(node)  # type: ignore[return-value]

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        return self._process_func(node)  # type: ignore[return-value]

    def visit_Module(self, node: ast.Module) -> ast.Module:
        self._module_file = getattr(node, '_module_file_path', '<script>')
        original = None
        if not self._keep_top and any(
                isinstance(sub, ast.Call) and (self._is_security_call(sub)
                                               or self._is_security_lower_tf_call(sub))
                for sub in ast.walk(node)):
            original = copy.deepcopy(node)
        node = self.generic_visit(node)  # type: ignore[assignment]

        if self._all_contexts:
            self._lift_helper_signals(node)
            self._analyze_dependencies(node)
            # Dependencies are only known on the lowered module. A deferred
            # signal that turns out to be a producer must stay in the top block
            # after all, so transform the module again with that decision fixed.
            # Sids are counted in visit order, so the second pass names every
            # context the same way.
            producers: set[str] = set()
            for ctx in self._all_contexts.values():
                deps = ctx['depends']
                if isinstance(deps, ast.List):
                    producers.update(dep.value for dep in deps.elts
                                     if isinstance(dep, ast.Constant))
            keep_top = self._deferred_by_reach & producers
            if keep_top and original is not None:
                second = SecurityTransformer()
                second._keep_top = frozenset(keep_top)
                return second.visit(original)
            self._mark_always_read(node)
            self._mark_signal_per_bar(node)

            # Add barmerge import if needed (SecurityTransformer runs AFTER ImportNormalizer,
            # so we must add it ourselves)
            if self._needs_barmerge:
                node.body.insert(0, ast.Import(
                    names=[ast.alias(name='pynecore.lib.barmerge', asname=None)]
                ))

            # __ltf_unzip__ transposes tuple-valued security_lower_tf() results.
            if self._needs_ltf_unzip:
                node.body.insert(0, ast.ImportFrom(
                    module='pynecore.core.security',
                    names=[ast.alias(name='__ltf_unzip__', asname=None)],
                    level=0,
                ))

            node.body.append(ast.Assign(
                targets=[ast.Name(id='__security_contexts__', ctx=ast.Store())],
                value=ast.Dict(
                    keys=[ast.Constant(value=sid) for sid in self._all_contexts],
                    values=[
                        ast.Dict(
                            keys=[ast.Constant(value=k) for k in ctx],
                            values=list(ctx.values())
                        )
                        for ctx in self._all_contexts.values()
                    ]
                )
            ))

        return node

# Statement / expression node types the dependency analyzer does not model.
# Any scope containing one of them falls back to "depends on every other sid"
# (a safe over-approximation: an unmodelled construct can only add flow).
_UNMODELLED_NODES: tuple[type[ast.AST], ...] = tuple(
    n for n in (
        getattr(ast, name, None) for name in (
            'Try', 'TryStar', 'With', 'AsyncWith', 'AsyncFor', 'Raise', 'Delete',
            'ClassDef', 'Lambda', 'ListComp', 'SetComp',
            'DictComp', 'GeneratorExp', 'Yield', 'YieldFrom', 'Await', 'Assert',
        )
    ) if n is not None
)

# Expression node types whose taint is simply the union of their children's.
_UNION_EXPRS: tuple[type[ast.AST], ...] = (
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp, ast.Tuple,
    ast.List, ast.Set, ast.Dict, ast.Slice, ast.Starred, ast.JoinedStr,
    ast.FormattedValue, ast.Subscript,
)

# Names that can never carry taint: ``lib`` is the library namespace root, so
# ``lib.plot(tainted)`` must not make every later ``lib.close`` tainted.
_UNTAINTABLE_ROOTS = frozenset({'lib'})

# ``lib`` namespaces whose functions mutate the collection passed as their
# first argument (``array.push(id, v)``, ``matrix.set(id, r, c, v)``, ...).
# Only these propagate taint back into an argument — doing it for every call
# would make an ordinary ``ta.sma(series, length)`` taint its length input.
_MUTATING_NAMESPACES = frozenset({'array', 'matrix', 'map'})

# Fixpoint safety net — the lattice is finite and monotone, so the loop always
# converges; the cap only guards against an unforeseen non-monotone edit.
_MAX_FIXPOINT_ROUNDS = 200

# Prefix marking a symbolic parameter token inside a taint set. A call site
# replaces the callee's own tokens with that call's argument taints, so two
# calls of one function never share their arguments' taint.
_PARAM_PREFIX = '\x00param\x00'

# Pseudo-parameter name carrying the control taint a call site runs under —
# the caller's ``if`` condition decides whether the callee's writes and
# mutations happen at all, so it is substituted like any other argument.
_CTRL_PARAM = '<ctrl>'


class _Scope:
    """One lexical scope of the analysed module (the module or a function).

    :ivar key: unique dotted path of the scope (``''`` for the module)
    :ivar node: the ``Module`` / ``FunctionDef`` the scope belongs to
    :ivar parent: key of the enclosing scope, ``None`` for the module
    :ivar locals: names bound inside the scope (parameters, assignment
        targets, nested ``def`` names) that are not declared ``global`` /
        ``nonlocal``
    :ivar globals: names the scope declares ``global``
    :ivar funcs: nested function name to scope key
    :ivar body: the scope's statement list
    """

    __slots__ = ('key', 'node', 'parent', 'locals', 'globals', 'funcs', 'body')

    def __init__(self, key: str, node: ast.AST, parent: str | None,
                 body: list[ast.stmt]):
        self.key = key
        self.node = node
        self.parent = parent
        self.locals: set[str] = set()
        self.globals: set[str] = set()
        self.funcs: dict[str, str] = {}
        self.body = body


class _DependencyAnalyzer:
    """Forward taint analysis over an already lowered security AST.

    After :class:`SecurityTransformer` has rewritten the module, every
    ``request.security()`` result is read through ``__sec_read__("<sid>", ...)``
    and every security expression is written through
    ``__sec_write__("<sid>", <expr>)``. This pass answers one question: which
    sids' results flow into which other sid's write expression.

    What the analysis models:

    - **Flow-insensitive on data**: a name's taint is the union over every
      assignment to it anywhere in its scope (so ``var`` backward flow and
      loop-carried values are covered without ordering rules).
    - **Lexically scoped names**: a name belongs to the scope that binds it —
      parameters and assignment targets are local, everything else resolves
      outwards, so a nested ``def`` reading a ``main`` local (the usual closure
      in a Pyne script) shares that local's taint while two same-named locals
      of unrelated functions do not. A ``global`` / ``nonlocal`` declaration
      rebinds the name to the module / the enclosing scope that owns it.
    - **Flow-sensitive on control**: a control-taint set accumulates the taint
      of enclosing ``if`` / ``for`` / ``while`` / ``match`` conditions, and a
      tainted ``return`` / ``break`` / ``continue`` taints the rest of its
      function.
    - **Alias classes**: ``a = b`` merges the two names into one taint class,
      so mutating a container through either alias is visible on both. A
      call whose callee returns one of its parameters (or an alias of it)
      merges the receiving name with that argument the same way.
    - **User functions, per call site**: each parameter carries a symbolic
      token instead of a concrete sid. A call's result is the callee's return
      taint with its own tokens replaced by this call's argument taints, so an
      argument only reaches the result when the callee's return actually draws
      on it, and two calls of one function do not mix. Taint the callee adds to
      a parameter (a mutated array, matrix or map) flows back into the caller's
      argument variable the same way. The control taint of the call site is a
      pseudo-parameter: a mutation or write the callee performs happens only if
      the caller reached the call, so it carries the caller's condition too. A
      call whose callee is not a modelled
      function keeps the conservative rule: every argument flows into the
      result. Tokens that survive into a write inside a callee are resolved
      against the union of that parameter's call sites, and a function that is
      called somewhere the analysis could not model resolves to every sid.
    - **Fallback**: a scope containing an unmodelled construct makes every
      write in it — and every call of it — depend on all other sids.

    Over-approximation is safe only towards EARLIER-sited producers, and the
    result is filtered to those: for them ``depends`` is just a wait filter, so
    a too-large set costs extra synchronisation and nothing else. A back-edge
    onto a later-sited producer is NOT safe — the child would block a write
    that the chart is waiting for before it ever reaches the producer's site —
    so every such edge is dropped (see :meth:`run`).
    """

    def __init__(self, module: ast.Module, all_sids: list[str]):
        self._module = module
        self._all: frozenset[str] = frozenset(all_sids)
        self.depends: dict[str, set[str]] = {sid: set() for sid in all_sids}
        self.in_loop: set[str] = set()
        self._funcs: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
        self._scopes: dict[str, _Scope] = {}
        self._key_of: dict[int, str] = {}
        self._called_names: set[str] = set()
        self._bound: set[str] = set()
        self._fallback_scopes: set[str] = set()
        self.fallback_reasons: dict[str, str] = {}
        self._ret: dict[str, set[str]] = {}
        self._ret_alias: dict[str, set[str]] = {}
        self._param_out: dict[str, dict[str, set[str]]] = {}
        self._param_in: dict[str, set[str]] = {}
        self._escape: dict[str, set[str]] = {}
        self._parent: dict[str, str] = {}
        self._taint: dict[str, set[str]] = {}
        self._ctrl: set[str] = set()
        self._order: dict[str, int] = {}
        self._scope: str = ''
        self._changed = False
        self._final = False

    # --- scopes ---

    @classmethod
    def _child_funcs(cls, node: ast.AST):
        """Yield the function definitions directly owned by ``node``'s scope."""
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield child
            else:
                yield from cls._child_funcs(child)

    @staticmethod
    def _params(fn: 'ast.FunctionDef | ast.AsyncFunctionDef') -> list[ast.arg]:
        spec = fn.args
        params = list(spec.posonlyargs) + list(spec.args) + list(spec.kwonlyargs)
        if spec.vararg is not None:
            params.append(spec.vararg)
        if spec.kwarg is not None:
            params.append(spec.kwarg)
        return params

    def _build_scopes(self) -> None:
        """Build the lexical scope tree and each scope's set of bound names."""
        root = _Scope('', self._module, None, self._module.body)
        self._scopes[''] = root
        self._key_of[id(self._module)] = ''
        work: list[_Scope] = [root]
        used: set[str] = {''}
        while work:
            scope = work.pop()
            declared: set[str] = set()
            for sub in self._iter_stmts_skip_funcs(scope.node):
                if isinstance(sub, ast.Global):
                    scope.globals.update(sub.names)
                    declared.update(sub.names)
                elif isinstance(sub, ast.Nonlocal):
                    declared.update(sub.names)
            if isinstance(scope.node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                scope.locals.update(p.arg for p in self._params(scope.node))
            for sub in self._iter_stmts_skip_funcs(scope.node):
                if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                    scope.locals.add(sub.id)
                elif isinstance(sub, (ast.Import, ast.ImportFrom)):
                    for alias in sub.names:
                        bound = alias.asname or alias.name.split('.')[0]
                        scope.locals.add(bound)
            for fn in self._child_funcs(scope.node):
                scope.locals.add(fn.name)
                key = (scope.key + '.' if scope.key else '') + fn.name
                while key in used:
                    key += '~'
                used.add(key)
                child = _Scope(key, fn, scope.key, fn.body)
                self._scopes[key] = child
                self._key_of[id(fn)] = key
                scope.funcs[fn.name] = key
                work.append(child)
            scope.locals -= declared

    def _q(self, name: str, scope_key: str | None = None) -> str:
        """Qualify a bare name with the key of the scope that binds it."""
        key = self._scope if scope_key is None else scope_key
        while True:
            scope = self._scopes.get(key)
            if scope is None:
                return '\x00' + name
            if name in scope.globals:
                return '\x00' + name
            if name in scope.locals or scope.parent is None:
                return key + '\x00' + name
            key = scope.parent

    def _lookup_func(self, name: str) -> str | None:
        """Scope key of the user function ``name`` resolves to, if any."""
        key: str | None = self._scope
        while key is not None:
            scope = self._scopes.get(key)
            if scope is None:
                break
            if name in scope.funcs:
                return scope.funcs[name]
            key = scope.parent
        fn = self._funcs.get(name)
        return self._key_of.get(id(fn)) if fn is not None else None

    # --- alias classes ---

    def _find(self, name: str) -> str:
        root = name
        while self._parent.get(root, root) != root:
            root = self._parent[root]
        # Path compression
        while self._parent.get(name, name) != name:
            self._parent[name], name = root, self._parent[name]
        return root

    def _union(self, a: str, b: str) -> None:
        ra, rb = self._find(a), self._find(b)
        if ra == rb:
            return
        self._parent[rb] = ra
        merged = self._taint.get(ra, set()) | self._taint.get(rb, set())
        self._taint.pop(rb, None)
        if merged != self._taint.get(ra, set()):
            self._changed = True
        self._taint[ra] = merged

    @staticmethod
    def _bare(name: str) -> str:
        """The bare name of a scope-qualified name."""
        return name.rpartition('\x00')[2]

    def _get_taint(self, name: str) -> set[str]:
        if self._bare(name) in _UNTAINTABLE_ROOTS:
            return set()
        return set(self._taint.get(self._find(name), ()))

    def _add_taint(self, name: str, taint: set[str]) -> None:
        if self._bare(name) in _UNTAINTABLE_ROOTS:
            return
        if not taint:
            self._taint.setdefault(self._find(name), set())
            return
        root = self._find(name)
        cur = self._taint.setdefault(root, set())
        if not taint <= cur:
            cur |= taint
            self._changed = True

    # --- parameter tokens ---

    @staticmethod
    def _token(qualified_param: str) -> str:
        return _PARAM_PREFIX + qualified_param

    @classmethod
    def _ctrl_token(cls, scope_key: str) -> str:
        """Token standing for the control taint a call of ``scope_key`` runs under."""
        return cls._token(scope_key + '\x00' + _CTRL_PARAM)

    @staticmethod
    def _subst(taint: set[str], token_map: dict[str, set[str]]) -> set[str]:
        """Replace this call's parameter tokens with the argument taints.

        Tokens the map does not name belong to an enclosing function (or to a
        parameter this call left at its default) and are carried through
        unchanged — :meth:`_resolve_tokens` settles them later.
        """
        out: set[str] = set()
        for item in taint:
            if item.startswith(_PARAM_PREFIX):
                mapped = token_map.get(item)
                if mapped is None:
                    out.add(item)
                else:
                    out |= mapped
            else:
                out.add(item)
        return out

    def _resolve_tokens(self, taint: set[str]) -> set[str]:
        """Turn a taint set into plain sids by expanding parameter tokens.

        A token expands to the union of the argument taints seen at the
        callee's call sites. A function the module calls somewhere the analysis
        never reached has no recorded call site, so its tokens expand to every
        sid instead of to nothing — but only in the final rounds
        (``_final``): before the first fixpoint a callee visited ahead of its
        call site merely has not been bound YET, and expanding it to every sid
        then would pin a spurious dependency that no later round can retract.
        """
        out: set[str] = set()
        seen: set[str] = set()
        work = list(taint)
        while work:
            item = work.pop()
            if not item.startswith(_PARAM_PREFIX):
                out.add(item)
                continue
            if item in seen:
                continue
            seen.add(item)
            key = item[len(_PARAM_PREFIX):].rpartition('\x00')[0]
            scope = self._scopes.get(key)
            if (self._final and key not in self._bound and scope is not None
                    and isinstance(scope.node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and scope.node.name in self._called_names):
                out |= self._all
                continue
            work.extend(self._param_in.get(item, ()))
        return out

    # --- helpers ---

    @staticmethod
    def _sid_arg(call: ast.Call) -> str | None:
        """Return the sid string of a ``__sec_read__`` / ``__sec_write__`` call."""
        if not call.args:
            return None
        first = call.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            return first.value
        return None

    @staticmethod
    def _root_name(node: ast.expr) -> str | None:
        """Base Name of a place expression (``a``, ``a.b``, ``a[i].c``)."""
        while isinstance(node, (ast.Attribute, ast.Subscript)):
            node = node.value
        return node.id if isinstance(node, ast.Name) else None

    @staticmethod
    def _is_plain_record(node: ast.ClassDef, factories: FactoryFields) -> bool:
        """Whether a class body only declares fields — a Pine ``type``.

        Such a body runs once, when the module is imported, before any security
        value exists, so it carries no taint and needs no fallback. Creating an
        instance is an ordinary call and assigning a field an ordinary attribute
        store, and both are modelled where they happen. Anything else in the
        body — a method above all — keeps the class unmodelled.

        A field default the compiler lowered to a ``default_factory`` lambda
        runs at every construction instead, so it only qualifies while its body
        reads nothing a script can bind: the lowering's reserved-name helpers
        and ``lib``.

        :param node: the class definition
        :param factories: the module's compiler-emitted field factories
        :return: whether the body holds nothing but field declarations
        """
        factory_calls = {id(call) for call in factories.of(node)}
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign):
                lambdas = [sub for sub in ast.walk(stmt) if isinstance(sub, ast.Lambda)]
                if not lambdas:
                    continue
                value = stmt.value
                if (len(lambdas) != 1 or value is None or id(value) not in factory_calls
                        or any(isinstance(sub, ast.Name)
                               and PYNE_RESERVED_NAME_CHAR not in sub.id
                               and sub.id != 'lib'
                               for sub in ast.walk(lambdas[0].body))):
                    return False
                continue
            if isinstance(stmt, ast.Pass):
                continue
            if (isinstance(stmt, ast.Assign)
                    and all(isinstance(t, ast.Name) for t in stmt.targets)):
                continue
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                continue
            return False
        return True

    @staticmethod
    def _iter_stmts_skip_funcs(node: ast.AST):
        """All descendant nodes of ``node``, not entering nested function defs."""
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            yield child
            yield from _DependencyAnalyzer._iter_stmts_skip_funcs(child)

    @classmethod
    def _walk_loop_depth(cls, node: ast.AST, depth: int):
        """Yield ``(node, loop_depth)`` pairs, not entering nested function defs.

        The whole ``for`` / ``while`` subtree counts as one level deeper — the
        iterator expression itself runs once, but treating it as in-loop only
        over-approximates the ``in_loop`` marking.
        """
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            sub_depth = depth + 1 if isinstance(
                child, (ast.For, ast.AsyncFor, ast.While)
            ) else depth
            yield child, sub_depth
            yield from cls._walk_loop_depth(child, sub_depth)

    # --- program order of the write sites ---

    def _compute_site_order(self) -> None:
        """Number every sid by the program order of its ``__sec_write__`` site.

        The walk starts at the module body and at every function nothing calls
        (``main`` and friends), and descends into a user function AT ITS CALL
        SITE — so a write inside a nested or cloned function takes the position
        of the call that instantiates it. Recursion is cut with a visiting set.

        Sites the walk never reaches (dead code) are numbered last, in
        registration order.
        """
        counter = 0
        visiting: set[str] = set()

        def walk(node: ast.AST) -> None:
            nonlocal counter
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                # Arguments are evaluated before the call they belong to
                walk(child)
                if not isinstance(child, ast.Call) or not isinstance(child.func, ast.Name):
                    continue
                name = child.func.id
                if name == '__sec_write__':
                    sid = self._sid_arg(child)
                    if sid is not None and sid not in self._order:
                        self._order[sid] = counter
                        counter += 1
                elif name in self._funcs and name not in visiting:
                    visiting.add(name)
                    walk(self._funcs[name])
                    visiting.discard(name)

        walk(self._module)
        for name, fn in self._funcs.items():
            if name not in self._called_names:
                walk(fn)
        for sid in self._all:
            if sid not in self._order:
                self._order[sid] = counter
                counter += 1

    # --- in_loop ---

    def _mark_in_loop(self) -> None:
        """Mark sids whose ``__sec_write__`` runs more than once per bar.

        Two sources: a write block standing directly in a ``for`` / ``while``
        body (nesting included), and a write inside a user function that is
        reached — through any number of call levels — from a loop body. The
        instantiation pass clones a security-bearing function per CALL SITE,
        not per iteration, so a single clone called from a loop writes its sids
        several times on one bar.
        """
        own_sids: dict[str, set[str]] = {}
        callees: dict[str, set[str]] = {}
        loop_callees: set[str] = set()

        scopes: list[tuple[str, ast.AST]] = [('', self._module)]
        scopes.extend((name, fn) for name, fn in self._funcs.items())

        for scope, root in scopes:
            own_sids.setdefault(scope, set())
            callees.setdefault(scope, set())
            for node, depth in self._walk_loop_depth(root, 0):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                    continue
                fname = node.func.id
                if fname == '__sec_write__':
                    sid = self._sid_arg(node)
                    if sid is not None:
                        own_sids[scope].add(sid)
                        if depth > 0:
                            self.in_loop.add(sid)
                elif fname in self._funcs:
                    callees[scope].add(fname)
                    if depth > 0:
                        loop_callees.add(fname)

        reached: set[str] = set()
        work = list(loop_callees)
        while work:
            name = work.pop()
            if name in reached:
                continue
            reached.add(name)
            work.extend(callees.get(name, ()))

        for name in reached:
            self.in_loop |= own_sids.get(name, set())

    # --- expression taint ---

    def _expr_taint(self, node: ast.expr | None) -> set[str]:
        if node is None or isinstance(node, ast.Constant):
            return set()
        if isinstance(node, ast.Name):
            return self._get_taint(self._q(node.id))
        if isinstance(node, ast.Attribute):
            return self._expr_taint(node.value)
        if isinstance(node, ast.Call):
            return self._call_taint(node)
        if isinstance(node, ast.NamedExpr):
            # Walrus assignments are injected by the lowering passes that run
            # before this one; they behave exactly like a plain assignment.
            taint = self._expr_taint(node.value)
            self._assign(node.target, taint | self._ctrl, node.value)
            return taint
        if isinstance(node, _UNION_EXPRS):
            taint: set[str] = set()
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.expr):
                    taint |= self._expr_taint(child)
            return taint
        # Unmodelled expression form: the scope falls back instead of the
        # value carrying every sid around through name taint.
        self._mark_fallback(node)
        return set()

    def _call_taint(self, node: ast.Call) -> set[str]:
        func = node.func
        if isinstance(func, ast.Name):
            if func.id == '__sec_read__':
                sid = self._sid_arg(node)
                return {sid} if sid is not None else set(self._all)
            if func.id == '__sec_write__':
                sid = self._sid_arg(node)
                value = node.args[1] if len(node.args) > 1 else None
                taint = self._resolve_tokens(self._expr_taint(value) | self._ctrl)
                if self._scope in self._fallback_scopes:
                    taint |= self._all
                if sid is not None:
                    target = self.depends.setdefault(sid, set())
                    new = (taint - {sid}) - target
                    if new:
                        target |= new
                        self._changed = True
                return set()
            if func.id in ('__sec_signal__', '__sec_wait__'):
                return set()

        taint = self._expr_taint(func) if not isinstance(func, ast.Name) else set()
        pos_taints = [self._expr_taint(a) for a in node.args]
        kw_taints = [(kw.arg, self._expr_taint(kw.value)) for kw in node.keywords]
        # A ``*iterable`` spread has an unknown length, so from its position on
        # no argument has a known parameter of its own.
        star_from: int | None = None
        for i, arg in enumerate(node.args):
            if isinstance(arg, ast.Starred):
                star_from = i
                break
        callee = self._lookup_func(func.id) if isinstance(func, ast.Name) else None
        if callee is not None:
            taint |= self._call_user_func(callee, node, pos_taints, kw_taints, star_from)
        else:
            # Unknown callee: every argument is assumed to reach the result.
            for arg_taint in pos_taints:
                taint |= arg_taint
            for _kw_name, arg_taint in kw_taints:
                taint |= arg_taint

        # Mutation: a collection API writes into the id passed as its first
        # argument (``array.push(id, v)``), and a method call writes into the
        # object it runs on (``id.push(v)``). Whether the write happens at all
        # depends on the enclosing conditions, so the control taint is part of
        # the collection's new state even when the pushed value is a constant.
        mutation = taint | self._ctrl
        if mutation and isinstance(func, ast.Attribute):
            if self._is_mutating_namespace_call(func):
                if node.args:
                    root = self._root_name(node.args[0])
                    if root is not None:
                        self._add_taint(self._q(root), mutation)
            else:
                root = self._root_name(func.value)
                if root is not None:
                    self._add_taint(self._q(root), mutation)
        return taint

    def _call_user_func(self, key: str, node: ast.Call,
                        pos_taints: list[set[str]],
                        kw_taints: list[tuple[str | None, set[str]]],
                        star_from: int | None) -> set[str]:
        """Taint of one call of a modelled user function.

        The call's arguments are matched to the callee's parameters, giving a
        substitution from the callee's parameter tokens to this call's argument
        taints. The result is the callee's return taint under that
        substitution — an argument the return does not draw on contributes
        nothing. The same substitution carries taint the callee added to a
        parameter (a mutated collection) back into the caller's variable.

        :param key: scope key of the callee
        :param node: the call site
        :param pos_taints: taint of each positional argument, in call order
        :param kw_taints: ``(keyword name, taint)`` pairs; the name is ``None``
            for a ``**mapping`` unpacking, whose targets are unknown
        :param star_from: index of the first ``*iterable`` in ``pos_taints``,
            ``None`` when the call spreads nothing
        :return: the taint of the call's result
        """
        fn = self._scopes[key].node
        assert isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
        arg_expr = self._arg_exprs(fn, node, star_from)
        bound: dict[str, set[str]] = {}
        # Taint whose target parameter is not known at compile time
        unknown: set[str] = set()
        positional = list(fn.args.posonlyargs) + list(fn.args.args)
        for i, arg_taint in enumerate(pos_taints):
            if star_from is not None and i >= star_from:
                unknown |= arg_taint
            elif i < len(positional):
                bound.setdefault(positional[i].arg, set()).update(arg_taint)
            else:
                unknown |= arg_taint
        by_name = {a.arg for a in positional + list(fn.args.kwonlyargs)}
        for kw_name, arg_taint in kw_taints:
            if kw_name is not None and kw_name in by_name:
                bound.setdefault(kw_name, set()).update(arg_taint)
            else:
                unknown |= arg_taint
        # A spread or a leftover argument can fill any parameter, so it binds
        # all of them (the catch-alls included).
        if unknown:
            for param in self._params(fn):
                bound.setdefault(param.arg, set()).update(unknown)

        token_map = {
            self._token(key + '\x00' + pname): arg_taint
            for pname, arg_taint in bound.items()
        }
        token_map[self._ctrl_token(key)] = set(self._ctrl)
        self._bound.add(key)
        for token, arg_taint in token_map.items():
            cur = self._param_in.setdefault(token, set())
            if not arg_taint <= cur:
                cur |= arg_taint
                self._changed = True

        result = self._subst(self._ret.get(key, set()), token_map)
        if key in self._fallback_scopes:
            # An unmodelled construct in the callee: its return can carry
            # anything, and so can what it writes into its arguments.
            result |= self._all
            for expr in arg_expr.values():
                root = self._root_name(expr)
                if root is not None:
                    self._add_taint(self._q(root), set(self._all))
            return result

        outs = self._param_out.get(key, {})
        for pname, expr in arg_expr.items():
            back = self._subst(outs.get(self._token(key + '\x00' + pname), set()), token_map)
            if not back:
                continue
            root = self._root_name(expr)
            if root is not None:
                self._add_taint(self._q(root), back)
        return result

    @classmethod
    def _arg_exprs(cls, fn: 'ast.FunctionDef | ast.AsyncFunctionDef', node: ast.Call,
                   star_from: int | None) -> dict[str, ast.expr]:
        """Map each parameter of ``fn`` to the argument expression ``node`` passes
        for it, for the arguments whose parameter is known at compile time."""
        positional = list(fn.args.posonlyargs) + list(fn.args.args)
        by_name = {a.arg for a in positional + list(fn.args.kwonlyargs)}
        out: dict[str, ast.expr] = {}
        for i, arg in enumerate(node.args):
            if star_from is not None and i >= star_from:
                break
            if i < len(positional):
                out[positional[i].arg] = arg
        for keyword in node.keywords:
            if keyword.arg is not None and keyword.arg in by_name:
                out[keyword.arg] = keyword.value
        return out

    def _alias_roots(self, expr: ast.expr) -> list[str]:
        """Root names of the collections ``expr`` may evaluate to an alias of:
        the root of a name/attribute/subscript chain, or — for a user-function
        call — the roots of the arguments its summary says it returns. Calls
        compose: a call-valued argument contributes its own alias roots, so
        ``f(g(store))`` reaches ``store`` when both return their argument."""
        if isinstance(expr, (ast.Name, ast.Attribute, ast.Subscript)):
            root = self._root_name(expr)
            return [] if root is None or root == 'lib' else [root]
        if not isinstance(expr, ast.Call) or not isinstance(expr.func, ast.Name):
            return []
        key = self._lookup_func(expr.func.id)
        if key is None:
            return []
        params = self._ret_alias.get(key)
        if not params:
            return []
        fn = self._scopes[key].node
        assert isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
        star_from = next(
            (i for i, a in enumerate(expr.args) if isinstance(a, ast.Starred)), None
        )
        exprs = self._arg_exprs(fn, expr, star_from)
        roots: list[str] = []
        for pname in params:
            arg = exprs.get(pname)
            if arg is not None:
                roots.extend(self._alias_roots(arg))
        return roots

    def _record_ret_alias(self, value: ast.expr | None) -> None:
        """Note which parameters the returned value shares an alias class with."""
        scope = self._scopes[self._scope]
        fn = scope.node
        if value is None or not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return
        aliased = self._ret_alias.setdefault(scope.key, set())
        for root in self._alias_roots(value):
            cls_root = self._find(self._q(root))
            for param in self._params(fn):
                if param.arg in aliased:
                    continue
                if self._find(scope.key + '\x00' + param.arg) == cls_root:
                    aliased.add(param.arg)
                    self._changed = True

    @staticmethod
    def _is_mutating_namespace_call(func: ast.Attribute) -> bool:
        """Whether ``func`` is ``lib.<array|matrix|map>.<fn>``."""
        owner = func.value
        return (isinstance(owner, ast.Attribute)
                and owner.attr in _MUTATING_NAMESPACES
                and isinstance(owner.value, ast.Name)
                and owner.value.id == 'lib')

    # --- assignment ---

    def _assign(self, target: ast.expr, taint: set[str], value: ast.expr | None) -> None:
        if self._scope in self._fallback_scopes:
            # The assigned value may come from an unmodelled construct whose
            # taint the analysis could not see.
            taint = taint | self._all
        if isinstance(target, ast.Name):
            self._add_taint(self._q(target.id), taint)
            if value is not None:
                for root in self._alias_roots(value):
                    self._union(self._q(target.id), self._q(root))
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._assign(elt, taint, None)
            return
        if isinstance(target, ast.Starred):
            self._assign(target.value, taint, None)
            return
        root = self._root_name(target)
        if root is not None:
            self._add_taint(self._q(root), taint)

    # --- statements ---

    def _visit_body(self, body: list[ast.stmt], ctrl: set[str]) -> set[str]:
        """Visit a statement list; return the control taint after it."""
        for stmt in body:
            ctrl = self._visit_stmt(stmt, ctrl)
        return ctrl

    def _visit_stmt(self, stmt: ast.stmt, ctrl: set[str]) -> set[str]:
        self._ctrl = ctrl
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return ctrl
        if isinstance(stmt, ast.Assign):
            taint = self._expr_taint(stmt.value) | ctrl
            for target in stmt.targets:
                self._assign(target, taint, stmt.value)
            return ctrl
        if isinstance(stmt, ast.AnnAssign):
            if stmt.value is not None:
                self._assign(stmt.target, self._expr_taint(stmt.value) | ctrl, stmt.value)
            return ctrl
        if isinstance(stmt, ast.AugAssign):
            taint = self._expr_taint(stmt.value) | self._expr_taint(stmt.target) | ctrl
            self._assign(stmt.target, taint, None)
            return ctrl
        if isinstance(stmt, ast.Expr):
            self._expr_taint(stmt.value)
            return ctrl
        if isinstance(stmt, ast.If):
            inner = ctrl | self._expr_taint(stmt.test)
            self._visit_body(stmt.body, inner)
            self._visit_body(stmt.orelse, inner)
            return ctrl
        if isinstance(stmt, (ast.For, ast.AsyncFor)):
            inner = ctrl | self._expr_taint(stmt.iter)
            self._assign(stmt.target, inner, None)
            self._visit_body(stmt.body, inner)
            self._visit_body(stmt.orelse, inner)
            return ctrl
        if isinstance(stmt, ast.While):
            inner = ctrl | self._expr_taint(stmt.test)
            self._visit_body(stmt.body, inner)
            self._visit_body(stmt.orelse, inner)
            return ctrl
        if isinstance(stmt, ast.Match):
            inner = ctrl | self._expr_taint(stmt.subject)
            for case in stmt.cases:
                case_ctrl = inner
                if case.guard is not None:
                    case_ctrl = inner | self._expr_taint(case.guard)
                self._visit_body(case.body, case_ctrl)
            return ctrl
        if isinstance(stmt, ast.Return):
            taint = self._expr_taint(stmt.value) | ctrl
            self._record_escape(self._ret, taint)
            self._record_ret_alias(stmt.value)
            self._record_escape(self._escape, ctrl)
            return ctrl | taint
        if isinstance(stmt, (ast.Break, ast.Continue)):
            # The decision to leave the block is itself tainted, so everything
            # after it runs conditionally on that taint.
            self._record_escape(self._escape, ctrl)
            return ctrl
        return ctrl

    def _mark_fallback(self, node: ast.AST | None = None) -> None:
        """Mark the current scope as containing an unmodelled construct.

        :param node: the construct that triggered the fallback, kept for
            diagnostics
        """
        if self._scope not in self._fallback_scopes:
            self._fallback_scopes.add(self._scope)
            self.fallback_reasons.setdefault(
                self._scope, type(node).__name__ if node is not None else '?'
            )
            self._changed = True

    def _record_escape(self, store: dict[str, set[str]], taint: set[str]) -> None:
        if not taint:
            store.setdefault(self._scope, set())
            return
        cur = store.setdefault(self._scope, set())
        if not taint <= cur:
            cur |= taint
            self._changed = True

    # --- per-scope pre/post steps ---

    def _seed_params(self) -> None:
        """Give every parameter its own symbolic token as a base taint."""
        for key, scope in self._scopes.items():
            if not isinstance(scope.node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for param in self._params(scope.node):
                qualified = key + '\x00' + param.arg
                self._add_taint(qualified, {self._token(qualified)})

    def _visit_defaults(self, scope: _Scope) -> None:
        """Fold parameter default expressions into the parameters' taint.

        Defaults are evaluated in the ENCLOSING scope, and a call that omits
        the argument leaves the default in place, so the taint belongs to the
        parameter unconditionally.
        """
        fn = scope.node
        assert isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
        spec = fn.args
        outer, self._scope = self._scope, scope.parent or ''
        ctrl, self._ctrl = self._ctrl, set()
        positional = list(spec.posonlyargs) + list(spec.args)
        pairs: list[tuple[ast.arg, ast.expr | None]] = list(
            zip(positional[len(positional) - len(spec.defaults):], spec.defaults)
        )
        pairs += list(zip(spec.kwonlyargs, spec.kw_defaults))
        taints = [
            (param, self._expr_taint(default))
            for param, default in pairs if default is not None
        ]
        self._scope, self._ctrl = outer, ctrl
        for param, taint in taints:
            self._add_taint(scope.key + '\x00' + param.arg, taint)

    def _record_param_out(self, scope: _Scope) -> None:
        """Snapshot every parameter's taint at the end of the callee's body."""
        fn = scope.node
        assert isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
        outs = self._param_out.setdefault(scope.key, {})
        for param in self._params(fn):
            qualified = scope.key + '\x00' + param.arg
            taint = self._get_taint(qualified)
            cur = outs.setdefault(self._token(qualified), set())
            if not taint <= cur:
                cur |= taint
                self._changed = True

    # --- driver ---

    def run(self) -> None:
        self._funcs = {
            n.name: n for n in ast.walk(self._module)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self._called_names = {
            n.func.id for n in ast.walk(self._module)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id in self._funcs
        }
        self._build_scopes()
        self._mark_in_loop()
        self._compute_site_order()

        factories = FactoryFields(self._module)
        record_nodes: set[int] = set()
        # Only a module-level class body runs at import time; one defined inside
        # a function runs with the function and stays unmodelled.
        for node in self._module.body:
            if isinstance(node, ast.ClassDef) and self._is_plain_record(node, factories):
                record_nodes.update(id(sub) for sub in ast.walk(node))
        for key, scope in self._scopes.items():
            for sub in self._iter_stmts_skip_funcs(scope.node):
                if id(sub) in record_nodes:
                    continue
                if isinstance(sub, _UNMODELLED_NODES):
                    self._fallback_scopes.add(key)
                    self.fallback_reasons.setdefault(key, type(sub).__name__)
                    break

        self._seed_params()
        ordered = list(self._scopes.values())
        for _ in range(_MAX_FIXPOINT_ROUNDS):
            self._changed = False
            for scope in ordered:
                self._scope = scope.key
                ctrl = set(self._escape.get(scope.key, set()))
                if scope.parent is not None:
                    self._visit_defaults(scope)
                    fn = scope.node
                    assert isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
                    if fn.name in self._called_names:
                        # Everything in a called function runs under the
                        # conditions of its call sites (see _CTRL_PARAM).
                        ctrl.add(self._ctrl_token(scope.key))
                self._visit_body(scope.body, ctrl)
                if scope.parent is not None:
                    self._record_param_out(scope)
            if self._changed:
                continue
            if self._final:
                break
            # Converged with every reachable call site bound: from here a
            # token still without a call site belongs to an unreached call.
            self._final = True

        # Only a producer whose site PRECEDES this write can be waited for.
        # A later-sited producer would deadlock the historical warmup batch:
        # the child replays many bars in one round, so blocking on a peer whose
        # site the chart has not reached yet stops the very write that would
        # release the round. Such an edge can only be a previous-bar carry
        # anyway; the child reads the default for it.
        for sid, deps in self.depends.items():
            own = self._order.get(sid, 0)
            self.depends[sid] = {
                d for d in deps
                if d != sid and d in self._all and self._order.get(d, own) < own
            }

class _CallReplacer(ast.NodeTransformer):
    """Replace marked request.security() call nodes with __sec_read__() calls."""

    def __init__(self, parent: SecurityTransformer):
        self._parent = parent

    # noinspection PyProtectedMember
    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = self.generic_visit(node)  # type: ignore[assignment]
        if hasattr(node, '_sec_id'):
            sec_id = getattr(node, '_sec_id')
            if sec_id in self._parent._ltf_sec_ids:
                ltf_arity = getattr(node, '_ltf_arity', None)
                return self._parent._sec_read_call_ltf(sec_id, ltf_arity)
            tuple_len = getattr(node, '_tuple_len', None)
            return self._parent._sec_read_call(sec_id, tuple_len)
        return node
