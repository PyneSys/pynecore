import ast
import copy
import hashlib
from collections.abc import Container


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
        by a hoistable binding (see :meth:`_hoistable_bindings`). Such an
        expression can be evaluated at the very start of the function, which is
        what lets its ``__sec_signal__`` move into the top block.

        :param node: expression to classify
        :param hoistable: names bound by hoistable top-level bindings
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
    ) -> dict[str, ast.Assign]:
        """Top-level bindings of ``func`` that may be moved to its first
        statements.

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
        :return: mapping of binding name to its assignment statement
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

        hoistable: dict[str, ast.Assign] = {}
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
                            and cls._is_simple_chain(stmt.value, hoistable)):
                        hoistable[name] = stmt
                        ok = True
            prefix = prefix and ok
        return hoistable

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
        # constants, lib.* values, or a local bound once from such a chain. In
        # the last case the binding itself is hoisted above the signal block, so
        # the value is already there. Everything else (function parameters,
        # series-dependent expressions) must be signalled inline, after the
        # variables it reads have been assigned.
        hoistable = self._hoistable_bindings(node)
        top_sec_ids = []
        runtime_sec_ids: set[str] = set()
        needed_names: set[str] = set()
        for sid in sec_ids:
            exprs = [e for e in self._signal_args[sid] if e is not None]
            if all(self._is_simple_chain(e, hoistable) for e in exprs):
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

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        return self._process_func(node)  # type: ignore[return-value]

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        return self._process_func(node)  # type: ignore[return-value]

    def visit_Module(self, node: ast.Module) -> ast.Module:
        self._module_file = getattr(node, '_module_file_path', '<script>')
        node = self.generic_visit(node)  # type: ignore[assignment]

        if self._all_contexts:
            self._analyze_dependencies(node)

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
            'Global', 'Nonlocal', 'ClassDef', 'Lambda', 'ListComp', 'SetComp',
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


class _DependencyAnalyzer:
    """Forward taint analysis over an already lowered security AST.

    After :class:`SecurityTransformer` has rewritten the module, every
    ``request.security()`` result is read through ``__sec_read__("<sid>", ...)``
    and every security expression is written through
    ``__sec_write__("<sid>", <expr>)``. This pass answers one question: which
    sids' results flow into which other sid's write expression.

    The analysis is deliberately coarse:

    - **Flow-insensitive on data**: a name's taint is the union over every
      assignment to it anywhere in the module (so ``var`` backward flow and
      loop-carried values are covered without ordering rules).
    - **Flat namespace**: names are not scoped per function. Two unrelated
      locals sharing a name merge their taint — an over-approximation only.
    - **Flow-sensitive on control**: a control-taint set accumulates the taint
      of enclosing ``if`` / ``for`` / ``while`` / ``match`` conditions, and a
      tainted ``return`` / ``break`` / ``continue`` taints the rest of its
      function.
    - **Alias classes**: ``a = b`` merges the two names into one taint class,
      so mutating a container through either alias is visible on both.
    - **User functions**: a call's result carries the callee's return taint
      plus the taint of every argument (arguments are conservatively assumed
      to flow into the result).
    - **Fallback**: a scope containing an unmodelled construct makes every
      write in it depend on all other sids.

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
        self._fallback_scopes: set[str] = set()
        self.fallback_reasons: dict[str, str] = {}
        self._ret: dict[str, set[str]] = {}
        self._escape: dict[str, set[str]] = {}
        self._parent: dict[str, str] = {}
        self._taint: dict[str, set[str]] = {}
        self._ctrl: set[str] = set()
        self._order: dict[str, int] = {}
        self._scope: str = ''
        self._changed = False

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

    def _get_taint(self, name: str) -> set[str]:
        if name in _UNTAINTABLE_ROOTS:
            return set()
        return set(self._taint.get(self._find(name), ()))

    def _add_taint(self, name: str, taint: set[str]) -> None:
        if name in _UNTAINTABLE_ROOTS:
            return
        if not taint:
            self._taint.setdefault(self._find(name), set())
            return
        root = self._find(name)
        cur = self._taint.setdefault(root, set())
        if not taint <= cur:
            cur |= taint
            self._changed = True

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

        called = {
            n.func.id for n in ast.walk(self._module)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id in self._funcs
        }
        walk(self._module)
        for name, fn in self._funcs.items():
            if name not in called:
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
            return self._get_taint(node.id)
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
                taint = self._expr_taint(value) | self._ctrl
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
        for arg_taint in pos_taints:
            taint |= arg_taint
        for _kw_name, arg_taint in kw_taints:
            taint |= arg_taint
        if isinstance(func, ast.Name) and func.id in self._funcs:
            self._bind_params(self._funcs[func.id], pos_taints, kw_taints, star_from)
            taint |= self._ret.get(func.id, set())

        # Mutation: a collection API writes into the id passed as its first
        # argument (``array.push(id, v)``), and a method call writes into the
        # object it runs on (``id.push(v)``).
        if taint and isinstance(func, ast.Attribute):
            if self._is_mutating_namespace_call(func):
                if node.args:
                    root = self._root_name(node.args[0])
                    if root is not None:
                        self._add_taint(root, taint)
            else:
                root = self._root_name(func.value)
                if root is not None:
                    self._add_taint(root, taint)
        return taint

    def _bind_params(self, fn: 'ast.FunctionDef | ast.AsyncFunctionDef',
                     pos_taints: list[set[str]],
                     kw_taints: list[tuple[str | None, set[str]]],
                     star_from: int | None = None) -> None:
        """Flow a call's argument taints into the callee's parameter names.

        The taint namespace is flat (one entry per bare name), so a parameter
        is just another name — but nothing writes it unless the argument taint
        is bound here. Without the binding a dependency only reaches a write
        inside the callee when the caller's variable happens to carry the same
        name as the parameter.

        :param fn: The callee's definition.
        :param pos_taints: Taint of each positional argument, in call order.
        :param kw_taints: ``(keyword name, taint)`` pairs; the name is ``None``
            for a ``**mapping`` unpacking, whose targets are unknown.
        :param star_from: Index of the first ``*iterable`` in ``pos_taints``,
            ``None`` when the call spreads nothing. From that index on the
            argument-to-parameter mapping is unknown at compile time.
        """
        spec = fn.args
        positional = list(spec.posonlyargs) + list(spec.args)
        every = positional + list(spec.kwonlyargs)
        extra_pos: set[str] = set()
        extra_kw: set[str] = set()
        spread: set[str] = set()
        for i, arg_taint in enumerate(pos_taints):
            if star_from is not None and i >= star_from:
                spread |= arg_taint
            elif i < len(positional):
                self._add_taint(positional[i].arg, arg_taint)
            else:
                extra_pos |= arg_taint
        by_name = {a.arg for a in every}
        unknown_kw: set[str] = set()
        for name, arg_taint in kw_taints:
            if name is None:
                unknown_kw |= arg_taint
            elif name in by_name:
                self._add_taint(name, arg_taint)
            else:
                extra_kw |= arg_taint
        # A spread of unknown length can fill every parameter from its own
        # position on, so its taint goes to all of them (and to ``*args``).
        if spread:
            for param in positional[star_from:]:
                self._add_taint(param.arg, spread)
            if spec.vararg is not None:
                self._add_taint(spec.vararg.arg, spread)
        # A ``**mapping`` of unknown keys can fill any keyword-addressable
        # parameter — positional-only ones are the sole exception.
        if unknown_kw:
            for param in list(spec.args) + list(spec.kwonlyargs):
                self._add_taint(param.arg, unknown_kw)
            if spec.kwarg is not None:
                self._add_taint(spec.kwarg.arg, unknown_kw)
        # An argument with no parameter of its own (``*args`` / ``**kwargs``,
        # or a starred call the caller spread) lands in the catch-all if there
        # is one, and over-approximates onto every parameter if there is not.
        if extra_pos:
            if spec.vararg is not None:
                self._add_taint(spec.vararg.arg, extra_pos)
            else:
                for param in every:
                    self._add_taint(param.arg, extra_pos)
        if extra_kw:
            if spec.kwarg is not None:
                self._add_taint(spec.kwarg.arg, extra_kw)
            else:
                for param in every:
                    self._add_taint(param.arg, extra_kw)

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
        if isinstance(target, ast.Name):
            self._add_taint(target.id, taint)
            if value is not None and isinstance(value, (ast.Name, ast.Attribute, ast.Subscript)):
                root = self._root_name(value)
                if root is not None and root != 'lib':
                    self._union(target.id, root)
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
            self._add_taint(root, taint)

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

    # --- driver ---

    def run(self) -> None:
        self._funcs = {
            n.name: n for n in ast.walk(self._module)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self._mark_in_loop()
        self._compute_site_order()

        scopes: list[tuple[str, list[ast.stmt]]] = [('', self._module.body)]
        scopes.extend((name, fn.body) for name, fn in self._funcs.items())
        roots: list[tuple[str, ast.AST]] = [('', self._module)]
        roots.extend((name, fn) for name, fn in self._funcs.items())
        for scope, root in roots:
            for sub in self._iter_stmts_skip_funcs(root):
                if isinstance(sub, _UNMODELLED_NODES):
                    self._fallback_scopes.add(scope)
                    self.fallback_reasons.setdefault(scope, type(sub).__name__)
                    break

        for _ in range(_MAX_FIXPOINT_ROUNDS):
            self._changed = False
            for scope, body in scopes:
                self._scope = scope
                self._visit_body(body, set(self._escape.get(scope, set())))
            if not self._changed:
                break

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
                if d != sid and self._order.get(d, own) < own
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
