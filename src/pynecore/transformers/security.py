import ast
import copy


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
    Non-constant symbol/timeframe values (e.g., function parameters) are stored as None
    in __security_contexts__ and resolved at runtime via __sec_signal__ arguments.

    Must be applied after ImportNormalizerTransformer, before PersistentSeriesTransformer.
    """

    def __init__(self):
        self._counter = 0
        self._all_contexts: dict[str, dict[str, ast.expr]] = {}
        self._signal_args: dict[str, tuple[ast.expr | None, ast.expr | None]] = {}
        self._needs_barmerge = False
        self._ltf_sec_ids: set[str] = set()

    def _gen_id(self) -> str:
        sec_id = f"sec\xb7{self._counter}"
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
        ast.expr | None, ast.expr | None
    ]:
        """Extract (symbol, timeframe, expression, gaps, ignore_invalid_symbol, currency)
        from request.security() call."""
        args = list(call.args)
        kwargs = {kw.arg: kw.value for kw in call.keywords if kw.arg is not None}
        return (
            kwargs.get('symbol', args[0] if len(args) > 0 else None),
            kwargs.get('timeframe', args[1] if len(args) > 1 else None),
            kwargs.get('expression', args[2] if len(args) > 2 else None),
            kwargs.get('gaps', args[3] if len(args) > 3 else None),
            kwargs.get('ignore_invalid_symbol', args[4] if len(args) > 4 else None),
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
            return SecurityTransformer._is_module_level_expr(node.func)
        return False

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

    def _is_none_check(self) -> ast.Compare:
        """Build: __active_security__ is None"""
        return ast.Compare(
            left=ast.Name(id='__active_security__', ctx=ast.Load()),
            ops=[ast.Is()], comparators=[ast.Constant(value=None)]
        )

    def _eq_check(self, sec_id: str) -> ast.Compare:
        """Build: __active_security__ == sec_id"""
        return ast.Compare(
            left=ast.Name(id='__active_security__', ctx=ast.Load()),
            ops=[ast.Eq()], comparators=[ast.Constant(value=sec_id)]
        )

    @staticmethod
    def _scope_id_ref() -> ast.Name:
        """Build: __scope_id__"""
        return ast.Name(id='__scope_id__', ctx=ast.Load())

    def _signal_block(self, sec_ids: list[str]) -> ast.If:
        """Build chart-context signal block for function start.

        Passes actual symbol and timeframe expressions to __sec_signal__
        so that runtime values (e.g., function parameters) are available.
        Also passes __scope_id__ for call_id-based context disambiguation.

        Only includes sec_ids whose symbol/timeframe are module-level
        expressions (constants or lib.* refs). Runtime-dependent signals
        are emitted inline before their write blocks by _transform_body.
        """
        body = []
        for s in sec_ids:
            args: list[ast.expr] = [ast.Constant(value=s)]
            sym_expr, tf_expr = self._signal_args[s]
            args.append(copy.deepcopy(sym_expr) if sym_expr is not None
                        else ast.Constant(value=None))
            args.append(copy.deepcopy(tf_expr) if tf_expr is not None
                        else ast.Constant(value=None))
            args.append(self._scope_id_ref())
            body.append(ast.Expr(value=self._func_call('__sec_signal__', *args)))
        return ast.If(
            test=self._is_none_check(),
            body=body,
            orelse=[]
        )

    def _inline_signal(self, sec_id: str) -> ast.If:
        """Build a single inline signal for runtime-dependent symbol/timeframe."""
        args: list[ast.expr] = [ast.Constant(value=sec_id)]
        sym_expr, tf_expr = self._signal_args[sec_id]
        args.append(copy.deepcopy(sym_expr) if sym_expr is not None
                    else ast.Constant(value=None))
        args.append(copy.deepcopy(tf_expr) if tf_expr is not None
                    else ast.Constant(value=None))
        args.append(self._scope_id_ref())
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
                ast.Expr(value=self._func_call(
                    '__sec_wait__', ast.Constant(value=s), self._scope_id_ref()
                ))
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
                    '__sec_write__', ast.Constant(value=sec_id), expression,
                    self._scope_id_ref()
                ))
            ],
            orelse=[]
        )

    def _sec_read_call(self, sec_id: str) -> ast.Call:
        """Build: __sec_read__("sec_id", lib.na, __scope_id__)"""
        return self._func_call(
            '__sec_read__', ast.Constant(value=sec_id), self._lib_na(),
            self._scope_id_ref()
        )

    def _sec_read_call_ltf(self, sec_id: str) -> ast.Call:
        """Build: __sec_read__("sec_id", [], __scope_id__)"""
        return self._func_call(
            '__sec_read__', ast.Constant(value=sec_id), ast.List(elts=[], ctx=ast.Load()),
            self._scope_id_ref()
        )

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
        runtime_sec_ids: set[str] | None = None
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
        if runtime_sec_ids is None:
            runtime_sec_ids = set()
        new_body: list[ast.stmt] = []
        replacer = _CallReplacer(self)

        for stmt in body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                new_body.append(stmt)
                continue

            self._recurse_subbodies(stmt, call_exprs, runtime_sec_ids)

            sec_ids_here = [
                n._sec_id  # type: ignore[attr-defined]
                for n in self._walk_skip_funcs(stmt)
                if isinstance(n, ast.Call) and hasattr(n, '_sec_id')
            ]

            if sec_ids_here:
                for sid in sec_ids_here:
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
            if is_ltf:
                symbol, timeframe, expression, ignore_invalid = (
                    self._extract_ltf_args(call)
                )
                gaps = None
            else:
                symbol, timeframe, expression, gaps, ignore_invalid, currency = (
                    self._extract_args(call)
                )

            call_exprs[sec_id] = expression if expression is not None else self._lib_na()

            # Store actual expressions for __sec_signal__ args (always passed at runtime)
            self._signal_args[sec_id] = (
                copy.deepcopy(symbol) if symbol is not None else None,
                copy.deepcopy(timeframe) if timeframe is not None else None,
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

            if ignore_invalid is not None:
                ctx['ignore_invalid_symbol'] = copy.deepcopy(ignore_invalid)

            if not is_ltf and currency is not None:
                ctx['currency'] = copy.deepcopy(currency)

            # Track if barmerge is used (only for non-LTF)
            if not is_ltf:
                if gaps is None:
                    self._needs_barmerge = True
                elif isinstance(gaps, ast.Attribute) and hasattr(gaps, 'value'):
                    v = gaps.value
                    if isinstance(v, ast.Attribute) and v.attr == 'barmerge':
                        self._needs_barmerge = True

            self._all_contexts[sec_id] = ctx
            sec_ids.append(sec_id)

        # Separate module-level (constant) signals from runtime-dependent ones.
        # Module-level signals can be emitted at function start for maximum
        # parallelism. Runtime signals must be emitted inline, after the
        # variables they reference have been assigned.
        top_sec_ids = []
        runtime_sec_ids: set[str] = set()
        for sid in sec_ids:
            sym_expr, tf_expr = self._signal_args[sid]
            sym_ok = sym_expr is None or self._is_module_level_expr(sym_expr)
            tf_ok = tf_expr is None or self._is_module_level_expr(tf_expr)
            if sym_ok and tf_ok:
                top_sec_ids.append(sid)
            else:
                runtime_sec_ids.add(sid)

        original_body = node.body
        top_block = [self._signal_block(top_sec_ids)] if top_sec_ids else []
        node.body = (
            top_block
            + self._transform_body(original_body, call_exprs, runtime_sec_ids)
            + [self._wait_block(sec_ids)]
        )

        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        return self._process_func(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        return self._process_func(node)

    def visit_Module(self, node: ast.Module) -> ast.Module:
        node = self.generic_visit(node)

        if self._all_contexts:
            # Add barmerge import if needed (SecurityTransformer runs AFTER ImportNormalizer,
            # so we must add it ourselves)
            if self._needs_barmerge:
                node.body.insert(0, ast.Import(
                    names=[ast.alias(name='pynecore.lib.barmerge', asname=None)]
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


class _CallReplacer(ast.NodeTransformer):
    """Replace marked request.security() call nodes with __sec_read__() calls."""

    def __init__(self, parent: SecurityTransformer):
        self._parent = parent

    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = self.generic_visit(node)
        if hasattr(node, '_sec_id'):
            sec_id = node._sec_id  # type: ignore[attr-defined]
            if sec_id in self._parent._ltf_sec_ids:
                return self._parent._sec_read_call_ltf(sec_id)
            return self._parent._sec_read_call(sec_id)
        return node
