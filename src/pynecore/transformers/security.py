import ast
import hashlib
import re
from collections.abc import Container

from . import ast_walk
from .lib_series_attrs import NON_SERIES_LIB_ATTRS
from .script_entry import is_script_entry
from .security_calls import has_security_call, is_security_call, is_security_lower_tf_call
from .security_dependencies import _DependencyAnalyzer


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

#: Name shape of the temps ``InlineSeriesHoistTransformer`` lifts an
#: ``inline_series`` call into (see :meth:`SecurityTransformer._scope_bindings`).
_HIST_TEMP_RE = re.compile(r'__hist_\d+__')

#: Annotations of an ``AnnAssign`` that DECLARE a history-bearing variable.
#: ``PersistentSeries`` / ``IBPersistentSeries`` are split into a persistent
#: half and a plain ``Series`` half by
#: :class:`~pynecore.transformers.persistent_series.PersistentSeriesTransformer`
#: (stage 501), so all three end up as a series. Bare ``Persistent`` /
#: ``IBPersistent`` are NOT here: they only say the value survives the bar.
_SERIES_ANNOTATIONS = frozenset({'Series', 'PersistentSeries', 'IBPersistentSeries'})


def _is_series_annotation(node: ast.expr | None) -> bool:
    """Whether an ``AnnAssign`` annotation declares a series variable.

    :param node: The annotation node, or None.
    :return: True for ``Series`` / ``PersistentSeries`` / ``IBPersistentSeries``,
        bare or subscripted.
    """
    if isinstance(node, ast.Subscript):
        node = node.value
    return isinstance(node, ast.Name) and node.id in _SERIES_ANNOTATIONS


def _is_param_series_annotation(node: ast.expr | None) -> bool:
    """Whether a PARAMETER annotation gives the parameter a history buffer.

    Only a plain ``Series`` / ``Series[T]``:
    :meth:`~pynecore.transformers.series.SeriesTransformer.visit_FunctionDef`
    converts exactly that, and ``PersistentSeriesTransformer`` rewrites
    declarations only, never a parameter's annotation.

    :param node: The annotation node, or None.
    :return: True for ``Series`` / ``Series[T]``.
    """
    if isinstance(node, ast.Subscript):
        node = node.value
    return isinstance(node, ast.Name) and node.id == 'Series'


def _is_history_index(node: ast.expr) -> bool:
    """Whether ``node`` is an int constant >= 1 (a history offset)."""
    return (isinstance(node, ast.Constant) and isinstance(node.value, int)
            and not isinstance(node.value, bool) and node.value >= 1)


def _is_inline_series_call(node: ast.expr | None) -> bool:
    """Whether ``node`` is the ``inline_series(expr, k)`` call PyneComp emits in
    place of an expression history reference, with ``k`` an offset >= 1.

    :param node: Expression node, or None.
    :return: True for such a call.
    """
    if not (isinstance(node, ast.Call) and len(node.args) == 2 and not node.keywords):
        return False
    func = node.func
    name = (func.id if isinstance(func, ast.Name)
            else func.attr if isinstance(func, ast.Attribute) else None)
    return name == 'inline_series' and _is_history_index(node.args[1])


def _is_lib_series_chain(node: ast.expr) -> bool:
    """Whether ``node`` is a subscript receiver
    :class:`~pynecore.transformers.lib_series.LibrarySeriesTransformer` turns
    into a series.

    Its criterion, mirrored from
    :meth:`~pynecore.transformers.lib_series.LibrarySeriesTransformer.
    visit_Subscript`: the receiver is an ``Attribute`` chain whose root ``Name``
    is ``lib`` and whose first segment after ``lib`` is not one of
    :data:`~pynecore.transformers.lib_series_attrs.NON_SERIES_LIB_ATTRS`. Nested
    namespaces count (``lib.ta.tr``), a ``Call`` never does — which is what
    keeps ``lib.ta.sma(...)[1]`` out.

    :param node: The subscript's receiver.
    :return: True when that pass would make a series of it.
    """
    if not isinstance(node, ast.Attribute):
        return False
    chain: list[str] = []
    cur: ast.expr = node
    while isinstance(cur, ast.Attribute):
        chain.append(cur.attr)
        cur = cur.value
    if not (isinstance(cur, ast.Name) and cur.id == 'lib'):
        return False
    chain.reverse()
    return not (chain and chain[0] in NON_SERIES_LIB_ATTRS)


#: Per function, the direct-``Name`` calls of its own body as ``(callee name,
#: call)`` pairs (see :meth:`SecurityTransformer._own_calls`)
_OwnCalls = dict[ast.FunctionDef | ast.AsyncFunctionDef, list[tuple[str, ast.Call]]]


class _SeriesScope:
    """What one function scope binds, as the passes AFTER this one see it.

    :ivar series: every name a series annotation or a ``Series``-annotated
        parameter gives a history buffer to, order-insensitively. Where the
        declaration stands relative to the reference does not matter here: this
        scope only makes the CANDIDATE, and
        :func:`~pynecore.transformers.security_closed_shift_check.
        verify_closed_shift` checks the lowered write exactly.
    :ivar locals_: every name the scope binds; one that is NOT in ``series``
        shadows a same-named parent series, exactly as
        :meth:`~pynecore.transformers.series.SeriesTransformer._lookup` has it
    :ivar hist: the value each ``__hist_N__`` temp of this scope holds
    """

    __slots__ = ('series', 'locals_', 'hist')

    def __init__(self, series: set[str], locals_: set[str],
                 hist: dict[str, ast.expr]):
        self.series = series
        self.locals_ = locals_
        self.hist = hist


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

# Compound statements whose statement lists ``_recurse_subbodies`` transforms
# on their own, and the fields holding those lists (``handlers`` / ``cases``
# hold them one level down, in each handler / case).
_SUB_BODY_STMTS: tuple[type[ast.AST], ...] = (
    ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith,
    ast.Try, ast.TryStar, ast.Match,
)
_SUB_BODY_FIELDS = frozenset({'body', 'orelse', 'finalbody', 'handlers', 'cases'})

# Lift rounds: each round moves a signal one call level up, so the cap only
# has to exceed the deepest unconditional helper chain a script can have.
_MAX_LIFT_ROUNDS = 16


class _SignalArgSubstituter(ast_walk.NodeTransformer):
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
                return self.visit(ast_walk.clone(self.bindings[node.id]))
            finally:
                self._active.discard(node.id)
        if node.id in self.mapping:
            # The caller's expression is already in the target scope — return
            # it unvisited so its own names are left alone.
            return ast_walk.clone(self.mapping[node.id])
        self.failed = True
        return node


class SecurityTransformer(ast_walk.NodeTransformer):
    """
    Transform request.security() calls into multiprocessing signal/write/read pattern.

    Transforms each lib.request.security(symbol, timeframe, expression, ...) call:

    1. Function start (chart context only):
       if __active_security__ is None:
           __sec_signal__("sec_id", symbol_expr, timeframe_expr)

    2. Original call position:
       if __active_security__ is not None and "sec_id" in __active_security__:
           __sec_write__("sec_id", expression)
       var = __sec_read__("sec_id", lib.na)

    3. Function end (chart context only):
       if __active_security__ is None:
           __sec_wait__("sec_id")

    Also creates module-level __security_contexts__ dict with metadata for each context.
    Non-constant symbol/timeframe/lookahead values (e.g., function parameters or
    input-derived expressions) are stored as None in __security_contexts__ and
    resolved at runtime via __sec_signal__ arguments.

    Every context also carries a ``group`` key: the ordinal of the group of
    contexts that are guaranteed to resolve to the very same feed, or None when
    the context stands alone. Two contexts share a group when their signal
    arguments are SYNTACTICALLY identical — the key is
    ``(symbol, timeframe, lookahead, is_ltf, ignore_invalid_symbol)``, each
    expression compared by ``ast.dump`` of the substituted signal argument (the
    module-level expression for a lookahead that has one, the runtime one
    otherwise). Only contexts signalled in the top block qualify: those run at
    function entry with the same bindings, so an identical expression there
    yields an identical value on every bar, input-derived timeframes included.
    A context whose signal arguments are STABLE — unchanged for the whole bar
    wherever they are evaluated (see :meth:`_stable_signal_args`) — is keyed by
    those arguments alone and groups wherever its signal stands, inline and
    behind branches included; such a group carries ``group_stable=True``.
    Lower-timeframe contexts, groups of one and groups whose reads one child
    could not answer unambiguously (see :meth:`_group_reads_are_unambiguous`)
    get no group.

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
        # ``_module_file`` as last hashed by :meth:`_gen_id`, and its hash
        self._hashed_file: str | None = None
        self._module_hash = ''
        # Sids whose __sec_signal__ is emitted in the function's top block
        # (module-level or hoistable arguments). Only these may be depended on.
        self._top_sec_ids: set[str] = set()
        self._sid_lineno: dict[str, int] = {}
        # Runtime-resolved sids kept out of the top block because their call is
        # not reached on every run, and the ones that must stay there anyway
        # because another context reads them (see ``visit_Module``).
        self._deferred_by_reach: set[str] = set()
        self._keep_top: frozenset[str] = frozenset()
        # Bindings of the function scope chain being processed, innermost last
        # (see :meth:`_scope_bindings`).
        self._scopes: list[_SeriesScope] = []
        # The own-body nodes :meth:`_may_defer` gathered, per function
        self._own_nodes: dict[ast.AST, list[ast.AST]] = {}
        # The binding counts :meth:`_lowered_hoistable` keeps per function
        self._binding_counts_memo: dict[ast.AST, tuple[dict[str, int], set[str]]] = {}

    def _gen_id(self) -> str:
        # The module hash keeps sec ids unique across modules: the main script and
        # any imported library may each have their own security calls, and their
        # contexts are merged into one registry by the runner
        if self._hashed_file != self._module_file:
            self._hashed_file = self._module_file
            self._module_hash = hashlib.sha1(self._module_file.encode()).hexdigest()[:8]
        sec_id = f"sec\xb7{self._module_hash}\xb7{self._counter}"
        self._counter += 1
        return sec_id

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
        yield from ast_walk.iter_descendants(node, (ast.FunctionDef, ast.AsyncFunctionDef))

    @staticmethod
    def _walk_skip_funcs_post(node: ast.AST):
        """Walk AST nodes in post-order (children before their parent),
        skipping nested function definitions."""
        # An explicit stack of (node, child iterator): a recursive generator
        # pays one ``yield from`` hop per tree level for every node
        stack = [(node, ast_walk.iter_child_nodes(node))]
        while stack:
            current, children = stack[-1]
            for child in children:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                stack.append((child, ast_walk.iter_child_nodes(child)))
                break
            else:
                stack.pop()
                yield current

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
        for sub in ast_walk.walk(expr):
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
                       for child in ast_walk.iter_child_nodes(node)
                       if isinstance(child, ast.expr))
        return False

    @staticmethod
    def _binding_counts(
            func: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> tuple[dict[str, int], set[str]]:
        """How often ``func`` binds or deletes each name, and what it declares
        ``global`` / ``nonlocal`` -- nested functions included.

        :param func: the function to count
        :return: name -> number of ``Store`` / ``Del`` occurrences, and the
            declared names
        """
        counts: dict[str, int] = {}
        declared: set[str] = set()
        for sub in ast_walk.walk(func):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, (ast.Store, ast.Del)):
                counts[sub.id] = counts.get(sub.id, 0) + 1
            elif isinstance(sub, (ast.Global, ast.Nonlocal)):
                declared.update(sub.names)
        return counts, declared

    @classmethod
    def _hoistable_bindings(
            cls, func: ast.FunctionDef | ast.AsyncFunctionDef,
            binding_counts: tuple[dict[str, int], set[str]] | None = None
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
        :param binding_counts: its :meth:`_binding_counts`, when already known
        :return: mapping of binding name to its assignment statement, and the
            set of stable parameter names
        """
        counts, declared = (cls._binding_counts(func) if binding_counts is None
                            else binding_counts)
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

    def _lowered_hoistable(
            self, func: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> tuple[dict[str, ast.Assign], set[str]]:
        """:meth:`_hoistable_bindings` of a function of the lowered module.

        The lift rounds and the stable-argument walk ask it of the same few
        functions (the script entry above all) over and over. Its expensive
        half, the :meth:`_binding_counts` walk over the whole function, is
        kept: a lift rewrites statement lists but binds no name (the signal
        arguments it moves are simple chains), so the counts stay exact for the
        rest of the pass. The scan of the body itself is redone every time, as
        a lift may have reordered it.

        :param func: a function of the module after the per-function lowering
        :return: what :meth:`_hoistable_bindings` returns for it
        """
        counts = self._binding_counts_memo.get(func)
        if counts is None:
            counts = self._binding_counts_memo[func] = self._binding_counts(func)
        return self._hoistable_bindings(func, counts)

    @staticmethod
    def _referenced_names(node: ast.expr) -> set[str]:
        """Names read by ``node`` (``lib`` excluded)."""
        return {n.id for n in ast_walk.walk(node)
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

    @staticmethod
    def _is_lookahead_on(node: ast.expr | None) -> bool:
        """Whether ``node`` literally names the lookahead-ON singleton.

        Conservative on purpose: only the spellings of the constant itself
        (``barmerge.lookahead_on``, a bare ``lookahead_on``, ``Lookahead.ON``)
        count. An input-derived lookahead is resolved at runtime and never
        reaches here with a value.
        """
        if isinstance(node, ast.Name):
            return node.id == 'lookahead_on'
        if isinstance(node, ast.Attribute):
            return node.attr == 'lookahead_on' or (
                node.attr == 'ON' and isinstance(node.value, ast.Name)
                and node.value.id == 'Lookahead')
        return False

    @staticmethod
    def _scope_bindings(
            func: ast.FunctionDef | ast.AsyncFunctionDef, own_nodes: list[ast.AST]
    ) -> _SeriesScope:
        """Collect what ONE function scope binds (see :class:`_SeriesScope`).

        Re-derived from the very source shapes the later passes consume: an
        ``AnnAssign`` with a series annotation and a ``Series``-annotated
        positional parameter declare a history buffer, every other binding is a
        plain local, and a ``__hist_N__`` assignment records what
        :class:`~pynecore.transformers.inline_series_hoist.
        InlineSeriesHoistTransformer` lifted into the temp — the standard
        non-repainting idiom (``request.security(..., close[1], lookahead_on) if
        flag else na``) puts the history reference in a ternary branch, so the
        split only ever sees the bare temp.

        Nested functions are skipped: each is a scope of its own, collected when
        it is visited. A temp bound twice in one scope is dropped rather than
        guessed about.

        :param func: The function whose own scope is being collected.
        :param own_nodes: The nodes of its own body
            (:meth:`_walk_skip_funcs_body`).
        :return: The scope's bindings.
        """
        series: set[str] = set()
        locals_: set[str] = set()
        hist: dict[str, ast.expr] = {}
        seen: set[str] = set()

        args = func.args
        for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs):
            locals_.add(arg.arg)
        if args.vararg is not None:
            locals_.add(args.vararg.arg)
        if args.kwarg is not None:
            locals_.add(args.kwarg.arg)
        for arg in args.args:
            if _is_param_series_annotation(arg.annotation):
                series.add(arg.arg)

        for node in own_nodes:
            if (isinstance(node, ast.Name)
                    and isinstance(node.ctx, (ast.Store, ast.Del))):
                locals_.add(node.id)
            if isinstance(node, ast.AnnAssign):
                if (isinstance(node.target, ast.Name)
                        and _is_series_annotation(node.annotation)):
                    series.add(node.target.id)
                continue
            if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
                continue
            target = node.targets[0]
            if not (isinstance(target, ast.Name)
                    and _HIST_TEMP_RE.fullmatch(target.id)):
                continue
            if target.id in seen:
                hist.pop(target.id, None)
                continue
            seen.add(target.id)
            hist[target.id] = node.value
        return _SeriesScope(series, locals_, hist)

    def _is_declared_series(self, name: str) -> bool:
        """Whether ``name`` resolves to a series along the enclosing scope chain.

        The same walk
        :meth:`~pynecore.transformers.series.SeriesTransformer._lookup` makes: a
        name bound in the CURRENT scope without a series declaration there
        shadows any parent series of that name, otherwise the innermost
        enclosing scope that declares it wins.

        Order-insensitive on purpose. Whether the declaration is REACHED before
        the reference — a name read as a plain list above its own
        ``Series[float]`` declaration, or inside the declaring statement the
        child's write is lifted out of — is a property of the lowered tree, and
        :func:`~pynecore.transformers.security_closed_shift_check.
        verify_closed_shift` decides it there on the emitted form instead of
        re-deriving it from source lines here.

        :param name: Source-level variable name.
        :return: True when the name may address a history buffer.
        """
        if not self._scopes:
            return False
        current = self._scopes[-1]
        if name in current.series:
            return True
        if name in current.locals_:
            return False
        for scope in reversed(self._scopes[:-1]):
            if name in scope.series:
                return True
        return False

    def _is_series_receiver(self, node: ast.expr) -> bool:
        """Whether a ``[k]`` subscript of ``node`` addresses a history buffer.

        Either a ``lib``-rooted chain :class:`~pynecore.transformers.lib_series.
        LibrarySeriesTransformer` converts, or a name that resolves to a series
        declaration / a ``Series``-annotated parameter (or to a hoisted
        ``inline_series`` temp).

        :param node: The subscript's receiver.
        :return: True when the subscript is a history read.
        """
        if isinstance(node, ast.Name):
            bound = self._scopes[-1].hist.get(node.id) if self._scopes else None
            if bound is not None and _is_inline_series_call(bound):
                return True
            return self._is_declared_series(node.id)
        return _is_lib_series_chain(node)

    def _is_history_reference(self, node: ast.expr) -> bool:
        """Whether ``node`` reads a CLOSED bar, i.e. lowers to a history read.

        A POSITIVE rule, mirroring the lowering: nothing else is a history
        reference at runtime, so the two shapes below are exhaustive.

        1. ``inline_series(expr, k>=1)`` — what PyneComp emits for an expression
           history reference (``ta.sma(close, 5)[1]``) — directly or through a
           ``__hist_N__`` temp bound to one in this scope.
        2. ``<receiver>[k>=1]`` where the receiver is what a later pass turns
           into a series: a ``lib`` chain for ``LibrarySeriesTransformer``
           (:func:`_is_lib_series_chain`) or a name declared / annotated series
           for ``SeriesTransformer`` (:meth:`_is_series_receiver`).

        Every other receiver — a call, an operator result, a collection, a name
        bound to any of them — is NOT a history reference: a subscript of it is
        element access or a tuple selection whose value follows the developing
        bar. The rule is the CANDIDATE: a shape it misses only costs a
        developing round that would have been skippable, and a source shape it
        reads too generously is caught on the lowered write by
        :func:`~pynecore.transformers.security_closed_shift_check.
        verify_closed_shift`.

        What is INSIDE a history reference does not matter: its value is fixed by
        the bars before the one being computed, and a developing re-tick of a
        security child re-runs the period's first tick from its own rolled-back
        baseline, so every tick of one period computes the same ``[k>=1]`` value.

        :param node: The expression to classify.
        :return: True when the whole expression reads closed bars only.
        """
        if isinstance(node, ast.Name) and self._scopes:
            bound = self._scopes[-1].hist.get(node.id)
            if bound is not None:
                node = bound
        if _is_inline_series_call(node):
            return True
        if isinstance(node, ast.Subscript):
            return (_is_history_index(node.slice)
                    and self._is_series_receiver(node.value))
        return False

    def _closed_shift(self, expression: ast.expr | None,
                      lookahead: ast.expr | None) -> bool:
        """Whether a context's value can only change when its HTF period closes.

        True for ``lookahead_on`` with an expression that is ENTIRELY a history
        reference in the sense of :meth:`_is_history_reference` — ``close[1]``, a
        history read of a declared series, the ``inline_series(..., 1)`` PyneComp
        emits for ``ta.sma(close, 5)[1]`` and ``(close + open)[1]``, and a
        tuple/list whose every element is one. The chart then runs a developing
        round only on the FIRST chart bar of each period (see
        ``core/security.py``); the other bars would recompute the same value.

        The classifier is a positive one and deliberately kept small: a shape it
        does not recognise keeps the period's developing rounds, which costs work
        and never correctness. It also decides the merge grouping, so it must
        run here, on the source shape; the exact check of the lowered write is
        :func:`~pynecore.transformers.security_closed_shift_check.
        verify_closed_shift`.

        ``close[1] + close`` is NOT one of them: its second term reads the
        developing bar, so it does change inside the period. (TradingView's own
        ``lookahead_on`` hands such an expression the period's FINAL value on
        every bar of it — lookahead PyneCore deliberately does not reproduce.)

        :param expression: The ``expression`` argument's AST, or None.
        :param lookahead: The ``lookahead`` argument's AST, or None.
        :return: Whether the developing rounds of the period may be skipped.
        """
        if expression is None or not self._is_lookahead_on(lookahead):
            return False
        if isinstance(expression, (ast.Tuple, ast.List)):
            return bool(expression.elts) and all(
                self._is_history_reference(elt) for elt in expression.elts)
        return self._is_history_reference(expression)

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
    def _in_active_check(sec_id: str) -> ast.BoolOp:
        """Build: __active_security__ is not None and sec_id in __active_security__

        ``__active_security__`` is the SET of contexts the running process
        serves — one security child can serve a whole group — so the guard is a
        membership test, like ``__same_context__``. It stays None in the chart
        process (that is what the signal / wait blocks test for), which the
        left conjunct guards against.
        """
        return ast.BoolOp(
            op=ast.And(),
            values=[
                ast.Compare(
                    left=ast.Name(id='__active_security__', ctx=ast.Load()),
                    ops=[ast.IsNot()], comparators=[ast.Constant(value=None)]
                ),
                ast.Compare(
                    left=ast.Constant(value=sec_id),
                    ops=[ast.In()],
                    comparators=[ast.Name(id='__active_security__', ctx=ast.Load())]
                ),
            ]
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
            args.append(ast_walk.clone(sym_expr) if sym_expr is not None
                        else ast.Constant(value=None))
            args.append(ast_walk.clone(tf_expr) if tf_expr is not None
                        else ast.Constant(value=None))
            # A lookahead that is not module-level evaluable (an input-derived
            # Pine "simple" value) is still passed here when its chain is
            # hoistable — the binding it reads is moved above this block.
            if la_expr is not None:
                args.append(ast_walk.clone(la_expr))
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
        args.append(ast_walk.clone(sym_expr) if sym_expr is not None
                    else ast.Constant(value=None))
        args.append(ast_walk.clone(tf_expr) if tf_expr is not None
                    else ast.Constant(value=None))
        if la_expr is not None:
            args.append(ast_walk.clone(la_expr))
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
        1. This IS a security process serving sec_id (sec_id in __active_security__,
           which is None in the chart process)
        2. This is the chart process and sec_id is same-context (sec_id in __same_context__)
        """
        return ast.If(
            test=ast.BoolOp(
                op=ast.Or(),
                values=[self._in_active_check(sec_id), self._in_same_context(sec_id)]
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
            self, own_nodes: list[ast.AST]
    ) -> list[tuple[ast.Call, str, bool]]:
        """
        Find all request.security() and request.security_lower_tf() calls in
        function body, skipping nested functions. Marks each call node with
        _sec_id attribute.

        :param own_nodes: The nodes of the function's own body
            (:meth:`_walk_skip_funcs_body`).
        :return: List of (call_node, sec_id, is_ltf) tuples
        """
        calls = []
        for node in own_nodes:
            if isinstance(node, ast.Call):
                if is_security_call(node):
                    sec_id = self._gen_id()
                    node._sec_id = sec_id  # type: ignore[attr-defined]
                    calls.append((node, sec_id, False))
                elif is_security_lower_tf_call(node):
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

            # Post-order: a call nested in another call's expression gets its
            # write block FIRST. Its producer then precedes the outer context
            # in program order, so the outer context's child waits for it and
            # pairs its value instead of answering the read with the default.
            # The sub-bodies are done (no marked call is left in them), so only
            # the statement's own parts are searched: re-walking them at every
            # nesting level would make a long ``elif`` chain quadratic.
            call_nodes_here = [
                n for part in self._own_parts(stmt)
                for n in self._walk_skip_funcs_post(part)
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
                    expr = ast_walk.clone(call_exprs[sid])
                    expr = replacer.visit(expr)
                    new_body.append(self._write_block(sid, expr))
                new_body.append(self._replace_own_calls(replacer, stmt))
            else:
                new_body.append(stmt)

        return new_body

    @staticmethod
    def _own_parts(stmt: ast.stmt):
        """The children of ``stmt`` that :meth:`_recurse_subbodies` does not
        transform, nested definitions excluded.

        For a compound statement that is everything but its statement lists
        (the ``except`` types, ``match`` patterns and guards included); any
        other statement keeps all of its children.

        :param stmt: a statement of the body being transformed
        """
        if not isinstance(stmt, _SUB_BODY_STMTS):
            for child in ast_walk.iter_child_nodes(stmt):
                if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    yield child
            return
        for holder in (stmt, *getattr(stmt, 'handlers', ()), *getattr(stmt, 'cases', ())):
            for field in holder._fields:
                if field in _SUB_BODY_FIELDS:
                    continue
                value = getattr(holder, field, None)
                if isinstance(value, ast.AST):
                    yield value
                elif isinstance(value, list):
                    yield from (item for item in value if isinstance(item, ast.AST))

    @staticmethod
    def _replace_own_calls(replacer: '_CallReplacer', stmt: ast.stmt) -> ast.stmt:
        """Replace the marked calls of ``stmt`` outside its transformed sub-bodies.

        :param replacer: the replacer of the body being transformed
        :param stmt: a statement holding marked calls in its own parts
        :return: the statement with those calls replaced
        """
        if not isinstance(stmt, _SUB_BODY_STMTS):
            return replacer.visit(stmt)
        for holder in (stmt, *getattr(stmt, 'handlers', ()), *getattr(stmt, 'cases', ())):
            for field in holder._fields:
                if field in _SUB_BODY_FIELDS:
                    continue
                value = getattr(holder, field, None)
                if isinstance(value, ast.AST):
                    setattr(holder, field, replacer.visit(value))
                elif isinstance(value, list):
                    value[:] = [replacer.visit(item) if isinstance(item, ast.AST) else item
                                for item in value]
        return stmt

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
        """Transform a function, with its own scope on the scope stack.

        The stack is what :meth:`_is_declared_series` resolves a name along, so
        every function pushes its bindings — a function without a
        ``request.security()`` of its own may still be the scope that DECLARES
        the series a nested one reads. The body below finishes its own contexts
        before the nested definitions are visited
        (:meth:`_visit_nested_defs`), so a nested scope always sees the chain
        above it.

        The nodes of the function's own body are gathered once, for both the
        bindings and the security calls (or taken from :meth:`_may_defer`).
        """
        own_nodes = self._own_nodes.pop(node, None)
        if own_nodes is None:
            own_nodes = list(self._walk_skip_funcs_body(node))
        self._scopes.append(self._scope_bindings(node, own_nodes))
        try:
            return self._process_func_body(node, own_nodes)
        finally:
            self._scopes.pop()

    def _process_func_body(self, node: ast.FunctionDef | ast.AsyncFunctionDef,
                           own_nodes: list[ast.AST]):
        """Transform a function containing request.security() / security_lower_tf() calls.

        :param node: The function.
        :param own_nodes: The nodes of its own body (:meth:`_walk_skip_funcs_body`).
        """
        calls = self._collect_calls(own_nodes)

        if not calls:
            self._visit_nested_defs(node)
            return node

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
                lookahead_rt = ast_walk.clone(lookahead)

            # Store actual expressions for __sec_signal__ args (always passed at runtime)
            self._signal_args[sec_id] = (
                ast_walk.clone(symbol) if symbol is not None else None,
                ast_walk.clone(timeframe) if timeframe is not None else None,
                lookahead_rt,
            )

            # For __security_contexts__ at module level: only use values that are
            # evaluable at module scope. Function parameters etc. become None.
            ctx: dict[str, ast.expr] = {}
            if symbol is not None:
                if self._is_module_level_expr(symbol):
                    ctx['symbol'] = ast_walk.clone(symbol)
                else:
                    ctx['symbol'] = ast.Constant(value=None)
            if timeframe is not None:
                if self._is_module_level_expr(timeframe):
                    ctx['timeframe'] = ast_walk.clone(timeframe)
                else:
                    ctx['timeframe'] = ast.Constant(value=None)

            if is_ltf:
                ctx['is_ltf'] = ast.Constant(value=True)
            else:
                gaps_expr = (
                    ast_walk.clone(gaps) if gaps is not None else self._default_gaps()
                )
                ctx['gaps'] = gaps_expr
                if lookahead is not None:
                    # A module-level lookahead (``barmerge.lookahead_*``) is
                    # consumed at module load to wire the HTF transport. An
                    # input-derived expression would NameError at import time,
                    # so store None as a placeholder — the runtime value passed
                    # to ``__sec_signal__`` resolves the mode on the first bar.
                    ctx['lookahead'] = (ast_walk.clone(lookahead) if lookahead_rt is None
                                        else ast.Constant(value=None))
                # Developing rounds inside one HTF period are skippable when
                # the whole expression is a ``[k>=1]`` history reference under
                # ``lookahead_on`` (see :meth:`_closed_shift`). Emitted for every
                # non-LTF context, True or False: the runtime decides whether the
                # skip applies, and ``PYNE_NO_SECURITY_DEV_SKIP`` turns it off
                # without touching the emitted tree.
                ctx['closed_shift'] = ast.Constant(
                    value=self._closed_shift(expression, lookahead))

            if ignore_invalid is not None:
                ctx['ignore_invalid_symbol'] = ast_walk.clone(ignore_invalid)

            if not is_ltf and currency is not None:
                ctx['currency'] = ast_walk.clone(currency)

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
        reached_ids = self._reached_ids(node.body)
        reached = {sid for call, sid, _is_ltf in calls if id(call) in reached_ids}
        top_sec_ids = []
        runtime_sec_ids: set[str] = set()
        needed_names: set[str] = set()
        for sid in sec_ids:
            sym_expr, tf_expr, _la_expr = self._signal_args[sid]
            runtime_resolved = self._resolved_at_runtime(sym_expr, tf_expr)
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

        self._visit_nested_defs(node)
        return node

    def _visit_nested_defs(self, node: ast.AST) -> None:
        """Process the function definitions standing in ``node``'s statements.

        What ``generic_visit`` would do, without walking every expression on
        the way: a definition only ever stands in a statement list, and it is
        the only node this pass rewrites — in place, handing back the same
        node — so the statements alone lead to every one, in the same order.
        A definition nested in another one is reached through that one's own
        processing.

        :param node: the module or function whose definitions to process
        """
        stack = [ast_walk.iter_child_statements(node)]
        while stack:
            for child in stack[-1]:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    self._process_func(child)
                else:
                    stack.append(ast_walk.iter_child_statements(child))
                    break
            else:
                stack.pop()

    @classmethod
    def _resolved_at_runtime(cls, symbol: ast.expr | None,
                             timeframe: ast.expr | None) -> bool:
        """Whether a context's feed is only known once the script runs.

        :param symbol: the ``symbol`` argument, or None
        :param timeframe: the ``timeframe`` argument, or None
        :return: True when either is not evaluable at module level
        """
        return any(e is not None and not cls._is_module_level_expr(e)
                   for e in (symbol, timeframe))

    @classmethod
    def _reached_ids(cls, body: list[ast.stmt]) -> set[int]:
        """Ids of the expressions evaluated every time ``body`` is entered.

        :param body: a function body
        :return: the ids of the :meth:`_unconditional_exprs` of its
            :meth:`_unconditional_stmts`
        """
        reached: set[int] = set()
        for stmt in cls._unconditional_stmts(body):
            reached.update(id(expr) for expr in cls._unconditional_exprs(stmt))
        return reached

    def _may_defer(self, module: ast.Module) -> bool:
        """Whether lowering ``module`` can defer a signal by reach.

        The decision :meth:`_process_func_body` makes per context, made up
        front: a call whose symbol or timeframe is resolved at runtime and that
        is not reached every time its function runs. It is made on the
        untransformed tree, which is exactly what every function body still is
        when its turn comes -- a function's lowering never edits the body of
        another one.

        The own-body nodes it walks are kept for :meth:`_process_func`, for
        the same reason: they are still what that function's turn would find.

        :param module: the module about to be lowered
        :return: True when at least one context will be deferred by reach
        """
        for func in ast_walk.walk_statements(module):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            own_nodes = self._own_nodes[func] = list(self._walk_skip_funcs_body(func))
            deferrable = []
            for node in own_nodes:
                if not isinstance(node, ast.Call):
                    continue
                if is_security_call(node):
                    symbol, timeframe = self._extract_args(node)[:2]
                elif is_security_lower_tf_call(node):
                    symbol, timeframe = self._extract_ltf_args(node)[:2]
                else:
                    continue
                if self._resolved_at_runtime(symbol, timeframe):
                    deferrable.append(node)
            if deferrable:
                reached = self._reached_ids(func.body)
                if any(id(call) not in reached for call in deferrable):
                    return True
        return False

    def _analyze_dependencies(self, node: ast.Module) -> None:
        """Run the taint analysis on the lowered module and record its result.

        Writes ``depends`` into every context: the sorted sids whose results
        flow into this sid's ``__sec_write__`` expression and whose producers
        come earlier in program order. The flowing sids whose producers come
        later go to ``late_reads`` (only when there are any), and a write that
        can run more than once per bar gets ``in_loop``.

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
            late = sorted(analyzer.late_reads.get(sid, set()) & known)
            if late:
                ctx['late_reads'] = ast.List(
                    elts=[ast.Constant(value=d) for d in late], ctx=ast.Load()
                )
            if sid in analyzer.in_loop:
                ctx['in_loop'] = ast.Constant(value=True)

    def _signal_sites(self, module: ast.Module, own_calls: _OwnCalls
                      ) -> dict[str, tuple[str, ...]]:
        """Where every ``__sec_signal__`` stands and with which arguments.

        Both halves are read off the FINAL tree, after
        :meth:`_lift_helper_signals`: a lifted signal is evaluated in the
        caller, with the arguments substituted into the caller's scope, and
        that — not the shape it had in the helper — is what decides the feed.

        :param module: the lowered module
        :param own_calls: the module's :meth:`_own_calls`
        :return: sid -> (host function key, ``ast.dump`` of each argument)
        """
        sites: dict[str, tuple[str, ...]] = {}
        for func, key in self._function_keys(module).items():
            for name, call in own_calls[func]:
                if name != '__sec_signal__':
                    continue
                sid = self._call_sid(call)
                if sid is None:
                    continue
                sites[sid] = (key, *(ast.dump(arg) for arg in call.args[1:]))
        return sites

    @classmethod
    def _own_calls(cls, module: ast.Module) -> _OwnCalls:
        """The direct-``Name`` calls of every function's OWN body.

        Read off the final tree once and shared by the passes over it that look
        for call sites and signals, instead of each walking every body again.

        :param module: the lowered module
        :return: function -> ``(callee name, call)`` pairs in
            :meth:`_walk_skip_funcs_body` order, for every function in
            :func:`ast_walk.walk_statements` order
        """
        found: _OwnCalls = {}
        for func in ast_walk.walk_statements(module):
            if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                found[func] = [(sub.func.id, sub) for sub in cls._walk_skip_funcs_body(func)
                               if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)]
        return found

    @staticmethod
    def _walk_skip_funcs_body(func: ast.FunctionDef | ast.AsyncFunctionDef):
        """Nodes of a function's OWN body, nested definitions excluded."""
        for stmt in func.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            yield from SecurityTransformer._walk_skip_funcs(stmt)

    @staticmethod
    def _function_keys(
            module: ast.Module
    ) -> dict[ast.FunctionDef | ast.AsyncFunctionDef, str]:
        """A unique key per function definition of the module.

        The key is the dotted path of the definition, with an occurrence index
        where one scope defines the same name twice, so two functions never
        share it. Only a definition standing directly in a scope's statement
        list gets one: a definition inside an ``if`` / loop / ``with`` / ``try``
        body has no key, so its signals never form a hosted group.
        """
        keys: dict[ast.FunctionDef | ast.AsyncFunctionDef, str] = {}

        def walk(body: list[ast.stmt], prefix: str) -> None:
            seen: dict[str, int] = {}
            for func in body:
                if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                count = seen.get(func.name, 0)
                seen[func.name] = count + 1
                key = f'{prefix}{func.name}' + (f'#{count}' if count else '')
                keys[func] = key
                walk(func.body, f'{key}.')

        walk(module.body, '')
        return keys

    @staticmethod
    def _call_sites_by_name(
            own_calls: _OwnCalls
    ) -> dict[str, list[tuple[ast.FunctionDef | ast.AsyncFunctionDef, ast.Call]]]:
        """Every direct call of every name.

        Unlike :meth:`_lift_candidates` this does not care whether a call is
        reached unconditionally — a stable signal argument is the same value
        wherever the call happens. What it does care about is seeing ALL of the
        sites: together with how often each name is read (see :meth:`_census`)
        they are what proves a name is only ever used by calling it, so
        substituting its parameters covers every way the function can run.

        :param own_calls: the module's :meth:`_own_calls`
        :return: callee name -> its call sites as ``(caller function, call
            node)``
        """
        sites: dict[str, list[tuple[ast.FunctionDef | ast.AsyncFunctionDef,
                                    ast.Call]]] = {}
        for caller, calls in own_calls.items():
            for name, call in calls:
                sites.setdefault(name, []).append((caller, call))
        return sites

    def _signal_call_sites(
            self, own_calls: _OwnCalls
    ) -> dict[str, tuple[ast.FunctionDef | ast.AsyncFunctionDef, list[ast.expr]]]:
        """Every ``__sec_signal__`` of the final tree, with its host function.

        :param own_calls: the module's :meth:`_own_calls`
        :return: sid -> ``(function the signal stands in, its arguments after
            the sid)``
        """
        found: dict[str, tuple[ast.FunctionDef | ast.AsyncFunctionDef,
                               list[ast.expr]]] = {}
        for func, calls in own_calls.items():
            for name, call in calls:
                if name != '__sec_signal__':
                    continue
                sid = self._call_sid(call)
                if sid is not None:
                    found[sid] = (func, list(call.args[1:]))
        return found

    def _stable_signal_args(self, own_calls: _OwnCalls,
                            load_counts: dict[str, int]) -> dict[str, tuple[str, ...]]:
        """The signal arguments of every context that cannot change during a bar.

        A context's signal decides WHICH feed it reads. When the expressions
        that decide it are STABLE, the feed is a property of the script rather
        than of the program point the signal happens to run at — and then two
        contexts with the same such expressions read the same feed even if one
        of them signals inline, behind a branch, or not at all on a given bar.
        That is what lets those contexts share a child process.

        Stable is decided by walking the arguments UP to the script entry,
        substituting them into each caller exactly as :meth:`_lift_signal`
        substitutes a lifted argument — the same :class:`_SignalArgSubstituter`,
        the same :meth:`_param_mapping` — with two differences:

        * the call need NOT be reached unconditionally. A value that cannot
          change does not depend on whether the call runs;
        * a function may have SEVERAL call sites, as long as every one of them
          substitutes to the same expression. That is the shape a helper
          wrapping ``request.security()`` and used a few times takes.

        At every level the argument must be a :meth:`_is_simple_chain` over the
        names that already hold their final value at function entry
        (:meth:`_hoistable_bindings`): constants, ``lib.*`` chains such as
        ``syminfo.tickerid``, ``input.*`` calls, operators over those, the
        function's never-rebound parameters and its single-assignment simple
        bindings. A local the body reassigns is therefore never stable, and
        neither is anything reading a series. A name used in any way OTHER than
        calling it (an alias, a callback) is not walked at all — the pass cannot
        see how it is then invoked.

        The walk ends at a function nobody calls — the script entry — whose
        parameters are its ``input.*`` defaults, fixed for the run.

        :param own_calls: the module's :meth:`_own_calls`
        :param load_counts: how often each name is read in the module
            (:meth:`_census`)
        :return: sid -> ``ast.dump`` of each fully substituted argument, for
            the contexts whose arguments are stable
        """
        call_sites = self._call_sites_by_name(own_calls)
        # Every context's walk passes through the script entry, and the tree
        # is only read here, so each function's bindings are computed once.
        bindings_of: dict[int, tuple[dict[str, ast.Assign], set[str]]] = {}

        def resolve(func: ast.FunctionDef | ast.AsyncFunctionDef,
                    args: list[ast.expr], depth: int) -> tuple[str, ...] | None:
            """The stable dumps of ``args`` as evaluated in ``func``, or None."""
            if depth > _MAX_LIFT_ROUNDS:
                return None
            found = bindings_of.get(id(func))
            if found is None:
                found = bindings_of[id(func)] = self._lowered_hoistable(func)
            hoistable, stable_params = found
            available = set(hoistable) | stable_params
            if not all(self._is_simple_chain(a, available) for a in args):
                return None
            sites = call_sites.get(func.name, ())
            if not sites:
                if load_counts.get(func.name, 0):
                    # Read without being called here: an alias or a callback,
                    # so how its parameters are bound is unknown.
                    return None
                # Nobody calls it: the script entry.
                return tuple(ast.dump(a) for a in args)
            if len(sites) != load_counts.get(func.name, 0):
                # Some read of the name is not one of these calls.
                return None
            bindings = {name: stmt.value for name, stmt in hoistable.items()}
            answers: set[tuple[str, ...]] = set()
            for caller, call in sites:
                if caller is func:
                    return None
                mapping = self._param_mapping(func, call)
                if mapping is None:
                    return None
                substituted: list[ast.expr] = []
                for arg in args:
                    substituter = _SignalArgSubstituter(mapping, bindings)
                    new_arg = substituter.visit(ast_walk.clone(arg))
                    if substituter.failed:
                        return None
                    substituted.append(new_arg)
                answer = resolve(caller, substituted, depth + 1)
                if answer is None:
                    return None
                answers.add(answer)
            # Several call sites agree only when they pass the same thing.
            return answers.pop() if len(answers) == 1 else None

        stable: dict[str, tuple[str, ...]] = {}
        for sid, (host, args) in self._signal_call_sites(own_calls).items():
            dumps = resolve(host, list(args), 0)
            if dumps is not None:
                stable[sid] = dumps
        return stable

    def _group_key(self, sid: str,
                   sites: dict[str, tuple[str, ...]],
                   stable_args: dict[str, tuple[str, ...]]
                   ) -> tuple[tuple[str | None, ...] | None, bool]:
        """Compile-time context key of ``sid``, or None when it may not group.

        The key is the syntax of what decides WHICH feed the context resolves
        to — symbol, timeframe, lookahead and ``ignore_invalid_symbol`` — plus
        ``closed_shift``, which decides the round protocol the shared child runs.
        ``gaps`` and ``currency`` are left out on purpose: they only shape how
        the chart side reads the result, not what the child loads. A
        lower-timeframe context never gets one (its write path is its own).

        There are two kinds of key, and they never match each other:

        * STABLE — the signal's arguments, walked up to the script entry, are
          expressions whose value cannot change during a bar (see
          :meth:`_stable_signal_args`). The feed then follows from the script
          alone, so the key is position-independent: contexts match wherever
          their signals stand, inline and behind branches included.
        * HOSTED — the arguments are not stable, so identical syntax only means
          an identical value at the SAME program point: two helpers may both
          signal ``(syminfo.tickerid, tf)`` and mean two different ``tf``. The
          host function is therefore part of the key, and only a context
          signalled in a TOP block qualifies at all — there the arguments are
          evaluated at function entry with the same bindings, so identical
          syntax does mean an identical value on every bar.

        :param sid: the context id
        :param sites: the signal sites of the final tree (see :meth:`_signal_sites`)
        :param stable_args: the stable substituted arguments per sid
        :return: ``(key, stable)``; the key is None when the context must stand
            alone
        """
        if sid in self._ltf_sec_ids:
            return None, False
        ctx = self._all_contexts[sid]
        # A runtime lookahead travels in the signal args and is already part of
        # the key below; a module-level one is only in the context.
        _sym, _tf, la_rt = self._signal_args[sid]
        la_expr = None if la_rt is not None else ctx.get('lookahead')
        # ``closed_shift`` decides whether the chart runs a developing round on
        # every chart bar of a period or only on its first, and one child runs
        # ONE round protocol for its whole group — so members of two different
        # modes may not share it.
        tail = (self._dump(la_expr), self._dump(ctx.get('ignore_invalid_symbol')),
                self._dump(ctx.get('closed_shift')))
        args = stable_args.get(sid)
        if args is not None:
            return ('stable', *args, *tail), True
        site = sites.get(sid)
        if site is None or sid not in self._top_sec_ids:
            return None, False
        return ('hosted', *site, *tail), False

    @staticmethod
    def _dump(expr: ast.expr | None) -> str | None:
        """``ast.dump`` of an optional expression."""
        return None if expr is None else ast.dump(expr)

    def _sid_set(self, sid: str, key: str) -> set[str]:
        """One context's ``depends`` / ``late_reads`` list, as a set of sids.

        :param sid: The context's id.
        :param key: ``'depends'`` or ``'late_reads'``.
        :return: The sids in that list (empty when the key is absent).
        """
        value = self._all_contexts[sid].get(key)
        if not isinstance(value, ast.List):
            return set()
        return {elt.value for elt in value.elts
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str)}

    def _group_reads_are_unambiguous(self, members: list[str]) -> bool:
        """Whether one child could answer every read of these contexts exactly.

        A child that serves a whole group answers a read of a context it serves
        from its own bar state, never from that context's ring — waiting on a
        frontier the child itself raises later in the same round would be a
        deadlock on itself. The bar state says WHAT was read, but not WHO read
        it: ``__sec_read__`` is handed the context being read, not the call site
        doing the reading.

        That is only enough while every site reading one context expects the
        same answer, and the two shapes that break it are:

        * a context read as a DEPENDENCY by one member (its write stands before
          that read, so the answer is this bar's value) and as a LATE READ by
          another (the producer writes afterwards, so the answer is the
          default);
        * a member of the group itself late-read by another member — the same
          split, with the ambiguous context inside the group.

        Rejecting the whole group in either case is what makes a merged run
        equal to an unmerged one value for value, and it costs nothing else: the
        members then run one child each, on the group clone, exactly as before.

        :param members: The candidate group's contexts.
        :return: Whether the group may be formed.
        """
        depends: set[str] = set()
        late: set[str] = set()
        for sid in members:
            depends |= self._sid_set(sid, 'depends')
            late |= self._sid_set(sid, 'late_reads')
        return not (depends & late) and not (set(members) & late)

    def _assign_groups(self, module: ast.Module, load_counts: dict[str, int]) -> None:
        """Write the ``group`` ordinal into every context.

        Contexts with an identical :meth:`_group_key` form one group and get
        its ordinal; a key with a single context, a group whose reads one child
        could not answer unambiguously
        (:meth:`_group_reads_are_unambiguous`), and every context without a key
        get None. Ordinals follow the order of the contexts themselves, so they
        are stable for a given module.

        ``group_stable`` says which of the two key rules the group formed under.
        It matters at runtime: a STABLE group's members are known to read the
        seed's feed before they have signalled at all, so the chart may resolve
        them from the seed and start them together; a hosted group's members
        must have signalled and resolved to the same thing first.

        :param module: the lowered module
        :param load_counts: how often each name is read in it (:meth:`_census`)
        """
        own_calls = self._own_calls(module)
        sites = self._signal_sites(module, own_calls)
        stable_args = self._stable_signal_args(own_calls, load_counts)
        by_key: dict[tuple[str | None, ...], list[str]] = {}
        stable_of_key: dict[tuple[str | None, ...], bool] = {}
        for sid in self._all_contexts:
            key, stable = self._group_key(sid, sites, stable_args)
            if key is not None:
                by_key.setdefault(key, []).append(sid)
                stable_of_key[key] = stable

        group_of: dict[str, int] = {}
        stable_of: dict[str, bool] = {}
        ordinal = 0
        for key, members in by_key.items():
            if len(members) < 2 or not self._group_reads_are_unambiguous(members):
                continue
            for sid in members:
                group_of[sid] = ordinal
                stable_of[sid] = stable_of_key[key]
            ordinal += 1

        for sid, ctx in self._all_contexts.items():
            ctx['group'] = ast.Constant(value=group_of.get(sid))
            if sid in stable_of:
                ctx['group_stable'] = ast.Constant(value=stable_of[sid])

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
        for node in ast_walk.walk_statements(module):
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
                for sub in self._unconditional_exprs(stmt):
                    if not (isinstance(sub, ast.Call)
                            and isinstance(sub.func, ast.Name)):
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

    def _mark_signal_per_bar(self, module: ast.Module, counts: dict[str, int]) -> None:
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
        :param counts: how many ``__sec_signal__`` calls of each sid it holds
            (:meth:`_census`)
        """
        entry: ast.FunctionDef | ast.AsyncFunctionDef | None = None
        for node in ast_walk.walk_statements(module):
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
            if self._can_exit(stmt):
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
            if cls._can_exit(stmt):
                return

    @staticmethod
    def _can_exit(stmt: ast.stmt) -> bool:
        """Whether ``stmt`` holds a statement that can end its body early.

        Every form of ``_EXIT_STMTS`` is a statement, so only the statement
        lists are searched, never an expression. A nested definition's exits
        end that function, not this body, so definitions are not entered.

        :param stmt: a statement that is not itself a definition
        :return: True when a ``return`` / ``raise`` / ``break`` / ``continue``
            stands in it
        """
        stack: list[ast.AST] = [stmt]
        while stack:
            node = stack.pop()
            if isinstance(node, _EXIT_STMTS):
                return True
            for child in ast_walk.iter_child_statements(node):
                if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    stack.append(child)
        return False

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

    @staticmethod
    def _unconditional_exprs(stmt: ast.stmt) -> list[ast.AST]:
        """The expressions evaluated every time ``stmt`` is evaluated, in pre-order.

        Walks only the expression children, stopping at any form that may skip
        its operands (see ``_CONDITIONAL_EXPRS``; the form itself is listed).
        Call arguments count as unconditional: Python evaluates every one of
        them before the call. Gathered in one walk per statement, so asking
        about every call of the statement costs no further walk.

        :param stmt: the statement whose expressions are wanted
        :return: the nodes in the order :meth:`_walk_skip_funcs` yields them
        """
        reached: list[ast.AST] = []
        stack = [child for child in reversed(list(ast_walk.iter_child_nodes(stmt)))
                 if isinstance(child, (ast.expr, ast.keyword))]
        while stack:
            node = stack.pop()
            reached.append(node)
            if isinstance(node, _CONDITIONAL_EXPRS):
                continue
            stack.extend(child for child in reversed(list(ast_walk.iter_child_nodes(node)))
                         if isinstance(child, (ast.expr, ast.keyword)))
        return reached

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
            self, module: ast.Module, load_counts: dict[str, int]
    ) -> list[tuple[ast.FunctionDef | ast.AsyncFunctionDef,
                    ast.FunctionDef | ast.AsyncFunctionDef, ast.Call, int]]:
        """Helper functions whose signal block may move up one call level.

        A candidate is a function with a chart-guard signal block whose name is
        read exactly once in the whole module, at a direct-``Name`` call site
        that stands in an enclosing function's own statement list, is evaluated
        unconditionally there, and has no statement ahead of it that can end
        that body early.

        A name read anywhere else (an alias, a callback, a second call) means
        the helper may run a different number of times than the single site
        suggests, hence ``load_counts``.

        :param module: the lowered module
        :param load_counts: how often each name is read in it (:meth:`_census`)
        :return: ``(helper, caller, call, stmt_index)`` tuples, ordered by call
            site so lifted signals keep their source order
        """
        funcs: list[ast.FunctionDef | ast.AsyncFunctionDef] = [
            n for n in ast_walk.walk_statements(module)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]

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
                    for sub in self._unconditional_exprs(stmt):
                        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
                            sites[sub.func.id] = (caller, sub, idx)
                # Anything beyond this statement may never be reached. A
                # nested def's own ``return`` belongs to that function, not to
                # this body, so nested scopes are skipped.
                blocked = self._can_exit(stmt)

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

        hoistable, stable_params = self._lowered_hoistable(caller)
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
                new_arg = substituter.visit(ast_walk.clone(arg))
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
            ast_walk.fix_missing_locations(new_stmt)
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
                        node is call for node in ast_walk.walk(trailing.value)):
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
            ast_walk.fix_missing_locations(new_stmt)
            caller_block.body.append(new_stmt)

    def _lift_helper_signals(
            self, module: ast.Module) -> tuple[dict[str, int], dict[str, int]]:
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
        :return: the :meth:`_census` of the module as the lift leaves it
        """
        for _ in range(_MAX_LIFT_ROUNDS):
            census = self._census(module)
            moved = False
            for helper, caller, call, _idx in self._lift_candidates(module, census[0]):
                if self._lift_signal(helper, caller, call):
                    moved = True
            if not moved:
                return census
        return self._census(module)

    @classmethod
    def _census(cls, module: ast.Module) -> tuple[dict[str, int], dict[str, int]]:
        """How often each name is read, and each sid signalled, in the module.

        :param module: the lowered module
        :return: name -> ``Load`` occurrences, and sid -> ``__sec_signal__``
            calls
        """
        loads: dict[str, int] = {}
        signals: dict[str, int] = {}
        for sub in ast_walk.walk(module):
            if isinstance(sub, ast.Name):
                if isinstance(sub.ctx, ast.Load):
                    loads[sub.id] = loads.get(sub.id, 0) + 1
            elif (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
                  and sub.func.id == '__sec_signal__'):
                sid = cls._call_sid(sub)
                if sid is not None:
                    signals[sid] = signals.get(sid, 0) + 1
        return loads, signals

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        return self._process_func(node)  # type: ignore[return-value]

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        return self._process_func(node)  # type: ignore[return-value]

    def visit_Module(self, node: ast.Module) -> ast.Module:
        self._module_file = getattr(node, '_module_file_path', '<script>')
        if not has_security_call(node):
            # Every rewrite of this pass starts at a security call: a module
            # without one would only be walked to be handed back unchanged
            return node
        # Only a context deferred by reach can call for the re-run below, which
        # starts from the module as it is now: the copy is kept only when the
        # lowering is going to defer one.
        original = (ast_walk.clone(node) if not self._keep_top and self._may_defer(node)
                    else None)
        self._visit_nested_defs(node)

        if self._all_contexts:
            load_counts, signal_counts = self._lift_helper_signals(node)
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
            if keep_top and not self._keep_top:
                # ``_may_defer`` foresaw every deferral
                assert original is not None
                second = SecurityTransformer()
                second._keep_top = frozenset(keep_top)
                return second.visit(original)
            self._mark_always_read(node)
            self._mark_signal_per_bar(node, signal_counts)
            # Groups are decided on the FINAL pass only: the re-run above
            # starts from the untransformed module and assigns its own.
            self._assign_groups(node, load_counts)

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


class _CallReplacer(ast_walk.NodeTransformer):
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
