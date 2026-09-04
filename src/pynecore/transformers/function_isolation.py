"""
Transform function call sites to the slot-based instance-state scheme.

Every isolated call site is classified at TRANSFORM time and emitted on one
of three routes (see ``work/benchmark`` plan, section 3.4):

- **fast** (provably state-carrying callee): the child instance's state
  lives in a compile-time-assigned slot of the CALLER's state vector::

      f((__st·__ if (__st·__ := __state__[5]) is not None
         else __resolve_slot·__(__state__, 5, f)), x, 12)

  Loop-shaped sites keep ONE instance shared by every iteration —
  TradingView's per-call-site state does not multiply with loop iterations
  (measured; see :func:`~pynecore.core.instance_state.__loop_state__`, which
  also runs the same-bar rollback of the callee's builtin machines)::

      f(__loop_state·__(__state__, 5, f), x)

- **direct** (provably stateless callee): plain call, zero overhead.

- **uniform** (anything not provable): the caller anchors a
  ``(callee, bound)`` pair in its own slot; the hot path is one identity
  check, ``__bind_any·__`` (re)binds on a miss. Loop-shaped sites fold the
  whole guard into ``__bind_loop·__`` (shared instance + same-bar rollback,
  like the fast loop form)::

      (__b·__[1] if (__b·__ := __state__[7]) is not None and __b·__[0] is f
       else __bind_any·__(__state__, 7, f))(x)
      __bind_loop·__(__state__, 7, f)(x)

The straight-line shapes above are assignment expressions, which Python
forbids anywhere inside a comprehension's ITERABLE expression (a lambda or a
nested comprehension in there does not lift the ban). Straight-line sites
under an iterable therefore fold the whole guard into one helper call —
``__slot_state·__`` / ``__bind_slot·__`` (the loop forms are helper-only
already and need no variant)::

      [x for x in __bind_slot·__(__state__, 7, f)()]

Classification sources:

- same-module functions: the shared :class:`ModuleLayout` (this transformer
  must run AFTER the Persistent and Series transformers) plus a carrier
  fixpoint over the module's call graph — a function carries state if it has
  own slots or any non-direct call site; a name whose LAST definition is
  decorated routes uniform (the runtime value is the decorator's return
  value — an ``overload`` dispatcher, an ``lru_cache`` wrapper, ...);
- cross-module callees (``lib.*``, user Pyne libraries): the callee module
  is imported at transform time and the object inspected —
  ``__pyne_bind__`` marks an overload dispatcher (uniform),
  ``__pyne_layout__`` proves state-carrying, a ``__pyne_slot_layout__``
  marker in the function's globals with no layout attribute proves
  stateless, everything else falls to uniform.

Unprovable always degrades to uniform (correct, only slower) — an error can
only come from a false proof, never from missing knowledge.

Overload pin and instance vector
--------------------------------

Two constants ride along on the emitted binder/state-creating calls. The
overload ``pin`` selects the callee's implementation; the instance ``vector``
configures the callee's INSTANCE, and both are decided by the type pass. A
generic body is shared by every context it is instantiated in, so a site
inside it whose pin differs per context cannot be a constant — the type pass
marks such sites (``get_varying`` on the definition), this pass reserves ONE
slot for the vector and the site reads its own entry out of it,
``__state__[k][j]``. A call site whose callee has such sites passes the
vector for the instance it creates, so the callee's body resolves per
instance. Everything else emits exactly what it emitted before; a scope with
no varying site gets no slot.

The vector reaches an instance where the instance is CREATED, so a call site
that creates none cannot hand one over. A ``@pine_method`` call is skipped
here entirely and binds through the method cache
(:mod:`~pynecore.core.pine_method`), and an overload dispatcher builds its
implementations' vectors itself (``overload._anchored``): a varying site
inside either falls back to value dispatch, with the type pass's
``context-dependent-pin`` diagnostic left standing.

Deliberately left untouched (raw calls): module-level call sites (a stateful
callee there raises a transform error, and one carrying an overload pin binds
in place through ``__bind_pinned·__`` — there is no anchor up there to resolve
the pin in, and the module body runs once), decorator and default-argument
expressions, class bodies, ``__test_*__`` functions (the test framework
calls them with fixtures, they must not grow a hidden parameter), and calls
whose callee is not a plain name/attribute. Calls inside lambdas are
anchored on the straight-line uniform route: a lambda body runs at its
consumer's cadence (a sort comparator, a filter predicate), which is not
loop-iteration semantics, so the loop form's same-bar rollback must not
apply there.
"""
from typing import cast, Any
import ast
import builtins
import importlib
import types

from ..core.pine_export import Exported
from ..utils.stdlib_checker import is_stdlib
from .pine_type_rules import (get_pin, get_pins, get_ty, get_varying, get_vector,
                              stamp_lowering)
# noinspection PyProtectedMember
from .slot_layout import DEFAULT_STATE_PARAM, ModuleLayout, scope_for_function

__all__ = ['FunctionIsolationTransformer', 'NON_TRANSFORMABLE_FUNCTIONS', 'HELPER_ALIASES']

# The runtime helpers are injected as a module-level import into the user's
# own namespace, so they get the same collision-safe middle-dot alias as the
# generated temporaries: a script variable (module global, parameter or local)
# named like a helper can neither shadow the import nor be clobbered by it.
# An alias stays a plain global lookup, so the emission costs nothing extra.
HELPER_ALIASES = {
    '__resolve_slot__': '__resolve_slot·__',
    '__bind_any__': '__bind_any·__',
    '__slot_state__': '__slot_state·__',
    '__bind_slot__': '__bind_slot·__',
    '__loop_state__': '__loop_state·__',
    '__bind_loop__': '__bind_loop·__',
    '__bind_pinned__': '__bind_pinned·__',
}

# Functions that should not be transformed because they:
# - don't return anything (plotting, display)
# - can't have Series values
# - are purely for output/display purposes
# This makes code run little bit faster
NON_TRANSFORMABLE_FUNCTIONS = {
    # Plot and display related (function-and-namespace modules appear as their
    # self-named function after the module property rewrite, e.g. lib.plot.plot)
    'lib.plot.plot', 'lib.plotchar', 'lib.plotshape', 'lib.plotarrow',
    'lib.label', 'lib.table', 'lib.box', 'lib.line', 'lib.hline.hline',
    'lib.fill', 'lib.bgcolor', 'lib.barcolor', 'lib.plotcandle',
    'lib.alert.alert', 'lib.alertcondition', 'lib.na',

    # Other builtin functions
    'lib.timestamp', 'lib.dayofmonth', 'lib.dayofweek.dayofweek', 'lib.hour', 'lib.minute', 'lib.month',
    'lib.second', 'lib.weekofyear', 'lib.year', 'lib.time', 'lib.time_close', 'lib.time_tradingday',
    'lib.timenow', 'lib.is_na', 'lib.nz', 'lib.timestamp',

    # Strategy
    'lib.strategy.entry', 'lib.strategy.order', 'lib.strategy.exit', 'lib.strategy.close',
    'lib.strategy.cancel', 'lib.strategy.cancel_all',
    'lib.strategy.equity', 'lib.strategy.eventrades', 'lib.strategy.initial_capital',
    'lib.strategy.grossloss', 'lib.strategy.grossprofit', 'lib.strategy.losstrades',
    'lib.strategy.max_drawdown', 'lib.strategy.max_runup', 'lib.strategy.netprofit',
    'lib.strategy.openprofit', 'lib.strategy.position_size', 'lib.strategy.position_avg_price',
    'lib.strategy.wintrades',
    'lib.strategy.closedtrades.commission', 'lib.strategy.closedtrades.entry_bar_index',
    'lib.strategy.closedtrades.entry_comment', 'lib.strategy.closedtrades.entry_id',
    'lib.strategy.closedtrades.entry_price', 'lib.strategy.closedtrades.entry_time',
    'lib.strategy.closedtrades.exit_bar_index', 'lib.strategy.closedtrades.exit_comment',
    'lib.strategy.closedtrades.exit_id', 'lib.strategy.closedtrades.exit_price',
    'lib.strategy.closedtrades.exit_time', 'lib.strategy.closedtrades.max_drawdown',
    'lib.strategy.closedtrades.max_drawdown_percent', 'lib.strategy.closedtrades.max_runup',
    'lib.strategy.closedtrades.max_runup_percent', 'lib.strategy.closedtrades.profit',
    'lib.strategy.closedtrades.profit_percent', 'lib.strategy.closedtrades.size',
    'lib.strategy.opentrades.commission', 'lib.strategy.opentrades.entry_bar_index',
    'lib.strategy.opentrades.entry_comment', 'lib.strategy.opentrades.entry_id',
    'lib.strategy.opentrades.entry_price', 'lib.strategy.opentrades.entry_time',
    'lib.strategy.opentrades.max_drawdown', 'lib.strategy.opentrades.max_drawdown_percent',
    'lib.strategy.opentrades.max_runup', 'lib.strategy.opentrades.max_runup_percent',
    'lib.strategy.opentrades.profit', 'lib.strategy.opentrades.profit_percent',
    'lib.strategy.opentrades.size',
    'lib.strategy.opentrades.opentrades', 'lib.strategy.closedtrades.closedtrades',

    # Input functions
    'lib.input', 'lib.input.int', 'lib.input.float', 'lib.input.bool', 'lib.input.string',
    'lib.input.source', 'lib.input.color',

    # Timeframe functions
    'lib.timeframe.in_seconds', 'lib.timeframe.from_seconds',

    # Logging
    'lib.log.info', 'lib.log.error', 'lib.log.warning',

    # Math functions
    'lib.math.abs', 'lib.math.acos', 'lib.math.asin', 'lib.math.atan', 'lib.math.avg', 'lib.math.ceil', 'lib.math.cos',
    'lib.math.exp', 'lib.math.floor', 'lib.math.log', 'lib.math.log10', 'lib.math.max', 'lib.math.min', 'lib.math.pow',
    'lib.math.round', 'lib.math.round_to_mintick', 'lib.math.sign', 'lib.math.sin', 'lib.math.sqrt',
    'lib.math.tan', 'lib.math.todegrees', 'lib.math.toradians',

    # String functions
    'lib.string.contains', 'lib.string.endswith', 'lib.string.format', 'lib.string.format_time', 'lib.string.length',
    'lib.string.lower', 'lib.string.match', 'lib.string.pos', 'lib.string.repeat', 'lib.string.replace',
    'lib.string.replace_all', 'lib.string.split', 'lib.string.startswith', 'lib.string.substring',
    'lib.string.tonumber', 'lib.string.tostring', 'lib.string.trim', 'lib.string.upper',

    # Array functions
    'lib.array.abs', 'lib.array.avg', 'lib.array.binary_search', 'lib.array.binary_search_leftmost',
    'lib.array.binary_search_rightmost', 'lib.array.clear', 'lib.array.concat', 'lib.array.copy',
    'lib.array.covariance', 'lib.array.every', 'lib.array.fill', 'lib.array.first', 'lib.array.from_items',
    'lib.array.get', 'lib.array.includes', 'lib.array.indexof', 'lib.array.insert', 'lib.array.join',
    'lib.array.last', 'lib.array.lastindexof', 'lib.array.max', 'lib.array.median', 'lib.array.min',
    'lib.array.mode', 'lib.array.percentrank', 'lib.array.percentile_linear_interpolation',
    'percentile_nearest_rank', 'percentile_nearest_rank', 'lib.array.pop', 'lib.array.push', 'lib.array.range',
    'lib.array.remove', 'lib.array.reverse', 'lib.array.set', 'lib.array.shift', 'lib.array.size', 'lib.array.slice',
    'lib.array.some', 'lib.array.sort', 'lib.array.sort_indices', 'lib.array.standardize', 'lib.array.stdev',
    'lib.array.sum', 'lib.array.unshift', 'lib.array.variance', 'lib.array.new',
    'lib.array.new_bool', 'lib.array.new_color', 'lib.array.new_float', 'lib.array.new_int', 'lib.array.new_string',

    # Map functions
    'lib.map.clear', 'lib.map.contains', 'lib.map.copy', 'lib.map.get', 'lib.map.keys', 'lib.map.new',
    'lib.map.put', 'lib.map.put_all', 'lib.map.remove', 'lib.map.size', 'lib.map.values',

    # Color functions
    'lib.color.new', 'lib.color.r', 'lib.color.g', 'lib.color.b', 'lib.color.a',
    'lib.color.rgb', 'lib.color.from_gradient',

    # Strategy functions
    "lib.strategy.fixed", "lib.strategy.cash", "lib.strategy.percent_of_equity", "lib.strategy.long",
    "lib.strategy.short", 'lib.strategy.direction', "lib.strategy.cancel", "lib.strategy.cancel_all",
    "lib.strategy.close", "lib.strategy.close_all", "lib.strategy.entry", "lib.strategy.exit",
    "lib.strategy.closedtrades", "lib.strategy.opentrades",

    # Other
    'lib.max_bars_back',

    'copy', 'dataclass', 'dccopy',
    'pytest.raises',

    'method_call', 'pine_range'
}

# Call-site routes decided at transform time. Same-module defs resolve to a
# ('same', scope_id) tuple first and collapse to fast/direct through the
# carrier fixpoint.
_SKIP = 'skip'
_DIRECT = 'direct'
_FAST = 'fast'
# Cross-module state-carrying callee marked ``__pyne_shared_call_site__``:
# emitted on the fast route with the STRAIGHT-LINE shape even inside a loop.
# Every loop site shares one instance across iterations now, so the marker no
# longer decides sharing — it decides the same-bar treatment: a marked machine
# advances once per EXECUTION on TradingView (the percentile machines,
# measured with a [5,9] / [5,na,9] length-loop probe), so it must skip the
# bar-keyed rollback ``__loop_state__`` applies to everything else
# (``ta.ema``/``ta.sma`` re-derive each call from bar-start state, measured
# with window-content and per-iteration probes). The straight-line emission is
# exactly that: shared slot, no rollback; the ``per_call`` layout flag
# (see ``_collect_builtins``) shields the same machines when they sit deeper
# in a rolled-back callee subtree.
_FAST_SHARED = 'fast-shared'
_UNIFORM = 'uniform'

_Route = str | tuple[str, str]


def _is_test_function(name: str) -> bool:
    """Whether a function follows the ``__test_*__`` convention (called by
    the test framework with fixtures — must stay untouched)."""
    return name.startswith('__test_') and name.endswith('__')


def _get_func_path(func: ast.expr) -> str | None:
    """Get the full dotted path of a callee expression."""
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        parts = []
        current: ast.expr = func
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
            return '.'.join(reversed(parts))
    return None


class _ScopeIndex(ast.NodeVisitor):
    """Pass 1a: per-scope name bindings (defs, classes, everything else
    assigned) and the module-level import map."""

    def __init__(self, layout: ModuleLayout):
        self.layout = layout
        # scope -> name -> (target scope id of the LAST def, has decorators);
        # the last definition wins, like the runtime name binding does
        self.defs: dict[str, dict[str, tuple[str, bool]]] = {'': {}}
        self.classes: dict[str, set[str]] = {'': set()}
        self.assigned: dict[str, set[str]] = {'': set()}
        # name -> (module path, attribute or None)
        self.import_map: dict[str, tuple[str, str | None]] = {}
        self._stack: list[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        outer = '·'.join(self._stack)
        segment = self.layout.scope_segment(node)
        target = f'{outer}·{segment}' if outer else segment
        self.defs[outer][node.name] = (target, bool(node.decorator_list))
        self._stack.append(segment)
        scope = '·'.join(self._stack)
        self.defs.setdefault(scope, {})
        self.classes.setdefault(scope, set())
        assigned = self.assigned.setdefault(scope, set())
        args = node.args
        for arg in args.args + args.posonlyargs + args.kwonlyargs:
            assigned.add(arg.arg)
        if args.vararg:
            assigned.add(args.vararg.arg)
        if args.kwarg:
            assigned.add(args.kwarg.arg)
        self.generic_visit(node)
        self._stack.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.classes['·'.join(self._stack)].add(node.name)
        # Class bodies are not isolation scopes — don't index their content

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.assigned['·'.join(self._stack)].add(node.id)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self.assigned['·'.join(self._stack)].add(node.name)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        scope = '·'.join(self._stack)
        for alias in node.names:
            bound = alias.asname or alias.name.split('.')[0]
            if scope:
                self.assigned[scope].add(bound)
            else:
                module = alias.name if alias.asname else alias.name.split('.')[0]
                self.import_map[bound] = (module, None)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        scope = '·'.join(self._stack)
        for alias in node.names:
            bound = alias.asname or alias.name
            if scope:
                self.assigned[scope].add(bound)
            elif node.module and not node.level:
                self.import_map[bound] = (node.module, alias.name)


class _RouteCollector(ast.NodeVisitor):
    """Pass 1b: prelim route of every call site per scope, input of the
    carrier fixpoint. Mirrors the transformer's skip rules (decorators,
    defaults, class bodies, test functions are not isolation territory)."""

    def __init__(self, transformer: 'FunctionIsolationTransformer'):
        self.transformer = transformer
        self.scope_routes: dict[str, list[_Route]] = {}
        # Scopes that will get an instance-vector slot once the body is
        # emitted. The slot does not exist yet, so the fixpoint cannot see it
        # in the layout — but it makes the definition state-carrying, and a
        # caller that routed around the state parameter would call it short
        self.pin_carriers: set[str] = set()
        self._stack: list[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if _is_test_function(node.name):
            return
        self._stack.append(self.transformer.layout.scope_segment(node))
        scope = '·'.join(self._stack)
        self.scope_routes.setdefault(scope, [])
        if get_varying(node):
            self.pin_carriers.add(scope)
        for stmt in node.body:
            self.visit(stmt)
        self._stack.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        pass

    def visit_Call(self, node: ast.Call) -> None:
        self.generic_visit(node)
        if self._stack and isinstance(node.func, (ast.Name, ast.Attribute)):
            route = self.transformer.route_for_callee(node.func, self._stack)
            self.scope_routes['·'.join(self._stack)].append(route)


def _stamped_call(bound: ast.expr, node: ast.Call) -> ast.Call:
    """
    Re-emit a call site through a bound callee, keeping its Pine type.

    The uniform route builds a NEW ``Call`` around the original arguments, so
    the result type would be lost on the very node the overload pin lands on;
    the dispatcher expression it calls through is machinery and types itself.

    :param bound: The expression yielding the callable
    :param node: The original call
    :return: The rewritten call, stamped
    """
    return stamp_lowering(
        ast.Call(func=bound, args=node.args, keywords=node.keywords), get_ty(node))


class FunctionIsolationTransformer(ast.NodeTransformer):
    """Rewrite call sites to the parent-slot / anchored emission (pass 2)."""

    def __init__(self, layout: ModuleLayout):
        self.layout = layout
        self.index = _ScopeIndex(layout)
        self.carrier: dict[str, bool] = {}
        self._pin_carriers: set[str] = set()
        self._scope_stack: list[str] = []
        self._loop_depth = 0
        self._lambda_depth = 0
        self._comp_iter_depth = 0
        self._ordinals: dict[str, int] = {}
        self._used_helpers: set[str] = set()
        self._resolve_cache: dict[str, Any] = {}
        # scope -> its instance-vector slot, and the index every varying site
        # of that scope reads out of it (by node identity)
        self._pin_slots: dict[str, int] = {}
        self._pin_index: dict[str, dict[int, int]] = {}

    # --- classification ----------------------------------------------------

    def route_for_callee(self, func: ast.expr, scope_stack: list[str]) -> _Route:
        """Classify a callee expression in a scope context.

        :param func: The callee (Name or Attribute).
        :param scope_stack: Function-name path of the call site's scope.
        :return: One of the route constants or ``('same', scope_id)``.
        """
        if self._is_series_slot_method(func, scope_stack):
            # Synthetic SeriesImpl method call emitted by the Series
            # transformer (__state__[N].add / .set) — stateless by
            # construction, and an anchor could never hit anyway (a bound
            # method is a fresh object on every attribute access)
            return _SKIP
        path = _get_func_path(func)
        if path is None:
            return _UNIFORM
        if path in NON_TRANSFORMABLE_FUNCTIONS:
            return _SKIP
        parts = path.split('.')
        base = parts[0]

        # Innermost-first scope-chain resolution of the base name
        for i in range(len(scope_stack), -1, -1):
            scope = '·'.join(scope_stack[:i])
            is_assigned = base in self.index.assigned.get(scope, ())
            entry = self.index.defs.get(scope, {}).get(base)
            if entry is not None:
                target, decorated = entry
                if is_assigned or len(parts) > 1 or decorated:
                    # Rebound name, attribute on a def, or a decorated def
                    # (the runtime value is the decorator's return value —
                    # an overload dispatcher, an lru_cache wrapper, ...)
                    return _UNIFORM
                return 'same', target
            if base in self.index.classes.get(scope, ()):
                # Constructor or class attribute — the legacy runtime guard
                # returned types untouched, skipping is the same net effect
                return _SKIP if not is_assigned else _UNIFORM
            if is_assigned:
                return _UNIFORM  # local value (function value, object, ...)

        entry = self.index.import_map.get(base)
        if entry is not None:
            if is_stdlib(entry[0]):
                return _SKIP
            obj = self._resolve_imported(path, parts)
            if obj is None:
                return _UNIFORM
            route = self._classify_object(obj)
            if route == _FAST and getattr(obj, '__pyne_shared_call_site__', False):
                return _FAST_SHARED
            return route
        if len(parts) == 1 and base in vars(builtins):
            return _SKIP
        if base.startswith('_'):
            return _SKIP  # unresolvable private name — legacy parity
        return _UNIFORM

    def _is_series_slot_method(self, func: ast.expr, scope_stack: list[str]) -> bool:
        """Whether a callee is a method of a series slot
        (``__state__[N].add`` / ``__state·scope__[N].set``).

        :param func: The callee expression.
        :param scope_stack: Function-name path of the call site's scope.
        :return: True if the slot under the attribute is a series slot.
        """
        if not (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Subscript)):
            return False
        sub = func.value
        if not (isinstance(sub.value, ast.Name) and isinstance(sub.slice, ast.Constant)
                and isinstance(sub.slice.value, int)):
            return False
        param = sub.value.id
        if param == DEFAULT_STATE_PARAM:
            scope_id = '·'.join(scope_stack)
        elif param.startswith('__state·') and param.endswith('__'):
            scope_id = param[len('__state·'):-2]
        else:
            return False
        scope = self.layout.scopes.get(scope_id)
        if scope is None:
            return False
        index = sub.slice.value
        return 0 <= index < len(scope.slots) and scope.slots[index].kind == 'series'

    def _resolve_imported(self, path: str, parts: list[str]) -> Any | None:
        """Resolve a dotted callee path through the module-level import map
        at transform time (imports are cached in sys.modules)."""
        try:
            return self._resolve_cache[path]
        except KeyError:
            pass
        module_name, attr = self.index.import_map[parts[0]]
        obj: Any | None
        try:
            obj = importlib.import_module(module_name)
            for name in ([attr] if attr else []) + parts[1:]:
                try:
                    obj = getattr(obj, name)
                except AttributeError:
                    # Submodule not yet loaded: the script's own import only
                    # runs after compilation, so import it here — otherwise
                    # the route would depend on what happens to be in
                    # sys.modules and the emission would not be deterministic
                    if not isinstance(obj, types.ModuleType):
                        raise
                    obj = importlib.import_module(f'{obj.__name__}.{name}')
        except Exception:  # noqa: any resolution failure means "unprovable"
            obj = None
        self._resolve_cache[path] = obj
        return obj

    @staticmethod
    def _classify_object(obj: Any) -> str:
        """Classify a transform-time resolved callee object."""
        if isinstance(obj, Exported):
            return _UNIFORM  # the anchor's bind unwraps it
        if isinstance(obj, type):
            return _SKIP
        bound_self = getattr(obj, '__self__', None)
        if bound_self is not None and isinstance(bound_self, type):
            return _SKIP  # classmethod
        if isinstance(obj, (types.BuiltinFunctionType, types.BuiltinMethodType)):
            return _SKIP
        if getattr(obj, '__pyne_bind__', None) is not None:
            # Overload dispatcher — the implementation is chosen at runtime.
            # Must be checked BEFORE the layout: functools.wraps copies the
            # first implementation's __dict__ (its __pyne_layout__ included)
            # onto the dispatcher.
            return _UNIFORM
        if getattr(obj, '__pyne_layout__', None) is not None:
            return _FAST
        if getattr(obj, '__module_property__', False):
            return _SKIP  # Pine-style module property getter — stateless by design
        if isinstance(obj, types.FunctionType) and '__pyne_slot_layout__' in obj.__globals__:
            return _DIRECT  # transformed module, no layout -> provably stateless
        return _UNIFORM

    def _is_carrier(self, scope_id: str) -> bool:
        """Whether a same-module scope carries state (fixpoint result)."""
        try:
            return self.carrier[scope_id]
        except KeyError:
            return self._own_state(scope_id)

    def _own_state(self, scope_id: str) -> bool:
        """Whether a scope carries state of its own: allocated slots, or the
        instance vector its varying sites are about to be given."""
        return self.layout.state_carrying(scope_id) or scope_id in self._pin_carriers

    def _run_fixpoint(self, scope_routes: dict[str, list[_Route]]) -> dict[str, bool]:
        """Carrier fixpoint: a scope carries state if it has own slots or any
        non-direct call site (fast/uniform, or same-module to a carrier)."""
        carrier = {scope: self._own_state(scope) for scope in scope_routes}
        for routes in scope_routes.values():
            for route in routes:
                if isinstance(route, tuple):
                    carrier.setdefault(route[1], self._own_state(route[1]))
        changed = True
        while changed:
            changed = False
            for scope, routes in scope_routes.items():
                if carrier[scope]:
                    continue
                for route in routes:
                    if (route in (_FAST, _FAST_SHARED, _UNIFORM)
                            or (isinstance(route, tuple) and carrier.get(route[1], False))):
                        carrier[scope] = True
                        changed = True
                        break
        return carrier

    # --- emission helpers ----------------------------------------------------

    def _state_param(self) -> str:
        return self.layout.state_param('·'.join(self._scope_stack))

    @staticmethod
    def _copy_callee(func: ast.expr) -> ast.expr:
        """Fresh, attribute-free copy of a callee expression. Other
        transformers hang ``parent`` backlinks on nodes, which would make a
        ``deepcopy`` drag the entire module tree along — rebuilding from
        source sidesteps that. The reparse stamps ``lineno=1`` on every node;
        those must be overwritten with the original callee's location, or the
        lazy-resolve branch emits line-1 line events mid-statement (double
        breakpoint hits and derailed step-over on the first bar)."""
        copy = cast(ast.expr, ast.parse(ast.unparse(func), mode='eval').body)
        for node in ast.walk(copy):
            ast.copy_location(node, func)
        return copy

    @staticmethod
    def _slot_ref(param: str, slot: int) -> ast.Subscript:
        return ast.Subscript(value=ast.Name(id=param, ctx=ast.Load()),
                             slice=ast.Constant(value=slot), ctx=ast.Load())

    def _helper(self, name: str) -> str:
        """Register a runtime helper for the import and return its alias."""
        self._used_helpers.add(name)
        return HELPER_ALIASES[name]

    def _helper_call(self, name: str, args: list[ast.expr]) -> ast.Call:
        """``<helper>(<args>)`` — a whole call site folded into one helper.

        Emitted in comprehension ITERABLE expressions, where Python rejects
        every assignment expression, so the inline guard forms are illegal.
        """
        return ast.Call(func=ast.Name(id=self._helper(name), ctx=ast.Load()),
                        args=args, keywords=[])

    def _emit_fast(self, node: ast.Call, slot: int, in_loop: bool,
                   vector: ast.expr | None = None) -> ast.Call:
        """Prepend the child-state expression as the hidden first argument.

        The instance vector, where there is one, is a trailing argument of the
        state-CREATING helper — every shape here has one, and each of them is
        the cold path, so the guard the hot path runs is unchanged.
        """
        param = self._state_param()
        callee_copy = self._copy_callee(node.func)
        extra: list[ast.expr] = [] if vector is None else [vector]
        state_expr: ast.expr
        if in_loop:
            # Shared instance + same-bar builtin rollback, folded into one
            # helper call — walrus-free, so it is legal in comprehension
            # iterables too
            state_expr = self._helper_call(
                '__loop_state__', [ast.Name(id=param, ctx=ast.Load()),
                                   ast.Constant(value=slot), callee_copy] + extra)
        elif self._comp_iter_depth:
            state_expr = self._helper_call(
                '__slot_state__', [ast.Name(id=param, ctx=ast.Load()),
                                   ast.Constant(value=slot), callee_copy] + extra)
        else:
            state_expr = ast.IfExp(
                test=ast.Compare(
                    left=ast.NamedExpr(target=ast.Name(id='__st·__', ctx=ast.Store()),
                                       value=self._slot_ref(param, slot)),
                    ops=[ast.IsNot()], comparators=[ast.Constant(value=None)]),
                body=ast.Name(id='__st·__', ctx=ast.Load()),
                orelse=ast.Call(func=ast.Name(id=self._helper('__resolve_slot__'),
                                              ctx=ast.Load()),
                                args=[ast.Name(id=param, ctx=ast.Load()),
                                      ast.Constant(value=slot), callee_copy] + extra,
                                keywords=[]))
        node.args.insert(0, state_expr)
        return node

    def _emit_uniform(self, node: ast.Call, slot: int, in_loop: bool,
                      pin_expr: ast.expr | None = None,
                      vector: ast.expr | None = None) -> ast.Call:
        """Wrap the call in the anchored bind form.

        Two trailing arguments ride along, and they are different quantities.
        The overload pin reaches the dispatcher's ``__pyne_bind__`` factory,
        which resolves it ONCE per anchor; a callee that publishes no such
        factory ignores it. The instance vector configures the state vector the
        binder creates for the callee. Both are absent on all but the sites
        that have one, so the emitted form is unchanged wherever there is
        nothing to say — and the vector needs the pin position filled in, since
        the binder takes them positionally.
        """
        param = self._state_param()
        pin: list[ast.expr] = []
        if pin_expr is not None:
            pin.append(pin_expr)
        if vector is not None:
            if not pin:
                pin.append(ast.Constant(value=None))
            pin.append(vector)
        if in_loop or self._comp_iter_depth:
            # Helper form: the callee expression is evaluated exactly once, as
            # the helper's argument, so a callee that runs code needs no
            # temporary and the identity check needs no walrus
            bound: ast.expr = self._helper_call(
                '__bind_loop__' if in_loop else '__bind_slot__',
                [ast.Name(id=param, ctx=ast.Load()),
                 ast.Constant(value=slot), node.func] + pin)
            return _stamped_call(bound, node)
        callee: ast.expr
        callee_copy: ast.expr
        bind: ast.expr | None = None
        if any(isinstance(n, ast.Call) for n in ast.walk(node.func)):
            # The callee expression RUNS CODE (a call hides in it), so the two
            # copies below would execute it twice -- side effects included, and
            # with the nested call's own state site duplicated. Bind it to a
            # temporary first and reference the name in both places.
            # The conjunct is a tautology ON PURPOSE: it must never short-circuit
            # past the ``__b·__`` walrus in the next conjunct.
            bind = ast.Compare(
                left=ast.NamedExpr(target=ast.Name(id='__c·__', ctx=ast.Store()),
                                   value=node.func),
                ops=[ast.Is()], comparators=[ast.Name(id='__c·__', ctx=ast.Load())])
            # Separate nodes on purpose -- AST nodes must not be shared
            callee = ast.Name(id='__c·__', ctx=ast.Load())
            callee_copy = ast.Name(id='__c·__', ctx=ast.Load())
        else:
            callee, callee_copy = node.func, self._copy_callee(node.func)
        pair = ast.Name(id='__b·__', ctx=ast.Load())
        bind_any = self._helper('__bind_any__')
        test: ast.BoolOp = ast.BoolOp(op=ast.And(), values=[
            ast.Compare(
                left=ast.NamedExpr(target=ast.Name(id='__b·__', ctx=ast.Store()),
                                   value=self._slot_ref(param, slot)),
                ops=[ast.IsNot()], comparators=[ast.Constant(value=None)]),
            ast.Compare(
                left=ast.Subscript(value=pair, slice=ast.Constant(value=0), ctx=ast.Load()),
                ops=[ast.Is()], comparators=[callee]),
        ])
        rebind: ast.expr = ast.Call(
            func=ast.Name(id=bind_any, ctx=ast.Load()),
            args=[ast.Name(id=param, ctx=ast.Load()), ast.Constant(value=slot),
                  callee_copy] + pin,
            keywords=[])
        if bind is not None:
            # FIRST operand: the callee (and any call site nested in it) must be
            # fully evaluated before the anchor read below binds ``__b·__`` --
            # an inner site writes that same name, and reading it first would
            # invoke the inner anchor's callable instead of this one
            test.values.insert(0, bind)
        bound = ast.IfExp(
            test=test,
            body=ast.Subscript(value=ast.Name(id='__b·__', ctx=ast.Load()),
                               slice=ast.Constant(value=1), ctx=ast.Load()),
            orelse=rebind)
        return _stamped_call(bound, node)

    # --- visitors ------------------------------------------------------------

    def visit_Module(self, node: ast.Module) -> ast.Module:
        self.layout.assign_scope_ids(node)
        self.index = _ScopeIndex(self.layout)
        self.index.visit(node)
        collector = _RouteCollector(self)
        collector.visit(node)
        self._pin_carriers = collector.pin_carriers
        self.carrier = self._run_fixpoint(collector.scope_routes)

        node = cast(ast.Module, self.generic_visit(node))

        if self._used_helpers:
            import_stmt = ast.ImportFrom(
                module='pynecore.core.instance_state',
                names=[ast.alias(name=name, asname=HELPER_ALIASES[name])
                       for name in sorted(self._used_helpers)],
                level=0)
            insert_pos = 0
            first = node.body[0] if node.body else None
            if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                insert_pos = 1
            for i in range(insert_pos, len(node.body)):
                if isinstance(node.body[i], (ast.Import, ast.ImportFrom)):
                    insert_pos = i + 1
                elif not isinstance(node.body[i], ast.Expr):
                    break
            node.body.insert(insert_pos, import_stmt)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        return node  # class bodies stay raw (no hidden-parameter injection path)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        if _is_test_function(node.name):
            return node
        self._scope_stack.append(self.layout.scope_segment(node))
        scope = '·'.join(self._scope_stack)
        scope_layout = scope_for_function(self.layout, scope, node)
        varying = get_varying(node)
        if varying:
            # Reserved BEFORE the body is walked, so an inner site can address
            # the slot with a literal index while the body is being emitted
            self._pin_index[scope] = {id(call): j for j, call in enumerate(varying)}
            self._pin_slots[scope] = scope_layout.add_pin(len(varying))
        old_loop, self._loop_depth = self._loop_depth, 0
        old_lambda, self._lambda_depth = self._lambda_depth, 0

        # Only the body is isolation territory (decorators and argument
        # defaults are evaluated outside the instance, legacy parity)
        node.body = [cast(ast.stmt, self.visit(stmt)) for stmt in node.body]

        self._loop_depth, self._lambda_depth = old_loop, old_lambda
        self._scope_stack.pop()
        return node

    def visit_For(self, node: ast.For) -> ast.For:
        node.iter = cast(ast.expr, self.visit(node.iter))
        self._loop_depth += 1
        node.body = [cast(ast.stmt, self.visit(stmt)) for stmt in node.body]
        node.orelse = [cast(ast.stmt, self.visit(stmt)) for stmt in node.orelse]
        self._loop_depth -= 1
        return node

    def visit_While(self, node: ast.While) -> ast.While:
        self._loop_depth += 1
        node.test = cast(ast.expr, self.visit(node.test))
        node.body = [cast(ast.stmt, self.visit(stmt)) for stmt in node.body]
        node.orelse = [cast(ast.stmt, self.visit(stmt)) for stmt in node.orelse]
        self._loop_depth -= 1
        return node

    def _visit_comprehension(self, node: ast.expr) -> ast.expr:
        """Comprehension parts run per element — loop context (walruses bind
        in the enclosing function scope per PEP 572, so counters work).

        The generator ITERABLE expressions are the exception, twice over. The
        outermost one is evaluated once, in the enclosing scope, so it is not a
        loop site at all. And Python rejects an assignment expression anywhere
        inside ANY of them — a lambda or a nested comprehension in there does
        not lift the ban — so every call site under an iterable has to use the
        walrus-free helper emission instead.
        """
        generators: list[ast.comprehension] = getattr(node, 'generators')
        for depth, generator in enumerate(generators):
            self._loop_depth += bool(depth)  # only a LATER iterable runs per element
            self._comp_iter_depth += 1
            generator.iter = cast(ast.expr, self.visit(generator.iter))
            self._comp_iter_depth -= 1
            self._loop_depth -= bool(depth)

        self._loop_depth += 1
        for generator in generators:
            generator.target = cast(ast.expr, self.visit(generator.target))
            generator.ifs = [cast(ast.expr, self.visit(test)) for test in generator.ifs]
        if isinstance(node, ast.DictComp):
            node.key = cast(ast.expr, self.visit(node.key))
            node.value = cast(ast.expr, self.visit(node.value))
        else:
            elt = cast(ast.ListComp | ast.SetComp | ast.GeneratorExp, node)
            elt.elt = cast(ast.expr, self.visit(elt.elt))
        self._loop_depth -= 1
        return node

    visit_ListComp = _visit_comprehension
    visit_SetComp = _visit_comprehension
    visit_DictComp = _visit_comprehension
    visit_GeneratorExp = _visit_comprehension

    def visit_Lambda(self, node: ast.Lambda) -> ast.Lambda:
        self._lambda_depth += 1
        node.body = cast(ast.expr, self.visit(node.body))
        self._lambda_depth -= 1
        return node

    def visit_Call(self, node: ast.Call) -> ast.expr:
        node.args = [cast(ast.expr, self.visit(arg)) for arg in node.args]
        node.keywords = [cast(ast.keyword, self.visit(kw)) for kw in node.keywords]
        if not isinstance(node.func, (ast.Name, ast.Attribute)):
            # Immediately-called expressions stay raw (legacy parity), but
            # calls inside the callee expression still get their own sites
            node.func = cast(ast.expr, self.visit(node.func))
            return node
        if isinstance(node.func, ast.Attribute) and _get_func_path(node.func) is None:
            # An attribute chain that does not bottom out in a name (``f().g``,
            # ``a[f()].g``, ...) hides an arbitrary expression, and the calls in
            # there are call sites of their own -- without this the state
            # argument of a nested state-carrying callee is never passed
            node.func.value = cast(ast.expr, self.visit(node.func.value))

        route = self.route_for_callee(node.func, self._scope_stack)
        if not self._scope_stack:
            if route in (_FAST, _FAST_SHARED) \
                    or (isinstance(route, tuple) and self._is_carrier(route[1])):
                raise SyntaxError("Stateful function calls are not supported at module level")
            pin = get_pin(node)
            if pin is not None and route == _UNIFORM:
                # An overload site up here has no anchor to resolve its pin in,
                # so it binds once, in place: the module body runs once, and
                # without the binding the site would dispatch from the VALUES
                # and take the float implementation for an int-typed argument
                return _stamped_call(self._helper_call(
                    '__bind_pinned__', [node.func, ast.Constant(value=pin)]), node)
            return node
        if isinstance(route, tuple):
            route = _FAST if self._is_carrier(route[1]) else _DIRECT
        if route in (_SKIP, _DIRECT):
            return node

        scope = '·'.join(self._scope_stack)
        if self._lambda_depth:
            # A lambda body runs at its consumer's cadence, not as a loop
            # iteration — the straight-line anchor keeps the shared state
            # without the loop form's same-bar rollback
            route, in_loop = _UNIFORM, False
        elif route == _FAST_SHARED:
            # A marked machine advances per EXECUTION on TradingView, so it
            # must skip the loop form's same-bar rollback: the straight-line
            # shape already shares its slot across iterations (see
            # ``_FAST_SHARED``)
            route, in_loop = _FAST, False
        else:
            in_loop = self._loop_depth > 0

        ordinal = self._ordinals.get(scope, 0)
        self._ordinals[scope] = ordinal + 1
        call_id = f'{scope}·{_get_func_path(node.func) or "<callee>"}·{ordinal}'
        scope_layout = self.layout.scope(scope)
        pin_expr, vector = self._channel_args(node, scope)
        if route == _FAST:
            slot = scope_layout.add_child(call_id, in_loop=in_loop)
            return self._emit_fast(node, slot, in_loop, vector)
        slot = scope_layout.add_anchor(call_id, in_loop=in_loop)
        return self._emit_uniform(node, slot, in_loop, pin_expr, vector)

    def _channel_args(self, node: ast.Call,
                      scope: str) -> tuple[ast.expr | None, ast.expr | None]:
        """The overload pin and the instance vector this call site passes on.

        Either is a constant where the type pass settled it for every instance
        of the enclosing body. Where it could not, the value belongs to the
        INSTANCE and lives in the caller's own vector, so the site reads its
        entry out of the slot instead: ``__state__[k][j]``, with ``j`` the
        site's index in that vector. A site is one kind or the other — an
        overload group is never instantiated per call site, so it never carries
        a vector, and a context-analysed callee has no overload to pin.

        :param node: The call node
        :param scope: Scope id of the call site
        :return: (pin expression, vector expression); either may be None
        """
        stamped = get_pin(node)
        constant = get_vector(node)
        pin: ast.expr | None = None if stamped is None else ast.Constant(value=stamped)
        vector: ast.expr | None = None if constant is None else ast.Constant(value=constant)
        index = self._pin_index.get(scope)
        position = index.get(id(node)) if index is not None else None
        if position is not None:
            entry = ast.Subscript(
                value=self._slot_ref(self._state_param(), self._pin_slots[scope]),
                slice=ast.Constant(value=position), ctx=ast.Load())
            if get_pins(node) is not None:
                pin = entry
            else:
                vector = entry
        return pin, vector
