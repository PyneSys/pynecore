"""
Transform function call sites to the slot-based instance-state scheme.

Every isolated call site is classified at TRANSFORM time and emitted on one
of three routes (see ``work/benchmark`` plan, section 3.4):

- **fast** (provably state-carrying callee): the child instance's state
  lives in a compile-time-assigned slot of the CALLER's state vector::

      f((__st__ if (__st__ := __state__[5]) is not None
         else __resolve_slot__(__state__, 5, f)), x, 12)

  Loop-shaped sites hold a child list indexed by a per-invocation counter
  (hoisted to the function prologue together with the list)::

      f((__chl_0__[__i__] if (__i__ := (__cnt_0__ := __cnt_0__ + 1) - 1)
         < len(__chl_0__) else __grow__(__chl_0__, f)), x)

- **direct** (provably stateless callee): plain call, zero overhead.

- **uniform** (anything not provable): the caller anchors a
  ``(callee, bound)`` pair in its own slot; the hot path is one identity
  check, ``__bind_any__`` / ``__bind_any_loop__`` (re)binds on a miss::

      (__b__[1] if (__b__ := __state__[7]) is not None and __b__[0] is f
       else __bind_any__(__state__, 7, f))(x)

Classification sources:

- same-module functions: the shared :class:`ModuleLayout` (this transformer
  must run AFTER the Persistent and Series transformers) plus a carrier
  fixpoint over the module's call graph — a function carries state if it has
  own slots or any non-direct call site;
- cross-module callees (``lib.*``, user Pyne libraries): the callee module
  is imported at transform time and the object inspected —
  ``__pyne_layout__`` proves state-carrying, a ``__pyne_slot_layout__``
  marker in the function's globals with no layout attribute proves
  stateless, ``overload`` dispatchers and everything else fall to uniform.

Unprovable always degrades to uniform (correct, only slower) — an error can
only come from a false proof, never from missing knowledge.

Deliberately left untouched (raw calls): module-level call sites (a stateful
callee there raises a transform error), decorator and default-argument
expressions, class bodies, ``__test_*__`` functions (the test framework
calls them with fixtures, they must not grow a hidden parameter), and calls
whose callee is not a plain name/attribute. Calls inside lambdas are
anchored on the straight-line uniform route (a loop counter would bind
lambda-local and break).
"""
from typing import cast, Any
import ast
import builtins
import importlib
import types

from ..core.pine_export import Exported
from ..utils.stdlib_checker import stdlib_checker
from .slot_layout import ModuleLayout, scope_for_function

__all__ = ['FunctionIsolationTransformer', 'NON_TRANSFORMABLE_FUNCTIONS']

# Functions that should not be transformed because they:
# - don't return anything (plotting, display)
# - can't have Series values
# - are purely for output/display purposes
# This makes code run little bit faster
NON_TRANSFORMABLE_FUNCTIONS = {
    # Plot and display related
    'lib.plot', 'lib.plotchar', 'lib.plotshape', 'lib.plotarrow',
    'lib.label', 'lib.table', 'lib.box', 'lib.line', 'lib.hline',
    'lib.fill', 'lib.bgcolor', 'lib.barcolor', 'lib.plotcandle',
    'lib.alert', 'lib.alertcondition', 'lib.na',

    # Other builtin functions
    'lib.timestamp', 'lib.dayofmonth', 'lib.dayofweek', 'lib.hour', 'lib.minute', 'lib.month', 'lib.second',
    'lib.weekofyear', 'lib.year', 'lib.time', 'lib.time_close', 'lib.time_tradingday', 'lib.timenow',
    'lib.is_na', 'lib.nz', 'lib.timestamp',

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

    def __init__(self):
        self.defs: dict[str, set[str]] = {'': set()}
        self.classes: dict[str, set[str]] = {'': set()}
        self.assigned: dict[str, set[str]] = {'': set()}
        # name -> (module path, attribute or None)
        self.import_map: dict[str, tuple[str, str | None]] = {}
        self._stack: list[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.defs['·'.join(self._stack)].add(node.name)
        self._stack.append(node.name)
        scope = '·'.join(self._stack)
        self.defs.setdefault(scope, set())
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
        self._stack: list[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if _is_test_function(node.name):
            return
        self._stack.append(node.name)
        self.scope_routes.setdefault('·'.join(self._stack), [])
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


class FunctionIsolationTransformer(ast.NodeTransformer):
    """Rewrite call sites to the parent-slot / anchored emission (pass 2)."""

    def __init__(self, layout: ModuleLayout):
        self.layout = layout
        self.index = _ScopeIndex()
        self.carrier: dict[str, bool] = {}
        self._scope_stack: list[str] = []
        self._loop_depth = 0
        self._lambda_depth = 0
        self._ordinals: dict[str, int] = {}
        # per-function pending loop hoists: (counter name, list name, slot)
        self._loop_hoists: list[list[tuple[str, str, int]]] = []
        self._used_helpers: set[str] = set()
        self._resolve_cache: dict[str, Any] = {}

    # --- classification ----------------------------------------------------

    def route_for_callee(self, func: ast.expr, scope_stack: list[str]) -> _Route:
        """Classify a callee expression in a scope context.

        :param func: The callee (Name or Attribute).
        :param scope_stack: Function-name path of the call site's scope.
        :return: One of the route constants or ``('same', scope_id)``.
        """
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
            if base in self.index.defs.get(scope, ()):
                if is_assigned or len(parts) > 1:
                    return _UNIFORM  # rebound name / attribute on a def
                return 'same', (f'{scope}·{base}' if scope else base)
            if base in self.index.classes.get(scope, ()):
                # Constructor or class attribute — the legacy runtime guard
                # returned types untouched, skipping is the same net effect
                return _SKIP if not is_assigned else _UNIFORM
            if is_assigned:
                return _UNIFORM  # local value (function value, object, ...)

        entry = self.index.import_map.get(base)
        if entry is not None:
            try:
                is_stdlib = stdlib_checker.is_stdlib(entry[0].split('.')[0])
            except Exception:  # noqa: BLE001 - e.g. dynamic modules without __spec__
                is_stdlib = False
            if is_stdlib:
                return _SKIP
            obj = self._resolve_imported(path, parts)
            return self._classify_object(obj) if obj is not None else _UNIFORM
        if len(parts) == 1 and base in vars(builtins):
            return _SKIP
        if base.startswith('_'):
            return _SKIP  # unresolvable private name — legacy parity
        return _UNIFORM

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
                obj = getattr(obj, name)
        except Exception:  # noqa: BLE001 - any resolution failure means "unprovable"
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
        if getattr(obj, '__pyne_layout__', None) is not None:
            return _FAST
        if getattr(obj, '__module_property__', False):
            return _SKIP  # Pine-style module property getter — stateless by design
        code = getattr(obj, '__code__', None)
        if code is not None and code.co_qualname == 'overload.<locals>.dispatcher':
            return _UNIFORM  # implementation chosen at runtime
        if isinstance(obj, types.FunctionType) and '__pyne_slot_layout__' in obj.__globals__:
            return _DIRECT  # transformed module, no layout -> provably stateless
        return _UNIFORM

    def _is_carrier(self, scope_id: str) -> bool:
        """Whether a same-module scope carries state (fixpoint result)."""
        try:
            return self.carrier[scope_id]
        except KeyError:
            return self.layout.state_carrying(scope_id)

    def _run_fixpoint(self, scope_routes: dict[str, list[_Route]]) -> dict[str, bool]:
        """Carrier fixpoint: a scope carries state if it has own slots or any
        non-direct call site (fast/uniform, or same-module to a carrier)."""
        carrier = {scope: self.layout.state_carrying(scope) for scope in scope_routes}
        for routes in scope_routes.values():
            for route in routes:
                if isinstance(route, tuple):
                    carrier.setdefault(route[1], self.layout.state_carrying(route[1]))
        changed = True
        while changed:
            changed = False
            for scope, routes in scope_routes.items():
                if carrier[scope]:
                    continue
                for route in routes:
                    if (route in (_FAST, _UNIFORM)
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
        source sidesteps that."""
        return cast(ast.expr, ast.parse(ast.unparse(func), mode='eval').body)

    @staticmethod
    def _slot_ref(param: str, slot: int) -> ast.Subscript:
        return ast.Subscript(value=ast.Name(id=param, ctx=ast.Load()),
                             slice=ast.Constant(value=slot), ctx=ast.Load())

    @staticmethod
    def _counter_walrus(counter: str) -> ast.NamedExpr:
        """``(__i__ := (<counter> := <counter> + 1) - 1)``"""
        increment = ast.NamedExpr(
            target=ast.Name(id=counter, ctx=ast.Store()),
            value=ast.BinOp(left=ast.Name(id=counter, ctx=ast.Load()),
                            op=ast.Add(), right=ast.Constant(value=1)))
        return ast.NamedExpr(
            target=ast.Name(id='__i__', ctx=ast.Store()),
            value=ast.BinOp(left=increment, op=ast.Sub(), right=ast.Constant(value=1)))

    def _add_loop_hoist(self, slot: int) -> tuple[str, str]:
        """Register a loop site's counter + hoisted list for the prologue."""
        k = len(self._loop_hoists[-1])
        counter, children = f'__cnt_{k}__', f'__chl_{k}__'
        self._loop_hoists[-1].append((counter, children, slot))
        return counter, children

    def _emit_fast(self, node: ast.Call, slot: int, in_loop: bool) -> ast.Call:
        """Prepend the child-state expression as the hidden first argument."""
        param = self._state_param()
        callee_copy = self._copy_callee(node.func)
        if not in_loop:
            self._used_helpers.add('__resolve_slot__')
            state_expr = ast.IfExp(
                test=ast.Compare(
                    left=ast.NamedExpr(target=ast.Name(id='__st__', ctx=ast.Store()),
                                       value=self._slot_ref(param, slot)),
                    ops=[ast.IsNot()], comparators=[ast.Constant(value=None)]),
                body=ast.Name(id='__st__', ctx=ast.Load()),
                orelse=ast.Call(func=ast.Name(id='__resolve_slot__', ctx=ast.Load()),
                                args=[ast.Name(id=param, ctx=ast.Load()),
                                      ast.Constant(value=slot), callee_copy],
                                keywords=[]))
        else:
            self._used_helpers.add('__grow__')
            counter, children = self._add_loop_hoist(slot)
            state_expr = ast.IfExp(
                test=ast.Compare(
                    left=self._counter_walrus(counter), ops=[ast.Lt()],
                    comparators=[ast.Call(func=ast.Name(id='len', ctx=ast.Load()),
                                          args=[ast.Name(id=children, ctx=ast.Load())],
                                          keywords=[])]),
                body=ast.Subscript(value=ast.Name(id=children, ctx=ast.Load()),
                                   slice=ast.Name(id='__i__', ctx=ast.Load()), ctx=ast.Load()),
                orelse=ast.Call(func=ast.Name(id='__grow__', ctx=ast.Load()),
                                args=[ast.Name(id=children, ctx=ast.Load()), callee_copy],
                                keywords=[]))
        node.args.insert(0, state_expr)
        return node

    def _emit_uniform(self, node: ast.Call, slot: int, in_loop: bool) -> ast.Call:
        """Wrap the call in the anchored bind form."""
        param = self._state_param()
        callee, callee_copy = node.func, self._copy_callee(node.func)
        pair = ast.Name(id='__b__', ctx=ast.Load())
        if not in_loop:
            self._used_helpers.add('__bind_any__')
            test: ast.expr = ast.BoolOp(op=ast.And(), values=[
                ast.Compare(
                    left=ast.NamedExpr(target=ast.Name(id='__b__', ctx=ast.Store()),
                                       value=self._slot_ref(param, slot)),
                    ops=[ast.IsNot()], comparators=[ast.Constant(value=None)]),
                ast.Compare(
                    left=ast.Subscript(value=pair, slice=ast.Constant(value=0), ctx=ast.Load()),
                    ops=[ast.Is()], comparators=[callee]),
            ])
            rebind: ast.expr = ast.Call(
                func=ast.Name(id='__bind_any__', ctx=ast.Load()),
                args=[ast.Name(id=param, ctx=ast.Load()), ast.Constant(value=slot), callee_copy],
                keywords=[])
        else:
            self._used_helpers.add('__bind_any_loop__')
            counter, children = self._add_loop_hoist(slot)
            test = ast.BoolOp(op=ast.And(), values=[
                ast.Compare(
                    left=self._counter_walrus(counter), ops=[ast.Lt()],
                    comparators=[ast.Call(func=ast.Name(id='len', ctx=ast.Load()),
                                          args=[ast.Name(id=children, ctx=ast.Load())],
                                          keywords=[])]),
                ast.Compare(
                    left=ast.Subscript(
                        value=ast.NamedExpr(
                            target=ast.Name(id='__b__', ctx=ast.Store()),
                            value=ast.Subscript(value=ast.Name(id=children, ctx=ast.Load()),
                                                slice=ast.Name(id='__i__', ctx=ast.Load()),
                                                ctx=ast.Load())),
                        slice=ast.Constant(value=0), ctx=ast.Load()),
                    ops=[ast.Is()], comparators=[callee]),
            ])
            rebind = ast.Call(
                func=ast.Name(id='__bind_any_loop__', ctx=ast.Load()),
                args=[ast.Name(id=children, ctx=ast.Load()),
                      ast.Name(id='__i__', ctx=ast.Load()), callee_copy],
                keywords=[])
        bound = ast.IfExp(
            test=test,
            body=ast.Subscript(value=ast.Name(id='__b__', ctx=ast.Load()),
                               slice=ast.Constant(value=1), ctx=ast.Load()),
            orelse=rebind)
        return ast.Call(func=bound, args=node.args, keywords=node.keywords)

    # --- visitors ------------------------------------------------------------

    def visit_Module(self, node: ast.Module) -> ast.Module:
        self.index = _ScopeIndex()
        self.index.visit(node)
        collector = _RouteCollector(self)
        collector.visit(node)
        self.carrier = self._run_fixpoint(collector.scope_routes)

        node = cast(ast.Module, self.generic_visit(node))

        if self._used_helpers:
            import_stmt = ast.ImportFrom(
                module='pynecore.core.instance_state',
                names=[ast.alias(name=name, asname=None)
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
        self._scope_stack.append(node.name)
        scope = '·'.join(self._scope_stack)
        scope_for_function(self.layout, scope, node)
        old_loop, self._loop_depth = self._loop_depth, 0
        old_lambda, self._lambda_depth = self._lambda_depth, 0
        self._loop_hoists.append([])

        # Only the body is isolation territory (decorators and argument
        # defaults are evaluated outside the instance, legacy parity)
        node.body = [cast(ast.stmt, self.visit(stmt)) for stmt in node.body]

        hoists = self._loop_hoists.pop()
        if hoists:
            param = self.layout.state_param(scope)
            prologue: list[ast.stmt] = []
            for counter, children, slot in hoists:
                prologue.append(ast.Assign(
                    targets=[ast.Name(id=counter, ctx=ast.Store())],
                    value=ast.Constant(value=0)))
                prologue.append(ast.Assign(
                    targets=[ast.Name(id=children, ctx=ast.Store())],
                    value=self._slot_ref(param, slot)))
            insert_pos = 0
            first = node.body[0] if node.body else None
            if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                insert_pos = 1
            node.body[insert_pos:insert_pos] = prologue

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

    def _visit_comprehension(self, node: ast.AST) -> ast.AST:
        """Comprehension parts run per element — loop context (walruses bind
        in the enclosing function scope per PEP 572, so counters work)."""
        self._loop_depth += 1
        node = self.generic_visit(node)
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

        route = self.route_for_callee(node.func, self._scope_stack)
        if not self._scope_stack:
            if route == _FAST or (isinstance(route, tuple) and self._is_carrier(route[1])):
                raise SyntaxError("Stateful function calls are not supported at module level")
            return node
        if isinstance(route, tuple):
            route = _FAST if self._is_carrier(route[1]) else _DIRECT
        if route in (_SKIP, _DIRECT):
            return node

        scope = '·'.join(self._scope_stack)
        if self._lambda_depth:
            # Loop counters would bind lambda-local; the straight-line
            # anchor is the only emission that stays correct inside a lambda
            route, in_loop = _UNIFORM, False
        else:
            in_loop = self._loop_depth > 0

        ordinal = self._ordinals.get(scope, 0)
        self._ordinals[scope] = ordinal + 1
        call_id = f'{scope}·{_get_func_path(node.func) or "<callee>"}·{ordinal}'
        scope_layout = self.layout.scope(scope)
        if route == _FAST:
            slot = scope_layout.add_child(call_id, in_loop=in_loop)
            return self._emit_fast(node, slot, in_loop)
        slot = scope_layout.add_anchor(call_id, in_loop=in_loop)
        return self._emit_uniform(node, slot, in_loop)
