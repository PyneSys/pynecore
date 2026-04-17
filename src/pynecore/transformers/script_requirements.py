"""
Detect broker capability requirements of a strategy script at compile time.

Scans the module AST for calls to ``strategy.entry``, ``strategy.exit``,
``strategy.order``, ``strategy.close``, and ``strategy.close_all``, and
from the keyword arguments present at each call site deduces which
:class:`~pynecore.core.broker.models.ScriptRequirements` flags the script
needs.

The detected :class:`ScriptRequirements` is injected as the
``_broker_requirements`` keyword of the ``@script.strategy(...)`` decorator
call on the script's ``main`` function, so the :class:`Script` object
carries the requirements at runtime — no need for a second AST pass or
metadata side channel.

Detection is **conservative**: if the keyword is syntactically present
(even with an ``na`` value), the requirement is taken to be needed. Better
to refuse to start against an under-capable exchange than to fail on the
first unexpected bar in live trading.
"""
from __future__ import annotations

import ast

__all__ = ['ScriptRequirementsTransformer']

# Flag names on ScriptRequirements — kept in sync with the dataclass in
# pynecore.core.broker.models.
_FLAG_MARKET = 'market_orders'
_FLAG_LIMIT = 'limit_orders'
_FLAG_STOP = 'stop_orders'
_FLAG_STOP_LIMIT = 'stop_limit_orders'
_FLAG_BRACKET = 'tp_sl_bracket'
_FLAG_TRAIL = 'trailing_stop'
_FLAG_STRATEGY_ORDER = 'strategy_order'
_FLAG_EXIT_ORDERS = 'exit_orders'


def _strategy_call_name(node: ast.Call) -> str | None:
    """
    Return ``"entry"`` / ``"exit"`` / ``"order"`` / ``"close"`` / ``"close_all"``
    if ``node`` is a call to ``(lib.)strategy.<that>``, else ``None``.

    Matches both ``strategy.entry(...)`` (when the script imported
    ``strategy`` directly) and ``lib.strategy.entry(...)`` (the form that
    earlier transformers like ``ImportNormalizer`` may produce).
    """
    func = node.func
    if not isinstance(func, ast.Attribute):
        return None
    method = func.attr
    parent = func.value
    # strategy.<method>
    if isinstance(parent, ast.Name) and parent.id == 'strategy':
        return method
    # lib.strategy.<method>
    if isinstance(parent, ast.Attribute) and parent.attr == 'strategy':
        grandparent = parent.value
        if isinstance(grandparent, ast.Name) and grandparent.id == 'lib':
            return method
    return None


def _kw_names(node: ast.Call) -> set[str]:
    """Keyword argument names syntactically present on the call."""
    return {kw.arg for kw in node.keywords if kw.arg is not None}


def _is_script_strategy_decorator(node: ast.expr) -> bool:
    """
    True if ``node`` is a ``@script.strategy(...)`` call — matches both the
    raw form and the ``@lib.script.strategy(...)`` form produced by
    :class:`ImportNormalizerTransformer`.
    """
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if not (isinstance(func, ast.Attribute) and func.attr == 'strategy'):
        return False
    parent = func.value
    # script.strategy
    if isinstance(parent, ast.Name) and parent.id == 'script':
        return True
    # lib.script.strategy
    if (isinstance(parent, ast.Attribute) and parent.attr == 'script'
            and isinstance(parent.value, ast.Name) and parent.value.id == 'lib'):
        return True
    return False


class ScriptRequirementsTransformer(ast.NodeTransformer):
    """
    Compute :class:`ScriptRequirements` for a strategy script and inject it
    into the ``@script.strategy(...)`` decorator as the
    ``_broker_requirements`` keyword argument.

    No-op on scripts that have no ``@script.strategy(...)`` decorator
    (indicator scripts).
    """

    def __init__(self) -> None:
        self._reqs: dict[str, bool] = {
            _FLAG_MARKET: False,
            _FLAG_LIMIT: False,
            _FLAG_STOP: False,
            _FLAG_STOP_LIMIT: False,
            _FLAG_BRACKET: False,
            _FLAG_TRAIL: False,
            _FLAG_STRATEGY_ORDER: False,
            _FLAG_EXIT_ORDERS: False,
        }
        self._strategy_decorator: ast.Call | None = None

    def visit_Module(self, node: ast.Module) -> ast.Module:
        self.generic_visit(node)
        if self._strategy_decorator is None:
            return node
        self._inject_requirements(node, self._strategy_decorator)
        ast.fix_missing_locations(node)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        for dec in node.decorator_list:
            if _is_script_strategy_decorator(dec):
                self._strategy_decorator = dec  # type: ignore[assignment]
                break
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        for dec in node.decorator_list:
            if _is_script_strategy_decorator(dec):
                self._strategy_decorator = dec  # type: ignore[assignment]
                break
        self.generic_visit(node)
        return node

    def visit_Call(self, node: ast.Call) -> ast.Call:
        self.generic_visit(node)
        name = _strategy_call_name(node)
        if name is None:
            return node
        kws = _kw_names(node)
        if name == 'entry':
            self._apply_entry_or_order(kws, is_strategy_order=False)
        elif name == 'order':
            self._apply_entry_or_order(kws, is_strategy_order=True)
        elif name == 'exit':
            self._apply_exit(kws)
            self._reqs[_FLAG_EXIT_ORDERS] = True
        elif name in ('close', 'close_all'):
            self._reqs[_FLAG_MARKET] = True
            self._reqs[_FLAG_EXIT_ORDERS] = True
        return node

    # === Detection rules (see design doc, "Detektálható Minták" table) ===

    def _apply_entry_or_order(self, kws: set[str], *, is_strategy_order: bool) -> None:
        has_limit = 'limit' in kws
        has_stop = 'stop' in kws
        if is_strategy_order:
            self._reqs[_FLAG_STRATEGY_ORDER] = True
        if has_limit and has_stop:
            self._reqs[_FLAG_STOP_LIMIT] = True
            self._reqs[_FLAG_LIMIT] = True
            self._reqs[_FLAG_STOP] = True
        elif has_limit:
            self._reqs[_FLAG_LIMIT] = True
        elif has_stop:
            self._reqs[_FLAG_STOP] = True
        else:
            self._reqs[_FLAG_MARKET] = True

    def _apply_exit(self, kws: set[str]) -> None:
        has_limit = 'limit' in kws
        has_stop = 'stop' in kws
        has_profit_ticks = 'profit' in kws
        has_loss_ticks = 'loss' in kws
        has_trail = (
            'trail_offset' in kws or 'trail_price' in kws or 'trail_points' in kws
        )

        # Full OCA-reduce bracket (both TP and SL)
        if (has_limit and has_stop) or (has_profit_ticks and has_loss_ticks):
            self._reqs[_FLAG_BRACKET] = True
            self._reqs[_FLAG_LIMIT] = True
            self._reqs[_FLAG_STOP] = True
        else:
            if has_limit or has_profit_ticks:
                self._reqs[_FLAG_LIMIT] = True
            if has_stop or has_loss_ticks:
                self._reqs[_FLAG_STOP] = True
        if has_trail:
            self._reqs[_FLAG_TRAIL] = True

    # === AST injection ===

    def _inject_requirements(self, module: ast.Module, decorator: ast.Call) -> None:
        """Append ``_broker_requirements=ScriptRequirements(...)`` and add an import."""
        # Build: ScriptRequirements(flag=True, ...)
        req_call = ast.Call(
            func=ast.Name(id='ScriptRequirements', ctx=ast.Load()),
            args=[],
            keywords=[
                ast.keyword(arg=flag, value=ast.Constant(value=value))
                for flag, value in self._reqs.items() if value
            ],
        )
        # Remove any existing _broker_requirements keyword (idempotency)
        decorator.keywords = [kw for kw in decorator.keywords
                              if kw.arg != '_broker_requirements']
        decorator.keywords.append(
            ast.keyword(arg='_broker_requirements', value=req_call)
        )

        # Add the import if the module does not already have it. We insert
        # as the first statement; ``ImportLifter`` runs before us, so any
        # docstring is still the zeroth statement.
        if not self._has_script_requirements_import(module):
            import_node = ast.ImportFrom(
                module='pynecore.core.broker.models',
                names=[ast.alias(name='ScriptRequirements', asname=None)],
                level=0,
            )
            # Insert after the module docstring (if any) to keep it valid
            insert_at = 0
            if (module.body and isinstance(module.body[0], ast.Expr)
                    and isinstance(module.body[0].value, ast.Constant)
                    and isinstance(module.body[0].value.value, str)):
                insert_at = 1
            module.body.insert(insert_at, import_node)

    @staticmethod
    def _has_script_requirements_import(module: ast.Module) -> bool:
        for stmt in module.body:
            if isinstance(stmt, ast.ImportFrom) and stmt.module == 'pynecore.core.broker.models':
                for alias in stmt.names:
                    if alias.name == 'ScriptRequirements':
                        return True
        return False
