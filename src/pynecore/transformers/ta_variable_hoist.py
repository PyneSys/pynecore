import ast
from typing import cast

# The stateful ``ta`` builtin variables. Their accumulators live in per-call-site
# Persistent state, but TradingView keeps ONE engine-level series per builtin
# variable: a read inside an ``if`` returns the same value as an unconditional one.
# Measured (probes m570/m572, BINANCE:BTCUSDT 30m, 28505+ bars): gated reads of
# nvi/wad/pvt and accdist/obv/pvi/vwap agree with an unconditional reference on
# every gated bar. ``iii`` and ``wvad`` are pure functions of the current bar, so
# a gated read already matches without any hoist; ``tr`` reads the runner's global
# ``lib._last_close`` window and is likewise stateless per bar, and calling it in
# place is cheaper than a mandatory per-bar evaluation.
HOISTED_TA_VARIABLES = frozenset({'accdist', 'nvi', 'obv', 'pvi', 'pvt', 'wad', 'vwap'})


class TaVariableHoistTransformer(ast.NodeTransformer):
    """
    Evaluate referenced stateful ``ta`` builtin variables once per bar, at the top
    of ``main``.

    Runs right after the ``ModulePropertyTransformer``, so a bare ``ta.nvi`` read is
    already a zero-argument ``lib.ta.nvi()`` call here. Every such call is replaced
    with a read of a module-global cache name, and ``main`` gets an unconditional
    prologue assigning each referenced cache from the real property call::

        global __ta·nvi
        __ta·nvi = lib.ta.nvi()

    The prologue call is an ordinary call site for the later series/persistent/
    isolation passes, so the variable's state advances exactly once per bar with a
    stable identity, no matter how many reference sites the script has or how they
    are gated. Zero-argument calls only: ``ta.vwap(src)`` and friends are the
    function forms with their own Pine semantics and stay untouched.

    Modules without a top-level ``main`` (library modules) are left alone — there
    is no per-bar entry point to hoist into.
    """

    def __init__(self):
        self.hoisted: list[str] = []  # referenced variable names, first-reference order

    @staticmethod
    def _cache_name(name: str) -> str:
        # Middle-dot namespace: reserved for the transformers, user scripts
        # spelling it are rejected before the pipeline runs
        return f'__ta·{name}'

    def visit_Module(self, node: ast.Module) -> ast.Module:
        main = next((stmt for stmt in node.body
                     if isinstance(stmt, ast.FunctionDef) and stmt.name == 'main'), None)
        if main is None:
            return node

        node = cast(ast.Module, self.generic_visit(node))
        if not self.hoisted:
            return node

        prologue: list[ast.stmt] = [
            ast.Global(names=[self._cache_name(name) for name in self.hoisted])
        ]
        for name in self.hoisted:
            prologue.append(ast.Assign(
                targets=[ast.Name(id=self._cache_name(name), ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Attribute(value=ast.Name(id='lib', ctx=ast.Load()),
                                            attr='ta', ctx=ast.Load()),
                        attr=name, ctx=ast.Load()),
                    args=[], keywords=[])))

        # Keep the docstring first; everything else in main may already read the caches
        insert_at = 0
        if (main.body and isinstance(main.body[0], ast.Expr)
                and isinstance(main.body[0].value, ast.Constant)
                and isinstance(main.body[0].value.value, str)):
            insert_at = 1
        main.body[insert_at:insert_at] = prologue
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = cast(ast.Call, self.generic_visit(node))
        if node.args or node.keywords:
            return node
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr in HOISTED_TA_VARIABLES):
            return node
        module = func.value
        if not (isinstance(module, ast.Attribute) and module.attr == 'ta'
                and isinstance(module.value, ast.Name) and module.value.id == 'lib'):
            return node
        if func.attr not in self.hoisted:
            self.hoisted.append(func.attr)
        return ast.copy_location(ast.Name(id=self._cache_name(func.attr), ctx=ast.Load()), node)
