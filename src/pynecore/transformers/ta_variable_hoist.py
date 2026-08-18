import ast
from typing import cast

# The stateful ``ta`` builtin variables. Their accumulators live in per-call-site
# Persistent state, but TradingView keeps ONE engine-level series per builtin
# variable in the script's global scope: a read inside an ``if`` returns the same
# value as an unconditional one. Measured (probes m570/m572, BINANCE:BTCUSDT 30m,
# 28505+ bars): gated global-scope reads of all of them agree with an
# unconditional reference on every gated bar.
#
# Function scope splits them in two (probes m573/m574, same data, a user function
# reading the variable called inside a gate):
#   - nvi/obv/pvi/pvt/wad stay engine-global INSIDE user functions too
#     (14255/14255 agreement), so their reads are rewired everywhere;
#   - vwap and accdist get a per-INSTANCE machine inside a function (594/14255
#     and 1/14255 — vwap agrees only on session-anchor bars where both reset),
#     so only their global-scope (main body) reads are rewired and a read inside
#     any other function keeps its own call-gated state.
#
# ``iii`` and ``wvad`` are pure functions of the current bar, so a gated read
# already matches without any hoist; ``tr`` reads the runner's global
# ``lib._last_close`` window and is likewise per-bar stateless (m574 confirms it
# engine-global in function scope too), and calling it in place is cheaper than
# a mandatory per-bar evaluation.
#
# Libraries draw the line at ``export`` — and the law is the same whether the
# library is imported or run directly as a study (probes m575-m578, same data,
# a private library whose exports return ``ta.nvi``/``ta.vwap``):
#   - an EXPORTED function's read is a per-call-site gated machine: a gated
#     call diverges from the unconditional global read even for the
#     engine-global set (nvi 2/14264, vwap 595/14264 — the vwap agreements are
#     the session-anchor bars), an unconditional call agrees on all 28527 bars,
#     and two gated call sites of the SAME export diverge from each other too;
#     m577 shows the identical signature with the library run as a study;
#   - the library's GLOBAL scope follows the script law: a gated read there is
#     engine-global (m577, 14264/14264);
#   - a NON-exported library function follows the user-function law of
#     m573/m574: engine-global (m578, 14264/14264).
ENGINE_GLOBAL_VARIABLES = frozenset({'nvi', 'obv', 'pvi', 'pvt', 'wad'})
MAIN_SCOPE_VARIABLES = frozenset({'accdist', 'vwap'})
HOISTED_TA_VARIABLES = ENGINE_GLOBAL_VARIABLES | MAIN_SCOPE_VARIABLES


class TaVariableHoistTransformer(ast.NodeTransformer):
    """
    Evaluate referenced stateful ``ta`` builtin variables once per bar, at the top
    of ``main``.

    Runs right after the ``ModulePropertyTransformer``, so a bare ``ta.nvi`` read is
    already a zero-argument ``lib.ta.nvi()`` call here. Every rewireable call is
    replaced with a read of a module-global cache name, and ``main`` gets an
    unconditional prologue assigning each referenced cache from the real property
    call::

        global __ta·nvi
        __ta·nvi = lib.ta.nvi()

    The prologue call is an ordinary call site for the later series/persistent/
    isolation passes, so the variable's state advances exactly once per bar with a
    stable identity, no matter how many reference sites the script has or how they
    are gated. Rewiring scope follows the measured TradingView behavior (see the
    variable sets above): the engine-global set is replaced in every function of
    the module, while ``vwap``/``accdist`` are replaced only directly in ``main``'s
    body — inside any other function they keep their own per-call-site machine.
    Zero-argument calls only: ``ta.vwap(src)`` and friends are the function forms
    with their own Pine semantics and stay untouched.

    Modules without a top-level ``main`` are left alone — there is no per-bar
    entry point to hoist into. In library modules (``main`` decorated with
    ``script.library``) the reference-site rules follow the measured m575-m578
    laws: reads in ``main``'s body and in functions nested in ``main`` are
    rewired like in a script, but ``@export``-decorated functions (and anything
    nested in them) keep their per-call-site gated machines, and so do
    module-level functions — in a hand-written library those are its export
    surface.
    """

    def __init__(self):
        self.hoisted: list[str] = []  # referenced variable names, first-reference order
        self.function_stack: list[str] = []
        self.export_depth = 0  # how many enclosing functions are @export-decorated
        self.is_library = False

    @staticmethod
    def _is_library_main(main: ast.FunctionDef) -> bool:
        for decorator in main.decorator_list:
            func = decorator.func if isinstance(decorator, ast.Call) else decorator
            if isinstance(func, ast.Attribute) and func.attr == 'library':
                return True
        return False

    @staticmethod
    def _is_export(node: ast.FunctionDef) -> bool:
        for decorator in node.decorator_list:
            func = decorator.func if isinstance(decorator, ast.Call) else decorator
            if isinstance(func, ast.Name) and func.id == 'export':
                return True
            if isinstance(func, ast.Attribute) and func.attr == 'export':
                return True
        return False

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
        self.is_library = self._is_library_main(main)

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

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        is_export = self._is_export(node)
        self.function_stack.append(node.name)
        self.export_depth += is_export
        try:
            return self.generic_visit(node)
        finally:
            self.function_stack.pop()
            self.export_depth -= is_export

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
        if not self.function_stack:
            # Module-level code runs once at import, outside the per-bar state
            # machinery — nothing to rewire to
            return node
        if self.is_library:
            if self.function_stack[0] != 'main':
                # Module-level functions of a hand-written library are its
                # export surface: per-call-site gated machines (m575/m576)
                return node
            if self.export_depth:
                # Exported functions keep per-call-site machines whether the
                # library is imported or run as a study (m575-m577)
                return node
        if func.attr in MAIN_SCOPE_VARIABLES and self.function_stack != ['main']:
            # Inside a user function these get their own per-instance machine on
            # TradingView — the call-gated property call already models that
            return node
        if func.attr not in self.hoisted:
            self.hoisted.append(func.attr)
        return ast.copy_location(ast.Name(id=self._cache_name(func.attr), ctx=ast.Load()), node)
