"""
Dynamic Default Transformer

Rewrites function-parameter defaults that reference per-bar runtime state
(any ``lib.*`` expression — ``lib.hl2``, ``lib.close``, ...) so they are
evaluated per CALL instead of at ``def`` time.

Why: Pine semantics. ``export ao(series float source = hl2)`` means "the
caller omitted the argument, use the CURRENT bar's hl2". A Python def-time
default freezes one value. Per-bar redefinition of the function does not
save the day either: an anchored call site (see ``instance_state``) binds
the callee closure once — an ``Exported`` library proxy keeps a stable
identity across bars, so the hot-path identity check keeps reusing the
FIRST bar's closure and with it the first bar's frozen default.

Transformation::

    # Before
    def ao(source: float = lib.hl2, shortLength: int = 5):
        ...

    # After
    def ao(source: float = __dyn_default__, shortLength: int = 5):
        if source is __dyn_default__:
            source = lib.hl2
        ...

Only defaults containing a ``lib`` reference are rewritten — plain constants
(``5``, ``'x'``) keep the zero-cost def-time path. Script entry points
(``@lib.script.indicator/strategy/library``) are skipped: their defaults are
``input.*()`` calls consumed by the input machinery at def time. Class bodies
(compiled Protocol stubs) are skipped too.

Must run after ImportNormalizerTransformer (references are ``lib.*``-
qualified) and before the series/isolation passes (the moved expressions
must participate in them like any other body statement).
"""

import ast

__all__ = ['DynamicDefaultTransformer']

_SCRIPT_ENTRY_DECORATORS = frozenset({'indicator', 'strategy', 'library'})
_SENTINEL_NAME = '__dyn_default__'


class DynamicDefaultTransformer(ast.NodeTransformer):
    """Move ``lib.*``-referencing parameter defaults into per-call prologues."""

    def __init__(self):
        self._changed = False

    @staticmethod
    def _is_script_entry(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        """Whether the function is a ``@script.indicator/strategy/library`` entry."""
        for dec in node.decorator_list:
            target = dec.func if isinstance(dec, ast.Call) else dec
            if not (isinstance(target, ast.Attribute)
                    and target.attr in _SCRIPT_ENTRY_DECORATORS):
                continue
            parent = target.value
            if isinstance(parent, ast.Name) and parent.id == 'script':
                return True
            if isinstance(parent, ast.Attribute) and parent.attr == 'script':
                return True
        return False

    @staticmethod
    def _is_dynamic(expr: ast.expr) -> bool:
        """Whether the default expression references runtime ``lib`` state."""
        return any(isinstance(n, ast.Name) and n.id == 'lib' for n in ast.walk(expr))

    def _prologue_if(self, param_name: str, default: ast.expr) -> ast.If:
        """Build ``if <param> is __dyn_default__: <param> = <default>``."""
        return ast.If(
            test=ast.Compare(
                left=ast.Name(id=param_name, ctx=ast.Load()),
                ops=[ast.Is()],
                comparators=[ast.Name(id=_SENTINEL_NAME, ctx=ast.Load())],
            ),
            body=[ast.Assign(
                targets=[ast.Name(id=param_name, ctx=ast.Store())],
                value=default,
            )],
            orelse=[],
        )

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        # Protocol stubs in compiled libraries carry the same dynamic defaults
        # in their signatures, but they are never called — leave them alone.
        return node

    def _process_func(self, node: ast.FunctionDef | ast.AsyncFunctionDef):
        if self._is_script_entry(node):
            # Entry defaults are input.*() calls consumed at def time by the
            # input machinery — but inner functions still need the rewrite.
            self.generic_visit(node)
            return node

        prologue: list[ast.If] = []

        positional = node.args.posonlyargs + node.args.args
        defaults = node.args.defaults
        offset = len(positional) - len(defaults)
        for i, default in enumerate(defaults):
            if not self._is_dynamic(default):
                continue
            pname = positional[offset + i].arg
            prologue.append(self._prologue_if(pname, default))
            defaults[i] = ast.Name(id=_SENTINEL_NAME, ctx=ast.Load())

        for i, default in enumerate(node.args.kw_defaults):
            if default is None or not self._is_dynamic(default):
                continue
            pname = node.args.kwonlyargs[i].arg
            prologue.append(self._prologue_if(pname, default))
            node.args.kw_defaults[i] = ast.Name(id=_SENTINEL_NAME, ctx=ast.Load())

        if prologue:
            self._changed = True
            # Keep a leading docstring first
            insert_at = 0
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                insert_at = 1
            node.body[insert_at:insert_at] = prologue

        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        return self._process_func(node)  # type: ignore[return-value]

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        return self._process_func(node)  # type: ignore[return-value]

    def visit_Module(self, node: ast.Module) -> ast.Module:
        self._changed = False
        node = self.generic_visit(node)  # type: ignore[assignment]
        if self._changed:
            node.body.insert(0, ast.ImportFrom(
                module='pynecore.core.instance_state',
                names=[ast.alias(name=_SENTINEL_NAME, asname=None)],
                level=0,
            ))
        return node
