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
``input.*()`` calls consumed by the input machinery at def time.

A UDT field defaulting to a bool na (``f: bool = na(bool)``) is the same kind
of frozen value: the bool na is what the RUNNING script's mode says it is
(see ``set_bool_na``), while a class body is evaluated once at import, under
the mode of the module that defines it — an imported library's own. The
default is therefore lowered to a ``dataclasses.field(default_factory=...)``
so each construction builds it under the mode in effect::

    # Before
    @udt
    class Flag:
        f: bool = lib.na(bool)

    # After
    from dataclasses import field as __pyne_field·__
    from pynecore.types.na import new_bool_na as __pyne_bool_na·__
    @udt
    class Flag:
        f: bool = __pyne_field·__(default_factory=lambda: __pyne_bool_na·__())

The factory is the canonical constructor, bound to a reserved name: what the
class statement would have called is resolved HERE, in source order (a
``lib.na`` call, or the constructor reached through any live ``pynecore``
binding -- a name, a module alias, a package alias, the bare root), so a
later rebinding of ``na`` / ``NA`` cannot reach into the default. Only
module-level classes are lowered; other class bodies (compiled Protocol
stubs) are left alone.

Must run after ImportNormalizerTransformer (references are ``lib.*``-
qualified) and before the series/isolation passes (the moved expressions
must participate in them like any other body statement).
"""

import ast

__all__ = ['DynamicDefaultTransformer', 'is_script_entry']

_SCRIPT_ENTRY_DECORATORS = frozenset({'indicator', 'strategy', 'library'})
_SENTINEL_NAME = '__dyn_default__'
#: The alias ``dataclasses.field`` is bound to for the lowered UDT defaults
_FIELD_NAME = '__pyne_field·__'
#: The alias the bool na factory (``pynecore.types.na.new_bool_na``) is bound to
_BOOL_NA_NAME = '__pyne_bool_na·__'
_UDT_DECORATORS = frozenset({'udt', 'dataclass'})


def _dotted_tail(node: ast.expr) -> str | None:
    """The last name of a ``Name`` / ``Attribute`` chain (the callee of a decorator call too)."""
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def _chain(node: ast.expr) -> list[str] | None:
    """The names of a ``a.b.c`` chain, None for anything else."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return parts[::-1]


class _NaBindings:
    """
    The spellings of the bool na constructor in effect at a top-level statement.

    Every name a ``pynecore`` import binds is kept with the ABSOLUTE path it
    denotes -- ``import pynecore as p`` binds ``p`` to ``pynecore``, ``from
    pynecore.types import na`` binds ``na`` to ``pynecore.types.na``, ``from
    pynecore.types.na import NA as X`` binds ``X`` to the constructor itself;
    a bare ``import pynecore[.x]`` binds the root, and importing the package
    loads the na module through ``pynecore.types``, so any chain reaching
    ``pynecore.types.na.NA`` through a live binding is the constructor. Any
    later binding of a name takes it away again, in source order.
    """

    __slots__ = ('paths',)

    #: The absolute paths that build a bool na
    CONSTRUCTORS = frozenset({'pynecore.types.na.NA', 'pynecore.lib.na'})

    def __init__(self):
        #: bound name -> the absolute dotted path it denotes
        self.paths: dict[str, str] = {}

    def is_bool_na(self, expr: ast.expr) -> bool:
        """Whether the expression is ``lib.na(bool)`` or the constructor called as ``(bool)``."""
        if not (isinstance(expr, ast.Call) and len(expr.args) == 1 and not expr.keywords
                and isinstance(expr.args[0], ast.Name) and expr.args[0].id == 'bool'):
            return False
        chain = _chain(expr.func)
        if chain is None:
            return False
        if chain == ['lib', 'na']:
            return True
        root = self.paths.get(chain[0])
        return root is not None and '.'.join([root, *chain[1:]]) in self.CONSTRUCTORS

    def track(self, stmt: ast.stmt) -> None:
        """Update the bindings past a top-level statement."""
        if isinstance(stmt, ast.ImportFrom):
            module = stmt.module or ''
            for alias in stmt.names:
                bound = alias.asname or alias.name
                self.paths.pop(bound, None)
                if not stmt.level and (module == 'pynecore' or module.startswith('pynecore.')):
                    self.paths[bound] = f'{module}.{alias.name}'
        elif isinstance(stmt, ast.Import):
            for alias in stmt.names:
                root = alias.name.split('.')[0]
                if alias.asname is not None:
                    self.paths.pop(alias.asname, None)
                    if root == 'pynecore':
                        self.paths[alias.asname] = alias.name
                else:
                    self.paths.pop(root, None)
                    if root == 'pynecore':
                        self.paths[root] = root
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            self.paths.pop(stmt.name, None)
        else:
            for node in ast.walk(stmt):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    self.paths.pop(node.id, None)


def _insert_import(body: list[ast.stmt], stmt: ast.ImportFrom) -> None:
    """Insert a generated import after the docstring and the ``__future__`` block."""
    insert_at = 0
    first = body[0] if body else None
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
            and isinstance(first.value.value, str):
        insert_at = 1
    while insert_at < len(body) and isinstance(body[insert_at], ast.ImportFrom) \
            and getattr(body[insert_at], 'module', None) == '__future__':
        insert_at += 1
    body.insert(insert_at, stmt)


def is_script_entry(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Whether the function is a ``@script.indicator/strategy/library`` entry.

    The runner calls such a function with NO arguments, so its parameter
    defaults are not fallbacks but the values it runs with, every bar. That is
    why this pass leaves them alone -- the input machinery consumes them at
    ``def`` time -- and it is the same fact the type inference reads them by.

    :param node: The definition to inspect
    :return: True for a script entry point
    """
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


class DynamicDefaultTransformer(ast.NodeTransformer):
    """Move ``lib.*``-referencing parameter defaults into per-call prologues."""

    def __init__(self):
        self._changed = False
        self._field_used = False

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

    def _lower_udt_defaults(self, node: ast.ClassDef, bindings: _NaBindings) -> None:
        """Bind a UDT field's bool na default to the factory, built per construction."""
        if not any(_dotted_tail(decorator) in _UDT_DECORATORS for decorator in node.decorator_list):
            return
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) and stmt.value is not None \
                    and bindings.is_bool_na(stmt.value):
                self._field_used = True
                stmt.value = ast.Call(
                    func=ast.Name(id=_FIELD_NAME, ctx=ast.Load()), args=[],
                    keywords=[ast.keyword(arg='default_factory', value=ast.Lambda(
                        args=ast.arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[],
                                           kw_defaults=[], kwarg=None, defaults=[]),
                        body=ast.Call(func=ast.Name(id=_BOOL_NA_NAME, ctx=ast.Load()),
                                      args=[], keywords=[])))])

    def _process_func(self, node: ast.FunctionDef | ast.AsyncFunctionDef):
        if is_script_entry(node):
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
        self._field_used = False
        bindings = _NaBindings()
        for stmt in node.body:
            if isinstance(stmt, ast.ClassDef):
                self._lower_udt_defaults(stmt, bindings)
            bindings.track(stmt)
        node = self.generic_visit(node)  # type: ignore[assignment]
        if self._field_used:
            _insert_import(node.body, ast.ImportFrom(
                module='pynecore.types.na',
                names=[ast.alias(name='new_bool_na', asname=_BOOL_NA_NAME)],
                level=0,
            ))
            _insert_import(node.body, ast.ImportFrom(
                module='dataclasses',
                names=[ast.alias(name='field', asname=_FIELD_NAME)],
                level=0,
            ))
        if self._changed:
            _insert_import(node.body, ast.ImportFrom(
                module='pynecore.core.instance_state',
                names=[ast.alias(name=_SENTINEL_NAME, asname=None)],
                level=0,
            ))
        return node
