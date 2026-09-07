"""
Define a library's exported functions once per run instead of once per bar.

A library's ``main`` is a per-bar entry point (the runner calls every
registered library main before the script's own main), and Pine's export
surface is emitted as ``@export``-decorated definitions inside it. Executed
as written, every bar allocates a fresh function object per export and runs
the ``export`` decorator over it.

MEASURED: the objects built after the first bar are never called. A call site
anchors on the module-level ``Exported`` proxy, whose identity is stable, so
``instance_state._bind_target`` runs exactly ONCE per run and keeps the
callable it unwrapped there — the bar-0 one — for the whole run (instrumented
probe over 50 bars, both run modes: 50 ``Exported.set`` calls, 1 bind, and the
bound object is the one bar 0 created). The rest are allocated, decorated and
dropped: ~292 ns per export per bar, ~5.6 us/bar for a 20-export library.

Defining them once is therefore not an approximation of the current behavior,
it IS the current behavior — with the dead allocations removed. The only free
variable an export closes over is ``main``'s hidden state parameter, and the
runner binds ``main`` to ONE root vector for the whole run, so the bar-0
closure stays correct for every later bar.

The latch is a ``Persistent`` slot of ``main``, never a module global: a
second run in the same process (``instance_state.reset()``) hands ``main`` a
NEW state vector, and a module-level flag would freeze the exports onto the
old one. Keyed to the root, a fresh vector re-runs the definitions.

The definitions bind through ``global`` so a read on a later bar — the
library's own global scope may call what it exports — resolves to the
module-level ``Exported`` proxy instead of an unbound local. Writing the name
up there changes nothing: ``export`` returns that very proxy.

An export whose name ``main`` also binds somewhere else is left out of the
latch entirely and keeps being rebuilt per bar. A library may export one
overload of a name and keep another unexported (``jason5480/chrono_utils``'
``is_bar_included``), and only one of the two bindings can own the name: latched
under ``global`` the per-bar one overwrites the module-level proxy, and latched
without it the export's own binding is written on bar 0 only, so from bar 1 the
name inside ``main`` resolves to the other definition — or to nothing, when that
other binding is conditional.
"""
import ast

__all__ = ['ExportOnceTransformer']

#: Per-root latch the definitions run under, a Persistent slot of ``main``
EXPORT_LATCH = '__exports·__'


def _bound_names(nodes: list[ast.stmt]) -> set[str]:
    """Every name the statements bind in ``main``'s OWN scope.

    Nested function and class bodies are not descended into: a name they bind
    belongs to their scope, not to ``main``'s, so it cannot collide with an
    export's binding. Everything else is walked, because an assignment at any
    depth of ``main``'s own code rebinds the name for the whole bar.

    :param nodes: The statements to scan.
    :return: The set of names the statements bind.
    """
    bound: set[str] = set()
    stack: list[ast.AST] = list(nodes)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # The definition binds its own name here; its body is another scope
            bound.add(node.name)
            continue
        if isinstance(node, ast.Lambda):
            continue
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        stack.extend(ast.iter_child_nodes(node))
    return bound


def _is_export(node: ast.FunctionDef) -> bool:
    """Whether a definition carries the ``@export`` decorator.

    :param node: The definition to check.
    :return: True if any decorator is ``export`` or ``<mod>.export``.
    """
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(target, ast.Name) and target.id == 'export':
            return True
        if isinstance(target, ast.Attribute) and target.attr == 'export':
            return True
    return False


class ExportOnceTransformer(ast.NodeTransformer):
    """Run ``main``'s ``@export`` definitions under a per-root latch."""

    def visit_Module(self, node: ast.Module) -> ast.Module:
        main: ast.FunctionDef | None = None
        for stmt in node.body:
            if isinstance(stmt, ast.FunctionDef) and stmt.name == 'main':
                main = stmt
                break
        if main is None:
            return node

        # A name the library binds anywhere ELSE in ``main`` stays out of the
        # latch: that other binding runs on every bar and would take the name
        # over from bar 1, while the export's own binding is written on bar 0
        # only. Rebuilt where it stands, such a definition behaves exactly as it
        # did before the latch, and the module-level ``Exported`` proxy the
        # decorator registered is the only thing a caller ever resolves to
        outside = _bound_names([
            stmt for stmt in main.body
            if not (isinstance(stmt, ast.FunctionDef) and _is_export(stmt))])

        # Contiguous runs of latchable export definitions, so nothing between
        # them is moved: a run is guarded where it stands
        runs: list[tuple[int, int]] = []
        names: list[str] = []
        start: int | None = None
        for i, stmt in enumerate(main.body):
            if (isinstance(stmt, ast.FunctionDef) and _is_export(stmt)
                    and stmt.name not in outside):
                if start is None:
                    start = i
                if stmt.name not in names:
                    names.append(stmt.name)
            elif start is not None:
                runs.append((start, i))
                start = None
        if start is not None:
            runs.append((start, len(main.body)))
        if not runs:
            return node

        body: list[ast.stmt] = []
        cursor = 0
        for n, (begin, end) in enumerate(runs):
            body.extend(main.body[cursor:begin])
            guarded = list(main.body[begin:end])
            if n == len(runs) - 1:
                # The last run closes the latch, so every earlier run still
                # sees it open on the bar the definitions are built
                guarded.append(ast.Assign(
                    targets=[ast.Name(id=EXPORT_LATCH, ctx=ast.Store())],
                    value=ast.Constant(value=True)))
            body.append(ast.If(
                test=ast.UnaryOp(op=ast.Not(),
                                 operand=ast.Name(id=EXPORT_LATCH, ctx=ast.Load())),
                body=guarded, orelse=[]))
            cursor = end
        body.extend(main.body[cursor:])

        prologue: list[ast.stmt] = [
            ast.Global(names=names),
            ast.AnnAssign(
                target=ast.Name(id=EXPORT_LATCH, ctx=ast.Store()),
                annotation=ast.Subscript(
                    value=ast.Name(id='Persistent', ctx=ast.Load()),
                    slice=ast.Name(id='bool', ctx=ast.Load()), ctx=ast.Load()),
                value=ast.Constant(value=False), simple=1),
        ]
        # Keep the docstring first; a ``global`` must precede every use of the
        # names it declares, so the prologue goes above any other statement
        first = body[0] if body else None
        insert_at = 1 if (isinstance(first, ast.Expr)
                          and isinstance(first.value, ast.Constant)
                          and isinstance(first.value.value, str)) else 0
        body[insert_at:insert_at] = prologue
        main.body = body
        return node
