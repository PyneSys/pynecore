"""Language rule: an object created outside a function may not be modified inside one.

A Pyne script may bind anything at module level — an array, a matrix, a map, a
user object, a color, an enum member — and read it from anywhere. What it may NOT
do is write into that object from inside a function::

    STORE = array.new_float(0)

    def main():
        array.push(STORE, close)   # rejected here

Module-level storage lives outside everything the runtime rolls back. A
``request.security`` child re-runs ``main()`` on the same developing bar and
discards the result, ``calc_on_order_fills`` re-executes the bar per fill, and a
live intrabar tick re-executes the bar per tick; all three restore the script's
own slots (``Persistent``, ``Series``) and none of them can restore a plain
Python object the script keeps for itself. Pine has no such storage at all, so
nothing that used to work is taken away — the check only names the error now.

This pass rejects the DIRECT forms, where the name being written is visibly the
module-level one. A write through an alias or through a parameter
(``def push(buf): array.push(buf, x)``) is not detected: the correct construct
for state that must survive is ``Persistent`` / ``IBPersistent``.

Placement in the pipeline: right after import normalization, so the ``lib.*``
chains are in their canonical shape, and well before function isolation, whose
per-call-site copies would otherwise report the same write several times.
Functions whose name starts with ``__test_`` are exempt: they are harness code,
not script code, and the AOT exporter skips them as well.
"""
import ast

from . import ast_walk

__all__ = ['OuterWriteTransformer']

#: Methods of the Python containers a script can hold that write the receiver.
#: A call ``NAME.<method>(...)`` on a module-level NAME is a write of that object.
_MUTATING_METHODS = frozenset({
    'append', 'extend', 'insert', 'pop', 'remove', 'clear', 'sort', 'reverse',
    'update', 'setdefault', 'popitem', 'add', 'discard',
})

#: Per collection namespace, the library functions that modify their first
#: argument in place. Collected from the library sources (``lib/array.py``,
#: ``lib/matrix.py``, ``lib/map.py``): a function is in here exactly when its
#: body writes the object its ``id`` parameter names. ``array.concat`` and
#: ``matrix.concat`` are in it because both extend their first argument and hand
#: it back; every other collection builtin returns a fresh value.
_MUTATING_LIB_CALLS: dict[str, frozenset[str]] = {
    'array': frozenset({
        'clear', 'concat', 'fill', 'insert', 'pop', 'push', 'remove', 'reverse',
        'set', 'shift', 'sort', 'unshift',
    }),
    'matrix': frozenset({
        'add_col', 'add_row', 'concat', 'fill', 'remove_col', 'remove_row',
        'reshape', 'reverse', 'set', 'sort', 'swap_columns', 'swap_rows',
    }),
    'map': frozenset({'clear', 'put', 'put_all', 'remove'}),
}

#: Prefix of the harness functions of a ``@pyne`` test script. They are not
#: script code — the AOT exporter drops them — so the rule does not apply to them.
_TEST_PREFIX = '__test_'


def _chain_root(node: ast.expr) -> ast.Name | None:
    """The base ``Name`` of an attribute / subscript chain, if it has one.

    :param node: The place expression (``a``, ``a.b``, ``a[i].c``).
    :return: The root name node, or None when the chain is rooted at a call or
        a literal.
    """
    root: ast.expr = node
    while isinstance(root, (ast.Attribute, ast.Subscript)):
        root = root.value
    return root if isinstance(root, ast.Name) else None


def _collection_call(func: ast.Attribute) -> tuple[str, str] | None:
    """``(namespace, function)`` of a ``lib.<array|matrix|map>.<fn>`` callee.

    :param func: The callee expression of a call.
    :return: The namespace and function name, or None for any other callee.
    """
    owner = func.value
    if (isinstance(owner, ast.Attribute) and owner.attr in _MUTATING_LIB_CALLS
            and isinstance(owner.value, ast.Name) and owner.value.id == 'lib'):
        return owner.attr, func.attr
    return None


def _module_names(module: ast.Module) -> set[str]:
    """Names the module's top level BINDS to an object of its own.

    Only assignment targets count. An import binds a module or a library
    function, a ``def`` or a ``class`` binds a callable — none of them is script
    storage, and a write into one is a different question this rule does not ask.
    Dunder names are the pipeline's own artifacts.

    :param module: The normalized module.
    :return: The module-level names.
    """
    names: set[str] = set()
    for stmt in module.body:
        targets: list[ast.expr] = []
        if isinstance(stmt, ast.Assign):
            targets = list(stmt.targets)
        elif isinstance(stmt, ast.AnnAssign):
            targets = [stmt.target]
        for target in targets:
            for sub in ast_walk.walk(target):
                if isinstance(sub, ast.Name) and not sub.id.startswith('__'):
                    names.add(sub.id)
    return names


#: The candidates :meth:`OuterWriteTransformer._check_node` can report
_WRITES: tuple[type[ast.AST], ...] = (ast.Global, ast.Assign, ast.AnnAssign, ast.AugAssign,
                                      ast.Delete, ast.Call)


def _nested_defs(node: ast.AST) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    """The function definitions standing directly in ``node``'s own scope.

    A ``class`` body is followed: its methods are function bodies the rule
    applies to just like any other. A ``def`` and a ``lambda`` are not — the
    first is returned instead of entered, the second runs no statements.
    """
    found: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    pending: list[ast.AST] = list(ast_walk.iter_child_nodes(node))
    while pending:
        child = pending.pop()
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found.append(child)
            continue
        if isinstance(child, ast.Lambda):
            continue
        pending.extend(ast_walk.iter_child_nodes(child))
    return found


def _scan_function(func: ast.FunctionDef | ast.AsyncFunctionDef) \
        -> tuple[set[str], list[ast.AST], list[ast.FunctionDef | ast.AsyncFunctionDef]]:
    """One walk of a function body for everything the check needs from it.

    The local names are the names the body binds itself, so they shadow the
    module's. Every parameter and every ``Store``-context ``Name`` of the body
    counts, wherever in the body it stands: Python binds a local for the whole
    call, so ``lastPivot: Pivot = lastPivot(this)`` makes ``lastPivot`` local
    even where the module binds the same spelling. Nested definitions are not
    entered — they get their own set, seeded with this one — but their names
    are locals, at any depth.

    The writes are the body's own nodes that may write an object, in the
    order a pre-order walk of the body meets them; they can only be judged
    once every local name is known.

    The nested definitions are the ones :func:`_nested_defs` finds, in its
    order: that walk takes the last child first, which for definitions (none
    of them encloses another) is the reverse of the pre-order this walk meets
    them in.

    :param func: The function definition.
    :return: The local names, the write candidates and the nested definitions.
    """
    names: set[str] = set()
    spec = func.args
    for arg in [*spec.posonlyargs, *spec.args, *spec.kwonlyargs, spec.vararg, spec.kwarg]:
        if arg is not None:
            names.add(arg.arg)
    writes: list[ast.AST] = []
    defs: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    # Pre-order, left to right: (node, whether it stands in a class body)
    pending: list[tuple[ast.AST, bool]] = [(stmt, False) for stmt in reversed(func.body)]
    while pending:
        node, in_class = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defs.append(node)
            continue
        if isinstance(node, ast.Lambda):
            continue
        if isinstance(node, ast.ClassDef):
            in_class = True
        elif not in_class:
            if isinstance(node, ast.Name):
                if isinstance(node.ctx, (ast.Store, ast.Del)):
                    names.add(node.id)
                continue
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    names.add(alias.asname or alias.name.split('.')[0])
            elif isinstance(node, _WRITES):
                writes.append(node)
        children = list(ast_walk.iter_child_nodes(node))
        pending.extend((child, in_class) for child in reversed(children))
    # Definitions and classes bind their names in the scope they stand in,
    # the ones nested deeper in this function included
    for stmt in ast_walk.walk_statements(func):
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                and stmt is not func:
            names.add(stmt.name)
    defs.reverse()
    return names, writes, defs


class OuterWriteTransformer(ast_walk.NodeTransformer):
    """Reject every direct write of a module-level object from inside a function.

    The pass only inspects; it returns the tree unchanged or raises.
    """

    def __init__(self) -> None:
        self._module_file: str = '<script>'
        self._outer: set[str] = set()

    def visit_Module(self, node: ast.Module) -> ast.Module:
        self._module_file = getattr(node, '_module_file_path', '<script>')
        self._outer = _module_names(node)
        for func in _nested_defs(node):
            self._check_function(func, frozenset())
        return node

    # --- checking ---

    def _check_function(self, func: ast.FunctionDef | ast.AsyncFunctionDef,
                        shadowed: frozenset[str]) -> None:
        """Check one function body and every definition nested in it.

        :param func: The function definition.
        :param shadowed: Names the enclosing functions already bind.
        """
        if func.name.startswith(_TEST_PREFIX):
            return
        local_names, writes, nested = _scan_function(func)
        locals_here = shadowed | local_names
        for node in writes:
            self._check_node(node, locals_here)
        for inner in nested:
            self._check_function(inner, frozenset(locals_here))

    def _check_node(self, node: ast.AST, shadowed: frozenset[str] | set[str]) -> None:
        """Report the node when it writes an object a module-level name holds."""
        if isinstance(node, ast.Global):
            # Whatever a ``global`` names is module-level storage, whether or
            # not the module's top level binds it too.
            self._reject(node, node.names[0], 'rebound')
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        elif isinstance(node, ast.Delete):
            targets = list(node.targets)
        for target in targets:
            if not isinstance(target, (ast.Attribute, ast.Subscript)):
                continue
            root = _chain_root(target)
            if root is not None and root.id not in shadowed and root.id in self._outer:
                self._reject(target, root.id, 'written')
        if not isinstance(node, ast.Call):
            return
        func = node.func
        if not isinstance(func, ast.Attribute):
            return
        collection = _collection_call(func)
        if collection is not None:
            namespace, fname = collection
            if fname not in _MUTATING_LIB_CALLS[namespace]:
                return
            first = self._first_argument(node)
            root = _chain_root(first) if first is not None else None
            if root is not None and root.id not in shadowed and root.id in self._outer:
                self._reject(node, root.id, 'written')
            return
        if func.attr not in _MUTATING_METHODS:
            return
        root = _chain_root(func.value)
        if root is not None and root.id not in shadowed and root.id in self._outer:
            self._reject(node, root.id, 'written')

    @staticmethod
    def _first_argument(node: ast.Call) -> ast.expr | None:
        """The collection a ``lib.<array|matrix|map>.<fn>`` call acts on.

        Every collection builtin takes it as its first parameter, named ``id``,
        so a keyword call names the very same object a positional one passes. An
        unpacked positional names nothing.
        """
        if node.args:
            first = node.args[0]
            return None if isinstance(first, ast.Starred) else first
        for kw in node.keywords:
            if kw.arg == 'id':
                return kw.value
        return None

    def _reject(self, node: ast.AST, name: str, kind: str) -> None:
        """Raise the located error naming the offending module-level binding."""
        verb = ('is rebound' if kind == 'rebound' else 'is modified')
        raise SyntaxError(
            f"'{name}' {verb} inside a function: an object created outside a "
            f"function cannot be modified inside one. Module-level state is not "
            f"restored when a bar is re-executed; declare the value as "
            f"'Persistent' or 'IBPersistent' inside the function instead.",
            (self._module_file, getattr(node, 'lineno', 0),
             getattr(node, 'col_offset', 0) + 1, None))
