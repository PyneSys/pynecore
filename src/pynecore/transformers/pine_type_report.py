"""
Where a module's types run out, reported once each.

The inference stamps every expression, and an UNKNOWN stamp means the value
has no Pine type. That is a fact about the program, not about this pass, and
it has ONE cause per cascade: ``b = a + 1`` is unknown because ``a`` is, and
``a`` is unknown because it was assigned from a call the pass could not type.
Reporting every unknown expression would repeat that cause once per use;
reporting the cascade's ROOT names it once, where the fix belongs.

The report walks the value positions only -- an expression whose type a
consumer reads: an argument, an operand, a right-hand side, a condition, a
return value, an index. What is not a value position says nothing about
types: a callee chain, a decorator, an annotation, a store target, the
namespace head of ``lib.close``. The plumbing the transforms emit is not a
program the user wrote and is left out too.

In hand-written code these diagnostics are the coverage meter -- how much of
the script the fast path can see -- and never stop anything. In an
``@pyne edge`` module the first one is an error.
"""
import ast
import re
from collections.abc import Iterator

from .node_ids import node_id
from .pine_type_rules import UNKNOWN, FactoryFields, get_ty, render_ty
from .pine_type_table import Diag, PineTypeTable, Unknown, qualify

__all__ = ['unknown_diags']

#: Names the transforms own: every dunder the emission spells, and every name
#: in the middle-dot namespace the transformers reserve (``__lib·bar_index``,
#: ``__slot_state·__``).
_SYNTHETIC = re.compile(r'^__.*__$|·')

#: How each provenance reason is remedied, for the diagnostic's fix-it.
_FIXES: dict[str, str] = {
    'unannotated-param': "annotate the parameter",
    'unannotated-import': "annotate the export in the module it comes from",
    'unknown-value': "assign it a value of a known type",
    'joined-branches': "make every branch assign the same type",
    'rebound-name': "do not rebind the name",
    'recursion': "annotate the parameters and the return of the recursive function",
    'import-cycle': "break the import cycle",
    'context-budget': "call it with fewer distinct argument types",
    'shape-mismatch': "make both sides the same type",
    'suppressed-import': "fix the imported module first",
    'unknown-name': "define it with a value of a known type",
    'function-value': "call it where it is used; a function is not a value",
    'unknown-return': "annotate the return",
    'unknown-lib': "give its arguments known types",
    'unknown-lib-name': "use a lib name that exists",
    'unknown-call': "call defined functions and lib names only",
    'unknown-field': "declare the field on a @udt class",
    'unknown-index': "index a series, a tuple, or an array through array.get()",
    'not-pine': "write it with Pine constructs only",
    'unknown-op': "give the operands types the operation accepts together",
}


def unknown_diags(tree: ast.Module, table: PineTypeTable) -> list[Diag]:
    """
    One diagnostic per root cause of an UNKNOWN value in the module.

    :param tree: The analysed, type-stamped module
    :param table: Its type table, for the provenance of the names
    :return: The diagnostics, in source order of the roots
    """
    return _Report(tree, table).run()


class _Report:
    """One pass over one stamped module."""

    def __init__(self, tree: ast.Module, table: PineTypeTable):
        self.tree = tree
        self.factory = FactoryFields(tree)
        self.table = table
        self.parent_of: dict[int, ast.AST] = {}
        self.scope_of: dict[int, str] = {}
        self.skip: set[int] = set()
        self.attr_bases: set[int] = set()
        self.statement_values: set[int] = set()
        #: (line, col) -> the value expressions starting there, for following
        #: a binding's provenance back to the expression it was assigned from
        self.at: dict[tuple[int, int], list[ast.expr]] = {}
        #: What the engine already reported: its diagnostics' origins, and
        #: the positions they stand at. A node it diagnosed is not diagnosed
        #: again, and a name whose provenance IS such a diagnostic is that
        #: diagnostic's cascade, not a root of its own
        self.seen: set[tuple[str, int, int, str]] = {
            (diag.origin.reason, diag.origin.line, diag.origin.col, diag.origin.detail)
            for diag in table.diags if diag.origin is not None}
        self.reported_at: set[tuple[int, int]] = {(diag.line, diag.col) for diag in table.diags}
        self.following: set[int] = set()
        self.diags: list[Diag] = []

    def run(self) -> list[Diag]:
        self._index(self.tree, '')
        for node in self._value_exprs():
            if get_ty(node) != UNKNOWN:
                continue
            if id(node) in self.attr_bases and isinstance(node, (ast.Name, ast.Attribute)):
                continue
            if not self._maximal(node) or id(node) in self.statement_values:
                continue
            if _is_plumbing(node) or not getattr(node, 'lineno', 0):
                continue
            for root in self._roots(node):
                self._report(root)
        self.diags.sort(key=lambda diag: (diag.line, diag.col))
        return self.diags

    # --- indexing --------------------------------------------------------

    def _index(self, node: ast.AST, scope: str) -> None:
        """Record every node's parent and scope, and what is not a value position."""
        self.scope_of[id(node)] = scope
        if isinstance(node, ast.Call):
            for sub in ast.walk(node.func):
                self.skip.add(id(sub))
            # ``method_call(delete, box)`` selects a method by its function:
            # the selector is a name of code, not a value
            if _dotted(node.func) in ('method_call', 'lib.method_call') and node.args:
                for sub in ast.walk(node.args[0]):
                    self.skip.add(id(sub))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for decorator in node.decorator_list:
                for sub in ast.walk(decorator):
                    self.skip.add(id(sub))
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    for sub in ast.walk(base):
                        self.skip.add(id(sub))
                # ``field(default_factory=...)`` as a UDT field's default: the
                # dataclass machinery builds it, the annotation types the field
                for value in self.factory.of(node):
                    for sub in ast.walk(value):
                        self.skip.add(id(sub))
            else:
                for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
                    if arg.annotation is not None:
                        for sub in ast.walk(arg.annotation):
                            self.skip.add(id(sub))
                if node.returns is not None:
                    for sub in ast.walk(node.returns):
                        self.skip.add(id(sub))
                scope = qualify(scope, node.name)
        elif isinstance(node, ast.AnnAssign):
            for sub in ast.walk(node.annotation):
                self.skip.add(id(sub))
        elif isinstance(node, ast.match_case):
            # A pattern matches, it does not evaluate: the ``match`` itself is
            # what is not Pine, and the structural gate says so
            for sub in ast.walk(node.pattern):
                self.skip.add(id(sub))
        elif isinstance(node, ast.ExceptHandler):
            # ``except TypeError:`` names a class to catch, not a value
            if node.type is not None:
                for sub in ast.walk(node.type):
                    self.skip.add(id(sub))
        if isinstance(node, ast.Expr):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                self.skip.add(id(node.value))
            else:
                self.statement_values.add(id(node.value))
        elif isinstance(node, ast.Attribute):
            self.attr_bases.add(id(node.value))
        elif isinstance(node, (ast.Name, ast.Subscript, ast.Tuple, ast.List, ast.Starred)) \
                and not isinstance(getattr(node, 'ctx', ast.Load()), ast.Load):
            self.skip.add(id(node))
            if isinstance(node, (ast.Tuple, ast.List)):
                for sub in ast.walk(node):
                    self.skip.add(id(sub))
        if isinstance(node, ast.expr) and hasattr(node, 'lineno'):
            self.at.setdefault((node.lineno, node.col_offset), []).append(node)
        for child in ast.iter_child_nodes(node):
            self.parent_of[id(child)] = node
            self._index(child, scope)

    def _value_exprs(self) -> Iterator[ast.expr]:
        for parent in ast.walk(self.tree):
            for _, value in ast.iter_fields(parent):
                for item in value if isinstance(value, list) else [value]:
                    if isinstance(item, ast.expr) and id(item) not in self.skip:
                        yield item

    def _maximal(self, node: ast.expr) -> bool:
        """Whether no enclosing value expression is unknown too (one cascade, one report)."""
        parent = self.parent_of.get(id(node))
        if isinstance(parent, ast.Call):
            # An argument of the transforms' own call is theirs, not the user's
            if node is not parent.func and _is_plumbing(parent):
                return False
            return node is parent.func or get_ty(parent) != UNKNOWN
        return not (isinstance(parent, ast.expr) and get_ty(parent) == UNKNOWN)

    # --- the root of a cascade -------------------------------------------

    def _roots(self, node: ast.expr) -> list[ast.expr]:
        """
        Descend through unknown children to the expressions that started it.

        Two unknown operands are two causes: ``foo + bar`` reports both. A
        transform may have put an unpositioned node on the way -- a closure
        parameter injected into a call, a hoisted temporary -- and a report
        has to point somewhere the user wrote: the descent stops at the last
        expression that has a position.
        """
        below = [child for child in self._value_children(node)
                 if get_ty(child) == UNKNOWN and not _is_plumbing(child)
                 and getattr(child, 'lineno', 0) and id(child) not in self.skip]
        if not below:
            return [node]
        return [root for child in below for root in self._roots(child)]

    def _value_children(self, node: ast.expr) -> list[ast.expr]:
        if isinstance(node, ast.Call):
            return [*node.args, *(keyword.value for keyword in node.keywords)]
        if isinstance(node, ast.Attribute):
            # A dotted chain is one name -- ``lib.volume`` is not a read of
            # ``lib`` -- but a local whose field is read is the cause itself
            value = node.value
            if isinstance(value, ast.Name) and self._binding(value) is not None:
                return [value]
            return [] if isinstance(value, (ast.Name, ast.Attribute)) else [value]
        if isinstance(node, ast.Lambda):
            return []
        return [child for child in ast.iter_child_nodes(node) if isinstance(child, ast.expr)]

    def _binding(self, name: ast.Name):
        """The binding a name reads, in its scope or any enclosing one."""
        scope = self.scope_of.get(id(name), '')
        while True:
            found = self.table.bindings.get(scope, {}).get(name.id)
            if found is not None:
                return found
            if not scope:
                return None
            scope = scope.rpartition('·')[0]

    def _is_user_function(self, callee: str, node: ast.AST) -> bool:
        """Whether a bare callee names a function this module defines, seen from the call."""
        scope = self.scope_of.get(id(node), '')
        while True:
            if qualify(scope, callee) in self.table.funcs:
                return True
            if not scope:
                return False
            scope = scope.rpartition('·')[0]

    def _covered(self, node: ast.expr) -> bool:
        """
        Whether a diagnostic already stands INSIDE an unknown expression.

        A put whose operand does not fit is reported at the operand, and the
        call it leaves unknown is that report's cascade: ``x =
        array.push(a, "s")`` has one cause, at the string. The same holds for
        every expression whose own reason is a report somewhere within it.
        """
        return any((getattr(sub, 'lineno', 0), getattr(sub, 'col_offset', 0)) in self.reported_at
                   for sub in ast.walk(node) if sub is not node and isinstance(sub, ast.expr))

    def _assigned_from(self, origin: Unknown) -> ast.expr | None:
        """The unknown value expression a binding's provenance points at, if it is one."""
        if origin.reason != 'unknown-value':
            return None
        for candidate in self.at.get((origin.line, origin.col), ()):
            if get_ty(candidate) == UNKNOWN and id(candidate) not in self.skip:
                return candidate
        return None

    # --- reporting -------------------------------------------------------

    def _report(self, node: ast.expr) -> None:
        line = getattr(node, 'lineno', 0)
        col = getattr(node, 'col_offset', 0)
        if (line, col) in self.reported_at or _is_plumbing(node) or self._covered(node):
            return
        if isinstance(node, ast.Name):
            binding = self._binding(node)
            origin = binding.unknown if binding is not None else None
            if origin is None:
                origin = Unknown(reason='unknown-name', line=line, col=col, detail=node.id)
            # A name assigned from an unknown expression is that expression's
            # cascade: follow the provenance to the root and report THAT once
            assigned = self._assigned_from(origin)
            if assigned is not None:
                if id(assigned) not in self.following:
                    self.following.add(id(assigned))
                    for root in self._roots(assigned):
                        self._report(root)
                return
            if origin.reason == 'unknown-name' and self._is_user_function(node.id, node):
                # ``g = other``: a function read as a value. Pine has no
                # function values, and the walk cannot follow what the alias
                # is then called as
                origin = Unknown(reason='function-value', line=line, col=col, detail=node.id)
                message = f"'{node.id}' is a function, not a value"
            else:
                message = f"'{node.id}' has no known type"
        elif isinstance(node, ast.Call):
            callee = _dotted(node.func)
            if callee is None:
                origin = Unknown(reason='unknown-call', line=line, col=col, detail='')
                message = 'this call has no known type'
            elif callee == 'method_call' and len(node.args) >= 2:
                # The plumbing of a method call: the METHOD is what has no
                # type, and the message names it, not the plumbing
                selector = node.args[0]
                if isinstance(selector, ast.Constant) and isinstance(selector.value, str):
                    method = selector.value
                else:
                    method = _dotted(selector) or ast.unparse(selector)
                origin = Unknown(reason='unknown-return', line=line, col=col, detail=method)
                message = f"the call to method '{method}' has no known type"
            elif callee.startswith('lib.'):
                origin = Unknown(reason='unknown-lib', line=line, col=col, detail=callee)
                message = f"'{callee[4:]}' has no known type here"
            elif self._is_user_function(callee, node):
                origin = Unknown(reason='unknown-return', line=line, col=col, detail=callee)
                message = f"the call to '{callee}' has no known type"
            else:
                origin = Unknown(reason='unknown-call', line=line, col=col, detail=callee)
                message = f"the call to '{callee}' has no known type"
        elif isinstance(node, ast.Attribute):
            dotted = _dotted(node)
            if dotted is not None and dotted.startswith('lib.'):
                origin = Unknown(reason='unknown-lib-name', line=line, col=col, detail=dotted)
                message = f"'{dotted[4:]}' is not a known lib name"
            else:
                origin = Unknown(reason='unknown-field', line=line, col=col, detail=node.attr)
                message = f"'.{node.attr}' has no known type on {render_ty(get_ty(node.value))}"
        elif isinstance(node, ast.Subscript):
            origin = Unknown(reason='unknown-index', line=line, col=col, detail='')
            message = f'indexing {render_ty(get_ty(node.value))} has no known type'
        elif isinstance(node, (ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp)):
            # A Pine operation over KNOWN operands that has no result type:
            # the operands are what the operation does not accept together
            operands = [node.body, node.orelse] if isinstance(node, ast.IfExp) \
                else self._value_children(node)
            over = ', '.join(render_ty(get_ty(operand)) for operand in operands)
            origin = Unknown(reason='unknown-op', line=line, col=col, detail=type(node).__name__)
            message = f"'{ast.unparse(node)[:40]}' has no known type over {over}"
        else:
            kind = type(node).__name__
            origin = Unknown(reason='not-pine', line=line, col=col, detail=kind)
            message = f"'{ast.unparse(node)[:40]}' is not a Pine expression"
        key = (origin.reason, origin.line, origin.col, origin.detail)
        if key in self.seen:
            return
        self.seen.add(key)
        fix = _FIXES.get(origin.reason, 'give it a known type')
        if origin.reason == 'unannotated-param' and origin.detail:
            fix = f"annotate '{origin.detail}'"
        elif origin.reason == 'unknown-return' and origin.detail:
            fix = f"annotate the return of '{origin.detail}'"
        self.diags.append(Diag(message=message, line=line, col=col, origin=origin, fix=fix))


def _is_plumbing(node: ast.AST) -> bool:
    """Whether an expression is something the transforms emitted, not the user."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and _SYNTHETIC.match(sub.id):
            return True
        if isinstance(sub, ast.Attribute) and sub.attr == '__class__':
            return True
        if isinstance(sub, ast.Call):
            callee = _dotted(sub.func) or ''
            if callee.startswith('__sec_') or callee in ('run', 'shadowed_namespace'):
                return True
    return False


def _dotted(node: ast.expr) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return '.'.join(reversed(parts))
