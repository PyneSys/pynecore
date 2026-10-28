"""
The ``field(default_factory=...)`` defaults the compiler emits for a module's UDT
fields, and the dotted-name reader they are recognized by.

A leaf module (``ast_walk`` only): the type rules, the type inference and the
security dependency analysis all read these, and the native ports pin the sources
of their Python steps by digest (see ``transformers/native.py``), so an edit of
the type algebra must not take a step that only needs this out of use.
"""
import ast

from . import ast_walk

__all__ = ['FactoryFields', 'dotted_name']


def dotted_name(node: ast.expr) -> str | None:
    """Render a dotted name expression, or None when it is not one."""
    parts: list[str] = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return '.'.join(reversed(parts))


class FactoryFields:
    """
    The ``field(default_factory=...)`` defaults of a module's UDT classes.

    Only a direct annotated field of a ``@udt`` / ``@dataclass`` class whose
    default calls the name bound to ``dataclasses.field`` AT THAT STATEMENT
    (a later import, definition or assignment of the name counterfeits it),
    with a zero-argument lambda as the factory -- the one form the compiler
    emits -- is dataclass plumbing: the machinery builds it and the annotation
    types the field. Any other call carrying a ``default_factory`` keyword is
    an ordinary call and is typed (and diagnosed) like one.
    """

    __slots__ = ('body', 'top_index')

    def __init__(self, tree: ast.Module):
        self.body = tree.body
        #: Every class's top-level statement index
        self.top_index: dict[int, int] = {}
        for index, stmt in enumerate(tree.body):
            for node in ast_walk.walk_statements(stmt):
                if isinstance(node, ast.ClassDef):
                    self.top_index[id(node)] = index

    def _binding(self, name: str, before: int) -> str | None:
        """What ``name`` denotes at statement ``before``: 'field', 'dataclasses' or other."""
        binding: str | None = None
        for stmt in self.body[:before]:
            if isinstance(stmt, ast.ImportFrom):
                for alias in stmt.names:
                    if (alias.asname or alias.name) == name:
                        binding = 'field' if (stmt.module == 'dataclasses' and not stmt.level
                                              and alias.name == 'field') else 'other'
            elif isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    if (alias.asname or alias.name.split('.')[0]) == name:
                        binding = 'dataclasses' if alias.name == 'dataclasses' else 'other'
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if stmt.name == name:
                    binding = 'other'
            else:
                for node in ast_walk.walk(stmt):
                    if isinstance(node, ast.Name) and node.id == name \
                            and isinstance(node.ctx, ast.Store):
                        binding = 'other'
        return binding

    def of(self, cls: ast.ClassDef) -> list[ast.Call]:
        """The factory-default calls of a class, empty for a class that is no UDT."""
        if not any((dotted_name(decorator) or '').rsplit('.', 1)[-1] in ('udt', 'dataclass')
                   for decorator in cls.decorator_list):
            return []
        before = self.top_index.get(id(cls), len(self.body))
        out: list[ast.Call] = []
        for stmt in cls.body:
            value = stmt.value if isinstance(stmt, ast.AnnAssign) else None
            if not isinstance(value, ast.Call) or len(value.keywords) != 1 or value.args:
                continue
            keyword = value.keywords[0]
            factory = keyword.value
            if keyword.arg != 'default_factory' or not isinstance(factory, ast.Lambda) \
                    or factory.args.args or factory.args.posonlyargs or factory.args.kwonlyargs \
                    or factory.args.vararg or factory.args.kwarg:
                continue
            func = value.func
            if (isinstance(func, ast.Name) and self._binding(func.id, before) == 'field') or (
                    isinstance(func, ast.Attribute) and func.attr == 'field'
                    and isinstance(func.value, ast.Name)
                    and self._binding(func.value.id, before) == 'dataclasses'):
                out.append(value)
        return out
