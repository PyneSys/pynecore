"""
Put ``Series`` outside a parameter's qualifier marker.

PyneComp writes a qualifier marker outside everything else of a parameter's
annotation: a ``simple float`` parameter whose history the body reads is
``Simple[Series[float]]`` (``types/qualifier.py``). Hand-written Pyne code may
spell the same as ``Series[Simple[float]]``. This pass rewrites the first form
into the second, so the steps that look for a ``Series`` annotation head -- the
series lowering above all -- see one, and the type pass reads the marker under
it (``pine_type_rules.annotation_qualifier``).

It runs right after import normalization, ahead of every step that reads a
parameter annotation.
"""
import ast

from . import ast_walk

__all__ = ['normalize_series_qualifiers']

#: The qualifier markers of ``types/qualifier.py``
_MARKERS = frozenset({'Simple', 'Const'})


def _head(node: ast.expr) -> str:
    """The name a subscript head spells, ``''`` for anything else."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ''


def _normalize(annotation: ast.expr | None) -> None:
    """Rewrite ``Marker[Series[T]]`` into ``Series[Marker[T]]`` in place.

    The two subscripts swap their heads: each node keeps its identity and its
    location, only what it subscripts changes.
    """
    if not isinstance(annotation, ast.Subscript) or _head(annotation.value) not in _MARKERS:
        return
    inner = annotation.slice
    if not isinstance(inner, ast.Subscript):
        return
    series = inner.value
    if not isinstance(series, ast.Name) or series.id != 'Series':
        return
    annotation.value, inner.value = series, annotation.value


def normalize_series_qualifiers(tree: ast.Module) -> ast.Module:
    """Rewrite every parameter annotation ``Marker[Series[T]]`` into ``Series[Marker[T]]``.

    :param tree: The module
    :return: The same module, rewritten in place
    """
    for node in ast_walk.walk_statements(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        arguments = node.args
        for arg in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs):
            _normalize(arg.annotation)
        if arguments.vararg is not None:
            _normalize(arguments.vararg.annotation)
        if arguments.kwarg is not None:
            _normalize(arguments.kwarg.annotation)
    return tree
