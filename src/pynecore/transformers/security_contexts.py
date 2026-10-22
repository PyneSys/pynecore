"""
Reading the ``__security_contexts__`` dict literal the security passes emit.
"""
import ast
from collections.abc import Iterator

__all__ = ['iter_contexts', 'ctx_get']


def iter_contexts(contexts: ast.Dict) -> Iterator[tuple[str, ast.Dict]]:
    """Yield ``(sid, context dict literal)`` pairs of ``__security_contexts__``."""
    for key, value in zip(contexts.keys, contexts.values):
        if (isinstance(key, ast.Constant) and isinstance(key.value, str)
                and isinstance(value, ast.Dict)):
            yield key.value, value


def ctx_get(ctx: ast.Dict, name: str) -> ast.expr | None:
    """One entry of a context dict literal."""
    for key, value in zip(ctx.keys, ctx.values):
        if isinstance(key, ast.Constant) and key.value == name:
            return value
    return None
