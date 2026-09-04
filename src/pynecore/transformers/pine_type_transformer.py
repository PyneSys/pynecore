"""
Pipeline entry for the Pine type inference.

Sits between the closure-argument pass and the series/isolation passes, which
is the last point where the tree still looks like Pine: the annotations are
intact (``SeriesTransformer`` rewrites the parameter ones and consumes the
declaration ones into the slot layout), the ``/`` is still a ``BinOp``
(``SafeDivisionTransformer`` wraps it into a call), and the security-bearing
functions have already been instantiated per call site, so each of them is
reached by exactly one caller here.

This pass changes nothing about the tree -- it clones no function and builds
no specialization. A generic helper is ANALYSED once per call-site context
and the answers are kept apart in the type table, while the tree keeps one
body carrying the join of what the contexts found. A call into an IMPORTED
module is the exception: it is typed from the interface that module publishes,
never from this call site, and every interface consulted is recorded in the
table's ``deps`` so the loader can invalidate what a moved signature broke. It
stamps ``_pine_ty`` on every expression and keeps the derived table on the
module node, so the passes that follow -- the overload pin, and the artifact
the AOT compiler consumes -- have the types without re-deriving them.
"""
import ast

from .pine_type_infer import infer_module
from .pine_type_table import Analyser, PineTypeTable

__all__ = ['PineTypeTransformer', 'TABLE_ATTR', 'module_table']

#: Attribute the module node carries its type table under.
TABLE_ATTR = '_pine_types'


class PineTypeTransformer:
    """
    Infer and stamp the Pine types of a module.

    Not an ``ast.NodeTransformer``: there is nothing to transform. It keeps
    the transformer shape (``visit`` returning the tree) so the pipeline reads
    uniformly.
    """

    def __init__(self, pyne_mode: str | None = None, *, analyse: Analyser | None = None,
                 pipeline_hash: str = ''):
        #: ``'lib'``, ``'edge'`` or None -- the strict gate keys off this
        self.pyne_mode = pyne_mode
        #: Re-derives an imported module's table from its source path. Injected
        #: rather than imported: the only real one lives in the import hook,
        #: which imports this pass
        self.analyse = analyse
        #: Digest of the pipeline an imported module's cached interface has to
        #: have been produced by
        self.pipeline_hash = pipeline_hash

    def visit(self, tree: ast.Module) -> ast.Module:
        """
        Stamp the tree and attach its type table.

        :param tree: The module being transformed
        :return: The same module
        """
        module_path = getattr(tree, '_module_file_path', '')
        table = infer_module(tree, module_path, analyse=self.analyse,
                             pipeline_hash=self.pipeline_hash)
        setattr(tree, TABLE_ATTR, table)
        return tree


def module_table(tree: ast.Module) -> PineTypeTable | None:
    """
    Read the table this pass attached, if it ran.

    :param tree: The module node
    :return: The table, or None
    """
    return getattr(tree, TABLE_ATTR, None)
