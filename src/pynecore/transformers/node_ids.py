"""
Stable per-node identifiers for diagnostics.

The Pine type inference stamps the type ON the node (``node._pine_ty``), so it
needs no identifier to carry a type table; the ids exist purely so a
diagnostic can name a node across passes -- "the UNKNOWN entered at #412" reads
better in a dump than a line/column pair that later passes rewrite.

The numbering is a deterministic pre-order walk, so the same source always
produces the same ids, which is what makes an artifact diffable.
"""
import ast

__all__ = ['assign_node_ids', 'node_id']

#: Attribute the id is stamped under. Leading underscore keeps it out of
#: ``ast.iter_fields`` and out of anything that reconstructs a node.
_ID_ATTR = '_pine_nid'


def assign_node_ids(tree: ast.AST, start: int = 0) -> int:
    """
    Number every node of a tree in pre-order.

    Re-running this on a tree that already carries ids overwrites them, which
    is deliberate: the ids are a debugging handle, never an identity.

    :param tree: The tree to number
    :param start: First id to hand out
    :return: The next unused id
    """
    nid = start
    stack: list[ast.AST] = [tree]
    while stack:
        node = stack.pop()
        setattr(node, _ID_ATTR, nid)
        nid += 1
        # Reversed so the children come off the stack in source order
        stack.extend(reversed(list(ast.iter_child_nodes(node))))
    return nid


def node_id(node: ast.AST) -> int | None:
    """
    Read a node's id.

    :param node: The node to look at
    :return: Its id, or None when the tree was never numbered
    """
    return getattr(node, _ID_ATTR, None)
