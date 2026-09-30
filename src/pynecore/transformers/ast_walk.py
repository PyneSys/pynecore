"""
AST traversal and copying for the transform pipeline.

A transform runs some thirty passes over the whole module, and most of its time
is the walking itself rather than the rewriting: the stdlib visitor builds the
``visit_<Class>`` name and looks it up on every node, reaches every field through
the ``iter_fields`` generator, and descends into nodes that can never hold a
child (expression contexts, operators, constants). ``copy.deepcopy`` of a tree
goes through the generic reduce protocol for every node.

The replacements here produce the same trees:

- :class:`NodeVisitor` / :class:`NodeTransformer` visit the same nodes in the
  same order and write back exactly what the stdlib classes write back. The
  visitor method is resolved once per (visitor class, node class) pair, and a
  node that has no child and no visitor method of its own is not entered at
  all -- visiting it would do nothing but hand it back.
- :func:`walk` and :func:`iter_child_nodes` yield what their ``ast``
  namesakes yield, in the same order.
- :func:`clone` is ``copy.deepcopy`` for trees: same memo protocol, same
  result, attributes the passes stamp on nodes included.
"""
import ast
import copy
from collections import deque
from collections.abc import Callable, Iterator
from typing import Any

__all__ = ['NodeVisitor', 'NodeTransformer', 'walk', 'iter_child_nodes', 'iter_descendants',
           'clone']

_AST = ast.AST

#: Node classes no field of which ever holds another node
_LEAF_TYPES: frozenset[type] = frozenset(
    cls for cls in vars(ast).values()
    if isinstance(cls, type) and issubclass(cls, _AST) and (
        issubclass(cls, (ast.expr_context, ast.operator, ast.unaryop, ast.cmpop, ast.boolop))
        or cls in (ast.Constant, ast.alias, ast.Pass, ast.Break, ast.Continue, ast.Global,
                   ast.Nonlocal, ast.TypeIgnore, ast.MatchSingleton)))

#: The stdlib's ``visit_Constant``, which only forwards to the deprecated
#: ``visit_Num`` / ``visit_Str`` / ... methods (absent in newer Pythons)
_STD_VISIT_CONSTANT = getattr(ast.NodeVisitor, 'visit_Constant', None)
_LEGACY_CONSTANT_VISITORS = ('visit_Num', 'visit_Str', 'visit_Bytes', 'visit_NameConstant',
                             'visit_Ellipsis')

#: Marks a node the visit leaves alone: a leaf without a visitor method
_SKIP = object()

#: (visitor class, node class) -> the function visiting it, or ``_SKIP``
_dispatch: dict[tuple[type, type], Any] = {}


def _resolve(visitor_cls: type, node_cls: type) -> Any:
    """Resolve and remember the function a visitor class visits a node class with."""
    method = getattr(visitor_cls, 'visit_' + node_cls.__name__, None)
    if method is not None and method is _STD_VISIT_CONSTANT and not any(
            hasattr(visitor_cls, name) for name in _LEGACY_CONSTANT_VISITORS):
        method = None
    if method is None:
        method = visitor_cls.generic_visit
        if node_cls in _LEAF_TYPES and method in _DEFAULT_GENERIC:
            method = _SKIP
    _dispatch[(visitor_cls, node_cls)] = method
    return method


class NodeVisitor(ast.NodeVisitor):
    """``ast.NodeVisitor`` with cached dispatch; see the module docstring."""

    def visit(self, node: ast.AST) -> Any:
        key = (self.__class__, node.__class__)
        method = _dispatch.get(key)
        if method is None:
            method = _resolve(*key)
        if method is _SKIP:
            return None
        return method(self, node)

    def generic_visit(self, node: ast.AST) -> None:
        cls = self.__class__
        if cls.visit is not NodeVisitor.visit:
            # A subclass that intercepts every visit must see every node
            for child in iter_child_nodes(node):
                self.visit(child)
            return
        dispatch = _dispatch
        for field in node._fields:
            try:
                value = getattr(node, field)
            except AttributeError:
                continue
            if isinstance(value, _AST):
                method = dispatch.get((cls, value.__class__))
                if method is None:
                    method = _resolve(cls, value.__class__)
                if method is not _SKIP:
                    method(self, value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, _AST):
                        method = dispatch.get((cls, item.__class__))
                        if method is None:
                            method = _resolve(cls, item.__class__)
                        if method is not _SKIP:
                            method(self, item)


class NodeTransformer(ast.NodeTransformer):
    """``ast.NodeTransformer`` with cached dispatch; see the module docstring.

    The write-back is the stdlib's to the letter -- every list field is
    reassigned in place and every node field set again -- because a visitor
    may edit its parent's fields while the parent is being walked, and the
    stdlib's write-back is what decides which of the two edits survives.
    """

    def visit(self, node: ast.AST) -> Any:
        key = (self.__class__, node.__class__)
        method = _dispatch.get(key)
        if method is None:
            method = _resolve(*key)
        if method is _SKIP:
            return node
        return method(self, node)

    def generic_visit(self, node: ast.AST) -> ast.AST:
        cls = self.__class__
        fast = cls.visit is NodeTransformer.visit
        dispatch = _dispatch
        for field in node._fields:
            try:
                old_value = getattr(node, field)
            except AttributeError:
                continue
            if isinstance(old_value, list):
                new_values = []
                for value in old_value:
                    if isinstance(value, _AST):
                        if fast:
                            method = dispatch.get((cls, value.__class__))
                            if method is None:
                                method = _resolve(cls, value.__class__)
                            if method is _SKIP:
                                new_values.append(value)
                                continue
                            value = method(self, value)
                        else:
                            value = self.visit(value)
                        if value is None:
                            continue
                        if not isinstance(value, _AST):
                            new_values.extend(value)
                            continue
                    new_values.append(value)
                old_value[:] = new_values
            elif isinstance(old_value, _AST):
                if fast:
                    method = dispatch.get((cls, old_value.__class__))
                    if method is None:
                        method = _resolve(cls, old_value.__class__)
                    if method is _SKIP:
                        continue
                    new_node = method(self, old_value)
                else:
                    new_node = self.visit(old_value)
                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)
        return node


#: The generic visits a leaf may be skipped for: the ones defined above
_DEFAULT_GENERIC = (NodeVisitor.generic_visit, NodeTransformer.generic_visit)


def iter_child_nodes(node: ast.AST) -> Iterator[ast.AST]:
    """The direct child nodes of ``node``, as :func:`ast.iter_child_nodes` yields them."""
    for field in node._fields:
        try:
            value = getattr(node, field)
        except AttributeError:
            continue
        if isinstance(value, _AST):
            yield value
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, _AST):
                    yield item


def walk(node: ast.AST) -> Iterator[ast.AST]:
    """Every node of a tree, breadth first, as :func:`ast.walk` yields them.

    As in :func:`ast.walk`, a node's children are taken before the node is
    handed out, so editing a yielded node does not change what is walked.
    """
    todo = deque([node])
    popleft = todo.popleft
    append = todo.append
    while todo:
        node = popleft()
        for field in node._fields:
            try:
                value = getattr(node, field)
            except AttributeError:
                continue
            if isinstance(value, _AST):
                append(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, _AST):
                        append(item)
        yield node


def iter_descendants(node: ast.AST, stop: type | tuple[type, ...] = ()) -> Iterator[ast.AST]:
    """Every descendant of ``node`` in pre-order, not yielding or entering ``stop`` nodes.

    The order, and the moment each node's children are read, are those of the
    recursive generator this replaces::

        for child in ast.iter_child_nodes(node):
            if not isinstance(child, stop):
                yield child
                yield from iter_descendants(child, stop)

    without its cost of one ``yield from`` hop per tree level for every node.

    :param node: The node whose descendants are walked (not yielded itself)
    :param stop: Node classes skipped together with their whole subtree
    """
    stack = [iter_child_nodes(node)]
    push = stack.append
    while stack:
        for child in stack[-1]:
            if isinstance(child, stop):
                continue
            yield child
            push(iter_child_nodes(child))
            break
        else:
            stack.pop()


#: Values ``copy.deepcopy`` hands back as they are
_ATOMIC: frozenset[type] = frozenset({
    type(None), int, float, bool, complex, bytes, str, type, type(Ellipsis),
    type(NotImplemented), range,
})

_deepcopy: Callable[..., Any] = copy.deepcopy


def clone(node: Any, memo: dict[int, Any] | None = None) -> Any:
    """``copy.deepcopy`` of a tree, list of nodes, or anything holding them.

    Same result and same memo protocol: a node referenced twice within one
    copy (one ``memo``) is copied once, and every attribute a pass stamped on
    a node is copied with it. Values that are neither nodes nor lists go
    through ``copy.deepcopy`` itself, with the same memo.

    :param node: What to copy
    :param memo: The ``copy.deepcopy`` memo, shared across calls to copy
                 several objects as one
    :return: The copy
    """
    if memo is None:
        memo = {}
    return _clone(node, memo)


def _clone(value: Any, memo: dict[int, Any]) -> Any:
    cls = value.__class__
    if cls in _ATOMIC:
        return value
    key = id(value)
    hit = memo.get(key, memo)
    if hit is not memo:
        return hit
    if cls is list:
        new_list: list[Any] = []
        memo[key] = new_list
        for item in value:
            new_list.append(item if item.__class__ in _ATOMIC else _clone(item, memo))
        _keep_alive(value, memo)
        return new_list
    if isinstance(value, _AST):
        state = value.__dict__
        fields = cls._fields
        # ``deepcopy`` rebuilds a node through its constructor, which fills an
        # absent field with its default; only a node with every field present
        # is guaranteed to come out the same without that call
        for field in fields:
            if field not in state:
                return _deepcopy(value, memo)
        new = cls.__new__(cls)
        memo[key] = new
        new_state = new.__dict__
        # The constructor sets the fields first, then the state is laid over
        # them: same key order as a ``deepcopy``
        for field in fields:
            item = state[field]
            new_state[field] = item if item.__class__ in _ATOMIC else _clone(item, memo)
        for name, item in state.items():
            if name not in new_state:
                new_state[name] = item if item.__class__ in _ATOMIC else _clone(item, memo)
        _keep_alive(value, memo)
        return new
    return _deepcopy(value, memo)


def _keep_alive(value: Any, memo: dict[int, Any]) -> None:
    """Keep a copied original alive for the memo's lifetime, as ``copy`` does."""
    try:
        memo[id(memo)].append(value)
    except KeyError:
        memo[id(memo)] = [value]
