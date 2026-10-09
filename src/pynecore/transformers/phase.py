"""
Rules and phases: several rewrites over ONE traversal of the tree.

Most of a pass's time is the walk itself, not the rewriting. A :class:`Rule` is a
pass written as per-node hooks instead of ``visit_<Class>`` methods, so the walk
can be shared: :func:`run_phase` runs several rules in a single post-order
traversal, while each rule stays its own class in its own module. A rule is
still a complete transformer -- ``rule.visit(tree)`` runs it alone, which is
exactly a phase of one rule.

The hooks a rule may define:

- ``enter_<Class>(node)`` -- before the node's children are visited;
- ``leave_<Class>(node)`` -- after them, returning the node that replaces it
  (the node itself when nothing changes);
- ``enter_function_body(node)`` / ``leave_function_body(node)`` -- around the
  ``body`` of a ``def`` or ``async def``. The body is a scope of its own, while
  the decorators, defaults and annotations, visited outside these two calls,
  are evaluated in the enclosing one.

At every node the rules' ``enter`` hooks run in the phase's order, then the
children are visited (by every rule), then the ``leave`` hooks run in the same
order. That makes a phase equivalent to running its rules one after the other
as long as two things hold, which the rules of a phase are chosen for:

- what a rule emits holds nothing a later rule of the phase rewrites, because
  in the phase the later rule never visits it;
- a rule's decision at a node does not depend on whether a LATER rule has
  already rewritten the node's children, because in the phase it has.

``PYNE_AST_SEQUENTIAL=1`` (see ``pipeline.py``) runs the same rules as separate
passes instead; that is the reference the phase is checked against.
"""
import ast
from collections.abc import Callable, Sequence
from typing import Any

from . import ast_walk

__all__ = ['Rule', 'run_phase']

#: One hook of one rule: the rule's position in the phase and the hook function
_Hook = tuple[int, Callable[..., Any]]

#: The node classes whose ``body`` field is a function body
_FUNCTION_CLASSES = ('FunctionDef', 'AsyncFunctionDef')

#: Hook name suffix of the function-body hooks
_FUNCTION_BODY = 'function_body'


def _visit_field(walker: ast_walk.NodeTransformer, node: ast.AST, field: str) -> None:
    """Visit one field of a node and write the result back as ``generic_visit`` does."""
    old_value = node.__dict__.get(field)
    if isinstance(old_value, list):
        new_values = []
        for value in old_value:
            if isinstance(value, ast.AST):
                value = walker.visit(value)
                if value is None:
                    continue
                if not isinstance(value, ast.AST):
                    new_values.extend(value)
                    continue
            new_values.append(value)
        old_value[:] = new_values
    elif isinstance(old_value, ast.AST):
        new_node = walker.visit(old_value)
        if new_node is None:
            delattr(node, field)
        else:
            setattr(node, field, new_node)


def _visit_method(enters: Sequence[_Hook], leaves: Sequence[_Hook],
                  body_enters: Sequence[_Hook] = (),
                  body_leaves: Sequence[_Hook] = ()) -> Callable[[Any, ast.AST], ast.AST]:
    """Build the visit method of one node class from the hooks the rules have for it."""

    if body_enters or body_leaves:
        def visit_children(walker: Any, node: ast.AST) -> ast.AST:
            rules = walker.rules
            # The fields in ``generic_visit`` order, the body bracketed by its hooks
            for field in node._fields:
                if field != 'body':
                    _visit_field(walker, node, field)
                    continue
                for index, hook in body_enters:
                    hook(rules[index], node)
                _visit_field(walker, node, field)
                for index, hook in body_leaves:
                    hook(rules[index], node)
            return node
    else:
        def visit_children(walker: Any, node: ast.AST) -> ast.AST:
            return walker.generic_visit(node)

    def visit(walker: Any, node: ast.AST) -> ast.AST:
        rules = walker.rules
        for index, hook in enters:
            hook(rules[index], node)
        node = visit_children(walker, node)
        for index, hook in leaves:
            node = hook(rules[index], node)
        return node

    return visit


def _visit_methods(rule_classes: Sequence[type]) -> dict[str, Callable[[Any, ast.AST], ast.AST]]:
    """The ``visit_<Class>`` methods that run these rules' hooks, in this order."""
    hooks: dict[tuple[str, str], list[_Hook]] = {}
    for index, rule_class in enumerate(rule_classes):
        for name in dir(rule_class):
            kind, _, target = name.partition('_')
            if kind in ('enter', 'leave') and target:
                hooks.setdefault((kind, target), []).append((index, getattr(rule_class, name)))
    body_enters = hooks.pop(('enter', _FUNCTION_BODY), [])
    body_leaves = hooks.pop(('leave', _FUNCTION_BODY), [])
    targets = {target for _, target in hooks}
    if body_enters or body_leaves:
        targets.update(_FUNCTION_CLASSES)
    methods = {}
    for target in targets:
        if not isinstance(getattr(ast, target, None), type):
            raise TypeError(f"hook for unknown node class {target!r}")
        function = target in _FUNCTION_CLASSES
        methods['visit_' + target] = _visit_method(
            hooks.get(('enter', target), []), hooks.get(('leave', target), []),
            body_enters if function else (), body_leaves if function else ())
    return methods


class Rule(ast_walk.NodeTransformer):
    """
    A pass written as per-node hooks (see the module docstring).

    Its ``visit_<Class>`` methods are generated from its hooks, so a rule run on
    its own is an ordinary transformer.
    """

    @property
    def rules(self) -> tuple['Rule']:
        """The rules the walk runs: a rule on its own is a phase of one."""
        return (self,)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # A phase walks with the hooks alone, so a visit method of the rule's
        # own would only ever run when the rule runs by itself
        own_visits = [name for name in vars(cls) if name.startswith('visit_')]
        if own_visits:
            raise TypeError(f"{cls.__name__} defines {', '.join(own_visits)}; a rule has hooks")
        for name, method in _visit_methods((cls,)).items():
            setattr(cls, name, method)


class _PhaseWalker(ast_walk.NodeTransformer):
    """The transformer that walks the rules of a phase; a subclass per set of rules
    carries the generated ``visit_<Class>`` methods."""

    def __init__(self, rules: Sequence[Rule]):
        self.rules = tuple(rules)


#: Rule classes of a phase -> the walker class that runs them together
_walkers: dict[tuple[type, ...], type[_PhaseWalker]] = {}


def run_phase(rules: Sequence[Rule], tree: ast.AST) -> ast.AST:
    """
    Run several rules over a tree in one traversal.

    :param rules: The rules, in the order they would run as separate passes.
    :param tree: The tree; it is transformed in place where the rules can.
    :return: The transformed tree.
    """
    key = tuple(rule.__class__ for rule in rules)
    walker_class = _walkers.get(key)
    if walker_class is None:
        walker_class = _walkers[key] = type('PhaseWalker', (_PhaseWalker,), _visit_methods(key))
    return walker_class(rules).visit(tree)
