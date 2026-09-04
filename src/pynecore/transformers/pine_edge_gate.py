"""
The ``@pyne edge`` gate: what a module may contain to be Pine.

An edge module is a promise: everything in it is Pine, so it runs on the fast
path, compiles ahead of time and runs on the web. The promise has two halves
and this module keeps the structural one -- the SYNTAX a Pine program can be
written with. The other half, that every value has a known Pine type, is what
the type inference reports through its own diagnostics; the gate in the
import hook raises the first one of either kind.

The rules are not written here. The Pyne Edge profile is defined once, in
the PyneIDE's ``pyneide_edge_rules.py``, and ``scripts/edge_rules_collector.py``
extracts it into ``edge_rules.json`` next to this file, versioned by the
spec's own ``EDGE_RULES_VERSION``. This module pins that version: a spec
revision that is not re-extracted fails at import rather than gating against
stale rules. What the data cannot express -- the structural rules: what a
class may contain, which callee a bare name may be, where a ``lambda`` may
stand -- is code here, and mirrors the IDE's checker rule for rule.

Only an ``@pyne edge`` module is gated. PyneCore's own ``@pyne lib`` modules
are the machines behind the lib, written in Python -- ``ta.py`` alone uses
well over a hundred constructs the profile has no place for -- and a profile
that allowed them would allow Python. Their mode word selects the series
semantics of a builtin machine, nothing about this gate.
"""
import ast
import json
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Final

from .pine_type_rules import FactoryFields
from .pine_type_table import Diag, Unknown

__all__ = ['EDGE_RULES_VERSION', 'STRICT_ENV', 'DIAG_ENV', 'edge_rules', 'strict_enabled',
           'diag_dump_enabled', 'gated', 'gate_module', 'render_diags']

#: The profile revision this gate was written against. ``edge_rules.json``
#: must carry the same one.
EDGE_RULES_VERSION: Final = '2026.07.1'

#: ``1`` makes an edge module's first diagnostic an error, ``0`` never
#: does; unset is ``0`` until the corpus is clean enough to flip the default.
STRICT_ENV: Final = 'PYNE_EDGE_STRICT'

#: ``1`` prints every diagnostic of every analysed module to stderr, in any
#: mode -- the coverage meter for hand-written code.
DIAG_ENV: Final = 'PYNE_TYPE_DIAG'

#: Script decorators, spelled ``@script.<name>(...)`` on the lib's ``script``.
SCRIPT_DECORATORS: Final = frozenset({'indicator', 'strategy', 'library'})

#: The two roots a ``script`` chain may hang from.
_LIB_MODULES: Final = ('pynecore.lib', 'pynecore')

#: How a rejected node is named in a message.
_NODE_LABELS: Final[dict[str, str]] = {
    'AsyncFunctionDef': "'async def'", 'AsyncFor': "'async for'",
    'AsyncWith': "'async with'", 'Await': "'await'",
    'Try': "'try'", 'TryStar': "'try'", 'Raise': "'raise'",
    'Assert': "'assert'", 'With': "'with'", 'Delete': "'del'",
    'Global': "'global'", 'Nonlocal': "'nonlocal'", 'Match': "'match'",
    'Yield': "'yield'", 'YieldFrom': "'yield from'",
    'List': 'a list literal', 'Dict': 'a dict literal',
    'Set': 'a set literal', 'ListComp': 'a list comprehension',
    'SetComp': 'a set comprehension', 'DictComp': 'a dict comprehension',
    'GeneratorExp': 'a generator expression', 'JoinedStr': 'an f-string',
    'NamedExpr': "a ':=' assignment", 'Starred': 'a starred expression',
    'Slice': 'a slice',
}

#: How a rejected operator is spelled in a message.
_OP_SYMBOLS: Final[dict[str, str]] = {
    'BitOr': '|', 'BitAnd': '&', 'BitXor': '^', 'LShift': '<<',
    'RShift': '>>', 'MatMult': '@', 'Invert': '~', 'Is': 'is',
    'IsNot': 'is not', 'In': 'in', 'NotIn': 'not in',
}

_SUFFIX: Final = 'is not Pine'

_rules: dict[str, Any] | None = None


def edge_rules() -> dict[str, Any]:
    """
    The extracted profile, read once.

    :return: The ``edge_rules.json`` document
    :raises RuntimeError: When the file was extracted from another revision of
                          the spec than this gate was written against
    """
    global _rules
    loaded = _rules
    if loaded is None:
        path = Path(__file__).with_name('edge_rules.json')
        loaded = json.loads(path.read_text(encoding='utf-8'))
        if loaded.get('rules_version') != EDGE_RULES_VERSION:
            raise RuntimeError(
                f'edge_rules.json carries profile {loaded.get("rules_version")!r}, this gate '
                f'is written against {EDGE_RULES_VERSION!r} -- rerun '
                f'scripts/edge_rules_collector.py and revise the gate')
        _rules = loaded
    return loaded


def strict_enabled() -> bool:
    """Whether an edge module's diagnostics are errors: ``PYNE_EDGE_STRICT=1``."""
    return os.environ.get(STRICT_ENV, '0').strip() == '1'


def diag_dump_enabled() -> bool:
    """Whether every module's diagnostics are printed: ``PYNE_TYPE_DIAG=1``."""
    return os.environ.get(DIAG_ENV, '0').strip() == '1'


def gated(pyne_mode: str | None) -> bool:
    """
    Whether a module's mode word puts it under the gate.

    :param pyne_mode: The ``@pyne`` mode word, None for a hand-written script
    :return: True for ``edge``
    """
    return pyne_mode == 'edge'


def render_diags(diags: Iterable[Diag], path: str) -> str:
    """
    The diagnostics as the dump prints them, one per line.

    :param diags: The diagnostics, in report order
    :param path: The module's source path
    :return: The lines, joined
    """
    lines = []
    for diag in diags:
        origin = ''
        if diag.origin is not None:
            origin = f' [{diag.origin.reason}@{diag.origin.line}:{diag.origin.col}'
            origin += f' {diag.origin.detail}]' if diag.origin.detail else ']'
        fix = f' fix: {diag.fix}' if diag.fix else ''
        lines.append(f'{path}:{diag.line}:{diag.col}: {diag.message}{origin}{fix}')
    return '\n'.join(lines)


def gate_module(tree: ast.Module) -> list[Diag]:
    """
    Every construct of a module the profile does not allow, in source order.

    Runs on the tree AS PARSED -- before any transform, because the transforms
    inject the plumbing (state parameters, security reads, hoisted
    temporaries) that no script wrote and no profile should judge. Each
    finding names the construct and the nearest Pine form; the walk does not
    descend into a rejected construct, so one construct is one finding.

    :param tree: The parsed module
    :return: The findings, as diagnostics with an ``edge-*`` origin
    """
    return _Gate(tree).run()


class _Gate:
    """One walk over one module."""

    def __init__(self, tree: ast.Module):
        rules = edge_rules()
        self.tree = tree
        self.nodes: frozenset[str] = frozenset(rules['nodes'])
        self.bin_ops: frozenset[str] = frozenset(rules['bin_ops'])
        self.unary_ops: frozenset[str] = frozenset(rules['unary_ops'])
        self.bool_ops: frozenset[str] = frozenset(rules['bool_ops'])
        self.cmp_ops: frozenset[str] = frozenset(rules['cmp_ops'])
        self.import_prefixes: tuple[str, ...] = tuple(rules['import_prefixes'])
        self.from_modules: dict[str, frozenset[str]] = {
            module: frozenset(names) for module, names in rules['from_modules'].items()}
        self.func_decorators: frozenset[tuple[str, str]] = frozenset(
            (module, name) for module, name in rules['func_decorators']
        ) | frozenset((module, name)
                      for module, name in rules['extras'].get('func_decorators', ()))
        self.class_decorators: frozenset[tuple[str, str]] = frozenset(
            (module, name) for module, name in rules['class_decorators'])
        self.builtin_calls: frozenset[str] = frozenset(rules['builtin_calls'])
        extras = rules['extras']
        self.dunder_names: frozenset[str] = frozenset(extras.get('dunder_names', ()))
        self.dunder_patterns: list[re.Pattern[str]] = [
            re.compile(pattern) for pattern in extras.get('dunder_patterns', ())]
        self.diags: list[Diag] = []
        #: bound name -> (module, original name) for every ``from X import y [as z]``
        self.from_imports: dict[str, tuple[str, str]] = {}
        #: Every name any import binds, allowed or not: a rejected import is
        #: reported once, its uses must not cascade
        self.import_bound: set[str] = set()
        #: Names bound to the lib itself (``lib``) or to its ``script``
        self.lib_names: set[str] = set()
        self.script_names: set[str] = set()
        self.def_names: set[str] = set()
        #: name -> index of the first module statement that binds it by
        #: assignment or definition: a rebound import is not the import
        self.module_bound_at: dict[str, int] = {}
        self.allowed_lambdas: set[int] = set()
        self.allowed_lists: set[int] = set()
        #: For each enclosing function, innermost last: name -> index of the
        #: body statement that first binds it (-1 for a parameter), and the
        #: index of the body statement being visited
        self._locals: list[dict[str, int]] = []
        self._local_index: list[int] = []
        self._stmt_index = 0
        self._reported: set[int] = set()

    def run(self) -> list[Diag]:
        self._collect()
        self._visit(self.tree)
        return self.diags

    # --- collection pre-pass ---------------------------------------------

    def _collect(self) -> None:
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.def_names.add(node.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    bound = alias.asname or alias.name
                    self.from_imports[bound] = (module, alias.name)
                    self.import_bound.add(bound)
                    if module == 'pynecore' and alias.name == 'lib':
                        self.lib_names.add(bound)
                    elif module == 'pynecore.lib' and alias.name == 'script':
                        self.script_names.add(bound)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    bound = alias.asname or alias.name.split('.')[0]
                    self.import_bound.add(bound)
                    if alias.asname is not None and alias.name == 'pynecore.lib':
                        self.lib_names.add(alias.asname)
        for index, stmt in enumerate(self.tree.body):
            for name in _stored_names(stmt):
                self.module_bound_at.setdefault(name, index)
        # ``field(default_factory=lambda: ...)`` is the one legitimate lambda:
        # a UDT field's default, which the compiler emits too -- and only
        # there, as the value of a field in a decorated class body (the
        # predicate is the type pass's, so the gate and the typing agree)
        factory = FactoryFields(self.tree)
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.ClassDef) or len(node.decorator_list) != 1 \
                    or not self._decorator_ok(node.decorator_list[0], self.class_decorators):
                continue
            for call in factory.of(node):
                self.allowed_lambdas.add(id(call.keywords[0].value))
        # ``__all__ = ['name', ...]`` at module level is the one legitimate
        # list literal: the library emitter produces it
        for stmt in self.tree.body:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 \
                    and isinstance(stmt.targets[0], ast.Name) \
                    and stmt.targets[0].id == '__all__' \
                    and isinstance(stmt.value, ast.List) \
                    and all(isinstance(e, ast.Constant) and isinstance(e.value, str)
                            for e in stmt.value.elts):
                self.allowed_lists.add(id(stmt.value))

    # --- the walk --------------------------------------------------------

    def _visit(self, node: ast.AST) -> None:
        kind = type(node).__name__
        if isinstance(node, (ast.expr_context, ast.boolop, ast.operator, ast.unaryop,
                             ast.cmpop)):
            return
        if kind == 'List' and id(node) in self.allowed_lists:
            return
        if kind not in self.nodes and hasattr(node, 'lineno'):
            label = _NODE_LABELS.get(kind, f"'{kind}'")
            self._report(node, 'edge-syntax', f'{label} {_SUFFIX}',
                         'write it with Pine constructs only')
            return
        if isinstance(node, ast.Module):
            for index, stmt in enumerate(node.body):
                self._stmt_index = index
                self._visit(stmt)
            return
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            self._check_import(node)
        elif isinstance(node, ast.FunctionDef):
            self._check_function(node)
            # The body is visited statement by statement, so a store on a
            # name knows whether the name is bound yet
            self._locals.append(_scope_bound(node))
            self._local_index.append(-1)
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.stmt):
                    continue
                self._visit(child)
            for index, stmt in enumerate(node.body):
                self._local_index[-1] = index
                self._visit(stmt)
            self._locals.pop()
            self._local_index.pop()
            return
        elif isinstance(node, ast.ClassDef):
            self._check_ident(node, node.name)
            if self._is_protocol_shim(node):
                # Only the ellipsis bodies are inert: the signatures still run
                for stmt in node.body:
                    if isinstance(stmt, ast.FunctionDef):
                        self._visit(stmt.args)
                        if stmt.returns is not None:
                            self._visit(stmt.returns)
                return
            self._check_class(node)
        elif isinstance(node, ast.Call):
            self._check_call(node)
        elif isinstance(node, ast.Lambda):
            if id(node) not in self.allowed_lambdas:
                self._report(node, 'edge-lambda',
                             f"'lambda' outside a field(default_factory=...) UDT field "
                             f'default {_SUFFIX}', 'define a function instead')
                return
        elif isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store):
            self._report(node, 'edge-subscript', f'subscript assignment {_SUFFIX}',
                         'use array.set()')
        elif isinstance(node, ast.Attribute):
            self._check_ident(node, node.attr)
            if isinstance(node.ctx, ast.Store):
                self._check_attribute_store(node)
        elif isinstance(node, ast.Name):
            self._check_ident(node, node.id)
        elif isinstance(node, ast.arg):
            self._check_ident(node, node.arg)
        elif isinstance(node, (ast.BinOp, ast.AugAssign)):
            self._check_op(node, node.op, self.bin_ops)
        elif isinstance(node, ast.UnaryOp):
            self._check_op(node, node.op, self.unary_ops)
        elif isinstance(node, ast.BoolOp):
            self._check_op(node, node.op, self.bool_ops)
        elif isinstance(node, ast.Compare):
            if len(node.ops) > 1:
                self._report(node, 'edge-syntax', f'a chained comparison {_SUFFIX}',
                             'compare pairwise, joined with and')
            for op in node.ops:
                self._check_op(node, op, self.cmp_ops)
        elif isinstance(node, (ast.For, ast.While)) and node.orelse:
            self._report(node.orelse[0], 'edge-syntax', f"a loop's else clause {_SUFFIX}",
                         'put the code after the loop')
        elif isinstance(node, ast.keyword):
            if node.arg is None:
                self._report(node.value, 'edge-syntax', f'keyword unpacking {_SUFFIX}',
                             'pass the arguments one by one')
            else:
                self._check_ident(node, node.arg)
        for child in ast.iter_child_nodes(node):
            self._visit(child)

    # --- structural rules ------------------------------------------------

    def _check_import(self, node: ast.Import | ast.ImportFrom) -> None:
        for alias in node.names:
            self._check_ident(node, alias.asname or alias.name.rpartition('.')[2])
        if isinstance(node, ast.ImportFrom):
            if node.level:
                self._report(node, 'edge-import', f'a relative import {_SUFFIX}',
                             'import from pynecore or lib.*')
                return
            if any(alias.name == '*' for alias in node.names):
                self._report(node, 'edge-import', f'a star import {_SUFFIX}',
                             'import the names one by one')
                return
            module = node.module or ''
            if self._has_allowed_prefix(module):
                return
            allowed = self.from_modules.get(module)
            if allowed is None:
                self._report(node, 'edge-import', f"importing from '{module}' {_SUFFIX}",
                             'only the PyneCore API and lib.* libraries are available')
                return
            for alias in node.names:
                if alias.name not in allowed:
                    self._report(node, 'edge-import', f"'{module}.{alias.name}' {_SUFFIX}",
                                 'only the PyneCore API and lib.* libraries are available')
            return
        for alias in node.names:
            if not self._has_allowed_prefix(alias.name):
                self._report(node, 'edge-import', f"importing '{alias.name}' {_SUFFIX}",
                             'only the PyneCore API and lib.* libraries are available')

    def _has_allowed_prefix(self, module: str) -> bool:
        return any(module == prefix or module.startswith(prefix + '.')
                   for prefix in self.import_prefixes)

    def _check_function(self, node: ast.FunctionDef) -> None:
        self._check_ident(node, node.name)
        if _has_special_parameters(node.args):
            self._report(node, 'edge-syntax',
                         f'*args, **kwargs, keyword-only and positional-only parameters '
                         f'{_SUFFIX}', 'declare plain positional parameters')
        for decorator in node.decorator_list:
            if not self._decorator_ok(decorator, self.func_decorators, allow_script=True):
                self._report(decorator, 'edge-decorator', f'this decorator {_SUFFIX}',
                             'only @script.indicator/strategy/library(...), @method, '
                             '@overload and @export exist')

    def _check_class(self, node: ast.ClassDef) -> None:
        if node.bases or node.keywords:
            self._report(node, 'edge-class', f'class inheritance {_SUFFIX}',
                         'a class is a plain @udt/@dataclass field list')
        decorators = node.decorator_list
        if len(decorators) != 1 or not self._decorator_ok(decorators[0], self.class_decorators):
            self._report(node, 'edge-class',
                         f'a class without exactly one @udt or @dataclass decorator {_SUFFIX}',
                         'decorate it with @udt')
        body = node.body
        if body and _is_docstring(body[0]):
            body = body[1:]
        for stmt in body:
            if not isinstance(stmt, (ast.AnnAssign, ast.Pass)):
                self._report(stmt, 'edge-class', f'a class body beyond annotated fields {_SUFFIX}',
                             'declare methods with @method outside the class')

    def _is_protocol_shim(self, node: ast.ClassDef) -> bool:
        """
        A library export's signature shim: ``class _ProtocolX(Protocol)`` with
        ellipsis-body method declarations only -- the compiler's typing
        scaffolding, erased at run time.
        """
        if len(node.bases) != 1 or node.keywords or node.decorator_list \
                or not isinstance(node.bases[0], ast.Name) \
                or not self._is_typing_protocol(node.bases[0].id):
            return False
        body = node.body
        if body and _is_docstring(body[0]):
            body = body[1:]
        return bool(body) and all(
            isinstance(stmt, ast.FunctionDef) and stmt.name == '__call__'
            and not stmt.decorator_list and len(stmt.body) == 1
            and isinstance(stmt.body[0], ast.Expr)
            and isinstance(stmt.body[0].value, ast.Constant)
            and stmt.body[0].value.value is Ellipsis
            for stmt in body)

    def _check_ident(self, node: ast.AST, name: str) -> None:
        """A double-underscore name is the pipeline's unless the compiler spells it."""
        if not _DUNDER.match(name) or name in self.dunder_names \
                or any(pattern.match(name) for pattern in self.dunder_patterns) \
                or id(node) in self._reported:
            return
        self._report(node, 'edge-name', f"the name '{name}' {_SUFFIX}",
                     'double-underscore names belong to the pipeline')

    def _check_call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Attribute):
            if _attribute_chain(func) is None:
                self._report(func, 'edge-call', f'calling a computed expression {_SUFFIX}',
                             'call a defined function or a lib name')
            return
        if not isinstance(func, ast.Name):
            self._report(func, 'edge-call', f'calling a computed expression {_SUFFIX}',
                         'call a defined function or a lib name')
            return
        name = func.id
        if name in self.def_names or name in self.import_bound or name in self.builtin_calls:
            return
        self._report(func, 'edge-call', f"calling '{name}' {_SUFFIX}",
                     f'only defined functions, imported names and '
                     f'{", ".join(sorted(self.builtin_calls))} can be called')

    def _check_attribute_store(self, node: ast.Attribute) -> None:
        chain = _attribute_chain(node)
        if chain is None:
            return
        root = chain[0]
        if any(root in bound and bound[root] <= index
               for bound, index in zip(self._locals, self._local_index)) \
                or self.module_bound_at.get(root, len(self.tree.body)) < self._stmt_index:
            # A value of that name stands in front of the function or module
            return
        if root in self.def_names or root in self.import_bound:
            self._report(node, 'edge-attr-store', f"assigning an attribute on '{root}' {_SUFFIX}",
                         'functions and modules are not objects')

    def _decorator_ok(self, decorator: ast.expr, allowed: frozenset[tuple[str, str]],
                      allow_script: bool = False) -> bool:
        if allow_script and isinstance(decorator, ast.Call):
            # A script decorator configures the script: it is always CALLED
            chain = _attribute_chain(decorator.func)
            if chain is not None:
                if chain[0] in self.lib_names:
                    chain = chain[1:]
                elif chain[0] in self.script_names:
                    chain = ['script', *chain[1:]]
                if len(chain) == 2 and chain[0] == 'script' and chain[1] in SCRIPT_DECORATORS:
                    return True
        if isinstance(decorator, ast.Name):
            return self.from_imports.get(decorator.id) in allowed
        return False

    def _is_typing_protocol(self, name: str) -> bool:
        """
        Whether ``name`` denotes ``typing.Protocol`` at the current statement.

        The binding in effect is the LAST one in source order before the
        statement: a later import, assignment or definition of the same name
        counterfeits the base, whatever an earlier ``from typing import``
        said.
        """
        binding: str | None = None
        for stmt in self.tree.body[:self._stmt_index]:
            if isinstance(stmt, ast.ImportFrom):
                for alias in stmt.names:
                    if (alias.asname or alias.name) == name:
                        binding = 'protocol' if (stmt.module == 'typing' and not stmt.level
                                                 and alias.name == 'Protocol') else 'other'
            elif isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    if (alias.asname or alias.name.split('.')[0]) == name:
                        binding = 'other'
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if stmt.name == name:
                    binding = 'other'
            elif name in _stored_names(stmt):
                binding = 'other'
        return binding == 'protocol'

    def _check_op(self, node: ast.AST, op: ast.AST, allowed: frozenset[str]) -> None:
        kind = type(op).__name__
        if kind not in allowed:
            symbol = _OP_SYMBOLS.get(kind, kind)
            self._report(node, 'edge-syntax', f"the '{symbol}' operator {_SUFFIX}",
                         'write it with Pine operators only')

    def _report(self, node: ast.AST, reason: str, message: str, fix: str) -> None:
        self._reported.add(id(node))
        line = getattr(node, 'lineno', 0)
        col = getattr(node, 'col_offset', 0)
        # A rejected EXPRESSION covers everything written inside it: the
        # typed half has nothing to add about its parts
        spans = isinstance(node, ast.expr)
        self.diags.append(Diag(message=message, line=line, col=col,
                               origin=Unknown(reason=reason, line=line, col=col,
                                              detail=type(node).__name__),
                               fix=fix,
                               end_line=getattr(node, 'end_lineno', 0) or 0 if spans else 0,
                               end_col=getattr(node, 'end_col_offset', 0) or 0 if spans else 0))


def _stored_names(stmt: ast.stmt) -> set[str]:
    """
    The names a statement binds to VALUES in its own scope.

    A nested definition binds a function or a class, which the definition
    rules answer for, and its body binds elsewhere: the walk stops there.
    """
    out: set[str] = set()
    pending: list[ast.AST] = [stmt]
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            out.add(node.id)
        pending.extend(ast.iter_child_nodes(node))
    return out


def _scope_bound(node: ast.FunctionDef) -> dict[str, int]:
    """
    Every name a function binds, with the body statement that first binds it.

    A parameter is bound before the body (-1); a stored name is bound from
    its statement on -- a store on it before that is not a store on the value.
    """
    args = node.args
    out = {arg.arg: -1 for arg in args.posonlyargs + args.args + args.kwonlyargs}
    if args.vararg is not None:
        out[args.vararg.arg] = -1
    if args.kwarg is not None:
        out[args.kwarg.arg] = -1
    for index, stmt in enumerate(node.body):
        for name in _stored_names(stmt):
            out.setdefault(name, index)
    return out


def _has_special_parameters(args: ast.arguments) -> bool:
    return bool(args.posonlyargs or args.kwonlyargs or args.vararg or args.kwarg)


def _is_docstring(stmt: ast.stmt) -> bool:
    return isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) \
        and isinstance(stmt.value.value, str)


#: A double-underscore identifier
_DUNDER = re.compile(r'^__\w+__$')


def _attribute_chain(node: ast.expr) -> list[str] | None:
    """``a.b.c`` as ``['a', 'b', 'c']``, None when the chain does not start at a name."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    parts.reverse()
    return parts
