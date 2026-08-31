"""
The Pine type inference engine.

Walks a module and gives every expression a Pine type, which it stamps on the
node itself (``node._pine_ty``). Later passes reuse the node objects, so the
stamp travels with the tree into the lowered form the AOT compiler consumes;
the passes that BUILD a wrapper node have to carry it over explicitly, which
is what ``inherit_ty`` is for.

Completeness on the Pine-expressible subset is the point, not best effort.
Int-ness has a CLOSED set of origins -- an int literal, an ``int``-ish
annotation, an int-returning lib name, an ``int()`` cast -- and travels over a
closed set of operators, so anything still UNKNOWN afterwards has genuinely
left the Pine world. Two leaks stay open in this phase and are closed later:
an unannotated user-function parameter (monomorphization) and a loop-carried
variable, which the bounded fixpoint here already handles.

This module is analysis-only: it rewrites nothing, clones nothing and pins
nothing. That keeps it testable one rule at a time, and keeps the rules
(``pine_type_rules``) separable from the walking done here.
"""
import ast
import json
from pathlib import Path
from typing import Any

from .node_ids import assign_node_ids
from .pine_type_rules import (
    INT, FLOAT, BOOL, STR, UNKNOWN, VOID, OBJECT, NUMERIC,
    join, binop_type, unaryop_type, compare_type, annotation_type,
    LIB_TYPE_OVERRIDES, BUILTIN_CALL_TYPES, TY_ATTR, get_ty, set_ty, inherit_ty,
    constant_type,
)
from .pine_type_table import Binding, CallSite, Diag, FuncSig, PineTypeTable, Unknown

__all__ = ['infer_module', 'lib_types', 'TY_ATTR', 'get_ty', 'set_ty', 'inherit_ty']

#: How many times a loop body is re-inferred before the types are declared
#: stable. The lattice is two high (int -> float -> unknown), so a binding can
#: only move twice; a third pass exists to OBSERVE that nothing moved.
_MAX_LOOP_PASSES = 3

_LIB_TYPES_PATH = Path(__file__).parent / 'lib_types.json'
_LIB_TYPES: dict[str, Any] = {}


def lib_types() -> dict[str, Any]:
    """
    The generated lib registry, loaded once.

    Read from JSON rather than by importing the lib: this module is imported
    by the import hook while it transforms pynecore's own lib modules, and an
    import there would re-enter a half-initialized package (the same reason
    ``const_fold`` defers its lib import).

    :return: name -> entry mapping
    """
    if not _LIB_TYPES:
        _LIB_TYPES.update(json.loads(_LIB_TYPES_PATH.read_text())['names'])
    return _LIB_TYPES


def infer_module(tree: ast.Module, module_path: str = '') -> PineTypeTable:
    """
    Infer and stamp the Pine types of a whole module.

    An unannotated helper's return type is only known once its body has been
    walked, so a call that appears BEFORE the definition -- a helper calling
    a helper defined further down, or two mutually recursive ones -- would
    read UNKNOWN on a single pass. When a pass hits such a call, the walk is
    repeated with what it learned, until the inferred return types stop
    moving; a module with no forward helper call is walked exactly once.

    :param tree: The module to walk; it is stamped in place
    :param module_path: Absolute source path, for diagnostics
    :return: The derived type table
    """
    assign_node_ids(tree)
    learned: dict[str, str] = {}
    engine = _Inference(module_path, learned)
    engine.run(tree)
    # Every pass resolves at least one more link of a forward call chain, and
    # a chain cannot be longer than the module has functions -- so the count
    # is the bound, not an arbitrary number that would leave a long chain
    # half-typed. The extra pass exists to OBSERVE that nothing moved.
    for _ in range(len(engine.table.funcs) + 1):
        settled = engine.inferred_returns()
        if not engine.saw_forward_call or settled == learned:
            break
        learned = settled
        engine = _Inference(module_path, learned)
        engine.run(tree)
    return engine.table


class _Inference:
    """The walker. One instance per module."""

    def __init__(self, module_path: str, learned_returns: dict[str, str] | None = None):
        self.table = PineTypeTable(module_path=module_path)
        #: Return types a previous pass inferred for the unannotated functions
        self._learned_returns = learned_returns or {}
        #: Scope ids, outermost first; the module scope is the empty string
        self._scopes: list[str] = ['']
        self.table.bindings[''] = {}
        #: Names bound by the enclosing lib import, e.g. ``lib``
        self._lib_aliases: set[str] = set()
        #: Whether each function spelled its return type out
        self._annotated_returns: dict[str, bool] = {}
        #: Set when a call resolved to a not-yet-walked function, which is the
        #: only thing another pass can improve on
        self.saw_forward_call = False

    # --- scope plumbing --------------------------------------------------

    @property
    def _scope(self) -> str:
        return self._scopes[-1]

    @staticmethod
    def _qualify(scope: str, name: str) -> str:
        """The scope-qualified identity of a name declared in one scope."""
        return f'{scope}·{name}' if scope else name

    def _resolve_func(self, name: str) -> str | None:
        """
        The signature key a call name resolves to, searching outward.

        Function signatures are keyed the same way bindings are, because a bare
        ``helper()`` means a DIFFERENT function in each enclosing scope that
        defines one; keying by the bare name alone let two same-named nested
        helpers overwrite each other's return type.

        :param name: The name as the call spells it
        :return: The key in ``table.funcs``, or None when nothing declares it
        """
        for scope in reversed(self._scopes):
            key = self._qualify(scope, name)
            if key in self.table.funcs:
                return key
        return None

    def _bindings(self) -> dict[str, Binding]:
        return self.table.bindings.setdefault(self._scope, {})

    def _lookup(self, name: str) -> Binding | None:
        """Find a name in the innermost scope that has it."""
        for scope in reversed(self._scopes):
            found = self.table.bindings.get(scope, {}).get(name)
            if found is not None:
                return found
        return None

    def _bind(self, name: str, ty: str, node: ast.AST, unknown: Unknown | None = None) -> None:
        """
        Record an assignment.

        Re-assigning a name JOINS with what it already had: Pine's variables
        are single-typed, and a branch that stores a float into an int-typed
        variable widens it for every later read.
        """
        bindings = self._bindings()
        existing = bindings.get(name)
        line = getattr(node, 'lineno', 0)
        if existing is None:
            bindings[name] = Binding(name=name, ty=ty, line=line, unknown=unknown)
            return
        joined = join(existing.ty, ty)
        existing.ty = joined
        if joined == UNKNOWN and existing.unknown is None:
            existing.unknown = unknown or self._unknown('joined-branches', node, name)

    def _unknown(self, reason: str, node: ast.AST, detail: str = '') -> Unknown:
        return Unknown(reason=reason, line=getattr(node, 'lineno', 0),
                       col=getattr(node, 'col_offset', 0), detail=detail)

    def _diag(self, message: str, node: ast.AST, origin: Unknown | None = None,
              fix: str = '') -> None:
        self.table.diags.append(Diag(
            message=message, line=getattr(node, 'lineno', 0),
            col=getattr(node, 'col_offset', 0), origin=origin, fix=fix))

    # --- entry point -----------------------------------------------------

    def run(self, tree: ast.Module) -> None:
        """Walk a module: collect the lib aliases, then the bodies."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == 'pynecore':
                self._lib_aliases.update(a.asname or a.name for a in node.names)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith('pynecore.lib'):
                        self._lib_aliases.add(alias.asname or alias.name.split('.')[0])
        # A module with no explicit import still spells lib references
        # ``lib.<name>`` after normalization
        self._lib_aliases.add('lib')

        self._declare_functions(tree.body)
        self._body(tree.body)

    def inferred_returns(self) -> dict[str, str]:
        """
        The return types this pass DERIVED, to seed the next one.

        Only the unannotated functions are reported, and only once they have
        a type: an annotated module then produces an empty mapping and needs
        no second pass.

        :return: scope-qualified function id -> return type
        """
        return {key: signature.ret for key, signature in self.table.funcs.items()
                if not self._annotated_returns.get(key) and signature.ret != UNKNOWN}

    def _declare_functions(self, body: list[ast.stmt], scope: str = '') -> None:
        """
        Record every function's annotated signature before walking bodies.

        Doing this first is what lets a call be typed from the callee's
        annotations regardless of definition order. The key is the function's
        own scope id, so a nested helper never collides with a same-named one
        under another parent.

        :param body: The statements to scan
        :param scope: Scope id the definitions live in, empty at module level
        """
        for stmt in body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                params = [annotation_type(a.annotation)
                          for a in list(stmt.args.posonlyargs) + list(stmt.args.args)]
                key = self._qualify(scope, stmt.name)
                declared = annotation_type(stmt.returns)
                if declared == UNKNOWN:
                    declared = self._learned_returns.get(key, UNKNOWN)
                self.table.funcs[key] = FuncSig(
                    name=stmt.name, params=params, ret=declared, line=_line(stmt))
                self._annotated_returns[key] = annotation_type(stmt.returns) != UNKNOWN
                self._declare_functions(stmt.body, key)

    # --- statements ------------------------------------------------------

    def _body(self, body: list[ast.stmt]) -> None:
        for stmt in body:
            self._stmt(stmt)

    def _stmt(self, stmt: ast.stmt) -> None:
        match stmt:
            case ast.FunctionDef() | ast.AsyncFunctionDef():
                self._function(stmt)
            case ast.Assign():
                ty = self._expr(stmt.value)
                for target in stmt.targets:
                    self._store(target, ty, stmt.value)
            case ast.AnnAssign():
                declared = annotation_type(stmt.annotation)
                if stmt.value is not None:
                    self._expr(stmt.value)
                # An explicit annotation is a DECLARATION: it wins over what
                # the initializer happens to be, the way Pine's `int x = ...`
                # does. That is the whole point of writing one.
                self._store(stmt.target, declared, stmt)
            case ast.AugAssign():
                value_ty = self._expr(stmt.value)
                current = self._target_type(stmt.target)
                self._store(stmt.target, binop_type(stmt.op, current, value_ty), stmt)
            case ast.Return():
                if stmt.value is not None:
                    self._expr(stmt.value)
            case ast.If():
                self._expr(stmt.test)
                self._body(stmt.body)
                self._body(stmt.orelse)
            case ast.While():
                self._loop(lambda: (self._expr(stmt.test), self._body(stmt.body)))
            case ast.For() | ast.AsyncFor():
                iter_ty = self._expr(stmt.iter)
                self._store(stmt.target, self._element_type(stmt.iter, iter_ty), stmt)
                self._loop(lambda: self._body(stmt.body))
                self._body(stmt.orelse)
            case ast.Expr():
                self._expr(stmt.value)
            case ast.With() | ast.AsyncWith():
                for item in stmt.items:
                    self._expr(item.context_expr)
                self._body(stmt.body)
            case ast.Try():
                self._body(stmt.body)
                for handler in stmt.handlers:
                    self._body(handler.body)
                self._body(stmt.orelse)
                self._body(stmt.finalbody)
            case ast.ClassDef():
                self._body(stmt.body)
            case _:
                # Import, Pass, Break, Continue, Global, Nonlocal, Delete:
                # nothing to type
                for child in ast.iter_child_nodes(stmt):
                    if isinstance(child, ast.expr):
                        self._expr(child)

    def _loop(self, walk) -> None:
        """
        Run a loop body until its bindings stop moving.

        A loop-carried variable is the one place a single forward pass is not
        enough: ``total = 0`` then ``total := total + close`` inside the body
        reads as int on the first pass and only becomes float on the second.
        The lattice is two high, so three passes are provably enough -- the
        third exists to confirm the second changed nothing.
        """
        for _ in range(_MAX_LOOP_PASSES):
            before = self._snapshot()
            walk()
            if self._snapshot() == before:
                return

    def _snapshot(self) -> dict[str, dict[str, str]]:
        return {scope: {name: b.ty for name, b in names.items()}
                for scope, names in self.table.bindings.items()}

    def _function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Walk a function body in its own scope."""
        for default in node.args.defaults:
            self._expr(default)
        for kw_default in node.args.kw_defaults:
            if kw_default is not None:
                self._expr(kw_default)

        self._scopes.append(self._qualify(self._scope, node.name))
        self.table.bindings.setdefault(self._scope, {})

        every = list(node.args.posonlyargs) + list(node.args.args) + list(node.args.kwonlyargs)
        for arg in every:
            ty = annotation_type(arg.annotation)
            unknown = None
            if ty == UNKNOWN:
                # The first of the two open leaks: closed by monomorphization,
                # which instantiates the function per call-site signature
                unknown = self._unknown('unannotated-param', arg, arg.arg)
            self._bindings()[arg.arg] = Binding(name=arg.arg, ty=ty,
                                                line=_line(arg), unknown=unknown)

        self._body(node.body)

        # The scope pushed above IS this function's signature key
        signature = self.table.funcs.get(self._scope)
        if signature is not None:
            signature.params = [self._bindings().get(a.arg, Binding(a.arg)).ty
                                for a in list(node.args.posonlyargs) + list(node.args.args)]
            if signature.ret == UNKNOWN:
                signature.ret = self._return_type(node)

        self._scopes.pop()

    def _return_type(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        """Join every ``return`` in a function body, ignoring nested functions."""
        result: str | None = None
        for stmt in _walk_own_scope(node):
            if isinstance(stmt, ast.Return):
                ty = VOID if stmt.value is None else get_ty(stmt.value)
                result = ty if result is None else join(result, ty)
        return VOID if result is None else result

    def _store(self, target: ast.expr, ty: str, source: ast.AST) -> None:
        """Bind an assignment target, recursing into tuple/list targets."""
        if isinstance(target, ast.Name):
            set_ty(target, ty)
            unknown = self._unknown('unknown-value', source) if ty == UNKNOWN else None
            self._bind(target.id, ty, source, unknown)
        elif isinstance(target, (ast.Tuple, ast.List)):
            set_ty(target, OBJECT)
            for element in target.elts:
                # A destructured element's own type is not modeled: the tuple
                # shapes Pine has (``request.security`` tuples) are opaque here
                self._store(element, UNKNOWN, source)
        elif isinstance(target, (ast.Attribute, ast.Subscript)):
            self._expr(target.value)
            set_ty(target, ty)

    def _target_type(self, target: ast.expr) -> str:
        """Current type of an augmented-assignment target."""
        if isinstance(target, ast.Name):
            found = self._lookup(target.id)
            return found.ty if found is not None else UNKNOWN
        return UNKNOWN

    def _element_type(self, iter_node: ast.expr, iter_ty: str) -> str:
        """
        Type of a ``for`` loop variable.

        MEASURED: TradingView does NOT truncate a Pine ``for``. With
        ``R = input.int(14)``, ``for i = R / 8 to R / 4`` iterates i = 1.75 and
        2.75, so the counter is an int-TYPED variable carrying a fractional
        value -- exactly the law this whole pass exists for. The type is
        therefore the join of the bounds, and a native ``range`` over int
        arguments yields an int.
        """
        if isinstance(iter_node, ast.Call):
            callee = _dotted(iter_node.func)
            if callee in ('range', 'pine_range', 'lib.pine_range'):
                bounds = [get_ty(a) for a in iter_node.args]
                if bounds and all(b in NUMERIC for b in bounds):
                    return INT if all(b == INT for b in bounds) else FLOAT
                return UNKNOWN
        return UNKNOWN if iter_ty != OBJECT else UNKNOWN

    # --- expressions -----------------------------------------------------

    def _expr(self, node: ast.expr) -> str:
        """Type one expression, stamping it and everything under it."""
        method = getattr(self, f'_e_{type(node).__name__}', None)
        if method is None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.expr):
                    self._expr(child)
            ty = UNKNOWN
        else:
            ty = method(node)
        set_ty(node, ty)
        return ty

    # Each ``_e_*`` returns the type; the caller stamps it.

    def _e_Constant(self, node: ast.Constant) -> str:
        # A stamp already on the node wins: the constant folder replaces a
        # Pine-typed subtree with its literal and records what the TYPE was,
        # which the Python literal alone can no longer tell (``14 / 8`` folds
        # to 1.75, indistinguishable from a float literal)
        existing = getattr(node, TY_ATTR, None)
        if existing is not None:
            return existing
        return constant_type(node.value)

    def _e_Name(self, node: ast.Name) -> str:
        found = self._lookup(node.id)
        if found is not None:
            return found.ty
        entry = lib_types().get(node.id)
        if entry is not None and entry.get('kind') == 'value':
            return entry['ty']
        return UNKNOWN

    def _e_Attribute(self, node: ast.Attribute) -> str:
        self._expr(node.value)
        name = self._lib_name(node)
        if name is None:
            return UNKNOWN
        override = LIB_TYPE_OVERRIDES.get(name)
        if isinstance(override, str) and len(override) == 1:
            return override
        entry = lib_types().get(name)
        if entry is None:
            return UNKNOWN
        if entry['kind'] == 'value':
            return entry['ty']
        # A bare reference to a lib function is the function itself
        return OBJECT

    def _e_BinOp(self, node: ast.BinOp) -> str:
        return binop_type(node.op, self._expr(node.left), self._expr(node.right))

    def _e_UnaryOp(self, node: ast.UnaryOp) -> str:
        return unaryop_type(node.op, self._expr(node.operand))

    def _e_BoolOp(self, node: ast.BoolOp) -> str:
        for value in node.values:
            self._expr(value)
        return BOOL

    def _e_Compare(self, node: ast.Compare) -> str:
        left = self._expr(node.left)
        right = left
        for comparator in node.comparators:
            right = self._expr(comparator)
        return compare_type(left, right)

    def _e_IfExp(self, node: ast.IfExp) -> str:
        self._expr(node.test)
        # MEASURED: ``d > 1 ? d : R`` is int, ``d > 1 ? d : 1.0`` is float --
        # the arms join, they do not widen unconditionally
        return join(self._expr(node.body), self._expr(node.orelse))

    def _e_NamedExpr(self, node: ast.NamedExpr) -> str:
        ty = self._expr(node.value)
        self._store(node.target, ty, node)
        return ty

    def _e_Subscript(self, node: ast.Subscript) -> str:
        base = self._expr(node.value)
        self._expr(node.slice)
        # MEASURED: ``d[1]`` on an int-typed ``d`` is int -- the history index
        # is type-preserving, it reads the same series one bar back
        return base

    def _e_Tuple(self, node: ast.Tuple) -> str:
        for element in node.elts:
            self._expr(element)
        return OBJECT

    _e_List = _e_Tuple
    _e_Set = _e_Tuple

    def _e_Dict(self, node: ast.Dict) -> str:
        for key in node.keys:
            if key is not None:
                self._expr(key)
        for value in node.values:
            self._expr(value)
        return OBJECT

    def _e_ListComp(self, node) -> str:
        """
        Walk a comprehension so nothing under it is left untyped.

        Comprehensions are outside Pine and the ``edge`` gate rejects them, but
        the lib's own code uses them, and an unvisited subtree would leave
        typed literals sitting under untyped nodes. The loop variables are
        bound in the ENCLOSING scope here rather than a scope of their own --
        an approximation Python does not make, and one that only matters for a
        name the comprehension shadows.
        """
        for generator in node.generators:
            iter_ty = self._expr(generator.iter)
            self._store(generator.target, self._element_type(generator.iter, iter_ty), node)
            for condition in generator.ifs:
                self._expr(condition)
        if isinstance(node, ast.DictComp):
            self._expr(node.key)
            self._expr(node.value)
        else:
            self._expr(node.elt)
        return OBJECT

    _e_SetComp = _e_ListComp
    _e_GeneratorExp = _e_ListComp
    _e_DictComp = _e_ListComp

    def _e_Call(self, node: ast.Call) -> str:
        for arg in node.args:
            self._expr(arg)
        for keyword in node.keywords:
            self._expr(keyword.value)

        callee = self._lib_name(node.func)
        if callee is None:
            self._expr(node.func)
            return self._user_call(node)

        argc = _call_arity(node)
        ty = self._lib_call_type(callee, node, argc)
        self.table.calls.append(CallSite(
            callee=callee, line=_line(node), col=_col(node), argc=argc, ty=ty))
        return ty

    def _lib_call_type(self, callee: str, node: ast.Call, argc: int | None) -> str:
        """
        Result type of a call to a lib name.

        The measured override wins over the annotation: the lib annotates
        ``math.round`` as a float because that is what Python returns, while
        TradingView types the one-argument form as an int.
        """
        entry = lib_types().get(callee)
        override = LIB_TYPE_OVERRIDES.get(callee)
        if override is not None:
            names = entry.get('names') if isinstance(entry, dict) else None
            resolved = self._apply_override(override, node, argc, names)
            if resolved is not None:
                return resolved

        if entry is None:
            return UNKNOWN
        if entry['kind'] == 'value':
            # A module property read that the property pass turned into a call
            return entry['ty']
        if entry['kind'] == 'function':
            return entry['ret']
        # An overload group: join what the arity-compatible implementations
        # return, so an unpinnable call still gets the tightest common type
        if argc is None:
            return UNKNOWN
        candidates = [impl['ret'] for impl in entry['impls']
                      if _arity_fits(impl, argc)]
        if not candidates:
            return UNKNOWN
        result = candidates[0]
        for candidate in candidates[1:]:
            result = join(result, candidate)
        return result

    def _apply_override(self, override: Any, node: ast.Call, argc: int | None,
                        param_names: list[str] | None = None) -> str | None:
        """
        Resolve one entry of the measured override table.

        ``param_names`` is the callee's declared parameter order, which is what
        turns a keyword spelling back into a position; without it only the
        positional arguments can be addressed.
        """
        if isinstance(override, dict):
            if argc is None:
                return UNKNOWN
            picked = override.get(argc)
            return None if picked is None else self._apply_override(
                picked, node, argc, param_names)
        if not isinstance(override, str):
            return None
        if override == 'all_int':
            # Every argument counts, however it was spelled: ``math.max`` is
            # int-typed exactly when all of them are -- and an unpacking hides
            # some of them, so there is nothing to decide on
            if argc is None:
                return UNKNOWN
            passed = [get_ty(a) for a in node.args]
            passed += [get_ty(k.value) for k in node.keywords if k.arg is not None]
            if not passed or any(t not in NUMERIC for t in passed):
                return UNKNOWN if passed else None
            return INT if all(t == INT for t in passed) else FLOAT
        if override.startswith('arg') and override[3:].isdigit():
            argument = _bound_arg(node, int(override[3:]), param_names)
            return UNKNOWN if argument is None else get_ty(argument)
        return override

    def _user_call(self, node: ast.Call) -> str:
        """Result type of a call to a function defined in this module."""
        name = _dotted(node.func) or ''
        key = self._resolve_func(name)
        signature = self.table.funcs.get(key) if key is not None else None
        if signature is None:
            # A module function shadows the builtin of the same name, so the
            # builtins are only consulted once the module has no such name
            builtin = BUILTIN_CALL_TYPES.get(name)
            if builtin is not None and isinstance(node.func, ast.Name):
                resolved = self._apply_override(builtin, node, _call_arity(node))
                if resolved is not None:
                    return resolved
            return UNKNOWN
        if signature.ret == UNKNOWN and not self._annotated_returns.get(key or ''):
            self.saw_forward_call = True
        self.table.calls.append(CallSite(
            callee=name, line=_line(node), col=_col(node),
            argc=_call_arity(node), ty=signature.ret))
        return signature.ret

    def _lib_name(self, node: ast.expr) -> str | None:
        """
        The registry key a lib reference resolves to.

        After import normalization every lib reference is spelled
        ``lib.<dotted path>``, so the key is that path with the alias stripped.

        :param node: The referenced expression
        :return: The dotted key, or None when this is not a lib reference
        """
        dotted = _dotted(node)
        if dotted is None:
            return None
        head, _, rest = dotted.partition('.')
        if head in self._lib_aliases and rest:
            return rest
        return None


def _line(node: ast.AST) -> int:
    """
    A node's line, tolerating the synthetic ones.

    Earlier passes emit nodes without positions; the pipeline only fills them
    in at the very end (``transformers/locations.py``), so anything read here
    has to survive their absence.
    """
    return getattr(node, 'lineno', 0)


def _col(node: ast.AST) -> int:
    """A node's column, tolerating the synthetic ones."""
    return getattr(node, 'col_offset', 0)


def _call_arity(node: ast.Call) -> int | None:
    """
    How many arguments a call actually passes.

    A keyword argument IS an argument: ``math.round(x, precision=2)`` is the
    two-argument -- float-typed -- form, and counting only ``node.args`` would
    resolve it to the one-argument int overload. An unpacking (``*seq``,
    ``**kw``) makes the count unknowable, which is what None says; the
    overrides and the overload groups then decline to pick rather than pick
    wrong.

    :param node: The call node
    :return: The argument count, or None when an unpacking hides it
    """
    if any(isinstance(a, ast.Starred) for a in node.args):
        return None
    if any(k.arg is None for k in node.keywords):
        return None
    return len(node.args) + len(node.keywords)


def _bound_arg(node: ast.Call, index: int, param_names: list[str] | None) -> ast.expr | None:
    """
    The expression bound to one declared parameter position.

    A type-preserving override names the parameter it copies from, and Python
    lets the caller spell that parameter either way, so the keywords have to be
    bound back to their declared position before the position can be read. An
    unpacking hides which position an argument landed on, and is unresolvable.

    :param node: The call node
    :param index: Declared parameter position, 0-based
    :param param_names: The callee's declared parameter order, when it is known
    :return: The bound expression, or None when it cannot be determined
    """
    if any(isinstance(a, ast.Starred) for a in node.args):
        return None
    if index < len(node.args):
        return node.args[index]
    if param_names is None or index >= len(param_names):
        return None
    wanted = param_names[index]
    for keyword in node.keywords:
        if keyword.arg == wanted:
            return keyword.value
    return None


def _arity_fits(impl: dict[str, Any], argc: int) -> bool:
    """Whether an overload implementation can take this many positional arguments."""
    if impl.get('vararg') is not None:
        return True
    params = impl['params']
    return len(params) - impl['defaults'] <= argc <= len(params)


def _dotted(node: ast.expr) -> str | None:
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


def _walk_own_scope(node: ast.AST):
    """Walk a function body without descending into nested function scopes."""
    stack = list(ast.iter_child_nodes(node))
    while stack:
        current = stack.pop()
        yield current
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        stack.extend(ast.iter_child_nodes(current))
