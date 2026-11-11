"""Per-call-site instantiation of security-bearing functions (Pine semantics).

In Pine Script every call of a user function creates a separate INSTANCE: a
``request.security()`` inside a function called from N sites is N distinct
data requests, each bound to its own symbol/timeframe arguments. PyneCore's
SecurityTransformer allocates one sec_id per *syntactic*
``request.security()`` call, and the runtime binds a sec_id to ONE resolved
(symbol, timeframe) on its first ``__sec_signal__`` — so multiple call sites
of the same function would silently share the FIRST call's binding (a
6-timeframe ``f_htf_trend(tf)`` helper would read the first timeframe's
series six times).

This pass restores Pine's instantiation semantics statically, BEFORE
SecurityTransformer runs: any function whose subtree contains a
``request.security[_lower_tf]`` call — or a direct call to another
security-bearing function — is cloned per direct-Name call site. Each clone
is a full deep copy inserted right after the original def, and exactly one
call site is rewritten to each clone, so SecurityTransformer then allocates
fresh sec_ids per call site and the whole downstream machinery (context
registry, ``--security`` discovery, subprocesses) works unchanged.

Bail-outs — the affected function keeps the legacy shared-context behavior:

- recursive functions (any name-level call-graph cycle),
- decorated functions (the runtime value is the decorator's return value),
- functions whose name is referenced outside a direct-call position
  (aliases, callbacks, stores) or rebound in their scope region (parameters,
  imports, class, ``except`` and ``match`` names),
- duplicate top-of-scope definitions of the same name (shadowing),
- attribute-style call sites (methods, cross-module library calls) are not
  rewritten — a library function with security calls instantiated from
  several script call sites remains a single shared context (documented
  limitation).

Must run after ImportNormalizerTransformer (security calls are in their
``lib.request.security`` form) and before SecurityTransformer.
"""
import ast
from collections.abc import Callable

from . import ast_walk
from .locations import anchor_all_at
from .security_calls import has_security_call, is_security_call, is_security_lower_tf_call

__all__ = ['SecurityInstantiationTransformer']

# The call graph is acyclic once recursive functions are excluded, so cloning
# always terminates, but a call chain that calls its callee more than once at
# every level grows exponentially. TradingView instantiates every call path as
# well and bounds it per function: the request values one function uses,
# transitively through the functions it calls, count as its "external
# elements", at most 254 of them, an int/float/bool value counting 2 (measured:
# ``f7() => f6() + f6()`` over a ``request.security(..., close)`` leaf, 128 calls,
# is rejected with 256 elements; ``f6``, 64 calls and 120 clones here, runs).
# The script's global scope (``main``) has no such bound (measured: 130
# identical global ``request.security`` float calls run). So a called function
# reaching more than 254 security calls cannot be a valid script, while every
# valid one keeps each function's instance count within 254 per call from the
# global scope -- a polynomial, not exponential, amount of cloning. Counting
# only ``request.security[_lower_tf]`` calls, at one element each, never
# rejects a script TradingView accepts.
_MAX_FUNCTION_REQUESTS = 254


class _FuncInfo:
    """One function definition eligible for instantiation analysis."""

    __slots__ = ('node', 'owner_body', 'index', 'region')

    def __init__(self, node: ast.FunctionDef | ast.AsyncFunctionDef,
                 owner_body: list[ast.stmt], index: int, region: ast.AST):
        self.node = node
        self.owner_body = owner_body
        self.index = index
        # The subtree in which this def's name is in scope (the whole module
        # for module-level defs, the enclosing function for nested defs).
        self.region = region


def _ordered_walk(node: ast.AST):
    """DFS in source order (``ast.walk`` is BFS; clone/call-site numbering
    must be stable and follow the source)."""
    yield node
    yield from ast_walk.iter_descendants(node)


class _RegionIndex:
    """Name references of one scope region, gathered in a single walk.

    ``_analyze`` asks the same two questions — "where is ``name`` called?" and
    "is ``name`` referenced any other way?" — for every candidate function of
    the region; answering them from one walk keeps an analysis round linear in
    the tree size instead of one full-region walk per candidate.
    """

    __slots__ = ('call_sites', 'non_call_refs')

    def __init__(self, region: ast.AST):
        # Direct-Name call sites per callee name, in source order. Sites
        # inside a nested def that redefines the name are not excluded:
        # the duplicate-name bail-out in ``_analyze`` disqualifies the
        # whole function instead
        self.call_sites: dict[str, list[ast.Call]] = {}
        # Names referenced outside a direct-call func position (alias,
        # callback argument, store, ...), bound as a parameter, import, class,
        # ``except`` or ``match`` name, or declared global / nonlocal
        self.non_call_refs: set[str] = set()
        call_funcs: set[int] = set()
        for n in _ordered_walk(region):
            if isinstance(n, ast.Call):
                func = n.func
                if isinstance(func, ast.Name):
                    # Pre-order: the call is seen before its func child
                    call_funcs.add(id(func))
                    self.call_sites.setdefault(func.id, []).append(n)
            elif isinstance(n, ast.Name):
                if id(n) not in call_funcs:
                    self.non_call_refs.add(n.id)
            elif isinstance(n, (ast.Global, ast.Nonlocal)):
                self.non_call_refs.update(n.names)
            elif isinstance(n, ast.arg):
                self.non_call_refs.add(n.arg)
            elif isinstance(n, ast.alias):
                self.non_call_refs.add((n.asname or n.name).partition('.')[0])
            elif isinstance(n, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar, ast.ClassDef)):
                if n.name is not None:
                    self.non_call_refs.add(n.name)
            elif isinstance(n, ast.MatchMapping):
                if n.rest is not None:
                    self.non_call_refs.add(n.rest)


class SecurityInstantiationTransformer:
    """Not an ``ast.NodeTransformer`` — a whole-module fixpoint pass with the
    same ``visit(tree) -> tree`` pipeline interface."""

    def __init__(self):
        self._clones_made = 0
        # Every candidate of the module as it is now (see :meth:`_candidates`);
        # dropped whenever a clone changes the module
        self._candidates_now: list[tuple[_FuncInfo, list[ast.Call]]] | None = None

    # --- collection ---

    @staticmethod
    def _is_security_call(node: ast.AST) -> bool:
        return isinstance(node, ast.Call) and (is_security_call(node)
                                               or is_security_lower_tf_call(node))

    def _collect_functions(self, module: ast.Module) -> list[_FuncInfo]:
        """Every FunctionDef with its owner body and scope region. Class
        bodies are skipped entirely (methods are called by attribute, which
        this pass never rewrites)."""
        result: list[_FuncInfo] = []

        def scan_body(body: list[ast.stmt], region: ast.AST) -> None:
            for idx, stmt in enumerate(body):
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    result.append(_FuncInfo(stmt, body, idx, region))
                    scan_body(stmt.body, stmt)
                elif isinstance(stmt, ast.ClassDef):
                    continue
                else:
                    # Compound statements can nest defs (if/try/with/for).
                    scan_sub(stmt, region)

        def scan_sub(stmt: ast.stmt, region: ast.AST) -> None:
            for field_body in ('body', 'orelse', 'finalbody'):
                sub = getattr(stmt, field_body, None)
                if isinstance(sub, list):
                    scan_body(sub, region)
            for handler in getattr(stmt, 'handlers', []) or []:
                scan_body(handler.body, region)
            for case in getattr(stmt, 'cases', []) or []:
                scan_body(case.body, region)

        scan_body(module.body, module)
        return result

    # --- analysis ---

    def _analyze(self, module: ast.Module, min_sites: int = 2
                 ) -> list[tuple[_FuncInfo, list[ast.Call]]]:
        """Return the eligible security-bearing functions with at least
        ``min_sites`` direct-Name call sites, each with its call sites in
        source order. Eligibility applies every bail-out.

        The candidates are only re-derived after the module has changed: the
        fixpoint's last round and the specialization ask about the very same
        module.
        """
        if self._candidates_now is None:
            self._candidates_now = self._candidates(module)
        return [(info, sites) for info, sites in self._candidates_now
                if len(sites) >= min_sites]

    def _candidates(self, module: ast.Module) -> list[tuple[_FuncInfo, list[ast.Call]]]:
        """Every security-bearing function that passes the bail-outs, with its
        direct-Name call sites in source order (possibly none)."""
        infos = self._collect_functions(module)

        by_name: dict[str, list[_FuncInfo]] = {}
        for info in infos:
            by_name.setdefault(info.node.name, []).append(info)

        # A function bears security when one instance of it holds a security
        # call, by the same lexical resolution the fan-out bound counts with:
        # a caller of a name that resolves to a stateless local definition is
        # not a bearer, so cloning never expands what the bound did not count.
        # The name-level call edges among module functions only serve the cycle
        # detection; a set, so the walk need not follow the source order.
        total = self._request_totals(module, infos)
        bearing: set[str] = {info.node.name for info in infos if total(info.node) > 0}
        edges: dict[str, set[str]] = {}
        for info in infos:
            callees = edges.setdefault(info.node.name, set())
            for n in ast_walk.walk(info.node):
                if isinstance(n, ast.Call):
                    func = n.func
                    if isinstance(func, ast.Name) and func.id in by_name:
                        callees.add(func.id)

        # Cycle detection over the bearing subgraph — any bearer on a cycle
        # (recursion, mutual recursion) is excluded, or cloning would never
        # converge.
        on_cycle: set[str] = set()

        def reaches(start: str, target: str, seen: set[str]) -> bool:
            for callee in edges.get(start, ()):
                if callee == target:
                    return True
                if callee not in seen:
                    seen.add(callee)
                    if reaches(callee, target, seen):
                        return True
            return False

        for name in bearing:
            if reaches(name, name, set()):
                on_cycle.add(name)

        # One index per scope region, built only for regions that hold a
        # candidate, and shared by every candidate of that region.
        regions: dict[int, _RegionIndex] = {}
        eligible: list[tuple[_FuncInfo, list[ast.Call]]] = []
        for info in infos:
            name = info.node.name
            if name not in bearing or name in on_cycle:
                continue
            if len(by_name[name]) > 1:  # shadowing / duplicate defs
                continue
            if info.node.decorator_list:
                continue
            index = regions.get(id(info.region))
            if index is None:
                index = regions[id(info.region)] = _RegionIndex(info.region)
            if name in index.non_call_refs:
                continue
            eligible.append((info, index.call_sites.get(name, [])))
        return eligible

    # --- request fan-out bound ---

    @staticmethod
    def _local_bindings(node: ast.FunctionDef | ast.AsyncFunctionDef
                        ) -> tuple[set[str], set[str]]:
        """The names a function binds in its own scope other than by a nested
        ``def``, and the names it declares ``global``.

        A binding is a parameter, an assignment, ``for`` / ``with`` / ``except``
        / ``match`` / walrus target, ``del``, an import or a class definition.
        Nested functions, lambdas and class bodies have scopes of their own, and
        a comprehension target binds only inside the comprehension.

        :param node: The function definition
        :return: The locally bound names and the ``global`` names
        """
        bound: set[str] = set()
        global_names: set[str] = set()
        arguments = node.args
        for arg in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs,
                    arguments.vararg, arguments.kwarg):
            if arg is not None:
                bound.add(arg.arg)
        todo: list[ast.AST] = list(node.body)
        while todo:
            n = todo.pop()
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            if isinstance(n, ast.ClassDef):
                bound.add(n.name)
                continue
            if isinstance(n, ast.comprehension):
                todo.append(n.iter)
                todo.extend(n.ifs)
                continue
            if isinstance(n, ast.Name):
                if not isinstance(n.ctx, ast.Load):
                    bound.add(n.id)
            elif isinstance(n, ast.alias):
                bound.add((n.asname or n.name).partition('.')[0])
            elif isinstance(n, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)):
                if n.name is not None:
                    bound.add(n.name)
            elif isinstance(n, ast.MatchMapping):
                if n.rest is not None:
                    bound.add(n.rest)
            elif isinstance(n, ast.Global):
                global_names.update(n.names)
            todo.extend(ast_walk.iter_child_nodes(n))
        return bound - global_names, global_names

    @classmethod
    def _request_totals(cls, module: ast.Module, infos: list[_FuncInfo]
                        ) -> Callable[[ast.FunctionDef | ast.AsyncFunctionDef], int]:
        """The number of security calls one instance of a function holds after
        instantiation, as a memoized function of its definition node.

        It counts the security calls of the body (lambdas and class bodies
        included, since cloning copies them too), plus the count of every
        function it calls, once per call site, plus the count of each function
        defined inside it that it does not call by name (a clone copies that
        definition as well). Calls on a recursive cycle count nothing: such
        functions are never cloned. A call resolves lexically, to the
        definitions of the nearest enclosing scope that defines the name; when
        that scope defines it more than once, the call counts as the largest of
        them: such a function is never cloned, but its callers are. A scope that
        binds the name in any other way (:meth:`_local_bindings`) and does not
        define it binds the call to that value, which counts nothing; a
        ``global`` declaration resolves the name in the module.

        :param module: The module the functions belong to
        :param infos: Every function of the module (:meth:`_collect_functions`)
        :return: The count of a function definition of ``infos``
        """
        nested: dict[int, list[ast.FunctionDef | ast.AsyncFunctionDef]] = {}
        scoped: dict[tuple[int, str], list[ast.FunctionDef | ast.AsyncFunctionDef]] = {}
        parent: dict[int, ast.AST] = {}
        for info in infos:
            nested.setdefault(id(info.region), []).append(info.node)
            scoped.setdefault((id(info.region), info.node.name), []).append(info.node)
            parent[id(info.node)] = info.region
        defined = {id(info.node) for info in infos}

        own: dict[int, int] = {}
        callees: dict[int, list[str]] = {}
        bound: dict[int, set[str]] = {}
        global_names: dict[int, set[str]] = {}
        for info in infos:
            bound[id(info.node)], global_names[id(info.node)] = cls._local_bindings(info.node)
            count = 0
            names: list[str] = []
            todo = list(ast_walk.iter_child_nodes(info.node))
            while todo:
                n = todo.pop()
                if id(n) in defined:
                    continue
                todo.extend(ast_walk.iter_child_nodes(n))
                if not isinstance(n, ast.Call):
                    continue
                if cls._is_security_call(n):
                    count += 1
                func = n.func
                if isinstance(func, ast.Name):
                    names.append(func.id)
            own[id(info.node)] = count
            callees[id(info.node)] = names

        totals: dict[int, int] = {}
        # Stack depth of every function under count; ``low`` is the smallest
        # depth a recursive call was cut at while counting the current function
        visiting: dict[int, int] = {}
        low = len(infos)

        def resolve(func_name: str, scope: ast.AST) -> int:
            # The nearest enclosing scope that defines the name binds the call;
            # a name it defines more than once is never cloned itself, but its
            # callers are, so it counts as its largest definition there. A
            # scope binding the name to another value calls that value
            while True:
                defs = scoped.get((id(scope), func_name))
                if defs:
                    return max(total(node) for node in defs)
                if scope is module or func_name in bound[id(scope)]:
                    return 0
                if func_name in global_names[id(scope)]:
                    scope = module
                else:
                    scope = parent[id(scope)]

        def total(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
            # A count cut at this function or one above it belongs to a
            # recursive cycle and depends on where the cycle was entered: it is
            # not memoized, so every count is the same whatever the order
            nonlocal low
            key = id(node)
            done = totals.get(key)
            if done is not None:
                return done
            cut_at = visiting.get(key)
            if cut_at is not None:
                low = min(low, cut_at)
                return 0
            depth = len(visiting)
            visiting[key] = depth
            outer_low, low = low, len(infos)
            called_names = callees[key]
            result = own[key] + sum(resolve(callee, node) for callee in called_names)
            for inner in nested.get(key, ()):
                if inner.name not in called_names:
                    result += total(inner)
            del visiting[key]
            if low > depth:
                totals[key] = result
            low = min(outer_low, low)
            return result

        return total

    def _check_request_fanout(self, module: ast.Module) -> None:
        """Reject a script in which a called function reaches more security
        calls than TradingView lets one function use.

        The count of a function is the number of security calls one instance
        of it holds after instantiation (:meth:`_request_totals`), computed on
        the call graph without cloning anything. A function called from
        nowhere is the global scope and is not bounded.

        :param module: The module before instantiation
        :raises SyntaxError: If a called function exceeds the bound
        """
        infos = self._collect_functions(module)
        total = self._request_totals(module, infos)
        by_name: dict[str, list[_FuncInfo]] = {}
        called: set[str] = set()
        for info in infos:
            by_name.setdefault(info.node.name, []).append(info)
            for n in ast_walk.walk(info.node):
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                    called.add(n.func.id)

        for name, defs in by_name.items():
            if len(defs) != 1 or name not in called:
                continue
            reached = total(defs[0].node)
            if reached > _MAX_FUNCTION_REQUESTS:
                raise SyntaxError(
                    f"function '{name}' reaches {reached} request.security calls through "
                    f"the functions it calls; a function can use at most "
                    f"{_MAX_FUNCTION_REQUESTS} request values"
                )

    # --- cloning ---

    def _unique_name(self, base: str, existing: set[str]) -> str:
        """Collision-free clone name against ``existing`` (every function name
        of the module); the global clone counter keeps names unique even
        across nested re-instantiation. The chosen name is added to
        ``existing``."""
        candidate = f"{base}__pyne_inst{self._clones_made}"
        while candidate in existing:
            candidate += "_"
        existing.add(candidate)
        return candidate

    def _clone_one(self, module: ast.Module) -> bool:
        """Clone the first eligible multi-site bearer; True if work was done."""
        eligible = self._analyze(module)
        if not eligible:
            return False
        info, sites = eligible[0]
        self._candidates_now = None
        name = info.node.name
        existing = {
            n.name for n in ast_walk.walk_statements(module)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        # Re-locate the def index (earlier clones may have shifted the body).
        index = info.owner_body.index(info.node)
        # Call site 1 keeps the original function; each further site gets a
        # fresh clone inserted after the original (source order preserved).
        for k, site in enumerate(sites[1:], start=2):
            self._clones_made += 1
            clone = ast_walk.clone(info.node)
            clone.name = self._unique_name(name, existing)
            info.owner_body.insert(index + k - 1, clone)
            site_func = site.func
            assert isinstance(site_func, ast.Name)
            site_func.id = clone.name
        return True

    # --- call-site specialization ---

    @classmethod
    def _is_static_arg(cls, node: ast.expr) -> bool:
        """Whether ``node`` can be duplicated into the callee without changing
        behaviour: a literal or a bare ``lib.*`` attribute chain.

        Calls are excluded on purpose — copying ``input.int(...)`` into the
        function body would register the input twice.
        """
        if isinstance(node, ast.Constant):
            return True
        if isinstance(node, ast.Attribute):
            return cls._is_static_arg(node.value)
        if isinstance(node, ast.Name):
            return node.id == 'lib'
        return False

    def _specialize(self, info: _FuncInfo, site: ast.Call) -> None:
        """Pin ``site``'s static arguments as bindings at the top of ``info``.

        After instantiation each security-bearing function has exactly one call
        site, so the timeframe/symbol a parameter carries is statically known.
        Writing it back as ``tf = "60"`` at the top of the body turns the
        parameter into a compile-time constant chain, which lets
        SecurityTransformer hoist the ``__sec_signal__`` into the top block
        instead of emitting it inline — a prerequisite for any other security
        expression to depend on this one.
        """
        fn = info.node
        fn_args = fn.args
        if fn_args.vararg is not None or fn_args.kwarg is not None:
            return
        if fn_args.posonlyargs:
            return
        if any(isinstance(a, ast.Starred) for a in site.args):
            return
        if any(kw.arg is None for kw in site.keywords):
            return

        positional = [a.arg for a in fn_args.args]
        by_name = positional + [a.arg for a in fn_args.kwonlyargs]
        bound: dict[str, ast.expr] = {}
        if len(site.args) > len(positional):
            return
        for idx, value in enumerate(site.args):
            bound[positional[idx]] = value
        for kw in site.keywords:
            if kw.arg not in by_name:
                return
            bound[kw.arg] = kw.value

        pinned: list[ast.stmt] = []
        for name in by_name:
            value = bound.get(name)
            if value is None or not self._is_static_arg(value):
                continue
            # The value is cloned from the call site, but it is bound at the
            # function's entry, as a parameter is
            pinned.append(anchor_all_at(ast.Assign(
                targets=[ast.Name(id=name, ctx=ast.Store())],
                value=ast_walk.clone(value),
            ), fn))
        if pinned:
            at = ast_walk.docstring_end(fn.body)
            fn.body[at:at] = pinned

    def _specialize_all(self, module: ast.Module) -> None:
        """Pin call-site constants into every single-site security bearer."""
        for info, sites in self._analyze(module, min_sites=1):
            if len(sites) != 1:
                continue
            self._specialize(info, sites[0])

    # --- pipeline API ---

    def visit(self, module: ast.Module) -> ast.Module:
        if not has_security_call(module):
            return module
        self._candidates_now = None
        self._check_request_fanout(module)
        # Fixpoint: cloning a caller duplicates its callees' call sites, so
        # re-analyze until no eligible multi-site bearer remains. It ends
        # because recursive functions are never cloned, and the fan-out bound
        # keeps the number of clones polynomial in the script's size.
        while self._clone_one(module):
            pass
        self._specialize_all(module)
        return module
