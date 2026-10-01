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
  (aliases, callbacks, stores),
- duplicate top-of-scope definitions of the same name (shadowing),
- attribute-style call sites (methods, cross-module library calls) are not
  rewritten — a library function with security calls instantiated from
  several script call sites remains a single shared context (documented
  limitation).

Must run after ImportNormalizerTransformer (security calls are in their
``lib.request.security`` form) and before SecurityTransformer.
"""
import ast

from . import ast_walk
from .security import SecurityTransformer

__all__ = ['SecurityInstantiationTransformer']

# Hard ceiling on clones per module — far above any real script (TradingView
# itself caps unique request.* calls at 40) but low enough to stop a
# pathological call-graph blow-up with a clear error instead of a hang.
_MAX_CLONES = 64


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
    must be stable and follow the source).

    Iterative: a recursive generator pays one ``yield from`` hop per tree level
    for every node it hands out.
    """
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(reversed(list(ast_walk.iter_child_nodes(current))))


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
        # callback argument, store, ...) or declared global / nonlocal
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


class SecurityInstantiationTransformer:
    """Not an ``ast.NodeTransformer`` — a whole-module fixpoint pass with the
    same ``visit(tree) -> tree`` pipeline interface."""

    def __init__(self):
        self._clones_made = 0

    # --- collection ---

    @staticmethod
    def _is_security_call(node: ast.AST) -> bool:
        return isinstance(node, ast.Call) and (
            SecurityTransformer._is_security_call(node)  # noqa
            or SecurityTransformer._is_security_lower_tf_call(node)  # noqa
        )

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
        source order. Eligibility applies every bail-out."""
        infos = self._collect_functions(module)

        by_name: dict[str, list[_FuncInfo]] = {}
        for info in infos:
            by_name.setdefault(info.node.name, []).append(info)

        # Direct bearers (subtree contains a security call) and name-level
        # call edges among module functions (for transitivity and cycle
        # detection), from one walk per function.
        bearing: set[str] = set()
        edges: dict[str, set[str]] = {}
        for info in infos:
            name = info.node.name
            callees = edges.setdefault(name, set())
            for n in _ordered_walk(info.node):
                if not isinstance(n, ast.Call):
                    continue
                if self._is_security_call(n):
                    bearing.add(name)
                func = n.func
                if isinstance(func, ast.Name) and func.id in by_name:
                    callees.add(func.id)

        # Transitive closure: calling a bearer makes the caller a bearer.
        changed = True
        while changed:
            changed = False
            for name, callees in edges.items():
                if name not in bearing and callees & bearing:
                    bearing.add(name)
                    changed = True

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
            sites = index.call_sites.get(name, [])
            if len(sites) >= min_sites:
                eligible.append((info, sites))
        return eligible

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
            if self._clones_made > _MAX_CLONES:
                raise SyntaxError(
                    f"security instantiation exceeded {_MAX_CLONES} function "
                    f"clones — the script's request.security call graph is "
                    f"too large to instantiate per call site"
                )
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
            assign = ast.Assign(
                targets=[ast.Name(id=name, ctx=ast.Store())],
                value=ast_walk.clone(value),
            )
            ast.copy_location(assign, fn)
            ast_walk.fix_missing_locations(assign)
            pinned.append(assign)
        if pinned:
            fn.body[0:0] = pinned

    def _specialize_all(self, module: ast.Module) -> None:
        """Pin call-site constants into every single-site security bearer."""
        for info, sites in self._analyze(module, min_sites=1):
            if len(sites) != 1:
                continue
            self._specialize(info, sites[0])

    # --- pipeline API ---

    def visit(self, module: ast.Module) -> ast.Module:
        if not any(self._is_security_call(n) for n in ast_walk.walk(module)):
            return module
        # Fixpoint: cloning a caller duplicates its callees' call sites, so
        # re-analyze until no eligible multi-site bearer remains. Bounded by
        # the clone cap (each iteration makes at least one clone).
        for _ in range(_MAX_CLONES + 1):
            if not self._clone_one(module):
                break
        self._specialize_all(module)
        return module
