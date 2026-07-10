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
import copy

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
    must be stable and follow the source)."""
    yield node
    for child in ast.iter_child_nodes(node):
        yield from _ordered_walk(child)


class SecurityInstantiationTransformer:
    """Not an ``ast.NodeTransformer`` — a whole-module fixpoint pass with the
    same ``visit(tree) -> tree`` pipeline interface."""

    def __init__(self):
        self._clones_made = 0
        self._module: ast.Module | None = None

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

    @staticmethod
    def _call_sites(region: ast.AST, name: str) -> list[ast.Call]:
        """Direct-Name call sites of ``name`` within ``region`` in source
        order, excluding sites inside a nested def that redefines the name
        (approximated: shadowing defs disqualify the whole function via
        duplicate-name bail-out in ``_analyze``)."""
        return [n for n in _ordered_walk(region)
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name) and n.func.id == name]

    @staticmethod
    def _non_call_refs(region: ast.AST, name: str,
                       own_def: ast.AST) -> bool:
        """Whether ``name`` is referenced outside its own def and outside a
        direct-call func position (alias, callback argument, store, ...)."""
        call_funcs = {
            id(n.func) for n in _ordered_walk(region)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        for n in _ordered_walk(region):
            if n is own_def:
                continue
            if isinstance(n, ast.Name) and n.id == name and id(n) not in call_funcs:
                return True
            if isinstance(n, (ast.Global, ast.Nonlocal)) and name in n.names:
                return True
        return False

    # --- analysis ---

    def _analyze(self, module: ast.Module) -> tuple[list[_FuncInfo], set[str]]:
        """Return (eligible security-bearing functions with >1 call site,
        bearing name set). Eligibility applies every bail-out."""
        infos = self._collect_functions(module)

        by_name: dict[str, list[_FuncInfo]] = {}
        for info in infos:
            by_name.setdefault(info.node.name, []).append(info)

        # Direct bearers: subtree contains a security call.
        bearing: set[str] = set()
        for info in infos:
            if any(self._is_security_call(n) for n in _ordered_walk(info.node)):
                bearing.add(info.node.name)

        # Name-level call edges among module functions (for transitivity and
        # cycle detection).
        edges: dict[str, set[str]] = {}
        for info in infos:
            callees = {
                n.func.id for n in _ordered_walk(info.node)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id in by_name
            }
            edges.setdefault(info.node.name, set()).update(callees)

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

        eligible: list[_FuncInfo] = []
        for info in infos:
            name = info.node.name
            if name not in bearing or name in on_cycle:
                continue
            if len(by_name[name]) > 1:  # shadowing / duplicate defs
                continue
            if info.node.decorator_list:
                continue
            if self._non_call_refs(info.region, name, info.node):
                continue
            if len(self._call_sites(info.region, name)) > 1:
                eligible.append(info)
        return eligible, bearing

    # --- cloning ---

    def _unique_name(self, base: str) -> str:
        """Collision-free clone name (module-wide check; the global clone
        counter keeps names unique even across nested re-instantiation)."""
        assert self._module is not None
        existing = {
            n.name for n in _ordered_walk(self._module)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        candidate = f"{base}__pyne_inst{self._clones_made}"
        while candidate in existing:
            candidate += "_"
        return candidate

    def _clone_one(self, module: ast.Module) -> bool:
        """Clone the first eligible multi-site bearer; True if work was done."""
        eligible, _ = self._analyze(module)
        if not eligible:
            return False
        info = eligible[0]
        name = info.node.name
        sites = self._call_sites(info.region, name)
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
            clone = copy.deepcopy(info.node)
            clone.name = self._unique_name(name)
            info.owner_body.insert(index + k - 1, clone)
            site_func = site.func
            assert isinstance(site_func, ast.Name)
            site_func.id = clone.name
        return True

    # --- pipeline API ---

    def visit(self, module: ast.Module) -> ast.Module:
        if not any(self._is_security_call(n) for n in ast.walk(module)):
            return module
        self._module = module
        # Fixpoint: cloning a caller duplicates its callees' call sites, so
        # re-analyze until no eligible multi-site bearer remains. Bounded by
        # the clone cap (each iteration makes at least one clone).
        for _ in range(_MAX_CLONES + 1):
            if not self._clone_one(module):
                break
        return module
