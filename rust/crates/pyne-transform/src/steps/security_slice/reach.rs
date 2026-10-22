//! Whether a context's write runs on every execution of the child's code (`_Reach`
//! and the statement-level rule behind it), and the `varip` scan
//! (`_declares_varip`): the two reasons `closed_shift` is cleared.

use std::collections::{BTreeSet, HashMap, HashSet};

use pyne_ast::walk::{child_statements, children, walk};
use pyne_ast::{Ast, Kind, NodeId};

use super::{
    MARKER_CALLS, PROTOCOL_CALLS, VARIP_TYPES, chain_root_name, contains_write, is_def,
    is_dispatch_test, own, protocol_sid, stmt_nodes, walk_own,
};
use crate::steps::security::collection::{arg_name, function_body, function_name};

/// Library functions whose own state is per EXECUTION rather than per bar
/// (`_PER_EXECUTION_LIB_CALLS`): a `varip` slot the child's re-tick rollback leaves
/// alone, so a period whose developing rounds are skipped would reach them a
/// different number of times. `ta.valuewhen` pushes its occurrence ring once per
/// execution; `math.sum`'s machine is a self-restoring `var` slot the rollback puts
/// back whole, so its value does not depend on the count.
const PER_EXECUTION_LIB_CALLS: [&str; 1] = ["valuewhen"];

/// Pure Python builtins a call may name without a body to read
/// (`_SAFE_BUILTIN_CALLS`): none of them touches state of the script, so an
/// unresolved one is not opaque. A builtin that runs arbitrary code (`eval`,
/// `exec`, `getattr`) must stay unresolvable.
const SAFE_BUILTIN_CALLS: [&str; 23] = [
    "abs",
    "all",
    "any",
    "bool",
    "dict",
    "enumerate",
    "filter",
    "float",
    "int",
    "len",
    "list",
    "map",
    "max",
    "min",
    "range",
    "reversed",
    "round",
    "set",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
];

/// What the target test of a reachability question matches: the sid's
/// `__sec_write__` call, or one exact node while the static call chain is
/// followed. A cached verdict is keyed by it (`_Reach._rules`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Target {
    Sid(String),
    Node(NodeId),
}

impl Target {
    fn hit(&self, ast: &Ast, node: NodeId) -> bool {
        match self {
            Target::Sid(sid) => protocol_sid(ast, node, "__sec_write__") == Some(sid.as_str()),
            Target::Node(target) => node == *target,
        }
    }
}

/// What one statement runs itself, nested definitions left out (`_OwnFacts`).
struct OwnFacts {
    /// The nodes [`walk_own`] yields for it
    ids: HashSet<NodeId>,
    /// Sids of the `__sec_write__` calls among them
    sids: HashSet<String>,
    /// Whether a write-skipping node (`return` / `raise`) stands in it
    exits: bool,
}

/// Everything one statement holds, nested definitions included (`_WholeFacts`).
struct WholeFacts {
    /// Every node `ast.walk` yields for it
    ids: HashSet<NodeId>,
    /// The function definitions in it, by name
    defs: HashMap<String, Vec<NodeId>>,
    /// The parameter names any of them declares
    params: HashSet<String>,
    /// The calls through a bare `Name`, by that name, in walk order
    calls: HashMap<String, Vec<NodeId>>,
    /// The `Name` nodes standing in callee position
    exempt: HashSet<NodeId>,
    /// Per name, the `Name` nodes of it outside callee position in this statement
    loose: HashMap<String, Vec<NodeId>>,
}

/// The nodes `ast.walk` yields for a node, in another order (`_whole_walk`): its own
/// walk, then the own walk of every definition nested in it.
fn whole_walk(ast: &Ast, node: NodeId) -> Vec<NodeId> {
    let mut out = Vec::new();
    let mut pending = vec![node];
    while let Some(next) = pending.pop() {
        let nodes = own(ast, next);
        for &sub in &nodes {
            for child in child_statements(ast, sub) {
                if is_def(ast, child) {
                    pending.push(child);
                }
            }
        }
        out.extend(nodes);
    }
    out
}

impl WholeFacts {
    fn new(ast: &Ast, stmt: NodeId) -> WholeFacts {
        let mut facts = WholeFacts {
            ids: HashSet::new(),
            defs: HashMap::new(),
            params: HashSet::new(),
            calls: HashMap::new(),
            exempt: HashSet::new(),
            loose: HashMap::new(),
        };
        let mut names = Vec::new();
        for sub in whole_walk(ast, stmt) {
            facts.ids.insert(sub);
            match ast.kind(sub) {
                Kind::Call { func, .. } => {
                    if let Kind::Name { id, .. } = ast.kind(*func) {
                        facts.calls.entry(id.clone()).or_default().push(sub);
                        facts.exempt.insert(*func);
                    }
                }
                Kind::Name { .. } => names.push(sub),
                Kind::FunctionDef { name, .. } | Kind::AsyncFunctionDef { name, .. } => {
                    facts.defs.entry(name.clone()).or_default().push(sub);
                }
                Kind::Arg { arg, .. } => {
                    facts.params.insert(arg.clone());
                }
                _ => {}
            }
        }
        for name in names {
            if !facts.exempt.contains(&name) {
                let Kind::Name { id, .. } = ast.kind(name) else { unreachable!() };
                facts.loose.entry(id.clone()).or_default().push(name);
            }
        }
        facts
    }
}

/// Whether a context's write runs on every execution of the child's code (`_Reach`).
///
/// What a statement holds does not depend on the question, so its two walks are
/// made once and cached, and so is the verdict of [`stmt_unconditional`] per
/// statement and target. The caches only serve code that does not change while it
/// is asked about: `main()`, which no part of the step edits, and the clones once
/// they are finished.
#[derive(Default)]
pub(super) struct Reach {
    own: HashMap<NodeId, OwnFacts>,
    whole: HashMap<NodeId, WholeFacts>,
    rules: HashMap<(NodeId, Target), bool>,
}

impl Reach {
    fn own_facts(&mut self, ast: &Ast, stmt: NodeId) -> &OwnFacts {
        self.own.entry(stmt).or_insert_with(|| {
            let walk = own(ast, stmt);
            let mut facts = OwnFacts { ids: HashSet::new(), sids: HashSet::new(), exits: false };
            for &sub in &walk[1..] {
                facts.ids.insert(sub);
                if let Some(sid) = protocol_sid(ast, sub, "__sec_write__") {
                    facts.sids.insert(sid.to_string());
                }
            }
            facts.exits = walk
                .iter()
                .any(|&sub| matches!(ast.kind(sub), Kind::Return { .. } | Kind::Raise { .. }));
            facts
        })
    }

    fn whole_facts(&mut self, ast: &Ast, stmt: NodeId) -> &WholeFacts {
        self.whole.entry(stmt).or_insert_with(|| WholeFacts::new(ast, stmt))
    }

    /// Whether the sid's write runs on every execution of `stmts` (`write_reached`).
    ///
    /// The statement-level rule answers first, for a write standing in `stmts`
    /// itself. A write inside a helper body is then judged along the static call
    /// chain ([`Reach::node_reached`]). One sid belongs to ONE call site there; two
    /// write nodes for one sid would not be unambiguous, and answer false.
    pub(super) fn write_reached(&mut self, ast: &Ast, stmts: &[NodeId], sid: &str) -> bool {
        let target = Target::Sid(sid.to_string());
        if self.unconditional(ast, stmts, &target) {
            return true;
        }
        let mut targets = Vec::new();
        for &stmt in stmts {
            let whole = self.whole_facts(ast, stmt);
            for &call in whole.calls.get("__sec_write__").into_iter().flatten() {
                if target.hit(ast, call) {
                    targets.push(call);
                }
            }
        }
        if targets.len() != 1 {
            return false;
        }
        self.node_reached(ast, targets[0], stmts, &BTreeSet::new())
    }

    /// [`write_unconditional`], read from the cached facts (`_unconditional`).
    fn unconditional(&mut self, ast: &Ast, stmts: &[NodeId], target: &Target) -> bool {
        for &stmt in stmts {
            if is_def(ast, stmt) {
                continue;
            }
            let own = self.own_facts(ast, stmt);
            let holds = match target {
                Target::Sid(sid) => own.sids.contains(sid),
                Target::Node(node) => own.ids.contains(node),
            };
            if holds {
                let rule = (stmt, target.clone());
                if let Some(&verdict) = self.rules.get(&rule) {
                    return verdict;
                }
                let verdict = stmt_unconditional(ast, stmt, &|ast, node| target.hit(ast, node));
                self.rules.insert(rule, verdict);
                return verdict;
            }
            if own.exits {
                return false;
            }
        }
        false
    }

    /// The nested `def`s standing between `body` and `target` (`_def_chain`): empty
    /// when the target stands in `body` itself, `None` when the chain is not one
    /// this pass models (the target is not in `body` at all, or is reached only
    /// through a `def` that is not a top-level statement of its scope).
    fn def_chain(&mut self, ast: &Ast, body: &[NodeId], target: NodeId) -> Option<Vec<NodeId>> {
        for &stmt in body {
            let holds = self.whole_facts(ast, stmt).ids.contains(&target);
            if is_def(ast, stmt) {
                if !holds {
                    continue;
                }
                // In a decorator or a default, or behind a conditional ``def``.
                let inner = self.def_chain(ast, &function_body(ast, stmt).clone(), target)?;
                let mut chain = vec![stmt];
                chain.extend(inner);
                return Some(chain);
            }
            if holds {
                return Some(Vec::new());
            }
        }
        None
    }

    /// Whether `func`'s body runs on every execution of `body`
    /// (`_def_called_unconditionally`).
    ///
    /// The callee resolution is the closed rule of the `varip` scan narrowed to a
    /// bare `Name` that resolves to this very `def`. A second `def` or binding of the
    /// name, a parameter shadowing it, or a reference outside callee position makes
    /// it unresolvable and the answer false. Every call site must then run on every
    /// execution of `body` itself; recursion answers false through `seen`.
    fn def_called_unconditionally(
        &mut self,
        ast: &Ast,
        func: NodeId,
        body: &[NodeId],
        seen: &BTreeSet<String>,
    ) -> bool {
        let name = function_name(ast, func).to_string();
        if seen.contains(&name) {
            return false;
        }
        let mut seen = seen.clone();
        seen.insert(name.clone());
        for &stmt in body {
            self.whole_facts(ast, stmt);
        }
        let facts: Vec<&WholeFacts> = body.iter().map(|stmt| &self.whole[stmt]).collect();
        let mut calls: Vec<NodeId> = Vec::new();
        for whole in &facts {
            if whole.params.contains(&name)
                || whole.defs.get(&name).into_iter().flatten().any(|&node| node != func)
            {
                return false;
            }
            // A node may stand in two statements, so a reference is in callee
            // position when ANY statement of the scope calls through it.
            for reference in whole.loose.get(&name).into_iter().flatten() {
                if !facts.iter().any(|other| other.exempt.contains(reference)) {
                    return false;
                }
            }
            calls.extend(whole.calls.get(&name).into_iter().flatten().copied());
        }
        if calls.is_empty() {
            return false;
        }
        calls.into_iter().all(|call| self.node_reached(ast, call, body, &seen))
    }

    /// Whether `target` runs on every execution of `body` (`_node_reached`).
    ///
    /// Either the target stands in `body` itself, and the statement-level rule
    /// decides, or it stands in a nested `def`, and both links have to hold: the
    /// target runs on every execution of that def's body, and the def is entered on
    /// every execution of `body`.
    fn node_reached(
        &mut self,
        ast: &Ast,
        target: NodeId,
        body: &[NodeId],
        seen: &BTreeSet<String>,
    ) -> bool {
        let Some(chain) = self.def_chain(ast, body, target) else { return false };
        let Some(&outer) = chain.first() else {
            return self.unconditional(ast, body, &Target::Node(target));
        };
        let outer_body = function_body(ast, outer).clone();
        self.node_reached(ast, target, &outer_body, seen)
            && self.def_called_unconditionally(ast, outer, body, seen)
    }
}

/// Whether the statement can end the round before the next statement (`_has_exit`).
fn has_exit(ast: &Ast, stmt: NodeId) -> bool {
    own(ast, stmt)
        .into_iter()
        .any(|sub| matches!(ast.kind(sub), Kind::Return { .. } | Kind::Raise { .. }))
}

/// Whether the target stands under no expression-level branch inside `node`
/// (`_expr_unguarded`).
///
/// The two expression-level branches are `IfExp` and the short-circuit `BoolOp`: a
/// target in a branch of the first or in a later operand of the second is guarded.
/// A lambda or a comprehension answers false as well: when its body runs is not
/// visible here.
pub(super) fn expr_unguarded(ast: &Ast, node: NodeId, hit: &dyn Fn(&Ast, NodeId) -> bool) -> bool {
    if hit(ast, node) {
        return true;
    }
    match ast.kind(node) {
        &Kind::IfExp { test, body, orelse } => {
            // Only the test itself runs unconditionally.
            contains_write(ast, test, hit)
                && ![body, orelse].into_iter().any(|branch| contains_write(ast, branch, hit))
                && expr_unguarded(ast, test, hit)
        }
        Kind::BoolOp { values, .. } => {
            // Only the first operand runs whatever the others evaluate to.
            let first = values[0];
            contains_write(ast, first, hit)
                && !values[1..].iter().any(|&value| contains_write(ast, value, hit))
                && expr_unguarded(ast, first, hit)
        }
        Kind::Lambda { .. }
        | Kind::ListComp { .. }
        | Kind::SetComp { .. }
        | Kind::DictComp { .. }
        | Kind::GeneratorExp { .. } => false,
        _ => children(ast, node).into_iter().all(|child| {
            is_def(ast, child)
                || !contains_write(ast, child, hit)
                || expr_unguarded(ast, child, hit)
        }),
    }
}

/// Whether the target runs on EVERY execution of `stmts` (`_write_unconditional`).
///
/// The target must stand in a TOP-LEVEL statement, with no early exit reachable
/// ahead of it, and with no branch of any kind around it. The only `if` that is not
/// a user branch is the protocol's own `__active_security__` dispatch. A target the
/// scan does not find at all answers false. The scan never enters a nested `def`.
fn write_unconditional(ast: &Ast, stmts: &[NodeId], hit: &dyn Fn(&Ast, NodeId) -> bool) -> bool {
    for &stmt in stmts {
        // A nested ``def`` only binds a name here, so neither its write nor its
        // ``return`` belongs to this statement list.
        if is_def(ast, stmt) {
            continue;
        }
        if walk_own(ast, stmt).into_iter().any(|sub| hit(ast, sub)) {
            return stmt_unconditional(ast, stmt, hit);
        }
        if has_exit(ast, stmt) {
            return false;
        }
    }
    false
}

/// Whether the statement holding the target performs it on every run
/// (`_stmt_unconditional`).
fn stmt_unconditional(ast: &Ast, stmt: NodeId, hit: &dyn Fn(&Ast, NodeId) -> bool) -> bool {
    if holds_write(ast, stmt, hit) {
        return expr_unguarded(ast, stmt, hit);
    }
    let Kind::If { test, body, orelse } = ast.kind(stmt) else { return false };
    if contains_write(ast, *test, hit) {
        // A target inside the test itself runs before the branch is taken.
        return expr_unguarded(ast, *test, hit);
    }
    if orelse.iter().any(|&sub| contains_write(ast, sub, hit)) {
        return false;
    }
    if body.iter().any(|&sub| contains_write(ast, sub, hit)) {
        if !(is_dispatch_test(ast, *test) && orelse.is_empty()) {
            return false;
        }
        return write_unconditional(ast, body, hit);
    }
    true
}

/// Whether a simple statement holds the target itself (`_holds_write`).
fn holds_write(ast: &Ast, stmt: NodeId, hit: &dyn Fn(&Ast, NodeId) -> bool) -> bool {
    matches!(
        ast.kind(stmt),
        Kind::Expr { .. }
            | Kind::Assign { .. }
            | Kind::AnnAssign { .. }
            | Kind::AugAssign { .. }
            | Kind::Return { .. }
    ) && walk_own(ast, stmt).into_iter().any(|sub| hit(ast, sub))
}

// --- the varip scan ---

/// Whether an annotation declares a `varip` (`IBPersistent`) variable
/// (`_is_varip_annotation`).
fn is_varip_annotation(ast: &Ast, annotation: NodeId) -> bool {
    match ast.kind(annotation) {
        Kind::Name { id, .. } => VARIP_TYPES.contains(&id.as_str()),
        Kind::Subscript { value, .. } => {
            matches!(ast.kind(*value), Kind::Name { id, .. } if VARIP_TYPES.contains(&id.as_str()))
        }
        Kind::Attribute { attr, .. } => VARIP_TYPES.contains(&attr.as_str()),
        _ => false,
    }
}

/// Whether a module path names the PyneCore package itself (`_is_pynecore_module`).
fn is_pynecore_module(module: Option<&str>) -> bool {
    module.is_some_and(|module| module == "pynecore" || module.starts_with("pynecore."))
}

/// The name an import alias binds: `asname`, or the first part of the dotted name.
fn alias_binding(ast: &Ast, alias: NodeId) -> (String, &str) {
    let Kind::Alias { name, asname } = ast.kind(alias) else { unreachable!("no alias") };
    let bound = match asname {
        Some(asname) => asname.clone(),
        None => name.split('.').next().unwrap_or(name).to_string(),
    };
    (bound, name)
}

/// Names an import binds from the pynecore package, or from any other module
/// (`_import_bindings`).
fn import_bindings(ast: &Ast, node: NodeId, pynecore: bool, out: &mut HashSet<String>) {
    match ast.kind(node) {
        Kind::ImportFrom { module, names, .. } => {
            if is_pynecore_module(module.as_deref()) != pynecore {
                return;
            }
            out.extend(names.iter().map(|&alias| alias_binding(ast, alias).0));
        }
        Kind::Import { names } => {
            for &alias in names {
                let (bound, name) = alias_binding(ast, alias);
                if is_pynecore_module(Some(name)) == pynecore {
                    out.insert(bound);
                }
            }
        }
        _ => {}
    }
}

/// Names an import of the PYNECORE package binds (`_pynecore_imported_names`).
///
/// Such a name stands for a library function, and the library's per-execution state
/// is enumerated, so a call through it needs no body: the same trust the
/// `lib`-rooted attribute spelling gets.
pub(super) fn pynecore_imported_names(ast: &Ast, nodes: &[NodeId]) -> HashSet<String> {
    let mut names = HashSet::new();
    for &node in nodes {
        import_bindings(ast, node, true, &mut names);
    }
    names
}

/// Names an assignment, a parameter, or a `for` / `with` / comprehension target
/// binds, and the names an import of a NON-pynecore module binds (`_rebound_names`).
///
/// A call through one of them is not resolvable by name: a parameter may SHADOW a
/// resolvable name, and `from helper import abs` binds an opaque user callable to a
/// spelling the safe builtins would otherwise trust.
pub(super) fn rebound_names(ast: &Ast, nodes: &[NodeId]) -> HashSet<String> {
    let mut names = HashSet::new();
    for &node in nodes {
        let targets: Vec<NodeId> = match ast.kind(node) {
            Kind::Import { .. } | Kind::ImportFrom { .. } => {
                import_bindings(ast, node, false, &mut names);
                continue;
            }
            Kind::FunctionDef { args, .. }
            | Kind::AsyncFunctionDef { args, .. }
            | Kind::Lambda { args, .. } => {
                let Kind::Arguments { posonlyargs, args, vararg, kwonlyargs, kwarg, .. } =
                    ast.kind(*args)
                else {
                    unreachable!("no arguments")
                };
                let params =
                    posonlyargs.iter().chain(args).chain(kwonlyargs).chain(vararg).chain(kwarg);
                names.extend(params.map(|&arg| arg_name(ast, arg).to_string()));
                continue;
            }
            Kind::Assign { targets, .. } => targets.clone(),
            Kind::AnnAssign { target, .. }
            | Kind::AugAssign { target, .. }
            | Kind::For { target, .. }
            | Kind::AsyncFor { target, .. }
            | Kind::Comprehension { target, .. } => vec![*target],
            Kind::WithItem { optional_vars, .. } => optional_vars.iter().copied().collect(),
            _ => continue,
        };
        for target in targets {
            for sub in walk(ast, target) {
                if let Kind::Name { id, .. } = ast.kind(sub) {
                    names.insert(id.clone());
                }
            }
        }
    }
    names
}

/// Whether the code these statements RUN carries `varip` state (`_declares_varip`).
///
/// The child's re-tick rollback restores the state VECTOR; `varip` (`IBPersistent`)
/// slots are deliberately outside it, so a period whose developing rounds are
/// skipped would end on a different number. The scan has to be COMPLETE over the
/// code the child can execute, so a call is followed only when its callee is
/// statically visible:
///
/// - a `lib`-rooted (or pynecore-import-rooted) attribute chain whose attribute is
///   not a per-execution library call,
/// - a bare `Name` the module defines with a `def`, whose bodies are then scanned
///   with the same rules,
/// - a bare `Name` bound by a pynecore import, one of the protocol / marker calls
///   the earlier passes emit, or one of the pure builtins.
///
/// A name an assignment, a parameter or a non-pynecore import also binds
/// (`rebound`) is never resolved. Every other callee gives the skip up.
pub(super) fn declares_varip(
    ast: &Ast,
    stmts: &[NodeId],
    funcs: &HashMap<String, Vec<NodeId>>,
    rebound: &HashSet<String>,
    pyne_imported: &HashSet<String>,
) -> bool {
    let mut seen: HashSet<NodeId> = HashSet::new();
    let mut work: Vec<NodeId> = Vec::new();
    for &stmt in stmts {
        work.extend(stmt_nodes(ast, stmt));
    }
    while let Some(node) = work.pop() {
        if !seen.insert(node) {
            continue;
        }
        for sub in own(ast, node) {
            match ast.kind(sub) {
                Kind::AnnAssign { annotation, .. } if is_varip_annotation(ast, *annotation) => {
                    return true;
                }
                Kind::Global { .. } | Kind::Nonlocal { .. } => return true,
                Kind::Call { func, .. } => match ast.kind(*func) {
                    Kind::Name { id, .. } => {
                        let id = id.as_str();
                        if rebound.contains(id) || PER_EXECUTION_LIB_CALLS.contains(&id) {
                            return true;
                        }
                        match funcs.get(id) {
                            Some(bodies) if !bodies.is_empty() => work.extend(bodies),
                            _ => {
                                if !pyne_imported.contains(id)
                                    && !PROTOCOL_CALLS.contains(&id)
                                    && !MARKER_CALLS.contains(&id)
                                    && !SAFE_BUILTIN_CALLS.contains(&id)
                                {
                                    return true;
                                }
                            }
                        }
                    }
                    Kind::Attribute { attr, .. } => {
                        if PER_EXECUTION_LIB_CALLS.contains(&attr.as_str()) {
                            return true;
                        }
                        match chain_root_name(ast, *func) {
                            Some(root)
                                if !rebound.contains(root)
                                    && (root == "lib" || pyne_imported.contains(root)) => {}
                            _ => return true,
                        }
                    }
                    // Neither a name nor an attribute chain: nothing to resolve.
                    _ => return true,
                },
                _ => {}
            }
        }
    }
    false
}
