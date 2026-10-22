//! Port of `transformers/security_dependencies.py`: which security contexts'
//! results flow into which other context's write (`_DependencyAnalyzer`), a forward
//! taint analysis over the module the security step has lowered. Its lexical half
//! (the scopes, name qualification) is what the slice pass reuses.
//!
//! What the analysis models, in short (the Python module documents each rule):
//!
//! - flow-insensitive on data: a name's taint is the union over every assignment
//!   to it in its scope, except a pure loop variable, which gets a cell per loop;
//! - lexically scoped names, `global` / `nonlocal` rebinding them;
//! - flow-sensitive on control: the conditions of enclosing `if` / loops / `match`
//!   accumulate, and a tainted `return` / `break` / `continue` taints what runs
//!   after it;
//! - collections: a call in the `array` / `matrix` / `map` namespace writes back
//!   into its first argument, unless it only reads it;
//! - alias classes: `a = b` merges two names into one taint class;
//! - user functions per call site: a parameter carries a symbolic token, replaced
//!   by the call's argument taint; the call site's control taint is one more
//!   pseudo-parameter;
//! - fallback: a scope holding an unmodelled construct depends on every sid.
//!
//! Only producers sited earlier than a write stay in `depends`; the later-sited
//! ones go to `late_reads`.
//!
//! The taint sets are ordered sets, where the Python ones iterate in hash order:
//! the analysis only ever unions them, so no result depends on the order.

use std::collections::{BTreeSet, HashMap, HashSet};

use indexmap::{IndexMap, IndexSet};
use pyne_ast::walk::{children, walk};
use pyne_ast::{Ast, Constant, ExprContext, Kind, NodeId};

use super::collection::{arg_name, function_args, function_name, is_function, params};
use crate::Error;
use crate::pine_type_rules::FactoryFields;

/// A set of sids and parameter tokens.
pub(crate) type Taint = BTreeSet<String>;

/// Fixpoint safety net: the lattice is finite and monotone, so the loop always
/// converges; the cap only guards against an unforeseen non-monotone edit.
const MAX_FIXPOINT_ROUNDS: usize = 200;

/// Prefix marking a symbolic parameter token inside a taint set.
const PARAM_PREFIX: &str = "\0param\0";

/// Pseudo-parameter carrying the control taint a call site runs under.
const CTRL_PARAM: &str = "<ctrl>";

/// Change-tracking cells of a function's summaries, followed by its scope key.
const RET_CELL: &str = "\u{1}ret\0";
const OUT_CELL: &str = "\u{1}out\0";
const ALIAS_CELL: &str = "\u{1}alias\0";

/// Names that can never carry taint: `lib` is the library namespace root.
const UNTAINTABLE_ROOTS: [&str; 1] = ["lib"];

/// `lib` namespaces holding the collection APIs (`_COLLECTION_NAMESPACES`).
pub(crate) const COLLECTION_NAMESPACES: [&str; 3] = ["array", "matrix", "map"];

/// Collection functions that only READ their collection (`_COLLECTION_READERS`):
/// an index, key or search argument of theirs does not become part of the
/// collection's taint.
pub(crate) fn is_collection_reader(namespace: &str, function: &str) -> bool {
    const ARRAY: &[&str] = &[
        "abs",
        "avg",
        "binary_search",
        "binary_search_leftmost",
        "binary_search_rightmost",
        "copy",
        "covariance",
        "every",
        "first",
        "from_items",
        "get",
        "includes",
        "indexof",
        "join",
        "last",
        "lastindexof",
        "max",
        "median",
        "min",
        "mode",
        "new",
        "new_bool",
        "new_box",
        "new_color",
        "new_float",
        "new_int",
        "new_label",
        "new_line",
        "new_linefill",
        "new_string",
        "new_table",
        "percentile_linear_interpolation",
        "percentile_nearest_rank",
        "percentrank",
        "range",
        "size",
        "some",
        "sort_indices",
        "standardize",
        "stdev",
        "sum",
        "variance",
    ];
    const MATRIX: &[&str] = &[
        "all",
        "avg",
        "col",
        "columns",
        "copy",
        "det",
        "diff",
        "eigenvalues",
        "eigenvectors",
        "elements_count",
        "get",
        "inv",
        "is_antidiagonal",
        "is_antisymmetric",
        "is_binary",
        "is_diagonal",
        "is_identity",
        "is_square",
        "is_stochastic",
        "is_symmetric",
        "is_triangular",
        "is_zero",
        "kron",
        "max",
        "median",
        "min",
        "mode",
        "mult",
        "new",
        "pinv",
        "pow",
        "rank",
        "row",
        "rows",
        "submatrix",
        "sum",
        "trace",
        "transpose",
    ];
    const MAP: &[&str] = &["contains", "copy", "get", "keys", "new", "size", "values"];
    let readers = match namespace {
        "array" => ARRAY,
        "matrix" => MATRIX,
        "map" => MAP,
        _ => return false,
    };
    readers.contains(&function)
}

/// Statement / expression node classes the analyzer does not model
/// (`_UNMODELLED_NODES`): a scope holding one falls back to "depends on every
/// other sid".
pub(crate) fn is_unmodelled(kind: &Kind) -> bool {
    matches!(
        kind,
        Kind::Try { .. }
            | Kind::TryStar { .. }
            | Kind::With { .. }
            | Kind::AsyncWith { .. }
            | Kind::AsyncFor { .. }
            | Kind::Raise { .. }
            | Kind::Delete { .. }
            | Kind::ClassDef { .. }
            | Kind::Lambda { .. }
            | Kind::ListComp { .. }
            | Kind::SetComp { .. }
            | Kind::DictComp { .. }
            | Kind::GeneratorExp { .. }
            | Kind::Yield { .. }
            | Kind::YieldFrom { .. }
            | Kind::Await { .. }
            | Kind::Assert { .. }
    )
}

/// Expression classes whose taint is the union of their children's (`_UNION_EXPRS`).
fn is_union_expr(kind: &Kind) -> bool {
    matches!(
        kind,
        Kind::BinOp { .. }
            | Kind::UnaryOp { .. }
            | Kind::BoolOp { .. }
            | Kind::Compare { .. }
            | Kind::IfExp { .. }
            | Kind::Tuple { .. }
            | Kind::List { .. }
            | Kind::Set { .. }
            | Kind::Dict { .. }
            | Kind::Slice { .. }
            | Kind::Starred { .. }
            | Kind::JoinedStr { .. }
            | Kind::FormattedValue { .. }
            | Kind::Subscript { .. }
    )
}

/// Nodes whose children may be skipped on a pass through the code around them
/// (`_CONDITIONAL_NODES`).
fn is_conditional_node(kind: &Kind) -> bool {
    matches!(
        kind,
        Kind::If { .. }
            | Kind::IfExp { .. }
            | Kind::BoolOp { .. }
            | Kind::Try { .. }
            | Kind::TryStar { .. }
            | Kind::Match { .. }
            | Kind::For { .. }
            | Kind::AsyncFor { .. }
            | Kind::While { .. }
            | Kind::ListComp { .. }
            | Kind::SetComp { .. }
            | Kind::DictComp { .. }
            | Kind::GeneratorExp { .. }
    )
}

fn is_loop(kind: &Kind) -> bool {
    matches!(kind, Kind::For { .. } | Kind::AsyncFor { .. } | Kind::While { .. })
}

/// The bare name of a scope-qualified name (`_bare`).
fn bare(name: &str) -> &str {
    name.rsplit('\0').next().unwrap_or(name)
}

fn is_untaintable(name: &str) -> bool {
    UNTAINTABLE_ROOTS.contains(&bare(name))
}

/// A parameter token (`_token`).
fn token(qualified_param: &str) -> String {
    format!("{PARAM_PREFIX}{qualified_param}")
}

/// The token standing for the control taint a call of a scope runs under
/// (`_ctrl_token`).
fn ctrl_token(scope_key: &str) -> String {
    token(&format!("{scope_key}\0{CTRL_PARAM}"))
}

/// Replace a call's parameter tokens with the argument taints (`_subst`); tokens
/// the map does not name are carried through.
fn subst(taint: &Taint, token_map: &HashMap<String, Taint>) -> Taint {
    let mut out = Taint::new();
    for item in taint {
        if item.starts_with(PARAM_PREFIX)
            && let Some(mapped) = token_map.get(item)
        {
            out.extend(mapped.iter().cloned());
        } else {
            out.insert(item.clone());
        }
    }
    out
}

/// The sid string of a `__sec_read__` / `__sec_write__` call (`_sid_arg`).
pub(crate) fn sid_arg(ast: &Ast, call: NodeId) -> Option<&str> {
    let Kind::Call { args, .. } = ast.kind(call) else { return None };
    match ast.kind(*args.first()?) {
        Kind::Constant { value: Constant::Str(sid), .. } => Some(sid),
        _ => None,
    }
}

/// The base name of a place expression (`a`, `a.b`, `a[i].c`) (`_root_name`).
pub(crate) fn root_name(ast: &Ast, mut node: NodeId) -> Option<&str> {
    loop {
        match ast.kind(node) {
            Kind::Attribute { value, .. } | Kind::Subscript { value, .. } => node = *value,
            Kind::Name { id, .. } => return Some(id),
            _ => return None,
        }
    }
}

/// The collection namespace of a `lib.<array|matrix|map>.<fn>` callee
/// (`_collection_namespace`).
fn collection_namespace(ast: &Ast, func: NodeId) -> Option<&str> {
    let Kind::Attribute { value: owner, .. } = ast.kind(func) else { return None };
    let Kind::Attribute { value, attr, .. } = ast.kind(*owner) else { return None };
    (COLLECTION_NAMESPACES.contains(&attr.as_str())
        && matches!(ast.kind(*value), Kind::Name { id, .. } if id == "lib"))
    .then_some(attr.as_str())
}

/// Whether a class body only declares fields, a Pine `type` (`_is_plain_record`).
fn is_plain_record(
    ast: &Ast,
    node: NodeId,
    factories: &FactoryFields,
    class_names: &HashSet<String>,
) -> bool {
    let factory_calls: HashSet<NodeId> = factories.of(ast, node).into_iter().collect();
    let Kind::ClassDef { body, .. } = ast.kind(node) else { unreachable!() };
    for &stmt in body {
        match ast.kind(stmt) {
            Kind::AnnAssign { value, .. } => {
                let lambdas: Vec<NodeId> = walk(ast, stmt)
                    .into_iter()
                    .filter(|&sub| matches!(ast.kind(sub), Kind::Lambda { .. }))
                    .collect();
                if lambdas.is_empty() {
                    continue;
                }
                let Kind::Lambda { body: lambda_body, .. } = ast.kind(lambdas[0]) else {
                    unreachable!()
                };
                let reads_script = || {
                    walk(ast, *lambda_body).into_iter().any(|sub| {
                        matches!(ast.kind(sub), Kind::Name { id, .. }
                            if !id.contains('\u{b7}') && id != "lib" && !class_names.contains(id))
                    })
                };
                if lambdas.len() != 1
                    || !value.is_some_and(|value| factory_calls.contains(&value))
                    || reads_script()
                {
                    return false;
                }
            }
            Kind::Pass => {}
            Kind::Assign { targets, .. }
                if targets.iter().all(|&target| matches!(ast.kind(target), Kind::Name { .. })) => {}
            Kind::Expr { value } if matches!(ast.kind(*value), Kind::Constant { .. }) => {}
            _ => return false,
        }
    }
    true
}

/// One lexical scope of the analysed module, the module or a function (`_Scope`).
/// Its key, the unique dotted path of the scope (`""` for the module), is the key
/// [`DependencyAnalyzer::scopes`] holds it under.
#[derive(Debug)]
pub(crate) struct Scope {
    /// The `Module` / `FunctionDef` the scope belongs to
    pub(crate) node: NodeId,
    /// Key of the enclosing scope, `None` for the module
    pub(crate) parent: Option<String>,
    /// Names bound inside the scope that are not declared `global` / `nonlocal`
    pub(crate) locals: HashSet<String>,
    /// Names the scope declares `global`
    pub(crate) globals: HashSet<String>,
    /// Nested function name to scope key
    pub(crate) funcs: HashMap<String, String>,
    /// The scope's statement list
    pub(crate) body: Vec<NodeId>,
    /// The direct-`Name` calls of the scope's own code in evaluation order, as
    /// `(callee name, call, inside a loop)`
    pub(crate) calls: Vec<(String, NodeId, bool)>,
}

/// A statement's last visit: the counter before it, the control taint in and out,
/// the `ctrl` it left behind, and the cells it read.
type MemoEntry = (u64, Taint, Taint, Taint, HashSet<String>);

/// Per loop, `name -> (occurrences, stored first)` (`_collect_loop_names`).
type LoopNames = IndexMap<NodeId, IndexMap<String, (usize, bool)>>;

/// The forward taint analysis (`_DependencyAnalyzer`).
pub(crate) struct DependencyAnalyzer<'a> {
    ast: &'a Ast,
    module: NodeId,
    all_list: Vec<String>,
    all: Taint,
    /// Per sid, the sids whose results flow into its write and whose producers
    /// come earlier in program order
    pub(crate) depends: IndexMap<String, Taint>,
    /// Sids whose write can run more than once per bar
    pub(crate) in_loop: HashSet<String>,
    /// Per sid, the flowing sids whose producers come later
    pub(crate) late_reads: HashMap<String, Taint>,
    funcs: IndexMap<String, NodeId>,
    pub(crate) scopes: IndexMap<String, Scope>,
    pub(crate) key_of: HashMap<NodeId, String>,
    called_names: HashSet<String>,
    bound: HashSet<String>,
    fallback_scopes: HashSet<String>,
    unmodelled: HashMap<String, Vec<NodeId>>,
    loop_scopes: HashSet<String>,
    ret: HashMap<String, Taint>,
    ret_alias: HashMap<String, IndexSet<String>>,
    param_out: HashMap<String, HashMap<String, Taint>>,
    param_in: HashMap<String, Taint>,
    escaped: Taint,
    broke: Taint,
    parent: HashMap<String, String>,
    taint: HashMap<String, Taint>,
    loop_cells: HashMap<NodeId, IndexMap<String, String>>,
    foreign_memo: HashMap<NodeId, HashMap<String, usize>>,
    name_cells: HashMap<String, String>,
    ctrl: Taint,
    order: HashMap<String, usize>,
    scope: String,
    changed: bool,
    final_round: bool,
    tick: u64,
    stamps: HashMap<String, u64>,
    reads: Option<HashSet<String>>,
    memo: HashMap<NodeId, MemoEntry>,
    escapes: usize,
}

impl<'a> DependencyAnalyzer<'a> {
    pub(crate) fn new(ast: &'a Ast, module: NodeId, all_sids: &[String]) -> Self {
        DependencyAnalyzer {
            ast,
            module,
            all_list: all_sids.to_vec(),
            all: all_sids.iter().cloned().collect(),
            depends: all_sids.iter().map(|sid| (sid.clone(), Taint::new())).collect(),
            in_loop: HashSet::new(),
            late_reads: HashMap::new(),
            funcs: IndexMap::new(),
            scopes: IndexMap::new(),
            key_of: HashMap::new(),
            called_names: HashSet::new(),
            bound: HashSet::new(),
            fallback_scopes: HashSet::new(),
            unmodelled: HashMap::new(),
            loop_scopes: HashSet::new(),
            ret: HashMap::new(),
            ret_alias: HashMap::new(),
            param_out: HashMap::new(),
            param_in: HashMap::new(),
            escaped: Taint::new(),
            broke: Taint::new(),
            parent: HashMap::new(),
            taint: HashMap::new(),
            loop_cells: HashMap::new(),
            foreign_memo: HashMap::new(),
            name_cells: HashMap::new(),
            ctrl: Taint::new(),
            order: HashMap::new(),
            scope: String::new(),
            changed: false,
            final_round: false,
            tick: 0,
            stamps: HashMap::new(),
            reads: None,
            memo: HashMap::new(),
            escapes: 0,
        }
    }

    // --- scopes ---

    /// Build the lexical scope tree and each scope's set of bound names
    /// (`_build_scopes`).
    pub(crate) fn build_scopes(&mut self) {
        let ast = self.ast;
        let Kind::Module { body, .. } = ast.kind(self.module) else { unreachable!() };
        let root = Scope {
            node: self.module,
            parent: None,
            locals: HashSet::new(),
            globals: HashSet::new(),
            funcs: HashMap::new(),
            body: body.clone(),
            calls: Vec::new(),
        };
        self.scopes.insert(String::new(), root);
        self.key_of.insert(self.module, String::new());
        let mut work: Vec<String> = vec![String::new()];
        let mut used: HashSet<String> = HashSet::from([String::new()]);
        while let Some(key) = work.pop() {
            let node = self.scopes[&key].node;
            if is_function(ast.kind(node)) {
                let names: Vec<String> =
                    params(ast, node).iter().map(|&arg| arg_name(ast, arg).to_string()).collect();
                self.scopes[&key].locals.extend(names);
            }
            let (funcs, declared) = self.scan_scope(&key);
            for function in funcs {
                let name = function_name(ast, function).to_string();
                self.scopes[&key].locals.insert(name.clone());
                let mut child_key =
                    if key.is_empty() { name.clone() } else { format!("{key}.{name}") };
                while used.contains(&child_key) {
                    child_key.push('~');
                }
                used.insert(child_key.clone());
                let body = super::collection::function_body(ast, function).clone();
                let child = Scope {
                    node: function,
                    parent: Some(key.clone()),
                    locals: HashSet::new(),
                    globals: HashSet::new(),
                    funcs: HashMap::new(),
                    body,
                    calls: Vec::new(),
                };
                self.scopes.insert(child_key.clone(), child);
                self.key_of.insert(function, child_key.clone());
                self.scopes[&key].funcs.insert(name, child_key.clone());
                work.push(child_key);
            }
            let scope = &mut self.scopes[&key];
            scope.locals.retain(|name| !declared.contains(name));
        }
    }

    /// Walk a scope's own code once (`_scan_scope`): its bound names, `global`
    /// names, loops, unmodelled constructs and direct-`Name` calls. Answers the
    /// function definitions the scope owns, in walk order, and the names it
    /// declares `global` / `nonlocal`.
    fn scan_scope(&mut self, key: &str) -> (Vec<NodeId>, HashSet<String>) {
        let ast = self.ast;
        let mut funcs = Vec::new();
        let mut declared = HashSet::new();
        let mut unmodelled = Vec::new();
        let mut bound: Vec<String> = Vec::new();
        let mut globals: Vec<String> = Vec::new();
        let mut calls: Vec<(String, NodeId, bool)> = Vec::new();
        let mut loops = 0usize;
        let mut has_loop = false;
        // Per node being walked: its remaining children, the node itself when it
        // is a direct-``Name`` call (recorded once its children are done, which
        // puts the arguments first), and whether it is a loop
        let scope_node = self.scopes[key].node;
        let mut stack: Vec<(std::vec::IntoIter<NodeId>, Option<NodeId>, bool)> =
            vec![(children(ast, scope_node).into_iter(), None, false)];
        while let Some((rest, _, _)) = stack.last_mut() {
            let mut pushed = None;
            for sub in rest.by_ref() {
                let kind = ast.kind(sub);
                match kind {
                    Kind::Name { id, ctx } => {
                        if *ctx == ExprContext::Store {
                            bound.push(id.clone());
                        }
                        continue;
                    }
                    Kind::Constant { .. } => continue,
                    Kind::Import { names } | Kind::ImportFrom { names, .. } => {
                        for &alias in names {
                            let Kind::Alias { name, asname } = ast.kind(alias) else {
                                unreachable!()
                            };
                            let bound_name = match asname {
                                Some(asname) => asname.clone(),
                                None => name.split('.').next().unwrap_or(name).to_string(),
                            };
                            bound.push(bound_name);
                        }
                        continue;
                    }
                    Kind::Global { names } => {
                        globals.extend(names.iter().cloned());
                        declared.extend(names.iter().cloned());
                        continue;
                    }
                    Kind::Nonlocal { names } => {
                        declared.extend(names.iter().cloned());
                        continue;
                    }
                    Kind::FunctionDef { .. } | Kind::AsyncFunctionDef { .. } => {
                        funcs.push(sub);
                        continue;
                    }
                    _ => {}
                }
                if is_loop(kind) {
                    if is_unmodelled(kind) {
                        unmodelled.push(sub);
                    }
                    has_loop = true;
                    loops += 1;
                    pushed = Some((children(ast, sub).into_iter(), None, true));
                } else if is_unmodelled(kind) {
                    unmodelled.push(sub);
                    pushed = Some((children(ast, sub).into_iter(), None, false));
                } else if let Kind::Call { func, .. } = kind {
                    let named = matches!(ast.kind(*func), Kind::Name { .. });
                    pushed = Some((children(ast, sub).into_iter(), named.then_some(sub), false));
                } else {
                    pushed = Some((children(ast, sub).into_iter(), None, false));
                }
                break;
            }
            match pushed {
                Some(entry) => stack.push(entry),
                None => {
                    let (_, call, is_loop_node) = stack.pop().expect("a frame");
                    if let Some(call) = call {
                        let Kind::Call { func, .. } = ast.kind(call) else { unreachable!() };
                        let Kind::Name { id, .. } = ast.kind(*func) else { unreachable!() };
                        calls.push((id.clone(), call, loops > 0));
                    }
                    if is_loop_node {
                        loops -= 1;
                    }
                }
            }
        }
        if has_loop {
            self.loop_scopes.insert(key.to_string());
        }
        let scope = &mut self.scopes[key];
        scope.locals.extend(bound);
        scope.globals.extend(globals);
        scope.calls.extend(calls);
        if !unmodelled.is_empty() {
            self.unmodelled.insert(key.to_string(), unmodelled);
        }
        (funcs, declared)
    }

    /// Qualify a bare name with the key of the scope that binds it (`_q`).
    pub(crate) fn q(&self, name: &str, scope_key: Option<&str>) -> String {
        if scope_key.is_none()
            && let Some(cell) = self.name_cells.get(name)
        {
            return cell.clone();
        }
        let mut key = scope_key.unwrap_or(&self.scope).to_string();
        loop {
            let Some(scope) = self.scopes.get(&key) else { return format!("\0{name}") };
            if scope.globals.contains(name) {
                return format!("\0{name}");
            }
            match &scope.parent {
                Some(parent) if !scope.locals.contains(name) => key = parent.clone(),
                _ => return format!("{key}\0{name}"),
            }
        }
    }

    /// The scope of a function definition of the module (`_scope_of`).
    fn scope_of(&self, function: NodeId) -> &Scope {
        &self.scopes[&self.key_of[&function]]
    }

    /// Scope key of the user function a name resolves to, if any (`_lookup_func`).
    fn lookup_func(&self, name: &str) -> Option<String> {
        let mut key = Some(self.scope.clone());
        while let Some(current) = key {
            let Some(scope) = self.scopes.get(&current) else { break };
            if let Some(found) = scope.funcs.get(name) {
                return Some(found.clone());
            }
            key = scope.parent.clone();
        }
        let function = self.funcs.get(name)?;
        self.key_of.get(function).cloned()
    }

    /// Record that a cell of the analysis state has just grown (`_touch`).
    fn touch(&mut self, cell: &str) {
        self.tick += 1;
        self.stamps.insert(cell.to_string(), self.tick);
        self.changed = true;
    }

    fn read(&mut self, cell: &str) {
        if let Some(reads) = &mut self.reads {
            reads.insert(cell.to_string());
        }
    }

    // --- alias classes ---

    fn find(&mut self, name: &str) -> String {
        let mut root = name.to_string();
        while let Some(parent) = self.parent.get(&root)
            && *parent != root
        {
            root = parent.clone();
        }
        // Path compression
        let mut current = name.to_string();
        while let Some(parent) = self.parent.get(&current).cloned()
            && parent != current
        {
            self.parent.insert(current, root.clone());
            current = parent;
        }
        root
    }

    fn union(&mut self, a: &str, b: &str) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        self.parent.insert(rb.clone(), ra.clone());
        let moved = self.taint.remove(&rb).unwrap_or_default();
        self.taint.entry(ra.clone()).or_default().extend(moved);
        // A merge is a change even when neither class gains taint: a later add to
        // one alias now reaches the other. A statement that read either root has
        // to look again.
        self.touch(&ra);
        self.stamps.insert(rb, self.tick);
    }

    fn get_taint(&mut self, name: &str) -> Taint {
        if is_untaintable(name) {
            return Taint::new();
        }
        let root = self.find(name);
        self.read(&root);
        self.taint.get(&root).cloned().unwrap_or_default()
    }

    fn add_taint(&mut self, name: &str, taint: &Taint) {
        if is_untaintable(name) {
            return;
        }
        let root = self.find(name);
        let current = self.taint.entry(root.clone()).or_default();
        if taint.is_subset(current) {
            return;
        }
        current.extend(taint.iter().cloned());
        self.touch(&root);
    }

    // --- parameter tokens ---

    /// Turn a taint set into plain sids by expanding parameter tokens
    /// (`_resolve_tokens`). A function the module calls only where the analysis
    /// never reached expands to every sid, in the final rounds only.
    fn resolve_tokens(&mut self, taint: Taint) -> Taint {
        let mut out = Taint::new();
        let mut seen: HashSet<String> = HashSet::new();
        let mut work: Vec<String> = taint.into_iter().collect();
        while let Some(item) = work.pop() {
            if !item.starts_with(PARAM_PREFIX) {
                out.insert(item);
                continue;
            }
            if !seen.insert(item.clone()) {
                continue;
            }
            self.read(&item);
            if self.final_round && self.of_unreached_call(&item) {
                out.extend(self.all.iter().cloned());
                continue;
            }
            if let Some(bound) = self.param_in.get(&item) {
                work.extend(bound.iter().cloned());
            }
        }
        out
    }

    /// Whether a token belongs to a function the module calls only where the
    /// analysis never reached (`_of_unreached_call`).
    fn of_unreached_call(&self, token: &str) -> bool {
        let rest = &token[PARAM_PREFIX.len()..];
        let key = match rest.rfind('\0') {
            Some(index) => &rest[..index],
            None => "",
        };
        let Some(scope) = self.scopes.get(key) else { return false };
        !self.bound.contains(key)
            && is_function(self.ast.kind(scope.node))
            && self.called_names.contains(function_name(self.ast, scope.node))
    }

    // --- program order of the write sites ---

    /// Number every sid by the program order of its `__sec_write__` site
    /// (`_compute_site_order`), descending into a user function at its first call
    /// site; sites the walk never reaches are numbered last, in registration order.
    fn compute_site_order(&mut self) -> Result<(), Error> {
        fn walk_scope(
            this: &mut DependencyAnalyzer<'_>,
            key: &str,
            counter: &mut usize,
            visiting: &mut HashSet<String>,
            walked: &mut HashSet<String>,
        ) {
            let calls = this.scopes[key].calls.clone();
            for (name, call, _) in calls {
                if name == "__sec_write__" {
                    if let Some(sid) = sid_arg(this.ast, call)
                        && !this.order.contains_key(sid)
                    {
                        this.order.insert(sid.to_string(), *counter);
                        *counter += 1;
                    }
                } else if let Some(&function) = this.funcs.get(&name)
                    && !visiting.contains(&name)
                    && !walked.contains(&name)
                {
                    visiting.insert(name.clone());
                    let callee = this.key_of[&function].clone();
                    walk_scope(this, &callee, counter, visiting, walked);
                    visiting.remove(&name);
                    walked.insert(name);
                }
            }
        }
        let mut counter = 0;
        let mut visiting = HashSet::new();
        let mut walked = HashSet::new();
        walk_scope(self, "", &mut counter, &mut visiting, &mut walked);
        let roots: Vec<(String, NodeId)> =
            self.funcs.iter().map(|(name, &function)| (name.clone(), function)).collect();
        for (name, function) in roots {
            if !self.called_names.contains(&name) {
                let key = self.key_of[&function].clone();
                walk_scope(self, &key, &mut counter, &mut visiting, &mut walked);
            }
        }
        let unreached: Vec<&String> =
            self.all_list.iter().filter(|sid| !self.order.contains_key(*sid)).collect();
        if unreached.len() > 1 {
            // The Python step numbers these in the iteration order of a string set,
            // which follows the interpreter's hash seed
            return Err(Error::Unsupported(format!(
                "security: {} write sites the site walk never reaches",
                unreached.len()
            )));
        }
        for sid in unreached {
            self.order.insert(sid.clone(), counter);
            counter += 1;
        }
        Ok(())
    }

    // --- loop variables ---

    /// Give each loop its own taint cell for the variables only it uses
    /// (`_mark_private_loop_vars`).
    fn mark_private_loop_vars(&mut self) {
        let keys: Vec<String> = self.scopes.keys().cloned().collect();
        for key in keys {
            if !self.loop_scopes.contains(&key) {
                // No loop of its own: nothing to give a cell to
                continue;
            }
            let (uses, loops, parents) = self.collect_loop_names(&key);
            let mut binders: IndexMap<String, Vec<NodeId>> = IndexMap::new();
            for (&loop_id, names) in &loops {
                for name in names.keys() {
                    binders.entry(name.clone()).or_default().push(loop_id);
                }
            }
            let mut index = 0;
            for (name, own) in &binders {
                let count = uses.get(name).copied().unwrap_or(0);
                if !is_loop_var(name, count, own, &loops, &parents) {
                    continue;
                }
                for &loop_id in own {
                    index += 1;
                    self.loop_cells
                        .entry(loop_id)
                        .or_default()
                        .insert(name.clone(), format!("{key}\0loop{index}\0{name}"));
                }
            }
        }
    }

    /// Walk a scope once in EVALUATION order, gathering everything the loop
    /// variable rule needs (`_collect_loop_names`): the occurrences of each name the
    /// scope owns; per loop, `name -> (occurrences, stored first)` for the scope's
    /// names it mentions; and each loop's enclosing loop.
    fn collect_loop_names(
        &mut self,
        key: &str,
    ) -> (HashMap<String, usize>, LoopNames, HashMap<NodeId, Option<NodeId>>) {
        let mut walker = LoopNameWalk {
            scope_node: self.scopes[key].node,
            uses: HashMap::new(),
            loops: IndexMap::new(),
            parents: HashMap::new(),
        };
        let mut stack = Vec::new();
        walker.visit(self, self.scopes[key].node, &mut stack, &HashSet::new(), &HashSet::new());
        let locals = &self.scopes[key].locals;
        let written: LoopNames = walker
            .loops
            .into_iter()
            .map(|(loop_id, names)| {
                (loop_id, names.into_iter().filter(|(name, _)| locals.contains(name)).collect())
            })
            .collect();
        (walker.uses, written, walker.parents)
    }

    /// Name occurrences inside a nested function, as an enclosing scope counts them
    /// (`_foreign_uses`): those not owned by a function at or below it.
    fn foreign_uses(&mut self, function: NodeId) -> HashMap<String, usize> {
        if let Some(found) = self.foreign_memo.get(&function) {
            return found.clone();
        }
        let ast = self.ast;
        // A lambda has no scope of its own: it is looked up as the module's
        let key = self.key_of.get(&function).cloned().unwrap_or_default();
        let own: HashSet<String> =
            self.scopes.get(&key).map(|scope| scope.locals.clone()).unwrap_or_default();
        let mut counts: HashMap<String, usize> = HashMap::new();
        let add = |counts: &mut HashMap<String, usize>, name: &str, count: usize| {
            *counts.entry(name.to_string()).or_default() += count;
        };
        let mut stack = children(ast, function);
        while let Some(node) = stack.pop() {
            match ast.kind(node) {
                Kind::Name { id, .. } => {
                    add(&mut counts, id, 1);
                    continue;
                }
                Kind::Import { names } | Kind::ImportFrom { names, .. } => {
                    for &alias in names {
                        let Kind::Alias { name, asname } = ast.kind(alias) else { unreachable!() };
                        let spelled = asname.as_deref().unwrap_or(name);
                        add(&mut counts, spelled.split('.').next().unwrap_or(spelled), 1);
                    }
                    continue;
                }
                Kind::Global { names } | Kind::Nonlocal { names } => {
                    for name in names {
                        add(&mut counts, name, 1);
                    }
                    continue;
                }
                Kind::FunctionDef { .. } | Kind::AsyncFunctionDef { .. } | Kind::Lambda { .. } => {
                    for (name, count) in self.foreign_uses(node) {
                        add(&mut counts, &name, count);
                    }
                    continue;
                }
                Kind::ExceptHandler { name: Some(name), .. }
                | Kind::MatchAs { name: Some(name), .. }
                | Kind::MatchStar { name: Some(name) }
                | Kind::MatchMapping { rest: Some(name), .. } => add(&mut counts, name, 1),
                _ => {}
            }
            stack.extend(children(ast, node));
        }
        let found: HashMap<String, usize> =
            counts.into_iter().filter(|(name, _)| !own.contains(name)).collect();
        self.foreign_memo.insert(function, found.clone());
        found
    }

    // --- in_loop ---

    /// Mark sids whose `__sec_write__` runs more than once per bar (`_mark_in_loop`):
    /// a write in a loop body, and a write in a user function reached from a loop.
    fn mark_in_loop(&mut self) {
        let mut own_sids: HashMap<String, HashSet<String>> = HashMap::new();
        let mut callees: HashMap<String, HashSet<String>> = HashMap::new();
        let mut loop_callees: IndexSet<String> = IndexSet::new();
        let mut written_in_loop: Vec<String> = Vec::new();

        let mut scopes: Vec<(String, &Scope)> = vec![(String::new(), &self.scopes[""])];
        scopes.extend(
            self.funcs.iter().map(|(name, &function)| (name.clone(), self.scope_of(function))),
        );
        for (scope, own) in scopes {
            let sids = own_sids.entry(scope.clone()).or_default();
            let called = callees.entry(scope).or_default();
            for (name, node, in_loop) in &own.calls {
                if name == "__sec_write__" {
                    if let Some(sid) = sid_arg(self.ast, *node) {
                        sids.insert(sid.to_string());
                        if *in_loop {
                            written_in_loop.push(sid.to_string());
                        }
                    }
                } else if self.funcs.contains_key(name) {
                    called.insert(name.clone());
                    if *in_loop {
                        loop_callees.insert(name.clone());
                    }
                }
            }
        }

        self.in_loop.extend(written_in_loop);
        let mut reached: HashSet<String> = HashSet::new();
        let mut work: Vec<String> = loop_callees.into_iter().collect();
        while let Some(name) = work.pop() {
            if !reached.insert(name.clone()) {
                continue;
            }
            if let Some(next) = callees.get(&name) {
                work.extend(next.iter().cloned());
            }
        }
        for name in reached {
            if let Some(sids) = own_sids.get(&name) {
                self.in_loop.extend(sids.iter().cloned());
            }
        }
    }

    // --- expression taint ---

    fn expr_taint(&mut self, node: Option<NodeId>) -> Taint {
        let Some(node) = node else { return Taint::new() };
        let ast = self.ast;
        match ast.kind(node) {
            Kind::Constant { .. } => Taint::new(),
            Kind::Name { id, .. } => {
                let qualified = self.q(id, None);
                self.get_taint(&qualified)
            }
            Kind::Attribute { value, .. } => self.expr_taint(Some(*value)),
            Kind::Call { .. } => self.call_taint(node),
            Kind::NamedExpr { target, value } => {
                // Walrus assignments are injected by the lowering passes that run
                // before this one; they behave exactly like a plain assignment.
                let taint = self.expr_taint(Some(*value));
                let mut assigned = taint.clone();
                assigned.extend(self.ctrl.iter().cloned());
                self.assign(*target, assigned, Some(*value));
                taint
            }
            kind if is_union_expr(kind) => {
                let mut taint = Taint::new();
                for child in children(ast, node) {
                    if ast.kind(child).class().category == pyne_ast::Category::Expr {
                        taint.extend(self.expr_taint(Some(child)));
                    }
                }
                taint
            }
            // Unmodelled expression form: the scope falls back instead of the value
            // carrying every sid around through name taint.
            _ => {
                self.mark_fallback();
                Taint::new()
            }
        }
    }

    fn call_taint(&mut self, node: NodeId) -> Taint {
        let ast = self.ast;
        let Kind::Call { func, args, keywords } = ast.kind(node) else { unreachable!() };
        let func = *func;
        if let Kind::Name { id, .. } = ast.kind(func) {
            match id.as_str() {
                "__sec_read__" => {
                    return match sid_arg(ast, node) {
                        Some(sid) => Taint::from([sid.to_string()]),
                        None => self.all.clone(),
                    };
                }
                "__sec_write__" => {
                    let sid = sid_arg(ast, node);
                    let value = args.get(1).copied();
                    let mut taint = self.expr_taint(value);
                    taint.extend(self.ctrl.iter().cloned());
                    let mut taint = self.resolve_tokens(taint);
                    if self.fallback_scopes.contains(&self.scope) {
                        taint.extend(self.all.iter().cloned());
                    }
                    if let Some(sid) = sid {
                        let target = self.depends.entry(sid.to_string()).or_default();
                        let mut grew = false;
                        for item in taint {
                            if item != sid && target.insert(item) {
                                grew = true;
                            }
                        }
                        if grew {
                            self.changed = true;
                        }
                    }
                    return Taint::new();
                }
                "__sec_signal__" | "__sec_wait__" => return Taint::new(),
                _ => {}
            }
        }

        let mut taint = match ast.kind(func) {
            Kind::Name { .. } => Taint::new(),
            _ => self.expr_taint(Some(func)),
        };
        let pos_taints: Vec<Taint> = args.iter().map(|&arg| self.expr_taint(Some(arg))).collect();
        let kw_taints: Vec<(Option<String>, Taint)> = keywords
            .iter()
            .map(|&keyword| {
                let Kind::Keyword { arg, value } = ast.kind(keyword) else { unreachable!() };
                (arg.clone(), self.expr_taint(Some(*value)))
            })
            .collect();
        // A ``*iterable`` spread has an unknown length, so from its position on no
        // argument has a known parameter of its own.
        let star_from = args.iter().position(|&arg| matches!(ast.kind(arg), Kind::Starred { .. }));
        let callee = match ast.kind(func) {
            Kind::Name { id, .. } => self.lookup_func(id),
            _ => None,
        };
        match callee {
            Some(key) => {
                let result = self.call_user_func(&key, node, &pos_taints, &kw_taints, star_from);
                taint.extend(result);
            }
            None => {
                // Unknown callee: every argument is assumed to reach the result.
                for arg_taint in pos_taints {
                    taint.extend(arg_taint);
                }
                for (_, arg_taint) in kw_taints {
                    taint.extend(arg_taint);
                }
            }
        }

        // Mutation: a collection API writes into the id passed as its first
        // argument (``array.push(id, v)``), and a method call writes into the
        // object it runs on (``id.push(v)``).
        if let Kind::Attribute { value: receiver, attr, .. } = ast.kind(func) {
            match collection_namespace(ast, func) {
                Some(namespace) => {
                    if !args.is_empty() {
                        self.mutate_collection(namespace, attr, node, &taint);
                    }
                }
                None => {
                    let mut mutation = taint.clone();
                    mutation.extend(self.ctrl.iter().cloned());
                    if !mutation.is_empty()
                        && let Some(root) = root_name(ast, *receiver)
                    {
                        let qualified = self.q(root, None);
                        self.add_taint(&qualified, &mutation);
                    }
                }
            }
        }
        taint
    }

    /// Write a collection call's effect back into its first argument
    /// (`_mutate_collection`): a pure reader writes nothing; anything else writes
    /// what the whole call touches plus the control taint.
    fn mutate_collection(&mut self, namespace: &str, function: &str, node: NodeId, taint: &Taint) {
        if is_collection_reader(namespace, function) {
            return;
        }
        let ast = self.ast;
        let Kind::Call { args, .. } = ast.kind(node) else { unreachable!() };
        let Some(root) = root_name(ast, args[0]) else { return };
        let mut back = taint.clone();
        back.extend(self.ctrl.iter().cloned());
        if !back.is_empty() {
            let qualified = self.q(root, None);
            self.add_taint(&qualified, &back);
        }
    }

    /// The taint of one call of a modelled user function (`_call_user_func`): the
    /// callee's return taint with its parameter tokens replaced by this call's
    /// argument taints, and what the callee added to a parameter written back into
    /// the caller's argument variable.
    fn call_user_func(
        &mut self,
        key: &str,
        node: NodeId,
        pos_taints: &[Taint],
        kw_taints: &[(Option<String>, Taint)],
        star_from: Option<usize>,
    ) -> Taint {
        let ast = self.ast;
        let function = self.scopes[key].node;
        let arg_expr = arg_exprs(ast, function, node, star_from);
        let mut bound: IndexMap<String, Taint> = IndexMap::new();
        // Taint whose target parameter is not known at compile time
        let mut unknown = Taint::new();
        let Kind::Arguments { posonlyargs, args, kwonlyargs, .. } =
            ast.kind(function_args(ast, function))
        else {
            unreachable!()
        };
        let positional: Vec<&str> =
            posonlyargs.iter().chain(args).map(|&arg| arg_name(ast, arg)).collect();
        for (index, arg_taint) in pos_taints.iter().enumerate() {
            if star_from.is_some_and(|star| index >= star) || index >= positional.len() {
                unknown.extend(arg_taint.iter().cloned());
            } else {
                bound
                    .entry(positional[index].to_string())
                    .or_default()
                    .extend(arg_taint.iter().cloned());
            }
        }
        let by_name: HashSet<&str> = positional
            .iter()
            .copied()
            .chain(kwonlyargs.iter().map(|&arg| arg_name(ast, arg)))
            .collect();
        for (name, arg_taint) in kw_taints {
            match name {
                Some(name) if by_name.contains(name.as_str()) => {
                    bound.entry(name.clone()).or_default().extend(arg_taint.iter().cloned());
                }
                _ => unknown.extend(arg_taint.iter().cloned()),
            }
        }
        // A spread or a leftover argument can fill any parameter, so it binds all
        // of them (the catch-alls included).
        if !unknown.is_empty() {
            for param in params(ast, function) {
                bound
                    .entry(arg_name(ast, param).to_string())
                    .or_default()
                    .extend(unknown.iter().cloned());
            }
        }

        let mut token_map: HashMap<String, Taint> = bound
            .into_iter()
            .map(|(name, arg_taint)| (token(&format!("{key}\0{name}")), arg_taint))
            .collect();
        token_map.insert(ctrl_token(key), self.ctrl.clone());
        self.bound.insert(key.to_string());
        for (token, arg_taint) in &token_map {
            let current = self.param_in.entry(token.clone()).or_default();
            if !arg_taint.is_subset(current) {
                current.extend(arg_taint.iter().cloned());
                self.touch(token);
            }
        }

        self.read(&format!("{RET_CELL}{key}"));
        self.read(&format!("{OUT_CELL}{key}"));
        let mut result = subst(&self.ret.get(key).cloned().unwrap_or_default(), &token_map);
        if self.fallback_scopes.contains(key) {
            // An unmodelled construct in the callee: its return can carry anything,
            // and so can what it writes into its arguments.
            result.extend(self.all.iter().cloned());
            let all = self.all.clone();
            for &expr in arg_expr.values() {
                if let Some(root) = root_name(ast, expr) {
                    let qualified = self.q(root, None);
                    self.add_taint(&qualified, &all);
                }
            }
            return result;
        }

        let outs = self.param_out.get(key).cloned().unwrap_or_default();
        for (name, &expr) in &arg_expr {
            let out = outs.get(&token(&format!("{key}\0{name}"))).cloned().unwrap_or_default();
            let back = subst(&out, &token_map);
            if back.is_empty() {
                continue;
            }
            if let Some(root) = root_name(ast, expr) {
                let qualified = self.q(root, None);
                self.add_taint(&qualified, &back);
            }
        }
        result
    }

    /// Root names of the collections an expression may evaluate to an alias of
    /// (`_alias_roots`): the root of a place chain, or the roots of the arguments a
    /// user function's summary says it returns.
    fn alias_roots(&mut self, expr: NodeId) -> Vec<String> {
        let ast = self.ast;
        match ast.kind(expr) {
            Kind::Name { .. } | Kind::Attribute { .. } | Kind::Subscript { .. } => {
                return match root_name(ast, expr) {
                    Some(root) if root != "lib" => vec![root.to_string()],
                    _ => Vec::new(),
                };
            }
            Kind::Call { .. } => {}
            _ => return Vec::new(),
        }
        let Kind::Call { func, args, .. } = ast.kind(expr) else { unreachable!() };
        let Kind::Name { id, .. } = ast.kind(*func) else { return Vec::new() };
        let Some(key) = self.lookup_func(id) else { return Vec::new() };
        self.read(&format!("{ALIAS_CELL}{key}"));
        let Some(params) = self.ret_alias.get(&key).cloned() else { return Vec::new() };
        if params.is_empty() {
            return Vec::new();
        }
        let function = self.scopes[&key].node;
        let star_from = args.iter().position(|&arg| matches!(ast.kind(arg), Kind::Starred { .. }));
        let exprs = arg_exprs(ast, function, expr, star_from);
        let mut roots = Vec::new();
        for name in &params {
            if let Some(&arg) = exprs.get(name) {
                roots.extend(self.alias_roots(arg));
            }
        }
        roots
    }

    /// Note which parameters the returned value shares an alias class with
    /// (`_record_ret_alias`).
    fn record_ret_alias(&mut self, value: Option<NodeId>) {
        let ast = self.ast;
        let key = self.scope.clone();
        let function = self.scopes[&key].node;
        let Some(value) = value else { return };
        if !is_function(ast.kind(function)) {
            return;
        }
        self.ret_alias.entry(key.clone()).or_default();
        for root in self.alias_roots(value) {
            let qualified = self.q(&root, None);
            let class_root = self.find(&qualified);
            self.read(&class_root);
            for param in params(ast, function) {
                let name = arg_name(ast, param);
                if self.ret_alias[&key].contains(name) {
                    continue;
                }
                let param_root = self.find(&format!("{key}\0{name}"));
                self.read(&param_root);
                if param_root == class_root {
                    self.ret_alias.get_mut(&key).expect("set above").insert(name.to_string());
                    self.touch(&format!("{ALIAS_CELL}{key}"));
                }
            }
        }
    }

    // --- assignment ---

    fn assign(&mut self, target: NodeId, mut taint: Taint, value: Option<NodeId>) {
        let ast = self.ast;
        if self.fallback_scopes.contains(&self.scope) {
            // The assigned value may come from an unmodelled construct whose taint
            // the analysis could not see.
            taint.extend(self.all.iter().cloned());
        }
        match ast.kind(target) {
            Kind::Name { id, .. } => {
                let qualified = self.q(id, None);
                self.add_taint(&qualified, &taint);
                if let Some(value) = value {
                    for root in self.alias_roots(value) {
                        let target_cell = self.q(id, None);
                        let root_cell = self.q(&root, None);
                        self.union(&target_cell, &root_cell);
                    }
                }
            }
            Kind::Tuple { elts, .. } | Kind::List { elts, .. } => {
                for &elt in elts {
                    self.assign(elt, taint.clone(), None);
                }
            }
            Kind::Starred { value, .. } => self.assign(*value, taint, None),
            _ => {
                if let Some(root) = root_name(ast, target) {
                    let qualified = self.q(root, None);
                    self.add_taint(&qualified, &taint);
                }
            }
        }
    }

    // --- statements ---

    /// Visit a statement list; answers the control taint after it (`_visit_body`).
    /// While the reads are recorded, a statement of a final round none of whose
    /// inputs changed since its last visit is skipped.
    fn visit_body(&mut self, body: &[NodeId], mut ctrl: Taint, top: bool) -> Taint {
        if self.reads.is_none() {
            for &stmt in body {
                ctrl = self.visit_stmt(stmt, ctrl);
            }
            return ctrl;
        }
        let skip = self.final_round;
        for &stmt in body {
            if skip
                && let Some((start, ctrl_in, ctrl_out, ctrl_last, cells)) = self.memo.get(&stmt)
                && *ctrl_in == ctrl
                && cells.iter().all(|cell| self.stamps.get(cell).copied().unwrap_or(0) <= *start)
            {
                // The enclosing statement's visit depends on these too
                let (ctrl_out, ctrl_last, cells) =
                    (ctrl_out.clone(), ctrl_last.clone(), cells.clone());
                self.reads.as_mut().expect("recording").extend(cells);
                // A ``match`` guard after a case body reads ``ctrl`` as the
                // body's last statement left it
                self.ctrl = ctrl_last;
                ctrl = ctrl_out;
                continue;
            }
            let outer = self.reads.replace(HashSet::new()).expect("recording");
            let start = self.tick;
            let escapes = self.escapes;
            let ctrl_out = self.visit_stmt(stmt, ctrl.clone());
            let cells = self.reads.take().expect("recording");
            let mut outer = outer;
            outer.extend(cells.iter().cloned());
            self.reads = Some(outer);
            // A statement holding a ``return`` / ``break`` / ``continue`` also
            // feeds the escape accumulators of its block, which the memo does not
            // replay, so such a statement is never skipped.
            if top || self.escapes == escapes {
                self.memo.insert(stmt, (start, ctrl, ctrl_out.clone(), self.ctrl.clone(), cells));
            }
            ctrl = ctrl_out;
        }
        ctrl
    }

    /// Visit a scope's statements, skipping those a visit cannot change
    /// (`_visit_scope_body`).
    fn visit_scope_body(&mut self, key: &str, ctrl: Taint) {
        self.reads = Some(HashSet::new());
        let body = self.scopes[key].body.clone();
        self.visit_body(&body, ctrl, true);
        self.reads = None;
    }

    /// Visit a nested block and collect the escapes taken inside it (`_visit_block`):
    /// answers the taint of the `return` escapes and of the `break` / `continue`
    /// escapes.
    fn visit_block(&mut self, body: &[NodeId], ctrl: Taint) -> (Taint, Taint) {
        let outer_returned = std::mem::take(&mut self.escaped);
        let outer_broke = std::mem::take(&mut self.broke);
        self.visit_body(body, ctrl, false);
        let returned = std::mem::take(&mut self.escaped);
        let broke = std::mem::take(&mut self.broke);
        self.escaped = &outer_returned | &returned;
        self.broke = &outer_broke | &broke;
        (returned, broke)
    }

    /// Visit a loop's blocks; answers the escape taint its successors inherit
    /// (`_visit_loop`). The loop swallows its own `break` / `continue` escapes.
    fn visit_loop(&mut self, body: &[NodeId], orelse: &[NodeId], inner: &Taint) -> Taint {
        let outer_broke = std::mem::take(&mut self.broke);
        let (mut returned, mut broke) = self.visit_block(body, inner.clone());
        if !broke.is_subset(inner) {
            // A tainted escape also decides whether the body runs AGAIN, so
            // everything in it is conditional on that taint. One extra pass over
            // the body under the widened control taint closes the back edge.
            let widened = inner | &broke;
            (returned, broke) = self.visit_block(body, widened);
        }
        let (more_returned, more_broke) = self.visit_block(orelse, inner.clone());
        broke.extend(more_broke);
        self.broke = outer_broke;
        let mut out = returned;
        out.extend(more_returned);
        out.extend(broke);
        out
    }

    /// Activate the private taint cells of a loop's own variables, if it has any
    /// (`_enter_loop_cells`); answers the overrides that were active before.
    fn enter_loop_cells(&mut self, stmt: NodeId) -> Vec<(String, Option<String>)> {
        let mut saved = Vec::new();
        if let Some(cells) = self.loop_cells.get(&stmt) {
            for (name, cell) in cells {
                saved.push((name.clone(), self.name_cells.insert(name.clone(), cell.clone())));
            }
        }
        saved
    }

    fn leave_loop_cells(&mut self, saved: Vec<(String, Option<String>)>) {
        for (name, previous) in saved {
            match previous {
                None => self.name_cells.remove(&name),
                Some(previous) => self.name_cells.insert(name, previous),
            };
        }
    }

    fn visit_stmt(&mut self, stmt: NodeId, ctrl: Taint) -> Taint {
        let ast = self.ast;
        self.ctrl = ctrl.clone();
        match ast.kind(stmt) {
            Kind::FunctionDef { .. } | Kind::AsyncFunctionDef { .. } => ctrl,
            Kind::Assign { targets, value, .. } => {
                let mut taint = self.expr_taint(Some(*value));
                taint.extend(ctrl.iter().cloned());
                for &target in targets {
                    self.assign(target, taint.clone(), Some(*value));
                }
                ctrl
            }
            Kind::AnnAssign { target, value, .. } => {
                if let Some(value) = value {
                    let mut taint = self.expr_taint(Some(*value));
                    taint.extend(ctrl.iter().cloned());
                    self.assign(*target, taint, Some(*value));
                }
                ctrl
            }
            Kind::AugAssign { target, value, .. } => {
                let mut taint = self.expr_taint(Some(*value));
                taint.extend(self.expr_taint(Some(*target)));
                taint.extend(ctrl.iter().cloned());
                self.assign(*target, taint, None);
                ctrl
            }
            Kind::Expr { value } => {
                self.expr_taint(Some(*value));
                ctrl
            }
            Kind::If { test, body, orelse } => {
                let inner = &ctrl | &self.expr_taint(Some(*test));
                let (returned, broke) = self.visit_block(body, inner.clone());
                let (more_returned, more_broke) = self.visit_block(orelse, inner);
                let mut out = ctrl;
                out.extend(returned);
                out.extend(broke);
                out.extend(more_returned);
                out.extend(more_broke);
                out
            }
            Kind::For { target, iter, body, orelse, .. }
            | Kind::AsyncFor { target, iter, body, orelse, .. } => {
                let inner = &ctrl | &self.expr_taint(Some(*iter));
                let saved = self.enter_loop_cells(stmt);
                self.assign(*target, inner.clone(), None);
                let escaped = self.visit_loop(body, orelse, &inner);
                self.leave_loop_cells(saved);
                &ctrl | &escaped
            }
            Kind::While { test, body, orelse } => {
                let inner = &ctrl | &self.expr_taint(Some(*test));
                let saved = self.enter_loop_cells(stmt);
                let escaped = self.visit_loop(body, orelse, &inner);
                self.leave_loop_cells(saved);
                &ctrl | &escaped
            }
            Kind::Match { subject, cases } => {
                let inner = &ctrl | &self.expr_taint(Some(*subject));
                let mut escaped = Taint::new();
                for &case in cases {
                    let Kind::MatchCase { guard, body, .. } = ast.kind(case) else {
                        unreachable!()
                    };
                    let mut case_ctrl = inner.clone();
                    if let Some(guard) = guard {
                        case_ctrl = &inner | &self.expr_taint(Some(*guard));
                    }
                    let (returned, broke) = self.visit_block(body, case_ctrl);
                    escaped.extend(returned);
                    escaped.extend(broke);
                }
                &ctrl | &escaped
            }
            Kind::Return { value } => {
                let mut taint = self.expr_taint(*value);
                taint.extend(ctrl.iter().cloned());
                self.record_return(&taint);
                self.record_ret_alias(*value);
                self.escaped.extend(taint.iter().cloned());
                self.escapes += 1;
                &ctrl | &taint
            }
            Kind::Break | Kind::Continue => {
                // The decision to leave the loop is itself tainted, so everything
                // that runs after the loop is conditional on that taint.
                self.broke.extend(ctrl.iter().cloned());
                self.escapes += 1;
                ctrl
            }
            _ => ctrl,
        }
    }

    /// Mark the current scope as containing an unmodelled construct (`_mark_fallback`).
    fn mark_fallback(&mut self) {
        if self.fallback_scopes.insert(self.scope.clone()) {
            self.changed = true;
        }
    }

    fn record_return(&mut self, taint: &Taint) {
        let current = self.ret.entry(self.scope.clone()).or_default();
        if taint.is_empty() || taint.is_subset(current) {
            return;
        }
        current.extend(taint.iter().cloned());
        let cell = format!("{RET_CELL}{}", self.scope);
        self.touch(&cell);
    }

    // --- per-scope pre/post steps ---

    /// Give every parameter its own symbolic token as a base taint (`_seed_params`).
    fn seed_params(&mut self) {
        let ast = self.ast;
        let scopes: Vec<(String, NodeId)> =
            self.scopes.iter().map(|(key, scope)| (key.clone(), scope.node)).collect();
        for (key, node) in scopes {
            if !is_function(ast.kind(node)) {
                continue;
            }
            for param in params(ast, node) {
                let qualified = format!("{key}\0{}", arg_name(ast, param));
                self.add_taint(&qualified, &Taint::from([token(&qualified)]));
            }
        }
    }

    /// Fold parameter default expressions into the parameters' taint
    /// (`_visit_defaults`); a default is evaluated in the ENCLOSING scope.
    fn visit_defaults(&mut self, key: &str) {
        let ast = self.ast;
        let function = self.scopes[key].node;
        let parent = self.scopes[key].parent.clone().unwrap_or_default();
        let outer = std::mem::replace(&mut self.scope, parent);
        let ctrl = std::mem::take(&mut self.ctrl);
        let Kind::Arguments { posonlyargs, args, kwonlyargs, kw_defaults, defaults, .. } =
            ast.kind(function_args(ast, function))
        else {
            unreachable!()
        };
        let positional: Vec<NodeId> = posonlyargs.iter().chain(args).copied().collect();
        let mut pairs: Vec<(NodeId, Option<NodeId>)> = positional
            [positional.len().saturating_sub(defaults.len())..]
            .iter()
            .copied()
            .zip(defaults.iter().map(|&default| Some(default)))
            .collect();
        pairs.extend(kwonlyargs.iter().copied().zip(kw_defaults.iter().copied()));
        let mut taints = Vec::new();
        for (param, default) in pairs {
            if default.is_some() {
                taints.push((param, self.expr_taint(default)));
            }
        }
        self.scope = outer;
        self.ctrl = ctrl;
        for (param, taint) in taints {
            let qualified = format!("{key}\0{}", arg_name(ast, param));
            self.add_taint(&qualified, &taint);
        }
    }

    /// Snapshot every parameter's taint at the end of the callee's body
    /// (`_record_param_out`).
    fn record_param_out(&mut self, key: &str) {
        let ast = self.ast;
        let function = self.scopes[key].node;
        self.param_out.entry(key.to_string()).or_default();
        for param in params(ast, function) {
            let qualified = format!("{key}\0{}", arg_name(ast, param));
            let taint = self.get_taint(&qualified);
            let current = self
                .param_out
                .get_mut(key)
                .expect("set above")
                .entry(token(&qualified))
                .or_default();
            if !taint.is_subset(current) {
                current.extend(taint);
                self.touch(&format!("{OUT_CELL}{key}"));
            }
        }
    }

    // --- driver ---

    /// Keep the first round's statement memo only where a final round would have
    /// visited alike (`_settle_first_round`).
    fn settle_first_round(&mut self, fallback: &HashSet<String>) {
        if self.fallback_scopes != *fallback {
            self.memo.clear();
            return;
        }
        let unreached: Vec<NodeId> = self
            .memo
            .iter()
            .filter(|(_, entry)| {
                entry
                    .4
                    .iter()
                    .any(|cell| cell.starts_with(PARAM_PREFIX) && self.of_unreached_call(cell))
            })
            .map(|(&stmt, _)| stmt)
            .collect();
        for stmt in unreached {
            self.memo.remove(&stmt);
        }
    }

    /// Whether every write already depends on every other sid (`_saturated`).
    fn saturated(&self) -> bool {
        self.depends
            .iter()
            .all(|(sid, deps)| self.all.iter().all(|other| other == sid || deps.contains(other)))
    }

    pub(crate) fn run(&mut self) -> Result<(), Error> {
        let ast = self.ast;
        for node in pyne_ast::walk::walk_statements(ast, self.module) {
            if is_function(ast.kind(node)) {
                self.funcs.insert(function_name(ast, node).to_string(), node);
            }
        }
        self.build_scopes();
        if self.all.len() < 2 {
            // ``depends`` and ``late_reads`` only ever name ANOTHER context, so a
            // lone one has neither, whatever the taint says. ``in_loop`` needs no
            // taint at all.
            self.mark_in_loop();
            return Ok(());
        }
        // The scopes' calls together are every call of the module
        self.called_names = self
            .scopes
            .values()
            .flat_map(|scope| scope.calls.iter().map(|(name, _, _)| name))
            .filter(|name| self.funcs.contains_key(*name))
            .cloned()
            .collect();
        self.mark_private_loop_vars();
        self.mark_in_loop();
        self.compute_site_order()?;

        let factories = FactoryFields::new(ast, self.module);
        let Kind::Module { body, .. } = ast.kind(self.module) else { unreachable!() };
        let class_names: HashSet<String> = body
            .iter()
            .filter_map(|&node| match ast.kind(node) {
                Kind::ClassDef { name, .. } => Some(name.clone()),
                _ => None,
            })
            .collect();
        let mut record_nodes: HashSet<NodeId> = HashSet::new();
        // Only a module-level class body runs at import time; one defined inside a
        // function runs with the function and stays unmodelled.
        for &node in body {
            if matches!(ast.kind(node), Kind::ClassDef { .. })
                && is_plain_record(ast, node, &factories, &class_names)
            {
                record_nodes.extend(walk(ast, node));
            }
        }
        for key in self.scopes.keys() {
            let Some(unmodelled) = self.unmodelled.get(key) else { continue };
            if unmodelled.iter().any(|sub| !record_nodes.contains(sub)) {
                self.fallback_scopes.insert(key.clone());
            }
        }

        self.seed_params();
        let ordered: Vec<String> = self.scopes.keys().cloned().collect();
        let fallback = self.fallback_scopes.clone();
        for _ in 0..MAX_FIXPOINT_ROUNDS {
            self.changed = false;
            for key in &ordered {
                self.scope = key.clone();
                self.escaped = Taint::new();
                self.broke = Taint::new();
                self.name_cells = HashMap::new();
                let mut ctrl = Taint::new();
                let function = self.scopes[key].node;
                let nested = self.scopes[key].parent.is_some();
                if nested {
                    self.visit_defaults(key);
                    if self.called_names.contains(function_name(ast, function)) {
                        // Everything in a called function runs under the conditions
                        // of its call sites
                        ctrl.insert(ctrl_token(key));
                    }
                }
                self.visit_scope_body(key, ctrl);
                if nested {
                    self.record_param_out(key);
                }
            }
            if self.final_round && !self.changed {
                break;
            }
            if self.saturated() {
                break;
            }
            if !self.final_round {
                self.settle_first_round(&fallback);
            }
            // After the first round ``bound`` holds every call site the analysis
            // reaches: from here a token still without a call site belongs to an
            // unreached call.
            self.final_round = true;
        }

        // Only a producer whose site PRECEDES this write can be waited for; a
        // later-sited one would deadlock the historical warmup batch. The dropped
        // edges are still reads of this write, kept apart in ``late_reads``.
        let order = &self.order;
        for (sid, deps) in self.depends.iter_mut() {
            let own = order.get(sid).copied().unwrap_or(0);
            let valid: Taint =
                deps.iter().filter(|&dep| dep != sid && self.all.contains(dep)).cloned().collect();
            let kept: Taint = valid
                .iter()
                .filter(|&dep| order.get(dep).copied().unwrap_or(own) < own)
                .cloned()
                .collect();
            let late: Taint = valid.difference(&kept).cloned().collect();
            *deps = kept;
            if !late.is_empty() {
                self.late_reads.insert(sid.clone(), late);
            }
        }
        Ok(())
    }
}

/// Each parameter of a function mapped to the argument expression a call passes
/// for it, for the arguments whose parameter is known at compile time
/// (`_arg_exprs`).
fn arg_exprs(
    ast: &Ast,
    function: NodeId,
    call: NodeId,
    star_from: Option<usize>,
) -> IndexMap<String, NodeId> {
    let Kind::Arguments { posonlyargs, args, kwonlyargs, .. } =
        ast.kind(function_args(ast, function))
    else {
        unreachable!()
    };
    let positional: Vec<&str> =
        posonlyargs.iter().chain(args).map(|&arg| arg_name(ast, arg)).collect();
    let by_name: HashSet<&str> = positional
        .iter()
        .copied()
        .chain(kwonlyargs.iter().map(|&arg| arg_name(ast, arg)))
        .collect();
    let Kind::Call { args: call_args, keywords, .. } = ast.kind(call) else { unreachable!() };
    let mut out = IndexMap::new();
    for (index, &arg) in call_args.iter().enumerate() {
        if star_from.is_some_and(|star| index >= star) {
            break;
        }
        if index < positional.len() {
            out.insert(positional[index].to_string(), arg);
        }
    }
    for &keyword in keywords {
        let Kind::Keyword { arg: Some(name), value } = ast.kind(keyword) else { continue };
        if by_name.contains(name.as_str()) {
            out.insert(name.clone(), *value);
        }
    }
    out
}

/// Whether a name lives only inside the loops that bind it (`_is_loop_var`).
fn is_loop_var(
    name: &str,
    uses: usize,
    binders: &[NodeId],
    loops: &LoopNames,
    parents: &HashMap<NodeId, Option<NodeId>>,
) -> bool {
    let binder_ids: HashSet<NodeId> = binders.iter().copied().collect();
    let mut found = 0;
    for loop_id in binders {
        // A binder nested in another one would carry the inner loop's value over
        // into the outer loop's body.
        let mut parent = parents[loop_id];
        while let Some(current) = parent {
            if binder_ids.contains(&current) {
                return false;
            }
            parent = parents[&current];
        }
        let (count, first_store) = loops[loop_id][name];
        // The loop must establish the name before reading it
        if !first_store {
            return false;
        }
        found += count;
    }
    found == uses
}

/// The walk of [`DependencyAnalyzer::collect_loop_names`], in evaluation order.
struct LoopNameWalk {
    scope_node: NodeId,
    uses: HashMap<String, usize>,
    loops: IndexMap<NodeId, IndexMap<String, (usize, bool)>>,
    parents: HashMap<NodeId, Option<NodeId>>,
}

impl LoopNameWalk {
    fn record(
        &mut self,
        name: &str,
        store: bool,
        stack: &[NodeId],
        cond: &HashSet<NodeId>,
        after: &HashSet<NodeId>,
    ) {
        *self.uses.entry(name.to_string()).or_default() += 1;
        for loop_id in stack {
            let seen = self.loops.get_mut(loop_id).expect("an entered loop");
            // A store counts as establishing the name only where every pass
            // through the loop body executes it.
            let (count, mut first_store) =
                seen.get(name).copied().unwrap_or((0, store && !cond.contains(loop_id)));
            if after.contains(loop_id) && !store {
                // A read in the loop's ``else`` clause also runs after zero
                // iterations, when it sees the value from before the loop.
                first_store = false;
            }
            seen.insert(name.to_string(), (count + 1, first_store));
        }
    }

    /// An occurrence no loop owns: the name can only be disqualified.
    fn foreign(&mut self, name: &str, count: usize) {
        *self.uses.entry(name.to_string()).or_default() += count;
    }

    fn target(
        &mut self,
        analyzer: &mut DependencyAnalyzer<'_>,
        node: NodeId,
        stack: &mut Vec<NodeId>,
        cond: &HashSet<NodeId>,
        after: &HashSet<NodeId>,
    ) {
        let ast = analyzer.ast;
        match ast.kind(node) {
            Kind::Name { id, .. } => self.record(id, true, stack, cond, after),
            Kind::Tuple { elts, .. } | Kind::List { elts, .. } => {
                for &elt in elts {
                    self.target(analyzer, elt, stack, cond, after);
                }
            }
            Kind::Starred { value, .. } => self.target(analyzer, *value, stack, cond, after),
            _ => self.visit(analyzer, node, stack, cond, after),
        }
    }

    fn visit(
        &mut self,
        analyzer: &mut DependencyAnalyzer<'_>,
        node: NodeId,
        stack: &mut Vec<NodeId>,
        cond: &HashSet<NodeId>,
        after: &HashSet<NodeId>,
    ) {
        let ast = analyzer.ast;
        let kind = ast.kind(node);
        match kind {
            Kind::Constant { .. } => return,
            Kind::Name { id, ctx } => {
                if *ctx == ExprContext::Load {
                    self.record(id, false, stack, cond, after);
                } else {
                    self.foreign(id, 1);
                }
                return;
            }
            Kind::Import { names } | Kind::ImportFrom { names, .. } => {
                for &alias in names {
                    let Kind::Alias { name, asname } = ast.kind(alias) else { unreachable!() };
                    let spelled = asname.as_deref().unwrap_or(name);
                    self.foreign(spelled.split('.').next().unwrap_or(spelled), 1);
                }
                return;
            }
            Kind::Global { names } | Kind::Nonlocal { names } => {
                for name in names {
                    self.foreign(name, 1);
                }
                return;
            }
            // ``except ... as name``, a match capture
            Kind::ExceptHandler { name, .. }
            | Kind::MatchAs { name, .. }
            | Kind::MatchStar { name } => {
                if let Some(name) = name {
                    self.foreign(name, 1);
                }
            }
            Kind::MatchMapping { rest: Some(rest), .. } => self.foreign(rest, 1),
            Kind::FunctionDef { .. } | Kind::AsyncFunctionDef { .. } | Kind::Lambda { .. } => {
                if node != self.scope_node {
                    // Everything below a nested function is an occurrence outside
                    // every loop of this scope
                    for (name, count) in analyzer.foreign_uses(node) {
                        self.foreign(&name, count);
                    }
                    return;
                }
            }
            Kind::Assign { targets, value, .. } => {
                self.visit(analyzer, *value, stack, cond, after);
                for &target in targets {
                    self.target(analyzer, target, stack, cond, after);
                }
                return;
            }
            Kind::AnnAssign { target, annotation, value, .. } => {
                // ``x: T = v`` evaluates ``v`` before binding ``x``; a bare
                // ``x: T`` binds nothing, so its name only counts as a use.
                if let Some(value) = value {
                    self.visit(analyzer, *value, stack, cond, after);
                }
                self.visit(analyzer, *annotation, stack, cond, after);
                match ast.kind(*target) {
                    Kind::Name { id, .. } if value.is_none() => {
                        self.record(id, false, stack, cond, after);
                    }
                    _ => self.target(analyzer, *target, stack, cond, after),
                }
                return;
            }
            Kind::AugAssign { target, value, .. } => {
                // ``acc += x`` reads the target before it stores into it.
                match ast.kind(*target) {
                    Kind::Name { id, .. } => self.record(id, false, stack, cond, after),
                    _ => self.visit(analyzer, *target, stack, cond, after),
                }
                self.visit(analyzer, *value, stack, cond, after);
                return;
            }
            _ => {}
        }
        if is_conditional_node(kind) {
            // Code below may be skipped on some pass through every loop already
            // entered, so nothing it stores establishes a name there.
            let mut cond: HashSet<NodeId> = cond.clone();
            cond.extend(stack.iter().copied());
            if is_loop(kind) {
                self.parents.insert(node, stack.last().copied());
                self.loops.insert(node, IndexMap::new());
                let mut inner_stack = stack.clone();
                inner_stack.push(node);
                // The ``else`` clause also runs after zero iterations, when nothing
                // in the body has executed.
                let mut else_cond = cond.clone();
                else_cond.insert(node);
                let (body, orelse) = match kind {
                    Kind::For { target, iter, body, orelse, .. }
                    | Kind::AsyncFor { target, iter, body, orelse, .. } => {
                        self.visit(analyzer, *iter, &mut inner_stack, &cond, after);
                        self.target(analyzer, *target, &mut inner_stack, &cond, after);
                        (body, orelse)
                    }
                    Kind::While { test, body, orelse } => {
                        self.visit(analyzer, *test, &mut inner_stack, &cond, after);
                        (body, orelse)
                    }
                    _ => unreachable!(),
                };
                for &child in body {
                    self.visit(analyzer, child, &mut inner_stack, &cond, after);
                }
                let mut else_after = after.clone();
                else_after.insert(node);
                for &child in orelse {
                    self.visit(analyzer, child, &mut inner_stack, &else_cond, &else_after);
                }
                return;
            }
            for child in children(ast, node) {
                self.visit(analyzer, child, stack, &cond, after);
            }
            return;
        }
        for child in children(ast, node) {
            self.visit(analyzer, child, stack, cond, after);
        }
    }
}
