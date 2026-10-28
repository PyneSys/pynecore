//! Port of `transformers/security_instantiation.py`: a security-bearing function is
//! cloned per direct-`Name` call site, so each call site gets its own security
//! contexts (Pine's instantiation semantics).
//!
//! A function is security-bearing when its subtree holds a
//! `lib.request.security[_lower_tf]` call or a direct call of another bearer. Each
//! further call site of a bearer gets a deep copy of it, inserted after the
//! original, until no bearer with more than one call site is left; then every
//! bearer with a single call site gets that site's static arguments pinned at the
//! top of its body. Recursive, decorated, shadowed and otherwise referenced
//! functions are left alone, as in the Python step.

use std::collections::{HashMap, HashSet};

use pyne_ast::locations::anchor_all_at;
use pyne_ast::walk::{clone, descendants, docstring_end, walk, walk_statements};
use pyne_ast::{Ast, ExprContext, FieldMut, FieldRef, Kind, NodeId, Unsupported};

use super::security::predicates::{has_security_call, is_security_call, is_security_lower_tf_call};
use crate::{Context, Error};

/// Hard ceiling on clones per module (`_MAX_CLONES`).
const MAX_CLONES: usize = 64;

/// A statement list: the list field `field` of node `owner`.
#[derive(Clone, Copy, Debug)]
struct Body {
    owner: NodeId,
    field: usize,
}

/// The index of a statement-list field of a node, `None` when the node has no
/// such list (`getattr(stmt, name, None)` not being a list).
fn list_field(ast: &Ast, node: NodeId, name: &str) -> Option<Body> {
    let kind = ast.kind(node);
    let field = kind.class().fields.iter().position(|field| field.name == name)?;
    matches!(kind.field(field), FieldRef::Nodes(_)).then_some(Body { owner: node, field })
}

fn statements(ast: &Ast, body: Body) -> &Vec<NodeId> {
    let FieldRef::Nodes(items) = ast.kind(body.owner).field(body.field) else { unreachable!() };
    items
}

fn statements_mut(ast: &mut Ast, body: Body) -> &mut Vec<NodeId> {
    let FieldMut::Nodes(items) = ast.kind_mut(body.owner).field_mut(body.field) else {
        unreachable!()
    };
    items
}

fn is_function(kind: &Kind) -> bool {
    matches!(kind, Kind::FunctionDef { .. } | Kind::AsyncFunctionDef { .. })
}

fn function_name(ast: &Ast, function: NodeId) -> &str {
    match ast.kind(function) {
        Kind::FunctionDef { name, .. } | Kind::AsyncFunctionDef { name, .. } => name,
        _ => unreachable!(),
    }
}

/// One function definition eligible for instantiation analysis (`_FuncInfo`).
#[derive(Clone, Copy, Debug)]
struct FuncInfo {
    node: NodeId,
    /// The statement list holding the definition
    owner: Body,
    /// The subtree in which the definition's name is in scope: the module for a
    /// module-level def, the enclosing function for a nested one
    region: NodeId,
}

/// Every function definition with its owner list and scope region, in the order
/// `_collect_functions` finds them. Class bodies are skipped entirely.
fn collect_functions(ast: &Ast, module: NodeId) -> Vec<FuncInfo> {
    let mut out = Vec::new();
    if let Some(body) = list_field(ast, module, "body") {
        scan_body(ast, body, module, &mut out);
    }
    out
}

fn scan_body(ast: &Ast, body: Body, region: NodeId, out: &mut Vec<FuncInfo>) {
    for &statement in statements(ast, body) {
        match ast.kind(statement) {
            Kind::FunctionDef { .. } | Kind::AsyncFunctionDef { .. } => {
                out.push(FuncInfo { node: statement, owner: body, region });
                if let Some(inner) = list_field(ast, statement, "body") {
                    scan_body(ast, inner, statement, out);
                }
            }
            Kind::ClassDef { .. } => {}
            // Compound statements can nest defs (if / try / with / for)
            _ => scan_sub(ast, statement, region, out),
        }
    }
}

fn scan_sub(ast: &Ast, statement: NodeId, region: NodeId, out: &mut Vec<FuncInfo>) {
    for name in ["body", "orelse", "finalbody"] {
        if let Some(body) = list_field(ast, statement, name) {
            scan_body(ast, body, region, out);
        }
    }
    for name in ["handlers", "cases"] {
        let Some(holders) = list_field(ast, statement, name) else { continue };
        for &holder in statements(ast, holders) {
            if let Some(body) = list_field(ast, holder, "body") {
                scan_body(ast, body, region, out);
            }
        }
    }
}

/// Name references of one scope region, gathered in a single walk (`_RegionIndex`).
struct RegionIndex {
    /// Direct-`Name` call sites per callee name, in source order
    call_sites: HashMap<String, Vec<NodeId>>,
    /// Names referenced outside a direct-call func position, or declared global /
    /// nonlocal
    non_call_refs: HashSet<String>,
}

impl RegionIndex {
    fn new(ast: &Ast, region: NodeId) -> RegionIndex {
        let mut index = RegionIndex { call_sites: HashMap::new(), non_call_refs: HashSet::new() };
        let mut call_funcs = HashSet::new();
        let mut nodes = vec![region];
        nodes.extend(descendants(ast, region, &|_| false));
        for node in nodes {
            match ast.kind(node) {
                Kind::Call { func, .. } => {
                    if let Kind::Name { id, .. } = ast.kind(*func) {
                        // Pre-order: the call is seen before its func child
                        call_funcs.insert(*func);
                        index.call_sites.entry(id.clone()).or_default().push(node);
                    }
                }
                Kind::Name { id, .. } => {
                    if !call_funcs.contains(&node) {
                        index.non_call_refs.insert(id.clone());
                    }
                }
                Kind::Global { names } | Kind::Nonlocal { names } => {
                    index.non_call_refs.extend(names.iter().cloned());
                }
                _ => {}
            }
        }
        index
    }
}

/// Whether `target` is reachable from `start` over at least one call edge.
fn reaches(edges: &HashMap<String, HashSet<String>>, start: &str, target: &str) -> bool {
    let mut seen: HashSet<&str> = HashSet::new();
    let mut todo = vec![start];
    while let Some(current) = todo.pop() {
        let Some(callees) = edges.get(current) else { continue };
        for callee in callees {
            if callee == target {
                return true;
            }
            if seen.insert(callee) {
                todo.push(callee);
            }
        }
    }
    false
}

/// A candidate: an eligible bearer with its direct-`Name` call sites in source order.
type Candidate = (FuncInfo, Vec<NodeId>);

/// Every security-bearing function that passes the bail-outs, with its call sites
/// (possibly none) (`_candidates`).
fn candidates(ast: &Ast, module: NodeId) -> Vec<Candidate> {
    let infos = collect_functions(ast, module);

    let mut by_name: HashMap<&str, usize> = HashMap::new();
    for info in &infos {
        *by_name.entry(function_name(ast, info.node)).or_default() += 1;
    }

    // Direct bearers and the name-level call edges among module functions
    let mut bearing: HashSet<String> = HashSet::new();
    let mut edges: HashMap<String, HashSet<String>> = HashMap::new();
    for info in &infos {
        let name = function_name(ast, info.node);
        let callees = edges.entry(name.to_string()).or_default();
        for node in walk(ast, info.node) {
            let Kind::Call { func, .. } = ast.kind(node) else { continue };
            if is_security_call(ast, node) || is_security_lower_tf_call(ast, node) {
                bearing.insert(name.to_string());
            }
            if let Kind::Name { id, .. } = ast.kind(*func)
                && by_name.contains_key(id.as_str())
            {
                callees.insert(id.clone());
            }
        }
    }

    // Transitive closure: calling a bearer makes the caller a bearer
    let mut changed = true;
    while changed {
        changed = false;
        for (name, callees) in &edges {
            if !bearing.contains(name) && callees.iter().any(|callee| bearing.contains(callee)) {
                bearing.insert(name.clone());
                changed = true;
            }
        }
    }

    // A bearer on a call-graph cycle is excluded, or cloning would never converge
    let on_cycle: HashSet<String> =
        bearing.iter().filter(|name| reaches(&edges, name, name)).cloned().collect();

    let mut regions: HashMap<NodeId, RegionIndex> = HashMap::new();
    let mut eligible = Vec::new();
    for info in infos {
        let name = function_name(ast, info.node);
        if !bearing.contains(name) || on_cycle.contains(name) {
            continue;
        }
        if by_name[name] > 1 {
            continue;
        }
        let decorated = match ast.kind(info.node) {
            Kind::FunctionDef { decorator_list, .. }
            | Kind::AsyncFunctionDef { decorator_list, .. } => !decorator_list.is_empty(),
            _ => unreachable!(),
        };
        if decorated {
            continue;
        }
        let index =
            regions.entry(info.region).or_insert_with(|| RegionIndex::new(ast, info.region));
        if index.non_call_refs.contains(name) {
            continue;
        }
        eligible.push((info, index.call_sites.get(name).cloned().unwrap_or_default()));
    }
    eligible
}

/// Whether an argument can be duplicated into the callee without changing
/// behaviour: a literal or a bare `lib.*` attribute chain (`_is_static_arg`).
fn is_static_arg(ast: &Ast, node: NodeId) -> bool {
    match ast.kind(node) {
        Kind::Constant { .. } => true,
        Kind::Attribute { value, .. } => is_static_arg(ast, *value),
        Kind::Name { id, .. } => id == "lib",
        _ => false,
    }
}

/// Pin a call site's static arguments as bindings at the top of the function
/// (`_specialize`).
fn specialize(ast: &mut Ast, function: NodeId, site: NodeId) -> Result<(), Unsupported> {
    let (Kind::FunctionDef { args, .. } | Kind::AsyncFunctionDef { args, .. }) = ast.kind(function)
    else {
        unreachable!()
    };
    let Kind::Arguments { posonlyargs, args, vararg, kwonlyargs, kwarg, .. } = ast.kind(*args)
    else {
        unreachable!()
    };
    if vararg.is_some() || kwarg.is_some() || !posonlyargs.is_empty() {
        return Ok(());
    }
    let Kind::Call { args: site_args, keywords, .. } = ast.kind(site) else { unreachable!() };
    if site_args.iter().any(|&arg| matches!(ast.kind(arg), Kind::Starred { .. })) {
        return Ok(());
    }
    let mut named = Vec::with_capacity(keywords.len());
    for &keyword in keywords {
        let Kind::Keyword { arg, value } = ast.kind(keyword) else { unreachable!() };
        // A `**mapping` argument
        let Some(arg) = arg else { return Ok(()) };
        named.push((arg, *value));
    }

    let arg_name = |arg: &NodeId| match ast.kind(*arg) {
        Kind::Arg { arg, .. } => arg.clone(),
        _ => unreachable!(),
    };
    let positional: Vec<String> = args.iter().map(arg_name).collect();
    let mut by_name = positional.clone();
    by_name.extend(kwonlyargs.iter().map(arg_name));
    if site_args.len() > positional.len() {
        return Ok(());
    }
    let mut bound: HashMap<&str, NodeId> = HashMap::new();
    for (index, &value) in site_args.iter().enumerate() {
        bound.insert(&positional[index], value);
    }
    for (arg, value) in named {
        if !by_name.contains(arg) {
            return Ok(());
        }
        bound.insert(arg, value);
    }

    let pins: Vec<(String, NodeId)> = by_name
        .iter()
        .filter_map(|name| bound.get(name.as_str()).map(|&value| (name.clone(), value)))
        .filter(|&(_, value)| is_static_arg(ast, value))
        .collect();
    let mut pinned = Vec::with_capacity(pins.len());
    for (name, value) in pins {
        let target = ast.add(Kind::Name { id: name, ctx: ExprContext::Store });
        let value = clone(ast, value, &mut HashMap::new());
        let assign = ast.add(Kind::Assign { targets: vec![target], value, type_comment: None });
        // The value is cloned from the call site, but it is bound at the function's
        // entry, as a parameter is
        pinned.push(anchor_all_at(ast, assign, function)?);
    }
    if !pinned.is_empty() {
        let body = list_field(ast, function, "body").expect("a function has a body");
        let at = docstring_end(ast, statements(ast, body));
        statements_mut(ast, body).splice(at..at, pinned);
    }
    Ok(())
}

/// The whole-module fixpoint (`SecurityInstantiationTransformer`).
struct Instantiation {
    clones_made: usize,
    /// Every candidate of the module as it is now; dropped whenever a clone
    /// changes the module
    candidates_now: Option<Vec<Candidate>>,
}

impl Instantiation {
    /// The eligible bearers with at least `min_sites` call sites (`_analyze`).
    fn analyze(&mut self, ast: &Ast, module: NodeId, min_sites: usize) -> Vec<Candidate> {
        let now = self.candidates_now.get_or_insert_with(|| candidates(ast, module));
        now.iter().filter(|(_, sites)| sites.len() >= min_sites).cloned().collect()
    }

    /// A collision-free clone name; the chosen name is added to `existing`.
    fn unique_name(&self, base: &str, existing: &mut HashSet<String>) -> String {
        let mut candidate = format!("{base}__pyne_inst{}", self.clones_made);
        while existing.contains(&candidate) {
            candidate.push('_');
        }
        existing.insert(candidate.clone());
        candidate
    }

    /// Clone the first eligible multi-site bearer; true if work was done (`_clone_one`).
    fn clone_one(&mut self, ast: &mut Ast, module: NodeId) -> Result<bool, Error> {
        let eligible = self.analyze(ast, module, 2);
        let Some((info, sites)) = eligible.into_iter().next() else { return Ok(false) };
        self.candidates_now = None;
        let name = function_name(ast, info.node).to_string();
        let mut existing: HashSet<String> = walk_statements(ast, module)
            .into_iter()
            .filter(|&node| is_function(ast.kind(node)))
            .map(|node| function_name(ast, node).to_string())
            .collect();
        let index = statements(ast, info.owner)
            .iter()
            .position(|&statement| statement == info.node)
            .ok_or_else(|| {
                Error::Unsupported("security instantiation: def not in its owner body".into())
            })?;
        // Call site 1 keeps the original function; each further site gets a fresh
        // clone inserted after the original (source order preserved)
        for (offset, &site) in sites.iter().enumerate().skip(1) {
            self.clones_made += 1;
            if self.clones_made > MAX_CLONES {
                return Err(Error::Rejected(format!(
                    "security instantiation exceeded {MAX_CLONES} function clones — the \
                     script's request.security call graph is too large to instantiate per \
                     call site"
                )));
            }
            let copy = clone(ast, info.node, &mut HashMap::new());
            let clone_name = self.unique_name(&name, &mut existing);
            match ast.kind_mut(copy) {
                Kind::FunctionDef { name, .. } | Kind::AsyncFunctionDef { name, .. } => {
                    *name = clone_name.clone();
                }
                _ => unreachable!(),
            }
            statements_mut(ast, info.owner).insert(index + offset, copy);
            let Kind::Call { func, .. } = ast.kind(site) else { unreachable!() };
            let func = *func;
            let Kind::Name { id, .. } = ast.kind_mut(func) else { unreachable!() };
            *id = clone_name;
        }
        Ok(true)
    }

    /// Pin call-site constants into every single-site bearer (`_specialize_all`).
    fn specialize_all(&mut self, ast: &mut Ast, module: NodeId) -> Result<(), Unsupported> {
        for (info, sites) in self.analyze(ast, module, 1) {
            if let [site] = sites[..] {
                specialize(ast, info.node, site)?;
            }
        }
        Ok(())
    }
}

pub fn run(ast: &mut Ast, module: NodeId, _ctx: &Context<'_>) -> Result<NodeId, Error> {
    if !has_security_call(ast, module) {
        return Ok(module);
    }
    let mut pass = Instantiation { clones_made: 0, candidates_now: None };
    // Fixpoint: cloning a caller duplicates its callees' call sites, so re-analyze
    // until no eligible multi-site bearer remains, bounded by the clone cap
    for _ in 0..=MAX_CLONES {
        if !pass.clone_one(ast, module)? {
            break;
        }
    }
    pass.specialize_all(ast, module)?;
    Ok(module)
}
