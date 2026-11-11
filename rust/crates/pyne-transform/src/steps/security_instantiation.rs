//! Port of `transformers/security_instantiation.py`: a security-bearing function is
//! cloned per direct-`Name` call site, so each call site gets its own security
//! contexts (Pine's instantiation semantics).
//!
//! A function is security-bearing when one instance of it holds a
//! `lib.request.security[_lower_tf]` call, its calls resolved lexically. Each
//! further call site of a bearer gets a deep copy of it, inserted after the
//! original, until no bearer with more than one call site is left; then every
//! bearer with a single call site gets that site's static arguments pinned at the
//! top of its body. Recursive, decorated, shadowed and otherwise referenced
//! functions are left alone, as in the Python step.

use std::collections::{HashMap, HashSet};

use pyne_ast::locations::anchor_all_at;
use pyne_ast::walk::{children, clone, descendants, docstring_end, walk, walk_statements};
use pyne_ast::{Ast, ExprContext, FieldMut, FieldRef, Kind, NodeId, Unsupported};

use super::security::predicates::{has_security_call, is_security_call, is_security_lower_tf_call};
use crate::{Context, Error};

/// The most security calls one called function may reach through the functions
/// it calls (`_MAX_FUNCTION_REQUESTS`): TradingView bounds a function's request
/// values at 254, and counting each security call once never rejects a script
/// it accepts.
const MAX_FUNCTION_REQUESTS: usize = 254;

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

/// The name an import alias binds: `asname`, or the first component of `name`.
fn alias_binding(name: &str, asname: Option<&str>) -> String {
    asname.unwrap_or(name).split('.').next().unwrap_or_default().to_string()
}

/// The names a function binds in its own scope other than by a nested `def`,
/// and the names it declares `global` (`_local_bindings`).
///
/// A binding is a parameter, an assignment, `for` / `with` / `except` / `match` /
/// walrus target, `del`, an import or a class definition. Nested functions,
/// lambdas and class bodies have scopes of their own, and a comprehension target
/// binds only inside the comprehension.
fn local_bindings(ast: &Ast, function: NodeId) -> (HashSet<String>, HashSet<String>) {
    let mut bound = HashSet::new();
    let mut global_names = HashSet::new();
    let (arguments, body) = match ast.kind(function) {
        Kind::FunctionDef { args, body, .. } | Kind::AsyncFunctionDef { args, body, .. } => {
            (*args, body.clone())
        }
        _ => unreachable!(),
    };
    if let Kind::Arguments { posonlyargs, args, vararg, kwonlyargs, kwarg, .. } =
        ast.kind(arguments)
    {
        let all = posonlyargs.iter().chain(args).chain(vararg).chain(kwonlyargs).chain(kwarg);
        for &arg in all {
            if let Kind::Arg { arg, .. } = ast.kind(arg) {
                bound.insert(arg.clone());
            }
        }
    }
    let mut todo = body;
    while let Some(node) = todo.pop() {
        match ast.kind(node) {
            Kind::FunctionDef { .. } | Kind::AsyncFunctionDef { .. } | Kind::Lambda { .. } => {
                continue;
            }
            Kind::ClassDef { name, .. } => {
                bound.insert(name.clone());
                continue;
            }
            Kind::Comprehension { iter, ifs, .. } => {
                todo.push(*iter);
                todo.extend(ifs.iter().copied());
                continue;
            }
            Kind::Name { id, ctx } => {
                if *ctx != ExprContext::Load {
                    bound.insert(id.clone());
                }
            }
            Kind::Alias { name, asname } => {
                bound.insert(alias_binding(name, asname.as_deref()));
            }
            Kind::ExceptHandler { name: Some(name), .. }
            | Kind::MatchAs { name: Some(name), .. }
            | Kind::MatchStar { name: Some(name) }
            | Kind::MatchMapping { rest: Some(name), .. } => {
                bound.insert(name.clone());
            }
            Kind::Global { names } => {
                global_names.extend(names.iter().cloned());
            }
            _ => {}
        }
        todo.extend(children(ast, node));
    }
    bound.retain(|name| !global_names.contains(name));
    (bound, global_names)
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
    /// Names referenced outside a direct-call func position, bound as a
    /// parameter, import, class, `except` or `match` name, or declared global /
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
                Kind::Arg { arg, .. } => {
                    index.non_call_refs.insert(arg.clone());
                }
                Kind::Alias { name, asname } => {
                    index.non_call_refs.insert(alias_binding(name, asname.as_deref()));
                }
                Kind::ExceptHandler { name: Some(name), .. }
                | Kind::MatchAs { name: Some(name), .. }
                | Kind::MatchStar { name: Some(name) }
                | Kind::ClassDef { name, .. }
                | Kind::MatchMapping { rest: Some(name), .. } => {
                    index.non_call_refs.insert(name.clone());
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

    // A function bears security when one instance of it holds a security call, by
    // the same lexical resolution the fan-out bound counts with, so cloning never
    // expands what the bound did not count. The name-level call edges among module
    // functions only serve the cycle detection
    let mut totals = RequestTotals::new(ast, module, &infos);
    let bearing: HashSet<String> = infos
        .iter()
        .filter(|info| totals.total(info.node) > 0)
        .map(|info| function_name(ast, info.node).to_string())
        .collect();
    let mut edges: HashMap<String, HashSet<String>> = HashMap::new();
    for info in &infos {
        let callees = edges.entry(function_name(ast, info.node).to_string()).or_default();
        for node in walk(ast, info.node) {
            if let Kind::Call { func, .. } = ast.kind(node)
                && let Kind::Name { id, .. } = ast.kind(*func)
                && by_name.contains_key(id.as_str())
            {
                callees.insert(id.clone());
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

/// The number of security calls one instance of a function holds after
/// instantiation, memoized per definition node (`_request_totals`).
///
/// It counts the security calls of the body (lambdas and class bodies included,
/// since cloning copies them too), plus the count of every function it calls, once
/// per call site, plus the count of each function defined inside it that it does
/// not call by name. Calls on a recursive cycle count nothing. A call resolves
/// lexically, to the definitions of the nearest enclosing scope that defines the
/// name; when that scope defines it more than once, the call counts as the largest
/// of them: such a function is never cloned, but its callers are. A scope that
/// binds the name in any other way ([`local_bindings`]) and does not define it
/// binds the call to that value, which counts nothing; a `global` declaration
/// resolves the name in the module.
struct RequestTotals<'a> {
    ast: &'a Ast,
    module: NodeId,
    scoped: HashMap<(NodeId, String), Vec<NodeId>>,
    parent: HashMap<NodeId, NodeId>,
    nested: HashMap<NodeId, Vec<NodeId>>,
    own: HashMap<NodeId, usize>,
    callees: HashMap<NodeId, Vec<String>>,
    bound: HashMap<NodeId, HashSet<String>>,
    global_names: HashMap<NodeId, HashSet<String>>,
    totals: HashMap<NodeId, usize>,
    /// Stack depth of every function under count.
    visiting: HashMap<NodeId, usize>,
    /// The smallest depth a recursive call was cut at while counting the current
    /// function.
    low: usize,
}

impl<'a> RequestTotals<'a> {
    fn new(ast: &'a Ast, module: NodeId, infos: &[FuncInfo]) -> RequestTotals<'a> {
        let mut nested: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        let mut scoped: HashMap<(NodeId, String), Vec<NodeId>> = HashMap::new();
        let mut parent: HashMap<NodeId, NodeId> = HashMap::new();
        for info in infos {
            let name = function_name(ast, info.node).to_string();
            scoped.entry((info.region, name)).or_default().push(info.node);
            parent.insert(info.node, info.region);
            nested.entry(info.region).or_default().push(info.node);
        }
        let defined: HashSet<NodeId> = infos.iter().map(|info| info.node).collect();

        let mut own: HashMap<NodeId, usize> = HashMap::new();
        let mut callees: HashMap<NodeId, Vec<String>> = HashMap::new();
        let mut bound: HashMap<NodeId, HashSet<String>> = HashMap::new();
        let mut global_names: HashMap<NodeId, HashSet<String>> = HashMap::new();
        for info in infos {
            let (locals, globals) = local_bindings(ast, info.node);
            bound.insert(info.node, locals);
            global_names.insert(info.node, globals);
            let mut count = 0;
            let mut names = Vec::new();
            let mut todo = children(ast, info.node);
            while let Some(node) = todo.pop() {
                if defined.contains(&node) {
                    continue;
                }
                todo.extend(children(ast, node));
                let Kind::Call { func, .. } = ast.kind(node) else { continue };
                if is_security_call(ast, node) || is_security_lower_tf_call(ast, node) {
                    count += 1;
                }
                if let Kind::Name { id, .. } = ast.kind(*func) {
                    names.push(id.clone());
                }
            }
            own.insert(info.node, count);
            callees.insert(info.node, names);
        }

        RequestTotals {
            ast,
            module,
            scoped,
            parent,
            nested,
            own,
            callees,
            bound,
            global_names,
            totals: HashMap::new(),
            visiting: HashMap::new(),
            low: usize::MAX,
        }
    }

    fn resolve(&mut self, name: &str, mut scope: NodeId) -> usize {
        // The nearest enclosing scope that defines the name binds the call; a
        // name it defines more than once is never cloned itself, but its callers
        // are, so it counts as its largest definition there. A scope binding the
        // name to another value calls that value
        loop {
            if let Some(defs) = self.scoped.get(&(scope, name.to_string())) {
                let defs = defs.clone();
                return defs.into_iter().map(|def| self.total(def)).max().unwrap_or(0);
            }
            if scope == self.module || self.bound[&scope].contains(name) {
                return 0;
            }
            scope = if self.global_names[&scope].contains(name) {
                self.module
            } else {
                self.parent[&scope]
            };
        }
    }

    fn total(&mut self, node: NodeId) -> usize {
        // A count cut at this function or one above it belongs to a recursive
        // cycle and depends on where the cycle was entered: it is not memoized, so
        // every count is the same whatever the order
        if let Some(&done) = self.totals.get(&node) {
            return done;
        }
        if let Some(&cut_at) = self.visiting.get(&node) {
            self.low = self.low.min(cut_at);
            return 0;
        }
        let depth = self.visiting.len();
        self.visiting.insert(node, depth);
        let outer_low = std::mem::replace(&mut self.low, usize::MAX);
        let names = self.callees[&node].clone();
        let mut result = self.own[&node];
        for callee in &names {
            result += self.resolve(callee, node);
        }
        let inner_defs = self.nested.get(&node).cloned().unwrap_or_default();
        for inner in inner_defs {
            let inner_name = function_name(self.ast, inner);
            if !names.iter().any(|n| n == inner_name) {
                result += self.total(inner);
            }
        }
        self.visiting.remove(&node);
        if self.low > depth {
            self.totals.insert(node, result);
        }
        self.low = self.low.min(outer_low);
        result
    }
}

/// Reject a script in which a called function reaches more security calls than
/// TradingView lets one function use (`_check_request_fanout`).
///
/// The count of a function is the number of security calls one instance of it
/// holds after instantiation ([`RequestTotals`]), worked out on the call graph
/// without cloning. A function called from nowhere is the global scope and is not
/// bounded.
fn check_request_fanout(ast: &Ast, module: NodeId) -> Result<(), Error> {
    let infos = collect_functions(ast, module);
    let mut totals = RequestTotals::new(ast, module, &infos);
    let mut order: Vec<String> = Vec::new();
    let mut by_name: HashMap<String, Vec<NodeId>> = HashMap::new();
    let mut called: HashSet<String> = HashSet::new();
    for info in &infos {
        let name = function_name(ast, info.node).to_string();
        let defs = by_name.entry(name.clone()).or_default();
        if defs.is_empty() {
            order.push(name);
        }
        defs.push(info.node);
        for node in walk(ast, info.node) {
            if let Kind::Call { func, .. } = ast.kind(node)
                && let Kind::Name { id, .. } = ast.kind(*func)
            {
                called.insert(id.clone());
            }
        }
    }

    for name in &order {
        let defs = &by_name[name];
        if defs.len() != 1 || !called.contains(name) {
            continue;
        }
        let reached = totals.total(defs[0]);
        if reached > MAX_FUNCTION_REQUESTS {
            return Err(Error::Rejected(format!(
                "function '{name}' reaches {reached} request.security calls through the \
                 functions it calls; a function can use at most {MAX_FUNCTION_REQUESTS} \
                 request values"
            )));
        }
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
    check_request_fanout(ast, module)?;
    // Fixpoint: cloning a caller duplicates its callees' call sites, so re-analyze
    // until no eligible multi-site bearer remains. It ends because recursive
    // functions are never cloned, and the fan-out bound keeps the number of clones
    // polynomial in the script's size
    while pass.clone_one(ast, module)? {}
    pass.specialize_all(ast, module)?;
    Ok(module)
}
