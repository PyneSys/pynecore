//! What the statements of `main()` and of the functions it reaches read, write and
//! mutate (`_Facts`, `_Collector`), the alias classes (`_Aliases`), and the backward
//! slice over them (`_build_index`, `_slice`).

use std::collections::{HashMap, HashSet};

use indexmap::IndexMap;
use pyne_ast::{Ast, ExprContext, Kind, NodeId};

use super::{
    Aliases, EXTERNAL, PROTOCOL_CALLS, chain_root_name, is_def, own, protocol_sid, stmt_nodes,
};
use crate::pine_type_rules::{BOOL, FLOAT, INT, STR, get_ty};
use crate::steps::security::dependencies::{
    COLLECTION_NAMESPACES, DependencyAnalyzer, is_collection_reader, is_unmodelled, root_name,
};

/// Fixpoint safety net for the call-graph summaries; the lattice is finite and
/// monotone, so the loop always converges well below this.
const MAX_ROUNDS: usize = 1000;

/// The scalar types whose runtime values are immutable (`_IMMUTABLE_SCALARS`). A
/// `Color` is not one: its `a` / `t` setters change it in place, so it keeps its
/// alias tracking.
const IMMUTABLE_SCALARS: [&str; 4] = [INT, FLOAT, BOOL, STR];

/// Per top-level statement of `main()`, the facts every slice needs (`_StmtIndex`).
#[derive(Default)]
pub(super) struct StmtIndex {
    /// Alias classes the statement reads
    reads: HashSet<String>,
    /// Alias classes it writes or may mutate
    defs: HashSet<String>,
    /// Sids whose `__sec_read__` the statement holds
    read_sids: HashSet<String>,
    /// Sids whose `__sec_write__` the statement holds
    pub(super) write_sids: HashSet<String>,
    /// A definition or an early exit: kept in every slice
    mandatory: bool,
}

/// What one statement (or one function body) does to the names around it (`_Facts`).
#[derive(Clone, Default)]
pub(super) struct Facts {
    /// Qualified names whose VALUE the code reads
    reads: HashSet<String>,
    /// Qualified names the code rebinds
    writes: HashSet<String>,
    /// Qualified names whose object the code may mutate in place
    mutates: HashSet<String>,
    /// Scope keys of the user functions the code calls
    calls: HashSet<String>,
    /// Sids the code reads through `__sec_read__`
    read_sids: HashSet<String>,
    /// Sids the code writes through `__sec_write__`
    write_sids: HashSet<String>,
}

impl Facts {
    /// Merge another summary's name and sid sets in; true when anything grew.
    fn absorb(&mut self, other: &Facts) -> bool {
        let before = self.sizes();
        self.reads.extend(other.reads.iter().cloned());
        self.writes.extend(other.writes.iter().cloned());
        self.mutates.extend(other.mutates.iter().cloned());
        self.read_sids.extend(other.read_sids.iter().cloned());
        self.write_sids.extend(other.write_sids.iter().cloned());
        before != self.sizes()
    }

    fn sizes(&self) -> [usize; 5] {
        [
            self.reads.len(),
            self.writes.len(),
            self.mutates.len(),
            self.read_sids.len(),
            self.write_sids.len(),
        ]
    }

    /// Mark the code as touching state the pass cannot model.
    fn external(&mut self) {
        self.reads.insert(EXTERNAL.into());
        self.writes.insert(EXTERNAL.into());
        self.mutates.insert(EXTERNAL.into());
    }
}

/// `(namespace, function)` of a `lib.<array|matrix|map>.<fn>` callee (`_collection_call`).
fn collection_call(ast: &Ast, func: NodeId) -> Option<(&str, &str)> {
    let Kind::Attribute { value: owner, attr: function, .. } = ast.kind(func) else {
        return None;
    };
    let Kind::Attribute { value, attr, .. } = ast.kind(*owner) else { return None };
    (COLLECTION_NAMESPACES.contains(&attr.as_str())
        && matches!(ast.kind(*value), Kind::Name { id, .. } if id == "lib"))
    .then_some((attr.as_str(), function.as_str()))
}

/// Whether an expression is an attribute chain rooted at `lib` (`_is_lib_chain`).
fn is_lib_chain(ast: &Ast, node: NodeId) -> bool {
    chain_root_name(ast, node) == Some("lib")
}

/// The collection a `lib.<array|matrix|map>.<fn>` call acts on (`_collection_target`).
///
/// Every collection builtin takes it as its first parameter, named `id`, so a
/// keyword call names the very same object a positional one passes. An unpacked
/// positional (`array.set(*args)`) names nothing at all.
fn collection_target(ast: &Ast, call: NodeId) -> Option<NodeId> {
    let Kind::Call { args, keywords, .. } = ast.kind(call) else { return None };
    if let Some(&first) = args.first() {
        return (!matches!(ast.kind(first), Kind::Starred { .. })).then_some(first);
    }
    keywords.iter().find_map(|&keyword| match ast.kind(keyword) {
        Kind::Keyword { arg: Some(arg), value } if arg == "id" => Some(*value),
        _ => None,
    })
}

/// A call's positional arguments, then its keyword values.
fn call_operands(ast: &Ast, call: NodeId) -> Vec<NodeId> {
    let Kind::Call { args, keywords, .. } = ast.kind(call) else { return Vec::new() };
    let mut out = args.clone();
    out.extend(keywords.iter().map(|&keyword| match ast.kind(keyword) {
        Kind::Keyword { value, .. } => *value,
        _ => unreachable!("no keyword"),
    }));
    out
}

/// Whether a call is a collection builtin mutating an unnameable id
/// (`_unresolved_collection`).
///
/// `lib.array.set(**args)` mutates a collection the pass cannot name, so the
/// mutation joins no alias class and the statement holding it would be dropped from
/// a slice that reads the very collection it writes. The whole module's
/// optimization is given up instead.
fn unresolved_collection(ast: &Ast, call: NodeId) -> bool {
    let Kind::Call { func, .. } = ast.kind(call) else { return false };
    let Some((namespace, function)) = collection_call(ast, *func) else { return false };
    if is_collection_reader(namespace, function) {
        return false;
    }
    collection_target(ast, call).is_none()
}

/// Scope keys of `main` and every user function it can reach (`_reachable_scopes`),
/// sorted, and the keys of the scopes that cannot be modelled: one holding a
/// construct the pass cannot classify, or a collection builtin called on an
/// unnameable id.
pub(super) fn reachable_scopes(
    ast: &Ast,
    scopes: &DependencyAnalyzer<'_>,
    main_key: &str,
) -> (Vec<String>, HashSet<String>) {
    let mut seen: HashSet<String> = HashSet::from([main_key.to_string()]);
    let mut work = vec![main_key.to_string()];
    let mut unmodelled: HashSet<String> = HashSet::new();
    while let Some(key) = work.pop() {
        let node = scopes.scopes[&key].node;
        let mut blocked = false;
        for sub in own(ast, node).into_iter().skip(1) {
            if is_unmodelled(ast.kind(sub)) {
                blocked = true;
            }
            let Kind::Call { func, .. } = ast.kind(sub) else { continue };
            if unresolved_collection(ast, sub) {
                blocked = true;
            }
            let Kind::Name { id, .. } = ast.kind(*func) else { continue };
            if let Some(callee) = scopes.lookup_func_from(id, &key)
                && seen.insert(callee.clone())
            {
                work.push(callee);
            }
        }
        // A nested definition is reachable through any call of it; its own body is
        // scanned when the call is found, but the definition's decorators and
        // defaults belong to the enclosing scope and are covered there.
        if blocked {
            unmodelled.insert(key);
        }
    }
    let mut reachable: Vec<String> = seen.into_iter().collect();
    reachable.sort();
    (reachable, unmodelled)
}

/// Names bound by the module's imports, `lib` left out (`_module_imports`): every
/// call through one is opaque.
pub(super) fn module_imports(ast: &Ast, statements: &[NodeId]) -> HashSet<String> {
    let mut names = HashSet::new();
    for &stmt in statements {
        let (Kind::Import { names: aliases } | Kind::ImportFrom { names: aliases, .. }) =
            ast.kind(stmt)
        else {
            continue;
        };
        for &alias in aliases {
            let Kind::Alias { name, asname } = ast.kind(alias) else { unreachable!() };
            names.insert(match asname {
                Some(asname) => asname.clone(),
                None => name.split('.').next().unwrap_or(name).to_string(),
            });
        }
    }
    names.remove("lib");
    names
}

/// Builds the [`Facts`] and alias classes of the statements of a scope (`_Collector`).
pub(super) struct Collector<'s, 'a> {
    ast: &'a Ast,
    scopes: &'s DependencyAnalyzer<'a>,
    aliases: Aliases,
    imports: HashSet<String>,
    library_calls: HashSet<String>,
    returns: HashMap<String, Vec<String>>,
    returning: HashSet<String>,
    /// Set when a mutation was seen that joins no alias class, so no slice of the
    /// module may drop statements.
    pub(super) unresolved: bool,
}

impl<'s, 'a> Collector<'s, 'a> {
    pub(super) fn new(
        ast: &'a Ast,
        scopes: &'s DependencyAnalyzer<'a>,
        imports: HashSet<String>,
        library_calls: HashSet<String>,
    ) -> Self {
        Collector {
            ast,
            scopes,
            aliases: Aliases::default(),
            imports,
            library_calls,
            returns: HashMap::new(),
            returning: HashSet::new(),
            unresolved: false,
        }
    }

    fn qualify(&self, name: &str, scope: &str) -> String {
        self.scopes.q(name, Some(scope))
    }

    // --- aliasing ---

    /// Qualified names `expr` may evaluate to an alias of (`alias_roots`).
    fn alias_roots(&mut self, expr: NodeId, scope: &str) -> Vec<String> {
        let ast = self.ast;
        if IMMUTABLE_SCALARS.contains(&get_ty(ast, expr)) {
            // An immutable scalar is a value, not an object: no later mutation can
            // reach it through a second name, so it joins no alias class.
            return Vec::new();
        }
        match ast.kind(expr) {
            Kind::Name { .. } | Kind::Attribute { .. } | Kind::Subscript { .. } => {
                return match root_name(ast, expr) {
                    None | Some("lib") => Vec::new(),
                    Some(root) => vec![self.qualify(root, scope)],
                };
            }
            &Kind::IfExp { body, orelse, .. } => {
                let mut roots = self.alias_roots(body, scope);
                roots.extend(self.alias_roots(orelse, scope));
                return roots;
            }
            Kind::BoolOp { values, .. } => {
                let mut roots = Vec::new();
                for value in values.clone() {
                    roots.extend(self.alias_roots(value, scope));
                }
                return roots;
            }
            Kind::Call { .. } => {}
            _ => return Vec::new(),
        }
        let Kind::Call { func, .. } = ast.kind(expr) else { unreachable!() };
        let func = *func;
        if matches!(ast.kind(func), Kind::Attribute { .. }) {
            if collection_call(ast, func).is_some() {
                // A reader does not hand the collection itself back, but it may hand
                // back an element of it (``array.get`` of a UDT) or a view sharing
                // its storage, and a mutation written through that element is a
                // mutation of the collection. A constructor keeps the objects it was
                // seeded with (``array.new(size, item)``, ``array.from_items(a,
                // b)``), so its result shares their storage too. Every argument is
                // therefore taken as alias provenance: a scalar index or size
                // contributes nothing, and widening a class only ever keeps more
                // statements.
                let mut roots = Vec::new();
                for arg in call_operands(ast, expr) {
                    roots.extend(self.alias_roots(arg, scope));
                }
                return roots;
            }
            if is_lib_chain(ast, func) {
                // Every other ``lib`` builtin returns a fresh value.
                return Vec::new();
            }
        }
        // A user function or an unknown callee may return one of its arguments.
        let mut roots = Vec::new();
        for arg in call_operands(ast, expr) {
            roots.extend(self.alias_roots(arg, scope));
        }
        // It may also return an object of an enclosing scope it never got as an
        // argument (a module-level collection built once and handed out by a
        // getter), which no argument of the call names.
        if let Kind::Name { id, .. } = ast.kind(func)
            && let Some(key) = self.scopes.lookup_func_from(id, scope)
        {
            roots.extend(self.returned_roots(&key));
        }
        roots
    }

    /// Qualified names the user function `key` may hand back (`_returned_roots`).
    ///
    /// The callee's `return` expressions are resolved in the callee's own scope, so
    /// a name it closes over or reads from module scope qualifies to the very object
    /// the caller's other statements mutate. A recursive callee is answered with
    /// [`EXTERNAL`]: the recursion cannot be unrolled here, and joining the external
    /// class keeps every statement that could reach the object.
    fn returned_roots(&mut self, key: &str) -> Vec<String> {
        if let Some(cached) = self.returns.get(key) {
            return cached.clone();
        }
        if self.returning.contains(key) {
            return vec![EXTERNAL.to_string()];
        }
        self.returning.insert(key.to_string());
        let mut roots = Vec::new();
        for sub in own(self.ast, self.scopes.scopes[key].node) {
            if let Kind::Return { value: Some(value) } = self.ast.kind(sub) {
                roots.extend(self.alias_roots(*value, key));
            }
        }
        self.returning.remove(key);
        self.returns.insert(key.to_string(), roots.clone());
        roots
    }

    /// Union every `target = <alias expression>` of the scope (`record_aliases`).
    pub(super) fn record_aliases(&mut self, scope: &str) {
        let ast = self.ast;
        for sub in own(ast, self.scopes.scopes[scope].node) {
            if matches!(ast.kind(sub), Kind::Call { .. }) {
                self.record_call_aliases(sub, scope);
            }
            let (targets, value) = match ast.kind(sub) {
                Kind::Assign { targets, value, .. } => (targets.clone(), *value),
                Kind::AnnAssign { target, value: Some(value), .. } => (vec![*target], *value),
                Kind::For { target, iter, .. } | Kind::AsyncFor { target, iter, .. } => {
                    (vec![*target], *iter)
                }
                _ => continue,
            };
            let roots = self.alias_roots(value, scope);
            if roots.is_empty() {
                continue;
            }
            for target in targets {
                let name = match root_name(ast, target) {
                    None | Some("lib") => continue,
                    Some(name) => name,
                };
                let qualified = self.qualify(name, scope);
                for root in &roots {
                    self.aliases.union(&qualified, root);
                }
            }
        }
    }

    /// Join the objects a call may store into one another (`_record_call_aliases`).
    ///
    /// `array.push(store, item)` puts `item` inside `store` without binding any name,
    /// so a later `array.get(store, 0).acc = x` mutates `item` as well. A user or
    /// unknown callee may keep one argument inside another the same way, and a
    /// method call may keep an argument inside its receiver. Joining the argument
    /// classes is conservative: a wider class only ever keeps more statements.
    fn record_call_aliases(&mut self, call: NodeId, scope: &str) {
        let ast = self.ast;
        let Kind::Call { func, .. } = ast.kind(call) else { unreachable!() };
        let func = *func;
        let mut args = call_operands(ast, call);
        if let Kind::Attribute { value: receiver, .. } = ast.kind(func) {
            if let Some((namespace, function)) = collection_call(ast, func) {
                if is_collection_reader(namespace, function) {
                    return;
                }
            } else if is_lib_chain(ast, func) {
                // Every other ``lib`` builtin keeps nothing of its arguments.
                return;
            } else if !matches!(root_name(ast, func), None | Some("lib")) {
                args.push(*receiver);
            }
        }
        let mut roots = Vec::new();
        for arg in args {
            roots.extend(self.alias_roots(arg, scope));
        }
        if let Some((first, rest)) = roots.split_first() {
            for other in rest {
                self.aliases.union(first, other);
            }
        }
    }

    // --- facts ---

    /// Collect what the given nodes do, not entering nested definitions (`facts`).
    fn facts(&mut self, nodes: &[NodeId], scope: &str) -> Facts {
        let mut facts = Facts::default();
        for &top in nodes {
            for sub in own(self.ast, top) {
                self.visit(sub, scope, &mut facts);
            }
        }
        facts
    }

    fn visit(&mut self, node: NodeId, scope: &str, facts: &mut Facts) {
        let ast = self.ast;
        match ast.kind(node) {
            Kind::Name { id, ctx } => {
                if id == "lib" {
                    return;
                }
                let qualified = self.qualify(id, scope);
                match ctx {
                    ExprContext::Store | ExprContext::Del => facts.writes.insert(qualified),
                    ExprContext::Load => facts.reads.insert(qualified),
                };
            }
            Kind::Attribute { ctx, .. } | Kind::Subscript { ctx, .. } => {
                if *ctx == ExprContext::Load {
                    return;
                }
                match root_name(ast, node) {
                    Some("lib") => {
                        // Writing onto the library namespace is global state.
                        facts.external();
                    }
                    Some(root) => {
                        let qualified = self.qualify(root, scope);
                        facts.reads.insert(qualified.clone());
                        facts.mutates.insert(qualified);
                    }
                    None => {
                        // The object written to is produced by an expression
                        // (``array.get(store, 0).acc = x``): the store reaches
                        // whatever that expression is an alias of.
                        let mut base = node;
                        while let Kind::Attribute { value, .. } | Kind::Subscript { value, .. } =
                            ast.kind(base)
                        {
                            base = *value;
                        }
                        let roots = self.alias_roots(base, scope);
                        if roots.is_empty() {
                            self.unresolved = true;
                        } else {
                            facts.reads.extend(roots.iter().cloned());
                            facts.mutates.extend(roots);
                        }
                    }
                }
            }
            Kind::AugAssign { target, .. } => {
                if let Some(root) = root_name(ast, *target)
                    && root != "lib"
                {
                    facts.reads.insert(self.qualify(root, scope));
                }
            }
            Kind::Call { .. } => self.visit_call(node, scope, facts),
            Kind::Global { names } | Kind::Nonlocal { names } => {
                for name in names {
                    facts.writes.insert(self.qualify(name, scope));
                }
            }
            _ => {}
        }
    }

    fn mutate_args(&mut self, args: &[NodeId], scope: &str, facts: &mut Facts) {
        for &arg in args {
            facts.mutates.extend(self.alias_roots(arg, scope));
        }
    }

    fn visit_call(&mut self, call: NodeId, scope: &str, facts: &mut Facts) {
        let ast = self.ast;
        let Kind::Call { func, .. } = ast.kind(call) else { unreachable!() };
        let func = *func;
        let args = call_operands(ast, call);
        match ast.kind(func) {
            Kind::Name { id, .. } => {
                if let Some(key) = self.scopes.lookup_func_from(id, scope) {
                    facts.calls.insert(key);
                    self.mutate_args(&args, scope, facts);
                    return;
                }
                if PROTOCOL_CALLS.contains(&id.as_str()) {
                    // A protocol call belongs to the context named by its first
                    // argument, not to any name of the script.
                    if let Some(sid) = protocol_sid(ast, call, "__sec_read__") {
                        facts.read_sids.insert(sid.to_string());
                    }
                    if let Some(sid) = protocol_sid(ast, call, "__sec_write__") {
                        facts.write_sids.insert(sid.to_string());
                    }
                    return;
                }
                if self.library_calls.contains(id) {
                    // A library function bound by a pynecore import (``cast_int``)
                    // or a marker call of an earlier pass: the trust the ``lib``
                    // chain gets, it touches no global of the script. It may still
                    // keep or change what it is handed.
                    self.mutate_args(&args, scope, facts);
                    return;
                }
                if self.imports.contains(id) {
                    facts.external();
                    return;
                }
                // An unresolvable callee: assume it does everything.
                facts.external();
                self.mutate_args(&args, scope, facts);
                return;
            }
            Kind::Attribute { .. } => {
                if let Some((namespace, function)) = collection_call(ast, func) {
                    if !is_collection_reader(namespace, function) {
                        // A mutator whose collection argument cannot be named
                        // (unpacking) is not modelled at all: the module was given
                        // up before this point (see ``_unresolved_collection``).
                        if let Some(target) = collection_target(ast, call) {
                            facts.mutates.extend(self.alias_roots(target, scope));
                        }
                    }
                    return;
                }
                if is_lib_chain(ast, func) {
                    return;
                }
                if let Some(root) = root_name(ast, func) {
                    if self.imports.contains(root) {
                        // A call into another module: its globals are invisible here.
                        facts.external();
                        self.mutate_args(&args, scope, facts);
                        return;
                    }
                    // A method call on an object of the script's own.
                    facts.mutates.insert(self.qualify(root, scope));
                    return;
                }
            }
            _ => {}
        }
        facts.external();
        self.mutate_args(&args, scope, facts);
    }
}

/// Transitive read / write / mutate summary of every reachable function
/// (`_function_summaries`).
pub(super) fn function_summaries(
    collector: &mut Collector<'_, '_>,
    scopes: &DependencyAnalyzer<'_>,
    reachable: &[String],
    main_key: &str,
) -> IndexMap<String, Facts> {
    let mut summaries: IndexMap<String, Facts> = IndexMap::new();
    for key in reachable {
        if key == main_key {
            continue;
        }
        let body = scopes.scopes[key].body.clone();
        let facts = collector.facts(&body, key);
        summaries.insert(key.clone(), facts);
    }
    for _ in 0..MAX_ROUNDS {
        let mut changed = false;
        for position in 0..summaries.len() {
            let calls: Vec<String> = summaries[position].calls.iter().cloned().collect();
            for callee in calls {
                match summaries.get(&callee).cloned() {
                    Some(other) => {
                        if summaries[position].absorb(&other) {
                            changed = true;
                        }
                    }
                    None => {
                        let facts = &mut summaries[position];
                        if !facts.reads.contains(EXTERNAL) {
                            facts.external();
                            changed = true;
                        }
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    summaries
}

/// Fold the summaries of the functions a statement calls into its facts (`_expand`).
fn expand(mut facts: Facts, summaries: &IndexMap<String, Facts>) -> Facts {
    let mut seen: HashSet<String> = HashSet::new();
    let mut work: Vec<String> = facts.calls.iter().cloned().collect();
    while let Some(key) = work.pop() {
        if !seen.insert(key.clone()) {
            continue;
        }
        match summaries.get(&key) {
            None => facts.external(),
            Some(other) => {
                facts.absorb(other);
                work.extend(other.calls.iter().cloned());
            }
        }
    }
    facts
}

/// Build the context-independent [`StmtIndex`] of `main()`'s body (`_build_index`).
pub(super) fn build_index(
    ast: &Ast,
    body: &[NodeId],
    collector: &mut Collector<'_, '_>,
    summaries: &IndexMap<String, Facts>,
    main_key: &str,
) -> Vec<StmtIndex> {
    let mut index = Vec::with_capacity(body.len());
    for &stmt in body {
        let facts = expand(collector.facts(&stmt_nodes(ast, stmt), main_key), summaries);
        let aliases = &collector.aliases;
        let mut defs = aliases.classes(&facts.writes);
        defs.extend(aliases.classes(&facts.mutates));
        let mandatory = if is_def(ast, stmt) {
            // A definition costs nothing per bar and may be called by anything the
            // slice keeps.
            true
        } else {
            // An early exit decides whether a later write block is reached.
            own(ast, stmt).into_iter().any(|sub| matches!(ast.kind(sub), Kind::Return { .. }))
        };
        index.push(StmtIndex {
            reads: aliases.classes(&facts.reads),
            defs,
            // The sids come from the same expansion as the names: a protocol call
            // inside a helper belongs to the STATEMENT THAT CALLS the helper, not to
            // the ``def`` that only binds it -- dropping the call site would leave
            // the write block unreached and the context silently unwritten.
            read_sids: facts.read_sids,
            write_sids: facts.write_sids,
            mandatory,
        });
    }
    index
}

/// Per alias class, the positions of the statements that define it (`_definers`).
pub(super) fn definers(index: &[StmtIndex]) -> HashMap<String, Vec<usize>> {
    let mut definers: HashMap<String, Vec<usize>> = HashMap::new();
    for (position, entry) in index.iter().enumerate() {
        for class in &entry.defs {
            definers.entry(class.clone()).or_default().push(position);
        }
    }
    definers
}

/// Backward slice of `main()`'s top-level statements for one unit (`_slice`).
///
/// A statement is kept when it defines an alias class a kept statement reads. Each
/// class is followed once, from the moment the first kept statement reads it, to
/// every statement defining it. Answers the sorted indexes of the kept statements,
/// or `None` when nothing was dropped.
pub(super) fn slice(
    index: &[StmtIndex],
    definers: &HashMap<String, Vec<usize>>,
    sids: &HashSet<String>,
    keep_reads: &HashSet<String>,
) -> Option<Vec<usize>> {
    let mut kept: HashSet<usize> = HashSet::new();
    for (position, entry) in index.iter().enumerate() {
        if entry.mandatory
            || !entry.write_sids.is_disjoint(sids)
            || !entry.read_sids.is_disjoint(keep_reads)
        {
            kept.insert(position);
        }
    }
    let mut needed: HashSet<&str> = HashSet::new();
    let mut work: Vec<usize> = kept.iter().copied().collect();
    while let Some(position) = work.pop() {
        for class in &index[position].reads {
            if !needed.insert(class) {
                continue;
            }
            for &definer in definers.get(class).into_iter().flatten() {
                if kept.insert(definer) {
                    work.push(definer);
                }
            }
        }
    }
    if kept.len() == index.len() {
        return None;
    }
    let mut kept: Vec<usize> = kept.into_iter().collect();
    kept.sort_unstable();
    Some(kept)
}
