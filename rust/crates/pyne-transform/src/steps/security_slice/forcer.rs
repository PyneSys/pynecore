//! Making the guarded writes of a clone unconditional (`_force_conditional_writes`,
//! `_Forcer`).
//!
//! TradingView HOISTS a `request.security()` call to global scope: the branch it is
//! written in decides where its RESULT lands on the chart, never whether the
//! requested context's series is computed. The security split leaves the
//! `__sec_write__` block where the call stood, so a call site inside a branch would
//! have the child re-evaluate that test against its own bars and publish on only
//! some of them. The clone is the child's own code, so the path is forced here:
//! every write, and every call of a helper performing one, is lifted in front of the
//! guards standing between it and the clone's body, inside the helper functions
//! first and then in the clone's own body. Branches the EXPRESSION depends on are
//! untouched.

use std::collections::{HashMap, HashSet};

use indexmap::IndexMap;
use pyne_ast::walk::{
    children, clone, copy_location, fix_missing_locations, walk, walk_statements,
};
use pyne_ast::{Ast, Category, ExprContext, FieldMut, FieldRef, Kind, NodeId};

use super::{
    Aliases, CLONE_PREFIX, VARIP_TYPES, contains_write, is_def, is_dispatch_test, module_body,
    protocol_sid, walk_own,
};
use crate::pine_type_rules::{SCALARS, get_ty};
use crate::steps::dynamic_default::is_script_entry;
use crate::steps::security::collection::{
    arg_name, function_args, function_body, function_body_mut, function_name,
};

/// The reserved separator of the names the transform injects
/// (`emission_settings.PYNE_RESERVED_NAME_CHAR`).
const PYNE_RESERVED_NAME_CHAR: char = '·';

/// Annotations of a declaration that owns per-call-site state
/// (`_STATEFUL_ANNOTATIONS`): running a copy of one would give the script a second,
/// independently advancing variable.
fn is_stateful_annotation(name: &str) -> bool {
    name == "Persistent" || name == "Series" || VARIP_TYPES.contains(&name)
}

/// The target test of the forcing: a `__sec_write__` call of one of the sids, or a
/// call of a helper whose call reaches one.
struct ForceTest<'s> {
    sids: &'s HashSet<String>,
    reaching: HashSet<String>,
}

impl ForceTest<'_> {
    fn hit(&self, ast: &Ast, node: NodeId) -> bool {
        // Both targets are calls, and ``hit`` holds for no other node
        let Kind::Call { func, .. } = ast.kind(node) else { return false };
        if protocol_sid(ast, node, "__sec_write__").is_some_and(|sid| self.sids.contains(sid)) {
            return true;
        }
        matches!(ast.kind(*func), Kind::Name { id, .. } if self.reaching.contains(id))
    }
}

/// A `__sec_write__` call of any of the sids (`_any_sid_write`).
fn any_sid_write(sids: &HashSet<String>) -> impl Fn(&Ast, NodeId) -> bool + '_ {
    move |ast, node| protocol_sid(ast, node, "__sec_write__").is_some_and(|sid| sids.contains(sid))
}

/// Make the writes of `sids` run on every bar the child computes
/// (`_force_conditional_writes`); answers whether anything was rewritten.
///
/// A writing helper defined at module level is shared with `main()`, so the clone
/// gets forced copies of the module-level definitions on the call chain as nested
/// `def`s, which shadow the originals inside the clone only.
pub(super) fn force_conditional_writes(
    ast: &mut Ast,
    module: NodeId,
    clone_id: NodeId,
    sids: &HashSet<String>,
) -> bool {
    let hit = any_sid_write(sids);
    let mut outer: IndexMap<String, Vec<NodeId>> = IndexMap::new();
    for &node in module_body(ast, module) {
        if is_def(ast, node)
            && !is_script_entry(ast, node)
            && !function_name(ast, node).starts_with(CLONE_PREFIX)
        {
            outer.entry(function_name(ast, node).to_string()).or_default().push(node);
        }
    }
    let mut nested = nested_defs(ast, clone_id);
    let shared: IndexMap<String, Vec<NodeId>> =
        outer.into_iter().filter(|(name, _)| !nested.contains_key(name)).collect();
    let reaching_shared = reaching_defs(ast, &shared, &hit);
    let mut names: Vec<&String> = reaching_shared.iter().collect();
    names.sort();
    let mut memo = HashMap::new();
    let mut copies = Vec::new();
    for name in names {
        for &node in &shared[name] {
            copies.push(clone(ast, node, &mut memo));
        }
    }

    // ``defs`` starts as a copy of ``nested`` whose lists ARE ``nested``'s: a
    // nested definition of a copy that shares a name with one of the clone's own
    // extends the clone's list, and is forced with the clone's definitions too.
    let mut defs = nested.clone();
    for &node in &copies {
        defs.entry(function_name(ast, node).to_string()).or_default().push(node);
        for (name, nodes) in nested_defs(ast, node) {
            defs.entry(name).or_default().extend(nodes);
        }
    }
    for (name, nodes) in nested.iter_mut() {
        nodes.clone_from(&defs[name]);
    }
    let reaching = reaching_defs(ast, &defs, &hit);
    let test = ForceTest { sids, reaching };

    let mut shared_forcer = Forcer::new(&test, plain_aliases(ast, &copies));
    for &node in &copies {
        let mut subs = vec![node];
        subs.extend(nested_defs(ast, node).into_values().flatten());
        for sub in subs {
            let body = shared_forcer.func_body(ast, sub);
            *function_body_mut(ast, sub) = body;
        }
    }
    let mut forcer = Forcer::new(&test, plain_aliases(ast, &[clone_id]));
    for nodes in nested.values() {
        for &node in nodes {
            let body = forcer.func_body(ast, node);
            *function_body_mut(ast, node) = body;
        }
    }
    let body = forcer.func_body(ast, clone_id);
    *function_body_mut(ast, clone_id) = body;
    if shared_forcer.changed {
        function_body_mut(ast, clone_id).splice(0..0, copies);
        // The clone comes located, and so does every node the forcing makes (each
        // takes the location of the statement it is lifted from); only the copied
        // module-level definitions may still lack one.
        fix_missing_locations(ast, clone_id);
    }
    forcer.changed || shared_forcer.changed
}

/// Names of the `def`s whose call can perform the target (`_reaching_defs`).
///
/// A direct writer holds the target in its own body; an indirect one calls a writer
/// by name. The closure is taken over the call graph the code spells out, so an
/// unresolvable callee simply leaves its caller out.
fn reaching_defs(
    ast: &Ast,
    defs: &IndexMap<String, Vec<NodeId>>,
    hit: &dyn Fn(&Ast, NodeId) -> bool,
) -> HashSet<String> {
    let mut reaching: HashSet<String> = defs
        .iter()
        .filter(|(_, nodes)| {
            nodes.iter().any(|&node| {
                function_body(ast, node)
                    .iter()
                    .any(|&stmt| !is_def(ast, stmt) && contains_write(ast, stmt, hit))
            })
        })
        .map(|(name, _)| name.clone())
        .collect();
    let callees: Vec<(&String, HashSet<&str>)> = defs
        .iter()
        .map(|(name, nodes)| {
            let mut called = HashSet::new();
            for &node in nodes {
                for sub in walk_own(ast, node) {
                    if let Kind::Call { func, .. } = ast.kind(sub)
                        && let Kind::Name { id, .. } = ast.kind(*func)
                    {
                        called.insert(id.as_str());
                    }
                }
            }
            (name, called)
        })
        .collect();
    for _ in 0..=defs.len() {
        let mut grown = false;
        for (name, called) in &callees {
            if !reaching.contains(*name) && called.iter().any(|callee| reaching.contains(*callee)) {
                reaching.insert((*name).clone());
                grown = true;
            }
        }
        if !grown {
            break;
        }
    }
    reaching
}

/// Every `def` standing anywhere inside `func`, by name (`_nested_defs`).
fn nested_defs(ast: &Ast, func: NodeId) -> IndexMap<String, Vec<NodeId>> {
    let mut defs: IndexMap<String, Vec<NodeId>> = IndexMap::new();
    for node in walk_statements(ast, func) {
        if is_def(ast, node) && node != func {
            defs.entry(function_name(ast, node).to_string()).or_default().push(node);
        }
    }
    defs
}

/// Whether running `stmt` performs the target, nested `def`s excluded (`_stmt_reaches`).
fn stmt_reaches(ast: &Ast, stmt: NodeId, test: &ForceTest<'_>) -> bool {
    !is_def(ast, stmt) && contains_write(ast, stmt, &|ast, node| test.hit(ast, node))
}

/// Expression shapes whose body runs at a time the forcing pass does not see
/// (`_DEFERRED_EXPRS`).
fn is_deferred(kind: &Kind) -> bool {
    matches!(
        kind,
        Kind::Lambda { .. }
            | Kind::ListComp { .. }
            | Kind::SetComp { .. }
            | Kind::DictComp { .. }
            | Kind::GeneratorExp { .. }
    )
}

/// Simple statements whose expressions are rewritten in place (`_FORCED_SIMPLE`).
fn is_forced_simple(kind: &Kind) -> bool {
    matches!(
        kind,
        Kind::Expr { .. }
            | Kind::Assign { .. }
            | Kind::AnnAssign { .. }
            | Kind::AugAssign { .. }
            | Kind::Return { .. }
    )
}

fn is_expr(ast: &Ast, node: NodeId) -> bool {
    ast.kind(node).class().category == Category::Expr
}

/// Rewrites a child's code so the targets run on every execution (`_Forcer`).
///
/// A user `if` stays where it is: only the statements that reach a target are
/// LIFTED in front of it, each behind copies of the plain assignments of the same
/// branch it reads. Everything else the branch does keeps running under its guard.
/// An expression-level guard (`IfExp`, short-circuit `BoolOp`) only ever stands
/// around a helper call, and is taken out. A loop, a `with`, a `try` and a `match`
/// are left as they are, and so is a lambda or a comprehension.
struct Forcer<'t, 's> {
    test: &'t ForceTest<'s>,
    aliases: Aliases,
    /// Whether anything was rewritten
    changed: bool,
    temps: usize,
    locals: HashSet<String>,
    certain: HashSet<String>,
}

impl<'t, 's> Forcer<'t, 's> {
    fn new(test: &'t ForceTest<'s>, aliases: Aliases) -> Self {
        Forcer {
            test,
            aliases,
            changed: false,
            temps: 0,
            locals: HashSet::new(),
            certain: HashSet::new(),
        }
    }

    fn reaches(&self, ast: &Ast, stmt: NodeId) -> bool {
        stmt_reaches(ast, stmt, self.test)
    }

    fn contains(&self, ast: &Ast, node: NodeId) -> bool {
        let test = self.test;
        contains_write(ast, node, &|ast, node| test.hit(ast, node))
    }

    /// One function's body, forced, with its local bindings tracked (`func_body`).
    ///
    /// A lift moves reads in front of the guard they were written under, so a local
    /// the guard itself is what binds would be read unbound on a pass that takes
    /// neither branch. A name bound nowhere in the function is a global, an import or
    /// a builtin and is always available, while every name the function binds is a
    /// local that has to be proven bound before the guard.
    fn func_body(&mut self, ast: &mut Ast, node: NodeId) -> Vec<NodeId> {
        let body = function_body(ast, node).clone();
        // Judged once per statement here: forcing one statement only ever rewrites
        // that statement, so whether a later one reaches a target cannot change on
        // the way.
        let reaches: Vec<bool> = body.iter().map(|&stmt| self.reaches(ast, stmt)).collect();
        if !reaches.iter().any(|&reach| reach) {
            // Nothing to force: the body comes back as it is.
            return body;
        }
        let Kind::Arguments { posonlyargs, args, vararg, kwonlyargs, kwarg, .. } =
            ast.kind(function_args(ast, node))
        else {
            unreachable!("no arguments")
        };
        let params: HashSet<String> = posonlyargs
            .iter()
            .chain(args)
            .chain(kwonlyargs)
            .chain(vararg)
            .chain(kwarg)
            .map(|&arg| arg_name(ast, arg).to_string())
            .collect();
        let mut locals = params.clone();
        for &stmt in &body {
            locals.extend(bound_names(ast, stmt));
        }
        let saved_locals = std::mem::replace(&mut self.locals, locals);
        let saved_certain = std::mem::replace(&mut self.certain, params);
        let out = self.body(ast, &body, Some(&reaches));
        self.locals = saved_locals;
        self.certain = saved_certain;
        out
    }

    /// The statement list with every reaching statement forced (`body`).
    fn body(&mut self, ast: &mut Ast, stmts: &[NodeId], reaches: Option<&[bool]>) -> Vec<NodeId> {
        let mut out = Vec::new();
        for (index, &stmt) in stmts.iter().enumerate() {
            let reach = match reaches {
                Some(reaches) => reaches[index],
                None => self.reaches(ast, stmt),
            };
            if reach {
                let replaced = self.stmt(ast, stmt);
                out.extend(replaced);
            } else {
                out.push(stmt);
            }
            let (bound, removed) = definite_bindings(ast, stmt);
            self.certain.retain(|name| !removed.contains(name));
            self.certain.extend(bound);
        }
        out
    }

    /// One reaching statement, as the statements that replace it (`stmt`).
    fn stmt(&mut self, ast: &mut Ast, stmt: NodeId) -> Vec<NodeId> {
        if matches!(ast.kind(stmt), Kind::If { .. }) {
            return self.if_stmt(ast, stmt);
        }
        if !is_forced_simple(ast.kind(stmt)) {
            return vec![stmt];
        }
        let mut hoisted = Vec::new();
        let count = ast.kind(stmt).class().fields.len();
        for index in 0..count {
            let child = match ast.kind(stmt).field(index) {
                FieldRef::Node(child) => *child,
                FieldRef::OptNode(Some(child)) => *child,
                _ => continue,
            };
            if !is_expr(ast, child) {
                continue;
            }
            let new = self.expr(ast, child, &mut hoisted);
            match ast.kind_mut(stmt).field_mut(index) {
                FieldMut::Node(slot) => *slot = new,
                FieldMut::OptNode(slot) => *slot = Some(new),
                _ => unreachable!(),
            }
        }
        let mut out = as_statements(ast, &hoisted, stmt);
        out.push(stmt);
        out
    }

    fn if_stmt(&mut self, ast: &mut Ast, stmt: NodeId) -> Vec<NodeId> {
        let mut hoisted = Vec::new();
        let test = if_parts(ast, stmt).0;
        if self.contains(ast, test) {
            let new = self.expr(ast, test, &mut hoisted);
            let Kind::If { test, .. } = ast.kind_mut(stmt) else { unreachable!() };
            *test = new;
        }
        let mut out = as_statements(ast, &hoisted, stmt);
        let (test, body, orelse) = if_parts(ast, stmt);
        let in_body = body.iter().any(|&sub| self.reaches(ast, sub));
        let in_else = orelse.iter().any(|&sub| self.reaches(ast, sub));
        // What is certainly bound in front of the ``if`` is what a lift out of either
        // branch may read; a branch's own bindings are undone for the other branch
        // and for everything after the statement.
        let outer = self.certain.clone();
        if is_dispatch_test(ast, test) || !(in_body || in_else) {
            // The protocol's own dispatch stays: the child always has its sid active.
            // A target in the test alone already runs on every pass.
            let new_body = self.body(ast, &body, None);
            set_if_branch(ast, stmt, false, new_body);
            self.certain = outer.clone();
            let orelse = if_parts(ast, stmt).2;
            let new_orelse = self.body(ast, &orelse, None);
            set_if_branch(ast, stmt, true, new_orelse);
            self.certain = outer;
            out.push(stmt);
            return out;
        }
        // Both branches may hold a target -- one context in each. TradingView
        // computes both series on every bar, so the child lifts both.
        for (is_else, branch) in [(false, body), (true, orelse)] {
            self.certain = outer.clone();
            let forced = self.body(ast, &branch, None);
            self.certain = outer.clone();
            let (kept, lifted) = self.lift(ast, &forced, &outer);
            if lifted.is_empty() {
                continue;
            }
            self.changed = true;
            out.extend(lifted);
            let replacement = if !kept.is_empty() {
                kept
            } else if !branch.is_empty() {
                let pass = ast.add(Kind::Pass);
                copy_location(ast, pass, stmt);
                vec![pass]
            } else {
                Vec::new()
            };
            set_if_branch(ast, stmt, is_else, replacement);
        }
        out.push(stmt);
        out
    }

    /// Split a branch into what stays guarded and what runs ahead of the guard
    /// (`_lift`).
    ///
    /// A reaching statement is lifted only when everything the branch does before it
    /// can run ahead of the guard unchanged: plain assignments of side-effect-free
    /// expressions, which the lift copies. Every local the lifted group reads must
    /// also be certainly bound in front of the guard, or bound by a copy of the group
    /// itself. A statement binding a name the branch reads or binds before it stays
    /// guarded too.
    fn lift(
        &mut self,
        ast: &mut Ast,
        stmts: &[NodeId],
        certain: &HashSet<String>,
    ) -> (Vec<NodeId>, Vec<NodeId>) {
        let mut kept: Vec<NodeId> = Vec::new();
        let mut lifted: Vec<NodeId> = Vec::new();
        let mut touched: HashSet<String> = HashSet::new();
        let mut used: HashSet<String> = HashSet::new();
        for &stmt in stmts {
            if !self.reaches(ast, stmt) {
                kept.push(stmt);
                used.extend(loaded_names(ast, stmt));
                used.extend(bound_names(ast, stmt));
                if !is_copyable_assignment(ast, stmt) {
                    touched.extend(operand_names(ast, stmt));
                }
                continue;
            }
            let mut feeding = feeding_copies(ast, stmt, &kept);
            if feeding.is_some() && !bound_names(ast, stmt).is_disjoint(&used) {
                // The statement would overtake an earlier read or write of a name it
                // binds.
                feeding = None;
            }
            let mut reads = operand_names(ast, stmt);
            let mut loaded = loaded_names(ast, stmt);
            let mut provided = certain.clone();
            for &copied in feeding.iter().flatten() {
                reads.extend(operand_names(ast, copied));
                loaded.extend(loaded_names(ast, copied));
                provided.extend(bound_names(ast, copied));
            }
            if feeding.is_some()
                && loaded.iter().any(|name| self.locals.contains(name) && !provided.contains(name))
            {
                // A local only the branch binds would be read unbound ahead of the
                // guard.
                feeding = None;
            }
            if feeding.is_some()
                && !self.aliases.classes(&touched).is_disjoint(&self.aliases.classes(&reads))
            {
                // An earlier effect of the branch -- a collection mutation, a call on
                // a script value -- may be what the statement reads.
                feeding = None;
            }
            let Some(feeding) = feeding else {
                kept.push(stmt);
                used.extend(loaded_names(ast, stmt));
                used.extend(bound_names(ast, stmt));
                touched.extend(operand_names(ast, stmt));
                continue;
            };
            self.detach(ast, &feeding, stmt);
            lifted.extend(feeding);
            lifted.push(stmt);
        }
        (kept, lifted)
    }

    /// Rebind the copied dependencies onto private names (`_detach`).
    ///
    /// The originals stay under the guard, so a copy keeping its own target would
    /// write that name ahead of the `if`. The copies therefore bind names of their
    /// own, and only the lifted statement follows them.
    fn detach(&mut self, ast: &mut Ast, feeding: &[NodeId], stmt: NodeId) {
        let mut renamed: IndexMap<String, String> = IndexMap::new();
        for &copied in feeding {
            let mut names: Vec<String> = bound_names(ast, copied).into_iter().collect();
            names.sort();
            for name in names {
                if !renamed.contains_key(&name) {
                    renamed.insert(
                        name,
                        format!("__sec_dep{PYNE_RESERVED_NAME_CHAR}{}__", self.temps),
                    );
                    self.temps += 1;
                }
            }
        }
        if renamed.is_empty() {
            return;
        }
        // A namespace root is never bound by a copy, so it is never renamed
        for &copied in feeding {
            for node in walk(ast, copied) {
                if let Kind::Name { id, .. } = ast.kind_mut(node)
                    && let Some(new) = renamed.get(id.as_str())
                {
                    *id = new.clone();
                }
            }
        }
        for node in walk(ast, stmt) {
            if let Kind::Name { id, ctx: ExprContext::Load } = ast.kind_mut(node)
                && let Some(new) = renamed.get(id.as_str())
            {
                *id = new.clone();
            }
        }
    }

    /// `node` with every guard around a reaching call removed (`expr`).
    ///
    /// An `IfExp` becomes the branch that reaches a target, and a guarded `BoolOp`
    /// operand is taken out of the chain. Whatever reaches a target but cannot stay
    /// in the expression is handed to `hoisted` and runs as a statement of its own
    /// ahead of the rewritten one.
    fn expr(&mut self, ast: &mut Ast, node: NodeId, hoisted: &mut Vec<NodeId>) -> NodeId {
        if is_deferred(ast.kind(node)) || !self.contains(ast, node) {
            return node;
        }
        match ast.kind(node) {
            &Kind::IfExp { test, .. } => {
                let new = self.expr(ast, test, hoisted);
                let Kind::IfExp { test, body, orelse } = ast.kind_mut(node) else { unreachable!() };
                *test = new;
                let (body, orelse) = (*body, *orelse);
                let in_body = self.contains(ast, body);
                let in_else = self.contains(ast, orelse);
                if !(in_body || in_else) {
                    return node;
                }
                self.changed = true;
                let body = self.expr(ast, body, hoisted);
                let orelse = self.expr(ast, orelse, hoisted);
                if in_body && in_else {
                    hoisted.push(orelse);
                }
                if in_body { body } else { orelse }
            }
            Kind::BoolOp { values, .. } => {
                let values = values.clone();
                let values: Vec<NodeId> =
                    values.into_iter().map(|value| self.expr(ast, value, hoisted)).collect();
                let mut kept = vec![values[0]];
                for &value in &values[1..] {
                    if self.contains(ast, value) {
                        hoisted.push(value);
                        self.changed = true;
                    } else {
                        kept.push(value);
                    }
                }
                if kept.len() == 1 {
                    return kept[0];
                }
                let Kind::BoolOp { values, .. } = ast.kind_mut(node) else { unreachable!() };
                *values = kept;
                node
            }
            _ => {
                let count = ast.kind(node).class().fields.len();
                for index in 0..count {
                    match ast.kind(node).field(index) {
                        FieldRef::Node(child) | FieldRef::OptNode(Some(child)) => {
                            let child = *child;
                            if !is_expr(ast, child) {
                                continue;
                            }
                            let new = self.expr(ast, child, hoisted);
                            match ast.kind_mut(node).field_mut(index) {
                                FieldMut::Node(slot) => *slot = new,
                                FieldMut::OptNode(slot) => *slot = Some(new),
                                _ => unreachable!(),
                            }
                        }
                        FieldRef::Nodes(items) => {
                            let items = items.clone();
                            let new: Vec<NodeId> = items
                                .into_iter()
                                .map(|item| {
                                    if is_expr(ast, item) {
                                        self.expr(ast, item, hoisted)
                                    } else {
                                        item
                                    }
                                })
                                .collect();
                            let FieldMut::Nodes(slot) = ast.kind_mut(node).field_mut(index) else {
                                unreachable!()
                            };
                            *slot = new;
                        }
                        FieldRef::OptNodes(items) => {
                            let items = items.clone();
                            let new: Vec<Option<NodeId>> = items
                                .into_iter()
                                .map(|item| match item {
                                    Some(item) if is_expr(ast, item) => {
                                        Some(self.expr(ast, item, hoisted))
                                    }
                                    other => other,
                                })
                                .collect();
                            let FieldMut::OptNodes(slot) = ast.kind_mut(node).field_mut(index)
                            else {
                                unreachable!()
                            };
                            *slot = new;
                        }
                        _ => {}
                    }
                }
                if let Kind::Call { keywords, .. } = ast.kind(node) {
                    for keyword in keywords.clone() {
                        let Kind::Keyword { value, .. } = ast.kind(keyword) else { unreachable!() };
                        let new = self.expr(ast, *value, hoisted);
                        let Kind::Keyword { value, .. } = ast.kind_mut(keyword) else {
                            unreachable!()
                        };
                        *value = new;
                    }
                }
                node
            }
        }
    }
}

/// An `if` statement's test, body and orelse.
fn if_parts(ast: &Ast, stmt: NodeId) -> (NodeId, Vec<NodeId>, Vec<NodeId>) {
    let Kind::If { test, body, orelse } = ast.kind(stmt) else { unreachable!("no if") };
    (*test, body.clone(), orelse.clone())
}

fn set_if_branch(ast: &mut Ast, stmt: NodeId, is_else: bool, stmts: Vec<NodeId>) {
    let Kind::If { body, orelse, .. } = ast.kind_mut(stmt) else { unreachable!("no if") };
    *(if is_else { orelse } else { body }) = stmts;
}

/// Expression statements running `exprs`, located at `at` (`_as_statements`).
fn as_statements(ast: &mut Ast, exprs: &[NodeId], at: NodeId) -> Vec<NodeId> {
    exprs
        .iter()
        .map(|&value| {
            let stmt = ast.add(Kind::Expr { value });
            copy_location(ast, stmt, at);
            stmt
        })
        .collect()
}

/// Every name the node reads (`_loaded_names`).
fn loaded_names(ast: &Ast, node: NodeId) -> HashSet<String> {
    walk(ast, node)
        .into_iter()
        .filter_map(|sub| match ast.kind(sub) {
            Kind::Name { id, ctx: ExprContext::Load } => Some(id.clone()),
            _ => None,
        })
        .collect()
}

/// Statements whose bindings only happen on some paths through them
/// (`_CONDITIONAL_STMT`).
fn is_conditional_stmt(kind: &Kind) -> bool {
    matches!(
        kind,
        Kind::If { .. }
            | Kind::For { .. }
            | Kind::AsyncFor { .. }
            | Kind::While { .. }
            | Kind::Try { .. }
            | Kind::TryStar { .. }
            | Kind::With { .. }
            | Kind::AsyncWith { .. }
            | Kind::Match { .. }
    )
}

/// The name an import alias binds (`(alias.asname or alias.name).split('.')[0]`).
fn alias_name(ast: &Ast, alias: NodeId) -> String {
    let Kind::Alias { name, asname } = ast.kind(alias) else { unreachable!("no alias") };
    let name = asname.as_ref().unwrap_or(name);
    name.split('.').next().unwrap_or(name).to_string()
}

/// What the statement certainly gives a value to, and what it may unbind
/// (`_definite_bindings`).
///
/// A name is local even when the statement that makes it one gives it no value
/// (`original: float`) or takes its value away (`del original`), and neither may let
/// a lift read it ahead of a guard. What binds is therefore listed by statement
/// kind, and only outside a compound statement. Deletions are subtracted wherever
/// they sit; an `except E as name` handler counts as one.
fn definite_bindings(ast: &Ast, stmt: NodeId) -> (HashSet<String>, HashSet<String>) {
    let mut nodes = vec![stmt];
    nodes.extend(walk_own(ast, stmt));
    let mut removed: HashSet<String> = HashSet::new();
    for &sub in &nodes {
        match ast.kind(sub) {
            Kind::Name { id, ctx: ExprContext::Del } => {
                removed.insert(id.clone());
            }
            Kind::ExceptHandler { name: Some(name), .. } => {
                removed.insert(name.clone());
            }
            _ => {}
        }
    }
    if is_conditional_stmt(ast.kind(stmt)) {
        return (HashSet::new(), removed);
    }
    let mut bound: HashSet<String> = HashSet::new();
    let targets: Vec<NodeId> = match ast.kind(stmt) {
        Kind::Assign { targets, .. } => targets.clone(),
        Kind::AnnAssign { target, value: Some(_), .. } | Kind::AugAssign { target, .. } => {
            vec![*target]
        }
        Kind::FunctionDef { name, .. }
        | Kind::AsyncFunctionDef { name, .. }
        | Kind::ClassDef { name, .. } => {
            bound.insert(name.clone());
            Vec::new()
        }
        Kind::Import { names } | Kind::ImportFrom { names, .. } => {
            bound.extend(names.iter().map(|&alias| alias_name(ast, alias)));
            Vec::new()
        }
        _ => Vec::new(),
    };
    for target in targets {
        for sub in walk(ast, target) {
            if let Kind::Name { id, ctx: ExprContext::Store } = ast.kind(sub) {
                bound.insert(id.clone());
            }
        }
    }
    bound.retain(|name| !removed.contains(name));
    (bound, removed)
}

/// Every local name the statement may bind, nested statements included
/// (`_bound_names`). Nested function definitions bind only their own name here.
fn bound_names(ast: &Ast, stmt: NodeId) -> HashSet<String> {
    let mut names = HashSet::new();
    let mut nodes = vec![stmt];
    nodes.extend(walk_own(ast, stmt));
    for node in nodes {
        match ast.kind(node) {
            Kind::Name { id, ctx: ExprContext::Store | ExprContext::Del } => {
                names.insert(id.clone());
            }
            Kind::FunctionDef { name, .. }
            | Kind::AsyncFunctionDef { name, .. }
            | Kind::ClassDef { name, .. } => {
                names.insert(name.clone());
            }
            Kind::Alias { .. } => {
                names.insert(alias_name(ast, node));
            }
            Kind::ExceptHandler { name: Some(name), .. } => {
                names.insert(name.clone());
            }
            _ => {}
        }
    }
    for child in children(ast, stmt) {
        if is_def(ast, child) {
            names.insert(function_name(ast, child).to_string());
        }
    }
    names
}

/// Whether the expression may be evaluated ahead of the guard it sits under
/// (`_is_repeatable`).
///
/// An allowlist: a constant, a bare name and an attribute chain rooted in the
/// compiler-owned `lib` binding. Everything else is rejected, since the guard is
/// usually what makes the expression valid at all: a call may advance state or draw,
/// a subscript needs the index the guard establishes, an attribute of a script value
/// the owner, and no operator can be shown safe without the operand types and
/// values.
fn is_repeatable(ast: &Ast, node: NodeId) -> bool {
    let mut nodes = vec![node];
    nodes.extend(walk(ast, node));
    nodes.into_iter().all(|sub| match ast.kind(sub) {
        // The context object an attribute or a name carries is walked as a node of
        // its own, and only a load is accepted
        Kind::Attribute { value, ctx, .. } => {
            *ctx == ExprContext::Load && super::chain_root_name(ast, *value) == Some("lib")
        }
        Kind::Name { ctx, .. } => *ctx == ExprContext::Load,
        Kind::Constant { .. } => true,
        _ => false,
    })
}

/// Union-find over the bare names a scope may make denote one object
/// (`_plain_aliases`): `b = a`, annotated or plain. The names are taken unqualified,
/// so two scopes spelling the same name share a class -- that only makes the lift
/// more conservative, never less.
fn plain_aliases(ast: &Ast, roots: &[NodeId]) -> Aliases {
    let mut aliases = Aliases::default();
    for &root in roots {
        // Both spellings are statements, so no expression is entered.
        for node in walk_statements(ast, root) {
            let (targets, value) = match ast.kind(node) {
                Kind::Assign { targets, value, .. } => (targets.clone(), Some(*value)),
                Kind::AnnAssign { target, value, .. } => (vec![*target], *value),
                _ => continue,
            };
            let Some(Kind::Name { id: source, .. }) = value.map(|value| ast.kind(value)) else {
                continue;
            };
            for target in targets {
                if let Kind::Name { id, .. } = ast.kind(target) {
                    aliases.union(id, source);
                }
            }
        }
    }
    aliases
}

/// The script values the node touches, the compiler namespace left out
/// (`_operand_names`).
///
/// Only `lib` is dropped: it holds nothing a statement of the branch can mutate. Any
/// other attribute owner is a value of the script itself. A name stamped with a
/// scalar Pine type is dropped as well: it holds an immutable value no call can
/// change in place.
fn operand_names(ast: &Ast, node: NodeId) -> HashSet<String> {
    let nodes = walk(ast, node);
    let mut roots: HashSet<NodeId> = HashSet::new();
    for &sub in &nodes {
        if let Kind::Attribute { value, .. } = ast.kind(sub) {
            let mut base = *value;
            while let Kind::Attribute { value, .. } = ast.kind(base) {
                base = *value;
            }
            if matches!(ast.kind(base), Kind::Name { id, .. } if id == "lib") {
                roots.insert(base);
            }
        }
    }
    nodes
        .into_iter()
        .filter_map(|sub| match ast.kind(sub) {
            Kind::Name { id, .. }
                if !roots.contains(&sub) && !SCALARS.contains(&get_ty(ast, sub)) =>
            {
                Some(id.clone())
            }
            _ => None,
        })
        .collect()
}

/// Whether the statement may run a second time ahead of its own guard
/// (`_is_copyable_assignment`): only a plain assignment of a repeatable expression.
fn is_copyable_assignment(ast: &Ast, stmt: NodeId) -> bool {
    let targets = plain_assignment_targets(ast, stmt);
    let value = match ast.kind(stmt) {
        Kind::Assign { value, .. } => Some(*value),
        Kind::AnnAssign { value, .. } => *value,
        _ => None,
    };
    let Some(value) = value else { return false };
    if targets.is_empty() || !is_repeatable(ast, value) {
        return false;
    }
    bound_names(ast, stmt).is_subset(&targets)
}

/// Copies of the branch statements the reaching statement reads (`_feeding_copies`),
/// in source order; `None` when a dependency cannot run ahead of the guard.
fn feeding_copies(ast: &mut Ast, stmt: NodeId, kept: &[NodeId]) -> Option<Vec<NodeId>> {
    let mut needed = loaded_names(ast, stmt);
    let mut feeding = Vec::new();
    for &earlier in kept.iter().rev() {
        if bound_names(ast, earlier).is_disjoint(&needed) {
            continue;
        }
        if !is_copyable_assignment(ast, earlier) {
            return None;
        }
        feeding.push(clone(ast, earlier, &mut HashMap::new()));
        needed.extend(loaded_names(ast, earlier));
    }
    feeding.reverse();
    // The originals stay under the guard and run again after the copies, so a copy
    // must not read a name the group itself binds at or after it: ``length = length
    // + 1`` would advance twice, and ``a = b`` followed by ``b = a + 1`` would
    // re-read the already advanced ``b``.
    for (index, &earlier) in feeding.iter().enumerate() {
        let mut later_targets: HashSet<String> = HashSet::new();
        for &other in &feeding[index..] {
            later_targets.extend(bound_names(ast, other));
        }
        if !later_targets.is_disjoint(&loaded_names(ast, earlier)) {
            return None;
        }
    }
    Some(feeding)
}

/// The local names a plain assignment binds, or nothing for anything else
/// (`_plain_assignment_targets`). An augmented assignment or a persistent / series
/// declaration advances state, so a lifted statement reads whatever value that state
/// has under its own guard.
fn plain_assignment_targets(ast: &Ast, stmt: NodeId) -> HashSet<String> {
    let targets = match ast.kind(stmt) {
        Kind::Assign { targets, .. } => targets.clone(),
        Kind::AnnAssign { target, annotation, value: Some(_), .. } => {
            let stateful = walk(ast, *annotation).into_iter().any(
                |node| matches!(ast.kind(node), Kind::Name { id, .. } if is_stateful_annotation(id)),
            );
            if stateful {
                return HashSet::new();
            }
            vec![*target]
        }
        _ => return HashSet::new(),
    };
    let mut names = HashSet::new();
    for target in targets {
        // Only a bare name: an attribute or a subscript target writes through an
        // object, and a tuple / list / starred one unpacks -- ``x, y = items`` raises
        // ``ValueError`` as soon as the guard above it is what gives ``items`` the
        // matching length.
        let Kind::Name { id, .. } = ast.kind(target) else { return HashSet::new() };
        names.insert(id.clone());
    }
    names
}
