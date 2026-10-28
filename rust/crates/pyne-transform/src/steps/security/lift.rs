//! The cross-function signal lift, and the reachability rules it shares with the
//! lowering and the group pass: which statements and expressions run every time a
//! body is entered, and the chart-guard blocks the step emits.

use std::collections::{HashMap, HashSet};

use indexmap::IndexMap;
use pyne_ast::locations::fix_locations;
use pyne_ast::walk::{
    Transformer, Visited, child_statements, children, clone, copy_location, docstring_end,
    generic_visit, walk,
};
use pyne_ast::{Ast, Category, CmpOperator, Constant, ExprContext, Kind, NodeId};

use super::Security;
use super::builders::{expr_stmt, func_call, guard_block, str_constant};
use super::collection::{
    arg_name, collect_hoisted, function_args, function_body, function_body_mut, function_name,
    is_function, is_simple_chain, referenced_names, str_value,
};

/// Lift rounds: each round moves a signal one call level up, so the cap only has
/// to exceed the deepest unconditional helper chain a script can have.
pub(crate) const MAX_LIFT_ROUNDS: usize = 16;

/// Statement forms that always run once when their body is entered
/// (`_UNCONDITIONAL_STMTS`).
fn is_unconditional_stmt(kind: &Kind) -> bool {
    matches!(
        kind,
        Kind::Expr { .. }
            | Kind::Assign { .. }
            | Kind::AugAssign { .. }
            | Kind::AnnAssign { .. }
            | Kind::Return { .. }
    )
}

/// Expression forms that may leave a subexpression unevaluated or defer it to
/// another invocation (`_CONDITIONAL_EXPRS`).
fn is_conditional_expr(kind: &Kind) -> bool {
    matches!(
        kind,
        Kind::IfExp { .. }
            | Kind::BoolOp { .. }
            | Kind::Lambda { .. }
            | Kind::ListComp { .. }
            | Kind::SetComp { .. }
            | Kind::DictComp { .. }
            | Kind::GeneratorExp { .. }
    )
}

/// Statements that can end the body early (`_EXIT_STMTS`).
fn is_exit_stmt(kind: &Kind) -> bool {
    matches!(kind, Kind::Return { .. } | Kind::Raise { .. } | Kind::Break | Kind::Continue)
}

fn is_definition(kind: &Kind) -> bool {
    is_function(kind) || matches!(kind, Kind::ClassDef { .. })
}

/// The statements of a body that run every time it is entered
/// (`_unconditional_stmts`): up to the first one that can end the body early,
/// nested definitions skipped.
pub(crate) fn unconditional_stmts(ast: &Ast, body: &[NodeId]) -> Vec<NodeId> {
    let mut out = Vec::new();
    for &stmt in body {
        if is_definition(ast.kind(stmt)) {
            continue;
        }
        if is_unconditional_stmt(ast.kind(stmt)) {
            out.push(stmt);
        }
        if can_exit(ast, stmt) {
            break;
        }
    }
    out
}

/// Whether a statement holds a statement that can end its body early (`_can_exit`).
/// Only statement lists are searched, and nested functions are not entered.
pub(crate) fn can_exit(ast: &Ast, stmt: NodeId) -> bool {
    let mut stack = vec![stmt];
    while let Some(node) = stack.pop() {
        if is_exit_stmt(ast.kind(node)) {
            return true;
        }
        for child in child_statements(ast, node) {
            if !is_function(ast.kind(child)) {
                stack.push(child);
            }
        }
    }
    false
}

/// Whether a child is an expression or a keyword, the nodes an unconditional
/// evaluation descends into.
fn is_expr_or_keyword(ast: &Ast, node: NodeId) -> bool {
    let kind = ast.kind(node);
    kind.class().category == Category::Expr || matches!(kind, Kind::Keyword { .. })
}

/// The expressions evaluated every time a statement is evaluated, in pre-order
/// (`_unconditional_exprs`): a form that may skip its operands is listed but not
/// entered; call arguments count as unconditional.
pub(crate) fn unconditional_exprs(ast: &Ast, stmt: NodeId) -> Vec<NodeId> {
    let mut reached = Vec::new();
    let mut stack: Vec<NodeId> = children(ast, stmt)
        .into_iter()
        .rev()
        .filter(|&child| is_expr_or_keyword(ast, child))
        .collect();
    while let Some(node) = stack.pop() {
        reached.push(node);
        if is_conditional_expr(ast.kind(node)) {
            continue;
        }
        stack.extend(
            children(ast, node).into_iter().rev().filter(|&child| is_expr_or_keyword(ast, child)),
        );
    }
    reached
}

/// The expressions evaluated every time a body is entered (`_reached_ids`).
pub(crate) fn reached_ids(ast: &Ast, body: &[NodeId]) -> HashSet<NodeId> {
    let mut reached = HashSet::new();
    for stmt in unconditional_stmts(ast, body) {
        reached.extend(unconditional_exprs(ast, stmt));
    }
    reached
}

/// The calls of `name` in a chart-guard block (`if __active_security__ is None:`
/// holding nothing but `name(...)` statements), `None` when the statement is no
/// such block (`_guard_block_calls`).
pub(crate) fn guard_block_calls(ast: &Ast, stmt: NodeId, name: &str) -> Option<Vec<NodeId>> {
    let Kind::If { test, body, orelse } = ast.kind(stmt) else { return None };
    if !orelse.is_empty() || body.is_empty() {
        return None;
    }
    let Kind::Compare { left, ops, comparators } = ast.kind(*test) else { return None };
    if ops[..] != [CmpOperator::Is]
        || !matches!(ast.kind(*left), Kind::Name { id, .. } if id == "__active_security__")
    {
        return None;
    }
    if !matches!(ast.kind(comparators[0]), Kind::Constant { value: Constant::None, .. }) {
        return None;
    }
    let mut calls = Vec::with_capacity(body.len());
    for &sub in body {
        let Kind::Expr { value } = ast.kind(sub) else { return None };
        let Kind::Call { func, .. } = ast.kind(*value) else { return None };
        if !matches!(ast.kind(*func), Kind::Name { id, .. } if id == name) {
            return None;
        }
        calls.push(*value);
    }
    Some(calls)
}

/// The first chart-guard block of `name` in a body, with its index (`_find_guard_block`).
pub(crate) fn find_guard_block(ast: &Ast, body: &[NodeId], name: &str) -> Option<(usize, NodeId)> {
    body.iter()
        .enumerate()
        .find(|&(_, &stmt)| guard_block_calls(ast, stmt, name).is_some())
        .map(|(index, &stmt)| (index, stmt))
}

/// The sid a `__sec_signal__` / `__sec_wait__` / `__sec_read__` call carries (`_call_sid`).
pub(crate) fn call_sid(ast: &Ast, call: NodeId) -> Option<&str> {
    let Kind::Call { args, .. } = ast.kind(call) else { return None };
    str_value(ast, *args.first()?)
}

/// Rewrites one lifted signal argument into the caller's scope
/// (`_SignalArgSubstituter`): a helper's parameter becomes the argument the caller
/// passed, a binding standing above the helper's signal block becomes its value
/// expression. A name with no such equivalent sets `failed`.
pub(crate) struct SignalArgSubstituter<'m> {
    mapping: &'m HashMap<String, NodeId>,
    bindings: &'m IndexMap<String, NodeId>,
    pub(crate) failed: bool,
    /// Guards a binding that (illegally) reads itself
    active: HashSet<String>,
}

impl<'m> SignalArgSubstituter<'m> {
    pub(crate) fn new(
        mapping: &'m HashMap<String, NodeId>,
        bindings: &'m IndexMap<String, NodeId>,
    ) -> Self {
        SignalArgSubstituter { mapping, bindings, failed: false, active: HashSet::new() }
    }

    /// The substituted copy of an argument.
    pub(crate) fn substitute(&mut self, ast: &mut Ast, arg: NodeId) -> NodeId {
        let copy = clone(ast, arg, &mut HashMap::new());
        self.visit_node(ast, copy)
    }

    fn visit_node(&mut self, ast: &mut Ast, node: NodeId) -> NodeId {
        match self.visit(ast, node) {
            Visited::Node(new) => new,
            _ => unreachable!("the substituter hands back a node"),
        }
    }
}

impl Transformer for SignalArgSubstituter<'_> {
    fn visit(&mut self, ast: &mut Ast, id: NodeId) -> Visited {
        let Kind::Name { id: name, .. } = ast.kind(id) else {
            generic_visit(self, ast, id);
            return Visited::Node(id);
        };
        if name == "lib" {
            return Visited::Node(id);
        }
        let name = name.clone();
        // Bindings come first: one of them may shadow a parameter, and then the
        // signal read the binding's value, not the argument the caller passed.
        if let Some(&value) = self.bindings.get(&name)
            && !self.active.contains(&name)
        {
            self.active.insert(name.clone());
            let copy = clone(ast, value, &mut HashMap::new());
            let new = self.visit_node(ast, copy);
            self.active.remove(&name);
            return Visited::Node(new);
        }
        if let Some(&value) = self.mapping.get(&name) {
            // The caller's expression is already in the target scope -- returned
            // unvisited so its own names are left alone.
            return Visited::Node(clone(ast, value, &mut HashMap::new()));
        }
        self.failed = true;
        Visited::Node(id)
    }
}

/// The `name = <expr>` bindings standing in `body[..limit]` (`_leading_bindings`),
/// `None` when that region is not such a shape: a statement that is no plain
/// single-name assignment, a name bound twice, a parameter rebound from anything
/// but a static expression, or a value reading a name bound below it.
pub(crate) fn leading_bindings(
    ast: &Ast,
    body: &[NodeId],
    limit: usize,
    params: &HashSet<String>,
) -> Option<IndexMap<String, NodeId>> {
    let mut assigned: Vec<(&str, NodeId)> = Vec::new();
    for &stmt in &body[..limit] {
        let Kind::Assign { targets, value, .. } = ast.kind(stmt) else { return None };
        let [target] = targets[..] else { return None };
        let Kind::Name { id, .. } = ast.kind(target) else { return None };
        if params.contains(id) && !referenced_names(ast, *value).is_empty() {
            return None;
        }
        assigned.push((id, *value));
    }

    let mut bindings = IndexMap::new();
    for (index, &(name, value)) in assigned.iter().enumerate() {
        if bindings.contains_key(name) {
            return None;
        }
        // Names bound BELOW this statement: a value reading one of them would be
        // substituted with that later value
        let later: HashSet<&str> = assigned[index + 1..].iter().map(|&(name, _)| name).collect();
        if referenced_names(ast, value).iter().any(|name| later.contains(name.as_str())) {
            return None;
        }
        bindings.insert(name.to_string(), value);
    }
    Some(bindings)
}

/// A function's parameters mapped to the expressions a call passes
/// (`_param_mapping`); `None` for any binding that is not statically decidable.
/// Parameters left to their default are absent.
pub(crate) fn param_mapping(
    ast: &Ast,
    function: NodeId,
    call: NodeId,
) -> Option<HashMap<String, NodeId>> {
    let Kind::Arguments { posonlyargs, args, vararg, kwonlyargs, kwarg, .. } =
        ast.kind(function_args(ast, function))
    else {
        unreachable!()
    };
    if vararg.is_some() || kwarg.is_some() {
        return None;
    }
    let Kind::Call { args: call_args, keywords, .. } = ast.kind(call) else { unreachable!() };
    if call_args.iter().any(|&arg| matches!(ast.kind(arg), Kind::Starred { .. })) {
        return None;
    }
    let keywords: Vec<(Option<&String>, NodeId)> = keywords
        .iter()
        .map(|&keyword| match ast.kind(keyword) {
            Kind::Keyword { arg, value } => (arg.as_ref(), *value),
            _ => unreachable!(),
        })
        .collect();
    if keywords.iter().any(|(arg, _)| arg.is_none()) {
        return None;
    }
    let positional: Vec<&str> =
        posonlyargs.iter().chain(args).map(|&arg| arg_name(ast, arg)).collect();
    let known: HashSet<&str> = positional
        .iter()
        .copied()
        .chain(kwonlyargs.iter().map(|&arg| arg_name(ast, arg)))
        .collect();
    if call_args.len() > positional.len() {
        return None;
    }
    let mut mapping = HashMap::new();
    for (index, &value) in call_args.iter().enumerate() {
        mapping.insert(positional[index].to_string(), value);
    }
    for (arg, value) in keywords {
        let arg = arg.expect("checked above");
        if !known.contains(arg.as_str()) || mapping.contains_key(arg) {
            return None;
        }
        mapping.insert(arg.clone(), value);
    }
    Some(mapping)
}

/// A lift candidate: helper, caller, call site, index of the statement it stands in.
type Candidate = (NodeId, NodeId, NodeId, usize);

impl Security {
    /// Helper functions whose signal block may move up one call level
    /// (`_lift_candidates`): the helper's name is read exactly once in the whole
    /// module, at a direct-`Name` call site evaluated unconditionally in an
    /// enclosing function's own statement list, with no statement ahead of it that
    /// can end that body early.
    fn lift_candidates(
        &self,
        ast: &Ast,
        module: NodeId,
        load_counts: &HashMap<String, usize>,
    ) -> Vec<Candidate> {
        let funcs: Vec<NodeId> = pyne_ast::walk::walk_statements(ast, module)
            .into_iter()
            .filter(|&node| is_function(ast.kind(node)))
            .collect();

        // Every unconditional direct-call site, keyed by callee name
        let mut sites: HashMap<&str, (NodeId, NodeId, usize)> = HashMap::new();
        for &caller in &funcs {
            for (index, &stmt) in function_body(ast, caller).iter().enumerate() {
                if is_definition(ast.kind(stmt)) {
                    // A definition neither calls nor exits this body
                    continue;
                }
                if is_unconditional_stmt(ast.kind(stmt)) {
                    for sub in unconditional_exprs(ast, stmt) {
                        if let Kind::Call { func, .. } = ast.kind(sub)
                            && let Kind::Name { id, .. } = ast.kind(*func)
                        {
                            sites.insert(id, (caller, sub, index));
                        }
                    }
                }
                // Anything beyond this statement may never be reached
                if can_exit(ast, stmt) {
                    break;
                }
            }
        }

        let mut candidates = Vec::new();
        for &helper in &funcs {
            if find_guard_block(ast, function_body(ast, helper), "__sec_signal__").is_none() {
                continue;
            }
            let name = function_name(ast, helper);
            if load_counts.get(name).copied().unwrap_or(0) != 1 {
                continue;
            }
            let Some(&(caller, call, index)) = sites.get(name) else { continue };
            if caller == helper {
                continue;
            }
            candidates.push((helper, caller, call, index));
        }
        candidates.sort_by_key(|candidate| candidate.3);
        candidates
    }

    /// Move every provably liftable signal of a helper into its caller
    /// (`_lift_signal`); true if at least one signal moved.
    fn lift_signal(&mut self, ast: &mut Ast, helper: NodeId, caller: NodeId, call: NodeId) -> bool {
        let Some((sig_index, sig_block)) =
            find_guard_block(ast, function_body(ast, helper), "__sec_signal__")
        else {
            return false;
        };
        let Some(mapping) = param_mapping(ast, helper, call) else { return false };
        let Kind::Arguments { posonlyargs, args, kwonlyargs, .. } =
            ast.kind(function_args(ast, helper))
        else {
            unreachable!()
        };
        let params: HashSet<String> = posonlyargs
            .iter()
            .chain(args)
            .chain(kwonlyargs)
            .map(|&arg| arg_name(ast, arg).to_string())
            .collect();
        let Some(bindings) = leading_bindings(ast, function_body(ast, helper), sig_index, &params)
        else {
            return false;
        };

        let (hoistable, stable_params) = self.lowered_hoistable(ast, caller);
        let mut available: HashSet<String> = hoistable.keys().cloned().collect();
        available.extend(stable_params);

        let mut lifted: Vec<(NodeId, String, Vec<NodeId>)> = Vec::new();
        let Kind::If { body: sig_body, .. } = ast.kind(sig_block) else { unreachable!() };
        for stmt in sig_body.clone() {
            let Kind::Expr { value } = ast.kind(stmt) else { continue };
            let value = *value;
            if !matches!(ast.kind(value), Kind::Call { .. }) {
                continue;
            }
            let Some(sid) = call_sid(ast, value).map(str::to_string) else { continue };
            let Kind::Call { args: signal_args, .. } = ast.kind(value) else { unreachable!() };
            let signal_args = signal_args[1..].to_vec();
            let mut new_args = Vec::new();
            let mut ok = true;
            for arg in signal_args {
                let mut substituter = SignalArgSubstituter::new(&mapping, &bindings);
                let new_arg = substituter.substitute(ast, arg);
                if substituter.failed || !is_simple_chain(ast, new_arg, &available) {
                    ok = false;
                    break;
                }
                new_args.push(new_arg);
            }
            if ok {
                lifted.push((stmt, sid, new_args));
            }
        }
        if lifted.is_empty() {
            return false;
        }

        let lifted_sids: HashSet<String> = lifted.iter().map(|(_, sid, _)| sid.clone()).collect();
        let mut needed: HashSet<String> = HashSet::new();
        for (_, _, new_args) in &lifted {
            for &arg in new_args {
                needed.extend(referenced_names(ast, arg));
            }
        }

        // The caller's signal block: reuse the one it already has, or start one at
        // the top of its body
        let caller_block = match find_guard_block(ast, function_body(ast, caller), "__sec_signal__")
        {
            Some((_, block)) => block,
            None => {
                let block = guard_block(ast, Vec::new());
                let at = docstring_end(ast, function_body(ast, caller));
                function_body_mut(ast, caller).insert(at, block);
                block
            }
        };

        for (stmt, sid, new_args) in lifted {
            remove_first(if_body_mut(ast, sig_block), stmt);
            let mut call_args = vec![str_constant(ast, &sid)];
            call_args.extend(new_args);
            let new_call = func_call(ast, "__sec_signal__", call_args);
            let new_stmt = expr_stmt(ast, new_call);
            copy_location(ast, new_stmt, call);
            if let Err(err) = fix_locations(ast, new_stmt, 1, 0) {
                self.error.get_or_insert(err);
            }
            if_body_mut(ast, caller_block).push(new_stmt);
        }
        if if_body(ast, sig_block).is_empty() {
            remove_first(function_body_mut(ast, helper), sig_block);
        }

        // Every binding the moved arguments read has to stand above the block
        let mut block_pos = position(function_body(ast, caller), caller_block);
        for stmt in collect_hoisted(ast, &hoistable, &needed) {
            let pos = position(function_body(ast, caller), stmt);
            if pos > block_pos {
                let body = function_body_mut(ast, caller);
                body.remove(pos);
                body.insert(block_pos, stmt);
                block_pos += 1;
            }
        }

        self.move_waits(ast, helper, caller, &lifted_sids, call);
        true
    }

    /// Follow the lifted signals with their `__sec_wait__` calls (`_move_waits`): the
    /// wait settles the round the signal launched, so it belongs to the end of
    /// whatever scope signalled.
    fn move_waits(
        &mut self,
        ast: &mut Ast,
        helper: NodeId,
        caller: NodeId,
        sids: &HashSet<String>,
        call: NodeId,
    ) {
        if let Some((_, wait_block)) =
            find_guard_block(ast, function_body(ast, helper), "__sec_wait__")
        {
            for stmt in if_body(ast, wait_block).clone() {
                let moved = match ast.kind(stmt) {
                    Kind::Expr { value } => {
                        matches!(ast.kind(*value), Kind::Call { .. })
                            && call_sid(ast, *value).is_some_and(|sid| sids.contains(sid))
                    }
                    _ => false,
                };
                if moved {
                    remove_first(if_body_mut(ast, wait_block), stmt);
                }
            }
            if if_body(ast, wait_block).is_empty() {
                remove_first(function_body_mut(ast, helper), wait_block);
            }
        }

        let caller_block = match find_guard_block(ast, function_body(ast, caller), "__sec_wait__") {
            Some((_, block)) => block,
            None => {
                let trailing = function_body(ast, caller).last().copied();
                if let Some(trailing) = trailing
                    && let Kind::Return { value } = ast.kind(trailing)
                {
                    if let Some(value) = value
                        && walk(ast, *value).contains(&call)
                    {
                        // The ``return`` itself runs the helper, and with it the
                        // reads: a wait in front of it would settle the round
                        // before the value is read. This scope has no statement
                        // after the reads, so the next signal and ``end_bar()``
                        // settle the round.
                        return;
                    }
                    let block = guard_block(ast, Vec::new());
                    copy_location(ast, block, call);
                    // Appended after the ``return``, the block would be dead code
                    let body = function_body_mut(ast, caller);
                    let at = body.len() - 1;
                    body.insert(at, block);
                    block
                } else {
                    let block = guard_block(ast, Vec::new());
                    copy_location(ast, block, call);
                    function_body_mut(ast, caller).push(block);
                    block
                }
            }
        };
        let present: HashSet<String> = guard_block_calls(ast, caller_block, "__sec_wait__")
            .unwrap_or_default()
            .into_iter()
            .filter_map(|wait| call_sid(ast, wait).map(str::to_string))
            .collect();
        let mut sorted: Vec<&String> = sids.iter().collect();
        sorted.sort();
        for sid in sorted {
            if present.contains(sid) {
                continue;
            }
            let sid = str_constant(ast, sid);
            let wait = func_call(ast, "__sec_wait__", vec![sid]);
            let new_stmt = expr_stmt(ast, wait);
            copy_location(ast, new_stmt, call);
            if let Err(err) = fix_locations(ast, new_stmt, 1, 0) {
                self.error.get_or_insert(err);
            }
            if_body_mut(ast, caller_block).push(new_stmt);
        }
    }

    /// Move helper-scoped `__sec_signal__` calls up to their caller, one call level
    /// per round (`_lift_helper_signals`); answers the module's census as the lift
    /// leaves it.
    pub(crate) fn lift_helper_signals(
        &mut self,
        ast: &mut Ast,
        module: NodeId,
    ) -> (HashMap<String, usize>, HashMap<String, usize>) {
        for _ in 0..MAX_LIFT_ROUNDS {
            let census = census(ast, module);
            let mut moved = false;
            for (helper, caller, call, _) in self.lift_candidates(ast, module, &census.0) {
                if self.lift_signal(ast, helper, caller, call) {
                    moved = true;
                }
            }
            if !moved {
                return census;
            }
        }
        census(ast, module)
    }
}

/// How often each name is read, and each sid signalled, in the module (`_census`).
pub(crate) fn census(
    ast: &Ast,
    module: NodeId,
) -> (HashMap<String, usize>, HashMap<String, usize>) {
    let mut loads: HashMap<String, usize> = HashMap::new();
    let mut signals: HashMap<String, usize> = HashMap::new();
    for sub in walk(ast, module) {
        match ast.kind(sub) {
            Kind::Name { id, ctx } => {
                if *ctx == ExprContext::Load {
                    *loads.entry(id.clone()).or_default() += 1;
                }
            }
            Kind::Call { func, .. } if matches!(ast.kind(*func), Kind::Name { id, .. } if id == "__sec_signal__") => {
                if let Some(sid) = call_sid(ast, sub) {
                    *signals.entry(sid.to_string()).or_default() += 1;
                }
            }
            _ => {}
        }
    }
    (loads, signals)
}

fn if_body(ast: &Ast, node: NodeId) -> &Vec<NodeId> {
    match ast.kind(node) {
        Kind::If { body, .. } => body,
        _ => unreachable!("no if"),
    }
}

fn if_body_mut(ast: &mut Ast, node: NodeId) -> &mut Vec<NodeId> {
    match ast.kind_mut(node) {
        Kind::If { body, .. } => body,
        _ => unreachable!("no if"),
    }
}

/// `list.remove(item)`: drop the first occurrence.
fn remove_first(list: &mut Vec<NodeId>, item: NodeId) {
    let index = position(list, item);
    list.remove(index);
}

/// `list.index(item)`.
fn position(list: &[NodeId], item: NodeId) -> usize {
    list.iter().position(|&node| node == item).expect("the statement stands in the list")
}
