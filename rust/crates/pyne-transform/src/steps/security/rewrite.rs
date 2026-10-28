//! The per-function lowering: each security call of a function's own body gets its
//! context, its signal (in the top block or inline), its write block and its read.

use std::collections::{HashMap, HashSet};

use pyne_ast::locations::anchor_all_at;
use pyne_ast::walk::{
    Transformer, Visited, child_statements, clone, docstring_end, generic_visit, visit_field,
};
use pyne_ast::{Ast, Constant, FieldRef, Int, Kind, NodeId, Value};

use super::builders::{
    constant, default_gaps, lib_na, sec_read_call, sec_read_call_ltf, str_constant, write_block,
};
use super::collection::{
    binding_counts, collect_hoisted, extract_args, extract_ltf_args, find_forbidden_strategy_state,
    function_body, function_body_mut, hoistable_bindings, is_function, is_module_level_expr,
    is_simple_chain, ohlcv_passthrough, referenced_names, resolved_at_runtime, scope_bindings,
    walk_skip_funcs_body, walk_skip_funcs_post,
};
use super::lift::reached_ids;
use super::predicates::{is_security_call, is_security_lower_tf_call};
use super::{Entries, Security};
use crate::Error;

/// Compound statements whose statement lists [`Security::recurse_subbodies`]
/// transforms on their own.
fn is_sub_body_stmt(kind: &Kind) -> bool {
    matches!(
        kind,
        Kind::If { .. }
            | Kind::For { .. }
            | Kind::AsyncFor { .. }
            | Kind::While { .. }
            | Kind::With { .. }
            | Kind::AsyncWith { .. }
            | Kind::Try { .. }
            | Kind::TryStar { .. }
            | Kind::Match { .. }
    )
}

/// The statement-list fields, or the handlers and cases holding statements.
const SUB_BODY_FIELDS: [&str; 5] = ["body", "orelse", "finalbody", "handlers", "cases"];

/// A statement and its handlers / cases: the nodes whose non-body fields are a
/// compound statement's own parts.
fn holders(ast: &Ast, stmt: NodeId) -> Vec<NodeId> {
    let mut out = vec![stmt];
    match ast.kind(stmt) {
        Kind::Try { handlers, .. } | Kind::TryStar { handlers, .. } => out.extend(handlers),
        Kind::Match { cases, .. } => out.extend(cases),
        _ => {}
    }
    out
}

/// The field indices of a holder that are no statement list.
fn own_fields(ast: &Ast, holder: NodeId) -> Vec<usize> {
    ast.kind(holder)
        .class()
        .fields
        .iter()
        .enumerate()
        .filter(|(_, field)| !SUB_BODY_FIELDS.contains(&field.name))
        .map(|(index, _)| index)
        .collect()
}

/// The children of a statement the sub-body recursion does not transform, nested
/// definitions excluded (`_own_parts`).
fn own_parts(ast: &Ast, stmt: NodeId) -> Vec<NodeId> {
    if !is_sub_body_stmt(ast.kind(stmt)) {
        return pyne_ast::walk::children(ast, stmt)
            .into_iter()
            .filter(|&child| !is_function(ast.kind(child)))
            .collect();
    }
    let mut out = Vec::new();
    for holder in holders(ast, stmt) {
        for index in own_fields(ast, holder) {
            match ast.kind(holder).field(index) {
                FieldRef::Node(child) => out.push(*child),
                FieldRef::OptNode(child) => out.extend(*child),
                FieldRef::Nodes(items) => out.extend(items),
                FieldRef::OptNodes(items) => out.extend(items.iter().flatten()),
                _ => {}
            }
        }
    }
    out
}

/// Replaces the marked security calls with their reads (`_CallReplacer`).
struct CallReplacer<'s> {
    ltf_sec_ids: &'s HashSet<String>,
}

/// An arity stamp of a marked call; this step alone writes them, as small ints.
fn int_extra(ast: &Ast, node: NodeId, name: &str) -> Option<i64> {
    match ast.extra(node, name) {
        Some(Value::Int(Int::Small(value))) => Some(*value),
        _ => None,
    }
}

impl Transformer for CallReplacer<'_> {
    fn visit(&mut self, ast: &mut Ast, id: NodeId) -> Visited {
        generic_visit(self, ast, id);
        if !matches!(ast.kind(id), Kind::Call { .. }) {
            return Visited::Node(id);
        }
        let Some(Value::Str(sec_id)) = ast.extra(id, "_sec_id") else {
            return Visited::Node(id);
        };
        let sec_id = sec_id.clone();
        if self.ltf_sec_ids.contains(&sec_id) {
            let arity = int_extra(ast, id, "_ltf_arity");
            return Visited::Node(sec_read_call_ltf(ast, &sec_id, arity));
        }
        let tuple_len = int_extra(ast, id, "_tuple_len");
        Visited::Node(sec_read_call(ast, &sec_id, tuple_len))
    }
}

impl CallReplacer<'_> {
    fn replace(&mut self, ast: &mut Ast, node: NodeId) -> NodeId {
        match self.visit(ast, node) {
            Visited::Node(new) => new,
            _ => unreachable!("the replacer hands back a node"),
        }
    }

    /// Replace the marked calls of a statement outside its transformed sub-bodies
    /// (`_replace_own_calls`).
    fn replace_own_calls(&mut self, ast: &mut Ast, stmt: NodeId) -> NodeId {
        if !is_sub_body_stmt(ast.kind(stmt)) {
            return self.replace(ast, stmt);
        }
        for holder in holders(ast, stmt) {
            for index in own_fields(ast, holder) {
                visit_field(self, ast, holder, index);
            }
        }
        stmt
    }
}

/// If a statement is a tuple-unpack assignment whose value is exactly `call`, the
/// target's arity (`_detect_tuple_arity`). Pine enforces target arity == value
/// arity for a tuple-returning `request.security()`, so the target is authoritative.
fn detect_tuple_arity(ast: &Ast, stmt: NodeId, call: NodeId) -> Option<i64> {
    let Kind::Assign { targets, value, .. } = ast.kind(stmt) else { return None };
    let [target] = targets[..] else { return None };
    let (Kind::Tuple { elts, .. } | Kind::List { elts, .. }) = ast.kind(target) else {
        return None;
    };
    if elts.iter().any(|&elt| matches!(ast.kind(elt), Kind::Starred { .. })) {
        return None;
    }
    if *value != call {
        return None;
    }
    Some(elts.len() as i64)
}

/// The named statement-list field of a node, when it has one.
fn list_field_index(ast: &Ast, node: NodeId, name: &str) -> Option<usize> {
    let kind = ast.kind(node);
    let index = kind.class().fields.iter().position(|field| field.name == name)?;
    matches!(kind.field(index), FieldRef::Nodes(_)).then_some(index)
}

fn take_list(ast: &mut Ast, node: NodeId, index: usize) -> Vec<NodeId> {
    match ast.kind_mut(node).field_mut(index) {
        pyne_ast::FieldMut::Nodes(items) => std::mem::take(items),
        _ => unreachable!(),
    }
}

fn put_list(ast: &mut Ast, node: NodeId, index: usize, items: Vec<NodeId>) {
    match ast.kind_mut(node).field_mut(index) {
        pyne_ast::FieldMut::Nodes(slot) => *slot = items,
        _ => unreachable!(),
    }
}

impl Security {
    /// Recursively transform a statement list (`_transform_body`): write blocks go
    /// before the statements holding security calls (an inline signal ahead of a
    /// runtime-resolved one), and the calls become reads.
    fn transform_body(
        &mut self,
        ast: &mut Ast,
        body: Vec<NodeId>,
        call_exprs: &HashMap<String, NodeId>,
        runtime_sec_ids: &HashSet<String>,
    ) -> Vec<NodeId> {
        let mut new_body = Vec::with_capacity(body.len());
        for stmt in body {
            if is_function(ast.kind(stmt)) {
                new_body.push(stmt);
                continue;
            }
            self.recurse_subbodies(ast, stmt, call_exprs, runtime_sec_ids);

            // Post-order: a call nested in another call's expression gets its write
            // block FIRST. Its producer then precedes the outer context in program
            // order, so the outer context's child waits for it and pairs its value
            // instead of answering the read with the default. The sub-bodies are
            // done (no marked call is left in them), so only the statement's own
            // parts are searched.
            let call_nodes_here: Vec<NodeId> = own_parts(ast, stmt)
                .into_iter()
                .flat_map(|part| walk_skip_funcs_post(ast, part))
                .filter(|&node| {
                    matches!(ast.kind(node), Kind::Call { .. })
                        && ast.extra(node, "_sec_id").is_some()
                })
                .collect();
            if call_nodes_here.is_empty() {
                new_body.push(stmt);
                continue;
            }
            for &call in &call_nodes_here {
                let Some(arity) = detect_tuple_arity(ast, stmt, call) else { continue };
                if ast.extra(call, "_tuple_len").is_none() {
                    ast.set_extra(call, "_tuple_len", Value::Int(Int::Small(arity)));
                }
                // LTF call with an opaque tuple expression (a function call, not a
                // literal): its own arity is unknowable here, but Pine enforces
                // target arity == value arity, so the unpack target is just as
                // authoritative -- wire the __ltf_unzip__ wrap from it.
                if self.ltf_sec_ids.contains(sid_of(ast, call))
                    && ast.extra(call, "_ltf_arity").is_none()
                {
                    ast.set_extra(call, "_ltf_arity", Value::Int(Int::Small(arity)));
                    self.needs_ltf_unzip = true;
                }
            }
            let mut replacer = CallReplacer { ltf_sec_ids: &self.ltf_sec_ids };
            // The blocks emitted for a sid, located at its call once the replacer is done
            let mut emitted: Vec<(NodeId, String)> = Vec::new();
            for &call in &call_nodes_here {
                let sid = sid_of(ast, call).to_string();
                if runtime_sec_ids.contains(&sid) {
                    let signal = self.inline_signal(ast, &sid);
                    emitted.push((signal, sid.clone()));
                }
                let expr = clone(ast, call_exprs[&sid], &mut HashMap::new());
                let expr = replacer.replace(ast, expr);
                emitted.push((write_block(ast, &sid, expr), sid));
            }
            let own = replacer.replace_own_calls(ast, stmt);
            for (block, sid) in emitted {
                new_body.push(self.at_call(ast, block, &sid));
            }
            new_body.push(own);
        }
        new_body
    }

    /// Transform the statement lists of a compound statement (`_recurse_subbodies`).
    fn recurse_subbodies(
        &mut self,
        ast: &mut Ast,
        stmt: NodeId,
        call_exprs: &HashMap<String, NodeId>,
        runtime_sec_ids: &HashSet<String>,
    ) {
        // Each statement list in source order: the holders' (handlers' / cases')
        // bodies stand between a ``try``'s body and its ``else``
        let lists: Vec<(NodeId, &str)> = match ast.kind(stmt) {
            Kind::If { .. } | Kind::For { .. } | Kind::AsyncFor { .. } | Kind::While { .. } => {
                vec![(stmt, "body"), (stmt, "orelse")]
            }
            Kind::With { .. } | Kind::AsyncWith { .. } => vec![(stmt, "body")],
            Kind::Try { handlers, .. } | Kind::TryStar { handlers, .. } => {
                let mut lists = vec![(stmt, "body")];
                lists.extend(handlers.iter().map(|&handler| (handler, "body")));
                lists.extend([(stmt, "orelse"), (stmt, "finalbody")]);
                lists
            }
            Kind::Match { cases, .. } => cases.iter().map(|&case| (case, "body")).collect(),
            _ => return,
        };
        for (node, name) in lists {
            self.transform_list(ast, node, name, call_exprs, runtime_sec_ids);
        }
    }

    fn transform_list(
        &mut self,
        ast: &mut Ast,
        node: NodeId,
        name: &str,
        call_exprs: &HashMap<String, NodeId>,
        runtime_sec_ids: &HashSet<String>,
    ) {
        let index = list_field_index(ast, node, name).expect("a statement list");
        let body = take_list(ast, node, index);
        let body = self.transform_body(ast, body, call_exprs, runtime_sec_ids);
        put_list(ast, node, index, body);
    }

    /// Transform a function with its own scope on the scope stack (`_process_func`).
    fn process_func(&mut self, ast: &mut Ast, node: NodeId) -> Result<(), Error> {
        let own_nodes = walk_skip_funcs_body(ast, node);
        self.scopes.push(scope_bindings(ast, node, &own_nodes)?);
        let result = self.process_func_body(ast, node, &own_nodes);
        self.scopes.pop();
        result
    }

    /// Lower the security calls of a function's own body (`_process_func_body`).
    fn process_func_body(
        &mut self,
        ast: &mut Ast,
        node: NodeId,
        own_nodes: &[NodeId],
    ) -> Result<(), Error> {
        let calls = self.collect_calls(ast, own_nodes);
        if calls.is_empty() {
            return self.visit_nested_defs(ast, node);
        }

        let mut call_exprs: HashMap<String, NodeId> = HashMap::new();
        let mut sec_ids: Vec<String> = Vec::new();
        for (call, sec_id, is_ltf) in &calls {
            let (call, is_ltf) = (*call, *is_ltf);
            let (symbol, timeframe, expression, gaps, lookahead, ignore_invalid, currency);
            if is_ltf {
                [symbol, timeframe, expression, ignore_invalid] = extract_ltf_args(ast, call);
                (gaps, lookahead, currency) = (None, None, None);
                // A tuple/list expression makes security_lower_tf() return one
                // array per element (column-major): the read is wrapped in
                // __ltf_unzip__
                if let Some(expression) = expression
                    && let Kind::Tuple { elts, .. } | Kind::List { elts, .. } = ast.kind(expression)
                {
                    let arity = elts.len() as i64;
                    ast.set_extra(call, "_ltf_arity", Value::Int(Int::Small(arity)));
                    self.needs_ltf_unzip = true;
                }
            } else {
                [symbol, timeframe, expression, gaps, lookahead, ignore_invalid, currency] =
                    extract_args(ast, call);
                // A tuple/list expression pins the arity at the call itself, which
                // the unpack target cannot show when the call is not the direct
                // value of an assignment
                if let Some(expression) = expression
                    && let Kind::Tuple { elts, .. } | Kind::List { elts, .. } = ast.kind(expression)
                {
                    let arity = elts.len() as i64;
                    ast.set_extra(call, "_tuple_len", Value::Int(Int::Small(arity)));
                }
            }

            if let Some(expression) = expression
                && let Some(bad) = find_forbidden_strategy_state(ast, expression)
            {
                let Kind::Attribute { attr, .. } = ast.kind(bad) else { unreachable!() };
                let function =
                    if is_ltf { "request.security_lower_tf" } else { "request.security" };
                return Err(Error::Rejected(format!(
                    "'strategy.{attr}' cannot be used as the expression argument of \
                     {function}() \u{2014} strategy state is only available in the chart \
                     context, not in a security context."
                )));
            }

            let expr = match expression {
                Some(expression) => expression,
                None => lib_na(ast),
            };
            call_exprs.insert(sec_id.clone(), expr);

            // Input-derived (Pine "simple") lookahead -- e.g. the standard TV
            // non-repaint HTF pattern ``repaint ? lookahead_off : lookahead_on``
            // -- cannot be evaluated at module level, so it is resolved at runtime
            // through __sec_signal__ like a deferred symbol/timeframe.
            let lookahead_rt = match lookahead {
                Some(lookahead) if !is_module_level_expr(ast, lookahead) => {
                    Some(clone(ast, lookahead, &mut HashMap::new()))
                }
                _ => None,
            };
            let symbol_copy = symbol.map(|symbol| clone(ast, symbol, &mut HashMap::new()));
            let timeframe_copy = timeframe.map(|tf| clone(ast, tf, &mut HashMap::new()));
            self.signal_args.insert(sec_id.clone(), [symbol_copy, timeframe_copy, lookahead_rt]);

            // For __security_contexts__ at module level: only values evaluable at
            // module scope; function parameters etc. become None
            let mut entries = Entries::new();
            for (key, value) in [("symbol", symbol), ("timeframe", timeframe)] {
                let Some(value) = value else { continue };
                let entry = if is_module_level_expr(ast, value) {
                    clone(ast, value, &mut HashMap::new())
                } else {
                    constant(ast, Constant::None)
                };
                entries.insert(key.into(), entry);
            }
            if is_ltf {
                entries.insert("is_ltf".into(), constant(ast, Constant::Bool(true)));
            } else {
                let gaps_expr = match gaps {
                    Some(gaps) => clone(ast, gaps, &mut HashMap::new()),
                    None => default_gaps(ast),
                };
                entries.insert("gaps".into(), gaps_expr);
                if let Some(lookahead) = lookahead {
                    // A module-level lookahead is consumed at module load to wire
                    // the HTF transport; an input-derived one would NameError at
                    // import time, so None stands in for it there
                    let entry = match lookahead_rt {
                        None => clone(ast, lookahead, &mut HashMap::new()),
                        Some(_) => constant(ast, Constant::None),
                    };
                    entries.insert("lookahead".into(), entry);
                }
                // Developing rounds inside one HTF period are skippable when the
                // whole expression is a ``[k>=1]`` history reference under
                // ``lookahead_on``. Emitted for every non-LTF context; the runtime
                // decides whether the skip applies.
                let shift = self.closed_shift(ast, expression, lookahead);
                entries.insert("closed_shift".into(), constant(ast, Constant::Bool(shift)));
            }
            if let Some(ignore_invalid) = ignore_invalid {
                let entry = clone(ast, ignore_invalid, &mut HashMap::new());
                entries.insert("ignore_invalid_symbol".into(), entry);
            }
            if !is_ltf && let Some(currency) = currency {
                let entry = clone(ast, currency, &mut HashMap::new());
                entries.insert("currency".into(), entry);
            }
            // Plain-OHLCV fast path: the security child serves the expression
            // straight from each bar without running main()
            if let Some((fields, is_tuple)) = ohlcv_passthrough(ast, expression) {
                let elts = fields.iter().map(|field| str_constant(ast, field)).collect();
                let list = ast.add(Kind::List { elts, ctx: pyne_ast::ExprContext::Load });
                entries.insert("ohlcv_fields".into(), list);
                entries.insert("ohlcv_tuple".into(), constant(ast, Constant::Bool(is_tuple)));
            }

            // Whether barmerge is used (only for non-LTF)
            if !is_ltf {
                match gaps {
                    None => self.needs_barmerge = true,
                    Some(gaps) => {
                        if let Kind::Attribute { value, .. } = ast.kind(gaps)
                            && matches!(ast.kind(*value), Kind::Attribute { attr, .. } if attr == "barmerge")
                        {
                            self.needs_barmerge = true;
                        }
                    }
                }
                if let Some(lookahead) = lookahead
                    && let Kind::Attribute { value, .. } = ast.kind(lookahead)
                    && matches!(ast.kind(*value), Kind::Attribute { attr, .. } if attr == "barmerge")
                {
                    self.needs_barmerge = true;
                }
            }

            self.all_contexts.insert(sec_id.clone(), entries);
            self.sid_call.insert(sec_id.clone(), call);
            sec_ids.push(sec_id.clone());
        }

        // Separate top-block signals from runtime-dependent ones. A signal can
        // start at function entry when every argument is a Pine "simple" chain:
        // constants, lib.* values, a parameter the body never rebinds, or a local
        // bound once from such a chain (that binding is then hoisted above the
        // signal block). A context whose symbol or timeframe is only known at
        // runtime moves up only when its call is reached every time the function
        // runs, or when another context depends on it (``keep_top``).
        let counts = binding_counts(ast, node);
        let (hoistable, stable_params) = hoistable_bindings(ast, node, &counts);
        let mut available: HashSet<String> = hoistable.keys().cloned().collect();
        available.extend(stable_params);
        let reached_nodes = reached_ids(ast, function_body(ast, node));
        let reached: HashSet<&String> = calls
            .iter()
            .filter(|(call, _, _)| reached_nodes.contains(call))
            .map(|(_, sid, _)| sid)
            .collect();
        let mut top_sec_ids: Vec<String> = Vec::new();
        let mut runtime_sec_ids: HashSet<String> = HashSet::new();
        let mut needed_names: HashSet<String> = HashSet::new();
        for sid in &sec_ids {
            let [symbol, timeframe, _] = self.signal_args[sid];
            let runtime_resolved = resolved_at_runtime(ast, symbol, timeframe);
            if runtime_resolved && !reached.contains(sid) && !self.keep_top.contains(sid) {
                self.deferred_by_reach.insert(sid.clone());
                runtime_sec_ids.insert(sid.clone());
                continue;
            }
            let exprs: Vec<NodeId> = self.signal_args[sid].iter().flatten().copied().collect();
            if exprs.iter().all(|&expr| is_simple_chain(ast, expr, &available)) {
                top_sec_ids.push(sid.clone());
                for expr in exprs {
                    needed_names.extend(referenced_names(ast, expr));
                }
            } else {
                runtime_sec_ids.insert(sid.clone());
            }
        }
        self.top_sec_ids.extend(top_sec_ids.iter().cloned());

        let hoisted = collect_hoisted(ast, &hoistable, &needed_names);
        let hoisted_ids: HashSet<NodeId> = hoisted.iter().copied().collect();
        let mut original_body: Vec<NodeId> = function_body(ast, node)
            .iter()
            .copied()
            .filter(|stmt| !hoisted_ids.contains(stmt))
            .collect();
        let rest = original_body.split_off(docstring_end(ast, &original_body));
        let mut new_body = original_body;
        // The top block runs at function entry, or right after the bindings hoisted
        // for it: it belongs to that point, whatever lines its arguments were cloned
        // from
        let anchor = hoisted.last().copied().unwrap_or(node);
        new_body.extend(hoisted);
        if !top_sec_ids.is_empty() {
            let block = self.signal_block(ast, &top_sec_ids);
            if let Err(err) = anchor_all_at(ast, block, anchor) {
                self.error.get_or_insert(err);
            }
            new_body.push(block);
        }
        new_body.extend(self.transform_body(ast, rest, &call_exprs, &runtime_sec_ids));
        let wait = self.wait_block(ast, &sec_ids);
        new_body.push(wait);
        *function_body_mut(ast, node) = new_body;

        self.visit_nested_defs(ast, node)
    }

    /// Process the function definitions standing in a node's statements
    /// (`_visit_nested_defs`): a definition only ever stands in a statement list,
    /// and a definition nested in another one is reached through that one's own
    /// processing.
    pub(crate) fn visit_nested_defs(&mut self, ast: &mut Ast, node: NodeId) -> Result<(), Error> {
        let mut stack: Vec<std::vec::IntoIter<NodeId>> =
            vec![child_statements(ast, node).into_iter()];
        while let Some(top) = stack.last_mut() {
            let Some(child) = top.next() else {
                stack.pop();
                continue;
            };
            if is_function(ast.kind(child)) {
                self.process_func(ast, child)?;
            } else {
                stack.push(child_statements(ast, child).into_iter());
            }
        }
        Ok(())
    }

    /// Whether lowering the module can defer a signal by reach (`_may_defer`): a
    /// call whose symbol or timeframe is resolved at runtime and that is not reached
    /// every time its function runs. Decided on the untransformed tree, which is
    /// what every function body still is when its turn comes.
    pub(crate) fn may_defer(&self, ast: &Ast, module: NodeId) -> bool {
        for function in pyne_ast::walk::walk_statements(ast, module) {
            if !is_function(ast.kind(function)) {
                continue;
            }
            let mut deferrable = Vec::new();
            for node in walk_skip_funcs_body(ast, function) {
                let (symbol, timeframe) = if is_security_call(ast, node) {
                    let args = extract_args(ast, node);
                    (args[0], args[1])
                } else if is_security_lower_tf_call(ast, node) {
                    let args = extract_ltf_args(ast, node);
                    (args[0], args[1])
                } else {
                    continue;
                };
                if resolved_at_runtime(ast, symbol, timeframe) {
                    deferrable.push(node);
                }
            }
            if !deferrable.is_empty() {
                let reached = reached_ids(ast, function_body(ast, function));
                if deferrable.iter().any(|call| !reached.contains(call)) {
                    return true;
                }
            }
        }
        false
    }
}

/// The sid a marked call carries.
fn sid_of(ast: &Ast, call: NodeId) -> &str {
    match ast.extra(call, "_sec_id") {
        Some(Value::Str(sid)) => sid,
        _ => unreachable!("a marked call"),
    }
}
