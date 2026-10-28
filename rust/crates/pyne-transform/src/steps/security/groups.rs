//! What the step reads off the final tree to mark the contexts: `always_read`,
//! `signal_per_bar`, and the feed groups (`group`, `group_stable`).

use std::collections::{HashMap, HashSet};

use indexmap::IndexMap;
use pyne_ast::dump::dump;
use pyne_ast::{Ast, Constant, Int, Kind, NodeId};

use super::Security;
use super::builders::constant;
use super::collection::{
    function_body, function_name, is_function, is_simple_chain, walk_skip_funcs_body,
};
use super::lift::{
    MAX_LIFT_ROUNDS, SignalArgSubstituter, call_sid, can_exit, guard_block_calls, param_mapping,
    unconditional_exprs, unconditional_stmts,
};
use crate::steps::dynamic_default::is_script_entry;

/// Per function, the direct-`Name` calls of its own body as `(callee name, call)`
/// pairs, for every function in `walk_statements` order (`_own_calls`).
type OwnCalls = IndexMap<NodeId, Vec<(String, NodeId)>>;

/// A group key: the kind (`stable` / `hosted`) and the dumps that decide the feed.
type GroupKey = Vec<Option<String>>;

/// `ast.dump` of an expression.
fn dumped(ast: &Ast, node: NodeId) -> String {
    dump(ast, node, false)
}

/// The direct-`Name` calls of every function's own body (`_own_calls`).
fn own_calls(ast: &Ast, module: NodeId) -> OwnCalls {
    let mut found = OwnCalls::new();
    for function in pyne_ast::walk::walk_statements(ast, module) {
        if !is_function(ast.kind(function)) {
            continue;
        }
        let calls = walk_skip_funcs_body(ast, function)
            .into_iter()
            .filter_map(|sub| match ast.kind(sub) {
                Kind::Call { func, .. } => match ast.kind(*func) {
                    Kind::Name { id, .. } => Some((id.clone(), sub)),
                    _ => None,
                },
                _ => None,
            })
            .collect();
        found.insert(function, calls);
    }
    found
}

/// A unique key per function definition standing directly in a scope's statement
/// list: its dotted path, with an occurrence index where one scope defines a name
/// twice (`_function_keys`).
fn function_keys(ast: &Ast, module: NodeId) -> IndexMap<NodeId, String> {
    fn walk_keys(ast: &Ast, body: &[NodeId], prefix: &str, keys: &mut IndexMap<NodeId, String>) {
        let mut seen: HashMap<&str, usize> = HashMap::new();
        for &function in body {
            if !is_function(ast.kind(function)) {
                continue;
            }
            let name = function_name(ast, function);
            let count = seen.get(name).copied().unwrap_or(0);
            seen.insert(name, count + 1);
            let mut key = format!("{prefix}{name}");
            if count > 0 {
                key.push_str(&format!("#{count}"));
            }
            keys.insert(function, key.clone());
            walk_keys(ast, function_body(ast, function), &format!("{key}."), keys);
        }
    }
    let mut keys = IndexMap::new();
    let Kind::Module { body, .. } = ast.kind(module) else { unreachable!() };
    walk_keys(ast, body, "", &mut keys);
    keys
}

/// Where every `__sec_signal__` stands and with which arguments (`_signal_sites`):
/// sid -> (host function key, dump of each argument).
fn signal_sites(ast: &Ast, module: NodeId, own: &OwnCalls) -> HashMap<String, Vec<String>> {
    let mut sites = HashMap::new();
    for (function, key) in function_keys(ast, module) {
        for (name, call) in &own[&function] {
            if name != "__sec_signal__" {
                continue;
            }
            let Some(sid) = call_sid(ast, *call) else { continue };
            let Kind::Call { args, .. } = ast.kind(*call) else { unreachable!() };
            let mut site = vec![key.clone()];
            site.extend(args[1..].iter().map(|&arg| dumped(ast, arg)));
            sites.insert(sid.to_string(), site);
        }
    }
    sites
}

/// Every direct call of every name, as `(caller, call)` (`_call_sites_by_name`).
fn call_sites_by_name(own: &OwnCalls) -> HashMap<&str, Vec<(NodeId, NodeId)>> {
    let mut sites: HashMap<&str, Vec<(NodeId, NodeId)>> = HashMap::new();
    for (&caller, calls) in own {
        for (name, call) in calls {
            sites.entry(name.as_str()).or_default().push((caller, *call));
        }
    }
    sites
}

/// Every `__sec_signal__` with its host function and its arguments after the sid
/// (`_signal_call_sites`).
fn signal_call_sites(ast: &Ast, own: &OwnCalls) -> IndexMap<String, (NodeId, Vec<NodeId>)> {
    let mut found = IndexMap::new();
    for (&function, calls) in own {
        for (name, call) in calls {
            if name != "__sec_signal__" {
                continue;
            }
            if let Some(sid) = call_sid(ast, *call) {
                let Kind::Call { args, .. } = ast.kind(*call) else { unreachable!() };
                found.insert(sid.to_string(), (function, args[1..].to_vec()));
            }
        }
    }
    found
}

/// The str elements of a context's `depends` / `late_reads` list (`_sid_set`).
fn sid_set(ast: &Ast, entries: &IndexMap<String, NodeId>, key: &str) -> HashSet<String> {
    let Some(&list) = entries.get(key) else { return HashSet::new() };
    let Kind::List { elts, .. } = ast.kind(list) else { return HashSet::new() };
    elts.iter()
        .filter_map(|&elt| match ast.kind(elt) {
            Kind::Constant { value: Constant::Str(sid), .. } => Some(sid.clone()),
            _ => None,
        })
        .collect()
}

impl Security {
    /// The signal arguments of every context that cannot change during a bar,
    /// substituted up to the script entry (`_stable_signal_args`): sid -> the dump
    /// of each fully substituted argument.
    fn stable_signal_args(
        &mut self,
        ast: &mut Ast,
        own: &OwnCalls,
        load_counts: &HashMap<String, usize>,
    ) -> HashMap<String, Vec<String>> {
        let call_sites: HashMap<String, Vec<(NodeId, NodeId)>> = call_sites_by_name(own)
            .into_iter()
            .map(|(name, sites)| (name.to_string(), sites))
            .collect();
        let mut stable = HashMap::new();
        for (sid, (host, args)) in signal_call_sites(ast, own) {
            if let Some(dumps) = self.resolve_stable(ast, &call_sites, load_counts, host, args, 0) {
                stable.insert(sid, dumps);
            }
        }
        stable
    }

    /// The stable dumps of `args` as evaluated in `function`, `None` when they are
    /// not stable (`_stable_signal_args.resolve`).
    fn resolve_stable(
        &mut self,
        ast: &mut Ast,
        call_sites: &HashMap<String, Vec<(NodeId, NodeId)>>,
        load_counts: &HashMap<String, usize>,
        function: NodeId,
        args: Vec<NodeId>,
        depth: usize,
    ) -> Option<Vec<String>> {
        if depth > MAX_LIFT_ROUNDS {
            return None;
        }
        let (hoistable, stable_params) = self.lowered_hoistable(ast, function);
        let mut available: HashSet<String> = hoistable.keys().cloned().collect();
        available.extend(stable_params);
        if !args.iter().all(|&arg| is_simple_chain(ast, arg, &available)) {
            return None;
        }
        let name = function_name(ast, function).to_string();
        let loads = load_counts.get(&name).copied().unwrap_or(0);
        let sites = call_sites.get(&name).map(Vec::as_slice).unwrap_or_default();
        if sites.is_empty() {
            if loads > 0 {
                // Read without being called here: an alias or a callback, so how
                // its parameters are bound is unknown
                return None;
            }
            // Nobody calls it: the script entry
            return Some(args.iter().map(|&arg| dumped(ast, arg)).collect());
        }
        if sites.len() != loads {
            // Some read of the name is not one of these calls
            return None;
        }
        let bindings: IndexMap<String, NodeId> = hoistable
            .iter()
            .map(|(name, &stmt)| match ast.kind(stmt) {
                Kind::Assign { value, .. } => (name.clone(), *value),
                _ => unreachable!(),
            })
            .collect();
        let mut answers: HashSet<Vec<String>> = HashSet::new();
        for &(caller, call) in sites {
            if caller == function {
                return None;
            }
            let mapping = param_mapping(ast, function, call)?;
            let mut substituted = Vec::with_capacity(args.len());
            for &arg in &args {
                let mut substituter = SignalArgSubstituter::new(&mapping, &bindings);
                let new_arg = substituter.substitute(ast, arg);
                if substituter.failed {
                    return None;
                }
                substituted.push(new_arg);
            }
            let answer =
                self.resolve_stable(ast, call_sites, load_counts, caller, substituted, depth + 1)?;
            answers.insert(answer);
        }
        // Several call sites agree only when they pass the same thing
        if answers.len() == 1 { answers.into_iter().next() } else { None }
    }

    /// The compile-time context key of a sid, `None` when it may not group
    /// (`_group_key`), and whether the key is a stable one.
    fn group_key(
        &self,
        ast: &Ast,
        sid: &str,
        sites: &HashMap<String, Vec<String>>,
        stable_args: &HashMap<String, Vec<String>>,
    ) -> (Option<GroupKey>, bool) {
        if self.ltf_sec_ids.contains(sid) {
            return (None, false);
        }
        let entries = &self.all_contexts[sid];
        // A runtime lookahead travels in the signal args and is already part of
        // the key below; a module-level one is only in the context
        let [_, _, lookahead_rt] = self.signal_args[sid];
        let lookahead = if lookahead_rt.is_some() { None } else { entries.get("lookahead") };
        let dump_entry = |node: Option<&NodeId>| node.map(|&node| dumped(ast, node));
        // ``closed_shift`` decides the round protocol, and one child runs ONE
        // protocol for its whole group
        let tail = [
            dump_entry(lookahead),
            dump_entry(entries.get("ignore_invalid_symbol")),
            dump_entry(entries.get("closed_shift")),
        ];
        if let Some(args) = stable_args.get(sid) {
            let mut key = vec![Some("stable".to_string())];
            key.extend(args.iter().cloned().map(Some));
            key.extend(tail);
            return (Some(key), true);
        }
        let Some(site) = sites.get(sid) else { return (None, false) };
        if !self.top_sec_ids.contains(sid) {
            return (None, false);
        }
        let mut key = vec![Some("hosted".to_string())];
        key.extend(site.iter().cloned().map(Some));
        key.extend(tail);
        (Some(key), false)
    }

    /// Whether one child could answer every read of these contexts exactly
    /// (`_group_reads_are_unambiguous`): no context is both a dependency of one
    /// member and a late read of another, and no member is late-read by another.
    fn group_reads_are_unambiguous(&self, ast: &Ast, members: &[String]) -> bool {
        let mut depends = HashSet::new();
        let mut late = HashSet::new();
        for sid in members {
            depends.extend(sid_set(ast, &self.all_contexts[sid], "depends"));
            late.extend(sid_set(ast, &self.all_contexts[sid], "late_reads"));
        }
        depends.is_disjoint(&late) && members.iter().all(|sid| !late.contains(sid))
    }

    /// Write the `group` ordinal into every context (`_assign_groups`), and
    /// `group_stable` into every grouped one.
    pub(crate) fn assign_groups(
        &mut self,
        ast: &mut Ast,
        module: NodeId,
        load_counts: &HashMap<String, usize>,
    ) {
        let own = own_calls(ast, module);
        let sites = signal_sites(ast, module, &own);
        let stable_args = self.stable_signal_args(ast, &own, load_counts);
        let mut by_key: IndexMap<GroupKey, Vec<String>> = IndexMap::new();
        let mut stable_of_key: HashMap<GroupKey, bool> = HashMap::new();
        for sid in self.all_contexts.keys() {
            let (key, stable) = self.group_key(ast, sid, &sites, &stable_args);
            if let Some(key) = key {
                by_key.entry(key.clone()).or_default().push(sid.clone());
                stable_of_key.insert(key, stable);
            }
        }

        let mut group_of: HashMap<String, i64> = HashMap::new();
        let mut stable_of: HashMap<String, bool> = HashMap::new();
        let mut ordinal = 0;
        for (key, members) in &by_key {
            if members.len() < 2 || !self.group_reads_are_unambiguous(ast, members) {
                continue;
            }
            for sid in members {
                group_of.insert(sid.clone(), ordinal);
                stable_of.insert(sid.clone(), stable_of_key[key]);
            }
            ordinal += 1;
        }

        let sids: Vec<String> = self.all_contexts.keys().cloned().collect();
        for sid in sids {
            let group = match group_of.get(&sid) {
                Some(&ordinal) => Constant::Int(Int::Small(ordinal)),
                None => Constant::None,
            };
            let group = constant(ast, group);
            self.all_contexts[&sid].insert("group".into(), group);
            if let Some(&stable) = stable_of.get(&sid) {
                let flag = constant(ast, Constant::Bool(stable));
                self.all_contexts[&sid].insert("group_stable".into(), flag);
            }
        }
    }

    /// Flag every context whose `__sec_read__` runs on every chart bar
    /// (`_mark_always_read`): a read standing unconditionally in the script entry's
    /// body, or in a helper that body reaches unconditionally.
    pub(crate) fn mark_always_read(&mut self, ast: &mut Ast, module: NodeId) {
        let mut funcs: HashMap<String, NodeId> = HashMap::new();
        let mut ambiguous: HashSet<String> = HashSet::new();
        let mut entry = None;
        for node in pyne_ast::walk::walk_statements(ast, module) {
            if !is_function(ast.kind(node)) {
                continue;
            }
            let name = function_name(ast, node).to_string();
            if funcs.contains_key(&name) {
                // Two definitions of one name: a call by that name cannot be
                // attributed to either, so neither is followed
                ambiguous.insert(name.clone());
            }
            funcs.insert(name, node);
            if entry.is_none() && is_script_entry(ast, node) {
                entry = Some(node);
            }
        }
        let Some(entry) = entry else { return };

        let mut pending = vec![entry];
        let mut seen = HashSet::from([entry]);
        let mut flagged = Vec::new();
        while let Some(function) = pending.pop() {
            for stmt in unconditional_stmts(ast, function_body(ast, function)) {
                for sub in unconditional_exprs(ast, stmt) {
                    let Kind::Call { func, .. } = ast.kind(sub) else { continue };
                    let Kind::Name { id, .. } = ast.kind(*func) else { continue };
                    if id == "__sec_read__" {
                        if let Some(sid) = call_sid(ast, sub)
                            && self.all_contexts.contains_key(sid)
                        {
                            flagged.push(sid.to_string());
                        }
                        continue;
                    }
                    if let Some(&callee) = funcs.get(id)
                        && !ambiguous.contains(id)
                        && seen.insert(callee)
                    {
                        pending.push(callee);
                    }
                }
            }
        }
        for sid in flagged {
            let flag = constant(ast, Constant::Bool(true));
            self.all_contexts[&sid].insert("always_read".into(), flag);
        }
    }

    /// Flag every context signalled exactly once per script entry run
    /// (`_mark_signal_per_bar`): its only `__sec_signal__` stands in a chart-guard
    /// block directly in the entry's body, ahead of anything that can end it early.
    pub(crate) fn mark_signal_per_bar(
        &mut self,
        ast: &mut Ast,
        module: NodeId,
        counts: &HashMap<String, usize>,
    ) {
        let entry = pyne_ast::walk::walk_statements(ast, module)
            .into_iter()
            .find(|&node| is_function(ast.kind(node)) && is_script_entry(ast, node));
        let Some(entry) = entry else { return };

        let mut flagged = Vec::new();
        for &stmt in function_body(ast, entry) {
            if is_function(ast.kind(stmt)) || matches!(ast.kind(stmt), Kind::ClassDef { .. }) {
                continue;
            }
            if let Some(calls) = guard_block_calls(ast, stmt, "__sec_signal__") {
                for call in calls {
                    let Some(sid) = call_sid(ast, call) else { continue };
                    if counts.get(sid) != Some(&1) {
                        continue;
                    }
                    if self.all_contexts.contains_key(sid) {
                        flagged.push(sid.to_string());
                    }
                }
            }
            if can_exit(ast, stmt) {
                break;
            }
        }
        for sid in flagged {
            let flag = constant(ast, Constant::Bool(true));
            self.all_contexts[&sid].insert("signal_per_bar".into(), flag);
        }
    }
}
