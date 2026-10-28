//! Definitions and their per-call-site contexts.

use std::collections::HashMap;
use std::rc::Rc;

use indexmap::IndexMap;
use pyne_ast::walk::child_statements;
use pyne_ast::{Kind, NodeId, Unsupported};

use super::facts::{self, arg_parts, every_param, spells_series};
use super::{Anchor, Engine, Frame, R};
use crate::pine_qualifier::{CONST, NOT_READ, SERIES};
use crate::pine_type_rules::{TYPELESS, UNKNOWN, VOID, join, overload_result, param_defaults};
use crate::pine_type_table::{Binding, ContextKey, ContextResult};
use crate::steps::dynamic_default::is_script_entry;

impl Engine<'_, '_> {
    /// Type a definition's defaults, and decide when its body is walked (`_definition`).
    pub(super) fn definition(&mut self, node: NodeId) -> R<()> {
        let key = crate::pine_type_table::qualify(&self.scope(), facts::def_name(self.ast, node));
        self.definition_keyed(node, key)
    }

    /// [`Self::definition`] of a definition whose id is not its name in this scope
    /// (a method, keyed under its class).
    pub(super) fn definition_keyed(&mut self, node: NodeId, key: String) -> R<()> {
        self.type_defaults(node, None)?;
        if self.overload_groups.contains(&key) || is_script_entry(self.ast, node) {
            self.declaration_context(&key, node)?;
            return Ok(());
        }
        self.pending.last_mut().expect("a pending list").push((key, node));
        Ok(())
    }

    /// Walk the definitions of this scope no call site ever typed (`_flush_pending`).
    pub(super) fn flush_pending(&mut self) -> R<()> {
        let waiting = std::mem::take(self.pending.last_mut().expect("a pending list"));
        let mut remaining: Vec<(String, NodeId, Rc<std::collections::HashSet<String>>)> =
            waiting.into_iter().map(|(key, node)| (key, node, self.called_names(node))).collect();
        let mut callers: Vec<i64> = remaining
            .iter()
            .enumerate()
            .map(|(index, (_, node, _))| {
                let name = facts::def_name(self.ast, *node);
                remaining
                    .iter()
                    .enumerate()
                    .filter(|(other, (_, _, names))| *other != index && names.contains(name))
                    .count() as i64
            })
            .collect();
        while !remaining.is_empty() {
            let picked = callers.iter().position(|&count| count == 0).unwrap_or(0);
            let (key, node, names) = remaining.remove(picked);
            callers.remove(picked);
            for (index, (_, waiting, _)) in remaining.iter().enumerate() {
                if names.contains(facts::def_name(self.ast, *waiting)) {
                    callers[index] -= 1;
                }
            }
            let walked = self.nid(node).is_some_and(|nid| self.walked.contains(&nid));
            if !walked {
                self.declaration_context(&key, node)?;
            }
        }
        Ok(())
    }

    /// The parameter types a definition states on its own (`_declared_params`).
    pub(super) fn declared_params(&self, node: NodeId) -> R<Vec<String>> {
        let defaults =
            if is_script_entry(self.ast, node) { param_defaults(self.ast, node)? } else { vec![] };
        let mut out = Vec::new();
        for arg in every_param(self.ast, node) {
            let (name, annotation) = arg_parts(self.ast, arg);
            let mut ty = self.annotation_type(annotation)?;
            if ty == UNKNOWN {
                let default = defaults.iter().find(|(known, _)| known == name).map(|(_, d)| *d);
                ty = default.map_or(UNKNOWN.to_string(), |default| self.ty_of(default));
            }
            out.push(ty);
        }
        Ok(out)
    }

    /// The parameter qualifiers a definition states on its own (`_declared_quals`).
    fn declared_quals(&self, node: NodeId) -> R<Vec<u8>> {
        let defaults =
            if is_script_entry(self.ast, node) { param_defaults(self.ast, node)? } else { vec![] };
        Ok(every_param(self.ast, node)
            .into_iter()
            .map(|arg| {
                let name = arg_parts(self.ast, arg).0;
                match defaults.iter().find(|(known, _)| known == name) {
                    None => SERIES,
                    Some((_, default)) => self.qual(*default),
                }
            })
            .collect())
    }

    /// Type a definition's defaults, which belong to the enclosing scope
    /// (`_type_defaults`).
    pub(super) fn type_defaults(&mut self, node: NodeId, frames: Option<Vec<usize>>) -> R<()> {
        let Some(args) = facts::args_of(self.ast, node) else { return Ok(()) };
        let defaults: Vec<NodeId> = args.defaults.to_vec();
        let kw_defaults: Vec<NodeId> = args.kw_defaults.iter().flatten().copied().collect();
        let outer = frames.map(|frames| std::mem::replace(&mut self.stack, frames));
        let walked = (|| {
            for default in defaults {
                self.expr(default)?;
            }
            for default in kw_defaults {
                self.expr(default)?;
            }
            Ok(())
        })();
        if let Some(outer) = outer {
            self.stack = outer;
        }
        walked
    }

    /// Analyse a body in the only context its own definition describes
    /// (`_declaration_context`).
    pub(super) fn declaration_context(&mut self, key: &str, node: NodeId) -> R<Option<usize>> {
        let params = self.declared_params(node)?;
        let quals = self.declared_quals(node)?;
        self.analyse(key, node, params, None, None, Some(quals))
    }

    /// Walk one function body under one parameter-type tuple (`_analyse`); the index
    /// of its result, `None` when it is re-entrant or over budget.
    pub(super) fn analyse(
        &mut self,
        key: &str,
        node: NodeId,
        params: Vec<String>,
        parents: Option<Vec<usize>>,
        site: Option<NodeId>,
        quals: Option<Vec<u8>>,
    ) -> R<Option<usize>> {
        let env = parents.unwrap_or_else(|| self.stack.clone());
        let nid = self.nid(node);
        let origin = if key.contains('·') { self.context() } else { 0 };
        let quals = match quals {
            Some(quals) => quals,
            None => self.declared_quals(node)?,
        };
        let memo = ContextKey {
            key: key.to_string(),
            params: params.clone(),
            origin,
            def: nid,
            env: self.free_env(node, &env),
            quals: quals.clone(),
        };
        if let Some(&found) = self.contexts.get(&memo) {
            return Ok(Some(found));
        }
        let guard = (key.to_string(), params.clone(), nid);
        if self.in_progress.contains(&guard) || self.contexts.len() >= super::MAX_CONTEXTS {
            return Ok(None);
        }
        let anchor: Anchor =
            (key.to_string(), params.clone(), origin, nid, site.and_then(|s| self.nid(s)));
        let cid = self.supersede(anchor, memo.clone())?;
        let mut result = ContextResult::new(cid, key, params.clone());
        result.quals = quals.clone();
        self.results.push(result);
        let index = self.results.len() - 1;
        self.contexts.insert(memo, index);
        self.in_progress.insert(guard.clone());
        let facts = self.scope_facts(node);
        let frame = Frame {
            scope: key.to_string(),
            declared_global: facts.declared_global.clone(),
            declared_nonlocal: facts.declared_nonlocal.clone(),
            local_names: facts
                .bound
                .iter()
                .filter(|name| {
                    !facts.declared_global.contains(*name)
                        && !facts.declared_nonlocal.contains(*name)
                })
                .cloned()
                .collect(),
            ..Frame::default()
        };
        let outer = std::mem::replace(&mut self.stack, env);
        self.frames.push(frame);
        let own = self.frames.len() - 1;
        self.stack.push(own);
        self.context_stack.push(cid);
        self.pin_sink.push(Some(index));
        self.pending.push(Vec::new());
        let conditions = std::mem::take(&mut self.conds);
        let walked = self.analyse_body(key, node, index, &params, &quals);
        self.conds = conditions;
        self.pending.pop();
        self.pin_sink.pop();
        self.context_stack.pop();
        self.stack = outer;
        self.in_progress.remove(&guard);
        walked?;
        if let Some(nid) = nid {
            self.walked.insert(nid);
        }
        let bound = std::mem::take(&mut self.frames[own].names);
        self.merge_bindings(key, &bound);
        self.frames[own].names = bound;
        self.record_signature(key, node, index);
        Ok(Some(index))
    }

    fn analyse_body(
        &mut self,
        _key: &str,
        node: NodeId,
        index: usize,
        params: &[String],
        quals: &[u8],
    ) -> R<()> {
        self.bind_params(node, params, quals);
        let body = facts::def_body(self.ast, node).to_vec();
        self.function_body(&body)?;
        let declared = self.annotation_type(facts::returns_annotation(self.ast, node))?;
        if declared != UNKNOWN {
            let kind = facts::na_widen(&declared);
            for stmt in self.returns(node).iter().copied().collect::<Vec<_>>() {
                if let Kind::Return { value: Some(value) } = self.kind(stmt).clone()
                    && self.ty_of(value) == TYPELESS
                {
                    self.record_widen(value, kind.clone())?;
                }
            }
        }
        let ret = if declared != UNKNOWN { declared } else { self.return_type(node)? };
        self.results[index].ret = ret;
        self.results[index].ret_qual = self.return_qual(node);
        self.flush_pending()
    }

    /// Walk a function body until no read saw a qualifier below its final one
    /// (`_function_body`).
    fn function_body(&mut self, body: &[NodeId]) -> R<()> {
        let frame = self.top();
        loop {
            let before: HashMap<String, u8> = self.frames[frame]
                .names
                .iter()
                .map(|(name, binding)| (name.clone(), binding.qual))
                .collect();
            for binding in self.frames[frame].names.values_mut() {
                binding.min_read = NOT_READ;
            }
            self.body(body)?;
            let names = &self.frames[frame].names;
            if names.values().all(|binding| binding.min_read >= binding.qual) {
                return Ok(());
            }
            if names.iter().all(|(name, binding)| before.get(name) == Some(&binding.qual)) {
                return Ok(());
            }
        }
    }

    /// The context id a fresh analysis of one call site takes over (`_supersede`).
    fn supersede(&mut self, anchor: Anchor, memo: ContextKey) -> R<i64> {
        let previous = self.anchors.insert(anchor, memo);
        match previous {
            None => {
                let cid = self.next_context;
                self.next_context += 1;
                Ok(cid)
            }
            Some(previous) => {
                let index = self
                    .contexts
                    .shift_remove(&previous)
                    .ok_or_else(|| Unsupported("a superseded context that is gone".into()))?;
                Ok(self.results[index].cid)
            }
        }
    }

    /// The types and qualifiers the enclosing scopes hold for the names a body reads
    /// (`_free_env`).
    fn free_env(&mut self, node: NodeId, frames: &[usize]) -> Vec<(String, String, u8)> {
        let (lexical, module_level) = self.free_names(node);
        let mut out = Vec::new();
        for name in lexical.iter() {
            for &frame in frames.iter().rev() {
                if let Some(found) = self.frames[frame].names.get(name) {
                    out.push((name.clone(), found.ty.clone(), found.qual));
                    break;
                }
            }
        }
        if let Some(&module) = frames.first() {
            for name in module_level.iter() {
                if let Some(found) = self.frames[module].names.get(name) {
                    out.push((name.clone(), found.ty.clone(), found.qual));
                }
            }
        }
        out.sort();
        out
    }

    /// The enclosing-scope names one definition's body reads, computed once
    /// (`_free_names`).
    fn free_names(&mut self, node: NodeId) -> super::FreeLists {
        let nid = self.nid(node);
        if let Some(nid) = nid
            && let Some(cached) = self.free.get(&nid)
        {
            return cached.clone();
        }
        let mut memo = std::mem::take(&mut self.scope_free_memo);
        let (lexical, module_level) = {
            let ast: &pyne_ast::Ast = self.ast;
            let facts_memo = &mut self.facts_memo;
            let mut facts_of = |scope: NodeId| -> Rc<facts::ScopeFacts> {
                facts_memo
                    .entry(scope)
                    .or_insert_with(|| Rc::new(facts::scope_facts(ast, scope)))
                    .clone()
            };
            facts::scope_free(ast, node, &mut memo, &mut facts_of)
        };
        self.scope_free_memo = memo;
        let name = facts::def_name(self.ast, node);
        let mut lexical: Vec<String> = lexical.iter().filter(|n| *n != name).cloned().collect();
        lexical.sort();
        let mut module_level: Vec<String> =
            module_level.iter().filter(|n| *n != name).cloned().collect();
        module_level.sort();
        let free = (Rc::new(lexical), Rc::new(module_level));
        if let Some(nid) = nid {
            self.free.insert(nid, free.clone());
        }
        free
    }

    /// What one scope's own statements bind, declare and read, computed once
    /// (`_scope_facts`).
    pub(super) fn scope_facts(&mut self, node: NodeId) -> Rc<facts::ScopeFacts> {
        if let Some(found) = self.facts_memo.get(&node) {
            return found.clone();
        }
        let found = Rc::new(facts::scope_facts(self.ast, node));
        self.facts_memo.insert(node, found.clone());
        found
    }

    /// Every bare name a definition calls, computed once (`_called_names`).
    fn called_names(&mut self, node: NodeId) -> Rc<std::collections::HashSet<String>> {
        if let Some(found) = self.called_memo.get(&node) {
            return found.clone();
        }
        let found = Rc::new(facts::called_names(self.ast, node));
        self.called_memo.insert(node, found.clone());
        found
    }

    /// A definition's own `return` statements, in the order the return type joins
    /// them (`_returns`).
    pub(super) fn returns(&mut self, node: NodeId) -> Rc<Vec<NodeId>> {
        if let Some(found) = self.returns_memo.get(&node) {
            return found.clone();
        }
        let mut returns = Vec::new();
        let mut stack = child_statements(self.ast, node);
        while let Some(current) = stack.pop() {
            match self.kind(current) {
                Kind::Return { .. } => returns.push(current),
                Kind::FunctionDef { .. } | Kind::AsyncFunctionDef { .. } => {}
                _ => stack.extend(child_statements(self.ast, current)),
            }
        }
        let found = Rc::new(returns);
        self.returns_memo.insert(node, found.clone());
        found
    }

    /// Bind a context's parameters in the freshly pushed frame (`_bind_params`).
    fn bind_params(&mut self, node: NodeId, params: &[String], quals: &[u8]) {
        let top = self.top();
        for (index, (arg, ty)) in every_param(self.ast, node).into_iter().zip(params).enumerate() {
            let (name, annotation) = arg_parts(self.ast, arg);
            let name = name.to_string();
            let unknown = if ty == UNKNOWN {
                Some(self.unknown("unannotated-param", arg, &name))
            } else {
                None
            };
            let mut binding = Binding::new(name.clone());
            binding.ty = ty.clone();
            binding.line = facts::line(self.ast, arg);
            binding.unknown = unknown;
            binding.series = spells_series(self.ast, annotation);
            binding.qual = quals.get(index).copied().unwrap_or(SERIES);
            binding.decl = self.nid(arg);
            self.frames[top].names.insert(name, binding);
        }
    }

    /// Fold a finished context's bindings into the scope's public view
    /// (`_merge_bindings`).
    fn merge_bindings(&mut self, scope: &str, names: &IndexMap<String, Binding>) {
        let target = self.bindings.entry(scope.to_string()).or_default();
        for (name, binding) in names {
            match target.get_mut(name) {
                None => {
                    let mut fresh = Binding::new(name.clone());
                    fresh.ty = binding.ty.clone();
                    fresh.line = binding.line;
                    fresh.unknown = binding.unknown.clone();
                    fresh.series = binding.series;
                    fresh.qual = binding.qual;
                    fresh.persistent = binding.persistent;
                    target.insert(name.clone(), fresh);
                }
                Some(existing) => {
                    existing.series = existing.series || binding.series;
                    existing.qual = existing.qual.max(binding.qual);
                    let joined = join(&existing.ty, &binding.ty);
                    existing.ty = joined.clone();
                    if joined == UNKNOWN && existing.unknown.is_none() {
                        existing.unknown = binding.unknown.clone();
                    }
                }
            }
        }
    }

    /// Fold a context's parameter and return types into the signature
    /// (`_record_signature`).
    fn record_signature(&mut self, key: &str, node: NodeId, index: usize) {
        if !self.funcs.contains_key(key) {
            return;
        }
        let positional = facts::args_of(self.ast, node)
            .map(|args| args.posonlyargs.len() + args.args.len())
            .unwrap_or(0);
        let result = &self.results[index];
        let observed: Vec<String> = result.params.iter().take(positional).cloned().collect();
        let result_ret = result.ret.clone();
        let signed = self.signed.contains(key);
        let is_group = self.overload_groups.contains(key);
        let annotated = self.annotated_returns.get(key).copied().unwrap_or(false);
        let slot =
            self.group_slot.get(&self.nid(node).filter(|&nid| nid != 0).unwrap_or(-1)).copied();
        let group_ret = if is_group {
            let returns = self.group_returns.entry(key.to_string()).or_default();
            if let Some(slot) = slot
                && slot < returns.len()
                && returns[slot] == UNKNOWN
            {
                returns[slot] = result_ret.clone();
            }
            Some(overload_result(returns).to_string())
        } else {
            None
        };
        let signature = self.funcs.get_mut(key).expect("signed");
        if signed && signature.params.len() == observed.len() {
            signature.params =
                signature.params.iter().zip(&observed).map(|(a, b)| join(a, b)).collect();
        } else {
            signature.params = observed;
        }
        signature.ret = match group_ret {
            Some(ret) => ret,
            None if annotated => result_ret,
            None if signed => join(&signature.ret, &result_ret),
            None => result_ret,
        };
        self.signed.insert(key.to_string());
    }

    /// Join every `return` in a function body (`_return_type`).
    fn return_type(&mut self, node: NodeId) -> R<String> {
        let mut result: Option<String> = None;
        for stmt in self.returns(node).iter().copied().collect::<Vec<_>>() {
            let ty = match self.kind(stmt) {
                Kind::Return { value: Some(value) } => self.ty_of(*value),
                _ => VOID.to_string(),
            };
            result = Some(match result {
                None => ty,
                Some(result) => self.joined(&result, &ty, stmt, "the function")?,
            });
        }
        Ok(result.unwrap_or_else(|| VOID.to_string()))
    }

    /// The qualifier a function body returns in this walk (`_return_qual`).
    fn return_qual(&mut self, node: NodeId) -> u8 {
        let returns = self.returns(node);
        returns.iter().map(|&stmt| self.qual(stmt)).max().unwrap_or(CONST)
    }
}
