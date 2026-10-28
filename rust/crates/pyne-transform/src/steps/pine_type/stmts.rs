//! Statements, stores and the loop fixpoint.

use indexmap::IndexMap;
use pyne_ast::walk::children;
use pyne_ast::{Kind, NodeId};

use super::facts::{self, enumerated, spells_persistent, spells_series, unchecked};
use super::{Engine, Frame, R};
use crate::pine_qualifier::{CONST, SERIES};
use crate::pine_type_rules::{
    FLOAT, INT, NUMERIC, OBJECT, PINE_LOOP, SCALARS, UNKNOWN, array_of, bare_wrapper, binop_type,
    class_of, element_of, elements_of, is_matrix, is_tuple, render_ty,
};
use crate::pine_type_table::{Unknown, qualify};

/// One pass of a loop body.
#[derive(Clone, Copy)]
enum LoopPass<'b> {
    /// A `while`: the test, then the body under it
    While(NodeId),
    /// A `for` body
    Body(&'b [NodeId]),
}

impl Engine<'_, '_> {
    pub(super) fn body(&mut self, body: &[NodeId]) -> R<()> {
        for &stmt in body {
            self.stmt(stmt)?;
        }
        Ok(())
    }

    fn stmt(&mut self, stmt: NodeId) -> R<()> {
        match self.kind(stmt).clone() {
            Kind::FunctionDef { .. } | Kind::AsyncFunctionDef { .. } => self.definition(stmt)?,
            Kind::Assign { targets, value, .. } => {
                let mut ty = self.expr(value)?;
                if targets.len() == 1
                    && let Kind::Name { id, .. } = self.kind(targets[0])
                    && let Some(found) = self.lookup(id)
                {
                    let declared = found.ty.clone();
                    ty = self.declared_constructor(&declared, value, &ty)?;
                }
                let qual = self.qual(value);
                for &target in &targets {
                    self.store(target, &ty, value, false, None, qual, false)?;
                    if ty == PINE_LOOP
                        && let Kind::Name { id, .. } = self.kind(target).clone()
                        && let Kind::Call { args, .. } = self.kind(value).clone()
                    {
                        let counter = self.loop_bound(value, "pine_loop", &args, None)?;
                        let top = self.top();
                        self.frames[top].loop_counters.insert(id, counter);
                    }
                }
            }
            Kind::AnnAssign { target, annotation, value, .. } => {
                let mut declared = Some(self.annotation_type(Some(annotation))?);
                let mut value_ty = match value {
                    Some(value) => Some(self.expr(value)?),
                    None => None,
                };
                if let (Some(declared), Some(ty), Some(value)) = (&declared, &value_ty, value) {
                    value_ty = Some(self.declared_constructor(declared, value, ty)?);
                }
                let mut unknown: Option<Unknown> = None;
                if bare_wrapper(self.ast, annotation) {
                    declared = value_ty.clone();
                } else if let (Some(was), Some(ty)) = (&declared, &value_ty)
                    && was != UNKNOWN
                    && !unchecked(ty)
                    && !facts::fits(was, ty)
                {
                    let spelled = self.unparse(target)?;
                    self.node_diag(
                        &format!(
                            "'{spelled}' is declared {} and assigned {}",
                            render_ty(was),
                            render_ty(ty)
                        ),
                        target,
                        "type-mismatch",
                        &spelled,
                        &format!("declare it {}, or assign {}", render_ty(ty), render_ty(was)),
                    );
                    unknown = Some(self.unknown("type-mismatch", target, &spelled));
                    declared = Some(UNKNOWN.to_string());
                }
                if let Some(declared) = declared {
                    let source = value.unwrap_or(stmt);
                    let qual = match value {
                        None => SERIES,
                        Some(value) => self.qual(value),
                    };
                    let series = spells_series(self.ast, Some(annotation));
                    let persistent = spells_persistent(self.ast, Some(annotation));
                    self.store(target, &declared, source, series, unknown, qual, persistent)?;
                } else if let Kind::Name { id, .. } = self.kind(target).clone()
                    && spells_series(self.ast, Some(annotation))
                {
                    let top = self.top();
                    self.frames[top].declared_series.insert(id);
                }
            }
            Kind::AugAssign { target, op, value } => {
                let value_ty = self.expr(value)?;
                let current = self.target_type(target)?;
                let qual = self.target_qual(target).max(self.qual(value));
                self.store(
                    target,
                    binop_type(op, &current, &value_ty),
                    stmt,
                    false,
                    None,
                    qual,
                    false,
                )?;
            }
            Kind::Return { value } => {
                let mut qual = CONST;
                if let Some(value) = value {
                    self.expr(value)?;
                    qual = self.qual(value);
                }
                if let Some(nid) = self.nid(stmt) {
                    let enclosing = self.conds.iter().max().copied().unwrap_or(CONST);
                    self.q.insert(nid, qual.max(enclosing));
                }
            }
            Kind::If { test, body, orelse } => {
                self.expr(test)?;
                self.conds.push(self.qual(test));
                let walked = self.body(&body).and_then(|_| self.body(&orelse));
                self.conds.pop();
                walked?;
            }
            Kind::While { .. } => self.run_loop(stmt, LoopPass::While(stmt))?,
            Kind::For { target: _, iter, body, orelse, .. }
            | Kind::AsyncFor { target: _, iter, body, orelse, .. } => {
                let mut iter_ty = self.expr(iter)?;
                if SCALARS.contains(&iter_ty.as_str()) {
                    let spelled = self.unparse_prefix(iter, 40)?;
                    let rendered = render_ty(&iter_ty);
                    self.node_diag(
                        &format!(
                            "'{spelled}' is a {rendered}; a for loop iterates a range or an array"
                        ),
                        iter,
                        "not-iterable",
                        &rendered,
                        "loop over range(...) or an array",
                    );
                    iter_ty = UNKNOWN.to_string();
                }
                self.store_iteration(stmt, &iter_ty)?;
                self.conds.push(self.qual(iter));
                let walked = self.run_loop(stmt, LoopPass::Body(&body));
                self.conds.pop();
                walked?;
                self.body(&orelse)?;
            }
            Kind::Expr { value } => {
                self.expr(value)?;
            }
            Kind::With { items, body, .. } | Kind::AsyncWith { items, body, .. } => {
                for item in items {
                    if let Kind::WithItem { context_expr, .. } = self.kind(item).clone() {
                        self.expr(context_expr)?;
                    }
                }
                self.body(&body)?;
            }
            Kind::Try { body, handlers, orelse, finalbody } => {
                self.body(&body)?;
                for handler in handlers {
                    if let Kind::ExceptHandler { body, .. } = self.kind(handler).clone() {
                        self.body(&body)?;
                    }
                }
                self.body(&orelse)?;
                self.body(&finalbody)?;
            }
            Kind::Match { subject, cases } => {
                self.expr(subject)?;
                for case in cases {
                    let Kind::MatchCase { guard, body, .. } = self.kind(case).clone() else {
                        continue;
                    };
                    let mut condition = self.qual(subject);
                    if let Some(guard) = guard {
                        self.expr(guard)?;
                        condition = condition.max(self.qual(guard));
                    }
                    self.conds.push(condition);
                    let walked = self.body(&body);
                    self.conds.pop();
                    walked?;
                }
            }
            Kind::ClassDef { .. } => self.class_body(stmt)?,
            _ => {
                // Import, Pass, Break, Continue, Global, Nonlocal, Delete: nothing to type
                for child in children(self.ast, stmt) {
                    if self.is_expr(child) {
                        self.expr(child)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// One pass of a loop.
    fn loop_pass(&mut self, pass: LoopPass<'_>) -> R<()> {
        match pass {
            LoopPass::While(stmt) => {
                let Kind::While { test, body, .. } = self.kind(stmt).clone() else {
                    return Ok(());
                };
                self.expr(test)?;
                self.conds.push(self.qual(test));
                let walked = self.body(&body);
                self.conds.pop();
                walked
            }
            LoopPass::Body(body) => self.body(body),
        }
    }

    /// Type a class body without letting its fields bind names outside it (`_class_body`).
    fn class_body(&mut self, stmt: NodeId) -> R<()> {
        for value in self.factory.of(self.ast, stmt) {
            self.factory_fields.insert(value);
        }
        let scope = self.scope();
        self.push_frame(Frame::new(&scope));
        let class_name = facts::def_name(self.ast, stmt).to_string();
        let body = facts::def_body(self.ast, stmt).to_vec();
        let walked = (|| {
            for member in body {
                if let Kind::FunctionDef { name, .. } = self.kind(member) {
                    // A method is keyed under its class (see `collect_members`)
                    let key = qualify(&self.scope(), &format!("{class_name}.{name}"));
                    self.definition_keyed(member, key)?;
                } else {
                    self.stmt(member)?;
                }
            }
            Ok(())
        })();
        self.stack.pop();
        walked
    }

    /// Run a loop body until its bindings stop moving (`_loop`).
    fn run_loop(&mut self, stmt: NodeId, pass: LoopPass<'_>) -> R<()> {
        self.loop_stack.push(stmt);
        let mut outcome = Ok(());
        for _ in 0..super::MAX_LOOP_PASSES {
            let before = self.snapshot();
            if let Err(err) = self.loop_pass(pass) {
                outcome = Err(err);
                break;
            }
            if self.snapshot() == before {
                break;
            }
        }
        self.loop_stack.pop();
        outcome
    }

    /// The bindings every live scope holds right now, type and qualifier (`_snapshot`).
    fn snapshot(&self) -> Vec<IndexMap<String, (String, u8)>> {
        self.stack
            .iter()
            .map(|&frame| {
                self.frames[frame]
                    .names
                    .iter()
                    .map(|(name, binding)| (name.clone(), (binding.ty.clone(), binding.qual)))
                    .collect()
            })
            .collect()
    }

    /// Bind an assignment target, recursing into tuple and list targets (`_store`).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn store(
        &mut self,
        target: NodeId,
        ty: &str,
        source: NodeId,
        series: bool,
        unknown: Option<Unknown>,
        qual: u8,
        persistent: bool,
    ) -> R<()> {
        match self.kind(target).clone() {
            Kind::Name { id, .. } => {
                self.stamp(target, ty);
                let unknown = match unknown {
                    None if ty == UNKNOWN => Some(self.unknown("unknown-value", source, "")),
                    other => other,
                };
                self.bind(&id, ty, source, unknown, series, qual, persistent)?;
            }
            Kind::Tuple { .. } | Kind::List { .. } => {
                self.stamp(target, if is_tuple(ty) { ty } else { OBJECT });
                self.distribute(target, ty, source, qual)?;
            }
            Kind::Attribute { value, .. } => {
                let base = self.expr(value)?;
                let stored = if self.is_expr(source) { Some(source) } else { None };
                self.field_store(&base, target, ty, stored)?;
                self.stamp(target, ty);
            }
            Kind::Subscript { value, .. } => {
                self.expr(value)?;
                self.stamp(target, ty);
            }
            _ => {}
        }
        Ok(())
    }

    /// Check a `obj.field = value` against the receiver's class (`_field_store`).
    fn field_store(&mut self, base: &str, node: NodeId, ty: &str, value: Option<NodeId>) -> R<()> {
        let Kind::Attribute { value: receiver, attr, .. } = self.kind(node).clone() else {
            return Ok(());
        };
        let mut ty = ty.to_string();
        if let Some(cid) = class_of(base) {
            let Some(sig) = self.class_sig(cid, node)? else { return Ok(()) };
            let found = sig.fields.get(&attr).cloned();
            if let (Some(found), Some(value)) = (&found, value) {
                ty = self.declared_constructor(found, value, &ty)?;
            }
            match found {
                None => {
                    let name = &sig.name;
                    self.node_diag(
                        &format!("'{name}' has no field '{attr}'"),
                        node,
                        "unknown-field",
                        &format!("{name}.{attr}"),
                        &format!("declare '{attr}' on '{name}', or assign a field it has"),
                    );
                }
                Some(found) if !unchecked(&ty) && !facts::fits(&found, &ty) => {
                    let name = &sig.name;
                    self.node_diag(
                        &format!(
                            "'{name}.{attr}' holds {} and is assigned {}",
                            render_ty(&found),
                            render_ty(&ty)
                        ),
                        node,
                        "type-mismatch",
                        &format!("{name}.{attr}"),
                        &format!("assign {}", render_ty(&found)),
                    );
                }
                Some(_) => {}
            }
            return Ok(());
        }
        if SCALARS.contains(&base) {
            self.node_diag(
                &format!("a {} has no field '{attr}' to assign", render_ty(base)),
                node,
                "unknown-field",
                &attr,
                "assign a field of a @udt object",
            );
        } else if base == OBJECT && !self.is_reference(receiver) {
            let spelled = self.spelled(receiver)?;
            self.node_diag(
                &format!(
                    "the class of '{spelled}' is not known here, so its field '{attr}' cannot be \
                     checked"
                ),
                node,
                "unknown-class",
                &attr,
                "annotate the value with the type it holds",
            );
        }
        Ok(())
    }

    /// Bind the names of an unpack, one element type each (`_distribute`).
    fn distribute(&mut self, target: NodeId, ty: &str, source: NodeId, qual: u8) -> R<()> {
        let elts = match self.kind(target) {
            Kind::Tuple { elts, .. } | Kind::List { elts, .. } => elts.clone(),
            _ => return Ok(()),
        };
        let names: Vec<NodeId> = elts
            .iter()
            .map(|&element| match self.kind(element) {
                Kind::Starred { value, .. } => *value,
                _ => element,
            })
            .collect();
        if elts.iter().any(|&element| facts::is_starred(self.ast, element)) {
            let origin = self.unknown("shape-mismatch", target, &render_ty(ty));
            self.diag(
                "a starred target takes an unknown number of elements, which Pine has no form \
                 for",
                target,
                Some(origin),
                "unpack exactly as many names as the tuple has elements",
            );
            for name in names {
                self.store(name, UNKNOWN, source, false, None, qual, false)?;
            }
            return Ok(());
        }
        let mut elements: Vec<String> = elements_of(ty).into_iter().map(str::to_string).collect();
        if !elements.is_empty() && elements.len() != names.len() {
            let origin = self.unknown(
                "shape-mismatch",
                target,
                &format!("{} vs {}", elements.len(), names.len()),
            );
            self.diag(
                &format!(
                    "{} has {} elements, and this unpacks {}",
                    render_ty(ty),
                    elements.len(),
                    names.len()
                ),
                target,
                Some(origin),
                &format!("unpack {} names, as many as the tuple has", elements.len()),
            );
            elements.clear();
        }
        for (index, name) in names.into_iter().enumerate() {
            let element =
                if elements.is_empty() { UNKNOWN.to_string() } else { elements[index].clone() };
            self.store(name, &element, source, false, None, qual, false)?;
        }
        Ok(())
    }

    /// Current qualifier of an augmented-assignment target (`_target_qual`).
    fn target_qual(&self, target: NodeId) -> u8 {
        match self.kind(target) {
            Kind::Name { id, .. } => self.lookup(id).map_or(SERIES, |found| found.qual),
            _ => SERIES,
        }
    }

    /// Current type of an augmented-assignment target (`_target_type`).
    fn target_type(&mut self, target: NodeId) -> R<String> {
        match self.kind(target).clone() {
            Kind::Name { id, .. } => {
                Ok(self.lookup(&id).map_or(UNKNOWN.to_string(), |found| found.ty.clone()))
            }
            Kind::Attribute { .. } => self.e_attribute(target),
            _ => Ok(UNKNOWN.to_string()),
        }
    }

    /// Type of a `for` loop variable (`_element_type`).
    pub(super) fn element_type(&self, iter_node: NodeId, iter_ty: &str) -> String {
        if let Kind::Call { func, args, .. } = self.kind(iter_node) {
            let callee = self.dotted(*func);
            if matches!(callee.as_deref(), Some("range" | "pine_range" | "lib.pine_range")) {
                let bounds: Vec<String> = args.iter().map(|&arg| self.ty_of(arg)).collect();
                if !bounds.is_empty()
                    && bounds.iter().all(|bound| NUMERIC.contains(&bound.as_str()))
                {
                    let all_int = bounds.iter().all(|bound| bound == INT);
                    return if all_int { INT } else { FLOAT }.to_string();
                }
                return UNKNOWN.to_string();
            }
        }
        if is_matrix(iter_ty) {
            return array_of(element_of(iter_ty));
        }
        element_of(iter_ty).to_string()
    }

    /// Bind what one pass of a `for` puts under its target (`_store_iteration`).
    fn store_iteration(&mut self, stmt: NodeId, iter_ty: &str) -> R<()> {
        let (target, iter) = match self.kind(stmt) {
            Kind::For { target, iter, .. } | Kind::AsyncFor { target, iter, .. } => {
                (*target, *iter)
            }
            _ => return Ok(()),
        };
        let indexed = enumerated(self.ast, iter);
        let pair = match self.kind(target) {
            Kind::Tuple { elts, .. } | Kind::List { elts, .. } if elts.len() == 2 => {
                Some((elts[0], elts[1]))
            }
            _ => None,
        };
        if let (Some((first, second)), Some(indexed)) = (pair, indexed) {
            let element = self.element_type(indexed, &self.ty_of(indexed));
            self.stamp(target, OBJECT);
            let qual = self.qual(iter);
            self.store(first, INT, stmt, false, None, qual, false)?;
            let qual = self.qual(iter);
            self.store(second, &element, stmt, false, None, qual, false)?;
            return Ok(());
        }
        let element = self.element_type(iter, iter_ty);
        let qual = self.qual(iter);
        self.store(target, &element, stmt, false, None, qual, false)
    }
}
