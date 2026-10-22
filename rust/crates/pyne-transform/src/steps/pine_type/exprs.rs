//! Expressions and their qualifiers.

use pyne_ast::walk::children;
use pyne_ast::{Constant, Int, Kind, NodeId, Value};

use super::facts::{self, call_arity, unchecked};
use super::{Engine, R};
use crate::data;
use crate::pine_qualifier::{
    CONST, SERIES, builtin_call_qualifier, lib_call_qualifier, lib_value_qualifier,
};
use crate::pine_type_rules::{
    BOOL, CallRule, DYN_DEFAULT, INT, NUMERIC, OBJECT, PINE_LOOP, SCALARS, TY_ATTR, TYPELESS,
    UNKNOWN, binop_type, builtin_name_type, class_of, compare_type, constant_type, elements_of,
    is_array, is_map, is_matrix, is_tuple, join, lib_type_override, render_ty, tuple_of,
    unaryop_type,
};

impl Engine<'_, '_> {
    /// Type one expression, stamping it and everything under it (`_expr`).
    pub(super) fn expr(&mut self, node: NodeId) -> R<String> {
        let ty = match self.kind(node) {
            Kind::Constant { .. } => self.e_constant(node),
            Kind::Name { .. } => self.e_name(node)?,
            Kind::Attribute { .. } => self.e_attribute(node)?,
            Kind::BinOp { .. } => self.e_binop(node)?,
            Kind::UnaryOp { op, operand } => {
                let (op, operand) = (*op, *operand);
                let operand_ty = self.expr(operand)?;
                unaryop_type(op, &operand_ty).to_string()
            }
            Kind::BoolOp { values, .. } => {
                for value in values.clone() {
                    self.expr(value)?;
                }
                BOOL.to_string()
            }
            Kind::Compare { left, comparators, .. } => {
                let (left, comparators) = (*left, comparators.clone());
                let left_ty = self.expr(left)?;
                let mut right = left_ty.clone();
                for comparator in comparators {
                    right = self.expr(comparator)?;
                }
                compare_type(&left_ty, &right).to_string()
            }
            Kind::IfExp { .. } => self.e_ifexp(node)?,
            Kind::NamedExpr { target, value } => {
                let (target, value) = (*target, *value);
                let ty = self.expr(value)?;
                let qual = self.qual(value);
                self.store(target, &ty, node, false, None, qual, false)?;
                ty
            }
            Kind::Subscript { .. } => self.e_subscript(node)?,
            Kind::Tuple { elts, .. } | Kind::List { elts, .. } => {
                let mut types = Vec::new();
                for element in elts.clone() {
                    types.push(self.expr(element)?);
                }
                tuple_of(&types)
            }
            Kind::Set { elts } => {
                for element in elts.clone() {
                    self.expr(element)?;
                }
                OBJECT.to_string()
            }
            Kind::Dict { keys, values } => {
                let (keys, values) = (keys.clone(), values.clone());
                for key in keys.into_iter().flatten() {
                    self.expr(key)?;
                }
                for value in values {
                    self.expr(value)?;
                }
                OBJECT.to_string()
            }
            Kind::ListComp { .. }
            | Kind::SetComp { .. }
            | Kind::GeneratorExp { .. }
            | Kind::DictComp { .. } => self.e_comprehension(node)?,
            Kind::Call { .. } => self.e_call(node)?,
            _ => {
                for child in children(self.ast, node) {
                    if self.is_expr(child) {
                        self.expr(child)?;
                    }
                }
                UNKNOWN.to_string()
            }
        };
        if let Some(nid) = self.nid(node) {
            let qual = self.expr_qual(node)?;
            self.q.insert(nid, qual);
            self.ty.insert(nid, ty.clone());
        }
        self.join_stamp(node, &ty);
        Ok(ty)
    }

    // --- qualifiers --------------------------------------------------------

    /// The qualifier this walk gave an expression it has visited (`_qual`).
    pub(super) fn qual(&self, node: NodeId) -> u8 {
        self.nid(node).and_then(|nid| self.q.get(&nid).copied()).unwrap_or(SERIES)
    }

    fn max_qual(&self, nodes: impl IntoIterator<Item = NodeId>) -> u8 {
        nodes.into_iter().map(|node| self.qual(node)).max().unwrap_or(CONST)
    }

    /// The qualifier of an expression whose children were just visited (`_expr_qual`).
    fn expr_qual(&mut self, node: NodeId) -> R<u8> {
        Ok(match self.kind(node).clone() {
            Kind::Constant { .. } => CONST,
            Kind::Name { id, .. } => self.name_qual(&id),
            Kind::Attribute { .. } => self.attribute_qual(node)?,
            Kind::Call { .. } => self.call_qual(node),
            Kind::Subscript { value, .. } => {
                if is_tuple(&self.ty_of(value)) {
                    self.qual(value)
                } else {
                    SERIES
                }
            }
            Kind::NamedExpr { value, .. } => self.qual(value),
            Kind::Lambda { .. }
            | Kind::ListComp { .. }
            | Kind::SetComp { .. }
            | Kind::DictComp { .. }
            | Kind::GeneratorExp { .. }
            | Kind::Await { .. }
            | Kind::Yield { .. }
            | Kind::YieldFrom { .. } => SERIES,
            Kind::BinOp { left, right, .. } => self.qual(left).max(self.qual(right)),
            Kind::UnaryOp { operand, .. } => self.qual(operand),
            Kind::IfExp { test, body, orelse } => {
                self.qual(test).max(self.qual(body)).max(self.qual(orelse))
            }
            Kind::BoolOp { values, .. } => self.max_qual(values),
            Kind::Compare { left, comparators, .. } => {
                self.qual(left).max(self.max_qual(comparators))
            }
            _ => {
                let operands: Vec<NodeId> = children(self.ast, node)
                    .into_iter()
                    .filter(|&child| self.is_expr(child))
                    .collect();
                self.max_qual(operands)
            }
        })
    }

    /// The qualifier of a bare name read (`_name_qual`).
    fn name_qual(&mut self, id: &str) -> u8 {
        if id == DYN_DEFAULT {
            return CONST;
        }
        if let Some(found) = self.lookup_mut(id) {
            found.min_read = found.min_read.min(found.qual);
            return found.qual;
        }
        match data::lib_entries().get(id) {
            Some(entry) if entry.kind == "value" => {
                lib_value_qualifier(id, entry.ty.as_deref().unwrap_or(UNKNOWN))
            }
            _ => CONST,
        }
    }

    /// The qualifier of an attribute read (`_attribute_qual`).
    fn attribute_qual(&mut self, node: NodeId) -> R<u8> {
        let Some(name) = self.lib_name(node) else {
            return Ok(if self.names_class(node)?.is_some() { CONST } else { SERIES });
        };
        Ok(match data::lib_entries().get(&name) {
            Some(entry) if entry.kind == "value" => {
                lib_value_qualifier(&name, entry.ty.as_deref().unwrap_or(UNKNOWN))
            }
            _ => CONST,
        })
    }

    /// The qualifier of a call's result (`_call_qual`).
    fn call_qual(&self, node: NodeId) -> u8 {
        let Some((func, args, keywords)) = facts::call_parts(self.ast, node) else {
            return SERIES;
        };
        let mut arguments: Vec<u8> = args.iter().map(|&arg| self.qual(arg)).collect();
        arguments.extend(
            keywords.iter().map(|&keyword| self.qual(facts::keyword_parts(self.ast, keyword).1)),
        );
        if let Some(callee) = self.cached_lib_name(func) {
            let argc = call_arity(self.ast, node).map(|count| count as usize);
            return lib_call_qualifier(&callee, arguments, argc);
        }
        if let Some(nid) = self.nid(node)
            && let Some(&qual) = self.call_q.get(&nid)
        {
            return qual;
        }
        if let Kind::Name { id, .. } = self.kind(func)
            && self.lookup(id).is_none()
            && let Some(builtin) = builtin_call_qualifier(id, arguments)
        {
            return builtin;
        }
        SERIES
    }

    // --- the rules ---------------------------------------------------------

    fn e_constant(&self, node: NodeId) -> String {
        if let Some(Value::Str(existing)) = self.ast.extra(node, TY_ATTR) {
            return existing.clone();
        }
        match self.kind(node) {
            Kind::Constant { value, .. } => constant_type(value).to_string(),
            _ => UNKNOWN.to_string(),
        }
    }

    /// Type a bare name read (`_e_name`).
    fn e_name(&self, node: NodeId) -> R<String> {
        let Kind::Name { id, .. } = self.kind(node) else { return Ok(UNKNOWN.to_string()) };
        if id == DYN_DEFAULT {
            return Ok(TYPELESS.to_string());
        }
        if let Some(found) = self.lookup(id) {
            return Ok(found.ty.clone());
        }
        if let Some(entry) = data::lib_entries().get(id.as_str()) {
            return Ok(if entry.kind == "value" {
                entry.ty.clone().unwrap_or_else(|| UNKNOWN.to_string())
            } else {
                OBJECT.to_string()
            });
        }
        if let Some(builtin) = builtin_name_type(id) {
            return Ok(builtin.to_string());
        }
        if self.lib_aliases.contains(id.as_str()) || data::lib_namespaces().contains(id.as_str()) {
            return Ok(OBJECT.to_string());
        }
        Ok(if self.names_type(node)? { OBJECT } else { UNKNOWN }.to_string())
    }

    /// The class a qualified spelling in value position names (`_names_class`).
    pub(super) fn names_class(&mut self, node: NodeId) -> R<Option<String>> {
        if !matches!(self.kind(node), Kind::Attribute { .. }) {
            return Ok(None);
        }
        let Some(spelled) = self.dotted(node) else { return Ok(None) };
        let parts: Vec<String> = spelled.split('.').map(str::to_string).collect();
        self.class_named(&parts, node)
    }

    /// Whether an expression in value position names a type (`_names_type`).
    pub(super) fn names_type(&self, node: NodeId) -> R<bool> {
        if let Kind::Name { id, .. } = self.kind(node)
            && self.lookup(id).is_some()
        {
            return Ok(false);
        }
        Ok(self.annotation_type(Some(node))? != UNKNOWN)
    }

    pub(super) fn e_attribute(&mut self, node: NodeId) -> R<String> {
        let Kind::Attribute { value, attr, .. } = self.kind(node).clone() else {
            return Ok(UNKNOWN.to_string());
        };
        let base = self.expr(value)?;
        if attr == "__class__" {
            return Ok(OBJECT.to_string());
        }
        let Some(name) = self.lib_name(node) else {
            if self.names_class(node)?.is_some() {
                return Ok(OBJECT.to_string());
            }
            if attr == "new" && self.names_class(value)?.is_some() {
                return Ok(OBJECT.to_string());
            }
            return self.field_read(&base, node);
        };
        if let Some(CallRule::Is(ty)) = lib_type_override(&name)
            && ty.chars().count() == 1
        {
            return Ok(ty.to_string());
        }
        let Some(entry) = data::lib_entries().get(&name) else {
            return Ok(
                if data::lib_namespaces().contains(&name) { OBJECT } else { UNKNOWN }.to_string()
            );
        };
        if entry.kind == "value" {
            return Ok(entry.ty.clone().unwrap_or_else(|| UNKNOWN.to_string()));
        }
        if name == "na" {
            return Ok(TYPELESS.to_string());
        }
        Ok(OBJECT.to_string())
    }

    /// Type a `obj.field` read from the receiver's class (`_field_read`).
    fn field_read(&mut self, base: &str, node: NodeId) -> R<String> {
        let Kind::Attribute { value: receiver, attr, .. } = self.kind(node).clone() else {
            return Ok(UNKNOWN.to_string());
        };
        if base == PINE_LOOP {
            return self.loop_field(node);
        }
        if let Some(cid) = class_of(base) {
            let Some(sig) = self.class_sig(cid, node)? else { return Ok(UNKNOWN.to_string()) };
            if let Some(found) = sig.fields.get(&attr) {
                return Ok(found.clone());
            }
            let declared_here =
                self.class_methods.get(cid).is_some_and(|methods| methods.contains_key(&attr));
            if sig.methods.contains_key(&attr) || declared_here {
                return Ok(OBJECT.to_string());
            }
            let name = sig.name.clone();
            self.node_diag(
                &format!("'{name}' has no field '{attr}'"),
                node,
                "unknown-field",
                &format!("{name}.{attr}"),
                &format!("declare '{attr}' on '{name}', or read a field it has"),
            );
            return Ok(UNKNOWN.to_string());
        }
        if base == OBJECT && !self.is_reference(receiver) {
            let spelled = self.spelled(receiver)?;
            self.node_diag(
                &format!(
                    "the class of '{spelled}' is not known here, so its field '{attr}' has no type"
                ),
                node,
                "unknown-class",
                &attr,
                "annotate the value with the type it holds",
            );
        }
        Ok(UNKNOWN.to_string())
    }

    /// Type a read of the compiled loop counter object (`_loop_field`).
    fn loop_field(&mut self, node: NodeId) -> R<String> {
        let Kind::Attribute { value, attr, .. } = self.kind(node).clone() else {
            return Ok(UNKNOWN.to_string());
        };
        if attr == "step" {
            return Ok(OBJECT.to_string());
        }
        if attr != "value" {
            self.node_diag(
                &format!("the loop counter has no field '{attr}'"),
                node,
                "unknown-field",
                &format!("PineLoop.{attr}"),
                "read 'value', or call 'step'",
            );
            return Ok(UNKNOWN.to_string());
        }
        match self.loop_counter(value) {
            Some(counter) if counter != UNKNOWN => Ok(counter),
            _ => {
                let spelled = self.unparse(value)?;
                self.node_diag(
                    &format!(
                        "the bounds of '{spelled}' are not known here, so its counter has no type"
                    ),
                    node,
                    "unknown-value",
                    "PineLoop.value",
                    "give the loop bounds known numeric types",
                );
                Ok(UNKNOWN.to_string())
            }
        }
    }

    /// The recorded counter type of a loop object, by the name that holds it
    /// (`_loop_counter`).
    pub(super) fn loop_counter(&self, receiver: NodeId) -> Option<String> {
        let Kind::Name { id, .. } = self.kind(receiver) else { return None };
        self.stack.iter().rev().find_map(|&frame| self.frames[frame].loop_counters.get(id).cloned())
    }

    /// Join a loop counter with the bounds one call passes (`_loop_bound`).
    pub(super) fn loop_bound(
        &mut self,
        node: NodeId,
        what: &str,
        bounds: &[NodeId],
        counter: Option<String>,
    ) -> R<String> {
        let mut result = counter.unwrap_or_else(|| INT.to_string());
        for &bound in bounds {
            let ty = self.ty_of(bound);
            if NUMERIC.contains(&ty.as_str()) {
                result = join(&result, &ty);
            } else if unchecked(&ty) {
                result = UNKNOWN.to_string();
            } else {
                self.node_diag(
                    &format!("'{what}' takes a number for a bound, {} passed", render_ty(&ty)),
                    node,
                    "bad-call",
                    what,
                    "pass int or float bounds",
                );
                result = UNKNOWN.to_string();
            }
        }
        Ok(result)
    }

    /// Type a `loop.step(to)` call (`_loop_step`).
    pub(super) fn loop_step(&mut self, node: NodeId, func: NodeId) -> R<String> {
        let Kind::Attribute { value, attr, .. } = self.kind(func).clone() else {
            return Ok(UNKNOWN.to_string());
        };
        if attr != "step" {
            return Ok(UNKNOWN.to_string());
        }
        let Some((_, args, keywords)) = facts::call_parts(self.ast, node) else {
            return Ok(BOOL.to_string());
        };
        let (args, has_keywords) = (args.to_vec(), !keywords.is_empty());
        if args.len() != 1 || has_keywords {
            let arity = call_arity(self.ast, node).map_or("None".to_string(), |n| n.to_string());
            self.node_diag(
                &format!("'step' takes 1 argument(s), {arity} passed"),
                node,
                "bad-call",
                "PineLoop.step",
                "pass the bound: 'step(to)'",
            );
            return Ok(BOOL.to_string());
        }
        let counter = self.loop_counter(value);
        if let (Some(counter), Kind::Name { id, .. }) = (counter, self.kind(value).clone()) {
            let joined = self.loop_bound(node, "step", &args, Some(counter))?;
            for index in (0..self.stack.len()).rev() {
                let frame = self.stack[index];
                if self.frames[frame].loop_counters.contains_key(&id) {
                    self.frames[frame].loop_counters.insert(id.clone(), joined);
                    break;
                }
            }
        }
        Ok(BOOL.to_string())
    }

    /// Whether an attribute's base is a module or a namespace (`_is_reference`).
    pub(super) fn is_reference(&self, node: NodeId) -> bool {
        let Kind::Name { id, .. } = self.kind(node) else { return false };
        self.imports.contains_key(id) || self.shadowed.contains_key(id) || self.lookup(id).is_none()
    }

    fn e_binop(&mut self, node: NodeId) -> R<String> {
        let Kind::BinOp { left, op, right } = *self.kind(node) else {
            return Ok(UNKNOWN.to_string());
        };
        let left_ty = self.expr(left)?;
        let right_ty = self.expr(right)?;
        let result = binop_type(op, &left_ty, &right_ty);
        for (operand, operand_ty) in [(left, &left_ty), (right, &right_ty)] {
            if operand_ty == TYPELESS {
                self.record_widen(operand, facts::na_widen(result))?;
            }
        }
        Ok(result.to_string())
    }

    fn e_ifexp(&mut self, node: NodeId) -> R<String> {
        let Kind::IfExp { test, body, orelse } = *self.kind(node) else {
            return Ok(UNKNOWN.to_string());
        };
        self.expr(test)?;
        let body_ty = self.expr(body)?;
        let orelse_ty = self.expr(orelse)?;
        let joined = self.joined(&body_ty, &orelse_ty, node, "the ternary")?;
        for (arm, arm_ty) in [(body, &body_ty), (orelse, &orelse_ty)] {
            if arm_ty == TYPELESS {
                self.record_widen(arm, facts::na_widen(&joined))?;
            }
        }
        Ok(joined)
    }

    fn e_subscript(&mut self, node: NodeId) -> R<String> {
        let Kind::Subscript { value, slice, .. } = *self.kind(node) else {
            return Ok(UNKNOWN.to_string());
        };
        let base = self.expr(value)?;
        self.expr(slice)?;
        if is_tuple(&base) {
            let elements = elements_of(&base);
            if let Kind::Constant { value: Constant::Int(Int::Small(index)), .. } = self.kind(slice)
                && *index >= 0
                && (*index as usize) < elements.len()
            {
                return Ok(elements[*index as usize].to_string());
            }
            return Ok(UNKNOWN.to_string());
        }
        if (is_array(&base) || is_matrix(&base) || is_map(&base)) && !self.is_series(value) {
            self.node_diag(
                &format!("indexing {} is not Pine", render_ty(&base)),
                node,
                "not-pine",
                "Subscript",
                "read elements with array.get / matrix.get / map.get",
            );
            return Ok(UNKNOWN.to_string());
        }
        if SCALARS.contains(&base.as_str()) && !self.is_series(value) {
            let spelled = self.unparse_prefix(value, 40)?;
            let rendered = render_ty(&base);
            self.node_diag(
                &format!("'{spelled}' is a plain {rendered}, not a series"),
                node,
                "not-series",
                &rendered,
                &format!("declare it Series[{rendered}] to read its history"),
            );
            return Ok(UNKNOWN.to_string());
        }
        Ok(base)
    }

    /// Whether an expression reads a series (`_is_series`).
    fn is_series(&self, node: NodeId) -> bool {
        if let Kind::Name { id, .. } = self.kind(node) {
            if let Some(found) = self.lookup(id) {
                return found.series || self.frames[self.top()].declared_series.contains(id);
            }
            return data::lib_entries().contains_key(id.as_str());
        }
        self.cached_lib_name(node).is_some()
    }

    /// Walk a comprehension so nothing under it is left untyped (`_e_listcomp`).
    fn e_comprehension(&mut self, node: NodeId) -> R<String> {
        let generators = match self.kind(node) {
            Kind::ListComp { generators, .. }
            | Kind::SetComp { generators, .. }
            | Kind::GeneratorExp { generators, .. }
            | Kind::DictComp { generators, .. } => generators.clone(),
            _ => Vec::new(),
        };
        for generator in generators {
            let Kind::Comprehension { target, iter, ifs, .. } = self.kind(generator).clone() else {
                continue;
            };
            let iter_ty = self.expr(iter)?;
            let element = self.element_type(iter, &iter_ty);
            self.store(target, &element, node, false, None, SERIES, false)?;
            for condition in ifs {
                self.expr(condition)?;
            }
        }
        match self.kind(node).clone() {
            Kind::DictComp { key, value, .. } => {
                self.expr(key)?;
                self.expr(value)?;
            }
            Kind::ListComp { elt, .. }
            | Kind::SetComp { elt, .. }
            | Kind::GeneratorExp { elt, .. } => {
                self.expr(elt)?;
            }
            _ => {}
        }
        Ok(OBJECT.to_string())
    }

    fn e_call(&mut self, node: NodeId) -> R<String> {
        let Some((func, args, keywords)) = facts::call_parts(self.ast, node) else {
            return Ok(UNKNOWN.to_string());
        };
        let (args, keywords) = (args.to_vec(), keywords.to_vec());
        if let Some(nid) = self.nid(node) {
            self.call_q.remove(&nid);
        }
        for arg in args {
            self.expr(arg)?;
        }
        for keyword in keywords {
            let value = facts::keyword_parts(self.ast, keyword).1;
            self.expr(value)?;
        }
        match self.lib_name(func) {
            None => {
                self.expr(func)?;
                self.user_call(node)
            }
            Some(callee) => self.lib_call(node, &callee),
        }
    }

    /// The registry key a lib reference resolves to (`_lib_name`).
    pub(super) fn lib_name(&mut self, node: NodeId) -> Option<String> {
        if let Some(cached) = self.lib_names.get(&node) {
            return cached.clone();
        }
        let name = self.resolve_lib_name(node);
        self.lib_names.insert(node, name.clone());
        name
    }

    /// `_lib_name` without its memo, for a reader that cannot write it.
    pub(super) fn cached_lib_name(&self, node: NodeId) -> Option<String> {
        match self.lib_names.get(&node) {
            Some(cached) => cached.clone(),
            None => self.resolve_lib_name(node),
        }
    }

    fn resolve_lib_name(&self, node: NodeId) -> Option<String> {
        let dotted = self.dotted(node)?;
        let (head, rest) = dotted.split_once('.').unwrap_or((&dotted, ""));
        (self.lib_aliases.contains(head) && !rest.is_empty()).then(|| rest.to_string())
    }
}
