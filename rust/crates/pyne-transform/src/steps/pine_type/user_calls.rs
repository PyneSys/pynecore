//! Calls to the module's own functions and groups, constructors, methods and the
//! security plumbing.

use std::rc::Rc;

use pyne_ast::{Constant, Kind, NodeId, Value};

use super::facts::{
    self, arg_parts, arity, call_arity, every_param, fits, keyword_parts, reaches_library_methods,
    unchecked,
};
use super::{Engine, R};
use crate::data;
use crate::pine_qualifier::SERIES;
use crate::pine_type_rules::ImplSig;
use crate::pine_type_rules::{
    NUMERIC, OBJECT, PINE_LOOP, TYPELESS, UNKNOWN, builtin_call_type, class_of, impl_sig, join,
    namespace_of, object_ty, overload_result, param_defaults, pin_for, render_ty,
};
use crate::pine_type_table::{CallSite, ClassSig, ExportSig};

/// The lib namespaces `method_call` dispatches a builtin method to (`_METHOD_NAMESPACES`).
const METHOD_NAMESPACES: &[&str] =
    &["array", "matrix", "map", "line", "box", "label", "table", "linefill", "polyline"];

/// What `udt` gives every user class besides its fields (`_UDT_ATTRS`).
const UDT_ATTRS: &[&str] = &["copy", "new"];

/// Whether a method implementation can take a receiver of a type (`_takes_receiver`).
fn takes_receiver(imp: &ImplSig, receiver: &str) -> bool {
    if class_of(receiver).is_none() {
        return true;
    }
    match imp.params.first() {
        None => imp.open_ended,
        Some(first) => first == receiver || first == UNKNOWN || first == OBJECT,
    }
}

impl Engine<'_, '_> {
    /// Result type of a call to a function this module defines or imports (`_user_call`).
    pub(super) fn user_call(&mut self, node: NodeId) -> R<String> {
        let (func, args, keywords) = facts::call_parts(self.ast, node).expect("a call");
        let (args, keywords) = (args.to_vec(), keywords.to_vec());
        let name = self.dotted(func).unwrap_or_default();
        if name == "__sec_read__" {
            return self.security_read(node);
        }
        if name == "method_call" {
            return match self.method_call(node)? {
                Some(dispatched) => Ok(dispatched),
                None => self.unknown_method(node),
            };
        }
        if self.factory_fields.contains(&node) {
            return Ok(OBJECT.to_string());
        }
        let (line, col) = (self.line(node), self.col(node));
        let Some((key, frame, shadowed)) = self.resolve_func(&name, node) else {
            if let Some(constructed) = self.constructor_call(&name, node)? {
                return Ok(constructed);
            }
            if let Some(invoked) = self.method_of(node)? {
                return Ok(invoked);
            }
            if let Some(imported) = self.imported_call(node, &name)? {
                return Ok(imported);
            }
            if let Some(rule) = builtin_call_type(&name)
                && matches!(self.kind(func), Kind::Name { .. })
            {
                let argc = call_arity(self.ast, node);
                if let Some(fallback) = self.apply_override(rule, &args, &keywords, argc, None)? {
                    return Ok(fallback);
                }
                self.node_diag(
                    &format!("the call to '{name}' has no known type"),
                    node,
                    "unknown-call",
                    &name,
                    &format!("call '{name}' the way Pine spells it"),
                );
                return Ok(UNKNOWN.to_string());
            }
            let message = if name.is_empty() {
                "the callee is not a name".to_string()
            } else {
                format!("'{name}' is not a function this module defines or imports, nor a lib name")
            };
            self.node_diag(
                &message,
                node,
                "unknown-call",
                &name,
                "define it, import it or call a lib name",
            );
            return Ok(UNKNOWN.to_string());
        };
        if shadowed {
            let origin = self.unknown("rebound-name", node, &name);
            self.diag(
                &format!("'{name}' is assigned as well as defined, so what it calls is unknown"),
                node,
                Some(origin),
                &format!("call '{name}' under a name nothing assigns to"),
            );
            return Ok(UNKNOWN.to_string());
        }
        let is_group = self.overload_groups.contains(&key);
        let argc = call_arity(self.ast, node);
        if is_group {
            self.ensure_group(&key, frame)?;
            if !self.group_call_fits(&key, &name, node, None)? {
                self.calls.push(CallSite {
                    callee: name,
                    line,
                    col,
                    argc,
                    ty: UNKNOWN.to_string(),
                    pin: None,
                });
                return Ok(UNKNOWN.to_string());
            }
        }
        let pin = self.pin(node, is_group)?;
        let ty = if is_group {
            self.group_type(&key, pin.as_deref(), &args, &keywords)?
        } else {
            self.call_context(&key, node, frame, None, true)?
        };
        self.calls.push(CallSite { callee: name, line, col, argc, ty: ty.clone(), pin });
        Ok(ty)
    }

    /// Whether some implementation of a user group takes a call's arguments
    /// (`_group_call_fits`).
    fn group_call_fits(
        &mut self,
        key: &str,
        name: &str,
        node: NodeId,
        args: Option<&[NodeId]>,
    ) -> R<bool> {
        let (_, own_args, keywords) = facts::call_parts(self.ast, node).expect("a call");
        if own_args.iter().any(|&arg| facts::is_starred(self.ast, arg))
            || keywords.iter().any(|&keyword| keyword_parts(self.ast, keyword).0.is_none())
        {
            return Ok(true);
        }
        let positional: Vec<NodeId> = args.unwrap_or(own_args).to_vec();
        let impls = self.defs.get(key).cloned().unwrap_or_default();
        if impls.is_empty() {
            return Ok(true);
        }
        for imp in impls {
            let open = facts::args_of(self.ast, imp)
                .is_some_and(|args| args.vararg.is_some() || args.kwarg.is_some());
            if open {
                return Ok(true);
            }
            if self.call_params(imp, node, &positional, false)?.is_some() {
                return Ok(true);
            }
        }
        self.bad_call(node, name, &format!("no overload of '{name}' takes these arguments"));
        Ok(false)
    }

    /// Type a call that builds an instance of a class (`_constructor_call`).
    fn constructor_call(&mut self, name: &str, node: NodeId) -> R<Option<String>> {
        let parts: Vec<String> =
            name.split('.').filter(|part| !part.is_empty()).map(str::to_string).collect();
        if parts.is_empty() {
            return Ok(None);
        }
        let mut built = None;
        if parts.len() > 1 && parts[parts.len() - 1] == "new" {
            built = self.class_named(&parts[..parts.len() - 1], node)?;
        }
        if built.is_none() {
            built = self.class_named(&parts, node)?;
        }
        let Some(built) = built else { return Ok(None) };
        if !self.constructor_fits(&built, name, node)? {
            return Ok(Some(UNKNOWN.to_string()));
        }
        Ok(Some(object_ty(&built)))
    }

    /// Whether a constructor call matches the class it builds (`_constructor_fits`).
    fn constructor_fits(&mut self, cid: &str, name: &str, node: NodeId) -> R<bool> {
        let (_, args, keywords) = facts::call_parts(self.ast, node).expect("a call");
        let (args, keywords) = (args.to_vec(), keywords.to_vec());
        if args.iter().any(|&arg| facts::is_starred(self.ast, arg))
            || keywords.iter().any(|&keyword| keyword_parts(self.ast, keyword).0.is_none())
        {
            return Ok(true);
        }
        let Some(sig) = self.class_sig(cid, node)? else { return Ok(true) };
        let fields: Vec<String> = sig.fields.keys().cloned().collect();
        let class = sig.name.clone();
        if args.len() > fields.len() {
            self.bad_call(
                node,
                name,
                &format!(
                    "'{class}' has {} field(s), {} argument(s) passed",
                    fields.len(),
                    args.len()
                ),
            );
            return Ok(false);
        }
        let mut bound: indexmap::IndexMap<String, NodeId> =
            fields.iter().cloned().zip(args.iter().copied()).collect();
        for &keyword in &keywords {
            let (arg, value) = keyword_parts(self.ast, keyword);
            let arg = arg.unwrap_or("").to_string();
            if !sig.fields.contains_key(&arg) {
                self.bad_call(node, name, &format!("'{class}' has no field '{arg}'"));
                return Ok(false);
            }
            if bound.contains_key(&arg) {
                self.bad_call(node, name, &format!("'{arg}' is passed to '{class}' twice"));
                return Ok(false);
            }
            bound.insert(arg, value);
        }
        for (index, field_name) in fields.iter().enumerate() {
            let Some(&passed) = bound.get(field_name) else {
                if (index as i64) < sig.required {
                    self.bad_call(
                        node,
                        name,
                        &format!("'{class}' needs a value for '{field_name}'"),
                    );
                    return Ok(false);
                }
                continue;
            };
            let declared = sig.fields[field_name].clone();
            let given = self.ty_of(passed);
            if declared != UNKNOWN && !unchecked(&given) && !fits(&declared, &given) {
                self.bad_call(
                    node,
                    name,
                    &format!(
                        "'{class}.{field_name}' holds {}, {} passed",
                        render_ty(&declared),
                        render_ty(&given)
                    ),
                );
                return Ok(false);
            }
            if given == TYPELESS {
                self.record_widen(passed, facts::na_widen(&declared))?;
            }
        }
        Ok(true)
    }

    /// The class a dotted path names (`_class_named`).
    pub(super) fn class_named(&mut self, parts: &[String], node: NodeId) -> R<Option<String>> {
        if parts.is_empty() || self.lookup(&parts[0]).is_some() {
            return Ok(None);
        }
        let cid = self.class_behind(parts, node)?;
        if let Some(cid) = &cid {
            self.class_spellings.entry(cid.clone()).or_insert_with(|| parts.join("."));
        }
        Ok(cid)
    }

    /// The class a dotted path names, the shadowing value ruled out (`_class_behind`).
    fn class_behind(&mut self, parts: &[String], node: NodeId) -> R<Option<String>> {
        let leaf = &parts[parts.len() - 1];
        if parts.len() == 1 {
            if let Some(own) = self.classes.get(leaf) {
                return Ok(Some(own.clone()));
            }
            let Some(binding) = self.imports.get(leaf).cloned() else { return Ok(None) };
            if binding.attrs.is_empty()
                || self.multi_imports.contains(leaf)
                || self.nearer_binding(leaf)
            {
                return Ok(None);
            }
            let (last, hops) = binding.attrs.split_last().expect("attrs");
            let Some(interface) = self.interface_of(&binding.module, hops, node)? else {
                return Ok(None);
            };
            return Ok(interface.classes.get(last).map(|sig| sig.id.clone()));
        }
        let Some(binding) = self.imports.get(&parts[0]).cloned() else { return Ok(None) };
        if self.multi_imports.contains(&parts[0]) || self.nearer_binding(&parts[0]) {
            return Ok(None);
        }
        let mut hops = binding.attrs.clone();
        hops.extend(parts[1..parts.len() - 1].iter().cloned());
        let Some(interface) = self.interface_of(&binding.module, &hops, node)? else {
            return Ok(None);
        };
        Ok(interface.classes.get(leaf).map(|sig| sig.id.clone()))
    }

    /// Type a `method_call(name, obj, ...)` the method transform emitted (`_method_call`).
    fn method_call(&mut self, node: NodeId) -> R<Option<String>> {
        let (_, all_args, _) = facts::call_parts(self.ast, node).expect("a call");
        if all_args.len() < 2 {
            return Ok(None);
        }
        let (selector, receiver) = (all_args[0], all_args[1]);
        let args: Vec<NodeId> = all_args[1..].to_vec();
        if matches!(self.kind(selector), Kind::Name { .. } | Kind::Attribute { .. }) {
            return self.named_method(selector, node, args);
        }
        let Some(name) = facts::str_constant(self.ast, selector).map(str::to_string) else {
            return Ok(None);
        };
        let receiver_ty = self.ty_of(receiver);
        if let Some(namespace) = namespace_of(&receiver_ty) {
            let callee = format!("{namespace}.{name}");
            if data::lib_entries().contains_key(&callee) {
                return self.builtin_method(&callee, node, &args).map(Some);
            }
        }
        if let Some(cid) = class_of(&receiver_ty) {
            let cid = cid.to_string();
            return self.method_result(&cid, &name, node, &args);
        }
        if self.resolve_func(&name, node).is_some() {
            return Ok(None);
        }
        if reaches_library_methods(&receiver_ty, &name)
            && let Some(published) = self.library_method(&name, node, None)?
        {
            return self.published_result(&published, &name, node, &args).map(Some);
        }
        let mut returns: Vec<String> = Vec::new();
        for candidate in METHOD_NAMESPACES {
            let Some(entry) = data::lib_entries().get(&format!("{candidate}.{name}")) else {
                continue;
            };
            if entry.kind != "function" {
                return Ok(None);
            }
            returns.push(entry.ret.clone().unwrap_or_else(|| UNKNOWN.to_string()));
        }
        if returns.is_empty() {
            return Ok(None);
        }
        let agreed = overload_result(&returns);
        Ok((agreed != UNKNOWN).then(|| agreed.to_string()))
    }

    /// Type a builtin reached through the method spelling (`_builtin_method`).
    fn builtin_method(&mut self, callee: &str, node: NodeId, args: &[NodeId]) -> R<String> {
        let entry = data::lib_entries().get(callee);
        if let Some(entry) = entry
            && !self.lib_call_fits(callee, entry, node, Some(args))?
        {
            return Ok(UNKNOWN.to_string());
        }
        let (_, _, keywords) = facts::call_parts(self.ast, node).expect("a call");
        let keywords = keywords.to_vec();
        let argc = arity(self.ast, args, &keywords);
        let ty = self.lib_call_type(callee, args, &keywords, argc, None)?;
        if let Some(entry) = entry {
            self.record_lib_na_args(callee, entry, args, &keywords)?;
        }
        Ok(ty)
    }

    /// Type a `method_call(<the method itself>, obj, ...)` (`_named_method`).
    fn named_method(
        &mut self,
        selector: NodeId,
        node: NodeId,
        args: Vec<NodeId>,
    ) -> R<Option<String>> {
        let Some(spelled) = self.dotted(selector) else { return Ok(None) };
        let imported = self.imported_selector(&spelled);
        let name = match &imported {
            None => spelled.rsplit('.').next().unwrap_or("").to_string(),
            Some((_, path)) => path[path.len() - 1].clone(),
        };
        let receiver_ty = args.first().map_or(UNKNOWN.to_string(), |&arg| self.ty_of(arg));
        if let Some(namespace) = namespace_of(&receiver_ty) {
            let callee = format!("{namespace}.{name}");
            if data::lib_entries().contains_key(&callee) {
                let captures = match self.ast.extra(node, "pyne_closure_args") {
                    Some(Value::Tuple(items) | Value::List(items)) => items.len(),
                    _ => 0,
                };
                let mut args = args;
                if captures > 0 {
                    let mut kept = vec![args[0]];
                    kept.extend(args.iter().skip(1 + captures).copied());
                    args = kept;
                }
                return self.builtin_method(&callee, node, &args).map(Some);
            }
        }
        let Some((key, frame, shadowed)) = self.resolve_func(&spelled, node) else {
            let Some((binding, path)) = imported else { return Ok(None) };
            return self.imported_method(&binding, &path, node, &args);
        };
        if shadowed {
            let origin = self.unknown("rebound-name", node, &spelled);
            self.diag(
                &format!("'{spelled}' is assigned as well as defined, so what it calls is unknown"),
                node,
                Some(origin),
                &format!("call '{spelled}' under a name nothing assigns to"),
            );
            return Ok(Some(UNKNOWN.to_string()));
        }
        if self.overload_groups.contains(&key) {
            self.ensure_group(&key, frame)?;
            if !self.group_call_fits(&key, &name, node, Some(&args))? {
                self.record_pin(node, None)?;
                return Ok(Some(UNKNOWN.to_string()));
            }
            let (_, _, keywords) = facts::call_parts(self.ast, node).expect("a call");
            let keywords = keywords.to_vec();
            let shapes = self.shapes(&args);
            let pin = match &shapes {
                Some(shapes) if keywords.is_empty() => pin_for(shapes),
                _ => None,
            };
            self.record_pin(node, pin.clone())?;
            return self.group_type(&key, pin.as_deref(), &args, &keywords).map(Some);
        }
        self.call_context(&key, node, frame, Some(&args), true).map(Some)
    }

    /// Report a `method_call` that reaches no method, naming the method
    /// (`_unknown_method`).
    fn unknown_method(&mut self, node: NodeId) -> R<String> {
        let (_, args, _) = facts::call_parts(self.ast, node).expect("a call");
        if args.len() < 2 {
            self.node_diag(
                "'method_call' takes a method and a receiver",
                node,
                "bad-call",
                "method_call",
                "spell it 'method_call(method, receiver, ...)'",
            );
            return Ok(UNKNOWN.to_string());
        }
        let (selector, receiver) = (args[0], args[1]);
        let spelled = match facts::str_constant(self.ast, selector) {
            Some(text) => text.to_string(),
            None => self.spelled(selector)?,
        };
        let receiver_ty = self.ty_of(receiver);
        if receiver_ty == UNKNOWN {
            return Ok(UNKNOWN.to_string());
        }
        if receiver_ty == OBJECT {
            let receiver_spelled = self.spelled(receiver)?;
            self.node_diag(
                &format!(
                    "the class of '{receiver_spelled}' is not known here, so its method '{spelled}' \
                     has no type"
                ),
                node,
                "unknown-class",
                &spelled,
                "annotate the receiver with the type it holds",
            );
            return Ok(UNKNOWN.to_string());
        }
        let cid = class_of(&receiver_ty).map(str::to_string);
        if matches!(self.kind(selector), Kind::Constant { .. })
            && let Some(cid) = cid
            && self.declares_method(&cid, &spelled, node)?
        {
            self.node_diag(
                &format!(
                    "'{spelled}' is declared on {}, but no export the runtime searches by name \
                     holds it",
                    render_ty(&receiver_ty)
                ),
                node,
                "unknown-method",
                &spelled,
                &format!("pass the method itself: 'method_call({spelled}, ...)'"),
            );
            return Ok(UNKNOWN.to_string());
        }
        self.node_diag(
            &format!("'{spelled}' is not a method of {} here", render_ty(&receiver_ty)),
            node,
            "unknown-method",
            &spelled,
            &format!("declare '{spelled}' on the receiver's class, or call a method its shape has"),
        );
        Ok(UNKNOWN.to_string())
    }

    /// Whether a module within reach declares a `@method` of that name on a class
    /// (`_declares_method`).
    fn declares_method(&mut self, cid: &str, name: &str, node: NodeId) -> R<bool> {
        if self.class_methods.get(cid).is_some_and(|methods| methods.contains_key(name)) {
            return Ok(true);
        }
        if let Some(sig) = self.class_sig(cid, node)?
            && sig.methods.contains_key(name)
        {
            return Ok(true);
        }
        for (_, binding) in self.import_sources() {
            let mut hops_list: Vec<Vec<String>> = vec![binding.attrs.clone()];
            let shorter = binding.attrs[..binding.attrs.len().saturating_sub(1)].to_vec();
            if shorter != binding.attrs {
                hops_list.push(shorter);
            }
            for hops in hops_list {
                if let Some(interface) = self.interface_of(&binding.module, &hops, node)?
                    && interface
                        .extensions
                        .get(cid)
                        .is_some_and(|methods| methods.contains_key(name))
                {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    /// The import a named selector comes from, and the path to it (`_imported_selector`).
    fn imported_selector(&self, spelled: &str) -> Option<(super::Import, Vec<String>)> {
        let (head, rest) = spelled.split_once('.').unwrap_or((spelled, ""));
        let binding = self.imports.get(head)?;
        if self.multi_imports.contains(head)
            || self.import_rebound(head)
            || self.nearer_binding(head)
        {
            return None;
        }
        let mut path = binding.attrs.clone();
        path.extend(rest.split('.').filter(|part| !part.is_empty()).map(str::to_string));
        if path.is_empty() { None } else { Some((binding.clone(), path)) }
    }

    /// Type a named method reached through an import (`_imported_method`).
    fn imported_method(
        &mut self,
        binding: &super::Import,
        path: &[String],
        node: NodeId,
        args: &[NodeId],
    ) -> R<Option<String>> {
        let (last, hops) = path.split_last().expect("a path");
        let Some(interface) = self.interface_of(&binding.module, hops, node)? else {
            return Ok(None);
        };
        let Some(sig) = interface.exports.get(last).cloned() else { return Ok(None) };
        self.published_result(&sig, last, node, args).map(Some)
    }

    /// Type an `obj.method(...)` call on a receiver whose class is known (`_method_of`).
    fn method_of(&mut self, node: NodeId) -> R<Option<String>> {
        let (func, args, _) = facts::call_parts(self.ast, node).expect("a call");
        let args = args.to_vec();
        let Kind::Attribute { value, attr, .. } = self.kind(func).clone() else { return Ok(None) };
        let receiver = self.ty_of(value);
        if receiver == PINE_LOOP {
            return self.loop_step(node, func).map(Some);
        }
        let Some(cid) = class_of(&receiver).map(str::to_string) else { return Ok(None) };
        let mut all = vec![value];
        all.extend(args);
        self.method_result(&cid, &attr, node, &all)
    }

    /// What a class id declares, loading the module that declares it if needed
    /// (`_class_sig`).
    pub(super) fn class_sig(&mut self, cid: &str, node: NodeId) -> R<Option<Rc<ClassSig>>> {
        if let Some(sig) = self.class_sigs.get(cid) {
            return Ok(Some(sig.clone()));
        }
        let source = cid.rsplit_once('#').map_or("", |(source, _)| source).to_string();
        if source.is_empty() || source == "lib" {
            return Ok(None);
        }
        let Some(interface) = self.interface_at(&source, &source, node)? else { return Ok(None) };
        for published in interface.classes.values() {
            self.class_sigs
                .entry(published.id.clone())
                .or_insert_with(|| Rc::new(published.clone()));
        }
        Ok(self.class_sigs.get(cid).cloned())
    }

    /// Type one call to a method of a known class, spelled by its name
    /// (`_method_result`).
    fn method_result(
        &mut self,
        cid: &str,
        name: &str,
        node: NodeId,
        args: &[NodeId],
    ) -> R<Option<String>> {
        if let Some(member) = self.member_result(cid, name, node, args)? {
            return Ok(Some(member));
        }
        let sig = self.class_sig(cid, node)?;
        let source = cid.rsplit_once('#').map_or("", |(source, _)| source).to_string();
        let builtin = source.is_empty() || source == "lib";
        let Some(sig) = sig else { return Ok(None) };
        if name.starts_with('_')
            || sig.fields.contains_key(name)
            || (!builtin && UDT_ATTRS.contains(&name))
        {
            return Ok(None);
        }
        let mut home = None;
        if !builtin && !self.class_sigs_table.contains_key(cid) {
            home = self.interface_at(&source, &source, node)?;
            if home.is_none() {
                return Ok(None);
            }
        }
        let Some(published) = self.library_method(name, node, home)? else { return Ok(None) };
        self.published_result(&published, name, node, args).map(Some)
    }

    /// Type a call to a method its class declares in its own body (`_member_result`):
    /// one of this module's classes in the context the call describes, another
    /// module's through the signature it publishes.
    fn member_result(
        &mut self,
        cid: &str,
        name: &str,
        node: NodeId,
        args: &[NodeId],
    ) -> R<Option<String>> {
        if let Some((key, scope)) =
            self.members.get(cid).and_then(|members| members.get(name)).cloned()
        {
            let Some(frame) =
                (0..self.stack.len()).rev().find(|&at| self.frames[self.stack[at]].scope == scope)
            else {
                return Ok(None);
            };
            let (func, _, _) = facts::call_parts(self.ast, node).expect("a call");
            let record = self.dotted(func).as_deref() != Some("method_call");
            return self.call_context(&key, node, frame, Some(args), record).map(Some);
        }
        let Some(sig) = self.class_sig(cid, node)? else { return Ok(None) };
        let Some(published) = sig.members.get(name).cloned() else { return Ok(None) };
        self.published_result(&published, name, node, args).map(Some)
    }

    /// Type a call to a method another module publishes (`_published_result`).
    fn published_result(
        &mut self,
        sig: &ExportSig,
        name: &str,
        node: NodeId,
        args: &[NodeId],
    ) -> R<String> {
        if !self.export_call_fits(node, name, sig, Some(args))? {
            return Ok(UNKNOWN.to_string());
        }
        if sig.kind == "group" {
            let receiver = args.first().map_or(UNKNOWN.to_string(), |&arg| self.ty_of(arg));
            let returns: Vec<&str> = sig
                .impls
                .iter()
                .filter(|imp| takes_receiver(imp, &receiver))
                .map(|imp| imp.ret.as_str())
                .collect();
            return Ok(overload_result(&returns).to_string());
        }
        Ok(if sig.annotated { sig.ret.clone() } else { UNKNOWN.to_string() })
    }

    /// Type a `__sec_read__('id', default)` the security transform emitted
    /// (`_security_read`).
    fn security_read(&mut self, node: NodeId) -> R<String> {
        let (_, args, _) = facts::call_parts(self.ast, node).expect("a call");
        let sec_id = args.first().copied();
        let default = args.get(1).copied();
        let mut parts: Vec<String> = Vec::new();
        if let Some(sec_id) = sec_id
            && let Kind::Constant { value: Constant::Str(id), .. } = self.kind(sec_id)
        {
            for &written in self.sec_writes.get(id).map(Vec::as_slice).unwrap_or(&[]) {
                parts.push(self.ty_of(written));
            }
        }
        if let Some(default) = default
            && self.dotted(default).as_deref() != Some("lib._na_none")
        {
            parts.push(self.ty_of(default));
        }
        let mut result = UNKNOWN.to_string();
        if let Some((first, rest)) = parts.split_first() {
            result = rest.iter().fold(first.clone(), |result, ty| join(&result, ty));
        }
        if let Some(default) = default {
            let default_ty = self.ty_of(default);
            for (value, path) in facts::typeless_nas(self.ast, default, &default_ty) {
                self.record_widen(value, facts::na_widen(&facts::element_at(&result, &path)))?;
            }
        }
        Ok(result)
    }

    /// Walk any implementation of a group the call got ahead of (`_ensure_group`).
    fn ensure_group(&mut self, key: &str, frame: usize) -> R<()> {
        let parents: Vec<usize> = self.stack[..=frame].to_vec();
        for imp in self.defs.get(key).cloned().unwrap_or_default() {
            let walked = self.nid(imp).is_some_and(|nid| self.walked.contains(&nid));
            if !walked {
                self.type_defaults(imp, Some(parents.clone()))?;
                let params = self.declared_params(imp)?;
                self.analyse(key, imp, params, Some(parents.clone()), None, None)?;
            }
        }
        Ok(())
    }

    /// Type of a call to a user overload group (`_group_type`).
    fn group_type(
        &mut self,
        key: &str,
        pin: Option<&str>,
        args: &[NodeId],
        keywords: &[NodeId],
    ) -> R<String> {
        let impls = self.impl_sigs(key)?;
        let shapes = self.shapes(args);
        let kw_shapes = self.keyword_shapes(keywords);
        let Some(choice) = self.choice(&impls, pin, shapes.as_deref(), kw_shapes.as_ref()) else {
            return Ok(self.funcs[key].ret.clone());
        };
        self.widen_group_nas(&choice, args, keywords)?;
        Ok(choice.ret)
    }

    /// Convert the typeless `na` arguments of a call the exact pass settled
    /// (`_widen_group_nas`).
    pub(super) fn widen_group_nas(
        &mut self,
        imp: &ImplSig,
        args: &[NodeId],
        keywords: &[NodeId],
    ) -> R<()> {
        let mut bound: Vec<(String, NodeId)> = args
            .iter()
            .enumerate()
            .filter(|(index, _)| *index < imp.params.len())
            .map(|(index, &arg)| (imp.params[index].clone(), arg))
            .collect();
        for &keyword in keywords {
            let (name, value) = keyword_parts(self.ast, keyword);
            if let Some(name) = name
                && let Some(at) = imp.names.iter().position(|known| known == name)
            {
                bound.push((imp.params[at].clone(), value));
            }
        }
        for (declared, value) in bound {
            if self.ty_of(value) == TYPELESS && !NUMERIC.contains(&declared.as_str()) {
                self.record_widen(value, facts::na_widen(&declared))?;
            }
        }
        Ok(())
    }

    /// Every implementation of a user group, as the static selection reads it
    /// (`_impl_sigs`).
    pub(super) fn impl_sigs(&mut self, key: &str) -> R<Vec<ImplSig>> {
        let returns = self.group_returns.get(key).cloned().unwrap_or_default();
        let mut out = Vec::new();
        for imp in self.defs.get(key).cloned().unwrap_or_default() {
            let required_kwonly = facts::args_of(self.ast, imp)
                .is_some_and(|args| args.kw_defaults.iter().any(Option::is_none));
            if required_kwonly {
                continue;
            }
            let slot =
                self.group_slot.get(&self.nid(imp).filter(|&nid| nid != 0).unwrap_or(-1)).copied();
            let ret = match slot {
                Some(slot) if slot < returns.len() => returns[slot].clone(),
                _ => UNKNOWN.to_string(),
            };
            out.push(impl_sig(
                self.ast,
                imp,
                &ret,
                &|value| self.ty_of(value),
                &self.classes_lookup,
            )?);
        }
        Ok(out)
    }

    /// Analyse the callee in the context this call site describes, and type it
    /// (`_call_context`).
    ///
    /// A site with `record` false (a method a `method_call` finds by name on the
    /// receiver's class) takes no part in the per-instance channel.
    fn call_context(
        &mut self,
        key: &str,
        node: NodeId,
        frame: usize,
        args: Option<&[NodeId]>,
        record: bool,
    ) -> R<String> {
        let Some(definitions) = self.defs.get(key).filter(|defs| !defs.is_empty()).cloned() else {
            return Ok(self.funcs[key].ret.clone());
        };
        let target = *definitions.last().expect("a definition");
        let parents: Vec<usize> = self.stack[..=frame].to_vec();
        let (func, own_args, _) = facts::call_parts(self.ast, node).expect("a call");
        let positional: Vec<NodeId> = args.unwrap_or(own_args).to_vec();
        let params = self.call_params(target, node, &positional, true)?;
        let nid = self.nid(node);
        let Some(params) = params else {
            if self.bad_calls.contains(&node) {
                return Ok(UNKNOWN.to_string());
            }
            let declared = self.declared_params(target)?;
            let Some(settled) = self.analyse(key, target, declared, Some(parents), None, None)?
            else {
                return Ok(UNKNOWN.to_string());
            };
            let (cid, ret, ret_qual) = {
                let result = &self.results[settled];
                (result.cid, result.ret.clone(), result.ret_qual)
            };
            self.unvectored.insert(cid);
            if let Some(nid) = nid {
                self.call_q.insert(nid, ret_qual);
            }
            return Ok(ret);
        };
        let spelled = self.dotted(func).unwrap_or_else(|| "None".to_string());
        let guard = (key.to_string(), params.clone(), self.nid(target));
        if self.in_progress.contains(&guard) {
            let origin = self.unknown("recursion", node, key);
            self.diag(
                &format!("'{spelled}' is re-entrant, so its result type is unknown"),
                node,
                Some(origin),
                "",
            );
            return Ok(UNKNOWN.to_string());
        }
        let quals = self.call_quals(target, node, &positional)?;
        let Some(result) =
            self.analyse(key, target, params, Some(parents), Some(node), Some(quals))?
        else {
            let origin = self.unknown("context-budget", node, key);
            self.diag(
                &format!("'{spelled}' hit the per-module context limit"),
                node,
                Some(origin),
                "",
            );
            return Ok(UNKNOWN.to_string());
        };
        let (cid, ret, ret_qual) = {
            let found = &self.results[result];
            (found.cid, found.ret.clone(), found.ret_qual)
        };
        if let Some(nid) = nid {
            self.call_q.insert(nid, ret_qual);
        }
        match nid {
            Some(nid) if record => {
                let context = self.context();
                self.call_ctx.entry(nid).or_default().insert(context, cid);
                self.callee_key.insert(nid, key.to_string());
            }
            _ => {
                self.unvectored.insert(cid);
            }
        }
        Ok(ret)
    }

    /// The parameter types one call site instantiates the callee with (`_call_params`).
    fn call_params(
        &mut self,
        target: NodeId,
        node: NodeId,
        args: &[NodeId],
        report: bool,
    ) -> R<Option<Vec<String>>> {
        let Some(declared) = facts::args_of(self.ast, target) else { return Ok(None) };
        let name = facts::def_name(self.ast, target).to_string();
        if declared.vararg.is_some() || declared.kwarg.is_some() {
            return Ok(None);
        }
        let positional = declared.positional();
        let kwonly = declared.kwonlyargs.to_vec();
        if args.iter().any(|&arg| facts::is_starred(self.ast, arg)) {
            return Ok(None);
        }
        let (_, _, keywords) = facts::call_parts(self.ast, node).expect("a call");
        let keywords = keywords.to_vec();
        if keywords.iter().any(|&keyword| keyword_parts(self.ast, keyword).0.is_none()) {
            return Ok(None);
        }
        if args.len() > positional.len() {
            if report {
                self.bad_call(
                    node,
                    &name,
                    &format!(
                        "'{name}' takes {} positional argument(s), {} passed",
                        positional.len(),
                        args.len()
                    ),
                );
            }
            return Ok(None);
        }
        let mut bound: indexmap::IndexMap<String, NodeId> = indexmap::IndexMap::new();
        for (&arg, &value) in positional.iter().zip(args) {
            bound.insert(facts::arg_name(self.ast, arg), value);
        }
        let declared_names: Vec<String> =
            positional.iter().chain(&kwonly).map(|&arg| facts::arg_name(self.ast, arg)).collect();
        for &keyword in &keywords {
            let (arg, value) = keyword_parts(self.ast, keyword);
            let arg = arg.unwrap_or("").to_string();
            if !declared_names.contains(&arg) {
                if report {
                    self.bad_call(node, &name, &format!("'{name}' has no parameter '{arg}'"));
                }
                return Ok(None);
            }
            if bound.contains_key(&arg) {
                if report {
                    self.bad_call(node, &name, &format!("'{arg}' is passed to '{name}' twice"));
                }
                return Ok(None);
            }
            bound.insert(arg, value);
        }
        let defaults = param_defaults(self.ast, target)?;
        let mut out = Vec::new();
        for arg in every_param(self.ast, target) {
            let (arg_name, annotation) = arg_parts(self.ast, arg);
            let arg_name = arg_name.to_string();
            let passed = bound.get(&arg_name).copied();
            let default = defaults.iter().find(|(known, _)| *known == arg_name).map(|(_, d)| *d);
            if passed.is_none() && default.is_none() {
                if report {
                    self.bad_call(
                        node,
                        &name,
                        &format!("'{name}' needs an argument for '{arg_name}'"),
                    );
                }
                return Ok(None);
            }
            let annotated = self.annotation_type(annotation)?;
            if annotated != UNKNOWN {
                let given_node = passed.or(default).expect("passed or defaulted");
                let given = self.ty_of(given_node);
                if passed.is_some() && !unchecked(&given) && !fits(&annotated, &given) {
                    if report {
                        self.bad_call(
                            node,
                            &name,
                            &format!(
                                "'{name}' takes {} for '{arg_name}', {} passed",
                                render_ty(&annotated),
                                render_ty(&given)
                            ),
                        );
                    }
                    return Ok(None);
                }
                if report && given == TYPELESS {
                    self.record_widen(given_node, facts::na_widen(&annotated))?;
                }
                out.push(annotated);
            } else if let (None, Some(default)) = (passed, default) {
                out.push(self.ty_of(default));
            } else if let (Some(passed), None) = (passed, default) {
                out.push(self.ty_of(passed));
            } else {
                let (passed, default) = (passed.expect("passed"), default.expect("default"));
                let joined = join(&self.ty_of(default), &self.ty_of(passed));
                if report && self.ty_of(passed) == TYPELESS {
                    self.record_widen(passed, facts::na_widen(&joined))?;
                }
                out.push(joined);
            }
        }
        Ok(Some(out))
    }

    /// The parameter qualifiers one call site instantiates the callee with
    /// (`_call_quals`).
    fn call_quals(&self, target: NodeId, node: NodeId, args: &[NodeId]) -> R<Vec<u8>> {
        let positional =
            facts::args_of(self.ast, target).map(|a| a.positional()).unwrap_or_default();
        let mut bound: std::collections::HashMap<String, NodeId> = std::collections::HashMap::new();
        for (&arg, &value) in positional.iter().zip(args) {
            bound.insert(facts::arg_name(self.ast, arg), value);
        }
        let (_, _, keywords) = facts::call_parts(self.ast, node).expect("a call");
        for &keyword in keywords {
            let (arg, value) = keyword_parts(self.ast, keyword);
            if let Some(arg) = arg {
                bound.insert(arg.to_string(), value);
            }
        }
        let defaults = param_defaults(self.ast, target)?;
        Ok(every_param(self.ast, target)
            .into_iter()
            .map(|arg| {
                let name = facts::arg_name(self.ast, arg);
                let passed = bound
                    .get(&name)
                    .copied()
                    .or_else(|| defaults.iter().find(|(known, _)| *known == name).map(|(_, d)| *d));
                passed.map_or(SERIES, |passed| self.qual(passed))
            })
            .collect())
    }

    /// A call whose shape or argument types the callee does not take (`_bad_call`).
    pub(super) fn bad_call(&mut self, node: NodeId, name: &str, message: &str) {
        self.bad_calls.insert(node);
        self.node_diag(
            message,
            node,
            "bad-call",
            name,
            &format!("call '{name}' the way it is declared"),
        );
    }
}
