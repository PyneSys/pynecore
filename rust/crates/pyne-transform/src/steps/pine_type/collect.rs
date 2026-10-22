//! What the walk records before it types anything: the lib aliases' classes, the
//! module's imports and classes, every definition with its annotated signature,
//! and, once every body is walked, the methods the classes publish.

use std::collections::HashMap;
use std::rc::Rc;
use std::sync::OnceLock;

use indexmap::IndexMap;
use pyne_ast::walk::walk_statements;
use pyne_ast::{Kind, NodeId};

use super::facts::{self, arg_parts, args_of, every_param, is_method, is_overload, pos};
use super::{Engine, Import, R, Shadowed};
use crate::data;
use crate::pine_type_rules::{
    UNKNOWN, builtin_class_id, class_id, class_of, impl_sig, overload_result,
};
use crate::pine_type_table::{ClassSig, ExportSig, FuncSig, qualify};

/// The classes the lib publishes, with the fields they declare (`lib_classes`).
pub(super) fn lib_classes() -> &'static IndexMap<String, ClassSig> {
    static CLASSES: OnceLock<IndexMap<String, ClassSig>> = OnceLock::new();
    CLASSES.get_or_init(|| {
        let mut names: Vec<&String> = data::classes().keys().collect();
        names.sort();
        names
            .into_iter()
            .map(|name| {
                let cid = builtin_class_id(name);
                let fields = data::classes()[name].iter().cloned().collect();
                let sig = ClassSig {
                    name: name.clone(),
                    id: cid.clone(),
                    fields,
                    required: 0,
                    methods: IndexMap::new(),
                };
                (cid, sig)
            })
            .collect()
    })
}

impl Engine<'_, '_> {
    /// Whether an imported module is one no Pine interface can come from
    /// (`_publishes_nothing`).
    pub(super) fn publishes_nothing(&self, module: &str) -> R<bool> {
        if module == "pynecore" || module.starts_with("pynecore.") {
            return Ok(true);
        }
        self.host.is_stdlib(module)
    }

    /// Record what the module's own imports bind (`_collect_imports`).
    pub(super) fn collect_imports(&mut self, body: &[NodeId]) -> R<()> {
        let statements = facts::module_statements(self.ast, body);
        for &stmt in &statements {
            match self.kind(stmt).clone() {
                Kind::Import { names } => {
                    for alias in names {
                        let Kind::Alias { name, asname } = self.kind(alias).clone() else {
                            continue;
                        };
                        match asname {
                            Some(asname) => self.add_import(
                                &asname,
                                Import { module: name, attrs: vec![] },
                                alias,
                            )?,
                            None => {
                                let head = name.split('.').next().unwrap_or("").to_string();
                                let binding = Import { module: head.clone(), attrs: vec![] };
                                self.add_import(&head, binding, alias)?;
                            }
                        }
                    }
                }
                Kind::ImportFrom { module: Some(module), names, level }
                    if !module.is_empty() && level.unwrap_or(0) == 0 =>
                {
                    for alias in names {
                        let Kind::Alias { name, asname } = self.kind(alias).clone() else {
                            continue;
                        };
                        let bound = asname.clone().unwrap_or_else(|| name.clone());
                        let binding = Import { module: module.clone(), attrs: vec![name.clone()] };
                        self.add_import(&bound, binding, alias)?;
                        if module == "pynecore.types" || module.starts_with("pynecore.types.") {
                            self.type_imports.insert(name.clone(), bound);
                        }
                    }
                }
                _ => {}
            }
        }
        for stmt in statements {
            self.add_shadowed(stmt);
        }
        Ok(())
    }

    /// Record one import binding, unless the module behind it is not ours (`_add_import`).
    fn add_import(&mut self, name: &str, binding: Import, node: NodeId) -> R<()> {
        if self.publishes_nothing(&binding.module)? {
            return Ok(());
        }
        if self.imports.contains_key(name) {
            self.multi_imports.insert(name.to_string());
        }
        self.imports.insert(name.to_string(), binding);
        let at = pos(self.ast, node);
        self.import_positions.entry(name.to_string()).or_default().insert(at);
        Ok(())
    }

    /// Record a `N = shadowed_namespace(<import>, lib.<ns>)` binding (`_add_shadowed`).
    fn add_shadowed(&mut self, stmt: NodeId) {
        let (targets, value) = match self.kind(stmt) {
            Kind::Assign { targets, value, .. } => (targets.clone(), Some(*value)),
            Kind::AnnAssign { target, value, .. } => (vec![*target], *value),
            _ => return,
        };
        let Some(value) = value else { return };
        let Kind::Call { func, args, .. } = self.kind(value).clone() else { return };
        if args.len() != 2 {
            return;
        }
        let Some(called) = self.dotted(func) else { return };
        if called.rsplit('.').next() != Some("shadowed_namespace") {
            return;
        }
        let Kind::Name { id: library, .. } = self.kind(args[0]).clone() else { return };
        let source = self.imports.get(&library).cloned();
        let namespace = self.lib_name(args[1]);
        let (Some(source), Some(namespace)) = (source, namespace) else { return };
        for target in targets {
            let Kind::Name { id, .. } = self.kind(target).clone() else { continue };
            if self.imports.contains_key(&id) || self.shadowed.contains_key(&id) {
                self.multi_imports.insert(id.clone());
            }
            self.shadowed.insert(
                id.clone(),
                Shadowed { source: source.clone(), namespace: namespace.clone() },
            );
            let at = pos(self.ast, target);
            self.import_positions.entry(id).or_default().insert(at);
        }
    }

    /// Every class an annotation of this module may name, with what it holds
    /// (`_collect_classes`).
    pub(super) fn collect_classes(
        &mut self,
        tree: NodeId,
        annotations: &IndexMap<String, NodeId>,
    ) -> R<()> {
        let own = self.own_class_nodes(tree);
        let module_key = self.module_path.clone();
        let names: IndexMap<String, String> =
            own.keys().map(|name| (name.clone(), class_id(&module_key, name))).collect();
        let mut imported: IndexMap<String, ClassSig> = IndexMap::new();
        for (spelled, &node) in annotations {
            let parts: Vec<&str> = spelled.split('.').collect();
            if names.contains_key(spelled) || imported.contains_key(spelled) {
                continue;
            }
            if self.multi_imports.contains(parts[0]) {
                continue;
            }
            let Some(binding) = self.imports.get(parts[0]).cloned() else { continue };
            let (hops, wanted) = if parts.len() == 1 {
                let Some((last, init)) = binding.attrs.split_last() else { continue };
                (init.to_vec(), last.clone())
            } else {
                let mut hops = binding.attrs.clone();
                hops.extend(parts[1..parts.len() - 1].iter().map(|part| part.to_string()));
                (hops, parts[parts.len() - 1].to_string())
            };
            let interface = self.interface_of(&binding.module, &hops, node)?;
            if let Some(found) = interface.and_then(|found| found.classes.get(&wanted).cloned()) {
                imported.insert(spelled.clone(), found);
            }
        }
        let mut classes: IndexMap<String, String> =
            imported.iter().map(|(name, sig)| (name.clone(), sig.id.clone())).collect();
        for (name, cid) in &names {
            classes.insert(name.clone(), cid.clone());
        }
        self.classes_lookup = classes.iter().map(|(a, b)| (a.clone(), b.clone())).collect();
        self.class_spellings = HashMap::new();
        for (spelled, cid) in &classes {
            self.class_spellings.insert(cid.clone(), spelled.clone());
        }
        self.classes = classes;
        for sig in imported.into_values() {
            self.class_sigs.entry(sig.id.clone()).or_insert_with(|| Rc::new(sig));
        }
        self.declare_classes(tree, &own, &names)
    }

    /// Type the fields of this module's own classes, and collect its methods
    /// (`_declare_classes`).
    fn declare_classes(
        &mut self,
        tree: NodeId,
        own: &IndexMap<String, NodeId>,
        names: &IndexMap<String, String>,
    ) -> R<()> {
        let methods = self.declared_methods(tree)?;
        for (name, &node) in own {
            let cid = names[name].clone();
            let mut fields: IndexMap<String, String> = IndexMap::new();
            let mut required = 0;
            for &stmt in facts::def_body(self.ast, node) {
                if let Kind::AnnAssign { target, annotation, value, .. } = self.kind(stmt)
                    && let Kind::Name { id, .. } = self.kind(*target)
                {
                    let ty = self.annotation_type(Some(*annotation))?;
                    fields.insert(id.clone(), ty);
                    if value.is_none() {
                        required += 1;
                    }
                }
            }
            let sig = ClassSig {
                name: name.clone(),
                id: cid.clone(),
                fields,
                required,
                methods: IndexMap::new(),
            };
            self.class_sigs_table.insert(cid, sig);
        }
        self.class_methods = methods;
        for (cid, sig) in &self.class_sigs_table {
            self.class_sigs.insert(cid.clone(), Rc::new(sig.clone()));
        }
        Ok(())
    }

    /// The Pine methods this module declares, by the class they attach to
    /// (`_declared_methods`).
    fn declared_methods(&self, tree: NodeId) -> R<IndexMap<String, IndexMap<String, NodeId>>> {
        let mut out: IndexMap<String, IndexMap<String, NodeId>> = IndexMap::new();
        for node in walk_statements(self.ast, tree) {
            let Kind::FunctionDef { name, .. } = self.kind(node) else { continue };
            if !is_method(self.ast, node) {
                continue;
            }
            let Some(args) = args_of(self.ast, node) else { continue };
            let positional = args.positional();
            let Some(&first) = positional.first() else { continue };
            let declared = self.annotation_type(arg_parts(self.ast, first).1)?;
            if let Some(cid) = class_of(&declared) {
                out.entry(cid.to_string()).or_default().insert(name.clone(), node);
            }
        }
        Ok(out)
    }

    /// The classes this module declares and still holds, by name (`_own_class_nodes`).
    fn own_class_nodes(&self, tree: NodeId) -> IndexMap<String, NodeId> {
        let mut declared: IndexMap<String, Vec<NodeId>> = IndexMap::new();
        for node in walk_statements(self.ast, tree) {
            if let Kind::ClassDef { name, .. } = self.kind(node) {
                declared.entry(name.clone()).or_default().push(node);
            }
        }
        let mut out = IndexMap::new();
        for (name, nodes) in declared {
            let positions: Vec<_> = nodes.iter().map(|&node| pos(self.ast, node)).collect();
            let kept = self
                .module_rebinds
                .get(&name)
                .is_none_or(|binds| binds.iter().all(|(at, _)| positions.contains(at)));
            if kept {
                out.insert(name, *nodes.last().expect("a class"));
            }
        }
        out
    }

    /// Record every definition and its annotated signature (`_collect`).
    pub(super) fn collect(&mut self, body: &[NodeId], scope: &str) -> R<()> {
        for &stmt in body {
            let (name, returns) = match self.kind(stmt) {
                Kind::FunctionDef { name, returns, .. }
                | Kind::AsyncFunctionDef { name, returns, .. } => (name.clone(), *returns),
                _ => {
                    for nested in facts::statement_lists(self.ast, stmt) {
                        self.collect(&nested, scope)?;
                    }
                    continue;
                }
            };
            let key = qualify(scope, &name);
            self.defs.entry(key.clone()).or_default().push(stmt);
            let args = args_of(self.ast, stmt).map(|args| args.positional()).unwrap_or_default();
            let mut params = Vec::with_capacity(args.len());
            for arg in args {
                params.push(self.annotation_type(arg_parts(self.ast, arg).1)?);
            }
            let mut declared = self.annotation_type(returns)?;
            let mut annotated = declared != UNKNOWN;
            if is_overload(self.ast, stmt) {
                self.overload_groups.insert(key.clone());
                let returns = self.group_returns.entry(key.clone()).or_default();
                if let Some(nid) = facts::nid(self.ast, stmt) {
                    self.group_slot.insert(nid, returns.len());
                }
                returns.push(declared);
                declared = overload_result(returns).to_string();
                annotated = annotated && self.annotated_returns.get(&key).copied().unwrap_or(true);
            }
            let line = facts::line(self.ast, stmt);
            self.funcs
                .insert(key.clone(), FuncSig { name: name.clone(), params, ret: declared, line });
            self.annotated_returns.insert(key.clone(), annotated);
            let mut own: Vec<String> = every_param(self.ast, stmt)
                .into_iter()
                .map(|arg| facts::arg_name(self.ast, arg))
                .collect();
            if let Some(args) = args_of(self.ast, stmt) {
                for arg in [args.vararg, args.kwarg].into_iter().flatten() {
                    own.push(facts::arg_name(self.ast, arg));
                }
            }
            let facts = self.scope_facts(stmt);
            let entry = self.rebound.entry(key.clone()).or_default();
            entry.extend(own);
            entry.extend(facts.bound.iter().cloned());
            let inner = facts::def_body(self.ast, stmt).to_vec();
            self.collect(&inner, &key)?;
        }
        Ok(())
    }

    /// Give every class the signatures of the methods declared on it
    /// (`_publish_methods`).
    pub(super) fn publish_methods(&mut self) -> R<()> {
        let mut keys: HashMap<NodeId, String> = HashMap::new();
        for (key, defs) in &self.defs {
            for &node in defs {
                keys.insert(node, key.clone());
            }
        }
        for (cid, methods) in self.class_methods.clone() {
            for (name, node) in methods {
                let key = keys.get(&node).cloned().unwrap_or_default();
                let Some(func) = self.funcs.get(&key) else { continue };
                let ret = func.ret.clone();
                let shape = impl_sig(
                    self.ast,
                    node,
                    &ret,
                    &|value| self.ty_of(value),
                    &self.classes_lookup,
                )?;
                let positional =
                    args_of(self.ast, node).map(|args| args.positional()).unwrap_or_default();
                let mut annotated = true;
                for arg in positional {
                    if self.annotation_type(arg_parts(self.ast, arg).1)? == UNKNOWN {
                        annotated = false;
                        break;
                    }
                }
                let published = ExportSig {
                    name: name.clone(),
                    kind: "function".into(),
                    params: shape.params,
                    required: shape.required as i64,
                    open_ended: shape.open_ended,
                    ret: shape.ret,
                    annotated,
                    impls: Vec::new(),
                    line: facts::line(self.ast, node),
                    names: shape.names,
                };
                match self.class_sigs_table.get_mut(&cid) {
                    Some(sig) => {
                        sig.methods.insert(name, published);
                    }
                    None => {
                        self.extensions.entry(cid.clone()).or_default().insert(name, published);
                    }
                }
            }
        }
        Ok(())
    }
}
