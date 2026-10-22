//! Calls into other modules, typed from the interfaces those modules publish, which
//! the host answers.

use std::rc::Rc;

use indexmap::IndexMap;
use pyne_ast::{Kind, NodeId};

use super::facts::{self, Pos, call_arity, fits, keyword_parts, unchecked};
use super::{Engine, Import, R, Shadowed};
use crate::pine_type_rules::{UNKNOWN, render_ty};
use crate::pine_type_table::{CallSite, ExportSig, ModuleInterface};

/// `str(Path(path).parent)` for a POSIX path.
fn path_parent(path: &str) -> String {
    let trimmed = path.trim_end_matches('/');
    if trimmed.is_empty() {
        return if path.starts_with('/') { "/".to_string() } else { ".".to_string() };
    }
    match trimmed.rfind('/') {
        None => ".".to_string(),
        Some(0) => "/".to_string(),
        Some(index) => trimmed[..index].trim_end_matches('/').to_string(),
    }
}

/// `Path(path).name` for a POSIX path.
fn path_name(path: &str) -> &str {
    let trimmed = path.trim_end_matches('/');
    trimmed.rsplit('/').next().unwrap_or("")
}

impl Engine<'_, '_> {
    /// Type a call whose callee comes from another module (`_imported_call`).
    pub(super) fn imported_call(&mut self, node: NodeId, name: &str) -> R<Option<String>> {
        let (head, rest) = name.split_once('.').unwrap_or((name, ""));
        let parts: Vec<String> =
            rest.split('.').filter(|part| !part.is_empty()).map(str::to_string).collect();
        let shadow = self.shadowed.get(head).cloned();
        let binding = match &shadow {
            Some(shadow) => Some(shadow.source.clone()),
            None => self.imports.get(head).cloned(),
        };
        let Some(binding) = binding else { return Ok(None) };
        if self.nearer_binding(head) {
            return Ok(Some(UNKNOWN.to_string()));
        }
        if self.multi_imports.contains(head) {
            let origin = self.unknown("rebound-name", node, head);
            self.diag(
                &format!("'{head}' is imported more than once, so what it calls is unknown"),
                node,
                Some(origin),
                &format!("import '{head}' once, under a name nothing else binds"),
            );
            return Ok(Some(UNKNOWN.to_string()));
        }
        if self.import_rebound(head) {
            let origin = self.unknown("rebound-name", node, head);
            self.diag(
                &format!("'{head}' is assigned as well as imported, so what it calls is unknown"),
                node,
                Some(origin),
                &format!("call '{head}' under a name nothing assigns to"),
            );
            return Ok(Some(UNKNOWN.to_string()));
        }
        if let Some(shadow) = shadow {
            return self.shadowed_call(node, name, &shadow, &parts).map(Some);
        }
        let mut path = binding.attrs.clone();
        path.extend(parts);
        let Some((last, hops)) = path.split_last() else {
            return Ok(Some(UNKNOWN.to_string()));
        };
        let Some(interface) = self.interface_of(&binding.module, hops, node)? else {
            return Ok(Some(UNKNOWN.to_string()));
        };
        match interface.exports.get(last).cloned() {
            None => Ok(Some(UNKNOWN.to_string())),
            Some(sig) => self.export_call(node, name, &sig, &interface).map(Some),
        }
    }

    /// Type a call through an alias that merges a library over a builtin namespace
    /// (`_shadowed_call`).
    fn shadowed_call(
        &mut self,
        node: NodeId,
        name: &str,
        shadow: &Shadowed,
        parts: &[String],
    ) -> R<String> {
        if parts.len() != 1 {
            return Ok(UNKNOWN.to_string());
        }
        let member = &parts[0];
        let Some(interface) =
            self.interface_of(&shadow.source.module, &shadow.source.attrs, node)?
        else {
            return Ok(UNKNOWN.to_string());
        };
        let published = match &interface.all {
            Some(all) => all.contains(member),
            None => interface.exports.contains_key(member),
        };
        if !published {
            return self.lib_call(node, &format!("{}.{member}", shadow.namespace));
        }
        match interface.exports.get(member).cloned() {
            None => Ok(UNKNOWN.to_string()),
            Some(sig) => self.export_call(node, name, &sig, &interface),
        }
    }

    /// Type one call against the signature its module publishes (`_export_call`).
    fn export_call(
        &mut self,
        node: NodeId,
        name: &str,
        sig: &ExportSig,
        interface: &ModuleInterface,
    ) -> R<String> {
        let is_group = sig.kind == "group";
        let (line, col) = (self.line(node), self.col(node));
        let argc = call_arity(self.ast, node);
        if !self.export_call_fits(node, name, sig, None)? {
            self.calls.push(CallSite {
                callee: name.to_string(),
                line,
                col,
                argc,
                ty: UNKNOWN.to_string(),
                pin: None,
            });
            return Ok(UNKNOWN.to_string());
        }
        let pin = self.pin(node, is_group);
        let ty = if is_group {
            let (_, args, keywords) = facts::call_parts(self.ast, node).expect("a call");
            let (args, keywords) = (args.to_vec(), keywords.to_vec());
            let shapes = self.shapes(&args);
            let kw_shapes = self.keyword_shapes(&keywords);
            match self.choice(&sig.impls, pin.as_deref(), shapes.as_deref(), kw_shapes.as_ref()) {
                None => sig.ret.clone(),
                Some(choice) => {
                    self.widen_group_nas(&choice, &args, &keywords)?;
                    choice.ret
                }
            }
        } else if sig.annotated {
            sig.ret.clone()
        } else {
            let place = format!("{}:{}", interface.path, sig.line);
            let origin = self.unknown("unannotated-import", node, &sig.name);
            self.diag(
                &format!(
                    "'{}' is imported from {place}, where its parameters carry no annotations, so \
                     what it returns is unknown",
                    sig.name
                ),
                node,
                Some(origin),
                &format!("annotate the parameters of '{}' in {place}", sig.name),
            );
            UNKNOWN.to_string()
        };
        self.calls.push(CallSite {
            callee: name.to_string(),
            line,
            col,
            argc,
            ty: ty.clone(),
            pin,
        });
        Ok(ty)
    }

    /// Whether a call matches the signature its module publishes (`_export_call_fits`).
    pub(super) fn export_call_fits(
        &mut self,
        node: NodeId,
        name: &str,
        sig: &ExportSig,
        args: Option<&[NodeId]>,
    ) -> R<bool> {
        let (_, own_args, keywords) = facts::call_parts(self.ast, node).expect("a call");
        let positional_args: Vec<NodeId> = args.unwrap_or(own_args).to_vec();
        let keywords = keywords.to_vec();
        if positional_args.iter().any(|&arg| facts::is_starred(self.ast, arg))
            || keywords.iter().any(|&keyword| keyword_parts(self.ast, keyword).0.is_none())
        {
            return Ok(true);
        }
        // (required, open-ended, parameter count, names) of every shape the export has
        let shapes: Vec<(i64, bool, usize, Vec<String>)> = if sig.kind == "group" {
            sig.impls
                .iter()
                .map(|imp| {
                    (imp.required as i64, imp.open_ended, imp.params.len(), imp.names.clone())
                })
                .collect()
        } else {
            vec![(sig.required, sig.open_ended, sig.params.len(), sig.names.clone())]
        };
        let argc = (positional_args.len() + keywords.len()) as i64;
        let takes = shapes.iter().any(|(required, open_ended, params, _)| {
            *required <= argc && (*open_ended || argc <= *params as i64)
        });
        if !takes {
            self.bad_call(node, name, &format!("'{name}' does not take {argc} argument(s)"));
            return Ok(false);
        }
        for &keyword in &keywords {
            let arg = keyword_parts(self.ast, keyword).0.unwrap_or("");
            if !shapes.iter().any(|(_, _, _, names)| names.iter().any(|known| known == arg)) {
                self.bad_call(node, name, &format!("'{name}' has no parameter '{arg}'"));
                return Ok(false);
            }
        }
        if sig.kind == "group" {
            return Ok(true);
        }
        let mut given: Vec<(String, String, NodeId)> = Vec::new();
        for (index, &arg) in positional_args.iter().take(sig.params.len()).enumerate() {
            let pname = sig.names.get(index).cloned().unwrap_or_else(|| index.to_string());
            given.push((pname, sig.params[index].clone(), arg));
        }
        let positional: Vec<&String> = sig.names.iter().take(positional_args.len()).collect();
        for &keyword in &keywords {
            let (arg, value) = keyword_parts(self.ast, keyword);
            let arg = arg.unwrap_or("").to_string();
            if positional.contains(&&arg) {
                self.bad_call(node, name, &format!("'{arg}' is passed to '{name}' twice"));
                return Ok(false);
            }
            if let Some(at) = sig.names.iter().position(|known| *known == arg) {
                given.push((arg, sig.params[at].clone(), value));
            }
        }
        for (pname, declared, arg) in given {
            let passed = self.ty_of(arg);
            if declared == UNKNOWN || unchecked(&passed) || fits(&declared, &passed) {
                continue;
            }
            self.bad_call(
                node,
                name,
                &format!(
                    "'{name}' takes {} for '{pname}', {} passed",
                    render_ty(&declared),
                    render_ty(&passed)
                ),
            );
            return Ok(false);
        }
        Ok(true)
    }

    /// The interface of the module a dotted callee reaches (`_interface_of`).
    pub(super) fn interface_of(
        &mut self,
        module: &str,
        hops: &[String],
        node: NodeId,
    ) -> R<Option<Rc<ModuleInterface>>> {
        let mut dotted = module.to_string();
        for hop in hops {
            dotted.push('.');
            dotted.push_str(hop);
        }
        let Some(path) = self.module_source(&dotted)? else { return Ok(None) };
        self.interface_at(&path, &dotted, node)
    }

    /// The interface of one module source, and the dependency on it (`_interface_at`).
    pub(super) fn interface_at(
        &mut self,
        path: &str,
        dotted: &str,
        node: NodeId,
    ) -> R<Option<Rc<ModuleInterface>>> {
        if !path.is_empty() && self.host.closes_cycle(path)? {
            self.node_diag(
                &format!(
                    "'{dotted}' imports this module back, so its signatures are not available yet"
                ),
                node,
                "import-cycle",
                dotted,
                &format!("break the import cycle between {} and {path}", self.module_path),
            );
            return Ok(None);
        }
        if !self.interfaces.contains_key(path) {
            let interface = self.host.interface(path)?.map(Rc::new);
            self.interfaces.insert(path.to_string(), interface.clone());
            if let Some(interface) = interface {
                if !interface.suppressed.is_empty() {
                    let origin = self.unknown("suppressed-import", node, dotted);
                    self.diag(
                        &format!(
                            "'{dotted}' is reported, and no type it publishes may drive a dispatch \
                             here: {}",
                            interface.suppressed
                        ),
                        node,
                        Some(origin),
                        &format!("fix '{dotted}' first"),
                    );
                    if self.pins_suppressed.is_none() {
                        self.pins_suppressed = self.diags.last().cloned();
                    }
                }
                for sig in interface.classes.values() {
                    self.class_sigs.entry(sig.id.clone()).or_insert_with(|| Rc::new(sig.clone()));
                }
                self.record_dep(&interface);
            }
        }
        Ok(self.interfaces[path].clone())
    }

    /// Record the dependency on one module and the dependencies it carries
    /// (`module_interface.add_dep`).
    fn record_dep(&mut self, interface: &ModuleInterface) {
        let record = interface.record.clone();
        self.deps.insert(record.path.clone(), record);
        for inherited in interface.deps.values() {
            if inherited.path == self.module_path {
                continue;
            }
            self.deps.entry(inherited.path.clone()).or_insert_with(|| inherited.clone());
        }
    }

    /// Record a dependency on every Pyne module this one imports (`_record_imports`).
    pub(super) fn record_imports(&mut self, body: &[NodeId]) -> R<()> {
        let mut paths: IndexMap<String, ()> = IndexMap::new();
        let anchor =
            if self.module_path.is_empty() { None } else { Some(path_parent(&self.module_path)) };
        for stmt in facts::module_statements(self.ast, body) {
            match self.kind(stmt).clone() {
                Kind::Import { names } => {
                    for alias in names {
                        let Kind::Alias { name, asname } = self.kind(alias).clone() else {
                            continue;
                        };
                        if self.publishes_nothing(&name)? {
                            continue;
                        }
                        let parts: Vec<&str> = name.split('.').collect();
                        let first = if asname.is_some() { parts.len() } else { 1 };
                        for end in first..=parts.len() {
                            if let Some(source) = self.module_source(&parts[..end].join("."))? {
                                paths.insert(source, ());
                            }
                        }
                    }
                }
                Kind::ImportFrom { module, names, level } => {
                    let level = level.unwrap_or(0);
                    let module = module.unwrap_or_default();
                    let mut package: Option<String> = None;
                    let source = if level != 0 {
                        let Some(anchor) = &anchor else { continue };
                        let mut directory = anchor.clone();
                        for _ in 0..level - 1 {
                            directory = path_parent(&directory);
                        }
                        let source = self.host.package_member(&directory, &module)?;
                        if module.is_empty() {
                            package = Some(directory);
                        }
                        source
                    } else if !module.is_empty() && !self.publishes_nothing(&module)? {
                        self.module_source(&module)?
                    } else {
                        continue;
                    };
                    if let Some(source) = source {
                        if path_name(&source) == "__init__.py" {
                            package = Some(path_parent(&source));
                        }
                        paths.insert(source, ());
                    }
                    let Some(package) = package else { continue };
                    for alias in names {
                        let Kind::Alias { name, .. } = self.kind(alias).clone() else { continue };
                        if name == "*" {
                            continue;
                        }
                        if let Some(member) = self.host.package_member(&package, &name)? {
                            paths.insert(member, ());
                        }
                    }
                }
                _ => {}
            }
        }
        for path in paths.into_keys() {
            if self.interfaces.contains_key(&path) {
                continue;
            }
            if let Some(interface) = self.host.interface(&path)? {
                self.record_dep(&interface);
            }
        }
        Ok(())
    }

    /// Where a module's source lives, without importing it (`_module_source`).
    fn module_source(&mut self, dotted: &str) -> R<Option<String>> {
        if let Some(found) = self.module_paths.get(dotted) {
            return Ok(found.clone());
        }
        let origin = self.host.module_spec_origin(dotted)?;
        self.module_paths.insert(dotted.to_string(), origin.clone());
        Ok(origin)
    }

    /// Whether a scope between the call and the module binds the imported name
    /// (`_nearer_binding`).
    pub(super) fn nearer_binding(&self, name: &str) -> bool {
        self.stack[1..].iter().any(|&frame| {
            self.rebound.get(&self.frames[frame].scope).is_some_and(|names| names.contains(name))
        })
    }

    /// Whether the module binds an imported name by anything but the import
    /// (`_import_rebound`).
    pub(super) fn import_rebound(&self, name: &str) -> bool {
        let own = self.import_positions.get(name);
        self.module_rebinds.get(name).is_some_and(|binds| {
            binds.iter().any(|(at, _)| !own.is_some_and(|own| own.contains(at)))
        })
    }

    /// The library export a method called by name reaches (`_library_method`).
    pub(super) fn library_method(
        &mut self,
        name: &str,
        node: NodeId,
        home: Option<Rc<ModuleInterface>>,
    ) -> R<Option<ExportSig>> {
        let sources = self.import_sources();
        for lookup in [name.to_string(), format!("{name}__ren__")] {
            if let Some(home) = &home
                && home.proxies.contains(&lookup)
            {
                return Ok(home.exports.get(&lookup).cloned());
            }
            if self.multi_imports.contains(&lookup)
                || self.import_rebound(&lookup)
                || (self.module_def_positions.contains_key(&lookup)
                    && (self.imports.contains_key(&lookup) || self.shadowed.contains_key(&lookup)))
            {
                return Ok(None);
            }
            for (bound, binding) in &sources {
                if *bound != lookup || self.shadowed.contains_key(bound) {
                    continue;
                }
                let Some((last, init)) = binding.attrs.split_last() else { continue };
                if self.interface_of(&binding.module, &binding.attrs, node)?.is_some() {
                    continue;
                }
                let Some(owner) = self.interface_of(&binding.module, init, node)? else {
                    return Ok(None);
                };
                if owner.proxies.contains(last) {
                    return Ok(owner.exports.get(last).cloned());
                }
            }
            for (bound, binding) in &sources {
                if self.multi_imports.contains(bound)
                    || self.import_rebound(bound)
                    || self.module_def_positions.contains_key(bound)
                {
                    return Ok(None);
                }
                let mut spelled = binding.module.clone();
                for attr in &binding.attrs {
                    spelled.push('.');
                    spelled.push_str(attr);
                }
                if !spelled.starts_with("lib.") {
                    continue;
                }
                let Some(interface) = self.interface_of(&binding.module, &binding.attrs, node)?
                else {
                    if let Some((_, init)) = binding.attrs.split_last()
                        && self.interface_of(&binding.module, init, node)?.is_some()
                    {
                        continue;
                    }
                    return Ok(None);
                };
                if !interface.proxies.contains(&lookup) {
                    continue;
                }
                if self.shadowed.contains_key(bound) {
                    let listed = match &interface.all {
                        Some(all) => all.contains(&lookup),
                        None => interface.exports.contains_key(&lookup) && !lookup.starts_with('_'),
                    };
                    if !listed {
                        continue;
                    }
                }
                return Ok(interface.exports.get(&lookup).cloned());
            }
        }
        Ok(None)
    }

    /// Every import of this module, with the name it binds, in the order of the
    /// globals (`_import_sources`).
    pub(super) fn import_sources(&self) -> Vec<(String, Import)> {
        let mut out: Vec<(String, Import)> = Vec::new();
        let mut seen: Vec<(String, Import)> = Vec::new();
        let candidates =
            self.imports.iter().map(|(name, binding)| (name.clone(), binding.clone())).chain(
                self.shadowed.iter().map(|(name, shadow)| (name.clone(), shadow.source.clone())),
            );
        for (name, binding) in candidates {
            let key = (name.clone(), binding.clone());
            if seen.contains(&key) {
                continue;
            }
            seen.push(key);
            out.push((name, binding));
        }
        let first_store = |name: &str| -> Pos {
            let mut stores: Vec<Pos> = self
                .module_rebinds
                .get(name)
                .map(|binds| binds.iter().map(|(at, _)| *at).collect())
                .unwrap_or_default();
            stores.extend(self.module_def_positions.get(name).cloned().unwrap_or_default());
            stores.into_iter().min().unwrap_or((0, 0))
        };
        out.sort_by_key(|(name, _)| first_store(name));
        out
    }
}
