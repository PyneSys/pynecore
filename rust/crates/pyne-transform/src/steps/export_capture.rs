//! Port of `transformers/export_capture.py`: validate library exports before closure
//! and state lowering.
//!
//! An exported function may read immutable scalar bindings from library main scope,
//! including per-bar values; objects, inputs and reassigned bindings stay private
//! (the module is rejected). A captured per-bar scalar is redeclared as a `Series` and
//! read through its library slot, so an export defined on the first bar keeps
//! reading the current value.
//!
//! An enum member counts as a constant once its declaration is proven from source:
//! the imported modules are located and read through the host ([`Host::module_origin`],
//! [`Host::read_source`]) where the Python step reads them, parsed into the same
//! arena, and never executed. The files read are stamped on the module as
//! `_export_capture_deps`.

use std::collections::{BTreeMap, HashMap, HashSet};

use pyne_ast::walk::{Transformer, Visited, children, copy_location, generic_visit, walk};
use pyne_ast::{Ast, Category, Constant, ExprContext, Int, Kind, NodeId, Value};

use crate::pine_qualifier::{
    CONST, builtin_call_qualifier, lib_call_qualifier, lib_value_qualifier,
};
use crate::pine_type_rules::{
    CallRule, Classes, UNKNOWN, annotation_type, builtin_call_type, is_scalar,
};
use crate::{Context, Error, Host, Unsupported};

/// The `enum` classes a member's class must derive from
const ENUM_BASES: &[&str] = &["Enum", "StrEnum", "IntEnum"];

/// Annotation wrappers whose binding is a series, not a constant
const SERIES_WRAPPERS: &[&str] = &["Series", "PersistentSeries", "IBPersistentSeries"];

/// Annotation wrappers whose binding keeps its own state slot already
const STATE_WRAPPERS: &[&str] =
    &["Persistent", "IBPersistent", "PersistentSeries", "IBPersistentSeries"];

type ScopeId = usize;

/// A lexical scope: the module, a function or a lambda (`_Scope`).
struct Scope {
    node: NodeId,
    parent: Option<ScopeId>,
    bindings: HashMap<String, Vec<NodeId>>,
    reads: Vec<NodeId>,
    children: Vec<ScopeId>,
    defaults: Vec<NodeId>,
    globals: HashSet<String>,
    nonlocals: HashSet<String>,
    /// A module scope's `_module_file_path`, empty when the tree carries none
    file: String,
}

impl Scope {
    fn new(node: NodeId, parent: Option<ScopeId>) -> Scope {
        Scope {
            node,
            parent,
            bindings: HashMap::new(),
            reads: Vec::new(),
            children: Vec::new(),
            defaults: Vec::new(),
            globals: HashSet::new(),
            nonlocals: HashSet::new(),
            file: String::new(),
        }
    }
}

/// What a dotted path resolves to (`_EnumResolver.resolve`): a class declared in
/// source, or one of the `enum` base classes by name.
#[derive(Clone)]
enum Resolved {
    Class(NodeId),
    EnumBase,
}

type ScopeName = (ScopeId, String);

/// A dotted identifier path, empty for other expressions (`_path`); an attribute of
/// some other expression keeps its leading dot, as in Python.
fn dotted_path(ast: &Ast, node: NodeId) -> String {
    match ast.kind(node) {
        Kind::Name { id, .. } => id.clone(),
        Kind::Attribute { value, attr, .. } => format!("{}.{attr}", dotted_path(ast, *value)),
        _ => String::new(),
    }
}

fn last_segment(path: &str) -> &str {
    path.rsplit('.').next().unwrap_or(path)
}

fn first_segment(path: &str) -> &str {
    path.split('.').next().unwrap_or(path)
}

fn split(path: &str) -> Vec<String> {
    path.split('.').map(str::to_string).collect()
}

/// The targets of an assignment-like statement, as the Python step lists them.
fn targets(kind: &Kind) -> Vec<NodeId> {
    match kind {
        Kind::Assign { targets, .. } => targets.clone(),
        Kind::AnnAssign { target, .. }
        | Kind::AugAssign { target, .. }
        | Kind::NamedExpr { target, .. } => vec![*target],
        _ => Vec::new(),
    }
}

/// The name an import alias binds in `collect` and `bind`: the alias, or the first
/// component of the module (`alias.asname or alias.name.split('.')[0]`).
fn import_binding(ast: &Ast, alias: NodeId) -> String {
    let Kind::Alias { name, asname } = ast.kind(alias) else { unreachable!() };
    asname.clone().unwrap_or_else(|| first_segment(name).to_string())
}

/// Whether a source file's bytes decode the way `ast.parse` decodes them, as UTF-8:
/// a BOM, a coding declaration other than UTF-8 or a NUL byte is left to Python.
fn source_text(data: &[u8]) -> Result<&str, Unsupported> {
    let refuse = |why: &str| Unsupported(format!("imported source {why}"));
    if data.starts_with(b"\xef\xbb\xbf") {
        return Err(refuse("with a BOM"));
    }
    if data.contains(&0) {
        return Err(refuse("with a NUL byte"));
    }
    let text = std::str::from_utf8(data).map_err(|_| refuse("not in UTF-8"))?;
    for line in text.split('\n').take(2) {
        let line = line.trim_start();
        if !line.starts_with('#') {
            continue;
        }
        let Some(at) = line.find("coding").map(|at| at + "coding".len()) else { continue };
        if !matches!(line[at..].chars().next(), Some(':' | '=')) {
            continue;
        }
        let name: String = line[at + 1..]
            .trim_start_matches([' ', '\t'])
            .chars()
            .take_while(|c| c.is_alphanumeric() || matches!(c, '-' | '_' | '.'))
            .collect::<String>()
            .to_lowercase()
            .replace('_', "-");
        if !(name == "utf-8" || name == "utf8" || name.starts_with("utf-8-")) {
            return Err(refuse(&format!("declaring coding {name}")));
        }
    }
    Ok(text)
}

struct Capture<'c> {
    ast: &'c mut Ast,
    host: &'c dyn Host,
    scopes: Vec<Scope>,
    /// `scopes`: definition node -> its scope, in first-insertion order
    scope_of: HashMap<NodeId, ScopeId>,
    order: Vec<NodeId>,
    mains: Vec<ScopeId>,
    namespace_names: HashSet<String>,
    /// `_EnumResolver.modules`: (name, importer when relative, level) -> module scope
    modules: HashMap<(String, String, Option<i64>), Option<ScopeId>>,
    /// `_EnumResolver.sources`: resolved path -> (mtime_ns, size)
    sources: BTreeMap<String, (i64, i64)>,
}

impl Capture<'_> {
    fn add_scope(&mut self, scope: Scope) -> ScopeId {
        self.scopes.push(scope);
        self.scopes.len() - 1
    }

    /// `scopes[node] = scope`, keeping the first insertion's position.
    fn register(&mut self, node: NodeId, scope: ScopeId) {
        if self.scope_of.insert(node, scope).is_none() {
            self.order.push(node);
        }
    }

    /// `scopes.values()`
    fn all_scopes(&self) -> Vec<ScopeId> {
        self.order.iter().map(|node| self.scope_of[node]).collect()
    }

    fn bind(&mut self, scope: ScopeId, name: String, node: NodeId) {
        self.scopes[scope].bindings.entry(name).or_default().push(node);
    }

    /// The scope binding a name, explicit scope declarations included (`_Scope.resolve`).
    fn resolve(&self, scope: ScopeId, name: &str) -> Option<ScopeId> {
        let mut current = Some(scope);
        let own = &self.scopes[scope];
        if own.globals.contains(name) {
            let mut top = scope;
            while let Some(parent) = self.scopes[top].parent {
                top = parent;
            }
            current = Some(top);
        } else if own.nonlocals.contains(name) {
            current = own.parent;
        }
        while let Some(id) = current {
            if self.scopes[id].bindings.contains_key(name) {
                return Some(id);
            }
            current = self.scopes[id].parent;
        }
        None
    }

    fn root(&self, mut scope: ScopeId) -> ScopeId {
        while let Some(parent) = self.scopes[scope].parent {
            scope = parent;
        }
        scope
    }

    fn collect(&mut self, current: NodeId, scope: ScopeId) {
        match self.ast.kind(current) {
            Kind::FunctionDef { name, args, body, decorator_list, .. }
            | Kind::AsyncFunctionDef { name, args, body, decorator_list, .. } => {
                let (name, args, body, decorators) =
                    (name.clone(), *args, body.clone(), decorator_list.clone());
                self.bind(scope, name, current);
                for decorator in decorators {
                    self.collect(decorator, scope);
                }
                self.collect_function(current, scope, args, body);
            }
            Kind::Lambda { args, body } => {
                let (args, body) = (*args, vec![*body]);
                self.collect_function(current, scope, args, body);
            }
            Kind::ClassDef { name, .. } => {
                let name = name.clone();
                self.bind(scope, name, current);
            }
            Kind::Import { names } | Kind::ImportFrom { names, .. } => {
                let names: Vec<String> =
                    names.iter().map(|&alias| import_binding(self.ast, alias)).collect();
                for name in names {
                    self.bind(scope, name, current);
                }
            }
            Kind::Global { names } => {
                let names = names.clone();
                self.scopes[scope].globals.extend(names);
            }
            Kind::Nonlocal { names } => {
                let names = names.clone();
                self.scopes[scope].nonlocals.extend(names);
            }
            Kind::Assign { value, .. }
            | Kind::AugAssign { value, .. }
            | Kind::NamedExpr { value, .. }
            | Kind::AnnAssign { value: Some(value), .. } => {
                let value = *value;
                self.collect_assignment(current, scope, Some(value));
            }
            Kind::AnnAssign { value: None, .. } => self.collect_assignment(current, scope, None),
            Kind::Name { id, ctx } => {
                if *ctx == ExprContext::Load {
                    self.scopes[scope].reads.push(current);
                } else if !self.scopes[scope].bindings.contains_key(id) {
                    let id = id.clone();
                    self.scopes[scope].bindings.insert(id, vec![current]);
                }
            }
            _ => {
                for child in children(self.ast, current) {
                    self.collect(child, scope);
                }
            }
        }
    }

    fn collect_function(
        &mut self,
        current: NodeId,
        scope: ScopeId,
        args: NodeId,
        body: Vec<NodeId>,
    ) {
        let child = self.add_scope(Scope::new(current, Some(scope)));
        self.register(current, child);
        self.scopes[scope].children.push(child);
        let Kind::Arguments {
            posonlyargs,
            args: positional,
            vararg,
            kwonlyargs,
            kwarg,
            defaults,
            kw_defaults,
        } = self.ast.kind(args)
        else {
            unreachable!()
        };
        let params: Vec<NodeId> = posonlyargs
            .iter()
            .chain(positional)
            .chain(kwonlyargs)
            .chain(vararg)
            .chain(kwarg)
            .copied()
            .collect();
        let defaults: Vec<NodeId> =
            defaults.iter().copied().chain(kw_defaults.iter().flatten().copied()).collect();
        for param in params {
            let Kind::Arg { arg, .. } = self.ast.kind(param) else { unreachable!() };
            let arg = arg.clone();
            self.scopes[child].bindings.insert(arg, vec![param]);
        }
        // Defaults can capture state even if the body only reads a parameter
        for default in defaults {
            let reads: Vec<NodeId> = walk(self.ast, default)
                .into_iter()
                .filter(|&item| {
                    matches!(self.ast.kind(item), Kind::Name { ctx: ExprContext::Load, .. })
                })
                .collect();
            self.scopes[child].defaults.extend(reads);
        }
        for statement in body {
            self.collect(statement, child);
        }
    }

    fn collect_assignment(&mut self, current: NodeId, scope: ScopeId, value: Option<NodeId>) {
        let targets = targets(self.ast.kind(current));
        for &target in &targets {
            for item in walk(self.ast, target) {
                if let Kind::Name { id, ctx: ExprContext::Store } = self.ast.kind(item) {
                    let id = id.clone();
                    self.bind(scope, id, current);
                }
            }
        }
        if let Some(value) = value {
            self.collect(value, scope);
        }
        for target in targets {
            if !matches!(self.ast.kind(target), Kind::Name { .. }) {
                self.collect(target, scope);
            }
        }
    }

    // --- the enum resolver (`_EnumResolver`) ----------------------------------

    /// The module scope of an import, read from source (`_EnumResolver.module`).
    fn module(
        &mut self,
        name: &str,
        owner: ScopeId,
        level: Option<i64>,
    ) -> Result<Option<ScopeId>, Error> {
        let source = self.scopes[self.root(owner)].file.clone();
        let relative = level.unwrap_or(0) != 0;
        let importer = if relative { source } else { String::new() };
        let key = (name.to_string(), importer.clone(), level);
        if let Some(&known) = self.modules.get(&key) {
            return Ok(known);
        }
        self.modules.insert(key.clone(), None);
        let Some(path) = self.host.module_origin(name, &importer, level.unwrap_or(0))? else {
            return Ok(None);
        };
        let Some(read) = self.host.read_source(&path)? else { return Ok(None) };
        let tree = pyne_parse::parse_module_into(self.ast, source_text(&read.data)?)?;
        self.sources.insert(read.resolved.clone(), (read.mtime_ns, read.size));
        let mut scope = Scope::new(tree, None);
        scope.file = read.resolved;
        let scope = self.add_scope(scope);
        let Kind::Module { body, .. } = self.ast.kind(tree) else { unreachable!() };
        for statement in body.clone() {
            self.bind_module(scope, statement);
        }
        self.modules.insert(key, Some(scope));
        Ok(Some(scope))
    }

    /// The module-level bindings of an imported module (`bind`).
    fn bind_module(&mut self, scope: ScopeId, node: NodeId) {
        match self.ast.kind(node) {
            Kind::ClassDef { name, .. }
            | Kind::FunctionDef { name, .. }
            | Kind::AsyncFunctionDef { name, .. } => {
                let name = name.clone();
                self.bind(scope, name, node);
            }
            Kind::Import { names } | Kind::ImportFrom { names, .. } => {
                let names: Vec<String> =
                    names.iter().map(|&alias| import_binding(self.ast, alias)).collect();
                for name in names {
                    self.bind(scope, name, node);
                }
            }
            Kind::Assign { .. }
            | Kind::AnnAssign { .. }
            | Kind::AugAssign { .. }
            | Kind::NamedExpr { .. } => {
                for target in targets(self.ast.kind(node)) {
                    for item in walk(self.ast, target) {
                        if let Kind::Name { id, ctx: ExprContext::Store } = self.ast.kind(item) {
                            let id = id.clone();
                            self.bind(scope, id, node);
                        }
                    }
                }
            }
            _ => {
                for child in children(self.ast, node) {
                    self.bind_module(scope, child);
                }
            }
        }
    }

    /// What a dotted path names, followed through aliases and imports
    /// (`_EnumResolver.resolve`).
    fn resolve_path(
        &mut self,
        parts: &[String],
        scope: ScopeId,
        seen: &HashSet<ScopeName>,
    ) -> Result<Option<(Resolved, ScopeId)>, Error> {
        let Some(first) = parts.first() else { return Ok(None) };
        let Some(owner) = self.resolve(scope, first) else { return Ok(None) };
        let key = (owner, parts.join("."));
        let bindings = &self.scopes[owner].bindings[first];
        if seen.contains(&key) || bindings.len() != 1 {
            return Ok(None);
        }
        let binding = bindings[0];
        let mut seen = seen.clone();
        seen.insert(key);
        let (module, tail, level) = match self.ast.kind(binding) {
            Kind::ClassDef { .. } => {
                return Ok((parts.len() == 1).then_some((Resolved::Class(binding), owner)));
            }
            Kind::Assign { value, .. } | Kind::AnnAssign { value: Some(value), .. } => {
                let alias = dotted_path(self.ast, *value);
                if alias.is_empty() {
                    return Ok(None);
                }
                let mut path = split(&alias);
                path.extend_from_slice(&parts[1..]);
                return self.resolve_path(&path, owner, &seen);
            }
            Kind::Import { names } => {
                let found =
                    names.iter().copied().find(|&alias| import_binding(self.ast, alias) == *first);
                let Some(alias) = found else { return Ok(None) };
                let Kind::Alias { name, asname } = self.ast.kind(alias) else { unreachable!() };
                let prefix = if asname.is_some() { vec![first.clone()] } else { split(name) };
                if parts.len() < prefix.len() || parts[..prefix.len()] != prefix[..] {
                    return Ok(None);
                }
                (name.clone(), parts[prefix.len()..].to_vec(), Some(0))
            }
            Kind::ImportFrom { module, names, level } => {
                let found = names.iter().copied().find(|&alias| {
                    let Kind::Alias { name, asname } = self.ast.kind(alias) else { unreachable!() };
                    asname.as_ref().unwrap_or(name) == first
                });
                let Some(alias) = found else { return Ok(None) };
                let Kind::Alias { name, .. } = self.ast.kind(alias) else { unreachable!() };
                let mut tail = vec![name.clone()];
                tail.extend_from_slice(&parts[1..]);
                (module.clone().unwrap_or_default(), tail, *level)
            }
            _ => return Ok(None),
        };
        if module == "enum"
            && level.unwrap_or(0) == 0
            && tail.len() == 1
            && ENUM_BASES.contains(&tail[0].as_str())
        {
            return Ok(Some((Resolved::EnumBase, owner)));
        }
        match self.module(&module, owner, level)? {
            Some(imported) => self.resolve_path(&tail, imported, &seen),
            None => Ok(None),
        }
    }

    /// Whether a dotted path is a member of an enum declared in source, bound once to
    /// a literal (`_EnumResolver.member`).
    fn member(&mut self, path: &str, scope: ScopeId) -> Result<bool, Error> {
        let parts = split(path);
        if parts.len() < 2 {
            return Ok(false);
        }
        let Some((Resolved::Class(definition), owner)) =
            self.resolve_path(&parts[..parts.len() - 1], scope, &HashSet::new())?
        else {
            return Ok(false);
        };
        let Kind::ClassDef { bases, keywords, body, decorator_list, .. } =
            self.ast.kind(definition)
        else {
            unreachable!()
        };
        if !decorator_list.is_empty() || !keywords.is_empty() {
            return Ok(false);
        }
        let (bases, body) = (bases.clone(), body.clone());
        // Every base is resolved, as the Python list is built: each may read a module
        let mut enum_base = false;
        for base in bases {
            let base = split(&dotted_path(self.ast, base));
            if let Some((Resolved::EnumBase, _)) =
                self.resolve_path(&base, owner, &HashSet::new())?
            {
                enum_base = true;
            }
        }
        if !enum_base {
            return Ok(false);
        }
        let member = &parts[parts.len() - 1];
        let bindings: Vec<NodeId> = body
            .into_iter()
            .filter(|&statement| {
                matches!(self.ast.kind(statement), Kind::Assign { .. } | Kind::AnnAssign { .. })
                    && targets(self.ast.kind(statement)).into_iter().any(|target| {
                        matches!(self.ast.kind(target), Kind::Name { id, .. } if id == member)
                    })
            })
            .collect();
        if bindings.len() != 1 || member.starts_with('_') {
            return Ok(false);
        }
        let value = match self.ast.kind(bindings[0]) {
            Kind::Assign { value, .. } | Kind::AnnAssign { value: Some(value), .. } => *value,
            _ => return Ok(false),
        };
        Ok(matches!(self.ast.kind(value), Kind::Constant { .. }))
    }

    // --- constants ------------------------------------------------------------

    fn constant(
        &mut self,
        value: NodeId,
        scope: ScopeId,
        visiting: &HashSet<ScopeName>,
        allow_series: bool,
    ) -> Result<bool, Error> {
        match self.ast.kind(value) {
            Kind::Constant { .. } => Ok(true),
            Kind::Name { id, .. } => {
                let id = id.clone();
                match self.resolve(scope, &id) {
                    Some(owner) => self.constant_binding(owner, &id, visiting, allow_series),
                    None => Ok(false),
                }
            }
            Kind::Attribute { .. } => {
                let path = dotted_path(self.ast, value);
                if let Some(key) = path.strip_prefix("lib.")
                    && let Some(entry) = self.host.lib_name(key)?
                {
                    if entry.kind != "value" || entry.callable {
                        return Ok(false);
                    }
                    let ty = entry.ty.ok_or_else(|| {
                        Error::Unsupported(format!("lib value {key} without a type"))
                    })?;
                    return Ok(
                        lib_value_qualifier(key, &ty) == CONST || allow_series && is_scalar(&ty)
                    );
                }
                let first = first_segment(&path);
                if let Some(owner) = self.resolve(scope, first) {
                    let local_enum = self.scopes[owner].bindings[first].iter().any(|&binding| {
                        let Kind::ClassDef { bases, .. } = self.ast.kind(binding) else {
                            return false;
                        };
                        bases.iter().any(|&base| {
                            ENUM_BASES.contains(&last_segment(&dotted_path(self.ast, base)))
                        })
                    });
                    if local_enum {
                        return Ok(true);
                    }
                }
                self.member(&path, scope)
            }
            Kind::UnaryOp { .. }
            | Kind::BinOp { .. }
            | Kind::BoolOp { .. }
            | Kind::Compare { .. }
            | Kind::IfExp { .. } => {
                for child in children(self.ast, value) {
                    if self.ast.kind(child).class().category != Category::Expr {
                        continue;
                    }
                    if !self.constant(child, scope, visiting, allow_series)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            Kind::Subscript { value: base, slice, .. } if allow_series => {
                let (base, slice) = (*base, *slice);
                Ok(self.constant(base, scope, visiting, true)?
                    && self.constant(slice, scope, visiting, true)?)
            }
            Kind::Call { func, args, keywords } => {
                let path = dotted_path(self.ast, *func);
                let mut arguments = args.clone();
                for &keyword in keywords {
                    let Kind::Keyword { value, .. } = self.ast.kind(keyword) else {
                        unreachable!()
                    };
                    arguments.push(*value);
                }
                self.constant_call(&path, &arguments, scope, visiting, allow_series)
            }
            _ => Ok(false),
        }
    }

    fn constant_call(
        &mut self,
        path: &str,
        arguments: &[NodeId],
        scope: ScopeId,
        visiting: &HashSet<ScopeName>,
        allow_series: bool,
    ) -> Result<bool, Error> {
        if self.namespace_names.contains(path) {
            return Ok(arguments.iter().all(|&arg| {
                matches!(self.ast.kind(arg), Kind::Name { .. } | Kind::Attribute { .. })
            }));
        }
        for &arg in arguments {
            if !self.constant(arg, scope, visiting, allow_series)? {
                return Ok(false);
            }
        }
        let consts = || std::iter::repeat_n(CONST, arguments.len());
        if let Some(key) = path.strip_prefix("lib.") {
            if lib_call_qualifier(key, consts(), Some(arguments.len())) == CONST {
                return Ok(true);
            }
            let returns = match self.host.lib_name(key)? {
                Some(entry) if entry.kind == "overloads" => entry.impl_rets,
                Some(entry) => vec![entry.ret],
                None => vec![None],
            };
            return Ok(
                allow_series && returns.iter().all(|ret| ret.as_deref().is_some_and(is_scalar))
            );
        }
        let owner = self.resolve(scope, path);
        if allow_series
            && matches!(builtin_call_type(path), Some(CallRule::Is(ty)) if is_scalar(ty))
        {
            let cast = |owner: ScopeId| {
                self.scopes[owner].bindings[path].iter().all(|&binding| {
                    matches!(self.ast.kind(binding),
                        Kind::ImportFrom { module: Some(module), .. }
                            if module == "pynecore.core.pine_cast")
                })
            };
            if owner.is_none_or(cast) {
                return Ok(true);
            }
        }
        Ok(owner.is_none() && builtin_call_qualifier(path, consts()) == Some(CONST))
    }

    fn constant_binding(
        &mut self,
        scope: ScopeId,
        name: &str,
        visiting: &HashSet<ScopeName>,
        allow_series: bool,
    ) -> Result<bool, Error> {
        let allow_series = allow_series && self.mains.contains(&scope);
        let key = (scope, name.to_string());
        if visiting.contains(&key) {
            return Ok(false);
        }
        let bindings = &self.scopes[scope].bindings[name];
        if bindings.len() != 1 {
            return Ok(false);
        }
        let binding = bindings[0];
        let value = match self.ast.kind(binding) {
            Kind::Import { .. } | Kind::ImportFrom { .. } | Kind::ClassDef { .. } => {
                return Ok(true);
            }
            Kind::Assign { targets, value, .. } => {
                if allow_series
                    && (targets.len() != 1
                        || !matches!(self.ast.kind(targets[0]), Kind::Name { .. }))
                {
                    return Ok(false);
                }
                *value
            }
            Kind::AnnAssign { annotation, value: Some(value), .. } => {
                let (annotation, value) = (*annotation, *value);
                if allow_series {
                    let ty = annotation_type(self.ast, Some(annotation), &Classes::new())?;
                    if ty != UNKNOWN && !is_scalar(&ty) {
                        return Ok(false);
                    }
                }
                let wrapper = match self.ast.kind(annotation) {
                    Kind::Subscript { value, .. } => *value,
                    _ => annotation,
                };
                if !allow_series
                    && SERIES_WRAPPERS.contains(&last_segment(&dotted_path(self.ast, wrapper)))
                {
                    return Ok(false);
                }
                value
            }
            _ => return Ok(false),
        };
        let mut visiting = visiting.clone();
        visiting.insert(key);
        self.constant(value, scope, &visiting, allow_series)
    }
}

/// Read captured scalars through their library slot (`RetainCaptures`).
struct RetainCaptures<'a> {
    reads: &'a HashSet<NodeId>,
    declarations: &'a HashMap<NodeId, NodeId>,
}

impl Transformer for RetainCaptures<'_> {
    fn visit(&mut self, ast: &mut Ast, id: NodeId) -> Visited {
        match ast.kind(id) {
            Kind::Name { .. } => {
                if !self.reads.contains(&id) {
                    return Visited::Node(id);
                }
                let zero =
                    ast.add(Kind::Constant { value: Constant::Int(Int::Small(0)), kind: None });
                let read =
                    ast.add(Kind::Subscript { value: id, slice: zero, ctx: ExprContext::Load });
                copy_location(ast, read, id);
                Visited::Node(read)
            }
            Kind::Subscript { value, slice, .. } if self.reads.contains(value) => {
                let slice = *slice;
                let Visited::Node(new) = self.visit(ast, slice) else { unreachable!() };
                let Kind::Subscript { slice, .. } = ast.kind_mut(id) else { unreachable!() };
                *slice = new;
                Visited::Node(id)
            }
            Kind::Assign { .. } | Kind::AnnAssign { .. } => {
                let node = self.declarations.get(&id).copied().unwrap_or(id);
                generic_visit(self, ast, node);
                Visited::Node(node)
            }
            _ => {
                generic_visit(self, ast, id);
                Visited::Node(id)
            }
        }
    }
}

/// The redeclaration of a captured per-bar scalar as a `Series`, `None` for a binding
/// that keeps a state slot of its own already.
fn declaration(ast: &mut Ast, binding: NodeId) -> Option<NodeId> {
    let (target, annotation, value) = match ast.kind(binding) {
        Kind::AnnAssign { target, annotation, value: Some(value), .. } => {
            let (target, annotation, value) = (*target, *annotation, *value);
            let wrapper = match ast.kind(annotation) {
                Kind::Subscript { value, .. } => *value,
                _ => annotation,
            };
            let wrapper = dotted_path(ast, wrapper);
            if STATE_WRAPPERS.contains(&last_segment(&wrapper)) {
                return None;
            }
            let annotation = if wrapper == "Series" {
                annotation
            } else {
                let series = ast.add(Kind::Name { id: "Series".into(), ctx: ExprContext::Load });
                ast.add(Kind::Subscript {
                    value: series,
                    slice: annotation,
                    ctx: ExprContext::Load,
                })
            };
            (target, annotation, value)
        }
        Kind::Assign { targets, value, .. } => {
            let (target, value) = (targets[0], *value);
            let series = ast.add(Kind::Name { id: "Series".into(), ctx: ExprContext::Load });
            (target, series, value)
        }
        _ => unreachable!(),
    };
    let declared = ast.add(Kind::AnnAssign { target, annotation, value: Some(value), simple: 1 });
    copy_location(ast, declared, binding);
    Some(declared)
}

pub fn run(ast: &mut Ast, module: NodeId, ctx: &Context<'_>) -> Result<NodeId, Error> {
    let Kind::Module { body, .. } = ast.kind(module) else {
        return Err(Error::Unsupported("ExportCapture on a non-module".into()));
    };
    let body = body.clone();
    let main_nodes: Vec<NodeId> = body
        .iter()
        .copied()
        .filter(|&statement| match ast.kind(statement) {
            Kind::FunctionDef { name, decorator_list, .. }
            | Kind::AsyncFunctionDef { name, decorator_list, .. } => {
                name == "main"
                    && decorator_list.iter().any(|&decorator| match ast.kind(decorator) {
                        Kind::Call { func, .. } => matches!(
                            dotted_path(ast, *func).as_str(),
                            "lib.script.library" | "script.library"
                        ),
                        _ => false,
                    })
            }
            _ => false,
        })
        .collect();
    if main_nodes.is_empty() {
        return Ok(module);
    }
    let file = match ast.extra(module, "_module_file_path") {
        None | Some(Value::None) => String::new(),
        Some(Value::Str(path)) => path.clone(),
        Some(_) => return Err(Error::Unsupported("a _module_file_path that is no str".into())),
    };
    let mut export_names: HashSet<String> = HashSet::from(["export".to_string()]);
    let mut namespace_names = HashSet::new();
    for &statement in &body {
        let Kind::ImportFrom { module: Some(from), names, .. } = ast.kind(statement) else {
            continue;
        };
        let wanted = match from.as_str() {
            "pynecore.core.pine_export" => "export",
            "pynecore.core.pine_import" => "shadowed_namespace",
            _ => continue,
        };
        for &alias in names {
            let Kind::Alias { name, asname } = ast.kind(alias) else { unreachable!() };
            if name == wanted {
                let bound = asname.clone().unwrap_or_else(|| name.clone());
                if wanted == "export" {
                    export_names.insert(bound);
                } else {
                    namespace_names.insert(bound);
                }
            }
        }
    }

    let mut root = Scope::new(module, None);
    root.file = file;
    let mut capture = Capture {
        ast,
        host: ctx.host,
        scopes: vec![root],
        scope_of: HashMap::new(),
        order: Vec::new(),
        mains: Vec::new(),
        namespace_names,
        modules: HashMap::new(),
        sources: BTreeMap::new(),
    };
    capture.register(module, 0);
    for &statement in &body {
        capture.collect(statement, 0);
    }
    capture.mains = main_nodes.iter().map(|main| capture.scope_of[main]).collect();
    let mut global_scopes: HashSet<ScopeId> = capture.mains.iter().copied().collect();
    global_scopes.insert(0);

    // Explicit outer writes invalidate an otherwise literal binding too
    for scope in capture.all_scopes() {
        let own = &capture.scopes[scope];
        let names: Vec<String> = own.globals.union(&own.nonlocals).cloned().collect();
        for name in names {
            let Some(owner) = capture.resolve(scope, &name) else { continue };
            if owner == scope {
                continue;
            }
            let written = capture.scopes[scope].bindings.get(&name).cloned().unwrap_or_default();
            capture.scopes[owner].bindings.get_mut(&name).expect("bound").extend(written);
        }
    }

    let default_reads: HashSet<NodeId> = capture
        .all_scopes()
        .into_iter()
        .flat_map(|scope| capture.scopes[scope].defaults.clone())
        .collect();
    let mut captures: Vec<(ScopeName, NodeId)> = Vec::new();
    let mut captured: HashMap<ScopeName, usize> = HashMap::new();
    for scope in capture.all_scopes() {
        let function = capture.scopes[scope].node;
        let (function_name, decorators) = match capture.ast.kind(function) {
            Kind::FunctionDef { name, decorator_list, .. }
            | Kind::AsyncFunctionDef { name, decorator_list, .. } => {
                (name.clone(), decorator_list.clone())
            }
            _ => continue,
        };
        let exported = decorators.iter().any(|&decorator| {
            let target = match capture.ast.kind(decorator) {
                Kind::Call { func, .. } => *func,
                _ => decorator,
            };
            let path = dotted_path(capture.ast, target);
            export_names.contains(&path) || path.ends_with(".export")
        });
        if !exported {
            continue;
        }
        let mut forbidden = global_scopes.clone();
        let mut ancestor = capture.scopes[scope].parent;
        while let Some(id) = ancestor {
            forbidden.insert(id);
            ancestor = capture.scopes[id].parent;
        }
        let mut pending = vec![scope];
        let mut seen: HashSet<ScopeId> = HashSet::new();
        while let Some(current) = pending.pop() {
            if !seen.insert(current) {
                continue;
            }
            let own = &capture.scopes[current];
            pending.extend(own.children.iter().copied());
            let mut reads: Vec<(NodeId, ScopeId)> =
                own.reads.iter().map(|&read| (read, current)).collect();
            if let Some(parent) = own.parent {
                reads.extend(own.defaults.iter().map(|&read| (read, parent)));
            }
            for (read, read_scope) in reads {
                let Kind::Name { id: name, .. } = capture.ast.kind(read) else { unreachable!() };
                let name = name.clone();
                let Some(owner) = capture.resolve(read_scope, &name) else { continue };
                let bindings = capture.scopes[owner].bindings[&name].clone();
                for binding in &bindings {
                    if let Some(&defined) = capture.scope_of.get(binding) {
                        pending.push(defined);
                    }
                }
                if !forbidden.contains(&owner)
                    || bindings.iter().all(|binding| capture.scope_of.contains_key(binding))
                {
                    continue;
                }
                if capture.constant_binding(owner, &name, &HashSet::new(), false)? {
                    continue;
                }
                if !default_reads.contains(&read)
                    && capture.constant_binding(owner, &name, &HashSet::new(), true)?
                {
                    let key = (owner, name);
                    match captured.get(&key) {
                        Some(&index) => captures[index].1 = bindings[0],
                        None => {
                            captured.insert(key.clone(), captures.len());
                            captures.push((key, bindings[0]));
                        }
                    }
                    continue;
                }
                return Err(Error::Rejected(format!(
                    "Exported function \"{function_name}\" cannot capture non-constant library \
                     global \"{name}\"."
                )));
            }
        }
    }

    // Export definitions survive across bars. A scalar held only in a Python closure
    // would freeze at its first-bar value, so read its library slot
    let mut declarations: HashMap<NodeId, NodeId> = HashMap::new();
    let mut retained: HashSet<ScopeName> = HashSet::new();
    for (key, binding) in captures {
        if let Some(declared) = declaration(capture.ast, binding) {
            declarations.insert(binding, declared);
            retained.insert(key);
        }
    }
    let mut reads: HashSet<NodeId> = HashSet::new();
    for scope in capture.all_scopes() {
        for &read in &capture.scopes[scope].reads {
            let Kind::Name { id: name, .. } = capture.ast.kind(read) else { unreachable!() };
            let owner = capture.resolve(scope, name);
            if let Some(owner) = owner
                && owner != scope
                && retained.contains(&(owner, name.clone()))
            {
                reads.insert(read);
            }
        }
    }
    let deps = Value::Tuple(
        capture
            .sources
            .iter()
            .map(|(path, &(mtime_ns, size))| {
                Value::Tuple(vec![
                    Value::Str(path.clone()),
                    Value::Int(Int::Small(mtime_ns)),
                    Value::Int(Int::Small(size)),
                ])
            })
            .collect(),
    );
    let ast = capture.ast;
    let Visited::Node(module) =
        RetainCaptures { reads: &reads, declarations: &declarations }.visit(ast, module)
    else {
        unreachable!()
    };
    ast.set_extra(module, "_export_capture_deps", deps);
    Ok(module)
}
