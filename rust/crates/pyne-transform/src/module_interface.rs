//! What one module publishes, derived natively (`module_interface`): the interface
//! a dependent's type pass reads, the routes digest and call routes its lowering
//! reads, and the digests both are compared by.
//!
//! The Python host builds these in Python and hands the native steps the result;
//! a host without Python ([`crate::resolver`]) builds them here, from a module it
//! analysed and lowered itself, so they have to come out exactly as the Python
//! functions make them -- the digests included, which are taken over the same
//! canonical JSON (`json.dumps(sort_keys=True, separators=(',', ':'))`) and over
//! `ast.dump` text of the interpreter the host names.

use std::collections::HashSet;

use indexmap::IndexMap;
use pyne_ast::walk::{child_statements, children};
use pyne_ast::{
    Ast, CLASSES, Category, Constant, ExprContext, FieldRef, Kind, NodeId, Unsupported,
};
use serde_json::{Value, json};

use crate::ast_dump::{dump, has_field};
use crate::pine_type_rules::{Classes, ImplSig, UNKNOWN, annotation_type, get_ty, impl_sig};
use crate::pine_type_table::{
    CallInterface, ClassSig, DepRecord, ExportSig, ModuleInterface, PineTypeTable, qualify,
};
use crate::sha256::sha256_hex;
use crate::slot_layout::ModuleLayout;

type R<T> = Result<T, Unsupported>;

/// Length of every digest (`_DIGEST_LEN`).
const DIGEST_LEN: usize = 16;

/// The digest of a dependency record that stands for a plain Python module
/// (`SOURCE_DIGEST`).
pub const SOURCE_DIGEST: &str = "source";

/// The reexport key of the star import a name the module does not bind after it
/// may come from (`STAR`).
pub const STAR: &str = "*";

const ROUTE_SKIP: &str = "skip";
const ROUTE_DIRECT: &str = "direct";
const ROUTE_FAST: &str = "fast";
const ROUTE_FAST_SHARED: &str = "fast-shared";
const ROUTE_UNIFORM: &str = "uniform";

/// Where the `udt` decorator lives (`_UDT`).
const UDT: (i64, &str, &str) = (0, "pynecore.core.pine_udt", "udt");

/// The marker of a function whose instance every call site in a loop shares
/// (`_SHARED_MARKER`).
const SHARED_MARKER: &str = "__pyne_shared_call_site__";

/// Everything one module publishes (`pine_type_table.ModuleInterface`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Interface {
    /// Resolved source path of the module
    pub path: String,
    pub exports: IndexMap<String, ExportSig>,
    /// The module's literal `__all__`, `None` when it spells none
    pub all: Option<Vec<String>>,
    pub classes: IndexMap<String, ClassSig>,
    pub extensions: IndexMap<String, IndexMap<String, ExportSig>>,
    pub digest: String,
    pub deps: IndexMap<String, DepRecord>,
    pub mtime_ns: i64,
    pub size: i64,
    pub suppressed: String,
    /// The routes digest, `""` until the lowering settled it
    pub routes: String,
    pub proxies: Vec<String>,
    pub call_routes: IndexMap<String, String>,
    pub reexports: IndexMap<String, (i64, String, String)>,
}

impl Interface {
    /// The state of this module as a dependent remembers it (`dep_record`).
    pub fn record(&self) -> DepRecord {
        DepRecord {
            path: self.path.clone(),
            mtime_ns: self.mtime_ns,
            size: self.size,
            digest: self.digest.clone(),
            routes: self.routes.clone(),
        }
    }

    /// What the type pass reads of it.
    pub fn types(&self) -> ModuleInterface {
        ModuleInterface {
            path: self.path.clone(),
            exports: self.exports.clone(),
            all: self.all.clone(),
            classes: self.classes.clone(),
            extensions: self.extensions.clone(),
            deps: self.deps.clone(),
            suppressed: self.suppressed.clone(),
            proxies: self.proxies.clone(),
            record: self.record(),
        }
    }

    /// What the call-site isolation reads of it.
    pub fn calls(&self) -> CallInterface {
        CallInterface {
            path: self.path.clone(),
            all: self.all.clone(),
            call_routes: self.call_routes.clone(),
            reexports: self.reexports.clone(),
            deps: self.deps.clone(),
            record: self.record(),
        }
    }
}

// --- the interface --------------------------------------------------------

/// Everything a module publishes, derived from its analysed tree (`build_interface`).
pub fn build_interface(
    ast: &Ast,
    tree: NodeId,
    table: &PineTypeTable,
    path: &str,
    fingerprint: (i64, i64),
) -> R<Interface> {
    let body = module_body(ast, tree);
    let proxies = exported_proxies(ast, body);
    let mut definitions = Vec::new();
    collect_defs(ast, tree, "", &mut definitions);
    let classes: Classes =
        table.classes.iter().map(|(name, id)| (name.clone(), id.clone())).collect();

    let mut exports: IndexMap<String, ExportSig> = IndexMap::new();
    let mut proxied: IndexMap<String, bool> = IndexMap::new();
    for (node, key, scope) in &definitions {
        let name = def_name(ast, *node);
        if scope.is_empty()
            && table.exportable.contains(name)
            && let Some(sig) = export_sig(ast, *node, key, table, &classes)?
        {
            exports.insert(name.to_string(), sig);
            proxied.insert(name.to_string(), proxies.contains(name) && is_exported(ast, *node));
        }
    }
    for (node, key, scope) in &definitions {
        let name = def_name(ast, *node);
        if !scope.is_empty()
            && proxies.contains(name)
            && is_exported(ast, *node)
            && let Some(sig) = export_sig(ast, *node, key, table, &classes)?
        {
            exports.insert(name.to_string(), sig);
            proxied.insert(name.to_string(), true);
        }
    }

    let all = module_all(ast, body);
    let mut bound: Vec<String> =
        proxied.into_iter().filter(|(_, bound)| *bound).map(|(name, _)| name).collect();
    bound.sort();
    let mut interface = Interface {
        path: path.to_string(),
        exports,
        classes: module_classes(ast, body, all.as_deref(), table),
        all,
        extensions: table.extensions.clone(),
        digest: String::new(),
        deps: table.deps.clone(),
        mtime_ns: fingerprint.0,
        size: fingerprint.1,
        suppressed: table
            .pins_suppressed
            .as_ref()
            .map(|diag| diag.message.clone())
            .unwrap_or_default(),
        routes: String::new(),
        proxies: bound,
        call_routes: IndexMap::new(),
        reexports: IndexMap::new(),
    };
    interface.digest = interface_digest(&interface);
    Ok(interface)
}

fn module_body(ast: &Ast, tree: NodeId) -> &[NodeId] {
    match ast.kind(tree) {
        Kind::Module { body, .. } => body,
        _ => &[],
    }
}

fn def_name(ast: &Ast, node: NodeId) -> &str {
    match ast.kind(node) {
        Kind::FunctionDef { name, .. } | Kind::AsyncFunctionDef { name, .. } => name,
        Kind::ClassDef { name, .. } => name,
        _ => "",
    }
}

fn decorators(ast: &Ast, node: NodeId) -> &[NodeId] {
    match ast.kind(node) {
        Kind::FunctionDef { decorator_list, .. }
        | Kind::AsyncFunctionDef { decorator_list, .. }
        | Kind::ClassDef { decorator_list, .. } => decorator_list,
        _ => &[],
    }
}

fn is_def(ast: &Ast, node: NodeId) -> bool {
    matches!(ast.kind(node), Kind::FunctionDef { .. } | Kind::AsyncFunctionDef { .. })
}

/// The published signature of one definition (`_export_sig`).
fn export_sig(
    ast: &Ast,
    node: NodeId,
    key: &str,
    table: &PineTypeTable,
    classes: &Classes,
) -> R<Option<ExportSig>> {
    let Some(func) = table.funcs.get(key) else { return Ok(None) };
    let line = ast[node].loc.lineno.unwrap_or(0);
    let name = def_name(ast, node).to_string();
    if let Some(impls) = table.groups.get(key).filter(|impls| !impls.is_empty()) {
        let first = &impls[0];
        return Ok(Some(ExportSig {
            name,
            kind: "group".into(),
            params: first.params.clone(),
            required: first.required as i64,
            open_ended: first.open_ended,
            ret: func.ret.clone(),
            annotated: impls.iter().all(|imp| !imp.params.iter().any(|param| param == UNKNOWN)),
            impls: impls.clone(),
            line,
            names: first.names.clone(),
        }));
    }
    let sig = impl_sig(ast, node, &func.ret, &|value| get_ty(ast, value).to_string(), classes)?;
    let args = match ast.kind(node) {
        Kind::FunctionDef { args, .. } | Kind::AsyncFunctionDef { args, .. } => *args,
        _ => return Ok(None),
    };
    let Kind::Arguments { posonlyargs, args, .. } = ast.kind(args) else { return Ok(None) };
    let mut annotated = true;
    for &arg in posonlyargs.iter().chain(args) {
        let annotation = match ast.kind(arg) {
            Kind::Arg { annotation, .. } => *annotation,
            _ => None,
        };
        if annotation_type(ast, annotation, classes)? == UNKNOWN {
            annotated = false;
            break;
        }
    }
    Ok(Some(ExportSig {
        name,
        kind: "function".into(),
        params: sig.params,
        required: sig.required as i64,
        open_ended: sig.open_ended,
        ret: sig.ret,
        annotated,
        impls: Vec::new(),
        line,
        names: sig.names,
    }))
}

/// Every definition of a tree, with the scope-qualified id the table keys it by
/// (`_collect_defs`).
fn collect_defs(ast: &Ast, node: NodeId, scope: &str, out: &mut Vec<(NodeId, String, String)>) {
    for child in child_statements(ast, node) {
        match ast.kind(child) {
            Kind::FunctionDef { name, .. } | Kind::AsyncFunctionDef { name, .. } => {
                let key = qualify(scope, name);
                out.push((child, key.clone(), scope.to_string()));
                collect_defs(ast, child, &key, out);
            }
            Kind::ClassDef { name, .. } => collect_defs(ast, child, &qualify(scope, name), out),
            _ => collect_defs(ast, child, scope, out),
        }
    }
}

/// The value and targets of a module-level assignment, plain or annotated.
fn assignment(ast: &Ast, stmt: NodeId) -> Option<(Vec<NodeId>, Option<NodeId>)> {
    match ast.kind(stmt) {
        Kind::Assign { targets, value, .. } => Some((targets.clone(), Some(*value))),
        Kind::AnnAssign { target, value, .. } => Some((vec![*target], *value)),
        _ => None,
    }
}

/// The names a compiled library binds an `Exported()` proxy to (`_exported_proxies`).
fn exported_proxies(ast: &Ast, body: &[NodeId]) -> HashSet<String> {
    let mut names = HashSet::new();
    for &stmt in body {
        let Some((targets, Some(value))) = assignment(ast, stmt) else { continue };
        let Kind::Call { func, .. } = ast.kind(value) else { continue };
        let called = match ast.kind(*func) {
            Kind::Name { id, .. } => id.as_str(),
            Kind::Attribute { attr, .. } => attr.as_str(),
            _ => "",
        };
        if called != "Exported" {
            continue;
        }
        for target in targets {
            if let Kind::Name { id, .. } = ast.kind(target) {
                names.insert(id.clone());
            }
        }
    }
    names
}

/// Whether a definition carries the `@export` decorator (`_is_exported`).
fn is_exported(ast: &Ast, node: NodeId) -> bool {
    decorators(ast, node).iter().any(|&decorator| {
        let target = match ast.kind(decorator) {
            Kind::Call { func, .. } => *func,
            _ => decorator,
        };
        match ast.kind(target) {
            Kind::Name { id, .. } => id == "export",
            Kind::Attribute { attr, .. } => attr == "export",
            _ => false,
        }
    })
}

/// The module's literal `__all__`, when it spells one (`_module_all`).
fn module_all(ast: &Ast, body: &[NodeId]) -> Option<Vec<String>> {
    let mut found = None;
    for &stmt in body {
        let Some((targets, value)) = assignment(ast, stmt) else { continue };
        if !targets
            .iter()
            .any(|&target| matches!(ast.kind(target), Kind::Name { id, .. } if id == "__all__"))
        {
            continue;
        }
        let Some(value) = value else { continue };
        let (Kind::List { elts, .. } | Kind::Tuple { elts, .. }) = ast.kind(value) else {
            continue;
        };
        found = Some(
            elts.iter()
                .filter_map(|&element| match ast.kind(element) {
                    Kind::Constant { value: Constant::Str(text), .. } => Some(text.clone()),
                    _ => None,
                })
                .collect(),
        );
    }
    found
}

/// The classes a module publishes, in source order (`_module_classes`).
fn module_classes(
    ast: &Ast,
    body: &[NodeId],
    all: Option<&[String]>,
    table: &PineTypeTable,
) -> IndexMap<String, ClassSig> {
    let mut out = IndexMap::new();
    for &stmt in body {
        let Kind::ClassDef { name, .. } = ast.kind(stmt) else { continue };
        if all.is_some_and(|published| !published.contains(name)) {
            continue;
        }
        let id = table.classes.get(name).map(String::as_str).unwrap_or("");
        if let Some(sig) = table.class_sigs.get(id) {
            out.insert(name.clone(), sig.clone());
        }
    }
    out
}

/// Digest of what a module publishes, blind to how it publishes it (`interface_digest`).
pub fn interface_digest(interface: &Interface) -> String {
    let exports_of = |sigs: &IndexMap<String, ExportSig>| -> Value {
        Value::Object(
            sigs.iter().map(|(name, sig)| (name.clone(), unlined(export_json(sig)))).collect(),
        )
    };
    let classes: serde_json::Map<String, Value> = interface
        .classes
        .iter()
        .map(|(name, sig)| {
            let fields: Vec<Value> =
                sig.fields.iter().map(|(field, ty)| json!([field, ty])).collect();
            (
                name.clone(),
                json!({
                    "fields": fields,
                    "required": sig.required,
                    "methods": exports_of(&sig.methods),
                    "members": exports_of(&sig.members),
                }),
            )
        })
        .collect();
    let extensions: serde_json::Map<String, Value> = interface
        .extensions
        .iter()
        .map(|(cid, methods)| (cid.clone(), exports_of(methods)))
        .collect();
    let payload = json!({
        "all": interface.all,
        "classes": classes,
        "extensions": extensions,
        "exports": exports_of(&interface.exports),
        "suppressed": interface.suppressed,
        "proxies": interface.proxies,
    });
    digest(canonical(&payload).as_bytes())
}

/// One published signature, in the payload's shape (`_export_json`).
fn export_json(sig: &ExportSig) -> Value {
    let impls: Vec<Value> = sig.impls.iter().map(impl_json).collect();
    json!({
        "kind": sig.kind,
        "params": sig.params,
        "required": sig.required,
        "open_ended": sig.open_ended,
        "ret": sig.ret,
        "annotated": sig.annotated,
        "impls": impls,
        "line": sig.line,
        "names": sig.names,
    })
}

fn impl_json(imp: &ImplSig) -> Value {
    json!({
        "params": imp.params,
        "required": imp.required,
        "open_ended": imp.open_ended,
        "ret": imp.ret,
        "fits": imp.fits,
        "names": imp.names,
    })
}

/// One signature with its line number taken out (`_unlined`).
fn unlined(mut payload: Value) -> Value {
    if let Value::Object(fields) = &mut payload {
        fields.shift_remove("line");
    }
    payload
}

/// The one hash every digest here is built with (`_digest`).
pub fn digest(data: &[u8]) -> String {
    let mut hex = sha256_hex(data);
    hex.truncate(DIGEST_LEN);
    hex
}

/// The one JSON spelling a digest may be taken over (`_canonical`): sorted keys, no
/// whitespace, every character outside printable ASCII escaped as `json.dumps` does.
pub fn canonical(value: &Value) -> String {
    let mut out = String::new();
    write_json(value, &mut out);
    out
}

fn write_json(value: &Value, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(number) => out.push_str(&number.to_string()),
        Value::String(text) => write_json_str(text, out),
        Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                write_json(item, out);
            }
            out.push(']');
        }
        Value::Object(fields) => {
            let mut keys: Vec<&String> = fields.keys().collect();
            keys.sort();
            out.push('{');
            for (index, key) in keys.into_iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                write_json_str(key, out);
                out.push(':');
                write_json(&fields[key], out);
            }
            out.push('}');
        }
    }
}

/// `json.dumps` of a string with `ensure_ascii`.
fn write_json_str(text: &str, out: &mut String) {
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            ' '..='~' => out.push(c),
            c => {
                let mut units = [0u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
        }
    }
    out.push('"');
}

// --- the routes -----------------------------------------------------------

/// The module-level structure a call into the module is routed by, bodies left out
/// (`routes_shape`), with the `ast.dump` of the interpreter `python` names.
pub fn routes_shape(ast: &Ast, tree: NodeId, python: (u32, u32)) -> R<String> {
    let mut shaped = Vec::new();
    for &stmt in module_body(ast, tree) {
        if !is_docstring(ast, stmt) {
            shaped.push(shape(ast, stmt, python)?);
        }
    }
    Ok(canonical(&Value::Array(shaped)))
}

/// Whether a node holds statement lists of its own (`_STATEMENT_LIKE`).
fn statement_like(ast: &Ast, node: NodeId) -> bool {
    let class = &CLASSES[ast.kind(node).class_index()];
    matches!(class.category, Category::Stmt | Category::Excepthandler) || class.name == "MatchCase"
}

/// One statement of the routes shape (`_shape`).
fn shape(ast: &Ast, node: NodeId, python: (u32, u32)) -> R<Value> {
    if is_def(ast, node) {
        let mut dumped = Vec::new();
        for &decorator in decorators(ast, node) {
            dumped.push(Value::String(dump(ast, decorator, false, python)?));
        }
        return Ok(json!(["def", def_name(ast, node), dumped]));
    }
    let kind = ast.kind(node);
    let class = &CLASSES[kind.class_index()];
    let mut shaped = vec![Value::String(class.name.to_string())];
    let item_of = |item: NodeId| -> R<Value> {
        if statement_like(ast, item) {
            shape(ast, item, python)
        } else {
            Ok(Value::String(dump(ast, item, false, python)?))
        }
    };
    for (index, field) in class.fields.iter().enumerate() {
        if !has_field(field.name, python) {
            continue;
        }
        let value = match kind.field(index) {
            FieldRef::Nodes(items) => {
                let mut out = Vec::new();
                for &item in items {
                    if !is_docstring(ast, item) {
                        out.push(item_of(item)?);
                    }
                }
                Value::Array(out)
            }
            FieldRef::OptNodes(items) => {
                let mut out = Vec::new();
                for item in items {
                    out.push(match item {
                        Some(item) if !is_docstring(ast, *item) => item_of(*item)?,
                        Some(_) => continue,
                        None => Value::Null,
                    });
                }
                Value::Array(out)
            }
            FieldRef::Strs(items) => json!(items),
            FieldRef::Node(child) => Value::String(dump(ast, *child, false, python)?),
            FieldRef::OptNode(child) => match child {
                Some(child) => Value::String(dump(ast, *child, false, python)?),
                None => Value::Null,
            },
            FieldRef::Str(text) => json!(text),
            FieldRef::OptStr(text) => json!(text),
            FieldRef::Int(value) => json!(value),
            FieldRef::OptInt(value) => json!(value),
            FieldRef::Operator(op) => Value::String(format!("{}()", op.name())),
            FieldRef::Constant(_)
            | FieldRef::ExprContext(_)
            | FieldRef::UnaryOperator(_)
            | FieldRef::BoolOperator(_)
            | FieldRef::CmpOperators(_) => {
                return Err(Unsupported(format!("a {} field of a statement", field.name)));
            }
        };
        shaped.push(json!([field.name, value]));
    }
    Ok(Value::Array(shaped))
}

/// Whether a statement is a bare string literal (`_is_docstring`).
fn is_docstring(ast: &Ast, node: NodeId) -> bool {
    let Kind::Expr { value } = ast.kind(node) else { return false };
    matches!(ast.kind(*value), Kind::Constant { value: Constant::Str(_), .. })
}

/// Digest of how every call into a module is routed, once its lowering settled it
/// (`routes_digest`).
pub fn routes_digest(shape: &str, ast: &Ast, tree: NodeId, layout: &ModuleLayout) -> String {
    let mut defs = Vec::new();
    reachable_defs(ast, tree, &mut defs);
    let carriers: Vec<Value> = defs
        .into_iter()
        .map(|node| {
            let carries = matches!(ast.kind(node), Kind::FunctionDef { .. })
                && layout.state_carrying(&layout.scope_segment(ast, node));
            json!([def_name(ast, node), carries])
        })
        .collect();
    digest(canonical(&json!([shape, carriers])).as_bytes())
}

/// The definitions of a module that do not stand inside another definition
/// (`_reachable_defs`).
fn reachable_defs(ast: &Ast, node: NodeId, out: &mut Vec<NodeId>) {
    for child in child_statements(ast, node) {
        if is_def(ast, child) {
            out.push(child);
        } else {
            reachable_defs(ast, child, out);
        }
    }
}

// --- the call routes ------------------------------------------------------

/// One statement binding a name in a module or class body (`_Binding`).
#[derive(Clone)]
struct Binding {
    index: usize,
    /// The definition or class it binds
    node: Option<NodeId>,
    conditional: bool,
    /// For an import, where it leads
    target: Option<(i64, String, String)>,
}

/// One store (or `del`) of an attribute on a name of the body (`_AttributeStore`).
struct AttributeStore {
    index: usize,
    attr: String,
    /// The value of a plain `name.attr = value`
    value: Option<NodeId>,
    conditional: bool,
}

/// What the statements of one body bind, in order (`_Bindings`).
struct Bindings {
    names: IndexMap<String, Vec<Binding>>,
    attributes: IndexMap<String, Vec<AttributeStore>>,
    /// (index, level, module, conditional) of every star import
    stars: Vec<(usize, i64, String, bool)>,
}

impl Bindings {
    fn new(ast: &Ast, body: &[NodeId]) -> Bindings {
        let mut found =
            Bindings { names: IndexMap::new(), attributes: IndexMap::new(), stars: Vec::new() };
        for (index, &stmt) in body.iter().enumerate() {
            found.statement(ast, stmt, index);
        }
        found
    }

    fn bind(&mut self, name: &str, binding: Binding) {
        self.names.entry(name.to_string()).or_default().push(binding);
    }

    fn store(&mut self, name: &str, store: AttributeStore) {
        self.attributes.entry(name.to_string()).or_default().push(store);
    }

    fn statement(&mut self, ast: &Ast, stmt: NodeId, index: usize) {
        match ast.kind(stmt) {
            Kind::FunctionDef { name, .. }
            | Kind::AsyncFunctionDef { name, .. }
            | Kind::ClassDef { name, .. } => {
                self.bind(
                    name,
                    Binding { index, node: Some(stmt), conditional: false, target: None },
                );
                self.header(ast, stmt, index);
            }
            Kind::Import { .. } | Kind::ImportFrom { .. } => self.import(ast, stmt, index, false),
            Kind::Assign { targets, value, .. } => {
                let single = targets.len() == 1;
                for &target in targets {
                    self.target(ast, target, index, if single { Some(*value) } else { None });
                }
                self.scan(ast, *value, index);
            }
            Kind::AnnAssign { target, annotation, value, .. } => {
                if let Some(value) = value {
                    self.target(ast, *target, index, Some(*value));
                    self.scan(ast, *value, index);
                }
                self.scan(ast, *annotation, index);
            }
            Kind::AugAssign { target, value, .. } => {
                self.target(ast, *target, index, None);
                self.scan(ast, *value, index);
            }
            Kind::Delete { targets } => {
                for &target in targets {
                    self.target(ast, target, index, None);
                }
            }
            _ => self.scan(ast, stmt, index),
        }
    }

    fn target(&mut self, ast: &Ast, target: NodeId, index: usize, value: Option<NodeId>) {
        match ast.kind(target) {
            Kind::Name { id, .. } => {
                self.bind(id, Binding { index, node: None, conditional: false, target: None });
            }
            Kind::Tuple { elts, .. } | Kind::List { elts, .. } => {
                for &element in elts {
                    self.target(ast, element, index, None);
                }
            }
            Kind::Starred { value: inner, .. } => self.target(ast, *inner, index, None),
            Kind::Attribute { value: base, attr, .. }
                if matches!(ast.kind(*base), Kind::Name { .. }) =>
            {
                let Kind::Name { id, .. } = ast.kind(*base) else { unreachable!() };
                let store = AttributeStore { index, attr: attr.clone(), value, conditional: false };
                self.store(id, store);
            }
            _ => self.scan(ast, target, index),
        }
    }

    fn import(&mut self, ast: &Ast, stmt: NodeId, index: usize, conditional: bool) {
        match ast.kind(stmt) {
            Kind::Import { names } => {
                for &alias in names {
                    let Kind::Alias { name, asname } = ast.kind(alias) else { continue };
                    // ``import a.b`` binds ``a``; ``import a.b as x`` binds ``a.b`` itself
                    let module = if asname.is_some() {
                        name.clone()
                    } else {
                        name.split('.').next().unwrap_or("").to_string()
                    };
                    let bound = asname.clone().unwrap_or_else(|| module.clone());
                    let target = Some((0, module, String::new()));
                    self.bind(&bound, Binding { index, node: Some(stmt), conditional, target });
                }
            }
            Kind::ImportFrom { module, names, level } => {
                let module = module.clone().unwrap_or_default();
                let level = level.unwrap_or(0);
                for &alias in names {
                    let Kind::Alias { name, asname } = ast.kind(alias) else { continue };
                    if name == "*" {
                        self.stars.push((index, level, module.clone(), conditional));
                    } else {
                        let bound = asname.clone().unwrap_or_else(|| name.clone());
                        let target = Some((level, module.clone(), name.clone()));
                        self.bind(&bound, Binding { index, node: Some(stmt), conditional, target });
                    }
                }
            }
            _ => {}
        }
    }

    /// What a definition evaluates in the body's scope (`_header`).
    fn header(&mut self, ast: &Ast, node: NodeId, index: usize) {
        let mut parts: Vec<NodeId> = decorators(ast, node).to_vec();
        match ast.kind(node) {
            Kind::ClassDef { bases, keywords, .. } => {
                parts.extend(bases);
                for &keyword in keywords {
                    if let Kind::Keyword { value, .. } = ast.kind(keyword) {
                        parts.push(*value);
                    }
                }
            }
            Kind::FunctionDef { args, returns, .. }
            | Kind::AsyncFunctionDef { args, returns, .. } => {
                parts.push(*args);
                parts.extend(returns);
            }
            _ => {}
        }
        for part in parts {
            self.scan(ast, part, index);
        }
    }

    /// Every binding below a node that lands in the body's scope, as conditional
    /// (`_scan`).
    fn scan(&mut self, ast: &Ast, node: NodeId, index: usize) {
        match ast.kind(node) {
            Kind::FunctionDef { name, .. }
            | Kind::AsyncFunctionDef { name, .. }
            | Kind::ClassDef { name, .. } => {
                self.bind(
                    name,
                    Binding { index, node: Some(node), conditional: true, target: None },
                );
                self.header(ast, node, index);
                return;
            }
            Kind::Import { .. } | Kind::ImportFrom { .. } => {
                self.import(ast, node, index, true);
                return;
            }
            Kind::Lambda { args, .. } => {
                self.scan(ast, *args, index);
                return;
            }
            Kind::Comprehension { iter, ifs, .. } => {
                self.scan(ast, *iter, index);
                for &test in ifs {
                    self.scan(ast, test, index);
                }
                return;
            }
            Kind::Name { id, ctx } => {
                if *ctx != ExprContext::Load {
                    self.bind(id, Binding { index, node: None, conditional: true, target: None });
                }
                return;
            }
            Kind::Attribute { value, attr, ctx } if *ctx != ExprContext::Load => {
                if let Kind::Name { id, .. } = ast.kind(*value) {
                    let store = AttributeStore {
                        index,
                        attr: attr.clone(),
                        value: None,
                        conditional: true,
                    };
                    self.store(id, store);
                    return;
                }
            }
            Kind::ExceptHandler { name: Some(name), .. }
            | Kind::MatchAs { name: Some(name), .. }
            | Kind::MatchStar { name: Some(name) } => {
                self.bind(name, Binding { index, node: None, conditional: true, target: None });
            }
            Kind::MatchMapping { rest: Some(rest), .. } => {
                self.bind(rest, Binding { index, node: None, conditional: true, target: None });
            }
            _ => {}
        }
        for child in children(ast, node) {
            self.scan(ast, child, index);
        }
    }
}

/// How a call through each module-level name of a module is routed, as far as its
/// analysed tree says (`CallPlan`).
#[derive(Default)]
pub struct CallPlan {
    /// Dotted path -> the route the tree alone proves
    pub fixed: IndexMap<String, String>,
    /// Dotted path -> the plain function a call reaches, and whether it is marked
    /// as sharing its call site
    pub defs: IndexMap<String, (NodeId, bool)>,
    pub reexports: IndexMap<String, (i64, String, String)>,
}

/// How a call through each module-level name is routed, from the analysed tree
/// (`call_plan`).
pub fn call_plan(ast: &Ast, tree: NodeId) -> CallPlan {
    let found = Bindings::new(ast, module_body(ast, tree));
    let mut plan = CallPlan::default();
    let last_star = found.stars.iter().map(|star| star.0 as i64).max().unwrap_or(-1);
    let star = match found.stars.as_slice() {
        [only] if !only.3 => Some(only),
        _ => None,
    };
    if let Some(star) = star {
        plan.reexports.insert(STAR.into(), (star.1, star.2.clone(), STAR.into()));
    }
    for (name, bindings) in &found.names {
        let last = bindings.last().expect("a bound name has a binding");
        if (last.index as i64) < last_star {
            continue;
        }
        let conditional = bindings.iter().any(|binding| binding.conditional);
        if !conditional {
            if let Some(target) = &last.target {
                plan.reexports.insert(name.clone(), target.clone());
                continue;
            }
            match last.node.map(|node| (node, ast.kind(node))) {
                Some((node, Kind::FunctionDef { decorator_list, .. }))
                | Some((node, Kind::AsyncFunctionDef { decorator_list, .. })) => {
                    let stores = found.attributes.get(name).map(Vec::as_slice).unwrap_or(&[]);
                    if let (true, Some(shared)) =
                        (decorator_list.is_empty(), shared(ast, stores, last.index))
                    {
                        plan.defs.insert(name.clone(), (node, shared));
                        continue;
                    }
                }
                Some((node, Kind::ClassDef { decorator_list, .. })) => {
                    let udt = match decorator_list.as_slice() {
                        [only] => match ast.kind(*only) {
                            Kind::Name { id, .. } => bound_to(&found, id, UDT, last.index),
                            _ => false,
                        },
                        _ => false,
                    };
                    if decorator_list.is_empty() || udt {
                        plan_class(ast, &mut plan, &found, name, node, last.index, udt);
                        continue;
                    }
                }
                _ => {}
            }
        }
        if star.is_some() {
            plan.fixed.insert(name.clone(), ROUTE_UNIFORM.into());
        }
    }
    plan
}

/// Whether a function is marked as sharing its call site (`_shared`).
fn shared(ast: &Ast, stores: &[AttributeStore], index: usize) -> Option<bool> {
    let mut shared = false;
    for store in stores {
        if store.index < index {
            continue;
        }
        let constant = store.value.and_then(|value| match ast.kind(value) {
            Kind::Constant { value, .. } => Some(value),
            _ => None,
        });
        let constant = constant.filter(|_| !store.conditional && store.attr == SHARED_MARKER)?;
        shared = truthy(constant)?;
    }
    Some(shared)
}

/// `bool(value)` of a constant; `None` where Python's answer is not modelled.
fn truthy(value: &Constant) -> Option<bool> {
    Some(match value {
        Constant::None => false,
        Constant::Bool(value) => *value,
        Constant::Int(pyne_ast::Int::Small(value)) => *value != 0,
        Constant::Int(pyne_ast::Int::Big(_)) => true,
        Constant::Float(value) => *value != 0.0,
        Constant::Complex { real, imag } => *real != 0.0 || *imag != 0.0,
        Constant::Str(text) => !text.is_empty(),
        Constant::Bytes(data) => !data.is_empty(),
        Constant::Ellipsis => true,
        Constant::Tuple(items) => !items.is_empty(),
        Constant::FrozenSet(items) => !items.is_empty(),
    })
}

/// Whether a name is an import of one particular object at a position of the body
/// (`_bound_to`).
fn bound_to(found: &Bindings, name: &str, target: (i64, &str, &str), index: usize) -> bool {
    let Some(bindings) = found.names.get(name) else { return false };
    let [binding] = bindings.as_slice() else { return false };
    !binding.conditional
        && binding.index < index
        && binding
            .target
            .as_ref()
            .is_some_and(|(level, module, attr)| (*level, module.as_str(), attr.as_str()) == target)
        && !found.stars.iter().any(|star| star.0 < index)
}

/// The routes through a class: the class itself, and the methods reached through it
/// (`_plan_class`).
fn plan_class(
    ast: &Ast,
    plan: &mut CallPlan,
    found: &Bindings,
    name: &str,
    node: NodeId,
    index: usize,
    udt: bool,
) {
    plan.fixed.insert(name.to_string(), ROUTE_SKIP.into());
    let mut replaced: Vec<String> = found
        .attributes
        .get(name)
        .map(|stores| {
            stores
                .iter()
                .filter(|store| store.index > index)
                .map(|store| store.attr.clone())
                .collect()
        })
        .unwrap_or_default();
    if udt {
        if !replaced.iter().any(|attr| attr == "new") {
            plan.fixed.insert(format!("{name}.new"), ROUTE_SKIP.into());
        }
        replaced.push("new".into());
        replaced.push("copy".into());
    }
    let Kind::ClassDef { body, .. } = ast.kind(node) else { return };
    let members = Bindings::new(ast, body);
    let builtins_free = found.stars.is_empty()
        && !["staticmethod", "classmethod"].iter().any(|decorator| {
            found.names.contains_key(*decorator) || members.names.contains_key(*decorator)
        });
    for (attr, bindings) in &members.names {
        let method = bindings.last().and_then(|binding| binding.node);
        if replaced.contains(attr)
            || members.attributes.contains_key(attr)
            || bindings.iter().any(|binding| binding.conditional)
        {
            continue;
        }
        let Some(method) = method.filter(|&method| is_def(ast, method)) else { continue };
        let names: Vec<Option<&str>> = decorators(ast, method)
            .iter()
            .map(|&decorator| match ast.kind(decorator) {
                Kind::Name { id, .. } => Some(id.as_str()),
                _ => None,
            })
            .collect();
        if !names.is_empty() && !builtins_free {
            continue;
        }
        if names == [Some("classmethod")] {
            plan.fixed.insert(format!("{name}.{attr}"), ROUTE_SKIP.into());
        } else if names.is_empty() || names == [Some("staticmethod")] {
            plan.defs.insert(format!("{name}.{attr}"), (method, false));
        }
    }
}

/// The call routes of a module, once its lowering settled which definitions keep
/// state (`call_routes`).
pub fn call_routes(plan: &CallPlan, ast: &Ast, layout: &ModuleLayout) -> IndexMap<String, String> {
    let mut routes = plan.fixed.clone();
    for (path, &(node, shared)) in &plan.defs {
        let carries = matches!(ast.kind(node), Kind::FunctionDef { .. })
            && layout.state_carrying(&layout.scope_segment(ast, node));
        let route = match (carries, shared) {
            (true, true) => ROUTE_FAST_SHARED,
            (true, false) => ROUTE_FAST,
            (false, _) => ROUTE_DIRECT,
        };
        routes.insert(path.clone(), route.into());
    }
    routes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_json_is_pythons() {
        let value = json!({"b": [1, true, null], "a": "x\"\\\n\u{7f}\u{e9}\u{1f600}"});
        let expected = format!(
            "{{\"a\":\"x\\\"\\\\\\n{b}u007f{b}u00e9{b}ud83d{b}ude00\",\"b\":[1,true,null]}}",
            b = '\\'
        );
        assert_eq!(canonical(&value), expected);
    }
}
