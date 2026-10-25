//! Port of `transformers/function_isolation.py`: every call site is routed to the
//! slot-based instance-state scheme at transform time.
//!
//! - **fast** (a provably state-carrying callee): the child instance lives in a slot
//!   of the caller's state vector, handed over as a hidden first argument;
//! - **direct** (a provably stateless callee): the call stays as it is;
//! - **uniform** (anything not provable): the caller anchors a `(callee, bound)`
//!   pair in a slot of its own and calls through it.
//!
//! Loop-shaped sites share one instance across iterations through the `__loop_state`
//! / `__bind_loop` helpers, and a site under a comprehension iterable folds the
//! guard into a helper call (Python forbids a walrus there). Same-module callees are
//! classified by the slot layout and a carrier fixpoint over the call graph;
//! cross-module ones by the generated lib routes, the interface a Pyne module
//! publishes, or (through the host) the object a module without one binds. The
//! routes of the imported callees extend the module's dependency records (the
//! native `PineTypeTable`'s `deps`).

use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::Path;
use std::sync::OnceLock;

use indexmap::IndexMap;
use pyne_ast::walk::{Transformer, Visited, children, copy_location, generic_visit, walk};
use pyne_ast::{
    Ast, BoolOperator, CmpOperator, Constant, ExprContext, FieldRef, Int, Kind, Loc, NodeId, Value,
};

use crate::data;
use crate::pine_qualifier::{SERIES_LEN_ATTR, SERIES_LENGTH_KEYWORD, SERIES_LENS_ATTR};
use crate::pine_type_rules::{
    PIN_ATTR, PINS_ATTR, VARYING_ATTR, VECTOR_ATTR, WIDENS_ATTR, stamp_lowering, ty_stamp,
};
use crate::pine_type_table::{CallInterface, PineTypeTable};
use crate::slot_layout::{DEFAULT_STATE_PARAM, ModuleLayout, SlotKind, scope_for_function};
use crate::{Context, Error, Host};

/// The runtime helpers by name, with the collision-safe alias a module imports them
/// under (`HELPER_ALIASES`).
pub const HELPER_ALIASES: &[(&str, &str)] = &[
    ("__resolve_slot__", "__resolve_slot·__"),
    ("__bind_any__", "__bind_any·__"),
    ("__slot_state__", "__slot_state·__"),
    ("__bind_slot__", "__bind_slot·__"),
    ("__loop_state__", "__loop_state·__"),
    ("__bind_loop__", "__bind_loop·__"),
    ("__bind_pinned__", "__bind_pinned·__"),
];

/// The callees that keep no per-call-site state, called plainly
/// (`NON_TRANSFORMABLE_FUNCTIONS`; `test_137` holds the two lists equal).
pub const NON_TRANSFORMABLE_FUNCTIONS: &[&str] = &include!("function_isolation_plain.in");

fn is_plain_callee(path: &str) -> bool {
    static PLAIN: OnceLock<HashSet<&'static str>> = OnceLock::new();
    PLAIN.get_or_init(|| NON_TRANSFORMABLE_FUNCTIONS.iter().copied().collect()).contains(path)
}

/// The prefix of the call-inlining pass's support aliases (`call_inline.SUPPORT_ALIAS_PREFIX`)
const SUPPORT_ALIAS_PREFIX: &str = "__inl·";

/// The package whose callees are routed from the generated `routes` table
const LIB_PACKAGE: &str = "pynecore.lib";

/// The reexport key of a star import (`module_interface.STAR`)
const STAR: &str = "*";

/// The module stamp the type pass leaves its table under (`TABLE_ATTR`)
const TABLE_ATTR: &str = "_pine_types";

/// A call-site route decided at transform time.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Route {
    Skip,
    Direct,
    Fast,
    /// A cross-module state-carrying callee marked `__pyne_shared_call_site__`: the
    /// fast route with the straight-line shape even inside a loop
    FastShared,
    Uniform,
    /// A same-module definition, by scope id: fast or direct once the carrier
    /// fixpoint settled it
    Same(String),
}

impl Route {
    /// A route word as the routes table, an interface or the host spell it.
    fn parse(word: &str) -> Result<Route, Error> {
        Ok(match word {
            "skip" => Route::Skip,
            "direct" => Route::Direct,
            "fast" => Route::Fast,
            "fast-shared" => Route::FastShared,
            "uniform" => Route::Uniform,
            other => return Err(Error::Unsupported(format!("a call route {other:?}"))),
        })
    }

    /// The routes decided by what the callee object carries (`_OBJECT_ROUTES`).
    fn is_object_route(&self) -> bool {
        matches!(self, Route::Direct | Route::Fast | Route::FastShared)
    }
}

fn host_error(err: pyne_ast::Unsupported) -> Error {
    Error::Unsupported(err.0)
}

fn is_test_function(name: &str) -> bool {
    name.starts_with("__test_") && name.ends_with("__")
}

/// The full dotted path of a callee expression (`_get_func_path`).
fn func_path(ast: &Ast, func: NodeId) -> Option<String> {
    match ast.kind(func) {
        Kind::Name { id, .. } => Some(id.clone()),
        Kind::Attribute { .. } => {
            let mut parts = Vec::new();
            let mut current = func;
            while let Kind::Attribute { value, attr, .. } = ast.kind(current) {
                parts.push(attr.as_str());
                current = *value;
            }
            let Kind::Name { id, .. } = ast.kind(current) else { return None };
            parts.push(id);
            parts.reverse();
            Some(parts.join("."))
        }
        _ => None,
    }
}

/// `str.isidentifier()` of an ASCII name.
fn is_ascii_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    chars.next().is_some_and(|first| first.is_ascii_alphabetic() || first == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// What reparsing a plain `a.b.c` callee and locating every node at its span
/// yields, built directly; `None` when the parser would not read it back unchanged
/// (`_copy_name_chain`).
fn copy_name_chain(ast: &mut Ast, func: NodeId, loc: Loc) -> Option<NodeId> {
    let (name, kind) = match ast.kind(func).clone() {
        Kind::Name { id, .. } => (id.clone(), Kind::Name { id, ctx: ExprContext::Load }),
        Kind::Attribute { value, attr, .. } => {
            let value = copy_name_chain(ast, value, loc)?;
            (attr.clone(), Kind::Attribute { value, attr, ctx: ExprContext::Load })
        }
        _ => return None,
    };
    // The parser normalizes a non-ASCII identifier and reads a keyword back as
    // something else; those take the reparse
    if !is_ascii_identifier(&name) || data::python_names().keywords.contains(&name) {
        return None;
    }
    Some(ast.add_at(kind, loc))
}

fn name(ast: &mut Ast, id: &str, ctx: ExprContext) -> NodeId {
    ast.add(Kind::Name { id: id.to_string(), ctx })
}

fn constant(ast: &mut Ast, value: Constant) -> NodeId {
    ast.add(Kind::Constant { value, kind: None })
}

fn int_constant(ast: &mut Ast, value: usize) -> NodeId {
    constant(ast, Constant::Int(Int::Small(value as i64)))
}

/// `<param>[slot]` (`_slot_ref`).
fn slot_ref(ast: &mut Ast, param: &str, slot: usize) -> NodeId {
    let value = name(ast, param, ExprContext::Load);
    let slice = int_constant(ast, slot);
    ast.add(Kind::Subscript { value, slice, ctx: ExprContext::Load })
}

/// The constant a stamped value becomes in an `ast.Constant`.
fn value_constant(value: &Value) -> Result<Constant, Error> {
    Ok(match value {
        Value::None => Constant::None,
        Value::Bool(flag) => Constant::Bool(*flag),
        Value::Int(int) => Constant::Int(int.clone()),
        Value::Float(float) => Constant::Float(*float),
        Value::Str(text) => Constant::Str(text.clone()),
        Value::Tuple(items) => {
            Constant::Tuple(items.iter().map(value_constant).collect::<Result<_, _>>()?)
        }
        other => return Err(Error::Unsupported(format!("a constant of {other:?}"))),
    })
}

/// A stamp read as Python's `getattr(node, attr, None)`: `None` when absent or None.
fn stamp<'a>(ast: &'a Ast, id: NodeId, attr: &str) -> Option<&'a Value> {
    ast.extra(id, attr).filter(|value| **value != Value::None)
}

/// The truth of a `_pine_series_len` stamp (`if get_series_len(node)`).
fn series_len(ast: &Ast, id: NodeId) -> Result<bool, Error> {
    match stamp(ast, id, SERIES_LEN_ATTR) {
        None => Ok(false),
        Some(Value::Bool(flag)) => Ok(*flag),
        Some(other) => Err(Error::Unsupported(format!("a series-length stamp {other:?}"))),
    }
}

/// A definition's instance-varying call sites (`get_varying`), empty when none.
fn varying(ast: &Ast, id: NodeId) -> Result<Vec<NodeId>, Error> {
    match stamp(ast, id, VARYING_ATTR) {
        None => Ok(Vec::new()),
        Some(Value::List(items)) => items
            .iter()
            .map(|item| match item {
                Value::Node(site) => Ok(*site),
                other => Err(Error::Unsupported(format!("a varying site {other:?}"))),
            })
            .collect(),
        Some(other) => Err(Error::Unsupported(format!("a varying stamp {other:?}"))),
    }
}

/// The node fields of a node with their names, in field order.
fn fields(ast: &Ast, id: NodeId) -> Vec<(&'static str, Vec<NodeId>)> {
    let kind = ast.kind(id);
    kind.class()
        .fields
        .iter()
        .enumerate()
        .filter_map(|(index, field)| {
            let ids = match kind.field(index) {
                FieldRef::Node(child) => vec![*child],
                FieldRef::OptNode(child) => child.iter().copied().collect(),
                FieldRef::Nodes(ids) => ids.clone(),
                FieldRef::OptNodes(ids) => ids.iter().flatten().copied().collect(),
                _ => return None,
            };
            Some((field.name, ids))
        })
        .collect()
}

/// Name -> (scope id of its LAST definition, has decorators)
type Defs = HashMap<String, (String, bool)>;

/// Pass 1 (`_ScopeIndex`): per-scope name bindings, the module-level import map,
/// and every call site of the isolation territory with the scope it stands in.
#[derive(Default)]
struct ScopeIndex {
    /// scope -> name -> (target scope id of the LAST def, has decorators)
    defs: HashMap<String, Defs>,
    classes: HashMap<String, HashSet<String>>,
    /// scope -> class name -> method name -> (scope id of its LAST def, has
    /// decorators)
    methods: HashMap<String, HashMap<String, Defs>>,
    assigned: HashMap<String, HashSet<String>>,
    /// name -> (module path, attribute)
    import_map: HashMap<String, (String, Option<String>)>,
    sites: Vec<(NodeId, Vec<String>)>,
    scope_routes: IndexMap<String, Vec<Route>>,
    pin_carriers: HashSet<String>,
    scope: String,
    segments: Vec<String>,
    territory: bool,
}

impl ScopeIndex {
    fn build(ast: &Ast, module: NodeId, layout: &ModuleLayout) -> Result<ScopeIndex, Error> {
        let mut index = ScopeIndex { territory: true, ..ScopeIndex::default() };
        index.defs.insert(String::new(), HashMap::new());
        index.classes.insert(String::new(), HashSet::new());
        index.assigned.insert(String::new(), HashSet::new());
        index.visit(ast, module, layout)?;
        Ok(index)
    }

    fn generic(&mut self, ast: &Ast, id: NodeId, layout: &ModuleLayout) -> Result<(), Error> {
        for child in children(ast, id) {
            self.visit(ast, child, layout)?;
        }
        Ok(())
    }

    fn assign(&mut self, name: &str) {
        self.assigned.entry(self.scope.clone()).or_default().insert(name.to_string());
    }

    /// Index one definition, without binding its name anywhere (`_index_def`):
    /// its scope id, and whether it is decorated.
    fn index_def(
        &mut self,
        ast: &Ast,
        id: NodeId,
        layout: &ModuleLayout,
    ) -> Result<(String, bool), Error> {
        let Kind::FunctionDef { name, args, decorator_list, .. } = ast.kind(id) else {
            unreachable!()
        };
        let outer = self.scope.clone();
        let segment = layout.scope_segment(ast, id);
        let scope = if outer.is_empty() { segment.clone() } else { format!("{outer}·{segment}") };
        self.scope = scope.clone();
        self.defs.entry(scope.clone()).or_default();
        self.classes.entry(scope.clone()).or_default();
        let Kind::Arguments { posonlyargs, args, vararg, kwonlyargs, kwarg, .. } = ast.kind(*args)
        else {
            unreachable!()
        };
        for &arg in args.iter().chain(posonlyargs).chain(kwonlyargs).chain(vararg).chain(kwarg) {
            let Kind::Arg { arg, .. } = ast.kind(arg) else { unreachable!() };
            self.assign(arg);
        }
        let territory = self.territory;
        let isolated = territory && !is_test_function(name);
        if isolated {
            self.scope_routes.entry(scope.clone()).or_default();
            if !varying(ast, id)?.is_empty() {
                self.pin_carriers.insert(scope.clone());
            }
        }
        self.segments.push(segment);
        // Every field is indexed, only the body of an isolated definition is
        // territory
        for (field, ids) in fields(ast, id) {
            self.territory = isolated && field == "body";
            for child in ids {
                self.visit(ast, child, layout)?;
            }
        }
        self.territory = territory;
        self.segments.pop();
        self.scope = outer;
        Ok((scope, !decorator_list.is_empty()))
    }

    fn visit(&mut self, ast: &Ast, id: NodeId, layout: &ModuleLayout) -> Result<(), Error> {
        match ast.kind(id) {
            Kind::FunctionDef { name, .. } => {
                let name = name.clone();
                let entry = self.index_def(ast, id, layout)?;
                self.defs.entry(self.scope.clone()).or_default().insert(name, entry);
            }
            Kind::ClassDef { name, body, .. } => {
                self.classes.entry(self.scope.clone()).or_default().insert(name.clone());
                // Only the methods of a class body are isolation scopes. The rest of
                // it runs once, when the class is created, and a method name binds
                // an attribute of the class, never a name of the enclosing scope
                let (class, body) = (name.clone(), body.clone());
                self.methods
                    .entry(self.scope.clone())
                    .or_default()
                    .entry(class.clone())
                    .or_default();
                for stmt in body {
                    let Kind::FunctionDef { name, .. } = ast.kind(stmt) else { continue };
                    let name = name.clone();
                    let entry = self.index_def(ast, stmt, layout)?;
                    let scope = self.scope.clone();
                    self.methods
                        .get_mut(&scope)
                        .and_then(|classes| classes.get_mut(&class))
                        .expect("class entry")
                        .insert(name, entry);
                }
            }
            Kind::Call { func, .. } => {
                let func = *func;
                self.generic(ast, id, layout)?;
                if self.territory
                    && !self.segments.is_empty()
                    && matches!(ast.kind(func), Kind::Name { .. } | Kind::Attribute { .. })
                {
                    self.sites.push((id, self.segments.clone()));
                }
            }
            Kind::Name { id: name, ctx } => {
                if matches!(ctx, ExprContext::Store | ExprContext::Del) {
                    self.assign(name);
                }
            }
            Kind::ExceptHandler { name, .. } => {
                if let Some(name) = name {
                    self.assign(name);
                }
                self.generic(ast, id, layout)?;
            }
            Kind::Import { names } => {
                for &alias in names {
                    let Kind::Alias { name, asname } = ast.kind(alias) else { unreachable!() };
                    let first = name.split('.').next().unwrap_or(name);
                    let bound = asname.as_deref().unwrap_or(first).to_string();
                    if !self.scope.is_empty() {
                        self.assign(&bound);
                    } else {
                        let module = if asname.is_some() { name.as_str() } else { first };
                        self.import_map.insert(bound, (module.to_string(), None));
                    }
                }
            }
            Kind::ImportFrom { module, names, level } => {
                for &alias in names {
                    let Kind::Alias { name, asname } = ast.kind(alias) else { unreachable!() };
                    let bound = asname.as_deref().unwrap_or(name).to_string();
                    if !self.scope.is_empty() {
                        self.assign(&bound);
                    } else if let Some(module) =
                        module.as_deref().filter(|module| !module.is_empty())
                        && level.unwrap_or(0) == 0
                    {
                        self.import_map.insert(bound, (module.to_string(), Some(name.clone())));
                    }
                }
            }
            _ => self.generic(ast, id, layout)?,
        }
        Ok(())
    }
}

/// The interface lookup, routing and dependency tracking a module's call sites
/// share.
struct FunctionIsolation<'a, 'h> {
    host: &'h dyn Host,
    layout: &'a mut ModuleLayout,
    index: ScopeIndex,
    carrier: HashMap<String, bool>,
    pin_carriers: HashSet<String>,
    scope_stack: Vec<String>,
    loop_depth: usize,
    lambda_depth: usize,
    comp_iter_depth: usize,
    ordinals: HashMap<String, usize>,
    used_helpers: BTreeSet<&'static str>,
    /// Callee path -> its route and the files it holds while (`_resolve_cache`)
    resolve_cache: HashMap<String, (Route, Vec<Option<String>>)>,
    /// The module's type table stamp, the dependency records the routes extend
    table: Option<Value>,
    tracked_files: HashSet<String>,
    /// The module's `_module_file_path`, `None` when it carries no usable one
    close_module: Option<String>,
    pin_slots: HashMap<String, usize>,
    pin_index: HashMap<String, HashMap<NodeId, usize>>,
    /// Call node -> its route, as the index's sites were classified
    routes: HashMap<NodeId, Route>,
    error: Option<Error>,
}

impl FunctionIsolation<'_, '_> {
    fn fail(&mut self, error: Error) {
        self.error.get_or_insert(error);
    }

    fn joined(&self) -> String {
        self.scope_stack.join("·")
    }

    // --- classification ------------------------------------------------------

    /// Classify a callee expression in a scope context (`route_for_callee`).
    fn route_for_callee(
        &mut self,
        ast: &mut Ast,
        func: NodeId,
        scope_stack: &[String],
    ) -> Result<Route, Error> {
        if self.is_series_slot_method(ast, func, scope_stack) {
            // A SeriesImpl method call the Series pass emitted: stateless by
            // construction, and an anchor could never hit anyway
            return Ok(Route::Skip);
        }
        let Some(path) = func_path(ast, func) else { return Ok(Route::Uniform) };
        if path.starts_with(SUPPORT_ALIAS_PREFIX) || is_plain_callee(&path) {
            return Ok(Route::Skip);
        }
        let parts: Vec<String> = path.split('.').map(str::to_string).collect();
        let base = parts[0].as_str();
        // Innermost-first scope-chain resolution of the base name
        for i in (0..=scope_stack.len()).rev() {
            let scope = scope_stack[..i].join("·");
            let is_assigned =
                self.index.assigned.get(&scope).is_some_and(|names| names.contains(base));
            if let Some((target, decorated)) =
                self.index.defs.get(&scope).and_then(|defs| defs.get(base))
            {
                if is_assigned || parts.len() > 1 || *decorated {
                    // Rebound name, attribute on a def, or a decorated def
                    return Ok(Route::Uniform);
                }
                return Ok(Route::Same(target.clone()));
            }
            if self.index.classes.get(&scope).is_some_and(|names| names.contains(base)) {
                if is_assigned {
                    return Ok(Route::Uniform);
                }
                if parts.len() == 2
                    && let Some((target, decorated)) = self
                        .index
                        .methods
                        .get(&scope)
                        .and_then(|classes| classes.get(base))
                        .and_then(|methods| methods.get(&parts[1]))
                {
                    // `Pair.m(p)` reaches the method's own function, which takes the
                    // state first like any definition of the module. A decorator
                    // decides what the attribute holds
                    return Ok(if *decorated {
                        Route::Uniform
                    } else {
                        Route::Same(target.clone())
                    });
                }
                // Constructor or class attribute
                return Ok(Route::Skip);
            }
            if is_assigned {
                return Ok(Route::Uniform);
            }
        }
        if let Some((module, _)) = self.index.import_map.get(base) {
            if data::is_stdlib(module) {
                return Ok(Route::Skip);
            }
            let (route, files) = self.route_imported(&path, &parts)?;
            if route.is_object_route() {
                self.track_route(ast, &files)?;
            }
            return Ok(route);
        }
        if parts.len() == 1 && data::python_names().builtins.contains(base) {
            return Ok(Route::Skip);
        }
        if base.starts_with('_') {
            return Ok(Route::Skip); // unresolvable private name -- legacy parity
        }
        Ok(Route::Uniform)
    }

    /// Whether a callee is a method of a series slot (`_is_series_slot_method`).
    fn is_series_slot_method(&self, ast: &Ast, func: NodeId, scope_stack: &[String]) -> bool {
        let Kind::Attribute { value, .. } = ast.kind(func) else { return false };
        let Kind::Subscript { value: param, slice, .. } = ast.kind(*value) else { return false };
        let (Kind::Name { id: param, .. }, Kind::Constant { value: index, .. }) =
            (ast.kind(*param), ast.kind(*slice))
        else {
            return false;
        };
        let index = match index {
            Constant::Int(Int::Small(index)) => *index,
            Constant::Bool(flag) => i64::from(*flag),
            _ => return false,
        };
        let scope_id = if param == DEFAULT_STATE_PARAM {
            scope_stack.join("·")
        } else if let Some(scope) =
            param.strip_prefix("__state·").and_then(|rest| rest.strip_suffix("__"))
        {
            scope.to_string()
        } else {
            return false;
        };
        let Some(scope) = self.layout.scopes.get(&scope_id) else { return false };
        usize::try_from(index)
            .ok()
            .and_then(|index| scope.slots.get(index))
            .is_some_and(|slot| slot.kind == SlotKind::Series)
    }

    /// The route of a dotted callee path that starts at a module-level import
    /// (`_route_imported`).
    fn route_imported(
        &mut self,
        path: &str,
        parts: &[String],
    ) -> Result<(Route, Vec<Option<String>>), Error> {
        if let Some(cached) = self.resolve_cache.get(path) {
            return Ok(cached.clone());
        }
        let (module, attr) = self.index.import_map[&parts[0]].clone();
        let names: Vec<String> = attr.into_iter().chain(parts[1..].iter().cloned()).collect();
        let result = self.resolve(module, None, names)?;
        self.resolve_cache.insert(path.to_string(), result.clone());
        Ok(result)
    }

    /// The route of a callee path in a module, read off what the modules publish
    /// (`_resolve`).
    fn resolve(
        &self,
        mut module: String,
        mut source: Option<String>,
        mut names: Vec<String>,
    ) -> Result<(Route, Vec<Option<String>>), Error> {
        let host = self.host;
        let mut files: Vec<Option<String>> = Vec::new();
        let mut star = false;
        let mut seen: HashSet<(String, Vec<String>)> = HashSet::new();
        loop {
            let target = std::iter::once(module.as_str())
                .chain(names.iter().map(String::as_str))
                .collect::<Vec<_>>()
                .join(".");
            if target == LIB_PACKAGE || target.starts_with("pynecore.lib.") {
                // Spelled the way a script spells it (`lib.ta.sma`)
                let route = if star {
                    Route::Uniform
                } else {
                    match host
                        .lib_route(&format!("lib{}", &target[LIB_PACKAGE.len()..]))
                        .map_err(host_error)?
                    {
                        Some(word) => Route::parse(&word)?,
                        None => Route::Uniform,
                    }
                };
                return Ok((route, files));
            }
            if names.is_empty() || !seen.insert((module.clone(), names.clone())) {
                // The module object itself, or imports that lead in a circle
                return Ok((Route::Uniform, files));
            }
            let pyne = !(module == "pynecore" || module.starts_with("pynecore."));
            if pyne && source.is_none() {
                source = host.module_spec_origin(&module).map_err(host_error)?;
            }
            let interface = match &source {
                Some(path) if pyne => self.interface(path)?,
                _ => None,
            };
            let (Some(interface), Some(path)) = (interface, source.clone()) else {
                let (word, read) = host.object_route(&module, &names, star).map_err(host_error)?;
                files.extend(read);
                return Ok((Route::parse(&word)?, files));
            };
            files.push(Some(path.clone()));
            let head = names[0].clone();
            if star {
                let hidden = match &interface.all {
                    Some(public) => !public.contains(&head),
                    None => head.starts_with('_'),
                };
                if hidden {
                    return Ok((Route::Uniform, files));
                }
                star = false;
            }
            if let Some(word) = interface.call_routes.get(&names.join(".")) {
                return Ok((Route::parse(word)?, files));
            }
            if interface.call_routes.contains_key(&head) {
                return Ok((Route::Uniform, files));
            }
            let Some((level, imported, attr)) =
                interface.reexports.get(&head).or_else(|| interface.reexports.get(STAR)).cloned()
            else {
                return Ok((Route::Uniform, files));
            };
            if level != 0 {
                // Relative to the package of the module the import stands in, found
                // by its directory as the type pass finds it
                let mut package: Vec<&str> = module.split('.').collect();
                package.pop();
                let level = usize::try_from(level)
                    .map_err(|_| Error::Unsupported(format!("an import level {level}")))?;
                if level > package.len() {
                    return Ok((Route::Uniform, files));
                }
                let mut directory = Path::new(&path)
                    .parent()
                    .ok_or_else(|| Error::Unsupported(format!("the directory of {path}")))?;
                for _ in 1..level {
                    directory = directory
                        .parent()
                        .ok_or_else(|| Error::Unsupported(format!("a parent of {path}")))?;
                }
                let directory = directory
                    .to_str()
                    .ok_or_else(|| Error::Unsupported(format!("the path {path}")))?
                    .to_string();
                package.truncate(package.len() + 1 - level);
                if !imported.is_empty() {
                    package.push(&imported);
                }
                module = package.join(".");
                source = host.package_member(&directory, &imported).map_err(host_error)?;
            } else {
                module = imported;
                source = None;
            }
            if attr == STAR {
                star = true;
            } else if !attr.is_empty() {
                names[0] = attr;
            } else {
                names.remove(0);
            }
        }
    }

    /// The settled interface a Pyne module publishes, when it has one to give
    /// (`_interface`).
    fn interface(&self, source: &str) -> Result<Option<CallInterface>, Error> {
        if Path::new(source).file_name().is_some_and(|name| name == "__init__.py") {
            return Ok(None);
        }
        self.host.call_interface(source, true).map_err(host_error)
    }

    /// Make the module's cache depend on what a callee's route was decided by
    /// (`_track_route`).
    fn track_route(&mut self, ast: &mut Ast, files: &[Option<String>]) -> Result<(), Error> {
        let Some(stamp) = self.table.clone() else { return Ok(()) };
        for file in files.iter().flatten() {
            if !self.tracked_files.insert(file.clone()) || !file.ends_with(".py") {
                continue;
            }
            let (key, in_package) = self.host.source_key(file).map_err(host_error)?;
            let table = table_mut(ast, &stamp)?;
            if key == table.module_path || table.deps.contains_key(&key) || in_package {
                continue;
            }
            if let Some(interface) = self.host.call_interface(&key, false).map_err(host_error)? {
                // `module_interface.add_dep`: the record, and the closure it carries
                let table = table_mut(ast, &stamp)?;
                let own = table.module_path.clone();
                table.deps.insert(interface.record.path.clone(), interface.record.clone());
                for inherited in interface.deps.values() {
                    if inherited.path != own {
                        table
                            .deps
                            .entry(inherited.path.clone())
                            .or_insert_with(|| inherited.clone());
                    }
                }
                continue;
            }
            if let Some(record) = self.host.source_record(&key).map_err(host_error)? {
                table_mut(ast, &stamp)?.deps.insert(key, record);
            }
        }
        Ok(())
    }

    /// Whether a same-module scope carries state (`_is_carrier`).
    fn is_carrier(&self, scope: &str) -> bool {
        match self.carrier.get(scope) {
            Some(carrier) => *carrier,
            None => self.own_state(scope),
        }
    }

    /// Allocated slots, or the instance vector its varying sites are about to be given.
    fn own_state(&self, scope: &str) -> bool {
        self.layout.state_carrying(scope) || self.pin_carriers.contains(scope)
    }

    /// The carrier fixpoint over the module's call graph (`_run_fixpoint`).
    fn run_fixpoint(&self, scope_routes: &IndexMap<String, Vec<Route>>) -> HashMap<String, bool> {
        let mut carrier: HashMap<String, bool> =
            scope_routes.keys().map(|scope| (scope.clone(), self.own_state(scope))).collect();
        for routes in scope_routes.values() {
            for route in routes {
                if let Route::Same(target) = route {
                    let own = self.own_state(target);
                    carrier.entry(target.clone()).or_insert(own);
                }
            }
        }
        let mut changed = true;
        while changed {
            changed = false;
            for (scope, routes) in scope_routes {
                if carrier[scope] {
                    continue;
                }
                let carries = routes.iter().any(|route| match route {
                    Route::Fast | Route::FastShared | Route::Uniform => true,
                    Route::Same(target) => carrier.get(target).copied().unwrap_or(false),
                    _ => false,
                });
                if carries {
                    carrier.insert(scope.clone(), true);
                    changed = true;
                }
            }
        }
        carrier
    }

    // --- emission helpers ----------------------------------------------------------

    fn state_param(&self) -> String {
        self.layout.state_param(&self.joined()).to_string()
    }

    /// Register a runtime helper for the import and return its alias (`_helper`).
    fn helper(&mut self, name: &'static str) -> &'static str {
        self.used_helpers.insert(name);
        HELPER_ALIASES.iter().find(|(helper, _)| *helper == name).expect("helper").1
    }

    /// `<helper>(<args>)` (`_helper_call`).
    fn helper_call(&mut self, ast: &mut Ast, name: &'static str, args: Vec<NodeId>) -> NodeId {
        let alias = self.helper(name);
        let func = self::name(ast, alias, ExprContext::Load);
        ast.add(Kind::Call { func, args, keywords: Vec::new() })
    }

    /// Fresh, attribute-free copy of a callee expression located at the callee
    /// (`_copy_callee`).
    fn copy_callee(ast: &mut Ast, func: NodeId, python: (u32, u32)) -> Result<NodeId, Error> {
        let loc = ast[func].loc;
        if let (Some(line), Some(col), Some(end_line), Some(end_col)) =
            (loc.lineno, loc.col_offset, loc.end_lineno, loc.end_col_offset)
            && let Some(copy) = copy_name_chain(ast, func, Loc::new(line, col, end_line, end_col))
        {
            return Ok(copy);
        }
        let text = crate::unparse::unparse(ast, func, python)?;
        if !text.is_ascii() {
            // The parser normalizes what is not ASCII
            return Err(Error::Unsupported(format!("reparsing the callee {text:?}")));
        }
        let copy = pyne_parse::parse_expression_into(ast, &text)?;
        // A callee another pass built without a location yields an unlocated copy,
        // which `fix_locations` anchors with its statement
        let located = [loc.lineno, loc.col_offset, loc.end_lineno, loc.end_col_offset]
            .iter()
            .all(Option::is_some);
        for node in walk(ast, copy) {
            if located {
                copy_location(ast, node, func);
            } else {
                ast[node].loc = Loc::default();
            }
        }
        Ok(copy)
    }

    /// Re-emit a call site through a bound callee, keeping its Pine type
    /// (`_stamped_call`).
    fn stamped_call(ast: &mut Ast, bound: NodeId, node: NodeId) -> Result<NodeId, Error> {
        let ty = ty_stamp(ast, node)?.to_string();
        let Kind::Call { args, keywords, .. } = ast.kind(node).clone() else { unreachable!() };
        let call = ast.add(Kind::Call { func: bound, args, keywords });
        Ok(stamp_lowering(ast, call, &ty))
    }

    /// Prepend the child-state expression as the hidden first argument (`_emit_fast`).
    fn emit_fast(
        &mut self,
        ast: &mut Ast,
        node: NodeId,
        slot: usize,
        in_loop: bool,
        vector: Option<NodeId>,
    ) -> Result<NodeId, Error> {
        let param = self.state_param();
        let Kind::Call { func, .. } = ast.kind(node) else { unreachable!() };
        let callee_copy = Self::copy_callee(ast, *func, self.host.python())?;
        let state_expr = if in_loop || self.comp_iter_depth > 0 {
            let helper = if in_loop { "__loop_state__" } else { "__slot_state__" };
            let state = name(ast, &param, ExprContext::Load);
            let index = int_constant(ast, slot);
            let args = [state, index, callee_copy].into_iter().chain(vector).collect();
            self.helper_call(ast, helper, args)
        } else {
            let target = name(ast, "__st·__", ExprContext::Store);
            let value = slot_ref(ast, &param, slot);
            let left = ast.add(Kind::NamedExpr { target, value });
            let none = constant(ast, Constant::None);
            let test = ast.add(Kind::Compare {
                left,
                ops: vec![CmpOperator::IsNot],
                comparators: vec![none],
            });
            let body = name(ast, "__st·__", ExprContext::Load);
            let alias = self.helper("__resolve_slot__");
            let func = name(ast, alias, ExprContext::Load);
            let state = name(ast, &param, ExprContext::Load);
            let index = int_constant(ast, slot);
            let args = [state, index, callee_copy].into_iter().chain(vector).collect();
            let orelse = ast.add(Kind::Call { func, args, keywords: Vec::new() });
            ast.add(Kind::IfExp { test, body, orelse })
        };
        let Kind::Call { args, .. } = ast.kind_mut(node) else { unreachable!() };
        args.insert(0, state_expr);
        Ok(node)
    }

    /// Wrap the call in the anchored bind form (`_emit_uniform`).
    fn emit_uniform(
        &mut self,
        ast: &mut Ast,
        node: NodeId,
        slot: usize,
        in_loop: bool,
        pin_expr: Option<NodeId>,
        vector: Option<NodeId>,
    ) -> Result<NodeId, Error> {
        let param = self.state_param();
        let mut pin: Vec<NodeId> = pin_expr.into_iter().collect();
        if let Some(vector) = vector {
            if pin.is_empty() {
                pin.push(constant(ast, Constant::None));
            }
            pin.push(vector);
        }
        let Kind::Call { func, .. } = ast.kind(node) else { unreachable!() };
        let func = *func;
        if in_loop || self.comp_iter_depth > 0 {
            // Helper form: the callee expression is evaluated exactly once
            let helper = if in_loop { "__bind_loop__" } else { "__bind_slot__" };
            let state = name(ast, &param, ExprContext::Load);
            let index = int_constant(ast, slot);
            let args = [state, index, func].into_iter().chain(pin).collect();
            let bound = self.helper_call(ast, helper, args);
            return Self::stamped_call(ast, bound, node);
        }
        let runs_code =
            walk(ast, func).into_iter().any(|n| matches!(ast.kind(n), Kind::Call { .. }));
        let (callee, callee_copy, bind) = if runs_code {
            // The callee expression runs code: bind it to a temporary first. The
            // conjunct is a tautology on purpose, it must never short-circuit past
            // the `__b·__` walrus in the next one
            let target = name(ast, "__c·__", ExprContext::Store);
            let left = ast.add(Kind::NamedExpr { target, value: func });
            let same = name(ast, "__c·__", ExprContext::Load);
            let bind = ast.add(Kind::Compare {
                left,
                ops: vec![CmpOperator::Is],
                comparators: vec![same],
            });
            let callee = name(ast, "__c·__", ExprContext::Load);
            let callee_copy = name(ast, "__c·__", ExprContext::Load);
            (callee, callee_copy, Some(bind))
        } else {
            (func, Self::copy_callee(ast, func, self.host.python())?, None)
        };
        let alias = self.helper("__bind_any__");
        let target = name(ast, "__b·__", ExprContext::Store);
        let value = slot_ref(ast, &param, slot);
        let left = ast.add(Kind::NamedExpr { target, value });
        let none = constant(ast, Constant::None);
        let hit =
            ast.add(Kind::Compare { left, ops: vec![CmpOperator::IsNot], comparators: vec![none] });
        let pair = name(ast, "__b·__", ExprContext::Load);
        let zero = int_constant(ast, 0);
        let first = ast.add(Kind::Subscript { value: pair, slice: zero, ctx: ExprContext::Load });
        let same = ast.add(Kind::Compare {
            left: first,
            ops: vec![CmpOperator::Is],
            comparators: vec![callee],
        });
        let func_name = name(ast, alias, ExprContext::Load);
        let state = name(ast, &param, ExprContext::Load);
        let index = int_constant(ast, slot);
        let args = [state, index, callee_copy].into_iter().chain(pin).collect();
        let rebind = ast.add(Kind::Call { func: func_name, args, keywords: Vec::new() });
        // First operand: the callee (and any call site nested in it) must be fully
        // evaluated before the anchor read binds `__b·__`
        let values = bind.into_iter().chain([hit, same]).collect();
        let test = ast.add(Kind::BoolOp { op: BoolOperator::And, values });
        let pair = name(ast, "__b·__", ExprContext::Load);
        let one = int_constant(ast, 1);
        let body = ast.add(Kind::Subscript { value: pair, slice: one, ctx: ExprContext::Load });
        let bound = ast.add(Kind::IfExp { test, body, orelse: rebind });
        Self::stamped_call(ast, bound, node)
    }

    // --- visitors ------------------------------------------------------------------

    /// Visit an expression, which always answers a node.
    fn expr(&mut self, ast: &mut Ast, id: NodeId) -> NodeId {
        match self.visit(ast, id) {
            Visited::Node(new) => new,
            other => {
                self.fail(Error::Unsupported(format!("{other:?} for an expression")));
                id
            }
        }
    }

    /// Visit each node of a list, as `[self.visit(n) for n in nodes]`.
    fn each(&mut self, ast: &mut Ast, nodes: Vec<NodeId>) -> Vec<NodeId> {
        nodes.into_iter().map(|id| self.expr(ast, id)).collect()
    }

    fn visit_function_def(&mut self, ast: &mut Ast, id: NodeId) -> Result<NodeId, Error> {
        let Kind::FunctionDef { name, body, .. } = ast.kind(id) else { unreachable!() };
        if is_test_function(name) {
            return Ok(id);
        }
        let body = body.clone();
        let segment = self.layout.scope_segment(ast, id);
        self.scope_stack.push(segment);
        let scope = self.joined();
        let sites = varying(ast, id)?;
        let scope_layout = scope_for_function(self.layout, ast, &scope, id);
        if !sites.is_empty() {
            // Reserved before the body is walked, so an inner site can address the
            // slot with a literal index
            let mut positions = HashMap::new();
            for (position, site) in sites.iter().enumerate() {
                positions.insert(*site, position);
            }
            let slot = scope_layout.add_pin(ast, sites.len());
            self.pin_index.insert(scope.clone(), positions);
            self.pin_slots.insert(scope, slot);
        }
        let (old_loop, old_lambda) = (self.loop_depth, self.lambda_depth);
        self.loop_depth = 0;
        self.lambda_depth = 0;
        // Only the body is isolation territory
        let body = self.each(ast, body);
        let Kind::FunctionDef { body: slot, .. } = ast.kind_mut(id) else { unreachable!() };
        *slot = body;
        self.loop_depth = old_loop;
        self.lambda_depth = old_lambda;
        self.scope_stack.pop();
        Ok(id)
    }

    fn visit_loop(&mut self, ast: &mut Ast, id: NodeId) {
        match ast.kind(id).clone() {
            Kind::For { iter, body, orelse, .. } => {
                let iter = self.expr(ast, iter);
                self.loop_depth += 1;
                let body = self.each(ast, body);
                let orelse = self.each(ast, orelse);
                self.loop_depth -= 1;
                let Kind::For { iter: a, body: b, orelse: c, .. } = ast.kind_mut(id) else {
                    unreachable!()
                };
                (*a, *b, *c) = (iter, body, orelse);
            }
            Kind::While { test, body, orelse } => {
                self.loop_depth += 1;
                let test = self.expr(ast, test);
                let body = self.each(ast, body);
                let orelse = self.each(ast, orelse);
                self.loop_depth -= 1;
                let Kind::While { test: a, body: b, orelse: c } = ast.kind_mut(id) else {
                    unreachable!()
                };
                (*a, *b, *c) = (test, body, orelse);
            }
            _ => unreachable!(),
        }
    }

    /// Comprehension parts run per element, the iterables aside (`_visit_comprehension`).
    fn visit_comprehension(&mut self, ast: &mut Ast, id: NodeId) {
        let generators = match ast.kind(id) {
            Kind::ListComp { generators, .. }
            | Kind::SetComp { generators, .. }
            | Kind::DictComp { generators, .. }
            | Kind::GeneratorExp { generators, .. } => generators.clone(),
            _ => unreachable!(),
        };
        for (depth, &generator) in generators.iter().enumerate() {
            // Only a later iterable runs per element
            self.loop_depth += usize::from(depth > 0);
            self.comp_iter_depth += 1;
            let Kind::Comprehension { iter, .. } = ast.kind(generator) else { unreachable!() };
            let iter = self.expr(ast, *iter);
            let Kind::Comprehension { iter: slot, .. } = ast.kind_mut(generator) else {
                unreachable!()
            };
            *slot = iter;
            self.comp_iter_depth -= 1;
            self.loop_depth -= usize::from(depth > 0);
        }
        self.loop_depth += 1;
        for &generator in &generators {
            let Kind::Comprehension { target, ifs, .. } = ast.kind(generator).clone() else {
                unreachable!()
            };
            let target = self.expr(ast, target);
            let ifs = self.each(ast, ifs);
            let Kind::Comprehension { target: a, ifs: b, .. } = ast.kind_mut(generator) else {
                unreachable!()
            };
            (*a, *b) = (target, ifs);
        }
        match ast.kind(id).clone() {
            Kind::DictComp { key, value, .. } => {
                let key = self.expr(ast, key);
                let value = self.expr(ast, value);
                let Kind::DictComp { key: a, value: b, .. } = ast.kind_mut(id) else {
                    unreachable!()
                };
                (*a, *b) = (key, value);
            }
            Kind::ListComp { elt, .. }
            | Kind::SetComp { elt, .. }
            | Kind::GeneratorExp { elt, .. } => {
                let elt = self.expr(ast, elt);
                match ast.kind_mut(id) {
                    Kind::ListComp { elt: slot, .. }
                    | Kind::SetComp { elt: slot, .. }
                    | Kind::GeneratorExp { elt: slot, .. } => *slot = elt,
                    _ => unreachable!(),
                }
            }
            _ => unreachable!(),
        }
        self.loop_depth -= 1;
    }

    fn visit_lambda(&mut self, ast: &mut Ast, id: NodeId) {
        let Kind::Lambda { body, .. } = ast.kind(id) else { unreachable!() };
        let body = *body;
        self.lambda_depth += 1;
        let body = self.expr(ast, body);
        let Kind::Lambda { body: slot, .. } = ast.kind_mut(id) else { unreachable!() };
        *slot = body;
        self.lambda_depth -= 1;
    }

    fn call_func(ast: &Ast, id: NodeId) -> NodeId {
        let Kind::Call { func, .. } = ast.kind(id) else { unreachable!() };
        *func
    }

    fn push_keyword(ast: &mut Ast, call: NodeId, arg: &str, value: NodeId) {
        let keyword = ast.add(Kind::Keyword { arg: Some(arg.to_string()), value });
        let Kind::Call { keywords, .. } = ast.kind_mut(call) else { unreachable!() };
        keywords.push(keyword);
    }

    fn visit_call(&mut self, ast: &mut Ast, node: NodeId) -> Result<NodeId, Error> {
        let Kind::Call { args, keywords, .. } = ast.kind(node).clone() else { unreachable!() };
        let args = self.each(ast, args);
        let keywords = self.each(ast, keywords);
        {
            let Kind::Call { args: a, keywords: k, .. } = ast.kind_mut(node) else {
                unreachable!()
            };
            (*a, *k) = (args, keywords);
        }
        let scope = self.joined();
        if stamp(ast, node, WIDENS_ATTR).is_some() {
            // A value whose implicit conversion differs per instance: the conversion
            // is this instance's vector entry
            if let Some(entry) = self.vector_entry(ast, node, &scope) {
                let Kind::Call { args, .. } = ast.kind_mut(node) else { unreachable!() };
                let Some(slot) = args.get_mut(1) else {
                    return Err(Error::Unsupported(
                        "a widened call without a second argument".into(),
                    ));
                };
                *slot = entry;
            }
        }
        if series_len(ast, node)? {
            let flag = constant(ast, Constant::Bool(true));
            Self::push_keyword(ast, node, SERIES_LENGTH_KEYWORD, flag);
        }
        let func = Self::call_func(ast, node);
        match ast.kind(func) {
            Kind::Name { .. } => {}
            Kind::Attribute { value, .. } => {
                if func_path(ast, func).is_none() {
                    // An attribute chain over an arbitrary expression: the calls in
                    // there are call sites of their own
                    let value = *value;
                    let value = self.expr(ast, value);
                    let Kind::Attribute { value: slot, .. } = ast.kind_mut(func) else {
                        unreachable!()
                    };
                    *slot = value;
                }
            }
            _ => {
                // Immediately-called expressions stay raw, but calls inside the
                // callee expression still get their own sites
                let new = self.expr(ast, func);
                let Kind::Call { func: slot, .. } = ast.kind_mut(node) else { unreachable!() };
                *slot = new;
                return Ok(node);
            }
        }
        let route = match self.routes.get(&node) {
            Some(route) => route.clone(),
            None => {
                let stack = self.scope_stack.clone();
                self.route_for_callee(ast, func, &stack)?
            }
        };
        let path = func_path(ast, func);
        if matches!(path.as_deref(), Some("lib.strategy.close" | "lib.strategy.close_all"))
            && !(0..=self.scope_stack.len()).any(|i| {
                self.index
                    .assigned
                    .get(&self.scope_stack[..i].join("·"))
                    .is_some_and(|names| names.contains("lib"))
            })
        {
            // The close statement, independent of the caller instance
            let Some(module) = &self.close_module else {
                return Err(Error::Unsupported("a close call in a module without a path".into()));
            };
            let loc = ast[node].loc;
            let (Some(line), Some(col)) = (loc.lineno, loc.col_offset) else {
                return Err(Error::Unsupported("a close call without a location".into()));
            };
            let site = constant(ast, Constant::Str(format!("{module}:{line}:{col}")));
            Self::push_keyword(ast, node, "_call_site", site);
        }
        if self.scope_stack.is_empty() {
            let carries = match &route {
                Route::Fast | Route::FastShared => true,
                Route::Same(target) => self.is_carrier(target),
                _ => false,
            };
            if carries {
                return Err(Error::Rejected(
                    "Stateful function calls are not supported at module level".into(),
                ));
            }
            if let Some(pin) = stamp(ast, node, PIN_ATTR).cloned()
                && route == Route::Uniform
            {
                // An overload site up here binds once, in place
                let pin = constant(ast, value_constant(&pin)?);
                let bound = self.helper_call(ast, "__bind_pinned__", vec![func, pin]);
                return Self::stamped_call(ast, bound, node);
            }
            return Ok(node);
        }
        let route = match route {
            Route::Same(target) => {
                if self.is_carrier(&target) {
                    Route::Fast
                } else {
                    Route::Direct
                }
            }
            other => other,
        };
        if matches!(route, Route::Skip | Route::Direct) {
            return Ok(node);
        }
        let (route, in_loop) = if self.lambda_depth > 0 {
            // A lambda body runs at its consumer's cadence, not as a loop iteration
            (Route::Uniform, false)
        } else if route == Route::FastShared {
            (Route::Fast, false)
        } else {
            (route, self.loop_depth > 0)
        };
        let ordinal = self.ordinals.entry(scope.clone()).or_insert(0);
        let call_id = format!(
            "{scope}·{}·{ordinal}",
            path.as_deref().filter(|path| !path.is_empty()).unwrap_or("<callee>")
        );
        *ordinal += 1;
        self.layout.scope(&scope);
        let (pin_expr, vector) = self.channel_args(ast, node, &scope)?;
        if route == Route::Fast {
            let slot = self.layout.scope(&scope).add_child(ast, &call_id, in_loop);
            return self.emit_fast(ast, node, slot, in_loop, vector);
        }
        let slot = self.layout.scope(&scope).add_anchor(ast, &call_id, in_loop);
        self.emit_uniform(ast, node, slot, in_loop, pin_expr, vector)
    }

    /// The overload pin and the instance vector this call site passes on
    /// (`_channel_args`).
    fn channel_args(
        &mut self,
        ast: &mut Ast,
        node: NodeId,
        scope: &str,
    ) -> Result<(Option<NodeId>, Option<NodeId>), Error> {
        let mut pin = match stamp(ast, node, PIN_ATTR).cloned() {
            Some(value) => Some(constant(ast, value_constant(&value)?)),
            None => None,
        };
        let mut vector = match stamp(ast, node, VECTOR_ATTR).cloned() {
            Some(value) => Some(constant(ast, value_constant(&value)?)),
            None => None,
        };
        if let Some(entry) = self.vector_entry(ast, node, scope) {
            if stamp(ast, node, SERIES_LENS_ATTR).is_some() {
                // A window call whose machine differs per instance takes it as an
                // argument
                Self::push_keyword(ast, node, SERIES_LENGTH_KEYWORD, entry);
            } else if stamp(ast, node, PINS_ATTR).is_some() {
                pin = Some(entry);
            } else {
                vector = Some(entry);
            }
        }
        Ok((pin, vector))
    }

    /// The read of an instance-varying site's own entry, `__state__[k][j]`
    /// (`_vector_entry`).
    fn vector_entry(&self, ast: &mut Ast, node: NodeId, scope: &str) -> Option<NodeId> {
        let position = *self.pin_index.get(scope)?.get(&node)?;
        let param = self.state_param();
        let value = slot_ref(ast, &param, self.pin_slots[scope]);
        let slice = int_constant(ast, position);
        Some(ast.add(Kind::Subscript { value, slice, ctx: ExprContext::Load }))
    }
}

fn table_mut<'a>(ast: &'a mut Ast, stamp: &Value) -> Result<&'a mut PineTypeTable, Error> {
    ast.native_mut::<PineTypeTable>(stamp)
        .ok_or_else(|| Error::Unsupported(format!("a {TABLE_ATTR} stamp {stamp:?}")))
}

impl Transformer for FunctionIsolation<'_, '_> {
    fn visit(&mut self, ast: &mut Ast, id: NodeId) -> Visited {
        if self.error.is_some() {
            return Visited::Node(id);
        }
        match ast.kind(id) {
            // Only the methods are isolation territory: the rest of a class body
            // runs once, when the class is created
            Kind::ClassDef { body, .. } => {
                for stmt in body.clone() {
                    if matches!(ast.kind(stmt), Kind::FunctionDef { .. })
                        && let Err(error) = self.visit_function_def(ast, stmt)
                    {
                        self.fail(error);
                    }
                }
            }
            Kind::FunctionDef { .. } => {
                return match self.visit_function_def(ast, id) {
                    Ok(id) => Visited::Node(id),
                    Err(error) => {
                        self.fail(error);
                        Visited::Node(id)
                    }
                };
            }
            Kind::For { .. } | Kind::While { .. } => self.visit_loop(ast, id),
            Kind::ListComp { .. }
            | Kind::SetComp { .. }
            | Kind::DictComp { .. }
            | Kind::GeneratorExp { .. } => self.visit_comprehension(ast, id),
            Kind::Lambda { .. } => self.visit_lambda(ast, id),
            Kind::Call { .. } => {
                return match self.visit_call(ast, id) {
                    Ok(new) => Visited::Node(new),
                    Err(error) => {
                        self.fail(error);
                        Visited::Node(id)
                    }
                };
            }
            _ => generic_visit(self, ast, id),
        }
        Visited::Node(id)
    }
}

pub fn run(ast: &mut Ast, module: NodeId, ctx: &Context<'_>) -> Result<NodeId, Error> {
    if !matches!(ast.kind(module), Kind::Module { .. }) {
        return Err(Error::Unsupported("FunctionIsolation on a non-module".into()));
    }
    // A source stamp, or (for a standalone tree) a digest of the tree: only a
    // close call reads it, which then needs the stamp
    let close_module = match ast.extra(module, "_module_file_path") {
        Some(Value::Str(path)) if !path.is_empty() => Some(path.clone()),
        _ => None,
    };
    let table = match ast.extra(module, TABLE_ATTR) {
        None | Some(Value::None) => None,
        Some(value) => Some(value.clone()),
    };
    let mut layout = ctx.slot_layout.borrow_mut();
    layout.assign_scope_ids(ast, module);
    let index = ScopeIndex::build(ast, module, &layout)?;
    let mut pass = FunctionIsolation {
        host: ctx.host,
        layout: &mut layout,
        index,
        carrier: HashMap::new(),
        pin_carriers: HashSet::new(),
        scope_stack: Vec::new(),
        loop_depth: 0,
        lambda_depth: 0,
        comp_iter_depth: 0,
        ordinals: HashMap::new(),
        used_helpers: BTreeSet::new(),
        resolve_cache: HashMap::new(),
        table,
        tracked_files: HashSet::new(),
        close_module,
        pin_slots: HashMap::new(),
        pin_index: HashMap::new(),
        routes: HashMap::new(),
        error: None,
    };
    let sites = std::mem::take(&mut pass.index.sites);
    let mut scope_routes = std::mem::take(&mut pass.index.scope_routes);
    for (site, stack) in sites {
        let func = FunctionIsolation::call_func(ast, site);
        let route = pass.route_for_callee(ast, func, &stack)?;
        scope_routes.entry(stack.join("·")).or_default().push(route.clone());
        // The emission reaches this site in the same scope and finds the same
        // callee, so it reuses the route
        pass.routes.insert(site, route);
    }
    pass.pin_carriers = std::mem::take(&mut pass.index.pin_carriers);
    pass.carrier = pass.run_fixpoint(&scope_routes);
    generic_visit(&mut pass, ast, module);
    if let Some(error) = pass.error {
        return Err(error);
    }
    if !pass.used_helpers.is_empty() {
        let names = pass
            .used_helpers
            .iter()
            .map(|helper| {
                let alias =
                    HELPER_ALIASES.iter().find(|(name, _)| name == helper).expect("helper").1;
                ast.add(Kind::Alias { name: helper.to_string(), asname: Some(alias.to_string()) })
            })
            .collect();
        let import = ast.add(Kind::ImportFrom {
            module: Some("pynecore.core.instance_state".into()),
            names,
            level: Some(0),
        });
        let Kind::Module { body, .. } = ast.kind(module) else { unreachable!() };
        let body = body.clone();
        let mut insert = 0;
        if let Some(&first) = body.first()
            && let Kind::Expr { value } = ast.kind(first)
            && matches!(ast.kind(*value), Kind::Constant { value: Constant::Str(_), .. })
        {
            insert = 1;
        }
        for (i, &stmt) in body.iter().enumerate().skip(insert) {
            match ast.kind(stmt) {
                Kind::Import { .. } | Kind::ImportFrom { .. } => insert = i + 1,
                Kind::Expr { .. } => {}
                _ => break,
            }
        }
        let Kind::Module { body, .. } = ast.kind_mut(module) else { unreachable!() };
        body.insert(insert, import);
    }
    Ok(module)
}
