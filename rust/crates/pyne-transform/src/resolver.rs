//! A host without Python: imported modules located, read and transformed natively.
//!
//! The Python host answers a step's questions about other modules from the live
//! interpreter: `find_spec`, the files on disk, and the import hook, which transforms
//! a Pyne dependency into its interface (`import_hook.compile_interface`). This is the
//! same machinery over a [`Files`] source -- a browser's virtual file system, or a
//! request that carries the files: a Pyne dependency is analysed and lowered here,
//! by the native steps, and published as `_publish`, `_lower_published` and
//! `_settle_deps` publish it, so a dependent reads the very interface the import
//! hook would have handed it.
//!
//! What only an interpreter can answer -- where `find_spec` finds a dotted name, and
//! the route of a callee read off the object a plain Python module binds -- the
//! [`Files`] source answers.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use pyne_ast::{Ast, Constant, FieldRef, Kind, NodeId, Unsupported, Value};

use crate::module_interface::{self as interfaces, Interface, SOURCE_DIGEST, build_interface};
use crate::pine_type_table::{
    CallInterface, DepRecord, ModuleInterface, PineTypeTable, TABLE_ATTR,
};
use crate::slot_layout::ModuleLayout;
use crate::{ANALYSIS, Context, Error, Host, LOWERING, SourceRead, half, run_steps};

type R<T> = Result<T, Unsupported>;

/// One file of a [`Files`] source.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct File {
    /// Its resolved path (`str(Path(path).resolve())`)
    pub resolved: String,
    pub mtime_ns: i64,
    pub size: i64,
    /// Its bytes, `None` when it cannot be read
    pub data: Option<Rc<[u8]>>,
}

/// What a host without an interpreter knows about the modules around the one it
/// transforms.
pub trait Files {
    /// The interpreter whose `repr` and `ast.unparse` the transform follows.
    fn python(&self) -> (u32, u32);

    /// The resolved directory of the `pynecore` package: a module below it is
    /// pynecore's own, not user code.
    fn package(&self) -> &str;

    /// The `.py` source a dotted module name resolves to, found without running a
    /// package (`module_paths.module_spec_origin`).
    fn spec_origin(&self, dotted: &str) -> R<Option<String>>;

    /// The `.py` source of an imported module as the enum capture finds it
    /// (`export_capture.module_origin`).
    fn module_origin(&self, name: &str, importer: &str, level: i64) -> R<Option<String>>;

    /// A file, `None` when there is no file at the path.
    fn file(&self, path: &str) -> R<Option<File>>;
}

/// How the modules a transform reaches are transformed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Options {
    /// Whether a dependency nothing published yet is transformed into its interface
    /// (`PipelineContext.analyse` set); without it only what this host transformed
    /// already answers
    pub analyse: bool,
    pub sequential: bool,
    pub security_slice_disabled: bool,
}

/// One transform of a dependency that a lookup started, while it runs (`Compile`).
struct Compile {
    path: String,
    /// The modules whose analysis was on the stack when it began
    outer: HashSet<String>,
    /// Whether it has reached one of them since
    cyclic: bool,
}

/// Why a dependency's transform published nothing, or stopped early.
enum Stop {
    /// Its analysis reached back into a module under analysis (`_ReachedBack`)
    ReachedBack,
    /// It is no valid module (`SyntaxError` and the loader's other refusals)
    Rejected,
    Unsupported(Unsupported),
}

impl From<Unsupported> for Stop {
    fn from(err: Unsupported) -> Stop {
        Stop::Unsupported(err)
    }
}

impl From<Error> for Stop {
    fn from(err: Error) -> Stop {
        match err {
            Error::Unsupported(message) => Stop::Unsupported(Unsupported(message)),
            Error::Rejected(_) => Stop::Rejected,
        }
    }
}

/// A [`Host`] over a [`Files`] source, with the registry, the analysing marks and the
/// transforms in progress `module_interface` and the import hook keep per process.
pub struct Resolver<F: Files> {
    pub files: F,
    options: Options,
    /// Resolved path -> the interface that module publishes (`_registry`)
    registry: RefCell<HashMap<String, Rc<Interface>>>,
    /// Paths whose analysis has not returned yet (`_analysing`)
    analysing: RefCell<HashSet<String>>,
    /// Per open analysing scope, the path it marked, `None` when it marked nothing
    scopes: RefCell<Vec<Option<String>>>,
    /// The dependency transforms in progress, innermost last (`_compiles`)
    compiles: RefCell<Vec<Compile>>,
    /// The Pyne modules whose transform is running (`import_hook._transforming`)
    transforming: RefCell<HashSet<String>>,
}

impl<F: Files> Resolver<F> {
    pub fn new(files: F, options: Options) -> Resolver<F> {
        Resolver {
            files,
            options,
            registry: RefCell::new(HashMap::new()),
            analysing: RefCell::new(HashSet::new()),
            scopes: RefCell::new(Vec::new()),
            compiles: RefCell::new(Vec::new()),
            transforming: RefCell::new(HashSet::new()),
        }
    }

    /// The identity a module is registered and looked up under (`_key`).
    fn key(&self, path: &str) -> R<String> {
        Ok(match self.files.file(path)? {
            Some(file) => file.resolved,
            None => path.to_string(),
        })
    }

    fn stat(&self, path: &str) -> R<Option<(i64, i64)>> {
        Ok(self.files.file(path)?.map(|file| (file.mtime_ns, file.size)))
    }

    fn register(&self, interface: Interface) -> R<Rc<Interface>> {
        let key = self.key(&interface.path)?;
        let interface = Rc::new(interface);
        self.registry.borrow_mut().insert(key, interface.clone());
        Ok(interface)
    }

    fn registered(&self, path: &str) -> R<Option<Rc<Interface>>> {
        let key = self.key(path)?;
        Ok(self.registry.borrow().get(&key).cloned())
    }

    /// Mark a module as being analysed until the matching [`Self::leave`]
    /// (`analysing_scope` entered).
    fn enter(&self, path: &str) -> R<()> {
        let marked = if path.is_empty() {
            None
        } else {
            let key = self.key(path)?;
            self.analysing.borrow_mut().insert(key.clone()).then_some(key)
        };
        self.scopes.borrow_mut().push(marked);
        Ok(())
    }

    fn leave(&self) {
        if let Some(Some(key)) = self.scopes.borrow_mut().pop() {
            self.analysing.borrow_mut().remove(&key);
        }
    }

    /// Run `body` inside an analysing scope of `path`.
    fn analysing_scope<T>(&self, path: &str, body: impl FnOnce() -> R<T>) -> R<T> {
        self.enter(path)?;
        let result = body();
        self.leave();
        result
    }

    /// Whether reaching a module closes an import cycle, noted where it matters
    /// (`_closes_cycle`).
    fn closes(&self, key: &str) -> bool {
        if !self.analysing.borrow().contains(key) {
            return false;
        }
        for compile in self.compiles.borrow_mut().iter_mut() {
            if compile.outer.contains(key) {
                compile.cyclic = true;
            }
        }
        true
    }

    /// A module's interface, from wherever it is cheapest to get (`lookup`).
    fn lookup(&self, path: &str, settled: bool) -> R<Option<Rc<Interface>>> {
        let key = self.key(path)?;
        if self.closes(&key) {
            return Ok(None);
        }
        let stat = self.stat(&key)?;
        let hit = self.registry.borrow().get(&key).cloned();
        if let Some(hit) = hit {
            if stat == Some((hit.mtime_ns, hit.size))
                && self.closure_current(&key, &hit, settled)?
            {
                if !settled || !hit.routes.is_empty() {
                    return Ok(Some(hit));
                }
            } else {
                self.registry.borrow_mut().remove(&key);
            }
        }
        if stat.is_none() || !self.options.analyse {
            return Ok(None);
        }
        let Some(interface) = self.compile_interface(&key)? else { return Ok(None) };
        self.registry.borrow_mut().insert(key, interface.clone());
        Ok((!settled || !interface.routes.is_empty()).then_some(interface))
    }

    /// Whether every module an interface was derived from still says the same thing
    /// (`_closure_current`).
    fn closure_current(&self, key: &str, interface: &Interface, settled: bool) -> R<bool> {
        if interface.deps.is_empty() {
            return Ok(true);
        }
        self.analysing_scope(key, || {
            for record in interface.deps.values() {
                if !self.dep_current(record, settled)? {
                    return Ok(false);
                }
            }
            Ok(true)
        })
    }

    /// Whether a dependency still means what the dependent was built against
    /// (`dep_current`).
    fn dep_current(&self, record: &DepRecord, settled: bool) -> R<bool> {
        let Some(stat) = self.stat(&record.path)? else { return Ok(false) };
        if stat == (record.mtime_ns, record.size) {
            return Ok(true);
        }
        if record.digest == SOURCE_DIGEST {
            return Ok(false);
        }
        let Some(interface) = self.lookup(&record.path, settled)? else { return Ok(false) };
        Ok(interface.digest == record.digest && (!settled || interface.routes == record.routes))
    }

    /// A dependency record with the routes it was made without, when they can be had
    /// (`settle`).
    fn settle(&self, record: &DepRecord) -> R<DepRecord> {
        if !record.routes.is_empty() {
            return Ok(record.clone());
        }
        let Some(interface) = self.lookup(&record.path, true)? else { return Ok(record.clone()) };
        if (interface.mtime_ns, interface.size) != (record.mtime_ns, record.size)
            || interface.digest != record.digest
        {
            return Ok(record.clone());
        }
        Ok(DepRecord { routes: interface.routes.clone(), ..record.clone() })
    }

    /// One Pyne module's interface for another module's transform
    /// (`import_hook.compile_interface`); there is no `.pyc` to read it from here.
    fn compile_interface(&self, path: &str) -> R<Option<Rc<Interface>>> {
        if self.transforming.borrow().contains(path) {
            return Ok(None);
        }
        let Some(file) = self.files.file(path)? else { return Ok(None) };
        let Some(data) = file.data.clone() else { return Ok(None) };
        if !starts_with_pyne(&data[..data.len().min(4096)]) {
            return Ok(None);
        }
        self.compile_into_cache(path, &file, &data)
    }

    /// Transform a module the way importing it would (`_compile_into_cache`).
    fn compile_into_cache(&self, path: &str, file: &File, data: &[u8]) -> R<Option<Rc<Interface>>> {
        let outer = self.analysing.borrow().clone();
        self.compiles.borrow_mut().push(Compile { path: self.key(path)?, outer, cyclic: false });
        let result = self.source_to_code(path, file, data);
        self.compiles.borrow_mut().pop();
        match result {
            Ok(()) | Err(Stop::ReachedBack) => self.registered(path),
            Err(Stop::Rejected) => Ok(None),
            Err(Stop::Unsupported(err)) => Err(err),
        }
    }

    /// Whether a dependency's transform reached back into a module under analysis
    /// (`reached_back`).
    fn reached_back(&self, path: &str) -> R<bool> {
        let key = self.key(path)?;
        Ok(self
            .compiles
            .borrow()
            .last()
            .is_some_and(|compile| compile.cyclic && compile.path == key))
    }

    /// `PyneLoader.source_to_code`, for the interface it publishes: the bytecode
    /// itself is no host's business here.
    fn source_to_code(&self, path: &str, file: &File, data: &[u8]) -> Result<(), Stop> {
        let Ok(source) = std::str::from_utf8(data) else { return Err(Stop::Rejected) };
        if !mentions_pyne(source) {
            return Ok(());
        }
        let (mut ast, module) = pyne_parse::parse_module_as(source, self.files.python())?;
        let (is_pyne, pyne_mode) = module_mode(&ast, module);
        if !is_pyne {
            return Ok(());
        }
        let fingerprint = (file.mtime_ns, file.size);
        let na_bool = script_bool_na(&ast, module).ok_or(Stop::Rejected)?;
        let key = self.key(path)?;
        let outermost = self.transforming.borrow_mut().insert(key.clone());
        let result =
            self.transform_module(&mut ast, module, source, &key, pyne_mode, na_bool, fingerprint);
        if outermost {
            self.transforming.borrow_mut().remove(&key);
        }
        result
    }

    /// Both halves of the pipeline, publishing what the module exports on the way
    /// (`_transform_module`, then `_settle_deps`).
    #[allow(clippy::too_many_arguments)]
    fn transform_module(
        &self,
        ast: &mut Ast,
        module: NodeId,
        source: &str,
        path: &str,
        pyne_mode: Option<String>,
        na_bool: bool,
        fingerprint: (i64, i64),
    ) -> Result<(), Stop> {
        let user_code = !in_package(path, self.files.package());
        let analysed = self.analyse_tree(ast, module, source, path, &pyne_mode, user_code)?;
        let table = module_table(ast, analysed)?;
        let published = match &table {
            Some(table) => {
                Some(self.register(build_interface(ast, analysed, table, path, fingerprint)?)?)
            }
            None => None,
        };
        if self.reached_back(path)? {
            return Err(Stop::ReachedBack);
        }
        let shape = match &published {
            Some(_) => interfaces::routes_shape(ast, analysed, self.files.python())?,
            None => String::new(),
        };
        let plan = published.as_ref().map(|_| interfaces::call_plan(ast, analysed));
        let ctx = self.context(path, "", &pyne_mode, user_code, na_bool);
        let lowered = run_steps(ast, analysed, &half(LOWERING, user_code), &ctx)?;
        let (Some(published), Some(plan)) = (published, plan) else { return Ok(()) };
        let layout = ctx.slot_layout.borrow();
        let mut interface = (*published).clone();
        interface.routes = interfaces::routes_digest(&shape, ast, lowered, &layout);
        interface.call_routes = interfaces::call_routes(&plan, ast, &layout);
        interface.reexports = plan.reexports;
        let published = self.register(interface)?;
        // `_settle_deps`, over the records the type pass and the lowering made
        let table = module_table(ast, analysed)?;
        let Some(table) = table.filter(|table| !table.deps.is_empty()) else { return Ok(()) };
        let mut deps = table.deps.clone();
        for record in deps.values_mut() {
            *record = self.settle(record)?;
        }
        let mut interface = (*published).clone();
        interface.deps = deps;
        self.register(interface)?;
        Ok(())
    }

    fn context<'h>(
        &'h self,
        path: &str,
        source: &str,
        pyne_mode: &Option<String>,
        user_code: bool,
        na_bool: bool,
    ) -> Context<'h> {
        Context {
            path: path.to_string(),
            pyne_mode: pyne_mode.clone(),
            source: source.to_string(),
            user_code,
            na_bool,
            sequential: self.options.sequential,
            security_slice_disabled: self.options.security_slice_disabled,
            emit_layout: true,
            host: self,
            slot_layout: RefCell::new(ModuleLayout::new(pyne_mode.as_deref() == Some("lib"))),
        }
    }

    /// The pipeline up to and including the type pass (`_analyse_tree`); the edge
    /// gate's findings only join the diagnostics, which no interface carries.
    fn analyse_tree(
        &self,
        ast: &mut Ast,
        module: NodeId,
        source: &str,
        path: &str,
        pyne_mode: &Option<String>,
        user_code: bool,
    ) -> Result<NodeId, Stop> {
        reject_reserved_names(ast, module, source)?;
        ast.set_extra(module, "_module_file_path", Value::Str(path.to_string()));
        let mut body = match ast.kind(module) {
            Kind::Module { body, .. } => body.clone(),
            _ => return Err(Stop::Unsupported(Unsupported("a module that is no Module".into()))),
        };
        body.retain(|&stmt| {
            !matches!(ast.kind(stmt), Kind::FunctionDef { name, .. }
                if name.starts_with("__test_") && name.ends_with("__"))
        });
        if let Kind::Module { body: kept, .. } = ast.kind_mut(module) {
            *kept = body;
        }
        if user_code {
            reject_pipeline_names(ast, module)?;
        }
        let ctx = self.context(path, source, pyne_mode, user_code, false);
        Ok(run_steps(ast, module, &half(ANALYSIS, user_code), &ctx)?)
    }
}

/// The type table the type pass attached to a module (`module_table`).
fn module_table(ast: &Ast, module: NodeId) -> R<Option<PineTypeTable>> {
    match ast.extra(module, TABLE_ATTR) {
        None | Some(Value::None) => Ok(None),
        Some(value) => ast
            .native::<PineTypeTable>(value)
            .cloned()
            .map(Some)
            .ok_or_else(|| Unsupported(format!("a {TABLE_ATTR} stamp {value:?}"))),
    }
}

/// Whether a resolved path lies below a directory (`Path.is_relative_to`).
fn in_package(path: &str, package: &str) -> bool {
    let package = package.trim_end_matches('/');
    !package.is_empty()
        && (path == package || path.strip_prefix(package).is_some_and(|rest| rest.starts_with('/')))
}

/// `Path(directory) / name`.
fn join(directory: &str, name: &str) -> String {
    let directory = directory.trim_end_matches('/');
    if directory.is_empty() { format!("/{name}") } else { format!("{directory}/{name}") }
}

/// The last separator-delimited part of a path and what precedes it.
fn split(path: &str) -> (&str, &str) {
    match path.rfind('/') {
        Some(at) => (&path[..at], &path[at + 1..]),
        None => ("", path),
    }
}

impl<F: Files> Host for Resolver<F> {
    fn python(&self) -> (u32, u32) {
        self.files.python()
    }

    fn module_origin(&self, name: &str, importer: &str, level: i64) -> R<Option<String>> {
        self.files.module_origin(name, importer, level)
    }

    fn read_source(&self, path: &str) -> R<Option<SourceRead>> {
        let Some(file) = self.files.file(path)? else { return Ok(None) };
        Ok(file.data.map(|data| SourceRead {
            resolved: file.resolved,
            data: data.to_vec(),
            mtime_ns: file.mtime_ns,
            size: file.size,
        }))
    }

    fn interface(&self, path: &str) -> R<Option<ModuleInterface>> {
        Ok(self.lookup(path, false)?.map(|interface| interface.types()))
    }

    fn closes_cycle(&self, path: &str) -> R<bool> {
        Ok(!path.is_empty() && self.closes(&self.key(path)?))
    }

    fn begin_analysis(&self, path: &str) -> R<()> {
        self.enter(path)
    }

    fn end_analysis(&self) -> R<()> {
        self.leave();
        Ok(())
    }

    fn module_spec_origin(&self, dotted: &str) -> R<Option<String>> {
        self.files.spec_origin(dotted)
    }

    /// `module_paths.package_member`: `name.py` before `name/__init__.py`.
    fn package_member(&self, directory: &str, dotted: &str) -> R<Option<String>> {
        let candidates = if dotted.is_empty() {
            vec![join(directory, "__init__.py")]
        } else {
            let target = dotted.split('.').fold(directory.to_string(), |at, part| join(&at, part));
            let (parent, name) = split(&target);
            vec![join(parent, &format!("{name}.py")), join(&target, "__init__.py")]
        };
        for candidate in candidates {
            if self.files.file(&candidate)?.is_some() {
                return Ok(Some(candidate));
            }
        }
        Ok(None)
    }

    fn call_interface(&self, path: &str, settled: bool) -> R<Option<CallInterface>> {
        Ok(self.lookup(path, settled)?.map(|interface| interface.calls()))
    }

    fn source_key(&self, path: &str) -> R<(String, bool)> {
        let key = self.key(path)?;
        let inside = in_package(&key, self.files.package());
        Ok((key, inside))
    }

    fn source_record(&self, path: &str) -> R<Option<DepRecord>> {
        let Some(file) = self.files.file(path)? else { return Ok(None) };
        Ok(Some(DepRecord {
            path: file.resolved,
            mtime_ns: file.mtime_ns,
            size: file.size,
            digest: SOURCE_DIGEST.into(),
            routes: SOURCE_DIGEST.into(),
        }))
    }
}

// --- the loader's own reading of a module ---------------------------------

/// `bytes.isspace` of one byte, as `\s` in a bytes pattern matches it.
fn byte_space(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\n' | b'\r' | b'\x0b' | b'\x0c')
}

/// Whether a source head is a Pyne module: its docstring begins with `@pyne`
/// (`source_starts_with_pyne`, `_PYNE_HEAD_RE`).
pub fn starts_with_pyne(head: &[u8]) -> bool {
    let mut at = 0;
    loop {
        while at < head.len() && byte_space(head[at]) {
            at += 1;
        }
        if head.get(at) != Some(&b'#') {
            break;
        }
        while at < head.len() && head[at] != b'\n' && head[at] != b'\r' {
            at += 1;
        }
        match head.get(at) {
            None => break,
            Some(b'\n') => at += 1,
            // ``\r?\n``: a carriage return alone ends no comment line
            _ if head.get(at + 1) == Some(&b'\n') => at += 2,
            _ => return false,
        }
    }
    while at < head.len() && b"rRbBuUfF".contains(&head[at]) {
        at += 1;
    }
    let rest = &head[at..];
    [&b"\"\"\""[..], b"'''", b"\"", b"'"].iter().any(|quote| {
        let Some(after) = rest.strip_prefix(*quote) else { return false };
        let start = after.iter().position(|&byte| !byte_space(byte)).unwrap_or(after.len());
        let Some(tail) = after[start..].strip_prefix(b"@pyne") else { return false };
        tail.is_empty() || byte_space(tail[0]) || tail.starts_with(quote)
    })
}

/// `str.isspace` of one character, as `\s` in a str pattern matches it.
fn char_space(c: char) -> bool {
    c.is_whitespace() || ('\x1c'..='\x1f').contains(&c)
}

/// The loader's prefilter: `@pyne` as a standalone token (`re.search(r'@pyne(\s|["\']|$)')`).
fn mentions_pyne(source: &str) -> bool {
    source.match_indices("@pyne").any(|(at, token)| {
        let rest = &source[at + token.len()..];
        match rest.chars().next() {
            None => true,
            Some(c) => char_space(c) || c == '"' || c == '\'',
        }
    })
}

/// The `@pyne` marker and mode word of a module's docstring (`_module_mode`).
pub fn module_mode(ast: &Ast, module: NodeId) -> (bool, Option<String>) {
    let Kind::Module { body, .. } = ast.kind(module) else { return (false, None) };
    let Some(&first) = body.first() else { return (false, None) };
    let Kind::Expr { value } = ast.kind(first) else { return (false, None) };
    let Kind::Constant { value: Constant::Str(text), .. } = ast.kind(*value) else {
        return (false, None);
    };
    let Some(rest) = text.trim_start_matches(char_space).strip_prefix("@pyne") else {
        return (false, None);
    };
    let ends = |rest: &str| rest.is_empty() || rest.starts_with(char_space);
    let gap = rest.trim_start_matches([' ', '\t']);
    if gap.len() < rest.len() {
        let word_end = gap.find(|c: char| !(c.is_alphanumeric() || c == '_')).unwrap_or(gap.len());
        if word_end > 0 && ends(&gap[word_end..]) {
            return (true, Some(gap[..word_end].to_string()));
        }
    }
    if ends(rest) { (true, None) } else { (false, None) }
}

/// The script's bool na choice off its `script.*` decorator (`_script_bool_na`),
/// false without one; None for a decorator the loader refuses.
pub fn script_bool_na(ast: &Ast, module: NodeId) -> Option<bool> {
    let Kind::Module { body, .. } = ast.kind(module) else { return Some(false) };
    let mut lib_names: HashSet<&str> = HashSet::new();
    let mut script_names: HashSet<&str> = HashSet::new();
    for &node in body {
        match ast.kind(node) {
            Kind::ImportFrom { module, names, .. } => {
                for &alias in names {
                    let Kind::Alias { name, asname } = ast.kind(alias) else { continue };
                    if module.as_deref() == Some("pynecore.lib") && name == "script" {
                        script_names.insert(asname.as_deref().unwrap_or(name));
                    }
                }
            }
            Kind::Import { names } => {
                for &alias in names {
                    let Kind::Alias { name, asname: Some(asname) } = ast.kind(alias) else {
                        continue;
                    };
                    if name == "pynecore.lib" {
                        lib_names.insert(asname);
                    }
                }
            }
            _ => {}
        }
    }
    let is_script_decorator = |func: NodeId| -> bool {
        let mut chain: Vec<&str> = Vec::new();
        let mut func = func;
        while let Kind::Attribute { value, attr, .. } = ast.kind(func) {
            chain.push(attr);
            func = *value;
        }
        let Kind::Name { id, .. } = ast.kind(func) else { return false };
        if chain.is_empty() {
            return false;
        }
        chain.push(id);
        chain.reverse();
        if !["indicator", "strategy", "library"].contains(chain.last().expect("a chain")) {
            return false;
        }
        let head = &chain[..chain.len() - 1];
        head == ["script"]
            || head.len() == 1 && script_names.contains(head[0])
            || head == ["pynecore", "lib", "script"]
            || head.len() == 2
                && head[1] == "script"
                && (head[0] == "lib" || lib_names.contains(head[0]))
    };
    let mut entry: Option<NodeId> = None;
    for &node in body {
        let Kind::FunctionDef { name, decorator_list, .. } = ast.kind(node) else { continue };
        for &decorator in decorator_list {
            let Kind::Call { func, .. } = ast.kind(decorator) else { continue };
            if !is_script_decorator(*func) {
                continue;
            }
            if name != "main" || entry.is_some() {
                return None;
            }
            entry = Some(decorator);
            break;
        }
    }
    let Some(entry) = entry else { return Some(false) };
    let Kind::Call { keywords, .. } = ast.kind(entry) else { return Some(false) };
    for &keyword in keywords {
        let Kind::Keyword { arg, value } = ast.kind(keyword) else { continue };
        if arg.as_deref() != Some("na_bool") {
            continue;
        }
        return match ast.kind(*value) {
            Kind::Constant { value: Constant::Bool(value), .. } => Some(*value),
            _ => None,
        };
    }
    Some(false)
}

/// The characters whose NFKC form holds the reserved separator `·`: it, U+0387 GREEK
/// ANO TELEIA and U+013F / U+0140 LATIN LETTER L WITH MIDDLE DOT.
const RESERVED_SOURCES: [char; 4] = ['\u{b7}', '\u{387}', '\u{13f}', '\u{140}'];

/// Every identifier a module spells (`_identifiers`): the string fields of every node
/// but a constant.
fn identifiers(ast: &Ast, module: NodeId) -> Vec<String> {
    let mut out = Vec::new();
    for node in pyne_ast::walk::walk(ast, module) {
        let kind = ast.kind(node);
        if matches!(kind, Kind::Constant { .. }) {
            continue;
        }
        let class = &pyne_ast::CLASSES[kind.class_index()];
        for index in 0..class.fields.len() {
            match kind.field(index) {
                FieldRef::Str(name) => out.push(name.clone()),
                FieldRef::OptStr(Some(name)) => out.push(name.clone()),
                FieldRef::Strs(names) => out.extend(names.iter().cloned()),
                _ => {}
            }
        }
    }
    out
}

/// Reject Pyne code that spells an identifier in the transformers' namespace
/// (`_reject_reserved_names`): an identifier binds a name holding the separator
/// exactly when it holds one of the characters that normalize to one.
pub fn reject_reserved_names(ast: &Ast, module: NodeId, source: &str) -> Result<(), Error> {
    if source.is_ascii() || !source.contains(RESERVED_SOURCES) {
        return Ok(());
    }
    if let Some(name) =
        identifiers(ast, module).into_iter().find(|name| name.contains(RESERVED_SOURCES))
    {
        return Err(Error::Rejected(format!(
            "'{name}' contains '\u{b7}', which is reserved for PyneCore's internal names in \
             Pyne code \u{2014} rename the identifier"
        )));
    }
    Ok(())
}

/// Reject user code that spells one of the plain names the transform emits
/// (`_reject_pipeline_names`). Every such name is ASCII; an identifier outside ASCII
/// is one the parser would have had to NFKC-normalize first.
pub fn reject_pipeline_names(ast: &Ast, module: NodeId) -> Result<(), Error> {
    for name in identifiers(ast, module) {
        if !name.is_ascii() {
            return Err(Error::Unsupported(format!("the identifier {name:?}")));
        }
        if is_pipeline_name(&name) {
            return Err(Error::Rejected(format!(
                "'{name}' is a name PyneCore's transform generates, reserved in Pyne code \
                 \u{2014} rename the identifier"
            )));
        }
    }
    Ok(())
}

/// Whether a name is one of the names the transform emits (`is_pipeline_name`).
pub fn is_pipeline_name(name: &str) -> bool {
    const WORDS: [&str; 19] = [
        "state",
        "attach_layout",
        "state_method",
        "resolve_slot",
        "bind_any",
        "bind_loop",
        "bind_slot",
        "bind_pinned",
        "loop_state",
        "slot_state",
        "dyn_default",
        "security_contexts",
        "active_security",
        "same_context",
        "sec_read",
        "sec_write",
        "sec_wait",
        "sec_signal",
        "ltf_unzip",
    ];
    let word = |c: char| c.is_alphanumeric() || c == '_';
    let Some(inner) = name.strip_prefix("__").and_then(|rest| rest.strip_suffix("__")) else {
        return false;
    };
    let digits = |rest: &str| !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit());
    if let Some(rest) = inner.strip_prefix("pyne_") {
        return !rest.is_empty() && rest.chars().all(word);
    }
    ["cmp", "bool", "hist_", "sec_main_"]
        .iter()
        .any(|prefix| inner.strip_prefix(prefix).is_some_and(digits))
        || WORDS.contains(&inner)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pyne_heads() {
        assert!(starts_with_pyne(b"\"\"\"\n@pyne\n\"\"\""));
        assert!(starts_with_pyne(b"# c\n\n  '''@pyne lib'''"));
        assert!(starts_with_pyne(b"r\"@pyne\""));
        assert!(starts_with_pyne(b"\"\"\"@pyne\"\"\""));
        assert!(!starts_with_pyne(b"\"\"\"@pyneX\"\"\""));
        assert!(!starts_with_pyne(b"x = 1\n\"\"\"@pyne\"\"\""));
    }

    #[test]
    fn pipeline_names() {
        assert!(is_pipeline_name("__pyne_x__"));
        assert!(is_pipeline_name("__pyne____"));
        assert!(is_pipeline_name("__hist_3__"));
        assert!(is_pipeline_name("__sec_main_1__"));
        assert!(is_pipeline_name("__bool0__"));
        assert!(!is_pipeline_name("__pyne___"));
        assert!(is_pipeline_name("__cmp12__"));
        assert!(is_pipeline_name("__state__"));
        assert!(!is_pipeline_name("__cmp__"));
        assert!(!is_pipeline_name("__pyne__"));
        assert!(!is_pipeline_name("____"));
        assert!(!is_pipeline_name("state"));
    }
}
