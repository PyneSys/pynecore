//! Port of `transformers/dynamic_default.py`: parameter defaults that reference
//! `lib.*` are evaluated per call, and a UDT field's bool na default per construction.
//!
//! A function's dynamic default becomes the `__dyn_default__` sentinel, and a
//! prologue `if <param> is __dyn_default__: <param> = <default>` takes its place.
//! Script entries keep their defaults (the input machinery consumes them at def
//! time). A module-level `@udt` / `@dataclass` field defaulting to the bool na
//! constructor becomes a `dataclasses.field(default_factory=...)`.

use std::collections::HashMap;

use pyne_ast::walk::{
    Transformer, Visited, docstring_end, future_end, generic_visit, walk, walk_statements,
};
use pyne_ast::{Ast, CmpOperator, ExprContext, Kind, NodeId};

use crate::dynamic_default_guard::SENTINEL_NAME;
use crate::{Context, Error};

const SCRIPT_ENTRY_DECORATORS: [&str; 3] = ["indicator", "strategy", "library"];
/// The alias `dataclasses.field` is bound to for the lowered UDT defaults
const FIELD_NAME: &str = "__pyne_field·__";
/// The alias the bool na factory (`pynecore.types.na.new_bool_na`) is bound to
const BOOL_NA_NAME: &str = "__pyne_bool_na·__";
const UDT_DECORATORS: [&str; 2] = ["udt", "dataclass"];
/// The absolute paths that build a bool na
const CONSTRUCTORS: [&str; 2] = ["pynecore.types.na.NA", "pynecore.lib.na"];

fn name(ast: &mut Ast, id: &str, ctx: ExprContext) -> NodeId {
    ast.add(Kind::Name { id: id.to_string(), ctx })
}

/// The last name of a `Name` / `Attribute` chain (the callee of a decorator call too).
fn dotted_tail(ast: &Ast, mut id: NodeId) -> Option<&str> {
    if let Kind::Call { func, .. } = ast.kind(id) {
        id = *func;
    }
    match ast.kind(id) {
        Kind::Attribute { attr, .. } => Some(attr),
        Kind::Name { id, .. } => Some(id),
        _ => None,
    }
}

/// The names of an `a.b.c` chain, `None` for anything else.
fn chain(ast: &Ast, mut id: NodeId) -> Option<Vec<&str>> {
    let mut parts = Vec::new();
    while let Kind::Attribute { value, attr, .. } = ast.kind(id) {
        parts.push(attr.as_str());
        id = *value;
    }
    let Kind::Name { id: root, .. } = ast.kind(id) else {
        return None;
    };
    parts.push(root);
    parts.reverse();
    Some(parts)
}

/// The spellings of the bool na constructor in effect at a top-level statement:
/// every name a `pynecore` import binds, with the absolute path it denotes.
#[derive(Default)]
struct NaBindings {
    paths: HashMap<String, String>,
}

impl NaBindings {
    /// Whether the expression is `lib.na(bool)` or the constructor called as `(bool)`.
    fn is_bool_na(&self, ast: &Ast, expr: NodeId) -> bool {
        let Kind::Call { func, args, keywords } = ast.kind(expr) else {
            return false;
        };
        if !(args.len() == 1
            && keywords.is_empty()
            && matches!(ast.kind(args[0]), Kind::Name { id, .. } if id == "bool"))
        {
            return false;
        }
        let Some(chain) = chain(ast, *func) else {
            return false;
        };
        if chain == ["lib", "na"] {
            return true;
        }
        self.paths.get(chain[0]).is_some_and(|root| {
            let mut path = root.clone();
            for part in &chain[1..] {
                path.push('.');
                path.push_str(part);
            }
            CONSTRUCTORS.contains(&path.as_str())
        })
    }

    /// Update the bindings past a top-level statement.
    fn track(&mut self, ast: &Ast, stmt: NodeId) {
        match ast.kind(stmt) {
            Kind::ImportFrom { module, names, level } => {
                let module = module.as_deref().unwrap_or("");
                let pynecore = module == "pynecore" || module.starts_with("pynecore.");
                for &alias in names {
                    let Kind::Alias { name, asname } = ast.kind(alias) else { unreachable!() };
                    let bound = asname.as_ref().unwrap_or(name);
                    self.paths.remove(bound);
                    if level.unwrap_or(0) == 0 && pynecore {
                        self.paths.insert(bound.clone(), format!("{module}.{name}"));
                    }
                }
            }
            Kind::Import { names } => {
                for &alias in names {
                    let Kind::Alias { name, asname } = ast.kind(alias) else { unreachable!() };
                    let root = name.split('.').next().unwrap_or(name);
                    match asname {
                        Some(asname) => {
                            self.paths.remove(asname);
                            if root == "pynecore" {
                                self.paths.insert(asname.clone(), name.clone());
                            }
                        }
                        None => {
                            self.paths.remove(root);
                            if root == "pynecore" {
                                self.paths.insert(root.to_string(), root.to_string());
                            }
                        }
                    }
                }
            }
            Kind::FunctionDef { name, .. }
            | Kind::AsyncFunctionDef { name, .. }
            | Kind::ClassDef { name, .. } => {
                self.paths.remove(name);
            }
            _ => {
                for node in walk(ast, stmt) {
                    if let Kind::Name { id, ctx: ExprContext::Store } = ast.kind(node) {
                        self.paths.remove(id);
                    }
                }
            }
        }
    }
}

/// Insert a generated import after the docstring and the `__future__` block.
fn insert_import(ast: &mut Ast, module: NodeId, stmt: NodeId) {
    let Kind::Module { body, .. } = ast.kind(module) else { unreachable!() };
    let insert_at = future_end(ast, body);
    let Kind::Module { body, .. } = ast.kind_mut(module) else { unreachable!() };
    body.insert(insert_at, stmt);
}

fn import_from(ast: &mut Ast, module: &str, name: &str, asname: Option<&str>) -> NodeId {
    let alias = ast.add(Kind::Alias { name: name.to_string(), asname: asname.map(str::to_string) });
    ast.add(Kind::ImportFrom {
        module: Some(module.to_string()),
        names: vec![alias],
        level: Some(0),
    })
}

/// The arguments, body and decorators of a (possibly async) function definition.
fn function_parts(kind: &Kind) -> Option<(NodeId, &Vec<NodeId>, &Vec<NodeId>)> {
    match kind {
        Kind::FunctionDef { args, body, decorator_list, .. }
        | Kind::AsyncFunctionDef { args, body, decorator_list, .. } => {
            Some((*args, body, decorator_list))
        }
        _ => None,
    }
}

/// Whether a function is a `@script.indicator/strategy/library` entry.
pub fn is_script_entry(ast: &Ast, function: NodeId) -> bool {
    let Some((_, _, decorators)) = function_parts(ast.kind(function)) else {
        return false;
    };
    decorators.iter().any(|&decorator| {
        let target = match ast.kind(decorator) {
            Kind::Call { func, .. } => *func,
            _ => decorator,
        };
        let Kind::Attribute { value, attr, .. } = ast.kind(target) else {
            return false;
        };
        if !SCRIPT_ENTRY_DECORATORS.contains(&attr.as_str()) {
            return false;
        }
        match ast.kind(*value) {
            Kind::Name { id, .. } => id == "script",
            Kind::Attribute { attr, .. } => attr == "script",
            _ => false,
        }
    })
}

/// Whether a default expression references runtime `lib` state.
fn is_dynamic(ast: &Ast, expr: NodeId) -> bool {
    walk(ast, expr)
        .into_iter()
        .any(|node| matches!(ast.kind(node), Kind::Name { id, .. } if id == "lib"))
}

/// Whether any parameter default references runtime `lib` state.
fn has_dynamic_default(ast: &Ast, args: NodeId) -> bool {
    let Kind::Arguments { defaults, kw_defaults, .. } = ast.kind(args) else { unreachable!() };
    defaults.iter().chain(kw_defaults.iter().flatten()).any(|&default| is_dynamic(ast, default))
}

/// Build `if <param> is __dyn_default__: <param> = <default>`.
fn prologue_if(ast: &mut Ast, param: &str, default: NodeId) -> NodeId {
    let left = name(ast, param, ExprContext::Load);
    let sentinel = name(ast, SENTINEL_NAME, ExprContext::Load);
    let test =
        ast.add(Kind::Compare { left, ops: vec![CmpOperator::Is], comparators: vec![sentinel] });
    let target = name(ast, param, ExprContext::Store);
    let assign =
        ast.add(Kind::Assign { targets: vec![target], value: default, type_comment: None });
    ast.add(Kind::If { test, body: vec![assign], orelse: vec![] })
}

/// Bind a UDT field's bool na default to the factory, built per construction;
/// whether a field was lowered.
fn lower_udt_defaults(ast: &mut Ast, class: NodeId, bindings: &NaBindings) -> bool {
    let Kind::ClassDef { body, decorator_list, .. } = ast.kind(class) else { unreachable!() };
    if !decorator_list.iter().any(|&decorator| {
        dotted_tail(ast, decorator).is_some_and(|tail| UDT_DECORATORS.contains(&tail))
    }) {
        return false;
    }
    let mut lowered = false;
    for stmt in body.clone() {
        let Kind::AnnAssign { value: Some(value), .. } = ast.kind(stmt) else {
            continue;
        };
        if !bindings.is_bool_na(ast, *value) {
            continue;
        }
        lowered = true;
        let factory = name(ast, BOOL_NA_NAME, ExprContext::Load);
        let factory_call = ast.add(Kind::Call { func: factory, args: vec![], keywords: vec![] });
        let no_args = ast.add(Kind::Arguments {
            posonlyargs: vec![],
            args: vec![],
            vararg: None,
            kwonlyargs: vec![],
            kw_defaults: vec![],
            kwarg: None,
            defaults: vec![],
        });
        let lambda = ast.add(Kind::Lambda { args: no_args, body: factory_call });
        let keyword =
            ast.add(Kind::Keyword { arg: Some("default_factory".to_string()), value: lambda });
        let field = name(ast, FIELD_NAME, ExprContext::Load);
        let call = ast.add(Kind::Call { func: field, args: vec![], keywords: vec![keyword] });
        let Kind::AnnAssign { value, .. } = ast.kind_mut(stmt) else { unreachable!() };
        *value = Some(call);
    }
    lowered
}

struct DynamicDefault {
    changed: bool,
    /// The first function the port does not handle
    error: Option<Error>,
}

impl DynamicDefault {
    fn process_function(&mut self, ast: &mut Ast, function: NodeId) -> Result<(), Error> {
        if is_script_entry(ast, function) {
            // Entry defaults are input.*() calls consumed at def time by the
            // input machinery, but inner functions still need the rewrite
            generic_visit(self, ast, function);
            return Ok(());
        }
        let (args, _, _) = function_parts(ast.kind(function)).expect("a function");
        let Kind::Arguments { posonlyargs, args: plain, kwonlyargs, kw_defaults, defaults, .. } =
            ast.kind(args).clone()
        else {
            unreachable!()
        };
        let arg_name = |ast: &Ast, arg: NodeId| -> String {
            let Kind::Arg { arg, .. } = ast.kind(arg) else { unreachable!() };
            arg.clone()
        };
        let positional: Vec<NodeId> = posonlyargs.iter().chain(&plain).copied().collect();
        let offset = positional.len().checked_sub(defaults.len()).ok_or_else(|| {
            Error::Unsupported("more parameter defaults than positional parameters".into())
        })?;

        let mut prologue = Vec::new();
        let mut new_defaults = defaults.clone();
        for (i, &default) in defaults.iter().enumerate() {
            if !is_dynamic(ast, default) {
                continue;
            }
            let param = arg_name(ast, positional[offset + i]);
            prologue.push(prologue_if(ast, &param, default));
            new_defaults[i] = name(ast, SENTINEL_NAME, ExprContext::Load);
        }
        let mut new_kw_defaults = kw_defaults.clone();
        for (i, default) in kw_defaults.iter().enumerate() {
            let Some(default) = *default else {
                continue;
            };
            if !is_dynamic(ast, default) {
                continue;
            }
            let param = arg_name(ast, kwonlyargs[i]);
            prologue.push(prologue_if(ast, &param, default));
            new_kw_defaults[i] = Some(name(ast, SENTINEL_NAME, ExprContext::Load));
        }
        let Kind::Arguments { defaults, kw_defaults, .. } = ast.kind_mut(args) else {
            unreachable!()
        };
        *defaults = new_defaults;
        *kw_defaults = new_kw_defaults;

        if !prologue.is_empty() {
            self.changed = true;
            let (_, body, _) = function_parts(ast.kind(function)).expect("a function");
            let insert_at = docstring_end(ast, body);
            let (Kind::FunctionDef { body, .. } | Kind::AsyncFunctionDef { body, .. }) =
                ast.kind_mut(function)
            else {
                unreachable!()
            };
            body.splice(insert_at..insert_at, prologue);
        }
        generic_visit(self, ast, function);
        Ok(())
    }
}

impl Transformer for DynamicDefault {
    fn visit(&mut self, ast: &mut Ast, id: NodeId) -> Visited {
        match ast.kind(id) {
            // Protocol stubs in compiled libraries carry the same dynamic defaults
            // in their signatures, but they are never called: left alone
            Kind::ClassDef { .. } => {}
            Kind::FunctionDef { .. } | Kind::AsyncFunctionDef { .. } => {
                if self.error.is_none()
                    && let Err(error) = self.process_function(ast, id)
                {
                    self.error = Some(error);
                }
            }
            _ => generic_visit(self, ast, id),
        }
        Visited::Node(id)
    }
}

pub fn run(ast: &mut Ast, module: NodeId, _ctx: &Context<'_>) -> Result<NodeId, Error> {
    let Kind::Module { body, .. } = ast.kind(module) else {
        return Err(Error::Unsupported("DynamicDefault on a non-module".into()));
    };
    let mut field_used = false;
    let mut bindings = NaBindings::default();
    for stmt in body.clone() {
        if matches!(ast.kind(stmt), Kind::ClassDef { .. }) {
            field_used |= lower_udt_defaults(ast, stmt, &bindings);
        }
        bindings.track(ast, stmt);
    }
    let mut pass = DynamicDefault { changed: false, error: None };
    // Only a function with a dynamic default is rewritten, and functions are
    // statements: without one, no expression of the module is visited
    if walk_statements(ast, module).into_iter().any(|stmt| {
        function_parts(ast.kind(stmt)).is_some_and(|(args, _, _)| {
            !is_script_entry(ast, stmt) && has_dynamic_default(ast, args)
        })
    }) {
        generic_visit(&mut pass, ast, module);
    }
    if let Some(error) = pass.error {
        return Err(error);
    }
    if field_used {
        let bool_na = import_from(ast, "pynecore.types.na", "new_bool_na", Some(BOOL_NA_NAME));
        insert_import(ast, module, bool_na);
        let field = import_from(ast, "dataclasses", "field", Some(FIELD_NAME));
        insert_import(ast, module, field);
    }
    if pass.changed {
        let sentinel = import_from(ast, "pynecore.core.instance_state", SENTINEL_NAME, None);
        insert_import(ast, module, sentinel);
    }
    Ok(module)
}
