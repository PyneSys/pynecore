//! Port of `transformers/closure_arguments_transformer.py`: the closure variables of
//! the functions defined in the decorated `main` become their arguments.
//!
//! A function takes its captures first (`def f(length, x)`, `f(length, x)`); a
//! method takes them right after its receiver, publishes their names as
//! `__pyne_closure__`, and a `method_call` passing them carries the names as
//! `pyne_closure_args` (or hands them over as `__pyne_captures__=(...)` when its
//! receiver is unpacked).

use std::collections::{BTreeSet, HashMap, HashSet};

use pyne_ast::locations::{anchor_at, fix_locations};
use pyne_ast::walk::{Transformer, Visited, children, generic_visit, walk, walk_statements};
use pyne_ast::{Ast, Constant, ExprContext, Kind, NodeId, Unsupported, Value};

use crate::{Context, Error};

/// The decorators that make an inner function a Pine method
const METHOD_DECORATORS: [&str; 2] = ["method", "pine_method.method"];

/// The decorators that make a `main` the script entry this pass works in
const MAIN_DECORATORS: [&str; 4] =
    ["lib.script.indicator", "lib.script.strategy", "script.indicator", "script.strategy"];

/// Whether an annotation is `Persistent[T]` or just `Persistent`.
fn is_persistent_annotation(ast: &Ast, annotation: NodeId) -> bool {
    match ast.kind(annotation) {
        Kind::Name { id, .. } => id == "Persistent",
        Kind::Subscript { value, .. } => {
            matches!(ast.kind(*value), Kind::Name { id, .. } if id == "Persistent")
        }
        _ => false,
    }
}

/// Whether an annotation is `Series` or `Series[T]`.
fn is_series_annotation(ast: &Ast, annotation: Option<NodeId>) -> bool {
    let Some(annotation) = annotation else { return false };
    match ast.kind(annotation) {
        Kind::Subscript { value, .. } => {
            matches!(ast.kind(*value), Kind::Name { id, .. } if id == "Series")
        }
        Kind::Name { id, .. } => id == "Series",
        _ => false,
    }
}

/// The inner type `T` of a `Persistent[T]` annotation; `None` for a bare `Persistent`.
fn extract_inner_type(ast: &Ast, annotation: NodeId) -> Option<NodeId> {
    match ast.kind(annotation) {
        Kind::Subscript { slice, .. } => Some(*slice),
        Kind::Name { id, .. } if id == "Persistent" => None,
        _ => Some(annotation),
    }
}

/// The dotted name of a decorator (the callee of a decorator call).
fn decorator_name(ast: &Ast, decorator: NodeId) -> Option<String> {
    match ast.kind(decorator) {
        Kind::Name { id, .. } => Some(id.clone()),
        Kind::Attribute { .. } => {
            let mut parts = Vec::new();
            let mut current = decorator;
            while let Kind::Attribute { value, attr, .. } = ast.kind(current) {
                parts.push(attr.as_str());
                current = *value;
            }
            if let Kind::Name { id, .. } = ast.kind(current) {
                parts.push(id);
            }
            parts.reverse();
            Some(parts.join("."))
        }
        Kind::Call { func, .. } => decorator_name(ast, *func),
        _ => None,
    }
}

fn has_decorator(ast: &Ast, decorators: &[NodeId], names: &[&str]) -> bool {
    decorators.iter().any(|&decorator| {
        decorator_name(ast, decorator).is_some_and(|n| names.contains(&n.as_str()))
    })
}

/// Whether a `def` stands anywhere inside a function.
fn has_nested_function(ast: &Ast, func: NodeId) -> bool {
    walk_statements(ast, func)
        .into_iter()
        .any(|stmt| stmt != func && matches!(ast.kind(stmt), Kind::FunctionDef { .. }))
}

/// The closure variables of every inner function of the decorated `main`.
#[derive(Default)]
struct Collector {
    current_function: Option<String>,
    function_stack: Vec<String>,
    /// The stack joined: the key of the current scope
    scope_key: String,
    scope_variables: HashMap<String, HashSet<String>>,
    scope_uses: HashMap<String, HashSet<String>>,
    /// Scope key -> its closure variables, sorted as the parameters list them
    closure_vars: HashMap<String, BTreeSet<String>>,
    /// `scope.var` -> the annotation it was declared with
    closure_var_types: HashMap<String, NodeId>,
    /// `scope.var` keys declared with a Persistent annotation
    persistent_vars: HashSet<String>,
    in_main_function: bool,
}

impl Collector {
    fn define(&mut self, name: &str) {
        self.scope_variables.entry(self.scope_key.clone()).or_default().insert(name.to_string());
    }

    /// Define a target that is a name, or the names directly in a tuple target.
    fn define_target(&mut self, ast: &Ast, target: NodeId) {
        match ast.kind(target) {
            Kind::Name { id, .. } => self.define(id),
            Kind::Tuple { elts, .. } => {
                for &elt in elts {
                    if let Kind::Name { id, .. } = ast.kind(elt) {
                        self.define(id);
                    }
                }
            }
            _ => {}
        }
    }

    fn visit_function(&mut self, ast: &Ast, id: NodeId) {
        let Kind::FunctionDef { name, args, decorator_list, .. } = ast.kind(id) else {
            unreachable!()
        };
        let is_main = name == "main" && has_decorator(ast, decorator_list, &MAIN_DECORATORS);
        let old_function = self.current_function.clone();
        let old_in_main = self.in_main_function;
        if is_main {
            self.in_main_function = true;
        }
        self.current_function = Some(name.clone());
        self.function_stack.push(name.clone());
        self.scope_key = self.function_stack.join(".");
        let scope_key = self.scope_key.clone();
        self.scope_variables.insert(scope_key.clone(), HashSet::new());
        self.scope_uses.insert(scope_key.clone(), HashSet::new());

        // An annotated parameter carries the same type information as an annotated
        // assignment does
        let Kind::Arguments { args: positional, .. } = ast.kind(*args) else { unreachable!() };
        for &param in positional {
            let Kind::Arg { arg, annotation, .. } = ast.kind(param) else { unreachable!() };
            self.scope_variables.get_mut(&scope_key).unwrap().insert(arg.clone());
            let Some(annotation) = *annotation else { continue };
            let arg_key = format!("{scope_key}.{arg}");
            self.closure_var_types.insert(arg_key.clone(), annotation);
            if is_persistent_annotation(ast, annotation) {
                self.persistent_vars.insert(arg_key);
            }
        }

        self.generic_visit(ast, id);

        if self.in_main_function
            && self.function_stack.len() > 1
            && self.function_stack[0] == "main"
        {
            let scope_key = self.scope_key.clone();
            let parent_scope = self.function_stack[..self.function_stack.len() - 1].join(".");
            let empty = HashSet::new();
            let own = self.scope_variables.get(&scope_key).unwrap_or(&empty);
            let parent = self.scope_variables.get(&parent_scope).unwrap_or(&empty);
            let found: BTreeSet<String> = self
                .scope_uses
                .get(&scope_key)
                .unwrap_or(&empty)
                .iter()
                .filter(|var| !own.contains(*var) && parent.contains(*var))
                .cloned()
                .collect();
            if !found.is_empty() {
                self.closure_vars.insert(scope_key, found);
            }
        }

        self.current_function = old_function;
        self.in_main_function = old_in_main;
        self.function_stack.pop();
        self.scope_key = self.function_stack.join(".");
    }

    fn visit(&mut self, ast: &Ast, id: NodeId) {
        let in_function = self.current_function.is_some();
        match ast.kind(id) {
            Kind::FunctionDef { .. } => return self.visit_function(ast, id),
            Kind::Name { id: name, ctx } if in_function => match ctx {
                ExprContext::Store => self.define(name),
                ExprContext::Load => {
                    let key = self.scope_key.clone();
                    self.scope_uses.entry(key).or_default().insert(name.clone());
                }
                ExprContext::Del => {}
            },
            Kind::Assign { targets, .. } if in_function => {
                for &target in targets {
                    self.define_target(ast, target);
                }
            }
            Kind::AnnAssign { target, annotation, .. } if in_function => {
                if let Kind::Name { id: name, .. } = ast.kind(*target) {
                    self.define(name);
                    let var_key = format!("{}.{name}", self.scope_key);
                    self.closure_var_types.insert(var_key.clone(), *annotation);
                    if is_persistent_annotation(ast, *annotation) {
                        self.persistent_vars.insert(var_key);
                    }
                }
            }
            Kind::For { target, .. } if in_function => self.define_target(ast, *target),
            Kind::ExceptHandler { name: Some(name), .. } if in_function => self.define(name),
            _ => {}
        }
        self.generic_visit(ast, id);
    }

    fn generic_visit(&mut self, ast: &Ast, id: NodeId) {
        for child in children(ast, id) {
            self.visit(ast, child);
        }
    }
}

struct ClosureArguments {
    in_main_function: bool,
    current_function: Option<String>,
    /// The inner functions of `main` met so far, by name
    inner_functions: HashSet<String>,
    closure_vars: HashMap<String, BTreeSet<String>>,
    closure_var_types: HashMap<String, NodeId>,
    /// Inner functions called as methods: their closure variables follow the receiver
    method_functions: HashSet<String>,
    error: Option<Unsupported>,
}

/// The key of an inner function of `main`.
fn function_key(name: &str) -> String {
    format!("main.{name}")
}

fn load_names(ast: &mut Ast, names: &[String]) -> Vec<NodeId> {
    names
        .iter()
        .map(|var| ast.add(Kind::Name { id: var.clone(), ctx: ExprContext::Load }))
        .collect()
}

impl ClosureArguments {
    /// The closure variables of an inner function, when it has any.
    fn closure_vars_of(&self, name: &str) -> Option<Vec<String>> {
        self.closure_vars
            .get(&function_key(name))
            .filter(|vars| !vars.is_empty())
            .map(|vars| vars.iter().cloned().collect())
    }

    fn visit_function(&mut self, ast: &mut Ast, id: NodeId) -> Visited {
        let Kind::FunctionDef { name, args, decorator_list, .. } = ast.kind(id) else {
            unreachable!()
        };
        let (name, args) = (name.clone(), *args);
        let is_main_decorated =
            name == "main" && has_decorator(ast, decorator_list, &MAIN_DECORATORS);
        let is_method = has_decorator(ast, decorator_list, &METHOD_DECORATORS);

        let old_in_main = self.in_main_function;
        let old_current = self.current_function.take();
        if is_main_decorated {
            self.in_main_function = true;
        }
        self.current_function = Some(name.clone());

        // The ``__pyne_closure__`` publication of a method's captures
        let mut marker = None;

        if self.in_main_function && old_current.as_deref() == Some("main") && name != "main" {
            if is_method {
                self.method_functions.insert(name.clone());
            }
            if let Some(closure_vars) = self.closure_vars_of(&name) {
                let mut new_args = Vec::with_capacity(closure_vars.len());
                for var in &closure_vars {
                    let mut annotation = self.closure_var_types.get(&function_key(var)).copied();
                    if let Some(found) = annotation
                        && is_persistent_annotation(ast, found)
                    {
                        annotation = extract_inner_type(ast, found);
                    }
                    new_args.push(ast.add(Kind::Arg {
                        arg: var.clone(),
                        annotation,
                        type_comment: None,
                    }));
                }
                let Kind::Arguments { args: positional, .. } = ast.kind(args) else {
                    unreachable!()
                };
                if self.method_functions.contains(&name) && !positional.is_empty() {
                    let Kind::Arguments { args: positional, .. } = ast.kind_mut(args) else {
                        unreachable!()
                    };
                    positional.splice(1..1, new_args);
                    let base = ast.add(Kind::Name { id: name.clone(), ctx: ExprContext::Load });
                    let target = ast.add(Kind::Attribute {
                        value: base,
                        attr: "__pyne_closure__".into(),
                        ctx: ExprContext::Store,
                    });
                    let elts = closure_vars
                        .iter()
                        .map(|var| {
                            ast.add(Kind::Constant {
                                value: Constant::Str(var.clone()),
                                kind: None,
                            })
                        })
                        .collect();
                    let value = ast.add(Kind::Tuple { elts, ctx: ExprContext::Load });
                    let assign =
                        ast.add(Kind::Assign { targets: vec![target], value, type_comment: None });
                    if let Err(err) = anchor_at(ast, assign, id)
                        .and_then(|assign| fix_locations(ast, assign, 1, 0))
                    {
                        self.error.get_or_insert(err);
                    }
                    marker = Some(assign);
                } else {
                    let Kind::Arguments { args: positional, .. } = ast.kind_mut(args) else {
                        unreachable!()
                    };
                    positional.splice(0..0, new_args);
                }
            }
            self.inner_functions.insert(name);
        }

        generic_visit(self, ast, id);

        self.in_main_function = old_in_main;
        self.current_function = old_current;

        match marker {
            Some(marker) => Visited::Many(vec![id, marker]),
            None => Visited::Node(id),
        }
    }

    fn visit_call(&mut self, ast: &mut Ast, id: NodeId) {
        generic_visit(self, ast, id);
        if !self.in_main_function {
            return;
        }
        let Kind::Call { func, args, .. } = ast.kind(id) else { unreachable!() };
        let Kind::Name { id: func_name, .. } = ast.kind(*func) else { return };

        if self.inner_functions.contains(func_name) {
            let Some(closure_vars) = self.closure_vars_of(func_name) else { return };
            if self.method_functions.contains(func_name) {
                // After the receiver, where the method declares them
                thread_after_receiver(ast, id, &closure_vars);
            } else {
                let names = load_names(ast, &closure_vars);
                let Kind::Call { args, .. } = ast.kind_mut(id) else { unreachable!() };
                args.splice(0..0, names);
            }
        } else if func_name == "method_call" && args.len() >= 2 {
            // Only the reference form ``method_call(method_function, this, ...)`` names
            // a function of this module; the string form resolves to a library method
            let Kind::Name { id: method_name, .. } = ast.kind(args[0]) else { return };
            if !self.inner_functions.contains(method_name) {
                return;
            }
            let Some(closure_vars) = self.closure_vars_of(method_name) else { return };
            let unpacked = matches!(ast.kind(args[1]), Kind::Starred { .. });
            let names = load_names(ast, &closure_vars);
            if unpacked {
                let tuple = ast.add(Kind::Tuple { elts: names, ctx: ExprContext::Load });
                let keyword =
                    ast.add(Kind::Keyword { arg: Some("__pyne_captures__".into()), value: tuple });
                let Kind::Call { keywords, .. } = ast.kind_mut(id) else { unreachable!() };
                keywords.push(keyword);
            } else {
                let Kind::Call { args, .. } = ast.kind_mut(id) else { unreachable!() };
                args.splice(2..2, names);
                let names = closure_vars.into_iter().map(Value::Str).collect();
                ast.set_extra(id, "pyne_closure_args", Value::Tuple(names));
            }
        }
    }
}

/// Pass a method's closure variables right after its receiver argument; a receiver
/// passed by keyword or unpacked leaves no position to follow, so they go by keyword.
fn thread_after_receiver(ast: &mut Ast, call: NodeId, closure_vars: &[String]) {
    let Kind::Call { args, .. } = ast.kind(call) else { unreachable!() };
    let positional =
        args.first().is_some_and(|&first| !matches!(ast.kind(first), Kind::Starred { .. }));
    if positional {
        let names = load_names(ast, closure_vars);
        let Kind::Call { args, .. } = ast.kind_mut(call) else { unreachable!() };
        args.splice(1..1, names);
    } else {
        let keywords: Vec<NodeId> = closure_vars
            .iter()
            .map(|var| {
                let value = ast.add(Kind::Name { id: var.clone(), ctx: ExprContext::Load });
                ast.add(Kind::Keyword { arg: Some(var.clone()), value })
            })
            .collect();
        let Kind::Call { keywords: slot, .. } = ast.kind_mut(call) else { unreachable!() };
        slot.extend(keywords);
    }
}

impl Transformer for ClosureArguments {
    fn visit(&mut self, ast: &mut Ast, id: NodeId) -> Visited {
        match ast.kind(id) {
            Kind::FunctionDef { .. } => return self.visit_function(ast, id),
            Kind::Call { .. } => self.visit_call(ast, id),
            _ => generic_visit(self, ast, id),
        }
        Visited::Node(id)
    }
}

pub fn run(ast: &mut Ast, module: NodeId, _ctx: &Context<'_>) -> Result<NodeId, Error> {
    // Only a function defined inside the decorated main can have closure variables
    // threaded: without one, neither pass would change anything
    let applies = walk_statements(ast, module).into_iter().any(|stmt| {
        matches!(ast.kind(stmt), Kind::FunctionDef { name, decorator_list, .. }
            if name == "main"
                && has_decorator(ast, decorator_list, &MAIN_DECORATORS)
                && has_nested_function(ast, stmt))
    });
    if !applies {
        return Ok(module);
    }

    let mut collector = Collector::default();
    collector.visit(ast, module);
    let mut method_functions = HashSet::new();
    for call in walk(ast, module) {
        if let Kind::Call { func, args, .. } = ast.kind(call)
            && matches!(ast.kind(*func), Kind::Name { id, .. } if id == "method_call")
            && let Some(&handed) = args.first()
            && let Kind::Name { id, .. } = ast.kind(handed)
        {
            method_functions.insert(id.clone());
        }
    }

    // A plain-Series free variable resolves through the scope chain, not through
    // argument passing; a PersistentSeries stays threaded
    let mut closure_vars = collector.closure_vars;
    for (scope_key, vars) in &mut closure_vars {
        let parent_scope = scope_key.rsplit_once('.').map_or(scope_key.as_str(), |(head, _)| head);
        vars.retain(|var| {
            let key = format!("{parent_scope}.{var}");
            collector.persistent_vars.contains(&key)
                || !is_series_annotation(ast, collector.closure_var_types.get(&key).copied())
        });
    }

    let mut transform = ClosureArguments {
        in_main_function: false,
        current_function: None,
        inner_functions: HashSet::new(),
        closure_vars,
        closure_var_types: collector.closure_var_types,
        method_functions,
        error: None,
    };
    generic_visit(&mut transform, ast, module);
    match transform.error {
        Some(err) => Err(err.into()),
        None => Ok(module),
    }
}
