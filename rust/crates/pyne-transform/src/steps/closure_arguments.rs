//! Port of `transformers/closure_arguments_transformer.py`: the closure variables of
//! the functions defined in the decorated `main` become their arguments.
//!
//! A function takes its captures first (`def f(length, x)`, `f(length, x)`); a
//! method takes them right after its receiver, publishes their names as
//! `__pyne_closure__`, and a `method_call` passing them carries the names as
//! `pyne_closure_args` (or hands them over as `__pyne_captures__=(...)` when its
//! receiver is unpacked). A direct call of such a method with an unpacked receiver
//! (`m(*args)`) is rejected.
//!
//! A call inside another inner function passes a capture by the CALLER's name for it,
//! so a caller binding that name itself (a parameter or a local) takes `main`'s value
//! as a capture of its own under `__capture·<name>__` and passes that on.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use pyne_ast::locations::{anchor_at, fix_locations};
use pyne_ast::walk::{
    Transformer, Visited, children, generic_visit, visit_field, walk, walk_statements,
};
use pyne_ast::{Ast, Constant, ExprContext, Kind, NodeId, Value};

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

/// The parameter that carries `main`'s `var` into a function that binds `var` itself.
fn capture_param(var: &str) -> String {
    format!("__capture·{var}__")
}

/// The name of the function a call reaches by name: its callee, or the method handed
/// to `method_call` by reference.
fn called_function(ast: &Ast, call: NodeId) -> Option<String> {
    let Kind::Call { func, args, .. } = ast.kind(call) else { return None };
    let Kind::Name { id, .. } = ast.kind(*func) else { return None };
    if id != "method_call" {
        return Some(id.clone());
    }
    match ast.kind(*args.first()?) {
        Kind::Name { id, .. } => Some(id.clone()),
        _ => None,
    }
}

/// The captures of every inner function of `main`: `main`'s variable -> the parameter
/// that carries it. A function takes its free variables under their own names; one
/// that calls a function taking `var` while binding `var` itself takes it as a capture
/// of its own under `__capture·var__`, which in turn reaches every function calling it.
fn thread_captures(
    closure_vars: &HashMap<String, BTreeSet<String>>,
    scope_variables: &HashMap<String, HashSet<String>>,
    scope_calls: &HashMap<String, HashSet<String>>,
) -> HashMap<String, BTreeMap<String, String>> {
    let mut captures: HashMap<String, BTreeMap<String, String>> = closure_vars
        .iter()
        .filter(|(_, vars)| !vars.is_empty())
        .map(|(key, vars)| (key.clone(), vars.iter().map(|v| (v.clone(), v.clone())).collect()))
        .collect();
    let empty = HashSet::new();
    let mut changed = true;
    while changed {
        changed = false;
        for (caller, callees) in scope_calls {
            if !caller.strip_prefix("main.").is_some_and(|rest| !rest.contains('.')) {
                continue;
            }
            let bound = scope_variables.get(caller).unwrap_or(&empty);
            let needed: Vec<String> = callees
                .iter()
                .filter_map(|callee| captures.get(&function_key(callee)))
                .flat_map(|callee| callee.keys())
                .filter(|var| bound.contains(*var))
                .cloned()
                .collect();
            let own = captures.entry(caller.clone()).or_default();
            for var in needed {
                if let std::collections::btree_map::Entry::Vacant(slot) = own.entry(var) {
                    let param = capture_param(slot.key());
                    slot.insert(param);
                    changed = true;
                }
            }
        }
    }
    captures.retain(|_, own| !own.is_empty());
    captures
}

/// Every parameter of a function, of every kind.
fn parameters(ast: &Ast, arguments: NodeId) -> Vec<NodeId> {
    let Kind::Arguments { posonlyargs, args, vararg, kwonlyargs, kwarg, .. } = ast.kind(arguments)
    else {
        unreachable!()
    };
    let mut params: Vec<NodeId> =
        posonlyargs.iter().chain(args).chain(kwonlyargs).copied().collect();
    params.extend(vararg.iter().chain(kwarg));
    params
}

/// Insert parameters at a position of the positional parameters: among the
/// positional-only ones when the position falls before their end.
fn insert_positional(ast: &mut Ast, arguments: NodeId, index: usize, new_args: Vec<NodeId>) {
    let Kind::Arguments { posonlyargs, args, .. } = ast.kind_mut(arguments) else { unreachable!() };
    if index < posonlyargs.len() {
        posonlyargs.splice(index..index, new_args);
    } else {
        let index = index - posonlyargs.len();
        args.splice(index..index, new_args);
    }
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
    /// Scope key -> the functions it calls by name
    scope_calls: HashMap<String, HashSet<String>>,
    /// Scope key -> its closure variables, sorted as the parameters list them
    closure_vars: HashMap<String, BTreeSet<String>>,
    /// `scope.var` -> the annotation it was declared with
    closure_var_types: HashMap<String, NodeId>,
    /// `scope.var` keys declared with a Persistent annotation
    persistent_vars: HashSet<String>,
    /// The inner functions of `main`, by name
    inner_functions: HashSet<String>,
    /// The inner functions of `main` declared methods
    method_definitions: HashSet<String>,
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
        let Kind::FunctionDef { name, args, body, decorator_list, returns, .. } = ast.kind(id)
        else {
            unreachable!()
        };
        let is_main = name == "main" && has_decorator(ast, decorator_list, &MAIN_DECORATORS);
        let old_function = self.current_function.clone();
        let old_in_main = self.in_main_function;

        if self.in_main_function && old_function.as_deref() == Some("main") && name != "main" {
            self.inner_functions.insert(name.clone());
            if has_decorator(ast, decorator_list, &METHOD_DECORATORS) {
                self.method_definitions.insert(name.clone());
            }
        }

        // The defaults, decorators and annotations belong to the enclosing scope: they
        // run there when the `def` executes
        for &decorator in decorator_list {
            self.visit(ast, decorator);
        }
        self.visit(ast, *args);
        if let Some(returns) = *returns {
            self.visit(ast, returns);
        }

        if is_main {
            self.in_main_function = true;
        }
        self.current_function = Some(name.clone());
        self.function_stack.push(name.clone());
        self.scope_key = self.function_stack.join(".");
        let scope_key = self.scope_key.clone();
        self.scope_variables.insert(scope_key.clone(), HashSet::new());
        self.scope_uses.insert(scope_key.clone(), HashSet::new());
        self.scope_calls.insert(scope_key.clone(), HashSet::new());

        // An annotated parameter carries the same type information as an annotated
        // assignment does
        for param in parameters(ast, *args) {
            let Kind::Arg { arg, annotation, .. } = ast.kind(param) else { unreachable!() };
            self.scope_variables.get_mut(&scope_key).unwrap().insert(arg.clone());
            let Some(annotation) = *annotation else { continue };
            let arg_key = format!("{scope_key}.{arg}");
            self.closure_var_types.insert(arg_key.clone(), annotation);
            if is_persistent_annotation(ast, annotation) {
                self.persistent_vars.insert(arg_key);
            }
        }

        for &stmt in body {
            self.visit(ast, stmt);
        }

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
            Kind::Call { .. } if in_function => {
                if let Some(callee) = called_function(ast, id) {
                    let key = self.scope_key.clone();
                    self.scope_calls.entry(key).or_default().insert(callee);
                }
            }
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
    /// The inner function of `main` the visit is in
    caller: Option<String>,
    /// The inner functions of `main`, by name
    inner_functions: HashSet<String>,
    /// Scope key -> its captures: `main`'s variable -> its parameter
    captures: HashMap<String, BTreeMap<String, String>>,
    closure_var_types: HashMap<String, NodeId>,
    /// Inner functions called as methods: their closure variables follow the receiver
    method_functions: HashSet<String>,
    error: Option<Error>,
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
    /// The captures of an inner function, when it has any.
    fn captures_of(&self, name: &str) -> Option<BTreeMap<String, String>> {
        self.captures.get(&function_key(name)).filter(|own| !own.is_empty()).cloned()
    }

    /// The names that pass `main`'s variables to a function taking them as captures,
    /// in its parameter order: `main`'s own variable in `main`, the caller's capture
    /// parameter for it inside an inner function (`main`'s value even where the caller
    /// binds the name itself).
    fn passed(&self, captures: &BTreeMap<String, String>) -> Vec<String> {
        let own =
            self.caller.as_deref().and_then(|caller| self.captures.get(&function_key(caller)));
        captures.keys().map(|var| own.and_then(|own| own.get(var)).unwrap_or(var).clone()).collect()
    }

    fn visit_function(&mut self, ast: &mut Ast, id: NodeId) -> Visited {
        let Kind::FunctionDef { name, args, decorator_list, .. } = ast.kind(id) else {
            unreachable!()
        };
        let (name, args) = (name.clone(), *args);
        let is_main_decorated =
            name == "main" && has_decorator(ast, decorator_list, &MAIN_DECORATORS);

        let old_in_main = self.in_main_function;
        let old_current = self.current_function.clone();

        // The ``__pyne_closure__`` publication of a method's captures
        let mut marker = None;

        let old_caller = self.caller.clone();
        let is_inner =
            self.in_main_function && old_current.as_deref() == Some("main") && name != "main";
        if is_inner && let Some(captures) = self.captures_of(&name) {
            let mut new_args = Vec::with_capacity(captures.len());
            for (var, param) in &captures {
                let mut annotation = self.closure_var_types.get(&function_key(var)).copied();
                if let Some(found) = annotation
                    && is_persistent_annotation(ast, found)
                {
                    annotation = extract_inner_type(ast, found);
                }
                new_args.push(ast.add(Kind::Arg {
                    arg: param.clone(),
                    annotation,
                    type_comment: None,
                }));
            }
            let Kind::Arguments { posonlyargs, args: positional, .. } = ast.kind(args) else {
                unreachable!()
            };
            if self.method_functions.contains(&name)
                && !(posonlyargs.is_empty() && positional.is_empty())
            {
                insert_positional(ast, args, 1, new_args);
                let base = ast.add(Kind::Name { id: name.clone(), ctx: ExprContext::Load });
                let target = ast.add(Kind::Attribute {
                    value: base,
                    attr: "__pyne_closure__".into(),
                    ctx: ExprContext::Store,
                });
                let elts = captures
                    .values()
                    .map(|param| {
                        ast.add(Kind::Constant { value: Constant::Str(param.clone()), kind: None })
                    })
                    .collect();
                let value = ast.add(Kind::Tuple { elts, ctx: ExprContext::Load });
                let assign =
                    ast.add(Kind::Assign { targets: vec![target], value, type_comment: None });
                if let Err(err) =
                    anchor_at(ast, assign, id).and_then(|assign| fix_locations(ast, assign, 1, 0))
                {
                    self.error.get_or_insert(err.into());
                }
                marker = Some(assign);
            } else {
                insert_positional(ast, args, 0, new_args);
            }
        }

        // The defaults, decorators and annotations run in the enclosing scope when the
        // `def` executes; only the body runs as the function itself
        let fields = ast.kind(id).class().fields;
        let body = fields.iter().position(|field| field.name == "body").unwrap();
        for index in (0..fields.len()).filter(|&index| index != body) {
            visit_field(self, ast, id, index);
        }
        if is_main_decorated {
            self.in_main_function = true;
        }
        self.current_function = Some(name.clone());
        if is_inner {
            self.caller = Some(name);
        }
        visit_field(self, ast, id, body);

        self.in_main_function = old_in_main;
        self.current_function = old_current;
        self.caller = old_caller;

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
            let Some(captures) = self.captures_of(func_name) else { return };
            let passed = self.passed(&captures);
            if self.method_functions.contains(func_name) {
                if args
                    .first()
                    .is_some_and(|&first| matches!(ast.kind(first), Kind::Starred { .. }))
                {
                    self.error.get_or_insert(Error::Rejected(format!(
                        "{func_name}() uses variables of main(), so its receiver cannot come \
                         from an unpacked argument; pass the receiver explicitly: \
                         {func_name}(receiver, *args)"
                    )));
                    return;
                }
                // After the receiver, where the method declares them
                thread_after_receiver(ast, id, &captures, &passed);
            } else {
                let names = load_names(ast, &passed);
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
            let Some(captures) = self.captures_of(method_name) else { return };
            let unpacked = matches!(ast.kind(args[1]), Kind::Starred { .. });
            let names = load_names(ast, &self.passed(&captures));
            if unpacked {
                let tuple = ast.add(Kind::Tuple { elts: names, ctx: ExprContext::Load });
                let keyword =
                    ast.add(Kind::Keyword { arg: Some("__pyne_captures__".into()), value: tuple });
                let Kind::Call { keywords, .. } = ast.kind_mut(id) else { unreachable!() };
                keywords.push(keyword);
            } else {
                let Kind::Call { args, .. } = ast.kind_mut(id) else { unreachable!() };
                args.splice(2..2, names);
                let names = captures.into_values().map(Value::Str).collect();
                ast.set_extra(id, "pyne_closure_args", Value::Tuple(names));
            }
        }
    }
}

/// Pass a method's closure variables (`passed`, in parameter order) right after its
/// receiver argument; a receiver passed by keyword leaves no position to follow, so
/// they go by keyword, under the parameter names.
fn thread_after_receiver(
    ast: &mut Ast,
    call: NodeId,
    captures: &BTreeMap<String, String>,
    passed: &[String],
) {
    let Kind::Call { args, .. } = ast.kind(call) else { unreachable!() };
    if !args.is_empty() {
        let names = load_names(ast, passed);
        let Kind::Call { args, .. } = ast.kind_mut(call) else { unreachable!() };
        args.splice(1..1, names);
    } else {
        let keywords: Vec<NodeId> = captures
            .values()
            .zip(passed)
            .map(|(param, var)| {
                let value = ast.add(Kind::Name { id: var.clone(), ctx: ExprContext::Load });
                ast.add(Kind::Keyword { arg: Some(param.clone()), value })
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
    let mut method_functions = collector.method_definitions;
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

    let captures =
        thread_captures(&closure_vars, &collector.scope_variables, &collector.scope_calls);
    let mut transform = ClosureArguments {
        in_main_function: false,
        current_function: None,
        caller: None,
        inner_functions: collector.inner_functions,
        captures,
        closure_var_types: collector.closure_var_types,
        method_functions,
        error: None,
    };
    generic_visit(&mut transform, ast, module);
    match transform.error {
        Some(err) => Err(err),
        None => Ok(module),
    }
}
