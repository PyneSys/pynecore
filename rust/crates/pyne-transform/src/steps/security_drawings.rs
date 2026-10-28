//! Port of `transformers/security_drawings.py`: no drawing may be created in the
//! expression of a `request.security` / `request.security_lower_tf` call.
//!
//! The step changes nothing; it follows each request expression through the
//! names, functions and arguments it depends on and rejects a module where one of
//! them creates a drawing. A rejected module answers [`Error::Rejected`], and the
//! Python host runs the Python step, which raises the source-located `SyntaxError`
//! the script author sees.

use std::collections::{HashMap, HashSet};

use pyne_ast::walk::{children, walk};
use pyne_ast::{Ast, Constant, ExprContext, Kind, NodeId};

use crate::{Context, Error};

/// The drawing constructors, by dotted path.
const DRAWING_CALLS: [&str; 9] = [
    "lib.label.new",
    "lib.label.copy",
    "lib.line.new",
    "lib.line.copy",
    "lib.box.new",
    "lib.box.copy",
    "lib.table.new",
    "lib.polyline.new",
    "lib.linefill.new",
];

const REQUESTS: [&str; 2] = ["lib.request.security", "lib.request.security_lower_tf"];

/// The last segment of every name in [`REQUESTS`]
const REQUEST_ATTRS: [&str; 2] = ["security", "security_lower_tf"];

struct Scope {
    parent: Option<usize>,
    bindings: HashMap<String, Vec<NodeId>>,
}

/// A node the Python step looks up in its owner map without having collected it:
/// the Python step fails there, so the port declines.
fn uncollected(node: NodeId) -> Error {
    Error::Unsupported(format!("security drawings: node {} has no scope", node.0))
}

struct Collector {
    scopes: Vec<Scope>,
    owners: HashMap<NodeId, usize>,
    /// A function or lambda -> the scope of its body
    bodies: HashMap<NodeId, usize>,
    /// Every call, in the order its subtree is finished
    requests: Vec<NodeId>,
    annotations: HashMap<(usize, String), NodeId>,
    /// The names `method_call` is imported under
    method_names: HashSet<String>,
}

impl Collector {
    fn new_scope(&mut self, parent: Option<usize>) -> usize {
        self.scopes.push(Scope { parent, bindings: HashMap::new() });
        self.scopes.len() - 1
    }

    fn bindings_of(&mut self, scope: usize, name: &str) -> &mut Vec<NodeId> {
        self.scopes[scope].bindings.entry(name.to_string()).or_default()
    }

    /// The lexical scope that binds a name.
    fn resolve(&self, mut scope: usize, name: &str) -> Option<usize> {
        loop {
            if self.scopes[scope].bindings.contains_key(name) {
                return Some(scope);
            }
            scope = self.scopes[scope].parent?;
        }
    }

    fn owner(&self, node: NodeId) -> Result<usize, Error> {
        self.owners.get(&node).copied().ok_or_else(|| uncollected(node))
    }

    fn collect(&mut self, ast: &Ast, current: NodeId, scope: usize) {
        self.owners.insert(current, scope);
        let kind = ast.kind(current);
        if let Kind::ImportFrom { module: Some(module), names, .. } = kind
            && module == "pynecore.core.pine_method"
        {
            for &alias in names {
                let Kind::Alias { name, asname } = ast.kind(alias) else { unreachable!() };
                if name == "method_call" {
                    let bound = match asname {
                        Some(asname) if !asname.is_empty() => asname,
                        _ => name,
                    };
                    self.method_names.insert(bound.clone());
                }
            }
        }
        match kind {
            Kind::FunctionDef { name, args, body, .. }
            | Kind::AsyncFunctionDef { name, args, body, .. } => {
                self.bindings_of(scope, name).push(current);
                self.collect_function(ast, current, scope, *args, body);
                return;
            }
            Kind::Lambda { args, body } => {
                self.collect_function(ast, current, scope, *args, std::slice::from_ref(body));
                return;
            }
            Kind::ClassDef { name, .. } => {
                self.bindings_of(scope, name).push(current);
                let child = self.new_scope(Some(scope));
                for statement in children(ast, current) {
                    self.collect(ast, statement, child);
                }
                return;
            }
            _ => {}
        }
        let assignment = match kind {
            Kind::Assign { targets, value, .. } => Some((targets.as_slice(), Some(*value), None)),
            Kind::AnnAssign { target, annotation, value, .. } => {
                Some((std::slice::from_ref(target), *value, Some(*annotation)))
            }
            Kind::NamedExpr { target, value } | Kind::AugAssign { target, value, .. } => {
                Some((std::slice::from_ref(target), Some(*value), None))
            }
            _ => None,
        };
        if let Some((targets, value, annotation)) = assignment {
            for &target in targets {
                for name in walk(ast, target) {
                    let Kind::Name { id, ctx: ExprContext::Store } = ast.kind(name) else {
                        continue;
                    };
                    let values = self.bindings_of(scope, id);
                    if let Some(value) = value {
                        values.push(value);
                    }
                    if let Some(annotation) = annotation {
                        self.annotations.insert((scope, id.clone()), annotation);
                    }
                }
            }
        }
        for item in children(ast, current) {
            self.collect(ast, item, scope);
        }
        if matches!(kind, Kind::Call { .. }) {
            self.requests.push(current);
        }
    }

    fn collect_function(
        &mut self,
        ast: &Ast,
        current: NodeId,
        scope: usize,
        args: NodeId,
        body: &[NodeId],
    ) {
        let child = self.new_scope(Some(scope));
        self.bodies.insert(current, child);
        let Kind::Arguments {
            posonlyargs,
            args: positional,
            vararg,
            kwonlyargs,
            kw_defaults,
            kwarg,
            defaults,
        } = ast.kind(args)
        else {
            unreachable!()
        };
        for &param in
            posonlyargs.iter().chain(positional).chain(kwonlyargs).chain(vararg).chain(kwarg)
        {
            let Kind::Arg { arg, annotation, .. } = ast.kind(param) else { unreachable!() };
            self.scopes[child].bindings.insert(arg.clone(), Vec::new());
            if let Some(annotation) = annotation {
                self.annotations.insert((child, arg.clone()), *annotation);
            }
        }
        for &default in defaults.iter().chain(kw_defaults.iter().flatten()) {
            self.collect(ast, default, scope);
        }
        for &statement in body {
            self.collect(ast, statement, child);
        }
    }

    /// The dotted path a reference resolves to through single static bindings
    /// ('' when none).
    fn path(
        &self,
        ast: &Ast,
        value: NodeId,
        seen: &mut Vec<(usize, String)>,
    ) -> Result<String, Error> {
        match ast.kind(value) {
            Kind::Attribute { value: base, attr, .. } => {
                Ok(format!("{}.{attr}", self.path(ast, *base, seen)?))
            }
            Kind::Name { id, .. } => {
                let Some(scope) = self.resolve(self.owner(value)?, id) else {
                    return Ok(id.clone());
                };
                let key = (scope, id.clone());
                if seen.contains(&key) {
                    return Ok(String::new());
                }
                let bindings = &self.scopes[scope].bindings[id];
                if bindings.len() == 1
                    && matches!(ast.kind(bindings[0]), Kind::Name { .. } | Kind::Attribute { .. })
                {
                    let depth = seen.len();
                    seen.push(key);
                    let result = self.path(ast, bindings[0], seen);
                    seen.truncate(depth);
                    return result;
                }
                Ok(String::new())
            }
            _ => Ok(String::new()),
        }
    }

    fn call_path(&self, ast: &Ast, call: NodeId) -> Result<String, Error> {
        let Kind::Call { func, .. } = ast.kind(call) else { unreachable!() };
        self.path(ast, *func, &mut Vec::new())
    }

    /// The drawing a call creates (`label.new`, ...): a drawing constructor, or
    /// `method_call` copying a receiver annotated as a drawing type.
    fn drawing_name(&self, ast: &Ast, call: NodeId) -> Result<Option<String>, Error> {
        let name = self.call_path(ast, call)?;
        if DRAWING_CALLS.contains(&name.as_str()) {
            return Ok(Some(name[4..].to_string()));
        }
        let Kind::Call { func, args, .. } = ast.kind(call) else { unreachable!() };
        let Kind::Name { id: func_name, .. } = ast.kind(*func) else { return Ok(None) };
        if !self.method_names.contains(func_name) || args.len() < 2 {
            return Ok(None);
        }
        let Kind::Constant { value: Constant::Str(method), .. } = ast.kind(args[0]) else {
            return Ok(None);
        };
        if method != "copy" {
            return Ok(None);
        }
        let Kind::Name { id: receiver, .. } = ast.kind(args[1]) else { return Ok(None) };
        let scope = self.resolve(self.owner(args[1])?, receiver);
        let mut annotation =
            scope.and_then(|scope| self.annotations.get(&(scope, receiver.clone())).copied());
        if let Some(Kind::Subscript { slice, .. }) = annotation.map(|id| ast.kind(id)) {
            annotation = Some(*slice);
        }
        let type_name = match annotation.map(|id| ast.kind(id)) {
            Some(Kind::Name { id, .. }) => id.as_str(),
            Some(Kind::Attribute { attr, .. }) => attr.as_str(),
            _ => "",
        };
        Ok(matches!(type_name, "Label" | "Line" | "Box")
            .then(|| format!("{}.copy", type_name.to_lowercase())))
    }
}

/// The expression of a request call: the `expression` keyword, else the third
/// positional argument.
fn request_expression(ast: &Ast, call: NodeId) -> Option<NodeId> {
    let Kind::Call { args, keywords, .. } = ast.kind(call) else { unreachable!() };
    keywords
        .iter()
        .find_map(|&keyword| match ast.kind(keyword) {
            Kind::Keyword { arg: Some(arg), value } if arg == "expression" => Some(*value),
            _ => None,
        })
        .or_else(|| args.get(2).copied())
}

pub fn run(ast: &mut Ast, module: NodeId, _ctx: &Context<'_>) -> Result<NodeId, Error> {
    let Kind::Module { body, .. } = ast.kind(module) else {
        return Err(Error::Unsupported("SecurityDrawings on a non-module".into()));
    };
    // A request resolves to a dotted name only through an attribute chain, so the
    // attribute carrying its last segment stands somewhere in the module
    let has_request = walk(ast, module).into_iter().any(|sub| match ast.kind(sub) {
        Kind::Attribute { attr, .. } => REQUEST_ATTRS.contains(&attr.as_str()),
        _ => false,
    });
    if !has_request {
        return Ok(module);
    }
    let mut collector = Collector {
        scopes: Vec::new(),
        owners: HashMap::new(),
        bodies: HashMap::new(),
        requests: Vec::new(),
        annotations: HashMap::new(),
        method_names: HashSet::from(["method_call".to_string()]),
    };
    let root = collector.new_scope(None);
    for &statement in body {
        collector.collect(ast, statement, root);
    }

    let mut request_scopes = HashSet::new();
    for &call in &collector.requests {
        if REQUESTS.contains(&collector.call_path(ast, call)?.as_str()) {
            request_scopes.insert(collector.owner(call)?);
        }
    }
    // A function called from a scope holding a request gets the call's arguments
    // as further bindings of its parameters
    for index in 0..collector.requests.len() {
        let call = collector.requests[index];
        let Kind::Call { func, args, keywords } = ast.kind(call) else { unreachable!() };
        let Kind::Name { id: func_name, .. } = ast.kind(*func) else { continue };
        let Some(owner) = collector.resolve(collector.owner(*func)?, func_name) else { continue };
        for function in collector.scopes[owner].bindings[func_name].clone() {
            let Some(&scope) = collector.bodies.get(&function) else { continue };
            if !request_scopes.contains(&scope) {
                continue;
            }
            let (Kind::FunctionDef { args: spec, .. }
            | Kind::AsyncFunctionDef { args: spec, .. }
            | Kind::Lambda { args: spec, .. }) = ast.kind(function)
            else {
                unreachable!()
            };
            let Kind::Arguments { posonlyargs, args: positional, .. } = ast.kind(*spec) else {
                unreachable!()
            };
            for (&argument, &value) in posonlyargs.iter().chain(positional).zip(args) {
                let Kind::Arg { arg, .. } = ast.kind(argument) else { unreachable!() };
                collector.bindings_of(scope, arg).push(value);
            }
            for &keyword in keywords {
                let Kind::Keyword { arg: Some(arg), value } = ast.kind(keyword) else { continue };
                if let Some(values) = collector.scopes[scope].bindings.get_mut(arg) {
                    values.push(*value);
                }
            }
        }
    }

    for &request in &collector.requests {
        let request_name = collector.call_path(ast, request)?;
        if !REQUESTS.contains(&request_name.as_str()) {
            continue;
        }
        let Some(expression) = request_expression(ast, request) else { continue };
        let mut pending = vec![expression];
        let mut seen = HashSet::new();
        while let Some(current) = pending.pop() {
            if !seen.insert(current) {
                continue;
            }
            let kind = ast.kind(current);
            if matches!(kind, Kind::Call { .. })
                && let Some(builtin) = collector.drawing_name(ast, current)?
            {
                return Err(Error::Rejected(format!(
                    "{builtin}() cannot be used in the expression of {}().",
                    &request_name[4..]
                )));
            }
            if let Kind::Name { id, ctx: ExprContext::Load } = kind
                && let Some(scope) = collector.resolve(collector.owner(current)?, id)
            {
                for &binding in &collector.scopes[scope].bindings[id] {
                    if !collector.bodies.contains_key(&binding) {
                        pending.push(binding);
                        continue;
                    }
                    let spec = match ast.kind(binding) {
                        Kind::FunctionDef { args, body, .. }
                        | Kind::AsyncFunctionDef { args, body, .. } => {
                            pending.extend(body);
                            *args
                        }
                        Kind::Lambda { args, body } => {
                            pending.push(*body);
                            *args
                        }
                        _ => unreachable!(),
                    };
                    let Kind::Arguments { defaults, kw_defaults, .. } = ast.kind(spec) else {
                        unreachable!()
                    };
                    pending.extend(defaults.iter().chain(kw_defaults.iter().flatten()));
                }
            }
            if matches!(
                kind,
                Kind::FunctionDef { .. } | Kind::AsyncFunctionDef { .. } | Kind::Lambda { .. }
            ) {
                continue;
            }
            pending.extend(children(ast, current));
        }
    }
    Ok(module)
}
