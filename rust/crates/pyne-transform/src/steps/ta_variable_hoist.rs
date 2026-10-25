//! Port of `transformers/ta_variable_hoist.py`: evaluate the referenced stateful `ta`
//! builtin variables once per bar, at the top of `main`.
//!
//! A rewireable zero-argument `lib.ta.<name>()` call becomes a read of the
//! module-global cache `__ta·<name>`, and `main` gets an unconditional prologue
//! assigning each referenced cache from the real call. Which reads are rewired
//! follows the measured TradingView scoping the Python module documents.

use pyne_ast::walk::{Transformer, Visited, copy_location, docstring_end, generic_visit};
use pyne_ast::{Ast, ExprContext, Kind, NodeId};

use crate::{Context, Error};

/// Engine-global inside user functions too: rewired everywhere
const ENGINE_GLOBAL_VARIABLES: [&str; 5] = ["nvi", "obv", "pvi", "pvt", "wad"];
/// Per-instance inside a function: rewired in `main`'s own body only
const MAIN_SCOPE_VARIABLES: [&str; 2] = ["accdist", "vwap"];

fn is_hoisted(name: &str) -> bool {
    ENGINE_GLOBAL_VARIABLES.contains(&name) || MAIN_SCOPE_VARIABLES.contains(&name)
}

/// The cache name of a variable; the middle-dot namespace is reserved for the transformers.
fn cache_name(name: &str) -> String {
    format!("__ta·{name}")
}

/// The callable a decorator names: the function of a decorator call, else the decorator.
fn decorator_func(ast: &Ast, decorator: NodeId) -> NodeId {
    match ast.kind(decorator) {
        Kind::Call { func, .. } => *func,
        _ => decorator,
    }
}

fn is_library_main(ast: &Ast, decorators: &[NodeId]) -> bool {
    decorators.iter().any(|&decorator| {
        matches!(ast.kind(decorator_func(ast, decorator)), Kind::Attribute { attr, .. } if attr == "library")
    })
}

fn is_export(ast: &Ast, decorators: &[NodeId]) -> bool {
    decorators.iter().any(|&decorator| match ast.kind(decorator_func(ast, decorator)) {
        Kind::Name { id, .. } => id == "export",
        Kind::Attribute { attr, .. } => attr == "export",
        _ => false,
    })
}

struct TaVariableHoist {
    /// Referenced variable names, in first-reference order
    hoisted: Vec<String>,
    function_stack: Vec<String>,
    /// How many enclosing functions are `@export`-decorated
    export_depth: usize,
    is_library: bool,
}

impl TaVariableHoist {
    fn visit_call(&mut self, ast: &mut Ast, id: NodeId) -> NodeId {
        generic_visit(self, ast, id);
        let Kind::Call { func, args, keywords } = ast.kind(id) else { unreachable!() };
        if !args.is_empty() || !keywords.is_empty() {
            return id;
        }
        let Kind::Attribute { value: module, attr, .. } = ast.kind(*func) else { return id };
        if !is_hoisted(attr) {
            return id;
        }
        let is_ta = matches!(ast.kind(*module), Kind::Attribute { value, attr, .. }
            if attr == "ta" && matches!(ast.kind(*value), Kind::Name { id, .. } if id == "lib"));
        if !is_ta {
            return id;
        }
        if self.function_stack.is_empty() {
            // Module-level code runs once at import, outside the per-bar state machinery
            return id;
        }
        if self.is_library && (self.function_stack[0] != "main" || self.export_depth > 0) {
            // Module-level functions of a hand-written library are its export
            // surface, and exported functions keep per-call-site machines
            return id;
        }
        if MAIN_SCOPE_VARIABLES.contains(&attr.as_str()) && self.function_stack != ["main"] {
            // Inside a user function these get their own per-instance machine
            return id;
        }
        let attr = attr.clone();
        let name = ast.add(Kind::Name { id: cache_name(&attr), ctx: ExprContext::Load });
        if !self.hoisted.contains(&attr) {
            self.hoisted.push(attr);
        }
        copy_location(ast, name, id);
        name
    }
}

impl Transformer for TaVariableHoist {
    fn visit(&mut self, ast: &mut Ast, id: NodeId) -> Visited {
        match ast.kind(id) {
            Kind::FunctionDef { name, decorator_list, .. } => {
                let export = is_export(ast, decorator_list);
                self.function_stack.push(name.clone());
                self.export_depth += usize::from(export);
                generic_visit(self, ast, id);
                self.function_stack.pop();
                self.export_depth -= usize::from(export);
            }
            Kind::Call { .. } => return Visited::Node(self.visit_call(ast, id)),
            _ => generic_visit(self, ast, id),
        }
        Visited::Node(id)
    }
}

pub fn run(ast: &mut Ast, module: NodeId, _ctx: &Context<'_>) -> Result<NodeId, Error> {
    let Kind::Module { body, .. } = ast.kind(module) else {
        return Err(Error::Unsupported("TaVariableHoist on a non-module".into()));
    };
    let main = body
        .iter()
        .copied()
        .find(|&stmt| matches!(ast.kind(stmt), Kind::FunctionDef { name, .. } if name == "main"));
    let Some(main) = main else { return Ok(module) };
    let Kind::FunctionDef { decorator_list, .. } = ast.kind(main) else { unreachable!() };
    let mut hoist = TaVariableHoist {
        hoisted: Vec::new(),
        function_stack: Vec::new(),
        export_depth: 0,
        is_library: is_library_main(ast, decorator_list),
    };
    generic_visit(&mut hoist, ast, module);
    if hoist.hoisted.is_empty() {
        return Ok(module);
    }

    let mut prologue = vec![
        ast.add(Kind::Global { names: hoist.hoisted.iter().map(|n| cache_name(n)).collect() }),
    ];
    for name in &hoist.hoisted {
        let target = ast.add(Kind::Name { id: cache_name(name), ctx: ExprContext::Store });
        let lib = ast.add(Kind::Name { id: "lib".into(), ctx: ExprContext::Load });
        let ta = ast.add(Kind::Attribute { value: lib, attr: "ta".into(), ctx: ExprContext::Load });
        let func =
            ast.add(Kind::Attribute { value: ta, attr: name.clone(), ctx: ExprContext::Load });
        let value = ast.add(Kind::Call { func, args: Vec::new(), keywords: Vec::new() });
        prologue.push(ast.add(Kind::Assign { targets: vec![target], value, type_comment: None }));
    }

    // Everything in main but the docstring may already read the caches
    let Kind::FunctionDef { body, .. } = ast.kind(main) else { unreachable!() };
    let insert_at = docstring_end(ast, body);
    let Kind::FunctionDef { body, .. } = ast.kind_mut(main) else { unreachable!() };
    body.splice(insert_at..insert_at, prologue);
    Ok(module)
}
