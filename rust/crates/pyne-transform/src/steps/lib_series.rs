//! Port of `transformers/lib_series.py`: a library variable read with an index
//! (`lib.close[1]`) gets a local `Series` declaration the series step then lays
//! out, and the read goes through that local.

use std::collections::{HashMap, HashSet};

use pyne_ast::walk::{Transformer, Visited, generic_visit, walk_statements};
use pyne_ast::{Ast, ExprContext, Kind, NodeId};

use crate::{Context, Error};

/// Builtin series that are always global, mapped to the element type their history
/// buffer stores.
fn builtin_global_series(name: &str) -> Option<&'static str> {
    match name {
        "open" | "high" | "low" | "close" | "volume" | "bid" | "ask" | "hl2" | "hlc3" | "ohlc4"
        | "hlcc4" => Some("float"),
        "time" => Some("int"),
        _ => None,
    }
}

/// Lib attributes that are not Series: an index into one is left alone.
const NON_SERIES_LIB_ATTRS: [&str; 1] = ["extra_fields"];

/// The name chain of an attribute reference (`lib.ta.x` -> `[lib, ta, x]`); a base
/// that is not a name contributes nothing.
fn attribute_chain(ast: &Ast, node: NodeId) -> Vec<String> {
    match ast.kind(node) {
        Kind::Name { id, .. } => vec![id.clone()],
        Kind::Attribute { value, attr, .. } => {
            let mut chain = attribute_chain(ast, *value);
            chain.push(attr.clone());
            chain
        }
        _ => Vec::new(),
    }
}

/// An attribute chain from a list of names, without locations.
fn create_attribute_chain(ast: &mut Ast, chain: &[String]) -> NodeId {
    let mut result = ast.add(Kind::Name { id: chain[0].clone(), ctx: ExprContext::Load });
    for name in &chain[1..] {
        result =
            ast.add(Kind::Attribute { value: result, attr: name.clone(), ctx: ExprContext::Load });
    }
    result
}

#[derive(Default)]
struct LibrarySeries {
    /// module -> {attribute key -> local name}
    lib_series_vars: HashMap<String, HashMap<String, String>>,
    /// (module, attribute key, function) triples already declared
    used_series: HashSet<(String, String, String)>,
    /// function name -> declarations
    declarations_to_insert: HashMap<String, Vec<NodeId>>,
    current_function: Option<String>,
    /// The enclosing function names of a nested function, outermost first
    parent_functions: Vec<String>,
    /// Declarations for the module level
    module_level_declarations: Vec<NodeId>,
}

impl LibrarySeries {
    fn visit_function_def(&mut self, ast: &mut Ast, id: NodeId) {
        let Kind::FunctionDef { name, .. } = ast.kind(id) else { unreachable!() };
        let name = name.clone();
        let old_function = self.current_function.take();
        if let Some(current) = &old_function {
            self.parent_functions.push(current.clone());
        }
        self.current_function = Some(name);
        generic_visit(self, ast, id);
        self.current_function = old_function;
        self.parent_functions.pop();
    }

    /// The local name of a library Series, declaring it in its target function (or
    /// the module) the first time that function uses it. Builtin series are chart
    /// global, so a nested function's declaration goes to the outermost function.
    fn process_series_usage(
        &mut self,
        ast: &mut Ast,
        module: &str,
        attr_chain: &[String],
    ) -> String {
        let builtin_type = match attr_chain {
            [name] => builtin_global_series(name),
            _ => None,
        };
        let (target_function, function_key) = match (builtin_type, self.parent_functions.first()) {
            (Some(_), Some(outermost)) => (Some(outermost.clone()), outermost.clone()),
            _ => (
                self.current_function.clone(),
                self.current_function.clone().unwrap_or_else(|| "__module__".into()),
            ),
        };
        let attr_key = attr_chain.join("·");
        let local_name = self
            .lib_series_vars
            .entry(module.to_string())
            .or_default()
            .entry(attr_key.clone())
            .or_insert_with(|| format!("__lib·{}", attr_key.replace('.', "·")))
            .clone();

        if self.used_series.insert((module.to_string(), attr_key, function_key)) {
            let annotation = match builtin_type {
                Some(element) => {
                    let series =
                        ast.add(Kind::Name { id: "Series".into(), ctx: ExprContext::Load });
                    let slice = ast.add(Kind::Name { id: element.into(), ctx: ExprContext::Load });
                    ast.add(Kind::Subscript { value: series, slice, ctx: ExprContext::Load })
                }
                None => ast.add(Kind::Name { id: "Series".into(), ctx: ExprContext::Load }),
            };
            let target = ast.add(Kind::Name { id: local_name.clone(), ctx: ExprContext::Store });
            let mut chain = vec![module.to_string()];
            chain.extend_from_slice(attr_chain);
            let value = create_attribute_chain(ast, &chain);
            let decl =
                ast.add(Kind::AnnAssign { target, annotation, value: Some(value), simple: 1 });
            match target_function {
                Some(function) => {
                    self.declarations_to_insert.entry(function).or_default().push(decl)
                }
                None => self.module_level_declarations.push(decl),
            }
        }
        local_name
    }

    fn visit_subscript(&mut self, ast: &mut Ast, id: NodeId) {
        let Kind::Subscript { value, .. } = ast.kind(id) else { unreachable!() };
        if matches!(ast.kind(*value), Kind::Attribute { .. }) {
            let attr_chain = attribute_chain(ast, *value);
            if attr_chain.first().is_some_and(|first| first == "lib") {
                if attr_chain.len() >= 2 && NON_SERIES_LIB_ATTRS.contains(&attr_chain[1].as_str()) {
                    generic_visit(self, ast, id);
                    return;
                }
                let local_name = self.process_series_usage(ast, "lib", &attr_chain[1..]);
                let name = ast.add(Kind::Name { id: local_name, ctx: ExprContext::Load });
                let Kind::Subscript { value, .. } = ast.kind_mut(id) else { unreachable!() };
                *value = name;
            }
        }
        generic_visit(self, ast, id);
    }
}

impl Transformer for LibrarySeries {
    fn visit(&mut self, ast: &mut Ast, id: NodeId) -> Visited {
        match ast.kind(id) {
            Kind::FunctionDef { .. } => self.visit_function_def(ast, id),
            Kind::Subscript { .. } => self.visit_subscript(ast, id),
            _ => generic_visit(self, ast, id),
        }
        Visited::Node(id)
    }
}

pub fn run(ast: &mut Ast, module: NodeId, _ctx: &Context<'_>) -> Result<NodeId, Error> {
    if !matches!(ast.kind(module), Kind::Module { .. }) {
        return Err(Error::Unsupported("LibrarySeries on a non-module".into()));
    }
    let mut series = LibrarySeries::default();
    generic_visit(&mut series, ast, module);

    // Every function of a name gets the declarations collected under that name
    if !series.declarations_to_insert.is_empty() {
        for stmt in walk_statements(ast, module) {
            let Kind::FunctionDef { name, .. } = ast.kind(stmt) else { continue };
            let Some(decls) = series.declarations_to_insert.get(name) else { continue };
            let decls = decls.clone();
            let Kind::FunctionDef { body, .. } = ast.kind_mut(stmt) else { unreachable!() };
            body.splice(0..0, decls);
        }
    }

    if !series.module_level_declarations.is_empty() {
        // After the imports and the ``__all__`` assignment leading the module
        let Kind::Module { body, .. } = ast.kind(module) else { unreachable!() };
        let mut insert_pos = 0;
        for (index, &stmt) in body.iter().enumerate() {
            let leading = match ast.kind(stmt) {
                Kind::Import { .. } | Kind::ImportFrom { .. } => true,
                Kind::Assign { targets, .. } => {
                    targets.len() == 1
                        && matches!(ast.kind(targets[0]), Kind::Name { id, .. } if id == "__all__")
                }
                _ => false,
            };
            if !leading {
                break;
            }
            insert_pos = index + 1;
        }
        let Kind::Module { body, .. } = ast.kind_mut(module) else { unreachable!() };
        body.splice(insert_pos..insert_pos, series.module_level_declarations);
    }
    Ok(module)
}
