//! Port of `transformers/input_transformer.py`: an input call that is a parameter's
//! default gets the parameter's name as its `_id`; a source input's parameter is
//! rebound at the top of its function to the series it names
//! (`x = __pyne_getattr__(lib, x, lib.na)`), the reserved alias imported once.

use indexmap::{IndexMap, IndexSet};
use pyne_ast::walk::children;
use pyne_ast::{Ast, Constant, ExprContext, Int, Kind, NodeId};

use crate::{Context, Error};

/// Builtin price series accepted by generic `input()` as source names
/// (`price_series.BUILTIN_PRICE_SERIES`).
const BUILTIN_PRICE_SERIES: &[&str] =
    &["open", "high", "low", "close", "volume", "bid", "ask", "hl2", "hlc3", "ohlc4", "hlcc4"];

/// The reserved alias of the `getattr` builtin
const GETATTR_ALIAS: &str = "__pyne_getattr__";

fn is_name(ast: &Ast, id: NodeId, name: &str) -> bool {
    matches!(ast.kind(id), Kind::Name { id, .. } if id == name)
}

/// Python's truth value of a constant.
fn constant_truthy(value: &Constant) -> bool {
    match value {
        Constant::None => false,
        Constant::Bool(flag) => *flag,
        Constant::Int(Int::Small(number)) => *number != 0,
        Constant::Int(Int::Big(_)) => true,
        Constant::Float(number) => *number != 0.0,
        Constant::Complex { real, imag } => *real != 0.0 || *imag != 0.0,
        Constant::Str(text) => !text.is_empty(),
        Constant::Bytes(bytes) => !bytes.is_empty(),
        Constant::Ellipsis => true,
        Constant::Tuple(items) | Constant::FrozenSet(items) => !items.is_empty(),
    }
}

/// Whether a call is `lib.input(...)`, `lib.input.<kind>(...)`, `input(...)` or
/// `input.<kind>(...)`.
fn is_input_call(ast: &Ast, call: NodeId) -> bool {
    let Kind::Call { func, .. } = ast.kind(call) else { return false };
    match ast.kind(*func) {
        Kind::Attribute { value, attr, .. } if attr == "input" => is_name(ast, *value, "lib"),
        Kind::Attribute { value, .. } => match ast.kind(*value) {
            Kind::Attribute { value: owner, attr, .. } => {
                is_name(ast, *owner, "lib") && attr == "input"
            }
            Kind::Name { id, .. } => id == "input",
            _ => false,
        },
        Kind::Name { id, .. } => id == "input",
        _ => false,
    }
}

struct Input {
    /// Function name -> the source-input parameters to rebind, in first-seen order
    function_source_vars: IndexMap<String, IndexSet<String>>,
    current_function: Option<String>,
    has_source_inputs: bool,
}

impl Input {
    /// Add `_id` to the input calls among a function's defaults and collect its
    /// source inputs.
    fn visit_arguments(&mut self, ast: &mut Ast, node: NodeId) {
        let Some(current) = self.current_function.clone() else { return };
        let Kind::Arguments { args, defaults, .. } = ast.kind(node) else { unreachable!() };
        // `zip(node.args[-len(node.defaults):], node.defaults)`
        let start = if defaults.is_empty() { 0 } else { args.len().saturating_sub(defaults.len()) };
        let pairs: Vec<(NodeId, NodeId)> =
            args[start..].iter().copied().zip(defaults.iter().copied()).collect();
        for (arg, default) in pairs {
            if !matches!(ast.kind(default), Kind::Call { .. }) || !is_input_call(ast, default) {
                continue;
            }
            let Kind::Arg { arg: arg_name, .. } = ast.kind(arg) else { unreachable!() };
            let arg_name = arg_name.clone();
            let value =
                ast.add(Kind::Constant { value: Constant::Str(arg_name.clone()), kind: None });
            let keyword = ast.add(Kind::Keyword { arg: Some("_id".into()), value });
            let Kind::Call { keywords, .. } = ast.kind_mut(default) else { unreachable!() };
            keywords.push(keyword);

            let Kind::Call { func, args, keywords } = ast.kind(default) else { unreachable!() };
            let (is_source_call, is_input_call) = match ast.kind(*func) {
                Kind::Attribute { attr, .. } if attr == "source" => (true, false),
                Kind::Attribute { attr, .. } if attr == "input" => (false, true),
                Kind::Name { id, .. } if id == "input" => (false, true),
                _ => (false, false),
            };
            let defval = args.first().copied().or_else(|| {
                keywords.iter().find_map(|&keyword| match ast.kind(keyword) {
                    Kind::Keyword { arg: Some(arg), value } if arg == "defval" => Some(*value),
                    _ => None,
                })
            });
            let Some(defval) = defval else { continue };
            if !(is_source_call || is_input_call) {
                continue;
            }
            // `input.source()` always returns the source NAME, so its defval is resolved
            // whatever shape it has; the generic `input()` only returns a name when the
            // defval is a builtin price series
            let source = match ast.kind(defval) {
                Kind::Constant { value, .. } if is_source_call => constant_truthy(value),
                Kind::Constant { .. } => false,
                Kind::Attribute { value, attr, .. } => {
                    is_name(ast, *value, "lib")
                        && (is_source_call || BUILTIN_PRICE_SERIES.contains(&attr.as_str()))
                        && !attr.is_empty()
                }
                Kind::Name { id, .. } => {
                    (is_source_call || BUILTIN_PRICE_SERIES.contains(&id.as_str()))
                        && !id.is_empty()
                }
                _ => false,
            };
            if source {
                self.function_source_vars.entry(current.clone()).or_default().insert(arg_name);
                self.has_source_inputs = true;
            }
        }
    }

    fn visit_function(&mut self, ast: &mut Ast, node: NodeId, name: String) {
        let previous = self.current_function.replace(name.clone());
        self.generic_visit(ast, node);
        let source_vars: Vec<String> = self
            .function_source_vars
            .get(&name)
            .map(|vars| vars.iter().cloned().collect())
            .unwrap_or_default();
        for var_name in source_vars {
            let target = ast.add(Kind::Name { id: var_name.clone(), ctx: ExprContext::Store });
            let func = ast.add(Kind::Name { id: GETATTR_ALIAS.into(), ctx: ExprContext::Load });
            let lib = ast.add(Kind::Name { id: "lib".into(), ctx: ExprContext::Load });
            let read = ast.add(Kind::Name { id: var_name, ctx: ExprContext::Load });
            let owner = ast.add(Kind::Name { id: "lib".into(), ctx: ExprContext::Load });
            let na = ast.add(Kind::Attribute {
                value: owner,
                attr: "na".into(),
                ctx: ExprContext::Load,
            });
            let value =
                ast.add(Kind::Call { func, args: vec![lib, read, na], keywords: Vec::new() });
            let assign = ast.add(Kind::Assign { targets: vec![target], value, type_comment: None });
            let Kind::FunctionDef { body, .. } = ast.kind_mut(node) else { unreachable!() };
            body.insert(0, assign);
        }
        self.current_function = previous;
    }

    fn generic_visit(&mut self, ast: &mut Ast, node: NodeId) {
        for child in children(ast, node) {
            self.visit(ast, child);
        }
    }

    fn visit(&mut self, ast: &mut Ast, node: NodeId) {
        match ast.kind(node) {
            Kind::FunctionDef { name, .. } => {
                let name = name.clone();
                self.visit_function(ast, node, name);
            }
            Kind::Arguments { .. } => self.visit_arguments(ast, node),
            _ => self.generic_visit(ast, node),
        }
    }
}

pub fn run(ast: &mut Ast, module: NodeId, _ctx: &Context<'_>) -> Result<NodeId, Error> {
    if !matches!(ast.kind(module), Kind::Module { .. }) {
        return Err(Error::Unsupported("Input on a non-module".into()));
    }
    let mut pass = Input {
        function_source_vars: IndexMap::new(),
        current_function: None,
        has_source_inputs: false,
    };
    pass.generic_visit(ast, module);
    if !pass.has_source_inputs {
        return Ok(module);
    }
    // The reserved alias of the builtin, bound at module scope so a function-local
    // `getattr` cannot shadow it
    let alias = ast.add(Kind::Alias { name: "getattr".into(), asname: Some(GETATTR_ALIAS.into()) });
    let import = ast.add(Kind::ImportFrom {
        module: Some("builtins".into()),
        names: vec![alias],
        level: Some(0),
    });
    let Kind::Module { body, .. } = ast.kind(module) else { unreachable!() };
    let mut position = 0;
    if let Some(&first) = body.first()
        && let Kind::Expr { value } = ast.kind(first)
        && matches!(ast.kind(*value), Kind::Constant { value: Constant::Str(_), .. })
    {
        position = 1;
    }
    while position < body.len()
        && matches!(ast.kind(body[position]),
            Kind::ImportFrom { module: Some(name), .. } if name == "__future__")
    {
        position += 1;
    }
    let Kind::Module { body, .. } = ast.kind_mut(module) else { unreachable!() };
    body.insert(position, import);
    Ok(module)
}
