//! Port of `transformers/input_transformer.py`: an input call that is a parameter
//! default of module-level `main()` (or of one of its generated clones) gets the
//! parameter's name as its `_id`; a source input's parameter is rebound at the top of
//! its function to the series it names (`x = __pyne_getattr__(lib, x, lib.na)`), the
//! reserved alias imported once. Any other input call rejects the module.

use std::collections::HashSet;

use indexmap::IndexSet;
use pyne_ast::walk::{children, docstring_end, future_end};
use pyne_ast::{Ast, Constant, ExprContext, Int, Kind, NodeId};

use crate::resolver::is_pipeline_name;
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

/// Whether a call is `lib.input(...)` or `lib.input.<kind>(...)`. The import normalizer
/// has already rewritten every name imported from `pynecore.lib` to its `lib.` path, so
/// a bare `input` is a name the script binds itself (or Python's builtin).
fn is_input_call(ast: &Ast, call: NodeId) -> bool {
    let Kind::Call { func, .. } = ast.kind(call) else { return false };
    match ast.kind(*func) {
        Kind::Attribute { value, attr, .. } if attr == "input" => is_name(ast, *value, "lib"),
        Kind::Attribute { value, .. } => match ast.kind(*value) {
            Kind::Attribute { value: owner, attr, .. } => {
                is_name(ast, *owner, "lib") && attr == "input"
            }
            _ => false,
        },
        _ => false,
    }
}

struct Input {
    has_source_inputs: bool,
    /// The input calls that are parameter defaults of an entry
    declared: HashSet<NodeId>,
    /// The module's own statements
    module_level: HashSet<NodeId>,
}

impl Input {
    /// Add `_id` to the input calls among an entry's defaults and collect its source
    /// inputs: the parameters to rebind to the series they name, in parameter order.
    fn declare_inputs(&mut self, ast: &mut Ast, function: NodeId) -> IndexSet<String> {
        let mut source_vars = IndexSet::new();
        let Kind::FunctionDef { args: spec, .. } = ast.kind(function) else { unreachable!() };
        let Kind::Arguments { posonlyargs, args, kwonlyargs, kw_defaults, defaults, .. } =
            ast.kind(*spec)
        else {
            unreachable!()
        };
        let positional: Vec<NodeId> = posonlyargs.iter().chain(args).copied().collect();
        let start = positional.len().saturating_sub(defaults.len());
        let mut pairs: Vec<(NodeId, NodeId)> =
            positional[start..].iter().copied().zip(defaults.iter().copied()).collect();
        pairs.extend(
            kwonlyargs
                .iter()
                .zip(kw_defaults)
                .filter_map(|(&arg, default)| default.map(|default| (arg, default))),
        );
        for (arg, default) in pairs {
            if !matches!(ast.kind(default), Kind::Call { .. }) || !is_input_call(ast, default) {
                continue;
            }
            self.declared.insert(default);
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
                source_vars.insert(arg_name);
                self.has_source_inputs = true;
            }
        }
        source_vars
    }

    fn visit_function(&mut self, ast: &mut Ast, node: NodeId, name: String) -> Result<(), Error> {
        // The runner reads inputs only from the parameter defaults of module-level
        // main(). The security slices are generated module-level clones of main() that
        // copy its parameters; their names are pipeline names no script can bind
        let source_vars =
            if self.module_level.contains(&node) && (name == "main" || is_pipeline_name(&name)) {
                self.declare_inputs(ast, node)
            } else {
                IndexSet::new()
            };
        self.generic_visit(ast, node)?;
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
            let Kind::FunctionDef { body, .. } = ast.kind(node) else { unreachable!() };
            let at = docstring_end(ast, body);
            let Kind::FunctionDef { body, .. } = ast.kind_mut(node) else { unreachable!() };
            body.insert(at, assign);
        }
        Ok(())
    }

    fn generic_visit(&mut self, ast: &mut Ast, node: NodeId) -> Result<(), Error> {
        for child in children(ast, node) {
            self.visit(ast, child)?;
        }
        Ok(())
    }

    fn visit(&mut self, ast: &mut Ast, node: NodeId) -> Result<(), Error> {
        match ast.kind(node) {
            Kind::FunctionDef { name, .. } => {
                let name = name.clone();
                self.visit_function(ast, node, name)
            }
            Kind::Call { .. } if !self.declared.contains(&node) && is_input_call(ast, node) => {
                // An input outside the parameter defaults gets no `_id`: it never
                // reaches the settings file and silently runs with its default
                let line =
                    ast[node].loc.lineno.map_or_else(|| "None".to_string(), |l| l.to_string());
                Err(Error::Rejected(format!(
                    "the input call on line {line} is not a main() parameter default; inputs \
                     must be declared as main() parameter defaults: \
                     def main(length=input.int(14, ...))"
                )))
            }
            _ => self.generic_visit(ast, node),
        }
    }
}

pub fn run(ast: &mut Ast, module: NodeId, _ctx: &Context<'_>) -> Result<NodeId, Error> {
    if !matches!(ast.kind(module), Kind::Module { .. }) {
        return Err(Error::Unsupported("Input on a non-module".into()));
    }
    let Kind::Module { body, .. } = ast.kind(module) else { unreachable!() };
    let module_level = body.iter().copied().collect();
    let mut pass = Input { has_source_inputs: false, declared: HashSet::new(), module_level };
    pass.generic_visit(ast, module)?;
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
        is_lazy: None,
    });
    let Kind::Module { body, .. } = ast.kind(module) else { unreachable!() };
    let position = future_end(ast, body);
    let Kind::Module { body, .. } = ast.kind_mut(module) else { unreachable!() };
    body.insert(position, import);
    Ok(module)
}
