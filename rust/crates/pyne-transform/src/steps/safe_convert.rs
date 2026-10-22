//! Port of `transformers/safe_convert_transformer.py`, a rule of the `Expression`
//! phase: `float()` / `int()` become the na-safe `safe_convert` casts (`int()` of a
//! `@pyne lib` module the native truncation), and the Python-native consumers of a
//! Pine int -- the arguments of the builtin `range()` and a subscript index typed
//! as a Pine int -- are truncated with `safe_convert.native_int`.

use std::collections::HashSet;

use pyne_ast::walk::{Visited, copy_location, walk};
use pyne_ast::{Ast, Constant, ExprContext, Int, Kind, NodeId};

use crate::phase::{Rule, run_rules, single};
use crate::pine_type_rules::{INT, get_ty, stamp_lowering, ty_stamp};
use crate::{Context, Error};

/// `safe_convert.<attr>`.
pub(crate) fn safe_convert_attr(ast: &mut Ast, attr: &str) -> NodeId {
    let value = ast.add(Kind::Name { id: "safe_convert".into(), ctx: ExprContext::Load });
    ast.add(Kind::Attribute { value, attr: attr.into(), ctx: ExprContext::Load })
}

/// Where the `safe_convert` import goes: after a leading constant statement (the
/// docstring) and after the imports that follow it.
pub(crate) fn import_position(ast: &Ast, body: &[NodeId]) -> usize {
    let mut position = 0;
    if let Some(&first) = body.first()
        && let Kind::Expr { value } = ast.kind(first)
        && matches!(ast.kind(*value), Kind::Constant { .. })
    {
        position = 1;
    }
    while position < body.len()
        && matches!(ast.kind(body[position]), Kind::Import { .. } | Kind::ImportFrom { .. })
    {
        position += 1;
    }
    position
}

/// `from pynecore.core import safe_convert`.
pub(crate) fn safe_convert_import(ast: &mut Ast) -> NodeId {
    let alias = ast.add(Kind::Alias { name: "safe_convert".into(), asname: None });
    ast.add(Kind::ImportFrom {
        module: Some("pynecore.core".into()),
        names: vec![alias],
        level: Some(0),
    })
}

/// Whether an `import pynecore.core.safe_convert as safe_convert` binds the module name.
pub(crate) fn binds_by_import(ast: &Ast, names: &[NodeId]) -> bool {
    names.iter().any(|&alias| {
        matches!(ast.kind(alias), Kind::Alias { name, asname: Some(asname) }
            if name == "pynecore.core.safe_convert" && asname == "safe_convert")
    })
}

/// `int(value)` of a float literal with an integral value.
fn integral(value: f64) -> Result<Int, Error> {
    // Every float in [-2**63, 2**63) converts exactly; beyond it Python makes a big int
    const LIMIT: f64 = 9_223_372_036_854_775_808.0;
    if (-LIMIT..LIMIT).contains(&value) {
        Ok(Int::Small(value as i64))
    } else {
        Err(Error::Unsupported(format!("an int index literal {value} beyond i64")))
    }
}

pub struct SafeConvert {
    /// The module is a `@pyne lib`: `int()` truncates to a native int
    lib: bool,
    /// A cast or a truncation was emitted, so the module needs `safe_convert`
    has_convert_functions: bool,
    /// The module being transformed
    module: Option<NodeId>,
    /// The module binds its own `range`; `None` until asked
    range_shadowed: Option<bool>,
    /// Loop counters bound by an enclosing `for ... in range(...)`
    range_vars: HashSet<String>,
    /// Per enclosing `for`, the counter it added to `range_vars`, if any
    for_counters: Vec<Option<String>>,
}

impl SafeConvert {
    pub fn new(ctx: &Context<'_>) -> SafeConvert {
        SafeConvert {
            lib: ctx.pyne_mode.as_deref() == Some("lib"),
            has_convert_functions: false,
            module: None,
            range_shadowed: None,
            range_vars: HashSet::new(),
            for_counters: Vec::new(),
        }
    }

    /// Whether a `range` call is the builtin, i.e. the module binds no `range`: a
    /// definition, a stored name or an import alias. Scanned once, at the first ask.
    fn is_builtin_range(&mut self, ast: &Ast) -> bool {
        let shadowed = *self.range_shadowed.get_or_insert_with(|| {
            self.module.is_some_and(|module| {
                walk(ast, module).into_iter().any(|id| match ast.kind(id) {
                    Kind::FunctionDef { name, .. }
                    | Kind::AsyncFunctionDef { name, .. }
                    | Kind::ClassDef { name, .. } => name == "range",
                    Kind::Name { id, ctx: ExprContext::Store } => id == "range",
                    Kind::Alias { name, asname } => asname.as_deref().unwrap_or(name) == "range",
                    _ => false,
                })
            })
        });
        !shadowed
    }

    /// Wrap a Python-native consumer's argument in the native truncation.
    fn native_int(&mut self, ast: &mut Ast, arg: NodeId) -> NodeId {
        self.has_convert_functions = true;
        let func = safe_convert_attr(ast, "native_int");
        let call = ast.add(Kind::Call { func, args: vec![arg], keywords: Vec::new() });
        copy_location(ast, call, arg);
        stamp_lowering(ast, call, INT)
    }

    /// Truncate a subscript index typed as a Pine int; leave any other alone.
    fn native_index(&mut self, ast: &mut Ast, index: NodeId) -> Result<NodeId, Error> {
        if get_ty(ast, index) != INT {
            return Ok(index);
        }
        match ast.kind(index) {
            &Kind::Constant { value: Constant::Float(value), .. } => {
                // A folded Pine int literal is a float (`2.0`): truncate it here
                if value.is_finite() && value.fract() == 0.0 {
                    let value = integral(value)?;
                    let Kind::Constant { value: slot, .. } = ast.kind_mut(index) else {
                        unreachable!()
                    };
                    *slot = Constant::Int(value);
                }
                return Ok(index);
            }
            Kind::Constant { .. } => return Ok(index),
            // The counter of a `range()` loop is already a native int
            Kind::Name { id, .. } if self.range_vars.contains(id) => return Ok(index),
            &Kind::Call { func, .. } => {
                if let Kind::Attribute { value, attr, .. } = ast.kind(func)
                    && attr == "safe_int"
                    && matches!(ast.kind(*value), Kind::Name { id, .. } if id == "safe_convert")
                {
                    // The index IS the cast: truncate natively instead
                    let Kind::Attribute { attr, .. } = ast.kind_mut(func) else { unreachable!() };
                    *attr = "native_int".into();
                    return Ok(index);
                }
            }
            _ => {}
        }
        Ok(self.native_int(ast, index))
    }

    fn enter_for(&mut self, ast: &Ast, node: NodeId) {
        let mut counter = None;
        if let &Kind::For { target, iter, .. } = ast.kind(node)
            && let Kind::Name { id, .. } = ast.kind(target)
            && let Kind::Call { func, .. } = ast.kind(iter)
            && matches!(ast.kind(*func), Kind::Name { id, .. } if id == "range")
            && !self.range_vars.contains(id)
            && self.is_builtin_range(ast)
        {
            counter = Some(id.clone());
            self.range_vars.insert(id.clone());
        }
        self.for_counters.push(counter);
    }

    fn leave_subscript(&mut self, ast: &mut Ast, node: NodeId) -> Result<(), Error> {
        let &Kind::Subscript { value: base, slice: index, .. } = ast.kind(node) else {
            return Ok(());
        };
        // A series buffer read: the series pass has rewritten the series to its
        // `<state param>[slot]` reference
        if let &Kind::Subscript { value, .. } = ast.kind(base)
            && let Kind::Name { id, .. } = ast.kind(value)
            && id.starts_with("__state")
            && id.ends_with("__")
        {
            return Ok(());
        }
        match *ast.kind(index) {
            Kind::Slice { lower, upper, step } => {
                if let Some(lower) = lower {
                    let lower = self.native_index(ast, lower)?;
                    let Kind::Slice { lower: slot, .. } = ast.kind_mut(index) else {
                        unreachable!()
                    };
                    *slot = Some(lower);
                }
                if let Some(upper) = upper {
                    let upper = self.native_index(ast, upper)?;
                    let Kind::Slice { upper: slot, .. } = ast.kind_mut(index) else {
                        unreachable!()
                    };
                    *slot = Some(upper);
                }
                if let Some(step) = step {
                    let step = self.native_index(ast, step)?;
                    let Kind::Slice { step: slot, .. } = ast.kind_mut(index) else {
                        unreachable!()
                    };
                    *slot = Some(step);
                }
            }
            Kind::Tuple { .. } => {}
            _ => {
                let index = self.native_index(ast, index)?;
                let Kind::Subscript { slice, .. } = ast.kind_mut(node) else { unreachable!() };
                *slice = index;
            }
        }
        Ok(())
    }

    fn leave_call(&mut self, ast: &mut Ast, node: NodeId) -> Result<NodeId, Error> {
        let Kind::Call { func, args, keywords } = ast.kind(node) else { return Ok(node) };
        let Kind::Name { id: callee, .. } = ast.kind(*func) else { return Ok(node) };
        if callee == "range" && keywords.is_empty() && self.is_builtin_range(ast) {
            let Kind::Call { args, .. } = ast.kind(node) else { unreachable!() };
            let args = args.clone();
            let mut new_args = Vec::with_capacity(args.len());
            for arg in args {
                let keep = matches!(
                    ast.kind(arg),
                    Kind::Starred { .. } | Kind::Constant { value: Constant::Int(_), .. }
                );
                new_args.push(if keep { arg } else { self.native_int(ast, arg) });
            }
            let Kind::Call { args, .. } = ast.kind_mut(node) else { unreachable!() };
            *args = new_args;
            return Ok(node);
        }
        let attr = match callee.as_str() {
            "int" if self.lib => "native_int",
            "int" => "safe_int",
            "float" => "safe_float",
            _ => return Ok(node),
        };
        self.has_convert_functions = true;
        // The na-safe form converts what the builtin converts: it keeps the cast's type
        let (args, keywords) = (args.clone(), keywords.clone());
        let ty = ty_stamp(ast, node)?.to_string();
        let func = safe_convert_attr(ast, attr);
        let call = ast.add(Kind::Call { func, args, keywords });
        Ok(stamp_lowering(ast, call, &ty))
    }

    fn leave_module(&mut self, ast: &mut Ast, node: NodeId) {
        if !self.has_convert_functions {
            return;
        }
        let Kind::Module { body, .. } = ast.kind(node) else { return };
        // An existing binding of the `safe_convert` module name is enough
        for &stmt in body {
            let bound = match ast.kind(stmt) {
                Kind::ImportFrom { module, names, .. } => {
                    let module = module.as_deref().unwrap_or("");
                    let module = module.strip_prefix("pynecore.").unwrap_or(module);
                    let alias_binds = |test: &dyn Fn(&str, Option<&str>) -> bool| {
                        names.iter().any(|&alias| match ast.kind(alias) {
                            Kind::Alias { name, asname } => test(name, asname.as_deref()),
                            _ => false,
                        })
                    };
                    match module {
                        "core.safe_convert" => {
                            alias_binds(&|_, asname| asname == Some("safe_convert"))
                        }
                        "core" => {
                            alias_binds(&|name, asname| asname.unwrap_or(name) == "safe_convert")
                        }
                        _ => continue,
                    }
                }
                Kind::Import { names } => binds_by_import(ast, names),
                _ => continue,
            };
            if bound {
                return;
            }
        }
        let position = import_position(ast, body);
        let import = safe_convert_import(ast);
        let Kind::Module { body, .. } = ast.kind_mut(node) else { unreachable!() };
        body.insert(position, import);
    }
}

impl Rule for SafeConvert {
    fn enter(&mut self, ast: &mut Ast, class: &'static str, node: NodeId) -> Result<(), Error> {
        match class {
            "Module" => self.module = Some(node),
            "For" => self.enter_for(ast, node),
            _ => {}
        }
        Ok(())
    }

    fn leave(
        &mut self,
        ast: &mut Ast,
        class: &'static str,
        node: Visited,
    ) -> Result<Visited, Error> {
        match class {
            "For" => {
                single(&node, class)?;
                if let Some(counter) = self.for_counters.pop().flatten() {
                    self.range_vars.remove(&counter);
                }
                Ok(node)
            }
            "Subscript" => {
                self.leave_subscript(ast, single(&node, class)?)?;
                Ok(node)
            }
            "Call" => Ok(Visited::Node(self.leave_call(ast, single(&node, class)?)?)),
            "Module" => {
                self.leave_module(ast, single(&node, class)?);
                Ok(node)
            }
            _ => Ok(node),
        }
    }
}

fn make(ctx: &Context<'_>) -> Box<dyn Rule> {
    Box::new(SafeConvert::new(ctx))
}

/// The rule on its own (sequential mode).
pub fn run(ast: &mut Ast, module: NodeId, ctx: &Context<'_>) -> Result<NodeId, Error> {
    run_rules(ast, module, vec![make(ctx)])
}

pub const RULE: crate::phase::RuleStep =
    crate::phase::RuleStep { name: "SafeConvert", make, user_code_only: false };
