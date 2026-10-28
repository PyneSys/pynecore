//! Port of `transformers/type_erasure.py`: replace `typing.cast(T, x)` with `x` and
//! resolve `if TYPE_CHECKING:` where that provably keeps the module's behaviour.
//!
//! A name is trusted only when it is bound by a module-level `typing` import and
//! bound nowhere else in the module; see the Python module for the full contract.

use std::collections::{HashMap, HashSet};

use pyne_ast::walk::{
    Transformer, Visited, Visitor, copy_location, generic_visit, generic_visit_ro, walk,
};
use pyne_ast::{Ast, ExprContext, Kind, NodeId, Operator};

use crate::{Context, Error};

/// Whether evaluating an expression can have no effect worth keeping: names,
/// attributes, subscripts, constants, tuples, lists and `|` unions only.
fn is_type_expression(ast: &Ast, id: NodeId) -> bool {
    walk(ast, id).into_iter().all(|child| match ast.kind(child) {
        Kind::Name { ctx, .. }
        | Kind::Attribute { ctx, .. }
        | Kind::Subscript { ctx, .. }
        | Kind::Tuple { ctx, .. }
        | Kind::List { ctx, .. } => *ctx == ExprContext::Load,
        Kind::Constant { .. } => true,
        Kind::BinOp { op, .. } => *op == Operator::BitOr,
        _ => false,
    })
}

/// Count, per name, every construct in a module that binds or unbinds it.
#[derive(Default)]
struct BindingCounter {
    bound: HashMap<String, usize>,
}

impl BindingCounter {
    fn bind(&mut self, name: Option<&str>) {
        if let Some(name) = name {
            *self.bound.entry(name.to_string()).or_default() += 1;
        }
    }
}

impl Visitor for BindingCounter {
    fn visit(&mut self, ast: &Ast, id: NodeId) {
        match ast.kind(id) {
            Kind::Name { id: name, ctx } => {
                if *ctx != ExprContext::Load {
                    self.bind(Some(name));
                }
            }
            Kind::Arg { arg, .. } => self.bind(Some(arg)),
            Kind::FunctionDef { name, .. }
            | Kind::AsyncFunctionDef { name, .. }
            | Kind::ClassDef { name, .. } => {
                self.bind(Some(name));
                generic_visit_ro(self, ast, id);
            }
            Kind::Import { names } => {
                for &alias in names {
                    let Kind::Alias { name, asname } = ast.kind(alias) else { unreachable!() };
                    self.bind(Some(
                        asname.as_deref().unwrap_or_else(|| name.split('.').next().unwrap()),
                    ));
                }
            }
            Kind::ImportFrom { names, .. } => {
                for &alias in names {
                    let Kind::Alias { name, asname } = ast.kind(alias) else { unreachable!() };
                    self.bind(Some(asname.as_deref().unwrap_or(name)));
                }
            }
            Kind::ExceptHandler { name, .. } | Kind::MatchAs { name, .. } => {
                self.bind(name.as_deref());
                generic_visit_ro(self, ast, id);
            }
            Kind::MatchStar { name } => self.bind(name.as_deref()),
            Kind::MatchMapping { rest, .. } => {
                self.bind(rest.as_deref());
                generic_visit_ro(self, ast, id);
            }
            // A ``global`` statement lets a function rebind the module name through a
            // plain assignment; the declaration is counted too, so the name is never trusted
            Kind::Global { names } | Kind::Nonlocal { names } => {
                for name in names {
                    self.bind(Some(name));
                }
            }
            _ => generic_visit_ro(self, ast, id),
        }
    }
}

struct TypeErasure {
    cast_names: HashSet<String>,
    flag_names: HashSet<String>,
    typing_names: HashSet<String>,
}

impl TypeErasure {
    fn is_typing_attribute(&self, ast: &Ast, id: NodeId, attribute: &str) -> bool {
        match ast.kind(id) {
            Kind::Attribute { value, attr, .. } if attr == attribute => {
                matches!(ast.kind(*value), Kind::Name { id, .. } if self.typing_names.contains(id))
            }
            _ => false,
        }
    }

    fn is_cast(&self, ast: &Ast, func: NodeId) -> bool {
        match ast.kind(func) {
            Kind::Name { id, .. } => self.cast_names.contains(id),
            _ => self.is_typing_attribute(ast, func, "cast"),
        }
    }

    fn is_flag(&self, ast: &Ast, test: NodeId) -> bool {
        match ast.kind(test) {
            Kind::Name { id, .. } => self.flag_names.contains(id),
            _ => self.is_typing_attribute(ast, test, "TYPE_CHECKING"),
        }
    }
}

impl Transformer for TypeErasure {
    fn visit(&mut self, ast: &mut Ast, id: NodeId) -> Visited {
        match ast.kind(id) {
            Kind::If { test, orelse, .. } if self.is_flag(ast, *test) => {
                if orelse.is_empty() {
                    // A statement list may not become empty, and the enclosing one
                    // is not known here
                    let pass = ast.add(Kind::Pass);
                    copy_location(ast, pass, id);
                    return Visited::Node(pass);
                }
                let mut kept = Vec::new();
                for stmt in orelse.clone() {
                    match self.visit(ast, stmt) {
                        Visited::Node(new) => kept.push(new),
                        Visited::Remove => {}
                        Visited::Many(many) => kept.extend(many),
                    }
                }
                Visited::Many(kept)
            }
            Kind::Call { .. } => {
                generic_visit(self, ast, id);
                let Kind::Call { func, args, keywords } = ast.kind(id) else { unreachable!() };
                if self.is_cast(ast, *func)
                    && args.len() == 2
                    && keywords.is_empty()
                    && !matches!(ast.kind(args[0]), Kind::Starred { .. })
                    && !matches!(ast.kind(args[1]), Kind::Starred { .. })
                    && is_type_expression(ast, args[0])
                {
                    return Visited::Node(args[1]);
                }
                Visited::Node(id)
            }
            _ => {
                generic_visit(self, ast, id);
                Visited::Node(id)
            }
        }
    }
}

pub fn run(ast: &mut Ast, module: NodeId, _ctx: &Context<'_>) -> Result<NodeId, Error> {
    let Kind::Module { body, .. } = ast.kind(module) else {
        return Err(Error::Unsupported("TypeErasure on a non-module".into()));
    };
    let mut cast_names: HashMap<String, usize> = HashMap::new();
    let mut flag_names: HashMap<String, usize> = HashMap::new();
    let mut typing_names: HashMap<String, usize> = HashMap::new();
    for &stmt in body {
        match ast.kind(stmt) {
            Kind::ImportFrom { module: Some(name), names, level }
                if name == "typing" && *level == Some(0) =>
            {
                for &alias in names {
                    let Kind::Alias { name, asname } = ast.kind(alias) else { unreachable!() };
                    let bound = asname.clone().unwrap_or_else(|| name.clone());
                    if name == "cast" {
                        *cast_names.entry(bound).or_default() += 1;
                    } else if name == "TYPE_CHECKING" {
                        *flag_names.entry(bound).or_default() += 1;
                    }
                }
            }
            Kind::Import { names } => {
                for &alias in names {
                    let Kind::Alias { name, asname } = ast.kind(alias) else { unreachable!() };
                    if name == "typing" {
                        let bound = asname.clone().unwrap_or_else(|| name.clone());
                        *typing_names.entry(bound).or_default() += 1;
                    }
                }
            }
            _ => {}
        }
    }
    if cast_names.is_empty() && flag_names.is_empty() && typing_names.is_empty() {
        return Ok(module);
    }
    let mut counter = BindingCounter::default();
    counter.visit(ast, module);
    // Trusted: every binding of the name in the whole module is one of the typing
    // imports counted above
    let trusted = |names: HashMap<String, usize>| -> HashSet<String> {
        names
            .into_iter()
            .filter(|(name, count)| counter.bound.get(name) == Some(count))
            .map(|(name, _)| name)
            .collect()
    };
    let mut erasure = TypeErasure {
        cast_names: trusted(cast_names),
        flag_names: trusted(flag_names),
        typing_names: trusted(typing_names),
    };
    if erasure.cast_names.is_empty()
        && erasure.flag_names.is_empty()
        && erasure.typing_names.is_empty()
    {
        return Ok(module);
    }
    generic_visit(&mut erasure, ast, module);
    Ok(module)
}
