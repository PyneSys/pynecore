//! Port of `transformers/type_checking_stripper.py`: remove `if TYPE_CHECKING:` blocks
//! and the `TYPE_CHECKING` import; any other read of the removed flag becomes `False`.
//!
//! A name means the flag only where it resolves to the flag's binding: a scope that
//! binds the name itself keeps its `if` and its reads, and a name the module rebinds
//! at module level keeps its import as well. The same holds for `typing` in a
//! qualified `typing.TYPE_CHECKING` guard.

use std::collections::HashSet;

use pyne_ast::walk::{
    Transformer, Visited, children, copy_location, generic_visit, walk, walk_statements,
};
use pyne_ast::{Ast, Constant, ExprContext, Kind, NodeId};

use crate::{Context, Unsupported};

type Names = HashSet<String>;

/// The module name a qualified `typing.TYPE_CHECKING` guard reads the flag through
const TYPING: &str = "typing";

fn is_comprehension(kind: &Kind) -> bool {
    matches!(
        kind,
        Kind::ListComp { .. }
            | Kind::SetComp { .. }
            | Kind::GeneratorExp { .. }
            | Kind::DictComp { .. }
    )
}

fn is_expr(kind: &Kind) -> bool {
    kind.class().category == pyne_ast::Category::Expr
}

/// An `import` alias that binds the name `typing` to the `typing` module itself.
fn binds_typing_module(name: &str, asname: Option<&str>) -> bool {
    match asname {
        None => name.split('.').next() == Some(TYPING),
        Some(asname) => asname == TYPING && name == TYPING,
    }
}

fn alias(ast: &Ast, id: NodeId) -> (&str, Option<&str>) {
    let Kind::Alias { name, asname } = ast.kind(id) else { unreachable!("not an alias") };
    (name, asname.as_deref())
}

/// The names the parameters of a function or lambda bind.
fn argument_names(ast: &Ast, args: NodeId) -> Names {
    let Kind::Arguments { posonlyargs, args: positional, vararg, kwonlyargs, kwarg, .. } =
        ast.kind(args)
    else {
        unreachable!("not arguments")
    };
    let arg_name = |id: &NodeId| match ast.kind(*id) {
        Kind::Arg { arg, .. } => arg.clone(),
        _ => unreachable!("not an arg"),
    };
    let mut names: Names =
        posonlyargs.iter().chain(positional).chain(kwonlyargs).map(arg_name).collect();
    names.extend(vararg.iter().map(arg_name));
    names.extend(kwarg.iter().map(arg_name));
    names
}

/// The default values of a parameter list: they are evaluated in the enclosing scope.
fn argument_defaults(ast: &Ast, args: NodeId) -> Vec<NodeId> {
    let Kind::Arguments { defaults, kw_defaults, .. } = ast.kind(args) else {
        unreachable!("not arguments")
    };
    defaults.iter().copied().chain(kw_defaults.iter().flatten().copied()).collect()
}

/// The names a scope's own `from typing import TYPE_CHECKING` imports bind.
fn flag_imports(ast: &Ast, nodes: &[NodeId]) -> Names {
    let mut names = Names::new();
    let mut stack = nodes.to_vec();
    while let Some(id) = stack.pop() {
        let kind = ast.kind(id);
        if matches!(
            kind,
            Kind::FunctionDef { .. }
                | Kind::AsyncFunctionDef { .. }
                | Kind::ClassDef { .. }
                | Kind::Lambda { .. }
        ) || is_comprehension(kind)
        {
            continue;
        }
        if let Kind::ImportFrom { module: Some(module), names: aliases, .. } = kind
            && module == TYPING
        {
            for &item in aliases {
                let (name, asname) = alias(ast, item);
                if name == "TYPE_CHECKING" {
                    names.insert(asname.unwrap_or(name).to_string());
                }
            }
        } else if !is_expr(kind) {
            stack.extend(children(ast, id));
        }
    }
    names
}

/// The names any `nonlocal` declaration in the nodes or below them declares.
fn nonlocal_names(ast: &Ast, nodes: &[NodeId]) -> Names {
    let mut names = Names::new();
    for &node in nodes {
        for child in walk(ast, node) {
            if let Kind::Nonlocal { names: declared } = ast.kind(child) {
                names.extend(declared.iter().cloned());
            }
        }
    }
    names
}

/// The names one scope binds, and the names it declares `global` / `nonlocal`.
fn scope_bindings(ast: &Ast, nodes: &[NodeId]) -> (Names, Names, Names) {
    let mut bound = Names::new();
    let mut declared_global = Names::new();
    let mut declared_nonlocal = Names::new();
    let mut stack = nodes.to_vec();
    while let Some(id) = stack.pop() {
        match ast.kind(id) {
            Kind::FunctionDef { name, decorator_list, args, .. }
            | Kind::AsyncFunctionDef { name, decorator_list, args, .. } => {
                bound.insert(name.clone());
                stack.extend(decorator_list);
                stack.extend(argument_defaults(ast, *args));
                continue;
            }
            Kind::ClassDef { name, decorator_list, bases, keywords, .. } => {
                bound.insert(name.clone());
                stack.extend(decorator_list);
                stack.extend(bases);
                for &keyword in keywords {
                    let Kind::Keyword { value, .. } = ast.kind(keyword) else { unreachable!() };
                    stack.push(*value);
                }
                continue;
            }
            Kind::Lambda { args, .. } => {
                stack.extend(argument_defaults(ast, *args));
                continue;
            }
            kind if is_comprehension(kind) => {
                let generators = match kind {
                    Kind::ListComp { generators, .. }
                    | Kind::SetComp { generators, .. }
                    | Kind::GeneratorExp { generators, .. }
                    | Kind::DictComp { generators, .. } => generators,
                    _ => unreachable!(),
                };
                let Kind::Comprehension { iter, .. } = ast.kind(generators[0]) else {
                    unreachable!()
                };
                stack.push(*iter);
                // An assignment expression in a comprehension binds in the enclosing scope
                for child in walk(ast, id) {
                    if let Kind::NamedExpr { target, .. } = ast.kind(child)
                        && let Kind::Name { id: name, .. } = ast.kind(*target)
                    {
                        bound.insert(name.clone());
                    }
                }
                continue;
            }
            Kind::Name { id: name, ctx } => {
                if *ctx != ExprContext::Load {
                    bound.insert(name.clone());
                }
            }
            Kind::Import { names } => {
                for &item in names {
                    let (name, asname) = alias(ast, item);
                    if !binds_typing_module(name, asname) {
                        let bound_name = asname.unwrap_or_else(|| name.split('.').next().unwrap());
                        bound.insert(bound_name.to_string());
                    }
                }
            }
            Kind::ImportFrom { names, .. } => {
                for &item in names {
                    let (name, asname) = alias(ast, item);
                    if name != "TYPE_CHECKING" {
                        bound.insert(asname.unwrap_or(name).to_string());
                    }
                }
            }
            Kind::ExceptHandler { name: Some(name), .. } => {
                bound.insert(name.clone());
            }
            Kind::MatchAs { name: Some(name), .. } | Kind::MatchStar { name: Some(name) } => {
                bound.insert(name.clone());
            }
            Kind::MatchMapping { rest: Some(rest), .. } => {
                bound.insert(rest.clone());
            }
            Kind::Global { names } => declared_global.extend(names.iter().cloned()),
            Kind::Nonlocal { names } => declared_nonlocal.extend(names.iter().cloned()),
            _ => {}
        }
        stack.extend(children(ast, id));
    }
    (bound, declared_global, declared_nonlocal)
}

struct Stripper {
    /// The names the flag goes by: `TYPE_CHECKING` and every alias of it
    names: Names,
    /// The names a scope can shadow: the flag names and `typing`
    tracked: Names,
    /// The names the removed `typing` import bound
    imported: Names,
    /// The flag names the module binds at module level besides the flag import
    unflagged: Names,
    /// The flag names that do not mean the flag in the scope being visited
    shadow: Names,
    /// The same for a function defined in the scope being visited
    function_shadow: Names,
}

impl Stripper {
    /// The flag names that do not mean the flag in a nested scope.
    fn scope_shadow(
        &self,
        ast: &Ast,
        outer: &Names,
        nodes: &[NodeId],
        params: Option<Names>,
        function: bool,
    ) -> Names {
        let (mut bound, declared_global, declared_nonlocal) = scope_bindings(ast, nodes);
        if let Some(params) = params {
            bound.extend(params);
        }
        if function {
            // A nested ``nonlocal`` needs the binding to survive: the import stays.
            // A nested function that rebinds ``typing`` makes it unknown here
            let mut flags = flag_imports(ast, nodes);
            flags.insert(TYPING.to_string());
            let nonlocal = nonlocal_names(ast, nodes);
            bound.extend(flags.intersection(&nonlocal).cloned());
        }
        let local: Names = bound
            .union(&declared_nonlocal)
            .filter(|name| self.tracked.contains(*name))
            .cloned()
            .collect();
        let mut result: Names =
            outer.union(&local).filter(|name| !declared_global.contains(*name)).cloned().collect();
        result.extend(declared_global.intersection(&self.unflagged).cloned());
        result
    }

    /// Visit a statement list, splicing what each visit returns as `generic_visit` does.
    fn visit_block(&mut self, ast: &mut Ast, stmts: Vec<NodeId>) -> Vec<NodeId> {
        let mut result = Vec::with_capacity(stmts.len());
        for stmt in stmts {
            match self.visit(ast, stmt) {
                Visited::Node(new) => result.push(new),
                Visited::Remove => {}
                Visited::Many(many) => result.extend(many),
            }
        }
        result
    }

    fn visit_expr(&mut self, ast: &mut Ast, id: NodeId) -> NodeId {
        match self.visit(ast, id) {
            Visited::Node(new) => new,
            other => panic!("an expression visit answered {other:?}"),
        }
    }

    fn visit_exprs(&mut self, ast: &mut Ast, ids: Vec<NodeId>) -> Vec<NodeId> {
        ids.into_iter().map(|id| self.visit_expr(ast, id)).collect()
    }

    fn visit_function(&mut self, ast: &mut Ast, id: NodeId) {
        // Decorators, defaults and annotations are evaluated in the enclosing scope
        let (decorators, args, returns, body) = match ast.kind(id) {
            Kind::FunctionDef { decorator_list, args, returns, body, .. }
            | Kind::AsyncFunctionDef { decorator_list, args, returns, body, .. } => {
                (decorator_list.clone(), *args, *returns, body.clone())
            }
            _ => unreachable!(),
        };
        let decorators = self.visit_exprs(ast, decorators);
        let args = self.visit_expr(ast, args);
        let returns = returns.map(|returns| self.visit_expr(ast, returns));
        set_function(ast, id, |d, a, r, _| {
            *d = decorators;
            *a = args;
            *r = returns;
        });
        let saved = (self.shadow.clone(), self.function_shadow.clone());
        let params = argument_names(ast, args);
        let outer = self.function_shadow.clone();
        let shadow = self.scope_shadow(ast, &outer, &body, Some(params), true);
        self.shadow = shadow.clone();
        self.function_shadow = shadow;
        let body = self.visit_block(ast, body);
        set_function(ast, id, |_, _, _, b| *b = body);
        (self.shadow, self.function_shadow) = saved;
    }

    fn visit_lambda(&mut self, ast: &mut Ast, id: NodeId) {
        let Kind::Lambda { args, body } = *ast.kind(id) else { unreachable!() };
        let args = self.visit_expr(ast, args);
        let saved = (self.shadow.clone(), self.function_shadow.clone());
        let params = argument_names(ast, args);
        let outer = self.function_shadow.clone();
        let shadow = self.scope_shadow(ast, &outer, &[body], Some(params), false);
        self.shadow = shadow.clone();
        self.function_shadow = shadow;
        let body = self.visit_expr(ast, body);
        *ast.kind_mut(id) = Kind::Lambda { args, body };
        (self.shadow, self.function_shadow) = saved;
    }

    fn visit_class(&mut self, ast: &mut Ast, id: NodeId) {
        let Kind::ClassDef { decorator_list, bases, keywords, body, .. } = ast.kind(id) else {
            unreachable!()
        };
        let (decorators, bases, keywords, body) =
            (decorator_list.clone(), bases.clone(), keywords.clone(), body.clone());
        let decorators = self.visit_exprs(ast, decorators);
        let bases = self.visit_exprs(ast, bases);
        let keywords = self.visit_exprs(ast, keywords);
        if let Kind::ClassDef { decorator_list, bases: b, keywords: k, .. } = ast.kind_mut(id) {
            *decorator_list = decorators;
            *b = bases;
            *k = keywords;
        }
        let saved = self.shadow.clone();
        let outer = self.shadow.clone();
        self.shadow = self.scope_shadow(ast, &outer, &body, None, false);
        let body = self.visit_block(ast, body);
        if let Kind::ClassDef { body: b, .. } = ast.kind_mut(id) {
            *b = body;
        }
        self.shadow = saved;
    }

    fn visit_comprehension(&mut self, ast: &mut Ast, id: NodeId) {
        let generators = match ast.kind(id) {
            Kind::ListComp { generators, .. }
            | Kind::SetComp { generators, .. }
            | Kind::GeneratorExp { generators, .. }
            | Kind::DictComp { generators, .. } => generators.clone(),
            _ => unreachable!(),
        };
        // The first iterable is evaluated in the enclosing scope
        let Kind::Comprehension { iter, .. } = *ast.kind(generators[0]) else { unreachable!() };
        let iter = self.visit_expr(ast, iter);
        if let Kind::Comprehension { iter: slot, .. } = ast.kind_mut(generators[0]) {
            *slot = iter;
        }
        let mut targets = Names::new();
        for &generator in &generators {
            let Kind::Comprehension { target, .. } = *ast.kind(generator) else { unreachable!() };
            for child in walk(ast, target) {
                if let Kind::Name { id: name, .. } = ast.kind(child) {
                    targets.insert(name.clone());
                }
            }
        }
        // A comprehension is a function scope: the class body around it does not
        // reach it, and its loop targets reach the lambdas defined in it
        let saved = (self.shadow.clone(), self.function_shadow.clone());
        let mut shadow = self.function_shadow.clone();
        shadow.extend(targets.intersection(&self.tracked).cloned());
        self.shadow = shadow.clone();
        self.function_shadow = shadow;
        for (index, &generator) in generators.iter().enumerate() {
            let Kind::Comprehension { target, iter, ifs, .. } = ast.kind(generator).clone() else {
                unreachable!()
            };
            let target = self.visit_expr(ast, target);
            let iter = if index > 0 { self.visit_expr(ast, iter) } else { iter };
            let ifs = self.visit_exprs(ast, ifs);
            if let Kind::Comprehension { target: t, iter: i, ifs: f, .. } = ast.kind_mut(generator)
            {
                *t = target;
                *i = iter;
                *f = ifs;
            }
        }
        match ast.kind(id).clone() {
            Kind::DictComp { key, value, .. } => {
                let key = self.visit_expr(ast, key);
                let value = self.visit_expr(ast, value);
                if let Kind::DictComp { key: k, value: v, .. } = ast.kind_mut(id) {
                    *k = key;
                    *v = value;
                }
            }
            Kind::ListComp { elt, .. }
            | Kind::SetComp { elt, .. }
            | Kind::GeneratorExp { elt, .. } => {
                let elt = self.visit_expr(ast, elt);
                match ast.kind_mut(id) {
                    Kind::ListComp { elt: e, .. }
                    | Kind::SetComp { elt: e, .. }
                    | Kind::GeneratorExp { elt: e, .. } => *e = elt,
                    _ => unreachable!(),
                }
            }
            _ => unreachable!(),
        }
        (self.shadow, self.function_shadow) = saved;
    }

    /// Match: `TYPE_CHECKING` (or an alias of it) / `typing.TYPE_CHECKING`.
    fn is_type_checking(&self, ast: &Ast, test: NodeId) -> bool {
        match ast.kind(test) {
            Kind::Name { id, .. } => self.names.contains(id) && !self.shadow.contains(id),
            Kind::Attribute { value, attr, .. } => {
                attr == "TYPE_CHECKING"
                    && matches!(ast.kind(*value), Kind::Name { id, .. } if id == TYPING)
                    && !self.shadow.contains(TYPING)
            }
            _ => false,
        }
    }
}

fn set_function(
    ast: &mut Ast,
    id: NodeId,
    set: impl FnOnce(&mut Vec<NodeId>, &mut NodeId, &mut Option<NodeId>, &mut Vec<NodeId>),
) {
    match ast.kind_mut(id) {
        Kind::FunctionDef { decorator_list, args, returns, body, .. }
        | Kind::AsyncFunctionDef { decorator_list, args, returns, body, .. } => {
            set(decorator_list, args, returns, body)
        }
        _ => unreachable!(),
    }
}

impl Transformer for Stripper {
    fn visit(&mut self, ast: &mut Ast, id: NodeId) -> Visited {
        match ast.kind(id) {
            Kind::FunctionDef { .. } | Kind::AsyncFunctionDef { .. } => {
                self.visit_function(ast, id)
            }
            Kind::Lambda { .. } => self.visit_lambda(ast, id),
            Kind::ClassDef { .. } => self.visit_class(ast, id),
            kind if is_comprehension(kind) => self.visit_comprehension(ast, id),
            Kind::Name { id: name, ctx } => {
                // Every other read of the flag outlives the removed import, so it
                // is the runtime value
                if self.imported.contains(name)
                    && *ctx == ExprContext::Load
                    && !self.shadow.contains(name)
                {
                    let constant =
                        ast.add(Kind::Constant { value: Constant::Bool(false), kind: None });
                    copy_location(ast, constant, id);
                    return Visited::Node(constant);
                }
            }
            Kind::If { test, orelse, .. } => {
                if self.is_type_checking(ast, *test) {
                    // The name is False at runtime: what runs is the else body
                    let orelse = orelse.clone();
                    let block = self.visit_block(ast, orelse);
                    return if block.is_empty() { Visited::Remove } else { Visited::Many(block) };
                }
                generic_visit(self, ast, id);
            }
            Kind::ImportFrom { module: Some(module), names, .. } if module == TYPING => {
                let kept: Vec<NodeId> = names
                    .iter()
                    .copied()
                    .filter(|&item| {
                        let (name, asname) = alias(ast, item);
                        name != "TYPE_CHECKING" || self.shadow.contains(asname.unwrap_or(name))
                    })
                    .collect();
                let empty = kept.is_empty();
                if let Kind::ImportFrom { names, .. } = ast.kind_mut(id) {
                    *names = kept;
                }
                if empty {
                    return Visited::Remove;
                }
            }
            _ => generic_visit(self, ast, id),
        }
        Visited::Node(id)
    }
}

pub fn run(ast: &mut Ast, module: NodeId, _ctx: &Context<'_>) -> Result<NodeId, Unsupported> {
    let mut stripper = Stripper {
        names: Names::from(["TYPE_CHECKING".to_string()]),
        tracked: Names::from(["TYPE_CHECKING".to_string(), TYPING.to_string()]),
        imported: Names::new(),
        unflagged: Names::new(),
        shadow: Names::new(),
        function_shadow: Names::new(),
    };
    // Both forms this pass rewrites are statements: a module without either is
    // left as it is, without visiting a single expression
    let mut relevant = false;
    for stmt in walk_statements(ast, module) {
        match ast.kind(stmt) {
            Kind::ImportFrom { module: Some(name), names, .. } if name == TYPING => {
                for &item in names {
                    let (name, asname) = alias(ast, item);
                    if name == "TYPE_CHECKING" {
                        relevant = true;
                        if let Some(asname) = asname {
                            stripper.names.insert(asname.to_string());
                        }
                        stripper.imported.insert(asname.unwrap_or(name).to_string());
                    }
                }
            }
            Kind::If { test, .. } if stripper.is_type_checking(ast, *test) => relevant = true,
            _ => {}
        }
    }
    if !relevant {
        return Ok(module);
    }
    stripper.tracked = stripper.names.clone();
    stripper.tracked.insert(TYPING.to_string());
    // A name the module binds itself is not (only) the flag
    let Kind::Module { body, .. } = ast.kind(module) else {
        return Err(Unsupported("TypeCheckingStripper on a non-module".into()));
    };
    let (mut bound, _, _) = scope_bindings(ast, &body.clone());
    for stmt in walk_statements(ast, module) {
        if let Kind::Global { names } = ast.kind(stmt) {
            bound.extend(names.iter().cloned());
        }
    }
    stripper.unflagged = bound.intersection(&stripper.tracked).cloned().collect();
    stripper.shadow = stripper.unflagged.clone();
    stripper.function_shadow = stripper.unflagged.clone();
    generic_visit(&mut stripper, ast, module);
    Ok(module)
}
