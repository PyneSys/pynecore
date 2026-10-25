//! Port of `transformers/persistent.py`: Persistent declarations and accesses become
//! state-vector slots.
//!
//! - `p: Persistent[float] = 0.0` -> slot allocated, declaration removed (the
//!   initial value moves into the layout's init template),
//! - non-literal initializers keep the lazy pattern: a value slot plus a flag slot
//!   and an `if not __state__[flag]: ...` guard at the declaration site,
//! - reads/writes become `__state__[N]`,
//! - `+=` stays a plain augmented assignment on the slot,
//! - a walrus write to a persistent is emitted through `__state__.__setitem__`
//!   inside a tuple expression, preserving the value.
//!
//! Nested definitions see parent persistents (resolved to the parent's state vector
//! through a closure reference on the parent's scope-qualified state parameter), and
//! a plain local assignment shadows a parent persistent of the same name.

use std::collections::{HashMap, HashSet};

use pyne_ast::walk::{Transformer, Visited, copy_location, generic_visit};
use pyne_ast::{Ast, Constant, ExprContext, Int, Kind, NodeId, UnaryOperator};

use crate::slot_layout::{ModuleLayout, scope_for_function};
use crate::{Context, Error};

const PERSISTENT_TYPES: &[&str] = &["Persistent", "IBPersistent", "IBPersistentSeries"];
const VARIP_TYPES: &[&str] = &["IBPersistent", "IBPersistentSeries"];

/// Whether an annotation names one of `types`: `T`, `T[...]` or `x.T`.
fn annotation_in(ast: &Ast, annotation: NodeId, types: &[&str]) -> bool {
    match ast.kind(annotation) {
        Kind::Name { id, .. } => types.contains(&id.as_str()),
        Kind::Subscript { value, .. } => {
            matches!(ast.kind(*value), Kind::Name { id, .. } if types.contains(&id.as_str()))
        }
        Kind::Attribute { attr, .. } => types.contains(&attr.as_str()),
        _ => false,
    }
}

/// Whether a node is a literal value or `na`.
fn is_literal_or_na(ast: &Ast, node: NodeId) -> bool {
    match ast.kind(node) {
        Kind::Constant { .. } => true,
        Kind::Name { id, .. } => id == "na",
        _ => false,
    }
}

/// The scalar a persistent declares (`Persistent[int]` -> `int`), else `None`.
fn declared_elem(ast: &Ast, annotation: NodeId) -> Option<String> {
    let Kind::Subscript { slice, .. } = ast.kind(annotation) else { return None };
    match ast.kind(*slice) {
        Kind::Name { id, .. } if matches!(id.as_str(), "int" | "float" | "bool" | "str") => {
            Some(id.clone())
        }
        _ => None,
    }
}

/// `float(n)` of a Python int literal.
fn int_to_float(value: &Int, negate: bool) -> Result<f64, Error> {
    match value {
        // `-n` is taken on the int, so `-0` is `0.0`; an i64 converts to the
        // nearest double, ties to even, as `float(int)` does
        Int::Small(n) => match if negate { n.checked_neg() } else { Some(*n) } {
            Some(n) => Ok(n as f64),
            None => Err(Error::Unsupported(format!("float(-{n})"))),
        },
        Int::Big(digits) => Err(Error::Unsupported(format!("float({digits})"))),
    }
}

fn name(ast: &mut Ast, id: &str, ctx: ExprContext) -> NodeId {
    ast.add(Kind::Name { id: id.to_string(), ctx })
}

fn constant(ast: &mut Ast, value: Constant) -> NodeId {
    ast.add(Kind::Constant { value, kind: None })
}

fn assign(ast: &mut Ast, target: NodeId, value: NodeId) -> NodeId {
    ast.add(Kind::Assign { targets: vec![target], value, type_comment: None })
}

struct Persistent<'l> {
    layout: &'l mut ModuleLayout,
    scope_stack: Vec<String>,
    current_scope: String,
    /// Scope -> var name -> value slot
    var_slots: HashMap<String, HashMap<String, usize>>,
    /// Every name `var_slots` holds in any scope: a name outside it is no persistent
    /// anywhere, which settles most lookups at once
    slot_names: HashSet<String>,
    persistent_declarations: HashMap<String, HashSet<String>>,
    local_vars: HashMap<String, HashSet<String>>,
    error: Option<Error>,
}

impl Persistent<'_> {
    fn fail(&mut self, error: Error) {
        self.error.get_or_insert(error);
    }

    /// Visit an expression, which always answers a node.
    fn expr(&mut self, ast: &mut Ast, id: NodeId) -> NodeId {
        match self.visit(ast, id) {
            Visited::Node(new) => new,
            other => {
                self.fail(Error::Unsupported(format!("{other:?} for an expression")));
                id
            }
        }
    }

    /// Resolve a name to its declaring scope and value slot; a name locally declared
    /// in the current scope (but not as Persistent) shadows any parent persistent of
    /// the same name.
    fn lookup(&self, var_name: &str) -> Option<(String, usize)> {
        if !self.slot_names.contains(var_name) {
            return None;
        }
        let scope = &self.current_scope;
        if self.local_vars.get(scope).is_some_and(|names| names.contains(var_name))
            && !self
                .persistent_declarations
                .get(scope)
                .is_some_and(|names| names.contains(var_name))
        {
            return None;
        }
        if let Some(&slot) = self.var_slots.get(scope).and_then(|slots| slots.get(var_name)) {
            return Some((scope.clone(), slot));
        }
        for i in (1..self.scope_stack.len()).rev() {
            let scope = self.scope_stack[..i].join("·");
            if let Some(&slot) = self.var_slots.get(&scope).and_then(|slots| slots.get(var_name)) {
                return Some((scope, slot));
            }
        }
        None
    }

    /// `<state param>[slot]` of a scope, in the given context.
    fn state_ref(&self, ast: &mut Ast, scope: &str, slot: usize, ctx: ExprContext) -> NodeId {
        let param = name(ast, self.layout.state_param(scope), ExprContext::Load);
        let index = constant(ast, Constant::Int(Int::Small(slot as i64)));
        ast.add(Kind::Subscript { value: param, slice: index, ctx })
    }

    /// The initializer of a numeric persistent, as the value the slot must hold: an
    /// int literal initializing an int- or float-typed persistent becomes the float
    /// literal (a Pine int is a double at runtime). A `@pyne lib` module keeps the
    /// Python int: its persistents are internal counters and ring indexes.
    fn numeric_init(
        &self,
        ast: &mut Ast,
        annotation: NodeId,
        value: NodeId,
    ) -> Result<NodeId, Error> {
        if self.layout.compacted_series
            || !declared_elem(ast, annotation).is_some_and(|elem| elem == "int" || elem == "float")
        {
            return Ok(value);
        }
        let float = match ast.kind(value) {
            Kind::Constant { value: Constant::Int(n), .. } => int_to_float(n, false)?,
            Kind::UnaryOp { op: op @ (UnaryOperator::USub | UnaryOperator::UAdd), operand } => {
                match ast.kind(*operand) {
                    Kind::Constant { value: Constant::Int(n), .. } => {
                        int_to_float(n, *op == UnaryOperator::USub)?
                    }
                    _ => return Ok(value),
                }
            }
            _ => return Ok(value),
        };
        let new = constant(ast, Constant::Float(float));
        copy_location(ast, new, value);
        Ok(new)
    }

    /// Strip the Persistent name from pynecore imports.
    fn visit_import_from(&mut self, ast: &mut Ast, id: NodeId) -> Visited {
        let Kind::ImportFrom { module, names, .. } = ast.kind(id) else { unreachable!() };
        if !module.as_deref().is_some_and(|module| module.starts_with("pynecore")) {
            return Visited::Node(id);
        }
        let kept: Vec<NodeId> = names
            .iter()
            .copied()
            .filter(|&alias| {
                !matches!(ast.kind(alias), Kind::Alias { name, .. } if name == "Persistent")
            })
            .collect();
        if kept.is_empty() {
            return Visited::Remove;
        }
        let Kind::ImportFrom { names, .. } = ast.kind_mut(id) else { unreachable!() };
        *names = kept;
        Visited::Node(id)
    }

    /// Track scopes, qualify the state parameter when nested defs exist.
    fn visit_function_def(&mut self, ast: &mut Ast, id: NodeId) -> Visited {
        let segment = self.layout.scope_segment(ast, id);
        self.scope_stack.push(segment);
        self.current_scope = self.scope_stack.join("·");
        let scope = self.current_scope.clone();

        scope_for_function(self.layout, ast, &scope, id);

        self.persistent_declarations.entry(scope.clone()).or_default();
        let locals = self.local_vars.entry(scope).or_default();
        let Kind::FunctionDef { args, .. } = ast.kind(id) else { unreachable!() };
        let Kind::Arguments { args: params, .. } = ast.kind(*args) else { unreachable!() };
        for &param in params {
            let Kind::Arg { arg, .. } = ast.kind(param) else { unreachable!() };
            locals.insert(arg.clone());
        }

        generic_visit(self, ast, id);

        // Persistent names are not closure variables anymore -- drop them from
        // nonlocal statements (shadowed locals keep theirs).
        let mut ancestors: HashSet<String> = HashSet::new();
        for i in (1..self.scope_stack.len()).rev() {
            let scope = self.scope_stack[..i].join("·");
            if let Some(names) = self.persistent_declarations.get(&scope) {
                ancestors.extend(names.iter().cloned());
            }
        }
        let Kind::FunctionDef { body, .. } = ast.kind(id) else { unreachable!() };
        let mut new_body = Vec::with_capacity(body.len());
        for stmt in body.clone() {
            if let Kind::Nonlocal { names } = ast.kind_mut(stmt) {
                names.retain(|name| !ancestors.contains(name));
                if names.is_empty() {
                    continue;
                }
            }
            new_body.push(stmt);
        }
        let Kind::FunctionDef { body, .. } = ast.kind_mut(id) else { unreachable!() };
        *body = new_body;

        self.scope_stack.pop();
        self.current_scope = self.scope_stack.join("·");
        Visited::Node(id)
    }

    /// Convert Persistent declarations into slot allocations.
    fn visit_ann_assign(&mut self, ast: &mut Ast, id: NodeId) -> Result<Visited, Error> {
        let Kind::AnnAssign { target, annotation, value, .. } = ast.kind(id) else {
            unreachable!()
        };
        let (target, annotation, value) = (*target, *annotation, *value);
        let target_name = match ast.kind(target) {
            Kind::Name { id, .. } => Some(id.clone()),
            _ => None,
        };
        let Some(var_name) =
            target_name.clone().filter(|_| annotation_in(ast, annotation, PERSISTENT_TYPES))
        else {
            if let Some(target_name) = target_name
                && !self.current_scope.is_empty()
            {
                // An annotated assignment declares a local -- it shadows a same-named
                // parent persistent, like a plain assignment does.
                self.local_vars.entry(self.current_scope.clone()).or_default().insert(target_name);
            }
            if let Some(value) = value {
                let new = self.expr(ast, value);
                let Kind::AnnAssign { value, .. } = ast.kind_mut(id) else { unreachable!() };
                *value = Some(new);
            }
            return Ok(Visited::Node(id));
        };
        if self.current_scope.is_empty() {
            return Err(Error::Rejected(
                "Persistent variables must be declared inside a function".into(),
            ));
        }

        let scope = self.current_scope.clone();
        // The Python pass indexes these two directly: a scope without them is a KeyError
        let (Some(declared), Some(locals)) =
            (self.persistent_declarations.get_mut(&scope), self.local_vars.get_mut(&scope))
        else {
            return Err(Error::Unsupported(format!("a persistent declaration in scope {scope:?}")));
        };
        declared.insert(var_name.clone());
        locals.insert(var_name.clone());
        let varip = annotation_in(ast, annotation, VARIP_TYPES);
        let elem = declared_elem(ast, annotation);

        let value = match value {
            Some(value) => {
                let value = self.numeric_init(ast, annotation, value)?;
                let Kind::AnnAssign { value: slot, .. } = ast.kind_mut(id) else { unreachable!() };
                *slot = Some(value);
                Some(value)
            }
            None => None,
        };
        if let Some(value) = value
            && !is_literal_or_na(ast, value)
        {
            // Lazy pattern: value slot + flag slot, initializer runs on first call
            let placeholder = constant(ast, Constant::None);
            let scope_layout = self.layout.scope(&scope);
            let slot = scope_layout.add_var(&var_name, placeholder, varip, elem.as_deref());
            self.var_slots.entry(scope.clone()).or_default().insert(var_name.clone(), slot);
            self.slot_names.insert(var_name.clone());
            let flag = self.layout.scope(&scope).add_flag(ast, &var_name, varip);
            let value = self.expr(ast, value);
            let operand = self.state_ref(ast, &scope, flag, ExprContext::Load);
            let test = ast.add(Kind::UnaryOp { op: UnaryOperator::Not, operand });
            let target = self.state_ref(ast, &scope, slot, ExprContext::Store);
            let store = assign(ast, target, value);
            let target = self.state_ref(ast, &scope, flag, ExprContext::Store);
            let done = constant(ast, Constant::Bool(true));
            let mark = assign(ast, target, done);
            let lazy = ast.add(Kind::If { test, body: vec![store, mark], orelse: Vec::new() });
            copy_location(ast, lazy, id);
            return Ok(Visited::Node(lazy));
        }

        // A bare-`na` default must be the na VALUE, not the `na` name -- at module
        // level `na` resolves to the `is_na` function object. Emit `lib.na()`.
        let init = match value {
            Some(value) => value,
            None => {
                let lib = name(ast, "lib", ExprContext::Load);
                let func = ast.add(Kind::Attribute {
                    value: lib,
                    attr: "na".into(),
                    ctx: ExprContext::Load,
                });
                ast.add(Kind::Call { func, args: Vec::new(), keywords: Vec::new() })
            }
        };
        let slot = self.layout.scope(&scope).add_var(&var_name, init, varip, elem.as_deref());
        self.var_slots.entry(scope).or_default().insert(var_name.clone(), slot);
        self.slot_names.insert(var_name);
        Ok(Visited::Remove)
    }

    /// Convert assignments to persistent variables into slot writes.
    fn visit_assign(&mut self, ast: &mut Ast, id: NodeId) -> Visited {
        let Kind::Assign { targets, value, .. } = ast.kind(id) else { unreachable!() };
        let (targets, value) = (targets.clone(), *value);
        if let [target] = targets[..]
            && let Kind::Name { id: var_name, .. } = ast.kind(target)
        {
            let var_name = var_name.clone();
            // The first plain assignment to a not-yet-known name declares a local,
            // shadowing a same-named parent persistent.
            self.local_vars.entry(self.current_scope.clone()).or_default().insert(var_name.clone());
            if let Some((scope, slot)) = self.lookup(&var_name) {
                let target = self.state_ref(ast, &scope, slot, ExprContext::Store);
                let value = self.expr(ast, value);
                let assign = assign(ast, target, value);
                copy_location(ast, assign, id);
                return Visited::Node(assign);
            }
        }
        let new_targets: Vec<NodeId> =
            targets.into_iter().map(|target| self.expr(ast, target)).collect();
        let new_value = self.expr(ast, value);
        let Kind::Assign { targets, value, .. } = ast.kind_mut(id) else { unreachable!() };
        *targets = new_targets;
        *value = new_value;
        Visited::Node(id)
    }

    /// Slot-target augmented assignment.
    fn visit_aug_assign(&mut self, ast: &mut Ast, id: NodeId) -> Visited {
        let Kind::AugAssign { target, value, .. } = ast.kind(id) else { unreachable!() };
        let (target, value) = (*target, *value);
        if let Kind::Name { id: var_name, .. } = ast.kind(target)
            && let Some((scope, slot)) = self.lookup(var_name)
        {
            let new_target = self.state_ref(ast, &scope, slot, ExprContext::Store);
            let Kind::AugAssign { target, .. } = ast.kind_mut(id) else { unreachable!() };
            *target = new_target;
            let new_value = self.expr(ast, value);
            let Kind::AugAssign { value, .. } = ast.kind_mut(id) else { unreachable!() };
            *value = new_value;
            return Visited::Node(id);
        }
        generic_visit(self, ast, id);
        Visited::Node(id)
    }

    /// Walrus write to a persistent: route through `__setitem__` so the construct
    /// stays a valid expression and still yields the value.
    fn visit_named_expr(&mut self, ast: &mut Ast, id: NodeId) -> Visited {
        let Kind::NamedExpr { target, value } = ast.kind(id) else { unreachable!() };
        let (target, value) = (*target, *value);
        if let Kind::Name { id: var_name, .. } = ast.kind(target)
            && let Some((scope, slot)) = self.lookup(var_name)
        {
            let value = self.expr(ast, value);
            let param = self.layout.state_param(&scope).to_string();
            let holder = name(ast, "__pyne_w__", ExprContext::Store);
            let walrus = ast.add(Kind::NamedExpr { target: holder, value });
            let state = name(ast, &param, ExprContext::Load);
            let func = ast.add(Kind::Attribute {
                value: state,
                attr: "__setitem__".into(),
                ctx: ExprContext::Load,
            });
            let index = constant(ast, Constant::Int(Int::Small(slot as i64)));
            let held = name(ast, "__pyne_w__", ExprContext::Load);
            let call = ast.add(Kind::Call { func, args: vec![index, held], keywords: Vec::new() });
            let tuple = ast.add(Kind::Tuple { elts: vec![walrus, call], ctx: ExprContext::Load });
            let zero = constant(ast, Constant::Int(Int::Small(0)));
            return Visited::Node(ast.add(Kind::Subscript {
                value: tuple,
                slice: zero,
                ctx: ExprContext::Load,
            }));
        }
        generic_visit(self, ast, id);
        Visited::Node(id)
    }

    /// Convert persistent references using scope-aware lookup.
    fn visit_name(&mut self, ast: &mut Ast, id: NodeId) -> Visited {
        let Kind::Name { id: var_name, ctx } = ast.kind(id) else { unreachable!() };
        let ctx = *ctx;
        match self.lookup(var_name) {
            Some((scope, slot)) => Visited::Node(self.state_ref(ast, &scope, slot, ctx)),
            None => Visited::Node(id),
        }
    }
}

impl Transformer for Persistent<'_> {
    fn visit(&mut self, ast: &mut Ast, id: NodeId) -> Visited {
        if self.error.is_some() {
            return Visited::Node(id);
        }
        match ast.kind(id) {
            Kind::Module { .. } => {
                self.layout.assign_scope_ids(ast, id);
                generic_visit(self, ast, id);
                Visited::Node(id)
            }
            Kind::ImportFrom { .. } => self.visit_import_from(ast, id),
            Kind::FunctionDef { .. } => self.visit_function_def(ast, id),
            Kind::AnnAssign { .. } => self.visit_ann_assign(ast, id).unwrap_or_else(|error| {
                self.fail(error);
                Visited::Node(id)
            }),
            Kind::Assign { .. } => self.visit_assign(ast, id),
            Kind::AugAssign { .. } => self.visit_aug_assign(ast, id),
            Kind::NamedExpr { .. } => self.visit_named_expr(ast, id),
            Kind::Name { .. } => self.visit_name(ast, id),
            _ => {
                generic_visit(self, ast, id);
                Visited::Node(id)
            }
        }
    }
}

pub fn run(ast: &mut Ast, module: NodeId, ctx: &Context<'_>) -> Result<NodeId, Error> {
    if !matches!(ast.kind(module), Kind::Module { .. }) {
        return Err(Error::Unsupported("Persistent on a non-module".into()));
    }
    let mut layout = ctx.slot_layout.borrow_mut();
    let mut pass = Persistent {
        layout: &mut layout,
        scope_stack: Vec::new(),
        current_scope: String::new(),
        var_slots: HashMap::new(),
        slot_names: HashSet::new(),
        persistent_declarations: HashMap::new(),
        local_vars: HashMap::new(),
        error: None,
    };
    let answer = pass.visit(ast, module);
    if let Some(error) = pass.error {
        return Err(error);
    }
    match answer {
        Visited::Node(module) => Ok(module),
        other => Err(Error::Unsupported(format!("{other:?} for the module"))),
    }
}
