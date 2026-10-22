//! Port of `transformers/series.py`: Series declarations and accesses become
//! state-vector slots.
//!
//! A series variable's circular buffer lives in a compile-time-assigned slot of its
//! scope's state vector, addressed with a literal index:
//!
//! - `s: Series[float] = value` -> `s = __state__[N].add(value)` (the local name
//!   keeps tracking the current scalar value, plain reads stay untouched),
//! - `s = value` -> `s = __state__[N].set(value)`,
//! - `s += value` -> `s = __state__[N].set(s + value)`,
//! - `s[idx]` -> `__state__[N][idx]`,
//! - `lib.max_bars_back(s, num)` in statement position -> `__state__[N].max_bars_back = num`,
//! - a `Series`-annotated parameter loses the Series wrapper from its annotation and
//!   gets `s = __state__[N].add(s)` prepended to the body.
//!
//! Subscript READS resolve through the scope chain: a nested definition reaches a
//! parent's series buffer through a closure reference on the parent's
//! scope-qualified state parameter. Writes stay same-scope only: a plain assignment
//! in a nested scope declares a local that shadows the parent's series.

use std::collections::{HashMap, HashSet};

use pyne_ast::walk::{Transformer, Visited, children, generic_visit};
use pyne_ast::{Ast, Category, Constant, ExprContext, Int, Kind, NodeId, Value};

use crate::dynamic_default_guard::is_dynamic_default_guard;
use crate::pine_type_rules::{OBJECT, TY_ATTR, UNKNOWN, stamp_lowering};
use crate::slot_layout::{ModuleLayout, scope_for_function};
use crate::{Context, Error};

/// `lib` no-ops that address a series buffer itself instead of its value, mapped to
/// the buffer attribute they set. Only a statement-position call on a series NAME is
/// converted; every other position is left to the runtime no-op in `pynecore.lib`.
const BUFFER_SETTERS: &[(&str, &str)] = &[("max_bars_back", "max_bars_back")];

fn is_expr(ast: &Ast, id: NodeId) -> bool {
    ast.kind(id).class().category == Category::Expr
}

/// A node's `_pine_ty` stamp as `get_ty` reads it: the value itself, `UNKNOWN` when
/// the node carries none.
fn ty_value(ast: &Ast, id: NodeId) -> Value {
    ast.extra(id, TY_ATTR).cloned().unwrap_or_else(|| Value::Str(UNKNOWN.to_string()))
}

/// `Unsupported` when typing a freshly built subtree (`stamp_lowering`) would meet a
/// `_pine_ty` stamp that is neither a string nor `None`: the Python pass keeps such a
/// stamp as it is, the native one reads only strings. Walks every expression the
/// typing could enter: the unstamped ones, from the root's children down.
fn check_lowering_stamps(ast: &Ast, root: NodeId) -> Result<(), Error> {
    fn check(ast: &Ast, id: NodeId) -> Result<(), Error> {
        match ast.extra(id, TY_ATTR) {
            Some(Value::Str(_)) => return Ok(()),
            None | Some(Value::None) => {}
            Some(other) => {
                return Err(Error::Unsupported(format!("a {TY_ATTR} stamp {other:?}")));
            }
        }
        if matches!(ast.kind(id), Kind::Constant { .. }) {
            return Ok(());
        }
        for child in children(ast, id) {
            if is_expr(ast, child) {
                check(ast, child)?;
            }
        }
        Ok(())
    }

    for child in children(ast, root) {
        if is_expr(ast, child) {
            check(ast, child)?;
        }
    }
    Ok(())
}

/// Type a freshly built subtree as `stamp_lowering(root, ty)` does, with the root's
/// type as any value the Python pass would put there.
fn stamp_lowering_value(ast: &mut Ast, root: NodeId, ty: Value) -> Result<(), Error> {
    check_lowering_stamps(ast, root)?;
    stamp_lowering(ast, root, OBJECT);
    ast.set_extra(root, TY_ATTR, ty);
    Ok(())
}

fn name(ast: &mut Ast, id: &str, ctx: ExprContext) -> NodeId {
    ast.add(Kind::Name { id: id.to_string(), ctx })
}

fn assign(ast: &mut Ast, target: NodeId, value: NodeId) -> NodeId {
    ast.add(Kind::Assign { targets: vec![target], value, type_comment: None })
}

/// Whether a type annotation is `Series` or `Series[...]`.
fn is_series_type(ast: &Ast, annotation: NodeId) -> bool {
    match ast.kind(annotation) {
        Kind::Subscript { value, .. } => {
            matches!(ast.kind(*value), Kind::Name { id, .. } if id == "Series")
        }
        Kind::Name { id, .. } => id == "Series",
        _ => false,
    }
}

/// Element type name of a `Series[T]` annotation when it is one of the scalars that
/// matter downstream (`float`, `int`, `bool`).
fn series_elem(ast: &Ast, annotation: NodeId) -> Option<String> {
    let Kind::Subscript { slice, .. } = ast.kind(annotation) else { return None };
    match ast.kind(*slice) {
        Kind::Name { id, .. } if matches!(id.as_str(), "float" | "int" | "bool") => {
            Some(id.clone())
        }
        _ => None,
    }
}

fn is_docstring(ast: &Ast, stmt: NodeId) -> bool {
    let Kind::Expr { value } = ast.kind(stmt) else { return false };
    matches!(ast.kind(*value), Kind::Constant { value: Constant::Str(_), .. })
}

struct Series<'l> {
    layout: &'l mut ModuleLayout,
    scope_stack: Vec<String>,
    current_scope: String,
    /// Scope -> var name -> series slot
    series_slots: HashMap<String, HashMap<String, usize>>,
    series_declarations: HashMap<String, HashSet<String>>,
    local_vars: HashMap<String, HashSet<String>>,
    /// Inside a parameter-default guard, where an assignment to a series parameter
    /// gives it its value instead of writing its history
    in_default_guard: bool,
    error: Option<Error>,
}

impl Series<'_> {
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

    /// Resolve a name to its declaring scope and series slot; a name locally assigned
    /// in the current scope (but not declared as a Series there) shadows any parent
    /// series of the same name.
    fn lookup(&self, var_name: &str) -> Option<(String, usize)> {
        let scope = &self.current_scope;
        if self.local_vars.get(scope).is_some_and(|names| names.contains(var_name))
            && !self.series_declarations.get(scope).is_some_and(|names| names.contains(var_name))
        {
            return None;
        }
        if let Some(&slot) = self.series_slots.get(scope).and_then(|slots| slots.get(var_name)) {
            return Some((scope.clone(), slot));
        }
        for i in (1..self.scope_stack.len()).rev() {
            let scope = self.scope_stack[..i].join("·");
            if let Some(&slot) = self.series_slots.get(&scope).and_then(|slots| slots.get(var_name))
            {
                return Some((scope, slot));
            }
        }
        None
    }

    /// The slot of a same-scope series, if `var_name` is one.
    fn own_slot(&self, var_name: &str) -> Option<usize> {
        self.series_slots.get(&self.current_scope).and_then(|slots| slots.get(var_name)).copied()
    }

    /// `<state param>[slot]` of a scope.
    fn state_ref(&self, ast: &mut Ast, scope: &str, slot: usize) -> NodeId {
        let param = name(ast, self.layout.state_param(scope), ExprContext::Load);
        let index =
            ast.add(Kind::Constant { value: Constant::Int(Int::Small(slot as i64)), kind: None });
        ast.add(Kind::Subscript { value: param, slice: index, ctx: ExprContext::Load })
    }

    /// `<state param>[slot].<method>(...)`, typed as the value it stores (both `add`
    /// and `set` return it).
    fn buffer_call(
        &self,
        ast: &mut Ast,
        scope: &str,
        slot: usize,
        method: &str,
        args: Vec<NodeId>,
    ) -> Result<NodeId, Error> {
        let ty = match args.first() {
            Some(&first) => ty_value(ast, first),
            None => Value::Str(OBJECT.to_string()),
        };
        let value = self.state_ref(ast, scope, slot);
        let func =
            ast.add(Kind::Attribute { value, attr: method.to_string(), ctx: ExprContext::Load });
        let call = ast.add(Kind::Call { func, args, keywords: Vec::new() });
        stamp_lowering_value(ast, call, ty)?;
        Ok(call)
    }

    /// Allocate a series slot for a declaration in the current scope.
    fn register(&mut self, ast: &mut Ast, var_name: &str, elem: Option<&str>) -> Option<usize> {
        let scope = self.current_scope.clone();
        let max_bars_back = ast.add(Kind::Constant { value: Constant::None, kind: None });
        let slot = self.layout.scope(&scope).add_series(ast, var_name, max_bars_back, elem);
        self.series_slots.entry(scope.clone()).or_default().insert(var_name.to_string(), slot);
        // The Python pass indexes these two directly: a scope without them is a KeyError
        let (Some(declared), true) =
            (self.series_declarations.get_mut(&scope), self.local_vars.contains_key(&scope))
        else {
            self.fail(Error::Unsupported(format!("a series declaration in scope {scope:?}")));
            return None;
        };
        declared.insert(var_name.to_string());
        self.local_vars.get_mut(&scope).expect("checked above").insert(var_name.to_string());
        Some(slot)
    }

    /// Strip the Series name from pynecore imports.
    fn visit_import_from(&mut self, ast: &mut Ast, id: NodeId) -> Visited {
        let Kind::ImportFrom { module, names, .. } = ast.kind(id) else { unreachable!() };
        if !module.as_deref().is_some_and(|module| module.starts_with("pynecore")) {
            return Visited::Node(id);
        }
        let kept: Vec<NodeId> = names
            .iter()
            .copied()
            .filter(
                |&alias| !matches!(ast.kind(alias), Kind::Alias { name, .. } if name == "Series"),
            )
            .collect();
        if kept.is_empty() {
            return Visited::Remove;
        }
        let Kind::ImportFrom { names, .. } = ast.kind_mut(id) else { unreachable!() };
        *names = kept;
        Visited::Node(id)
    }

    /// Track scopes and convert Series-annotated parameters.
    fn visit_function_def(&mut self, ast: &mut Ast, id: NodeId) -> Result<Visited, Error> {
        let segment = self.layout.scope_segment(ast, id);
        self.scope_stack.push(segment);
        self.current_scope = self.scope_stack.join("·");
        let scope = self.current_scope.clone();
        scope_for_function(self.layout, ast, &scope, id);
        self.local_vars.entry(scope.clone()).or_default();
        self.series_declarations.entry(scope.clone()).or_default();

        let Kind::FunctionDef { args, .. } = ast.kind(id) else { unreachable!() };
        let Kind::Arguments { args: params, .. } = ast.kind(*args) else { unreachable!() };
        let mut param_inits = Vec::new();
        for param in params.clone() {
            let Kind::Arg { arg, annotation, .. } = ast.kind(param) else { unreachable!() };
            let (arg, annotation) = (arg.clone(), *annotation);
            self.local_vars.get_mut(&scope).expect("entered above").insert(arg.clone());
            let Some(annotation) = annotation else { continue };
            if !is_series_type(ast, annotation) {
                continue;
            }
            let elem = series_elem(ast, annotation);
            let Some(slot) = self.register(ast, &arg, elem.as_deref()) else {
                return Ok(Visited::Node(id));
            };
            let stripped = match ast.kind(annotation) {
                Kind::Subscript { slice, .. } => Some(*slice),
                _ => None,
            };
            let Kind::Arg { annotation, .. } = ast.kind_mut(param) else { unreachable!() };
            *annotation = stripped;
            let read = name(ast, &arg, ExprContext::Load);
            let value = self.buffer_call(ast, &scope, slot, "add", vec![read])?;
            let target = name(ast, &arg, ExprContext::Store);
            param_inits.push(assign(ast, target, value));
        }

        generic_visit(self, ast, id);

        if !param_inits.is_empty() {
            let Kind::FunctionDef { body, .. } = ast.kind(id) else { unreachable!() };
            let mut insert_pos = 0;
            if body.first().is_some_and(|&first| is_docstring(ast, first)) {
                insert_pos = 1;
            }
            // A parameter whose default is dynamic arrives as the sentinel and gets
            // its value in the dynamic-default guard block; recording it above that
            // would put the sentinel into the series. The block is contiguous but
            // need not be first -- an earlier pass may have prepended a hoisted
            // declaration of its own
            if let Some(guard) = body.iter().position(|&stmt| is_dynamic_default_guard(ast, stmt)) {
                insert_pos = guard;
                while insert_pos < body.len() && is_dynamic_default_guard(ast, body[insert_pos]) {
                    insert_pos += 1;
                }
            }
            let Kind::FunctionDef { body, .. } = ast.kind_mut(id) else { unreachable!() };
            body.splice(insert_pos..insert_pos, param_inits);
        }

        self.scope_stack.pop();
        self.current_scope = self.scope_stack.join("·");
        Ok(Visited::Node(id))
    }

    /// Convert Series declarations into slot allocations with `add()`.
    fn visit_ann_assign(&mut self, ast: &mut Ast, id: NodeId) -> Result<Visited, Error> {
        let Kind::AnnAssign { target, annotation, value, .. } = ast.kind(id) else {
            unreachable!()
        };
        let (target, annotation, value) = (*target, *annotation, *value);
        let target_name = match ast.kind(target) {
            Kind::Name { id, .. } => Some(id.clone()),
            _ => None,
        };
        let Some(var_name) = target_name.clone().filter(|_| is_series_type(ast, annotation)) else {
            if let Some(target_name) = target_name
                && !self.current_scope.is_empty()
            {
                // An annotated assignment declares a local -- it shadows a same-named
                // parent series, like a plain assignment does
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
                "Series variables must be declared inside a function".into(),
            ));
        }
        let elem = series_elem(ast, annotation);
        let Some(slot) = self.register(ast, &var_name, elem.as_deref()) else {
            return Ok(Visited::Node(id));
        };
        let Some(value) = value else { return Ok(Visited::Remove) };
        let value = self.expr(ast, value);
        let scope = self.current_scope.clone();
        let call = self.buffer_call(ast, &scope, slot, "add", vec![value])?;
        let target = name(ast, &var_name, ExprContext::Store);
        Ok(Visited::Node(assign(ast, target, call)))
    }

    /// Leave a parameter-default guard's own assignment alone: the guard gives a
    /// PARAMETER its value, and the series recording of that parameter happens right
    /// after the guard block.
    fn visit_if(&mut self, ast: &mut Ast, id: NodeId) -> Visited {
        if !is_dynamic_default_guard(ast, id) {
            generic_visit(self, ast, id);
            return Visited::Node(id);
        }
        let outer = std::mem::replace(&mut self.in_default_guard, true);
        let Kind::If { body, .. } = ast.kind(id) else { unreachable!() };
        let mut new_body = Vec::new();
        for stmt in body.clone() {
            match self.visit(ast, stmt) {
                Visited::Node(new) => new_body.push(new),
                // The Python pass puts whatever the visit answers into the list as it is
                other => self.fail(Error::Unsupported(format!("{other:?} in a default guard"))),
            }
        }
        let Kind::If { body, .. } = ast.kind_mut(id) else { unreachable!() };
        *body = new_body;
        self.in_default_guard = outer;
        Visited::Node(id)
    }

    /// Convert assignments to same-scope series variables into buffer writes.
    fn visit_assign(&mut self, ast: &mut Ast, id: NodeId) -> Result<Visited, Error> {
        let Kind::Assign { targets, value, .. } = ast.kind(id) else { unreachable!() };
        let (targets, value) = (targets.clone(), *value);
        if let [target] = targets[..]
            && let Kind::Name { id: var_name, .. } = ast.kind(target)
        {
            let var_name = var_name.clone();
            let slot = self.own_slot(&var_name);
            if self.in_default_guard && slot.is_some() {
                // The guard resolves the parameter; the buffer write follows it
                let new = self.expr(ast, value);
                let Kind::Assign { value, .. } = ast.kind_mut(id) else { unreachable!() };
                *value = new;
                return Ok(Visited::Node(id));
            }
            if let Some(slot) = slot {
                let value = self.expr(ast, value);
                let scope = self.current_scope.clone();
                let call = self.buffer_call(ast, &scope, slot, "set", vec![value])?;
                let target = name(ast, &var_name, ExprContext::Store);
                return Ok(Visited::Node(assign(ast, target, call)));
            }
            if !self.current_scope.is_empty() {
                self.local_vars.entry(self.current_scope.clone()).or_default().insert(var_name);
            }
        }

        // Tuple/list unpacking (`a, b = f()`): a series element is declared bare
        // (`a: Series` with no initializer, so no per-bar `add()`) and the unpack only
        // binds the local name -- its buffer never advances. Emit an `add()` per
        // series element to record history so `a[1]` works.
        if let [target] = targets[..]
            && let Kind::Tuple { elts, .. } | Kind::List { elts, .. } = ast.kind(target)
            && !self.current_scope.is_empty()
            && elts.iter().all(|&elt| matches!(ast.kind(elt), Kind::Name { .. }))
        {
            let elts = elts.clone();
            let slots = self.series_slots.get(&self.current_scope).cloned().unwrap_or_default();
            let new = self.expr(ast, value);
            let Kind::Assign { value, .. } = ast.kind_mut(id) else { unreachable!() };
            *value = new;
            let mut stmts = vec![id];
            let scope = self.current_scope.clone();
            for elt in elts {
                let Kind::Name { id: elt_name, .. } = ast.kind(elt) else { unreachable!() };
                let elt_name = elt_name.clone();
                match slots.get(&elt_name) {
                    Some(&slot) => {
                        let read = name(ast, &elt_name, ExprContext::Load);
                        let call = self.buffer_call(ast, &scope, slot, "add", vec![read])?;
                        let target = name(ast, &elt_name, ExprContext::Store);
                        stmts.push(assign(ast, target, call));
                    }
                    None => {
                        self.local_vars.entry(scope.clone()).or_default().insert(elt_name);
                    }
                }
            }
            return Ok(if stmts.len() > 1 { Visited::Many(stmts) } else { Visited::Node(id) });
        }

        generic_visit(self, ast, id);
        Ok(Visited::Node(id))
    }

    /// Convert augmented assignments to same-scope series into `set()`.
    fn visit_aug_assign(&mut self, ast: &mut Ast, id: NodeId) -> Result<Visited, Error> {
        let Kind::AugAssign { target, op, value } = ast.kind(id) else { unreachable!() };
        let (target, op, value) = (*target, *op, *value);
        if let Kind::Name { id: var_name, .. } = ast.kind(target)
            && let Some(slot) = self.own_slot(var_name)
        {
            let var_name = var_name.clone();
            // The target's stamp is the type of the AUGMENTED value, which is what
            // both the read below and the whole expression carry
            let augmented = ty_value(ast, target);
            let left = name(ast, &var_name, ExprContext::Load);
            ast.set_extra(left, TY_ATTR, augmented.clone());
            let right = self.expr(ast, value);
            let binop = ast.add(Kind::BinOp { left, op, right });
            stamp_lowering_value(ast, binop, augmented)?;
            let scope = self.current_scope.clone();
            let call = self.buffer_call(ast, &scope, slot, "set", vec![binop])?;
            let target = name(ast, &var_name, ExprContext::Store);
            return Ok(Visited::Node(assign(ast, target, call)));
        }
        generic_visit(self, ast, id);
        Ok(Visited::Node(id))
    }

    /// Rewrite series indexing to address the buffer in its slot.
    fn visit_subscript(&mut self, ast: &mut Ast, id: NodeId) -> Visited {
        let Kind::Subscript { value, slice, .. } = ast.kind(id) else { unreachable!() };
        let (value, slice) = (*value, *slice);
        if let Kind::Name { id: var_name, .. } = ast.kind(value)
            && let Some((scope, slot)) = self.lookup(var_name)
        {
            let state = self.state_ref(ast, &scope, slot);
            let Kind::Subscript { value, .. } = ast.kind_mut(id) else { unreachable!() };
            *value = state;
            let new = self.expr(ast, slice);
            let Kind::Subscript { slice, .. } = ast.kind_mut(id) else { unreachable!() };
            *slice = new;
            return Visited::Node(id);
        }
        generic_visit(self, ast, id);
        Visited::Node(id)
    }

    /// Convert a statement-position buffer setter call into an attribute assignment
    /// on the buffer (see [`BUFFER_SETTERS`]).
    fn visit_expr_stmt(&mut self, ast: &mut Ast, id: NodeId) -> Visited {
        let Kind::Expr { value: call } = ast.kind(id) else { unreachable!() };
        if let Kind::Call { func, args, .. } = ast.kind(*call)
            && let Kind::Attribute { value: owner, attr, .. } = ast.kind(*func)
            && let Some(&(_, setter)) = BUFFER_SETTERS.iter().find(|(name, _)| name == attr)
            && matches!(ast.kind(*owner), Kind::Name { id, .. } if id == "lib")
            && args.len() >= 2
            && let Kind::Name { id: series, .. } = ast.kind(args[0])
            && let Some((scope, slot)) = self.lookup(series)
        {
            let amount = args[1];
            let buffer = self.state_ref(ast, &scope, slot);
            let target = ast.add(Kind::Attribute {
                value: buffer,
                attr: setter.to_string(),
                ctx: ExprContext::Store,
            });
            let value = self.expr(ast, amount);
            return Visited::Node(assign(ast, target, value));
        }
        generic_visit(self, ast, id);
        Visited::Node(id)
    }
}

impl Transformer for Series<'_> {
    fn visit(&mut self, ast: &mut Ast, id: NodeId) -> Visited {
        if self.error.is_some() {
            return Visited::Node(id);
        }
        let answer = match ast.kind(id) {
            Kind::Module { .. } => {
                self.layout.assign_scope_ids(ast, id);
                generic_visit(self, ast, id);
                Ok(Visited::Node(id))
            }
            Kind::ImportFrom { .. } => Ok(self.visit_import_from(ast, id)),
            Kind::FunctionDef { .. } => self.visit_function_def(ast, id),
            Kind::AnnAssign { .. } => self.visit_ann_assign(ast, id),
            Kind::If { .. } => Ok(self.visit_if(ast, id)),
            Kind::Assign { .. } => self.visit_assign(ast, id),
            Kind::AugAssign { .. } => self.visit_aug_assign(ast, id),
            Kind::Subscript { .. } => Ok(self.visit_subscript(ast, id)),
            Kind::Expr { .. } => Ok(self.visit_expr_stmt(ast, id)),
            _ => {
                generic_visit(self, ast, id);
                Ok(Visited::Node(id))
            }
        };
        answer.unwrap_or_else(|error| {
            self.fail(error);
            Visited::Node(id)
        })
    }
}

pub fn run(ast: &mut Ast, module: NodeId, ctx: &Context<'_>) -> Result<NodeId, Error> {
    if !matches!(ast.kind(module), Kind::Module { .. }) {
        return Err(Error::Unsupported("Series on a non-module".into()));
    }
    let mut layout = ctx.slot_layout.borrow_mut();
    let mut pass = Series {
        layout: &mut layout,
        scope_stack: Vec::new(),
        current_scope: String::new(),
        series_slots: HashMap::new(),
        series_declarations: HashMap::new(),
        local_vars: HashMap::new(),
        in_default_guard: false,
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
