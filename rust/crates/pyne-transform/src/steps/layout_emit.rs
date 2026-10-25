//! Port of `transformers/layout_emit.py` (`apply_layout`): the module's slot layout
//! written into the tree.
//!
//! - the module-level `__pyne_slot_layout__` dict (one entry per state-carrying
//!   scope),
//! - a `func.__pyne_layout__ = __pyne_slot_layout__['<scope>']` attach after every
//!   state-carrying definition; a decorated one gets an innermost
//!   `@__attach_layout__(...)` decorator instead, and a method of a class body the
//!   `__state_method__` descriptor besides,
//! - the hidden state parameter, first, on every state-carrying definition.
//!
//! The step runs only when the context asks for the CPython materialization
//! (`emit_layout`).

use std::collections::BTreeSet;

use pyne_ast::locations::anchor_at;
use pyne_ast::{Ast, Category, Constant, ExprContext, FieldMut, FieldRef, Int, Kind, NodeId};

use crate::pine_type_rules::{OBJECT, stamp_lowering};
use crate::slot_layout::{ModuleLayout, ScopeLayout, SlotKind};
use crate::{Context, Error};

fn constant(ast: &mut Ast, value: Constant) -> NodeId {
    ast.add(Kind::Constant { value, kind: None })
}

fn int(ast: &mut Ast, value: usize) -> NodeId {
    constant(ast, Constant::Int(Int::Small(value as i64)))
}

fn text(ast: &mut Ast, value: Option<&str>) -> NodeId {
    constant(ast, value.map_or(Constant::None, |value| Constant::Str(value.to_string())))
}

fn tuple(ast: &mut Ast, elts: Vec<NodeId>) -> NodeId {
    ast.add(Kind::Tuple { elts, ctx: ExprContext::Load })
}

fn name(ast: &mut Ast, id: &str, ctx: ExprContext) -> NodeId {
    ast.add(Kind::Name { id: id.to_string(), ctx })
}

/// The layout dict literal of one scope (`_scope_entry_ast`).
fn scope_entry(ast: &mut Ast, scope: &ScopeLayout, compacted: bool) -> NodeId {
    let init = tuple(ast, scope.slots.iter().map(|slot| slot.init).collect());
    let series = scope
        .slots
        .iter()
        .filter(|slot| slot.kind == SlotKind::Series)
        .map(|slot| {
            let index = int(ast, slot.index);
            let max_bars_back = slot.max_bars_back.unwrap_or_else(|| text(ast, None));
            let elem = text(ast, slot.series_elem.as_deref());
            tuple(ast, vec![index, max_bars_back, elem])
        })
        .collect();
    let series = tuple(ast, series);
    let varip =
        scope.slots.iter().filter(|slot| slot.varip).map(|slot| int(ast, slot.index)).collect();
    let varip = tuple(ast, varip);
    let children = scope
        .slots
        .iter()
        .filter(|slot| matches!(slot.kind, SlotKind::Child | SlotKind::Anchor))
        .map(|slot| {
            let index = int(ast, slot.index);
            let call_id = text(ast, slot.call_id.as_deref());
            let in_loop = constant(ast, Constant::Bool(slot.in_loop));
            tuple(ast, vec![index, call_id, in_loop])
        })
        .collect();
    let children = tuple(ast, children);
    let names = scope.slots.iter().map(|slot| text(ast, Some(&slot.name))).collect();
    let names = tuple(ast, names);
    let mut keys = vec!["init", "series", "varip", "children", "names"];
    let mut values = vec![init, series, varip, children, names];
    if let Some(pin) = scope.pin_slot {
        keys.push("pin");
        values.push(int(ast, pin));
    }
    if compacted {
        keys.push("compacted");
        values.push(constant(ast, Constant::Bool(true)));
    }
    let keys = keys.into_iter().map(|key| Some(text(ast, Some(key)))).collect();
    ast.add(Kind::Dict { keys, values })
}

/// The module-level `__pyne_slot_layout__` assignment (`_layout_assign_ast`).
fn layout_assign(ast: &mut Ast, layout: &ModuleLayout) -> NodeId {
    let carrying: Vec<&ScopeLayout> =
        layout.scopes.values().filter(|scope| !scope.slots.is_empty()).collect();
    let keys = carrying.iter().map(|scope| Some(text(ast, Some(&scope.scope)))).collect();
    let values =
        carrying.iter().map(|scope| scope_entry(ast, scope, layout.compacted_series)).collect();
    let dict = ast.add(Kind::Dict { keys, values });
    // The literal swallows the declarations' inits, which keep their stamps; the
    // table around them is machinery
    let value = stamp_lowering(ast, dict, OBJECT);
    let target = name(ast, "__pyne_slot_layout__", ExprContext::Store);
    ast.add(Kind::Assign { targets: vec![target], value, type_comment: None })
}

/// `__pyne_slot_layout__['<scope>']`.
fn layout_entry(ast: &mut Ast, scope_id: &str) -> NodeId {
    let value = name(ast, "__pyne_slot_layout__", ExprContext::Load);
    let slice = text(ast, Some(scope_id));
    ast.add(Kind::Subscript { value, slice, ctx: ExprContext::Load })
}

/// `func.__pyne_layout__ = __pyne_slot_layout__['scope']` (`_attach_ast`).
fn attach(ast: &mut Ast, func_name: &str, scope_id: &str) -> NodeId {
    let func = name(ast, func_name, ExprContext::Load);
    let target = ast.add(Kind::Attribute {
        value: func,
        attr: "__pyne_layout__".into(),
        ctx: ExprContext::Store,
    });
    let value = layout_entry(ast, scope_id);
    ast.add(Kind::Assign { targets: vec![target], value, type_comment: None })
}

/// `__attach_layout__(__pyne_slot_layout__['scope'])` (`_attach_decorator_ast`).
fn attach_decorator(ast: &mut Ast, scope_id: &str) -> NodeId {
    let func = name(ast, "__attach_layout__", ExprContext::Load);
    let entry = layout_entry(ast, scope_id);
    ast.add(Kind::Call { func, args: vec![entry], keywords: Vec::new() })
}

/// Run [`process_defs`] over every statement list hanging off a node
/// (`_process_nested_bodies`).
fn process_nested_bodies(
    ast: &mut Ast,
    node: NodeId,
    prefix: &str,
    layout: &mut ModuleLayout,
    used: &mut BTreeSet<&'static str>,
) -> Result<(), Error> {
    for index in 0..ast.kind(node).class().fields.len() {
        let FieldRef::Nodes(items) = ast.kind(node).field(index) else { continue };
        let Some(&first) = items.first() else { continue };
        let items = items.clone();
        if ast.kind(first).class().category == Category::Stmt {
            let body = process_defs(ast, items, prefix, layout, used, false)?;
            let FieldMut::Nodes(slot) = ast.kind_mut(node).field_mut(index) else { unreachable!() };
            *slot = body;
            continue;
        }
        for item in items {
            if matches!(ast.kind(item), Kind::ExceptHandler { .. } | Kind::MatchCase { .. }) {
                process_nested_bodies(ast, item, prefix, layout, used)?;
            }
        }
    }
    Ok(())
}

/// Wire a state-carrying method of a class body to its layout (`_process_method`).
///
/// The hidden state parameter stands first, like on every other definition, so the
/// class attribute is the plain function; an instance reaches it through the
/// `__state_method__` descriptor. A static method needs only the layout; any other
/// decorator decides itself what the attribute holds, so its state could not be
/// routed.
fn process_method(
    ast: &mut Ast,
    stmt: NodeId,
    scope_id: &str,
    used: &mut BTreeSet<&'static str>,
) -> Result<(), Error> {
    let Kind::FunctionDef { name: func_name, decorator_list, .. } = ast.kind(stmt) else {
        unreachable!()
    };
    let decorators = decorator_list.clone();
    let static_method = decorators.len() == 1
        && matches!(ast.kind(decorators[0]), Kind::Name { id, .. } if id == "staticmethod");
    if !decorators.is_empty() && !static_method {
        let line = ast[stmt].loc.lineno.map_or_else(|| "None".to_string(), |l| l.to_string());
        return Err(Error::Rejected(format!(
            "method '{func_name}' keeps state (a 'ta' call, a series or a persistent), \
             which only a plain or a @staticmethod method can do (line {line})"
        )));
    }
    let mut added = Vec::with_capacity(2);
    if decorators.is_empty() {
        added.push(name(ast, "__state_method__", ExprContext::Load));
        used.insert("__state_method__");
    }
    added.push(attach_decorator(ast, scope_id));
    used.insert("__attach_layout__");
    let Kind::FunctionDef { decorator_list, .. } = ast.kind_mut(stmt) else { unreachable!() };
    decorator_list.extend(added);
    Ok(())
}

/// Inject hidden state parameters and layout attaches into a statement list
/// (`_process_defs`); `in_class` when the list is a class body, whose definitions
/// are methods.
fn process_defs(
    ast: &mut Ast,
    body: Vec<NodeId>,
    prefix: &str,
    layout: &mut ModuleLayout,
    used: &mut BTreeSet<&'static str>,
    in_class: bool,
) -> Result<Vec<NodeId>, Error> {
    let mut new_body = Vec::with_capacity(body.len());
    for stmt in body {
        new_body.push(stmt);
        if let Kind::ClassDef { body, .. } = ast.kind(stmt) {
            // A class body is no scope of its own: its methods belong to the
            // enclosing one, like the scope segments say
            let inner = process_defs(ast, body.clone(), prefix, layout, used, true)?;
            let Kind::ClassDef { body, .. } = ast.kind_mut(stmt) else { unreachable!() };
            *body = inner;
            continue;
        }
        let Kind::FunctionDef { name, args, body, decorator_list, .. } = ast.kind(stmt) else {
            process_nested_bodies(ast, stmt, prefix, layout, used)?;
            continue;
        };
        let (func_name, args, inner, decorated) =
            (name.clone(), *args, body.clone(), !decorator_list.is_empty());
        let segment = layout.scope_segment(ast, stmt);
        let scope_id = if prefix.is_empty() { segment } else { format!("{prefix}·{segment}") };
        let inner = process_defs(ast, inner, &scope_id, layout, used, false)?;
        let Kind::FunctionDef { body, .. } = ast.kind_mut(stmt) else { unreachable!() };
        *body = inner;
        if !layout.state_carrying(&scope_id) {
            continue;
        }
        let param = ast.add(Kind::Arg {
            arg: layout.state_param(&scope_id).to_string(),
            annotation: None,
            type_comment: None,
        });
        let Kind::Arguments { posonlyargs, args: params, .. } = ast.kind_mut(args) else {
            unreachable!()
        };
        // First in the signature: positional-only parameters precede the rest
        if posonlyargs.is_empty() { params } else { posonlyargs }.insert(0, param);
        if in_class {
            process_method(ast, stmt, &scope_id, used)?;
        } else if decorated {
            let decorator = attach_decorator(ast, &scope_id);
            let Kind::FunctionDef { decorator_list, .. } = ast.kind_mut(stmt) else {
                unreachable!()
            };
            decorator_list.push(decorator);
            used.insert("__attach_layout__");
        } else {
            let attach = attach(ast, &func_name, &scope_id);
            new_body.push(anchor_at(ast, attach, stmt)?);
        }
    }
    Ok(new_body)
}

/// Index right after the module docstring and the leading import block (`_insert_index`).
fn insert_index(ast: &Ast, body: &[NodeId]) -> usize {
    let mut index = 0;
    for (i, &stmt) in body.iter().enumerate() {
        let leading = match ast.kind(stmt) {
            Kind::Expr { value } => {
                matches!(ast.kind(*value), Kind::Constant { value: Constant::Str(_), .. })
            }
            Kind::Import { .. } | Kind::ImportFrom { .. } => true,
            _ => false,
        };
        if !leading {
            break;
        }
        index = i + 1;
    }
    index
}

pub fn run(ast: &mut Ast, module: NodeId, ctx: &Context<'_>) -> Result<NodeId, Error> {
    if !ctx.emit_layout {
        return Ok(module);
    }
    let Kind::Module { body, .. } = ast.kind(module) else {
        return Err(Error::Unsupported("ApplyLayout on a non-module".into()));
    };
    let body = body.clone();
    let mut layout = ctx.slot_layout.borrow_mut();
    // The dict is emitted even when empty: its presence marks the module as
    // transformed, which the cross-module call-site classification relies on
    layout.assign_scope_ids(ast, module);
    let mut used = BTreeSet::new();
    let mut body = process_defs(ast, body, "", &mut layout, &mut used, false)?;
    let index = insert_index(ast, &body);
    let assign = layout_assign(ast, &layout);
    body.insert(index, assign);
    if !used.is_empty() {
        let names = used
            .iter()
            .map(|name| ast.add(Kind::Alias { name: name.to_string(), asname: None }))
            .collect();
        let import = ast.add(Kind::ImportFrom {
            module: Some("pynecore.core.instance_state".into()),
            names,
            level: Some(0),
        });
        body.insert(index, import);
    }
    let Kind::Module { body: slot, .. } = ast.kind_mut(module) else { unreachable!() };
    *slot = body;
    Ok(module)
}
