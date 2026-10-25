//! Port of `transformers/implicit_conversion.py`: Pine's implicit conversions of a
//! value where it goes, which the type pass marked on the value.
//!
//! An int that goes into a float container becomes a float literal or
//! `float(...)`; an int container copied into a float one is wrapped in
//! `safe_convert.safe_float_items` / `safe_float_matrix`; a number passed to a lib
//! `bool` parameter in `safe_convert.safe_bool`. A typeless `na` becomes
//! the na of the type it goes in as (`''`, `lib._na_float`, the text `'na'`,
//! `lib.na(<class>)`); one held rather than spelled is wrapped in
//! `safe_convert.implicit_convert(value, kind)`, and a value converted per
//! instance is wrapped in the call the type pass prepared for it. A field default
//! naming a class its class body cannot reach yet is deferred to a
//! `field(default_factory=lambda: ...)`.
//!
//! The step reads only the per-node stamps (`_pine_widen`, `_pine_widen_site`,
//! `_pine_na_class`, `_pine_ty`), never the module's `PineTypeTable`.

use std::collections::{HashMap, HashSet};

use pyne_ast::walk::{Transformer, Visited, copy_location, generic_visit, walk};
use pyne_ast::{
    Ast, Category, Constant, ExprContext, Int, Kind, NodeId, UnaryOperator, Unsupported, Value,
};

use crate::pine_type_rules::{
    BOOL, CONTAINER_NA_WIDENS, FLOAT, OBJECT, STR, WIDEN_ATTR, WIDEN_BOOL, WIDEN_FLOAT,
    WIDEN_ITEMS, WIDEN_MATRIX, WIDEN_NA_TEXT, WIDEN_NUMBER_NA, WIDEN_SITE_ATTR, WIDEN_STRING_NA,
    array_of, class_of, get_na_class, get_ty, is_na_literal, is_na_widen, matrix_of, set_ty,
    set_widen, set_widen_site, stamp_lowering,
};
use crate::{Context, Error};

fn is_expr(ast: &Ast, id: NodeId) -> bool {
    ast.kind(id).class().category == Category::Expr
}

fn name(ast: &mut Ast, id: &str) -> NodeId {
    ast.add(Kind::Name { id: id.to_string(), ctx: ExprContext::Load })
}

/// `<module>.<attr>`, both loaded.
fn module_attr(ast: &mut Ast, module: &str, attr: &str) -> NodeId {
    let value = name(ast, module);
    ast.add(Kind::Attribute { value, attr: attr.to_string(), ctx: ExprContext::Load })
}

fn constant(ast: &mut Ast, value: Constant) -> NodeId {
    ast.add(Kind::Constant { value, kind: None })
}

fn call(ast: &mut Ast, func: NodeId, args: Vec<NodeId>) -> NodeId {
    ast.add(Kind::Call { func, args, keywords: vec![] })
}

/// A node built at a value's location (`ast.copy_location(new, old)`).
fn located(ast: &mut Ast, new: NodeId, old: NodeId) -> NodeId {
    copy_location(ast, new, old);
    new
}

/// The conversion a value needs where it goes (`get_widen`); an unexpected stamp is
/// left to Python.
fn widen_of(ast: &Ast, id: NodeId) -> Result<Option<String>, Unsupported> {
    match ast.extra(id, WIDEN_ATTR) {
        None | Some(Value::None) => Ok(None),
        Some(Value::Str(kind)) => Ok(Some(kind.clone())),
        Some(other) => Err(Unsupported(format!("a {WIDEN_ATTR} stamp {other:?}"))),
    }
}

/// The per-instance conversion call a value is wrapped in (`get_widen_site`).
fn widen_site_of(ast: &Ast, id: NodeId) -> Result<Option<NodeId>, Unsupported> {
    match ast.extra(id, WIDEN_SITE_ATTR) {
        None | Some(Value::None) => Ok(None),
        Some(Value::Node(site)) => Ok(Some(*site)),
        Some(other) => Err(Unsupported(format!("a {WIDEN_SITE_ATTR} stamp {other:?}"))),
    }
}

/// The helper a copied-in container is wrapped in, and the type it returns.
fn copy_helper(kind: &str) -> Option<(&'static str, String)> {
    match kind {
        WIDEN_ITEMS => Some(("safe_float_items", array_of(FLOAT))),
        WIDEN_MATRIX => Some(("safe_float_matrix", matrix_of(FLOAT))),
        _ => None,
    }
}

#[derive(Default)]
struct ImplicitConversion {
    needs_import: bool,
    /// The classes the module defines at its top level
    module_classes: HashSet<String>,
    /// The module's classes whose definition has run before the statement visited
    defined_classes: HashSet<String>,
    /// The direct field defaults of the class bodies being visited
    field_defaults: HashSet<NodeId>,
    /// The name `dataclasses.field` is called by in a deferred field default
    field_alias: Option<String>,
    /// Whether the module has to import `dataclasses.field` under that name
    needs_field_import: bool,
    /// How many times the module binds each name, imports and definitions included
    bindings: HashMap<String, usize>,
    /// The names a `from dataclasses import field` ahead of every class binds
    field_imports: HashSet<String>,
    /// Why the pass cannot answer for the module, once it met such a construct
    unsupported: Option<String>,
}

impl ImplicitConversion {
    fn binding_count(&self, name: &str) -> usize {
        self.bindings.get(name).copied().unwrap_or(0)
    }

    /// Convert one visited expression the type pass marked.
    fn convert(&mut self, ast: &mut Ast, node: NodeId) -> Result<NodeId, Unsupported> {
        let original = node;
        if let Some(site) = widen_site_of(ast, node)? {
            // Wrapped once, like a conversion: a clone must not wrap it again
            set_widen_site(ast, node, None);
            match ast.kind_mut(site) {
                Kind::Call { args, .. } if !args.is_empty() => args[0] = node,
                _ => return Err(Unsupported("a widen site that is no call with arguments".into())),
            }
            self.needs_import = true;
            let ty = get_ty(ast, node).to_string();
            return Ok(stamp_lowering(ast, site, &ty));
        }
        let Some(kind) = widen_of(ast, node)? else {
            return Ok(node);
        };
        // Converted once: a clone of this node (a security slice) must not convert again
        set_widen(ast, node, None);
        let kind = kind.as_str();
        let is_container_na = CONTAINER_NA_WIDENS.contains(&kind);
        if is_na_widen(Some(kind)) && !is_na_literal(ast, node) {
            self.needs_import = true;
            let (converted, result) = if class_of(kind).is_some() {
                // The na of a class: the conversion is handed the class itself
                (class_expr(ast, node), kind)
            } else {
                let result = if kind == WIDEN_NUMBER_NA {
                    FLOAT
                } else if is_container_na {
                    OBJECT
                } else {
                    STR
                };
                (constant(ast, Constant::Str(kind.to_string())), result)
            };
            let func = module_attr(ast, "safe_convert", "implicit_convert");
            let wrapped = call(ast, func, vec![node, converted]);
            let wrapped = located(ast, wrapped, node);
            let value = stamp_lowering(ast, wrapped, result);
            return Ok(self.field_default(ast, original, value, result));
        }
        if class_of(kind).is_some() || is_container_na {
            // The na of a class or of a container: `lib.na(<class>)`
            let result = if is_container_na { OBJECT } else { kind };
            let func = module_attr(ast, "lib", "na");
            let class = class_expr(ast, node);
            let na = call(ast, func, vec![class]);
            let na = located(ast, na, node);
            let value = stamp_lowering(ast, na, result);
            return Ok(self.field_default(ast, original, value, result));
        }
        match kind {
            WIDEN_STRING_NA => {
                let empty = constant(ast, Constant::Str(String::new()));
                let empty = located(ast, empty, node);
                Ok(set_ty(ast, empty, STR))
            }
            WIDEN_NUMBER_NA => {
                let nan = module_attr(ast, "lib", "_na_float");
                let nan = located(ast, nan, node);
                Ok(set_ty(ast, nan, FLOAT))
            }
            WIDEN_NA_TEXT => {
                let text = constant(ast, Constant::Str("na".to_string()));
                let text = located(ast, text, node);
                Ok(set_ty(ast, text, STR))
            }
            WIDEN_BOOL => {
                self.needs_import = true;
                let func = module_attr(ast, "safe_convert", "safe_bool");
                let converted = call(ast, func, vec![node]);
                let converted = located(ast, converted, node);
                Ok(stamp_lowering(ast, converted, BOOL))
            }
            WIDEN_FLOAT => {
                if let Some(literal) = int_literal(ast, node)? {
                    // Rounded to nearest, ties to even, as Python's `float(int)`
                    let float = constant(ast, Constant::Float(literal as f64));
                    let float = located(ast, float, node);
                    return Ok(set_ty(ast, float, FLOAT));
                }
                let func = name(ast, "float");
                let cast = call(ast, func, vec![node]);
                let cast = located(ast, cast, node);
                Ok(stamp_lowering(ast, cast, FLOAT))
            }
            _ => {
                let (helper, result) = copy_helper(kind)
                    .ok_or_else(|| Unsupported(format!("an unknown conversion {kind:?}")))?;
                self.needs_import = true;
                let func = module_attr(ast, "safe_convert", helper);
                let copy = call(ast, func, vec![node]);
                let copy = located(ast, copy, node);
                Ok(stamp_lowering(ast, copy, &result))
            }
        }
    }

    /// Defer a field default that names a class its class body cannot reach yet.
    fn field_default(
        &mut self,
        ast: &mut Ast,
        original: NodeId,
        value: NodeId,
        result: &str,
    ) -> NodeId {
        let spelled = get_na_class(ast, original);
        let reachable = match spelled {
            None => true,
            Some(spelled) => {
                !self.module_classes.contains(spelled) || self.defined_classes.contains(spelled)
            }
        };
        if !self.field_defaults.contains(&original) || reachable {
            return value;
        }
        if self.field_alias.is_none() {
            self.field_alias = Some(self.dataclass_field_alias());
        }
        let alias = self.field_alias.clone().unwrap_or_default();
        let args = ast.add(Kind::Arguments {
            posonlyargs: vec![],
            args: vec![],
            vararg: None,
            kwonlyargs: vec![],
            kw_defaults: vec![],
            kwarg: None,
            defaults: vec![],
        });
        let factory = ast.add(Kind::Lambda { args, body: value });
        let factory = located(ast, factory, original);
        let keyword =
            ast.add(Kind::Keyword { arg: Some("default_factory".to_string()), value: factory });
        let func = name(ast, &alias);
        let field = ast.add(Kind::Call { func, args: vec![], keywords: vec![keyword] });
        let field = located(ast, field, original);
        stamp_lowering(ast, field, result)
    }

    /// The name a deferred field default calls `dataclasses.field` by.
    fn dataclass_field_alias(&mut self) -> String {
        let mut alias = "_dataclass_field".to_string();
        loop {
            if self.field_imports.contains(&alias) && self.binding_count(&alias) == 1 {
                return alias;
            }
            if self.binding_count(&alias) == 0 {
                self.needs_field_import = true;
                return alias;
            }
            alias.push('_');
        }
    }

    fn visit_class_def(&mut self, ast: &mut Ast, id: NodeId) {
        let Kind::ClassDef { body, .. } = ast.kind(id) else { unreachable!() };
        let fields: Vec<NodeId> = body
            .iter()
            .filter_map(|&stmt| match ast.kind(stmt) {
                Kind::AnnAssign { value: Some(value), .. } => Some(*value),
                _ => None,
            })
            .collect();
        self.field_defaults.extend(fields.iter().copied());
        generic_visit(self, ast, id);
        for field in &fields {
            self.field_defaults.remove(field);
        }
    }

    fn visit_module(&mut self, ast: &mut Ast, module: NodeId) {
        let Kind::Module { body, .. } = ast.kind(module) else { unreachable!() };
        let body = body.clone();
        for &stmt in &body {
            match ast.kind(stmt) {
                Kind::ClassDef { name, .. } => {
                    self.module_classes.insert(name.clone());
                }
                Kind::ImportFrom { module: Some(from), names, level }
                    if from == "dataclasses"
                        && level.unwrap_or(0) == 0
                        && self.module_classes.is_empty() =>
                {
                    for &alias in names {
                        let Kind::Alias { name, asname } = ast.kind(alias) else { continue };
                        if name == "field" {
                            self.field_imports.insert(asname.as_ref().unwrap_or(name).clone());
                        }
                    }
                }
                _ => {}
            }
        }
        for child in walk(ast, module) {
            let bound = match ast.kind(child) {
                Kind::Name { id, ctx } if *ctx != ExprContext::Load => id.clone(),
                Kind::FunctionDef { name, .. }
                | Kind::AsyncFunctionDef { name, .. }
                | Kind::ClassDef { name, .. } => name.clone(),
                Kind::Alias { name, asname } => {
                    let spelled = asname.as_ref().unwrap_or(name);
                    spelled.split('.').next().unwrap_or(spelled).to_string()
                }
                _ => continue,
            };
            *self.bindings.entry(bound).or_insert(0) += 1;
        }
        let mut new_body = Vec::with_capacity(body.len());
        for &stmt in &body {
            match self.visit(ast, stmt) {
                Visited::Node(new) => new_body.push(new),
                Visited::Remove => {}
                Visited::Many(many) => new_body.extend(many),
            }
            if let Kind::ClassDef { name, .. } = ast.kind(stmt) {
                self.defined_classes.insert(name.clone());
            }
        }
        set_body(ast, module, new_body);
        if self.needs_import && !binds_safe_convert(ast, module) {
            let alias = ast.add(Kind::Alias { name: "safe_convert".to_string(), asname: None });
            let import = ast.add(Kind::ImportFrom {
                module: Some("pynecore.core".to_string()),
                names: vec![alias],
                level: Some(0),
            });
            let position = import_position(ast, module);
            insert_statement(ast, module, position, import);
        }
        if self.needs_field_import {
            let alias = ast
                .add(Kind::Alias { name: "field".to_string(), asname: self.field_alias.clone() });
            let import = ast.add(Kind::ImportFrom {
                module: Some("dataclasses".to_string()),
                names: vec![alias],
                level: Some(0),
            });
            let position = import_position(ast, module);
            insert_statement(ast, module, position, import);
        }
    }
}

impl Transformer for ImplicitConversion {
    fn visit(&mut self, ast: &mut Ast, id: NodeId) -> Visited {
        if matches!(ast.kind(id), Kind::ClassDef { .. }) {
            self.visit_class_def(ast, id);
        } else {
            generic_visit(self, ast, id);
        }
        if !is_expr(ast, id) || self.unsupported.is_some() {
            return Visited::Node(id);
        }
        match self.convert(ast, id) {
            Ok(new) => Visited::Node(new),
            Err(Unsupported(message)) => {
                self.unsupported = Some(message);
                Visited::Node(id)
            }
        }
    }
}

fn module_body(ast: &Ast, module: NodeId) -> &Vec<NodeId> {
    let Kind::Module { body, .. } = ast.kind(module) else { unreachable!() };
    body
}

fn set_body(ast: &mut Ast, module: NodeId, new_body: Vec<NodeId>) {
    let Kind::Module { body, .. } = ast.kind_mut(module) else { unreachable!() };
    *body = new_body;
}

fn insert_statement(ast: &mut Ast, module: NodeId, position: usize, stmt: NodeId) {
    let Kind::Module { body, .. } = ast.kind_mut(module) else { unreachable!() };
    body.insert(position, stmt);
}

/// Where a generated import goes: after the docstring and the module's own imports.
fn import_position(ast: &Ast, module: NodeId) -> usize {
    let body = module_body(ast, module);
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

/// The expression naming the class whose na a typeless `na` becomes, at the value's
/// location.
fn class_expr(ast: &mut Ast, node: NodeId) -> NodeId {
    let spelled = get_na_class(ast, node).unwrap_or("").to_string();
    let mut parts = spelled.split('.');
    let head = parts.next().unwrap_or("");
    let mut expr = name(ast, head);
    for attr in parts {
        expr = ast.add(Kind::Attribute {
            value: expr,
            attr: attr.to_string(),
            ctx: ExprContext::Load,
        });
    }
    located(ast, expr, node)
}

/// The value of an int literal, a signed one included (`-3` is a `UnaryOp`).
fn int_literal(ast: &Ast, node: NodeId) -> Result<Option<i128>, Unsupported> {
    let mut sign = 1i128;
    let mut node = node;
    if let &Kind::UnaryOp { op: op @ (UnaryOperator::USub | UnaryOperator::UAdd), operand } =
        ast.kind(node)
    {
        if op == UnaryOperator::USub {
            sign = -1;
        }
        node = operand;
    }
    match ast.kind(node) {
        Kind::Constant { value: Constant::Int(Int::Small(value)), .. } => {
            Ok(Some(sign * i128::from(*value)))
        }
        Kind::Constant { value: Constant::Int(Int::Big(_)), .. } => {
            Err(Unsupported("a float conversion of a big int literal".into()))
        }
        _ => Ok(None),
    }
}

/// Whether the module already binds the `safe_convert` module name at its top level.
fn binds_safe_convert(ast: &Ast, module: NodeId) -> bool {
    let alias_of = |alias: NodeId| -> (&str, Option<&str>) {
        let Kind::Alias { name, asname } = ast.kind(alias) else { unreachable!() };
        (name.as_str(), asname.as_deref())
    };
    for &stmt in module_body(ast, module) {
        match ast.kind(stmt) {
            Kind::ImportFrom { module: from, names, .. } => {
                let from = from.as_deref().unwrap_or("");
                let from = from.strip_prefix("pynecore.").unwrap_or(from);
                if from == "core.safe_convert"
                    && names.iter().any(|&alias| alias_of(alias).1 == Some("safe_convert"))
                {
                    return true;
                }
                if from == "core"
                    && names.iter().any(|&alias| {
                        let (name, asname) = alias_of(alias);
                        asname.unwrap_or(name) == "safe_convert"
                    })
                {
                    return true;
                }
            }
            Kind::Import { names }
                if names.iter().any(|&alias| {
                    alias_of(alias) == ("pynecore.core.safe_convert", Some("safe_convert"))
                }) =>
            {
                return true;
            }
            _ => {}
        }
    }
    false
}

pub fn run(ast: &mut Ast, module: NodeId, _ctx: &Context<'_>) -> Result<NodeId, Error> {
    if !matches!(ast.kind(module), Kind::Module { .. }) {
        return Err(Error::Unsupported("ImplicitConversion on a non-module".into()));
    }
    let mut step = ImplicitConversion::default();
    step.visit_module(ast, module);
    match step.unsupported {
        Some(message) => Err(Error::Unsupported(message)),
        None => Ok(module),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NoHost;

    fn ctx() -> Context<'static> {
        Context {
            path: String::new(),
            pyne_mode: None,
            source: String::new(),
            user_code: true,
            na_bool: false,
            sequential: false,
            security_slice_disabled: false,
            emit_layout: true,
            host: &NoHost,
            slot_layout: Default::default(),
        }
    }

    /// `push(a, <value>)` as the only statement, the value stamped `_pine_widen`.
    fn module(ast: &mut Ast, value: Constant, widen: Value) -> (NodeId, NodeId) {
        let value = constant(ast, value);
        ast.set_extra(value, WIDEN_ATTR, widen);
        let func = name(ast, "push");
        let a = name(ast, "a");
        let call = call(ast, func, vec![a, value]);
        let statement = ast.add(Kind::Expr { value: call });
        (ast.add(Kind::Module { body: vec![statement], type_ignores: vec![] }), call)
    }

    #[test]
    fn an_int_literal_into_a_float_container_becomes_a_float_literal() {
        let mut ast = Ast::new();
        let (root, call) =
            module(&mut ast, Constant::Int(Int::Small(-3)), Value::Str(WIDEN_FLOAT.into()));
        assert_eq!(run(&mut ast, root, &ctx()), Ok(root));
        let Kind::Call { args, .. } = ast.kind(call) else { unreachable!() };
        assert!(matches!(ast.kind(args[1]), Kind::Constant { value: Constant::Float(v), .. }
            if *v == -3.0));
        assert_eq!(get_ty(&ast, args[1]), FLOAT);
    }

    #[test]
    fn what_the_port_cannot_reproduce_is_left_to_python() {
        let cases = [
            (Constant::Int(Int::Big("1".repeat(400))), Value::Str(WIDEN_FLOAT.into())),
            (Constant::Int(Int::Small(1)), Value::Opaque(0)),
            (Constant::Int(Int::Small(1)), Value::Str("?".into())),
        ];
        for (value, widen) in cases {
            let mut ast = Ast::new();
            let (root, _) = module(&mut ast, value, widen);
            assert!(matches!(run(&mut ast, root, &ctx()), Err(Error::Unsupported(_))));
        }
    }
}
