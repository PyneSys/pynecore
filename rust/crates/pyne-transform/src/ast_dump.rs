//! `ast.dump` as a given interpreter writes it, for the digests taken over its text:
//! the routes shape of a module's interface (`module_interface.routes_shape`) and the
//! content stamp of a module without a source path (`function_isolation`).
//!
//! The text follows the interpreter the host names: before 3.12 a definition has no
//! `type_params`, before 3.13 an empty list is written out, and a value is written as
//! that interpreter's `repr` writes it.

use pyne_ast::{Ast, CLASSES, Constant, FieldRef, NodeId, Unsupported};

use crate::unparse::py_repr;

type R<T> = Result<T, Unsupported>;

/// Whether a class has a field on this interpreter.
pub fn has_field(field: &str, python: (u32, u32)) -> bool {
    field != "type_params" || python >= (3, 12)
}

/// `ast.dump(node, include_attributes=...)` as the interpreter `python` names writes it.
pub fn dump(ast: &Ast, node: NodeId, include_attributes: bool, python: (u32, u32)) -> R<String> {
    let mut out = String::new();
    Dumper { ast, include_attributes, python }.node(node, &mut out)?;
    Ok(out)
}

struct Dumper<'a> {
    ast: &'a Ast,
    include_attributes: bool,
    python: (u32, u32),
}

impl Dumper<'_> {
    fn node(&self, node: NodeId, out: &mut String) -> R<()> {
        let (ast, python) = (self.ast, self.python);
        let kind = ast.kind(node);
        let class = &CLASSES[kind.class_index()];
        let omit_empty = python >= (3, 13);
        out.push_str(class.name);
        out.push('(');
        let mut first = true;
        for (index, field) in class.fields.iter().enumerate() {
            if !has_field(field.name, python) {
                continue;
            }
            let mut text = String::new();
            let shown = match kind.field(index) {
                FieldRef::Node(child) => {
                    self.node(*child, &mut text)?;
                    true
                }
                FieldRef::OptNode(child) => match child {
                    Some(child) => {
                        self.node(*child, &mut text)?;
                        true
                    }
                    None => false,
                },
                FieldRef::Nodes(items) => {
                    text.push('[');
                    for (i, &item) in items.iter().enumerate() {
                        if i > 0 {
                            text.push_str(", ");
                        }
                        self.node(item, &mut text)?;
                    }
                    text.push(']');
                    !(omit_empty && items.is_empty())
                }
                FieldRef::OptNodes(items) => {
                    text.push('[');
                    for (i, item) in items.iter().enumerate() {
                        if i > 0 {
                            text.push_str(", ");
                        }
                        match item {
                            Some(item) => self.node(*item, &mut text)?,
                            None => text.push_str("None"),
                        }
                    }
                    text.push(']');
                    !(omit_empty && items.is_empty())
                }
                FieldRef::Str(value) => {
                    text.push_str(&py_repr(&Constant::Str(value.clone()), python)?);
                    true
                }
                FieldRef::OptStr(value) => match value {
                    Some(value) => {
                        text.push_str(&py_repr(&Constant::Str(value.clone()), python)?);
                        true
                    }
                    None => false,
                },
                FieldRef::Strs(items) => {
                    text.push('[');
                    for (i, item) in items.iter().enumerate() {
                        if i > 0 {
                            text.push_str(", ");
                        }
                        text.push_str(&py_repr(&Constant::Str(item.clone()), python)?);
                    }
                    text.push(']');
                    !(omit_empty && items.is_empty())
                }
                FieldRef::Int(value) => {
                    text.push_str(&value.to_string());
                    true
                }
                FieldRef::OptInt(value) => match value {
                    Some(value) => {
                        text.push_str(&value.to_string());
                        true
                    }
                    None => false,
                },
                FieldRef::Constant(value) => {
                    text.push_str(&py_repr(value, python)?);
                    true
                }
                FieldRef::ExprContext(value) => {
                    text.push_str(value.name());
                    text.push_str("()");
                    true
                }
                FieldRef::Operator(value) => {
                    text.push_str(value.name());
                    text.push_str("()");
                    true
                }
                FieldRef::UnaryOperator(value) => {
                    text.push_str(value.name());
                    text.push_str("()");
                    true
                }
                FieldRef::BoolOperator(value) => {
                    text.push_str(value.name());
                    text.push_str("()");
                    true
                }
                FieldRef::CmpOperators(items) => {
                    text.push('[');
                    for (i, item) in items.iter().enumerate() {
                        if i > 0 {
                            text.push_str(", ");
                        }
                        text.push_str(item.name());
                        text.push_str("()");
                    }
                    text.push(']');
                    !(omit_empty && items.is_empty())
                }
            };
            if !shown {
                continue;
            }
            if !first {
                out.push_str(", ");
            }
            first = false;
            out.push_str(field.name);
            out.push('=');
            out.push_str(&text);
        }
        if self.include_attributes && class.located {
            let loc = ast[node].loc;
            for (name, value) in [
                ("lineno", loc.lineno),
                ("col_offset", loc.col_offset),
                ("end_lineno", loc.end_lineno),
                ("end_col_offset", loc.end_col_offset),
            ] {
                let Some(value) = value else { continue };
                if !first {
                    out.push_str(", ");
                }
                first = false;
                out.push_str(name);
                out.push('=');
                out.push_str(&value.to_string());
            }
        }
        out.push(')');
        Ok(())
    }
}
