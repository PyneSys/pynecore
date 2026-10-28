//! `ast.dump(node, include_attributes=...)` as CPython 3.13+ writes it (single line,
//! empty lists and absent optional fields left out), so a tree built without
//! Python can be compared with one built by CPython as text.
//!
//! Values are written as Python's `repr` writes them. The one approximation is
//! `str.isprintable` for non-ASCII characters: the control, format and
//! separator characters a source plausibly holds are escaped, other non-ASCII
//! characters are taken as printable.

use crate::{Ast, CLASSES, Constant, FieldRef, Int, NodeId};

/// The dump of the tree under `id`.
pub fn dump(ast: &Ast, id: NodeId, include_attributes: bool) -> String {
    let mut out = String::new();
    node(ast, id, include_attributes, &mut out);
    out
}

fn node(ast: &Ast, id: NodeId, attrs: bool, out: &mut String) {
    let data = &ast[id];
    let class = &CLASSES[data.kind.class_index()];
    out.push_str(class.name);
    out.push('(');
    let mut first = true;
    let mut sep = |out: &mut String| {
        if !first {
            out.push_str(", ");
        }
        first = false;
    };
    for (index, field) in class.fields.iter().enumerate() {
        let value = data.kind.field(index);
        let mut text = String::new();
        let skip = match value {
            FieldRef::Node(child) => {
                node(ast, *child, attrs, &mut text);
                false
            }
            FieldRef::OptNode(child) => match child {
                Some(child) => {
                    node(ast, *child, attrs, &mut text);
                    false
                }
                None => true,
            },
            FieldRef::Nodes(items) => {
                list(items.iter().map(|&item| Some(item)), ast, attrs, &mut text);
                items.is_empty()
            }
            FieldRef::OptNodes(items) => {
                list(items.iter().copied(), ast, attrs, &mut text);
                items.is_empty()
            }
            FieldRef::Str(value) => {
                str_repr(value, &mut text);
                false
            }
            FieldRef::OptStr(value) => match value {
                Some(value) => {
                    str_repr(value, &mut text);
                    false
                }
                None => true,
            },
            FieldRef::Strs(items) => {
                text.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        text.push_str(", ");
                    }
                    str_repr(item, &mut text);
                }
                text.push(']');
                items.is_empty()
            }
            FieldRef::Int(value) => {
                text.push_str(&value.to_string());
                false
            }
            FieldRef::OptInt(value) => match value {
                Some(value) => {
                    text.push_str(&value.to_string());
                    false
                }
                None => true,
            },
            FieldRef::Constant(value) => {
                // ``value`` is a required field: None is written, not left out
                constant(value, &mut text);
                false
            }
            FieldRef::ExprContext(value) => {
                text.push_str(value.name());
                text.push_str("()");
                false
            }
            FieldRef::Operator(value) => {
                text.push_str(value.name());
                text.push_str("()");
                false
            }
            FieldRef::UnaryOperator(value) => {
                text.push_str(value.name());
                text.push_str("()");
                false
            }
            FieldRef::BoolOperator(value) => {
                text.push_str(value.name());
                text.push_str("()");
                false
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
                items.is_empty()
            }
        };
        if skip {
            continue;
        }
        sep(out);
        out.push_str(field.name);
        out.push('=');
        out.push_str(&text);
    }
    if attrs && class.located {
        let loc = data.loc;
        for (name, value) in [
            ("lineno", loc.lineno),
            ("col_offset", loc.col_offset),
            ("end_lineno", loc.end_lineno),
            ("end_col_offset", loc.end_col_offset),
        ] {
            if let Some(value) = value {
                sep(out);
                out.push_str(name);
                out.push('=');
                out.push_str(&value.to_string());
            }
        }
    }
    out.push(')');
}

fn list(items: impl Iterator<Item = Option<NodeId>>, ast: &Ast, attrs: bool, out: &mut String) {
    out.push('[');
    for (i, item) in items.enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        match item {
            Some(item) => node(ast, item, attrs, out),
            None => out.push_str("None"),
        }
    }
    out.push(']');
}

/// `repr` of a constant value.
pub fn constant(value: &Constant, out: &mut String) {
    match value {
        Constant::None => out.push_str("None"),
        Constant::Bool(true) => out.push_str("True"),
        Constant::Bool(false) => out.push_str("False"),
        Constant::Int(Int::Small(value)) => out.push_str(&value.to_string()),
        Constant::Int(Int::Big(digits)) => out.push_str(digits),
        Constant::Float(value) => out.push_str(&float_repr(*value)),
        Constant::Complex { real, imag } => {
            if *real == 0.0 && real.is_sign_positive() {
                out.push_str(&float_repr_no_point(*imag));
                out.push('j');
            } else {
                out.push('(');
                out.push_str(&float_repr_no_point(*real));
                let imag_text = float_repr_no_point(*imag);
                if !imag_text.starts_with('-') {
                    out.push('+');
                }
                out.push_str(&imag_text);
                out.push_str("j)");
            }
        }
        Constant::Str(value) => str_repr(value, out),
        Constant::Bytes(value) => bytes_repr(value, out),
        Constant::Ellipsis => out.push_str("Ellipsis"),
        Constant::Tuple(items) => {
            out.push('(');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                constant(item, out);
            }
            if items.len() == 1 {
                out.push(',');
            }
            out.push(')');
        }
        Constant::FrozenSet(items) => {
            if items.is_empty() {
                out.push_str("frozenset()");
                return;
            }
            out.push_str("frozenset({");
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                constant(item, out);
            }
            out.push_str("})");
        }
    }
}

/// The shortest round-trip digits and the decimal exponent of a finite non-zero
/// `|value|`, as `(digits, exponent)` with the value `0.digits * 10^exponent`.
fn shortest(value: f64) -> (String, i32) {
    // ``{:e}`` writes the shortest digits that round-trip: ``d.ddde<exp>``
    let text = format!("{:e}", value.abs());
    let (mantissa, exponent) = text.split_once('e').unwrap();
    let digits: String = mantissa.chars().filter(|c| c.is_ascii_digit()).collect();
    (digits, exponent.parse::<i32>().unwrap() + 1)
}

/// `repr(float)`: the shortest round-trip digits, positional while the decimal
/// exponent is in [-4, 16), scientific otherwise.
pub fn float_repr(value: f64) -> String {
    float_format(value, true)
}

/// The `r` format without a forced `.0`, as complex parts are written.
fn float_repr_no_point(value: f64) -> String {
    float_format(value, false)
}

fn float_format(value: f64, add_point: bool) -> String {
    if value.is_nan() {
        return "nan".into();
    }
    if value.is_infinite() {
        return if value > 0.0 { "inf".into() } else { "-inf".into() };
    }
    let sign = if value.is_sign_negative() { "-" } else { "" };
    if value == 0.0 {
        return format!("{sign}0{}", if add_point { ".0" } else { "" });
    }
    let (digits, point) = shortest(value);
    // Python's ``repr`` switches to scientific outside 1e-4 <= |x| < 1e16
    let exponent = point - 1;
    let mut out = String::from(sign);
    if (-4..16).contains(&exponent) {
        if point <= 0 {
            out.push_str("0.");
            out.push_str(&"0".repeat((-point) as usize));
            out.push_str(&digits);
        } else if point as usize >= digits.len() {
            out.push_str(&digits);
            out.push_str(&"0".repeat(point as usize - digits.len()));
            if add_point {
                out.push_str(".0");
            }
        } else {
            out.push_str(&digits[..point as usize]);
            out.push('.');
            out.push_str(&digits[point as usize..]);
        }
    } else {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        out.push('e');
        out.push(if exponent < 0 { '-' } else { '+' });
        out.push_str(&format!("{:02}", exponent.abs()));
    }
    out
}

/// Whether Python's `str.isprintable` holds for a character (see the module docs).
fn printable(c: char) -> bool {
    let code = c as u32;
    !(code < 0x20
        || (0x7f..=0xa0).contains(&code)
        || code == 0xad
        || (0x200b..=0x200f).contains(&code)
        || (0x2028..=0x202e).contains(&code)
        || (0x2060..=0x2064).contains(&code)
        || code == 0xfeff
        || (0xd800..=0xdfff).contains(&code)
        || (0xe000..=0xf8ff).contains(&code)
        || code == 0x3000
        || (0x2000..=0x200a).contains(&code)
        || code == 0x1680
        || code == 0x205f)
}

/// `repr(str)`.
pub fn str_repr(value: &str, out: &mut String) {
    let quote = if value.contains('\'') && !value.contains('"') { '"' } else { '\'' };
    out.push(quote);
    for c in value.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if printable(c) => out.push(c),
            c => {
                let code = c as u32;
                if code <= 0xff {
                    out.push_str(&format!("\\x{code:02x}"));
                } else if code <= 0xffff {
                    out.push_str(&format!("\\u{code:04x}"));
                } else {
                    out.push_str(&format!("\\U{code:08x}"));
                }
            }
        }
    }
    out.push(quote);
}

/// `repr(bytes)`.
pub fn bytes_repr(value: &[u8], out: &mut String) {
    let quote = if value.contains(&b'\'') && !value.contains(&b'"') { '"' } else { '\'' };
    out.push('b');
    out.push(quote);
    for &byte in value {
        match byte {
            b'\\' => out.push_str("\\\\"),
            b'\n' => out.push_str("\\n"),
            b'\r' => out.push_str("\\r"),
            b'\t' => out.push_str("\\t"),
            b if b as char == quote => {
                out.push('\\');
                out.push(b as char);
            }
            0x20..=0x7e => out.push(byte as char),
            b => out.push_str(&format!("\\x{b:02x}")),
        }
    }
    out.push(quote);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floats_as_python_writes_them() {
        for (value, text) in [
            (1.0, "1.0"),
            (0.1, "0.1"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (1.5e-5, "1.5e-05"),
            (0.0001, "0.0001"),
            (-0.0, "-0.0"),
            (123.456, "123.456"),
            (1e-10, "1e-10"),
            (2.5e300, "2.5e+300"),
            (f64::INFINITY, "inf"),
        ] {
            assert_eq!(float_repr(value), text, "{value}");
        }
    }

    #[test]
    fn strings_as_python_writes_them() {
        let mut out = String::new();
        str_repr("it's \"x\"\n\u{b7}", &mut out);
        assert_eq!(out, "'it\\'s \"x\"\\n\u{b7}'");
        out.clear();
        str_repr("it's", &mut out);
        assert_eq!(out, "\"it's\"");
    }
}
