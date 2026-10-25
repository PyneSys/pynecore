//! Port of `transformers/pine_type_rules.py`: the Pine type lattice and the measured
//! TradingView type algebra.
//!
//! A Pine type is a string, exactly as in Python: one character for a scalar
//! (`i f b s c o v ? *`), and a small grammar for the shaped ones:
//!
//! ```text
//! ty   := <char> | 'o:' <class-id> | 'a:' <ty> | 'm:' <ty>
//!       | 'M:' <key-char> ':' <ty> | 'T:' <item>+
//! item := <decimal length> ':' <ty>
//! ```
//!
//! Lengths and positions are counted in characters, as Python counts them: a
//! class id carries a module path, which may hold any character. The module
//! keeps no AST walking of its own beyond what the Python module does
//! (`stamp_lowering`, the annotation readers, `FactoryFields`), so the algebra can
//! be checked one rule at a time against the Python one.

use std::collections::HashMap;
use std::sync::OnceLock;

use pyne_ast::walk::{children, walk, walk_statements};
use pyne_ast::{
    Ast, Category, Constant, ExprContext, Int, Kind, NodeId, Operator, UnaryOperator, Unsupported,
    Value,
};

// --- the stamps -----------------------------------------------------------

/// Attribute a typed expression node carries its Pine type under.
pub const TY_ATTR: &str = "_pine_ty";
/// Attribute a call site carries its overload pin under.
pub const PIN_ATTR: &str = "_pine_pin";
/// Attribute a call site carries its per-context pins under.
pub const PINS_ATTR: &str = "_pine_pins";
/// Attribute a call site carries the instance vector it configures its callee with.
pub const VECTOR_ATTR: &str = "_pine_vector";
/// Attribute a definition carries its instance-varying inner call sites under.
pub const VARYING_ATTR: &str = "_pine_varying";
/// Attribute a value carries the implicit conversion it needs where it goes under.
pub const WIDEN_ATTR: &str = "_pine_widen";
/// Attribute a typeless `na` carries the spelling of the class it becomes the na of.
pub const NA_CLASS_ATTR: &str = "_pine_na_class";
/// Attribute a value carries its per-instance conversion call under.
pub const WIDEN_SITE_ATTR: &str = "_pine_widen_site";
/// Attribute a per-instance conversion call carries its per-context conversions under.
pub const WIDENS_ATTR: &str = "_pine_widens";

/// One int element, converted to float
pub const WIDEN_FLOAT: &str = "f";
/// A number passed to a `bool` parameter of a lib function, converted by Pine's
/// tolerant float-to-bool rule, na included
pub const WIDEN_BOOL: &str = "B";
/// An int array whose elements are copied in, converted
pub const WIDEN_ITEMS: &str = "a";
/// An int matrix whose elements are copied in, converted
pub const WIDEN_MATRIX: &str = "m";
/// A typeless `na` where a string goes: the string na
pub const WIDEN_STRING_NA: &str = "s";
/// A typeless `na` where a number goes: NaN
pub const WIDEN_NUMBER_NA: &str = "n";
/// A typeless `na` passed as compile-time text of the study
pub const WIDEN_NA_TEXT: &str = "t";
/// A typeless `na` where an array goes
pub const WIDEN_ARRAY_NA: &str = "A";
/// A typeless `na` where a map goes
pub const WIDEN_MAP_NA: &str = "M";
/// A typeless `na` where a matrix goes
pub const WIDEN_MATRIX_NA: &str = "X";
/// The conversions of a typeless `na` into the na of a container
pub const CONTAINER_NA_WIDENS: &[&str] = &[WIDEN_ARRAY_NA, WIDEN_MAP_NA, WIDEN_MATRIX_NA];

/// The spellings of Pine's typeless `na` in value position.
pub const NA_LITERALS: &[&str] = &["na", "lib.na", "_na_none", "lib._na_none"];

/// The parameters TradingView compiles into the study's metadata, per lib callee.
pub const NA_TEXT_PARAMS: &[(&str, &[&str])] = &[
    ("plot", &["title"]),
    ("plot.plot", &["title"]),
    ("plotchar", &["title", "char", "text"]),
    ("plotshape", &["title", "text"]),
    ("plotarrow", &["title"]),
    ("plotbar", &["title"]),
    ("plotcandle", &["title"]),
    ("hline", &["title"]),
    ("hline.hline", &["title"]),
    ("bgcolor", &["title"]),
    ("barcolor", &["title"]),
    ("fill", &["title"]),
    ("alertcondition", &["title", "message"]),
];

/// Whether a lib callee's parameter is compile-time text of the study (`NA_TEXT_PARAMS`).
pub fn is_na_text_param(callee: &str, param: &str) -> bool {
    NA_TEXT_PARAMS.iter().any(|&(name, params)| name == callee && params.contains(&param))
}

/// Lib parameters the registry types as strings that are no Pine string.
pub const NOT_PINE_STRINGS: &[(&str, &str)] =
    &[("matrix.sort", "order"), ("array.sort", "sort_field"), ("array.sort_indices", "sort_field")];

/// Whether a lib callee's parameter is one of `NOT_PINE_STRINGS`.
pub fn is_not_pine_string(callee: &str, param: &str) -> bool {
    NOT_PINE_STRINGS.contains(&(callee, param))
}

/// A stamped string attribute, `None` when absent or `None`.
fn extra_str<'a>(ast: &'a Ast, id: NodeId, name: &str) -> Option<&'a str> {
    match ast.extra(id, name) {
        Some(Value::Str(text)) => Some(text),
        _ => None,
    }
}

/// A `dict[int, str | None]` stamp, in insertion order.
fn extra_context_map(ast: &Ast, id: NodeId, name: &str) -> Option<Vec<(i64, Option<String>)>> {
    let Some(Value::Dict(items)) = ast.extra(id, name) else {
        return None;
    };
    items
        .iter()
        .map(|(key, value)| {
            let Value::Int(Int::Small(context)) = key else {
                return None;
            };
            match value {
                Value::None => Some((*context, None)),
                Value::Str(text) => Some((*context, Some(text.clone()))),
                _ => None,
            }
        })
        .collect()
}

fn context_map_value(map: Option<&[(i64, Option<String>)]>) -> Value {
    match map {
        None => Value::None,
        Some(items) => Value::Dict(
            items
                .iter()
                .map(|(context, text)| {
                    let text = text.as_ref().map_or(Value::None, |text| Value::Str(text.clone()));
                    (Value::Int(Int::Small(*context)), text)
                })
                .collect(),
        ),
    }
}

fn opt_str_value(text: Option<&str>) -> Value {
    text.map_or(Value::None, |text| Value::Str(text.to_string()))
}

/// A call site's overload pin (`get_pin`).
pub fn get_pin(ast: &Ast, id: NodeId) -> Option<&str> {
    extra_str(ast, id, PIN_ATTR)
}

/// Stamp a call site's overload pin, `None` included (`set_pin`).
pub fn set_pin(ast: &mut Ast, id: NodeId, pin: Option<&str>) -> NodeId {
    ast.set_extra(id, PIN_ATTR, opt_str_value(pin));
    id
}

/// The per-context pins of a call site, context id -> pin (`get_pins`).
pub fn get_pins(ast: &Ast, id: NodeId) -> Option<Vec<(i64, Option<String>)>> {
    extra_context_map(ast, id, PINS_ATTR)
}

/// Stamp the per-context pins of a call site, `None` included (`set_pins`).
pub fn set_pins(ast: &mut Ast, id: NodeId, pins: Option<&[(i64, Option<String>)]>) -> NodeId {
    ast.set_extra(id, PINS_ATTR, context_map_value(pins));
    id
}

/// The instance vector a call site configures its callee with (`get_vector`).
pub fn get_vector(ast: &Ast, id: NodeId) -> Option<&Value> {
    match ast.extra(id, VECTOR_ATTR) {
        None | Some(Value::None) => None,
        Some(vector) => Some(vector),
    }
}

/// Stamp the instance vector a call site hands its callee (`set_vector`).
pub fn set_vector(ast: &mut Ast, id: NodeId, vector: Option<Value>) -> NodeId {
    ast.set_extra(id, VECTOR_ATTR, vector.unwrap_or(Value::None));
    id
}

/// A definition's instance-varying inner call sites, in source order (`get_varying`).
pub fn get_varying(ast: &Ast, id: NodeId) -> Option<Vec<NodeId>> {
    let Some(Value::List(items)) = ast.extra(id, VARYING_ATTR) else {
        return None;
    };
    items
        .iter()
        .map(|item| match item {
            Value::Node(site) => Some(*site),
            _ => None,
        })
        .collect()
}

/// Stamp a definition's instance-varying inner call sites (`set_varying`).
pub fn set_varying(ast: &mut Ast, id: NodeId, sites: Option<&[NodeId]>) -> NodeId {
    let value = sites.map_or(Value::None, |sites| {
        Value::List(sites.iter().map(|&site| Value::Node(site)).collect())
    });
    ast.set_extra(id, VARYING_ATTR, value);
    id
}

/// The implicit conversion a value needs where it goes (`get_widen`).
pub fn get_widen(ast: &Ast, id: NodeId) -> Option<&str> {
    extra_str(ast, id, WIDEN_ATTR)
}

/// Stamp the conversion a value needs, `None` included (`set_widen`).
pub fn set_widen(ast: &mut Ast, id: NodeId, kind: Option<&str>) -> NodeId {
    ast.set_extra(id, WIDEN_ATTR, opt_str_value(kind));
    id
}

/// Whether a conversion turns Pine's typeless `na` into the na of a type.
pub fn is_na_widen(kind: Option<&str>) -> bool {
    let Some(kind) = kind else {
        return false;
    };
    [WIDEN_STRING_NA, WIDEN_NUMBER_NA, WIDEN_NA_TEXT].contains(&kind)
        || CONTAINER_NA_WIDENS.contains(&kind)
        || kind.starts_with(OBJ)
}

/// How the module spells the class a typeless `na` becomes the na of (`get_na_class`).
pub fn get_na_class(ast: &Ast, id: NodeId) -> Option<&str> {
    extra_str(ast, id, NA_CLASS_ATTR)
}

/// Stamp the spelling of the class a typeless `na` becomes the na of (`set_na_class`).
pub fn set_na_class(ast: &mut Ast, id: NodeId, spelled: &str) -> NodeId {
    ast.set_extra(id, NA_CLASS_ATTR, Value::Str(spelled.to_string()));
    id
}

/// The per-instance conversion call a value is to be wrapped in (`get_widen_site`).
pub fn get_widen_site(ast: &Ast, id: NodeId) -> Option<NodeId> {
    match ast.extra(id, WIDEN_SITE_ATTR) {
        Some(Value::Node(site)) => Some(*site),
        _ => None,
    }
}

/// Stamp the per-instance conversion call of a value, `None` included (`set_widen_site`).
pub fn set_widen_site(ast: &mut Ast, id: NodeId, site: Option<NodeId>) -> NodeId {
    ast.set_extra(id, WIDEN_SITE_ATTR, site.map_or(Value::None, Value::Node));
    id
}

/// The per-context conversions of a per-instance conversion call (`get_widens`).
pub fn get_widens(ast: &Ast, id: NodeId) -> Option<Vec<(i64, Option<String>)>> {
    extra_context_map(ast, id, WIDENS_ATTR)
}

/// Stamp the per-context conversions of a per-instance conversion call (`set_widens`).
pub fn set_widens(ast: &mut Ast, id: NodeId, widens: Option<&[(i64, Option<String>)]>) -> NodeId {
    ast.set_extra(id, WIDENS_ATTR, context_map_value(widens));
    id
}

/// A node's Pine type, `UNKNOWN` when it was never typed (`get_ty`).
pub fn get_ty(ast: &Ast, id: NodeId) -> &str {
    extra_str(ast, id, TY_ATTR).unwrap_or(UNKNOWN)
}

/// A node's Pine type as [`get_ty`] reads it, for a pass that carries the stamp
/// over to another node: a stamp that is no string is `Unsupported`, where the
/// Python pass would carry that very value.
pub fn ty_stamp(ast: &Ast, id: NodeId) -> Result<&str, Unsupported> {
    match ast.extra(id, TY_ATTR) {
        None => Ok(UNKNOWN),
        Some(Value::Str(ty)) => Ok(ty),
        Some(other) => Err(Unsupported(format!("a {TY_ATTR} stamp {other:?}"))),
    }
}

/// Stamp a node's Pine type (`set_ty`).
pub fn set_ty(ast: &mut Ast, id: NodeId, ty: &str) -> NodeId {
    ast.set_extra(id, TY_ATTR, Value::Str(ty.to_string()));
    id
}

/// Carry a type stamp onto a freshly built wrapper node (`inherit_ty`).
pub fn inherit_ty(ast: &mut Ast, new: NodeId, old: NodeId) -> NodeId {
    match ast.extra(old, TY_ATTR) {
        None | Some(Value::None) => {}
        Some(ty) => {
            let ty = ty.clone();
            ast.set_extra(new, TY_ATTR, ty);
        }
    }
    new
}

// --- the lattice ----------------------------------------------------------

pub const INT: &str = "i";
pub const FLOAT: &str = "f";
pub const BOOL: &str = "b";
pub const STR: &str = "s";
pub const COLOR: &str = "c";
/// A known non-scalar: array, matrix, map, drawing, UDT, tuple
pub const OBJECT: &str = "o";
/// A call that returns nothing
pub const VOID: &str = "v";
/// Untypable
pub const UNKNOWN: &str = "?";
/// Pine's `na`: a value that carries no type of its own
pub const TYPELESS: &str = "*";

/// The types the arithmetic algebra is defined over.
pub const NUMERIC: &[&str] = &[INT, FLOAT];
/// Everything the gate accepts.
pub const KNOWN: &[&str] = &[INT, FLOAT, BOOL, STR, COLOR, OBJECT, VOID, TYPELESS];
/// The scalars: what a map key may be and what a shape ends in.
pub const SCALARS: &[&str] = &[INT, FLOAT, BOOL, STR, COLOR];

pub fn is_numeric(ty: &str) -> bool {
    NUMERIC.contains(&ty)
}

pub fn is_known(ty: &str) -> bool {
    KNOWN.contains(&ty)
}

pub fn is_scalar(ty: &str) -> bool {
    SCALARS.contains(&ty)
}

// --- shaped types ---------------------------------------------------------

/// Module key the classes the lib publishes are identified by.
pub const LIB_MODULE: &str = "lib";
/// What separates a class's module key from its name in a class id.
pub const CLASS_SEP: &str = "#";

const OBJ: &str = "o:";
const ARRAY: &str = "a:";
const MATRIX: &str = "m:";
const MAP: &str = "M:";
const TUPLE: &str = "T:";

/// The rest of a string after its first `count` characters, empty when it is shorter.
fn skip_chars(text: &str, count: usize) -> &str {
    match text.char_indices().nth(count) {
        Some((index, _)) => &text[index..],
        None => "",
    }
}

/// Whether a string is exactly one character long, as Python's `len(ty) == 1`.
fn one_char(text: &str) -> bool {
    let mut chars = text.chars();
    chars.next().is_some() && chars.next().is_none()
}

/// The identity of one class: `<module key>#<name>`.
pub fn class_id(module_key: &str, name: &str) -> String {
    format!("{module_key}{CLASS_SEP}{name}")
}

/// The identity of a class the lib publishes.
pub fn builtin_class_id(name: &str) -> String {
    class_id(LIB_MODULE, name)
}

/// The type of an instance of one class.
pub fn object_ty(cid: &str) -> String {
    format!("{OBJ}{cid}")
}

/// The counter object a compiled `for` loop iterates (`core/pine_range`'s `PineLoop`).
pub const PINE_LOOP: &str = "o:lib#PineLoop";

/// The type of an array over one element type; `OBJECT` when the element is not known.
pub fn array_of(element: &str) -> String {
    if element == UNKNOWN || element == VOID {
        OBJECT.to_string()
    } else {
        format!("{ARRAY}{element}")
    }
}

/// The type of a matrix over one element type; `OBJECT` when the element is not known.
pub fn matrix_of(element: &str) -> String {
    if element == UNKNOWN || element == VOID {
        OBJECT.to_string()
    } else {
        format!("{MATRIX}{element}")
    }
}

/// The type of a map from a scalar key type; `OBJECT` when either half is not known.
pub fn map_of(key: &str, value: &str) -> String {
    if !is_scalar(key) || value == UNKNOWN || value == VOID {
        OBJECT.to_string()
    } else {
        format!("{MAP}{key}:{value}")
    }
}

/// The type of a Pine tuple over its element types, each one length-prefixed;
/// `OBJECT` when there are none.
pub fn tuple_of<S: AsRef<str>>(elements: &[S]) -> String {
    if elements.is_empty() {
        return OBJECT.to_string();
    }
    let mut out = String::from(TUPLE);
    for element in elements {
        let element = element.as_ref();
        out.push_str(&element.chars().count().to_string());
        out.push(':');
        out.push_str(element);
    }
    out
}

/// The lattice character a type behaves as: itself when it is one, `OBJECT` for a shape.
pub fn head(ty: &str) -> &str {
    if one_char(ty) { ty } else { OBJECT }
}

/// Whether a type carries more than its lattice character.
pub fn is_shaped(ty: &str) -> bool {
    ty.chars().nth(1).is_some()
}

/// The class id an object type names.
pub fn class_of(ty: &str) -> Option<&str> {
    ty.strip_prefix(OBJ)
}

pub fn is_array(ty: &str) -> bool {
    ty.starts_with(ARRAY)
}

pub fn is_matrix(ty: &str) -> bool {
    ty.starts_with(MATRIX)
}

pub fn is_map(ty: &str) -> bool {
    ty.starts_with(MAP)
}

pub fn is_tuple(ty: &str) -> bool {
    ty.starts_with(TUPLE)
}

/// The element types of a tuple, in order; empty when the type is not a tuple this
/// module wrote. A length is ASCII digits, the only ones `tuple_of` writes.
pub fn elements_of(ty: &str) -> Vec<&str> {
    let Some(mut rest) = ty.strip_prefix(TUPLE) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    while !rest.is_empty() {
        let Some(colon) = rest.find(':') else {
            return Vec::new();
        };
        let digits = &rest[..colon];
        if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
            return Vec::new();
        }
        // A length beyond `usize` is beyond the string as well
        let Ok(size) = digits.parse::<usize>() else {
            return Vec::new();
        };
        let after = &rest[colon + 1..];
        let end = if size == 0 {
            0
        } else {
            match after.char_indices().nth(size - 1) {
                Some((index, last)) => index + last.len_utf8(),
                None => return Vec::new(),
            }
        };
        out.push(&after[..end]);
        rest = &after[end..];
    }
    out
}

/// How many elements a tuple holds, 0 when the type is not a tuple.
pub fn arity(ty: &str) -> usize {
    elements_of(ty).len()
}

/// The element type of an array or a matrix, `UNKNOWN` when the type is neither.
pub fn element_of(ty: &str) -> &str {
    if ty.starts_with(ARRAY) || ty.starts_with(MATRIX) { &ty[2..] } else { UNKNOWN }
}

/// The key type of a map, `UNKNOWN` when the type is not a map (or has no key, where
/// Python raises).
pub fn key_of(ty: &str) -> &str {
    if !ty.starts_with(MAP) {
        return UNKNOWN;
    }
    let rest = &ty[MAP.len()..];
    match rest.chars().next() {
        Some(key) => &rest[..key.len_utf8()],
        None => UNKNOWN,
    }
}

/// The value type of a map, `UNKNOWN` when the type is not a map.
pub fn value_of(ty: &str) -> &str {
    if ty.starts_with(MAP) { skip_chars(ty, MAP.len() + 2) } else { UNKNOWN }
}

/// Whether two types cannot both be right where one value goes.
pub fn shape_mismatch(left: &str, right: &str) -> bool {
    if left == right || left == UNKNOWN || left == TYPELESS || right == UNKNOWN || right == TYPELESS
    {
        return false;
    }
    if is_tuple(left) || is_tuple(right) {
        return true;
    }
    head(left) == OBJECT && head(right) == OBJECT
}

/// The pair of types that keeps two types from being joined, if any.
pub fn shape_conflict<'a>(left: &'a str, right: &'a str) -> Option<(&'a str, &'a str)> {
    if is_tuple(left) && is_tuple(right) {
        let items = elements_of(left);
        let others = elements_of(right);
        if items.len() != others.len() {
            return Some((left, right));
        }
        return items.into_iter().zip(others).find_map(|(item, other)| shape_conflict(item, other));
    }
    if shape_mismatch(left, right) { Some((left, right)) } else { None }
}

/// How each lattice character is spelled in a message.
const RENDERED: &[(&str, &str)] = &[
    (INT, "int"),
    (FLOAT, "float"),
    (BOOL, "bool"),
    (STR, "string"),
    (COLOR, "color"),
    (OBJECT, "object"),
    (VOID, "void"),
    (UNKNOWN, "unknown"),
    (TYPELESS, "na"),
];

/// The part after the last occurrence of `sep`, the whole text when there is none
/// (`str.rpartition(sep)[2]`).
fn after_last<'a>(text: &'a str, sep: &str) -> &'a str {
    text.rfind(sep).map_or(text, |index| &text[index + sep.len()..])
}

/// A type in Pine's own spelling, for a message a user reads.
pub fn render_ty(ty: &str) -> String {
    if let Some(cid) = class_of(ty) {
        return after_last(cid, CLASS_SEP).to_string();
    }
    if ty.starts_with(ARRAY) {
        return format!("array<{}>", render_ty(element_of(ty)));
    }
    if ty.starts_with(MATRIX) {
        return format!("matrix<{}>", render_ty(element_of(ty)));
    }
    if ty.starts_with(MAP) {
        return format!("map<{}, {}>", render_ty(key_of(ty)), render_ty(value_of(ty)));
    }
    if is_tuple(ty) {
        let items: Vec<String> = elements_of(ty).into_iter().map(render_ty).collect();
        return format!("[{}]", items.join(", "));
    }
    RENDERED
        .iter()
        .find(|&&(char, _)| char == ty)
        .map_or_else(|| ty.to_string(), |&(_, name)| name.to_string())
}

/// The classes the lib publishes as Pine objects, by the name an annotation spells them with.
pub const BUILTIN_CLASS_NAMES: &[&str] = &[
    "Line",
    "LineFill",
    "Label",
    "Box",
    "Table",
    "Polyline",
    "HLine",
    "ChartPoint",
    "VolumeRow",
    "Currency",
    "DayOfWeek",
    "Extend",
    "Footprint",
    "Position",
    "Size",
    "Format",
    "AlignEnum",
    "FontFamilyEnum",
    "FormatEnum",
];

/// The object type of a class the lib publishes (`BUILTIN_CLASSES.get(name)`).
pub fn builtin_class(name: &str) -> Option<String> {
    BUILTIN_CLASS_NAMES.contains(&name).then(|| object_ty(&builtin_class_id(name)))
}

/// The lib namespace a builtin object's methods live in, by class name.
pub const BUILTIN_NAMESPACES: &[(&str, &str)] = &[
    ("Line", "line"),
    ("LineFill", "linefill"),
    ("Label", "label"),
    ("Box", "box"),
    ("Table", "table"),
    ("Polyline", "polyline"),
];

/// The lib namespace a shaped receiver's methods are looked up in.
pub fn namespace_of(ty: &str) -> Option<&'static str> {
    if ty.starts_with(ARRAY) {
        return Some("array");
    }
    if ty.starts_with(MATRIX) {
        return Some("matrix");
    }
    if ty.starts_with(MAP) {
        return Some("map");
    }
    let cid = class_of(ty)?;
    let (module, name) = match cid.rfind(CLASS_SEP) {
        Some(index) => (&cid[..index], &cid[index + CLASS_SEP.len()..]),
        None => ("", cid),
    };
    if module != LIB_MODULE {
        return None;
    }
    BUILTIN_NAMESPACES.iter().find(|&&(class, _)| class == name).map(|&(_, namespace)| namespace)
}

/// Least upper bound of two types.
pub fn join(left: &str, right: &str) -> String {
    if left == right || right == TYPELESS {
        return left.to_string();
    }
    if left == TYPELESS {
        return right.to_string();
    }
    if is_tuple(left) && is_tuple(right) {
        // A tuple joins POSITION BY POSITION; two arities are two different types
        let items = elements_of(left);
        let others = elements_of(right);
        if items.len() != others.len() {
            return UNKNOWN.to_string();
        }
        let joined: Vec<String> =
            items.into_iter().zip(others).map(|(item, other)| join(item, other)).collect();
        return tuple_of(&joined);
    }
    if is_numeric(left) && is_numeric(right) {
        return FLOAT.to_string();
    }
    UNKNOWN.to_string()
}

// --- operators ------------------------------------------------------------

/// The arithmetic operators, all following the same int/float algebra.
fn is_arithmetic(op: Operator) -> bool {
    matches!(
        op,
        Operator::Add
            | Operator::Sub
            | Operator::Mult
            | Operator::Div
            | Operator::Mod
            | Operator::FloorDiv
            | Operator::Pow
    )
}

/// Type of a binary arithmetic expression (MEASURED: `int op int -> int` for every
/// arithmetic operator, the division and the modulo included).
pub fn binop_type(op: Operator, left: &str, right: &str) -> &'static str {
    let (mut left, mut right) = (head(left), head(right));
    // An untyped ``na`` operand takes the OTHER operand's type; two of them are int
    if left == TYPELESS && right == TYPELESS {
        left = INT;
        right = INT;
    } else if left == TYPELESS {
        left = right;
    } else if right == TYPELESS {
        right = left;
    }
    if op == Operator::Add && left == STR && right == STR {
        return STR;
    }
    if is_arithmetic(op) && is_numeric(left) && is_numeric(right) {
        return if left == INT && right == INT { INT } else { FLOAT };
    }
    // A bool operand is a Pine type error, and bit operators do not exist in Pine
    UNKNOWN
}

/// Type of a unary expression: the sign is type-preserving, `not` is a bool.
pub fn unaryop_type(op: UnaryOperator, operand: &str) -> &'static str {
    match op {
        UnaryOperator::Not => BOOL,
        UnaryOperator::UAdd | UnaryOperator::USub => match operand {
            TYPELESS => TYPELESS,
            INT => INT,
            FLOAT => FLOAT,
            _ => UNKNOWN,
        },
        UnaryOperator::Invert => UNKNOWN,
    }
}

/// Type of a comparison: always bool in Pine.
pub fn compare_type(_left: &str, _right: &str) -> &'static str {
    BOOL
}

/// Pine type of a Python literal.
pub fn constant_type(value: &Constant) -> &'static str {
    match value {
        Constant::None => VOID,
        Constant::Bool(_) => BOOL,
        Constant::Int(_) => INT,
        Constant::Float(_) => FLOAT,
        Constant::Str(_) => STR,
        _ => UNKNOWN,
    }
}

// --- call result rules ----------------------------------------------------

/// How the result of a call is typed: a value of `LIB_TYPE_OVERRIDES` or
/// `BUILTIN_CALL_TYPES`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CallRule {
    /// A type (a character or a whole shape) or one of the named forms (`'arg0'`,
    /// `'all_int'`, `'same_array0'`, `'put_array:1'`, ...), exactly as Python spells it
    Is(&'static str),
    /// Argument count -> the rule for that count
    ByArity(&'static [(usize, CallRule)]),
    /// Rules tried in order until one answers
    FirstOf(&'static [CallRule]),
}

use CallRule::{ByArity, FirstOf, Is};

/// Names a module calls under their bare spelling, with the Pine type of the result.
pub const BUILTIN_CALL_TYPES: &[(&str, CallRule)] = &[
    ("int", Is(INT)),
    ("float", Is(FLOAT)),
    ("bool", Is(BOOL)),
    ("str", Is(STR)),
    ("len", Is(INT)),
    ("abs", Is("arg0")),
    ("NA", FirstOf(&[Is("na_arg"), Is(TYPELESS)])),
    ("run", Is(VOID)),
    ("__sec_write__", Is(VOID)),
    ("__sec_signal__", Is(VOID)),
    ("__sec_wait__", Is(VOID)),
    ("__pyne_bool_na·__", Is(BOOL)),
    ("__ltf_unzip__", Is(OBJECT)),
    ("range", Is(OBJECT)),
    ("pine_range", Is(OBJECT)),
    ("pine_loop", Is(PINE_LOOP)),
    ("inline_series", Is("arg0")),
    ("udt_copy", Is("arg0")),
    ("Exported", Is(OBJECT)),
    ("shadowed_namespace", Is(OBJECT)),
    ("round", ByArity(&[(1, Is(INT)), (2, Is(FLOAT))])),
    ("cast_int", Is(INT)),
    ("cast_float", Is(FLOAT)),
    ("cast_bool", Is(BOOL)),
    ("cast_string", Is(STR)),
    ("cast_color", Is(COLOR)),
    ("cast_label", Is(OBJECT)),
    ("cast_table", Is(OBJECT)),
    ("cast_box", Is(OBJECT)),
    ("cast_line", Is(OBJECT)),
    ("cast_linefill", Is(OBJECT)),
    ("max", Is("all_int")),
    ("min", Is("all_int")),
    ("print", Is(VOID)),
    ("enumerate", Is(OBJECT)),
];

/// Bare names the pipeline's own plumbing binds, with the type a read of them has.
pub const BUILTIN_NAME_TYPES: &[(&str, &str)] = &[
    ("__name__", STR),
    ("__file__", STR),
    ("__active_security__", OBJECT),
    ("__same_context__", OBJECT),
];

/// Lib names whose TradingView type is not what the Python annotation says, keyed by
/// the dotted lib path. See the Python table for the forms and the measurements.
pub const LIB_TYPE_OVERRIDES: &[(&str, CallRule)] = &[
    ("math.max", Is("all_int")),
    ("math.min", Is("all_int")),
    ("math.abs", Is("arg0")),
    ("math.sum", Is(FLOAT)),
    ("math.avg", Is(FLOAT)),
    ("math.sign", Is(FLOAT)),
    ("math.round", ByArity(&[(1, Is(INT)), (2, Is(FLOAT))])),
    ("math.floor", Is(INT)),
    ("math.ceil", Is(INT)),
    ("math.sqrt", Is(FLOAT)),
    ("math.pow", Is(FLOAT)),
    ("math.log", Is(FLOAT)),
    ("math.log10", Is(FLOAT)),
    ("math.exp", Is(FLOAT)),
    ("math.sin", Is(FLOAT)),
    ("math.cos", Is(FLOAT)),
    ("math.tan", Is(FLOAT)),
    ("math.asin", Is(FLOAT)),
    ("math.acos", Is(FLOAT)),
    ("math.atan", Is(FLOAT)),
    ("math.todegrees", Is(FLOAT)),
    ("math.toradians", Is(FLOAT)),
    ("math.random", Is(FLOAT)),
    ("math.round_to_mintick", Is(FLOAT)),
    ("math.pi", Is(FLOAT)),
    ("math.e", Is(FLOAT)),
    ("nz", Is("join_args")),
    ("na", FirstOf(&[Is("na_arg"), Is(BOOL)])),
    ("_na_none", Is(TYPELESS)),
    ("_na_float", Is(FLOAT)),
    ("ta.change", Is("arg0")),
    ("fixnan", Is("arg0")),
    ("ta.valuewhen", Is("arg1")),
    ("ta.median", Is("arg0")),
    ("ta.range", Is("arg0")),
    ("ta.mode", Is("arg0")),
    ("int", Is(INT)),
    ("float", Is(FLOAT)),
    ("bool", Is(BOOL)),
    ("string", Is(STR)),
    ("color", Is(COLOR)),
    ("input", Is("arg0")),
    ("input.int", Is(INT)),
    ("input.float", Is(FLOAT)),
    ("input.bool", Is(BOOL)),
    ("input.string", Is(STR)),
    ("input.text_area", Is(STR)),
    ("input.color", Is(COLOR)),
    ("input.symbol", Is(STR)),
    ("input.timeframe", Is(STR)),
    ("input.session", Is(STR)),
    ("input.source", Is(FLOAT)),
    ("input.price", Is(FLOAT)),
    ("input.time", Is(INT)),
    ("input.enum", Is(OBJECT)),
    // --- the containers
    ("array.get", Is("elem0")),
    ("array.first", Is("elem0")),
    ("array.last", Is("elem0")),
    ("array.pop", Is("elem0")),
    ("array.shift", Is("elem0")),
    ("array.remove", Is("elem0")),
    ("array.min", Is("elem0")),
    ("array.max", Is("elem0")),
    ("array.mode", Is("elem0")),
    ("array.median", Is("elem0")),
    ("array.range", Is("elem0")),
    ("array.avg", Is("elem0")),
    ("array.sum", Is("elem0")),
    ("array.stdev", Is("elem0")),
    ("array.variance", Is("elem0")),
    ("array.percentile_nearest_rank", Is("elem0")),
    ("array.percentile_linear_interpolation", Is("elem0")),
    ("array.percentrank", Is("elem0")),
    ("array.abs", Is("same_array0")),
    ("array.standardize", Is("same_array0")),
    ("array.copy", Is("same_array0")),
    ("array.slice", Is("same_array0")),
    ("array.concat", Is("merge_array")),
    ("array.push", Is("put_array:1")),
    ("array.unshift", Is("put_array:1")),
    ("array.fill", Is("put_array:1")),
    ("array.set", Is("put_array:2")),
    ("array.insert", Is("put_array:2")),
    ("map.put", Is("put_map:2")),
    ("matrix.set", Is("put_matrix:3")),
    ("matrix.fill", Is("put_matrix:1")),
    ("matrix.add_row", Is("put_matrix_array:2")),
    ("matrix.add_col", Is("put_matrix_array:2")),
    ("array.from_items", Is("array_of_join_args")),
    ("array.new", Is("array_of_arg1")),
    ("matrix.get", Is("elem0")),
    ("matrix.avg", Is("elem0")),
    ("matrix.min", Is("elem0")),
    ("matrix.max", Is("elem0")),
    ("matrix.mode", Is("elem0")),
    ("matrix.median", Is("elem0")),
    ("matrix.det", Is("elem0")),
    ("matrix.trace", Is("elem0")),
    ("matrix.row", Is("array_of_elem0")),
    ("matrix.col", Is("array_of_elem0")),
    ("matrix.remove_row", Is("array_of_elem0")),
    ("matrix.remove_col", Is("array_of_elem0")),
    ("matrix.copy", Is("same_matrix0")),
    ("matrix.transpose", Is("same_matrix0")),
    ("matrix.inv", Is("same_matrix0")),
    ("matrix.pinv", Is("same_matrix0")),
    ("matrix.pow", Is("same_matrix0")),
    ("matrix.submatrix", Is("same_matrix0")),
    ("matrix.eigenvalues", Is("array_of_elem0")),
    ("matrix.eigenvectors", Is("same_matrix0")),
    ("matrix.kron", Is("merge_matrix")),
    ("matrix.concat", Is("merge_matrix")),
    ("matrix.sum", Is("merge_matrix_or_scalar")),
    ("matrix.diff", Is("merge_matrix_or_scalar")),
    ("matrix.mult", Is("matrix_mult")),
    ("matrix.new", Is("matrix_of_arg2")),
    ("map.get", Is("map_value0")),
    ("map.remove", Is("map_value0")),
    ("map.keys", Is("array_of_map_keys0")),
    ("map.values", Is("array_of_map_values0")),
    ("map.copy", Is("same_map0")),
];

fn rule_index(table: &'static [(&'static str, CallRule)]) -> HashMap<&'static str, CallRule> {
    table.iter().copied().collect()
}

/// The measured type rule of a dotted lib path (`LIB_TYPE_OVERRIDES.get(path)`).
pub fn lib_type_override(path: &str) -> Option<CallRule> {
    static INDEX: OnceLock<HashMap<&'static str, CallRule>> = OnceLock::new();
    INDEX.get_or_init(|| rule_index(LIB_TYPE_OVERRIDES)).get(path).copied()
}

/// The result rule of a bare builtin call (`BUILTIN_CALL_TYPES.get(name)`).
pub fn builtin_call_type(name: &str) -> Option<CallRule> {
    static INDEX: OnceLock<HashMap<&'static str, CallRule>> = OnceLock::new();
    INDEX.get_or_init(|| rule_index(BUILTIN_CALL_TYPES)).get(name).copied()
}

/// The type a read of a bare plumbing name has (`BUILTIN_NAME_TYPES.get(name)`).
pub fn builtin_name_type(name: &str) -> Option<&'static str> {
    BUILTIN_NAME_TYPES.iter().find(|&&(known, _)| known == name).map(|&(_, ty)| ty)
}

/// Declared parameter order of the lib names whose module has no registry entry.
pub const OVERRIDE_PARAM_NAMES: &[(&str, &[&str])] =
    &[("input", &["defval", "title", "tooltip", "inline", "group", "display", "active"])];

/// The declared parameter order of an override name (`OVERRIDE_PARAM_NAMES.get(name)`).
pub fn override_param_names(name: &str) -> Option<&'static [&'static str]> {
    OVERRIDE_PARAM_NAMES.iter().find(|&&(known, _)| known == name).map(|&(_, names)| names)
}

// --- the lowering stamp ---------------------------------------------------

fn is_expr(ast: &Ast, id: NodeId) -> bool {
    ast.kind(id).class().category == Category::Expr
}

/// Type a subtree a lowering pass has just built: the plumbing from its children up,
/// the root as `result`. A node that already carries a stamp is left as it is,
/// subtree and all.
pub fn stamp_lowering(ast: &mut Ast, root: NodeId, result: &str) -> NodeId {
    for child in children(ast, root) {
        if is_expr(ast, child) {
            stamp_emitted(ast, child);
        }
    }
    set_ty(ast, root, result)
}

/// Stamp every expression child of a node, returning nothing.
fn stamp_expr_children(ast: &mut Ast, id: NodeId) {
    for child in children(ast, id) {
        if is_expr(ast, child) {
            stamp_emitted(ast, child);
        }
    }
}

/// Type one emitted node from its children up.
fn stamp_emitted(ast: &mut Ast, id: NodeId) -> String {
    if let Some(existing) = extra_str(ast, id, TY_ATTR) {
        return existing.to_string();
    }
    let ty: String = match ast.kind(id) {
        Kind::Constant { value, .. } => constant_type(value).to_string(),
        &Kind::BinOp { left, op, right } => {
            let left = stamp_emitted(ast, left);
            let right = stamp_emitted(ast, right);
            binop_type(op, &left, &right).to_string()
        }
        &Kind::UnaryOp { op, operand } => {
            let operand = stamp_emitted(ast, operand);
            unaryop_type(op, &operand).to_string()
        }
        &Kind::NamedExpr { value, .. } => stamp_emitted(ast, value),
        &Kind::IfExp { test, body, orelse } => {
            stamp_emitted(ast, test);
            let body = stamp_emitted(ast, body);
            let orelse = stamp_emitted(ast, orelse);
            join(&body, &orelse)
        }
        Kind::Compare { .. } | Kind::BoolOp { .. } => {
            stamp_expr_children(ast, id);
            BOOL.to_string()
        }
        _ => {
            // The machinery's own references: known, carrying no numeric algebra
            stamp_expr_children(ast, id);
            OBJECT.to_string()
        }
    };
    set_ty(ast, id, &ty);
    ty
}

// --- annotations ----------------------------------------------------------

/// Names that map straight onto a lattice type wherever they appear in an annotation.
pub const ANNOTATION_TYPES: &[(&str, &str)] = &[
    ("int", INT),
    ("PyneInt", INT),
    ("float", FLOAT),
    ("PyneFloat", FLOAT),
    ("bool", BOOL),
    ("PyneBool", BOOL),
    ("str", STR),
    ("PyneStr", STR),
    ("Color", COLOR),
    ("None", VOID),
];

/// Annotation heads whose payload is the type itself.
const TRANSPARENT_SUBSCRIPTS: &[&str] = &[
    "Series",
    "PersistentSeries",
    "NA",
    "Persistent",
    "IBPersistent",
    "IBPersistentSeries",
    "Optional",
];
/// Annotation heads whose payload is the array element type.
const ARRAY_SUBSCRIPTS: &[&str] = &["list", "Array", "SequenceView"];
/// Annotation heads whose payload is the matrix element type.
const MATRIX_SUBSCRIPTS: &[&str] = &["Matrix"];
/// Annotation heads whose payload is a (key, value) pair.
const MAP_SUBSCRIPTS: &[&str] = &["dict", "Map"];
/// Annotation heads whose payload is the tuple's element types.
const TUPLE_SUBSCRIPTS: &[&str] = &["tuple", "Tuple"];
/// Annotation heads that denote a known non-scalar carrying no element type.
const OBJECT_SUBSCRIPTS: &[&str] = &["set"];
/// Container names with no payload at all.
const CONTAINER_NAMES: &[&str] =
    &["Array", "Matrix", "Map", "SequenceView", "list", "tuple", "dict", "set"];
/// Annotation names that accept `None` whatever they wrap.
const NONE_TAKING_NAMES: &[&str] = &["Any", "object", "NoneType"];
/// Annotation heads that add a `None` member to whatever they wrap.
const OPTIONAL_SUBSCRIPTS: &[&str] = &["Optional"];

/// Class spelling -> class id, for every class visible where an annotation stands.
pub type Classes = HashMap<String, String>;

/// The leaf name of a `Name` or an `Attribute`, empty for anything else.
fn leaf_name(ast: &Ast, id: NodeId) -> &str {
    match ast.kind(id) {
        Kind::Name { id: name, .. } => name,
        Kind::Attribute { attr, .. } => attr,
        _ => "",
    }
}

/// Render a dotted name expression, `None` when it is not one (`_dotted`).
pub fn dotted(ast: &Ast, node: NodeId) -> Option<String> {
    let mut parts: Vec<&str> = Vec::new();
    let mut current = node;
    while let Kind::Attribute { value, attr, .. } = ast.kind(current) {
        parts.push(attr);
        current = *value;
    }
    let Kind::Name { id, .. } = ast.kind(current) else {
        return None;
    };
    parts.push(id);
    parts.reverse();
    Some(parts.join("."))
}

/// Whether an annotation is a series/persistence wrapper that names no type.
pub fn bare_wrapper(ast: &Ast, node: NodeId) -> bool {
    TRANSPARENT_SUBSCRIPTS.contains(&leaf_name(ast, node))
}

/// Python keywords; a forward reference spelling one is not read natively.
const KEYWORDS: &[&str] = &[
    "and", "as", "assert", "async", "await", "break", "class", "continue", "def", "del", "elif",
    "else", "except", "finally", "for", "from", "global", "if", "import", "in", "is", "lambda",
    "nonlocal", "not", "or", "pass", "raise", "return", "try", "while", "with", "yield",
];

/// Parse a stringized annotation (`ast.parse(text, mode='eval').body`) into its own
/// arena.
///
/// Only a spelling whose parse provably agrees with CPython's is read: printable ASCII
/// on one line, not indented, without the punctuation of statements, slices, stars,
/// comments or line continuations, and without keywords. Everything else, and a
/// spelling the parser refuses, answers `Unsupported`, so the Python step decides.
pub(crate) fn forward_reference(text: &str) -> Result<(Ast, NodeId), Unsupported> {
    let refuse = || Unsupported(format!("forward reference {text:?}"));
    if text.starts_with(' ')
        || !text
            .bytes()
            .all(|byte| (0x20..0x7f).contains(&byte) && !b"#\\:;=*@{}`$?!".contains(&byte))
    {
        return Err(refuse());
    }
    // A number running into letters (``1if``, ``0x1for``) is tokenized differently
    // by the two parsers' rules on literal suffixes, so it is refused with the keywords
    let mut words = text.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'));
    if words.any(|word| {
        KEYWORDS.contains(&word)
            || word.starts_with(|c: char| c.is_ascii_digit())
                && word.contains(|c: char| c.is_ascii_alphabetic())
    }) {
        return Err(refuse());
    }
    let mut ast = Ast::new();
    let body = pyne_parse::parse_expression_into(&mut ast, text).map_err(|_| refuse())?;
    Ok((ast, body))
}

/// Map a Python annotation onto a Pine type (`annotation_type`).
///
/// :param node: The annotation expression, or `None` when there is none
/// :param classes: Class spelling -> class id, visible where the annotation stands
pub fn annotation_type(
    ast: &Ast,
    node: Option<NodeId>,
    classes: &Classes,
) -> Result<String, Unsupported> {
    let Some(node) = node else {
        return Ok(UNKNOWN.to_string());
    };
    match ast.kind(node) {
        Kind::Constant { value, .. } => match value {
            Constant::None => Ok(VOID.to_string()),
            // A stringized forward reference: ``'_ExitOrderKey'``, ``'list[Pivot]'``
            Constant::Str(text) => {
                let (parsed, body) = forward_reference(text)?;
                annotation_type(&parsed, Some(body), classes)
            }
            _ => Ok(UNKNOWN.to_string()),
        },
        Kind::Name { id, .. } => Ok(named_type(id, classes)),
        Kind::Attribute { attr, .. } => {
            // A qualified spelling is resolved as a whole first; only one nothing
            // published under its full path falls back to the leaf
            if let Some(found) = dotted(ast, node).and_then(|spelled| classes.get(&spelled)) {
                return Ok(object_ty(found));
            }
            Ok(named_type(attr, classes))
        }
        Kind::BinOp { op: Operator::BitOr, .. } => union_type(ast, node, classes),
        &Kind::Subscript { value, slice, .. } => subscript_type(ast, value, slice, classes),
        _ => Ok(UNKNOWN.to_string()),
    }
}

/// The elements of a `Tuple` slice, `None` for any other slice.
fn tuple_elts(ast: &Ast, slice: NodeId) -> Option<&[NodeId]> {
    match ast.kind(slice) {
        Kind::Tuple { elts, .. } => Some(elts),
        _ => None,
    }
}

/// Type of a subscripted annotation (`_subscript_type`).
fn subscript_type(
    ast: &Ast,
    value: NodeId,
    slice: NodeId,
    classes: &Classes,
) -> Result<String, Unsupported> {
    let name = leaf_name(ast, value);
    if TRANSPARENT_SUBSCRIPTS.contains(&name) {
        return annotation_type(ast, Some(slice), classes);
    }
    if ARRAY_SUBSCRIPTS.contains(&name) {
        return Ok(array_of(&annotation_type(ast, Some(slice), classes)?));
    }
    if MATRIX_SUBSCRIPTS.contains(&name) {
        return Ok(matrix_of(&annotation_type(ast, Some(slice), classes)?));
    }
    if MAP_SUBSCRIPTS.contains(&name) {
        let pair = tuple_elts(ast, slice).unwrap_or(&[]);
        if pair.len() != 2 {
            return Ok(OBJECT.to_string());
        }
        let key = annotation_type(ast, Some(pair[0]), classes)?;
        let value = annotation_type(ast, Some(pair[1]), classes)?;
        return Ok(map_of(&key, &value));
    }
    if TUPLE_SUBSCRIPTS.contains(&name) {
        // ``tuple[int]`` subscripts with the element itself, ``tuple[int, float]``
        // with a tuple of them
        let single = [slice];
        let items = tuple_elts(ast, slice).unwrap_or(&single);
        let open = items.iter().any(|&item| {
            matches!(ast.kind(item), Kind::Constant { value: Constant::Ellipsis, .. })
        });
        if open {
            // ``tuple[X, ...]`` has no fixed arity, so it is no Pine tuple
            return Ok(OBJECT.to_string());
        }
        let types = items
            .iter()
            .map(|&item| annotation_type(ast, Some(item), classes))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(tuple_of(&types));
    }
    if OBJECT_SUBSCRIPTS.contains(&name) {
        return Ok(OBJECT.to_string());
    }
    Ok(UNKNOWN.to_string())
}

/// Resolve a bare annotation name to a type (`_named_type`).
fn named_type(name: &str, classes: &Classes) -> String {
    if let Some(&(_, ty)) = ANNOTATION_TYPES.iter().find(|&&(known, _)| known == name) {
        return ty.to_string();
    }
    // A container spelled without its payload stays an object of unknown elements,
    // even where the module has the container class itself in scope
    if CONTAINER_NAMES.contains(&name) {
        return OBJECT.to_string();
    }
    // A class the module declares or imports wins over the builtin of the same name
    if let Some(found) = classes.get(name) {
        return object_ty(found);
    }
    builtin_class(name).unwrap_or_else(|| UNKNOWN.to_string())
}

/// Whether a union member only says "this may be absent" (`_is_typeless`).
fn is_typeless_member(ast: &Ast, node: NodeId) -> bool {
    match ast.kind(node) {
        Kind::Constant { value, .. } => *value == Constant::None,
        Kind::Name { id, .. } => id == "NA",
        Kind::Attribute { attr, .. } => attr == "NA",
        _ => false,
    }
}

/// Type of an annotation union, ignoring its absence markers (`_union_type`).
fn union_type(ast: &Ast, node: NodeId, classes: &Classes) -> Result<String, Unsupported> {
    fn flatten(ast: &Ast, expr: NodeId, members: &mut Vec<NodeId>) {
        match ast.kind(expr) {
            &Kind::BinOp { left, op: Operator::BitOr, right } => {
                flatten(ast, left, members);
                flatten(ast, right, members);
            }
            _ => members.push(expr),
        }
    }
    let mut members = Vec::new();
    flatten(ast, node, &mut members);
    let mut contributing = members.into_iter().filter(|&member| !is_typeless_member(ast, member));
    let Some(first) = contributing.next() else {
        // ``NA | None`` and friends: absence only, no type to speak of
        return Ok(UNKNOWN.to_string());
    };
    let mut result = annotation_type(ast, Some(first), classes)?;
    for member in contributing {
        result = join(&result, &annotation_type(ast, Some(member), classes)?);
    }
    Ok(result)
}

/// Whether a parameter's annotation accepts a `None` argument (`annotation_takes_none`,
/// mirroring `core/overload.py::_check_type(None, hint, strict=True)`).
pub fn annotation_takes_none(ast: &Ast, node: Option<NodeId>) -> Result<bool, Unsupported> {
    let Some(node) = node else {
        // A parameter with no hint is read as ``Any``
        return Ok(true);
    };
    match ast.kind(node) {
        Kind::Constant { value, .. } => match value {
            Constant::None => Ok(true),
            Constant::Str(text) => {
                let (parsed, body) = forward_reference(text)?;
                annotation_takes_none(&parsed, Some(body))
            }
            _ => Ok(false),
        },
        Kind::Name { id, .. } => Ok(NONE_TAKING_NAMES.contains(&id.as_str())),
        Kind::Attribute { attr, .. } => Ok(NONE_TAKING_NAMES.contains(&attr.as_str())),
        &Kind::BinOp { left, op: Operator::BitOr, right } => {
            Ok(annotation_takes_none(ast, Some(left))? || annotation_takes_none(ast, Some(right))?)
        }
        &Kind::Subscript { value, slice, .. } => {
            let name = leaf_name(ast, value);
            if OPTIONAL_SUBSCRIPTS.contains(&name) {
                return Ok(true);
            }
            if name == "Union" {
                let single = [slice];
                for &member in tuple_elts(ast, slice).unwrap_or(&single) {
                    if annotation_takes_none(ast, Some(member))? {
                        return Ok(true);
                    }
                }
            }
            Ok(false)
        }
        _ => Ok(false),
    }
}

/// Whether a type is the one that changes behavior.
pub fn is_int_typed(ty: &str) -> bool {
    ty == INT
}

// --- the overload pin -----------------------------------------------------

/// Types a pin can name: one witness value stands for the whole type.
pub const PINNABLE: &[&str] = &[INT, FLOAT, BOOL, STR];

/// Pin character of an array or a matrix of one scalar.
pub const PIN_CONTAINERS: &[(&str, char)] = &[
    ("a:i", 'I'),
    ("a:f", 'F'),
    ("a:b", 'B'),
    ("a:s", 'S'),
    ("m:i", '1'),
    ("m:f", '2'),
    ("m:b", '3'),
    ("m:s", '4'),
];

/// Pin character for a position the runtime has no witness value for.
pub const PIN_ANY: char = '*';

/// The pin character of a scalar container type.
fn pin_container(ty: &str) -> Option<char> {
    PIN_CONTAINERS.iter().find(|&&(shaped, _)| shaped == ty).map(|&(_, char)| char)
}

/// The scalar container type a pin character names (`_PIN_CONTAINER_TYPES`).
fn pin_container_type(want: &str) -> Option<&'static str> {
    PIN_CONTAINERS
        .iter()
        .find(|&&(_, char)| one_char(want) && want.starts_with(char))
        .map(|&(shaped, _)| shaped)
}

/// The overload pin a call site's argument types justify, `None` where the site is
/// not worth pinning.
pub fn pin_for<S: AsRef<str>>(arg_types: &[S]) -> Option<String> {
    let chars = pin_chars(arg_types);
    let worth = chars.chars().any(|c| c == 'i' || PIN_CONTAINERS.iter().any(|&(_, p)| p == c));
    worth.then_some(chars)
}

/// A call site's argument types as the selector's exact pass reads them.
pub fn pin_chars<S: AsRef<str>>(arg_types: &[S]) -> String {
    arg_types
        .iter()
        .map(|ty| {
            let ty = ty.as_ref();
            if let Some(char) = pin_container(ty) {
                char.to_string()
            } else if PINNABLE.contains(&ty) {
                ty.to_string()
            } else {
                PIN_ANY.to_string()
            }
        })
        .collect()
}

/// Type of a call to an overload group: the implementations' agreed return type.
pub fn overload_result<S: AsRef<str>>(returns: &[S]) -> &str {
    let Some(first) = returns.first() else {
        return UNKNOWN;
    };
    let first = first.as_ref();
    if returns[1..].iter().all(|ret| ret.as_ref() == first) { first } else { UNKNOWN }
}

/// Type character of a literal `None` default.
pub const NONE_DEFAULT: &str = "0";
/// A parameter whose default satisfies its own declared type.
pub const FIT_OMISSIBLE: char = 'y';
/// A parameter with no usable default.
pub const FIT_REQUIRED: char = 'n';
/// A parameter whose default cannot be typed here.
pub const FIT_UNSURE: char = '?';

/// How one implementation answers a pin in one pass of the selector.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Verdict {
    No,
    Maybe,
    Yes,
}

/// What a parameter contributes to a call that leaves it to its default.
///
/// :param default: The default's type, `TYPELESS`, `NONE_DEFAULT`, or `None` when the
///                 parameter has none
pub fn default_fit(declared: &str, default: Option<&str>, takes_none: bool) -> char {
    let Some(default) = default else {
        return FIT_REQUIRED;
    };
    if default == NONE_DEFAULT {
        return if takes_none { FIT_OMISSIBLE } else { FIT_REQUIRED };
    }
    if default == TYPELESS {
        return FIT_OMISSIBLE;
    }
    if default == UNKNOWN || declared == UNKNOWN {
        return FIT_UNSURE;
    }
    // Heads, not shapes: the runtime checks a default with an ``isinstance``
    if head(default) == head(declared) { FIT_OMISSIBLE } else { FIT_REQUIRED }
}

/// One overload implementation's shape, as the static selection reads it.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ImplSig {
    /// Type of each positional parameter, in declaration order
    pub params: Vec<String>,
    /// How many of them a call has to pass
    pub required: usize,
    /// Whether a `*args` lets it take any number of them
    pub open_ended: bool,
    /// What the implementation returns
    pub ret: String,
    /// One `FIT_*` character per parameter, the positional ones first, then the
    /// keyword-only ones
    pub fits: String,
    /// Name of each positional parameter
    pub names: Vec<String>,
}

/// The characters of a pin as one-character strings, as Python's `tuple(pin)`.
fn pin_items(pin: &str) -> Vec<&str> {
    pin.char_indices().map(|(index, char)| &pin[index..index + char.len_utf8()]).collect()
}

/// Return type of the implementation a pinned call site selects (`overload_pick`).
pub fn overload_pick<'a, S: AsRef<str>>(
    impls: &'a [ImplSig],
    pin: &str,
    shapes: &[S],
) -> Option<&'a str> {
    overload_choice(impls, pin, shapes).map(|index| impls[index].ret.as_str())
}

/// The implementation a pinned call site selects, by its index in `impls`
/// (`overload_choice`): the selector's exact pass, positional half first.
pub fn overload_choice<S: AsRef<str>>(impls: &[ImplSig], pin: &str, shapes: &[S]) -> Option<usize> {
    let wanted = pin_items(pin);
    for exact_arity in [true, false] {
        for (index, imp) in impls.iter().enumerate() {
            match takes(imp, &wanted, exact_arity, shapes) {
                Verdict::No => continue,
                Verdict::Yes => return Some(index),
                Verdict::Maybe => return None,
            }
        }
    }
    None
}

fn shape_at<S: AsRef<str>>(shapes: &[S], index: usize) -> &str {
    shapes.get(index).map_or(UNKNOWN, AsRef::as_ref)
}

/// How one implementation answers a pin in one half of the exact pass (`_takes`).
fn takes<S: AsRef<str>>(
    imp: &ImplSig,
    wanted: &[&str],
    exact_arity: bool,
    shapes: &[S],
) -> Verdict {
    if imp.open_ended {
        return Verdict::Maybe;
    }
    let fits: Vec<char> = imp.fits.chars().collect();
    if exact_arity {
        // The positional half counts EVERY visible parameter against the arguments
        if fits.len() != wanted.len() || imp.params.len() != wanted.len() {
            return Verdict::No;
        }
    } else if !(imp.required <= wanted.len() && wanted.len() <= imp.params.len()) {
        return Verdict::No;
    }
    let mut verdict = Verdict::Yes;
    for (index, (declared, want)) in imp.params.iter().zip(wanted).enumerate() {
        let taken = takes_position(declared, want, shape_at(shapes, index));
        if taken == Verdict::No {
            return Verdict::No;
        }
        verdict = verdict.min(taken);
    }
    if exact_arity {
        return verdict;
    }
    for &fit in fits.iter().skip(wanted.len()) {
        if fit == FIT_REQUIRED {
            return Verdict::No;
        }
        if fit == FIT_UNSURE {
            verdict = Verdict::Maybe;
        }
    }
    verdict
}

/// How one parameter answers one argument of the exact pass (`_takes_position`).
fn takes_position(declared: &str, want: &str, given: &str) -> Verdict {
    if one_char(want) && want.starts_with(PIN_ANY) {
        if given == TYPELESS {
            // The selector takes a typeless na for whatever the position declares
            return if declared == UNKNOWN { Verdict::Maybe } else { Verdict::Yes };
        }
        // No witness: an object of a known class is taken by its class alone
        if class_of(given).is_some() && declared != UNKNOWN && declared != OBJECT {
            return if declared == given { Verdict::Yes } else { Verdict::No };
        }
        return Verdict::Maybe;
    }
    if let Some(shaped) = pin_container_type(want) {
        if declared == shaped {
            return Verdict::Yes;
        }
        if declared == UNKNOWN
            || declared == OBJECT
            || (((is_array(declared) && is_array(shaped))
                || (is_matrix(declared) && is_matrix(shaped)))
                && element_of(declared) == UNKNOWN)
        {
            return Verdict::Maybe;
        }
        return Verdict::No;
    }
    let declared = head(declared);
    if declared == UNKNOWN {
        return Verdict::Maybe;
    }
    if declared == want { Verdict::Yes } else { Verdict::No }
}

/// Return type of the implementation a call with keyword arguments selects
/// (`overload_bind`).
pub fn overload_bind<'a, S: AsRef<str>>(
    impls: &'a [ImplSig],
    chars: &str,
    shapes: &[S],
    kw_chars: &[(String, String)],
    kw_shapes: &[(String, String)],
) -> Option<&'a str> {
    overload_binding(impls, chars, shapes, kw_chars, kw_shapes)
        .map(|index| impls[index].ret.as_str())
}

/// The implementation a call with keyword arguments selects, by its index in `impls`
/// (`overload_binding`).
///
/// :param kw_chars: Parameter name -> the keyword argument's pin character, in order
/// :param kw_shapes: Parameter name -> the keyword argument's static type
pub fn overload_binding<S: AsRef<str>>(
    impls: &[ImplSig],
    chars: &str,
    shapes: &[S],
    kw_chars: &[(String, String)],
    kw_shapes: &[(String, String)],
) -> Option<usize> {
    let wanted = pin_items(chars);
    for (index, imp) in impls.iter().enumerate() {
        match binds(imp, &wanted, shapes, kw_chars, kw_shapes) {
            Verdict::No => continue,
            Verdict::Yes => return Some(index),
            Verdict::Maybe => return None,
        }
    }
    None
}

/// How one implementation answers a call with keyword arguments (`_binds`).
fn binds<S: AsRef<str>>(
    imp: &ImplSig,
    wanted: &[&str],
    shapes: &[S],
    kw_chars: &[(String, String)],
    kw_shapes: &[(String, String)],
) -> Verdict {
    if imp.open_ended {
        return Verdict::Maybe;
    }
    if wanted.len() > imp.params.len() {
        return Verdict::No;
    }
    let mut verdict = Verdict::Yes;
    let mut bound: Vec<usize> = (0..wanted.len()).collect();
    for (index, (declared, want)) in imp.params.iter().zip(wanted).enumerate() {
        let taken = takes_position(declared, want, shape_at(shapes, index));
        if taken == Verdict::No {
            return Verdict::No;
        }
        verdict = verdict.min(taken);
    }
    let fit_count = imp.fits.chars().count();
    for (name, want) in kw_chars {
        let Some(index) = imp.names.iter().position(|known| known == name) else {
            // A keyword-only parameter is not named in the shape: one may take the
            // keyword, none means the bind fails
            return if fit_count > imp.params.len() { Verdict::Maybe } else { Verdict::No };
        };
        if bound.contains(&index) {
            return Verdict::No;
        }
        bound.push(index);
        let given = kw_shapes
            .iter()
            .find(|(known, _)| known == name)
            .map_or(UNKNOWN, |(_, shape)| shape.as_str());
        let declared = imp.params.get(index).map_or(UNKNOWN, String::as_str);
        let taken = takes_position(declared, want, given);
        if taken == Verdict::No {
            return Verdict::No;
        }
        verdict = verdict.min(taken);
    }
    for (index, fit) in imp.fits.chars().enumerate() {
        if bound.contains(&index) {
            continue;
        }
        if fit == FIT_REQUIRED {
            return Verdict::No;
        }
        if fit == FIT_UNSURE {
            verdict = Verdict::Maybe;
        }
    }
    verdict
}

// --- reading a definition -------------------------------------------------

/// Name `DynamicDefaultTransformer` leaves in place of a default that reads per-bar
/// lib state.
pub const DYN_DEFAULT: &str = "__dyn_default__";

/// The parameter lists of a definition's `arguments` node.
struct Params<'a> {
    positional: Vec<NodeId>,
    vararg: bool,
    kwonly: &'a [NodeId],
    kw_defaults: &'a [Option<NodeId>],
    defaults: &'a [NodeId],
}

fn params(ast: &Ast, func: NodeId) -> Result<Params<'_>, Unsupported> {
    let args = match ast.kind(func) {
        Kind::FunctionDef { args, .. } | Kind::AsyncFunctionDef { args, .. } => *args,
        _ => return Err(Unsupported(format!("a definition expected, not {}", ast.name(func)))),
    };
    let Kind::Arguments { posonlyargs, args, vararg, kwonlyargs, kw_defaults, defaults, .. } =
        ast.kind(args)
    else {
        return Err(Unsupported("a definition without arguments".into()));
    };
    Ok(Params {
        positional: posonlyargs.iter().chain(args).copied().collect(),
        vararg: vararg.is_some(),
        kwonly: kwonlyargs,
        kw_defaults,
        defaults,
    })
}

fn arg_parts(ast: &Ast, arg: NodeId) -> (&str, Option<NodeId>) {
    match ast.kind(arg) {
        Kind::Arg { arg, annotation, .. } => (arg, *annotation),
        _ => ("", None),
    }
}

/// One definition's shape, as the static overload selection reads it (`impl_sig`).
///
/// :param ret: What it returns, the inference's answer
/// :param ty_of: Types a default expression (`get_ty` reads the stamps)
pub fn impl_sig(
    ast: &Ast,
    func: NodeId,
    ret: &str,
    ty_of: &dyn Fn(NodeId) -> String,
    classes: &Classes,
) -> Result<ImplSig, Unsupported> {
    let found = params(ast, func)?;
    let defaults = param_defaults(ast, func)?;
    let default_of =
        |name: &str| defaults.iter().find(|(known, _)| known == name).map(|&(_, default)| default);
    let mut types = Vec::with_capacity(found.positional.len());
    for &arg in &found.positional {
        types.push(annotation_type(ast, arg_parts(ast, arg).1, classes)?);
    }
    let mut fits = String::new();
    for &arg in found.positional.iter().chain(found.kwonly) {
        let default = default_of(arg_parts(ast, arg).0);
        fits.push(param_fit(ast, arg, default, ty_of, classes)?);
    }
    Ok(ImplSig {
        params: types,
        required: found.positional.len().saturating_sub(found.defaults.len()),
        open_ended: found.vararg,
        ret: ret.to_string(),
        fits,
        names: found.positional.iter().map(|&arg| arg_parts(ast, arg).0.to_string()).collect(),
    })
}

/// What one parameter contributes to a call that omits it (`param_fit`).
pub fn param_fit(
    ast: &Ast,
    arg: NodeId,
    default: Option<NodeId>,
    ty_of: &dyn Fn(NodeId) -> String,
    classes: &Classes,
) -> Result<char, Unsupported> {
    let Some(default) = default else {
        return Ok(FIT_REQUIRED);
    };
    let Some(annotation) = arg_parts(ast, arg).1 else {
        return Ok(FIT_OMISSIBLE);
    };
    let declared = annotation_type(ast, Some(annotation), classes)?;
    let default = default_ty(ast, default, ty_of);
    let takes_none = annotation_takes_none(ast, Some(annotation))?;
    Ok(default_fit(&declared, Some(&default), takes_none))
}

/// Type of a default expression, as the overload selector reads it (`default_ty`).
pub fn default_ty(ast: &Ast, default: NodeId, ty_of: &dyn Fn(NodeId) -> String) -> String {
    if matches!(ast.kind(default), Kind::Constant { value: Constant::None, .. }) {
        return NONE_DEFAULT.to_string();
    }
    if let Some(spelled) = dotted(ast, default) {
        let last = spelled.rsplit('.').next().unwrap_or("");
        if last == "na" || last == DYN_DEFAULT {
            return TYPELESS.to_string();
        }
    }
    ty_of(default)
}

/// Each parameter's default expression, by parameter name, in Python's dict order
/// (`_param_defaults`).
pub fn param_defaults(ast: &Ast, func: NodeId) -> Result<Vec<(String, NodeId)>, Unsupported> {
    let found = params(ast, func)?;
    let mut out: Vec<(String, NodeId)> = Vec::new();
    let mut put =
        |name: &str, default: NodeId| match out.iter_mut().find(|(known, _)| known == name) {
            Some(slot) => slot.1 = default,
            None => out.push((name.to_string(), default)),
        };
    // The defaults align to the LAST parameters (``positional[len - len(defaults):]``)
    let count = found.positional.len() as isize;
    let mut start = count - found.defaults.len() as isize;
    if start < 0 {
        start = (count + start).max(0);
    }
    for (&arg, &default) in found.positional[start as usize..].iter().zip(found.defaults) {
        put(arg_parts(ast, arg).0, default);
    }
    for (&kwarg, kw_default) in found.kwonly.iter().zip(found.kw_defaults) {
        if let Some(default) = kw_default {
            put(arg_parts(ast, kwarg).0, *default);
        }
    }
    Ok(out)
}

/// Whether a value spells Pine's typeless `na`, a signed one included.
pub fn is_na_literal(ast: &Ast, node: NodeId) -> bool {
    let mut node = node;
    while let &Kind::UnaryOp { op: UnaryOperator::USub, operand } = ast.kind(node) {
        node = operand;
    }
    matches!(ast.kind(node), Kind::Name { .. } | Kind::Attribute { .. })
        && dotted(ast, node).is_some_and(|spelled| NA_LITERALS.contains(&spelled.as_str()))
}

// --- dataclass field factories --------------------------------------------

/// What a name denotes at one statement of the module.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Binding {
    /// `from dataclasses import field`
    Field,
    /// `import dataclasses`
    Dataclasses,
    Other,
}

/// The `field(default_factory=...)` defaults of a module's UDT classes
/// (`FactoryFields`); the module body is read as it was when this was built.
pub struct FactoryFields {
    body: Vec<NodeId>,
    /// Every class's top-level statement index
    top_index: HashMap<NodeId, usize>,
}

impl FactoryFields {
    pub fn new(ast: &Ast, module: NodeId) -> FactoryFields {
        let body = match ast.kind(module) {
            Kind::Module { body, .. } | Kind::Interactive { body } => body.clone(),
            _ => Vec::new(),
        };
        let mut top_index = HashMap::new();
        for (index, &stmt) in body.iter().enumerate() {
            for node in walk_statements(ast, stmt) {
                if matches!(ast.kind(node), Kind::ClassDef { .. }) {
                    top_index.insert(node, index);
                }
            }
        }
        FactoryFields { body, top_index }
    }

    /// What `name` denotes at statement `before`.
    fn binding(&self, ast: &Ast, name: &str, before: usize) -> Option<Binding> {
        let mut binding = None;
        for &stmt in &self.body[..before.min(self.body.len())] {
            match ast.kind(stmt) {
                Kind::ImportFrom { module, names, level } => {
                    for &alias in names {
                        let Kind::Alias { name: imported, asname } = ast.kind(alias) else {
                            continue;
                        };
                        let bound = asname.as_deref().filter(|bound| !bound.is_empty());
                        if bound.unwrap_or(imported) == name {
                            let field = module.as_deref() == Some("dataclasses")
                                && level.unwrap_or(0) == 0
                                && imported == "field";
                            binding = Some(if field { Binding::Field } else { Binding::Other });
                        }
                    }
                }
                Kind::Import { names } => {
                    for &alias in names {
                        let Kind::Alias { name: imported, asname } = ast.kind(alias) else {
                            continue;
                        };
                        let bound = asname.as_deref().filter(|bound| !bound.is_empty());
                        let top = imported.split('.').next().unwrap_or("");
                        if bound.unwrap_or(top) == name {
                            binding = Some(if imported == "dataclasses" {
                                Binding::Dataclasses
                            } else {
                                Binding::Other
                            });
                        }
                    }
                }
                Kind::FunctionDef { name: defined, .. }
                | Kind::AsyncFunctionDef { name: defined, .. }
                | Kind::ClassDef { name: defined, .. } => {
                    if defined == name {
                        binding = Some(Binding::Other);
                    }
                }
                _ => {
                    for node in walk(ast, stmt) {
                        if let Kind::Name { id, ctx: ExprContext::Store } = ast.kind(node)
                            && id == name
                        {
                            binding = Some(Binding::Other);
                        }
                    }
                }
            }
        }
        binding
    }

    /// The factory-default calls of a class, empty for a class that is no UDT.
    pub fn of(&self, ast: &Ast, cls: NodeId) -> Vec<NodeId> {
        let Kind::ClassDef { body, decorator_list, .. } = ast.kind(cls) else {
            return Vec::new();
        };
        let udt = decorator_list.iter().any(|&decorator| {
            let spelled = dotted(ast, decorator).unwrap_or_default();
            let last = spelled.rsplit('.').next().unwrap_or("");
            last == "udt" || last == "dataclass"
        });
        if !udt {
            return Vec::new();
        }
        let before = self.top_index.get(&cls).copied().unwrap_or(self.body.len());
        let mut out = Vec::new();
        for &stmt in body {
            let Kind::AnnAssign { value: Some(value), .. } = ast.kind(stmt) else {
                continue;
            };
            let Kind::Call { func, args, keywords } = ast.kind(*value) else {
                continue;
            };
            if keywords.len() != 1 || !args.is_empty() {
                continue;
            }
            let Kind::Keyword { arg, value: factory } = ast.kind(keywords[0]) else {
                continue;
            };
            if arg.as_deref() != Some("default_factory") || !is_bare_lambda(ast, *factory) {
                continue;
            }
            let plumbing = match ast.kind(*func) {
                Kind::Name { id, .. } => self.binding(ast, id, before) == Some(Binding::Field),
                Kind::Attribute { value: owner, attr, .. } if attr == "field" => {
                    match ast.kind(*owner) {
                        Kind::Name { id, .. } => {
                            self.binding(ast, id, before) == Some(Binding::Dataclasses)
                        }
                        _ => false,
                    }
                }
                _ => false,
            };
            if plumbing {
                out.push(*value);
            }
        }
        out
    }
}

/// Whether a node is a lambda taking no parameter at all.
fn is_bare_lambda(ast: &Ast, node: NodeId) -> bool {
    let Kind::Lambda { args, .. } = ast.kind(node) else {
        return false;
    };
    matches!(
        ast.kind(*args),
        Kind::Arguments { posonlyargs, args, vararg: None, kwonlyargs, kwarg: None, .. }
            if posonlyargs.is_empty() && args.is_empty() && kwonlyargs.is_empty()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tuples_count_characters() {
        let class = object_ty(&class_id("/Users/Ádám/x:y, z.py", "Pivot"));
        let tuple = tuple_of(&[class.as_str(), INT, "a:f"]);
        assert_eq!(elements_of(&tuple), vec![class.as_str(), INT, "a:f"]);
        let widened = tuple_of(&[class.as_str(), FLOAT, "a:f"]);
        assert_eq!(join(&tuple, &widened), widened);
        assert!(elements_of("T:3:ab").is_empty());
        assert!(elements_of("T:x:i").is_empty());
    }

    #[test]
    fn arithmetic_follows_the_measured_algebra() {
        assert_eq!(binop_type(Operator::Div, INT, INT), INT);
        assert_eq!(binop_type(Operator::Add, TYPELESS, TYPELESS), INT);
        assert_eq!(binop_type(Operator::Add, STR, TYPELESS), STR);
        assert_eq!(binop_type(Operator::BitOr, INT, INT), UNKNOWN);
        assert_eq!(unaryop_type(UnaryOperator::USub, TYPELESS), TYPELESS);
    }

    #[test]
    fn pin_line_is_the_lib_loop_class() {
        assert_eq!(PINE_LOOP, object_ty(&builtin_class_id("PineLoop")));
    }
}
