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

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use indexmap::IndexMap;

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
/// A typeless `na` where a bool goes: the bool na of the mode in effect
pub const WIDEN_BOOL_NA: &str = "b";
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
    [WIDEN_STRING_NA, WIDEN_NUMBER_NA, WIDEN_BOOL_NA, WIDEN_NA_TEXT].contains(&kind)
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

/// The type of a map from one key type to one value type (`map_of`): a scalar key is
/// one character, an enum key a length-prefixed item; `OBJECT` when either half is not
/// known or the key is neither a scalar nor an object of one of `enums`.
pub fn map_of(key: &str, value: &str, enums: &Enums) -> String {
    if value == UNKNOWN || value == VOID {
        return OBJECT.to_string();
    }
    if is_scalar(key) {
        return format!("{MAP}{key}:{value}");
    }
    if class_of(key).is_some_and(|cid| enums.contains(cid)) {
        return format!("{MAP}{}:{key}:{value}", key.chars().count());
    }
    OBJECT.to_string()
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

/// The enum key of a map type and the byte index its value type starts at
/// (`_item_key`); `None` when the key is no length-prefixed item. A length is ASCII
/// digits, the only ones `map_of` writes.
fn item_key(ty: &str) -> Option<(&str, usize)> {
    let rest = &ty[MAP.len()..];
    let colon = rest.find(':')?;
    let digits = &rest[..colon];
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    // A length beyond `usize` is beyond the string as well
    let size = digits.parse::<usize>().ok()?;
    let after = &rest[colon + 1..];
    let end = if size == 0 {
        0
    } else {
        let (index, last) = after.char_indices().nth(size - 1)?;
        index + last.len_utf8()
    };
    if !after[end..].starts_with(':') {
        return None;
    }
    Some((&after[..end], MAP.len() + colon + 1 + end + 1))
}

/// The key type of a map, `UNKNOWN` when the type is not a map (or has no key, where
/// Python raises).
pub fn key_of(ty: &str) -> &str {
    if !ty.starts_with(MAP) {
        return UNKNOWN;
    }
    if let Some((key, _)) = item_key(ty) {
        return key;
    }
    let rest = &ty[MAP.len()..];
    match rest.chars().next() {
        Some(key) => &rest[..key.len_utf8()],
        None => UNKNOWN,
    }
}

/// The value type of a map, `UNKNOWN` when the type is not a map.
pub fn value_of(ty: &str) -> &str {
    if !ty.starts_with(MAP) {
        return UNKNOWN;
    }
    match item_key(ty) {
        Some((_, start)) => &ty[start..],
        None => skip_chars(ty, MAP.len() + 2),
    }
}

/// The scalar a type is where it meets another type (`constant_scalar`): the scalar
/// of a lib constant class that is no set of flags, any other type unchanged.
pub fn constant_scalar(ty: &str) -> &str {
    static SCALARS: OnceLock<HashMap<String, String>> = OnceLock::new();
    if !ty.starts_with("o:") {
        return ty;
    }
    let scalars = SCALARS.get_or_init(|| {
        let flags = crate::data::bit_sets();
        crate::data::scalar_classes()
            .iter()
            .filter(|(name, _)| !flags.contains(name))
            .map(|(name, scalar)| (object_ty(&builtin_class_id(name)), scalar.clone()))
            .collect()
    });
    scalars.get(ty).map_or(ty, String::as_str)
}

/// Whether a type is a lib class that derives from `StrLiteral` (`is_str_literal`): the
/// selector takes any string for a parameter of one, in both of its passes.
pub fn is_str_literal(ty: &str) -> bool {
    static LITERALS: OnceLock<HashSet<String>> = OnceLock::new();
    if !ty.starts_with("o:") {
        return false;
    }
    LITERALS
        .get_or_init(|| {
            crate::data::str_literals()
                .iter()
                .map(|name| object_ty(&builtin_class_id(name)))
                .collect()
        })
        .contains(ty)
}

/// Whether a ternary of two container types has its first branch's type
/// (`ternary_takes_first`): a float array, matrix or map value before the same
/// container of ints.
pub fn ternary_takes_first(body: &str, orelse: &str) -> bool {
    if is_array(body) && is_array(orelse) || is_matrix(body) && is_matrix(orelse) {
        return element_of(body) == FLOAT && element_of(orelse) == INT;
    }
    if is_map(body) && is_map(orelse) {
        return key_of(body) == key_of(orelse)
            && value_of(body) == FLOAT
            && value_of(orelse) == INT;
    }
    false
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
    if head(left) != OBJECT || head(right) != OBJECT {
        return false;
    }
    // Two lib constants are scalars to each other, see `constant_scalar`
    constant_scalar(left) == left || constant_scalar(right) == right
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
    let (scalar, other) = (constant_scalar(left), constant_scalar(right));
    if scalar != left || other != right {
        // A lib constant meeting a scalar, or a constant of another class
        return join(scalar, other);
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
    ("request.seed", Is("arg2")),
    ("ta.vwap", Is("vwap")),
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
    // An enum input holds a member of its default's enum
    ("input.enum", Is("arg0")),
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
pub const OVERRIDE_PARAM_NAMES: &[(&str, &[&str])] = &[
    ("input", &["defval", "title", "tooltip", "inline", "group", "display", "active"]),
    (
        "input.enum",
        &[
            "defval", "title", "options", "tooltip", "inline", "group", "confirm", "display",
            "active",
        ],
    ),
    ("ta.vwap", &["source", "anchor", "stdev_mult"]),
];

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

/// Annotation heads whose payload is the type itself, the qualifier markers
/// (`QUALIFIER_MARKERS`) included.
const TRANSPARENT_SUBSCRIPTS: &[&str] = &[
    "Series",
    "PersistentSeries",
    "NA",
    "Persistent",
    "IBPersistent",
    "IBPersistentSeries",
    "Optional",
    "Simple",
    "Const",
];
/// A parameter's qualifier marker, one `ImplSig::quals` character per positional
/// parameter: the `pine_qualifier` lattice's values, lowest first.
pub const QUAL_CONST: char = '0';
/// The `ImplSig::quals` character of a `Simple[T]` parameter.
pub const QUAL_SIMPLE: char = '1';
/// The `ImplSig::quals` character of a parameter without a marker.
pub const QUAL_PLAIN: char = '2';
/// Qualifier marker annotation head -> its `ImplSig::quals` character.
const QUALIFIER_MARKERS: &[(&str, char)] = &[("Const", QUAL_CONST), ("Simple", QUAL_SIMPLE)];

/// The qualifier a parameter annotation's marker declares (`annotation_qualifier`):
/// `QUAL_SIMPLE` for `Simple[T]`, `QUAL_CONST` for `Const[T]`, `None` for no marker. A
/// parameter whose history is read carries the marker under `Series`: `Series[Simple[T]]`.
pub fn annotation_qualifier(ast: &Ast, node: Option<NodeId>) -> Option<char> {
    let Kind::Subscript { value, slice, .. } = ast.kind(node?) else {
        return None;
    };
    let name = leaf_name(ast, *value);
    if name == "Series" {
        return annotation_qualifier(ast, Some(*slice));
    }
    QUALIFIER_MARKERS.iter().find(|(head, _)| *head == name).map(|&(_, qual)| qual)
}
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

/// The class ids of the enums among a module's `Classes`, the only classes a map key
/// may be.
pub type Enums = HashSet<String>;

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
/// arena; `None` where CPython raises `SyntaxError`.
///
/// Only a spelling whose parse provably agrees with CPython's is read: printable ASCII
/// on one line, not indented, without the punctuation of statements, slices, stars,
/// comments or line continuations, and without keywords. A blank spelling, and one
/// indented before its first token, is a `SyntaxError` on every interpreter.
/// Everything else, and a spelling the parser refuses, answers `Unsupported`, so the
/// Python step decides.
pub(crate) fn forward_reference(text: &str) -> Result<Option<(Ast, NodeId)>, Unsupported> {
    let refuse = || Unsupported(format!("forward reference {text:?}"));
    let token = text.trim_start_matches([' ', '\t']);
    if text.chars().all(|c| matches!(c, ' ' | '\t' | '\n' | '\r' | '\x0c'))
        || token.len() < text.len()
            && token.starts_with(|c: char| c.is_ascii_graphic() && c != '\\' && c != '#')
    {
        return Ok(None);
    }
    if !text.bytes().all(|byte| (0x20..0x7f).contains(&byte) && !b"#\\:;=*@{}`$?!".contains(&byte))
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
    Ok(Some((ast, body)))
}

/// Map a Python annotation onto a Pine type (`annotation_type`).
///
/// :param node: The annotation expression, or `None` when there is none
/// :param classes: Class spelling -> class id, visible where the annotation stands
/// :param enums: The class ids of the enums among them
pub fn annotation_type(
    ast: &Ast,
    node: Option<NodeId>,
    classes: &Classes,
    enums: &Enums,
) -> Result<String, Unsupported> {
    let Some(node) = node else {
        return Ok(UNKNOWN.to_string());
    };
    match ast.kind(node) {
        Kind::Constant { value, .. } => match value {
            Constant::None => Ok(VOID.to_string()),
            // A stringized forward reference: ``'_ExitOrderKey'``, ``'list[Pivot]'``
            Constant::Str(text) => match forward_reference(text)? {
                Some((parsed, body)) => annotation_type(&parsed, Some(body), classes, enums),
                None => Ok(UNKNOWN.to_string()),
            },
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
        Kind::BinOp { op: Operator::BitOr, .. } => union_type(ast, node, classes, enums),
        &Kind::Subscript { value, slice, .. } => subscript_type(ast, value, slice, classes, enums),
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
    enums: &Enums,
) -> Result<String, Unsupported> {
    let name = leaf_name(ast, value);
    if TRANSPARENT_SUBSCRIPTS.contains(&name) {
        return annotation_type(ast, Some(slice), classes, enums);
    }
    if ARRAY_SUBSCRIPTS.contains(&name) {
        return Ok(array_of(&annotation_type(ast, Some(slice), classes, enums)?));
    }
    if MATRIX_SUBSCRIPTS.contains(&name) {
        return Ok(matrix_of(&annotation_type(ast, Some(slice), classes, enums)?));
    }
    if MAP_SUBSCRIPTS.contains(&name) {
        let pair = tuple_elts(ast, slice).unwrap_or(&[]);
        if pair.len() != 2 {
            return Ok(OBJECT.to_string());
        }
        let key = annotation_type(ast, Some(pair[0]), classes, enums)?;
        let value = annotation_type(ast, Some(pair[1]), classes, enums)?;
        return Ok(map_of(&key, &value, enums));
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
            .map(|&item| annotation_type(ast, Some(item), classes, enums))
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
    // A class the module declares or imports wins over the builtin of the same
    // name, and its own `Map` is no container
    let found = classes.get(name);
    if let Some(found) = found
        && !found.starts_with(&format!("{LIB_MODULE}{CLASS_SEP}"))
    {
        return object_ty(found);
    }
    // A container spelled without its payload stays an object of unknown elements,
    // even where the module has the container class itself in scope
    if CONTAINER_NAMES.contains(&name) {
        return OBJECT.to_string();
    }
    if let Some(found) = found {
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
fn union_type(
    ast: &Ast,
    node: NodeId,
    classes: &Classes,
    enums: &Enums,
) -> Result<String, Unsupported> {
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
    let mut result = annotation_type(ast, Some(first), classes, enums)?;
    for member in contributing {
        result = join(&result, &annotation_type(ast, Some(member), classes, enums)?);
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
            Constant::Str(text) => match forward_reference(text)? {
                Some((parsed, body)) => annotation_takes_none(&parsed, Some(body)),
                None => Ok(false),
            },
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

/// Separator of a pin's implementation index (`PIN_INDEX`): `<characters>=<index>` names
/// the implementation by its position in the group's registration order.
pub const PIN_INDEX: char = '=';

/// Every pin character the runtime has a witness value for (`PIN_WITNESSED`).
pub const PIN_WITNESSED: &str = "ifbsIFBS1234";

/// Separator of the two passes of an acceptance spec (`ACCEPTS_SEP`).
pub const ACCEPTS_SEP: char = '/';

/// The acceptance spec of a parameter whose check of a witness is not known
/// (`ACCEPTS_UNKNOWN`).
pub const ACCEPTS_UNKNOWN: &str = "?";

/// The pin that names one implementation of a group by its registration index
/// (`indexed_pin`).
pub fn indexed_pin<S: AsRef<str>>(arg_types: &[S], slot: i64) -> String {
    format!("{}{PIN_INDEX}{slot}", pin_chars(arg_types))
}

/// The implementation index a pin names, where it names one (`pin_index`).
pub fn pin_index(pin: Option<&str>) -> Option<i64> {
    let (_, index) = pin?.split_once(PIN_INDEX)?;
    // Python's ``str.isdigit`` then ``int``: ASCII digits only here, which is all the
    // type pass ever writes
    if index.is_empty() || !index.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    index.parse().ok()
}

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

/// A call site's argument types as the selector reads the values of an unpinned call
/// (`value_chars`): a float and a scalar container read as `PIN_ANY`, an int, a bool and
/// a string keep their characters.
pub fn value_chars<S: AsRef<str>>(arg_types: &[S]) -> String {
    pin_chars(arg_types)
        .chars()
        .map(|char| {
            let text = char.to_string();
            if text == FLOAT || PIN_CONTAINERS.iter().any(|&(_, pin)| pin == char) {
                PIN_ANY
            } else {
                char
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

/// What a parameter contributes to a call that leaves it to its default, in the
/// widening pass (`_default_fit_wide`): an int default for a float and a float default
/// for an int are taken there too.
fn default_fit_wide(declared: &str, default: Option<&str>, takes_none: bool) -> char {
    let fit = default_fit(declared, default, takes_none);
    if fit == FIT_REQUIRED
        && default.is_some_and(|default| is_numeric(head(default)))
        && is_numeric(head(declared))
    {
        return FIT_OMISSIBLE;
    }
    fit
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
    /// One character per positional parameter, `'1'` where it has no annotation and
    /// the selector takes every value; empty where the shape does not say
    pub bare: String,
    /// Position of the implementation in its group's registration order, the ones the
    /// static selection leaves out included; -1 where it is not known
    pub slot: i64,
    /// How the selector checks a witness at each positional parameter where its type
    /// does not say it: `""` reads the type, `ACCEPTS_UNKNOWN` knows nothing, otherwise
    /// `<exact>/<widening>`; empty where no position needs one
    pub accepts: Vec<String>,
    /// The `fits` of the widening pass; empty where not known (`_wide_fits`)
    pub wide_fits: String,
    /// Whether a `**kwargs` parameter is among the ones the selector counts
    pub kwarg: bool,
    /// How many implementations the group registers; -1 where it is not known
    pub size: i64,
    /// One `QUAL_*` character per positional parameter, the qualifier its marker
    /// declares (`annotation_qualifier`), `QUAL_PLAIN` where it has none; empty where
    /// no parameter has one (`qualifier_choice`)
    pub quals: String,
}

/// Whether an implementation's positional parameter has no annotation (`_is_bare`).
fn is_bare(imp: &ImplSig, index: usize) -> bool {
    imp.bare.as_bytes().get(index) == Some(&b'1')
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
    pass_choice(impls, &pin_items(pin), shapes, true).1
}

/// How one pass of the selector answers a call, positional half first (`_pass_choice`).
fn pass_choice<S: AsRef<str>>(
    impls: &[ImplSig],
    wanted: &[&str],
    shapes: &[S],
    strict: bool,
) -> (Verdict, Option<usize>) {
    for exact_arity in [true, false] {
        for (index, imp) in impls.iter().enumerate() {
            match takes(imp, wanted, exact_arity, shapes, strict, false) {
                Verdict::No => continue,
                Verdict::Yes => return (Verdict::Yes, Some(index)),
                Verdict::Maybe => return (Verdict::Maybe, None),
            }
        }
    }
    (Verdict::No, None)
}

/// The implementation the runtime runs for a pin, decided statically, by its index in
/// `impls` (`resolve_pin`): the witnesses' exact pass then widening pass, or the
/// wildcard rule; `None` where unsure, or where `impls` are not the whole group.
pub fn resolve_pin(impls: &[ImplSig], pin: &str) -> Option<usize> {
    let chars = pin.split_once(PIN_INDEX).map_or(pin, |(chars, _)| chars);
    let size = impls.len();
    if size == 0
        || impls
            .iter()
            .enumerate()
            .any(|(slot, imp)| imp.size != size as i64 || imp.slot != slot as i64)
        || chars.chars().any(|char| !PIN_WITNESSED.contains(char) && char != PIN_ANY)
    {
        return None;
    }
    let wanted = pin_items(chars);
    let no_shapes: &[&str] = &[];
    if chars.contains(PIN_ANY) {
        let verdicts_for = |strict: bool| -> Vec<Verdict> {
            impls.iter().map(|imp| takes(imp, &wanted, false, no_shapes, strict, true)).collect()
        };
        let exact = verdicts_for(true);
        let widening = verdicts_for(false);
        if exact.contains(&Verdict::Maybe) || widening.contains(&Verdict::Maybe) {
            return None;
        }
        for (strict, verdicts) in [(true, &exact), (false, &widening)] {
            let found: Vec<usize> = verdicts
                .iter()
                .enumerate()
                .filter(|(_, verdict)| **verdict == Verdict::Yes)
                .map(|(index, _)| index)
                .collect();
            if found.len() > 1 {
                return None;
            }
            let Some(&index) = found.first() else {
                continue;
            };
            if strict {
                // The values may fail the chosen one at a wildcard position and
                // fall through to the widening pass: only an implementation that
                // declares the same types there cannot be the one they then take
                let chosen = &impls[index].params;
                let wild: Vec<usize> = chars
                    .chars()
                    .enumerate()
                    .filter(|(_, char)| *char == PIN_ANY)
                    .map(|(k, _)| k)
                    .collect();
                let differs = widening.iter().enumerate().any(|(other, verdict)| {
                    let params = &impls[other].params;
                    *verdict == Verdict::Yes
                        && other != index
                        && wild.iter().any(|&k| {
                            k >= params.len() || k >= chosen.len() || params[k] != chosen[k]
                        })
                });
                if differs {
                    return None;
                }
            }
            return Some(index);
        }
        return None;
    }
    for strict in [true, false] {
        let (verdict, chosen) = pass_choice(impls, &wanted, no_shapes, strict);
        if verdict != Verdict::No {
            return chosen;
        }
    }
    None
}

/// The implementations the runtime selector can run for a call site, by index
/// (`overload_candidates`): the exact pass walked like `overload_choice`, each
/// implementation that may take the call collected, the first that surely takes it
/// closing the set; `None` where the widening pass may run one left out.
pub fn overload_candidates<S: AsRef<str>>(
    impls: &[ImplSig],
    pin: &str,
    shapes: &[S],
) -> Option<Vec<usize>> {
    let wanted = pin_items(pin);
    let mut found: Vec<usize> = Vec::new();
    for exact_arity in [true, false] {
        for (index, imp) in impls.iter().enumerate() {
            let verdict = takes(imp, &wanted, exact_arity, shapes, true, false);
            if verdict == Verdict::No {
                continue;
            }
            if !found.contains(&index) {
                found.push(index);
            }
            if verdict == Verdict::Yes {
                return Some(found);
            }
        }
    }
    let settled = impls
        .iter()
        .enumerate()
        .all(|(index, imp)| found.contains(&index) || out_of_reach(imp, &wanted, shapes));
    if found.is_empty() || !settled { None } else { Some(found) }
}

/// Whether the selector's widening pass rejects an implementation for every value
/// (`_out_of_reach`): the call's arity rules it out, or a parameter it fills takes no
/// value of the argument's type in that pass either.
fn out_of_reach<S: AsRef<str>>(imp: &ImplSig, wanted: &[&str], shapes: &[S]) -> bool {
    if imp.open_ended || imp.kwarg {
        return false;
    }
    let argc = wanted.len();
    if !(imp.required <= argc && argc <= imp.params.len()) && argc != imp.fits.chars().count() {
        return true;
    }
    imp.params
        .iter()
        .zip(wanted)
        .enumerate()
        .any(|(index, (declared, want))| widening_rejects(declared, given_at(want, shapes, index)))
}

/// The static type of one argument: its shape where the caller has it, else the type
/// its character stands for (`_given_at`).
fn given_at<'a, S: AsRef<str>>(want: &'a str, shapes: &'a [S], index: usize) -> &'a str {
    let given = shape_at(shapes, index);
    if given != UNKNOWN {
        return given;
    }
    if PINNABLE.contains(&want) {
        return want;
    }
    pin_container_type(want).unwrap_or(UNKNOWN)
}

/// Whether the widening pass rejects every value of a type for a parameter
/// (`_widening_rejects`): values of different kinds never meet there, but an `NA`
/// object passes for an int, float, string or bool parameter, and a color may be the
/// typeless na.
fn widening_rejects(declared: &str, given: &str) -> bool {
    let (Some(kind), Some(declared_kind)) = (kind_of(given), kind_of(declared)) else {
        return false;
    };
    if kind == declared_kind || given == COLOR {
        return false;
    }
    !([INT, FLOAT, BOOL, STR].contains(&declared) && ![INT, FLOAT, STR].contains(&given))
}

/// The kind of value the selector checks a value of a type as (`_kind`); `None` where
/// the type does not say one.
fn kind_of(ty: &str) -> Option<&str> {
    if NUMERIC.contains(&ty) {
        return Some(FLOAT);
    }
    if [BOOL, STR, COLOR].contains(&ty) {
        return Some(ty);
    }
    if is_array(ty) {
        return Some(ARRAY);
    }
    if is_matrix(ty) {
        return Some(MATRIX);
    }
    if is_map(ty) {
        return Some(MAP);
    }
    if class_of(ty).is_some() && constant_scalar(ty) == ty {
        return Some(ty);
    }
    None
}

/// Whether every value of a type passes the exact pass's check against the type itself
/// (`_strictly_held`).
fn strictly_held(ty: &str) -> bool {
    if [BOOL, STR, COLOR].contains(&ty) {
        return true;
    }
    if is_array(ty) || is_matrix(ty) {
        return strictly_held(element_of(ty));
    }
    if is_map(ty) {
        return ty.chars().count() > MAP.len()
            && strictly_held(key_of(ty))
            && strictly_held(value_of(ty));
    }
    class_of(ty).is_some() && constant_scalar(ty) == ty
}

/// The implementation Pine itself selects for a positional call, by its index in
/// `impls` (`pine_choice`): an exact match of every declared type first, then one taking
/// an int for a float; decided only where every argument type is settled, every
/// implementation the arity reaches declares the positions the call fills, and exactly
/// one implementation answers the first pass any answers.
pub fn pine_choice<S: AsRef<str>>(impls: &[ImplSig], shapes: &[S]) -> Option<usize> {
    match pine_matches(impls, shapes) {
        Some(found) if found.len() == 1 => Some(found[0]),
        _ => None,
    }
}

/// The implementations that answer the first of Pine's two passes any answers, by
/// their index in `impls` (`_pine_matches`): the candidates [`pine_choice`] decides
/// among; empty where no pass answers, `None` where the types are not ones the passes
/// read.
fn pine_matches<S: AsRef<str>>(impls: &[ImplSig], shapes: &[S]) -> Option<Vec<usize>> {
    let argc = shapes.len();
    if !shapes.iter().all(|given| pine_settled(given.as_ref())) {
        return None;
    }
    let mut reachable: Vec<usize> = Vec::new();
    for (index, imp) in impls.iter().enumerate() {
        if imp.open_ended {
            return None;
        }
        if !(imp.required <= argc && argc <= imp.params.len()) {
            continue;
        }
        if (0..argc).any(|position| is_bare(imp, position) || !pine_settled(&imp.params[position]))
        {
            return None;
        }
        reachable.push(index);
    }
    for widening in [false, true] {
        let found: Vec<usize> = reachable
            .iter()
            .copied()
            .filter(|&index| {
                shapes.iter().enumerate().all(|(position, given)| {
                    pine_takes(&impls[index].params[position], given.as_ref(), widening)
                })
            })
            .collect();
        if !found.is_empty() {
            return Some(found);
        }
    }
    Some(Vec::new())
}

/// The implementation Pine selects among ones that differ only by a parameter's
/// qualifier, by its index in `impls` (`qualifier_choice`): of the candidates the
/// types leave ([`pine_matches`]), the ones whose markers admit every argument's
/// qualifier, and of those the one whose every parameter is at least as narrow as
/// every other's. A call with keyword arguments is read the same way, each keyword at
/// the parameter of its name ([`pine_keyword_matches`]); one bound to a keyword-only
/// parameter carries no marker. Undecided where no implementation carries a marker and
/// where no one implementation is the narrowest.
///
/// :param quals: The `pine_qualifier` qualifier of each positional argument
/// :param kw_shapes: Parameter name -> the keyword argument's static type, each keyword
///                   spelled as the parameter it binds
/// :param kw_quals: Parameter name -> the keyword argument's qualifier, spelled as
///                  `kw_shapes`
pub fn qualifier_choice<S: AsRef<str>>(
    impls: &[ImplSig],
    shapes: &[S],
    quals: &[u8],
    kw_shapes: &[(String, String)],
    kw_quals: &[(String, u8)],
) -> Option<usize> {
    if quals.len() != shapes.len() || impls.iter().all(|imp| imp.quals.is_empty()) {
        return None;
    }
    let shaped: HashSet<&str> = kw_shapes.iter().map(|(name, _)| name.as_str()).collect();
    let qualified: HashSet<&str> = kw_quals.iter().map(|(name, _)| name.as_str()).collect();
    if shaped != qualified {
        return None;
    }
    let found = if kw_shapes.is_empty() {
        pine_matches(impls, shapes)?
    } else {
        pine_keyword_matches(impls, shapes, kw_shapes)?
    };
    let bound = |index: usize| -> IndexMap<usize, u8> {
        let names = &impls[index].names;
        let mut out: IndexMap<usize, u8> = quals.iter().copied().enumerate().collect();
        for (name, qual) in kw_quals {
            if let Some(position) = names.iter().position(|known| known == name) {
                out.insert(position, *qual);
            }
        }
        out
    };
    let rank = |index: usize, position: usize| -> u8 {
        impls[index].quals.as_bytes().get(position).copied().unwrap_or(QUAL_PLAIN as u8) - b'0'
    };
    let admitted: Vec<usize> = found
        .into_iter()
        .filter(|&index| {
            bound(index).iter().all(|(&position, &qual)| qual <= rank(index, position))
        })
        .collect();
    let narrowest: Vec<usize> = admitted
        .iter()
        .copied()
        .filter(|&index| {
            let positions = bound(index);
            admitted.iter().all(|&other| {
                positions.keys().all(|&position| rank(index, position) <= rank(other, position))
            })
        })
        .collect();
    if narrowest.len() == 1 { Some(narrowest[0]) } else { None }
}

/// The implementation Pine itself selects for a call with keyword arguments, by its
/// index in `impls` (`pine_keyword_choice`): every implementation that binds the call
/// (`keyword_bound`) is reached, and the two passes of [`pine_choice`] compare each
/// bound argument; a keyword the shape does not name binds a keyword-only parameter
/// untyped. Undecided where a type is not settled, an implementation takes `*args` or
/// `**kwargs`, or several answer the first pass any answers.
///
/// :param kw_shapes: Parameter name -> the keyword argument's static type, each keyword
///     spelled as the parameter it binds
pub fn pine_keyword_choice<S: AsRef<str>>(
    impls: &[ImplSig],
    shapes: &[S],
    kw_shapes: &[(String, String)],
) -> Option<usize> {
    match pine_keyword_matches(impls, shapes, kw_shapes) {
        Some(found) if found.len() == 1 => Some(found[0]),
        _ => None,
    }
}

/// Whether another implementation of the group is indistinguishable from the chosen one
/// (`has_twin`): equal positional types, witness specs, required count and qualifier
/// markers; keyword-only parameters and parameter names take no part in it.
pub fn has_twin(impls: &[ImplSig], chosen: usize) -> bool {
    let key = |imp: &ImplSig| {
        let count = imp.params.len();
        let prefix = |text: &str| text.chars().take(count).collect::<String>();
        (
            imp.params.clone(),
            imp.required,
            imp.open_ended,
            imp.bare.clone(),
            imp.accepts.clone(),
            imp.quals.clone(),
            prefix(&imp.fits),
            prefix(&imp.wide_fits),
        )
    };
    let own = key(&impls[chosen]);
    impls.iter().enumerate().any(|(index, imp)| index != chosen && imp.slot >= 0 && key(imp) == own)
}

/// The implementations that answer the first of Pine's two passes any answers, keywords
/// bound, by their index in `impls` (`_pine_keyword_matches`): the candidates
/// [`pine_keyword_choice`] decides among; empty where no pass answers, `None` where the
/// types are not ones the passes read.
fn pine_keyword_matches<S: AsRef<str>>(
    impls: &[ImplSig],
    shapes: &[S],
    kw_shapes: &[(String, String)],
) -> Option<Vec<usize>> {
    if !shapes.iter().all(|given| pine_settled(given.as_ref()))
        || !kw_shapes.iter().all(|(_, given)| pine_settled(given))
    {
        return None;
    }
    let mut reachable: Vec<(usize, Vec<(String, String)>)> = Vec::new();
    for (index, imp) in impls.iter().enumerate() {
        if imp.open_ended || imp.kwarg {
            return None;
        }
        let Some(bound) = keyword_bound(imp, shapes, kw_shapes) else { continue };
        if bound.iter().any(|(declared, _)| !pine_settled(declared)) {
            return None;
        }
        reachable.push((index, bound));
    }
    for widening in [false, true] {
        let found: Vec<usize> = reachable
            .iter()
            .filter(|(_, bound)| {
                bound.iter().all(|(declared, given)| pine_takes(declared, given, widening))
            })
            .map(|(index, _)| *index)
            .collect();
        if !found.is_empty() {
            return Some(found);
        }
    }
    Some(Vec::new())
}

/// The (declared, given) type pairs a call with keyword arguments binds in one
/// implementation (`_keyword_bound`): UNKNOWN declared for a parameter without an
/// annotation, nothing for a keyword-only one; None where it does not bind the call.
fn keyword_bound<S: AsRef<str>>(
    imp: &ImplSig,
    shapes: &[S],
    kw_shapes: &[(String, String)],
) -> Option<Vec<(String, String)>> {
    if shapes.len() > imp.params.len() {
        return None;
    }
    let declared_at = |position: usize| {
        if is_bare(imp, position) { UNKNOWN.to_string() } else { imp.params[position].clone() }
    };
    let mut filled: Vec<usize> = (0..shapes.len()).collect();
    let mut bound: Vec<(String, String)> = shapes
        .iter()
        .enumerate()
        .map(|(position, given)| (declared_at(position), given.as_ref().to_string()))
        .collect();
    let keyword_only = imp.fits.chars().count() > imp.params.len();
    for (name, given) in kw_shapes {
        let Some(position) = imp.names.iter().position(|known| known == name) else {
            if !keyword_only {
                return None;
            }
            continue;
        };
        if filled.contains(&position) {
            return None;
        }
        filled.push(position);
        bound.push((declared_at(position), given.clone()));
    }
    if imp
        .fits
        .chars()
        .take(imp.params.len())
        .enumerate()
        .any(|(position, fit)| fit == FIT_REQUIRED && !filled.contains(&position))
    {
        return None;
    }
    Some(bound)
}

/// Whether the value dispatch may run an implementation for a reason only an indexed pin
/// overrules (`value_rival`): it declares another element type of an argument's container
/// kind, or, where `numbers`, an int for a float argument or the other way round.
pub fn value_rival<S: AsRef<str>>(imp: &ImplSig, shapes: &[S], numbers: bool) -> bool {
    for (declared, given) in imp.params.iter().zip(shapes) {
        let given = given.as_ref();
        if declared == given {
            continue;
        }
        if is_array(given) || is_matrix(given) || is_map(given) {
            if kind_of(declared) == kind_of(given) {
                return true;
            }
        } else if numbers && NUMERIC.contains(&given) && NUMERIC.contains(&declared.as_str()) {
            return true;
        }
    }
    false
}

/// Whether an argument type leaves the overload a call runs to its value
/// (`is_unsettled`): UNKNOWN, an object whose class was lost, a container of either.
pub fn is_unsettled(ty: &str) -> bool {
    if ty == UNKNOWN || ty == OBJECT {
        return true;
    }
    if is_array(ty) || is_matrix(ty) {
        return is_unsettled(element_of(ty));
    }
    if is_map(ty) {
        return is_unsettled(key_of(ty)) || is_unsettled(value_of(ty));
    }
    false
}

/// Whether a type is one Pine's overload resolution compares as it stands
/// (`_pine_settled`).
fn pine_settled(ty: &str) -> bool {
    if SCALARS.contains(&ty) {
        return true;
    }
    if is_array(ty) || is_matrix(ty) {
        return pine_settled(element_of(ty));
    }
    if is_map(ty) {
        return pine_settled(key_of(ty)) && pine_settled(value_of(ty));
    }
    class_of(ty).is_some()
}

/// Whether Pine passes an argument of one type to a parameter of another
/// (`_pine_takes`).
fn pine_takes(declared: &str, given: &str, widening: bool) -> bool {
    if declared == given || constant_scalar(given) == declared {
        return true;
    }
    widening && given == INT && declared == FLOAT
}

fn shape_at<S: AsRef<str>>(shapes: &[S], index: usize) -> &str {
    shapes.get(index).map_or(UNKNOWN, AsRef::as_ref)
}

/// How one implementation answers a pin in one half of one pass of the selector
/// (`_takes`).
///
/// :param strict: The exact pass; the widening pass otherwise
/// :param wild: Whether a wildcard position is left out of the match (`resolve_pin`)
fn takes<S: AsRef<str>>(
    imp: &ImplSig,
    wanted: &[&str],
    exact_arity: bool,
    shapes: &[S],
    strict: bool,
    wild: bool,
) -> Verdict {
    if imp.open_ended || imp.kwarg {
        return Verdict::Maybe;
    }
    let argc = wanted.len();
    let fits: Vec<char> = if strict { imp.fits.chars().collect() } else { wide_fits(imp) };
    if exact_arity {
        // The positional half counts EVERY visible parameter against the arguments, a
        // keyword-only one included
        if imp.fits.chars().count() != argc {
            return Verdict::No;
        }
        if imp.params.len() != argc {
            return Verdict::Maybe;
        }
    } else if !(imp.required <= argc && argc <= imp.params.len()) {
        return Verdict::No;
    }
    let mut verdict = Verdict::Yes;
    for (index, (declared, want)) in imp.params.iter().zip(wanted).enumerate() {
        if is_bare(imp, index) || (wild && is_pin_any(want)) {
            continue;
        }
        let taken = position(imp, index, declared, want, shape_at(shapes, index), strict);
        if taken == Verdict::No {
            return Verdict::No;
        }
        verdict = verdict.min(taken);
    }
    if exact_arity {
        return verdict;
    }
    for &fit in fits.iter().skip(argc) {
        if fit == FIT_REQUIRED {
            return Verdict::No;
        }
        if fit == FIT_UNSURE {
            verdict = Verdict::Maybe;
        }
    }
    verdict
}

/// Whether a pin character is the wildcard.
fn is_pin_any(want: &str) -> bool {
    one_char(want) && want.starts_with(PIN_ANY)
}

/// What each parameter contributes to the widening pass when it is omitted
/// (`_wide_fits`): the shape's own, else what the exact pass's say of it.
fn wide_fits(imp: &ImplSig) -> Vec<char> {
    if imp.wide_fits.chars().count() == imp.fits.chars().count() {
        return imp.wide_fits.chars().collect();
    }
    imp.fits
        .chars()
        .enumerate()
        .map(
            |(index, fit)| {
                if fit != FIT_REQUIRED || index < imp.required { fit } else { FIT_UNSURE }
            },
        )
        .collect()
}

/// How one positional parameter answers one argument in one pass (`_position`).
fn position(
    imp: &ImplSig,
    index: usize,
    declared: &str,
    want: &str,
    given: &str,
    strict: bool,
) -> Verdict {
    if is_pin_any(want) {
        return if strict { takes_position(declared, want, given) } else { Verdict::Maybe };
    }
    let spec = imp.accepts.get(index).map_or("", String::as_str);
    if spec == ACCEPTS_UNKNOWN {
        return Verdict::Maybe;
    }
    if !spec.is_empty() {
        let (exact, wide) = spec.split_once(ACCEPTS_SEP).unwrap_or((spec, ""));
        let taken = if strict { exact } else { wide };
        return if taken.contains(want) { Verdict::Yes } else { Verdict::No };
    }
    if strict { takes_position(declared, want, given) } else { widens_position(declared, want) }
}

/// How one parameter answers a witness in the widening pass, by its declared type
/// (`_widens_position`).
fn widens_position(declared: &str, want: &str) -> Verdict {
    let exact = takes_position(declared, want, UNKNOWN);
    if exact != Verdict::No {
        return exact;
    }
    let Some(shaped) = pin_container_type(want) else {
        return if is_numeric(head(declared)) && is_numeric(want) {
            Verdict::Yes
        } else {
            Verdict::No
        };
    };
    if (is_array(declared) && is_array(shaped)) || (is_matrix(declared) && is_matrix(shaped)) {
        return if is_numeric(element_of(declared)) && is_numeric(element_of(shaped)) {
            Verdict::Yes
        } else {
            Verdict::No
        };
    }
    Verdict::No
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
        if given == COLOR {
            return if declared == COLOR { Verdict::Yes } else { Verdict::Maybe };
        }
        // A container is checked by what it is, then by a sampled element
        let declared_kind = kind_of(declared);
        if !(is_array(given) || is_matrix(given) || is_map(given)) || declared_kind.is_none() {
            return Verdict::Maybe;
        }
        if kind_of(given) != declared_kind {
            return Verdict::No;
        }
        return if declared == given && strictly_held(given) {
            Verdict::Yes
        } else {
            Verdict::Maybe
        };
    }
    if let Some(shaped) = pin_container_type(want) {
        if declared == shaped {
            return Verdict::Yes;
        }
        if element_of(shaped) == STR
            && is_str_literal(element_of(declared))
            && ((is_array(declared) && is_array(shaped))
                || (is_matrix(declared) && is_matrix(shaped)))
        {
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
    if want == STR && is_str_literal(declared) {
        // The selector takes any string for a ``StrLiteral`` subclass
        return Verdict::Yes;
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
    if imp.open_ended || imp.kwarg {
        return Verdict::Maybe;
    }
    if wanted.len() > imp.params.len() {
        return Verdict::No;
    }
    let mut verdict = Verdict::Yes;
    let mut bound: Vec<usize> = (0..wanted.len()).collect();
    for (index, (declared, want)) in imp.params.iter().zip(wanted).enumerate() {
        if is_bare(imp, index) {
            continue;
        }
        let taken = position(imp, index, declared, want, shape_at(shapes, index), true);
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
        if is_bare(imp, index) {
            continue;
        }
        let given = kw_shapes
            .iter()
            .find(|(known, _)| known == name)
            .map_or(UNKNOWN, |(_, shape)| shape.as_str());
        let declared = imp.params.get(index).map_or(UNKNOWN, String::as_str);
        let taken = position(imp, index, declared, want, given, true);
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
    kwarg: bool,
    kwonly: &'a [NodeId],
    kw_defaults: &'a [Option<NodeId>],
    defaults: &'a [NodeId],
}

fn params(ast: &Ast, func: NodeId) -> Result<Params<'_>, Unsupported> {
    let args = match ast.kind(func) {
        Kind::FunctionDef { args, .. } | Kind::AsyncFunctionDef { args, .. } => *args,
        _ => return Err(Unsupported(format!("a definition expected, not {}", ast.name(func)))),
    };
    let Kind::Arguments { posonlyargs, args, vararg, kwonlyargs, kw_defaults, kwarg, defaults } =
        ast.kind(args)
    else {
        return Err(Unsupported("a definition without arguments".into()));
    };
    Ok(Params {
        positional: posonlyargs.iter().chain(args).copied().collect(),
        vararg: vararg.is_some(),
        kwarg: kwarg.is_some(),
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
    enums: &Enums,
) -> Result<ImplSig, Unsupported> {
    let found = params(ast, func)?;
    let defaults = param_defaults(ast, func)?;
    let default_of =
        |name: &str| defaults.iter().find(|(known, _)| known == name).map(|&(_, default)| default);
    let mut types = Vec::with_capacity(found.positional.len());
    let mut bare = String::with_capacity(found.positional.len());
    let mut accepts = Vec::with_capacity(found.positional.len());
    let mut quals = String::with_capacity(found.positional.len());
    for &arg in &found.positional {
        let annotation = arg_parts(ast, arg).1;
        let declared = annotation_type(ast, annotation, classes, enums)?;
        accepts.push(own_accepts(ast, annotation, &declared, classes)?);
        types.push(declared);
        bare.push(if annotation.is_none() { '1' } else { '0' });
        quals.push(annotation_qualifier(ast, annotation).unwrap_or(QUAL_PLAIN));
    }
    if quals.chars().all(|qual| qual == QUAL_PLAIN) {
        quals.clear();
    }
    let mut fits = String::new();
    let mut wide_fits = String::new();
    for &arg in found.positional.iter().chain(found.kwonly) {
        let default = default_of(arg_parts(ast, arg).0);
        fits.push(param_fit(ast, arg, default, ty_of, classes, enums, true)?);
        wide_fits.push(param_fit(ast, arg, default, ty_of, classes, enums, false)?);
    }
    if accepts.iter().all(String::is_empty) {
        accepts.clear();
    }
    Ok(ImplSig {
        params: types,
        required: found.positional.len().saturating_sub(found.defaults.len()),
        open_ended: found.vararg,
        ret: ret.to_string(),
        fits,
        names: found.positional.iter().map(|&arg| arg_parts(ast, arg).0.to_string()).collect(),
        bare,
        slot: -1,
        accepts,
        wide_fits,
        kwarg: found.kwarg,
        size: -1,
        quals,
    })
}

/// What one parameter contributes to a call that omits it (`param_fit`).
///
/// :param strict: The selector's exact pass; its widening pass otherwise
pub fn param_fit(
    ast: &Ast,
    arg: NodeId,
    default: Option<NodeId>,
    ty_of: &dyn Fn(NodeId) -> String,
    classes: &Classes,
    enums: &Enums,
    strict: bool,
) -> Result<char, Unsupported> {
    let Some(default) = default else {
        return Ok(FIT_REQUIRED);
    };
    let Some(annotation) = arg_parts(ast, arg).1 else {
        return Ok(FIT_OMISSIBLE);
    };
    let declared = annotation_type(ast, Some(annotation), classes, enums)?;
    let default = default_ty(ast, default, ty_of);
    if PINNABLE.contains(&default.as_str()) {
        let spec = own_accepts(ast, Some(annotation), &declared, classes)?;
        if spec == ACCEPTS_UNKNOWN {
            return Ok(FIT_UNSURE);
        }
        if !spec.is_empty() {
            let (exact, wide) = spec.split_once(ACCEPTS_SEP).unwrap_or((&spec, ""));
            let taken = if strict { exact } else { wide };
            return Ok(if taken.contains(default.as_str()) { FIT_OMISSIBLE } else { FIT_REQUIRED });
        }
    }
    let takes_none = annotation_takes_none(ast, Some(annotation))?;
    Ok(if strict {
        default_fit(&declared, Some(&default), takes_none)
    } else {
        default_fit_wide(&declared, Some(&default), takes_none)
    })
}

/// The acceptance spec `ImplSig.accepts` records for one parameter (`_own_accepts`):
/// nothing where the declared type reads the witnesses the way the selector checks
/// them, or where the type is not known.
fn own_accepts(
    ast: &Ast,
    annotation: Option<NodeId>,
    declared: &str,
    classes: &Classes,
) -> Result<String, Unsupported> {
    let Some(annotation) = annotation else {
        return Ok(String::new());
    };
    if declared == UNKNOWN {
        return Ok(String::new());
    }
    let spec = annotation_accepts(ast, annotation, classes)?;
    Ok(if derived_accepts(declared).as_deref() == Some(spec.as_str()) {
        String::new()
    } else {
        spec
    })
}

/// The acceptance spec a declared type reads as (`_derived_accepts`), `None` where it
/// leaves a witness undecided.
fn derived_accepts(declared: &str) -> Option<String> {
    let mut passes = Vec::with_capacity(2);
    for strict in [true, false] {
        let mut taken = String::new();
        for want in pin_items(PIN_WITNESSED) {
            let verdict = if strict {
                takes_position(declared, want, UNKNOWN)
            } else {
                widens_position(declared, want)
            };
            match verdict {
                Verdict::Maybe => return None,
                Verdict::Yes => taken.push_str(want),
                Verdict::No => {}
            }
        }
        passes.push(taken);
    }
    Some(passes.join(&ACCEPTS_SEP.to_string()))
}

/// Every witness character in both passes: what `Any` and `object` take.
const EVERY_WITNESS: &str = "ifbsIFBS1234/ifbsIFBS1234";

/// No witness in either pass: what a class, a map, a tuple and `None` take.
const NO_WITNESS: &str = "/";

/// The acceptance spec of an annotation name, measured (`_NAME_ACCEPTS`).
const NAME_ACCEPTS: &[(&str, &str)] = &[
    ("int", "i/if"),
    ("float", "f/if"),
    ("bool", "b/b"),
    ("str", "s/s"),
    ("PyneInt", "if/if"),
    ("PyneFloat", "f/if"),
    ("PyneBool", "b/b"),
    ("PyneStr", "s/s"),
    ("Any", EVERY_WITNESS),
    ("object", EVERY_WITNESS),
    ("StrLiteral", "s/s"),
    ("list", "IFBS/IFBS"),
    ("Matrix", "1234/1234"),
    ("Color", NO_WITNESS),
    ("NA", NO_WITNESS),
    ("dict", NO_WITNESS),
    ("tuple", NO_WITNESS),
    ("set", NO_WITNESS),
    ("NoneType", NO_WITNESS),
];

/// Annotation heads whose payload is checked as the annotation itself
/// (`_SAME_ACCEPTS_SUBSCRIPTS`).
const SAME_ACCEPTS_SUBSCRIPTS: &[&str] =
    &["Persistent", "IBPersistent", "Optional", "Series", "Simple", "Const"];

/// Annotation heads that take no witness whatever their payload (`_NO_WITNESS_SUBSCRIPTS`).
const NO_WITNESS_SUBSCRIPTS: &[&str] = &["NA", "dict", "tuple", "Tuple", "set"];

/// Container annotation head -> the witness character of each scalar element
/// (`_CONTAINER_WITNESSES`).
fn container_witnesses(name: &str) -> Option<[(char, char); 4]> {
    match name {
        "list" => Some([('i', 'I'), ('f', 'F'), ('b', 'B'), ('s', 'S')]),
        "Matrix" => Some([('i', '1'), ('f', '2'), ('b', '3'), ('s', '4')]),
        _ => None,
    }
}

/// Which witness values a parameter annotation takes, in each pass of the selector
/// (`annotation_accepts`): `<exact>/<widening>`, or `ACCEPTS_UNKNOWN`.
pub fn annotation_accepts(
    ast: &Ast,
    node: NodeId,
    classes: &Classes,
) -> Result<String, Unsupported> {
    let unknown = || Ok(ACCEPTS_UNKNOWN.to_string());
    match ast.kind(node) {
        Kind::Constant { value, .. } => match value {
            Constant::None => Ok(NO_WITNESS.to_string()),
            Constant::Str(text) => match forward_reference(text)? {
                Some((parsed, body)) => annotation_accepts(&parsed, body, classes),
                None => unknown(),
            },
            _ => unknown(),
        },
        &Kind::BinOp { left, op: Operator::BitOr, right } => {
            union_accepts(ast, &[left, right], classes)
        }
        &Kind::Subscript { value, slice, .. } => {
            let name = leaf_name(ast, value);
            if name == "Union" {
                let single = [slice];
                return union_accepts(ast, tuple_elts(ast, slice).unwrap_or(&single), classes);
            }
            if SAME_ACCEPTS_SUBSCRIPTS.contains(&name) {
                return annotation_accepts(ast, slice, classes);
            }
            if NO_WITNESS_SUBSCRIPTS.contains(&name) {
                return Ok(NO_WITNESS.to_string());
            }
            let Some(witnesses) = container_witnesses(name) else {
                return unknown();
            };
            let element = annotation_accepts(ast, slice, classes)?;
            if element == ACCEPTS_UNKNOWN {
                return unknown();
            }
            let passes: Vec<String> = element
                .split(ACCEPTS_SEP)
                .map(|taken| {
                    taken
                        .chars()
                        .filter_map(|char| {
                            witnesses.iter().find(|(scalar, _)| *scalar == char).map(|(_, c)| *c)
                        })
                        .collect()
                })
                .collect();
            Ok(passes.join(&ACCEPTS_SEP.to_string()))
        }
        Kind::Name { .. } | Kind::Attribute { .. } => {
            let leaf = leaf_name(ast, node);
            let mut found = dotted(ast, node).and_then(|spelled| classes.get(&spelled).cloned());
            if found.is_none()
                && let Some(&(_, spec)) = NAME_ACCEPTS.iter().find(|&&(known, _)| known == leaf)
            {
                return Ok(spec.to_string());
            }
            if found.is_none() {
                found = classes.get(leaf).cloned();
            }
            let ty = match found {
                Some(found) => object_ty(&found),
                None => match builtin_class(leaf) {
                    Some(builtin) => builtin,
                    None => return unknown(),
                },
            };
            Ok(if is_str_literal(&ty) { "s/s".to_string() } else { NO_WITNESS.to_string() })
        }
        _ => unknown(),
    }
}

/// The acceptance spec of a union: what any of its members takes (`_union_accepts`).
fn union_accepts(ast: &Ast, members: &[NodeId], classes: &Classes) -> Result<String, Unsupported> {
    let mut exact = String::new();
    let mut wide = String::new();
    for &member in members {
        let spec = annotation_accepts(ast, member, classes)?;
        if spec == ACCEPTS_UNKNOWN {
            return Ok(spec);
        }
        let (member_exact, member_wide) = spec.split_once(ACCEPTS_SEP).unwrap_or((&spec, ""));
        exact.push_str(member_exact);
        wide.push_str(member_wide);
    }
    let order = |taken: &str| -> String {
        PIN_WITNESSED.chars().filter(|char| taken.contains(*char)).collect()
    };
    Ok(format!("{}{ACCEPTS_SEP}{}", order(&exact), order(&wide)))
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
                Kind::ImportFrom { module, names, level, .. } => {
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
                Kind::Import { names, .. } => {
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
