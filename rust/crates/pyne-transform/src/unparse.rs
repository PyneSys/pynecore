//! `ast.unparse` of an expression, native.
//!
//! A port of CPython 3.13's `ast._Unparser` for the expression classes a step
//! spells into a message: names, constants, the operators with their precedence
//! rules, calls, subscripts and slices, attributes, displays, conditional
//! expressions, lambdas and f-string fields. A class or a constant outside that set
//! (a comprehension, an `await`, a bytes or complex literal, a string with
//! characters whose `repr` depends on Unicode's printability tables) answers
//! [`Unsupported`], so the Python step decides.

use std::collections::HashMap;

use pyne_ast::{
    Ast, BoolOperator, CmpOperator, Constant, Int, Kind, NodeId, Operator, UnaryOperator,
    Unsupported,
};

/// `_Precedence`, lowest binding first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Prec {
    NamedExpr,
    Tuple,
    Yield,
    Test,
    Or,
    And,
    Not,
    Cmp,
    /// `EXPR`, which `BOR` is an alias of
    Bor,
    Bxor,
    Band,
    Shift,
    Arith,
    Term,
    Factor,
    Power,
    Await,
    Atom,
}

impl Prec {
    fn next(self) -> Prec {
        use Prec::*;
        match self {
            NamedExpr => Tuple,
            Tuple => Yield,
            Yield => Test,
            Test => Or,
            Or => And,
            And => Not,
            Not => Cmp,
            Cmp => Bor,
            Bor => Bxor,
            Bxor => Band,
            Band => Shift,
            Shift => Arith,
            Arith => Term,
            Term => Factor,
            Factor => Power,
            Power => Await,
            Await | Atom => Atom,
        }
    }
}

/// `repr(float)` with the unparser's substitutions for the infinities and NaN.
pub fn float_repr(value: f64) -> String {
    const INFSTR: &str = "1e309";
    if value.is_nan() {
        return format!("({INFSTR}-{INFSTR})");
    }
    if value.is_infinite() {
        return if value < 0.0 { format!("-{INFSTR}") } else { INFSTR.to_string() };
    }
    py_float_repr(value)
}

/// Python's `repr(float)` of a finite value: the shortest digits that round-trip,
/// in fixed notation while the decimal point stands within 16 digits, exponent
/// notation outside.
pub fn py_float_repr(value: f64) -> String {
    let sign = if value.is_sign_negative() { "-" } else { "" };
    let magnitude = value.abs();
    if magnitude == 0.0 {
        return format!("{sign}0.0");
    }
    // ``{:e}`` writes the shortest round-trip digits as ``d.ddde<exp>``
    let scientific = format!("{magnitude:e}");
    let (mantissa, exponent) = scientific.split_once('e').expect("exponent");
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let exponent: i32 = exponent.parse().expect("exponent digits");
    let decpt = exponent + 1;
    if decpt <= -4 || decpt > 16 {
        let rest = &digits[1..];
        let point = if rest.is_empty() { String::new() } else { format!(".{rest}") };
        let exp_sign = if exponent < 0 { '-' } else { '+' };
        return format!("{sign}{}{point}e{exp_sign}{:02}", &digits[..1], exponent.abs());
    }
    if decpt <= 0 {
        return format!("{sign}0.{}{digits}", "0".repeat((-decpt) as usize));
    }
    let decpt = decpt as usize;
    if decpt >= digits.len() {
        return format!("{sign}{digits}{}.0", "0".repeat(decpt - digits.len()));
    }
    format!("{sign}{}.{}", &digits[..decpt], &digits[decpt..])
}

/// Whether `str.isprintable` holds for a non-ASCII character, where that is known here
/// without the Unicode database: Latin-1 and the General Punctuation punctuation.
/// `None` for every other character.
fn printable_beyond_ascii(c: char) -> Option<bool> {
    match c as u32 {
        // C1 controls, the no-break space and the soft hyphen
        0x80..=0xa0 | 0xad => Some(false),
        0xa1..=0xff | 0x2010..=0x2027 | 0x2030..=0x205e => Some(true),
        _ => None,
    }
}

/// Python's `repr(str)`, for text whose characters are ASCII, Latin-1 or General
/// Punctuation.
pub fn str_repr(text: &str) -> Result<String, Unsupported> {
    if text.chars().any(|c| !c.is_ascii() && printable_beyond_ascii(c).is_none()) {
        return Err(Unsupported(format!("repr of non-ASCII text {text:?}")));
    }
    let quote = if text.contains('\'') && !text.contains('"') { '"' } else { '\'' };
    let mut out = String::with_capacity(text.len() + 2);
    out.push(quote);
    for c in text.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (' '..='~').contains(&c) => out.push(c),
            c if printable_beyond_ascii(c) == Some(true) => out.push(c),
            c => out.push_str(&format!("\\x{:02x}", c as u32)),
        }
    }
    out.push(quote);
    Ok(out)
}

fn constant_repr(value: &Constant) -> Result<String, Unsupported> {
    Ok(match value {
        Constant::None => "None".to_string(),
        Constant::Bool(true) => "True".to_string(),
        Constant::Bool(false) => "False".to_string(),
        Constant::Int(Int::Small(value)) => value.to_string(),
        Constant::Int(Int::Big(digits)) => digits.clone(),
        Constant::Float(value) => float_repr(*value),
        Constant::Str(text) => str_repr(text)?,
        Constant::Ellipsis => "...".to_string(),
        _ => return Err(Unsupported("unparse of a bytes, complex or tuple constant".into())),
    })
}

struct Unparser<'a> {
    ast: &'a Ast,
    out: String,
    precedences: HashMap<NodeId, Prec>,
}

impl Unparser<'_> {
    fn precedence(&self, node: NodeId) -> Prec {
        self.precedences.get(&node).copied().unwrap_or(Prec::Test)
    }

    fn set(&mut self, precedence: Prec, nodes: &[NodeId]) {
        for &node in nodes {
            self.precedences.insert(node, precedence);
        }
    }

    fn open_if(&mut self, condition: bool, start: &str) -> bool {
        if condition {
            self.out.push_str(start);
        }
        condition
    }

    fn close_if(&mut self, opened: bool, end: &str) {
        if opened {
            self.out.push_str(end);
        }
    }

    fn interleave(&mut self, nodes: &[NodeId]) -> Result<(), Unsupported> {
        for (index, &node) in nodes.iter().enumerate() {
            if index > 0 {
                self.out.push_str(", ");
            }
            self.traverse(node)?;
        }
        Ok(())
    }

    fn items_view(&mut self, nodes: &[NodeId]) -> Result<(), Unsupported> {
        if nodes.len() == 1 {
            self.traverse(nodes[0])?;
            self.out.push(',');
            Ok(())
        } else {
            self.interleave(nodes)
        }
    }

    fn traverse(&mut self, node: NodeId) -> Result<(), Unsupported> {
        let ast = self.ast;
        match ast.kind(node) {
            Kind::Name { id, .. } => self.out.push_str(id),
            Kind::Constant { value, kind } => {
                if kind.as_deref() == Some("u") {
                    self.out.push('u');
                }
                let text = constant_repr(value)?;
                self.out.push_str(&text);
            }
            Kind::NamedExpr { target, value } => {
                let opened = self.open_if(self.precedence(node) > Prec::NamedExpr, "(");
                self.set(Prec::Atom, &[*target, *value]);
                self.traverse(*target)?;
                self.out.push_str(" := ");
                self.traverse(*value)?;
                self.close_if(opened, ")");
            }
            Kind::UnaryOp { op, operand } => {
                let (operator, precedence) = match op {
                    UnaryOperator::Invert => ("~", Prec::Factor),
                    UnaryOperator::Not => ("not", Prec::Not),
                    UnaryOperator::UAdd => ("+", Prec::Factor),
                    UnaryOperator::USub => ("-", Prec::Factor),
                };
                let opened = self.open_if(self.precedence(node) > precedence, "(");
                self.out.push_str(operator);
                if precedence != Prec::Factor {
                    self.out.push(' ');
                }
                self.set(precedence, &[*operand]);
                self.traverse(*operand)?;
                self.close_if(opened, ")");
            }
            Kind::BinOp { left, op, right } => {
                let (operator, precedence) = match op {
                    Operator::Add => ("+", Prec::Arith),
                    Operator::Sub => ("-", Prec::Arith),
                    Operator::Mult => ("*", Prec::Term),
                    Operator::MatMult => ("@", Prec::Term),
                    Operator::Div => ("/", Prec::Term),
                    Operator::Mod => ("%", Prec::Term),
                    Operator::LShift => ("<<", Prec::Shift),
                    Operator::RShift => (">>", Prec::Shift),
                    Operator::BitOr => ("|", Prec::Bor),
                    Operator::BitXor => ("^", Prec::Bxor),
                    Operator::BitAnd => ("&", Prec::Band),
                    Operator::FloorDiv => ("//", Prec::Term),
                    Operator::Pow => ("**", Prec::Power),
                };
                let opened = self.open_if(self.precedence(node) > precedence, "(");
                let (left_prec, right_prec) = if operator == "**" {
                    (precedence.next(), precedence)
                } else {
                    (precedence, precedence.next())
                };
                self.set(left_prec, &[*left]);
                self.traverse(*left)?;
                self.out.push_str(&format!(" {operator} "));
                self.set(right_prec, &[*right]);
                self.traverse(*right)?;
                self.close_if(opened, ")");
            }
            Kind::Compare { left, ops, comparators } => {
                let opened = self.open_if(self.precedence(node) > Prec::Cmp, "(");
                self.set(Prec::Cmp.next(), &[*left]);
                self.set(Prec::Cmp.next(), comparators);
                self.traverse(*left)?;
                for (op, &comparator) in ops.iter().zip(comparators) {
                    let spelled = match op {
                        CmpOperator::Eq => "==",
                        CmpOperator::NotEq => "!=",
                        CmpOperator::Lt => "<",
                        CmpOperator::LtE => "<=",
                        CmpOperator::Gt => ">",
                        CmpOperator::GtE => ">=",
                        CmpOperator::Is => "is",
                        CmpOperator::IsNot => "is not",
                        CmpOperator::In => "in",
                        CmpOperator::NotIn => "not in",
                    };
                    self.out.push_str(&format!(" {spelled} "));
                    self.traverse(comparator)?;
                }
                self.close_if(opened, ")");
            }
            Kind::BoolOp { op, values } => {
                let (operator, mut precedence) = match op {
                    BoolOperator::And => ("and", Prec::And),
                    BoolOperator::Or => ("or", Prec::Or),
                };
                let opened = self.open_if(self.precedence(node) > precedence, "(");
                for (index, &value) in values.iter().enumerate() {
                    if index > 0 {
                        self.out.push_str(&format!(" {operator} "));
                    }
                    precedence = precedence.next();
                    self.set(precedence, &[value]);
                    self.traverse(value)?;
                }
                self.close_if(opened, ")");
            }
            Kind::IfExp { test, body, orelse } => {
                let opened = self.open_if(self.precedence(node) > Prec::Test, "(");
                self.set(Prec::Test.next(), &[*body, *test]);
                self.traverse(*body)?;
                self.out.push_str(" if ");
                self.traverse(*test)?;
                self.out.push_str(" else ");
                self.set(Prec::Test, &[*orelse]);
                self.traverse(*orelse)?;
                self.close_if(opened, ")");
            }
            Kind::Attribute { value, attr, .. } => {
                self.set(Prec::Atom, &[*value]);
                self.traverse(*value)?;
                if matches!(
                    ast.kind(*value),
                    Kind::Constant { value: Constant::Int(_) | Constant::Bool(_), .. }
                ) {
                    self.out.push(' ');
                }
                self.out.push('.');
                self.out.push_str(attr);
            }
            Kind::Call { func, args, keywords } => {
                self.set(Prec::Atom, &[*func]);
                self.traverse(*func)?;
                self.out.push('(');
                let mut comma = false;
                for &arg in args.iter().chain(keywords) {
                    if comma {
                        self.out.push_str(", ");
                    }
                    comma = true;
                    self.traverse(arg)?;
                }
                self.out.push(')');
            }
            Kind::Keyword { arg, value } => {
                match arg {
                    None => self.out.push_str("**"),
                    Some(arg) => {
                        self.out.push_str(arg);
                        self.out.push('=');
                    }
                }
                self.traverse(*value)?;
            }
            Kind::Subscript { value, slice, .. } => {
                self.set(Prec::Atom, &[*value]);
                self.traverse(*value)?;
                self.out.push('[');
                match ast.kind(*slice) {
                    Kind::Tuple { elts, .. } if !elts.is_empty() => self.items_view(elts)?,
                    _ => self.traverse(*slice)?,
                }
                self.out.push(']');
            }
            Kind::Starred { value, .. } => {
                self.out.push('*');
                self.set(Prec::Bor, &[*value]);
                self.traverse(*value)?;
            }
            Kind::Slice { lower, upper, step } => {
                if let Some(lower) = lower {
                    self.traverse(*lower)?;
                }
                self.out.push(':');
                if let Some(upper) = upper {
                    self.traverse(*upper)?;
                }
                if let Some(step) = step {
                    self.out.push(':');
                    self.traverse(*step)?;
                }
            }
            Kind::List { elts, .. } => {
                self.out.push('[');
                self.interleave(elts)?;
                self.out.push(']');
            }
            Kind::Tuple { elts, .. } => {
                let opened =
                    self.open_if(elts.is_empty() || self.precedence(node) > Prec::Tuple, "(");
                self.items_view(elts)?;
                self.close_if(opened, ")");
            }
            Kind::Set { elts } => {
                if elts.is_empty() {
                    self.out.push_str("{*()}");
                } else {
                    self.out.push('{');
                    self.interleave(elts)?;
                    self.out.push('}');
                }
            }
            Kind::Dict { keys, values } => {
                self.out.push('{');
                for (index, (key, &value)) in keys.iter().zip(values).enumerate() {
                    if index > 0 {
                        self.out.push_str(", ");
                    }
                    match key {
                        None => {
                            self.out.push_str("**");
                            self.set(Prec::Bor, &[value]);
                            self.traverse(value)?;
                        }
                        Some(key) => {
                            self.traverse(*key)?;
                            self.out.push_str(": ");
                            self.traverse(value)?;
                        }
                    }
                }
                self.out.push('}');
            }
            Kind::ListComp { elt, generators } => self.comprehension("[", *elt, generators, "]")?,
            Kind::SetComp { elt, generators } => self.comprehension("{", *elt, generators, "}")?,
            Kind::GeneratorExp { elt, generators } => {
                self.comprehension("(", *elt, generators, ")")?
            }
            Kind::DictComp { key, value, generators } => {
                self.out.push('{');
                self.traverse(*key)?;
                self.out.push_str(": ");
                self.traverse(*value)?;
                self.generators(generators)?;
                self.out.push('}');
            }
            Kind::Lambda { args, body } => {
                let opened = self.open_if(self.precedence(node) > Prec::Test, "(");
                self.out.push_str("lambda");
                let mark = self.out.len();
                self.out.push(' ');
                self.arguments(*args)?;
                if self.out.len() == mark + 1 {
                    self.out.truncate(mark);
                }
                self.out.push_str(": ");
                self.set(Prec::Test, &[*body]);
                self.traverse(*body)?;
                self.close_if(opened, ")");
            }
            Kind::FormattedValue { .. } => self.formatted_value(node)?,
            Kind::Arg { arg, annotation, .. } => {
                self.out.push_str(arg);
                if let Some(annotation) = annotation {
                    self.out.push_str(": ");
                    self.traverse(*annotation)?;
                }
            }
            _ => return Err(Unsupported(format!("unparse of {}", ast.name(node)))),
        }
        Ok(())
    }

    fn comprehension(
        &mut self,
        start: &str,
        elt: NodeId,
        generators: &[NodeId],
        end: &str,
    ) -> Result<(), Unsupported> {
        self.out.push_str(start);
        self.traverse(elt)?;
        self.generators(generators)?;
        self.out.push_str(end);
        Ok(())
    }

    fn generators(&mut self, generators: &[NodeId]) -> Result<(), Unsupported> {
        for &generator in generators {
            let Kind::Comprehension { target, iter, ifs, is_async } = self.ast.kind(generator)
            else {
                return Err(Unsupported("unparse of a malformed comprehension".into()));
            };
            self.out.push_str(if *is_async != 0 { " async for " } else { " for " });
            self.set(Prec::Tuple, &[*target]);
            self.traverse(*target)?;
            self.out.push_str(" in ");
            self.set(Prec::Test.next(), &[*iter]);
            self.set(Prec::Test.next(), ifs);
            self.traverse(*iter)?;
            for &condition in ifs {
                self.out.push_str(" if ");
                self.traverse(condition)?;
            }
        }
        Ok(())
    }

    fn arguments(&mut self, args: NodeId) -> Result<(), Unsupported> {
        let Kind::Arguments { posonlyargs, args, vararg, kwonlyargs, kw_defaults, kwarg, defaults } =
            self.ast.kind(args)
        else {
            return Err(Unsupported("lambda without arguments".into()));
        };
        let all: Vec<NodeId> = posonlyargs.iter().chain(args).copied().collect();
        let missing = all.len().saturating_sub(defaults.len());
        let mut first = true;
        for (index, &arg) in all.iter().enumerate() {
            if !first {
                self.out.push_str(", ");
            }
            first = false;
            self.traverse(arg)?;
            if index >= missing {
                self.out.push('=');
                self.traverse(defaults[index - missing])?;
            }
            if index + 1 == posonlyargs.len() {
                self.out.push_str(", /");
            }
        }
        if vararg.is_some() || !kwonlyargs.is_empty() {
            if !first {
                self.out.push_str(", ");
            }
            first = false;
            self.out.push('*');
            if let Some(vararg) = vararg {
                self.traverse(*vararg)?;
            }
        }
        for (&arg, default) in kwonlyargs.iter().zip(kw_defaults) {
            self.out.push_str(", ");
            self.traverse(arg)?;
            if let Some(default) = default {
                self.out.push('=');
                self.traverse(*default)?;
            }
        }
        if let Some(kwarg) = kwarg {
            if !first {
                self.out.push_str(", ");
            }
            self.out.push_str("**");
            self.traverse(*kwarg)?;
        }
        Ok(())
    }

    fn formatted_value(&mut self, node: NodeId) -> Result<(), Unsupported> {
        let Kind::FormattedValue { value, conversion, format_spec } = self.ast.kind(node) else {
            unreachable!()
        };
        self.out.push('{');
        let mut inner = Unparser { ast: self.ast, out: String::new(), precedences: HashMap::new() };
        inner.set(Prec::Test.next(), &[*value]);
        inner.traverse(*value)?;
        if inner.out.starts_with('{') {
            self.out.push(' ');
        }
        self.out.push_str(&inner.out);
        if *conversion != -1 {
            let conversion = u32::try_from(*conversion)
                .ok()
                .and_then(char::from_u32)
                .ok_or_else(|| Unsupported("f-string conversion".into()))?;
            self.out.push('!');
            self.out.push(conversion);
        }
        if let Some(spec) = format_spec {
            self.out.push(':');
            self.fstring_inner(*spec, true)?;
        }
        self.out.push('}');
        Ok(())
    }

    fn fstring_inner(&mut self, node: NodeId, is_format_spec: bool) -> Result<(), Unsupported> {
        match self.ast.kind(node) {
            Kind::JoinedStr { values } => {
                for &value in values {
                    self.fstring_inner(value, is_format_spec)?;
                }
            }
            Kind::Constant { value: Constant::Str(text), .. } => {
                let mut value = text.replace('{', "{{").replace('}', "}}");
                if is_format_spec {
                    value = value
                        .replace('\\', "\\\\")
                        .replace('\'', "\\'")
                        .replace('"', "\\\"")
                        .replace('\n', "\\n");
                }
                self.out.push_str(&value);
            }
            Kind::FormattedValue { .. } => self.formatted_value(node)?,
            _ => return Err(Unsupported("unexpected node inside an f-string".into())),
        }
        Ok(())
    }
}

/// `ast.unparse(node)` of an expression.
pub fn unparse(ast: &Ast, node: NodeId) -> Result<String, Unsupported> {
    let mut unparser = Unparser { ast, out: String::new(), precedences: HashMap::new() };
    unparser.traverse(node)?;
    Ok(unparser.out)
}

/// `ast.unparse(node)[:limit]`: the first `limit` characters.
pub fn unparse_prefix(ast: &Ast, node: NodeId, limit: usize) -> Result<String, Unsupported> {
    Ok(unparse(ast, node)?.chars().take(limit).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round(source: &str) -> String {
        let (ast, module) = pyne_parse::parse_module(source).expect("parses");
        let Kind::Module { body, .. } = ast.kind(module) else { panic!() };
        let Kind::Expr { value } = ast.kind(body[0]) else { panic!() };
        unparse(&ast, *value).expect("unparses")
    }

    #[test]
    fn precedence_and_spacing_follow_cpython() {
        assert_eq!(round("(a+b)*c"), "(a + b) * c");
        assert_eq!(round("a+b*c"), "a + b * c");
        assert_eq!(round("-x"), "-x");
        assert_eq!(round("not (a and b)"), "not (a and b)");
        assert_eq!(round("a[1:2]"), "a[1:2]");
        assert_eq!(round("x[::2]"), "x[::2]");
        assert_eq!(round("f(a, *b, k=1, **d)"), "f(a, *b, k=1, **d)");
        assert_eq!(round("lambda x, y=1: x+y"), "lambda x, y=1: x + y");
        assert_eq!(round("lambda: 0"), "lambda: 0");
        assert_eq!(round("2**3**4"), "2 ** 3 ** 4");
        assert_eq!(round("(2**3)**4"), "(2 ** 3) ** 4");
        assert_eq!(round("a if b else c"), "a if b else c");
        assert_eq!(round("(1,)"), "(1,)");
        assert_eq!(round("t[1, 2]"), "t[1, 2]");
    }

    #[test]
    fn constants_spell_their_repr() {
        assert_eq!(round("1e16"), "1e+16");
        assert_eq!(round("1e15"), "1000000000000000.0");
        assert_eq!(round("1e-5"), "1e-05");
        assert_eq!(round("0.0001"), "0.0001");
        assert_eq!(round("1.5"), "1.5");
        assert_eq!(round("\"it's\""), "\"it's\"");
        assert_eq!(round("'a\\nb'"), "'a\\nb'");
        assert_eq!(round("'Ended \u{b7} '"), "'Ended \u{b7} '");
        assert_eq!(round("'\u{a0}x\u{ad}'"), "'\\xa0x\\xad'");
    }

    #[test]
    fn comprehensions_follow_cpython() {
        assert_eq!(round("[x for x in y if x if z]"), "[x for x in y if x if z]");
        assert_eq!(round("{k: v for k, v in d.items()}"), "{k: v for k, v in d.items()}");
        assert_eq!(round("f(x for x in y)"), "f((x for x in y))");
        assert_eq!(round("{a for (a, b) in c}"), "{a for a, b in c}");
        assert_eq!(round("[(a, b) for a in (lambda: 1)()]"), "[(a, b) for a in (lambda: 1)()]");
        assert_eq!(round("[x for x in (a if b else c)]"), "[x for x in (a if b else c)]");
    }
}
