//! Python source to the CPython-shaped [`pyne_ast`] tree.
//!
//! The syntax is Ruff's parser; this crate lowers its tree into the node classes
//! and fields CPython's `ast.parse` produces, with the locations CPython reports
//! (1-based lines, UTF-8 byte columns). Where the two trees disagree the CPython
//! shape wins: an `elif` becomes a nested `If`, a `def` starts at its keyword
//! rather than its first decorator, a lone generator argument spans the call's
//! parentheses, and so on. The parity gate compares the result with
//! `ast.dump(ast.parse(source), include_attributes=True)` over the corpus.
//!
//! What this lowering does not reproduce yet is refused with [`Unsupported`]
//! (f-strings and t-strings among them), so a host falls back to CPython's own
//! parser for such a module.

use pyne_ast::{
    Ast, BoolOperator, CmpOperator, Constant, ExprContext, Int, Kind, Loc, NodeId, Operator,
    UnaryOperator, Unsupported,
};
use ruff_python_ast as ra;
use ruff_text_size::{Ranged, TextRange, TextSize};

type Result<T> = std::result::Result<T, Unsupported>;

/// Parse a module into a fresh arena.
///
/// :return: The arena and the `Module` node.
pub fn parse_module(source: &str) -> Result<(Ast, NodeId)> {
    let mut ast = Ast::new();
    let module = parse_module_into(&mut ast, source)?;
    Ok((ast, module))
}

/// Parse a module into an existing arena.
pub fn parse_module_into(ast: &mut Ast, source: &str) -> Result<NodeId> {
    let parsed = ruff_python_parser::parse_module(source)
        .map_err(|err| Unsupported(format!("syntax error: {err}")))?;
    let module = parsed.into_syntax();
    let mut lower = Lowerer::new(ast, source);
    let body = lower.suite(&module.body)?;
    Ok(lower.ast.add(Kind::Module {
        body,
        type_ignores: Vec::new(),
    }))
}

/// Parse a single expression (`ast.parse(source, mode='eval').body`).
pub fn parse_expression_into(ast: &mut Ast, source: &str) -> Result<NodeId> {
    let parsed = ruff_python_parser::parse_expression(source)
        .map_err(|err| Unsupported(format!("syntax error: {err}")))?;
    let expression = parsed.into_syntax();
    let mut lower = Lowerer::new(ast, source);
    lower.expr(&expression.body)
}

struct Lowerer<'a> {
    ast: &'a mut Ast,
    source: &'a str,
    /// Byte offset of the first character of every line
    line_starts: Vec<usize>,
}

impl<'a> Lowerer<'a> {
    fn new(ast: &'a mut Ast, source: &'a str) -> Lowerer<'a> {
        let bytes = source.as_bytes();
        let mut line_starts = vec![0];
        let mut index = 0;
        while index < bytes.len() {
            match bytes[index] {
                b'\n' => line_starts.push(index + 1),
                b'\r' => {
                    if bytes.get(index + 1) == Some(&b'\n') {
                        index += 1;
                    }
                    line_starts.push(index + 1);
                }
                _ => {}
            }
            index += 1;
        }
        Lowerer {
            ast,
            source,
            line_starts,
        }
    }

    /// The 1-based line and the byte column of an offset.
    fn position(&self, offset: TextSize) -> (i64, i64) {
        let offset = offset.to_usize();
        let line = self.line_starts.partition_point(|&start| start <= offset) - 1;
        ((line + 1) as i64, (offset - self.line_starts[line]) as i64)
    }

    fn loc(&self, range: TextRange) -> Loc {
        let (lineno, col) = self.position(range.start());
        let (end_lineno, end_col) = self.position(range.end());
        Loc::new(lineno, col, end_lineno, end_col)
    }

    fn add(&mut self, kind: Kind, range: TextRange) -> NodeId {
        let loc = self.loc(range);
        self.ast.add_at(kind, loc)
    }

    fn suite(&mut self, body: &[ra::Stmt]) -> Result<Vec<NodeId>> {
        body.iter().map(|stmt| self.stmt(stmt)).collect()
    }

    fn exprs(&mut self, items: &[ra::Expr]) -> Result<Vec<NodeId>> {
        items.iter().map(|item| self.expr(item)).collect()
    }

    fn opt_expr(&mut self, item: Option<&ra::Expr>) -> Result<Option<NodeId>> {
        item.map(|item| self.expr(item)).transpose()
    }

    /// Where a compound statement starts for CPython: its keyword, not its decorators.
    fn keyword_start(&self, name: &ra::Identifier, keyword: &str) -> Result<TextSize> {
        let bytes = self.source.as_bytes();
        let mut index = name.range.start().to_usize();
        index = skip_space_back(bytes, index);
        let start = index
            .checked_sub(keyword.len())
            .filter(|&start| &bytes[start..index] == keyword.as_bytes())
            .ok_or_else(|| Unsupported(format!("no `{keyword}` before `{}`", name.id)))?;
        // ``async def``: the statement starts at ``async``
        let before = skip_space_back(bytes, start);
        if keyword == "def" && before >= 5 && &bytes[before - 5..before] == b"async" {
            return Ok(TextSize::try_from(before - 5).unwrap());
        }
        Ok(TextSize::try_from(start).unwrap())
    }

    fn decorators(&mut self, decorators: &[ra::Decorator]) -> Result<Vec<NodeId>> {
        decorators
            .iter()
            .map(|decorator| self.expr(&decorator.expression))
            .collect()
    }

    fn stmt(&mut self, stmt: &ra::Stmt) -> Result<NodeId> {
        let range = stmt.range();
        let kind = match stmt {
            ra::Stmt::FunctionDef(def) => {
                let start = self.keyword_start(&def.name, "def")?;
                let decorator_list = self.decorators(&def.decorator_list)?;
                let args = self.parameters(Some(&def.parameters))?;
                let returns = self.opt_expr(def.returns.as_deref())?;
                let body = self.suite(&def.body)?;
                let type_params = self.type_params(def.type_params.as_deref())?;
                let name = def.name.id.to_string();
                let kind = if def.is_async {
                    Kind::AsyncFunctionDef {
                        name,
                        args,
                        body,
                        decorator_list,
                        returns,
                        type_comment: None,
                        type_params,
                    }
                } else {
                    Kind::FunctionDef {
                        name,
                        args,
                        body,
                        decorator_list,
                        returns,
                        type_comment: None,
                        type_params,
                    }
                };
                return Ok(self.add(kind, TextRange::new(start, range.end())));
            }
            ra::Stmt::ClassDef(def) => {
                let start = self.keyword_start(&def.name, "class")?;
                let decorator_list = self.decorators(&def.decorator_list)?;
                let (bases, keywords) = match def.arguments.as_deref() {
                    Some(arguments) => self.call_arguments(arguments)?,
                    None => (Vec::new(), Vec::new()),
                };
                let body = self.suite(&def.body)?;
                let type_params = self.type_params(def.type_params.as_deref())?;
                let kind = Kind::ClassDef {
                    name: def.name.id.to_string(),
                    bases,
                    keywords,
                    body,
                    decorator_list,
                    type_params,
                };
                return Ok(self.add(kind, TextRange::new(start, range.end())));
            }
            ra::Stmt::Return(node) => Kind::Return {
                value: self.opt_expr(node.value.as_deref())?,
            },
            ra::Stmt::Delete(node) => Kind::Delete {
                targets: self.exprs(&node.targets)?,
            },
            ra::Stmt::TypeAlias(node) => Kind::TypeAlias {
                name: self.expr(&node.name)?,
                type_params: self.type_params(node.type_params.as_deref())?,
                value: self.expr(&node.value)?,
            },
            ra::Stmt::Assign(node) => Kind::Assign {
                targets: self.exprs(&node.targets)?,
                value: self.expr(&node.value)?,
                type_comment: None,
            },
            ra::Stmt::AugAssign(node) => Kind::AugAssign {
                target: self.expr(&node.target)?,
                op: operator(node.op),
                value: self.expr(&node.value)?,
            },
            ra::Stmt::AnnAssign(node) => Kind::AnnAssign {
                target: self.expr(&node.target)?,
                annotation: self.expr(&node.annotation)?,
                value: self.opt_expr(node.value.as_deref())?,
                simple: i64::from(node.simple),
            },
            ra::Stmt::For(node) => {
                let target = self.expr(&node.target)?;
                let iter = self.expr(&node.iter)?;
                let body = self.suite(&node.body)?;
                let orelse = self.suite(&node.orelse)?;
                if node.is_async {
                    Kind::AsyncFor {
                        target,
                        iter,
                        body,
                        orelse,
                        type_comment: None,
                    }
                } else {
                    Kind::For {
                        target,
                        iter,
                        body,
                        orelse,
                        type_comment: None,
                    }
                }
            }
            ra::Stmt::While(node) => Kind::While {
                test: self.expr(&node.test)?,
                body: self.suite(&node.body)?,
                orelse: self.suite(&node.orelse)?,
            },
            ra::Stmt::If(node) => {
                let test = self.expr(&node.test)?;
                let body = self.suite(&node.body)?;
                let orelse = self.elif_else(&node.elif_else_clauses)?;
                Kind::If { test, body, orelse }
            }
            ra::Stmt::With(node) => {
                let items = node
                    .items
                    .iter()
                    .map(|item| {
                        let context_expr = self.expr(&item.context_expr)?;
                        let optional_vars = self.opt_expr(item.optional_vars.as_deref())?;
                        Ok(self.ast.add(Kind::WithItem {
                            context_expr,
                            optional_vars,
                        }))
                    })
                    .collect::<Result<Vec<_>>>()?;
                let body = self.suite(&node.body)?;
                if node.is_async {
                    Kind::AsyncWith {
                        items,
                        body,
                        type_comment: None,
                    }
                } else {
                    Kind::With {
                        items,
                        body,
                        type_comment: None,
                    }
                }
            }
            ra::Stmt::Match(node) => {
                let subject = self.expr(&node.subject)?;
                let cases = node
                    .cases
                    .iter()
                    .map(|case| {
                        let pattern = self.pattern(&case.pattern)?;
                        let guard = self.opt_expr(case.guard.as_deref())?;
                        let body = self.suite(&case.body)?;
                        Ok(self.ast.add(Kind::MatchCase {
                            pattern,
                            guard,
                            body,
                        }))
                    })
                    .collect::<Result<Vec<_>>>()?;
                Kind::Match { subject, cases }
            }
            ra::Stmt::Raise(node) => Kind::Raise {
                exc: self.opt_expr(node.exc.as_deref())?,
                cause: self.opt_expr(node.cause.as_deref())?,
            },
            ra::Stmt::Try(node) => {
                let body = self.suite(&node.body)?;
                let handlers = node
                    .handlers
                    .iter()
                    .map(|handler| {
                        let ra::ExceptHandler::ExceptHandler(handler) = handler;
                        let r#type = self.opt_expr(handler.type_.as_deref())?;
                        let name = handler.name.as_ref().map(|name| name.id.to_string());
                        let body = self.suite(&handler.body)?;
                        Ok(self.add(Kind::ExceptHandler { r#type, name, body }, handler.range))
                    })
                    .collect::<Result<Vec<_>>>()?;
                let orelse = self.suite(&node.orelse)?;
                let finalbody = self.suite(&node.finalbody)?;
                if node.is_star {
                    Kind::TryStar {
                        body,
                        handlers,
                        orelse,
                        finalbody,
                    }
                } else {
                    Kind::Try {
                        body,
                        handlers,
                        orelse,
                        finalbody,
                    }
                }
            }
            ra::Stmt::Assert(node) => Kind::Assert {
                test: self.expr(&node.test)?,
                msg: self.opt_expr(node.msg.as_deref())?,
            },
            ra::Stmt::Import(node) => {
                if node.is_lazy {
                    return Err(Unsupported("lazy import".into()));
                }
                Kind::Import {
                    names: self.aliases(&node.names),
                }
            }
            ra::Stmt::ImportFrom(node) => {
                if node.is_lazy {
                    return Err(Unsupported("lazy import".into()));
                }
                Kind::ImportFrom {
                    module: node.module.as_ref().map(|module| module.id.to_string()),
                    names: self.aliases(&node.names),
                    level: Some(i64::from(node.level)),
                }
            }
            ra::Stmt::Global(node) => Kind::Global {
                names: node.names.iter().map(|name| name.id.to_string()).collect(),
            },
            ra::Stmt::Nonlocal(node) => Kind::Nonlocal {
                names: node.names.iter().map(|name| name.id.to_string()).collect(),
            },
            ra::Stmt::Expr(node) => Kind::Expr {
                value: self.expr(&node.value)?,
            },
            ra::Stmt::Pass(_) => Kind::Pass,
            ra::Stmt::Break(_) => Kind::Break,
            ra::Stmt::Continue(_) => Kind::Continue,
            ra::Stmt::IpyEscapeCommand(_) => return Err(Unsupported("IPython escape".into())),
        };
        Ok(self.add(kind, range))
    }

    /// An `elif` chain as CPython nests it: each `elif` an `If` in the `orelse` of
    /// the one before, spanning from its keyword to the end of the whole chain.
    fn elif_else(&mut self, clauses: &[ra::ElifElseClause]) -> Result<Vec<NodeId>> {
        let Some((first, rest)) = clauses.split_first() else {
            return Ok(Vec::new());
        };
        match &first.test {
            None => self.suite(&first.body),
            Some(test) => {
                let test = self.expr(test)?;
                let body = self.suite(&first.body)?;
                let orelse = self.elif_else(rest)?;
                let end = clauses.last().unwrap().range.end();
                let range = TextRange::new(first.range.start(), end);
                Ok(vec![self.add(Kind::If { test, body, orelse }, range)])
            }
        }
    }

    fn aliases(&mut self, names: &[ra::Alias]) -> Vec<NodeId> {
        names
            .iter()
            .map(|alias| {
                let kind = Kind::Alias {
                    name: alias.name.id.to_string(),
                    asname: alias.asname.as_ref().map(|name| name.id.to_string()),
                };
                self.add(kind, alias.range)
            })
            .collect()
    }

    fn arg(&mut self, parameter: &ra::Parameter) -> Result<NodeId> {
        let annotation = self.opt_expr(parameter.annotation.as_deref())?;
        let kind = Kind::Arg {
            arg: parameter.name.id.to_string(),
            annotation,
            type_comment: None,
        };
        // CPython starts ``*args`` / ``**kwargs`` at the name, after the stars
        Ok(self.add(
            kind,
            TextRange::new(parameter.name.range.start(), parameter.range.end()),
        ))
    }

    fn parameters(&mut self, parameters: Option<&ra::Parameters>) -> Result<NodeId> {
        let mut posonlyargs = Vec::new();
        let mut args = Vec::new();
        let mut kwonlyargs = Vec::new();
        let mut defaults = Vec::new();
        let mut kw_defaults = Vec::new();
        let mut vararg = None;
        let mut kwarg = None;
        if let Some(parameters) = parameters {
            for parameter in &parameters.posonlyargs {
                posonlyargs.push(self.arg(&parameter.parameter)?);
                if let Some(default) = &parameter.default {
                    defaults.push(self.expr(default)?);
                }
            }
            for parameter in &parameters.args {
                args.push(self.arg(&parameter.parameter)?);
                if let Some(default) = &parameter.default {
                    defaults.push(self.expr(default)?);
                }
            }
            vararg = parameters
                .vararg
                .as_deref()
                .map(|p| self.arg(p))
                .transpose()?;
            for parameter in &parameters.kwonlyargs {
                kwonlyargs.push(self.arg(&parameter.parameter)?);
                kw_defaults.push(self.opt_expr(parameter.default.as_deref())?);
            }
            kwarg = parameters
                .kwarg
                .as_deref()
                .map(|p| self.arg(p))
                .transpose()?;
        }
        Ok(self.ast.add(Kind::Arguments {
            posonlyargs,
            args,
            vararg,
            kwonlyargs,
            kw_defaults,
            kwarg,
            defaults,
        }))
    }

    fn type_params(&mut self, params: Option<&ra::TypeParams>) -> Result<Vec<NodeId>> {
        let Some(params) = params else {
            return Ok(Vec::new());
        };
        params
            .type_params
            .iter()
            .map(|param| {
                let kind = match param {
                    ra::TypeParam::TypeVar(param) => Kind::TypeVar {
                        name: param.name.id.to_string(),
                        bound: self.opt_expr(param.bound.as_deref())?,
                        default_value: self.opt_expr(param.default.as_deref())?,
                    },
                    ra::TypeParam::TypeVarTuple(param) => Kind::TypeVarTuple {
                        name: param.name.id.to_string(),
                        default_value: self.opt_expr(param.default.as_deref())?,
                    },
                    ra::TypeParam::ParamSpec(param) => Kind::ParamSpec {
                        name: param.name.id.to_string(),
                        default_value: self.opt_expr(param.default.as_deref())?,
                    },
                };
                Ok(self.add(kind, param.range()))
            })
            .collect()
    }

    fn call_arguments(&mut self, arguments: &ra::Arguments) -> Result<(Vec<NodeId>, Vec<NodeId>)> {
        let lone_generator = arguments.args.len() == 1 && arguments.keywords.is_empty();
        let mut args = Vec::with_capacity(arguments.args.len());
        for arg in &arguments.args {
            match arg {
                // ``f(x for x in y)``: CPython spans the generator over the call's parentheses
                ra::Expr::Generator(generator) if lone_generator && !generator.parenthesized => {
                    let id = self.generator(generator)?;
                    self.ast[id].loc = self.loc(arguments.range);
                    args.push(id);
                }
                _ => args.push(self.expr(arg)?),
            }
        }
        let keywords = arguments
            .keywords
            .iter()
            .map(|keyword| {
                let value = self.expr(&keyword.value)?;
                let arg = keyword.arg.as_ref().map(|arg| arg.id.to_string());
                Ok(self.add(Kind::Keyword { arg, value }, keyword.range))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok((args, keywords))
    }

    fn comprehensions(&mut self, generators: &[ra::Comprehension]) -> Result<Vec<NodeId>> {
        generators
            .iter()
            .map(|generator| {
                let target = self.expr(&generator.target)?;
                let iter = self.expr(&generator.iter)?;
                let ifs = self.exprs(&generator.ifs)?;
                Ok(self.ast.add(Kind::Comprehension {
                    target,
                    iter,
                    ifs,
                    is_async: i64::from(generator.is_async),
                }))
            })
            .collect()
    }

    fn generator(&mut self, node: &ra::ExprGenerator) -> Result<NodeId> {
        let elt = self.expr(&node.elt)?;
        let generators = self.comprehensions(&node.generators)?;
        Ok(self.add(Kind::GeneratorExp { elt, generators }, node.range))
    }

    fn expr(&mut self, expr: &ra::Expr) -> Result<NodeId> {
        let range = expr.range();
        let kind = match expr {
            ra::Expr::BoolOp(node) => Kind::BoolOp {
                op: match node.op {
                    ra::BoolOp::And => BoolOperator::And,
                    ra::BoolOp::Or => BoolOperator::Or,
                },
                values: self.exprs(&node.values)?,
            },
            ra::Expr::Named(node) => Kind::NamedExpr {
                target: self.expr(&node.target)?,
                value: self.expr(&node.value)?,
            },
            ra::Expr::BinOp(node) => Kind::BinOp {
                left: self.expr(&node.left)?,
                op: operator(node.op),
                right: self.expr(&node.right)?,
            },
            ra::Expr::UnaryOp(node) => Kind::UnaryOp {
                op: match node.op {
                    ra::UnaryOp::Invert => UnaryOperator::Invert,
                    ra::UnaryOp::Not => UnaryOperator::Not,
                    ra::UnaryOp::UAdd => UnaryOperator::UAdd,
                    ra::UnaryOp::USub => UnaryOperator::USub,
                },
                operand: self.expr(&node.operand)?,
            },
            ra::Expr::Lambda(node) => Kind::Lambda {
                args: self.parameters(node.parameters.as_deref())?,
                body: self.expr(&node.body)?,
            },
            ra::Expr::If(node) => Kind::IfExp {
                test: self.expr(&node.test)?,
                body: self.expr(&node.body)?,
                orelse: self.expr(&node.orelse)?,
            },
            ra::Expr::Dict(node) => {
                let mut keys = Vec::with_capacity(node.items.len());
                let mut values = Vec::with_capacity(node.items.len());
                for item in &node.items {
                    keys.push(self.opt_expr(item.key.as_ref())?);
                    values.push(self.expr(&item.value)?);
                }
                Kind::Dict { keys, values }
            }
            ra::Expr::Set(node) => Kind::Set {
                elts: self.exprs(&node.elts)?,
            },
            ra::Expr::ListComp(node) => Kind::ListComp {
                elt: self.expr(&node.elt)?,
                generators: self.comprehensions(&node.generators)?,
            },
            ra::Expr::SetComp(node) => Kind::SetComp {
                elt: self.expr(&node.elt)?,
                generators: self.comprehensions(&node.generators)?,
            },
            ra::Expr::DictComp(node) => Kind::DictComp {
                key: match node.key.as_deref() {
                    Some(key) => self.expr(key)?,
                    None => return Err(Unsupported("dict comprehension without a key".into())),
                },
                value: self.expr(&node.value)?,
                generators: self.comprehensions(&node.generators)?,
            },
            ra::Expr::Generator(node) => return self.generator(node),
            ra::Expr::Await(node) => Kind::Await {
                value: self.expr(&node.value)?,
            },
            ra::Expr::Yield(node) => Kind::Yield {
                value: self.opt_expr(node.value.as_deref())?,
            },
            ra::Expr::YieldFrom(node) => Kind::YieldFrom {
                value: self.expr(&node.value)?,
            },
            ra::Expr::Compare(node) => {
                let (first, rest) = node.operands.split_first().expect("a comparison operand");
                Kind::Compare {
                    left: self.expr(first)?,
                    ops: node.ops.iter().map(|&op| cmp_operator(op)).collect(),
                    comparators: self.exprs(rest)?,
                }
            }
            ra::Expr::Call(node) => {
                let func = self.expr(&node.func)?;
                let (args, keywords) = self.call_arguments(&node.arguments)?;
                Kind::Call {
                    func,
                    args,
                    keywords,
                }
            }
            ra::Expr::FString(_) => return Err(Unsupported("f-string".into())),
            ra::Expr::TString(_) => return Err(Unsupported("t-string".into())),
            ra::Expr::StringLiteral(node) => Kind::Constant {
                value: Constant::Str(node.value.to_str().to_string()),
                kind: node.value.is_unicode().then(|| "u".to_string()),
            },
            ra::Expr::BytesLiteral(node) => Kind::Constant {
                value: Constant::Bytes(node.value.bytes().collect()),
                kind: None,
            },
            ra::Expr::NumberLiteral(node) => Kind::Constant {
                value: match &node.value {
                    ra::Number::Int(_) => Constant::Int(self.int_literal(range)?),
                    ra::Number::Float(value) => Constant::Float(*value),
                    ra::Number::Complex { real, imag } => Constant::Complex {
                        real: *real,
                        imag: *imag,
                    },
                },
                kind: None,
            },
            ra::Expr::BooleanLiteral(node) => Kind::Constant {
                value: Constant::Bool(node.value),
                kind: None,
            },
            ra::Expr::NoneLiteral(_) => Kind::Constant {
                value: Constant::None,
                kind: None,
            },
            ra::Expr::EllipsisLiteral(_) => Kind::Constant {
                value: Constant::Ellipsis,
                kind: None,
            },
            ra::Expr::Attribute(node) => Kind::Attribute {
                value: self.expr(&node.value)?,
                attr: node.attr.id.to_string(),
                ctx: context(node.ctx)?,
            },
            ra::Expr::Subscript(node) => Kind::Subscript {
                value: self.expr(&node.value)?,
                slice: self.expr(&node.slice)?,
                ctx: context(node.ctx)?,
            },
            ra::Expr::Starred(node) => Kind::Starred {
                value: self.expr(&node.value)?,
                ctx: context(node.ctx)?,
            },
            ra::Expr::Name(node) => Kind::Name {
                id: node.id.to_string(),
                ctx: context(node.ctx)?,
            },
            ra::Expr::List(node) => Kind::List {
                elts: self.exprs(&node.elts)?,
                ctx: context(node.ctx)?,
            },
            ra::Expr::Tuple(node) => Kind::Tuple {
                elts: self.exprs(&node.elts)?,
                ctx: context(node.ctx)?,
            },
            ra::Expr::Slice(node) => Kind::Slice {
                lower: self.opt_expr(node.lower.as_deref())?,
                upper: self.opt_expr(node.upper.as_deref())?,
                step: self.opt_expr(node.step.as_deref())?,
            },
            ra::Expr::IpyEscapeCommand(_) => return Err(Unsupported("IPython escape".into())),
        };
        Ok(self.add(kind, range))
    }

    /// An integer literal's value, from its spelling (Ruff keeps a too-large one as text).
    fn int_literal(&self, range: TextRange) -> Result<Int> {
        let text: String = self.source[range].chars().filter(|&c| c != '_').collect();
        let lower = text.to_ascii_lowercase();
        let (digits, radix) = if let Some(rest) = lower.strip_prefix("0x") {
            (rest, 16)
        } else if let Some(rest) = lower.strip_prefix("0o") {
            (rest, 8)
        } else if let Some(rest) = lower.strip_prefix("0b") {
            (rest, 2)
        } else {
            (lower.as_str(), 10)
        };
        if let Ok(value) = i64::from_str_radix(digits, radix) {
            return Ok(Int::Small(value));
        }
        decimal_digits(digits, radix)
            .map(Int::Big)
            .ok_or_else(|| Unsupported(format!("integer literal {text}")))
    }

    fn pattern(&mut self, pattern: &ra::Pattern) -> Result<NodeId> {
        let range = pattern.range();
        let kind = match pattern {
            ra::Pattern::MatchValue(node) => Kind::MatchValue {
                value: self.expr(&node.value)?,
            },
            ra::Pattern::MatchSingleton(node) => Kind::MatchSingleton {
                value: match node.value {
                    ra::Singleton::None => Constant::None,
                    ra::Singleton::True => Constant::Bool(true),
                    ra::Singleton::False => Constant::Bool(false),
                },
            },
            ra::Pattern::MatchSequence(node) => Kind::MatchSequence {
                patterns: node
                    .patterns
                    .iter()
                    .map(|p| self.pattern(p))
                    .collect::<Result<_>>()?,
            },
            ra::Pattern::MatchMapping(node) => Kind::MatchMapping {
                keys: self.exprs(&node.keys)?,
                patterns: node
                    .patterns
                    .iter()
                    .map(|p| self.pattern(p))
                    .collect::<Result<_>>()?,
                rest: node.rest.as_ref().map(|rest| rest.id.to_string()),
            },
            ra::Pattern::MatchClass(node) => Kind::MatchClass {
                cls: self.expr(&node.cls)?,
                patterns: node
                    .arguments
                    .patterns
                    .iter()
                    .map(|p| self.pattern(p))
                    .collect::<Result<_>>()?,
                kwd_attrs: node
                    .arguments
                    .keywords
                    .iter()
                    .map(|k| k.attr.id.to_string())
                    .collect(),
                kwd_patterns: node
                    .arguments
                    .keywords
                    .iter()
                    .map(|k| self.pattern(&k.pattern))
                    .collect::<Result<_>>()?,
            },
            ra::Pattern::MatchStar(node) => Kind::MatchStar {
                name: node.name.as_ref().map(|name| name.id.to_string()),
            },
            ra::Pattern::MatchAs(node) => Kind::MatchAs {
                pattern: node
                    .pattern
                    .as_deref()
                    .map(|p| self.pattern(p))
                    .transpose()?,
                name: node.name.as_ref().map(|name| name.id.to_string()),
            },
            ra::Pattern::MatchOr(node) => Kind::MatchOr {
                patterns: node
                    .patterns
                    .iter()
                    .map(|p| self.pattern(p))
                    .collect::<Result<_>>()?,
            },
        };
        Ok(self.add(kind, range))
    }
}

/// The offset before the whitespace (and line continuations) that end at `index`.
fn skip_space_back(bytes: &[u8], mut index: usize) -> usize {
    loop {
        while index > 0 && matches!(bytes[index - 1], b' ' | b'\t' | b'\x0c') {
            index -= 1;
        }
        // A backslash continuation: ``def \`` + newline + name
        if index >= 2 && bytes[index - 1] == b'\n' && bytes[index - 2] == b'\\' {
            index -= 2;
        } else if index >= 3 && &bytes[index - 3..index] == b"\\\r\n" {
            index -= 3;
        } else {
            return index;
        }
    }
}

/// The decimal spelling of an unsigned integer written in `radix`.
fn decimal_digits(digits: &str, radix: u32) -> Option<String> {
    // Little-endian limbs of 10^9
    let mut limbs: Vec<u64> = vec![0];
    for c in digits.chars() {
        let digit = u64::from(c.to_digit(radix)?);
        let mut carry = digit;
        for limb in limbs.iter_mut() {
            let value = *limb * u64::from(radix) + carry;
            *limb = value % 1_000_000_000;
            carry = value / 1_000_000_000;
        }
        while carry > 0 {
            limbs.push(carry % 1_000_000_000);
            carry /= 1_000_000_000;
        }
    }
    let mut out = limbs.last().unwrap().to_string();
    for limb in limbs.iter().rev().skip(1) {
        out.push_str(&format!("{limb:09}"));
    }
    Some(out)
}

fn operator(op: ra::Operator) -> Operator {
    match op {
        ra::Operator::Add => Operator::Add,
        ra::Operator::Sub => Operator::Sub,
        ra::Operator::Mult => Operator::Mult,
        ra::Operator::MatMult => Operator::MatMult,
        ra::Operator::Div => Operator::Div,
        ra::Operator::Mod => Operator::Mod,
        ra::Operator::Pow => Operator::Pow,
        ra::Operator::LShift => Operator::LShift,
        ra::Operator::RShift => Operator::RShift,
        ra::Operator::BitOr => Operator::BitOr,
        ra::Operator::BitXor => Operator::BitXor,
        ra::Operator::BitAnd => Operator::BitAnd,
        ra::Operator::FloorDiv => Operator::FloorDiv,
    }
}

fn cmp_operator(op: ra::CmpOp) -> CmpOperator {
    match op {
        ra::CmpOp::Eq => CmpOperator::Eq,
        ra::CmpOp::NotEq => CmpOperator::NotEq,
        ra::CmpOp::Lt => CmpOperator::Lt,
        ra::CmpOp::LtE => CmpOperator::LtE,
        ra::CmpOp::Gt => CmpOperator::Gt,
        ra::CmpOp::GtE => CmpOperator::GtE,
        ra::CmpOp::Is => CmpOperator::Is,
        ra::CmpOp::IsNot => CmpOperator::IsNot,
        ra::CmpOp::In => CmpOperator::In,
        ra::CmpOp::NotIn => CmpOperator::NotIn,
    }
}

fn context(ctx: ra::ExprContext) -> Result<ExprContext> {
    match ctx {
        ra::ExprContext::Load => Ok(ExprContext::Load),
        ra::ExprContext::Store => Ok(ExprContext::Store),
        ra::ExprContext::Del => Ok(ExprContext::Del),
        ra::ExprContext::Invalid => Err(Unsupported("invalid expression context".into())),
    }
}
