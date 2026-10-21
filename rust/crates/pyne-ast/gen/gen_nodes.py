"""
Generate ``src/nodes.rs``: the CPython ``ast`` node classes as one Rust enum.

The schema is read from the running interpreter's ``ast`` module (``_fields``,
``_field_types`` and ``_attributes``; ``_field_types`` needs Python 3.13+), so the
Rust model is the CPython one field for field. Run it with the newest supported
Python: older interpreters lack some classes and fields, which the Python bridge
handles when it converts.

    python3.14 rust/crates/pyne-ast/gen/gen_nodes.py

The generated file is checked in; ``tests`` in the bridge compare it with the
interpreter that runs them.
"""
import ast
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / 'src' / 'nodes.rs'

#: Abstract bases whose concrete subclasses are plain Rust enums, not nodes
ENUM_BASES = {
    'expr_context': 'ExprContext',
    'operator': 'Operator',
    'unaryop': 'UnaryOperator',
    'boolop': 'BoolOperator',
    'cmpop': 'CmpOperator',
}

#: Classes CPython keeps only for backward compatibility; the parser never makes them
DEPRECATED = {'AugLoad', 'AugStore', 'Param', 'Suite', 'ExtSlice', 'Index', 'slice'}

#: List fields that hold ``None`` entries next to nodes (``{**x}``, a keyword-only
#: parameter without a default)
OPTIONAL_ITEMS = {('Dict', 'keys'), ('arguments', 'kw_defaults')}

#: ``object``-typed fields; ``Interpolation.str`` holds the source text
CONSTANT_FIELDS = {('Constant', 'value'), ('MatchSingleton', 'value')}
STRING_OBJECT_FIELDS = {('Interpolation', 'str')}


def subclasses(base: type) -> list[type]:
    found = []
    for cls in base.__subclasses__():
        found.append(cls)
        found.extend(subclasses(cls))
    return found


def field_kind(cls: type, name: str, annotation: object) -> str:
    """The FieldKind of one field."""
    key = (cls.__name__, name)
    if key in CONSTANT_FIELDS:
        return 'Constant'
    if key in STRING_OBJECT_FIELDS:
        return 'Str'
    text = annotation if isinstance(annotation, str) else repr(annotation)
    if isinstance(annotation, type):
        if annotation is str:
            return 'Str'
        if annotation is int:
            return 'Int'
        if annotation.__name__ in ENUM_BASES:
            return ENUM_BASES[annotation.__name__]
        if issubclass(annotation, ast.AST):
            return 'Node'
        raise ValueError(f'{key}: {annotation!r}')
    text = str(annotation)
    if text.startswith('list['):
        inner = text[5:-1]
        if inner == 'str':
            return 'Strs'
        if inner == 'ast.cmpop':
            return 'CmpOperators'
        if inner.startswith('ast.'):
            return 'OptNodes' if key in OPTIONAL_ITEMS else 'Nodes'
        raise ValueError(f'{key}: {text}')
    if text.endswith(' | None'):
        inner = text[:-len(' | None')]
        if inner == 'str':
            return 'OptStr'
        if inner == 'int':
            return 'OptInt'
        if inner.startswith('ast.'):
            return 'OptNode'
        raise ValueError(f'{key}: {text}')
    raise ValueError(f'{key}: {text}')


RUST_TYPES = {
    'Node': 'NodeId',
    'OptNode': 'Option<NodeId>',
    'Nodes': 'Vec<NodeId>',
    'OptNodes': 'Vec<Option<NodeId>>',
    'Str': 'String',
    'OptStr': 'Option<String>',
    'Strs': 'Vec<String>',
    'Int': 'i64',
    'OptInt': 'Option<i64>',
    'Constant': 'Constant',
    'ExprContext': 'ExprContext',
    'Operator': 'Operator',
    'UnaryOperator': 'UnaryOperator',
    'BoolOperator': 'BoolOperator',
    'CmpOperators': 'Vec<CmpOperator>',
}

#: Field kinds that can hold a node: the ones a walk enters
NODE_KINDS = {'Node', 'OptNode', 'Nodes', 'OptNodes'}


def category(base: str) -> str:
    """The Category variant of an abstract base class name."""
    return 'Other' if base == 'AST' else ''.join(part.capitalize() for part in base.split('_'))


#: Rust variant names of the lowercase CPython classes
VARIANTS = {'alias': 'Alias', 'arg': 'Arg', 'arguments': 'Arguments',
            'comprehension': 'Comprehension', 'keyword': 'Keyword', 'match_case': 'MatchCase',
            'withitem': 'WithItem'}


def variant(cls: type) -> str:
    return VARIANTS.get(cls.__name__, cls.__name__)


def rust_field(name: str) -> str:
    return 'r#type' if name == 'type' else name


def main() -> None:
    if sys.version_info < (3, 14):
        raise SystemExit('run with Python 3.14+')
    all_classes = subclasses(ast.AST)
    enums: dict[str, list[str]] = {rust: [] for rust in ENUM_BASES.values()}
    nodes = []
    for cls in sorted(set(all_classes), key=lambda c: c.__name__):
        if cls.__name__ in DEPRECATED or cls.__subclasses__():
            continue
        if not cls._fields and not cls._attributes and cls.__bases__[0].__name__ in ENUM_BASES:
            enums[ENUM_BASES[cls.__bases__[0].__name__]].append(cls.__name__)
            continue
        if cls.__name__ in ENUM_BASES or cls is ast.AST:
            continue
        types = cls._field_types
        fields = [(name, field_kind(cls, name, types[name])) for name in cls._fields]
        nodes.append((cls, fields))

    out = []
    w = out.append
    w('// @generated by gen/gen_nodes.py from the CPython 3.14 `ast` module. Do not edit.')
    w('#![allow(clippy::all)]')
    w('')
    w('use crate::{Constant, NodeId};')
    w('')
    for rust, members in enums.items():
        w('#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]')
        w(f'pub enum {rust} {{')
        for member in members:
            w(f'    {member},')
        w('}')
        w('')
        w(f'impl {rust} {{')
        w('    pub const ALL: &[Self] = &[' + ', '.join(f'Self::{m}' for m in members) + '];')
        w('')
        w('    /// The CPython class name.')
        w('    pub fn name(self) -> &\'static str {')
        w('        match self {')
        for member in members:
            w(f'            Self::{member} => "{member}",')
        w('        }')
        w('    }')
        w('')
        w('    pub fn from_name(name: &str) -> Option<Self> {')
        w('        match name {')
        for member in members:
            w(f'            "{member}" => Some(Self::{member}),')
        w('            _ => None,')
        w('        }')
        w('    }')
        w('}')
        w('')

    w('/// The category a concrete node class belongs to (its CPython abstract base).')
    w('#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]')
    w('pub enum Category {')
    bases = sorted({cls.__bases__[0].__name__ for cls, _ in nodes})
    for base in bases:
        w(f'    {category(base)},')
    w('}')
    w('')

    w('/// What a field holds.')
    w('#[derive(Clone, Copy, Debug, PartialEq, Eq)]')
    w('pub enum FieldKind {')
    for kind in RUST_TYPES:
        w(f'    {kind},')
    w('}')
    w('')
    w('/// One field of a node class.')
    w('#[derive(Debug)]')
    w('pub struct FieldInfo {')
    w('    pub name: &\'static str,')
    w('    pub kind: FieldKind,')
    w('}')
    w('')
    w('/// One concrete node class.')
    w('#[derive(Debug)]')
    w('pub struct ClassInfo {')
    w('    pub name: &\'static str,')
    w('    pub category: Category,')
    w('    pub fields: &\'static [FieldInfo],')
    w('    /// Whether the class carries `lineno` / `col_offset` / `end_*`')
    w('    pub located: bool,')
    w('}')
    w('')
    w('pub static CLASSES: &[ClassInfo] = &[')
    for cls, fields in nodes:
        base = cls.__bases__[0].__name__
        cat = category(base)
        located = 'true' if cls._attributes else 'false'
        field_list = ', '.join(
            f'FieldInfo {{ name: "{name}", kind: FieldKind::{kind} }}' for name, kind in fields)
        w(f'    ClassInfo {{ name: "{cls.__name__}", category: Category::{cat}, '
          f'fields: &[{field_list}], located: {located} }},')
    w('];')
    w('')

    w('/// A node: one variant per concrete CPython class, its fields in `_fields` order.')
    w('#[derive(Clone, Debug, PartialEq)]')
    w('pub enum Kind {')
    for cls, fields in nodes:
        if fields:
            w(f'    {variant(cls)} {{')
            for name, kind in fields:
                w(f'        {rust_field(name)}: {RUST_TYPES[kind]},')
            w('    },')
        else:
            w(f'    {variant(cls)},')
    w('}')
    w('')

    w('/// A borrowed field value, for code that walks every class alike.')
    w('#[derive(Debug)]')
    w('pub enum FieldRef<\'a> {')
    for kind, rust in RUST_TYPES.items():
        w(f'    {kind}(&\'a {rust}),')
    w('}')
    w('')
    w('/// A mutably borrowed field value.')
    w('#[derive(Debug)]')
    w('pub enum FieldMut<\'a> {')
    for kind, rust in RUST_TYPES.items():
        w(f'    {kind}(&\'a mut {rust}),')
    w('}')
    w('')

    w('impl Kind {')
    w('    /// Index of the class in [`CLASSES`].')
    w('    pub fn class_index(&self) -> usize {')
    w('        match self {')
    for index, (cls, fields) in enumerate(nodes):
        pattern = f'{variant(cls)} {{ .. }}' if fields else variant(cls)
        w(f'            Kind::{pattern} => {index},')
    w('        }')
    w('    }')
    w('')
    w('    pub fn class(&self) -> &\'static ClassInfo {')
    w('        &CLASSES[self.class_index()]')
    w('    }')
    w('')
    w('    /// The CPython class name.')
    w('    pub fn name(&self) -> &\'static str {')
    w('        self.class().name')
    w('    }')
    w('')
    w('    /// The field at `index` of the class\'s `_fields`.')
    w('    pub fn field(&self, index: usize) -> FieldRef<\'_> {')
    w('        match (self, index) {')
    for cls, fields in nodes:
        for index, (name, kind) in enumerate(fields):
            w(f'            (Kind::{variant(cls)} {{ {rust_field(name)}, .. }}, {index}) => '
              f'FieldRef::{kind}({rust_field(name)}),')
    w('            _ => panic!("{} has no field {}", self.name(), index),')
    w('        }')
    w('    }')
    w('')
    w('    pub fn field_mut(&mut self, index: usize) -> FieldMut<\'_> {')
    w('        let name = self.name();')
    w('        match (self, index) {')
    for cls, fields in nodes:
        for index, (name, kind) in enumerate(fields):
            w(f'            (Kind::{variant(cls)} {{ {rust_field(name)}, .. }}, {index}) => '
              f'FieldMut::{kind}({rust_field(name)}),')
    w('            _ => panic!("{} has no field {}", name, index),')
    w('        }')
    w('    }')
    w('')
    w('    /// A node of class `index` with every field at its CPython default: empty')
    w('    /// lists, `None`, `0`, `Load` and the first enum member. The bridge fills')
    w('    /// it field by field.')
    w('    pub fn empty(index: usize) -> Kind {')
    w('        match index {')
    defaults = {
        'Node': 'NodeId::PLACEHOLDER', 'OptNode': 'None', 'Nodes': 'Vec::new()',
        'OptNodes': 'Vec::new()', 'Str': 'String::new()', 'OptStr': 'None',
        'Strs': 'Vec::new()', 'Int': '0', 'OptInt': 'None', 'Constant': 'Constant::None',
        'ExprContext': 'ExprContext::Load', 'Operator': 'Operator::Add',
        'UnaryOperator': 'UnaryOperator::Invert', 'BoolOperator': 'BoolOperator::And',
        'CmpOperators': 'Vec::new()',
    }
    for index, (cls, fields) in enumerate(nodes):
        if fields:
            body = ', '.join(f'{rust_field(name)}: {defaults[kind]}' for name, kind in fields)
            w(f'            {index} => Kind::{variant(cls)} {{ {body} }},')
        else:
            w(f'            {index} => Kind::{variant(cls)},')
    w('            _ => panic!("no node class {}", index),')
    w('        }')
    w('    }')
    w('}')
    w('')
    w('/// Index of a class in [`CLASSES`] by its CPython name.')
    w('pub fn class_index(name: &str) -> Option<usize> {')
    w('    match name {')
    for index, (cls, _) in enumerate(nodes):
        w(f'        "{cls.__name__}" => Some({index}),')
    w('        _ => None,')
    w('    }')
    w('}')
    OUT.write_text('\n'.join(out) + '\n')
    print(f'wrote {OUT} ({len(nodes)} classes)')


if __name__ == '__main__':
    main()
