"""
The Pine type lattice and the measured TradingView type algebra.

Pine's ``int`` is a STATIC type only: at run time every number is a double, an
int-typed expression keeps its fractional value (``14 / 8`` is int-typed and
1.75), and truncation happens where an integer is consumed. The type therefore
decides only two things -- which overload a call site resolves to, and which
consuming slots truncate -- and both are decided before the program runs.

This module holds the vocabulary and the rules; it deliberately has no AST
walking of its own, so the algebra can be unit-tested one rule at a time.

The rules here are MEASURED on TradingView, not inferred from the Python
annotations: ``math.round(x)`` is annotated ``PyneFloat`` in the lib but is
int-typed on TradingView, and ``int / int`` is int-typed there while Python
gives a float. Where the two disagree, the measurement wins.
"""
import ast
from typing import Final, TypeVar

#: Any expression node -- ``stamp_lowering`` hands back exactly what it got.
_E = TypeVar('_E', bound=ast.expr)

__all__ = [
    'INT', 'FLOAT', 'BOOL', 'STR', 'COLOR', 'OBJECT', 'VOID', 'UNKNOWN',
    'NUMERIC', 'KNOWN',
    'join', 'binop_type', 'unaryop_type', 'compare_type',
    'annotation_type', 'ANNOTATION_TYPES', 'LIB_TYPE_OVERRIDES',
    'is_int_typed', 'TY_ATTR', 'get_ty', 'set_ty', 'inherit_ty',
    'constant_type', 'stamp_lowering', 'BUILTIN_CALL_TYPES',
]

# --- the stamp ------------------------------------------------------------

#: Attribute a typed expression node carries. The type travels ON the node,
#: not in a side table keyed by position: the later passes reuse the node
#: objects, so the stamp survives the lowering for free. A pass that BUILDS a
#: replacement node has to call ``inherit_ty``.
TY_ATTR: Final = '_pine_ty'


def get_ty(node) -> str:
    """
    Read a node's Pine type.

    :param node: The node to look at
    :return: Its type character, UNKNOWN when it was never typed
    """
    return getattr(node, TY_ATTR, UNKNOWN)


def set_ty(node, ty: str):
    """
    Stamp a node's Pine type.

    :param node: The node to stamp
    :param ty: The type character
    :return: ``node``
    """
    setattr(node, TY_ATTR, ty)
    return node


def inherit_ty(new, old):
    """
    Carry a type stamp onto a freshly built wrapper node.

    A pass that returns the SAME node object keeps the stamp for free; one
    that builds a replacement -- the ``safe_div(a, b)`` wrapper around a
    division, the state subscript the series pass emits -- drops it unless it
    says so here.

    :param new: The node being returned in place of ``old``
    :param old: The node it replaces
    :return: ``new``, stamped
    """
    ty = getattr(old, TY_ATTR, None)
    if ty is not None:
        setattr(new, TY_ATTR, ty)
    return new


# --- the lattice ----------------------------------------------------------

# One character per type so a per-node stamp costs one interned str, and the
# artifact stays compact. The chain is int -> float -> unknown; everything
# else is a flat peer that joins to UNKNOWN with anything but itself.
INT: Final = 'i'
FLOAT: Final = 'f'
BOOL: Final = 'b'
STR: Final = 's'
COLOR: Final = 'c'
#: A known non-scalar: array, matrix, map, drawing, UDT, tuple. Known is the
#: point -- it is NOT a typing failure, it simply carries no numeric algebra.
OBJECT: Final = 'o'
#: A call that returns nothing (``plot()``, ``strategy.entry()``).
VOID: Final = 'v'
#: Untypable. Carries provenance in the inference engine, and is what the
#: ``@pyne edge`` gate rejects.
UNKNOWN: Final = '?'

#: The types the arithmetic algebra is defined over.
NUMERIC: Final = frozenset({INT, FLOAT})
#: Everything the gate accepts -- UNKNOWN is the only failure.
KNOWN: Final = frozenset({INT, FLOAT, BOOL, STR, COLOR, OBJECT, VOID})


def join(left: str, right: str) -> str:
    """
    Least upper bound of two types.

    Used wherever one expression has two possible types: the arms of a
    ternary, the branches feeding a variable, the iterations of a loop.
    ``int`` widens to ``float`` because every int-typed value already IS a
    double; anything else that disagrees is UNKNOWN.

    :param left: One type character
    :param right: The other type character
    :return: The joined type character
    """
    if left == right:
        return left
    if left in NUMERIC and right in NUMERIC:
        return FLOAT
    return UNKNOWN


# --- operators ------------------------------------------------------------

#: Arithmetic operators, all of which follow the same int/float algebra.
_ARITHMETIC: Final = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod,
                      ast.FloorDiv, ast.Pow)


def binop_type(op: ast.operator, left: str, right: str) -> str:
    """
    Type of a binary arithmetic expression.

    MEASURED on TradingView (FX:EURUSD@60, ``d = (R + z) / 8``, int-typed with
    the value 1.75): ``int op int -> int`` for EVERY arithmetic operator, the
    division and the modulo INCLUDED -- ``d / 2`` and ``d % 2`` are both
    int-typed, and ``d * 100`` plots 175 rather than 1. A float operand widens
    the result, exactly as in the classic numeric tower.

    ``+`` over two strings is string concatenation; that is the only
    non-numeric arithmetic Pine has.

    :param op: The operator node
    :param left: Type of the left operand
    :param right: Type of the right operand
    :return: Type of the result
    """
    if isinstance(op, ast.Add) and left == STR and right == STR:
        return STR
    if isinstance(op, _ARITHMETIC):
        if left in NUMERIC and right in NUMERIC:
            return INT if left == INT and right == INT else FLOAT
        if UNKNOWN in (left, right):
            return UNKNOWN
        # A bool operand is a Pine type error, not a silent widening
        return UNKNOWN
    # Bit operators do not exist in Pine
    return UNKNOWN


def unaryop_type(op: ast.unaryop, operand: str) -> str:
    """
    Type of a unary expression.

    MEASURED: ``-d`` on an int-typed 1.75 stays int-typed (and plots -1.75),
    so the unary sign is type-preserving. ``not`` is always bool.

    :param op: The operator node
    :param operand: Type of the operand
    :return: Type of the result
    """
    if isinstance(op, ast.Not):
        return BOOL
    if isinstance(op, (ast.UAdd, ast.USub)):
        return operand if operand in NUMERIC else UNKNOWN
    return UNKNOWN


def compare_type(_left: str, _right: str) -> str:
    """
    Type of a comparison. Always bool in Pine, whatever the operands are.

    :param _left: Type of the left operand, unused
    :param _right: Type of the right operand, unused
    :return: ``BOOL``
    """
    return BOOL


def constant_type(value: object) -> str:
    """
    Pine type of a Python literal.

    :param value: The literal's value
    :return: Its type character
    """
    if value is None:
        return VOID
    if isinstance(value, bool):
        return BOOL
    if isinstance(value, int):
        return INT
    if isinstance(value, float):
        return FLOAT
    if isinstance(value, str):
        return STR
    return UNKNOWN


#: Python builtins a Pyne script calls directly, with the Pine type of the
#: result. The Pine casts of the same name reach the inference as ``lib.int``
#: and friends (see ``LIB_TYPE_OVERRIDES``); these are the bare spellings that
#: survive import normalization. ``'arg0'`` means the first argument's type.
BUILTIN_CALL_TYPES: Final[dict[str, object]] = {
    'int': INT,
    'float': FLOAT,
    'bool': BOOL,
    'str': STR,
    'len': INT,
    'abs': 'arg0',
    # Python's own arity split, and the same one TradingView measures for
    # ``math.round``: without a precision the result is an integer
    'round': {1: INT, 2: FLOAT},
}


def stamp_lowering(root: _E, result: str) -> _E:
    """
    Type a subtree a lowering pass has just built.

    The Pine types are upward-closed over expressions: an unstamped node may
    never CONTAIN a stamped one, or the type of a value is lost exactly where
    its consumers -- the overload pin, the AOT front end -- look for it. A pass
    that wraps typed operands in fresh plumbing therefore has to type the
    plumbing too, and the plumbing's types are mechanical: the tests it emits
    are bools, the references it emits are objects, and the arithmetic it emits
    follows the ordinary algebra over its operands.

    Only the ROOT carries what this cannot derive -- what the rewritten
    expression means -- so the caller passes it in. A node that already carries
    a stamp is a preserved operand and is left untouched, subtree and all.

    :param root: The freshly built expression
    :param result: The Pine type of the whole rewritten expression
    :return: ``root``, stamped
    """
    for child in ast.iter_child_nodes(root):
        if isinstance(child, ast.expr):
            _stamp_emitted(child)
    return set_ty(root, result)


def _stamp_emitted(node: ast.expr) -> str:
    """
    Type one emitted node from its children up.

    :param node: The node to type
    :return: Its type character
    """
    existing = getattr(node, TY_ATTR, None)
    if existing is not None:
        return existing
    if isinstance(node, ast.Constant):
        literal = constant_type(node.value)
        set_ty(node, literal)
        return literal
    if isinstance(node, ast.BinOp):
        ty = binop_type(node.op, _stamp_emitted(node.left), _stamp_emitted(node.right))
    elif isinstance(node, ast.UnaryOp):
        ty = unaryop_type(node.op, _stamp_emitted(node.operand))
    elif isinstance(node, ast.NamedExpr):
        ty = _stamp_emitted(node.value)
    elif isinstance(node, ast.IfExp):
        _stamp_emitted(node.test)
        ty = join(_stamp_emitted(node.body), _stamp_emitted(node.orelse))
    elif isinstance(node, (ast.Compare, ast.BoolOp)):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.expr):
                _stamp_emitted(child)
        ty = BOOL
    else:
        # A Name, an Attribute, a Subscript, a Call: the machinery's own
        # references -- a state tuple, a bound dispatcher, a helper function.
        # Known, and carrying no numeric algebra of their own.
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.expr):
                _stamp_emitted(child)
        ty = OBJECT
    set_ty(node, ty)
    return ty


# --- annotations ----------------------------------------------------------

#: Names that map straight onto a lattice type wherever they appear in an
#: annotation. The ``Pyne*`` aliases are unions of the bare type, its ``NA``
#: and its ``Series`` (see ``types/pine_types.py``), so all three spellings of
#: one Pine type land on the same character.
ANNOTATION_TYPES: Final[dict[str, str]] = {
    'int': INT, 'PyneInt': INT,
    'float': FLOAT, 'PyneFloat': FLOAT,
    'bool': BOOL, 'PyneBool': BOOL,
    'str': STR, 'PyneStr': STR,
    'Color': COLOR,
    'None': VOID,
}

#: Annotation heads whose payload is the element type: ``Series[int]``,
#: ``NA[float]``, ``Persistent[int]``, ``Optional[int]``.
_TRANSPARENT_SUBSCRIPTS: Final = frozenset({
    'Series', 'PersistentSeries', 'NA', 'Persistent', 'IBPersistent',
    'IBPersistentSeries', 'Optional',
})

#: Annotation heads that denote a known non-scalar.
_OBJECT_SUBSCRIPTS: Final = frozenset({'list', 'tuple', 'dict', 'set', 'Array', 'Matrix'})

#: Concrete lib classes: known objects, not typing failures.
_OBJECT_NAMES: Final = frozenset({
    'Array', 'Matrix', 'Map', 'Line', 'LineFill', 'Label', 'Box', 'Table',
    'Polyline', 'HLine', 'ChartPoint', 'VolumeRow', 'Currency', 'DayOfWeek',
    'Extend', 'Footprint', 'Position', 'Size', 'Format', 'AlignEnum',
    'FontFamilyEnum', 'FormatEnum', 'list', 'tuple', 'dict', 'set',
})


def annotation_type(node: ast.expr | None) -> str:
    """
    Map a Python annotation onto a Pine type character.

    Unions are joined, so ``int | NA`` is ``int`` and ``float | int`` is
    ``float``; the ``Series``/``NA``/``Persistent`` wrappers are transparent,
    since they change the storage, not the Pine type. An annotation this does
    not recognize is UNKNOWN, which is what makes a missing annotation and an
    exotic one behave alike.

    :param node: The annotation expression, or None when there is none
    :return: The type character
    """
    if node is None:
        return UNKNOWN

    if isinstance(node, ast.Constant):
        if node.value is None:
            return VOID
        # A stringized forward reference: ``'_ExitOrderKey'``
        if isinstance(node.value, str):
            return _named_type(node.value)
        return UNKNOWN

    if isinstance(node, ast.Name):
        return _named_type(node.id)

    if isinstance(node, ast.Attribute):
        return _named_type(node.attr)

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _union_type(node)

    if isinstance(node, ast.Subscript):
        head = node.value
        name = head.id if isinstance(head, ast.Name) else \
            (head.attr if isinstance(head, ast.Attribute) else '')
        if name in _TRANSPARENT_SUBSCRIPTS:
            return annotation_type(node.slice)
        if name in _OBJECT_SUBSCRIPTS:
            return OBJECT
        return UNKNOWN

    return UNKNOWN


def _named_type(name: str) -> str:
    """Resolve a bare annotation name to a type character."""
    if name in ANNOTATION_TYPES:
        return ANNOTATION_TYPES[name]
    if name in _OBJECT_NAMES:
        return OBJECT
    return UNKNOWN


def _is_typeless(node: ast.expr) -> bool:
    """
    Whether a union member carries no type of its own.

    A bare ``NA`` and a ``None`` say "this may be absent", not "this is of
    some other type", so they must not drag ``int | NA`` or ``int | None``
    away from int. A genuinely conflicting member still must -- which is why
    this is a test on the SPELLING and not on the resulting UNKNOWN: treating
    every unknown member as absent turned ``int | float | str | bool | NA``
    into bool by letting the later members overwrite the conflict.
    """
    if isinstance(node, ast.Constant) and node.value is None:
        return True
    if isinstance(node, ast.Name):
        return node.id == 'NA'
    if isinstance(node, ast.Attribute):
        return node.attr == 'NA'
    return False


def _union_type(node: ast.BinOp) -> str:
    """
    Type of an annotation union, ignoring its absence markers.

    :param node: The ``X | Y`` annotation node
    :return: The joined type character
    """
    members: list[ast.expr] = []

    def flatten(expr: ast.expr) -> None:
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.BitOr):
            flatten(expr.left)
            flatten(expr.right)
        else:
            members.append(expr)

    flatten(node)
    contributing = [m for m in members if not _is_typeless(m)]
    if not contributing:
        # ``NA | None`` and friends: absence only, no type to speak of
        return UNKNOWN

    result = annotation_type(contributing[0])
    for member in contributing[1:]:
        result = join(result, annotation_type(member))
    return result


def is_int_typed(ty: str) -> bool:
    """
    Whether a type is the one that changes behavior.

    :param ty: A type character
    :return: True for ``int``
    """
    return ty == INT


# --- measured overrides ---------------------------------------------------

#: Lib functions whose TradingView type is NOT what the Python annotation
#: says, keyed by the dotted lib path. A value is either a type character
#: (fixed result), the string ``'arg0'``..``'arg9'`` (echoes that argument's
#: type), ``'all_int'`` (int when EVERY argument is int, float otherwise), or
#: a dict mapping an argument COUNT to any of those (an arity split).
#:
#: MEASURED on TradingView (FX:EURUSD@60, ``d = (R + z) / 8`` int-typed 1.75):
#: ``math.max(d, 1)`` is int while ``math.max(d, 1.0)`` is float,
#: ``math.abs(d)`` is int, ``math.round(d)`` is int but ``math.round(d, 2)``
#: is float, ``math.floor(d)`` and ``math.ceil(d)`` are int, and
#: ``math.sqrt(d)`` is float.
LIB_TYPE_OVERRIDES: Final[dict[str, object]] = {
    'math.max': 'all_int',
    'math.min': 'all_int',
    'math.abs': 'arg0',
    'math.sum': 'arg0',
    'math.avg': 'all_int',
    'math.sign': 'arg0',
    # The arity split is the whole reason ``math.round`` needed a fix: with no
    # precision it is the int overload, with one it is the float overload
    'math.round': {1: INT, 2: FLOAT},
    'math.floor': INT,
    'math.ceil': INT,
    'math.sqrt': FLOAT,
    'math.pow': FLOAT,
    'math.log': FLOAT,
    'math.log10': FLOAT,
    'math.exp': FLOAT,
    'math.sin': FLOAT,
    'math.cos': FLOAT,
    'math.tan': FLOAT,
    'math.asin': FLOAT,
    'math.acos': FLOAT,
    'math.atan': FLOAT,
    'math.todegrees': FLOAT,
    'math.toradians': FLOAT,
    'math.random': FLOAT,
    'math.round_to_mintick': FLOAT,
    # ``nz`` and the history index are type-preserving: ``nz(d)`` and ``d[1]``
    # are both int-typed on TradingView
    'nz': 'arg0',
    'na': BOOL,
    'fixnan': 'arg0',
    # The casts are the only place the TYPE and the VALUE move together
    'int': INT,
    'float': FLOAT,
    'bool': BOOL,
    'string': STR,
    'color': COLOR,
    # Inputs are typed by the constructor that made them
    'input.int': INT,
    'input.float': FLOAT,
    'input.bool': BOOL,
    'input.string': STR,
    'input.text_area': STR,
    'input.color': COLOR,
    'input.symbol': STR,
    'input.timeframe': STR,
    'input.session': STR,
    'input.source': FLOAT,
    'input.price': FLOAT,
    'input.time': INT,
    'input.enum': OBJECT,
}
