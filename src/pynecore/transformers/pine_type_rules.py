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
from typing import Final, NamedTuple, TypeVar

#: Any expression node -- ``stamp_lowering`` hands back exactly what it got.
_E = TypeVar('_E', bound=ast.expr)

__all__ = [
    'INT', 'FLOAT', 'BOOL', 'STR', 'COLOR', 'OBJECT', 'VOID', 'UNKNOWN',
    'NUMERIC', 'KNOWN',
    'join', 'binop_type', 'unaryop_type', 'compare_type',
    'annotation_type', 'ANNOTATION_TYPES', 'LIB_TYPE_OVERRIDES',
    'is_int_typed', 'TY_ATTR', 'get_ty', 'set_ty', 'inherit_ty',
    'constant_type', 'stamp_lowering', 'BUILTIN_CALL_TYPES',
    'PIN_ATTR', 'PINS_ATTR', 'PINNABLE', 'get_pin', 'set_pin', 'get_pins', 'set_pins',
    'VECTOR_ATTR', 'VARYING_ATTR', 'get_vector', 'set_vector', 'get_varying', 'set_varying',
    'pin_for', 'overload_result', 'ImplSig', 'overload_pick',
    'TYPELESS', 'NONE_DEFAULT', 'FIT_OMISSIBLE', 'FIT_REQUIRED', 'FIT_UNSURE',
    'default_fit', 'annotation_takes_none',
]

# --- the stamp ------------------------------------------------------------

#: Attribute a typed expression node carries. The type travels ON the node,
#: not in a side table keyed by position: the later passes reuse the node
#: objects, so the stamp survives the lowering for free. A pass that BUILDS a
#: replacement node has to call ``inherit_ty``.
TY_ATTR: Final = '_pine_ty'


#: Attribute a call site carries its overload pin under. One type character
#: per positional argument, or absent when the site is not pinnable.
PIN_ATTR: Final = '_pine_pin'

#: Attribute a call site carries its PER-CONTEXT pins under, present only
#: where the contexts a shared body was analysed in disagree.
PINS_ATTR: Final = '_pine_pins'

#: Attribute a call site carries the INSTANCE VECTOR it configures its callee
#: with, present only where every context reaching the site agrees on it.
VECTOR_ATTR: Final = '_pine_vector'

#: Attribute a definition carries its instance-varying inner call sites under,
#: in source order. Their index in this list IS their index in the vector.
VARYING_ATTR: Final = '_pine_varying'


def get_pin(node) -> str | None:
    """
    Read a call site's overload pin.

    :param node: The call node
    :return: The pin string, or None when there is none
    """
    return getattr(node, PIN_ATTR, None)


def set_pin(node, pin: str | None):
    """
    Stamp a call site's overload pin, ``None`` included.

    Writing the None is the point: a body is walked more than once -- the loop
    fixpoint re-walks it, and a generic function is analysed once per call-site
    context -- and a later walk that decides AGAINST the pin has to erase the
    one an earlier walk wrote. Leaving it would emit a pin the inference itself
    no longer stands behind.

    :param node: The call node
    :param pin: One type character per positional argument, or None
    :return: ``node``
    """
    setattr(node, PIN_ATTR, pin)
    return node


def get_pins(node) -> dict[int, str | None] | None:
    """
    Read the per-context pins of a call site the contexts disagree on.

    :param node: The call node
    :return: context id -> pin, or None when one pin holds for every context
    """
    return getattr(node, PINS_ATTR, None)


def set_pins(node, pins: dict[int, str | None] | None):
    """
    Stamp the per-context pins of a call site, ``None`` included.

    A generic function's body is shared by every context it is instantiated
    in, so one call site inside it can justify ``'i'`` under an int caller and
    nothing at all under a float one. That is not a correction to erase -- both
    are true -- so the single pin goes away and this map takes its place, for a
    later pass to hand each instance the character that belongs to it. Writing
    the None matters for the same reason it does for the single pin: a walk
    that brings the contexts back into agreement has to clear the map.

    :param node: The call node
    :param pins: context id -> pin, or None when the contexts agree
    :return: ``node``
    """
    setattr(node, PINS_ATTR, pins)
    return node


def get_vector(node):
    """
    Read the instance vector a call site configures its callee with.

    :param node: The call node
    :return: The vector, or None when the site configures nothing
    """
    return getattr(node, VECTOR_ATTR, None)


def set_vector(node, vector: tuple | None):
    """
    Stamp the instance vector a call site hands its callee.

    :param node: The call node
    :param vector: The nested vector, or None
    :return: ``node``
    """
    setattr(node, VECTOR_ATTR, vector)
    return node


def get_varying(node) -> list | None:
    """
    Read a definition's instance-varying inner call sites, in source order.

    :param node: The function definition node
    :return: The call nodes, or None when the body varies with nothing
    """
    return getattr(node, VARYING_ATTR, None)


def set_varying(node, sites: list | None):
    """
    Stamp a definition's instance-varying inner call sites.

    :param node: The function definition node
    :param sites: The call nodes in source order, or None
    :return: ``node``
    """
    setattr(node, VARYING_ATTR, sites)
    return node


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


#: Names a module calls under their BARE spelling, with the Pine type of the
#: result -- the ones import normalization leaves alone. Two families live
#: here: the Python builtins a hand-written Pyne script uses, and the
#: ``pine_cast`` helpers a compiled script gets its Pine casts as (the lib
#: spellings ``lib.int`` and friends are in ``LIB_TYPE_OVERRIDES`` instead).
#: ``'arg0'`` means the first argument's type.
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
    # ``core/pine_cast.py``: what a Pine cast compiles to. Every compiled
    # script imports the ones it uses by name, so an unlisted cast would drop
    # the type of every value that passes through it
    'cast_int': INT,
    'cast_float': FLOAT,
    'cast_bool': BOOL,
    'cast_string': STR,
    'cast_color': COLOR,
    'cast_label': OBJECT,
    'cast_table': OBJECT,
    'cast_box': OBJECT,
    'cast_line': OBJECT,
    'cast_linefill': OBJECT,
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


#: Types a pin can name. The runtime turns each character back into a WITNESS
#: value and runs the ordinary selector on it, so a character may only appear
#: here when one value stands for the whole type. An object does not qualify:
#: what discriminates two container overloads is the ELEMENT type, and no
#: single witness carries that.
PINNABLE: Final = frozenset({INT, FLOAT, BOOL, STR})


def pin_for(arg_types: list[str]) -> str | None:
    """
    The overload pin a call site's argument types justify.

    A pin is only worth emitting where the static answer can DIFFER from the
    dynamic one, and that is exactly where an int-typed argument is involved:
    ``int / int`` keeps the int TYPE while the value is a Python float, so the
    runtime selector -- which sees values, not types -- widens it to the float
    implementation. Every other shape already agrees, and is left on the
    ordinary path rather than pinned for nothing.

    :param arg_types: Type of each positional argument, in order
    :return: One character per argument, or None when the site is not pinnable
    """
    if not arg_types or any(t not in PINNABLE for t in arg_types):
        return None
    if INT not in arg_types:
        return None
    return ''.join(arg_types)


def overload_result(returns: list[str]) -> str:
    """
    Type of a call to an overload group, from its implementations' returns.

    Pine resolves the overload STATICALLY, so exactly one of these IS the
    call's type -- but which one is a question this pass deliberately does not
    answer, because the selector lives in the runtime and duplicating it is
    what the pin exists to avoid. Where the implementations agree, the answer
    needs no selection. Where they do not, the type is genuinely unknown, and
    joining them would be worse than saying so: the join is a GUESS, and a
    guess that reaches an enclosing pin makes that pin select an
    implementation neither TradingView nor the value-driven dispatch would.

    :param returns: Each implementation's return type, in declaration order
    :return: The agreed type, or UNKNOWN
    """
    if not returns:
        return UNKNOWN
    first = returns[0]
    return first if all(ret == first for ret in returns[1:]) else UNKNOWN


#: Type character of a default that carries no type of its own: ``na`` and the
#: ``__dyn_default__`` sentinel the dynamic-default pass leaves behind. The
#: selector's exact pass takes such a default for whatever the parameter
#: declares -- MEASURED against ``core/overload.py::_check_type``, a typeless
#: ``na`` answers every annotation (``NA.type is None`` returns True outright),
#: and the sentinel is skipped before the check runs at all.
TYPELESS: Final = '*'

#: Type character of a literal ``None`` default. It is NOT typeless: the
#: selector type-checks it like any other bound value, and ``_check_type``
#: accepts ``None`` only where the annotation has a ``None`` member of its own
#: (``int | None``, ``Optional[int]``), or is ``Any``/``object``/``NoneType``.
#: Against a plain ``int`` it is REJECTED, which takes the implementation out
#: of the exact pass -- so the fit is decided against the annotation's
#: None-acceptance, not against its Pine type character.
NONE_DEFAULT: Final = '0'

#: What one parameter contributes to a call that OMITS it: its default
#: satisfies its own declared type, it has no usable default at all (none, or
#: one the exact pass would reject), or its default cannot be typed here.
FIT_OMISSIBLE: Final = 'y'
FIT_REQUIRED: Final = 'n'
FIT_UNSURE: Final = '?'

#: How one implementation answers a pin in one pass of the selector.
_NO: Final = 0
_MAYBE: Final = 1
_YES: Final = 2


def default_fit(declared: str, default: str | None, takes_none: bool) -> str:
    """
    What a parameter contributes to a call that leaves it to its default.

    The runtime binds the omitted arguments and type-checks the defaults with
    everything else, so a defaulted parameter is not simply absent from the
    match -- it is one more type the exact pass has to accept.

    :param declared: The parameter's declared type character
    :param default: The default's type character, ``TYPELESS`` for one that
                    carries no type, ``NONE_DEFAULT`` for a literal ``None``,
                    or None when the parameter has no default
    :param takes_none: Whether the parameter's FULL annotation accepts a
                       ``None`` value, which is what decides a ``None`` default
    :return: One of the ``FIT_*`` characters
    """
    if default is None:
        return FIT_REQUIRED
    if default == NONE_DEFAULT:
        return FIT_OMISSIBLE if takes_none else FIT_REQUIRED
    if default == TYPELESS:
        return FIT_OMISSIBLE
    if default == UNKNOWN or declared == UNKNOWN:
        return FIT_UNSURE
    return FIT_OMISSIBLE if default == declared else FIT_REQUIRED


#: Annotation names that accept ``None`` whatever they wrap.
_NONE_TAKING_NAMES: Final = frozenset({'Any', 'object', 'NoneType'})

#: Annotation heads that add a ``None`` member to whatever they wrap.
_OPTIONAL_SUBSCRIPTS: Final = frozenset({'Optional'})


def annotation_takes_none(node: ast.expr | None) -> bool:
    """
    Whether a parameter's annotation accepts a ``None`` argument.

    This is not a question about the Pine type: ``int | None`` and ``int`` are
    both int-typed here, yet only the first one takes the ``None`` a default
    binds into it. What is mirrored is
    ``core/overload.py::_check_type(None, hint, strict=True)``, which answers
    True for ``Any``, ``object``, ``NoneType``, and for a union with a ``None``
    member -- and False for everything else, ``NA[int]`` and ``Series[float]``
    included, since neither is an instance check ``None`` passes.

    A MISSING annotation takes None too: the runtime reads a parameter with no
    hint as ``Any``.

    :param node: The annotation expression, or None when there is none
    :return: True when a ``None`` default satisfies the annotation
    """
    if node is None:
        return True

    if isinstance(node, ast.Constant):
        if node.value is None:
            return True
        if not isinstance(node.value, str):
            return False
        # A stringized forward reference: the runtime resolves it before
        # checking, so ``'int | None'`` takes None as much as the bare
        # spelling does
        try:
            return annotation_takes_none(ast.parse(node.value, mode='eval').body)
        except SyntaxError:
            return False

    if isinstance(node, ast.Name):
        return node.id in _NONE_TAKING_NAMES

    if isinstance(node, ast.Attribute):
        return node.attr in _NONE_TAKING_NAMES

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return annotation_takes_none(node.left) or annotation_takes_none(node.right)

    if isinstance(node, ast.Subscript):
        head = node.value
        name = head.id if isinstance(head, ast.Name) else \
            (head.attr if isinstance(head, ast.Attribute) else '')
        if name in _OPTIONAL_SUBSCRIPTS:
            return True
        if name == 'Union':
            members = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            return any(annotation_takes_none(member) for member in members)
        return False

    return False


class ImplSig(NamedTuple):
    """One overload implementation's shape, as the static selection reads it."""
    #: Type character of each positional parameter, in declaration order
    params: tuple[str, ...]
    #: How many of them a call has to pass
    required: int
    #: Whether a ``*args`` lets it take any number of them
    open_ended: bool
    #: What the implementation returns
    ret: str
    #: One ``FIT_*`` character per parameter -- the positional ones first, in
    #: ``params`` order, then the keyword-only ones. Its LENGTH is what says
    #: whether the implementation has keyword-only parameters at all.
    fits: str


def overload_pick(impls: list[ImplSig], pin: str) -> str | None:
    """
    Return type of the implementation a PINNED call site selects.

    The runtime resolves a pin by turning its characters back into witness
    values and running the ordinary selector, whose first pass is EXACT -- no
    int-where-a-float-is-wanted widening. This is that first pass and nothing
    else, both halves of it: the selector tries a positional match on the
    implementations whose arity is exactly the call's, and only then binds the
    arguments to the parameters and type-checks the DEFAULTS along with them.
    Reading the first half alone left ``h(1)`` unanswered where the group
    spells ``h(x: int, y: int = 0)``, which the runtime settles outright.

    Anything the exact pass does not settle is left to ``overload_result``,
    because a guess that reaches an enclosing pin selects an implementation
    neither TradingView nor the value-driven dispatch would.

    Declaration order decides between implementations that match equally well
    WITHIN one half, so an earlier one that could take the witnesses too (an
    unannotated parameter takes anything) makes the site unanswerable rather
    than overruling the runtime's own choice.

    :param impls: Every implementation of the group, in declaration order
    :param pin: The call site's pin, one character per positional argument
    :return: The selected implementation's return type, or None
    """
    wanted = tuple(pin)
    for exact_arity in (True, False):
        for impl in impls:
            verdict = _takes(impl, wanted, exact_arity)
            if verdict == _NO:
                continue
            return impl.ret if verdict == _YES else None
    return None


def _takes(impl: ImplSig, wanted: tuple[str, ...], exact_arity: bool) -> int:
    """
    How one implementation answers a pin in one half of the exact pass.

    :param impl: The implementation's shape
    :param wanted: The pin, one character per positional argument
    :param exact_arity: The positional half, which only reaches an
                        implementation whose every parameter the call fills
    :return: ``_YES``, ``_MAYBE`` or ``_NO``
    """
    if impl.open_ended:
        return _MAYBE
    if exact_arity:
        # The positional half counts EVERY visible parameter against the
        # argument count, so a keyword-only one puts the implementation out
        # of its reach however it is defaulted
        if len(impl.fits) != len(wanted) or len(impl.params) != len(wanted):
            return _NO
    elif not impl.required <= len(wanted) <= len(impl.params):
        return _NO
    verdict = _YES
    for declared, want in zip(impl.params, wanted):
        if declared == UNKNOWN:
            verdict = _MAYBE
        elif declared != want:
            return _NO
    if exact_arity:
        return verdict
    for fit in impl.fits[len(wanted):]:
        if fit == FIT_REQUIRED:
            return _NO
        if fit == FIT_UNSURE:
            verdict = _MAYBE
    return verdict


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
