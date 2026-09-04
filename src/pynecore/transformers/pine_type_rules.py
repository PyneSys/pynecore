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
from collections.abc import Callable, Mapping, Sequence
from typing import Final, NamedTuple, TypeVar

#: Any expression node -- ``stamp_lowering`` hands back exactly what it got.
_E = TypeVar('_E', bound=ast.expr)

__all__ = [
    'INT', 'FLOAT', 'BOOL', 'STR', 'COLOR', 'OBJECT', 'PINE_LOOP', 'VOID', 'UNKNOWN',
    'NUMERIC', 'KNOWN', 'SCALARS',
    'LIB_MODULE', 'CLASS_SEP', 'class_id', 'builtin_class_id', 'object_ty',
    'array_of', 'matrix_of', 'map_of', 'tuple_of', 'head', 'is_shaped', 'is_array',
    'is_matrix', 'is_map', 'is_tuple', 'arity', 'elements_of', 'class_of',
    'element_of', 'key_of', 'value_of', 'shape_mismatch', 'shape_conflict', 'render_ty',
    'BUILTIN_CLASSES', 'BUILTIN_NAMESPACES', 'namespace_of',
    'join', 'binop_type', 'unaryop_type', 'compare_type',
    'annotation_type', 'bare_wrapper', 'ANNOTATION_TYPES', 'LIB_TYPE_OVERRIDES',
    'OVERRIDE_PARAM_NAMES',
    'is_int_typed', 'TY_ATTR', 'get_ty', 'set_ty', 'inherit_ty',
    'constant_type', 'stamp_lowering', 'BUILTIN_CALL_TYPES', 'BUILTIN_NAME_TYPES',
    'PIN_ATTR', 'PINS_ATTR', 'PINNABLE', 'get_pin', 'set_pin', 'get_pins', 'set_pins',
    'VECTOR_ATTR', 'VARYING_ATTR', 'get_vector', 'set_vector', 'get_varying', 'set_varying',
    'pin_for', 'overload_result', 'ImplSig', 'overload_pick',
    'FactoryFields',
    'TYPELESS', 'NONE_DEFAULT', 'FIT_OMISSIBLE', 'FIT_REQUIRED', 'FIT_UNSURE',
    'default_fit', 'annotation_takes_none', 'impl_sig', 'param_fit', 'default_ty',
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

#: Pine's ``na``: a value that carries no type of its own and takes the one of
#: whatever it meets. It is what an untyped ``na`` literal reads as, and what
#: a default that carries no type is -- the selector's exact pass takes such a
#: default for whatever the parameter declares (MEASURED against
#: ``core/overload.py::_check_type``: a typeless ``na`` answers every
#: annotation, ``NA.type is None`` returns True outright, and the
#: ``__dyn_default__`` sentinel is skipped before the check runs at all).
TYPELESS: Final = '*'

#: The types the arithmetic algebra is defined over.
NUMERIC: Final = frozenset({INT, FLOAT})
#: Everything the gate accepts -- UNKNOWN is the only failure, and a typeless
#: ``na`` is a value the language has rather than a type it could not find.
KNOWN: Final = frozenset({INT, FLOAT, BOOL, STR, COLOR, OBJECT, VOID, TYPELESS})
#: The types a Pine value can be spelled as a single character: the scalars a
#: map key is allowed to be, and the only tails a shape ever ends in.
SCALARS: Final = frozenset({INT, FLOAT, BOOL, STR, COLOR})


# --- shaped types ---------------------------------------------------------

# Pine's types are fully static and fully spelled: an object KNOWS its class,
# an ``array<int>`` is a different type from an ``array<float>``, and a
# ``map<string, array<float>>`` holds float arrays. A single character cannot
# say any of that, so the representation stays a ``str`` and grows a grammar:
#
#     ty  := <char> | 'o:' <class-id> | 'a:' <ty> | 'm:' <ty>
#          | 'M:' <key-char> ':' <ty> | 'T:' <item>+
#     item := <decimal length> ':' <ty>
#
# array, matrix and map, in that order; a map key is a Pine scalar so one
# character is enough for it and the TAIL is the value type, which makes the
# grammar unambiguous read left to right. A bare ``'o'`` stays what it always
# was -- an object whose class was lost -- and is what the diagnostics point at.
#
# The tuple is the one form holding SEVERAL types in a row, and none of the
# others is self-delimiting inside a sequence: a class id ends in a module key
# that is a filesystem path, which may hold any character a path may hold --
# commas, colons, spaces, even the ``#`` the id itself is spelled with. So an
# item carries its own LENGTH, and a reader takes exactly that many characters
# and stops. Nothing has to be escaped, nothing is ambiguous, and a shape of
# any depth round-trips through ``tuple_of``/``elements_of`` unchanged.

#: Module key the classes the LIB publishes are identified by. A real module
#: key is an absolute source path, so this word can never collide with one.
LIB_MODULE: Final = 'lib'

#: What separates a class's module key from its name in a class id.
CLASS_SEP: Final = '#'

_OBJ: Final = 'o:'
_ARRAY: Final = 'a:'
_MATRIX: Final = 'm:'
_MAP: Final = 'M:'
_TUPLE: Final = 'T:'


def class_id(module_key: str, name: str) -> str:
    """
    The identity of one class, which is (module, name) and never the bare name.

    :param module_key: Resolved source path of the module declaring it
    :param name: The class name
    :return: The class id
    """
    return f'{module_key}{CLASS_SEP}{name}'


def builtin_class_id(name: str) -> str:
    """
    The identity of a class the lib publishes (``Line``, ``Label``, ...).

    :param name: The class name
    :return: The class id
    """
    return f'{LIB_MODULE}{CLASS_SEP}{name}'


def object_ty(cid: str) -> str:
    """
    The type of an instance of one class.

    :param cid: The class id
    :return: The shaped type
    """
    return _OBJ + cid


#: The counter object a compiled ``for`` loop iterates: ``core/pine_range``'s
#: ``PineLoop``, whose ``value`` is the counter and whose ``step(to)`` advances it
PINE_LOOP: Final = object_ty(builtin_class_id('PineLoop'))


def array_of(element: str) -> str:
    """
    The type of an array over one element type.

    A shape whose element is UNKNOWN collapses to a bare object: keeping it
    would claim a shape while carrying no information, and two such arrays
    would then read as a shape MISMATCH rather than as the one thing they
    honestly are -- containers nothing is known about.

    :param element: The element type
    :return: The shaped type, or OBJECT when the element is not known
    """
    return OBJECT if element in (UNKNOWN, VOID) else _ARRAY + element


def matrix_of(element: str) -> str:
    """
    The type of a matrix over one element type.

    :param element: The element type
    :return: The shaped type, or OBJECT when the element is not known
    """
    return OBJECT if element in (UNKNOWN, VOID) else _MATRIX + element


def map_of(key: str, value: str) -> str:
    """
    The type of a map from one scalar key type to one value type.

    :param key: The key type, which Pine requires to be a scalar
    :param value: The value type
    :return: The shaped type, or OBJECT when either half is not known
    """
    if key not in SCALARS or value in (UNKNOWN, VOID):
        return OBJECT
    return f'{_MAP}{key}:{value}'


def tuple_of(elements: Sequence[str]) -> str:
    """
    The type of a Pine tuple over its element types, in order.

    Every element is length-prefixed, so an element of any shape -- a class id
    whose module key is a path full of punctuation included -- survives being
    put next to another one. An element type is kept exactly as it is: an
    UNKNOWN one only makes that POSITION unknown, and collapsing the whole
    tuple for it would take the types of the other positions down with it,
    which is the opposite of what an unpack needs. A typeless element stays
    typeless for the same reason -- ``[na, na]`` takes its types from the
    branch it meets.

    :param elements: The element types, in order
    :return: The shaped type, or OBJECT when there are no elements
    """
    if not elements:
        return OBJECT
    return _TUPLE + ''.join(f'{len(element)}:{element}' for element in elements)


def head(ty: str) -> str:
    """
    The lattice character a type behaves as.

    Every shape is an object, so this is what the char-based rules -- the
    arithmetic, the pin, the overload selection -- read. It is also why the
    pin wire format is untouched by shapes: ``pin_for(['a:i', 'i'])`` and
    ``pin_for(['o', 'i'])`` are the same question.

    :param ty: Any type
    :return: Its single-character head
    """
    return ty if len(ty) == 1 else OBJECT


def is_shaped(ty: str) -> bool:
    """
    Whether a type carries more than its lattice character.

    :param ty: Any type
    :return: True for a class, an array, a matrix or a map
    """
    return len(ty) > 1


def class_of(ty: str) -> str | None:
    """
    The class id an object type names.

    :param ty: Any type
    :return: The class id, or None when the type is not a classed object
    """
    return ty[len(_OBJ):] if ty.startswith(_OBJ) else None


def is_array(ty: str) -> bool:
    """
    Whether a type is an array of a known element type.

    :param ty: Any type
    :return: True for ``a:<ty>``
    """
    return ty.startswith(_ARRAY)


def is_matrix(ty: str) -> bool:
    """
    Whether a type is a matrix of a known element type.

    :param ty: Any type
    :return: True for ``m:<ty>``
    """
    return ty.startswith(_MATRIX)


def is_map(ty: str) -> bool:
    """
    Whether a type is a map of known key and value types.

    :param ty: Any type
    :return: True for ``M:<key>:<ty>``
    """
    return ty.startswith(_MAP)


def is_tuple(ty: str) -> bool:
    """
    Whether a type is a Pine tuple.

    :param ty: Any type
    :return: True for ``T:<item>+``
    """
    return ty.startswith(_TUPLE)


def elements_of(ty: str) -> tuple[str, ...]:
    """
    The element types of a tuple, in order.

    The encoding is read the way it was written: a decimal length, a colon,
    and exactly that many characters. Anything that does not read back that
    way is not a tuple this module produced, and has no elements to give.

    :param ty: Any type
    :return: The element types, empty when the type is not a tuple
    """
    if not ty.startswith(_TUPLE):
        return ()
    out: list[str] = []
    index = len(_TUPLE)
    while index < len(ty):
        colon = ty.find(':', index)
        digits = ty[index:colon]
        if colon < 0 or not digits.isdigit():
            return ()
        size = int(digits)
        start = colon + 1
        if start + size > len(ty):
            return ()
        out.append(ty[start:start + size])
        index = start + size
    return tuple(out)


def arity(ty: str) -> int:
    """
    How many elements a tuple holds.

    :param ty: Any type
    :return: The element count, 0 when the type is not a tuple
    """
    return len(elements_of(ty))


def element_of(ty: str) -> str:
    """
    The element type of an array or a matrix.

    :param ty: Any type
    :return: The element type, UNKNOWN when the type is neither
    """
    if ty.startswith(_ARRAY) or ty.startswith(_MATRIX):
        return ty[2:]
    return UNKNOWN


def key_of(ty: str) -> str:
    """
    The key type of a map.

    :param ty: Any type
    :return: The key character, UNKNOWN when the type is not a map
    """
    return ty[len(_MAP)] if ty.startswith(_MAP) else UNKNOWN


def value_of(ty: str) -> str:
    """
    The value type of a map.

    :param ty: Any type
    :return: The value type, UNKNOWN when the type is not a map
    """
    return ty[len(_MAP) + 2:] if ty.startswith(_MAP) else UNKNOWN


def shape_mismatch(left: str, right: str) -> bool:
    """
    Whether two types are objects of DIFFERENT shape.

    Pine rejects a variable whose branches produce two different types, so
    this is the case worth a diagnostic of its own: the join is UNKNOWN
    because the program is wrong, not because anything was untypable. A bare
    object counts as different from a shape -- the class was lost somewhere,
    and that is the thing to point at.

    A tuple is the one shape that also disagrees with the SCALARS: ``[float,
    int]`` and ``float`` are two types, and a function returning one on one
    path and the other on another is rejected the same way.

    :param left: One type
    :param right: The other type
    :return: True when the two are types that cannot both be right here
    """
    if left == right or left in (UNKNOWN, TYPELESS) or right in (UNKNOWN, TYPELESS):
        return False
    if is_tuple(left) or is_tuple(right):
        return True
    return head(left) == OBJECT and head(right) == OBJECT


def shape_conflict(left: str, right: str) -> tuple[str, str] | None:
    """
    The pair of types that keeps two types from being joined, if any.

    Two tuples of one arity join position by position, so their conflict --
    when they have one -- sits at a POSITION: ``[array<int>, int]`` against
    ``[array<float>, int]`` disagrees in the first element and nowhere else,
    and the join keeps the second position while the first is unknown. That
    partial answer is still a Pine error, and the element pair is the thing
    worth pointing at. Two tuples of different arity, or a tuple against a
    scalar, conflict as wholes.

    :param left: One type
    :param right: The other type
    :return: The two types that disagree, or None when the join is clean
    """
    if is_tuple(left) and is_tuple(right):
        items, others = elements_of(left), elements_of(right)
        if len(items) != len(others):
            return left, right
        for item, other in zip(items, others):
            found = shape_conflict(item, other)
            if found is not None:
                return found
        return None
    return (left, right) if shape_mismatch(left, right) else None


#: How each type is spelled in a message: Pine's own spelling, not the
#: character.
_RENDERED: Final[dict[str, str]] = {
    INT: 'int', FLOAT: 'float', BOOL: 'bool', STR: 'string', COLOR: 'color',
    OBJECT: 'object', VOID: 'void', UNKNOWN: 'unknown', TYPELESS: 'na',
}


def render_ty(ty: str) -> str:
    """
    A type in Pine's own spelling, for a message a user reads.

    ``'o:/lib/zigzag.py#Pivot'`` renders as ``Pivot``, ``'a:i'`` as
    ``array<int>``, ``'M:s:a:f'`` as ``map<string, array<float>>`` and a tuple
    as ``[float, int]``, which is how Pine spells one.

    :param ty: Any type
    :return: Its Pine spelling
    """
    cid = class_of(ty)
    if cid is not None:
        return cid.rpartition(CLASS_SEP)[2]
    if ty.startswith(_ARRAY):
        return f'array<{render_ty(element_of(ty))}>'
    if ty.startswith(_MATRIX):
        return f'matrix<{render_ty(element_of(ty))}>'
    if ty.startswith(_MAP):
        return f'map<{render_ty(key_of(ty))}, {render_ty(value_of(ty))}>'
    if is_tuple(ty):
        return f"[{', '.join(render_ty(element) for element in elements_of(ty))}]"
    return _RENDERED.get(ty, ty)


#: The classes the lib publishes as Pine objects, by the name an annotation
#: spells them with. They are the one family whose module key is not a path --
#: a script names ``Line`` without importing anything.
BUILTIN_CLASSES: Final[dict[str, str]] = {
    name: object_ty(builtin_class_id(name)) for name in (
        'Line', 'LineFill', 'Label', 'Box', 'Table', 'Polyline', 'HLine',
        'ChartPoint', 'VolumeRow', 'Currency', 'DayOfWeek', 'Extend',
        'Footprint', 'Position', 'Size', 'Format', 'AlignEnum',
        'FontFamilyEnum', 'FormatEnum',
    )
}

#: The lib namespace a builtin object's methods live in, by class name. This
#: is what lets ``method_call('get_top', box)`` resolve to ``box.get_top``
#: once the receiver's class is known.
BUILTIN_NAMESPACES: Final[dict[str, str]] = {
    'Line': 'line', 'LineFill': 'linefill', 'Label': 'label', 'Box': 'box',
    'Table': 'table', 'Polyline': 'polyline',
}


def namespace_of(ty: str) -> str | None:
    """
    The lib namespace a shaped receiver's methods are looked up in.

    :param ty: The receiver's type
    :return: The namespace, or None when the shape names none
    """
    if ty.startswith(_ARRAY):
        return 'array'
    if ty.startswith(_MATRIX):
        return 'matrix'
    if ty.startswith(_MAP):
        return 'map'
    cid = class_of(ty)
    if cid is None:
        return None
    module, _, name = cid.rpartition(CLASS_SEP)
    return BUILTIN_NAMESPACES.get(name) if module == LIB_MODULE else None


def join(left: str, right: str) -> str:
    """
    Least upper bound of two types.

    Used wherever one expression has two possible types: the arms of a
    ternary, the branches feeding a variable, the iterations of a loop.
    ``int`` widens to ``float`` because every int-typed value already IS a
    double; anything else that disagrees is UNKNOWN -- two different shapes
    included, since Pine rejects such a variable outright and a silent
    widening to "some object" would hide exactly the class the next field read
    needs.

    A TYPELESS operand carries no type of its own and cannot disagree with
    anything, so it takes the other side's -- a typed ``na`` already IS the
    shape it names and needs no rule. A TUPLE joins element by element, which
    is the same rule one level down.

    :param left: One type
    :param right: The other type
    :return: The joined type
    """
    if left == right:
        return left
    if left == TYPELESS:
        return right
    if right == TYPELESS:
        return left
    if is_tuple(left) and is_tuple(right):
        # A tuple joins POSITION BY POSITION: the branches of ``[a, b] = cond
        # ? [x, 1] : [y, 2]`` each say something about both halves, and one
        # position disagreeing says nothing about the other. Two different
        # arities are two different types, which is the mismatch case.
        items, others = elements_of(left), elements_of(right)
        if len(items) != len(others):
            return UNKNOWN
        return tuple_of([join(item, other) for item, other in zip(items, others)])
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
    left, right = head(left), head(right)
    # MEASURED (FX:EURUSD@60, ``na_probe1``/``3``/``4``): an untyped ``na``
    # operand takes the OTHER operand's type -- ``int a = na + (R + z)``,
    # ``float h = na + (R + z)`` and ``string g = na + 'x'`` all compile, while
    # ``int i = na + 1.0`` is rejected with CE10173 ("const float"). Two of
    # them are int: the same channel reports ``na + na`` as "const int"
    if left == TYPELESS and right == TYPELESS:
        left = right = INT
    elif left == TYPELESS:
        left = right
    elif right == TYPELESS:
        right = left
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
    so the unary sign is type-preserving. ``not`` is always bool. An untyped
    ``na`` stays untyped under a sign, which is what makes ``int e = -na``
    compile (MEASURED, ``na_probe3``).

    :param op: The operator node
    :param operand: Type of the operand
    :return: Type of the result
    """
    if isinstance(op, ast.Not):
        return BOOL
    if isinstance(op, (ast.UAdd, ast.USub)):
        if operand == TYPELESS:
            return TYPELESS
        return operand if head(operand) in NUMERIC else UNKNOWN
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
#: result -- the ones import normalization leaves alone. Three families live
#: here: the Python builtins a hand-written Pyne script uses, the
#: ``pine_cast`` helpers a compiled script gets its Pine casts as (the lib
#: spellings ``lib.int`` and friends are in ``LIB_TYPE_OVERRIDES`` instead),
#: and the plumbing the pipeline itself emits around a script.
#: The values take the same forms ``LIB_TYPE_OVERRIDES`` does.
BUILTIN_CALL_TYPES: Final[dict[str, object]] = {
    'int': INT,
    'float': FLOAT,
    'bool': BOOL,
    'str': STR,
    'len': INT,
    'abs': 'arg0',
    # ``NA(int)``, ``NA(Line)``: a typed na IS of the type it names. Without a
    # type to name it is the na object itself, which is Pine's typeless marker
    'NA': ['na_arg', TYPELESS],
    # The tail every compiled script carries: ``if __name__ == '__main__':
    # run(__file__)``. ``run`` is the runner entry point and returns nothing a
    # script can read
    'run': VOID,
    # ``SecurityTransformer``'s plumbing. The write, the signal and the wait
    # are statements; the unzip transposes an intrabar buffer into arrays. The
    # READ is the one that carries a value, and it is typed from the write of
    # the same id (see ``_security_read``)
    '__sec_write__': VOID,
    '__sec_signal__': VOID,
    '__sec_wait__': VOID,
    # ``DynamicDefaultTransformer``'s plumbing: the bool na factory a UDT
    # field's ``na(bool)`` default is bound to
    '__pyne_bool_na·__': BOOL,
    '__ltf_unzip__': OBJECT,
    # A Pine ``for`` iterates over one of these; the loop VARIABLE's type is
    # the join of the bounds (see ``_element_type``), while the iterable
    # itself is an ordinary known non-scalar
    'range': OBJECT,
    'pine_range': OBJECT,
    # The compiled forms of a loop with a re-read bound and of an inline
    # history read: ``pine_loop(from, step)`` builds the counter object whose
    # ``value`` the walker types from the bounds (see ``_loop_counter``);
    # ``inline_series(expr, n)`` is ``expr[n]``, type-preserving
    'pine_loop': PINE_LOOP,
    'inline_series': 'arg0',
    # ``core/pine_export.Exported()``: the module-level proxy a compiled
    # library binds every export to, which ``@export`` fills in at run time.
    # A callable reference, like a bare function name
    'Exported': OBJECT,
    # ``import_normalizer``'s builtin-namespace merge: the value is a
    # namespace object, and its members are resolved per member
    'shadowed_namespace': OBJECT,
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
    # The edge profile's own builtins: ``max``/``min`` are ``math.max``/
    # ``math.min`` (MEASURED: all-int arguments give an int), ``print`` is a
    # statement, ``enumerate`` is the iterable a ``for [i, x] in arr`` compiles to
    'max': 'all_int',
    'min': 'all_int',
    'print': VOID,
    'enumerate': OBJECT,
}


#: Bare names the pipeline's own plumbing binds, with the type a READ of them
#: has. None of them is a Pine value: they are the module's dunders and the
#: security transformer's process-identity globals, and they only ever appear
#: inside a comparison the ``Compare`` rule already types as a bool. Naming
#: them here is what keeps the comparison's operands from being the untyped
#: nodes under a typed one.
BUILTIN_NAME_TYPES: Final[dict[str, str]] = {
    '__name__': STR,
    '__file__': STR,
    '__active_security__': OBJECT,
    '__same_context__': OBJECT,
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

#: Annotation heads whose payload is the ARRAY element type. PyneComp emits
#: ``T[]`` as ``list[T]``; ``SequenceView`` is what ``array.slice`` hands back,
#: and the runtime dispatches a view to the array namespace like an array.
_ARRAY_SUBSCRIPTS: Final = frozenset({'list', 'Array', 'SequenceView'})

#: Annotation heads whose payload is the MATRIX element type.
_MATRIX_SUBSCRIPTS: Final = frozenset({'Matrix'})

#: Annotation heads whose payload is a (key, value) pair. PyneComp emits
#: ``map<K, V>`` as ``dict[K, V]``.
_MAP_SUBSCRIPTS: Final = frozenset({'dict', 'Map'})

#: Annotation heads whose payload is the tuple's element types, one per
#: position: ``tuple[float, int]`` is Pine's ``[float, int]``.
_TUPLE_SUBSCRIPTS: Final = frozenset({'tuple', 'Tuple'})

#: Annotation heads that denote a known non-scalar carrying no element type.
_OBJECT_SUBSCRIPTS: Final = frozenset({'set'})

#: Container names with no payload at all: known objects whose element type
#: the annotation simply does not spell.
_CONTAINER_NAMES: Final = frozenset({
    'Array', 'Matrix', 'Map', 'SequenceView', 'list', 'tuple', 'dict', 'set',
})

#: The class map of a caller that has none to offer.
NO_CLASSES: Final[Mapping[str, str]] = {}


def bare_wrapper(node: ast.expr) -> bool:
    """
    Whether an annotation is a series/persistence wrapper that names no type.

    ``x: Series = ta.ema(close, 9)`` and ``var x = ...`` in Pine say HOW the
    variable lives, not what it holds: the wrapper without a subscript is a
    storage marker, and the type is whatever the value has. ``Series[int]``,
    by contrast, declares.

    :param node: The annotation
    :return: True for a bare ``Series``, ``Persistent``, ``NA`` and their kin
    """
    name = node.id if isinstance(node, ast.Name) else \
        (node.attr if isinstance(node, ast.Attribute) else '')
    return name in _TRANSPARENT_SUBSCRIPTS


def annotation_type(node: ast.expr | None, classes: Mapping[str, str] = NO_CLASSES) -> str:
    """
    Map a Python annotation onto a Pine type character.

    Unions are joined, so ``int | NA`` is ``int`` and ``float | int`` is
    ``float``; the ``Series``/``NA``/``Persistent`` wrappers are transparent,
    since they change the storage, not the Pine type. An annotation this does
    not recognize is UNKNOWN, which is what makes a missing annotation and an
    exotic one behave alike.

    A name in ``classes`` is an object OF THAT CLASS, not an unknown and not
    an anonymous object. A UDT is a type a script declares and a library
    publishes, and a parameter typed by one is fully annotated -- reading it
    as unknown made such a parameter behave like an unannotated one, which is
    what took the whole export out of the typed world for its callers, while
    reading it as a bare object lost the field types the class declares.

    The containers carry their element type the same way: ``list[int]`` is an
    ``array<int>``, ``Matrix[float]`` a ``matrix<float>`` and ``dict[str,
    list[float]]`` a ``map<string, array<float>>`` -- the spellings PyneComp
    emits for Pine's ``array<T>``, ``matrix<T>`` and ``map<K, V>``. A
    ``tuple[float, int]`` is Pine's ``[float, int]``, one type per position.

    :param node: The annotation expression, or None when there is none
    :param classes: Class SPELLING -> class id, for every class visible where
                    the annotation stands, the imported ones included. An
                    imported class is keyed by the spelling that reaches it
                    (``Settings`` for ``from m import Settings``, ``m.Settings``
                    for a namespace import), because identity is (module, name)
                    and two modules' same-named classes are two types
    :return: The type
    """
    if node is None:
        return UNKNOWN

    if isinstance(node, ast.Constant):
        if node.value is None:
            return VOID
        # A stringized forward reference: ``'_ExitOrderKey'``, ``'list[Pivot]'``
        if isinstance(node.value, str):
            try:
                return annotation_type(ast.parse(node.value, mode='eval').body, classes)
            except SyntaxError:
                return UNKNOWN
        return UNKNOWN

    if isinstance(node, ast.Name):
        return _named_type(node.id, classes)

    if isinstance(node, ast.Attribute):
        # A QUALIFIED spelling is resolved as a whole first: ``a.Settings`` and
        # ``b.Settings`` are two different classes, and the leaf cannot tell
        # them apart. Only a spelling nothing published under its full path
        # falls back to the leaf -- a class nested in another one, and the lib
        # classes a script names through ``lib.``
        spelled = _dotted(node)
        if spelled is not None and spelled in classes:
            return object_ty(classes[spelled])
        return _named_type(node.attr, classes)

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _union_type(node, classes)

    if isinstance(node, ast.Subscript):
        return _subscript_type(node, classes)

    return UNKNOWN


def _subscript_type(node: ast.Subscript, classes: Mapping[str, str]) -> str:
    """
    Type of a subscripted annotation.

    :param node: The ``X[...]`` annotation node
    :param classes: Class name -> class id, visible where it stands
    :return: The type
    """
    value = node.value
    name = value.id if isinstance(value, ast.Name) else \
        (value.attr if isinstance(value, ast.Attribute) else '')
    if name in _TRANSPARENT_SUBSCRIPTS:
        return annotation_type(node.slice, classes)
    if name in _ARRAY_SUBSCRIPTS:
        return array_of(annotation_type(node.slice, classes))
    if name in _MATRIX_SUBSCRIPTS:
        return matrix_of(annotation_type(node.slice, classes))
    if name in _MAP_SUBSCRIPTS:
        pair = node.slice.elts if isinstance(node.slice, ast.Tuple) else ()
        if len(pair) != 2:
            return OBJECT
        return map_of(annotation_type(pair[0], classes),
                      annotation_type(pair[1], classes))
    if name in _TUPLE_SUBSCRIPTS:
        # ``tuple[int]`` subscripts with the element itself, ``tuple[int,
        # float]`` with a tuple of them: one element or many, the arity is
        # fixed either way
        items = node.slice.elts if isinstance(node.slice, ast.Tuple) else (node.slice,)
        if any(isinstance(item, ast.Constant) and item.value is Ellipsis for item in items):
            # ``tuple[X, ...]`` is a sequence of unknown length, which is not
            # a Pine tuple at all -- Pine's has a fixed arity, and that arity
            # is what an unpack is checked against
            return OBJECT
        return tuple_of([annotation_type(item, classes) for item in items])
    if name in _OBJECT_SUBSCRIPTS:
        return OBJECT
    return UNKNOWN


def _named_type(name: str, classes: Mapping[str, str] = NO_CLASSES) -> str:
    """Resolve a bare annotation name to a type."""
    if name in ANNOTATION_TYPES:
        return ANNOTATION_TYPES[name]
    # A container spelled without its payload is an object whose element type
    # the annotation does not say -- and it stays that even where the module
    # has the container class itself in scope, which is how the lib spells its
    # own ``Matrix`` and ``SequenceView`` returns
    if name in _CONTAINER_NAMES:
        return OBJECT
    # A class the module declares or imports wins over the builtin of the same
    # name: a script's own ``Line`` is the class it wrote, not the lib's
    found = classes.get(name)
    if found is not None:
        return object_ty(found)
    builtin = BUILTIN_CLASSES.get(name)
    if builtin is not None:
        return builtin
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


def _union_type(node: ast.BinOp, classes: Mapping[str, str] = NO_CLASSES) -> str:
    """
    Type of an annotation union, ignoring its absence markers.

    :param node: The ``X | Y`` annotation node
    :param classes: Class name -> class id, visible where the annotation stands
    :return: The joined type
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

    result = annotation_type(contributing[0], classes)
    for member in contributing[1:]:
        result = join(result, annotation_type(member, classes))
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
    # A shaped argument pins as its head, which is an object -- and an object
    # is not pinnable, so the wire format never sees a shape
    arg_types = [head(t) for t in arg_types]
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
    # Heads, not shapes: what the runtime checks a default against is an
    # ``isinstance``, which cannot see a class or an element type
    return FIT_OMISSIBLE if head(default) == head(declared) else FIT_REQUIRED


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
    #: Name of each positional parameter, so a keyword argument can be matched
    names: tuple[str, ...] = ()


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
        declared = head(declared)
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


# --- reading a definition -------------------------------------------------

#: Name ``DynamicDefaultTransformer`` leaves in place of a default that
#: reads per-bar lib state. The selector skips such a parameter, so the
#: sentinel fits whatever the parameter declares.
_DYN_DEFAULT: Final = '__dyn_default__'


def impl_sig(node: ast.FunctionDef | ast.AsyncFunctionDef, ret: str,
             ty_of: Callable[[ast.AST], str] = get_ty,
             classes: Mapping[str, str] = NO_CLASSES) -> ImplSig:
    """
    One definition's shape, as the static overload selection reads it.

    The single place a definition becomes an ``ImplSig``: the selection reads
    these to answer a pinned call site, and a module's published interface
    carries the same ones so another module's call sites get the same answer.

    :param node: The definition to measure
    :param ret: What it returns -- the inference's answer, not the annotation's
    :param ty_of: Types a default expression; the walk passes its OWN view, a
                  consumer reading a finished tree the node stamps
    :param classes: Class name -> class id, visible in the definition's module
    :return: The implementation's shape
    """
    args = node.args
    positional = list(args.posonlyargs) + list(args.args)
    defaults = _param_defaults(node)
    return ImplSig(
        params=tuple(annotation_type(a.annotation, classes) for a in positional),
        required=len(positional) - len(args.defaults),
        open_ended=args.vararg is not None, ret=ret,
        fits=''.join(param_fit(arg, defaults.get(arg.arg), ty_of, classes)
                     for arg in positional + list(args.kwonlyargs)),
        names=tuple(a.arg for a in positional))


def param_fit(arg: ast.arg, default: ast.expr | None,
              ty_of: Callable[[ast.AST], str] = get_ty,
              classes: Mapping[str, str] = NO_CLASSES) -> str:
    """
    What one parameter contributes to a call that omits it.

    An UNANNOTATED parameter contributes nothing: the selector has no declared
    type to check its default against, so it takes whatever the default is.
    That is not the same as an annotation this pass cannot read, which
    ``default_fit`` declines on.

    :param arg: The parameter
    :param default: Its default expression, or None when it has none
    :param ty_of: Types the default expression
    :param classes: Class name -> class id, visible where the annotation stands
    :return: One of the ``FIT_*`` characters
    """
    if default is None:
        return FIT_REQUIRED
    if arg.annotation is None:
        return FIT_OMISSIBLE
    return default_fit(annotation_type(arg.annotation, classes), default_ty(default, ty_of),
                       annotation_takes_none(arg.annotation))


def default_ty(default: ast.expr, ty_of: Callable[[ast.AST], str] = get_ty) -> str:
    """
    Type of a default expression, as the overload selector reads it.

    ``na`` carries no type of its own, and neither does the sentinel
    ``DynamicDefaultTransformer`` leaves where a default read per-bar lib
    state -- the selector skips that one outright. A literal ``None`` is a
    VALUE the selector type-checks like any other, so it gets its own
    character and is decided against the annotation, not against the
    annotation's type.

    :param default: The default expression
    :param ty_of: Types anything the two special forms do not cover
    :return: Its type character, ``NONE_DEFAULT`` or ``TYPELESS``
    """
    if isinstance(default, ast.Constant) and default.value is None:
        return NONE_DEFAULT
    dotted = _dotted(default)
    if dotted is not None and dotted.split('.')[-1] in ('na', _DYN_DEFAULT):
        return TYPELESS
    return ty_of(default)


def _param_defaults(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, ast.expr]:
    """
    Each parameter's default expression, by parameter name.

    A default is one half of what types a parameter: at a call site the type is
    JOIN(default, argument), and where the caller omits the argument the
    default IS the value passed. A SCRIPT ENTRY POINT is the extreme of that
    same rule -- ``run_main()`` passes no arguments, so an entry's parameter is
    its default on every bar, which is how a compiled script receives its
    inputs (``main(length=input.int(14))``, unannotated, because Pine's
    ``input.int``'s first parameter is the input's default VALUE).

    What a default never does on its own is DECLARE a type: with no call site
    to join with, ``def helper(x=0)`` says nothing about ``x``, because
    ``helper(1.5)`` is a legal call.

    :param node: The definition to read
    :return: parameter name -> default expression, for the ones that have one
    """
    args = node.args
    positional = list(args.posonlyargs) + list(args.args)
    out: dict[str, ast.expr] = {}
    # The defaults align to the LAST parameters, one per default
    for arg, default in zip(positional[len(positional) - len(args.defaults):], args.defaults):
        out[arg.arg] = default
    for kwarg, kw_default in zip(args.kwonlyargs, args.kw_defaults):
        if kw_default is not None:
            out[kwarg.arg] = kw_default
    return out


def _dotted(node: ast.expr) -> str | None:
    """Render a dotted name expression, or None when it is not one."""
    parts: list[str] = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return '.'.join(reversed(parts))


# --- measured overrides ---------------------------------------------------

#: Lib names whose TradingView type is NOT what the Python annotation says,
#: keyed by the dotted lib path. A value is either a TYPE (a fixed result: a
#: character, or a whole shape such as ``'a:i'``), one of the forms below, a
#: dict mapping an argument COUNT to any of those (an arity split), or a list
#: of any of those tried in order until one answers.
#:
#: The forms, with ``N`` a declared parameter position:
#:
#: ``'argN'``            echoes that argument's type, UNKNOWN when it has none
#: ``'all_int'``         int when EVERY argument is int, float otherwise
#: ``'join_args'``       the join of every argument's type
#: ``'na_arg'``          the type the first argument NAMES (typed ``na``)
#: ``'same_arrayN'``     that argument's own type, but only while it is an ARRAY
#: ``'same_matrixN'``    that argument's own type, but only while it is a MATRIX
#: ``'same_mapN'``       that argument's own type, but only while it is a MAP
#: ``'merge_array'``     the first argument's array, after checking that the
#:                       SECOND is an array whose elements fit into it
#: ``'merge_matrix'``    the same for two matrices
#: ``'merge_matrix_or_scalar'``  the same, the second operand a matrix or a
#:                       number that fits the elements
#: ``'elemN'``           the element type of that array or matrix
#: ``'map_keyN'``        the key type of that map
#: ``'map_valueN'``      the value type of that map
#: ``'array_of_argN'``   an array over that argument's type
#: ``'array_of_elemN'``  an array over that array's or matrix's element type
#: ``'array_of_map_keysN'``    an array over that map's key type
#: ``'array_of_map_valuesN'``  an array over that map's value type
#: ``'matrix_of_argN'``  a matrix over that argument's type
#: ``'array_of_join_args'``    an array over the join of every argument
#: ``'matrix_mult'``     ``matrix.mult``, whose result follows its SECOND
#:                       operand: a matrix by an array yields an array
#:
#: Every form from ``'same_arrayN'`` down DECLINES -- answers nothing at all --
#: where the shape it reads is not known, so the call falls through to the
#: annotation instead of claiming an UNKNOWN the annotation could have typed.
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
    'math.pi': FLOAT,
    'math.e': FLOAT,
    # MEASURED (FX:EURUSD@60, ``R = input.int(14)``): ``int e = nz(R, 1.0)`` is
    # rejected with CE10173 "simple float", while ``int e = nz(R, 2)`` and
    # ``int e = nz(R)`` both compile -- so ``nz`` widens with its REPLACEMENT
    # and is not simply type-preserving. The history index still is: ``d[1]``
    # on an int-typed ``d`` is int
    'nz': 'join_args',
    # ``na(x)`` is the predicate; ``na(int)`` and ``na(Line)`` are the typed-na
    # constructors the compiled form spells a declared na with
    'na': ['na_arg', BOOL],
    # The interned typeless na ``module_property`` rewrites a value-position
    # bare ``na`` to. Pine's ``na`` is TYPELESS: it carries no type of its own
    # and takes the one of whatever it meets, which is why ``x = cond ?
    # line.new(...) : na`` is a line and ``float x = na`` is a float. MEASURED
    # (``na_probe3``): ``int f = na`` compiles, and so does every other
    # declared type -- so joining it with a type must yield that type, not the
    # UNKNOWN a void would give
    '_na_none': TYPELESS,
    # MEASURED: ``int a = ta.change(bar_index)``, ``float b = ta.change(close)``
    # and ``bool c = ta.change(close > open)`` all compile, so the result is the
    # source's own type
    'ta.change': 'arg0',
    'fixnan': 'arg0',
    # The casts are the only place the TYPE and the VALUE move together
    'int': INT,
    'float': FLOAT,
    'bool': BOOL,
    'string': STR,
    'color': COLOR,
    # Inputs are typed by the constructor that made them, and the GENERIC one
    # by its default value -- MEASURED: ``int g = input(14)``,
    # ``float h = input(1.5)``, ``bool i = input(true)``,
    # ``string j = input("x")`` and ``color k = input(color.red)`` all compile
    'input': 'arg0',
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

    # --- the containers ---------------------------------------------------
    # MEASURED (FX:EURUSD@60, probes ``shape_probe1``/``3``/``4``/``5``, an
    # ``array<int>`` holding the int-typed 3.5, an ``matrix<int>`` and a
    # ``map<string, int>``): a container reduction returns the ELEMENT type,
    # not a float. ``int v = array.avg(ai)`` compiles, and so do ``sum``,
    # ``min``, ``max``, ``range``, ``median``, ``mode``, ``stdev``,
    # ``variance``, ``percentile_nearest_rank``,
    # ``percentile_linear_interpolation`` and ``percentrank`` -- while
    # ``int v = math.sqrt(4 + z)`` on the same probe is rejected with CE10173,
    # which is what makes the channel a measurement. ``array.covariance`` is
    # the ONE exception measured: it is a float over an int array.
    # The derivations follow the same law -- ``abs``, ``standardize``,
    # ``copy``, ``slice``, ``concat`` and ``from`` over an ``array<int>`` all
    # read back as int, ``array.from(int, float)`` widens to ``array<float>``,
    # and every matrix and map accessor answers with its own element type.
    'array.get': 'elem0',
    'array.first': 'elem0',
    'array.last': 'elem0',
    'array.pop': 'elem0',
    'array.shift': 'elem0',
    'array.remove': 'elem0',
    'array.min': 'elem0',
    'array.max': 'elem0',
    'array.mode': 'elem0',
    'array.median': 'elem0',
    'array.range': 'elem0',
    'array.avg': 'elem0',
    'array.sum': 'elem0',
    'array.stdev': 'elem0',
    'array.variance': 'elem0',
    'array.percentile_nearest_rank': 'elem0',
    'array.percentile_linear_interpolation': 'elem0',
    'array.percentrank': 'elem0',
    'array.abs': 'same_array0',
    'array.standardize': 'same_array0',
    'array.copy': 'same_array0',
    'array.slice': 'same_array0',
    'array.concat': 'merge_array',
    # One element goes INTO the array: it has to fit the element type, or the
    # array holds something its type does not say (see ``_merge_override``)
    'array.push': 'put_array:1',
    'array.unshift': 'put_array:1',
    'array.fill': 'put_array:1',
    'array.set': 'put_array:2',
    'array.insert': 'put_array:2',
    'map.put': 'put_map:2',
    'matrix.set': 'put_matrix:3',
    'matrix.fill': 'put_matrix:1',
    'matrix.add_row': 'put_matrix_array:2',
    'matrix.add_col': 'put_matrix_array:2',
    'array.from_items': 'array_of_join_args',
    # ``array.new(size, initial)`` is the generic constructor a compiled
    # ``array.new<T>()`` becomes; the typed ``na`` it passes NAMES the element
    'array.new': 'array_of_arg1',
    'matrix.get': 'elem0',
    'matrix.avg': 'elem0',
    'matrix.min': 'elem0',
    'matrix.max': 'elem0',
    'matrix.mode': 'elem0',
    'matrix.median': 'elem0',
    'matrix.det': 'elem0',
    'matrix.trace': 'elem0',
    'matrix.row': 'array_of_elem0',
    'matrix.col': 'array_of_elem0',
    'matrix.remove_row': 'array_of_elem0',
    'matrix.remove_col': 'array_of_elem0',
    'matrix.eigenvalues': 'array_of_elem0',
    'matrix.copy': 'same_matrix0',
    'matrix.transpose': 'same_matrix0',
    'matrix.inv': 'same_matrix0',
    'matrix.pinv': 'same_matrix0',
    'matrix.pow': 'same_matrix0',
    'matrix.submatrix': 'same_matrix0',
    'matrix.eigenvectors': 'same_matrix0',
    'matrix.kron': 'merge_matrix',
    'matrix.concat': 'merge_matrix',
    'matrix.sum': 'merge_matrix_or_scalar',
    'matrix.diff': 'merge_matrix_or_scalar',
    'matrix.mult': 'matrix_mult',
    'matrix.new': 'matrix_of_arg2',
    'map.get': 'map_value0',
    'map.remove': 'map_value0',
    'map.keys': 'array_of_map_keys0',
    'map.values': 'array_of_map_values0',
    'map.copy': 'same_map0',
}

#: Declared parameter order of the lib names whose module lives outside
#: ``pynecore/lib`` and therefore has no registry entry to read it from. A
#: type-preserving override names the parameter it copies from, and a keyword
#: spelling (``input(title='WMA Length', defval=10)``) only binds back to that
#: position with the order in hand.
OVERRIDE_PARAM_NAMES: Final[dict[str, list[str]]] = {
    # ``core/script.py::_Input.__call__``
    'input': ['defval', 'title', 'tooltip', 'inline', 'group', 'display', 'active'],
}


class FactoryFields:
    """
    The ``field(default_factory=...)`` defaults of a module's UDT classes.

    Only a direct annotated field of a ``@udt`` / ``@dataclass`` class whose
    default calls the name bound to ``dataclasses.field`` AT THAT STATEMENT
    (a later import, definition or assignment of the name counterfeits it),
    with a zero-argument lambda as the factory -- the one form the compiler
    emits -- is dataclass plumbing: the machinery builds it and the annotation
    types the field. Any other call carrying a ``default_factory`` keyword is
    an ordinary call and is typed (and diagnosed) like one.
    """

    __slots__ = ('body', 'top_index')

    def __init__(self, tree: ast.Module):
        self.body = tree.body
        #: Every class's top-level statement index
        self.top_index: dict[int, int] = {}
        for index, stmt in enumerate(tree.body):
            for node in ast.walk(stmt):
                if isinstance(node, ast.ClassDef):
                    self.top_index[id(node)] = index

    def _binding(self, name: str, before: int) -> str | None:
        """What ``name`` denotes at statement ``before``: 'field', 'dataclasses' or other."""
        binding: str | None = None
        for stmt in self.body[:before]:
            if isinstance(stmt, ast.ImportFrom):
                for alias in stmt.names:
                    if (alias.asname or alias.name) == name:
                        binding = 'field' if (stmt.module == 'dataclasses' and not stmt.level
                                              and alias.name == 'field') else 'other'
            elif isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    if (alias.asname or alias.name.split('.')[0]) == name:
                        binding = 'dataclasses' if alias.name == 'dataclasses' else 'other'
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if stmt.name == name:
                    binding = 'other'
            else:
                for node in ast.walk(stmt):
                    if isinstance(node, ast.Name) and node.id == name \
                            and isinstance(node.ctx, ast.Store):
                        binding = 'other'
        return binding

    def of(self, cls: ast.ClassDef) -> list[ast.Call]:
        """The factory-default calls of a class, empty for a class that is no UDT."""
        if not any((_dotted(decorator) or '').rsplit('.', 1)[-1] in ('udt', 'dataclass')
                   for decorator in cls.decorator_list):
            return []
        before = self.top_index.get(id(cls), len(self.body))
        out: list[ast.Call] = []
        for stmt in cls.body:
            value = stmt.value if isinstance(stmt, ast.AnnAssign) else None
            if not isinstance(value, ast.Call) or len(value.keywords) != 1 or value.args:
                continue
            keyword = value.keywords[0]
            factory = keyword.value
            if keyword.arg != 'default_factory' or not isinstance(factory, ast.Lambda) \
                    or factory.args.args or factory.args.posonlyargs or factory.args.kwonlyargs \
                    or factory.args.vararg or factory.args.kwarg:
                continue
            func = value.func
            if (isinstance(func, ast.Name) and self._binding(func.id, before) == 'field') or (
                    isinstance(func, ast.Attribute) and func.attr == 'field'
                    and isinstance(func.value, ast.Name)
                    and self._binding(func.value.id, before) == 'dataclasses'):
                out.append(value)
        return out
