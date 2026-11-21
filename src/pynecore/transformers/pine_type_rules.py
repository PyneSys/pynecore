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

from . import ast_walk
from .factory_fields import dotted_name
from .lib_registry import lib_registry

#: Any expression node -- ``stamp_lowering`` hands back exactly what it got.
_E = TypeVar('_E', bound=ast.expr)

__all__ = [
    'INT', 'FLOAT', 'BOOL', 'STR', 'COLOR', 'OBJECT', 'PINE_LOOP', 'VOID', 'UNKNOWN',
    'NUMERIC', 'KNOWN', 'SCALARS',
    'LIB_MODULE', 'CLASS_SEP', 'class_id', 'builtin_class_id', 'object_ty',
    'array_of', 'matrix_of', 'map_of', 'tuple_of', 'head', 'is_shaped', 'is_array',
    'is_matrix', 'is_map', 'is_tuple', 'arity', 'elements_of', 'class_of',
    'element_of', 'key_of', 'value_of', 'shape_mismatch', 'shape_conflict', 'ternary_takes_first',
    'render_ty',
    'BUILTIN_CLASSES', 'BUILTIN_NAMESPACES', 'namespace_of',
    'join', 'binop_type', 'unaryop_type', 'compare_type',
    'annotation_type', 'bare_wrapper', 'ANNOTATION_TYPES', 'LIB_TYPE_OVERRIDES',
    'OVERRIDE_PARAM_NAMES',
    'is_int_typed', 'TY_ATTR', 'get_ty', 'set_ty', 'inherit_ty',
    'constant_type', 'stamp_lowering', 'BUILTIN_CALL_TYPES', 'BUILTIN_NAME_TYPES',
    'PIN_ATTR', 'PINS_ATTR', 'PINNABLE', 'PIN_CONTAINERS', 'PIN_ANY', 'get_pin', 'set_pin', 'get_pins', 'set_pins',
    'VECTOR_ATTR', 'VARYING_ATTR', 'get_vector', 'set_vector', 'get_varying', 'set_varying',
    'WIDEN_ATTR', 'WIDEN_FLOAT', 'WIDEN_BOOL', 'WIDEN_ITEMS', 'WIDEN_MATRIX', 'WIDEN_STRING_NA', 'WIDEN_NUMBER_NA',
    'WIDEN_BOOL_NA',
    'WIDEN_NA_TEXT', 'WIDEN_ARRAY_NA', 'WIDEN_MAP_NA', 'WIDEN_MATRIX_NA', 'CONTAINER_NA_WIDENS',
    'NA_LITERALS', 'is_na_literal', 'is_na_widen',
    'NA_CLASS_ATTR', 'get_na_class', 'set_na_class',
    'NA_TEXT_PARAMS', 'NOT_PINE_STRINGS', 'get_widen',
    'set_widen', 'WIDEN_SITE_ATTR', 'WIDENS_ATTR', 'get_widen_site', 'set_widen_site',
    'get_widens', 'set_widens',
    'pin_chars', 'pin_for', 'value_chars', 'overload_result', 'ImplSig', 'overload_pick',
    'overload_bind', 'overload_choice', 'overload_candidates', 'overload_binding',
    'PIN_INDEX', 'indexed_pin', 'pin_index', 'pine_choice', 'value_rival',
    'resolve_pin', 'PIN_WITNESSED', 'ACCEPTS_SEP', 'ACCEPTS_UNKNOWN', 'annotation_accepts',
    'is_str_literal',
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

#: Attribute a value carries when Pine converts it implicitly where it goes: an
#: int into a float-element container, a number into a lib ``bool`` parameter, a
#: typeless ``na`` into a string, a number or the na of a class (see
#: ``implicit_conversion.py``). Present only where every
#: context agrees on it.
WIDEN_ATTR: Final = '_pine_widen'

#: The value is one int element: it is converted to float
WIDEN_FLOAT: Final = 'f'

#: The value is a number passed to a ``bool`` parameter of a lib function: it is
#: converted by Pine's tolerant float-to-bool rule, na included
WIDEN_BOOL: Final = 'B'

#: The value is an int array whose elements are copied in: they are converted
WIDEN_ITEMS: Final = 'a'

#: The value is an int matrix whose elements are copied in: they are converted
WIDEN_MATRIX: Final = 'm'

#: The value is a typeless ``na`` that goes where a string goes: it is the string na,
#: the empty string
WIDEN_STRING_NA: Final = 's'

#: The value is a typeless ``na`` that goes where a float or an int goes: it is the
#: numeric na, NaN (what ``NA(float)`` and ``NA(int)`` both are)
WIDEN_NUMBER_NA: Final = 'n'

#: The value is a typeless ``na`` that goes where a bool goes: it is the bool na the
#: script runs with -- ``NA(bool)``, Pine v4/v5's third state under ``na_bool``, plain
#: ``False`` otherwise (v6 refuses the typeless na there at compile time)
WIDEN_BOOL_NA: Final = 'b'

#: The value is a typeless ``na`` passed as compile-time text of the study
#: (``NA_TEXT_PARAMS``): TradingView writes it out as the text ``na``
WIDEN_NA_TEXT: Final = 't'

#: The value is a typeless ``na`` that goes where an array goes: it is the na of an
#: array, ``NA(list)``
WIDEN_ARRAY_NA: Final = 'A'

#: The value is a typeless ``na`` that goes where a map goes: it is the na of a map,
#: ``NA(dict)``
WIDEN_MAP_NA: Final = 'M'

#: The value is a typeless ``na`` that goes where a matrix goes: it is the na of a
#: matrix, ``NA(Matrix)``
WIDEN_MATRIX_NA: Final = 'X'

#: The conversions of a typeless ``na`` into the na of a container
CONTAINER_NA_WIDENS: Final = frozenset({WIDEN_ARRAY_NA, WIDEN_MAP_NA, WIDEN_MATRIX_NA})

#: A typeless ``na`` that goes where an object of a class goes -- a UDT, a drawing
#: -- is the na OF that class, and its conversion kind is the object type itself
#: (``o:<class id>``), so two contexts agree on it only when they agree on the
#: class. The value then carries the class's spelling in this attribute: the
#: expression, valid at the module's top level, that names the class in value
#: position (``Pivot``, ``chr.SessionTimeRange``, the ``Line`` an import binds).
#: One that becomes the na of a container (``CONTAINER_NA_WIDENS``) carries the
#: container's class the same way (``list``, ``dict``, the ``Matrix`` an import binds)
NA_CLASS_ATTR: Final = '_pine_na_class'

#: The spellings of Pine's typeless ``na`` in value position: the name itself, and
#: what ``module_property`` rewrites it to. One of these, signed or not, is replaced by
#: the na it is converted to; any other value that holds the typeless na -- a defaulted
#: parameter, a variable -- is converted at run time, which leaves every other value as
#: it is
NA_LITERALS: Final = frozenset({'na', 'lib.na', '_na_none', 'lib._na_none'})

#: The parameters whose value TradingView compiles into the study's metadata rather
#: than evaluating at run time, per lib callee. A typeless ``na`` there is not the
#: string na: MEASURED, ``plot(title=na)`` is titled ``na``, ``plotchar(char=na)``
#: draws ``n`` and ``alertcondition(message=na)`` reads ``na``, while ``title=""``
#: stays empty. (``indicator`` and ``input.*`` titles reject a na at compile time.)
NA_TEXT_PARAMS: Final[dict[str, frozenset[str]]] = {
    'plot': frozenset({'title'}),
    'plot.plot': frozenset({'title'}),
    'plotchar': frozenset({'title', 'char', 'text'}),
    'plotshape': frozenset({'title', 'text'}),
    'plotarrow': frozenset({'title'}),
    'plotbar': frozenset({'title'}),
    'plotcandle': frozenset({'title'}),
    'hline': frozenset({'title'}),
    'hline.hline': frozenset({'title'}),
    'bgcolor': frozenset({'title'}),
    'barcolor': frozenset({'title'}),
    'fill': frozenset({'title'}),
    'alertcondition': frozenset({'title', 'message'}),
}

#: Lib parameters the registry types as strings that are no Pine string: an enum
#: Pine spells as a constant (``matrix.sort``'s ``order``) and a Python-only
#: extension (``sort_field``). A typeless ``na`` passed there is left as it is
NOT_PINE_STRINGS: Final[frozenset[tuple[str, str]]] = frozenset({
    ('matrix.sort', 'order'),
    ('array.sort', 'sort_field'),
    ('array.sort_indices', 'sort_field'),
})

#: Attribute a value carries where the contexts of the body it is in disagree on
#: its conversion: the call ``safe_convert.implicit_convert(value, kind)`` that
#: takes the value's place, an instance-varying site whose vector entry is the
#: conversion of that instance (see ``implicit_conversion.py``).
WIDEN_SITE_ATTR: Final = '_pine_widen_site'

#: Attribute that call carries its PER-CONTEXT conversions under.
WIDENS_ATTR: Final = '_pine_widens'


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


def get_widen(node) -> str | None:
    """
    Read the implicit conversion a value needs where it goes.

    :param node: The value node
    :return: ``WIDEN_FLOAT``, ``WIDEN_ITEMS``, ``WIDEN_MATRIX``, ``WIDEN_STRING_NA``,
             ``WIDEN_NUMBER_NA``, ``WIDEN_BOOL_NA``, ``WIDEN_NA_TEXT`` or an object type
             (the na of that class), or None
    """
    return getattr(node, WIDEN_ATTR, None)


def set_widen(node, kind: str | None):
    """
    Stamp the conversion a value needs, ``None`` included (a later context that
    decides against it has to erase what an earlier one wrote).

    :param node: The value node
    :param kind: ``WIDEN_FLOAT``, ``WIDEN_BOOL``, ``WIDEN_ITEMS``, ``WIDEN_MATRIX``,
                 ``WIDEN_STRING_NA``, ``WIDEN_NUMBER_NA``, ``WIDEN_BOOL_NA``,
                 ``WIDEN_NA_TEXT``, an object type or None
    :return: ``node``
    """
    setattr(node, WIDEN_ATTR, kind)
    return node


def is_na_widen(kind: str | None) -> bool:
    """
    Whether a conversion turns Pine's typeless ``na`` into the na of a type.

    :param kind: A conversion kind, or None
    :return: True for the string na, the numeric na, the bool na, the text ``na``, the
             na of a container and the na of a class (an object type)
    """
    return kind in (WIDEN_STRING_NA, WIDEN_NUMBER_NA, WIDEN_BOOL_NA, WIDEN_NA_TEXT) \
        or kind in CONTAINER_NA_WIDENS \
        or (kind is not None and kind.startswith(_OBJ))


def get_na_class(node) -> str | None:
    """
    Read how the module spells the class whose na a typeless ``na`` becomes.

    :param node: The value node
    :return: The class's spelling in value position, or None
    """
    return getattr(node, NA_CLASS_ATTR, None)


def set_na_class(node, spelled: str):
    """
    Stamp the spelling of the class whose na a typeless ``na`` becomes.

    :param node: The value node
    :param spelled: The class's spelling in value position, valid at module level
    :return: ``node``
    """
    setattr(node, NA_CLASS_ATTR, spelled)
    return node


def get_widen_site(node) -> ast.Call | None:
    """
    Read the per-instance conversion call a value is to be wrapped in.

    :param node: The value node
    :return: The ``safe_convert.implicit_convert`` call, or None when the contexts agree
    """
    return getattr(node, WIDEN_SITE_ATTR, None)


def set_widen_site(node, site: ast.Call | None):
    """
    Stamp the per-instance conversion call of a value, ``None`` included.

    :param node: The value node
    :param site: The ``safe_convert.implicit_convert`` call, or None
    :return: ``node``
    """
    setattr(node, WIDEN_SITE_ATTR, site)
    return node


def get_widens(node) -> dict[int, str | None] | None:
    """
    Read the per-context conversions of a per-instance conversion call.

    :param node: A call node
    :return: context id -> ``WIDEN_*`` or None, or None when the node is no such call
    """
    return getattr(node, WIDENS_ATTR, None)


def set_widens(node, widens: dict[int, str | None] | None):
    """
    Stamp the per-context conversions of a per-instance conversion call.

    :param node: The ``safe_convert.implicit_convert`` call
    :param widens: context id -> ``WIDEN_*`` or None (no conversion in that context)
    :return: ``node``
    """
    setattr(node, WIDENS_ATTR, widens)
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
# published interface stays compact. The chain is int -> float -> unknown; everything
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
#          | 'M:' <key> ':' <ty> | 'T:' <item>+
#     key := <char> | <item>
#     item := <decimal length> ':' <ty>
#
# array, matrix and map, in that order; the TAIL of a map is its value type,
# which makes the grammar unambiguous read left to right. A map key is a Pine
# scalar or a Pine enum: a scalar is one character, and an enum -- an object
# whose class id may hold any character -- is an item, which carries its own
# length the way a tuple's elements do. No scalar character is a digit, so the
# first character of a key tells the two apart. A bare ``'o'`` stays what it
# always was -- an object whose class was lost -- and is what the diagnostics
# point at.
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

#: The enum set of a caller that has none to offer: every class key is refused.
NO_ENUMS: Final[frozenset[str]] = frozenset()


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


def map_of(key: str, value: str, enums: frozenset[str] = NO_ENUMS) -> str:
    """
    The type of a map from one key type to one value type.

    Pine takes a scalar or an enum as a map key and nothing else: a map keyed by
    a UDT or a drawing is no Pine type, and it reads as a bare object, the same
    as a map whose key is not known.

    :param key: The key type
    :param value: The value type
    :param enums: The class ids of the enums visible where the type is spelled
    :return: The shaped type, or OBJECT when either half is not known or the key
             is neither a scalar nor an object of one of ``enums``
    """
    if value in (UNKNOWN, VOID):
        return OBJECT
    if key in SCALARS:
        return f'{_MAP}{key}:{value}'
    if class_of(key) in enums:
        return f'{_MAP}{len(key)}:{key}:{value}'
    return OBJECT


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
    arithmetic, the overload selection -- read. The pin reads the shape itself
    where it has a witness for it (``PIN_CONTAINERS``).

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


def _item_key(ty: str) -> tuple[str, int] | None:
    """
    The enum key of a map type, read the way ``map_of`` writes it.

    An enum key is an item: a decimal length, a colon, exactly that many
    characters, and the colon the value type follows. A key that does not read
    back that way is a scalar's single character, or no key this module wrote.

    :param ty: A map type
    :return: The key and the index its value type starts at, or None when the key
             is no item
    """
    start = len(_MAP)
    colon = ty.find(':', start)
    digits = ty[start:colon]
    if colon < 0 or not digits or any(digit not in '0123456789' for digit in digits):
        return None
    end = colon + 1 + int(digits)
    if ty[end:end + 1] != ':':
        return None
    return ty[colon + 1:end], end + 1


def key_of(ty: str) -> str:
    """
    The key type of a map.

    :param ty: Any type
    :return: The key type -- a scalar character or an enum's object type --,
             UNKNOWN when the type is not a map
    """
    if not ty.startswith(_MAP):
        return UNKNOWN
    item = _item_key(ty)
    return ty[len(_MAP)] if item is None else item[0]


def value_of(ty: str) -> str:
    """
    The value type of a map.

    :param ty: Any type
    :return: The value type, UNKNOWN when the type is not a map
    """
    if not ty.startswith(_MAP):
        return UNKNOWN
    item = _item_key(ty)
    return ty[len(_MAP) + 2:] if item is None else ty[item[1]:]


#: Lib class type -> the scalar its constants are, filled on first use
_CONSTANT_SCALARS: Final[dict[str, str]] = {}


def constant_scalar(ty: str) -> str:
    """
    The scalar a type is where it meets another type.

    ``format.volume`` is a ``Format`` and ``text.align_right`` an ``AlignEnum``
    so that the parameter taking one can say which constants it means, but Pine
    spells every such constant as a ``const string`` (or a ``const int``): it
    is one type with ``'#.##'``, and with a constant of another such class. A
    set of flags (``display.all``) is the exception, a type of its own.

    :param ty: Any type
    :return: The scalar of a lib constant class, any other type unchanged
    """
    if not ty.startswith(_OBJ):
        return ty
    if not _CONSTANT_SCALARS:
        registry = lib_registry()
        flags = set(registry.get('bit_sets') or ())
        _CONSTANT_SCALARS.update(
            {object_ty(builtin_class_id(name)): scalar
             for name, scalar in (registry.get('scalar_classes') or {}).items()
             if name not in flags})
    return _CONSTANT_SCALARS.get(ty, ty)


#: The types of the lib's ``StrLiteral`` classes, filled on first use
_STR_LITERALS: Final[set[str]] = set()


def is_str_literal(ty: str) -> bool:
    """
    Whether a type is a lib class that derives from ``StrLiteral``.

    The overload selector takes any string for a parameter of such a class, in
    both of its passes (``core/overload._check_type``): ``xloc.bar_index`` and
    ``'bar_index'`` are the same argument to it. Another string class of the
    lib takes only its own instances.

    :param ty: Any type
    :return: True for the object type of such a class
    """
    if not ty.startswith(_OBJ):
        return False
    if not _STR_LITERALS:
        _STR_LITERALS.update(object_ty(builtin_class_id(name))
                             for name in lib_registry().get('str_literals') or ())
    return ty in _STR_LITERALS


def ternary_takes_first(body: str, orelse: str) -> bool:
    """
    Whether a ternary of two container types has its first branch's type.

    MEASURED on TradingView: ``cond ? <float container> : <int container>``
    compiles, and the expression is the float container -- ``float[] c = cond ?
    floats : ints`` is accepted and ``int[] c = ...`` is not. It holds for an
    array, a matrix and a map's value, for this element pair and this order
    only: ``cond ? ints : floats`` is rejected, and so is every other pair of
    element types in either order. Nothing is converted, the second branch's
    container is the value as it is.

    :param body: The first branch's type
    :param orelse: The second branch's type
    :return: True when the ternary is of ``body``'s type
    """
    if is_array(body) and is_array(orelse) or is_matrix(body) and is_matrix(orelse):
        return element_of(body) == FLOAT and element_of(orelse) == INT
    if is_map(body) and is_map(orelse):
        return key_of(body) == key_of(orelse) \
            and value_of(body) == FLOAT and value_of(orelse) == INT
    return False


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
    if head(left) != OBJECT or head(right) != OBJECT:
        return False
    # Two lib constants are scalars to each other, see ``constant_scalar``
    return constant_scalar(left) == left or constant_scalar(right) == right


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
    scalar, other = constant_scalar(left), constant_scalar(right)
    if scalar != left or other != right:
        # A lib constant meeting a scalar, or a constant of another class
        return join(scalar, other)
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
    # ``core/pine_udt.udt_copy``: what ``x.copy()`` of a user type compiles to.
    # The copy is of the value's own type
    'udt_copy': 'arg0',
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
    for child in ast_walk.iter_child_nodes(root):
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
        for child in ast_walk.iter_child_nodes(node):
            if isinstance(child, ast.expr):
                _stamp_emitted(child)
        ty = BOOL
    else:
        # A Name, an Attribute, a Subscript, a Call: the machinery's own
        # references -- a state tuple, a bound dispatcher, a helper function.
        # Known, and carrying no numeric algebra of their own.
        for child in ast_walk.iter_child_nodes(node):
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


def annotation_type(node: ast.expr | None, classes: Mapping[str, str] = NO_CLASSES,
                    enums: frozenset[str] = NO_ENUMS) -> str:
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
    A map key is a scalar or an enum (``dict[Speed, float]`` is a
    ``map<Speed, float>``); a map keyed by any other class is no Pine type and
    reads as a bare object (``map_of``).

    :param node: The annotation expression, or None when there is none
    :param classes: Class SPELLING -> class id, for every class visible where
                    the annotation stands, the imported ones included. An
                    imported class is keyed by the spelling that reaches it
                    (``Settings`` for ``from m import Settings``, ``m.Settings``
                    for a namespace import), because identity is (module, name)
                    and two modules' same-named classes are two types
    :param enums: The class ids of the enums among ``classes``, the only
                  classes a map key may be
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
                return annotation_type(ast.parse(node.value, mode='eval').body, classes, enums)
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
        spelled = dotted_name(node)
        if spelled is not None and spelled in classes:
            return object_ty(classes[spelled])
        return _named_type(node.attr, classes)

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _union_type(node, classes, enums)

    if isinstance(node, ast.Subscript):
        return _subscript_type(node, classes, enums)

    return UNKNOWN


def _subscript_type(node: ast.Subscript, classes: Mapping[str, str],
                    enums: frozenset[str]) -> str:
    """
    Type of a subscripted annotation.

    :param node: The ``X[...]`` annotation node
    :param classes: Class name -> class id, visible where it stands
    :param enums: The class ids of the enums among them
    :return: The type
    """
    value = node.value
    name = value.id if isinstance(value, ast.Name) else \
        (value.attr if isinstance(value, ast.Attribute) else '')
    if name in _TRANSPARENT_SUBSCRIPTS:
        return annotation_type(node.slice, classes, enums)
    if name in _ARRAY_SUBSCRIPTS:
        return array_of(annotation_type(node.slice, classes, enums))
    if name in _MATRIX_SUBSCRIPTS:
        return matrix_of(annotation_type(node.slice, classes, enums))
    if name in _MAP_SUBSCRIPTS:
        pair = node.slice.elts if isinstance(node.slice, ast.Tuple) else ()
        if len(pair) != 2:
            return OBJECT
        return map_of(annotation_type(pair[0], classes, enums),
                      annotation_type(pair[1], classes, enums), enums)
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
        return tuple_of([annotation_type(item, classes, enums) for item in items])
    if name in _OBJECT_SUBSCRIPTS:
        return OBJECT
    return UNKNOWN


def _named_type(name: str, classes: Mapping[str, str] = NO_CLASSES) -> str:
    """Resolve a bare annotation name to a type."""
    if name in ANNOTATION_TYPES:
        return ANNOTATION_TYPES[name]
    # A class the module declares or imports wins over the builtin of the same
    # name: a script's own ``Line`` is the class it wrote, not the lib's, and
    # its own ``Map`` is no container
    found = classes.get(name)
    if found is not None and not found.startswith(LIB_MODULE + CLASS_SEP):
        return object_ty(found)
    # A container spelled without its payload is an object whose element type
    # the annotation does not say -- and it stays that even where the module
    # has the container class itself in scope, which is how the lib spells its
    # own ``Matrix`` and ``SequenceView`` returns
    if name in _CONTAINER_NAMES:
        return OBJECT
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


def _union_type(node: ast.BinOp, classes: Mapping[str, str] = NO_CLASSES,
                enums: frozenset[str] = NO_ENUMS) -> str:
    """
    Type of an annotation union, ignoring its absence markers.

    :param node: The ``X | Y`` annotation node
    :param classes: Class name -> class id, visible where the annotation stands
    :param enums: The class ids of the enums among them
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

    result = annotation_type(contributing[0], classes, enums)
    for member in contributing[1:]:
        result = join(result, annotation_type(member, classes, enums))
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
#: here when one value stands for the whole type. A drawing, a user type or a
#: map does not qualify; an array or a matrix does when its element is a scalar
#: (see ``PIN_CONTAINERS``).
PINNABLE: Final = frozenset({INT, FLOAT, BOOL, STR})

#: Pin character of an array or a matrix of one scalar. Its witness holds one
#: element of that scalar (``core/overload.py``'s ``_PIN_WITNESSES``), which is
#: exactly what the selector samples, so the static element type decides even
#: where the runtime head element would not: an int literal in a float array, a
#: double-valued Pine int in an int array, an empty container
PIN_CONTAINERS: Final = {
    _ARRAY + INT: 'I', _ARRAY + FLOAT: 'F', _ARRAY + BOOL: 'B', _ARRAY + STR: 'S',
    _MATRIX + INT: '1', _MATRIX + FLOAT: '2', _MATRIX + BOOL: '3', _MATRIX + STR: '4',
}

#: Pin character of an array or a matrix of a scalar -> the shaped type it names
_PIN_CONTAINER_TYPES: Final = {char: ty for ty, char in PIN_CONTAINERS.items()}

_PIN_CONTAINER_CHARS: Final = frozenset(_PIN_CONTAINER_TYPES)

#: Pin character for a position the runtime has no witness value for
PIN_ANY: Final = '*'

#: Separator of a pin's implementation index: ``<characters>=<index>`` names the
#: implementation the runtime runs by its position in the group's registration
#: order (``core/overload.py``'s ``_anchored``). It is no pin character, so the
#: characters before it keep their meaning, one per positional argument.
PIN_INDEX: Final = '='

#: Every pin character the runtime has a witness value for, in the order of
#: ``core/overload.py``'s ``_PIN_WITNESSES``: the scalars, then their arrays,
#: then their matrices
PIN_WITNESSED: Final = 'ifbsIFBS1234'

#: Separator of the two passes of an acceptance spec (``ImplSig.accepts``):
#: ``<characters the exact pass takes>/<characters the widening pass takes>``
ACCEPTS_SEP: Final = '/'

#: The acceptance spec of a parameter whose check of a witness is not known
ACCEPTS_UNKNOWN: Final = '?'


def indexed_pin(arg_types: Sequence[str], slot: int) -> str:
    """
    The pin that names one implementation of a group by its registration index.

    :param arg_types: Type of each positional argument, in order
    :param slot: The implementation's position in the group's registration order
    :return: ``pin_chars`` of the arguments, then ``PIN_INDEX`` and the index
    """
    return f'{pin_chars(arg_types)}{PIN_INDEX}{slot}'


def pin_index(pin: str | None) -> int | None:
    """
    The implementation index a pin names, where it names one.

    :param pin: A pin, or None
    :return: The registration index after ``PIN_INDEX``, or None
    """
    if pin is None:
        return None
    _, separator, index = pin.partition(PIN_INDEX)
    return int(index) if separator and index.isdigit() else None


def pin_for(arg_types: list[str]) -> str | None:
    """
    The overload pin a call site's argument types justify.

    A pin is only worth emitting where the static answer can DIFFER from the
    dynamic one: where an int-typed argument is involved -- ``int / int`` keeps
    the int TYPE while the value is a Python float, so the runtime selector,
    which sees values, not types, widens it to the float implementation -- and
    where an array or a matrix of a scalar is, whose head element the runtime
    samples (an empty one has none). Every other shape already agrees, and is
    left on the ordinary path rather than pinned for nothing.

    A position the runtime cannot witness -- a container, a drawing, a user
    type, an unsettled one -- pins as ``PIN_ANY``. It says nothing about that
    argument, and the runtime treats it as such: the implementation is pinned
    only where the witnessed positions alone name exactly one. An argument like
    that sitting next to an int-typed one is common (``f(len, myEnum, src)``),
    and refusing the whole site over it would leave the int argument to the
    value dispatch, which is the divergence the pin exists to close.

    :param arg_types: Type of each positional argument, in order
    :return: One character per argument, or None when the site is not pinnable
    """
    chars = pin_chars(arg_types)
    if INT not in chars and not any(c in _PIN_CONTAINER_CHARS for c in chars):
        return None
    return chars


def pin_chars(arg_types: Sequence[str]) -> str:
    """
    A call site's argument types as the selector's exact pass reads them.

    One character per argument: the witness character of a scalar or of a
    scalar container, ``PIN_ANY`` for every position the runtime has no witness
    for. ``pin_for`` emits these only where they are worth a pin; the static
    selection (``overload_pick``) reads them at every positional call site.

    :param arg_types: Type of each positional argument, in order
    :return: One character per argument
    """
    return ''.join(PIN_CONTAINERS.get(t) or (t if t in PINNABLE else PIN_ANY)
                   for t in arg_types)


def value_chars(arg_types: Sequence[str]) -> str:
    """
    A call site's argument types as the selector reads the VALUES of an unpinned call.

    An unpinned call dispatches on what its arguments hold at run time. A
    float-typed value may be a Python int there (``float x = 1``), which the
    exact pass takes for an int, and an array or a matrix of a scalar is matched
    by its head element, which an empty one does not have: each of those
    positions reads as ``PIN_ANY``, and the selection reads it from the
    argument's static type instead (``_takes_position``). An int, a bool and a
    string keep their characters.

    :param arg_types: Type of each positional argument, in order
    :return: One character per argument
    """
    return ''.join(PIN_ANY if char == FLOAT or char in _PIN_CONTAINER_CHARS else char
                   for char in pin_chars(arg_types))


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


def _default_fit_wide(declared: str, default: str | None, takes_none: bool) -> str:
    """
    What a parameter contributes to a call that leaves it to its default, in the widening pass.

    The selector's second pass checks the defaults the way it checks the
    arguments, with an int taken where a float is declared and a float where
    an int is (``core/overload._check_type``); everything ``default_fit`` says
    of the exact pass holds here otherwise.

    :param declared: The parameter's declared type character
    :param default: The default's type character, as ``default_fit`` reads it
    :param takes_none: Whether the parameter's annotation accepts ``None``
    :return: One of the ``FIT_*`` characters
    """
    fit = default_fit(declared, default, takes_none)
    if fit == FIT_REQUIRED and default is not None and head(default) in NUMERIC \
            and head(declared) in NUMERIC:
        return FIT_OMISSIBLE
    return fit


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
        origin = node.value
        name = origin.id if isinstance(origin, ast.Name) else \
            (origin.attr if isinstance(origin, ast.Attribute) else '')
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
    #: One character per positional parameter, ``'1'`` where it has no
    #: annotation: the selector checks such a parameter against ``Any``, which
    #: takes every value in both of its passes. Empty where the shape does not
    #: say (a published one), which reads every position by its type.
    bare: str = ''
    #: Position of the implementation in its group's REGISTRATION order -- every
    #: ``@overload`` definition of the name in its scope, the ones the static
    #: selection leaves out included -- which is what an indexed pin names
    #: (``PIN_INDEX``); -1 where it is not known
    slot: int = -1
    #: How the selector checks a witness value at each positional parameter,
    #: where its declared type does not say it (``_position``): ``''`` reads the
    #: type, ``ACCEPTS_UNKNOWN`` knows nothing, and otherwise the witness
    #: characters each pass takes, ``ACCEPTS_SEP`` between them -- ``'if/if'``
    #: for an ``int | float``, ``'s/s'`` for a ``StrLiteral``. Empty where no
    #: position needs one.
    accepts: tuple[str, ...] = ()
    #: The ``fits`` of the selector's widening pass, one character per parameter
    #: in the same order; empty where it is not known (``_wide_fits``)
    wide_fits: str = ''
    #: Whether a ``**kwargs`` parameter is among the ones the selector counts
    kwarg: bool = False
    #: How many implementations the group registers, the ones the static
    #: selection leaves out included; -1 where it is not known
    size: int = -1


def _is_bare(impl: ImplSig, index: int) -> bool:
    """
    Whether an implementation's positional parameter has no annotation.

    :param impl: The implementation's shape
    :param index: The parameter's position
    :return: True when the selector takes every value there
    """
    return index < len(impl.bare) and impl.bare[index] == '1'


def overload_pick(impls: list[ImplSig], pin: str,
                  shapes: Sequence[str] = ()) -> str | None:
    """
    Return type of the implementation a PINNED call site selects (``overload_choice``).

    :param impls: Every implementation of the group, in declaration order
    :param pin: The call site's pin, one character per positional argument
    :param shapes: The static type of each positional argument, where the caller
                   has them; empty when it does not
    :return: The selected implementation's return type, or None
    """
    choice = overload_choice(impls, pin, shapes)
    return None if choice is None else choice.ret


def overload_choice(impls: list[ImplSig], pin: str,
                    shapes: Sequence[str] = ()) -> ImplSig | None:
    """
    The implementation a PINNED call site selects.

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
    WITHIN one half, so an earlier one that could take the witnesses too makes
    the site unanswerable rather than overruling the runtime's own choice. A
    parameter without an annotation takes anything (``ImplSig.bare``), so the
    earlier implementation is then the one that runs.

    A position the pin carries no witness for (``PIN_ANY``) is read from the
    argument's static type, the way ``_takes_position`` reads it: by the kind of
    value it is, and by its whole declared type where every value of that type
    passes the selector's check of it.

    It is the exact pass of ``resolve_pin``, the same rules read the same way.

    :param impls: Every implementation of the group, in declaration order
    :param pin: The call site's pin, one character per positional argument
    :param shapes: The static type of each positional argument, where the caller
                   has them; empty when it does not
    :return: The selected implementation, or None
    """
    return _pass_choice(impls, tuple(pin), shapes, True)[1]


def _pass_choice(impls: Sequence[ImplSig], wanted: tuple[str, ...], shapes: Sequence[str],
                 strict: bool) -> tuple[int, ImplSig | None]:
    """
    How one pass of the selector answers a call: its positional half, then its binding half.

    Each half takes the first implementation, in declaration order, that takes
    the call; an earlier one that may take it too leaves the pass unanswered.

    :param impls: Every implementation of the group, in declaration order
    :param wanted: One character per positional argument
    :param shapes: The static type of each positional argument, or nothing
    :param strict: The exact pass; the widening pass otherwise
    :return: ``(_YES, implementation)`` where the pass runs it, ``(_MAYBE, None)``
             where it may run one this cannot name, ``(_NO, None)`` where it runs none
    """
    for exact_arity in (True, False):
        for impl in impls:
            verdict = _takes(impl, wanted, exact_arity, shapes, strict)
            if verdict == _NO:
                continue
            return (_YES, impl) if verdict == _YES else (_MAYBE, None)
    return _NO, None


def resolve_pin(impls: Sequence[ImplSig], pin: str) -> ImplSig | None:
    """
    The implementation a pinned call site runs, selected from the static types.

    The selection follows ``core/overload.py``'s value selector (``_select``)
    on one value of each pinned type, rule for rule, and its answer is the
    index the runtime binds the call site to. A pin without a wildcard takes
    the selector's exact pass (``overload_choice``), and where no
    implementation takes the types there, its widening pass, which takes an
    int where a float is declared and a float where an int is, the defaults
    checked the same way. A pin with wildcards ignores those positions and is
    an answer only where exactly one implementation takes the rest, in the
    first pass any implementation does: where more than one survives, the
    position the pin knows nothing about is the one that decides, and the
    values know it and the pin does not.

    There is no answer where the runtime has no witness for a character, where
    a declared type or a default does not say how the selector checks a value
    (``_MAYBE``), and where the implementations are not the whole group in its
    registration order (``ImplSig.slot``, ``ImplSig.size``): the one the static
    selection leaves out may still be the one the values select. A call site
    without an answer dispatches from its values.

    :param impls: Every implementation of the group, in registration order
    :param pin: The call site's pin; an index after it is not read
    :return: The implementation the runtime runs, or None where this cannot say
             or the runtime runs none by the pin
    """
    chars = pin.partition(PIN_INDEX)[0]
    size = len(impls)
    if not size or any(impl.size != size or impl.slot != slot for slot, impl in enumerate(impls)) \
            or any(char not in PIN_WITNESSED and char != PIN_ANY for char in chars):
        return None
    wanted = tuple(chars)
    if PIN_ANY in wanted:
        for strict in (True, False):
            verdicts = [_takes(impl, wanted, False, (), strict, True) for impl in impls]
            found = [impl for impl, verdict in zip(impls, verdicts) if verdict == _YES]
            if _MAYBE in verdicts or len(found) > 1:
                return None
            if found:
                return found[0]
        return None
    for strict in (True, False):
        verdict, chosen = _pass_choice(impls, wanted, (), strict)
        if verdict != _NO:
            return chosen
    return None


def overload_candidates(impls: list[ImplSig], pin: str,
                        shapes: Sequence[str] = ()) -> list[int] | None:
    """
    The implementations the runtime selector can run for a call site, by index.

    The walk is ``overload_choice``'s -- the exact pass, positional half first,
    the implementations in declaration order -- but an implementation that MAY
    take the call joins the set instead of ending the walk, and the first one
    that surely takes it closes the set: the exact pass selects the first
    implementation that takes the call, so it runs one of those collected up
    to there.

    Where no implementation surely takes the call, the exact pass may take
    none, and the selector's widening pass (``core/overload._select``) then
    runs the first implementation that takes the arguments with an int where a
    float is wanted, or a float where an int is. The set still holds every
    implementation that can run only where each one left out is out of reach
    of that pass too (``_out_of_reach``); otherwise the answer is None.

    :param impls: Every implementation of the group, in declaration order
    :param pin: One character per positional argument: the call site's pin, or
                its arguments read as values (``value_chars``)
    :param shapes: The static type of each positional argument, where the caller
                   has them; empty when it does not
    :return: The indices of the implementations that can run, in the order the
             walk met them, or None where the runtime may run another one
    """
    wanted = tuple(pin)
    found: list[int] = []
    for exact_arity in (True, False):
        for index, impl in enumerate(impls):
            verdict = _takes(impl, wanted, exact_arity, shapes)
            if verdict == _NO:
                continue
            if index not in found:
                found.append(index)
            if verdict == _YES:
                return found
    if not found or not all(index in found or _out_of_reach(impl, wanted, shapes)
                            for index, impl in enumerate(impls)):
        return None
    return found


def _out_of_reach(impl: ImplSig, wanted: tuple[str, ...], shapes: Sequence[str]) -> bool:
    """
    Whether the selector's widening pass rejects an implementation for every value.

    The widening pass binds the arguments the way the exact pass does, so an
    implementation the call's arity rules out stays out. A parameter it fills
    stays out when the argument's kind of value cannot meet it in that pass
    either (``_widening_rejects``). A default the exact pass rejects is no
    proof: the widening pass may take an int default for a float parameter.

    :param impl: The implementation's shape
    :param wanted: The call's characters, one per positional argument
    :param shapes: The static type of each positional argument, or nothing
    :return: True when the widening pass never runs the implementation
    """
    if impl.open_ended or impl.kwarg:
        return False
    argc = len(wanted)
    if not impl.required <= argc <= len(impl.params) and argc != len(impl.fits):
        return True
    return any(_widening_rejects(declared, _given_at(want, shapes, index))
               for index, (declared, want) in enumerate(zip(impl.params, wanted)))


def _given_at(want: str, shapes: Sequence[str], index: int) -> str:
    """
    The static type of one argument: its shape where the caller has it, else
    the type its character stands for.

    :param want: The argument's character
    :param shapes: The static type of each positional argument, or nothing
    :param index: The argument's position
    :return: The type, UNKNOWN where neither says one
    """
    given = shapes[index] if index < len(shapes) else UNKNOWN
    if given != UNKNOWN:
        return given
    if want in PINNABLE:
        return want
    return _PIN_CONTAINER_TYPES.get(want, UNKNOWN)


def _widening_rejects(declared: str, given: str) -> bool:
    """
    Whether the widening pass rejects every value of a type for a parameter.

    Values of different kinds (``_kind``) never meet, with one exception: an
    ``NA`` object -- the na of a class, of a container, and the bool na of a
    script that keeps a third bool state -- passes for an ``int``, ``float``,
    ``str`` or ``bool`` parameter in that pass (``core/overload._check_type``
    answers ``not strict`` for it). A number's na is a NaN and a string's is
    ``''``, plain values of their own kind. A color may be the typeless na,
    which no pass rejects. A same-kind mismatch never qualifies: an int meets a
    float parameter there, and a container's sampled element is checked the
    same way.

    :param declared: The parameter's type
    :param given: The argument's static type
    :return: True when no value of ``given`` passes
    """
    kind = _kind(given)
    declared_kind = _kind(declared)
    if kind is None or declared_kind is None or kind == declared_kind or given == COLOR:
        return False
    return not (declared in (INT, FLOAT, BOOL, STR) and given not in (INT, FLOAT, STR))


def _takes(impl: ImplSig, wanted: tuple[str, ...], exact_arity: bool,
           shapes: Sequence[str] = (), strict: bool = True, wild: bool = False) -> int:
    """
    How one implementation answers a pin in one half of one pass of the selector.

    A position without an annotation takes every argument (``ImplSig.bare``);
    every other one answers by its type (``_position``).

    The positional half reaches an implementation whose visible parameters the
    call's arguments number exactly, and checks each argument against the
    parameter in its position: where a keyword-only parameter is among them, it
    is checked against that one, which is not read here. A ``*args`` or a
    ``**kwargs`` counts as a parameter there and takes a shape this does not
    model in the binding half either.

    :param impl: The implementation's shape
    :param wanted: The pin, one character per positional argument
    :param exact_arity: The positional half, which only reaches an
                        implementation whose every parameter the call fills
    :param shapes: The static type of each positional argument, or nothing
    :param strict: The exact pass; the widening pass otherwise
    :param wild: Whether a wildcard position is left out of the match
                 (``resolve_pin``) instead of being read from its static type
    :return: ``_YES``, ``_MAYBE`` or ``_NO``
    """
    if impl.open_ended or impl.kwarg:
        return _MAYBE
    argc = len(wanted)
    if exact_arity:
        # The positional half counts EVERY visible parameter against the
        # argument count, a keyword-only one included
        if len(impl.fits) != argc:
            return _NO
        if len(impl.params) != argc:
            return _MAYBE
    elif not impl.required <= argc <= len(impl.params):
        return _NO
    verdict = _YES
    for index, (declared, want) in enumerate(zip(impl.params, wanted)):
        if _is_bare(impl, index) or (wild and want == PIN_ANY):
            continue
        taken = _position(impl, index, declared, want,
                          shapes[index] if index < len(shapes) else UNKNOWN, strict)
        if taken == _NO:
            return _NO
        verdict = min(verdict, taken)
    if exact_arity:
        return verdict
    for fit in (impl.fits if strict else _wide_fits(impl))[argc:]:
        if fit == FIT_REQUIRED:
            return _NO
        if fit == FIT_UNSURE:
            verdict = _MAYBE
    return verdict


def _wide_fits(impl: ImplSig) -> str:
    """
    What each parameter of an implementation contributes to the widening pass when it is omitted.

    Where the shape does not carry it (``ImplSig.wide_fits``), it follows from
    the exact pass's: a default that pass takes is taken here too, a parameter
    with no default at all is required in both, and a default that pass rejects
    may or may not be taken.

    :param impl: The implementation's shape
    :return: One ``FIT_*`` character per parameter, ``ImplSig.fits`` order
    """
    if len(impl.wide_fits) == len(impl.fits):
        return impl.wide_fits
    return ''.join(fit if fit != FIT_REQUIRED or index < impl.required else FIT_UNSURE
                   for index, fit in enumerate(impl.fits))


def _position(impl: ImplSig, index: int, declared: str, want: str, given: str,
              strict: bool) -> int:
    """
    How one positional parameter of an implementation answers one argument in one pass.

    A witness is checked the way the shape's acceptance spec says
    (``ImplSig.accepts``), or else the way the declared type says
    (``_takes_position``, ``_widens_position``). A wildcard position is read
    from its static type in the exact pass and is not modelled in the widening
    pass.

    :param impl: The implementation's shape
    :param index: The parameter's position
    :param declared: The parameter's type
    :param want: The argument's pin character
    :param given: The argument's static type, for a position without a witness
    :param strict: The exact pass; the widening pass otherwise
    :return: ``_YES``, ``_MAYBE`` or ``_NO``
    """
    if want == PIN_ANY:
        return _takes_position(declared, want, given) if strict else _MAYBE
    spec = impl.accepts[index] if index < len(impl.accepts) else ''
    if spec == ACCEPTS_UNKNOWN:
        return _MAYBE
    if spec:
        exact, _, wide = spec.partition(ACCEPTS_SEP)
        return _YES if want in (exact if strict else wide) else _NO
    return _takes_position(declared, want, given) if strict else _widens_position(declared, want)


def _widens_position(declared: str, want: str) -> int:
    """
    How one parameter answers a witness in the selector's widening pass, by its declared type.

    The pass takes what the exact pass takes (``_takes_position``), and an int
    where a float is declared or a float where an int is: a number, and the
    sampled element of an array or a matrix of numbers alike.

    :param declared: The parameter's type
    :param want: The argument's pin character, one with a witness
    :return: ``_YES``, ``_MAYBE`` or ``_NO``
    """
    exact = _takes_position(declared, want, UNKNOWN)
    if exact != _NO:
        return exact
    shaped = _PIN_CONTAINER_TYPES.get(want)
    if shaped is None:
        return _YES if head(declared) in NUMERIC and want in NUMERIC else _NO
    if (is_array(declared) and is_array(shaped)) or (is_matrix(declared) and is_matrix(shaped)):
        return _YES if element_of(declared) in NUMERIC and element_of(shaped) in NUMERIC else _NO
    return _NO


def _takes_position(declared: str, want: str, given: str) -> int:
    """
    How one parameter answers one argument of the exact pass.

    A position without a witness (``PIN_ANY``) is read from the argument's
    static type, for every value that type admits at run time: a value of the
    type, or the na the conversion pass builds for it (``implicit_conversion``)
    -- a ``NaN`` for a number, ``''`` for a string, ``NA(<class>)`` for an
    object, ``NA(list)`` / ``NA(dict)`` / ``NA(Matrix)`` for a container. The
    selector checks such a value against the parameter's type
    (``core/overload._check_type``), and the answer follows from what it checks:

    * A typeless ``na`` is taken for whatever the position declares.
    * An object of a known class is taken by a parameter of its class alone.
    * A container is checked by what it is -- a ``list``, a ``dict``, a
      ``Matrix`` -- so a parameter of another kind (``_kind``) never takes it,
      and one of the same kind only for some values: it is then matched by a
      SAMPLED element, which an empty one does not have (it passes for the
      first ``list[...]`` parameter whatever its element). Where every element
      type it spells passes its own check -- an object, a string, a bool, a
      color, not a number (``_strictly_held``) -- the declared type itself is
      taken for sure.
    * A color is taken by a color parameter. Anything else may take it too:
      the compiler leaves a color ``na`` as the typeless ``na``
      (``pine_type_infer._na_widen`` converts none for a color), which the
      exact pass takes for every parameter.

    :param declared: The parameter's type
    :param want: The argument's pin character
    :param given: The argument's static type, for a position without a witness
    :return: ``_YES``, ``_MAYBE`` or ``_NO``
    """
    if want == PIN_ANY:
        if given == TYPELESS:
            # A typeless na: the selector takes it for whatever the position
            # declares (``_check_type``), so the first implementation the other
            # positions let through runs, exactly as at run time
            return _MAYBE if declared == UNKNOWN else _YES
        # No witness: an object of a known class is taken by its class
        # alone, anything else by whatever declares the position
        if class_of(given) is not None and declared not in (UNKNOWN, OBJECT):
            return _YES if declared == given else _NO
        if given == COLOR:
            return _YES if declared == COLOR else _MAYBE
        declared_kind = _kind(declared)
        if not (is_array(given) or is_matrix(given) or is_map(given)) or declared_kind is None:
            return _MAYBE
        if _kind(given) != declared_kind:
            return _NO
        return _YES if declared == given and _strictly_held(given) else _MAYBE
    shaped = _PIN_CONTAINER_TYPES.get(want)
    if shaped is not None:
        # The witness is a one-element container, which the selector takes
        # by its container and that element: exactly the declared shape
        # takes it, a bare or element-unknown one of the same container may
        if declared == shaped:
            return _YES
        if element_of(shaped) == STR and is_str_literal(element_of(declared)) and (
                is_array(declared) and is_array(shaped)
                or is_matrix(declared) and is_matrix(shaped)):
            return _YES
        if declared in (UNKNOWN, OBJECT) or (
                (is_array(declared) and is_array(shaped)
                 or is_matrix(declared) and is_matrix(shaped))
                and element_of(declared) == UNKNOWN):
            return _MAYBE
        return _NO
    if want == STR and is_str_literal(declared):
        # The selector takes any string for a ``StrLiteral`` subclass
        return _YES
    declared = head(declared)
    if declared == UNKNOWN:
        return _MAYBE
    return _YES if declared == want else _NO


def _kind(ty: str) -> str | None:
    """
    The kind of value the selector checks a value of a type as.

    The selector's check of a value starts with its Python type: a number is an
    ``int`` or a ``float`` (either one for either Pine type), a container a
    ``list``, a ``dict`` or a ``Matrix``, an object its class. Two types of
    different kinds have no value in common. A lib constant class is no kind
    here: its constants are strings or ints themselves, which a ``str`` or an
    ``int`` parameter takes.

    :param ty: Any type
    :return: The kind, or None where the type does not say one
    """
    if ty in NUMERIC:
        return FLOAT
    if ty in (BOOL, STR, COLOR):
        return ty
    if is_array(ty):
        return _ARRAY
    if is_matrix(ty):
        return _MATRIX
    if is_map(ty):
        return _MAP
    if class_of(ty) is not None and constant_scalar(ty) == ty:
        return ty
    return None


def _strictly_held(ty: str) -> bool:
    """
    Whether every value of a type passes the exact pass's check against the type itself.

    A number does not: an int-typed value may be a Python float, a float-typed
    one a Python int. An object, a string, a bool and a color do, their na
    included (a color na is the typeless na, which passes everything). A
    container does where every element type it spells does, since the selector
    checks a sampled element against the declared one.

    :param ty: Any type
    :return: True when the declared type takes every value of itself
    """
    if ty in (BOOL, STR, COLOR):
        return True
    if is_array(ty) or is_matrix(ty):
        return _strictly_held(element_of(ty))
    if is_map(ty):
        return len(ty) > len(_MAP) and _strictly_held(key_of(ty)) \
            and _strictly_held(value_of(ty))
    return class_of(ty) is not None and constant_scalar(ty) == ty


def pine_choice(impls: Sequence[ImplSig], shapes: Sequence[str]) -> int | None:
    """
    The implementation Pine itself selects for a positional call, by its index in ``impls``.

    Pine resolves an overload from the STATIC argument types: an implementation
    whose every parameter declares exactly the type of its argument first, and
    only where none does, one that takes them with an int argument where a
    float is declared. That is decided only where every argument has a settled
    type (``_pine_settled``) and every implementation the call's arity reaches
    declares every position the call fills -- an unannotated or unreadable
    parameter, or a ``*args``, takes arguments by a rule this does not model --
    and where exactly one implementation answers the first of the two passes
    that any implementation answers.

    The runtime's value dispatch can run another implementation for the same
    types (an empty container matches the first implementation of its kind, a
    float-typed value holding a Python int takes an int parameter); an indexed
    pin (``indexed_pin``) is what makes it run this one.

    :param impls: Every implementation of the group the static selection reads,
                  in declaration order
    :param shapes: The static type of each positional argument
    :return: The index of the selected implementation in ``impls``, or None
    """
    argc = len(shapes)
    if not all(_pine_settled(given) for given in shapes):
        return None
    reachable: list[int] = []
    for index, impl in enumerate(impls):
        if impl.open_ended:
            return None
        if not impl.required <= argc <= len(impl.params):
            continue
        if any(_is_bare(impl, position) or not _pine_settled(impl.params[position])
               for position in range(argc)):
            return None
        reachable.append(index)
    for widening in (False, True):
        found = [index for index in reachable
                 if all(_pine_takes(impls[index].params[position], given, widening)
                        for position, given in enumerate(shapes))]
        if found:
            return found[0] if len(found) == 1 else None
    return None


def value_rival(impl: ImplSig, shapes: Sequence[str], numbers: bool) -> bool:
    """
    Whether the value dispatch may run an implementation for a reason only an indexed pin overrules.

    Pine never selects such an implementation for these argument types, and
    the selector can still run it from the values: where it declares another
    element type of the container kind an argument is (an empty container
    matches the first implementation of its kind, ``_takes_position``), and,
    for a call no pin reaches (``numbers``), where it declares an int for a
    float argument or the other way round (a float-typed value may be a Python
    int, an int-typed one a Python float).

    :param impl: The implementation's shape
    :param shapes: The static type of each positional argument
    :param numbers: Whether the call is selected from its values alone
    :return: True when one of its positions is such a rival of the argument
    """
    for declared, given in zip(impl.params, shapes):
        if declared == given:
            continue
        if is_array(given) or is_matrix(given) or is_map(given):
            if _kind(declared) == _kind(given):
                return True
        elif numbers and given in (INT, FLOAT) and declared in (INT, FLOAT):
            return True
    return False


def _pine_settled(ty: str) -> bool:
    """
    Whether a type is one Pine's overload resolution compares as it stands.

    :param ty: Any type
    :return: True for a scalar, a color, an object of a known class, and a
             container whose element types are all of those
    """
    if ty in (INT, FLOAT, BOOL, STR, COLOR):
        return True
    if is_array(ty) or is_matrix(ty):
        return _pine_settled(element_of(ty))
    if is_map(ty):
        return _pine_settled(key_of(ty)) and _pine_settled(value_of(ty))
    return class_of(ty) is not None


def _pine_takes(declared: str, given: str, widening: bool) -> bool:
    """
    Whether Pine passes an argument of one type to a parameter of another.

    A lib constant (``format.volume``, ``text.align_right``) is the scalar Pine
    spells it as (``constant_scalar``).

    :param declared: The parameter's type
    :param given: The argument's static type
    :param widening: The second pass, which takes an int for a float
    :return: True when the parameter takes the argument in that pass
    """
    if declared == given or constant_scalar(given) == declared:
        return True
    return widening and given == INT and declared == FLOAT


def overload_bind(impls: list[ImplSig], chars: str, shapes: Sequence[str],
                  kw_chars: Mapping[str, str], kw_shapes: Mapping[str, str]) -> str | None:
    """
    Return type of the implementation a call WITH KEYWORD ARGUMENTS selects
    (``overload_binding``).

    :param impls: Every implementation of the group, in declaration order
    :param chars: One pin character per positional argument
    :param shapes: The static type of each positional argument
    :param kw_chars: Parameter name -> the keyword argument's pin character
    :param kw_shapes: Parameter name -> the keyword argument's static type
    :return: The selected implementation's return type, or None
    """
    choice = overload_binding(impls, chars, shapes, kw_chars, kw_shapes)
    return None if choice is None else choice.ret


def overload_binding(impls: list[ImplSig], chars: str, shapes: Sequence[str],
                     kw_chars: Mapping[str, str],
                     kw_shapes: Mapping[str, str]) -> ImplSig | None:
    """
    The implementation a call WITH KEYWORD ARGUMENTS selects.

    Such a call is never pinned; the selector binds its arguments to each
    implementation's parameters in declaration order, applies the defaults, and
    takes the first implementation whose every bound value fits its parameter's
    type in the exact pass (``core/overload._select``, the keyword path). This is
    that pass read statically, with the positions read the way ``overload_pick``
    reads them, so an implementation an earlier one could take too leaves the
    site unanswerable.

    :param impls: Every implementation of the group, in declaration order
    :param chars: One pin character per positional argument
    :param shapes: The static type of each positional argument
    :param kw_chars: Parameter name -> the keyword argument's pin character
    :param kw_shapes: Parameter name -> the keyword argument's static type
    :return: The selected implementation, or None
    """
    for impl in impls:
        verdict = _binds(impl, chars, shapes, kw_chars, kw_shapes)
        if verdict == _NO:
            continue
        return impl if verdict == _YES else None
    return None


def _binds(impl: ImplSig, chars: str, shapes: Sequence[str],
           kw_chars: Mapping[str, str], kw_shapes: Mapping[str, str]) -> int:
    """
    How one implementation answers a call with keyword arguments.

    A position without an annotation takes every argument, positional or
    keyword (``ImplSig.bare``), as in ``_takes``.

    :param impl: The implementation's shape
    :param chars: One pin character per positional argument
    :param shapes: The static type of each positional argument
    :param kw_chars: Parameter name -> the keyword argument's pin character
    :param kw_shapes: Parameter name -> the keyword argument's static type
    :return: ``_YES``, ``_MAYBE`` or ``_NO``
    """
    if impl.open_ended or impl.kwarg:
        return _MAYBE
    if len(chars) > len(impl.params):
        return _NO
    verdict = _YES
    bound: set[int] = set(range(len(chars)))
    for index, (declared, want) in enumerate(zip(impl.params, chars)):
        if _is_bare(impl, index):
            continue
        taken = _position(impl, index, declared, want,
                          shapes[index] if index < len(shapes) else UNKNOWN, True)
        if taken == _NO:
            return _NO
        verdict = min(verdict, taken)
    for name, want in kw_chars.items():
        if name not in impl.names:
            # A keyword-only parameter is not named in the shape: one may take
            # the keyword, none means the bind fails
            return _MAYBE if len(impl.fits) > len(impl.params) else _NO
        index = impl.names.index(name)
        if index in bound:
            return _NO
        bound.add(index)
        if _is_bare(impl, index):
            continue
        taken = _position(impl, index, impl.params[index], want, kw_shapes.get(name, UNKNOWN),
                          True)
        if taken == _NO:
            return _NO
        verdict = min(verdict, taken)
    for index, fit in enumerate(impl.fits):
        if index in bound:
            continue
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
             classes: Mapping[str, str] = NO_CLASSES,
             enums: frozenset[str] = NO_ENUMS, slot: int = -1) -> ImplSig:
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
    :param enums: The class ids of the enums among them
    :param slot: Its position in its group's registration order, -1 for none
    :return: The implementation's shape
    """
    args = node.args
    positional = list(args.posonlyargs) + list(args.args)
    every = positional + list(args.kwonlyargs)
    defaults = _param_defaults(node)
    params = tuple(annotation_type(a.annotation, classes, enums) for a in positional)
    accepts = tuple(_own_accepts(arg, declared, classes)
                    for arg, declared in zip(positional, params))
    return ImplSig(
        params=params,
        required=len(positional) - len(args.defaults),
        open_ended=args.vararg is not None, ret=ret,
        fits=''.join(param_fit(arg, defaults.get(arg.arg), ty_of, classes, enums)
                     for arg in every),
        names=tuple(a.arg for a in positional),
        bare=''.join('1' if a.annotation is None else '0' for a in positional), slot=slot,
        accepts=accepts if any(accepts) else (),
        wide_fits=''.join(param_fit(arg, defaults.get(arg.arg), ty_of, classes, enums, False)
                          for arg in every),
        kwarg=args.kwarg is not None)


def param_fit(arg: ast.arg, default: ast.expr | None,
              ty_of: Callable[[ast.AST], str] = get_ty,
              classes: Mapping[str, str] = NO_CLASSES,
              enums: frozenset[str] = NO_ENUMS, strict: bool = True) -> str:
    """
    What one parameter contributes to a call that omits it.

    An UNANNOTATED parameter contributes nothing: the selector has no declared
    type to check its default against, so it takes whatever the default is.
    That is not the same as an annotation this pass cannot read, which
    ``default_fit`` declines on.

    A scalar default of a parameter whose annotation the selector checks
    otherwise than its Pine type says (``annotation_accepts``: a union, a
    ``StrLiteral``) is checked the way the annotation says, like a witness of
    its type.

    :param arg: The parameter
    :param default: Its default expression, or None when it has none
    :param ty_of: Types the default expression
    :param classes: Class name -> class id, visible where the annotation stands
    :param enums: The class ids of the enums among them
    :param strict: The selector's exact pass; its widening pass otherwise
    :return: One of the ``FIT_*`` characters
    """
    if default is None:
        return FIT_REQUIRED
    if arg.annotation is None:
        return FIT_OMISSIBLE
    declared = annotation_type(arg.annotation, classes, enums)
    ty = default_ty(default, ty_of)
    if ty in PINNABLE:
        spec = _own_accepts(arg, declared, classes)
        if spec == ACCEPTS_UNKNOWN:
            return FIT_UNSURE
        if spec:
            exact, _, wide = spec.partition(ACCEPTS_SEP)
            return FIT_OMISSIBLE if ty in (exact if strict else wide) else FIT_REQUIRED
    fit = default_fit if strict else _default_fit_wide
    return fit(declared, ty, annotation_takes_none(arg.annotation))


def _own_accepts(arg: ast.arg, declared: str, classes: Mapping[str, str]) -> str:
    """
    The acceptance spec ``ImplSig.accepts`` records for one parameter.

    Nothing where the declared type reads the witnesses the way the selector
    checks them (``_derived_accepts``), so a shape stays as it was wherever the
    annotation is an ordinary one, and nothing for an annotation whose type is
    not known either, which reads every witness as undecided already.

    :param arg: The parameter
    :param declared: Its declared type
    :param classes: Class name -> class id, visible where the annotation stands
    :return: ``''``, ``ACCEPTS_UNKNOWN`` or a spec
    """
    if arg.annotation is None or declared == UNKNOWN:
        return ''
    spec = annotation_accepts(arg.annotation, classes)
    return '' if spec == _derived_accepts(declared) else spec


def _derived_accepts(declared: str) -> str | None:
    """
    The acceptance spec a declared type reads as (``_takes_position``, ``_widens_position``).

    :param declared: A parameter's declared type
    :return: The spec, or None where the type leaves a witness undecided
    """
    passes: list[str] = []
    for strict in (True, False):
        taken = ''
        for want in PIN_WITNESSED:
            verdict = _takes_position(declared, want, UNKNOWN) if strict \
                else _widens_position(declared, want)
            if verdict == _MAYBE:
                return None
            if verdict == _YES:
                taken += want
        passes.append(taken)
    return ACCEPTS_SEP.join(passes)


#: Every witness character, in both passes: what ``Any`` and ``object`` take
_EVERY_WITNESS: Final = PIN_WITNESSED + ACCEPTS_SEP + PIN_WITNESSED

#: No witness, in either pass: what a class, a map, a tuple and ``None`` take
_NO_WITNESS: Final = ACCEPTS_SEP

#: The acceptance spec of an annotation NAME, MEASURED with
#: ``core/overload._check_type`` on the witness values (``test_107`` checks it).
#: ``PyneInt`` is ``float | int | Series[int]``, so it takes a float in the exact
#: pass; an ``NA`` takes no witness at all.
_NAME_ACCEPTS: Final[dict[str, str]] = {
    'int': 'i/if', 'float': 'f/if', 'bool': 'b/b', 'str': 's/s',
    'PyneInt': 'if/if', 'PyneFloat': 'f/if', 'PyneBool': 'b/b', 'PyneStr': 's/s',
    'Any': _EVERY_WITNESS, 'object': _EVERY_WITNESS, 'StrLiteral': 's/s',
    'list': 'IFBS/IFBS', 'Matrix': '1234/1234',
    'Color': _NO_WITNESS, 'NA': _NO_WITNESS, 'dict': _NO_WITNESS,
    'tuple': _NO_WITNESS, 'set': _NO_WITNESS, 'NoneType': _NO_WITNESS,
}

#: Annotation heads whose payload is checked as the annotation itself: the
#: ``Persistent`` wrappers are ``T | NA[T] | Series[T]``, and only ``T`` takes a
#: witness; a parameter's ``Series[T]`` is ``T`` at run time, the series
#: transform leaves the payload in its place (``SeriesTransformer``)
_SAME_ACCEPTS_SUBSCRIPTS: Final = frozenset({'Persistent', 'IBPersistent', 'Optional', 'Series'})

#: Annotation heads that take no witness whatever their payload
_NO_WITNESS_SUBSCRIPTS: Final = frozenset({'NA', 'dict', 'tuple', 'Tuple', 'set'})

#: Container annotation head -> the witness character of each scalar element
_CONTAINER_WITNESSES: Final[dict[str, dict[str, str]]] = {
    'list': {INT: 'I', FLOAT: 'F', BOOL: 'B', STR: 'S'},
    'Matrix': {INT: '1', FLOAT: '2', BOOL: '3', STR: '4'},
}


def annotation_accepts(node: ast.expr, classes: Mapping[str, str] = NO_CLASSES) -> str:
    """
    Which witness values a parameter annotation takes, in each pass of the selector.

    This is ``core/overload._check_type`` read off the annotation: a union
    takes what any of its members takes, ``list[E]`` the array witness whose
    element ``E`` takes, a class none (no witness is an instance of a class) --
    except a lib ``StrLiteral`` class, which takes any string. An annotation
    this does not model is ``ACCEPTS_UNKNOWN``.

    :param node: The annotation expression
    :param classes: Class spelling -> class id, visible where the annotation stands
    :return: ``<exact>/<widening>``, the witness characters each pass takes, or
             ``ACCEPTS_UNKNOWN``
    """
    if isinstance(node, ast.Constant):
        if node.value is None:
            return _NO_WITNESS
        if isinstance(node.value, str):
            try:
                return annotation_accepts(ast.parse(node.value, mode='eval').body, classes)
            except SyntaxError:
                return ACCEPTS_UNKNOWN
        return ACCEPTS_UNKNOWN
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _union_accepts([node.left, node.right], classes)
    if isinstance(node, ast.Subscript):
        value = node.value
        name = value.id if isinstance(value, ast.Name) else \
            (value.attr if isinstance(value, ast.Attribute) else '')
        if name == 'Union':
            members = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            return _union_accepts(list(members), classes)
        if name in _SAME_ACCEPTS_SUBSCRIPTS:
            return annotation_accepts(node.slice, classes)
        if name in _NO_WITNESS_SUBSCRIPTS:
            return _NO_WITNESS
        witnesses = _CONTAINER_WITNESSES.get(name)
        if witnesses is None:
            return ACCEPTS_UNKNOWN
        element = annotation_accepts(node.slice, classes)
        if element == ACCEPTS_UNKNOWN:
            return ACCEPTS_UNKNOWN
        return ACCEPTS_SEP.join(
            ''.join(witnesses[char] for char in taken if char in witnesses)
            for taken in element.split(ACCEPTS_SEP))
    if isinstance(node, (ast.Name, ast.Attribute)):
        spelled = dotted_name(node)
        leaf = node.id if isinstance(node, ast.Name) else node.attr
        found = classes.get(spelled) if spelled is not None else None
        if found is None and leaf in _NAME_ACCEPTS:
            return _NAME_ACCEPTS[leaf]
        if found is None:
            found = classes.get(leaf)
        if found is None and leaf in BUILTIN_CLASSES:
            return 's/s' if is_str_literal(BUILTIN_CLASSES[leaf]) else _NO_WITNESS
        if found is None:
            return ACCEPTS_UNKNOWN
        return 's/s' if is_str_literal(object_ty(found)) else _NO_WITNESS
    return ACCEPTS_UNKNOWN


def _union_accepts(members: list[ast.expr], classes: Mapping[str, str]) -> str:
    """
    The acceptance spec of a union: what any of its members takes.

    :param members: The union's members
    :param classes: Class spelling -> class id, visible where the annotation stands
    :return: The spec, or ``ACCEPTS_UNKNOWN`` where a member's is not known
    """
    exact = wide = ''
    for member in members:
        spec = annotation_accepts(member, classes)
        if spec == ACCEPTS_UNKNOWN:
            return ACCEPTS_UNKNOWN
        member_exact, _, member_wide = spec.partition(ACCEPTS_SEP)
        exact += member_exact
        wide += member_wide
    return ''.join(char for char in PIN_WITNESSED if char in exact) + ACCEPTS_SEP + \
        ''.join(char for char in PIN_WITNESSED if char in wide)


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
    dotted = dotted_name(default)
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


def is_na_literal(node: ast.AST) -> bool:
    """
    Whether a value spells Pine's typeless ``na``: the name itself (``NA_LITERALS``), a
    signed one included -- the na object's negation hands the na itself back.

    :param node: An expression
    :return: True when the value is the typeless na whatever runs
    """
    while isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        node = node.operand
    return isinstance(node, (ast.Name, ast.Attribute)) and dotted_name(node) in NA_LITERALS


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
#: ``'vwap'``            ``ta.vwap``, a float, or a tuple of three once its
#:                       band multiplier is passed
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
    # MEASURED (2026-09-25, compile-only ``int a = <expr>``): ``math.sum(bar_index, 3)``,
    # ``math.avg(bar_index, 2)``, ``math.avg(1, 2)`` and ``math.sign(bar_index)`` are all
    # rejected as float -- none of them follows its int arguments
    'math.sum': FLOAT,
    'math.avg': FLOAT,
    'math.sign': FLOAT,
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
    # The numeric na a typeless ``na`` becomes where a float or an int goes
    # (``implicit_conversion``)
    '_na_float': FLOAT,
    # MEASURED: ``int a = ta.change(bar_index)``, ``float b = ta.change(close)``
    # and ``bool c = ta.change(close > open)`` all compile, so the result is the
    # source's own type
    'ta.change': 'arg0',
    'fixnan': 'arg0',
    # MEASURED (2026-09-25, compile-only ``int a = <expr>``): ``ta.valuewhen(c, bar_index,
    # 0)``, ``ta.median(bar_index, 4)``, ``ta.range(bar_index, 4)`` and ``ta.mode(bar_index,
    # 4)`` compile, while the same calls over ``close`` are rejected as float, so each result
    # is its source's own type
    'ta.valuewhen': 'arg1',
    'ta.median': 'arg0',
    'ta.range': 'arg0',
    'ta.mode': 'arg0',
    # ``request.seed`` evaluates its expression in the seed's context, like
    # ``request.security``, and yields a value of the expression's own type
    'request.seed': 'arg2',
    # ``ta.vwap`` is one value, and the value with its upper and lower band once a
    # band multiplier is passed. Read bare (``ta.vwap``) it is the call without
    # arguments
    'ta.vwap': 'vwap',
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
    # An enum input holds a member of its default's enum
    'input.enum': 'arg0',

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
    'matrix.copy': 'same_matrix0',
    'matrix.transpose': 'same_matrix0',
    'matrix.inv': 'same_matrix0',
    'matrix.pinv': 'same_matrix0',
    'matrix.pow': 'same_matrix0',
    'matrix.submatrix': 'same_matrix0',
    # MEASURED (CAPITALCOM:EURUSD@60, a ``matrix<int>`` [[1, 2], [3, 4]]): the
    # eigen results keep the INT element type while holding fractional values.
    # ``array<float> v = matrix.eigenvalues(mi)`` and ``matrix<float> e =
    # matrix.eigenvectors(mi)`` are rejected with CE10173 as ``array<int>`` and
    # ``matrix<int>``, and ``array<int> v = mi.eigenvalues()`` compiles and
    # holds [-0.3722813233, 5.3722813233]
    'matrix.eigenvalues': 'array_of_elem0',
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

#: Declared parameter order of the lib names the registry holds no parameters
#: for: those whose module lives outside ``pynecore/lib``, and a property that
#: is also called, which the registry records as a value. A
#: type-preserving override names the parameter it copies from, and a keyword
#: spelling (``input(title='WMA Length', defval=10)``) only binds back to that
#: position with the order in hand.
OVERRIDE_PARAM_NAMES: Final[dict[str, list[str]]] = {
    # ``core/script.py::_Input.__call__``
    'input': ['defval', 'title', 'tooltip', 'inline', 'group', 'display', 'active'],
    # ``core/script.py::_Input.enum``
    'input.enum': ['defval', 'title', 'options', 'tooltip', 'inline', 'group', 'confirm',
                   'display', 'active'],
    # ``lib/ta.py::vwap``
    'ta.vwap': ['source', 'anchor', 'stdev_mult'],
}
