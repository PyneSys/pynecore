"""Shape the missing value of a ``request.security`` read after type inference."""
import ast

from . import ast_walk
from .pine_type_rules import (BOOL, FLOAT, NUMERIC, OBJECT, STR, elements_of, get_ty, is_tuple,
                              set_ty, stamp_lowering, tuple_of)


_BOOL_NA_NAME = '__sec_bool_na·__'


class SecurityDefaultTransformer(ast_walk.NodeTransformer):
    """Resolve bool defaults and tuple widths after type inference.

    The bool factory reads the running script's mode, so a library's defaults
    follow its caller just like bool series history and UDT fields do. A read of
    a tuple whose width the security transform could not see gets one ``na`` per
    element. Other defaults stay as the security transform supplied them.

    A ``request.security_lower_tf`` read of a tuple gives one array per element:
    a read the security transform left raw, because neither the expression nor an
    unpack showed the width (the read stands in a function's ``return``), gets the
    ``__ltf_unzip__`` transposition here, with the width of the written tuple.

    A tuple read unpacked into fewer names than the tuple has gives its leading
    values, as on TradingView: ``a, b = request.security(s, tf, f(1))`` with ``f``
    giving three values, also when the read stands in another function's
    ``return``. Such an unpack takes a ``[:n]`` slice of its value.
    """

    def __init__(self):
        self._used = False

    def _default(self, default: ast.expr, ty: str) -> ast.expr:
        """Build the missing value of a bool, or of a tuple.

        :param default: The default emitted by the security transform.
        :param ty: The inferred result type.
        :return: The default with bool elements resolved under the runtime mode.
        """
        if ty == BOOL:
            self._used = True
            return ast.copy_location(stamp_lowering(ast.Call(
                func=ast.Name(id=_BOOL_NA_NAME, ctx=ast.Load()),
                args=[], keywords=[]), BOOL), default)
        if is_tuple(ty):
            types = elements_of(ty)
            if isinstance(default, ast.Tuple) and len(default.elts) <= len(types):
                # The security transform took the width from the unpack target,
                # which may name fewer values than the tuple gives: the leading ones
                defaults = default.elts
                types = types[:len(defaults)]
                ty = tuple_of(types)
                values = [self._default(value, item) for value, item in zip(defaults, types)]
                if all(value is old for value, old in zip(values, defaults)):
                    return default
            elif not isinstance(default, (ast.Tuple, ast.List)):
                # A scalar default of a tuple read: the security transform could not
                # see the width (the expression is a call of a tuple-returning
                # function, and the read stands in a ``return``), the type pass did.
                # One ``na`` per element, or the unpack at the call site raises
                values = [self._element_default(default, item) for item in types]
            else:
                return default
            return ast.copy_location(stamp_lowering(
                ast.Tuple(elts=values, ctx=ast.Load()), ty), default)
        return default

    def _element_default(self, default: ast.expr, ty: str) -> ast.expr:
        """Build the missing value of one element of a tuple read from a scalar default.

        The na of the element's type, as the implicit conversion makes it from a
        typeless ``na`` in a tuple-literal default: the numeric na, the string na,
        the bool na of the runtime mode. Any other element keeps the typeless ``na``.

        :param default: The scalar default emitted by the security transform.
        :param ty: The element's inferred type.
        :return: A new node for the element.
        """
        if ty in NUMERIC:
            return set_ty(ast.copy_location(ast.Attribute(
                value=ast.Name(id='lib', ctx=ast.Load()), attr='_na_float', ctx=ast.Load()),
                default), FLOAT)
        if ty == STR:
            return set_ty(ast.copy_location(ast.Constant(value=''), default), STR)
        return self._default(ast_walk.clone(default), ty)

    @staticmethod
    def _unzip_lower_tf(node: ast.Module) -> bool:
        """Transpose the raw lower-timeframe reads of a tuple into one array per element.

        :param node: The module.
        :return: Whether a read was wrapped.
        """
        widths: dict[str, int | None] = {}
        for sub in ast_walk.walk(node):
            if not (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
                    and sub.func.id == '__sec_write__' and len(sub.args) == 2):
                continue
            key = sub.args[0]
            if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                continue
            ty = get_ty(sub.args[1])
            width = len(elements_of(ty)) if is_tuple(ty) else None
            sec_id = key.value
            widths[sec_id] = width if widths.get(sec_id, width) == width else None
        wrapped = False
        for parent in ast_walk.walk(node):
            if (isinstance(parent, ast.Call) and isinstance(parent.func, ast.Name)
                    and parent.func.id == '__ltf_unzip__'):
                continue
            for field, value in ast.iter_fields(parent):
                children = value if isinstance(value, list) else [value]
                for index, child in enumerate(children):
                    width = _lower_tf_width(child, widths)
                    if width is None:
                        continue
                    unzip = ast.copy_location(ast.Call(
                        func=ast.Name(id='__ltf_unzip__', ctx=ast.Load()),
                        args=[child, ast.Constant(value=width)], keywords=[]), child)
                    unzip = stamp_lowering(unzip, OBJECT)
                    if isinstance(value, list):
                        value[index] = unzip
                    else:
                        setattr(parent, field, unzip)
                    wrapped = True
        return wrapped

    @staticmethod
    def _take_leading(node: ast.Module):
        """Slice the requested tuple of an unpack that names fewer values than it gives.

        :param node: The module.
        """
        requested = requested_functions(node)
        returned = _returned_widths(node, requested)
        for stmt in ast_walk.walk_statements(node):
            if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
                continue
            target = stmt.targets[0]
            if (not isinstance(target, (ast.Tuple, ast.List))
                    or any(isinstance(elt, ast.Starred) for elt in target.elts)):
                continue
            value = stmt.value
            names = len(target.elts)
            if not is_requested(value, requested):
                continue
            ty = get_ty(value)
            if is_tuple(ty):
                width = len(elements_of(ty))
                ty = tuple_of(elements_of(ty)[:names])
            else:
                width = _width(value, returned)
            if width is None or width <= names:
                continue
            sliced = ast.copy_location(ast.Subscript(
                value=value,
                slice=ast.Slice(lower=None, upper=ast.Constant(value=names), step=None),
                ctx=ast.Load()), value)
            stmt.value = stamp_lowering(sliced, ty)

    def visit_Module(self, node: ast.Module) -> ast.Module:
        self._used = False
        # Security reads are only ever emitted together with the module's
        # ``__security_contexts__`` registry: without it there is nothing to
        # resolve, and most scripts have none.
        if not any(isinstance(stmt, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == '__security_contexts__'
                           for target in stmt.targets)
                   for stmt in node.body):
            return node
        # Each read is resolved on its own: a default never holds another
        # read, so neither the order nor a rebuild of the tree around them
        # matters, and a plain walk finds them all.
        for sub in ast_walk.walk(node):
            if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
                    and sub.func.id == '__sec_read__' and len(sub.args) == 2):
                sub.args[1] = self._default(sub.args[1], get_ty(sub))
        if (self._unzip_lower_tf(node)
                and not any(isinstance(stmt, ast.ImportFrom)
                            and stmt.module == 'pynecore.core.security'
                            and any(alias.name == '__ltf_unzip__' and alias.asname is None
                                    for alias in stmt.names)
                            for stmt in node.body)):
            node.body.insert(ast_walk.future_end(node.body), ast.ImportFrom(
                module='pynecore.core.security',
                names=[ast.alias(name='__ltf_unzip__', asname=None)], level=0))
        self._take_leading(node)
        if self._used:
            insert_at = 0
            first = node.body[0] if node.body else None
            if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                insert_at = 1
            while insert_at < len(node.body):
                stmt = node.body[insert_at]
                if not (isinstance(stmt, ast.ImportFrom) and stmt.module == '__future__'):
                    break
                insert_at += 1
            node.body.insert(insert_at, ast.ImportFrom(
                module='pynecore.types.na',
                names=[ast.alias(name='new_bool_na', asname=_BOOL_NA_NAME)], level=0))
        return node


def requested_functions(node: ast.Module) -> set[str]:
    """The names of the functions whose every ``return`` gives a security read.

    Directly (``return __sec_read__(...)``) or through another such function. A
    name defined more than once counts only when every definition qualifies.

    :param node: The module.
    :return: The function names.
    """
    definitions: dict[str, list[ast.FunctionDef]] = {}
    for sub in ast_walk.walk(node):
        if isinstance(sub, ast.FunctionDef):
            definitions.setdefault(sub.name, []).append(sub)
    returns = {
        name: [[sub.value for sub in ast_walk.iter_descendants(
            fn, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef))
            if isinstance(sub, ast.Return)] for fn in fns]
        for name, fns in definitions.items()
    }
    requested: set[str] = set()
    changed = True
    while changed:
        changed = False
        for name, values in returns.items():
            if name in requested:
                continue
            if all(fn_values and all(is_requested(value, requested) for value in fn_values)
                   for fn_values in values):
                requested.add(name)
                changed = True
    return requested


def is_requested(value: ast.expr | None, functions: set[str]) -> bool:
    """Whether ``value`` is a security read, or a call of a function giving one.

    Such a tuple may be unpacked into fewer names than it gives: they take its
    leading values, as on TradingView.

    :param value: The expression.
    :param functions: The names of the functions giving a security read, from
        :func:`requested_functions`.
    :return: Whether it is.
    """
    if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Name):
        return False
    if value.func.id == '__sec_read__':
        return len(value.args) == 2
    if value.func.id == '__ltf_unzip__':
        return len(value.args) == 2 and is_requested(value.args[0], functions)
    return value.func.id in functions


def _lower_tf_width(node: object, widths: dict[str, int | None]) -> int | None:
    """The width of the tuple a raw lower-timeframe read gives, if it is one.

    :param node: A node of the tree.
    :param widths: Security id -> the width of the tuple its writes give, None
        where they give no tuple, or tuples of different widths.
    :return: The width, or None when ``node`` is no such read.
    """
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == '__sec_read__' and len(node.args) == 2):
        return None
    key, default = node.args
    if not (isinstance(key, ast.Constant) and isinstance(key.value, str)
            and isinstance(default, ast.List) and not default.elts):
        return None
    return widths.get(key.value)


def _width(value: ast.expr, returned: dict[str, int]) -> int | None:
    """The width of the tuple a requested value gives, where its type does not tell.

    :param value: A requested value.
    :param returned: Function name -> the width of the tuple its every ``return`` gives.
    :return: The width, or None when it is not known.
    """
    if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Name):
        return None
    if value.func.id == '__ltf_unzip__':
        width = value.args[1] if len(value.args) == 2 else None
        return width.value if isinstance(width, ast.Constant) and isinstance(width.value, int) \
            else None
    ty = get_ty(value)
    if is_tuple(ty):
        return len(elements_of(ty))
    return returned.get(value.func.id)


def _returned_widths(node: ast.Module, functions: set[str]) -> dict[str, int]:
    """The width of the tuple each requested function's every ``return`` gives.

    :param node: The module.
    :param functions: The requested functions, from :func:`requested_functions`.
    :return: Function name -> width, for the functions whose returns agree on one.
    """
    returns: dict[str, list[ast.expr | None]] = {}
    for sub in ast_walk.walk(node):
        if isinstance(sub, ast.FunctionDef) and sub.name in functions:
            returns.setdefault(sub.name, []).extend(
                ret.value for ret in ast_walk.iter_descendants(
                    sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef))
                if isinstance(ret, ast.Return))
    widths: dict[str, int] = {}
    changed = True
    while changed:
        changed = False
        for name, values in returns.items():
            if name in widths:
                continue
            found = {_width(value, widths) if value is not None else None for value in values}
            if len(found) == 1 and None not in found:
                widths[name] = found.pop()
                changed = True
    return widths
