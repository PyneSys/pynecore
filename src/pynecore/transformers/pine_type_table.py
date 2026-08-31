"""
The result of the Pine type inference, and the error it raises.

The inference stamps each expression's type on the node itself; what lives
here is the derived index a consumer wants -- the per-scope bindings, the
function signatures (originals and monomorphized clones), the call sites with
their overload pin, and the diagnostics.

The diagnostics are the same information in both modes. A hand-written script
collects them and keeps running with runtime dispatch; ``@pyne edge`` raises on
the first one. One code path, one message; only the raising is conditional.
"""
from dataclasses import dataclass, field

from .pine_type_rules import UNKNOWN

__all__ = ['Unknown', 'Binding', 'FuncSig', 'CallSite', 'Diag', 'PineTypeTable',
           'PineTypeError']


@dataclass(slots=True, frozen=True)
class Unknown:
    """
    Why a type could not be determined, and where it was lost.

    An UNKNOWN that carries no provenance is a dead end for the user: the
    error can only point at the expression that failed, not at the parameter
    or the call three lines up that made it unknowable. So the origin travels
    with the type, first-wins -- the FIRST place the type was lost is the one
    worth fixing, not the last place it was noticed.
    """
    #: Short machine-readable cause, e.g. ``'unannotated-param'``
    reason: str
    #: 1-based line the type was lost on
    line: int
    #: 0-based column offset
    col: int
    #: Human-readable detail, e.g. the offending name
    detail: str = ''

    def __str__(self) -> str:
        where = f'line {self.line}'
        return f'{self.reason} at {where}' + (f': {self.detail}' if self.detail else '')


@dataclass(slots=True)
class Binding:
    """A name in a scope, with the type its assignments join to."""
    name: str
    ty: str = UNKNOWN
    #: 1-based line of the binding that first established the type
    line: int = 0
    #: Provenance when ``ty`` is UNKNOWN
    unknown: Unknown | None = None


@dataclass(slots=True)
class FuncSig:
    """
    A user function's inferred signature.

    ``origin`` names the function a monomorphized clone was made from, so a
    consumer can fold the clones back together; it is None for an original.
    """
    name: str
    params: list[str] = field(default_factory=list)
    ret: str = UNKNOWN
    line: int = 0
    origin: str | None = None


@dataclass(slots=True)
class CallSite:
    """
    One call, with the overload pin the inference decided on.

    ``pin`` is one type character per positional argument, or None when the
    call site is not pinnable (no overload group, a keyword or star argument,
    an unknown argument type). The runtime turns those characters into witness
    values and runs the ORDINARY selector on them, so the static and the
    dynamic decision can never drift apart.
    """
    #: Dotted callee as written, e.g. ``'ta.highest'``
    callee: str
    line: int
    col: int
    #: Arguments passed, keyword ones included; None when an unpacking hides
    #: the count
    argc: int | None
    ty: str = UNKNOWN
    pin: str | None = None


@dataclass(slots=True, frozen=True)
class Diag:
    """One typing complaint, in the shape both modes report."""
    message: str
    line: int
    col: int
    #: Where the type was lost, when that differs from where it was noticed
    origin: Unknown | None = None
    #: Concrete remedy, e.g. "annotate 'length' as int"
    fix: str = ''

    def render(self) -> str:
        """Full one-line form: what happened, where it came from, how to fix it."""
        parts = [self.message]
        if self.origin is not None and (self.origin.line, self.origin.col) != (self.line, self.col):
            parts.append(f'(type lost at line {self.origin.line}: {self.origin.reason})')
        if self.fix:
            parts.append(f'-- {self.fix}')
        return ' '.join(parts)


@dataclass(slots=True)
class PineTypeTable:
    """Everything the inference learned about one module."""
    #: Absolute source path of the module
    module_path: str = ''
    #: scope id -> name -> binding; the module scope is the empty string
    bindings: dict[str, dict[str, Binding]] = field(default_factory=dict)
    #: scope-qualified function id (clone names included) -> signature. The
    #: key is the function's OWN scope id, the same identity ``bindings``
    #: uses, so two same-named nested helpers stay apart
    funcs: dict[str, FuncSig] = field(default_factory=dict)
    calls: list[CallSite] = field(default_factory=list)
    diags: list[Diag] = field(default_factory=list)

    def binding(self, scope: str, name: str) -> Binding | None:
        """
        Look one name up in one scope, without walking outwards.

        :param scope: Scope id, empty for module level
        :param name: The name to find
        :return: The binding, or None
        """
        return self.bindings.get(scope, {}).get(name)


class PineTypeError(SyntaxError):
    """
    A typing failure that stops the transform.

    Raised only in ``@pyne edge`` mode, where an untypable construct is a
    contract violation rather than a missed optimization. It is a
    ``SyntaxError`` on purpose: Python renders the 4-tuple with a real caret
    under the offending column, the same way the security and series passes
    report theirs.
    """

    @classmethod
    def from_diag(cls, diag: Diag, filename: str, text: str | None = None) -> 'PineTypeError':
        """
        Build the error a diagnostic describes.

        :param diag: The diagnostic to raise
        :param filename: Source path, for the traceback header
        :param text: The offending source line, when it is available
        :return: The error, ready to raise
        """
        return cls(diag.render(), (filename, diag.line, diag.col + 1, text))
