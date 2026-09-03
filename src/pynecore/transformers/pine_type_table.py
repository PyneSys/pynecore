"""
The result of the Pine type inference, and the error it raises.

The inference stamps each expression's type on the node itself; what lives
here is the derived index a consumer wants -- the per-scope bindings, the
function signatures, the per-call-site contexts a generic function was
analysed in, the call sites with their overload pin, and the diagnostics.

A generic function has ONE body, analysed once per distinct parameter-type
tuple. Each analysis lands in ``contexts``, while ``bindings`` and ``funcs``
report the JOIN over them -- the same thing the node stamps carry, and for the
same reason: the tree the later passes read is one tree.

The diagnostics are the same information in both modes. A hand-written script
collects them and keeps running with runtime dispatch; ``@pyne edge`` raises on
the first one. One code path, one message; only the raising is conditional.
"""
import ast
from collections.abc import Callable
from dataclasses import dataclass, field

from .pine_type_rules import UNKNOWN, ImplSig

__all__ = ['Unknown', 'Binding', 'FuncSig', 'CallSite', 'ContextKey',
           'ContextResult', 'Diag', 'DepRecord', 'ExportSig', 'ClassSig',
           'ModuleInterface', 'PineTypeTable', 'PineTypeError', 'Analyser',
           'SCOPE_SEP', 'qualify']

#: Separator the scope-qualified ids are spelled with. It is the same character
#: the transformers reserve for their injected names, so a scope-qualified id
#: can never collide with a name a script is allowed to spell.
SCOPE_SEP = '·'


def qualify(scope: str, name: str) -> str:
    """
    The scope-qualified identity of a name declared in one scope.

    :param scope: Scope id, empty at module level
    :param name: The name declared there
    :return: The id ``bindings`` and ``funcs`` key it under
    """
    return f'{scope}{SCOPE_SEP}{name}' if scope else name


#: How a context is addressed: the callee's scope id, the parameter types it
#: was instantiated with, the context its caller was running in, the node id of
#: the DEFINITION analysed (an overload group spells several under one scope
#: id) and the types the enclosing scopes held for the names its body reads.
ContextKey = tuple[str, tuple[str, ...], int, int | None, tuple[tuple[str, str], ...]]


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
    #: Declared as a series (``Series[...]``): its history ``x[n]`` is readable
    series: bool = False


@dataclass(slots=True)
class FuncSig:
    """
    A user function's inferred signature.

    A generic function has one signature and several contexts, so ``params``
    and ``ret`` are the JOIN over every context it was analysed in -- what
    holds for the function as a whole. What a particular call site sees is in
    ``PineTypeTable.contexts``.
    """
    name: str
    params: list[str] = field(default_factory=list)
    ret: str = UNKNOWN
    line: int = 0


@dataclass(slots=True)
class ContextResult:
    """
    One analysis of one function body under one parameter-type tuple.

    MEASURED: TradingView types an unannotated parameter as JOIN(default,
    argument) at the call site and behaves as if the body were instantiated per
    distinct tuple. Here the body is walked again per tuple and only the
    ANSWERS are kept apart, which is what a consumer needs to hand one instance
    a different overload pin from another.
    """
    #: Identity of this context within the module; 0 is the module body itself
    cid: int
    #: Scope-qualified id of the function analysed
    key: str
    #: Type of each parameter, positional first then keyword-only
    params: tuple[str, ...]
    ret: str = UNKNOWN
    #: Call node id -> the overload pin this context justified there. Where two
    #: contexts disagree, the node itself carries no single pin and a later
    #: pass has to hand each instance its own out of these
    pins: dict[int, str | None] = field(default_factory=dict)


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
    #: End of the construct the complaint is about, when it spans one (a
    #: structural rejection covers everything written inside it); 0 = a point
    end_line: int = 0
    end_col: int = 0

    def render(self) -> str:
        """Full one-line form: what happened, where it came from, how to fix it."""
        parts = [self.message]
        if self.origin is not None and (self.origin.line, self.origin.col) != (self.line, self.col):
            parts.append(f'(type lost at line {self.origin.line}: {self.origin.reason})')
        if self.fix:
            parts.append(f'-- {self.fix}')
        return ' '.join(parts)


@dataclass(slots=True, frozen=True)
class DepRecord:
    """
    One module another module's types were derived from.

    The stat pair is what makes the common case free: an untouched dependency
    answers from ``os.stat`` alone, without parsing anything. It is only when
    the file moved that the digest has to be re-derived and compared -- an
    edit that leaves the INTERFACE alone (a body change, a comment) then keeps
    the dependent's cached bytecode valid.
    """
    #: Resolved source path of the dependency
    path: str
    mtime_ns: int
    size: int
    #: The dependency's interface digest when the dependent was transformed
    digest: str


@dataclass(slots=True, frozen=True)
class ExportSig:
    """
    One name a module publishes, as a consumer of the module reads it.

    A group has no single shape -- each implementation has its own -- so
    ``impls`` is where the authoritative shapes live and the fields beside it
    mirror the FIRST implementation, the one declaration order reaches first.
    """
    name: str
    #: ``'function'`` for a plain definition, ``'group'`` for an ``@overload``
    #: group
    kind: str
    #: Type character of each positional parameter, in declaration order
    params: tuple[str, ...] = ()
    #: How many of them a call has to pass
    required: int = 0
    #: Whether a ``*args`` lets it take any number of them
    open_ended: bool = False
    ret: str = UNKNOWN
    #: Whether every positional parameter carries an annotation this pass can
    #: read. An export that does not is one a caller cannot type its arguments
    #: against, whatever the arguments are
    annotated: bool = False
    #: Every implementation of a group, in declaration order; empty otherwise
    impls: tuple[ImplSig, ...] = ()
    line: int = 0
    #: Name of each positional parameter (a group's: the first implementation's)
    names: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class ClassSig:
    """
    One user-defined type, as everything that reads a value of it needs it.

    A Pine object KNOWS its class, so ``obj.field`` has the field's DECLARED
    type and ``obj.method(...)`` the method's. That is only answerable while
    the class travels with the value, which is what the class id in the type
    (``'o:<module>#Pivot'``) is for -- and why the id is (module, name) and
    never the bare name: two libraries publishing a ``Settings`` publish two
    different types.
    """
    #: The class name, as an annotation of its own module spells it
    name: str
    #: ``<module key>#<name>``, the identity a shaped type carries
    id: str
    #: Field name -> its declared type, in declaration order
    fields: dict[str, str] = field(default_factory=dict)
    #: How many of them a constructor call has to pass: the fields without a
    #: default, which a dataclass keeps in front
    required: int = 0
    #: Method name -> its signature. A Pine method is a free function whose
    #: first parameter is annotated with the class; it is published with the
    #: class because that is how a receiver reaches it
    methods: dict[str, ExportSig] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ModuleInterface:
    """
    Everything one module publishes, and nothing about how it does it.

    This is the whole cross-module contract: a dependent's types are a
    function of its own source and of the INTERFACES it imports, so a
    dependency whose interface digest is unchanged cannot invalidate it.
    """
    #: Resolved source path of the module
    path: str
    exports: dict[str, ExportSig]
    #: The module's literal ``__all__``, or None when it spells none. A
    #: namespace import reads the exports through it, so it is part of the
    #: contract and not merely a hint
    all: tuple[str, ...] | None
    #: The classes the module publishes, ``__all__``-filtered, by name. A
    #: dependent annotating a parameter with one of them means an object OF
    #: THAT CLASS, so the whole class -- its id, its field types and its
    #: methods -- is part of the contract: a field whose type moves changes
    #: what every dependent reading that field resolves to
    classes: dict[str, ClassSig]
    #: The methods this module declares on ANOTHER module's class, by class id
    #: and method name. Pine lets one library extend another library's UDT, and
    #: the runtime finds such a method by searching the importing script's
    #: library modules -- so what a receiver of that class resolves to depends
    #: on this module too, and it is part of the contract
    extensions: dict[str, dict[str, ExportSig]]
    #: Digest of ``exports``, ``all``, ``classes`` and ``extensions``, blind to
    #: every body
    digest: str
    #: Every module this one's types were derived from, its own dependencies'
    #: dependencies included. An interface can be INFERRED from a third
    #: module -- an export with annotated parameters and no return annotation
    #: takes its return from whatever it calls -- so a consumer that only
    #: checked this module's own source would keep believing a signature that
    #: moved two modules away. Validation metadata, deliberately outside the
    #: digest: what a module publishes does not change because a dependency's
    #: file was touched
    deps: dict[str, DepRecord] = field(default_factory=dict)
    #: The source stat this interface was derived from. It travels WITH the
    #: interface so a consumer can neither pair a fresh stat with an old
    #: digest nor keep a registry entry whose file has since moved
    mtime_ns: int = 0
    #: Size of that same stat; -1 when the source could not be stat'd at all,
    #: which no later stat matches
    size: int = -1
    #: Why NO pin may be built on anything this module publishes, or ``''``:
    #: a container of its was mutated with something its type does not
    #: hold and the aliases are not tracked, so a return type derived from
    #: one may be a confident wrong answer. Part of the contract -- an
    #: importer that consults such a module has to give its own pins up too
    suppressed: str = ''


@dataclass(slots=True)
class PineTypeTable:
    """Everything the inference learned about one module."""
    #: Absolute source path of the module
    module_path: str = ''
    #: scope id -> name -> binding; the module scope is the empty string
    bindings: dict[str, dict[str, Binding]] = field(default_factory=dict)
    #: scope-qualified function id -> signature. The
    #: key is the function's OWN scope id, the same identity ``bindings``
    #: uses, so two same-named nested helpers stay apart
    funcs: dict[str, FuncSig] = field(default_factory=dict)
    #: (callee id, parameter tuple, calling context) -> what that context found
    contexts: dict[ContextKey, ContextResult] = field(default_factory=dict)
    calls: list[CallSite] = field(default_factory=list)
    diags: list[Diag] = field(default_factory=list)
    #: Scope-qualified id of every ``@overload`` group -> its implementations
    #: in declaration order. The selection reads these, and so does the
    #: interface a dependent module resolves the group's calls against
    groups: dict[str, tuple[ImplSig, ...]] = field(default_factory=dict)
    #: Resolved path -> the state of every module this one's types were
    #: derived from. A dependent's cached bytecode is only valid while every
    #: record here still describes the file on disk
    deps: dict[str, DepRecord] = field(default_factory=dict)
    #: Every class name an annotation in this module may name -> its class
    #: id: the ones the module declares in any scope, and the ones its imports
    #: bring in. An annotation naming one of these is an object of that class,
    #: not an unknown and not an anonymous object
    classes: dict[str, str] = field(default_factory=dict)
    #: The classes this module DECLARES, by id, with their fields and methods.
    #: An imported one is not here -- it belongs to the module that declares
    #: it, and travels in that module's interface
    class_sigs: dict[str, ClassSig] = field(default_factory=dict)
    #: The methods this module declares on classes it does NOT declare, by
    #: class id and method name. A ``@method`` whose receiver is an imported
    #: class extends that class for everyone who imports this module, which is
    #: why it travels in the interface rather than with the class
    extensions: dict[str, dict[str, ExportSig]] = field(default_factory=dict)
    #: Why NO call site of the module carries an overload pin, when none
    #: does: a container was mutated in place with something its type does
    #: not hold, and the aliases of that container -- another name, a field,
    #: a parameter, a ``var`` read earlier in source order on the next bar --
    #: are not tracked, so no type downstream may drive a dispatch. None
    #: while every pin the walk decided on stands.
    pins_suppressed: Diag | None = None
    #: Module-scope names an importer is guaranteed to receive a DEFINITION
    #: under: the name's last module-level binding is a ``def`` that no branch
    #: guards, and no other binding of it stands after that def. What the
    #: module interface publishes -- every other name reaches the importer as
    #: whatever ran last, which is not a static question
    exportable: frozenset[str] = frozenset()
    #: Call node id -> the source position of that call. The pins a context
    #: records are keyed by node id, which describes the tree the inference
    #: walked and not the one a later pass emits, so anything written out has
    #: to name the position instead
    call_pos: dict[int, tuple[int, int]] = field(default_factory=dict)

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


#: Re-derives one module's tree and type table from its source path, without
#: compiling or executing anything. The import hook owns the only real one
#: (``core.import_hook.analyse_source``); it is passed in rather than imported
#: so the analysis stays free of the loader.
#:
#: The third element is the ``(mtime_ns, size)`` of the very bytes that were
#: parsed, read as one pair with them, or None when no such pair could be had.
#: An interface built from the tree is only allowed to carry THAT fingerprint:
#: a stat taken before the read describes whatever the file was then, and
#: pairing it with signatures read afterwards records a state that never was.
Analyser = Callable[[str], tuple[ast.Module, PineTypeTable, tuple[int, int] | None] | None]
