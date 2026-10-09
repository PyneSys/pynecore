"""
The AST transform pipeline as data: every step, its order and why it stands where it does.

The loader (:mod:`pynecore.core.import_hook`) runs the two halves declared here. The
ANALYSIS half normalizes the tree into the form the Pine type pass reads and stamps
the types onto it, without emitting any of the state plumbing; splitting it out is
what lets an imported module's signatures be derived without compiling or running
anything. The LOWERING half turns the analysed tree into the state-plumbed form the
runtime executes.

Each step is one pass over the module. Its position is a contract with the steps
around it, and the comment above it says which one; a step that only applies to
user code (scripts and their libraries, not pynecore's own ``@pyne`` lib modules) is
marked so. A :class:`Phase` groups steps written as rules (see ``phase.py``) that
run in ONE traversal; ``PYNE_AST_SEQUENTIAL=1`` runs them as separate passes
instead, which is the reference a phase must reproduce. ``PYNE_AST_TIMING=1``
prints the wall time of every step (every phase, every rule in sequential mode)
to stderr.
"""
import ast
import os
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pynecore.transformers.builtin_shadow import BuiltinShadowTransformer
from pynecore.transformers.call_inline import CallInlineTransformer
from pynecore.transformers.closure_arguments_transformer import ClosureArgumentsTransformer
from pynecore.transformers.const_fold import ConstFoldTransformer
from pynecore.transformers.dynamic_default import DynamicDefaultTransformer
from pynecore.transformers.export_capture import ExportCaptureTransformer
from pynecore.transformers.export_once import ExportOnceTransformer
from pynecore.transformers.float_tolerance import FloatToleranceTransformer
from pynecore.transformers.function_isolation import FunctionIsolationTransformer
from pynecore.transformers.import_lifter import ImportLifterTransformer
from pynecore.transformers.import_normalizer import ImportNormalizerTransformer
from pynecore.transformers.inline_series_hoist import InlineSeriesHoistTransformer
from pynecore.transformers.input_transformer import InputTransformer
from pynecore.transformers.lib_series import LibrarySeriesTransformer
from pynecore.transformers import locations
from pynecore.transformers.module_property import ModulePropertyTransformer
from pynecore.transformers.outer_write import OuterWriteTransformer
from pynecore.transformers.persistent import PersistentTransformer
from pynecore.transformers.persistent_series import PersistentSeriesTransformer
from pynecore.transformers.phase import Rule, run_phase
from pynecore.transformers.pine_truthiness import PineTruthinessTransformer
from pynecore.transformers.pine_type_transformer import PineTypeTransformer
from pynecore.transformers.plot_scope import PlotScopeTransformer
from pynecore.transformers.safe_convert_transformer import SafeConvertTransformer
from pynecore.transformers.safe_division_transformer import SafeDivisionTransformer
from pynecore.transformers.script_requirements import ScriptRequirementsTransformer
from pynecore.transformers.security import SecurityTransformer
from pynecore.transformers.security_closed_shift_check import verify_closed_shift
from pynecore.transformers.security_default import SecurityDefaultTransformer
from pynecore.transformers.security_drawings import SecurityDrawingsTransformer
from pynecore.transformers.security_instantiation import SecurityInstantiationTransformer
from pynecore.transformers.security_slice import SecuritySliceTransformer, finalize_clone_defaults
from pynecore.transformers.series import SeriesTransformer
from pynecore.transformers.slot_layout import ModuleLayout, apply_layout
from pynecore.transformers.ta_variable_hoist import TaVariableHoistTransformer
from pynecore.transformers.type_checking_stripper import TypeCheckingStripperTransformer
from pynecore.transformers.type_erasure import erase_type_calls
from pynecore.transformers.unused_series_detector import UnusedSeriesDetectorTransformer

if TYPE_CHECKING:
    from pynecore.transformers.pine_type_table import Analyser

__all__ = ['PipelineContext', 'Step', 'RuleStep', 'Phase', 'ANALYSIS', 'LOWERING', 'run_steps']


@dataclass(slots=True)
class PipelineContext:
    """What the steps of one module's transform share besides the tree."""

    #: Source path; the script / lib profile is picked from it
    path: Path
    #: The module's mode word, None for a hand-written script
    pyne_mode: str | None
    #: Full module source, for error locations
    source: str = ''
    #: Whether the module is user code rather than one of pynecore's own lib modules
    user_code: bool = True
    #: Answers an imported module's interface without executing it (type pass)
    analyse: 'Analyser | None' = None
    #: Digest of the pipeline the module is transformed by (type pass)
    pipeline_hash: str = ''
    #: Whether the module keeps Pine's three-state bool (see FloatToleranceTransformer)
    na_bool: bool = False
    #: Whether ``apply_layout`` materializes the slot layout for CPython
    emit_layout: bool = True
    #: The shared slot allocator of the lowering half (see slot_layout.py)
    slot_layout: ModuleLayout = field(default_factory=ModuleLayout)


@dataclass(frozen=True, slots=True)
class Step:
    """One pass over the module."""

    #: Name reported by ``PYNE_AST_TIMING``
    name: str
    #: The pass; it returns the (possibly replaced) module
    run: Callable[[ast.Module, PipelineContext], ast.Module]
    #: Only user code gets this step, never pynecore's own ``@pyne`` lib modules
    user_code_only: bool = False


@dataclass(frozen=True, slots=True)
class RuleStep:
    """One rule of a :class:`Phase`: a pass written as hooks (see ``phase.py``)."""

    #: Name reported by ``PYNE_AST_TIMING`` in sequential mode
    name: str
    #: Builds the rule for one module
    make: Callable[[PipelineContext], Rule]
    #: Only user code gets this rule, never pynecore's own ``@pyne`` lib modules
    user_code_only: bool = False


@dataclass(frozen=True, slots=True)
class Phase:
    """Rules that run in one traversal, equivalent to running them in this order."""

    #: Name reported by ``PYNE_AST_TIMING``
    name: str
    rules: Sequence[RuleStep]


def _visit(transformer: type) -> Callable[[ast.Module, PipelineContext], ast.Module]:
    return lambda tree, ctx: transformer().visit(tree)


def _lower_layout(transformer: type) -> Callable[[ast.Module, PipelineContext], ast.Module]:
    return lambda tree, ctx: transformer(ctx.slot_layout).visit(tree)


def _in_place(fn: Callable[[ast.Module], object]) -> Callable[[ast.Module, PipelineContext], ast.Module]:
    def run(tree: ast.Module, ctx: PipelineContext) -> ast.Module:
        fn(tree)
        return tree
    return run


def _pine_types(tree: ast.Module, ctx: PipelineContext) -> ast.Module:
    return PineTypeTransformer(
        ctx.pyne_mode, analyse=ctx.analyse, pipeline_hash=ctx.pipeline_hash,
        qualify_windows=ctx.user_code,
    ).visit(tree)


def _function_isolation(tree: ast.Module, ctx: PipelineContext) -> ast.Module:
    return FunctionIsolationTransformer(
        ctx.slot_layout, analyse=ctx.analyse, pipeline_hash=ctx.pipeline_hash,
    ).visit(tree)


def _apply_layout(tree: ast.Module, ctx: PipelineContext) -> ast.Module:
    return apply_layout(tree, ctx.slot_layout) if ctx.emit_layout else tree


ANALYSIS: Sequence[Step] = (
    Step('ImportLifter', _visit(ImportLifterTransformer)),
    Step('TypeCheckingStripper', _visit(TypeCheckingStripperTransformer)),
    # Before import normalization: the pass trusts a callee by the module's own
    # ``typing`` imports, which it reads as written
    Step('TypeErasure', lambda tree, ctx: erase_type_calls(tree)),
    # The builtin-namespace fallback must run before import normalization so the
    # lib.<ns>.<name> chains it emits get their imports added there
    Step('BuiltinShadow', _visit(BuiltinShadowTransformer)),
    Step('ImportNormalizer', _visit(ImportNormalizerTransformer)),
    Step('PlotScope', lambda tree, ctx: PlotScopeTransformer(ctx.source).visit(tree),
         user_code_only=True),
    # The language rule that an object created outside a function may not be
    # modified inside one. It reads the normalized ``lib.*`` chains and runs well
    # before function isolation, whose per-call-site copies would report the same
    # write once per copy. Only user code is subject to it -- pynecore's own lib
    # modules ARE the module-level machinery
    Step('OuterWrite', _visit(OuterWriteTransformer), user_code_only=True),
    Step('ExportCapture', _visit(ExportCaptureTransformer), user_code_only=True),
    Step('SecurityDrawings', _visit(SecurityDrawingsTransformer), user_code_only=True),
    # TradingView folds constant subtrees at parse time with fdlibm transcendentals
    # and a 16-decimal embedding cap, while runtime series-fed calls use the
    # Intel-LIBM intrinsics (lib.math / core.pine_math); the fold pass replays that
    # split. It needs the normalized lib.math.* chains, and pynecore's own lib
    # modules must keep their raw expressions
    Step('ConstFold', _visit(ConstFoldTransformer), user_code_only=True),
    # Per-call evaluation of lib.*-referencing parameter defaults; must precede the
    # series/isolation passes so the moved expressions get their series slots and
    # call-site anchors like any body statement
    Step('DynamicDefault', _visit(DynamicDefaultTransformer)),
    # Lazy-context history hoist must run before call-site anchoring: the hoisted
    # statements are the anchorable call sites
    Step('InlineSeriesHoist', _visit(InlineSeriesHoistTransformer)),
    # Pine's tolerant float-to-bool conversion, over the script's OWN bool contexts:
    # it runs before the passes that emit control flow of their own (lazy-init flags
    # and friends), whose tests are bools by construction. User code only -- see the
    # comparison rewrite at the end of the lowering half for the same rule
    Step('PineTruthiness', _visit(PineTruthinessTransformer), user_code_only=True),
    # Pine instantiation semantics: clone security-bearing functions per call site
    # so each call site gets its own security contexts
    Step('SecurityInstantiation', _visit(SecurityInstantiationTransformer)),
    Step('Security', _visit(SecurityTransformer)),
    Step('PersistentSeries', _visit(PersistentSeriesTransformer)),
    Step('LibrarySeries', _visit(LibrarySeriesTransformer)),
    Step('ModuleProperty', _visit(ModulePropertyTransformer)),
    # Stateful ta builtin variables become one unconditional per-bar evaluation at
    # the top of main (TradingView keeps a single engine series per builtin
    # variable, gates notwithstanding); must follow the property transformer (bare
    # reads are calls by now) and precede the series/persistent/isolation passes so
    # the hoisted call site is anchored like any hand-written statement
    Step('TaVariableHoist', _visit(TaVariableHoistTransformer)),
    Step('ClosureArguments', _visit(ClosureArgumentsTransformer)),
    # Pine's static types, stamped on the nodes. The last point where the tree still
    # looks like Pine: the annotations are intact (the series pass rewrites and
    # consumes them), the `/` is still a BinOp (safe division wraps it into a call),
    # and the security-bearing functions are already instantiated per call site.
    # Analysis only -- it stamps, it does not rewrite. It also decides the machine
    # of every window call (``ta.highest`` and its kin) from the Pine qualifier of
    # the length -- in user code, not in pynecore's own lib modules, whose machines
    # ARE those calls
    Step('PineType', _pine_types),
    # Per-context backward slices of main(), emitted as ordinary module-level
    # functions the security children run instead of main(). Last of this half: the
    # clones must see the tree every earlier step produced (the hoisted ta
    # assignments among them), and the lowering half then lays them out like any
    # other function, with their own slot layout
    Step('SecuritySlice', _visit(SecuritySliceTransformer)),
)

LOWERING: Sequence[Step | Phase] = (
    # Security reads now carry their expression's inferred type, including tuple
    # fields; resolve bool defaults before the state plumbing is emitted
    Step('SecurityDefault', _visit(SecurityDefaultTransformer)),
    # A library's export surface is defined once per RUN, not once per bar: the
    # latch it runs under is a Persistent slot, so it must precede the slot
    # transformers, and the guarded definitions must reach them in their final
    # position
    Step('ExportOnce', _visit(ExportOnceTransformer)),
    Step('UnusedSeriesDetector', lambda tree, ctx: UnusedSeriesDetectorTransformer().optimize(tree)),
    Step('Series', _lower_layout(SeriesTransformer)),
    # The series step has just emitted the only two forms a runtime history
    # reference can take, so the source-level ``closed_shift`` candidate can now be
    # checked exactly; the verifier only ever CLEARS the flag
    Step('VerifyClosedShift', lambda tree, ctx: verify_closed_shift(tree, ctx.slot_layout)),
    Step('Persistent', _lower_layout(PersistentTransformer)),
    # Trivial builtin wrappers are copied into their call sites BEFORE the
    # isolation step, so a site that is no longer a call gets no anchor slot and no
    # binding guard. Its arguments are already lowered here, and the comparisons it
    # emits are marked ``pine_exact`` so the float tolerance rewrite below leaves
    # the copied raw na tests alone
    Step('CallInline', _visit(CallInlineTransformer)),
    # Call-site classification needs the var/series slots, so the isolation step
    # must run after Persistent and Series
    Step('FunctionIsolation', _function_isolation),
    Step('ScriptRequirements', _visit(ScriptRequirementsTransformer)),
    # Not a rule of the phase below: a source-input parameter it rebinds at the top of
    # the function can be named ``range``, which SafeConvert's binding scan must see
    Step('Input', _visit(InputTransformer)),
    # The expression-level rewrites, in one post-order traversal. It equals the three
    # passes in this order: none of them emits what a later one rewrites, and the one
    # decision that reads a child a later rule may already have rewritten --
    # SafeDivision's operand types -- reads the same type either way. A comparison
    # FloatTolerance rewrote is typed bool, as the type pass typed it, and the types
    # SafeConvert gives what it wraps only reach untyped plumbing (state reads), which
    # is never numeric
    Phase('Expression', (
        RuleStep('SafeConvert', lambda ctx: SafeConvertTransformer(lib=ctx.pyne_mode == 'lib')),
        RuleStep('SafeDivision', lambda ctx: SafeDivisionTransformer()),
        # After SafeDivision so wrapped operands (safe_div calls) are bound once by
        # the walrus instead of evaluating twice. Only user code gets Pine's
        # tolerant comparison semantics: pynecore's own lib modules implement the
        # natively bit-exact builtins and use the raw ``x != x`` nan idiom, both of
        # which the rewrite would break
        RuleStep('FloatTolerance', lambda ctx: FloatToleranceTransformer(na_bool=ctx.na_bool),
                 user_code_only=True),
    )),
    Step('FinalizeCloneDefaults', _in_place(finalize_clone_defaults)),
    Step('ApplyLayout', _apply_layout),
    # Debugger-safe variant of ast.fix_missing_locations: synthetic nodes get point
    # anchors, so no prologue bytecode maps onto the function's last line (see
    # transformers/locations.py)
    Step('FixLocations', lambda tree, ctx: locations.fix_locations(tree)),
)

#: Whether every step's wall time is reported on stderr
_TIMING = bool(os.environ.get('PYNE_AST_TIMING'))

#: Whether the rules of a phase run as separate passes instead of in one traversal
_SEQUENTIAL = bool(os.environ.get('PYNE_AST_SEQUENTIAL'))


def run_steps(steps: Sequence[Step | Phase], tree: ast.Module, ctx: PipelineContext) -> ast.Module:
    """Run the steps over a module in their declared order.

    :param steps: One half of the pipeline (:data:`ANALYSIS` or :data:`LOWERING`).
    :param tree: The module; steps transform it in place where they can.
    :param ctx: What the steps share besides the tree.
    :return: The transformed module.
    """
    for step in steps:
        if isinstance(step, Phase):
            rules = [(rule.name, rule.make(ctx)) for rule in step.rules
                     if ctx.user_code or not rule.user_code_only]
            if _SEQUENTIAL:
                for name, rule in rules:
                    tree = _run(name, ctx, rule.visit, tree)
            else:
                tree = _run(step.name, ctx, run_phase, [rule for _, rule in rules], tree)
        elif ctx.user_code or not step.user_code_only:
            tree = _run(step.name, ctx, step.run, tree, ctx)
    return tree


def _run(name: str, ctx: PipelineContext, run: Callable[..., Any], *args: Any) -> ast.Module:
    """Run one pass over the module, timed when ``PYNE_AST_TIMING`` asks for it."""
    if not _TIMING:
        return run(*args)
    start = time.perf_counter()
    tree = run(*args)
    sys.stderr.write(f'[pyne-ast] {ctx.path.name}: {name} '
                     f'{1000 * (time.perf_counter() - start):.2f} ms\n')
    return tree
