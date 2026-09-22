"""
@pyne
"""
import ast

from pynecore.transformers.security import SecurityTransformer
from pynecore.transformers.security_slice import (
    CLONE_PREFIX, SecuritySliceTransformer,
)


def __test_helper_transform(source: str) -> ast.Module:
    """Run the security split and the slicer over a source string.

    The two passes are run alone: the slicer only needs a lowered security tree
    (write blocks, ``__sec_read__`` calls and ``__security_contexts__``), which
    :class:`SecurityTransformer` alone produces.

    :param source: The module source.
    :return: The transformed module.
    """
    tree = ast.parse(source)
    tree = SecurityTransformer().visit(tree)
    tree = SecuritySliceTransformer().visit(tree)
    ast.fix_missing_locations(tree)
    return tree


def __test_helper_clones(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    """Every emitted clone of the module, keyed by its name."""
    clones: dict[str, ast.FunctionDef] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith(CLONE_PREFIX):
            clones[node.name] = node
    return clones


def __test_helper_only_clone(tree: ast.Module) -> ast.FunctionDef:
    """The single clone the module is expected to carry."""
    clones = __test_helper_clones(tree)
    assert len(clones) == 1, f"expected exactly one clone, got {sorted(clones)}"
    return next(iter(clones.values()))


def __test_helper_body(clone: ast.FunctionDef) -> str:
    """The clone's body as source text."""
    return '\n'.join(ast.unparse(stmt) for stmt in clone.body)


def __test_helper_calls(clone: ast.FunctionDef) -> set[str]:
    """Names called by the clone's own top-level statements."""
    names: set[str] = set()
    for stmt in clone.body:
        if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)):
            continue
        func = stmt.value.func
        if isinstance(func, ast.Name):
            names.add(func.id)
    return names


def __test_helper_slice_names(tree: ast.Module) -> list[str]:
    """The ``slice_main`` entries recorded in ``__security_contexts__``."""
    names: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        target = node.targets[0]
        if not (isinstance(target, ast.Name) and target.id == '__security_contexts__'):
            continue
        assert isinstance(node.value, ast.Dict)
        for ctx in node.value.values:
            assert isinstance(ctx, ast.Dict)
            for key, value in zip(ctx.keys, ctx.values):
                if (isinstance(key, ast.Constant) and key.value == 'slice_main'
                        and isinstance(value, ast.Constant)):
                    names.append(value.value)
    return names


def __test_slice_keeps_the_expression_cone__(log):
    """The clone keeps what the expression reads and drops the rest

    The persistent counter is written BEFORE and read by the expression, the
    plots and the unrelated array live only after it — and the signal / wait
    blocks are chart-only no-ops in a child.
    """
    source = """
@script.indicator("t")
def main():
    counter: Persistent[int] = 0
    counter += 1
    store = lib.array.new_float(0)
    sma = lib.ta.sma(lib.close, 14)
    d = lib.request.security(lib.syminfo.tickerid, "1D", sma + counter)
    lib.array.push(store, d)
    lib.plot(d)
    lib.plot(lib.array.size(store))
"""
    tree = __test_helper_transform(source)
    clone = __test_helper_only_clone(tree)
    body = __test_helper_body(clone)

    assert 'counter' in body, "the bar-crossing persistent write was dropped"
    assert 'lib.ta.sma' in body, "the expression's own input was dropped"
    assert '__sec_write__' in body, "the write block itself was dropped"
    assert 'lib.plot' not in body, "an unrelated plot survived the slice"
    assert 'lib.array.push' not in body, "an unrelated mutation survived the slice"
    assert '__sec_signal__' not in body and '__sec_wait__' not in body, \
        "a chart-only protocol block survived the slice"
    assert __test_helper_slice_names(tree) == [clone.name], \
        "the clone was not recorded in __security_contexts__"
    log.info("slice kept %d of %d statements", len(clone.body), 9)


def __test_slice_keeps_mutation_of_a_read_collection__(log):
    """A collection the expression reads keeps every statement that mutates it

    The ``array.push`` stands AFTER the expression, so a position-based slice
    would drop it — but ``main()`` is a loop body, and the next bar's expression
    reads what this bar pushed.
    """
    source = """
@script.indicator("t")
def main():
    store = lib.array.new_float(0)
    noise = lib.array.new_float(0)
    d = lib.request.security(lib.syminfo.tickerid, "1D", lib.array.size(store))
    lib.array.push(store, lib.close)
    lib.array.push(noise, lib.close)
    lib.plot(d)
"""
    tree = __test_helper_transform(source)
    body = __test_helper_body(__test_helper_only_clone(tree))

    assert 'lib.array.push(store' in body, \
        "the mutation feeding the next bar's expression was dropped"
    assert 'lib.array.push(noise' not in body, \
        "a mutation of an unrelated collection was kept"
    assert 'lib.plot' not in body
    log.info("the read collection's mutation survived, the unrelated one did not")


def __test_slice_keeps_an_alias_mutation__(log):
    """Mutating a collection through an alias counts as mutating the original"""
    source = """
@script.indicator("t")
def main():
    store = lib.array.new_float(0)
    d = lib.request.security(lib.syminfo.tickerid, "1D", lib.array.size(store))
    alias = store
    lib.array.push(alias, lib.close)
    lib.plot(d)
"""
    tree = __test_helper_transform(source)
    body = __test_helper_body(__test_helper_only_clone(tree))

    assert 'alias = store' in body, "the alias binding was dropped"
    assert 'lib.array.push(alias' in body, "the aliased mutation was dropped"
    log.info("alias unification kept the indirect mutation")


def __test_slice_keeps_an_early_return__(log):
    """A statement that can end main() early is never dropped"""
    source = """
@script.indicator("t")
def main():
    if lib.bar_index < 10:
        return
    lib.plot(lib.close)
    d = lib.request.security(lib.syminfo.tickerid, "1D", lib.ta.sma(lib.close, 3))
    lib.plot(d)
"""
    tree = __test_helper_transform(source)
    body = __test_helper_body(__test_helper_only_clone(tree))

    assert 'return' in body, "the early exit was dropped from the slice"
    assert body.count('lib.plot') == 0, "an unrelated plot survived the slice"
    log.info("the early return survived while both plots were dropped")


def __test_slice_keeps_peer_and_own_reads__(log):
    """Every ``__sec_read__`` of a peer this context depends on is mandatory"""
    source = """
@script.indicator("t")
def main():
    a = lib.request.security(lib.syminfo.tickerid, "1D", lib.ta.sma(lib.close, 3))
    b = lib.request.security(lib.syminfo.tickerid, "240", a + 1.0)
    lib.plot(b)
"""
    tree = __test_helper_transform(source)
    clones = __test_helper_clones(tree)
    assert len(clones) == 2, f"expected a clone per context, got {sorted(clones)}"
    consumer = __test_helper_body(clones[f'{CLONE_PREFIX}1__'])

    assert consumer.count('__sec_read__') == 2, \
        "the consumer's clone lost a mandatory peer or own read"
    assert 'lib.plot' not in consumer
    log.info("both the peer read and the own read survived in the consumer clone")


def __test_no_clone_for_unmodelled_construct__(log):
    """A ``try`` anywhere in main() turns the whole optimization off"""
    source = """
@script.indicator("t")
def main():
    try:
        x = 1.0
    finally:
        x = 2.0
    d = lib.request.security(lib.syminfo.tickerid, "1D", lib.ta.sma(lib.close, 3))
    lib.plot(d)
"""
    tree = __test_helper_transform(source)
    assert not __test_helper_clones(tree), "a clone was emitted for an unmodelled main()"
    assert not __test_helper_slice_names(tree), "slice_main was recorded without a clone"
    log.info("the unmodelled construct fell back to the whole-main child")


def __test_runtime_resolved_context_is_sliced__(log):
    """An input-driven symbol / timeframe does not block the slice

    Which feed the context resolves to is decided by the chart at spawn time;
    ``main()``'s statements — and therefore the slice — are the same either
    way, and the child reads its clone name from the very context meta the
    chart hands it after the deferred resolution.
    """
    source = """
@script.indicator("t")
def main(tf: str = lib.input.timeframe("60"), sym: str = lib.input.symbol("")):
    d = lib.request.security(sym, tf, lib.ta.sma(lib.close, 3))
    lib.plot(d)
"""
    tree = __test_helper_transform(source)
    body = __test_helper_body(__test_helper_only_clone(tree))

    assert '__sec_write__' in body, "the write block is missing from the clone"
    assert 'lib.plot' not in body, "an unrelated plot survived the slice"
    assert __test_helper_slice_names(tree), "the clone was not recorded"
    log.info("the runtime-resolved context got a slice like any other")


def __test_skip_reasons_are_recorded__(log):
    """The transformer names why a context or a whole module was not sliced"""
    unmodelled = """
@script.indicator("t")
def main():
    try:
        x = 1.0
    finally:
        x = 2.0
    d = lib.request.security(lib.syminfo.tickerid, "1D", lib.ta.sma(lib.close, 3))
"""
    tree = ast.parse(unmodelled)
    tree = SecurityTransformer().visit(tree)
    transformer = SecuritySliceTransformer()
    transformer.visit(tree)
    assert transformer.module_skip is not None
    assert transformer.module_skip.startswith('unmodelled:Try:'), \
        f"unexpected module skip reason: {transformer.module_skip}"

    passthrough = """
@script.indicator("t")
def main():
    d = lib.request.security(lib.syminfo.tickerid, "1D", lib.close)
    lib.plot(d)
"""
    tree = ast.parse(passthrough)
    tree = SecurityTransformer().visit(tree)
    transformer = SecuritySliceTransformer()
    transformer.visit(tree)
    assert transformer.module_skip is None
    assert set(transformer.skipped.values()) == {'ohlcv_passthrough'}, \
        f"unexpected per-context skip reasons: {transformer.skipped}"
    log.info("both skip reasons were reported")


def __test_no_clone_for_lower_tf_or_passthrough__(log):
    """An LTF context and a plain-OHLCV one both keep the existing child path

    ``request.security_lower_tf`` is out of scope, and a plain-OHLCV expression
    already skips ``main()`` entirely in the child — a clone would only be dead
    weight there.
    """
    ltf = """
@script.indicator("t")
def main():
    d = lib.request.security_lower_tf(lib.syminfo.tickerid, "1", lib.ta.sma(lib.close, 3))
    lib.plot(lib.array.size(d))
"""
    passthrough = """
@script.indicator("t")
def main():
    d = lib.request.security(lib.syminfo.tickerid, "1D", lib.close)
    lib.plot(d)
"""
    for name, source in (("lower_tf", ltf), ("plain-OHLCV", passthrough)):
        tree = __test_helper_transform(source)
        assert not __test_helper_clones(tree), f"{name} context got a clone"
        assert not __test_helper_slice_names(tree), f"{name} recorded a slice_main"
    log.info("neither the LTF nor the plain-OHLCV context was sliced")


def __test_slice_follows_a_user_function__(log):
    """A helper's free-name writes are what its call site contributes"""
    source = """
@script.indicator("t")
def main():
    total: Persistent[float] = 0.0
    other: Persistent[float] = 0.0

    def bump():
        nonlocal total
        total = total + 1.0

    def unrelated():
        nonlocal other
        other = other + 1.0

    bump()
    unrelated()
    d = lib.request.security(lib.syminfo.tickerid, "1D", total)
"""
    tree = __test_helper_transform(source)

    calls = __test_helper_calls(__test_helper_only_clone(tree))
    assert 'bump' in calls, "the call that writes the read closure name was dropped"
    assert 'unrelated' not in calls, \
        "a helper writing nothing the slice reads was kept"
    log.info("the helper call survived through its free-name write")


def __test_slice_keeps_the_call_site_of_a_helper_write__(log):
    """A write block inside a helper keeps the statement that CALLS the helper

    The ``def`` alone binds a name; dropping the call would leave the context
    unwritten and the chart reading ``na`` forever. The unrelated helper, whose
    body holds no protocol call, may still go.
    """
    source = """
@script.indicator("t")
def main():
    def ratio(a, b):
        x = lib.ta.sma(lib.close, 3)
        return lib.request.security(a, "1D", x) / lib.request.security(b, "1D", x)

    def noise():
        return lib.ta.sma(lib.close, 9)

    lib.plot(noise())
    lib.plot(ratio("A:X", "B:Y"))
"""
    tree = __test_helper_transform(source)
    clones = __test_helper_clones(tree)
    assert len(clones) == 2, f"expected a clone per context, got {sorted(clones)}"
    for name, clone in clones.items():
        body = __test_helper_body(clone)
        assert 'ratio(' in body.replace('def ratio(', ''), \
            f"{name}: the call site of the helper holding the write was dropped"
        assert 'noise()' not in body.replace('def noise()', ''), \
            f"{name}: an unrelated helper call survived the slice"
    log.info("the helper's call site survived in both clones")


def __test_helper_slice_name_per_context(tree: ast.Module) -> list[str | None]:
    """The ``slice_main`` of every context, in context order (None when absent)."""
    names: list[str | None] = []
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        target = node.targets[0]
        if not (isinstance(target, ast.Name) and target.id == '__security_contexts__'):
            continue
        assert isinstance(node.value, ast.Dict)
        for ctx in node.value.values:
            assert isinstance(ctx, ast.Dict)
            found: str | None = None
            for key, value in zip(ctx.keys, ctx.values):
                if (isinstance(key, ast.Constant) and key.value == 'slice_main'
                        and isinstance(value, ast.Constant)):
                    found = value.value
            names.append(found)
    return names


def __test_grouped_contexts_share_one_clone__(log):
    """One group, one clone: it holds both write blocks and both cones

    The third context is on another timeframe, so it is its own group and keeps
    its own clone — and neither clone carries the other's input.
    """
    source = """
@script.indicator("t")
def main():
    fast = lib.ta.sma(lib.close, 3)
    slow = lib.ta.ema(lib.close, 9)
    noise = lib.ta.rma(lib.close, 21)
    a = lib.request.security(lib.syminfo.tickerid, "1D", fast)
    b = lib.request.security(lib.syminfo.tickerid, "1D", slow)
    c = lib.request.security(lib.syminfo.tickerid, "240", noise)
    lib.plot(a + b + c)
"""
    tree = __test_helper_transform(source)
    clones = __test_helper_clones(tree)
    names = __test_helper_slice_name_per_context(tree)

    assert names[0] == names[1] is not None, "the group members got different clones"
    assert names[2] not in (None, names[0]), "the lone context lost its own clone"
    assert len(clones) == 2, f"expected one clone per unit, got {sorted(clones)}"

    merged = __test_helper_body(clones[names[0]])
    assert merged.count('__sec_write__') == 2, "a group member's write block is missing"
    assert 'lib.ta.sma' in merged and 'lib.ta.ema' in merged, \
        "a group member's own input was dropped"
    assert 'lib.ta.rma' not in merged, "the other group's input leaked in"
    assert 'lib.plot' not in merged, "an unrelated plot survived the slice"

    alone = __test_helper_body(clones[names[2]])
    assert alone.count('__sec_write__') == 1, "the lone clone gained a foreign write block"
    assert 'lib.ta.rma' in alone and 'lib.ta.sma' not in alone
    log.info("the group shares one clone, the lone context keeps its own")


def __test_a_group_keeps_its_ohlcv_member__(log):
    """A plain-OHLCV member joins its group: its write block is one statement"""
    source = """
@script.indicator("t")
def main():
    fast = lib.ta.sma(lib.close, 3)
    a = lib.request.security(lib.syminfo.tickerid, "1D", fast)
    b = lib.request.security(lib.syminfo.tickerid, "1D", lib.close)
    lib.plot(a + b)
"""
    tree = __test_helper_transform(source)
    names = __test_helper_slice_name_per_context(tree)
    assert names[0] == names[1] is not None, \
        "the plain-OHLCV member did not join the group's clone"

    body = __test_helper_body(__test_helper_only_clone(tree))
    assert body.count('__sec_write__') == 2, "the OHLCV member's write block was dropped"
    assert 'lib.ta.sma' in body, "the sliced member's input was dropped"
    assert 'lib.plot' not in body
    log.info("the group clone carries the plain-OHLCV write block too")


def __test_an_all_ohlcv_group_gets_no_clone__(log):
    """Nothing in such a group needs ``main()``, so the fast path stays"""
    source = """
@script.indicator("t")
def main():
    a = lib.request.security(lib.syminfo.tickerid, "1D", lib.close)
    b = lib.request.security(lib.syminfo.tickerid, "1D", lib.high)
    lib.plot(a + b)
"""
    tree = ast.parse(source)
    tree = SecurityTransformer().visit(tree)
    transformer = SecuritySliceTransformer()
    tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)

    assert not __test_helper_clones(tree), "an all-OHLCV group got a clone"
    assert set(transformer.skipped.values()) == {'ohlcv_passthrough'}, \
        f"unexpected per-context skip reasons: {transformer.skipped}"
    log.info("the all-OHLCV group kept the main()-free child path")


def __test_helper_closed_shift(tree: ast.Module) -> list[bool | None]:
    """The ``closed_shift`` flag of every context, in context order."""
    flags: list[bool | None] = []
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        target = node.targets[0]
        if not (isinstance(target, ast.Name) and target.id == '__security_contexts__'):
            continue
        assert isinstance(node.value, ast.Dict)
        for ctx in node.value.values:
            assert isinstance(ctx, ast.Dict)
            found: bool | None = None
            for key, value in zip(ctx.keys, ctx.values):
                if (isinstance(key, ast.Constant) and key.value == 'closed_shift'
                        and isinstance(value, ast.Constant)):
                    found = value.value
            flags.append(found)
    return flags


def __test_closed_shift_survives_a_varip_free_slice__(log):
    """A slice with no ``varip`` of its own keeps the flag the transformer set

    The counter lives in the OTHER context's expression, and that context's
    write block is not in this one's slice — which is exactly the shape the
    round-sequence fingerprint of the equivalence tests needs.
    """
    source = """
@lib.script.indicator("t")
def main():
    def counted():
        n: IBPersistent[int] = 0
        n += 1
        return n
    shifted = lib.request.security(lib.syminfo.tickerid, "D", lib.close[1],
                               lookahead=lib.barmerge.lookahead_on)
    ticks = lib.request.security(lib.syminfo.tickerid, "60", counted(),
                            lookahead=lib.barmerge.lookahead_on)
    lib.plot(shifted)
    lib.plot(ticks)
"""
    tree = __test_helper_transform(source)
    assert __test_helper_closed_shift(tree) == [True, False]
    log.info("the shifted context kept closed_shift beside a varip context")


def __test_a_varip_in_the_slice_clears_closed_shift__(log):
    """A ``varip`` the child runs is not rolled back, so no round may be skipped"""
    source = """
@lib.script.indicator("t")
def main():
    tick: IBPersistent[int] = 0
    tick += 1
    shifted = lib.request.security(lib.syminfo.tickerid, "D", (lib.close + tick)[1],
                               lookahead=lib.barmerge.lookahead_on)
    other = lib.request.security(lib.syminfo.tickerid, "60", lib.ta.sma(lib.close, 3),
                             lookahead=lib.barmerge.lookahead_on)
    lib.plot(shifted)
    lib.plot(other)
"""
    tree = __test_helper_transform(source)
    assert __test_helper_closed_shift(tree) == [False, False]
    log.info("the varip declaration in the kept statements cleared the flag")


def __test_a_varip_helper_the_slice_calls_clears_closed_shift__(log):
    """The scan follows the calls: a counter one level down counts too"""
    source = """
@lib.script.indicator("t")
def main():
    def counted():
        n: IBPersistent[int] = 0
        n += 1
        return n
    shifted = lib.request.security(lib.syminfo.tickerid, "D", counted()[1],
                               lookahead=lib.barmerge.lookahead_on)
    other = lib.request.security(lib.syminfo.tickerid, "60", lib.ta.sma(lib.close, 3),
                             lookahead=lib.barmerge.lookahead_on)
    lib.plot(shifted)
    lib.plot(other)
"""
    tree = __test_helper_transform(source)
    assert __test_helper_closed_shift(tree) == [False, False]
    log.info("the varip inside the called helper cleared the flag")


def __test_a_per_execution_library_call_clears_closed_shift__(log):
    """``ta.valuewhen`` fills its ring once per execution, re-ticks included"""
    source = """
@lib.script.indicator("t")
def main():
    shifted = lib.request.security(lib.syminfo.tickerid, "D",
                               lib.ta.valuewhen(lib.close > lib.open, lib.close, 1)[1],
                               lookahead=lib.barmerge.lookahead_on)
    other = lib.request.security(lib.syminfo.tickerid, "60", lib.ta.sma(lib.close, 3),
                             lookahead=lib.barmerge.lookahead_on)
    lib.plot(shifted)
    lib.plot(other)
"""
    tree = __test_helper_transform(source)
    assert __test_helper_closed_shift(tree) == [False, False]
    log.info("the per-execution library call cleared the flag")


def __test_an_unsliced_context_is_judged_on_the_whole_main__(log):
    """Without a clone the child runs ``main()``, so its ``varip`` counts

    ``PYNE_NO_SECURITY_SLICE`` is the shape of it that needs no contrived
    script: no clone is emitted at all, every child runs the whole ``main()``,
    and the counter standing in that body is state the child carries.
    """
    import os

    source = """
@lib.script.indicator("t")
def main():
    tick: IBPersistent[int] = 0
    tick += 1
    shifted = lib.request.security(lib.syminfo.tickerid, "D", lib.close[1],
                                   lookahead=lib.barmerge.lookahead_on)
    lib.plot(shifted + tick)
"""
    assert __test_helper_closed_shift(__test_helper_transform(source)) == [True]
    os.environ['PYNE_NO_SECURITY_SLICE'] = '1'
    try:
        tree = __test_helper_transform(source)
    finally:
        os.environ.pop('PYNE_NO_SECURITY_SLICE', None)
    assert not __test_helper_clones(tree), "expected no clone"
    assert __test_helper_closed_shift(tree) == [False]
    log.info("the unsliced context was judged on the whole main()")


def __test_a_call_through_a_parameter_clears_closed_shift__(log):
    """An opaque callee may run anything, ``varip`` included

    A callable PARAMETER names no definition of the module, so the scan cannot
    see the counter it is handed — the context keeps its developing rounds.
    """
    source = """
@lib.script.indicator("t")
def main():
    def counter():
        n: IBPersistent[int] = 0
        n += 1
        return n
    def invoke(fn):
        return fn()
    shifted = lib.request.security(lib.syminfo.tickerid, "D", invoke(counter)[1],
                               lookahead=lib.barmerge.lookahead_on)
    other = lib.request.security(lib.syminfo.tickerid, "60", lib.ta.sma(lib.close, 3),
                             lookahead=lib.barmerge.lookahead_on)
    lib.plot(shifted)
    lib.plot(other)
"""
    tree = __test_helper_transform(source)
    assert __test_helper_closed_shift(tree) == [False, False]
    log.info("the call through the parameter cleared the flag")


def __test_an_imported_helper_clears_closed_shift__(log):
    """An import of another module binds a body this module does not hold

    Only ``pynecore`` imports are trusted (their per-execution state is
    enumerated); ``from helper import counter`` may well bring ``varip`` state
    along, so the context keeps its developing rounds.
    """
    source = """
from helper import counter

@lib.script.indicator("t")
def main():
    shifted = lib.request.security(lib.syminfo.tickerid, "D",
                               inline_series(counter(), 1),
                               lookahead=lib.barmerge.lookahead_on)
    other = lib.request.security(lib.syminfo.tickerid, "60", lib.ta.sma(lib.close, 3),
                             lookahead=lib.barmerge.lookahead_on)
    lib.plot(shifted)
    lib.plot(other)
"""
    tree = __test_helper_transform(source)
    assert __test_helper_closed_shift(tree) == [False, False]
    log.info("the imported helper cleared the flag")


def __test_a_parameter_shadowing_a_builtin_clears_closed_shift__(log):
    """A callable parameter named like a builtin is still opaque

    ``abs`` resolves without a body only as the BUILTIN; a parameter of that
    name stands for whatever was passed in, so the shadowed spelling must not
    answer for it.
    """
    source = """
@lib.script.indicator("t")
def main():
    def counter():
        n: IBPersistent[int] = 0
        n += 1
        return n
    def invoke(abs):
        return abs()
    shifted = lib.request.security(lib.syminfo.tickerid, "D",
                               inline_series(invoke(counter), 1),
                               lookahead=lib.barmerge.lookahead_on)
    other = lib.request.security(lib.syminfo.tickerid, "60", lib.ta.sma(lib.close, 3),
                             lookahead=lib.barmerge.lookahead_on)
    lib.plot(shifted)
    lib.plot(other)
"""
    tree = __test_helper_transform(source)
    assert __test_helper_closed_shift(tree) == [False, False]
    log.info("the builtin-shadowing parameter cleared the flag")


def __test_an_unnamed_callee_clears_closed_shift__(log):
    """A callee that is neither a name nor an attribute chain resolves to nothing

    ``getattr(helper, "counter")()`` calls the RESULT of a call and
    ``callbacks[0]()`` a subscript of a container: neither spelling names a body
    the scan could read, so both must keep the developing rounds.
    """
    source = """
import helper

@lib.script.indicator("t")
def main():
    shifted = lib.request.security(lib.syminfo.tickerid, "D",
                               inline_series(getattr(helper, "counter")(), 1),
                               lookahead=lib.barmerge.lookahead_on)
    other = lib.request.security(lib.syminfo.tickerid, "60", lib.ta.sma(lib.close, 3),
                             lookahead=lib.barmerge.lookahead_on)
    lib.plot(shifted)
    lib.plot(other)
"""
    tree = __test_helper_transform(source)
    assert __test_helper_closed_shift(tree) == [False, False]

    source = """
@lib.script.indicator("t")
def main():
    def counter():
        n: IBPersistent[int] = 0
        n += 1
        return n
    callbacks = [counter]
    shifted = lib.request.security(lib.syminfo.tickerid, "D",
                               inline_series(callbacks[0](), 1),
                               lookahead=lib.barmerge.lookahead_on)
    other = lib.request.security(lib.syminfo.tickerid, "60", lib.ta.sma(lib.close, 3),
                             lookahead=lib.barmerge.lookahead_on)
    lib.plot(shifted)
    lib.plot(other)
"""
    tree = __test_helper_transform(source)
    assert __test_helper_closed_shift(tree) == [False, False]
    log.info("the getattr call and the subscript callee both cleared the flag")


def __test_a_module_qualified_helper_clears_closed_shift__(log):
    """``helper.counter()`` is as opaque as the bare-import spelling of it

    A qualified call is resolvable only through the module its root names: the
    ``pynecore`` library namespace has its per-execution state enumerated, a
    user module does not, so the context keeps its developing rounds.
    """
    source = """
import helper

@lib.script.indicator("t")
def main():
    shifted = lib.request.security(lib.syminfo.tickerid, "D",
                               inline_series(helper.counter(), 1),
                               lookahead=lib.barmerge.lookahead_on)
    other = lib.request.security(lib.syminfo.tickerid, "60", lib.ta.sma(lib.close, 3),
                             lookahead=lib.barmerge.lookahead_on)
    lib.plot(shifted)
    lib.plot(other)
"""
    tree = __test_helper_transform(source)
    assert __test_helper_closed_shift(tree) == [False, False]
    log.info("the module-qualified helper cleared the flag")


def __test_an_import_shadowing_a_builtin_clears_closed_shift__(log):
    """``from helper import abs`` binds a user callable, not the builtin

    A builtin name resolves without a body only while nothing rebinds it; an
    import of a non-pynecore module is exactly such a rebinding, so the call
    must answer "may carry varip state" and keep the developing rounds.
    """
    source = """
from helper import abs

@lib.script.indicator("t")
def main():
    shifted = lib.request.security(lib.syminfo.tickerid, "D",
                               inline_series(abs(), 1),
                               lookahead=lib.barmerge.lookahead_on)
    other = lib.request.security(lib.syminfo.tickerid, "60", lib.ta.sma(lib.close, 3),
                             lookahead=lib.barmerge.lookahead_on)
    lib.plot(shifted)
    lib.plot(other)
"""
    tree = __test_helper_transform(source)
    assert __test_helper_closed_shift(tree) == [False, False]
    log.info("the builtin-shadowing import cleared the flag")


def __test_an_import_alias_shadowing_a_builtin_clears_closed_shift__(log):
    """``import helper as abs`` is the same rebinding one spelling further"""
    source = """
import helper as abs

@lib.script.indicator("t")
def main():
    shifted = lib.request.security(lib.syminfo.tickerid, "D",
                               inline_series(abs.counter(), 1),
                               lookahead=lib.barmerge.lookahead_on)
    other = lib.request.security(lib.syminfo.tickerid, "60", lib.ta.sma(lib.close, 3),
                             lookahead=lib.barmerge.lookahead_on)
    lib.plot(shifted)
    lib.plot(other)
"""
    tree = __test_helper_transform(source)
    assert __test_helper_closed_shift(tree) == [False, False]
    log.info("the aliased module import cleared the flag")


def __test_a_module_level_helper_is_scanned_through__(log):
    """A bare name the module defines is resolved and its body scanned (R1)

    The helper standing at module level is reached by name, so the ``varip`` in
    it counts; the same shape without ``varip`` keeps the flag.
    """
    source = """
def helper():
    n: IBPersistent[int] = 0
    n += 1
    return n

@lib.script.indicator("t")
def main():
    shifted = lib.request.security(lib.syminfo.tickerid, "D",
                               inline_series(helper(), 1),
                               lookahead=lib.barmerge.lookahead_on)
    lib.plot(shifted)
"""
    assert __test_helper_closed_shift(__test_helper_transform(source)) == [False]

    source = """
def helper():
    return lib.close * 2

@lib.script.indicator("t")
def main():
    shifted = lib.request.security(lib.syminfo.tickerid, "D",
                               inline_series(helper(), 1),
                               lookahead=lib.barmerge.lookahead_on)
    lib.plot(shifted)
"""
    assert __test_helper_closed_shift(__test_helper_transform(source)) == [True]
    log.info("the module-level helper was scanned through")


def __test_a_global_helper_clears_closed_shift__(log):
    """A module global the helper increments is outside the rollback (R0)

    The child's re-tick rollback restores the state vector, not the module's own
    names, so a period whose developing rounds are skipped ends on a different
    count.
    """
    source = """
ticks = 0

def counter():
    global ticks
    ticks += 1
    return float(ticks)

@lib.script.indicator("t")
def main():
    shifted = lib.request.security(lib.syminfo.tickerid, "D",
                               inline_series(counter(), 1),
                               lookahead=lib.barmerge.lookahead_on)
    lib.plot(shifted)
"""
    assert __test_helper_closed_shift(__test_helper_transform(source)) == [False]
    log.info("the global helper cleared the flag")


def __test_an_early_series_return_clears_closed_shift__(log):
    """``if close <= open: return`` is decided by the developing bar"""
    source = """
@lib.script.indicator("t")
def main():
    if lib.close <= lib.open:
        return
    shifted = lib.request.security(lib.syminfo.tickerid, "D", lib.close[1],
                                   lookahead=lib.barmerge.lookahead_on)
    lib.plot(shifted)
"""
    assert __test_helper_closed_shift(__test_helper_transform(source)) == [False]
    log.info("the series-guarded early return cleared the flag")


def __test_a_forced_series_guard_keeps_closed_shift__(log):
    """An ``if`` around the write is forced out of the clone, so the flag stays"""
    source = """
@lib.script.indicator("t")
def main():
    shifted = lib.na
    if lib.close > 0:
        shifted = lib.request.security(lib.syminfo.tickerid, "D", lib.close[1],
                                       lookahead=lib.barmerge.lookahead_on)
    lib.plot(shifted)
"""
    tree = __test_helper_transform(source)
    assert __test_helper_closed_shift(tree) == [True]
    clone = __test_helper_clones(tree)['__sec_main_0__']
    assert __test_helper_write_guards(clone) == []
    log.info("the child runs the write unguarded, and the flag is kept")


def __test_a_helper_called_unconditionally_keeps_closed_shift__(log):
    """The write sits in a helper body; the one call site is unconditional"""
    source = """
@lib.script.indicator("t")
def main():
    def htf(tf):
        return lib.request.security(lib.syminfo.tickerid, tf, lib.close[1],
                                    lookahead=lib.barmerge.lookahead_on)
    lib.plot(htf("D"))
"""
    assert __test_helper_closed_shift(__test_helper_transform(source)) == [True]
    log.info("the unconditional call site kept the flag")


def __test_a_helper_called_under_a_forced_series_guard_keeps_closed_shift__(log):
    """A call site behind a bar-data guard is forced out of the clone too"""
    source = """
@lib.script.indicator("t")
def main():
    def htf(tf):
        return lib.request.security(lib.syminfo.tickerid, tf, lib.close[1],
                                    lookahead=lib.barmerge.lookahead_on)
    shifted = lib.na
    if lib.close > lib.open:
        shifted = htf("D")
    lib.plot(shifted)
"""
    assert __test_helper_closed_shift(__test_helper_transform(source)) == [True]
    log.info("the forced call site kept the flag")


def __test_a_helper_called_in_a_loop_clears_closed_shift__(log):
    """A call site inside a ``for`` is not modelled, so the write is not either"""
    source = """
@lib.script.indicator("t")
def main():
    def htf(tf):
        return lib.request.security(lib.syminfo.tickerid, tf, lib.close[1],
                                    lookahead=lib.barmerge.lookahead_on)
    shifted = lib.na
    for i in range(2):
        shifted = htf("D")
    lib.plot(shifted)
"""
    assert __test_helper_closed_shift(__test_helper_transform(source)) == [False]
    log.info("the call site inside the loop cleared the flag")


def __test_an_early_raise_clears_closed_shift__(log):
    """A ``raise`` ahead of the write is an early exit like a ``return``"""
    in_main = """
@lib.script.indicator("t")
def main():
    if lib.close <= lib.open:
        raise ValueError
    shifted = lib.request.security(lib.syminfo.tickerid, "D", lib.close[1],
                                   lookahead=lib.barmerge.lookahead_on)
    lib.plot(shifted)
"""
    in_helper = """
@lib.script.indicator("t")
def main():
    def htf(tf):
        if lib.close <= lib.open:
            raise ValueError
        return lib.request.security(lib.syminfo.tickerid, tf, lib.close[1],
                                    lookahead=lib.barmerge.lookahead_on)
    lib.plot(htf("D"))
"""
    assert __test_helper_closed_shift(__test_helper_transform(in_main)) == [False]
    assert __test_helper_closed_shift(__test_helper_transform(in_helper)) == [False]
    log.info("the early raise cleared the flag in main and in a helper")


def __test_an_early_return_in_a_helper_clears_closed_shift__(log):
    """An early exit ahead of the write may skip it for some bars"""
    source = """
@lib.script.indicator("t")
def main():
    def htf(tf):
        if lib.close <= lib.open:
            return lib.na
        return lib.request.security(lib.syminfo.tickerid, tf, lib.close[1],
                                    lookahead=lib.barmerge.lookahead_on)
    lib.plot(htf("D"))
"""
    assert __test_helper_closed_shift(__test_helper_transform(source)) == [False]
    log.info("the early return inside the helper cleared the flag")


def __test_two_helper_copies_each_keep_closed_shift__(log):
    """Both isolation copies of a helper are called unconditionally"""
    source = """
@lib.script.indicator("t")
def main():
    def htf(tf):
        return lib.request.security(lib.syminfo.tickerid, tf, lib.close[1],
                                    lookahead=lib.barmerge.lookahead_on)
    def htf__pyne_inst1(tf):
        return lib.request.security(lib.syminfo.tickerid, tf, lib.close[1],
                                    lookahead=lib.barmerge.lookahead_on)
    lib.plot(htf("D"))
    lib.plot(htf__pyne_inst1("W"))
"""
    assert __test_helper_closed_shift(__test_helper_transform(source)) == [True, True]
    log.info("both isolation copies kept the flag")


def __test_only_a_write_run_on_every_round_keeps_closed_shift__(log):
    """The flag survives a write the child runs on every round, and no other

    The write below stands at the top level of a helper that the entry calls at
    its own top level, so it runs on every round. An ``if`` around the very same
    write is forced out of the child's clone, so it keeps the flag as well. A
    loop around it cannot be forced, and clears the flag.
    """
    unguarded = """
@lib.script.indicator("t")
def main():
    def htf():
        return lib.request.security(lib.syminfo.tickerid, "D", lib.close[1],
                                    lookahead=lib.barmerge.lookahead_on)
    lib.plot(htf())
"""
    guarded = """
@lib.script.indicator("t")
def main():
    flag = lib.input.bool(True, "use")
    shifted = lib.na
    if flag:
        shifted = lib.request.security(lib.syminfo.tickerid, "D", lib.close[1],
                                       lookahead=lib.barmerge.lookahead_on)
    lib.plot(shifted)
"""
    looped = """
@lib.script.indicator("t")
def main():
    shifted = lib.na
    for _i in range(2):
        shifted = lib.request.security(lib.syminfo.tickerid, "D", lib.close[1],
                                       lookahead=lib.barmerge.lookahead_on)
    lib.plot(shifted)
"""
    assert __test_helper_closed_shift(__test_helper_transform(unguarded)) == [True]
    assert __test_helper_closed_shift(__test_helper_transform(guarded)) == [True]
    assert __test_helper_closed_shift(__test_helper_transform(looped)) == [False]
    log.info("the unguarded and the forced write kept the flag, the looped one lost it")


def __test_helper_write_guards(clone: ast.FunctionDef) -> list[str]:
    """The user guards still standing around a write or a helper call in a clone."""
    guards: list[str] = []
    for node in ast.walk(clone):
        if isinstance(node, ast.If) and '__active_security__' not in ast.unparse(node.test):
            # The user's ``if`` itself stays; what may not stay under it is a
            # write or a call of the writing helper.
            guarded = ast.unparse(ast.Module(body=[*node.body, *node.orelse], type_ignores=[]))
            if '__sec_write__' in guarded or 'htf(' in guarded:
                guards.append(ast.unparse(node.test))
        elif isinstance(node, (ast.IfExp, ast.BoolOp)) and 'htf' in ast.unparse(node):
            guards.append(ast.unparse(node))
    return guards


def __test_only_the_write_leaves_its_branch__(log):
    """The branch keeps its guard: the write is lifted with the assignment it reads

    The slicing is switched off so the clone holds the whole ``main()``: the
    drawing and the conditionally advancing counter must stay under the ``if``.
    """
    import os
    source = """
@lib.script.indicator("t")
def main():
    count: Persistent[int] = 0
    value = lib.na
    if lib.bar_index % 3 != 1:
        count += 1
        length = 3
        lib.label.new(lib.bar_index, lib.high, "x")
        value = lib.request.security(lib.syminfo.tickerid, "D", lib.ta.sma(lib.close, length))
    lib.plot(value)
    lib.plot(count)
"""
    os.environ['PYNE_NO_SECURITY_SLICE'] = '1'
    try:
        clone = __test_helper_only_clone(__test_helper_transform(source))
    finally:
        os.environ.pop('PYNE_NO_SECURITY_SLICE', None)
    assert __test_helper_write_guards(clone) == []
    guard = next(node for node in clone.body
                 if isinstance(node, ast.If) and 'bar_index' in ast.unparse(node.test))
    guarded = ast.unparse(ast.Module(body=guard.body, type_ignores=[]))
    assert 'count += 1' in guarded and 'label.new' in guarded
    ahead = ast.unparse(ast.Module(body=clone.body[:clone.body.index(guard)], type_ignores=[]))
    # The copy binds a private name: the original still runs under the guard,
    # so a copy writing ``length`` would leave the branch's value behind on a
    # pass that never took the branch
    assert '__sec_dep' in ahead and '= 3' in ahead and '__sec_write__' in ahead
    assert 'length = 3' not in ahead
    assert 'count += 1' not in ahead and 'label.new' not in ahead
    assert 'length = 3' in guarded
    log.info("write and its plain assignment lifted, counter and drawing still guarded")


def __test_an_alias_of_a_mutated_collection_keeps_the_write_guarded__(log):
    """A mutation written through a second name still holds the write back

    ``alias = store`` makes both names denote one array, so ``lib.array.set()``
    on the alias is a mutation of what the request reads through ``store``.
    Lifting the write ahead of it would publish what the array held before the
    branch ran.
    """
    import os
    source = """
@lib.script.indicator("t")
def main():
    store = lib.array.new_float(1, 2.0)
    alias = store
    value = lib.na
    if lib.bar_index % 3 != 1:
        lib.array.set(alias, 0, 3.0)
        value = lib.request.security(lib.syminfo.tickerid, "D", lib.array.get(store, 0))
    lib.plot(value)
"""
    os.environ['PYNE_NO_SECURITY_SLICE'] = '1'
    try:
        tree = __test_helper_transform(source)
    finally:
        os.environ.pop('PYNE_NO_SECURITY_SLICE', None)
    assert __test_helper_clones(tree) == {}
    guarded = __test_helper_guarded_write(tree)
    assert '__sec_write__' in guarded and 'array.set' in guarded
    log.info("the alias tied the write to the mutation")


def __test_an_annotated_alias_of_a_mutated_collection_keeps_the_write_guarded__(log):
    """An annotation does not turn an alias into a fresh object

    ``alias: list[float] = store`` binds the very same array as ``alias =
    store`` does, so the mutation written through the alias must hold the
    write back exactly as the plain spelling does.
    """
    import os
    source = """
@lib.script.indicator("t")
def main():
    store = lib.array.new_float(1, 2.0)
    alias: list[float] = store
    value = lib.na
    if lib.bar_index % 3 != 1:
        lib.array.set(alias, 0, 3.0)
        value = lib.request.security(lib.syminfo.tickerid, "D", lib.array.get(store, 0))
    lib.plot(value)
"""
    os.environ['PYNE_NO_SECURITY_SLICE'] = '1'
    try:
        tree = __test_helper_transform(source)
    finally:
        os.environ.pop('PYNE_NO_SECURITY_SLICE', None)
    assert __test_helper_clones(tree) == {}
    guarded = __test_helper_guarded_write(tree)
    assert '__sec_write__' in guarded and 'array.set' in guarded
    log.info("the annotated alias tied the write to the mutation")


def __test_a_lifted_dependency_does_not_change_its_own_guard__(log):
    """A copied dependency binds a private name, not the one the test reads

    ``x`` is both the guard's input and the request's: a copy keeping the name
    would run ``x = lib.open`` before ``x > 100``, turning a true branch false
    and leaving that value behind on a pass that never took the branch.
    """
    import os
    source = """
@lib.script.indicator("t")
def main():
    x = lib.close
    value = lib.na
    if x > 100:
        x = lib.open
        value = lib.request.security(lib.syminfo.tickerid, "D", x)
    lib.plot(value)
    lib.plot(x)
"""
    os.environ['PYNE_NO_SECURITY_SLICE'] = '1'
    try:
        clone = __test_helper_only_clone(__test_helper_transform(source))
    finally:
        os.environ.pop('PYNE_NO_SECURITY_SLICE', None)
    assert __test_helper_write_guards(clone) == []
    guard = next(node for node in clone.body
                 if isinstance(node, ast.If) and 'x > 100' in ast.unparse(node.test))
    ahead = ast.unparse(ast.Module(body=clone.body[:clone.body.index(guard)], type_ignores=[]))
    assert '__sec_dep' in ahead and '__sec_write__' in ahead
    assert 'x = lib.open' not in ahead
    assert 'x = lib.open' in ast.unparse(ast.Module(body=guard.body, type_ignores=[]))
    log.info("the guard still reads the value it read before the lift")


def __test_helper_guarded_write(tree: ast.Module) -> str:
    """The body of ``main()``'s ``bar_index`` guard, as source text."""
    main = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == 'main')
    guard = next(node for node in ast.walk(main)
                 if isinstance(node, ast.If) and 'bar_index' in ast.unparse(node.test))
    return ast.unparse(ast.Module(body=guard.body, type_ignores=[]))


def __test_a_collection_mutated_before_the_request_keeps_the_write_guarded__(log):
    """A branch effect the request reads is never overtaken by the lift

    ``lib.array.set()`` binds no name, so only the value it touches ties it to
    the request: lifting the write ahead of it would publish what the
    collection held before the branch ran. Nothing can be lifted here, so no
    forced clone is emitted at all and the write stays under its guard.
    """
    import os
    source = """
@lib.script.indicator("t")
def main():
    store = lib.array.new_float(1, 2.0)
    value = lib.na
    if lib.bar_index % 3 != 1:
        lib.array.set(store, 0, 3.0)
        value = lib.request.security(lib.syminfo.tickerid, "D", lib.array.get(store, 0))
    lib.plot(value)
"""
    os.environ['PYNE_NO_SECURITY_SLICE'] = '1'
    try:
        tree = __test_helper_transform(source)
    finally:
        os.environ.pop('PYNE_NO_SECURITY_SLICE', None)
    assert __test_helper_clones(tree) == {}
    guarded = __test_helper_guarded_write(tree)
    assert '__sec_write__' in guarded and 'array.set' in guarded
    log.info("the write stayed behind the mutation it reads")


def __test_a_self_referential_assignment_keeps_the_write_guarded__(log):
    """A dependency reading what the same branch binds is not copied ahead

    ``length = length + 1`` cannot be copied in front of the guard: the
    original runs again under it, so the counter would advance twice.
    """
    import os
    source = """
@lib.script.indicator("t")
def main():
    length = 2
    value = lib.na
    if lib.bar_index % 3 != 1:
        length = length + 1
        value = lib.request.security(lib.syminfo.tickerid, "D", lib.ta.sma(lib.close, length))
    lib.plot(value)
    lib.plot(length)
"""
    os.environ['PYNE_NO_SECURITY_SLICE'] = '1'
    try:
        tree = __test_helper_transform(source)
    finally:
        os.environ.pop('PYNE_NO_SECURITY_SLICE', None)
    assert __test_helper_clones(tree) == {}
    guarded = __test_helper_guarded_write(tree)
    assert guarded.count('length = length + 1') == 1 and '__sec_write__' in guarded
    log.info("the increment runs once, under its own guard")


def __test_a_guarded_helper_call_is_forced_in_the_clone__(log):
    """A ternary or a short circuit around the writing helper's call is removed"""
    ternary = """
@lib.script.indicator("t")
def main():
    def htf():
        return lib.request.security(lib.syminfo.tickerid, "D", lib.ta.sma(lib.close, 3))
    gate = lib.bar_index % 3 != 1
    lib.plot(htf() if gate else lib.na)
"""
    short_circuit = """
@lib.script.indicator("t")
def main():
    def htf():
        return lib.request.security(lib.syminfo.tickerid, "D", lib.close > lib.open)
    gate = lib.bar_index % 3 != 1
    flag = gate and htf()
    lib.plot(1 if flag else 0)
"""
    for name, source in (("ternary", ternary), ("short circuit", short_circuit)):
        clone = __test_helper_only_clone(__test_helper_transform(source))
        assert __test_helper_write_guards(clone) == [], name
        assert 'htf()' in __test_helper_body(clone), name
    log.info("neither expression-level guard survived in the clone")


def __test_a_context_in_each_branch_keeps_both_writes__(log):
    """``if`` / ``else`` holding a context each: the child runs both branches"""
    source = """
@lib.script.indicator("t")
def main():
    gate = lib.bar_index % 3 != 1
    value = 0.0
    if gate:
        value = lib.request.security(lib.syminfo.tickerid, "D", lib.ta.sma(lib.high, 3))
    else:
        value = lib.request.security(lib.syminfo.tickerid, "D", lib.ta.sma(lib.low, 3))
    lib.plot(value)
"""
    tree = __test_helper_transform(source)
    names = __test_helper_slice_name_per_context(tree)
    assert names[0] is not None and names[0] == names[1], names
    clone = __test_helper_clones(tree)[names[0]]
    assert __test_helper_write_guards(clone) == []
    body = __test_helper_body(clone)
    assert 'lib.high' in body and 'lib.low' in body
    log.info("the one clone of the group writes both contexts unguarded")


def __test_a_guarded_write_gets_a_whole_clone_where_nothing_is_sliced__(log):
    """An LTF context, or any context while slicing is off, is still forced"""
    import os

    ltf_guarded = """
@lib.script.indicator("t")
def main():
    vals = lib.array.new_float(0)
    if lib.bar_index % 3 != 1:
        vals = lib.request.security_lower_tf(lib.syminfo.tickerid, "1", lib.ta.sma(lib.close, 3))
    lib.plot(lib.array.sum(vals))
"""
    htf_guarded = """
@lib.script.indicator("t")
def main():
    value = 0.0
    if lib.bar_index % 3 != 1:
        value = lib.request.security(lib.syminfo.tickerid, "D", lib.ta.sma(lib.close, 3))
    lib.plot(value)
"""
    htf_plain = """
@lib.script.indicator("t")
def main():
    lib.plot(lib.request.security(lib.syminfo.tickerid, "D", lib.ta.sma(lib.close, 3)))
"""
    clone = __test_helper_only_clone(__test_helper_transform(ltf_guarded))
    assert __test_helper_write_guards(clone) == []
    # Nothing was dropped: the chart-side statements are all still there
    assert 'lib.plot' in __test_helper_body(clone)

    os.environ['PYNE_NO_SECURITY_SLICE'] = '1'
    try:
        clone = __test_helper_only_clone(__test_helper_transform(htf_guarded))
        assert __test_helper_write_guards(clone) == []
        assert 'lib.plot' in __test_helper_body(clone)
        assert __test_helper_clones(__test_helper_transform(htf_plain)) == {}
    finally:
        os.environ.pop('PYNE_NO_SECURITY_SLICE', None)
    log.info("the unsliced contexts got a forced clone of the whole main()")


def __test_a_module_level_helper_is_forced_on_a_private_copy__(log):
    """The guard inside a shared helper goes from the clone's copy only"""
    source = """
def htf(flag):
    value = lib.na
    if flag:
        value = lib.request.security(lib.syminfo.tickerid, "D", lib.ta.sma(lib.close, 3))
    return value

@lib.script.indicator("t")
def main():
    lib.plot(htf(lib.bar_index % 3 != 1))
"""
    tree = __test_helper_transform(source)
    clone = __test_helper_only_clone(tree)
    copies = [node for node in clone.body
              if isinstance(node, ast.FunctionDef) and node.name == 'htf']
    assert len(copies) == 1
    assert __test_helper_write_guards(copies[0]) == []
    shared = [node for node in tree.body
              if isinstance(node, ast.FunctionDef) and node.name == 'htf']
    assert len(shared) == 1
    assert __test_helper_write_guards(shared[0]) == ['flag']
    log.info("main() still calls the guarded helper, the clone its forced copy")


def __test_a_method_mutation_of_a_script_value_keeps_the_write_guarded__(log):
    """A method receiver is a script value the branch may mutate

    ``store.append(3.0)`` mutates ``store`` exactly as ``lib.array.push()``
    does; only the receiver name ties the effect to the request, so the write
    must stay behind it instead of publishing the collection as it was before
    the branch ran.
    """
    import os
    source = """
@lib.script.indicator("t")
def main():
    store = lib.array.new_float(1, 2.0)
    value = lib.na
    if lib.bar_index % 3 != 1:
        store.append(3.0)
        value = lib.request.security(lib.syminfo.tickerid, "D", lib.array.size(store))
    lib.plot(value)
"""
    os.environ['PYNE_NO_SECURITY_SLICE'] = '1'
    try:
        tree = __test_helper_transform(source)
    finally:
        os.environ.pop('PYNE_NO_SECURITY_SLICE', None)
    assert __test_helper_clones(tree) == {}
    guarded = __test_helper_guarded_write(tree)
    assert '__sec_write__' in guarded and 'store.append' in guarded
    log.info("the method receiver tied the write to the mutation")


def __test_a_subscript_dependency_is_never_copied_ahead_of_its_guard__(log):
    """The guard is what makes the index valid, so the copy may not run

    ``x = store[0]`` holds no call, but the ``array.size()`` test above it is
    what establishes that element: evaluating a copy on a pass that never
    takes the branch would raise where the original never runs.
    """
    import os
    source = """
@lib.script.indicator("t")
def main():
    store = lib.array.new_float(0)
    value = lib.na
    if lib.array.size(store) > 0:
        x = store[0]
        value = lib.request.security(lib.syminfo.tickerid, "D", x + lib.close)
    lib.plot(value)
"""
    os.environ['PYNE_NO_SECURITY_SLICE'] = '1'
    try:
        tree = __test_helper_transform(source)
    finally:
        os.environ.pop('PYNE_NO_SECURITY_SLICE', None)
    assert __test_helper_clones(tree) == {}
    main = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == 'main')
    guard = next(node for node in ast.walk(main)
                 if isinstance(node, ast.If) and 'array.size' in ast.unparse(node.test))
    guarded = ast.unparse(ast.Module(body=guard.body, type_ignores=[]))
    assert '__sec_write__' in guarded and 'store[0]' in guarded
    # No private copy of the subscript was hoisted ahead of the guard
    assert '__sec_dep' not in ast.unparse(main)
    log.info("the subscript dependency stayed under the guard")



def __test_helper_pipeline(source: str) -> ast.Module:
    """Run the whole analysis half of the pipeline over a source string.

    The slicer's copy analysis reads the tree the earlier passes produce, so a
    dependency hoist is only visible end to end.

    :param source: The module source.
    :return: The analysed module, locations fixed up for unparsing.
    """
    import os
    from pathlib import Path
    from pynecore.core.import_hook import _analyse_tree, _module_mode
    os.environ['PYNE_SAVE_SCRIPT_TOML'] = '0'
    tree = ast.parse(source)
    path = Path(__file__).parent / 'security_slice_probe.py'
    tree = _analyse_tree(tree, source, path, _module_mode(tree)[1])
    ast.fix_missing_locations(tree)
    return tree


def __test_an_attribute_dependency_is_never_copied_ahead_of_its_guard__(log):
    """The guard is what makes the attribute owner valid, so the copy may not run

    ``x = item.value`` holds no call, but the ``item is not None`` test above it
    is what establishes the owner: a copy evaluated on a pass that never takes
    the branch would read the attribute off ``na`` and raise where the original
    never runs.
    """
    source = '''"""
@pyne
"""
from pynecore import lib
from pynecore.lib import script, request, syminfo, close, na, plot

@script.indicator("t")
def main(item=None):
    value = na
    if item is not None:
        x = item.value
        value = request.security(syminfo.tickerid, "D", x + close)
    plot(value)
'''
    tree = __test_helper_pipeline(source)
    clone = __test_helper_only_clone(tree)
    body = __test_helper_body(clone)
    # The attribute read stayed under its own guard, with no copy ahead of it
    assert '__sec_dep' not in body, body
    guard = next(node for node in ast.walk(clone)
                 if isinstance(node, ast.If) and 'is not None' in ast.unparse(node.test)
                 and '__active_security__' not in ast.unparse(node.test))
    guarded = ast.unparse(ast.Module(body=guard.body, type_ignores=[]))
    assert '__sec_write__' in guarded and 'item.value' in guarded, guarded
    log.info("the attribute dependency stayed under the guard")


def __test_an_arithmetic_dependency_is_never_copied_ahead_of_its_guard__(log):
    """Operators may raise outside their guard, so no operand copy may run early

    ``x = 10 // item`` holds no call, no subscript and no attribute, yet the
    ``item != 0`` test above it is what makes the division defined: a copy
    evaluated on a pass that never takes the branch raises ``ZeroDivisionError``
    where the original never runs.
    """
    source = '''"""
@pyne
"""
from pynecore import lib
from pynecore.lib import script, request, syminfo, close, na, plot

@script.indicator("t")
def main(item=0):
    value = na
    if item != 0:
        x = 10 // item
        value = request.security(syminfo.tickerid, "D", x + close)
    plot(value)
'''
    tree = __test_helper_pipeline(source)
    clone = __test_helper_only_clone(tree)
    body = __test_helper_body(clone)
    # The division stayed under its own guard, with no copy ahead of it
    assert '__sec_dep' not in body, body
    guard = next(node for node in ast.walk(clone)
                 if isinstance(node, ast.If) and '!= 0' in ast.unparse(node.test)
                 and '__active_security__' not in ast.unparse(node.test))
    guarded = ast.unparse(ast.Module(body=guard.body, type_ignores=[]))
    assert '__sec_write__' in guarded and '//' in guarded, guarded
    log.info("the arithmetic dependency stayed under the guard")


def __test_an_unpacking_dependency_is_never_copied_ahead_of_its_guard__(log):
    """Unpacking needs the length the guard establishes, so no copy may run early

    ``x, y = items`` holds no call and no subscript, yet the ``len(items) == 2``
    test above it is what makes the unpacking match: a copy evaluated on a pass
    that never takes the branch raises ``ValueError`` where the original never
    runs.
    """
    source = '''"""
@pyne
"""
from pynecore import lib
from pynecore.lib import script, request, syminfo, close, na, plot

@script.indicator("t")
def main(items=()):
    value = na
    if len(items) == 2:
        x, y = items
        value = request.security(syminfo.tickerid, "D", x + close)
    plot(value)
'''
    tree = __test_helper_pipeline(source)
    clone = __test_helper_only_clone(tree)
    body = __test_helper_body(clone)
    # The unpacking stayed under its own guard, with no copy ahead of it
    assert '__sec_dep' not in body, body
    guard = next(node for node in ast.walk(clone)
                 if isinstance(node, ast.If) and 'len(items)' in ast.unparse(node.test))
    guarded = ast.unparse(ast.Module(body=guard.body, type_ignores=[]))
    assert '__sec_write__' in guarded, guarded
    log.info("the unpacking dependency stayed under the guard")


def __test_a_conditionally_bound_local_is_never_read_ahead_of_its_guard__(log):
    """A local only a guard binds may not be read in front of another guard

    ``original`` is bound in the first ``if`` alone, so the second ``if`` may
    not be lifted: on a pass taking neither branch the lifted read would raise
    ``UnboundLocalError`` where the original never runs.
    """
    source = '''"""
@pyne
"""
from pynecore import lib
from pynecore.lib import script, request, syminfo, close, na, plot

@script.indicator("t")
def main(flag=False):
    value = na
    if flag:
        original = close
    if flag:
        x = original
        value = request.security(syminfo.tickerid, "D", x + close)
    plot(value)
'''
    tree = __test_helper_pipeline(source)
    clone = __test_helper_only_clone(tree)
    body = __test_helper_body(clone)
    assert '__sec_dep' not in body, body
    guard = next(node for node in ast.walk(clone)
                 if isinstance(node, ast.If) and 'flag' in ast.unparse(node.test)
                 and 'x = original' in ast.unparse(ast.Module(body=node.body,
                                                              type_ignores=[])))
    guarded = ast.unparse(ast.Module(body=guard.body, type_ignores=[]))
    assert '__sec_write__' in guarded, guarded
    log.info("the conditionally bound local stayed under the guard")


def __test_an_annotation_only_declaration_binds_no_value__(log):
    """A declared but unassigned local may not be read in front of a guard

    ``original: float`` makes the name a local without giving it a value, so
    the only binding is the one the first ``if`` writes: lifting the second
    ``if`` would read it unbound.
    """
    source = '''"""
@pyne
"""
from pynecore import lib
from pynecore.lib import script, request, syminfo, close, na, plot

@script.indicator("t")
def main(flag=False):
    value = na
    original: float
    if flag:
        original = close
    if flag:
        x = original
        value = request.security(syminfo.tickerid, "D", x + close)
    plot(value)
'''
    tree = __test_helper_pipeline(source)
    clone = __test_helper_only_clone(tree)
    body = __test_helper_body(clone)
    assert '__sec_dep' not in body, body
    log.info("the annotation-only declaration did not license the lift")


def __test_a_deleted_local_is_not_certainly_bound__(log):
    """A local a branch may delete may not be read in front of a later guard

    ``original`` is bound unconditionally but ``del`` may take that binding
    away, so the read must stay under its own guard.
    """
    source = '''"""
@pyne
"""
from pynecore import lib
from pynecore.lib import script, request, syminfo, close, na, plot

@script.indicator("t")
def main(flag=False):
    value = na
    original = close
    if flag:
        del original
    if flag:
        x = original
        value = request.security(syminfo.tickerid, "D", x + close)
    plot(value)
'''
    module = ast.unparse(__test_helper_pipeline(source))
    assert '__sec_dep' not in module, module
    log.info("the deleted local did not license the lift")


def __test_a_short_circuited_walrus_binds_nothing_certain__(log):
    """An assignment inside a short-circuiting expression is not a binding

    ``flag and (original := close)`` writes ``original`` only when ``flag`` is
    true, so the later guard's read of it may not run ahead of the guard.
    """
    source = '''"""
@pyne
"""
from pynecore import lib
from pynecore.lib import script, request, syminfo, close, na, plot

@script.indicator("t")
def main(flag=False):
    value = na
    flag and (original := close)
    if flag:
        x = original
        value = request.security(syminfo.tickerid, "D", x + close)
    plot(value)
'''
    tree = __test_helper_pipeline(source)
    clone = __test_helper_only_clone(tree)
    body = __test_helper_body(clone)
    assert '__sec_dep' not in body, body
    log.info("the short-circuited walrus did not license the lift")


def __test_an_except_handler_target_is_not_certainly_bound__(log):
    """A name an ``except ... as`` handler holds is unbound when it leaves

    Python deletes the handler's target on exit, so the earlier
    ``original = close`` no longer holds after the ``try``: the later read
    must stay under its own guard.
    """
    source = '''"""
@pyne
"""
from pynecore import lib
from pynecore.lib import script, request, syminfo, close, na, plot

@script.indicator("t")
def main(flag=False):
    value = na
    original = close
    try:
        pass
    except ValueError as original:
        pass
    if flag:
        x = original
        value = request.security(syminfo.tickerid, "D", x + close)
    plot(value)
'''
    module = ast.unparse(__test_helper_pipeline(source))
    assert '__sec_dep' not in module, module
    log.info("the except-handler target did not license the lift")
