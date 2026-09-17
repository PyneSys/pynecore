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
