"""
Behavior tests of the post-lowering ``closed_shift`` verifier.

The source-level classifier in ``SecurityTransformer`` is only the candidate:
it must never be wrong in the direction that SKIPS a developing round, and
whether a subscript is a history read is a fact of the LOWERED tree. These
tests run both halves of the real pipeline and check the flag as the runtime
gets it — a candidate whose lowered write indexes anything but a series slot or
an ``inline_series`` call is cleared, alone or together with its merge group.
"""
import ast
from pathlib import Path

from pynecore.core.import_hook import _analyse_tree, _lower_tree
from pynecore.transformers.security_closed_shift_check import _binds_name, _name_binding


def __test_helper_flags(source: str) -> list[bool | None]:
    """The ``closed_shift`` flag of every context after the full pipeline.

    :param source: The module source.
    :return: One flag per context in context order, None where there is none.
    """
    path = Path('<closed-shift-test>.py')
    analysed = _analyse_tree(ast.parse(source), source, path, None)
    lowered, _ = _lower_tree(analysed, path, None, emit_layout=False)
    flags: list[bool | None] = []
    for stmt in lowered.body:
        if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1):
            continue
        target = stmt.targets[0]
        if not (isinstance(target, ast.Name)
                and target.id == '__security_contexts__'):
            continue
        assert isinstance(stmt.value, ast.Dict)
        for ctx in stmt.value.values:
            assert isinstance(ctx, ast.Dict)
            found: bool | None = None
            for key, value in zip(ctx.keys, ctx.values):
                if (isinstance(key, ast.Constant) and key.value == 'closed_shift'
                        and isinstance(value, ast.Constant)):
                    found = value.value
            flags.append(found)
    return flags


def __test_a_slot_history_read_keeps_the_flag__(log):
    """The two forms that DO index history survive the exact check

    ``x: Series[float] = lib.close`` read as ``x[1]`` lowers to a slot
    subscript, and the compiled ``inline_series(expr, 1)`` stays the call the
    verifier recognises.
    """
    source = """
@lib.script.indicator("T")
def main():
    x: Series[float] = lib.close
    a = lib.request.security(lib.syminfo.tickerid, "D", x[1],
                             lookahead=lib.barmerge.lookahead_on)
    lib.plot(a)
"""
    assert __test_helper_flags(source) == [True]

    source = """
@lib.script.indicator("T")
def main():
    a = lib.request.security(lib.syminfo.tickerid, "D",
                             inline_series(lib.ta.sma(lib.close, 3), 1),
                             lookahead=lib.barmerge.lookahead_on)
    lib.plot(a)
"""
    assert __test_helper_flags(source) == [True]
    log.info("the slot read and the inline_series call both kept the flag")


def __test_a_list_indexed_above_its_declaration_is_cleared__(log):
    """A plain list read as ``data[1]`` is element access, not history

    The series pass registers the slot where it REACHES the declaration, so the
    write lifted above it indexes the LIST. The candidate says True (the name is
    declared a series somewhere in the scope), the verifier clears it; the
    second context, whose read stands below the declaration, keeps it.
    """
    source = """
@lib.script.indicator("T")
def main():
    data = [lib.close, lib.close]
    a = lib.request.security(lib.syminfo.tickerid, "D", data[1],
                             lookahead=lib.barmerge.lookahead_on)
    data: Series[float] = lib.close
    b = lib.request.security(lib.syminfo.tickerid, "240", data[1],
                             lookahead=lib.barmerge.lookahead_on)
    lib.plot(a + b)
"""
    assert __test_helper_flags(source) == [False, True]
    log.info("the read above the declaration was cleared, the one below kept")


def __test_a_reference_in_the_declaring_statement_is_cleared__(log):
    """Extracting the request lifts the write ABOVE the declaration

    The name still holds the plain list where the write lands, so the subscript
    is element access there too.
    """
    source = """
@lib.script.indicator("T")
def main():
    data = [lib.close, lib.close]
    data: Series[float] = lib.request.security(
        lib.syminfo.tickerid, "D", data[1], lookahead=lib.barmerge.lookahead_on)
    lib.plot(data)
"""
    assert __test_helper_flags(source) == [False]
    log.info("the reference in the declaring statement was cleared")


def __test_an_alias_of_a_list_is_cleared__(log):
    """``alias = data`` then ``alias[1]``: the alias is no series either

    Nothing declares ``alias`` a series, so the lowered write indexes the list
    the name holds — the candidate's own answer here is False as well, and the
    verifier confirms it on the emission.
    """
    source = """
@lib.script.indicator("T")
def main():
    data = [lib.close, lib.close]
    alias = data
    a = lib.request.security(lib.syminfo.tickerid, "D", alias[1],
                             lookahead=lib.barmerge.lookahead_on)
    lib.plot(a)
"""
    assert __test_helper_flags(source) == [False]
    log.info("the aliased list was cleared")


def __test_a_cleared_member_clears_its_whole_group__(log):
    """One child serves a merge group, so its round protocol must be uniform

    Both contexts resolve to the same feed and share a clone; the second one's
    candidate is True (``data`` IS declared a series in the scope) while its
    lowered write indexes the list the name holds there, so neither may skip a
    developing round.
    """
    source = """
@lib.script.indicator("T")
def main():
    x: Series[float] = lib.close
    data = [lib.close, lib.close]
    a = lib.request.security(lib.syminfo.tickerid, "D", x[1],
                             lookahead=lib.barmerge.lookahead_on)
    b = lib.request.security(lib.syminfo.tickerid, "D", data[1],
                             lookahead=lib.barmerge.lookahead_on)
    data: Series[float] = lib.close
    lib.plot(a + b)
"""
    flags = __test_helper_flags(source)
    assert flags == [False, False], f"the group was not cleared uniformly: {flags}"
    log.info("the group's members were cleared together")


def __test_a_shadowed_inline_series_is_cleared__(log):
    """A script's own ``def inline_series`` is an ordinary function

    The spelling means the history primitive only while the module has not bound
    it; the lowered write calls the user function, which reads no history at all.
    """
    source = """
def inline_series(value, offset):
    return value

@lib.script.indicator("T")
def main():
    a = lib.request.security(lib.syminfo.tickerid, "D",
                             inline_series(lib.close, 1),
                             lookahead=lib.barmerge.lookahead_on)
    lib.plot(a)
"""
    assert __test_helper_flags(source) == [False]
    log.info("the shadowed inline_series was cleared")


def __test_a_hoisted_inline_series_keeps_the_flag__(log):
    """The hoist's temporary is resolved back to the history call

    An ``inline_series`` standing in a conditional expression is lifted into a
    scope-local temporary, so the lowered write carries a bare name. The name is
    bound once, in front of the write, in the same statement list — the value is
    exactly the history call, and the flag survives.
    """
    source = """
@lib.script.indicator("T")
def main():
    mode = lib.input.string("SMA")
    a = (lib.request.security(lib.syminfo.tickerid, "D",
                              inline_series(lib.ta.sma(lib.close, 3), 1),
                              lookahead=lib.barmerge.lookahead_on)
         if mode == "SMA" else
         lib.request.security(lib.syminfo.tickerid, "D",
                              inline_series(lib.ta.ema(lib.close, 3), 1),
                              lookahead=lib.barmerge.lookahead_on))
    lib.plot(a)
"""
    flags = __test_helper_flags(source)
    assert flags == [True, True], f"the hoisted history reads were cleared: {flags}"
    log.info("both hoisted inline_series temporaries kept the flag")


def __test_helper_binding(source: str) -> ast.expr | None:
    """Resolve ``h`` at the ``use(...)`` call of a one-function module.

    :param source: A module whose single function holds both.
    :return: What :func:`_name_binding` resolves the name to.
    """
    module = ast.parse(source)
    scope = module.body[0]
    assert isinstance(scope, ast.FunctionDef)
    target = None
    for node in ast.walk(scope):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == 'use':
            target = node
    assert target is not None
    return _name_binding('h', scope, target)


def __test_only_a_certain_binding_is_resolved__(log):
    """Every doubtful binding of the temporary stays unresolved

    The rule answers for ONE shape: a single plain assignment in the scope's own
    statement list, in front of the read. A second binding, a parameter, a
    binding that stands in a branch, and a read behind a nested definition all
    give None, so the write is judged on the bare name and the flag is cleared.
    """
    assert __test_helper_binding("""
def f():
    h = src(1)
    use(h)
""") is not None
    assert __test_helper_binding("""
def f():
    h = src(1)
    h = src(2)
    use(h)
""") is None
    assert __test_helper_binding("""
def f(h):
    use(h)
""") is None
    assert __test_helper_binding("""
def f():
    if cond:
        h = src(1)
    use(h)
""") is None
    assert __test_helper_binding("""
def f():
    h = src(1)
    for h in items:
        pass
    use(h)
""") is None
    assert __test_helper_binding("""
def f():
    h = src(1)
    def g():
        use(h)
""") is None
    assert __test_helper_binding("""
def f():
    use(h)
    h = src(1)
""") is None
    log.info("only the single unconditional same-scope binding resolved")


def __test_the_primitive_import_is_not_a_shadow__(log):
    """The import lifter's own import of ``inline_series`` is the primitive

    The lowered module always carries ``from pynecore.core.series import
    inline_series``; counting it as a binding would call every module's
    ``inline_series`` an ordinary function and clear every flag. An import of the
    same spelling from anywhere else still shadows it.
    """
    primitive = ast.parse("from pynecore.core.series import inline_series")
    assert not _binds_name(primitive, 'inline_series')
    foreign = ast.parse("from helper import inline_series")
    assert _binds_name(foreign, 'inline_series')
    aliased = ast.parse("import pynecore.core.series as inline_series")
    assert _binds_name(aliased, 'inline_series')
    log.info("only a foreign binding shadows the primitive")
