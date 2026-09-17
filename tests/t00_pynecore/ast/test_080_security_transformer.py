"""
@pyne
"""
import ast

import pytest

from pynecore.transformers.security import SecurityTransformer
from pynecore.transformers.security_instantiation import SecurityInstantiationTransformer


def _transform(source: str) -> str:
    """Parse source, apply SecurityTransformer, return unparsed code."""
    tree = ast.parse(source)
    tree = SecurityTransformer().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _transform_tree(source: str) -> ast.Module:
    """Parse source, apply SecurityTransformer, return AST tree."""
    tree = ast.parse(source)
    tree = SecurityTransformer().visit(tree)
    ast.fix_missing_locations(tree)
    return tree


def _find_func(tree: ast.Module, name: str = 'main') -> ast.FunctionDef:
    """Find a FunctionDef by name in the module body."""
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"FunctionDef '{name}' not found")


def _find_contexts(tree: ast.Module) -> ast.Assign:
    """Find the __security_contexts__ assignment in the module body."""
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == '__security_contexts__':
                return node
    raise AssertionError("__security_contexts__ not found")


def __test_simple_security__(log):
    """Simple request.security() call"""
    source = """
def main():
    sma = lib.ta.sma(lib.close, 20)
    daily = lib.request.security(lib.syminfo.tickerid, "1D", sma)
    lib.plot(daily)
"""
    result = _transform(source)

    assert 'lib.request.security' not in result
    assert '__sec_signal__' in result
    assert '__sec_write__' in result
    assert '__sec_read__' in result
    assert '__sec_wait__' in result
    assert '__security_contexts__' in result

    tree = _transform_tree(source)
    func = _find_func(tree)

    # First statement: signal block (if __active_security__ is None)
    signal_if = func.body[0]
    assert isinstance(signal_if, ast.If)
    assert isinstance(signal_if.test, ast.Compare)
    assert isinstance(signal_if.test.ops[0], ast.Is)
    assert len(signal_if.body) == 1  # one signal call
    signal_call = signal_if.body[0].value
    assert signal_call.func.id == '__sec_signal__'

    # Statement 2: original sma assignment (unchanged)
    assert isinstance(func.body[1], ast.Assign)

    # Statement 3: write block (if __active_security__ == sec_id or sec_id in __same_context__)
    write_if = func.body[2]
    assert isinstance(write_if, ast.If)
    assert isinstance(write_if.test, ast.BoolOp)
    assert isinstance(write_if.test.op, ast.Or)
    assert isinstance(write_if.test.values[0].ops[0], ast.Eq)
    assert isinstance(write_if.test.values[1].ops[0], ast.In)
    write_call = write_if.body[0].value
    assert write_call.func.id == '__sec_write__'

    # Statement 4: daily = __sec_read__(sec_id, lib.na)
    read_assign = func.body[3]
    assert isinstance(read_assign, ast.Assign)
    assert read_assign.value.func.id == '__sec_read__'
    assert read_assign.targets[0].id == 'daily'

    # Statement 5: lib.plot(daily) (unchanged)
    assert isinstance(func.body[4], ast.Expr)

    # Statement 6: wait block
    wait_if = func.body[5]
    assert isinstance(wait_if, ast.If)
    wait_call = wait_if.body[0].value
    assert wait_call.func.id == '__sec_wait__'

    # Module-level __security_contexts__ dict
    ctx_assign = _find_contexts(tree)
    assert ctx_assign.targets[0].id == '__security_contexts__'


def __test_multiple_security_calls__(log):
    """Multiple request.security() calls in one function"""
    source = """
def main():
    sma = lib.ta.sma(lib.close, 20)
    daily_sma = lib.request.security(lib.syminfo.tickerid, "1D", sma)
    daily_high = lib.request.security(lib.syminfo.tickerid, "1D", lib.high)
    lib.plot(daily_sma + daily_high)
"""
    result = _transform(source)

    assert result.count('__sec_signal__') == 2
    assert result.count('__sec_write__') == 2
    assert result.count('__sec_read__') == 2
    assert result.count('__sec_wait__') == 2
    assert 'lib.request.security' not in result

    tree = _transform_tree(source)
    func = _find_func(tree)

    # Signal block should have 2 signals
    signal_if = func.body[0]
    assert len(signal_if.body) == 2

    # Wait block should have 2 waits
    wait_if = func.body[-1]
    assert len(wait_if.body) == 2

    # __security_contexts__ dict should have 2 entries
    ctx_assign = _find_contexts(tree)
    ctx_dict = ctx_assign.value
    assert len(ctx_dict.keys) == 2

    # IDs should be different
    id0 = ctx_dict.keys[0].value
    id1 = ctx_dict.keys[1].value
    assert id0.endswith('\xb70')
    assert id1.endswith('\xb71')


def __test_no_security_calls__(log):
    """Function without request.security() is unchanged"""
    source = """
def main():
    sma = lib.ta.sma(lib.close, 20)
    lib.plot(sma)
"""
    result = _transform(source)

    assert '__sec_signal__' not in result
    assert '__sec_write__' not in result
    assert '__sec_read__' not in result
    assert '__sec_wait__' not in result
    assert '__security_contexts__' not in result


def __test_conditional_security__(log):
    """request.security() inside if-block — write/read stay inside the conditional"""
    source = """
def main():
    sma = lib.ta.sma(lib.close, 20)
    if lib.bar_index > 100:
        daily = lib.request.security(lib.syminfo.tickerid, "1D", sma)
        lib.plot(daily)
"""
    tree = _transform_tree(source)
    func = _find_func(tree)

    # Signal block at function start (unconditional)
    signal_if = func.body[0]
    assert isinstance(signal_if, ast.If)
    assert signal_if.body[0].value.func.id == '__sec_signal__'

    # Original sma assignment
    assert isinstance(func.body[1], ast.Assign)

    # if bar_index > 100: — write and read are INSIDE
    cond_if = func.body[2]
    assert isinstance(cond_if, ast.If)

    # Inside the conditional: write block, read assignment, plot
    write_if = cond_if.body[0]
    assert isinstance(write_if, ast.If)
    assert write_if.body[0].value.func.id == '__sec_write__'

    read_assign = cond_if.body[1]
    assert isinstance(read_assign, ast.Assign)
    assert read_assign.value.func.id == '__sec_read__'

    # Wait block at function end (outside conditional)
    wait_if = func.body[-1]
    assert isinstance(wait_if, ast.If)
    assert wait_if.body[0].value.func.id == '__sec_wait__'


def __test_nested_functions__(log):
    """request.security() in nested function is handled separately"""
    source = """
def main():
    daily = lib.request.security(lib.syminfo.tickerid, "1D", lib.close)

    def helper():
        weekly = lib.request.security(lib.syminfo.tickerid, "1W", lib.close)
        return weekly

    lib.plot(daily)
"""
    tree = _transform_tree(source)

    # main function
    main_func = _find_func(tree)
    assert main_func.name == 'main'

    # main's signal block should have 1 signal (only daily, not weekly)
    signal_if = main_func.body[0]
    assert len(signal_if.body) == 1

    # Find the helper function inside main's body
    helper = None
    for stmt in main_func.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name == 'helper':
            helper = stmt
            break
    assert helper is not None

    # helper should have its own signal/wait blocks
    helper_signal = helper.body[0]
    assert isinstance(helper_signal, ast.If)
    assert helper_signal.body[0].value.func.id == '__sec_signal__'

    helper_wait = helper.body[-1]
    assert isinstance(helper_wait, ast.If)
    assert helper_wait.body[0].value.func.id == '__sec_wait__'

    # __security_contexts__ should have 2 entries total (daily + weekly)
    ctx_assign = _find_contexts(tree)
    assert len(ctx_assign.value.keys) == 2


def __test_keyword_arguments__(log):
    """request.security() with keyword arguments"""
    source = """
def main():
    val = lib.request.security(
        symbol=lib.syminfo.tickerid,
        timeframe="1D",
        expression=lib.close,
        gaps=lib.barmerge.gaps_on
    )
"""
    tree = _transform_tree(source)
    func = _find_func(tree)

    # Should still generate signal/write/read/wait
    assert isinstance(func.body[0], ast.If)  # signal
    assert func.body[0].body[0].value.func.id == '__sec_signal__'

    # Check __security_contexts__ has the gaps parameter
    ctx_assign = _find_contexts(tree)
    ctx_dict = ctx_assign.value.values[0]  # first context's dict
    ctx_keys = [k.value for k in ctx_dict.keys]
    assert 'gaps' in ctx_keys

    # gaps should be lib.barmerge.gaps_on (not default gaps_off)
    gaps_idx = ctx_keys.index('gaps')
    gaps_val = ctx_dict.values[gaps_idx]
    assert isinstance(gaps_val, ast.Attribute)
    assert gaps_val.attr == 'gaps_on'


def __test_expression_in_larger_expr__(log):
    """request.security() inside a larger expression (not direct assignment)"""
    source = """
def main():
    x = lib.request.security(lib.syminfo.tickerid, "1D", lib.close) + 1
"""
    tree = _transform_tree(source)
    func = _find_func(tree)

    # Write block before the assignment
    write_if = func.body[1]
    assert isinstance(write_if, ast.If)
    assert write_if.body[0].value.func.id == '__sec_write__'

    # Assignment: x = __sec_read__(...) + 1
    assign = func.body[2]
    assert isinstance(assign, ast.Assign)
    assert isinstance(assign.value, ast.BinOp)
    assert assign.value.left.func.id == '__sec_read__'
    assert isinstance(assign.value.right, ast.Constant)
    assert assign.value.right.value == 1


def __test_default_gaps__(log):
    """Default gaps=lib.barmerge.gaps_off when not specified"""
    source = """
def main():
    val = lib.request.security(lib.syminfo.tickerid, "1D", lib.close)
"""
    tree = _transform_tree(source)

    # Check __security_contexts__ has default gaps_off
    ctx_assign = _find_contexts(tree)
    ctx_dict = ctx_assign.value.values[0]
    ctx_keys = [k.value for k in ctx_dict.keys]
    assert 'gaps' in ctx_keys

    gaps_idx = ctx_keys.index('gaps')
    gaps_val = ctx_dict.values[gaps_idx]
    assert isinstance(gaps_val, ast.Attribute)
    assert gaps_val.attr == 'gaps_off'
    assert gaps_val.value.attr == 'barmerge'


def __test_security_contexts_structure__(log):
    """__security_contexts__ dict has correct structure"""
    source = """
def main():
    daily = lib.request.security("AAPL", "1D", lib.close)
    hourly = lib.request.security(lib.syminfo.tickerid, "60", lib.high, gaps=lib.barmerge.gaps_on)
"""
    tree = _transform_tree(source)

    ctx_assign = _find_contexts(tree)
    assert isinstance(ctx_assign, ast.Assign)
    assert ctx_assign.targets[0].id == '__security_contexts__'

    ctx_dict = ctx_assign.value
    assert len(ctx_dict.keys) == 2

    # First context: symbol="AAPL", timeframe="1D", gaps=gaps_off
    ctx0 = ctx_dict.values[0]
    ctx0_keys = [k.value for k in ctx0.keys]
    assert 'symbol' in ctx0_keys
    assert 'timeframe' in ctx0_keys
    assert 'gaps' in ctx0_keys
    sym_idx = ctx0_keys.index('symbol')
    assert isinstance(ctx0.values[sym_idx], ast.Constant)
    assert ctx0.values[sym_idx].value == 'AAPL'

    # Second context: gaps=gaps_on
    ctx1 = ctx_dict.values[1]
    ctx1_keys = [k.value for k in ctx1.keys]
    gaps_idx = ctx1_keys.index('gaps')
    assert ctx1.values[gaps_idx].attr == 'gaps_on'


def __test_ltf_simple__(log):
    """Simple request.security_lower_tf() call"""
    source = """
def main():
    intrabars = lib.request.security_lower_tf(lib.syminfo.tickerid, "1", lib.close)
    lib.plot(intrabars.size())
"""
    result = _transform(source)

    assert 'lib.request.security_lower_tf' not in result
    assert '__sec_signal__' in result
    assert '__sec_write__' in result
    assert '__sec_read__' in result
    assert '__sec_wait__' in result
    assert '__security_contexts__' in result

    tree = _transform_tree(source)
    func = _find_func(tree)

    # Read call should use [] as default (not lib.na)
    read_assign = func.body[2]
    assert isinstance(read_assign, ast.Assign)
    read_call = read_assign.value
    assert read_call.func.id == '__sec_read__'
    default_arg = read_call.args[1]
    assert isinstance(default_arg, ast.List)
    assert default_arg.elts == []


def __test_ltf_tuple_expression_unzip__(log):
    """A tuple expression makes security_lower_tf() return one array per element.

    The read is wrapped in ``__ltf_unzip__(__sec_read__(...), N)`` so the
    row-major intrabar buffer is transposed into N column arrays, and the helper
    import is added.
    """
    source = """
def main():
    a, b, c = lib.request.security_lower_tf(lib.syminfo.tickerid, "1", (lib.high, lib.low, lib.close))
    lib.plot(a.size())
"""
    result = _transform(source)

    assert 'lib.request.security_lower_tf' not in result
    assert '__ltf_unzip__' in result
    assert 'from pynecore.core.security import __ltf_unzip__' in result

    tree = _transform_tree(source)
    func = _find_func(tree)

    # Find the tuple-unpack assignment; its RHS is __ltf_unzip__(__sec_read__, 3)
    unpack = next(
        s for s in func.body
        if isinstance(s, ast.Assign) and isinstance(s.targets[0], ast.Tuple)
    )
    call = unpack.value
    assert isinstance(call, ast.Call)
    assert call.func.id == '__ltf_unzip__'
    inner = call.args[0]
    assert isinstance(inner, ast.Call) and inner.func.id == '__sec_read__'
    assert isinstance(call.args[1], ast.Constant) and call.args[1].value == 3


def __test_ltf_scalar_no_unzip__(log):
    """A scalar security_lower_tf() expression is read directly, never wrapped."""
    source = """
def main():
    intrabars = lib.request.security_lower_tf(lib.syminfo.tickerid, "1", lib.close)
    lib.plot(intrabars.size())
"""
    result = _transform(source)
    assert '__ltf_unzip__' not in result
    assert 'import __ltf_unzip__' not in result


def __test_ltf_opaque_tuple_expression_lhs_arity__(log):
    """An opaque (function-call) tuple expression gets its arity from the
    LHS unpack target.

    Pine enforces LHS-arity == RHS-arity, so ``a, b = security_lower_tf(...,
    fn())`` must wrap the read in ``__ltf_unzip__(..., 2)`` even though the
    expression itself reveals no arity.
    """
    source = """
def main():
    up, dn = lib.request.security_lower_tf(lib.syminfo.tickerid, "1", upDnVolumes())
    lib.plot(up.size())
"""
    result = _transform(source)

    assert '__ltf_unzip__' in result
    assert 'from pynecore.core.security import __ltf_unzip__' in result

    tree = _transform_tree(source)
    func = _find_func(tree)
    unpack = next(
        s for s in func.body
        if isinstance(s, ast.Assign) and isinstance(s.targets[0], ast.Tuple)
    )
    call = unpack.value
    assert isinstance(call, ast.Call)
    assert call.func.id == '__ltf_unzip__'
    assert isinstance(call.args[1], ast.Constant) and call.args[1].value == 2


def __test_sec_ids_unique_across_modules__(log):
    """The sec id embeds a hash of ``_module_file_path``, so a script and an
    imported library with their own security calls get distinct contexts.

    Without a distinct per-module hash the ids collide (same counter, same
    fallback hash) and the runner merges different contexts under one id.
    """
    source = """
def main():
    daily = lib.request.security(lib.syminfo.tickerid, "1D", lib.close)
"""

    def _ids(module_path: str | None) -> list[str]:
        tree = ast.parse(source)
        if module_path is not None:
            tree._module_file_path = module_path
        tree = SecurityTransformer().visit(tree)
        ast.fix_missing_locations(tree)
        ctx = _find_contexts(tree)
        return [k.value for k in ctx.value.keys]

    ids_a = _ids('/some/where/script.py')
    ids_b = _ids('/some/where/lib_module.py')
    ids_a2 = _ids('/some/where/script.py')

    assert ids_a != ids_b
    assert ids_a == ids_a2  # same module path -> stable ids (parent vs child process)


def __test_ltf_context_metadata__(log):
    """security_lower_tf context has is_ltf=True and no gaps key"""
    source = """
def main():
    intrabars = lib.request.security_lower_tf("AAPL", "1", lib.close)
"""
    tree = _transform_tree(source)
    ctx_assign = _find_contexts(tree)
    ctx_dict = ctx_assign.value.values[0]
    ctx_keys = [k.value for k in ctx_dict.keys]

    assert 'is_ltf' in ctx_keys
    assert 'gaps' not in ctx_keys

    ltf_idx = ctx_keys.index('is_ltf')
    assert ctx_dict.values[ltf_idx].value is True


def __test_ltf_mixed_with_htf__(log):
    """Both request.security() and request.security_lower_tf() in same function"""
    source = """
def main():
    daily = lib.request.security(lib.syminfo.tickerid, "1D", lib.close)
    intrabars = lib.request.security_lower_tf(lib.syminfo.tickerid, "1", lib.close)
    lib.plot(daily)
"""
    tree = _transform_tree(source)

    # Two contexts total
    ctx_assign = _find_contexts(tree)
    ctx_dict = ctx_assign.value
    assert len(ctx_dict.keys) == 2

    # First context (HTF): has gaps, no is_ltf
    ctx0_keys = [k.value for k in ctx_dict.values[0].keys]
    assert 'gaps' in ctx0_keys
    assert 'is_ltf' not in ctx0_keys

    # Second context (LTF): has is_ltf, no gaps
    ctx1_keys = [k.value for k in ctx_dict.values[1].keys]
    assert 'is_ltf' in ctx1_keys
    assert 'gaps' not in ctx1_keys

    func = _find_func(tree)

    # Signal block at top should have 2 signals
    signal_if = func.body[0]
    assert len(signal_if.body) == 2

    # Find both read calls — first should use lib.na, second should use []
    read_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == '__sec_read__':
                read_calls.append(node)
    assert len(read_calls) == 2

    # HTF read: default is lib.na (ast.Attribute)
    assert isinstance(read_calls[0].args[1], ast.Attribute)
    assert read_calls[0].args[1].attr == 'na'

    # LTF read: default is [] (ast.List)
    assert isinstance(read_calls[1].args[1], ast.List)


def __test_ltf_no_barmerge_import__(log):
    """security_lower_tf alone should not trigger barmerge import"""
    source = """
def main():
    intrabars = lib.request.security_lower_tf(lib.syminfo.tickerid, "1", lib.close)
"""
    result = _transform(source)

    assert 'pynecore.lib.barmerge' not in result


def __test_tuple_unpack_emits_tuple_default__(log):
    """LHS tuple-unpack must produce a tuple-of-na default in __sec_read__.

    Pine semantics: ``request.security()`` returning a tuple yields a
    tuple-of-na on every no-data path. A scalar ``lib.na`` default would
    crash the unpack with TypeError on the first / between-period bars.
    """
    source = """
def main():
    (a, b, c, d, e, f) = lib.request.security(lib.syminfo.tickerid, "1D", f_six())
    lib.plot(a)
"""
    tree = _transform_tree(source)

    read_call = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == '__sec_read__':
                read_call = node
                break
    assert read_call is not None

    default = read_call.args[1]
    assert isinstance(default, ast.Tuple)
    assert len(default.elts) == 6
    for elt in default.elts:
        assert isinstance(elt, ast.Attribute) and elt.attr == 'na'


def __test_list_target_unpack_emits_tuple_default__(log):
    """``[a, b] = security(...)`` (list-target) is also tuple-unpack."""
    source = """
def main():
    [a, b] = lib.request.security(lib.syminfo.tickerid, "1D", f_two())
    lib.plot(a)
"""
    tree = _transform_tree(source)

    read_call = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == '__sec_read__'
    )
    default = read_call.args[1]
    assert isinstance(default, ast.Tuple)
    assert len(default.elts) == 2


def __test_scalar_assign_keeps_scalar_default__(log):
    """Single-target scalar assignment keeps the scalar ``lib.na`` default."""
    source = """
def main():
    x = lib.request.security(lib.syminfo.tickerid, "1D", lib.close)
    lib.plot(x)
"""
    tree = _transform_tree(source)

    read_call = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == '__sec_read__'
    )
    default = read_call.args[1]
    assert isinstance(default, ast.Attribute) and default.attr == 'na'


def __test_star_unpack_falls_back_to_scalar__(log):
    """Star-unpack arity is unknown — must NOT bake a fixed-arity tuple."""
    source = """
def main():
    a, *rest = lib.request.security(lib.syminfo.tickerid, "1D", f_three())
    lib.plot(a)
"""
    tree = _transform_tree(source)

    read_call = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == '__sec_read__'
    )
    default = read_call.args[1]
    assert isinstance(default, ast.Attribute) and default.attr == 'na'


def __test_call_wrapped_in_ifexp_falls_back_to_scalar__(log):
    """When the call is not the direct RHS, arity is not knowable — scalar."""
    source = """
def main():
    a, b = lib.request.security(lib.syminfo.tickerid, "1D", f_two()) if cond else (0, 0)
    lib.plot(a)
"""
    tree = _transform_tree(source)

    read_call = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == '__sec_read__'
    )
    default = read_call.args[1]
    assert isinstance(default, ast.Attribute) and default.attr == 'na'


def __test_strategy_position_size_in_security_rejected__(log):
    """Direct strategy.position_size as security expression must raise SyntaxError."""
    source = """
def main():
    daily = lib.request.security(lib.syminfo.tickerid, "1D", lib.strategy.position_size)
    lib.plot(daily)
"""
    with pytest.raises(SyntaxError, match="strategy.position_size"):
        _transform(source)


def __test_strategy_state_in_binop_rejected__(log):
    """strategy.equity inside an arithmetic expression in security() is detected."""
    source = """
def main():
    daily = lib.request.security(lib.syminfo.tickerid, "1D", lib.strategy.equity + 100)
    lib.plot(daily)
"""
    with pytest.raises(SyntaxError, match="strategy.equity"):
        _transform(source)


def __test_strategy_state_in_security_lower_tf_rejected__(log):
    """request.security_lower_tf() applies the same rule."""
    source = """
def main():
    arr = lib.request.security_lower_tf(lib.syminfo.tickerid, "1", lib.strategy.netprofit)
    lib.plot(arr)
"""
    with pytest.raises(SyntaxError, match="strategy.netprofit.*security_lower_tf"):
        _transform(source)


def __test_strategy_state_via_local_alias_passes__(log):
    """Strategy state passed via a local alias compiles cleanly (not rejected at AST level).

    Transitive (local-alias) bind is intentionally NOT detected at AST level —
    runtime guard in the strategy module handles it via 0.0/0 inert defaults."""
    source = """
def main():
    ps = lib.strategy.position_size
    daily = lib.request.security(lib.syminfo.tickerid, "1D", ps)
    lib.plot(daily)
"""
    # Should compile cleanly; runtime falls back to inert default in security child.
    _transform(source)


def __test_strategy_state_in_chart_body_passes__(log):
    """strategy.* in chart-body code (outside security expr) compiles fine."""
    source = """
def main():
    if lib.strategy.position_size > 0:
        daily = lib.request.security(lib.syminfo.tickerid, "1D", lib.close)
        lib.plot(daily)
"""
    _transform(source)


def __test_all_thirteen_strategy_state_attrs_rejected__(log):
    """All 13 strategy state accessors are rejected when used directly in security()."""
    forbidden = [
        "equity", "eventrades", "grossloss", "grossprofit", "initial_capital",
        "losstrades", "max_drawdown", "max_runup", "netprofit", "openprofit",
        "position_avg_price", "position_size", "wintrades",
    ]
    for attr in forbidden:
        source = f"""
def main():
    x = lib.request.security(lib.syminfo.tickerid, "1D", lib.strategy.{attr})
"""
        with pytest.raises(SyntaxError, match=f"strategy.{attr}"):
            _transform(source)


def __test_lookahead_last_closed_kwarg_stored__(log):
    """`lookahead=barmerge.lookahead_last_closed` is parsed and stored in ctx."""
    source = """
def main():
    val = lib.request.security(
        lib.syminfo.tickerid, "1D", lib.close,
        lookahead=lib.barmerge.lookahead_last_closed,
    )
"""
    tree = _transform_tree(source)
    ctx_assign = _find_contexts(tree)
    ctx_dict = ctx_assign.value.values[0]
    ctx_keys = [k.value for k in ctx_dict.keys]
    assert 'lookahead' in ctx_keys

    la_idx = ctx_keys.index('lookahead')
    la_val = ctx_dict.values[la_idx]
    assert isinstance(la_val, ast.Attribute)
    assert la_val.attr == 'lookahead_last_closed'
    assert la_val.value.attr == 'barmerge'


def __test_lookahead_positional_at_index_4__(log):
    """Pine v6 positional order: lookahead is the 5th positional argument."""
    source = """
def main():
    val = lib.request.security(
        lib.syminfo.tickerid, "1D", lib.close,
        lib.barmerge.gaps_off,
        lib.barmerge.lookahead_last_closed,
    )
"""
    tree = _transform_tree(source)
    ctx_assign = _find_contexts(tree)
    ctx_dict = ctx_assign.value.values[0]
    ctx_keys = [k.value for k in ctx_dict.keys]
    assert 'lookahead' in ctx_keys

    la_idx = ctx_keys.index('lookahead')
    la_val = ctx_dict.values[la_idx]
    assert la_val.attr == 'lookahead_last_closed'


def __test_lookahead_omitted_no_ctx_key__(log):
    """Omitting lookahead leaves no 'lookahead' key in ctx (runtime defaults to off)."""
    source = """
def main():
    val = lib.request.security(lib.syminfo.tickerid, "1D", lib.close)
"""
    tree = _transform_tree(source)
    ctx_assign = _find_contexts(tree)
    ctx_dict = ctx_assign.value.values[0]
    ctx_keys = [k.value for k in ctx_dict.keys]
    assert 'lookahead' not in ctx_keys


def __test_lookahead_on_kwarg_stored__(log):
    """`lookahead=barmerge.lookahead_on` is parsed and stored in ctx."""
    source = """
def main():
    val = lib.request.security(
        lib.syminfo.tickerid, "1D", lib.close,
        lookahead=lib.barmerge.lookahead_on,
    )
"""
    tree = _transform_tree(source)
    ctx_assign = _find_contexts(tree)
    ctx_dict = ctx_assign.value.values[0]
    ctx_keys = [k.value for k in ctx_dict.keys]
    assert 'lookahead' in ctx_keys

    la_idx = ctx_keys.index('lookahead')
    la_val = ctx_dict.values[la_idx]
    assert isinstance(la_val, ast.Attribute)
    assert la_val.attr == 'lookahead_on'
    assert la_val.value.attr == 'barmerge'


def __test_lookahead_on_positional_stored__(log):
    """Positional `lookahead_on` at the 5th arg slot is parsed."""
    source = """
def main():
    val = lib.request.security(
        lib.syminfo.tickerid, "1D", lib.close,
        lib.barmerge.gaps_off,
        lib.barmerge.lookahead_on,
    )
"""
    tree = _transform_tree(source)
    ctx_assign = _find_contexts(tree)
    ctx_dict = ctx_assign.value.values[0]
    ctx_keys = [k.value for k in ctx_dict.keys]
    assert 'lookahead' in ctx_keys

    la_idx = ctx_keys.index('lookahead')
    la_val = ctx_dict.values[la_idx]
    assert la_val.attr == 'lookahead_on'


def __test_lookahead_runtime_expr_deferred__(log):
    """Runtime `lookahead` expressions are deferred to __sec_signal__.

    An input-derived (Pine "simple") lookahead — the standard TV non-repaint
    HTF pattern — cannot be inlined into the module-level
    ``__security_contexts__`` literal (it would NameError at import time).
    The transformer stores None there instead and passes the actual
    expression as the 4th ``__sec_signal__`` argument, emitted INLINE after
    the local it references has been assigned (not at function start).
    """
    source = """
def main():
    mode = lib.barmerge.lookahead_on if cond else lib.barmerge.lookahead_off
    val = lib.request.security(
        lib.syminfo.tickerid, "1D", lib.close,
        lookahead=mode,
    )
"""
    tree = _transform_tree(source)

    # Module-level ctx stores None as the lookahead placeholder
    ctx_assign = _find_contexts(tree)
    ctx_dict = ctx_assign.value.values[0]
    ctx_keys = [k.value for k in ctx_dict.keys]
    la_idx = ctx_keys.index('lookahead')
    la_val = ctx_dict.values[la_idx]
    assert isinstance(la_val, ast.Constant) and la_val.value is None

    # The signal is inline (after the `mode` assignment), not at function
    # start, and carries the runtime expression as its 4th argument
    func = _find_func(tree)
    assert isinstance(func.body[0], ast.Assign)  # mode = ...
    signal_if = func.body[1]
    assert isinstance(signal_if, ast.If)
    signal_call = signal_if.body[0].value
    assert signal_call.func.id == '__sec_signal__'
    assert len(signal_call.args) == 4
    assert isinstance(signal_call.args[3], ast.Name)
    assert signal_call.args[3].id == 'mode'


def __test_lookahead_constant_signal_has_no_lookahead_arg__(log):
    """A constant lookahead stays in ctx; __sec_signal__ keeps 3 args."""
    source = """
def main():
    val = lib.request.security(
        lib.syminfo.tickerid, "1D", lib.close,
        lookahead=lib.barmerge.lookahead_on,
    )
"""
    tree = _transform_tree(source)
    func = _find_func(tree)
    signal_if = func.body[0]
    assert isinstance(signal_if, ast.If)
    signal_call = signal_if.body[0].value
    assert signal_call.func.id == '__sec_signal__'
    assert len(signal_call.args) == 3


def __test_symbol_call_with_local_arg_deferred__(log):
    """``ticker.heikinashi(<local>)`` must not reach the module-level ctx.

    The callee is a safe ``lib.*`` chain, but the argument is a function
    parameter (an ``input.symbol()`` in Pine), so the whole call has to be
    deferred to ``__sec_signal__`` — inlining it would NameError at import.
    """
    source = """
def main(sym):
    val = lib.request.security(lib.ticker.heikinashi(sym), "1D", lib.close)
"""
    tree = _transform_tree(source)

    ctx_assign = _find_contexts(tree)
    ctx_dict = ctx_assign.value.values[0]
    ctx_keys = [k.value for k in ctx_dict.keys]
    sym_val = ctx_dict.values[ctx_keys.index('symbol')]
    assert isinstance(sym_val, ast.Constant) and sym_val.value is None

    # Signal is emitted inline and carries the actual call expression
    func = _find_func(tree)
    signal_if = func.body[0]
    assert isinstance(signal_if, ast.If)
    signal_call = signal_if.body[0].value
    assert signal_call.func.id == '__sec_signal__'
    assert isinstance(signal_call.args[1], ast.Call)


def __test_symbol_call_with_lib_arg_stays_module_level__(log):
    """``ticker.heikinashi(lib.syminfo.tickerid)`` stays in the module ctx."""
    source = """
def main():
    val = lib.request.security(
        lib.ticker.heikinashi(lib.syminfo.tickerid), "1D", lib.close
    )
"""
    tree = _transform_tree(source)

    ctx_assign = _find_contexts(tree)
    ctx_dict = ctx_assign.value.values[0]
    ctx_keys = [k.value for k in ctx_dict.keys]
    sym_val = ctx_dict.values[ctx_keys.index('symbol')]
    assert isinstance(sym_val, ast.Call)


def __test_tuple_expression_pins_arity_without_unpack_target__(log):
    """A tuple EXPRESSION sets the read default even with no unpack target.

    ``return lib.request.security(..., (a, b, c), ...)`` is not the RHS of an
    assignment, so the unpack-target rule sees nothing — but the expression
    itself is authoritative. Without it the inactive-context read returns a
    scalar ``na`` and the caller's unpack raises ``na is not iterable``.
    """
    source = """
def main():
    def f():
        return lib.request.security(lib.syminfo.tickerid, "D",
                                    (lib.volume, lib.high, lib.low))
    a, b, c = f()
"""
    result = _transform(source)
    assert '__sec_read__' in result
    read_line = next(ln for ln in result.splitlines() if '__sec_read__' in ln)
    assert read_line.count('lib.na') == 3, read_line


# --- dependency analysis, in_loop marking and signal hoisting ---


def _sec_meta(source: str, instantiate: bool = False) -> dict[str, dict]:
    """Transform ``source`` and return per-context metadata.

    Contexts are keyed by their timeframe literal (every fixture below uses a
    distinct one), and ``depends`` entries are translated to the same keys, so
    the assertions stay readable instead of carrying opaque sec ids.
    """
    tree = ast.parse(source)
    if instantiate:
        tree = SecurityInstantiationTransformer().visit(tree)
    tree = SecurityTransformer().visit(tree)
    ast.fix_missing_locations(tree)

    ctx_assign = _find_contexts(tree)
    tf_of: dict[str, str] = {}
    raw: dict[str, dict] = {}
    for key, val in zip(ctx_assign.value.keys, ctx_assign.value.values):
        entry: dict = {}
        for k, v in zip(val.keys, val.values):
            if k.value == 'depends':
                entry['depends'] = [e.value for e in v.elts]
            elif k.value == 'late_reads':
                entry['late_reads'] = [e.value for e in v.elts]
            elif k.value == 'in_loop':
                entry['in_loop'] = v.value
            elif k.value == 'timeframe' and isinstance(v, ast.Constant):
                tf_of[key.value] = v.value
        raw[key.value] = entry

    return {
        tf_of.get(sid, sid): {
            'depends': sorted(tf_of.get(d, d) for d in entry.get('depends', [])),
            'late_reads': sorted(tf_of.get(d, d) for d in entry.get('late_reads', [])),
            'in_loop': entry.get('in_loop', False),
        }
        for sid, entry in raw.items()
    }


def __test_depends_monthly_rsi_chain__(log):
    """The dependent monthly-RSI form: the SMA context consumes the RSI one."""
    source = """
def main():
    rsiMonthly = lib.request.security(lib.syminfo.tickerid, "M", lib.ta.rsi(lib.close, 14))
    rsiSma = lib.request.security(lib.syminfo.tickerid, "W", lib.ta.sma(rsiMonthly, 14))
"""
    meta = _sec_meta(source)
    assert meta['M']['depends'] == []
    assert meta['W']['depends'] == ['M']
    log.info("monthly RSI dependency OK")


def __test_depends_longer_chain_is_direct_only__(log):
    """A three-link chain records DIRECT producers only — the transitive
    closure is the runtime's business."""
    source = """
def main():
    a = lib.request.security(lib.syminfo.tickerid, "D", lib.close)
    b = lib.request.security(lib.syminfo.tickerid, "W", lib.ta.sma(a, 3))
    c = lib.request.security(lib.syminfo.tickerid, "M", lib.ta.sma(b, 3))
"""
    meta = _sec_meta(source)
    assert meta['D']['depends'] == []
    assert meta['W']['depends'] == ['D']
    assert meta['M']['depends'] == ['W']
    log.info("chain dependency OK")


def __test_back_edge_onto_later_site_is_dropped__(log):
    """A value carried back from a LATER site is not a dependency.

    The consumer's write would block on a producer whose site the chart has not
    reached yet, and in the historical warmup batch that stops the very write
    that releases the round. Such a value can only be a previous-bar carry, so
    the edge is dropped and the child reads the default.
    """
    source = """
def main():
    acc = 0.0
    b = lib.request.security(lib.syminfo.tickerid, "W", acc)
    a = lib.request.security(lib.syminfo.tickerid, "D", lib.close)
    acc = acc + a
"""
    meta = _sec_meta(source)
    assert meta['W']['depends'] == []
    assert meta['D']['depends'] == []
    log.info("back edge dropped OK")


def __test_depends_implicit_flow_via_if_condition__(log):
    """A write block standing under a tainted condition depends on it even
    though no value flows into the expression."""
    source = """
def main():
    a = lib.request.security(lib.syminfo.tickerid, "D", lib.close)
    if a > 0:
        b = lib.request.security(lib.syminfo.tickerid, "W", lib.close)
"""
    meta = _sec_meta(source)
    assert meta['W']['depends'] == ['D']
    log.info("implicit flow OK")


def __test_depends_alias_through_container__(log):
    """Mutating a container through an alias taints the container itself."""
    source = """
def main():
    box = lib.array.new_float()
    a = lib.request.security(lib.syminfo.tickerid, "D", lib.close)
    alias = box
    lib.array.push(alias, a)
    b = lib.request.security(lib.syminfo.tickerid, "W", lib.array.get(box, 0))
"""
    meta = _sec_meta(source)
    assert meta['W']['depends'] == ['D']
    log.info("alias/mutation flow OK")


def __test_depends_through_user_function__(log):
    """Both directions of a user call carry taint: the callee's return value
    and the arguments the caller passes in."""
    source = """
def producer():
    return lib.request.security(lib.syminfo.tickerid, "D", lib.close)
def scale(x):
    return x * 2
def main():
    p = producer()
    b = lib.request.security(lib.syminfo.tickerid, "W", lib.ta.sma(p, 3))
    c = lib.request.security(lib.syminfo.tickerid, "M", scale(b))
"""
    meta = _sec_meta(source)
    assert meta['W']['depends'] == ['D']
    assert meta['M']['depends'] == ['W']
    log.info("user function flow OK")


def __test_depends_tainted_return_taints_what_follows__(log):
    """A ``return`` taken under a tainted condition makes everything after it
    conditional on that taint."""
    source = """
def main():
    a = lib.request.security(lib.syminfo.tickerid, "D", lib.close)
    if a > 0:
        return
    b = lib.request.security(lib.syminfo.tickerid, "W", lib.close)
"""
    meta = _sec_meta(source)
    assert meta['W']['depends'] == ['D']
    log.info("tainted return OK")


def __test_depends_unhandled_construct_falls_back_to_all__(log):
    """An unmodelled construct in a scope makes every write in it depend on
    every EARLIER-sited context (safe over-approximation)."""
    source = """
def main():
    a = lib.request.security(lib.syminfo.tickerid, "D", lib.close)
    b = lib.request.security(lib.syminfo.tickerid, "W", lib.close)
    c = lib.request.security(lib.syminfo.tickerid, "M", lib.close)
    try:
        lib.plot(a)
    except Exception:
        lib.plot(b)
"""
    meta = _sec_meta(source)
    assert meta['D']['depends'] == []
    assert meta['W']['depends'] == ['D']
    assert meta['M']['depends'] == ['D', 'W']
    log.info("fallback to all earlier sids OK")


def __test_depends_field_only_class_is_modelled__(log):
    """A Pine ``type`` — a class of field declarations — keeps the analysis
    precise: its body runs at import, before any security value exists."""
    source = """
from pynecore import lib

@udt
class Level:
    top: float = lib.na(float)
    bot: float = lib.na(float)

LENGTH = 9

def main():
    a = lib.request.security(lib.syminfo.tickerid, "D", lib.ta.ema(lib.close, LENGTH))
    b = lib.request.security(lib.syminfo.tickerid, "W", lib.ta.ema(lib.high, LENGTH))
    lvl = Level(top=a, bot=b)
    lib.plot(lvl.top)
"""
    meta = _sec_meta(source)
    assert meta['D']['depends'] == []
    assert meta['W']['depends'] == []
    log.info("field-only class keeps the dependencies precise")


def __test_depends_compiler_field_factory_is_modelled__(log):
    """A bool field default lowered to ``field(default_factory=lambda: ...)``
    reading only the lowering's reserved helper keeps the class a record."""
    source = """
from pynecore import lib
from dataclasses import field as __pyne_field·__
from pynecore.types.na import new_bool_na as __pyne_bool_na·__

@udt
class Flag:
    on: bool = __pyne_field·__(default_factory=lambda: __pyne_bool_na·__())

LENGTH = 9

def main():
    a = lib.request.security(lib.syminfo.tickerid, "D", lib.ta.ema(lib.close, LENGTH))
    b = lib.request.security(lib.syminfo.tickerid, "W", lib.ta.ema(lib.high, LENGTH))
    lib.plot(a + b)
"""
    meta = _sec_meta(source)
    assert meta['W']['depends'] == []
    log.info("compiler field factory keeps the dependencies precise")


def __test_depends_factory_reading_a_script_name_falls_back__(log):
    """A factory lambda reading a name the script can bind stays unmodelled."""
    source = """
from pynecore import lib
from dataclasses import field as __pyne_field·__

@udt
class Flag:
    on: float = __pyne_field·__(default_factory=lambda: LENGTH)

LENGTH = 9

def main():
    a = lib.request.security(lib.syminfo.tickerid, "D", lib.ta.ema(lib.close, LENGTH))
    b = lib.request.security(lib.syminfo.tickerid, "W", lib.ta.ema(lib.high, LENGTH))
    lib.plot(a + b)
"""
    meta = _sec_meta(source)
    assert meta['W']['depends'] == ['D']
    log.info("factory reading a script name falls back")


def __test_depends_break_taints_the_whole_loop_body__(log):
    """A tainted ``break`` decides whether the body runs again, so everything
    in the body is conditional on it — the statements before it included."""
    source = """
def main():
    a = lib.request.security(lib.syminfo.tickerid, "D", lib.close)
    total = 0.0
    for i in range(10):
        b = lib.request.security(lib.syminfo.tickerid, "W", lib.close)
        total += b
        if a > 0:
            break
    lib.plot(total)
"""
    meta = _sec_meta(source)
    assert meta['W']['depends'] == ['D']
    log.info("break taint covers the loop body it can cut short")


def __test_depends_break_taints_a_later_write__(log):
    """A write placed after the loop does run conditionally on the ``break``."""
    source = """
def main():
    a = lib.request.security(lib.syminfo.tickerid, "D", lib.close)
    for i in range(3):
        if a > 0:
            break
    b = lib.request.security(lib.syminfo.tickerid, "W", lib.close)
    lib.plot(b)
"""
    meta = _sec_meta(source)
    assert meta['W']['depends'] == ['D']
    log.info("break taint reaches the following write")


def __test_depends_collection_reader_leaves_the_container_alone__(log):
    """``array.get`` reads: a tainted index reaches the VALUE it returns, never
    the array it was read from."""
    source = """
def main():
    a = lib.request.security(lib.syminfo.tickerid, "D", lib.close)
    store = lib.array.new_float(4, 0.0)
    value = lib.array.get(store, a)
    b = lib.request.security(lib.syminfo.tickerid, "W", lib.array.size(store))
    c = lib.request.security(lib.syminfo.tickerid, "M", value)
    lib.plot(b + c)
"""
    meta = _sec_meta(source)
    assert meta['W']['depends'] == []
    assert meta['M']['depends'] == ['D']
    log.info("reader keeps the index out of the container")


def __test_depends_mutator_writes_back_index_and_value__(log):
    """``array.set`` stores its value, and the position it writes at decides
    which slot a later read observes — both are part of what the array holds."""
    source = """
def main():
    a = lib.request.security(lib.syminfo.tickerid, "D", lib.close)
    at_index = lib.array.new_float(4, 0.0)
    lib.array.set(at_index, a, 1.0)
    of_value = lib.array.new_float(4, 0.0)
    lib.array.set(of_value, 0, a)
    b = lib.request.security(lib.syminfo.tickerid, "W", lib.array.size(at_index))
    c = lib.request.security(lib.syminfo.tickerid, "M", lib.array.size(of_value))
    lib.plot(b + c)
"""
    meta = _sec_meta(source)
    assert meta['W']['depends'] == ['D']
    assert meta['M']['depends'] == ['D']
    log.info("mutator writes back its index as well as its value")


def __test_depends_map_put_stores_key_and_value__(log):
    """``map.put`` stores both arguments: a map hands its keys back out, so a
    tainted key is part of what the map holds."""
    source = """
def main():
    a = lib.request.security(lib.syminfo.tickerid, "D", lib.close)
    of_key = lib.map.new()
    lib.map.put(of_key, a, 1.0)
    of_value = lib.map.new()
    lib.map.put(of_value, 1.0, a)
    b = lib.request.security(lib.syminfo.tickerid, "W", lib.array.size(lib.map.keys(of_key)))
    c = lib.request.security(lib.syminfo.tickerid, "M", lib.map.size(of_value))
    lib.plot(b + c)
"""
    meta = _sec_meta(source)
    assert meta['W']['depends'] == ['D']
    assert meta['M']['depends'] == ['D']
    log.info("map.put stores its key as well as its value")


def __test_depends_unknown_collection_function_stays_mutating__(log):
    """A collection function the analysis does not know keeps the conservative
    rule: everything the call touches flows into the collection."""
    source = """
def main():
    a = lib.request.security(lib.syminfo.tickerid, "D", lib.close)
    store = lib.array.new_float(4, 0.0)
    lib.array.no_such_function(store, a)
    b = lib.request.security(lib.syminfo.tickerid, "W", lib.array.size(store))
    lib.plot(b)
"""
    meta = _sec_meta(source)
    assert meta['W']['depends'] == ['D']
    log.info("unknown collection function stays mutating")


def __test_depends_loop_variable_is_private_to_its_loop__(log):
    """A counter used by nothing but its own loops gets a taint cell per loop,
    so a tainted loop does not taint the next one's counter."""
    source = """
def main():
    a = lib.request.security(lib.syminfo.tickerid, "D", lib.close)
    tainted = lib.array.new_float(4, 0.0)
    lib.array.push(tainted, a)
    for i in range(lib.array.size(tainted)):
        lib.plot(i)
    other = lib.array.new_float()
    for i in range(3):
        lib.array.push(other, i)
    b = lib.request.security(lib.syminfo.tickerid, "W", lib.array.size(other))
    lib.plot(b)
"""
    meta = _sec_meta(source)
    assert meta['W']['depends'] == []
    log.info("loop variable stays private to its loop")


def __test_depends_loop_variable_read_after_the_loop_stays_shared__(log):
    """A counter read outside its loops keeps the single scope-wide cell."""
    source = """
def main():
    a = lib.request.security(lib.syminfo.tickerid, "D", lib.close)
    tainted = lib.array.new_float(4, 0.0)
    lib.array.push(tainted, a)
    for i in range(lib.array.size(tainted)):
        lib.plot(i)
    lib.plot(i)
    other = lib.array.new_float()
    for i in range(3):
        lib.array.push(other, i)
    b = lib.request.security(lib.syminfo.tickerid, "W", lib.array.size(other))
    lib.plot(b)
"""
    meta = _sec_meta(source)
    assert meta['W']['depends'] == ['D']
    log.info("loop variable read outside its loop stays shared")


def __test_depends_conditional_loop_store_stays_shared__(log):
    """A store the loop body may skip does not establish the name: the value a
    skipped pass reads is still the one an earlier loop left behind."""
    source = """
def main():
    a = lib.request.security(lib.syminfo.tickerid, "D", lib.close)
    acc = 0.0
    for i in range(3):
        acc = lib.ta.sma(a, 3)
    for i in range(3):
        if lib.close < 0:
            acc = 0.0
        b = lib.request.security(lib.syminfo.tickerid, "W", acc)
        lib.plot(b)
"""
    meta = _sec_meta(source)
    assert meta['W']['depends'] == ['D']
    log.info("conditional loop store keeps the shared cell")


def __test_nested_call_writes_inner_context_first__(log):
    """A call nested in another call's expression is written first, so the
    outer context depends on it instead of reading it as a late producer."""
    source = """
def main():
    v = lib.request.security("EXCH:OUT", "60", lib.ta.sma(lib.request.security("EXCH:INN", "D", lib.close), 3))
    lib.plot(v)
"""
    meta = _sec_meta(source)
    assert meta['60']['depends'] == ['D']
    assert meta['60']['late_reads'] == []

    func = _find_func(_transform_tree(source))
    writes = [
        stmt.body[0].value.args[0].value
        for stmt in func.body
        if isinstance(stmt, ast.If) and isinstance(stmt.body[0], ast.Expr)
        and getattr(stmt.body[0].value.func, 'id', None) == '__sec_write__'
    ]
    assert [w.rsplit('·', 1)[1] for w in writes] == ['1', '0']
    log.info("nested call writes the inner context first")


def __test_late_producer_read_is_kept_in_late_reads__(log):
    """A loop-carried read of a producer sited later is no dependency (it
    cannot be waited for) but stays a read of the expression in ``late_reads``."""
    source = """
def main():
    y = 0.0
    for i in range(2):
        x = lib.request.security("EXCH:A", "D", y + 1.0)
        y = lib.request.security("EXCH:B", "W", lib.close * 2.0)
    lib.plot(x)
"""
    meta = _sec_meta(source)
    assert meta['D']['depends'] == []
    assert meta['D']['late_reads'] == ['W']
    assert meta['W']['late_reads'] == []
    log.info("late producer read kept apart from depends")


def __test_depends_annotated_loop_accumulator_carries_between_loops__(log):
    """``acc: float = acc + 1.0`` evaluates its value before binding the
    name, so the loop reads the value an earlier loop left behind."""
    source = """
def main():
    a = lib.request.security(lib.syminfo.tickerid, "D", lib.close)
    for i in range(2):
        acc = lib.ta.sma(a, 3)
    for j in range(2):
        acc: float = acc + 1.0
        b = lib.request.security(lib.syminfo.tickerid, "W", acc)
    lib.plot(b)
"""
    meta = _sec_meta(source)
    assert meta['W']['depends'] == ['D']
    log.info("annotated read-modify-write keeps the shared cell")


def __test_depends_loop_else_read_stays_shared__(log):
    """A loop's ``else`` also runs after zero iterations, so a read there can
    see the value from before the loop even when the body stores first."""
    source = """
def main():
    a = lib.request.security(lib.syminfo.tickerid, "D", lib.close)
    for i in range(2):
        acc = lib.ta.sma(a, 3)
    for j in range(0):
        acc = 0.0
    else:
        b = lib.request.security(lib.syminfo.tickerid, "W", acc)
        lib.plot(b)
"""
    meta = _sec_meta(source)
    assert meta['W']['depends'] == ['D']
    log.info("loop-else read keeps the shared cell")


def __test_depends_loop_accumulator_carries_between_loops__(log):
    """``acc = acc + x`` READS the name before it stores into it, so the loop
    does not own it: the value an earlier loop left behind is carried over."""
    source = """
def main():
    a = lib.request.security(lib.syminfo.tickerid, "D", lib.close)
    for i in range(2):
        acc = lib.ta.sma(a, 3)
    for j in range(2):
        acc = acc + 1
        b = lib.request.security(lib.syminfo.tickerid, "W", acc)
    lib.plot(b)
"""
    meta = _sec_meta(source)
    assert meta['W']['depends'] == ['D']
    log.info("read-modify-write keeps the shared cell")


def __test_depends_named_expr_accumulator_carries_between_loops__(log):
    """``(acc := acc + x)`` evaluates its value before binding, and an
    assignment expression never establishes a loop-private name."""
    source = """
def main():
    a = lib.request.security(lib.syminfo.tickerid, "D", lib.close)
    for i in range(2):
        acc = lib.ta.sma(a, 3)
    for j in range(2):
        (acc := acc + 1.0)
        b = lib.request.security(lib.syminfo.tickerid, "W", acc)
    lib.plot(b)
"""
    meta = _sec_meta(source)
    assert meta['W']['depends'] == ['D']
    log.info("assignment-expression accumulator keeps the shared cell")


def __test_depends_loop_augmented_accumulator_carries_between_loops__(log):
    """``acc += x`` is a read-modify-write too."""
    source = """
def main():
    a = lib.request.security(lib.syminfo.tickerid, "D", lib.close)
    for i in range(2):
        acc = lib.ta.sma(a, 3)
    for j in range(2):
        acc += 1
        b = lib.request.security(lib.syminfo.tickerid, "W", acc)
    lib.plot(b)
"""
    meta = _sec_meta(source)
    assert meta['W']['depends'] == ['D']
    log.info("augmented accumulator keeps the shared cell")


def __test_depends_loop_variable_read_by_a_closure_stays_shared__(log):
    """A nested function is visited as its own scope and cannot see a loop's
    private cell, so a name it closes over keeps the scope-wide one."""
    source = """
def main():
    a = lib.request.security(lib.syminfo.tickerid, "D", lib.close)
    tainted = lib.array.new_float(4, 0.0)
    lib.array.push(tainted, a)
    for j in range(2):
        i = lib.array.size(tainted) + j

        def inner():
            return i

        lib.plot(inner())
    for i in range(3):
        b = lib.request.security(lib.syminfo.tickerid, "W", i)
    lib.plot(b)
"""
    meta = _sec_meta(source)
    assert meta['W']['depends'] == ['D']
    log.info("closure read keeps the loop variable shared")


def __test_depends_factory_naming_another_record_is_modelled__(log):
    """A field defaulting to ``na(<another UDT>)`` keeps the class a record:
    a class name is an import-time constant binding, taint-free."""
    source = """
from pynecore import lib
from dataclasses import field as __pyne_field·__

@udt
class Inner:
    top: float = lib.na(float)

@udt
class Outer:
    info: Inner = __pyne_field·__(default_factory=lambda: lib.na(Inner))

LENGTH = 9

def main():
    a = lib.request.security(lib.syminfo.tickerid, "D", lib.ta.ema(lib.close, LENGTH))
    b = lib.request.security(lib.syminfo.tickerid, "W", lib.ta.ema(lib.high, LENGTH))
    lib.plot(a + b)
"""
    meta = _sec_meta(source)
    assert meta['W']['depends'] == []
    log.info("factory naming another record keeps the analysis precise")


def __test_depends_field_only_class_still_carries_instance_taint__(log):
    """A value stored into an instance field flows on through that instance."""
    source = """
from pynecore import lib

@udt
class Level:
    top: float = lib.na(float)

def main():
    a = lib.request.security(lib.syminfo.tickerid, "D", lib.close)
    lvl = Level(top=a)
    b = lib.request.security(lib.syminfo.tickerid, "W", lvl.top)
    lib.plot(b)
"""
    meta = _sec_meta(source)
    assert meta['W']['depends'] == ['D']
    log.info("instance field taint flows OK")


def __test_depends_class_with_a_method_falls_back__(log):
    """A class body with anything but field declarations stays unmodelled."""
    source = """
from pynecore import lib

class Box:
    top: float = 0.0

    def grow(self, v):
        self.top = v

LENGTH = 9

def main():
    a = lib.request.security(lib.syminfo.tickerid, "D", lib.ta.ema(lib.close, LENGTH))
    b = lib.request.security(lib.syminfo.tickerid, "W", lib.ta.ema(lib.high, LENGTH))
    lib.plot(a + b)
"""
    meta = _sec_meta(source)
    assert meta['W']['depends'] == ['D']
    log.info("class with a method falls back OK")


def __test_depends_local_class_falls_back__(log):
    """A class defined inside a function runs with it, so its field defaults can
    read security values: it stays unmodelled."""
    source = """
from pynecore import lib

def main():
    a = lib.request.security(lib.syminfo.tickerid, "D", lib.close)

    class Box:
        value = a

    b = lib.request.security(lib.syminfo.tickerid, "W", Box.value)
    lib.plot(b)
"""
    meta = _sec_meta(source)
    assert meta['W']['depends'] == ['D']
    log.info("local class falls back OK")


def __test_mutual_cycle_keeps_only_the_forward_edge__(log):
    """A two-way flow collapses to the forward (earlier-sited) edge only."""
    source = """
def main():
    prevB = 0.0
    a = lib.request.security(lib.syminfo.tickerid, "D", prevB)
    b = lib.request.security(lib.syminfo.tickerid, "W", a)
    prevB = b
"""
    meta = _sec_meta(source)
    assert meta['D']['depends'] == []
    assert meta['W']['depends'] == ['D']
    log.info("forward edge only OK")


def __test_depends_independent_calls_have_none__(log):
    """Unrelated contexts keep an empty depends list."""
    source = """
def main():
    a = lib.request.security(lib.syminfo.tickerid, "D", lib.close)
    b = lib.request.security(lib.syminfo.tickerid, "W", lib.high)
    lib.plot(a + b)
"""
    meta = _sec_meta(source)
    assert meta['D']['depends'] == []
    assert meta['W']['depends'] == []
    log.info("independent contexts OK")


def __test_in_loop_direct_and_nested__(log):
    """A write block inside a ``for`` / ``while`` body is marked in_loop, at
    any nesting depth; a call outside every loop is not."""
    source = """
def main():
    plain = lib.request.security(lib.syminfo.tickerid, "D", lib.close)
    for i in range(3):
        looped = lib.request.security(lib.syminfo.tickerid, "W", lib.close)
    for i in range(3):
        while i > 0:
            nested = lib.request.security(lib.syminfo.tickerid, "M", lib.close)
"""
    meta = _sec_meta(source)
    assert meta['D']['in_loop'] is False
    assert meta['W']['in_loop'] is True
    assert meta['M']['in_loop'] is True
    log.info("direct in_loop marking OK")


def __test_in_loop_propagates_through_call_chain__(log):
    """The marking follows the call graph: a function reached from a loop
    body writes its sids several times per bar, however deep the chain."""
    source = """
def deep():
    return lib.request.security(lib.syminfo.tickerid, "M", lib.close)
def mid():
    return deep()
def outside():
    return lib.request.security(lib.syminfo.tickerid, "W", lib.close)
def main():
    a = outside()
    for i in range(3):
        b = mid()
    c = lib.request.security(lib.syminfo.tickerid, "D", lib.close)
"""
    meta = _sec_meta(source)
    assert meta['M']['in_loop'] is True
    assert meta['W']['in_loop'] is False
    assert meta['D']['in_loop'] is False
    log.info("call-chain in_loop propagation OK")


def __test_in_loop_call_after_loop_not_marked__(log):
    """A call site standing AFTER the loop is not in the loop body."""
    source = """
def helper():
    return lib.request.security(lib.syminfo.tickerid, "W", lib.close)
def main():
    for i in range(3):
        lib.plot(lib.close)
    a = helper()
"""
    meta = _sec_meta(source)
    assert meta['W']['in_loop'] is False
    log.info("post-loop call not marked OK")


def __test_in_loop_marks_per_clone_call_site__(log):
    """Instantiation clones per call site, so only the clone whose site is in
    a loop body gets marked."""
    source = """
def f(tf):
    return lib.request.security(lib.syminfo.tickerid, tf, lib.close)
def main():
    a = f("60")
    for i in range(3):
        b = f("240")
"""
    tree = ast.parse(source)
    tree = SecurityInstantiationTransformer().visit(tree)
    tree = SecurityTransformer().visit(tree)
    ast.fix_missing_locations(tree)

    # The clone is inserted right after the original, so context order is
    # (original = "60" site, clone = "240" site inside the loop).
    ctxs = _find_contexts(tree).value.values
    flags = [
        dict(zip([k.value for k in c.keys], c.values)).get('in_loop')
        for c in ctxs
    ]
    assert len(flags) == 2
    assert flags[0] is None
    assert flags[1] is not None and flags[1].value is True
    log.info("per-clone in_loop OK")


def __test_hoist_name_bound_timeframe__(log):
    """A timeframe bound once from a literal is hoisted above the signal
    block, so the signal can start at function entry."""
    source = """
def main():
    lib.plot(lib.close)
    tf = "240"
    v = lib.request.security(lib.syminfo.tickerid, tf, lib.close)
"""
    tree = _transform_tree(source)
    func = _find_func(tree)

    hoisted = func.body[0]
    assert isinstance(hoisted, ast.Assign)
    assert hoisted.targets[0].id == 'tf'

    signal_if = func.body[1]
    assert isinstance(signal_if, ast.If)
    signal_call = signal_if.body[0].value
    assert signal_call.func.id == '__sec_signal__'
    assert isinstance(signal_call.args[2], ast.Name)
    assert signal_call.args[2].id == 'tf'

    # The original statement order is otherwise kept and the binding is not
    # duplicated.
    assert _transform(source).count("tf = '240'") == 1
    log.info("literal binding hoisting OK")


def __test_hoist_input_derived_timeframe__(log):
    """``input.timeframe()`` results are hoistable too — the module-level ctx
    keeps None and the top-block signal carries the runtime value."""
    source = """
def main():
    lib.plot(lib.close)
    tf = lib.input.timeframe("60", "HTF")
    v = lib.request.security(lib.syminfo.tickerid, tf, lib.close)
"""
    tree = _transform_tree(source)
    func = _find_func(tree)
    assert isinstance(func.body[0], ast.Assign)
    assert func.body[0].targets[0].id == 'tf'
    signal_if = func.body[1]
    assert isinstance(signal_if, ast.If)
    assert signal_if.body[0].value.func.id == '__sec_signal__'

    ctx_dict = _find_contexts(tree).value.values[0]
    ctx_keys = [k.value for k in ctx_dict.keys]
    tf_val = ctx_dict.values[ctx_keys.index('timeframe')]
    assert isinstance(tf_val, ast.Constant) and tf_val.value is None

    # The input is evaluated exactly once — the binding moves, it is not copied
    assert _transform(source).count('lib.input.timeframe') == 1
    log.info("input-derived hoisting OK")


def __test_hoist_only_what_the_signal_needs__(log):
    """Bindings no signal reads stay where the author put them."""
    source = """
def main():
    lib.plot(lib.close)
    unrelated = "240"
    v = lib.request.security(lib.syminfo.tickerid, "D", lib.close)
"""
    tree = _transform_tree(source)
    func = _find_func(tree)
    # Signal block first, plot second, unrelated binding still third
    assert isinstance(func.body[0], ast.If)
    assert isinstance(func.body[1], ast.Expr)
    assert isinstance(func.body[2], ast.Assign)
    assert func.body[2].targets[0].id == 'unrelated'
    log.info("selective hoisting OK")


def __test_stable_parameters_are_hoisted__(log):
    """Parameters the body never rebinds hold the caller's value for the whole
    call, so their signals start in the top block."""
    source = """
def main(sym1, tf1, sym2, tf2):
    lib.plot(lib.close)
    a = lib.request.security(sym1, tf1, lib.close)
    b = lib.request.security(sym2, tf2, lib.close)
"""
    tree = _transform_tree(source)
    func = _find_func(tree)
    signal_if = func.body[0]
    assert isinstance(signal_if, ast.If)
    assert len(signal_if.body) == 2
    assert all(stmt.value.func.id == '__sec_signal__' for stmt in signal_if.body)
    # The plot call follows the top block, untouched
    assert isinstance(func.body[1], ast.Expr)
    log.info("stable parameter signals hoisted OK")


def __test_rebound_parameter_is_not_hoisted__(log):
    """A parameter the body reassigns is not known at function entry, so its
    signal stays inline."""
    source = """
def main(tf):
    lib.plot(lib.close)
    tf = "60"
    v = lib.request.security(lib.syminfo.tickerid, tf, lib.close)
"""
    tree = _transform_tree(source)
    func = _find_func(tree)
    # No top block: the first statement is the untouched plot call
    assert isinstance(func.body[0], ast.Expr)
    assert isinstance(func.body[1], ast.Assign)
    signal_if = func.body[2]
    assert isinstance(signal_if, ast.If)
    assert signal_if.body[0].value.func.id == '__sec_signal__'
    log.info("rebound parameter stays inline OK")


def __test_runtime_resolved_signal_in_branch_stays_inline__(log):
    """A context resolved at runtime keeps its signal inside the branch.

    Its first signal locates and loads its data, so a top-block signal would
    demand the feed on every run, even one that never takes the branch.
    """
    source = """
def main(use_htf, htf_tf):
    lib.plot(lib.close)
    if use_htf:
        v = lib.request.security(lib.syminfo.tickerid, htf_tf, lib.close)
"""
    tree = _transform_tree(source)
    func = _find_func(tree)
    # No top block: the first statement is the untouched plot call
    assert isinstance(func.body[0], ast.Expr)
    branch = func.body[1]
    assert isinstance(branch, ast.If)
    signal_if = branch.body[0]
    assert isinstance(signal_if, ast.If)
    assert signal_if.body[0].value.func.id == '__sec_signal__'
    log.info("runtime-resolved signal in a branch stays inline OK")


def __test_runtime_resolved_producer_in_branch_is_hoisted__(log):
    """A runtime-resolved context another context reads keeps its top-block
    signal even behind a branch: the consumer's child waits for its record,
    which the chart only sends at that signal."""
    source = """
def main(tf='60'):
    v = 0.0
    if lib.close > 10:
        v = lib.request.security(lib.syminfo.tickerid, tf, lib.close)
    w = lib.request.security(lib.syminfo.tickerid, "D", v + 1)
    lib.plot(w)
"""
    tree = _transform_tree(source)
    func = _find_func(tree)
    signal_if = func.body[0]
    assert isinstance(signal_if, ast.If)
    signalled = [stmt.value.args[0].value for stmt in signal_if.body]
    assert len(signalled) == 2
    branch = next(stmt for stmt in func.body if isinstance(stmt, ast.If)
                  and isinstance(stmt.test, ast.Compare)
                  and isinstance(stmt.test.left, ast.Attribute)
                  and stmt.test.left.attr == 'close')
    inline = [sub for sub in ast.walk(branch) if isinstance(sub, ast.Call)
              and isinstance(sub.func, ast.Name) and sub.func.id == '__sec_signal__']
    assert inline == []
    log.info("runtime-resolved producer in a branch is hoisted OK")


def __test_static_signal_in_branch_is_still_hoisted__(log):
    """A context with a module-level symbol and timeframe is resolved at setup,
    so its signal still starts in the top block."""
    source = """
def main(use_htf):
    lib.plot(lib.close)
    if use_htf:
        v = lib.request.security(lib.syminfo.tickerid, "60", lib.close)
"""
    tree = _transform_tree(source)
    func = _find_func(tree)
    signal_if = func.body[0]
    assert isinstance(signal_if, ast.If)
    assert signal_if.body[0].value.func.id == '__sec_signal__'
    log.info("static signal in a branch is still hoisted OK")


def __test_depends_on_runtime_context_is_recorded_not_rejected__(log):
    """A producer whose context is only resolved while the bar runs is a
    legal dependency — the runtime hands the resolved context to the child
    over the registry pipe, so the transformer only records the edge."""
    source = """
def main(sym):
    a = lib.request.security(sym, "D", lib.close)
    b = lib.request.security(lib.syminfo.tickerid, "W", lib.ta.sma(a, 3))
"""
    tree = ast.parse(source)
    tree = SecurityTransformer().visit(tree)
    ast.fix_missing_locations(tree)
    ctx_assign = _find_contexts(tree)
    sids = [k.value for k in ctx_assign.value.keys]
    ctx_b = ctx_assign.value.values[1]
    keys_b = [k.value for k in ctx_b.keys]
    assert [e.value for e in ctx_b.values[keys_b.index('depends')].elts] == [sids[0]]
    log.info("runtime-context dependency recorded OK")


def __test_hoisted_context_can_be_depended_on__(log):
    """A hoistable (input-derived) producer is a legal dependency."""
    source = """
def main():
    tf = lib.input.timeframe("D", "HTF")
    a = lib.request.security(lib.syminfo.tickerid, tf, lib.close)
    b = lib.request.security(lib.syminfo.tickerid, "W", lib.ta.sma(a, 3))
"""
    tree = ast.parse(source)
    tree = SecurityTransformer().visit(tree)
    ast.fix_missing_locations(tree)
    ctx_assign = _find_contexts(tree)
    sids = [k.value for k in ctx_assign.value.keys]
    ctx_b = ctx_assign.value.values[1]
    keys_b = [k.value for k in ctx_b.keys]
    deps = [e.value for e in ctx_b.values[keys_b.index('depends')].elts]
    assert deps == [sids[0]]
    log.info("hoisted producer dependency OK")


def __test_length_argument_is_not_tainted_by_the_series__(log):
    """``ta.sma(<security result>, len)`` must not taint ``len``.

    A shared length input reused by a second security expression would
    otherwise link two completely independent contexts (measured on the wild
    corpus' Swing Data script).
    """
    source = """
def main():
    length = lib.input.int(50)
    vol = lib.request.security(lib.syminfo.tickerid, "60", lib.volume)
    ma = lib.ta.sma(vol, length)
    avg = lib.request.security(lib.syminfo.tickerid, "W", ma)
    volDa = lib.request.security(lib.syminfo.tickerid, "D", lib.volume)
    maDa = lib.request.security(lib.syminfo.tickerid, "M", lib.ta.sma(volDa, length))
"""
    meta = _sec_meta(source)
    assert meta['60']['depends'] == []
    assert meta['W']['depends'] == ['60']
    assert meta['D']['depends'] == []
    assert meta['M']['depends'] == ['D']
    log.info("length argument not tainted OK")


def __test_display_calls_do_not_taint_their_arguments__(log):
    """``plot`` / ``fill`` / ``table.cell`` consume values, they do not feed
    them back into their arguments."""
    source = """
def main():
    a = lib.request.security(lib.syminfo.tickerid, "D", lib.close)
    shared = lib.input.float(1.0)
    lib.plot(a * shared)
    lib.plot(a, shared)
    b = lib.request.security(lib.syminfo.tickerid, "W", lib.ta.sma(lib.close, shared))
"""
    meta = _sec_meta(source)
    assert meta['W']['depends'] == []
    log.info("display calls do not taint OK")


def __test_nested_helper_function_is_analysed_not_a_fallback__(log):
    """A ``def`` nested in ``main()`` is analysed like any other scope — it is
    not an unmodelled construct, so independent contexts stay independent."""
    source = """
def main(tfSlope):
    def ma(kind, src, length):
        if kind == "EMA":
            return lib.ta.ema(src, length)
        return lib.ta.sma(src, length)
    adr = lib.request.security(lib.syminfo.tickerid, "D", lib.ta.sma(lib.high - lib.low, 21))
    slope = lib.request.security(lib.syminfo.tickerid, tfSlope, ma("SMA", lib.close, 20))
    rising = lib.request.security(lib.syminfo.tickerid, "W", lib.ta.rising(slope, 1))
"""
    meta = _sec_meta(source)
    assert meta['D']['depends'] == []
    # The tfSlope context has no timeframe literal, so it is keyed by its sid
    slope_key = next(k for k in meta if k not in ('D', 'W'))
    assert meta[slope_key]['depends'] == []
    assert meta['W']['depends'] == [slope_key]
    log.info("nested helper analysed OK")


def __test_scoped_fallback_does_not_leak_to_other_functions__(log):
    """An unmodelled construct only affects the writes of the scope it stands
    in, not every context in the module."""
    source = """
def risky():
    try:
        lib.plot(lib.close)
    except Exception:
        pass
    return lib.request.security(lib.syminfo.tickerid, "D", lib.close)
def main():
    a = risky()
    b = lib.request.security(lib.syminfo.tickerid, "W", lib.close)
    c = lib.request.security(lib.syminfo.tickerid, "M", lib.close)
"""
    meta = _sec_meta(source)
    # risky() is called first, so its write is the earliest site: the fallback
    # has nothing earlier to depend on, and it does not leak into main()
    assert meta['D']['depends'] == []
    assert meta['W']['depends'] == []
    assert meta['M']['depends'] == []
    log.info("scoped fallback OK")


def __test_symbol_alias_and_series_annotation__(log):
    """``s = syminfo.tickerid`` aliases and ``Series``-annotated assignments
    carry no taint of their own."""
    source = """
def main():
    s = lib.syminfo.tickerid
    a = lib.request.security(s, "D", (lib.bar_index + 1) / 252)
    b: Series = lib.request.security(s, "W", lib.close)
    c = lib.request.security(s, "M", lib.close if b > 0 else lib.open)
"""
    meta = _sec_meta(source)
    assert meta['D']['depends'] == []
    assert meta['W']['depends'] == []
    assert meta['M']['depends'] == ['W']
    log.info("alias / annotation / conditional OK")


def __test_walrus_is_modelled_as_an_assignment__(log):
    """A walrus binding (injected by the earlier lowering passes) carries taint
    like a plain assignment instead of falling back to every context."""
    source = """
def main():
    a = lib.request.security(lib.syminfo.tickerid, "D", lib.close)
    b = lib.request.security(lib.syminfo.tickerid, "W", lib.ta.sma((carry := a), 3))
    c = lib.request.security(lib.syminfo.tickerid, "M", lib.close)
    lib.plot(carry)
"""
    meta = _sec_meta(source)
    assert meta['D']['depends'] == []
    assert meta['W']['depends'] == ['D']
    # No fallback: the unrelated third context stays independent
    assert meta['M']['depends'] == []
    log.info("walrus modelled OK")


def __test_back_edge_through_user_function_is_dropped__(log):
    """The site of a write inside a user function is the site of the CALL that
    instantiates it, so a call standing later cannot be depended on."""
    source = """
def late():
    return lib.request.security(lib.syminfo.tickerid, "M", lib.close)
def main():
    carry = 0.0
    early = lib.request.security(lib.syminfo.tickerid, "D", carry)
    carry = late()
    after = lib.request.security(lib.syminfo.tickerid, "W", carry)
"""
    meta = _sec_meta(source)
    assert meta['D']['depends'] == []
    assert meta['M']['depends'] == []
    assert meta['W']['depends'] == ['M']
    log.info("call-site ordering OK")


def __test_depends_same_function_different_call_sites__(log):
    """One helper called from several expressions does not link them.

    The taint of an argument belongs to its own call site: a tainted call of
    ``fmt()`` must not put that taint on the results of the two untainted
    calls.
    """
    source = """
def main():
    def fmt(v):
        return lib.math.round(v)
    a = lib.request.security(lib.syminfo.tickerid, "D", lib.close)
    lbl = fmt(a)
    lib.plot(lbl)
    b = lib.request.security(lib.syminfo.tickerid, "W", fmt(lib.high))
    c = lib.request.security(lib.syminfo.tickerid, "M", fmt(lib.low))
"""
    meta = _sec_meta(source)
    assert meta['W']['depends'] == []
    assert meta['M']['depends'] == []
    log.info("per-call-site argument taint OK")


def __test_depends_argument_reaching_the_return__(log):
    """An argument the callee's return actually draws on IS a dependency."""
    source = """
def main():
    def scaled(v):
        return v * 2
    a = lib.request.security(lib.syminfo.tickerid, "D", lib.close)
    b = lib.request.security(lib.syminfo.tickerid, "W", scaled(a))
"""
    meta = _sec_meta(source)
    assert meta['D']['depends'] == []
    assert meta['W']['depends'] == ['D']
    log.info("argument reaching the return OK")


def __test_depends_argument_mutated_inside_the_callee__(log):
    """A callee that pushes a peer's value into its argument array taints the
    caller's array — the consumer reading it depends on the peer."""
    source = """
def main():
    def collect(box, v):
        lib.array.push(box, v)
    store = lib.array.new_float()
    a = lib.request.security(lib.syminfo.tickerid, "D", lib.close)
    collect(store, a)
    b = lib.request.security(lib.syminfo.tickerid, "W", lib.array.get(store, 0))
"""
    meta = _sec_meta(source)
    assert meta['D']['depends'] == []
    assert meta['W']['depends'] == ['D']
    log.info("argument mutation write-back OK")


def __test_same_named_locals_do_not_share_taint__(log):
    """Two functions with a same-named parameter are separate taint classes."""
    source = """
def main():
    def left(x):
        return x
    def right(x):
        return lib.ta.sma(lib.close, x)
    a = lib.request.security(lib.syminfo.tickerid, "D", lib.close)
    used = left(a)
    lib.plot(used)
    b = lib.request.security(lib.syminfo.tickerid, "W", right(lib.high))
"""
    meta = _sec_meta(source)
    assert meta['D']['depends'] == []
    assert meta['W']['depends'] == []
    log.info("scoped locals OK")


def __test_closure_over_a_main_local_keeps_the_dependency__(log):
    """A nested function reading a ``main`` local sees that local's taint —
    the scoping separates same-named locals, it does not cut closures."""
    source = """
def main():
    def carried():
        return lib.ta.sma(shared, 3)
    a = lib.request.security(lib.syminfo.tickerid, "D", lib.close)
    shared = a
    b = lib.request.security(lib.syminfo.tickerid, "W", carried())
"""
    meta = _sec_meta(source)
    assert meta['D']['depends'] == []
    assert meta['W']['depends'] == ['D']
    log.info("closure taint OK")


def __test_nonlocal_write_in_helper_taints_the_enclosing_variable__(log):
    """A helper assigning a peer's value to a ``nonlocal`` name writes the
    enclosing scope's variable, so a later reader of it depends on the peer
    even though the helper's own result is discarded."""
    source = """
def main():
    shared = 0.0
    def helper():
        nonlocal shared
        shared = lib.request.security(lib.syminfo.tickerid, "D", lib.close)
    helper()
    b = lib.request.security(lib.syminfo.tickerid, "W", shared)
"""
    meta = _sec_meta(source)
    assert meta['D']['depends'] == []
    assert meta['W']['depends'] == ['D']
    log.info("nonlocal write-back OK")


def __test_returned_argument_alias_carries_later_mutations__(log):
    """A callee returning its collection argument makes the receiving name an
    alias of that argument: pushing into the alias afterwards is visible to a
    reader of the original."""
    source = """
def main():
    def identity(box):
        return box
    store = lib.array.new_float()
    alias = identity(store)
    a = lib.request.security(lib.syminfo.tickerid, "D", lib.close)
    lib.array.push(alias, a)
    b = lib.request.security(lib.syminfo.tickerid, "W", lib.array.get(store, 0))
"""
    meta = _sec_meta(source)
    assert meta['D']['depends'] == []
    assert meta['W']['depends'] == ['D']
    log.info("returned alias OK")


def __test_returned_alias_composes_through_nested_calls__(log):
    """Alias summaries compose: a wrapper that returns another identity call,
    and an identity call fed with an identity call, both still alias the
    original collection, so a later push through the alias reaches it."""
    wrapper = """
def main():
    def identity(box):
        return box
    def wrapper(box):
        return identity(box)
    store = lib.array.new_float()
    alias = wrapper(store)
    a = lib.request.security(lib.syminfo.tickerid, "D", lib.close)
    lib.array.push(alias, a)
    b = lib.request.security(lib.syminfo.tickerid, "W", lib.array.get(store, 0))
"""
    nested_argument = """
def main():
    def identity(box):
        return box
    store = lib.array.new_float()
    alias = identity(identity(store))
    a = lib.request.security(lib.syminfo.tickerid, "D", lib.close)
    lib.array.push(alias, a)
    b = lib.request.security(lib.syminfo.tickerid, "W", lib.array.get(store, 0))
"""
    for source in (wrapper, nested_argument):
        meta = _sec_meta(source)
        assert meta['D']['depends'] == []
        assert meta['W']['depends'] == ['D']
    log.info("composed alias OK")


def __test_conditional_constant_mutation_carries_the_control_taint__(log):
    """Pushing a constant into a collection under a tainted condition changes
    the collection's state depending on that taint — directly and through a
    helper called from the conditional branch."""
    direct = """
def main():
    store = lib.array.new_float()
    a = lib.request.security(lib.syminfo.tickerid, "D", lib.close)
    if a > 0:
        lib.array.push(store, 1)
    b = lib.request.security(lib.syminfo.tickerid, "W", lib.array.size(store))
"""
    through_helper = """
def main():
    def collect(box):
        lib.array.push(box, 1)
    store = lib.array.new_float()
    a = lib.request.security(lib.syminfo.tickerid, "D", lib.close)
    if a > 0:
        collect(store)
    b = lib.request.security(lib.syminfo.tickerid, "W", lib.array.size(store))
"""
    for source in (direct, through_helper):
        meta = _sec_meta(source)
        assert meta['D']['depends'] == []
        assert meta['W']['depends'] == ['D']
    log.info("conditional mutation control taint OK")


def __test_write_inside_conditionally_called_helper_depends_on_the_condition__(log):
    """A security write inside a helper that is only called under a tainted
    condition depends on that condition's producer."""
    source = """
def main():
    def writer():
        return lib.request.security(lib.syminfo.tickerid, "W", lib.close)
    a = lib.request.security(lib.syminfo.tickerid, "D", lib.close)
    if a > 0:
        writer()
"""
    meta = _sec_meta(source)
    assert meta['D']['depends'] == []
    assert meta['W']['depends'] == ['D']
    log.info("call-site control taint OK")


def __test_helper_signal_sids(source: str) -> dict[str, list[str]]:
    """Sids signalled from each function's chart-guard signal block.

    Only blocks standing directly in a function's statement list are read, and
    every function of the module gets an entry (an empty list when it signals
    nothing), so a test can assert both where a signal went and where it did
    not.
    """
    tree = _transform_tree(source)
    result: dict[str, list[str]] = {}
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        sids: list[str] = []
        for stmt in func.body:
            calls = SecurityTransformer._guard_block_calls(stmt, '__sec_signal__')
            for call in calls or []:
                sids.append(call.args[0].value)
        result[func.name] = sids
    return result


def __test_helper_wait_sids(source: str) -> dict[str, list[str]]:
    """Sids waited on from each function's chart-guard wait block."""
    tree = _transform_tree(source)
    result: dict[str, list[str]] = {}
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        sids: list[str] = []
        for stmt in func.body:
            calls = SecurityTransformer._guard_block_calls(stmt, '__sec_wait__')
            for call in calls or []:
                sids.append(call.args[0].value)
        result[func.name] = sids
    return result


def __test_unconditional_helper_call_lifts_the_signal_to_main__(log):
    """A helper called once, unconditionally, from main's own statement list
    signals from main's top block — its round starts before any read."""
    source = """
def main():
    def htf(tf):
        return lib.request.security(lib.syminfo.tickerid, tf, lib.close)
    a = htf("D")
    lib.plot(a)
"""
    sids = __test_helper_signal_sids(source)
    assert len(sids['main']) == 1
    assert sids['htf'] == []

    # The wait follows the signal: it settles the round main now launches.
    waits = __test_helper_wait_sids(source)
    assert waits['main'] == sids['main']
    assert waits['htf'] == []

    # The substituted argument is the call site's own timeframe.
    tree = _transform_tree(source)
    func = _find_func(tree)
    signal_call = func.body[0].body[0].value
    assert ast.unparse(signal_call.args[2]) == "'D'"
    log.info("unconditional helper call lifted")


def __test_lift_into_a_returning_caller_puts_no_wait_before_the_reads__(log):
    """A caller that returns the helper call reads inside its ``return``.

    A wait placed in front of that ``return`` would settle the round before its
    value is read, so the chart would wait for the child's whole bar on every
    call instead of being released at the write. The lifted signal still starts
    the caller, and no wait block precedes the reads.
    """
    source = """
def main():
    def htf(tf):
        fast = lib.request.security(lib.syminfo.tickerid, tf, lib.close)
        return fast
    def outer(tf):
        return htf(tf)
    if lib.close > 0:
        lib.plot(outer("D"))
"""
    sids = __test_helper_signal_sids(source)
    assert len(sids['outer']) == 1
    assert sids['htf'] == []

    waits = __test_helper_wait_sids(source)
    assert waits['outer'] == []
    assert waits['htf'] == []

    tree = _transform_tree(source)
    outer = next(n for n in ast.walk(tree)
                 if isinstance(n, ast.FunctionDef) and n.name == 'outer')
    assert isinstance(outer.body[-1], ast.Return)
    log.info("no wait in front of a returning caller's reads")


def __test_helper_call_inside_if_is_not_lifted__(log):
    """A conditional call site would make main signal on bars the helper never
    ran on."""
    source = """
def main():
    def htf(tf):
        return lib.request.security(lib.syminfo.tickerid, tf, lib.close)
    if lib.close > 0:
        a = htf("D")
        lib.plot(a)
"""
    sids = __test_helper_signal_sids(source)
    assert sids['main'] == []
    assert len(sids['htf']) == 1
    log.info("conditional call site not lifted")


def __test_helper_call_in_short_circuit_is_not_lifted__(log):
    """A call under a ternary or an ``and`` may not be evaluated at all."""
    ternary = """
def main():
    def htf(tf):
        return lib.request.security(lib.syminfo.tickerid, tf, lib.close)
    a = htf("D") if lib.close > 0 else 0.0
    lib.plot(a)
"""
    short_circuit = """
def main():
    def htf(tf):
        return lib.request.security(lib.syminfo.tickerid, tf, lib.close)
    a = lib.close > 0 and htf("D")
    lib.plot(a)
"""
    for source in (ternary, short_circuit):
        sids = __test_helper_signal_sids(source)
        assert sids['main'] == []
        assert len(sids['htf']) == 1
    log.info("short-circuited call site not lifted")


def __test_helper_call_in_loop_body_is_not_lifted__(log):
    """A call in a loop body runs once per iteration, not once per bar."""
    source = """
def main():
    def htf(tf):
        return lib.request.security(lib.syminfo.tickerid, tf, lib.close)
    for i in (1, 2):
        a = htf("D")
        lib.plot(a)
"""
    sids = __test_helper_signal_sids(source)
    assert sids['main'] == []
    assert len(sids['htf']) == 1
    log.info("loop-body call site not lifted")


def __test_rebound_helper_parameter_is_not_lifted__(log):
    """A parameter rebound from itself no longer carries the value the call
    site passed, so the signal stays in the helper."""
    source = """
def main():
    def htf(tf):
        tf = tf + "x"
        return lib.request.security(lib.syminfo.tickerid, tf, lib.close)
    a = htf("D")
    lib.plot(a)
"""
    sids = __test_helper_signal_sids(source)
    assert sids['main'] == []
    assert len(sids['htf']) == 1
    log.info("rebound parameter not lifted")


def __test_nested_unconditional_helpers_lift_to_main__(log):
    """Two unconditional call levels lift the signal all the way to main."""
    source = """
def main():
    def inner(tf):
        return lib.request.security(lib.syminfo.tickerid, tf, lib.close)
    def outer(tf):
        return inner(tf)
    a = outer("D")
    lib.plot(a)
"""
    sids = __test_helper_signal_sids(source)
    assert len(sids['main']) == 1
    assert sids['outer'] == []
    assert sids['inner'] == []

    tree = _transform_tree(source)
    func = _find_func(tree)
    signal_call = func.body[0].body[0].value
    assert ast.unparse(signal_call.args[2]) == "'D'"
    log.info("nested unconditional helpers lifted")


def __test_conditionally_bound_call_argument_is_not_lifted__(log):
    """A call-site argument assigned in a branch is not available at main's
    top, so the signal cannot move there."""
    source = """
def main():
    def htf(tf):
        return lib.request.security(lib.syminfo.tickerid, tf, lib.close)
    tfv = "D"
    if lib.close > 0:
        tfv = "60"
    a = htf(tfv)
    lib.plot(a)
"""
    sids = __test_helper_signal_sids(source)
    assert sids['main'] == []
    assert len(sids['htf']) == 1
    log.info("conditionally bound argument not lifted")


def __test_helper_always_read_sids(source: str) -> dict[str, bool]:
    """``always_read`` of every context of ``source``, keyed by sid.

    :param source: the script source to transform
    :return: sid -> whether the transformer flagged the read unconditional
    """
    tree = _transform_tree(source)
    contexts = _find_contexts(tree)
    assert isinstance(contexts.value, ast.Dict)
    flags: dict[str, bool] = {}
    for sid_node, ctx_node in zip(contexts.value.keys, contexts.value.values):
        assert isinstance(sid_node, ast.Constant)
        assert isinstance(ctx_node, ast.Dict)
        value = False
        for key, val in zip(ctx_node.keys, ctx_node.values):
            if (isinstance(key, ast.Constant) and key.value == 'always_read'
                    and isinstance(val, ast.Constant)):
                value = bool(val.value)
        flags[str(sid_node.value)] = value
    return flags


def __test_top_level_read_is_always_read__(log):
    """A read standing in the entry's own body runs on every bar."""
    source = """
@lib.script.indicator("T")
def main():
    daily = lib.request.security(lib.syminfo.tickerid, "1D", lib.close)
    lib.plot(daily)
"""
    flags = __test_helper_always_read_sids(source)
    assert list(flags.values()) == [True]
    log.info("top-level read flagged always_read")


def __test_read_in_ternary_is_not_always_read__(log):
    """Either branch of a ternary may go unevaluated for the whole run."""
    source = """
@lib.script.indicator("T")
def main():
    def htf(tf):
        return (lib.request.security(lib.syminfo.tickerid, tf, lib.close)
                if lib.close > 0
                else lib.request.security(lib.syminfo.tickerid, tf, lib.open))
    a = htf("D")
    lib.plot(a)
"""
    flags = __test_helper_always_read_sids(source)
    assert len(flags) == 2
    assert not any(flags.values())
    log.info("ternary branches not flagged always_read")


def __test_read_in_loop_body_is_not_always_read__(log):
    """A loop body may run zero times, so its read is not unconditional."""
    source = """
@lib.script.indicator("T")
def main():
    a = 0.0
    for i in lib.pine_range(0, 2, 1):
        a = lib.request.security(lib.syminfo.tickerid, "1D", lib.close)
    lib.plot(a)
"""
    flags = __test_helper_always_read_sids(source)
    assert list(flags.values()) == [False]
    log.info("loop-body read not flagged always_read")


def __test_read_through_unconditional_helper_is_always_read__(log):
    """A helper the entry always calls carries its reads with it."""
    source = """
@lib.script.indicator("T")
def main():
    def htf(tf):
        return lib.request.security(lib.syminfo.tickerid, tf, lib.close)
    a = htf("D")
    lib.plot(a)
"""
    flags = __test_helper_always_read_sids(source)
    assert list(flags.values()) == [True]
    log.info("read behind an unconditional helper flagged always_read")
