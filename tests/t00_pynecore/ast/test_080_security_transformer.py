"""
@pyne
"""
import ast

import pytest

from pynecore.transformers.security import SecurityTransformer


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


def __test_lookahead_runtime_expr_rejected__(log):
    """Runtime `lookahead` expressions are rejected at transform time.

    Pine requires ``lookahead`` to be a ``barmerge.lookahead_*`` constant.
    A local variable or IfExp would be inlined into the module-level
    ``__security_contexts__`` literal and NameError at import time, so the
    transformer raises a clear SyntaxError instead.
    """
    source = """
def main():
    mode = lib.barmerge.lookahead_on if cond else lib.barmerge.lookahead_off
    val = lib.request.security(
        lib.syminfo.tickerid, "1D", lib.close,
        lookahead=mode,
    )
"""
    with pytest.raises(SyntaxError, match="must be a constant"):
        _transform(source)
