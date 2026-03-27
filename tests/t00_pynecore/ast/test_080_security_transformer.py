"""
@pyne
"""
import ast

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

    # Statement 3: write block (if __active_security__ == sec_id)
    write_if = func.body[2]
    assert isinstance(write_if, ast.If)
    assert isinstance(write_if.test.ops[0], ast.Eq)
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
