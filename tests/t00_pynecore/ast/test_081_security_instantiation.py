"""
@pyne
"""
import ast

import pytest

from pynecore.transformers.security_instantiation import SecurityInstantiationTransformer


def _transform(source: str) -> str:
    tree = ast.parse(source)
    tree = SecurityInstantiationTransformer().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def __test_multi_call_site_clones__(log):
    """A security-bearing function called from N sites is cloned per site,
    each site rewired to its own clone (Pine instantiation semantics)."""
    source = """
def main():
    def f_htf_trend(tf):
        c = lib.request.security(lib.syminfo.tickerid, tf, lib.close)
        e = lib.request.security(lib.syminfo.tickerid, tf, lib.ta.ema(lib.close, 50))
        return 1 if c > e else -1 if c < e else 0
    t1 = f_htf_trend("1")
    t2 = f_htf_trend("5")
    t3 = f_htf_trend("15")
"""
    result = _transform(source)
    assert result.count('def f_htf_trend') == 3
    assert 't2 = f_htf_trend__pyne_inst' in result
    assert 't3 = f_htf_trend__pyne_inst' in result
    # Each clone carries its own security calls -> 6 syntactic calls total.
    assert result.count('lib.request.security(') == 6
    log.info("multi-call-site cloning OK")


def __test_transitive_instantiation__(log):
    """A caller of a security-bearing function is itself security-bearing:
    its call sites instantiate too, multiplying through the call chain."""
    source = """
def g(tf):
    return lib.request.security(lib.syminfo.tickerid, tf, lib.close)
def f(tf):
    return g(tf) + g("D")
def main():
    a = f("5")
    b = f("15")
"""
    result = _transform(source)
    # f cloned once (2 sites), each f-instance holds 2 g-sites -> 4 g defs.
    assert result.count('def f') == 2
    assert result.count('def g') == 4
    assert result.count('lib.request.security(') == 4
    log.info("transitive instantiation OK")


def __test_single_site_untouched__(log):
    """One call site needs no clone — the module must be left as-is."""
    source = """
def f(tf):
    return lib.request.security(lib.syminfo.tickerid, tf, lib.close)
def main():
    a = f("5")
"""
    result = _transform(source)
    assert '__pyne_inst' not in result
    assert result.count('def f') == 1
    log.info("single-site pass-through OK")


def __test_recursion_bails__(log):
    """Recursive security-bearing functions keep the shared context (no
    clone) — cloning a cycle would never converge."""
    source = """
def f(tf, n):
    x = lib.request.security(lib.syminfo.tickerid, tf, lib.close)
    return x if n == 0 else f(tf, n - 1)
def main():
    a = f("5", 1)
    b = f("15", 1)
"""
    result = _transform(source)
    assert '__pyne_inst' not in result
    log.info("recursion bail-out OK")


def __test_alias_reference_bails__(log):
    """A function referenced outside a direct-call position (alias) is not
    cloned — the alias would keep calling the original."""
    source = """
def f(tf):
    return lib.request.security(lib.syminfo.tickerid, tf, lib.close)
def main():
    g = f
    a = f("5")
    b = f("15")
"""
    result = _transform(source)
    assert '__pyne_inst' not in result
    log.info("alias bail-out OK")


def __test_security_lower_tf_clones_too__(log):
    """``request.security_lower_tf`` marks a function security-bearing the
    same way as plain ``request.security``."""
    source = """
def main():
    def f(tf):
        return lib.request.security_lower_tf(lib.syminfo.tickerid, tf, lib.close)
    a = f("1")
    b = f("5")
"""
    result = _transform(source)
    assert result.count('def f') == 2
    log.info("security_lower_tf cloning OK")


def __test_no_security_no_op__(log):
    """Modules without security calls are returned untouched (fast path)."""
    source = """
def f(x):
    return x + 1
def main():
    a = f(1)
    b = f(2)
"""
    result = _transform(source)
    assert '__pyne_inst' not in result
    log.info("no-security fast path OK")


def __test_call_site_constant_pinned_into_clone__(log):
    """Each instance gets its call site's static arguments written back as
    bindings at the top of its body.

    After instantiation a security-bearing function has exactly one call site,
    so its timeframe/symbol parameters are statically known. Pinning them makes
    the ``__sec_signal__`` hoistable, which is what lets another
    ``request.security()`` depend on this one.
    """
    source = """
def f(tf):
    return lib.request.security(lib.syminfo.tickerid, tf, lib.close)
def main():
    a = f("60")
    b = f("240")
"""
    result = _transform(source)
    assert "tf = '60'" in result
    assert "tf = '240'" in result
    log.info("call-site constant pinning OK")


def __test_single_call_site_constant_pinned__(log):
    """A function with one call site is specialized too — no clone needed."""
    source = """
def f(tf):
    return lib.request.security(lib.syminfo.tickerid, tf, lib.close)
def main():
    a = f(lib.timeframe.period)
"""
    result = _transform(source)
    assert '__pyne_inst' not in result
    assert 'tf = lib.timeframe.period' in result
    log.info("single-site specialization OK")


def __test_call_site_call_argument_not_pinned__(log):
    """A call-valued argument is never copied into the callee — duplicating
    ``input.timeframe()`` would register the input twice."""
    source = """
def f(tf):
    return lib.request.security(lib.syminfo.tickerid, tf, lib.close)
def main():
    a = f(lib.input.timeframe("60", "HTF"))
"""
    result = _transform(source)
    assert result.count('lib.input.timeframe') == 1
    assert 'tf = lib.input' not in result
    log.info("call argument not pinned OK")


def __test_specialization_skips_bailed_out_functions__(log):
    """A function excluded from instantiation (alias reference) keeps its
    parameters untouched — its call sites are not statically paired."""
    source = """
def f(tf):
    return lib.request.security(lib.syminfo.tickerid, tf, lib.close)
def main():
    g = f
    a = f("5")
    b = f("15")
"""
    result = _transform(source)
    assert "tf = '5'" not in result
    assert "tf = '15'" not in result
    log.info("bail-out functions not specialized OK")


def __test_helper_fan_source(depth: int, leaf_calls: int = 1) -> str:
    """``f{i}`` calls ``f{i-1}`` twice, down to a leaf with ``leaf_calls``
    security calls, so ``f{depth}`` reaches ``leaf_calls * 2 ** depth`` of them."""
    leaf = " + ".join(["lib.request.security(lib.syminfo.tickerid, 'D', lib.close)"] * leaf_calls)
    lines = [f"def f0():\n    return {leaf}"]
    for i in range(1, depth + 1):
        lines.append(f"def f{i}():\n    return f{i - 1}() + f{i - 1}()")
    lines.append(f"def main():\n    v = f{depth}()")
    return "\n".join(lines) + "\n"


def __test_fanout_within_tv_bound_instantiates__(log):
    """64 call paths below one called function run on TradingView (measured),
    so their 120 clones must not trip any ceiling."""
    result = _transform(__test_helper_fan_source(6))
    assert result.count('lib.request.security(') == 64
    assert result.count('def f0') == 64
    log.info("TV-valid fan-out instantiated OK")


def __test_global_scope_is_not_bounded__(log):
    """main (the global scope) may hold any number of security calls; only a
    called function is bounded."""
    calls = "\n".join(
        f"    a{i} = lib.request.security(lib.syminfo.tickerid, 'D', lib.close)" for i in range(300)
    )
    result = _transform(f"def f():\n    return 1\ndef main():\n{calls}\n")
    assert result.count('lib.request.security(') == 300
    log.info("global scope unbounded OK")


def __test_fanout_beyond_tv_bound_rejected__(log):
    """A called function reaching more than 254 security calls is rejected
    (TradingView rejects 128 float requests, 256 elements) before cloning."""
    _transform(__test_helper_fan_source(6, leaf_calls=2))
    with pytest.raises(SyntaxError, match="reaches 256 request.security calls"):
        _transform(__test_helper_fan_source(7, leaf_calls=2))
    with pytest.raises(SyntaxError, match="'f8' reaches 256 "):
        _transform(__test_helper_fan_source(40))
    log.info("runaway fan-out rejected OK")


def __test_fanout_counts_every_cloned_subtree__(log):
    """Cloning copies a function's lambdas, class bodies and nested defs, so
    security calls there count toward the bound as well."""
    security = "lib.request.security(lib.syminfo.tickerid, 'D', lib.close)"
    lambda_leaf = __test_helper_fan_source(8).replace(
        f"return {security}", f"return (lambda: {security})()")
    with pytest.raises(SyntaxError, match="'f8' reaches 256 "):
        _transform(lambda_leaf)

    method_leaf = __test_helper_fan_source(8).replace(
        f"return {security}",
        f"class C:\n        def m(self):\n            return {security}\n    return C().m()")
    with pytest.raises(SyntaxError, match="'f8' reaches 256 "):
        _transform(method_leaf)

    lines = [f"def f0():\n    return {security}"]
    for i in range(1, 9):
        lines.append(f"def f{i}():\n    def g{i}():\n        return f{i - 1}() + f{i - 1}()\n"
                     f"    return 1")
    lines.append("def main():\n    v = f8()")
    with pytest.raises(SyntaxError, match="'f8' reaches 256 "):
        _transform("\n".join(lines) + "\n")

    lines = [f"def f0():\n    return {security}"]
    for i in range(1, 9):
        lines.append(f"def f{i}():\n    def g{i}():\n        return f{i - 1}()\n"
                     f"    return g{i}() + g{i}()")
    lines.append("def main():\n    v = f8()")
    with pytest.raises(SyntaxError, match="'f8' reaches 256 "):
        _transform("\n".join(lines) + "\n")
    log.info("lambda / class / nested-def fan-out rejected OK")


def __test_fanout_counts_ambiguous_callees__(log):
    """A duplicate-defined function is never cloned, but its callers are, so a
    call to it counts as its largest definition."""
    security = "lib.request.security(lib.syminfo.tickerid, 'D', lib.close)"
    source = __test_helper_fan_source(7)
    _transform(source)
    duplicated = source.replace("def f1():", f"def f0():\n    return {security} + {security}\ndef f1():", 1)
    with pytest.raises(SyntaxError, match="'f7' reaches 256 "):
        _transform(duplicated)
    log.info("ambiguous-callee fan-out rejected OK")


def __test_fanout_resolves_calls_lexically__(log):
    """A call binds to the nearest enclosing definition, so calls to a
    stateless local function that shadows a security-bearing global one count
    nothing."""
    security = "lib.request.security(lib.syminfo.tickerid, 'D', lib.close)"
    calls = " + ".join(["sample()"] * 128)
    source = (f"def sample():\n    return {security} + {security}\n"
              f"def f():\n    def sample():\n        return 1\n    return {calls}\n"
              f"def main():\n    v = f()\n")
    result = _transform(source)
    assert result.count('lib.request.security(') == 2
    log.info("lexically shadowed callee accepted OK")


def __test_shadowed_stateless_chain_not_cloned__(log):
    """A caller of a stateless local function that shadows a security-bearing
    global one is not a bearer, so a doubling chain above it stays uncloned."""
    security = "lib.request.security(lib.syminfo.tickerid, 'D', lib.close)"
    source = (f"def sample():\n    return {security} + {security}\n"
              f"def f0():\n    def sample():\n        return 1\n    return sample()\n")
    for i in range(1, 9):
        source += f"def f{i}():\n    return f{i - 1}() + f{i - 1}()\n"
    source += "def main():\n    v = f8()\n"
    result = _transform(source)
    assert result.count('def f') == 9
    assert '__pyne_inst' not in result
    log.info("shadowed stateless chain left uncloned OK")


def __test_fanout_stops_at_local_bindings__(log):
    """A call to a name the function binds to another value (assignment,
    parameter, import) calls that value and counts nothing; a ``global``
    declaration still resolves it to the module definition."""
    security = "lib.request.security(lib.syminfo.tickerid, 'D', lib.close)"
    calls = " + ".join(["sample(1)"] * 128)
    head = f"def sample(x):\n    return {security} + {security}\n"
    tail = "def main():\n    v = f(abs)\n"
    for body in ("    sample = abs\n", "    from math import fabs as sample\n",
                 "    for sample in (abs,):\n        pass\n"):
        result = _transform(f"{head}def f(fn):\n{body}    return {calls}\n{tail}")
        assert result.count('lib.request.security(') == 2
        assert '__pyne_inst' not in result
    result = _transform(f"{head}def f(sample):\n    return {calls}\n{tail}")
    assert result.count('lib.request.security(') == 2
    with pytest.raises(SyntaxError, match="'f' reaches 256 "):
        _transform(f"{head}def f(fn):\n    global sample\n    return {calls}\n{tail}")
    log.info("local bindings stop the lexical lookup OK")


def __test_recursive_cycle_count_ignores_definition_order__(log):
    """A caller of a recursive cycle counts the same, and so clones the same,
    whichever function of the cycle is defined first."""
    f = "def f(n):\n    return lib.request.security(lib.syminfo.tickerid, 'D', lib.close) + g(n)\n"
    g = "def g(n):\n    return f(n - 1)\n"
    h = "def h():\n    return g(1) + g(2)\n"
    tail = "def main():\n    a = h()\n    b = h()\n"
    for order in ((f, g, h), (g, f, h), (h, f, g), (h, g, f)):
        module = ast.parse("".join(order) + tail)
        infos = SecurityInstantiationTransformer()._collect_functions(module)
        total = SecurityInstantiationTransformer._request_totals(module, infos)
        assert {info.node.name: total(info.node) for info in infos} == {
            'f': 1, 'g': 1, 'h': 2, 'main': 4}
        result = _transform("".join(order) + tail)
        assert result.count('def h') == 2
        assert result.count('def f') == 1
        assert result.count('def g') == 1
    log.info("recursive cycle count is order-independent OK")
