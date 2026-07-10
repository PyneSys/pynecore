"""
@pyne
"""
import ast

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
