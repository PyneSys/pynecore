"""
Tests for :class:`ScriptRequirementsTransformer` detection and for the
startup-time :func:`validate_at_startup` pure function.

The transformer tests run the transformer directly on synthetic AST modules
and assert on the injected ``_broker_requirements`` keyword of the
``@script.strategy(...)`` decorator — no ScriptRunner, no actual execution.
"""
from __future__ import annotations

import ast
import textwrap

from pynecore.core.broker.models import ScriptRequirements, ExchangeCapabilities
from pynecore.core.broker.validation import validate_at_startup
from pynecore.transformers.script_requirements import ScriptRequirementsTransformer


def _transform(src: str) -> ast.Module:
    tree = ast.parse(textwrap.dedent(src))
    return ScriptRequirementsTransformer().visit(tree)


def _get_requirements_keyword(tree: ast.Module) -> dict[str, bool] | None:
    """Return the flags dict injected into @script.strategy's call, or None."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for dec in node.decorator_list:
                if (isinstance(dec, ast.Call)
                        and isinstance(dec.func, ast.Attribute)
                        and dec.func.attr == 'strategy'):
                    for kw in dec.keywords:
                        if kw.arg == '_broker_requirements' and isinstance(kw.value, ast.Call):
                            return {
                                k.arg: k.value.value  # type: ignore[attr-defined]
                                for k in kw.value.keywords
                                if k.arg is not None
                            }
    return None


def __test_indicator_script_has_no_injection__():
    """A script without @script.strategy must be left alone."""
    tree = _transform("""
        @script.indicator('Foo')
        def main():
            pass
    """)
    assert _get_requirements_keyword(tree) is None


def __test_market_only_entry_detects_market_orders__():
    tree = _transform("""
        @script.strategy('S')
        def main():
            strategy.entry('Long', strategy.long, qty=1)
    """)
    assert _get_requirements_keyword(tree) == {'market_orders': True}


def __test_entry_with_limit_detects_limit_orders__():
    tree = _transform("""
        @script.strategy('S')
        def main():
            strategy.entry('Long', strategy.long, qty=1, limit=50000.0)
    """)
    assert _get_requirements_keyword(tree) == {'limit_orders': True}


def __test_entry_with_stop_detects_stop_orders__():
    tree = _transform("""
        @script.strategy('S')
        def main():
            strategy.entry('Long', strategy.long, qty=1, stop=45000.0)
    """)
    assert _get_requirements_keyword(tree) == {'stop_orders': True}


def __test_entry_with_limit_and_stop_detects_stop_limit__():
    tree = _transform("""
        @script.strategy('S')
        def main():
            strategy.entry('Long', strategy.long, qty=1, limit=50000.0, stop=49500.0)
    """)
    flags = _get_requirements_keyword(tree)
    assert flags == {
        'limit_orders': True, 'stop_orders': True, 'stop_limit_orders': True,
    }


def __test_exit_price_bracket_detects_tp_sl__():
    """strategy.exit with both limit and stop → OCA reduce bracket."""
    tree = _transform("""
        @script.strategy('S')
        def main():
            strategy.exit('TP', from_entry='Long', limit=60000.0, stop=45000.0)
    """)
    flags = _get_requirements_keyword(tree)
    assert flags == {
        'limit_orders': True, 'stop_orders': True, 'tp_sl_bracket': True,
        'exit_orders': True,
    }


def __test_exit_tick_bracket_detects_tp_sl__():
    """strategy.exit with profit+loss ticks also requires the bracket capability."""
    tree = _transform("""
        @script.strategy('S')
        def main():
            strategy.exit('TP', from_entry='Long', profit=100, loss=50)
    """)
    flags = _get_requirements_keyword(tree)
    assert flags == {
        'limit_orders': True, 'stop_orders': True, 'tp_sl_bracket': True,
        'exit_orders': True,
    }


def __test_exit_trail_offset_detects_trailing_stop__():
    tree = _transform("""
        @script.strategy('S')
        def main():
            strategy.exit('TR', from_entry='Long', trail_offset=50, trail_points=100)
    """)
    flags = _get_requirements_keyword(tree)
    assert flags == {'trailing_stop': True, 'exit_orders': True}


def __test_strategy_order_detects_strategy_order_flag__():
    """strategy.order() bypasses pyramiding → needs its own capability flag."""
    tree = _transform("""
        @script.strategy('S')
        def main():
            strategy.order('X', strategy.long, qty=1)
    """)
    flags = _get_requirements_keyword(tree)
    assert flags == {'market_orders': True, 'strategy_order': True}


def __test_close_detects_market_orders__():
    tree = _transform("""
        @script.strategy('S')
        def main():
            strategy.close('Long')
    """)
    assert _get_requirements_keyword(tree) == {
        'market_orders': True, 'exit_orders': True,
    }


def __test_close_all_detects_exit_orders__():
    tree = _transform("""
        @script.strategy('S')
        def main():
            strategy.close_all()
    """)
    assert _get_requirements_keyword(tree) == {
        'market_orders': True, 'exit_orders': True,
    }


def __test_plain_exit_detects_exit_orders__():
    """A bracket-less strategy.exit still requires reduce-only semantics."""
    tree = _transform("""
        @script.strategy('S')
        def main():
            strategy.exit('X', from_entry='Long')
    """)
    assert _get_requirements_keyword(tree) == {'exit_orders': True}


def __test_import_is_injected_when_requirements_present__():
    tree = _transform("""
        @script.strategy('S')
        def main():
            strategy.entry('Long', strategy.long, qty=1, limit=50000.0)
    """)
    imports = [stmt for stmt in tree.body if isinstance(stmt, ast.ImportFrom)]
    assert any(
        imp.module == 'pynecore.core.broker.models'
        and any(a.name == 'ScriptRequirements' for a in imp.names)
        for imp in imports
    )


def __test_lib_strategy_prefix_is_also_detected__():
    """After ImportNormalizer rewrites, calls appear as lib.strategy.entry(...)."""
    tree = _transform("""
        @script.strategy('S')
        def main():
            lib.strategy.entry('Long', lib.strategy.long, qty=1, limit=50000.0)
    """)
    assert _get_requirements_keyword(tree) == {'limit_orders': True}


def __test_lib_script_strategy_decorator_is_detected__():
    """
    The ``@script.strategy(...)`` decorator is rewritten to
    ``@lib.script.strategy(...)`` by ``ImportNormalizer``; injection must
    still reach it.
    """
    tree = _transform("""
        @lib.script.strategy('S')
        def main():
            lib.strategy.entry('Long', lib.strategy.long, qty=1)
    """)
    assert _get_requirements_keyword(tree) == {'market_orders': True}


# === validate_at_startup ===

def __test_validate_empty_when_requirements_satisfied__():
    reqs = ScriptRequirements(tp_sl_bracket=True)
    caps = ExchangeCapabilities(tp_sl_bracket=True)
    assert validate_at_startup(reqs, caps) == []


def __test_validate_reports_missing_bracket__():
    reqs = ScriptRequirements(tp_sl_bracket=True)
    caps = ExchangeCapabilities()
    errors = validate_at_startup(reqs, caps)
    assert len(errors) == 1
    assert 'TP+SL' in errors[0]


def __test_validate_collects_all_missing_capabilities__():
    reqs = ScriptRequirements(
        stop_orders=True, stop_limit_orders=True,
        tp_sl_bracket=True, trailing_stop=True,
        exit_orders=True,
    )
    caps = ExchangeCapabilities()
    errors = validate_at_startup(reqs, caps)
    assert len(errors) == 5


def __test_validate_rejects_exit_without_reduce_only_capability__():
    """A script that uses strategy.exit/close must refuse to start on an
    exchange that doesn't honour reduce-only semantics — otherwise a
    later-arriving exit can flip the book to the other side."""
    reqs = ScriptRequirements(exit_orders=True)
    caps = ExchangeCapabilities()  # reduce_only=False
    errors = validate_at_startup(reqs, caps)
    assert len(errors) == 1
    assert 'reduce-only' in errors[0]


def __test_validate_accepts_exit_with_reduce_only_capability__():
    reqs = ScriptRequirements(exit_orders=True)
    caps = ExchangeCapabilities(reduce_only=True)
    assert validate_at_startup(reqs, caps) == []
