"""
Unit tests for the ModulePropertyTransformer rewrite rules.

The transformer is registry-driven (module_properties.json): bare property reads
become calls, function-and-namespace module calls are routed to the module's
self-named function, unknown names on known pynecore.lib modules raise at
transform time, and everything else stays a plain attribute access.
"""
import ast

import pytest

from pynecore.transformers.module_property import ModulePropertyTransformer


def _transform(src: str) -> str:
    tree = ast.parse(src)
    transformed = ModulePropertyTransformer().visit(tree)
    ast.fix_missing_locations(transformed)
    return ast.unparse(transformed)


def __test_property_bare_read_becomes_call__():
    assert _transform("x = lib.ta.tr") == "x = lib.ta.tr()"
    assert _transform("x = lib.time") == "x = lib.time()"
    assert _transform("x = lib.na") == "x = lib.na()"


def __test_property_explicit_call_untouched__():
    assert _transform("x = lib.ta.tr(True)") == "x = lib.ta.tr(True)"
    assert _transform("x = lib.year(t, 'UTC')") == "x = lib.year(t, 'UTC')"


def __test_variable_is_plain_read__():
    assert _transform("x = lib.close") == "x = lib.close"
    assert _transform("x = lib.plot.style_line") == "x = lib.plot.style_line"
    assert _transform("x = lib.dayofweek.monday") == "x = lib.dayofweek.monday"
    assert _transform("x = lib.syminfo.mincontract") == "x = lib.syminfo.mincontract"


def __test_function_and_namespace_module_call_routed__():
    assert _transform("lib.plot(x)") == "lib.plot.plot(x)"
    assert _transform("lib.hline(1.0)") == "lib.hline.hline(1.0)"
    assert _transform("lib.alert('msg')") == "lib.alert.alert('msg')"
    assert _transform("x = lib.dayofweek(t)") == "x = lib.dayofweek.dayofweek(t)"


def __test_promoted_bare_property_routed_to_self_named_function__():
    assert _transform("x = lib.dayofweek") == "x = lib.dayofweek.dayofweek()"
    assert _transform("x = lib.strategy.opentrades") == "x = lib.strategy.opentrades.opentrades()"
    assert _transform("x = lib.strategy.closedtrades") == "x = lib.strategy.closedtrades.closedtrades()"


def __test_namespaced_function_call_untouched__():
    assert _transform("x = lib.strategy.opentrades.commission(0)") == \
           "x = lib.strategy.opentrades.commission(0)"


def __test_submodule_reference_is_plain_read__():
    assert _transform("x = lib.ta") == "x = lib.ta"
    assert _transform("x = lib.plot") == "x = lib.plot"


def __test_underscore_name_is_plain_read__():
    assert _transform("x = lib._datetime") == "x = lib._datetime"


def __test_function_reference_is_plain_read__():
    assert _transform("f = lib.ta.sma") == "f = lib.ta.sma"


def __test_user_library_path_is_plain_read__():
    assert _transform("x = lib.PineCoders.getSeries.v1.something") == \
           "x = lib.PineCoders.getSeries.v1.something"


def __test_unknown_name_on_known_module_raises__():
    with pytest.raises(SyntaxError, match="nosuchname"):
        _transform("x = lib.ta.nosuchname")
    with pytest.raises(SyntaxError, match="currency__"):
        _transform("x = lib.syminfo.currency__")


def __test_annotation_untouched__():
    assert _transform("x: lib.ta.tr = 1") == "x: lib.ta.tr = 1"


def __test_non_lib_rooted_untouched__():
    assert _transform("x = other.ta.tr") == "x = other.ta.tr"
