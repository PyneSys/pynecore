"""Smoke test for the vendored-SDK import shim (``_sdk.py``).

Guards the vendor-shadow order: every plugin module reaches the DNSE SDK through
``from ._sdk import ...``, and a pip-installed ``dnse`` package (present in this
venv, see ``pip show dnse``) must never win over the vendored copy under
``_vendor/dnse`` — that would silently swap in the wrong client implementation.
"""
import importlib
import sys


def __test_sdk_import_resolves_vendored_dnse__():
    importlib.import_module("pynecore_dnse._sdk")

    assert "dnse" in sys.modules, "importing _sdk must import the top-level 'dnse' package"
    dnse_file = sys.modules["dnse"].__file__ or ""
    assert "_vendor" in dnse_file, (
        f"dnse resolved to {dnse_file!r}, expected the vendored copy under '_vendor' "
        "(a pip-installed 'dnse' must never shadow it)"
    )


def __test_sdk_reexports_expected_names__():
    from pynecore_dnse import _sdk

    assert "DNSEClient" in _sdk.__all__ and "TradingClient" in _sdk.__all__, \
        f"core SDK entry points missing from __all__: {sorted(_sdk.__all__)}"
    missing = [name for name in _sdk.__all__ if not hasattr(_sdk, name)]
    assert not missing, f"_sdk.__all__ lists names that aren't actually exported: {missing}"
    assert _sdk.DNSEClient is not None, "DNSEClient must resolve to a real class, not a placeholder"
    assert _sdk.TradingClient is not None, "TradingClient must resolve to a real class, not a placeholder"
