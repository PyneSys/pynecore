"""Phase 0a repro — the written contract must be TRUE (no type checker guards it).

Round-1 item 0 refuted the plan's premise: nothing type-checks the plugin
(pyright includes only ``src``, no CI job, tool not installed), so the value
of Phase 0 is the WRITTEN contract itself — and today it lies twice:

- two ``@override`` markers claim base-class overrides that do not exist
  anywhere in the MRO (``_identity_for``, ``_drain_pending_oco``) — a reader
  ports refactors against a fictional base contract;
- the contract validator certifies a ``...``-stub ``watch_orders`` as a
  working order stream (``overridden()`` is an identity check,
  core/broker/validation.py:223-224) — the engine would ``async for`` over
  ``None`` at runtime, after validation said the plugin was fine.

The GREEN control pins today's MRO order exactly: the 0a refactor
(config extraction + generic provider) must leave resolution byte-identical.
"""
import pytest
import pynecore.lib as lib

lib.bar_index = 0

from pynecore_dnse.broker import DNSEBroker
from pynecore.core.plugin.broker import BrokerPlugin
from pynecore.core.broker.validation import validate_plugin_contract


# --- RED 1: every @override marker must name a real base owner ---------------

def __test_every_override_marker_has_a_base_owner__():
    """A marked override with no owner in any base is a lie about the
    contract — the next reader ports reference code against a base method
    that does not exist (round-2 item 0, finding 14)."""
    bases = DNSEBroker.__mro__[1:]
    liars = []
    for name, attr in vars(DNSEBroker).items():
        func = getattr(attr, "__func__", attr)
        if not getattr(func, "__override__", False):
            continue
        if not any(name in vars(base) for base in bases):
            liars.append(name)
    assert not liars, (
        f"@override on {liars} but no base in the MRO defines them — "
        f"remove the markers or the methods are misnamed")


# --- RED 2: the validator must not certify a stub order stream ---------------

def __test_validator_rejects_stub_watch_orders__():
    """``overridden()`` is an identity check: a class whose only
    ``watch_orders`` is ``async def watch_orders(self): ...`` passes
    validation with watch_orders declared supported — then the engine
    ``async for``s over None at runtime. The validator must flag it.
    (Red-first control, round-2 probe: a plugin with NO watch_orders at all
    is correctly flagged today — only the stub false-passes.)"""
    from pynecore.core.broker.models import CapabilityLevel

    class _StubStreamBroker(DNSEBroker):
        async def watch_orders(self):  # a stub, NOT an async generator
            ...

        def get_capabilities(self):
            caps = super().get_capabilities()
            assert caps.watch_orders.is_supported, "premise: DNSE declares support"
            return caps

    from pynecore_dnse.broker import DNSEBrokerConfig
    config = DNSEBrokerConfig(api_key="k", api_secret="s", account_no="A",
                              trading_token="t", token_file="/nonexistent")
    plugin = _StubStreamBroker(symbol="VN30F1M", timeframe="15", config=config)

    errors, warnings = validate_plugin_contract(plugin)
    text = " ".join(errors + warnings)
    assert "watch_orders" in text, (
        "a declared-supported watch_orders whose override cannot produce an "
        "async iterator must be flagged — certifying the stub is the "
        "validator false-pass (round-2 item 0, finding 13)")


# --- GREEN control: today's MRO order is pinned through the 0a refactor ------

def __test_mro_order_is_pinned__():
    """DNSEProvider must precede BrokerPlugin (round 1: the verbatim
    reference shape breaks DNSE) and the 0a config extraction must leave
    resolution order byte-identical."""
    names = [c.__name__ for c in DNSEBroker.__mro__]
    assert names.index("DNSEProvider") < names.index("BrokerPlugin"), names
    assert names[0] == "DNSEBroker"


# --- panel anchors (#66): controls for the validator fix and the 0a move -----

def __test_real_watch_orders_passes_the_shape_check__():
    """The fixed validator must keep certifying the REAL stream: DNSEBroker's
    watch_orders is an async generator function (as are all three reference
    plugins' — measured by the panel), so the stub-flag must not fire."""
    import inspect
    assert inspect.isasyncgenfunction(inspect.unwrap(DNSEBroker.watch_orders))


def __test_config_reexport_and_new_home_are_the_same_object__():
    """0a moved DNSEBrokerConfig to config.py; broker.py re-exports it so
    every existing import keeps working (testing/ and tools/ import it from
    broker — the #55 lesson's sweep gate)."""
    from pynecore_dnse.broker import DNSEBrokerConfig as from_broker
    from pynecore_dnse.config import DNSEBrokerConfig as from_config
    assert from_broker is from_config
