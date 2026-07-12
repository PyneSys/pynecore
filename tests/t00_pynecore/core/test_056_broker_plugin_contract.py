"""
Tests for the startup-time :func:`validate_plugin_contract` probe.

Unlike the duck-typed mocks in the runner tests, every plugin here is a REAL
:class:`BrokerPlugin` subclass — the probe works by method-identity comparison
against the base class, so only genuine overrides count. The abstract surface
is stubbed minimally; nothing is ever connected or executed.
"""
from pynecore.core.broker.models import (
    CancelDispositionOutcome,
    CapabilityLevel,
    ExchangeCapabilities,
)
from pynecore.core.broker.validation import validate_plugin_contract
from pynecore.core.plugin.broker import BrokerPlugin


def _caps(**overrides) -> ExchangeCapabilities:
    """A conforming capability set (Capital.com-shaped), tweakable per test."""
    base = dict(
        stop_order=CapabilityLevel.NATIVE,
        trailing_stop=CapabilityLevel.NATIVE,
        tp_sl_bracket=CapabilityLevel.NATIVE,
        partial_qty_bracket_exit=CapabilityLevel.SOFTWARE,
        partial_qty_bracket_exit_pyramiding=CapabilityLevel.SOFTWARE,
        oca_cancel=CapabilityLevel.SOFTWARE,
        amend_order=CapabilityLevel.SOFTWARE,
        cancel_all=CapabilityLevel.SOFTWARE,
        reduce_only=CapabilityLevel.SOFTWARE,
        watch_orders=CapabilityLevel.SOFTWARE,
        fetch_position=CapabilityLevel.NATIVE,
        idempotency=CapabilityLevel.SOFTWARE,
    )
    base.update(overrides)
    return ExchangeCapabilities(**base)


class _ConformingPlugin(BrokerPlugin):
    """Minimal plugin that passes the contract probe with zero findings."""

    capabilities: ExchangeCapabilities = _caps()

    # --- ProviderPlugin abstracts (never called by the probe) ---

    @classmethod
    def to_tradingview_timeframe(cls, timeframe):
        return timeframe

    @classmethod
    def to_exchange_timeframe(cls, timeframe):
        return timeframe

    def get_list_of_symbols(self, *args, **kwargs):
        return []

    def update_symbol_info(self):
        raise AssertionError("not used by the contract probe")

    def download_ohlcv(self, time_from, time_to, on_progress=None,
                       limit=None, with_extra=False):
        raise AssertionError("not used by the contract probe")

    # --- LiveProviderPlugin abstracts ---

    async def connect(self):
        pass

    async def disconnect(self):
        pass

    @property
    def is_connected(self) -> bool:
        return False

    async def watch_ohlcv(self, symbol, timeframe):
        raise AssertionError("not used by the contract probe")

    # --- BrokerPlugin abstracts + contract overrides ---

    async def execute_entry(self, envelope):
        return []

    async def execute_exit(self, envelope):
        return []

    async def execute_close(self, envelope):
        raise AssertionError("not used by the contract probe")

    async def execute_cancel(self, envelope):
        return True

    async def execute_cancel_with_outcome(self, envelope):
        return CancelDispositionOutcome.UNKNOWN

    async def get_open_orders(self, symbol=None):
        return []

    async def get_position(self, symbol):
        return None

    async def get_balance(self):
        return {}

    async def watch_orders(self):
        return
        # noinspection PyUnreachableCode
        yield  # pragma: no cover — makes this an async generator

    def get_capabilities(self) -> ExchangeCapabilities:
        return self.capabilities


class _CompletePort:
    """Object carrying the full PositionPort surface."""

    async def fetch_raw_positions(self, symbol):
        return []

    async def get_volume_quantizer(self, symbol):
        return lambda v: int(v)

    async def close_leg(self, symbol, leg_id, volume, coid):
        pass

    async def reject_out_of_range(self, envelope, qty):
        pass

    async def place_leg(self, envelope, qty):
        return []

    async def amend_bracket(self, symbol, leg_id, *, side, tp_price,
                            sl_price, trail_offset, coid):
        pass


# === Conforming baseline ===


def __test_conforming_plugin_is_clean__():
    """The baseline plugin produces zero errors and zero warnings."""
    errors, warnings = validate_plugin_contract(_ConformingPlugin())
    assert errors == []
    assert warnings == []


# === Capability declaration ===


def __test_bool_capability_field_rejected__():
    """A bool capability value (classic new-author mistake) is an error."""
    class _BoolCaps(_ConformingPlugin):
        capabilities = _caps()

    plugin = _BoolCaps()
    # Dataclasses don't validate field types — smuggle the bool in the same
    # way an untyped call site would.
    object.__setattr__(plugin.capabilities, 'stop_order', True)
    errors, _ = validate_plugin_contract(plugin)
    assert any('stop_order' in e and 'CapabilityLevel' in e for e in errors)


def __test_idempotency_unsupported_rejected__():
    """idempotency=UNSUPPORTED must refuse live trading."""
    class _NoIdem(_ConformingPlugin):
        capabilities = _caps(idempotency=CapabilityLevel.UNSUPPORTED)

    errors, _ = validate_plugin_contract(_NoIdem())
    assert any('idempotency' in e for e in errors)


def __test_watch_orders_declared_but_not_overridden__():
    """A supported watch_orders declaration needs the method overridden."""
    class _NoStream(_ConformingPlugin):
        watch_orders = BrokerPlugin.watch_orders

    errors, _ = validate_plugin_contract(_NoStream())
    assert any('watch_orders' in e for e in errors)


def __test_missing_watch_orders_only_warns_when_undeclared__():
    """UNSUPPORTED watch_orders without override is legal but warned about."""
    class _PollOnly(_ConformingPlugin):
        capabilities = _caps(watch_orders=CapabilityLevel.UNSUPPORTED)
        watch_orders = BrokerPlugin.watch_orders

    errors, warnings = validate_plugin_contract(_PollOnly())
    assert errors == []
    assert any('disappearance' in w for w in warnings)


def __test_amend_declared_without_modify_override__():
    """NATIVE/PARTIAL_NATIVE amend_order needs a modify_* override."""
    class _AmendClaim(_ConformingPlugin):
        capabilities = _caps(amend_order=CapabilityLevel.PARTIAL_NATIVE)

    errors, _ = validate_plugin_contract(_AmendClaim())
    assert any('amend_order' in e for e in errors)

    class _AmendWired(_AmendClaim):
        async def modify_exit(self, old, new):
            return []

    errors, _ = validate_plugin_contract(_AmendWired())
    assert errors == []


# === Override pairs ===


def __test_residual_override_requires_cancel_ref__():
    """Overriding the residual enumerator without cancel_broker_order_ref fails."""
    class _ResidualOnly(_ConformingPlugin):
        def get_residual_orders_after_bracket_attach_reject(self, context):
            return ["broker-ref-1"]

    errors, _ = validate_plugin_contract(_ResidualOnly())
    assert any('cancel_broker_order_ref' in e for e in errors)

    class _ResidualPair(_ResidualOnly):
        async def cancel_broker_order_ref(self, ref):
            pass

    errors, _ = validate_plugin_contract(_ResidualPair())
    assert errors == []


def __test_default_cancel_with_outcome_warns__():
    """Keeping the UNKNOWN-only cancel outcome default is a warning."""
    class _NoOutcome(_ConformingPlugin):
        execute_cancel_with_outcome = BrokerPlugin.execute_cancel_with_outcome

    errors, warnings = validate_plugin_contract(_NoOutcome())
    assert errors == []
    assert any('execute_cancel_with_outcome' in w for w in warnings)


# === PositionPort surface ===


def __test_position_port_surface_checked__():
    """A partial PositionPort object is rejected with the missing methods named."""
    class _HalfPort:
        async def fetch_raw_positions(self, symbol):
            return []

    plugin = _ConformingPlugin()
    plugin.position_port = _HalfPort()
    errors, _ = validate_plugin_contract(plugin)
    assert any('close_leg' in e and 'amend_bracket' in e for e in errors)

    plugin.position_port = _CompletePort()
    errors, _ = validate_plugin_contract(plugin)
    assert errors == []


# === Lifecycle ===


def __test_account_id_required_on_broker_path__():
    """require_account_id=True rejects the "default" sentinel, and only that."""
    plugin = _ConformingPlugin()

    errors, _ = validate_plugin_contract(plugin, require_account_id=True)
    assert any('account_id' in e for e in errors)

    errors, _ = validate_plugin_contract(plugin, require_account_id=False)
    assert errors == []

    plugin._account_id = "testbroker-demo-123"
    errors, _ = validate_plugin_contract(plugin, require_account_id=True)
    assert errors == []
