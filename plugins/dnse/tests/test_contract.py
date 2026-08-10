"""Contract probe (step-0 gate) — the DNSE broker satisfies the enforceable parts of the
``BrokerPlugin`` authoring contract, the same probe the engine runs at startup
(``pynecore/cli/commands/run.py`` -> ``validate_plugin_contract``). A violation here
would otherwise only surface as a fail-fast error the first time you go ``--broker``.
"""
from pynecore.core.broker.validation import validate_plugin_contract
from pynecore_dnse.broker import DNSEBroker, DNSEBrokerConfig


def _broker() -> DNSEBroker:
    cfg = DNSEBrokerConfig(api_key="k", api_secret="s", account_no="0001672126")
    return DNSEBroker(symbol="VN30F1M", timeframe="5", config=cfg)


def __test_plugin_contract_has_no_errors__():
    errors, _warnings = validate_plugin_contract(_broker())
    assert errors == [], f"BrokerPlugin contract violations: {errors}"


def __test_plugin_contract_has_no_warnings__():
    # Currently clean; pin it so a future contract warning is surfaced for review.
    _errors, warnings = validate_plugin_contract(_broker())
    assert warnings == [], f"BrokerPlugin contract warnings: {warnings}"
