"""Authoritative coverage matrix for the offline broker conformance lab."""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys


@dataclass(frozen=True)
class Requirement:
    """One audited broker failure family and its current lab evidence."""

    key: str
    broker: str
    priority: str
    title: str
    status: str
    evidence: tuple[str, ...] = ()
    note: str = ""


REQUIREMENTS = (
    Requirement(
        "common-market-roundtrip",
        "All",
        "P1",
        "Market long/short open-close and repeated rounds",
        "covered",
        (
            "core:market-entry-fill",
            "core:keyed-close-does-not-repeat-after-restart",
            "core:market-long-short-open-close-repeated-rounds",
        ),
    ),
    Requirement(
        "common-order-lifecycle",
        "All",
        "P0",
        "Limit/stop create, amend, cancel and cancel-all",
        "covered",
        (
            "core:entry-amend-replaces-one-physical-order",
            "core:cancel-ack-without-push-is-terminal",
            "core:cancel-all-terminalizes-every-active-order",
        ),
    ),
    Requirement(
        "common-partial-close",
        "All",
        "P0",
        "Partial and percentage close without residual or over-close errors",
        "covered",
        (
            "core:partial-close-preserves-exact-residual",
            "core:over-close-is-capped-at-live-position",
            "core:percentage-close-preserves-derived-residual",
        ),
    ),
    Requirement(
        "common-reversal",
        "All",
        "P0",
        "Long-short and short-long reversal",
        "covered",
        (
            "core:long-to-short-reversal-has-exact-economic-size",
            "core:short-to-long-reversal-has-exact-economic-size",
        ),
    ),
    Requirement(
        "common-restart",
        "All",
        "P0",
        "Restart with working order or position",
        "covered",
        (
            "core:restart-adopts-working-order",
            "core:completed-intent-does-not-resurrect-after-restart",
            "core:per-entry-hedge-reconstruction-survives-restart",
        ),
    ),
    Requirement(
        "common-same-id-recreate",
        "All",
        "P0",
        "Same-ID recreate after cancel",
        "covered",
        ("core:same-id-cancel-then-recreate-uses-new-physical-order",),
    ),
    Requirement(
        "common-event-ordering",
        "All",
        "P0",
        "Duplicate, delayed, reordered or missing ACK/PUSH",
        "covered",
        (
            "core:duplicate-fill-is-idempotent",
            "core:reordered-partial-fills-are-idempotent",
            "core:cancel-ack-without-push-is-terminal",
            "ctrader:ctrader-missed-fill-push-is-recovered-once-from-reconcile-and-deal-history",
        ),
    ),
    Requirement(
        "common-rejection-observability",
        "All",
        "P0",
        "Bounded rejection and below-minimum skip observability",
        "covered",
        (
            "core:bounded-entry-rejection",
            "core:minimum-below-order-is-observable-and-bounded",
        ),
    ),
    Requirement(
        "common-shutdown-lifecycle",
        "All",
        "P1",
        "Shutdown, pending cancel and subprocess lifecycle",
        "covered",
        (
            "subprocess:real-pyne-run-clean-lifecycle",
            "subprocess:real-pyne-run-shutdown-timeout",
            "subprocess:real-pyne-run-startup-exception-closes-store",
        ),
    ),
    Requirement(
        "common-credential-redaction",
        "All",
        "P0",
        "Credential redaction in CLI failures",
        "covered",
        ("subprocess:real-pyne-run-redacts-credential-from-unexpected-traceback",),
    ),
    Requirement(
        "bybit-stop-limit",
        "Bybit",
        "P0",
        "Dormant and already-triggered stop-limit in both directions",
        "covered",
        (
            "bybit:bybit-buy-stop-limit-stays-dormant-until-rising-trigger",
            "bybit:bybit-sell-stop-limit-stays-dormant-until-falling-trigger",
            "bybit:bybit-already-crossed-buy-stop-limit-drops-trigger",
            "bybit:bybit-already-crossed-sell-stop-limit-drops-trigger",
        ),
    ),
    Requirement(
        "bybit-global-bracket",
        "Bybit",
        "P0",
        "Global bracket over two entries",
        "covered",
        ("bybit:bybit-two-entry-global-bracket-creates-four-physical-legs",),
    ),
    Requirement(
        "bybit-bracket-lifecycle",
        "Bybit",
        "P0",
        "Bracket create, amend, cancel and reduce-only sibling sweep",
        "covered",
        (
            "bybit:bybit-bracket-amend-and-cancel-retains-no-orphan-leg",
            "bybit:bybit-filled-bracket-leg-cancels-reduce-only-sibling",
        ),
    ),
    Requirement(
        "bybit-concurrent-ownership",
        "Bybit",
        "P0",
        "Concurrent same-symbol run ownership and restart",
        "covered",
        ("bybit:bybit-concurrent-runs-restart-and-close-only-owned-exposure",),
    ),
    Requirement(
        "bybit-oca",
        "Bybit",
        "P1",
        "strategy.order OCA cancel and reduce",
        "covered",
        (
            "bybit:bybit-strategy-order-oca-cancel-sweeps-sibling-once",
            "bybit:bybit-strategy-order-oca-reduce-amends-sibling-once",
        ),
    ),
    Requirement(
        "bybit-product-mapping",
        "Bybit",
        "P1",
        "Linear, spot and inverse request mapping and quantity grid",
        "covered",
        (
            "bybit:bybit-market-entry-uses-real-transport-shape",
            "bybit:bybit-spot-market-entry-uses-base-coin-denomination",
            "bybit:bybit-inverse-market-entry-uses-contract-denomination-and-anchor",
            "bybit:bybit-quantity-grid-residual-is-not-under-rounded",
        ),
    ),
    Requirement(
        "capital-marketable-whole-row",
        "Capital.com",
        "P0",
        "Marketable whole-row limit and stop exit",
        "covered",
        (
            "capital:capitalcom-marketable-limit-exit-uses-one-immediate-close",
            "capital:capitalcom-marketable-stop-exit-uses-one-immediate-close",
        ),
    ),
    Requirement(
        "capital-partial-marketable",
        "Capital.com",
        "P0",
        "Partial marketable exit preserves residual",
        "covered",
        ("capital:capitalcom-partial-marketable-exit-uses-exact-opposite-post",),
    ),
    Requirement(
        "capital-durable-activity-replay",
        "Capital.com",
        "P0",
        "Activity fill replay across restart",
        "covered",
        (
            "capital:capitalcom-activity-replay-is-deduplicated",
            "capital:capitalcom-activity-replay-after-restart-is-durable",
        ),
    ),
    Requirement(
        "capital-shared-ownership",
        "Capital.com",
        "P0",
        "Shared netting account with two run owners",
        "covered",
        ("capital:capitalcom-shared-netting-runs-adopt-and-close-only-own-ledger",),
    ),
    Requirement(
        "capital-bracket-lifecycle",
        "Capital.com",
        "P0",
        "Bracket and trailing attach, amend, clear, cancel and restart",
        "covered",
        (
            "capital:capitalcom-position-bracket-attach-replace-shape",
            "capital:capitalcom-bracket-cancel-stays-cleared-after-restart",
            "capital:capitalcom-trailing-attach-amend-clear-survives-restart",
        ),
    ),
    Requirement(
        "capital-working-order-reversal",
        "Capital.com",
        "P1",
        "Working-order lifecycle and netting reversal",
        "covered",
        (
            "capital:capitalcom-restart-does-not-repeat-working-order",
            "capital:capitalcom-netting-reversal-activity-closes-and-opens-exactly-once",
        ),
    ),
    Requirement(
        "capital-pagination-fallback",
        "Capital.com",
        "P1",
        "Pagination, gaps and WS/REST fallback",
        "covered",
        (
            "capital:capitalcom-price-pagination-has-no-overlap-or-gap",
            "capital:capitalcom-missing-activity-falls-back-to-rest-position-snapshot",
        ),
    ),
    Requirement(
        "ctrader-adopted-cancel",
        "cTrader",
        "P0",
        "Adopted working-order cancel without PUSH",
        "covered",
        ("ctrader:ctrader-cancel-ack-without-push-terminalizes",),
    ),
    Requirement(
        "ctrader-hedged-restart-close",
        "cTrader",
        "P0",
        "Multiple HEDGED legs restart and keyed close",
        "covered",
        (
            "ctrader:ctrader-hedged-two-leg-restart-keyed-close-targets-only-one-position",
        ),
    ),
    Requirement(
        "ctrader-concurrent-ownership",
        "cTrader",
        "P0",
        "Concurrent same/opposite-direction run ownership",
        "covered",
        (
            "ctrader:ctrader-concurrent-opposite-runs-remain-owned-while-account-net-is-zero",
        ),
    ),
    Requirement(
        "ctrader-push-ordering",
        "cTrader",
        "P0",
        "Deal/fill PUSH, snapshot and reconcile ordering",
        "covered",
        (
            "ctrader:ctrader-duplicate-fill-push-and-restart-snapshot-are-exactly-once",
            "ctrader:ctrader-missed-fill-push-is-recovered-once-from-reconcile-and-deal-history",
        ),
    ),
    Requirement(
        "ctrader-partial-reversal",
        "cTrader",
        "P0",
        "Partial close and reversal in HEDGED mode",
        "covered",
        (
            "ctrader:ctrader-hedged-partial-close-preserves-target-leg-residual",
            "ctrader:ctrader-hedged-reversal-closes-old-leg-before-opening-residual",
        ),
    ),
    Requirement(
        "ctrader-startup-retry",
        "cTrader",
        "P1",
        "Startup retry and backoff fault injection",
        "covered",
        (
            "subprocess:real-pyne-run-transient-connect-retry",
            "subprocess:real-pyne-run-permanent-connect-fails-fast",
            "ctrader:ctrader-real-connect-boundary-retries-transient-fault-once",
            "ctrader:ctrader-real-connect-boundary-fails-fast-on-permanent-fault",
        ),
    ),
    Requirement(
        "ctrader-reconnect-disposition",
        "cTrader",
        "P0",
        "Reconnect before/after write and unknown disposition",
        "covered",
        (
            "ctrader:ctrader-pre-write-disconnect-retries-once-without-duplicate-order",
            "ctrader:ctrader-post-write-disconnect-parks-disposition-without-redispatch",
        ),
    ),
    Requirement(
        "live-auth-identity",
        "All",
        "live",
        "Authentication and account identity",
        "live-only",
    ),
    Requirement(
        "live-rate-limit", "All", "live", "Real rate limits and throttling", "live-only"
    ),
    Requirement(
        "live-network-soak",
        "All",
        "live",
        "Handshake, latency and reconnect soak",
        "live-only",
    ),
    Requirement(
        "live-matching",
        "All",
        "live",
        "Matching, spread, slippage and partial-fill timing",
        "live-only",
    ),
    Requirement(
        "live-instrument-rules",
        "All",
        "probe",
        "Current instrument rules and tradability",
        "live-only",
    ),
    Requirement(
        "live-bybit-spot-fees",
        "Bybit",
        "probe",
        "Spot fee currency and balance update",
        "live-only",
    ),
    Requirement(
        "live-bybit-spent-coid",
        "Bybit",
        "probe",
        "Spent orderLinkId venue behavior",
        "live-only",
    ),
    Requirement(
        "live-bybit-bracket-autocancel",
        "Bybit",
        "probe",
        "Native bracket auto-cancel",
        "live-only",
    ),
    Requirement(
        "live-capital-close-semantics",
        "Capital.com",
        "probe",
        "Position close endpoint semantics",
        "live-only",
    ),
    Requirement(
        "live-capital-protection-rules",
        "Capital.com",
        "probe",
        "Native protection distance rules",
        "live-only",
    ),
    Requirement(
        "live-capital-session",
        "Capital.com",
        "live",
        "Multiple sessions and throttling",
        "live-only",
    ),
    Requirement(
        "live-capital-market-data",
        "Capital.com",
        "live",
        "Market status and weekend data",
        "live-only",
    ),
    Requirement(
        "live-ctrader-account-mode",
        "cTrader",
        "probe",
        "HEDGED flag and opaque venue IDs",
        "live-only",
    ),
    Requirement(
        "live-ctrader-reset",
        "cTrader",
        "live",
        "Real connection reset timing",
        "live-only",
    ),
    Requirement(
        "live-ohlcv-continuity",
        "All",
        "live",
        "OHLCV continuity and venue gaps",
        "live-only",
    ),
)


SUITES = {
    "core": Path(__file__).with_name("suite.py"),
    "subprocess": Path(__file__).with_name("subprocess_suite.py"),
    "bybit": Path(__file__).resolve().parents[2]
    / "plugins/pynecore-bybit/broker_lab/suite.py",
    "capital": Path(__file__).resolve().parents[2]
    / "plugins/pynecore-capitalcom/broker_lab/suite.py",
    "ctrader": Path(__file__).resolve().parents[2]
    / "plugins/pynecore-ctrader/broker_lab/suite.py",
}


def _scenario_names(path: Path) -> set[str]:
    spec = importlib.util.spec_from_file_location(
        f"broker_lab_coverage_{path.parent.parent.name}", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load broker-lab suite: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return {scenario.name for scenario in module.build_suite(mode="extended", seed=0)}


def validate(requirements: Sequence[Requirement] = REQUIREMENTS) -> list[str]:
    """Return structural and stale-evidence errors for the coverage matrix."""
    errors: list[str] = []
    keys: set[str] = set()
    loaded: dict[str, set[str]] = {}
    valid_statuses = {"covered", "partial", "missing", "live-only"}
    broker_suites = {
        "Bybit": {"bybit", "core", "subprocess"},
        "Capital.com": {"capital", "core", "subprocess"},
        "cTrader": {"ctrader", "core", "subprocess"},
    }
    for requirement in requirements:
        if requirement.key in keys:
            errors.append(f"duplicate requirement key: {requirement.key}")
        keys.add(requirement.key)
        if requirement.status not in valid_statuses:
            errors.append(f"{requirement.key}: invalid status {requirement.status!r}")
        if requirement.status in ("covered", "partial") and not requirement.evidence:
            errors.append(
                f"{requirement.key}: {requirement.status} requires scenario evidence"
            )
        if requirement.status in ("missing", "live-only") and requirement.evidence:
            errors.append(
                f"{requirement.key}: {requirement.status} must not claim scenario evidence"
            )
        for reference in requirement.evidence:
            suite, separator, scenario = reference.partition(":")
            if not separator or suite not in SUITES:
                errors.append(
                    f"{requirement.key}: invalid evidence reference {reference!r}"
                )
                continue
            allowed = broker_suites.get(requirement.broker)
            if allowed is not None and suite not in allowed:
                errors.append(
                    f"{requirement.key}: {suite!r} evidence belongs to another broker"
                )
                continue
            if suite not in loaded:
                loaded[suite] = _scenario_names(SUITES[suite])
            if scenario not in loaded[suite]:
                errors.append(
                    f"{requirement.key}: missing scenario evidence {reference!r}"
                )
    return errors


def _print_markdown() -> None:
    print("| Key | Broker | Priority | Status | Failure family | Evidence / note |")
    print("|---|---|---:|---|---|---|")
    for item in REQUIREMENTS:
        detail = ", ".join(f"`{value}`" for value in item.evidence)
        if item.note:
            detail = f"{detail}; {item.note}" if detail else item.note
        print(
            f"| `{item.key}` | {item.broker} | {item.priority} | {item.status} | "
            f"{item.title} | {detail} |"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="validate scenario evidence"
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="also fail while any offline requirement is not covered",
    )
    args = parser.parse_args(argv)
    errors = validate() if args.check or args.require_complete else []
    if args.require_complete:
        errors.extend(
            f"{item.key}: status is {item.status}"
            for item in REQUIREMENTS
            if item.status not in ("covered", "live-only")
        )
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    _print_markdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
