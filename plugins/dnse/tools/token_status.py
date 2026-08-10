#!/usr/bin/env python3
"""Check the DNSE trading-token status, and refresh it by hand if it isn't good.

Meant to be run manually at ~08:05 (just after the 08:00 cron) to answer one question:
**did the job leave us with a token that actually works?** It shows:

  * mint time + age vs the 8h TTL, and whether it was minted after 08:00 today,
  * the tail of the cron log (what the morning job actually did),
  * a LIVE liveness probe — a harmless cancel of a bogus order id. DNSE checks the
    trading-token header before it looks the order up, so an ``INVALID_TRADING_TOKEN``
    reply means the token is bad, while ANY other reply (not-found, session-closed, …)
    means the token was accepted, i.e. good.

If the token is missing / stale / rejected (or you pass ``--refresh``), it walks you
through a manual mint: it sends the email OTP, you read it and type the code, it writes
the new token to the file the plugin reads.

    .venv/bin/python plugins/dnse/tools/token_status.py [--refresh]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # tools/ (for refresh_token)
import refresh_token as rt  # noqa: E402
from pynecore_dnse import errors  # noqa: E402
from pynecore_dnse.client import DNSEClient  # noqa: E402

ICT = timezone(timedelta(hours=7))
TTL_HOURS = 8
_PROBE_ORDER_ID = "TOKENCHECK0000"  # a bogus id — a cancel of it never touches a real order


def read_state(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError):
        return None


def resolve_account(client: DNSEClient) -> str | None:
    status, body = client.get_accounts()
    if status == 200 and isinstance(body, dict):
        accounts = body.get("accounts") or []
        if accounts:
            return accounts[0].get("id")
    return None


def token_is_live(client: DNSEClient, account: str, token: str) -> tuple[bool, str]:
    """Probe the token with a harmless cancel of a bogus order id.

    Returns ``(accepted, reason)``. Only ``INVALID_TRADING_TOKEN`` / auth means the
    token is bad; every other reply means DNSE accepted it.
    """
    status, body = client.cancel_order(account, _PROBE_ORDER_ID, "DERIVATIVE",
                                       token, order_category="STOP")
    if status == 0:
        return False, f"could not reach DNSE ({errors.code_of(body) or 'network error'})"
    classified = errors.classify(status, body, is_write=True)
    if classified and classified.disposition in (errors.Disposition.AUTH_TOKEN,
                                                  errors.Disposition.AUTH):
        return False, f"DNSE rejected the token ({classified.code})"
    return True, f"accepted (probe -> {errors.code_of(body) or ('http ' + str(status))})"


def show_cron_log(state_path: Path) -> None:
    log = state_path.parent / "refresh_token.log"
    print(f"\ncron log ({log}):")
    if not log.exists():
        print("  (none yet — the 08:00 cron hasn't run, or it logs elsewhere)")
        return
    for line in log.read_text().splitlines()[-6:]:
        print(f"  {line}")


def interactive_refresh(client: DNSEClient, state_path: Path) -> bool:
    """Send an OTP, prompt for the code, mint + write. Returns True on success."""
    if not sys.stdin.isatty():
        print("  (not a terminal — run this yourself to refresh interactively)")
        return False
    print("\nSending an email OTP to your DNSE account…")
    rt.send_otp(client)
    code = input("Enter the OTP code from your email (blank to abort): ").strip()
    if not code:
        print("aborted.")
        return False
    status, body = client.create_trading_token("email_otp", code)
    if status not in (200, 201) or not isinstance(body, dict) or not body.get("tradingToken"):
        print(f"✗ mint failed: {status} {body}")
        return False
    rt.write_token(state_path, body["tradingToken"])
    print(f"✓ new token written to {state_path}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Check + manually refresh the DNSE trading token.")
    parser.add_argument("--config", type=Path, default=rt.DEFAULT_CONFIG)
    parser.add_argument("--state", type=Path, default=rt.DEFAULT_STATE)
    parser.add_argument("--refresh", action="store_true", help="refresh even if the token looks good")
    args = parser.parse_args()

    client = DNSEClient(*rt.load_credentials(args.config))
    now = datetime.now(ICT)
    print(f"=== DNSE token status @ {now:%Y-%m-%d %H:%M %Z} ===")

    state = read_state(args.state)
    good = False
    if not state or not state.get("trading_token"):
        print(f"token file:  MISSING or empty  ({args.state})")
    else:
        minted_at = state.get("minted_at")
        if minted_at:
            minted = datetime.fromtimestamp(minted_at, ICT)
            age_h = (time.time() - minted_at) / 3600
            within_ttl = age_h < TTL_HOURS
            after_8_today = minted.date() == now.date() and minted.hour >= 8
            print(f"minted:      {minted:%Y-%m-%d %H:%M %Z}  (age {age_h:.1f}h; "
                  f"TTL {TTL_HOURS}h -> {'within' if within_ttl else 'EXPIRED'})")
            print(f"fresh cron:  {'yes — minted after 08:00 today' if after_8_today else 'NO — not minted after 08:00 today'}")
        else:
            within_ttl = False
            print("minted:      (unknown — file has no minted_at)")

        account = resolve_account(client)
        if not account:
            good = within_ttl
            print("liveness:    (could not resolve account to probe — using the TTL heuristic)")
        else:
            live, why = token_is_live(client, account, state["trading_token"])
            good = live
            print(f"liveness:    {'GOOD — ' if live else 'BAD — '}{why}")

    show_cron_log(args.state)
    print(f"\nVERDICT: {'GOOD — the plugin can place orders' if good else 'NOT GOOD — refresh needed'}")

    if args.refresh or not good:
        if not sys.stdin.isatty():
            return 0 if good else 1
        default_yes = args.refresh or not good
        answer = input(f"\nRefresh the token now? [{'Y/n' if default_yes else 'y/N'}] ").strip().lower()
        if answer == "y" or (default_yes and answer == ""):
            interactive_refresh(client, args.state)
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
