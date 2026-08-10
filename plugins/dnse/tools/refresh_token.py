#!/usr/bin/env python3
"""Mint a DNSE trading token into the state file the plugin reads.

The plugin is a pure CONSUMER of ``workdir/state/dnse_trading_token.json``; this is
the ONLY producer. A trading token is valid ~8h and self-invalidating (requesting a
new OTP kills the previous code), so run this once each trading morning — one daily
cron at 08:00 ICT covers the whole session (morning + afternoon, expires ~16:00).

Modes
-----
* ``refresh_token.py``            auto (cron): send an email OTP, read the newest DNSE
                                  OTP from Gmail (IMAP), create the token, write it.
* ``refresh_token.py --otp CODE`` manual: you read the code yourself and pass it in
                                  (no Gmail creds needed — the reliable fallback).
* ``refresh_token.py --send``     just send the OTP email, then exit (read it, then
                                  re-run with ``--otp CODE``).

Auto mode needs a Gmail **app password** (not your login password) in the environment:
``DNSE_GMAIL_USER`` + ``DNSE_GMAIL_APP_PASSWORD`` (optionally ``DNSE_OTP_FROM`` to
narrow the sender, default ``dnse``).

Security: the state file is order-placement authority. It is written ``0600`` under
``workdir/state/`` (gitignored) via an atomic temp+rename; the token is never printed.
"""
from __future__ import annotations

import argparse
import email
import email.utils
import imaplib
import json
import os
import re
import sys
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path

# The minter reuses the plugin's version-pinned, TLS-verifying client for signing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # plugins/dnse
from pynecore_dnse.client import DNSEClient  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO / "workdir/config/plugins/dnse_broker.toml"
DEFAULT_STATE = REPO / "workdir/state/dnse_trading_token.json"
_SIX_DIGITS = re.compile(r"\b(\d{6})\b")


def load_credentials(config_path: Path) -> tuple[str, str]:
    if not config_path.exists():
        sys.exit(f"config not found: {config_path} (pass --config)")
    cfg = tomllib.loads(config_path.read_text())
    api_key, api_secret = cfg.get("api_key"), cfg.get("api_secret")
    if not api_key or not api_secret:
        sys.exit(f"{config_path} is missing api_key / api_secret")
    return api_key, api_secret


def write_token(state_path: Path, token: str) -> None:
    """Atomically write the state file the plugin reads (temp + os.replace, 0600)."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"trading_token": token,
                          "minted_at": int(time.time()),
                          "otp_type": "email_otp"})
    tmp = state_path.with_name(state_path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, payload.encode())
    finally:
        os.close(fd)
    os.replace(tmp, state_path)          # atomic on the same filesystem
    os.chmod(state_path, 0o600)


def send_otp(client: DNSEClient) -> float:
    """Send the email OTP; return the send timestamp (to reject older codes)."""
    sent_at = time.time()
    status, body = client.send_email_otp()
    if status not in (200, 201):
        sys.exit(f"send_email_otp failed: {status} {body}")
    return sent_at


def _message_text(message: email.message.Message) -> str:
    """Best-effort plaintext of an email (text/plain, else de-tagged text/html)."""
    parts = message.walk() if message.is_multipart() else [message]
    chunks = []
    for part in parts:
        if part.get_content_type() in ("text/plain", "text/html"):
            payload = part.get_payload(decode=True) or b""
            chunks.append(payload.decode(part.get_content_charset() or "utf-8", "replace"))
    text = "\n".join(chunks)
    return re.sub(r"<[^>]+>", " ", text)  # strip any HTML tags


def _extract_otp(text: str) -> str | None:
    """The 6-digit code, preferring one that follows an OTP/code/mã keyword."""
    keyed = re.search(r"(?:otp|code|m[aã]|passcode)[^0-9]{0,20}(\d{6})", text, re.I)
    if keyed:
        return keyed.group(1)
    loose = _SIX_DIGITS.search(text)
    return loose.group(1) if loose else None


def read_otp_from_gmail(after_ts: float, *, timeout: int = 180, poll: int = 10) -> str:
    """Poll Gmail (IMAP) for the newest DNSE OTP that arrived AFTER ``after_ts``."""
    user = os.environ.get("DNSE_GMAIL_USER")
    app_pw = os.environ.get("DNSE_GMAIL_APP_PASSWORD")
    sender = os.environ.get("DNSE_OTP_FROM", "dnse")
    if not user or not app_pw:
        sys.exit("auto mode needs DNSE_GMAIL_USER + DNSE_GMAIL_APP_PASSWORD (a Gmail "
                 "app password). Or use manual mode: --otp <code>.")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with imaplib.IMAP4_SSL("imap.gmail.com") as imap:
                imap.login(user, app_pw)
                imap.select("INBOX")
                _typ, data = imap.search(None, f'(FROM "{sender}")')
                for msg_id in reversed((data[0] or b"").split()[-20:]):  # newest first
                    _typ, raw = imap.fetch(msg_id, "(RFC822)")
                    if not raw or not raw[0]:
                        continue
                    message = email.message_from_bytes(raw[0][1])
                    date = email.utils.parsedate_to_datetime(message.get("Date", ""))
                    if date and date.timestamp() < after_ts - 5:  # older than our send
                        continue
                    code = _extract_otp(_message_text(message))
                    if code:
                        return code
        except imaplib.IMAP4.error as error:
            sys.exit(f"Gmail IMAP error: {error}")
        time.sleep(poll)
    sys.exit("no DNSE OTP arrived in time — retry, or use manual mode (--otp <code>).")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mint a DNSE trading token.")
    parser.add_argument("--otp", metavar="CODE", help="OTP you read yourself (manual mode)")
    parser.add_argument("--send", action="store_true", help="send the OTP email, then exit")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    args = parser.parse_args(argv)

    if args.otp and args.send:
        parser.error("--send and --otp are mutually exclusive")

    client = DNSEClient(*load_credentials(args.config))

    if args.send:  # send-only leg
        send_otp(client)
        print("OTP email sent — read it, then re-run: refresh_token.py --otp <code>")
        return 0

    if args.otp:  # manual leg (OTP already delivered)
        code = args.otp
    else:          # auto leg
        sent_at = send_otp(client)
        print("OTP email sent; reading the newest DNSE OTP from Gmail…")
        code = read_otp_from_gmail(sent_at)

    status, body = client.create_trading_token("email_otp", code)
    if status not in (200, 201) or not isinstance(body, dict) or not body.get("tradingToken"):
        sys.exit(f"create_trading_token failed: {status} {body}")
    write_token(args.state, body["tradingToken"])
    now = datetime.now(timezone.utc).astimezone()
    print(f"✓ token minted {now:%Y-%m-%d %H:%M %Z} -> {args.state} "
          f"(valid ~8h; the plugin picks it up on its next read)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
