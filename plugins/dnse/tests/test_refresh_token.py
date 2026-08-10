"""Tests for the DNSE trading-token minter — no live network, no real Gmail, tmp FS only."""
import json
import os
import sys
from pathlib import Path

import pytest

# tools/ is not a package — put it on the path to import the minter.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import refresh_token as rt  # noqa: E402


def _config(tmp_path):
    cfg = tmp_path / "dnse.toml"
    cfg.write_text('api_key = "k"\napi_secret = "s"\n')
    return cfg


class _FakeClient:
    """Stand-in for DNSEClient: the OTP endpoints, no network."""
    def __init__(self, *a, **k):
        self.sent = 0

    def send_email_otp(self):
        self.sent += 1
        return (200, {})

    def create_trading_token(self, otp_type, passcode):
        assert otp_type == "email_otp", "the minter must request an email OTP"
        return (200, {"tradingToken": f"tok-{passcode}"})


def __test_write_token_is_atomic_and_private__(tmp_path):
    state = tmp_path / "state" / "dnse_trading_token.json"
    rt.write_token(state, "TKN-123")

    data = json.loads(state.read_text())
    assert data["trading_token"] == "TKN-123", "plugin reads the 'trading_token' key"
    assert isinstance(data["minted_at"], int)
    assert oct(os.stat(state).st_mode & 0o777) == "0o600", "token file must be 0600"
    assert not state.with_name(state.name + ".tmp").exists(), "no temp file left behind"


@pytest.mark.parametrize("text, want", [
    ("Mã OTP của bạn là 123456", "123456"),
    ("Your OTP code: 654321. Do not share.", "654321"),
    ("passcode 246810 expires soon", "246810"),
    ("order 111222 total 999999", "111222"),   # loose fallback: first 6-digit
    ("no six digit code here 12345", None),     # 5 digits -> no match
])
def __test_extract_otp__(text, want):
    assert rt._extract_otp(text) == want


def __test_message_text_strips_html__():
    import email
    msg = email.message_from_string(
        "Content-Type: text/html\n\n<p>Your code is <b>135790</b></p>")
    assert rt._extract_otp(rt._message_text(msg)) == "135790"


def __test_manual_mode_writes_token_without_sending__(tmp_path, monkeypatch):
    fake = _FakeClient()
    monkeypatch.setattr(rt, "DNSEClient", lambda *a, **k: fake)
    state = tmp_path / "dnse_trading_token.json"

    rc = rt.main(["--otp", "424242", "--config", str(_config(tmp_path)), "--state", str(state)])

    assert rc == 0
    assert json.loads(state.read_text())["trading_token"] == "tok-424242"
    assert fake.sent == 0, "manual --otp must NOT send a fresh OTP (that would invalidate it)"


def __test_auto_mode_sends_then_scrapes_gmail__(tmp_path, monkeypatch):
    fake = _FakeClient()
    monkeypatch.setattr(rt, "DNSEClient", lambda *a, **k: fake)
    monkeypatch.setattr(rt, "read_otp_from_gmail", lambda after_ts, **k: "777888")
    state = tmp_path / "s.json"

    rc = rt.main(["--config", str(_config(tmp_path)), "--state", str(state)])

    assert rc == 0
    assert fake.sent == 1, "auto mode sends the OTP before scraping"
    assert json.loads(state.read_text())["trading_token"] == "tok-777888"


def __test_send_and_otp_are_mutually_exclusive__(tmp_path, monkeypatch):
    monkeypatch.setattr(rt, "DNSEClient", lambda *a, **k: _FakeClient())
    with pytest.raises(SystemExit):
        rt.main(["--send", "--otp", "1", "--config", str(_config(tmp_path))])
