"""Tests for the DNSE token-status tool — pure logic, no live network."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import token_status as ts  # noqa: E402


def __test_read_state_missing_valid_and_malformed__(tmp_path):
    assert ts.read_state(tmp_path / "nope.json") is None
    good = tmp_path / "s.json"
    good.write_text(json.dumps({"trading_token": "T", "minted_at": 1}))
    assert ts.read_state(good)["trading_token"] == "T"
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert ts.read_state(bad) is None


def __test_resolve_account__(fake_client):
    assert ts.resolve_account(
        fake_client(get_accounts=(200, {"accounts": [{"id": "0001672126"}]}))) == "0001672126"
    assert ts.resolve_account(fake_client(get_accounts=(200, {"accounts": []}))) is None
    assert ts.resolve_account(fake_client(get_accounts=(500, {}))) is None


@pytest.mark.parametrize("reply, accepted", [
    ((400, {"code": "INVALID_TRADING_TOKEN"}), False),                       # token rejected
    ((401, {}), False),                                                      # auth failure
    ((404, {"code": "RESOURCE_NOT_FOUND"}), True),                           # accepted, id not found
    ((400, {"code": "CANNOT_CANCEL_THE_ORDER_IN_THE_ATO_SESSION"}), True),   # accepted, session
    ((0, {}), False),                                                        # could not reach DNSE
])
def __test_token_is_live__(fake_client, reply, accepted):
    fake = fake_client(cancel_order=reply)
    live, _why = ts.token_is_live(fake, "ACC", "tok")
    assert live is accepted, f"{reply} -> live should be {accepted}"
