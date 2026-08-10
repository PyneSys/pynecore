"""Probe: does DNSE accept a RESTING stop order for VN30F1M derivatives?

Places candidate stop orders 3% AWAY from market (so a genuine stop RESTS,
fill 0) and reads each back. Tries the full matrix of orderCategory × orderType
× trigger-field because the real combination is unknown and DNSE accepts bogus
inputs with a 200 — a successful POST proves nothing; only a readback showing
`fillQuantity == 0` with a live status proves a resting stop.

SAFETY: every placed order is cancelled AND any resulting position flattened in
a finally block, even on error. A marketable variant WILL fill and open a real
1-contract position; this script must never leave naked exposure.

Run:  needs an OPEN session (09:00-11:30 / 13:00-14:45 ICT) and a fresh trading
      token. Pass the token as argv[1], or set it in dnse_broker.toml and let
      the harness read it.

    python probe_conditional_order.py <TRADING_TOKEN>
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    from dnse_client import client_from_env
except ImportError:                      # running from repo root
    sys.path.insert(0, "workdir/dnse_lab")
    from dnse_client import client_from_env

MARKET = "DERIVATIVE"

# The field/type names are GUESSES to be discovered — not asserted. A buy order
# whose trigger is 3% ABOVE market must REST if it is a real stop; if it fills,
# that variant is not a resting stop (or not a stop at all).
ORDER_CATEGORIES = ("NORMAL", "CONDITIONAL", "STOP")
ORDER_TYPES = ("STO", "MTL", "MOK", "LO")          # LO kept only as a control
TRIGGER_FIELDS = ("stopPrice", "triggerPrice", "activationPrice", "conditionPrice")


def resolve(client):
    a = client.get_accounts()[1]["accounts"][0]["id"]
    inst = client.request("GET", "/instruments", query={"limit": 200})[1]["data"]
    contract = next(r["symbol"] for r in inst if r["symbolType"] == "VN30F1M")
    now = int(time.time())
    last = float(client.get_ohlc(MARKET, {"symbol": "VN30F1M", "resolution": "1",
                 "from": now - 1800, "to": now})[1]["c"][-1])
    loan = client.get_loan_packages(a, MARKET)[1]["loanPackages"][0]["id"]
    return a, contract, last, loan


def main() -> None:
    token = sys.argv[1] if len(sys.argv) > 1 else None
    if not token:
        import re
        cfg = Path("workdir/config/plugins/dnse_broker.toml")
        m = re.search(r'trading_token\s*=\s*"([^"]*)"', cfg.read_text()) if cfg.exists() else None
        token = m.group(1) if m else None
    if not token or token.startswith("fake"):
        raise SystemExit("need a real trading token (argv[1] or dnse_broker.toml)")

    client = client_from_env()
    account, contract, last, loan = resolve(client)
    stop_px = round(last * 1.03, 1)          # 3% ABOVE -> a real BUY stop rests
    print(f"account …{account[-4:]}  contract={contract}  last={last}  "
          f"buy-stop trigger={stop_px} (3% above)\n")

    placed_ids: list[str] = []
    results = []
    try:
        for category in ORDER_CATEGORIES:
            for otype in ORDER_TYPES:
                # LO is the control: a buy LO 3% above market is marketable; a
                # fill there says NOTHING about stop support.
                field_choices = ("price",) if otype == "LO" else TRIGGER_FIELDS
                for field in field_choices:
                    payload = {"accountNo": account, "symbol": contract,
                               "side": "NB", "orderType": otype, "quantity": 1,
                               "loanPackageId": loan, field: stop_px}
                    st, r = client.post_order(MARKET, payload, token,
                                              order_category=category)
                    line = {"category": category, "type": otype, "field": field,
                            "http": st}
                    if st in (200, 201) and isinstance(r, dict) and r.get("id"):
                        oid = str(r["id"])
                        placed_ids.append(oid)
                        time.sleep(1.5)
                        d = client.get_order_detail(account, oid, MARKET)[1]
                        line["status"] = d.get("orderStatus")
                        line["fill"] = d.get("fillQuantity")
                        line["rested"] = (str(d.get("orderStatus", "")).lower()
                                          in ("new", "pending", "pendingnew")
                                          and float(d.get("fillQuantity") or 0) == 0
                                          and otype != "LO")
                    else:
                        line["error"] = (r.get("code"), str(r.get("message", ""))[:70]) \
                            if isinstance(r, dict) else str(r)[:70]
                    results.append(line)
                    print(f"  {category:11} {otype:4} {field:16} -> "
                          f"{json.dumps({k: v for k, v in line.items() if k not in ('category','type','field')}, ensure_ascii=False)[:150]}")
    finally:
        # cancel everything, then flatten any position a marketable variant opened
        for oid in placed_ids:
            try:
                client.cancel_order(account, oid, MARKET, token)
            except Exception as e:
                print(f"  cleanup cancel {oid} failed: {e}")
        time.sleep(2)
        for pos in [p for p in (client.get_positions(account, MARKET)[1].get("positions") or [])
                    if p.get("openQuantity")]:
            try:
                client.request("POST", f"/accounts/positions/{pos['id']}/close",
                               query={"marketType": MARKET},
                               headers={"trading-token": token})
                print(f"  FLATTENED residual {pos['side']} {pos['openQuantity']}")
            except Exception as e:
                print(f"  cleanup flatten failed: {e}")
        time.sleep(2)
        p = [x for x in (client.get_positions(account, MARKET)[1].get("positions") or [])
             if x.get("openQuantity")]
        o = [x for x in (client.get_orders(account, MARKET, None)[1].get("orders") or [])
             if x.get("orderStatus") not in ("Canceled", "Cancelled", "Filled", "Rejected", "Expired")]
        print(f"\n  CLEANUP: positions={len(p)} live_orders={len(o)}")

    resting = [r for r in results if r.get("rested")]
    print("\n=== VERDICT ===")
    if resting:
        print("  DNSE ACCEPTS A RESTING STOP. Wire this into _place:")
        for r in resting:
            print(f"    category={r['category']} orderType={r['type']} field={r['field']}")
    else:
        accepted = [r for r in results if r.get("http") in (200, 201)]
        print("  NO variant rested (fill 0, live status).")
        print(f"  {len(accepted)} POSTs were accepted but none rested -> UNKNOWN,")
        print("  NOT proof of absence. Check DNSE docs/support for the real stop")
        print("  order payload before concluding stops are unsupported.")


if __name__ == "__main__":
    main()
