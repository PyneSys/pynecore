"""A fake DNSE order API that reproduces the venue's MEASURED quirks (card #10).

Not a general simulator — it models exactly the behaviours observed live on 2026-08-13,
so the fixes for #19 and #20 can be verified end-to-end through the real
:class:`DNSEBroker` code path (``_place`` -> ``_cancel_one`` -> ``_cancel_took_effect`` ->
``_cancel_dependent_exits`` -> ``get_open_orders``) at any hour, without a trading session.

Modelled quirks, each traceable to a live observation:

* **Cancel is an ACK, not a completion** (#20). ``cancel_order`` answers ``200`` with the
  order object while ``orderStatus`` stays ``New`` for ``ack_lag`` subsequent reads, then
  flips to ``Canceled``. Live: three ``200 OK`` cancels on d9umsv21a4skcecmf4ag left it
  ``New`` for >12s.
* **No cascade** (#19). Cancelling an entry does NOT touch any other order; an exit leg
  survives as ``New``. Live: T4's entry went ``Canceled`` while X4 stayed ``New``.
* **Two books with different id shapes.** NORMAL ids are ints, conditional ids are
  strings — the plugin routes cancels by book, so the distinction matters.
* **A cancel against the wrong book 404s** with ``RESOURCE_NOT_FOUND``, which is how the
  plugin probes for the right one.
"""
from __future__ import annotations

import itertools


class FakeDNSEVenue:
    """Minimal stand-in for the DNSE client wrapper, with the measured quirks."""

    def __init__(self, *, ack_lag: int = 0):
        """:param ack_lag: how many reads AFTER a cancel still report ``New``."""
        self.ack_lag = ack_lag
        self.orders: dict[str, dict] = {}
        self._int_ids = itertools.count(400001)
        self._str_ids = itertools.count(1)
        self.calls: list[tuple] = []

    # --- helpers -------------------------------------------------------------
    def _new_id(self, category: str) -> str:
        return (str(next(self._int_ids)) if category == "NORMAL"
                else f"d9fake{next(self._str_ids):04d}")

    def _book_of(self, order_id: str) -> str | None:
        row = self.orders.get(order_id)
        return row["category"] if row else None

    def working_ids(self) -> list[str]:
        return [i for i, r in self.orders.items() if r["orderStatus"] == "New"]

    # --- the client surface the broker uses ----------------------------------
    def get_security_definition(self, *a, **k):
        return (200, [{"ceilingPrice": "2075.8", "floorPrice": "1804.2",
                       "securityGroupId": "FU"}])

    def get_loan_packages(self, *a, **k):
        return (200, {"loanPackages": [{"id": 42}]})

    def get_instruments(self, *a, **k):
        """Alias -> tradable contract, so ``resolve_contract`` works offline."""
        return (200, {"instruments": [
            {"symbol": "VN30F1M", "underlyingSymbol": "VN30F1M",
             "tradingSymbol": "41I1G8000"}]})

    def get_ohlc(self, market, params):
        return (200, {"t": [1], "o": [1912.0], "h": [1913.0],
                      "l": [1911.0], "c": [1912.0], "v": [100]})

    def post_order(self, account, market, payload, token, order_category="NORMAL"):
        self.calls.append(("post_order", order_category))
        oid = self._new_id(order_category)
        self.orders[oid] = {
            "id": oid, "symbol": payload["symbol"], "side": payload["side"],
            "quantity": payload["quantity"], "fillQuantity": 0,
            "price": payload.get("price"), "stopPrice": payload.get("stopPrice"),
            "orderStatus": "New", "category": order_category,
            "_cancel_pending": 0,
        }
        return (201, dict(self.orders[oid]))

    def cancel_order(self, account, order_id, market, token, order_category=None):
        self.calls.append(("cancel_order", order_id, order_category))
        row = self.orders.get(str(order_id))
        if row is None or (order_category and row["category"] != order_category):
            # probing the wrong book is how the plugin finds the right one
            return (404, {"status": 404, "code": "RESOURCE_NOT_FOUND",
                          "message": f"Order not found with id: {order_id}"})
        if row["orderStatus"] != "New":
            return (400, {"code": "CO-ORD-013", "message": "Order Is Done"})
        # ACK only: the status flips after ack_lag further reads (#20). NOTE: no cascade
        # to any other order (#19) — that is the venue behaviour, deliberately preserved.
        row["_cancel_pending"] = self.ack_lag + 1
        if row["_cancel_pending"] <= 1:
            row["orderStatus"] = "Canceled"
            row["_cancel_pending"] = 0
        return (200, dict(row))

    def get_order_detail(self, account, order_id, market, order_category=None,
                         dry_run=False):
        self.calls.append(("get_order_detail", order_id, order_category))
        row = self.orders.get(str(order_id))
        if row is None or (order_category and row["category"] != order_category):
            return (404, {"code": "RESOURCE_NOT_FOUND"})
        if row["_cancel_pending"] > 0:
            row["_cancel_pending"] -= 1
            if row["_cancel_pending"] == 0:
                row["orderStatus"] = "Canceled"
        return (200, dict(row))

    def get_orders(self, account, market, order_category=None, page_index=0,
                   page_size=100):
        self.calls.append(("get_orders", order_category))
        rows = [dict(r) for r in self.orders.values()
                if r["category"] == order_category]
        return (200, {"orders": rows})

    def get_positions(self, account, market):
        return (200, {"positions": []})
