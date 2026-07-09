"""
Regression: the strategy order book must re-index a price level that was reused
after being removed mid-walk.

``PriceOrderBook`` keeps a sorted ``price_levels`` list (what the intrabar walk
iterates) alongside a ``price -> [orders]`` map. ``iter_orders`` snapshots the
level list, so a fill that removes a level still listed later in the snapshot
would, when the bucket map was a ``defaultdict(list)``, resurrect an empty
bucket on the stale read. That orphan key then shadowed the price in
``add_order`` (its guard checked the bucket map, not ``price_levels``), so a
later order reusing that exact price never got its level inserted -- the leg
silently vanished from the walk and only "filled" as a next-bar gap-through at
the wrong price. Observed in the wild ThinkTech strategy, where two same-day
bracket exits shared an identical take-profit price and the second one was
dropped, corrupting the equity path.
"""
from pynecore.lib.strategy import Order, PriceOrderBook


def __test_orderbook_reindexes_reused_level_after_midwalk_removal__():
    """A price reused after a mid-walk removal is registered as a walkable level."""
    ob = PriceOrderBook()
    o_low = Order("low", -1.0, limit=110.0)
    o_high = Order("high", -1.0, limit=120.0)
    ob.add_order(o_low)
    ob.add_order(o_high)
    assert ob.price_levels == [110.0, 120.0]

    # Walk the range and remove the higher-level order while positioned on the
    # lower one, so the generator's snapshot still lists 120.0 after its level
    # is gone. A defaultdict read at that stale level would resurrect an empty
    # 120.0 bucket.
    seen = []
    for order in ob.iter_orders(min_price=105.0, max_price=125.0):
        seen.append(order)
        if order is o_low:
            ob.remove_order(o_high)
    assert seen == [o_low]  # o_high's level was gone before the walk reached it

    # Invariant: no orphan/empty buckets; the bucket map and the level list stay
    # in lock-step.
    assert all(bucket for bucket in ob.orders_at_price.values())
    assert set(ob.orders_at_price) == set(ob.price_levels)

    # A new order reusing 120.0 must become a walkable level again -- the bug
    # skipped the level insertion when an orphan empty bucket shadowed the price.
    o_new = Order("new", -1.0, limit=120.0)
    ob.add_order(o_new)
    assert 120.0 in ob.price_levels
    assert o_new in list(ob.iter_orders(min_price=115.0, max_price=125.0))
