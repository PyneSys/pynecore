from ...types.strategy import Spread

#
# Constants
#
# PyneCore's own extension, no TradingView counterpart: how the strategy
# simulator charges the bid/ask spread (``strategy(spread_type=...)``).

none = Spread("none")
trade = Spread("trade")
opposite = Spread("opposite")
