from ..types.display import Display

# TradingView's own bit values, read from the compiled script metadata
# (``metaInfo.defaults.styles.<plot>.display``): ``all`` is every bit, not the
# union of the known places.
all: Display = Display(0xFFFFFFFF)  # noqa F821
data_window: Display = Display(2)
none: Display = Display(0)
pane: Display = Display(1)
pine_screener: Display = Display(16)
price_scale: Display = Display(4)
status_line: Display = Display(8)
