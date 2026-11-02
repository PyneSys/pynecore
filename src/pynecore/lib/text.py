from ..types.text import AlignEnum, FormatEnum, WrapEnum

# Text alignment constants
align_left = AlignEnum('left')
align_center = AlignEnum('center')
align_right = AlignEnum('right')
align_top = AlignEnum('top')
align_bottom = AlignEnum('bottom')

# Text format constants: TradingView's own bit values, read from the drawing data of a
# chart (``tfrm`` of labels, boxes and table cells); combined with ``+`` and ``-``
format_bold = FormatEnum(1)
format_italic = FormatEnum(2)
format_none = FormatEnum(0)

# Text wrap constants
wrap_auto = WrapEnum('auto')
wrap_none = WrapEnum('none')
