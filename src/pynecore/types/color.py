def _channel(value: float) -> int:
    """
    Bring a Pine rgb channel argument into the 0-255 integer range.

    :param value: Channel value, possibly fractional, out of range or na
    :return: The channel as an integer between 0 and 255
    """
    if not (value == value):  # is_na_arg
        return 0
    if value <= 0:
        return 0
    if value >= 255:
        return 255
    return int(value)


class Color:
    """
    Color class that stores RGBA values in a single 32-bit integer.
    The fmt is 0xRRGGBBAA where each component is 8 bits.
    """

    __slots__ = ('value',)

    def __init__(self, hexstr: str) -> None:
        # Remove leading '#' if present
        hexstr = hexstr.lstrip('#').upper()

        # Add default alpha if not provided
        if len(hexstr) == 6:
            hexstr += 'FF'

        # Convert hex string to int
        self.value = int(hexstr, 16)

    def __repr__(self) -> str:
        return f'Color("#{self.value:08X}")'

    def __lt__(self, other: 'Color') -> bool:
        return self.value < other.value

    def __eq__(self, other: 'Color') -> bool:
        return self.value == other.value

    def __hash__(self) -> int:
        # Value-based, consistent with __eq__. Defining __eq__ otherwise sets
        # __hash__ to None (unhashable), which makes a Color an illegal
        # dataclass field default (``@udt`` uses dataclass) and forbids use as a
        # dict key / set member -- both of which Pine colors legitimately need.
        return hash(self.value)

    @property
    def r(self) -> int:
        """Red component (0-255)"""
        return (self.value >> 24) & 0xFF

    @property
    def g(self) -> int:
        """Green component (0-255)"""
        return (self.value >> 16) & 0xFF

    @property
    def b(self) -> int:
        """Blue component (0-255)"""
        return (self.value >> 8) & 0xFF

    @property
    def a(self) -> int:
        """Alpha component (0-255)"""
        return self.value & 0xFF

    @a.setter
    def a(self, alpha: int) -> None:
        """
        Set alpha component (0-255)
        0: fully transparent
        255: fully opaque

        :param alpha: Alpha value (0-255)
        """
        if not (0 <= alpha <= 255):
            raise ValueError("Alpha must be between 0 and 255")
        self.value = (self.value & 0xFFFFFF00) | alpha

    @property
    def t(self) -> float:
        """
        Transparency component (0-100)
        0: not transparent (fully opaque)
        100: invisible
        """
        return 100 - (self.value & 0xFF) / 255.0 * 100

    @t.setter
    def t(self, transp: float) -> None:
        """
        Set transparency component (0-100)
        0: not transparent (fully opaque)
        100: invisible

        :param transp: Transparency percentage (0-100)
        """
        if not (0 <= transp <= 100):
            raise ValueError("Transparency must be between 0 and 100")
        self.value = (self.value & 0xFFFFFF00) | int((1 - transp / 100.0) * 255)

    @classmethod
    def rgb(cls, r: float, g: float, b: float, transp: float = 0) -> 'Color':
        """
        Create a Color object from RGB values and transparency.

        Fractional arguments are truncated, out of range arguments are clipped, and an
        na argument counts as 0 for a channel and as fully transparent for ``transp``.

        :param r: Red component (0-255)
        :param g: Green component (0-255)
        :param b: Blue component (0-255)
        :param transp: Transparency percentage (0-100, 0: not transparent, 100: invisible)
        :return: Color object
        """
        # Measured on TradingView (BINANCE:BTCUSDT 1D): a fractional channel is
        # TRUNCATED, not rounded (127.4/127.5/127.6 all give 127, 126.5 gives 126,
        # 254.7 gives 254), every argument is clipped instead of rejected (300 -> 255,
        # -20 -> 0, transp 110 -> 100, transp -10 -> 0), and an na argument yields a
        # solid color: an na channel reads back as 0, an na transparency as 100.
        if not (transp == transp):  # is_na_arg
            transp = 100.0
        elif transp <= 0:
            transp = 0.0
        elif transp >= 100:
            transp = 100.0
        return cls(f'#{_channel(r):02X}{_channel(g):02X}{_channel(b):02X}'
                   f'{int((1 - transp / 100.0) * 255):02X}')
