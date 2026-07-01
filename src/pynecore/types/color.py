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
    def rgb(cls, r: int, g: int, b: int, transp: float = 0) -> 'Color':
        """
        Create a Color object from RGB values and transparency.

        :param r: Red component (0-255)
        :param g: Green component (0-255)
        :param b: Blue component (0-255)
        :param transp: Transparency percentage (0-100, 0: not transparent, 100: invisible)
        :return: Color object
        """
        if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
            raise ValueError("RGB values must be between 0 and 255")
        return cls(f'#{r:02X}{g:02X}{b:02X}{int((1 - transp / 100.0) * 255):02X}')
