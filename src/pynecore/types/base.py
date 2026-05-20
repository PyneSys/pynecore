class IntEnum(int):
    """
    IntEnum class that auto-increments values.
    """

    def __init_subclass__(cls, start: int = 0, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._counter = start  # Each subclass gets its own counter

    def __new__(cls):
        # Create new object with the current counter value
        value = cls._counter
        cls._counter += 1
        # noinspection PyTypeChecker
        return super().__new__(cls, value)


class StrLiteral(str):
    """
    StrLiteral class to store string literals.
    """
