class IntEnum(int):
    """
    IntEnum class that auto-increments values.
    """

    __slots__ = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._counter = 0  # Minden leszármazottnak külön számlálója lesz

    def __new__(cls):
        # Az új objektum létrehozása a számláló aktuális értékével
        value = cls._counter
        cls._counter += 1
        return super().__new__(cls, value)


class StrLiteral(str):
    """
    StrLiteral class to store string literals.
    """

    __slots__ = ()
