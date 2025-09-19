from .base import IntEnum

__all__ = [
    'ScriptType',
    'indicator', 'strategy', 'library',
]


class ScriptType(IntEnum):
    __slots__ = ()


indicator = ScriptType()
strategy = ScriptType()
library = ScriptType()
