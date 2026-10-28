"""
The annotations that declare a ``varip`` variable: ``IBPersistent`` state, whose
slot keeps its value across the re-ticks of a bar.

A leaf module, so a step that only needs the names (the security slice's ``varip``
scan) does not depend on the persistent transformer's sources: the native ports pin
the sources of their Python steps by digest (see ``transformers/native.py``).
"""

__all__ = ['VARIP_TYPES']

VARIP_TYPES = ('IBPersistent', 'IBPersistentSeries')
