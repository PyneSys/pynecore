"""
The ``lib`` attributes that are no series: the one fact the library series pass
and the security pass's history-reference rule both read.

A leaf module (no imports), so an edit of the library series pass does not take
the native port of a step that only needs this set out of use (see
``transformers/native.py``).
"""

__all__ = ['NON_SERIES_LIB_ATTRS']

# Lib attributes that are NOT Series — subscript access on these should not be transformed
NON_SERIES_LIB_ATTRS = frozenset({
    'extra_fields',
})
