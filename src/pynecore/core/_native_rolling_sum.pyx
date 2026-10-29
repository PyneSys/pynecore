# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: infer_types=False, initializedcheck=False
"""Compiled twin of :class:`pynecore.core.rolling_sum.SumMachine`.

The SAME machine as the pure-Python original, branch for branch and operation for
operation, only compiled: the Python class stays the reference implementation and the
fallback wherever this extension is not built, and it is swapped for this one at import
when it is. The measured TradingView laws behind every branch are documented on the
original; the comments here only mark what the compiled form does differently to keep
the results identical.

Bit-identity rests on the C compiler evaluating every expression exactly as written: no
fused multiply-add contraction and no reassociation (the build passes
``-ffp-contract=off``, ``/fp:precise`` on MSVC, and never ``-ffast-math``). The
compensated adds are exactly the kind of arithmetic a reassociating compiler would
"simplify" away.

The window lives in an ``array('d')`` rather than a raw buffer: a rollback token holds
the array of the state it was taken in, and a step that grows it leaves the token's
array untouched, so the reference keeps it alive.
"""
cimport cython
from cpython cimport array
from cpython.float cimport PyFloat_AS_DOUBLE
from cpython.long cimport PyLong_AsLongLongAndOverflow
from libc.stdlib cimport free, realloc

import array as _array

from . import rolling_sum as _rs
from .safe_convert import native_int as _native_int
from .series import SeriesImpl as _SeriesImpl
from ..types.na import na_float as _na_float

__all__ = ['SumMachine']

cdef array.array _DOUBLES = _array.array('d')

# A length this large behaves exactly like any larger one: it exceeds every possible
# stored-bar count, and on a stored bar it fails the ``max_bars_back`` limit anyway
cdef long long _LENGTH_CAP = 1LL << 62

cdef str _LENGTH_ERROR = "Invalid length, length must be greater than 0!"

cdef str _STALE_TOKEN_ERROR = _rs._STALE_TOKEN_ERROR


cdef inline array.array _zeros(Py_ssize_t n):
    return array.clone(_DOUBLES, n, True)


cdef struct _Undo:
    unsigned long long values_gen
    long long win_at
    double win_old


# The records live in a C array instead of a list of tuples, so journaling a step
# allocates nothing. A chain of journals is released node by node rather than by
# recursion, however long an old token kept it.
@cython.trashcan(True)
cdef class _Journal:
    """What the steps taken since one snapshot overwrote in place (see the original)."""
    cdef _Undo* records
    cdef Py_ssize_t count, alloc
    cdef _Journal following

    def __cinit__(self):
        self.records = NULL
        self.count = 0
        self.alloc = 0

    def __dealloc__(self):
        free(self.records)

    cdef int append(self, unsigned long long values_gen, long long win_at,
                    double win_old) except -1:
        cdef Py_ssize_t size
        cdef _Undo* records
        cdef _Undo* record
        if self.count == self.alloc:
            size = self.alloc * 2 if self.alloc else 4
            records = <_Undo*> realloc(self.records, size * sizeof(_Undo))
            if records == NULL:
                raise MemoryError()
            self.records = records
            self.alloc = size
        record = &self.records[self.count]
        record.values_gen = values_gen
        record.win_at = win_at
        record.win_old = win_old
        self.count += 1
        return 0


cdef class _SumToken:
    """Rollback token of a :class:`SumMachine` (see ``__pyne_snapshot__``)."""
    cdef _Journal journal
    cdef unsigned long long values_gen
    cdef double summ, compensation
    cdef long long seen, window, capacity
    cdef long long values_cap, values_size, values_pos
    cdef array.array values
    cdef object values_bar


cdef class SumMachine:
    """The rolling-sum machine of one ``math.sum`` call site (compiled twin)."""
    cdef double summ, compensation
    cdef long long seen, window, capacity
    cdef array.array values
    cdef long long values_cap, values_size, values_pos
    cdef object values_bar
    cdef object lib
    cdef bint memo_valid
    cdef long long memo_n, memo_prev_w, memo_shift
    cdef double memo_s, memo_c
    cdef double* memo_sums
    cdef double* memo_comps
    cdef Py_ssize_t memo_len, memo_alloc
    cdef _Journal journal
    cdef unsigned long long values_gen, counter

    def __cinit__(self):
        self.memo_sums = NULL
        self.memo_comps = NULL
        self.memo_len = 0
        self.memo_alloc = 0

    def __init__(self):
        self.lib = _rs._bar_source()
        self.summ = 0.0
        self.compensation = 0.0
        self.seen = 0
        self.window = 0
        self.capacity = _SeriesImpl.DEFAULT_MAX_BARS_BACK
        self.values = _zeros(0)
        self.values_cap = _SeriesImpl.DEFAULT_MAX_BARS_BACK + 1
        self.values_size = 0
        self.values_pos = 0
        self.values_bar = -1
        self.memo_valid = False
        self.journal = None
        self.values_gen = 0
        self.counter = 0

    def __dealloc__(self):
        free(self.memo_sums)
        free(self.memo_comps)

    cdef int _memo_reserve(self, Py_ssize_t n) except -1:
        cdef Py_ssize_t size
        cdef double* sums
        cdef double* comps
        if n <= self.memo_alloc:
            return 0
        size = self.memo_alloc * 2
        if size < n:
            size = n
        sums = <double*> realloc(self.memo_sums, size * sizeof(double))
        if sums == NULL:
            raise MemoryError()
        self.memo_sums = sums
        comps = <double*> realloc(self.memo_comps, size * sizeof(double))
        if comps == NULL:
            raise MemoryError()
        self.memo_comps = comps
        self.memo_alloc = size
        return 0

    cdef inline double _offset(self, long long k):
        # Window offset ``k`` (0 = newest); the caller checked it is stored
        cdef long long pos = self.values_pos - 1 - k
        if pos < 0:
            pos += self.values_cap
        return self.values.data.as_doubles[pos]

    cdef int _check_stop(self, long long stop) except -1:
        # The bound check of a ``SeriesImpl`` slice read
        if stop > self.values_size:
            raise IndexError("Slice stop index out of range!")
        return 0

    cdef int _resize_values(self, long long max_bars_back) except -1:
        # ``SeriesImpl.max_bars_back`` setter semantics, see the original
        cdef long long new_cap, size, pos, old_cap, keep, start, i
        cdef array.array old, fresh
        if max_bars_back <= 0:
            raise ValueError("The max_bars_back must be a positive integer!")
        if max_bars_back > _SeriesImpl.MAXIMUM_MAX_BARS_BACK:
            raise ValueError(
                f"The max_bars_back cannot exceed {_SeriesImpl.MAXIMUM_MAX_BARS_BACK}!")
        new_cap = max_bars_back + 1
        if new_cap == self.values_cap:
            return 0
        size = self.values_size
        pos = self.values_pos
        if new_cap > size and size == pos:
            self.values_cap = new_cap
            return 0
        old = self.values
        old_cap = self.values_cap
        keep = size if size < new_cap else new_cap
        start = (pos - keep) % old_cap
        if start < 0:
            start += old_cap
        fresh = _zeros(keep)
        for i in range(keep):
            fresh.data.as_doubles[i] = old.data.as_doubles[(start + i) % old_cap]
        self.values = fresh
        self.counter += 1
        self.values_gen = self.counter
        self.values_cap = new_cap
        self.values_size = keep
        self.values_pos = keep
        return 0

    def step(self, source, length):
        """One ``math.sum`` call (see :meth:`pynecore.core.rolling_sum.SumMachine.step`)."""
        cdef bint source_na
        cdef double value = 0.0
        cdef double s, c, v, t, d
        cdef double win_old = 0.0
        cdef long long win_at = -1
        cdef unsigned long long values_gen
        cdef long long clen, n, prev_w, new_w, pos, size, need, grow
        cdef long long i, shift, top, walked, k, stop, start
        cdef int overflow = 0
        cdef double* vals
        cdef object bar

        if type(source) is float:
            value = PyFloat_AS_DOUBLE(source)
            source_na = value != value
            if source_na:
                value = 0.0
        else:
            source_na = not (source == source)
            if not source_na:
                value = float(source)

        # ``native_int`` of the original, with its two common shapes kept in C: a
        # float length truncates toward zero exactly like ``int()``, and a non-finite
        # one (an na) takes the Python path, where the assert rejects it
        if type(length) is float and PyFloat_AS_DOUBLE(length) - PyFloat_AS_DOUBLE(length) == 0.0:
            d = PyFloat_AS_DOUBLE(length)
            if d >= <double> _LENGTH_CAP:
                clen = _LENGTH_CAP
            elif d <= -<double> _LENGTH_CAP:
                clen = -1
            else:
                clen = <long long> d
            assert clen > 0, _LENGTH_ERROR
        else:
            if type(length) is not int:
                length = _native_int(length)
            if type(length) is int:
                clen = PyLong_AsLongLongAndOverflow(length, &overflow)
                if overflow > 0 or clen > _LENGTH_CAP:
                    clen = _LENGTH_CAP
                elif overflow < 0:
                    clen = -1
                assert clen > 0, _LENGTH_ERROR
            else:
                assert length > 0, _LENGTH_ERROR
                clen = length

        n = self.seen
        if not source_na:
            n += 1
        prev_w = self.window
        new_w = clen if n >= clen else n
        if source_na and prev_w == new_w:
            return self.summ + self.compensation if new_w >= clen else _na_float

        values_gen = self.values_gen

        if not source_na:
            bar = self.lib.bar_index
            size = self.values_size
            if self.values_bar == bar:
                if size:
                    pos = self.values_pos - 1
                    if pos < 0:
                        pos += self.values_cap
                    vals = self.values.data.as_doubles
                    win_at = pos
                    win_old = vals[pos]
                    vals[pos] = value
            else:
                if size < self.values_cap:
                    # The window is linear while it fills: the write lands at ``size``
                    pos = self.values_pos
                    if pos >= len(self.values):
                        grow = len(self.values) * 2
                        if grow < pos + 1:
                            grow = pos + 1
                        elif grow > self.values_cap:
                            grow = self.values_cap
                        array.resize(self.values, grow)
                    self.values.data.as_doubles[pos] = value
                    self.values_size = size + 1
                    self.values_pos += 1
                else:
                    pos = self.values_pos
                    if pos >= self.values_cap:
                        pos = 0
                        self.values_pos = 1
                    else:
                        self.values_pos = pos + 1
                    vals = self.values.data.as_doubles
                    win_at = pos
                    win_old = vals[pos]
                    vals[pos] = value
                self.values_bar = bar
            if clen > self.capacity:
                self.capacity = clen
                self._resize_values(clen)

        c = self.compensation
        s = self.summ

        if source_na:
            shift = 0
        else:
            shift = 1

        if prev_w + shift > new_w:
            top = prev_w - 1 + shift
            if top > new_w:
                need = top - new_w
                if not (self.memo_valid and self.memo_n == n and self.memo_prev_w == prev_w
                        and self.memo_shift == shift and self.memo_s == s and self.memo_c == c):
                    self._memo_reserve(1)
                    self.memo_valid = True
                    self.memo_n = n
                    self.memo_prev_w = prev_w
                    self.memo_shift = shift
                    self.memo_s = s
                    self.memo_c = c
                    self.memo_sums[0] = s
                    self.memo_comps[0] = c
                    self.memo_len = 1
                walked = self.memo_len - 1
                if need <= walked:
                    s = self.memo_sums[need]
                    c = self.memo_comps[need]
                else:
                    s = self.memo_sums[walked]
                    c = self.memo_comps[walked]
                    # The memo is extended before the window read, as the original
                    # pads its lists before the read can raise
                    self._memo_reserve(need + 1)
                    for i in range(walked + 1, need + 1):
                        self.memo_sums[i] = 0.0
                        self.memo_comps[i] = 0.0
                    self.memo_len = need + 1
                    stop = top - walked + 1
                    self._check_stop(stop)
                    i = walked + 1
                    k = stop - 1
                    while k >= new_w + 1:
                        v = -self._offset(k)
                        t = s + v
                        if (s if s >= 0.0 else -s) >= (v if v >= 0.0 else -v):
                            c += (s - t) + v
                        else:
                            c += (v - t) + s
                        s = t
                        self.memo_sums[i] = s
                        self.memo_comps[i] = c
                        i += 1
                        k -= 1
        elif new_w > prev_w + shift or (new_w < n and not source_na):
            if source_na:
                top = new_w
            else:
                top = new_w + 1 if new_w < n else n
            self._check_stop(top)
            start = prev_w + shift
            if start > top:
                start = top
            # Newest first: offsets ``start``..``top - 1``
            k = start
            while k < top:
                v = self._offset(k)
                t = s + v
                if (s if s >= 0.0 else -s) >= (v if v >= 0.0 else -v):
                    c += (s - t) + v
                else:
                    c += (v - t) + s
                s = t
                k += 1

        if new_w < n and (shift or prev_w > new_w):
            self._check_stop(new_w + 1)
            v = -self._offset(new_w)
            t = s + v
            if (s if s >= 0.0 else -s) >= (v if v >= 0.0 else -v):
                c += (s - t) + v
            else:
                c += (v - t) + s
            s = t
        if not source_na:
            t = s + value
            if (s if s >= 0.0 else -s) >= (value if value >= 0.0 else -value):
                c += (s - t) + value
            else:
                c += (value - t) + s
            s = t
        self.summ = s
        self.compensation = c
        self.seen = n
        self.window = new_w
        if self.journal is not None:
            self.journal.append(values_gen, win_at, win_old)

        return s + c if new_w >= clen else _na_float

    def __pyne_snapshot__(self):
        """Rollback token of the current state (see ``__pyne_restore__``)."""
        cdef _SumToken token = _SumToken.__new__(_SumToken)
        cdef _Journal journal = self.journal
        cdef _Journal fresh
        if journal is None or journal.count:
            fresh = _Journal.__new__(_Journal)
            if journal is not None:
                journal.following = fresh
            self.journal = journal = fresh
        token.journal = journal
        token.summ = self.summ
        token.compensation = self.compensation
        token.seen = self.seen
        token.window = self.window
        token.capacity = self.capacity
        token.values = self.values
        token.values_cap = self.values_cap
        token.values_size = self.values_size
        token.values_pos = self.values_pos
        token.values_bar = self.values_bar
        token.values_gen = self.values_gen
        return token

    cdef void _unwind(self, _Journal journal, _SumToken token):
        # The records of one journal, newest first, put back into the token's array
        cdef Py_ssize_t i = journal.count
        cdef _Undo* record
        cdef double* values = token.values.data.as_doubles
        while i > 0:
            i -= 1
            record = &journal.records[i]
            if record.win_at >= 0 and record.values_gen == token.values_gen:
                values[record.win_at] = record.win_old

    def __pyne_restore__(self, _SumToken token not None):
        """Roll the machine back to a ``__pyne_snapshot__`` token (see the original)."""
        cdef _Journal journal = token.journal
        cdef _Journal node, last
        cdef list nodes
        cdef Py_ssize_t i
        if journal is self.journal:
            if not journal.count:
                return
        else:
            nodes = []
            last = journal
            node = journal.following
            while node is not None:
                nodes.append(node)
                last = node
                node = node.following
            if last is not self.journal:
                raise RuntimeError(_STALE_TOKEN_ERROR)
            i = len(nodes)
            while i > 0:
                i -= 1
                self._unwind(<_Journal> nodes[i], token)
        self._unwind(journal, token)
        if self.memo_valid and self.memo_n > token.seen + 1:
            self.memo_valid = False
        self.summ = token.summ
        self.compensation = token.compensation
        self.seen = token.seen
        self.window = token.window
        self.capacity = token.capacity
        self.values = token.values
        self.values_cap = token.values_cap
        self.values_size = token.values_size
        self.values_pos = token.values_pos
        self.values_bar = token.values_bar
        self.values_gen = token.values_gen
        journal.count = 0
        journal.following = None
        self.journal = journal


def _install():
    """Rebind the Python module's machine to this compiled twin.

    Runs at the end of this module's import, which ``rolling_sum`` triggers from its
    last lines, so the module is fully defined by then and its ``PYTHON_SUM_MACHINE``
    keeps the original reachable.
    """
    _rs.SumMachine = SumMachine
    _rs.rolling_sum_step = SumMachine.step


_install()
