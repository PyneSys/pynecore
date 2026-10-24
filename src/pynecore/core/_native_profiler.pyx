# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, binding=False
"""Compiled ``sys.monitoring`` callbacks of :mod:`pynecore.core.profiler`.

Every callback runs on a line event or a call of profiled code, so its own cost
lands on the measured lines. Written in Python, that cost was 230-370 ns per line
event and it skewed the result toward cheap lines that run often; compiled, with C
arrays as the accumulators, it is 40-90 ns. The callbacks are plain C functions
(``binding=False``), the clock is CPython's own ``perf_counter`` clock read
without creating a Python object (``PyTime_PerfCounterRaw``, Python 3.13+), and no
code object is ever hashed: a code object's hash covers its bytecode and constants
and is recomputed on every call, so code objects are keyed by identity.

Two kinds of code are tracked, both on one frame stack:

- **profiled code** (the script and its libraries): line events attribute the time
  from one line event to the next one of the same frame to the earlier line, and
  count the line's executions. A callee's time stays on the calling line too.

  A line event is not always a new execution. Python steps back onto a line it
  already ran in the same call when a ``with`` block exits, when a statement or a
  call spanning several lines finishes, and when PyneCore's ``request.security``
  protocol waits for a context's result at the end of the body. Outside a loop a
  body never runs a line twice, so a line event that goes backwards counts an
  execution only on a loop header (every iteration's check of a ``for`` or a
  ``while``, ``while True`` included, has a line event on its header), and only
  when it comes from outside the header: a step back from a later line of a header
  spanning several lines is the same check. The time of every event still counts.
  The one layout this cannot tell apart is a header spanning several lines whose
  last line also carries the body (``for i in range(\\n 3): x = i``): there the
  header's own step back and the loop's jump back land on the same line, so the
  header's count comes out too high; the times stay exact.
- **regions** (runtime functions of a category: the strategy engine, the output
  writers, the data feed, the security waits, the runner itself): only their
  entry and exit is tracked.

Category time is exclusive: at every moment the innermost tracked frame owns it.
Time outside the outermost frame (the consumer of the runner's iterator) is not
counted at all.
"""
import sys

from libc.stdint cimport int64_t
from libc.stdlib cimport calloc, free

cdef extern from *:
    """
    #if PY_VERSION_HEX >= 0x030D0000
    #define PYNE_PROFILE_CLOCK 1
    static inline int64_t pyne_profile_now(void) {
        PyTime_t t;
        PyTime_PerfCounterRaw(&t);
        return (int64_t)t;
    }
    #else
    #define PYNE_PROFILE_CLOCK 0
    static inline int64_t pyne_profile_now(void) { return 0; }
    #endif
    """
    int PYNE_PROFILE_CLOCK
    int64_t pyne_profile_now() nogil

cdef extern from "pythread.h":
    unsigned long PyThread_get_thread_ident() nogil

#: Number of time categories (see ``profiler.CATEGORIES``)
cdef enum:
    N_CATEGORIES = 8
    MAX_DEPTH = 4096

#: Whether this build has the clock the profiler needs (Python 3.13+)
CLOCK_AVAILABLE = bool(PYNE_PROFILE_CLOCK)

# Per-line accumulators of every profiled file, one block of slots per file, and
# in the same layout the last line of the loop header starting on a line (0 where
# no loop header starts)
cdef int64_t *_line_ns = NULL
cdef int64_t *_line_hits = NULL
cdef int *_loop_head = NULL
cdef Py_ssize_t _n_slots = 0

cdef int64_t _cat_ns[N_CATEGORIES]
cdef int _cur_cat = -1
cdef int64_t _cat_start = 0

# The frame stack: code identity, slot block (-1 for a region), block size,
# category, current line, the start of the current line's slice and the furthest
# line the frame got to since it last entered a loop
cdef Py_ssize_t _frame_code[MAX_DEPTH]
cdef Py_ssize_t _frame_base[MAX_DEPTH]
cdef Py_ssize_t _frame_size[MAX_DEPTH]
cdef int _frame_cat[MAX_DEPTH]
cdef int _frame_line[MAX_DEPTH]
cdef int64_t _frame_start[MAX_DEPTH]
cdef int _frame_reach[MAX_DEPTH]
cdef Py_ssize_t _depth = 0

cdef unsigned long _thread = 0
cdef bint _running = False
cdef int _tool = 2

#: Profiled files by ``co_filename``: (slot base, slot count, category)
cdef dict _files = {}
#: Tracked code objects by identity: (slot base or -1, slot count, category)
cdef dict _kinds = {}
#: Every code object local events were set on, to clear them at the stop
cdef list _codes = []
#: Profiled file names in slot order, for the results
cdef list _file_order = []

cdef object _DISABLE = None
cdef object _monitoring = None
cdef int _profiled_events = 0
cdef int _region_events = 0


cdef inline void _switch(int64_t now, int cat) noexcept:
    """Close the running category slice and open one for ``cat``."""
    global _cur_cat, _cat_start
    if _cur_cat >= 0:
        _cat_ns[_cur_cat] += now - _cat_start
    _cur_cat = cat
    _cat_start = now


cdef inline void _push(Py_ssize_t key, Py_ssize_t base, Py_ssize_t size, int cat) noexcept:
    global _depth
    cdef int64_t now = pyne_profile_now()
    if _depth < MAX_DEPTH:
        _frame_code[_depth] = key
        _frame_base[_depth] = base
        _frame_size[_depth] = size
        _frame_cat[_depth] = cat
        _frame_line[_depth] = 0
        _frame_start[_depth] = now
        _frame_reach[_depth] = 0
        _switch(now, cat)
    _depth += 1


cdef inline void _pop() noexcept:
    global _depth
    cdef int64_t now
    cdef Py_ssize_t base
    cdef int line
    if _depth <= 0:
        return
    _depth -= 1
    if _depth >= MAX_DEPTH:
        return
    now = pyne_profile_now()
    base = _frame_base[_depth]
    line = _frame_line[_depth]
    if base >= 0 and line:
        _line_ns[base + line] += now - _frame_start[_depth]
    _switch(now, _frame_cat[_depth - 1] if _depth > 0 else -1)


cdef inline bint _is_top(object code) noexcept:
    return 0 < _depth <= MAX_DEPTH and _frame_code[_depth - 1] == <Py_ssize_t><void *>code


def on_start(code, int offset):
    if PyThread_get_thread_ident() != _thread:
        return None
    cdef Py_ssize_t key = <Py_ssize_t><void *>code
    entry = _kinds.get(key)
    if entry is None:
        entry = _files.get(code.co_filename)
        if entry is None:
            return _DISABLE
        _kinds[key] = entry
        _codes.append(code)
        _monitoring.set_local_events(_tool, code, _profiled_events)
    base, size, cat = entry
    _push(key, base, size, cat)
    return None


def on_resume(code, int offset):
    if PyThread_get_thread_ident() != _thread:
        return None
    cdef Py_ssize_t key = <Py_ssize_t><void *>code
    entry = _kinds.get(key)
    if entry is not None:
        base, size, cat = entry
        _push(key, base, size, cat)
    return None


def on_line(code, int line):
    if PyThread_get_thread_ident() != _thread:
        return None
    cdef Py_ssize_t d = _depth - 1
    cdef Py_ssize_t base
    cdef int64_t now
    cdef int prev, head_end
    if d < 0 or d >= MAX_DEPTH:
        return None
    base = _frame_base[d]
    if base < 0 or line < 1 or line > _frame_size[d]:
        return None
    now = pyne_profile_now()
    prev = _frame_line[d]
    if prev:
        _line_ns[base + prev] += now - _frame_start[d]
    _frame_line[d] = line
    _frame_start[d] = now
    head_end = _loop_head[base + line]
    if line > _frame_reach[d] or (head_end and not line < prev <= head_end):
        _frame_reach[d] = line
        _line_hits[base + line] += 1
    return None


def on_exit(code, int offset, value):
    if PyThread_get_thread_ident() == _thread and _is_top(code):
        _pop()
    return None


def on_throw(code, int offset, exc):
    if PyThread_get_thread_ident() == _thread:
        on_resume(code, offset)
    return None


def start(object files, object regions, int tool):
    """Start profiling.

    :param files: ``(co_filename, line count, category, loop headers)`` of every
                  profiled file, a loop header given as its first and last line.
    :param regions: ``(code, category)`` of every region.
    :param tool: The ``sys.monitoring`` tool id to use.
    :raises RuntimeError: If this build has no profiling clock.
    :raises ValueError: If the tool id is already in use.
    """
    global _running
    if not PYNE_PROFILE_CLOCK:
        raise RuntimeError("profiling needs Python 3.13 or newer")
    if _running:
        raise RuntimeError("a profiler is already running in this process")
    monitoring = sys.monitoring
    monitoring.use_tool_id(tool, "pynecore-profiler")
    try:
        _start(monitoring, files, regions, tool)
    except BaseException:
        _release()
        _teardown(monitoring, tool)
        raise
    _running = True


cdef _start(object monitoring, object files, object regions, int tool):
    global _line_ns, _line_hits, _loop_head, _n_slots, _cur_cat, _depth, _thread
    global _tool, _DISABLE, _monitoring, _profiled_events, _region_events
    events = monitoring.events
    _release()
    cdef Py_ssize_t total = 0, base
    cdef int i, line, end
    _files.clear()
    _kinds.clear()
    _codes.clear()
    _file_order.clear()
    for filename, line_count, category, _loops in files:
        # Slot 0 of a block is never written: line numbers start at 1
        _files[filename] = (total, line_count, category)
        _file_order.append(filename)
        total += line_count + 1
    _line_ns = <int64_t *>calloc(total + 1, sizeof(int64_t))
    _line_hits = <int64_t *>calloc(total + 1, sizeof(int64_t))
    _loop_head = <int *>calloc(total + 1, sizeof(int))
    if _line_ns == NULL or _line_hits == NULL or _loop_head == NULL:
        raise MemoryError()
    _n_slots = total
    for filename, line_count, _category, loops in files:
        base = _files[filename][0]
        for line, end in loops:
            if 1 <= line <= line_count:
                _loop_head[base + line] = min(max(end, line), line_count)
    for i in range(N_CATEGORIES):
        _cat_ns[i] = 0

    _tool = tool
    _monitoring = monitoring
    _DISABLE = monitoring.DISABLE
    _profiled_events = events.LINE | events.PY_RETURN | events.PY_YIELD | events.PY_RESUME
    _region_events = events.PY_RETURN | events.PY_YIELD | events.PY_RESUME
    for code, category in regions:
        _kinds[<Py_ssize_t><void *>code] = (-1, 0, category)
        _codes.append(code)
        monitoring.set_local_events(tool, code, _region_events)

    _thread = PyThread_get_thread_ident()
    _cur_cat = -1
    _depth = 0

    monitoring.register_callback(tool, events.PY_START, on_start)
    monitoring.register_callback(tool, events.PY_RESUME, on_resume)
    monitoring.register_callback(tool, events.LINE, on_line)
    monitoring.register_callback(tool, events.PY_RETURN, on_exit)
    monitoring.register_callback(tool, events.PY_YIELD, on_exit)
    monitoring.register_callback(tool, events.PY_UNWIND, on_exit)
    monitoring.register_callback(tool, events.PY_THROW, on_throw)
    # A code object an earlier run answered with DISABLE must be seen again
    monitoring.restart_events()
    monitoring.set_events(tool, events.PY_START | events.PY_UNWIND | events.PY_THROW)


def running():
    """Whether profiling is running."""
    return _running


def stop():
    """Stop profiling and release the tool id. The results stay readable."""
    global _running, _depth
    if not _running:
        return
    _running = False
    _switch(pyne_profile_now(), -1)
    _teardown(_monitoring, _tool)
    _depth = 0


cdef _teardown(object monitoring, int tool):
    """Switch every event off, drop the callbacks and release the tool id."""
    events = monitoring.events
    monitoring.set_events(tool, 0)
    for code in _codes:
        monitoring.set_local_events(tool, code, 0)
    for event in (events.PY_START, events.PY_RESUME, events.LINE, events.PY_RETURN,
                  events.PY_YIELD, events.PY_UNWIND, events.PY_THROW):
        monitoring.register_callback(tool, event, None)
    monitoring.free_tool_id(tool)
    _codes.clear()
    _kinds.clear()


def results():
    """The measurements of the last run.

    :return: ``(files, categories)``: per profiled file name a dict of line number to
             ``(nanoseconds, executions)`` for every line that ran, and the
             nanoseconds of each category.
    """
    cdef Py_ssize_t base, line_count, line
    files = {}
    for filename in _file_order:
        base, line_count, _ = _files[filename]
        lines = {}
        if _line_ns != NULL:
            for line in range(1, line_count + 1):
                if _line_hits[base + line] or _line_ns[base + line]:
                    lines[line] = (_line_ns[base + line], _line_hits[base + line])
        files[filename] = lines
    return files, [_cat_ns[i] for i in range(N_CATEGORIES)]


cdef void _release() noexcept:
    global _line_ns, _line_hits, _loop_head, _n_slots
    free(_line_ns)
    free(_line_hits)
    free(_loop_head)
    _line_ns = NULL
    _line_hits = NULL
    _loop_head = NULL
    _n_slots = 0
