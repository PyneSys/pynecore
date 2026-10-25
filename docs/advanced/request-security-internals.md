<!--
---
weight: 1006
title: "request.security() Internals"
description: "Technical deep-dive into the multiprocessing architecture behind request.security()"
icon: "memory"
date: "2026-03-27"
lastmod: "2026-10-25"
draft: false
toc: true
categories: ["Advanced"]
tags: ["request-security", "multiprocessing", "shared-memory", "ipc", "ast"]
---
-->

# request.security() Internals

This page covers the internal architecture of `request.security()`. For usage documentation,
examples, and data preparation, see the [request.security() Library page](../lib/request-security.md).

## Architecture

```
                           Chart Process
                          +----------------------------+
                          |  ScriptRunner.run_iter()   |
                          |    for each bar:           |
                          |      __sec_signal__(id)    |-----> advance_event.set()
                          |      ...                   |
                          |      __sec_read__(id, na)  |<----- data_ready.wait()
                          |      ...                   |
                          |      __sec_wait__(id)      |<----- done_event.wait()
                          +----------------------------+
                                    |    ^
                              SharedMemory (SyncBlock + ResultBlocks)
                                    v    |
                          +----------------------------+
                          |  Security Process          |
                          |    advance_event.wait()    |
                          |    run bars to target_time |
                          |      __sec_write__(id, v)  |-----> write to ResultBlock
                          |    data_ready.set()        |
                          |    done_event.set()        |
                          +----------------------------+
```

The diagram shows one context's per-bar handshake. Which contexts get a process of their own is
decided at run time; see [Processes and Context Groups](#processes-and-context-groups).

## AST Transformation

The `SecurityTransformer` rewrites each `lib.request.security()` call into four protocol
functions. Around it the pipeline runs `SecurityDrawings` and `SecurityInstantiation` (before it),
and `SecuritySlice`, `SecurityDefault` and `VerifyClosedShift` (after it); see the step tables on
the [AST Transformation](./ast-transformations.md#the-steps) page. For example:

**Before:**

```python
def main():
    daily = lib.request.security(lib.syminfo.tickerid, "1D", lib.ta.sma(lib.close, 20))
```

**After (conceptual):**

```python
def main():
    if __active_security__ is None:
        __sec_signal__("sec-abc-0")  # chart: signal security process

    if __active_security__ is not None and "sec-abc-0" in __active_security__:
        __sec_write__("sec-abc-0", lib.ta.sma(lib.close, 20))  # security: write result
    daily = __sec_read__("sec-abc-0", lib.na)  # both: read result

    if __active_security__ is None:
        __sec_wait__("sec-abc-0")  # chart: wait for completion
```

The `__active_security__` variable is `None` in the chart process and the set of security IDs the
process serves in a security process. This makes the same compiled code run correctly in all contexts.

Write/read blocks stay at the **same scope level** as the original call, preventing `NameError`
for conditionally-scoped variables.

## Shared Memory Layout

Three kinds of shared memory blocks carry the data between processes. The byte-level layout is
defined, and documented field by field, at the top of `core/security_shm.py`; it changes with the
protocol, so treat that file as the reference.

**SyncBlock** — one per script run. It holds a fixed-size slot per security context and, after
the slots, a watermark matrix:

- per slot: the producer's **frontier close** (every bar of the context that closes at or before
  this instant is already published), the result block version and size, the target time the
  context should advance to, state flags, ring bookkeeping (entry count, head, version, bytes
  used), the developing higher-timeframe bar's OHLCV and time (used by `lookahead_on`), and the
  current round's tick instant, finished-round counter and park state;
- the watermark matrix: for each consumer (one row per context, plus one for the chart) and each
  producer, the oldest instant the consumer may still need. The ring garbage collector keeps
  everything at or above the minimum over a producer's consumers.

**ResultBlock** — one per security context, holds the pickled latest result. Reallocated with
doubled size when the data outgrows the block.

**RingBlock** — one per security context that has consumers (other contexts or the chart in a
batch round): an append-only history of `(open, close, value)` entries ordered by close.
Reallocated with doubled capacity when full; the version is part of the shared memory name, so
readers notice and re-attach.

## Processes and Context Groups

A `request.security()` call does not always get a process of its own:

- A context that resolves to the chart's own symbol and timeframe has no process; it is evaluated
  inline in the chart process. So does a context whose symbol was dropped by
  `ignore_invalid_symbol=True`.
- The transformer marks contexts that are guaranteed to resolve to one and the same feed as a
  group. Members that the script reads unconditionally may share **one** child process. The
  member whose `__sec_signal__` runs first becomes the group's primary and is the only one the
  chart schedules; every member keeps its own result block, ring, `data_ready` event and
  `gaps`/`currency` handling. A member behind a branch that this run does not take is never
  evaluated by the group's child; it starts alone at its first read, or not at all.
- Every other context runs in its own process, started lazily at its first `__sec_signal__`.

## Security Process Lifecycle

1. **Start** — `multiprocessing.Process` from the `forkserver` context where the platform has
   it (Linux, macOS), `spawn` otherwise (Windows); see `core/security_mp.py`
2. **Init** — re-register import hooks, open shared memory by name, load OHLCV + syminfo
3. **Import** — re-import script module through the import hook (it loads the same transformed bytecode as the chart; fresh Series state)
4. **Configure** — `lib._lib_semaphore = True`, `__active_security__` = the set of security IDs this process serves; if the context has a `slice_main` clone (`SecuritySlice`), the process runs that function instead of `main()`
5. **Loop** — `advance_event.wait()` → run bars to `target_time` → `data_ready.set()` + `done_event.set()`
6. **Shutdown** — `stop_event` detected → cleanup and exit

## Skipping Developing Rounds

With `barmerge.lookahead_on` and an expression that is entirely history-indexed (`close[1]`,
`ta.sma(close, 5)[1]`, a tuple of such elements), the value a context produces is the same on every
chart bar of one higher-timeframe period. PyneCore marks such a context `closed_shift` when the script
is loaded, and the
runtime then runs the period's **developing** round only on its first chart bar; the later bars read
the value that round produced. For a daily context on a 30-minute chart this is 48x fewer child
rounds and 48x fewer `main()` executions per period.

The flag is cleared wherever the skipped rounds would be observable:

- the context does not resolve to a higher timeframe, or it has dependents, consumers or gaps,
- the child's code declares `varip` (`IBPersistent`) state, which is deliberately outside the
  re-tick rollback, or calls a library function whose own state is per execution (`ta.valuewhen`),
- the context's write does not run on every round of the child: it stands under a branch, or an
  early `return` or a `raise` can be reached ahead of it, in `main()` or in a helper it is called
  through.

Everything else the child runs is restored by the re-tick rollback, which is what makes the skip
unobservable. The one kind of state the rollback cannot restore is a plain Python object created at
module level. A language rule covers it instead: a script must not write such an object from inside a
function (see [Module-Level Objects Are Read-Only Inside Functions](../scripting.md)). The direct
forms are rejected with a `SyntaxError` when the script is loaded; a write through an alias or a
parameter is not detected, and a script that does it anyway can get different values with the skip
on and off. The dev-skip rests on that rule, not on a static proof about arbitrary Python.

Set `PYNE_NO_SECURITY_DEV_SKIP=1` to turn the skip off at runtime and run every developing round, as
before. It is a runtime switch, not a load-time one: the flag is always emitted and the decision
is made when the context is prepared, so bytecode built with and without the switch is identical.

## Cross-Context Reads

A security process that reads another context's value (a nested dependency) reads it from that
producer's ring. It first waits until the producer has published everything closing at or before
the consumer's own as-of instant (`RingReader.wait_for_close`), then takes the last entry closing
at or before it; a lower-timeframe producer yields the slice of intrabars that belongs to the
consumer's bar. Before waiting, the consumer parks its watermark at that instant, which tells the
producer whom a publication can release and which ring entries must be kept. A peer that nothing
produces for (a dropped symbol) answers with the default value instead of waiting.

## File Reference

| File                       | Purpose                                                   |
|----------------------------|-----------------------------------------------------------|
| `transformers/security.py` | AST transformation (SecurityTransformer)                  |
| `core/security_shm.py`     | Shared memory: SyncBlock, ResultBlock, RingBlock, readers |
| `core/security_mp.py`      | Process start method (forkserver / spawn) and its context |
| `core/security.py`         | Runtime protocol, SecurityState, context groups           |
| `core/security_process.py` | Security process entry point and bar loop                 |
| `core/script_runner.py`    | Integration: spawn, inject, cleanup                       |
