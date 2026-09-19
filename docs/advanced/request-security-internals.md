<!--
---
weight: 1006
title: "request.security() Internals"
description: "Technical deep-dive into the multiprocessing architecture behind request.security()"
icon: "memory"
date: "2026-03-27"
lastmod: "2026-03-28"
draft: false
toc: true
categories: ["Advanced", "Technical Implementation"]
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
                          |  Security Process (per ID) |
                          |    advance_event.wait()    |
                          |    run bars to target_time |
                          |      __sec_write__(id, v)  |-----> write to ResultBlock
                          |    data_ready.set()        |
                          |    done_event.set()        |
                          +----------------------------+
```

## AST Transformation

The `SecurityTransformer` rewrites each `lib.request.security()` call into four protocol
functions. For example:

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

    if __active_security__ == "sec-abc-0":
        __sec_write__("sec-abc-0", lib.ta.sma(lib.close, 20))  # security: write result
    daily = __sec_read__("sec-abc-0", lib.na)  # both: read result

    if __active_security__ is None:
        __sec_wait__("sec-abc-0")  # chart: wait for completion
```

The `__active_security__` variable is `None` in the chart process and set to the security ID
in each security process. This makes the same compiled code run correctly in all contexts.

Write/read blocks stay at the **same scope level** as the original call, preventing `NameError`
for conditionally-scoped variables.

## Shared Memory Layout

**SyncBlock** — one per script run, contains metadata for all security slots:

| Field            | Type   | Offset | Description                              |
|------------------|--------|--------|------------------------------------------|
| `last_timestamp` | int64  | 0      | Last bar timestamp from security process |
| `version`        | uint32 | 8      | ResultBlock version (reallocation count) |
| `result_size`    | uint32 | 12     | Current pickle data size in bytes        |
| `target_time`    | int64  | 16     | Time the process should advance to       |
| `flags`          | uint8  | 24     | State flags                              |

**ResultBlock** — one per security context, holds pickled result. Auto-reallocates with doubled
size when data outgrows the block.

## Security Process Lifecycle

1. **Spawn** — `multiprocessing.Process` with `spawn` mode
2. **Init** — re-register import hooks, open shared memory by name, load OHLCV + syminfo
3. **Import** — re-import script module (AST transforms run again, fresh Series state)
4. **Configure** — `lib._lib_semaphore = True`, `__active_security__ = sec_id`
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

Security processes reading other contexts' values (nested dependencies) get **immediate** reads
without waiting. This prevents deadlocks between security processes.

## File Reference

| File                       | Purpose                                       |
|----------------------------|-----------------------------------------------|
| `transformers/security.py` | AST transformation (SecurityTransformer)      |
| `core/security_shm.py`     | Shared memory: SyncBlock, ResultBlock, Reader |
| `core/security.py`         | Runtime protocol, SecurityState, event logic  |
| `core/security_process.py` | Security process entry point and bar loop     |
| `core/script_runner.py`    | Integration: spawn, inject, cleanup           |
