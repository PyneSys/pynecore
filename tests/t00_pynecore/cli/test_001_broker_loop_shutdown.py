"""
Regression tests for :func:`pynecore.cli.commands.run._shutdown_broker_event_loop`.

The broker event loop runs ``run_forever`` on a dedicated pump thread while the
(synchronous) Pine script thread submits coroutines via
``run_coroutine_threadsafe``. At CLI teardown the loop must be stopped and only
then closed — closing a still-running loop raises
``RuntimeError: Cannot close a running event loop`` (observed after a normal
cTrader market round trip on SIGINT shutdown).

These tests exercise the shutdown sequencing with a real event loop on a real
thread (no broker, fully offline) plus the degenerate cases.
"""
import asyncio
import threading

from pynecore.cli.commands.run import _shutdown_broker_event_loop


def _run_loop_on_thread() -> tuple[asyncio.AbstractEventLoop, threading.Thread]:
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True, name="test-loop")
    thread.start()
    # Wait (event-driven) until the loop is actually running before returning.
    started = threading.Event()
    loop.call_soon_threadsafe(started.set)
    assert started.wait(timeout=5.0), "loop failed to start"
    return loop, thread


def __test_shutdown_stops_and_closes_running_loop__():
    """A running loop is stopped, its thread joined, and the loop closed."""
    loop, thread = _run_loop_on_thread()
    closed = _shutdown_broker_event_loop(loop, thread, timeout=5.0)
    assert closed is True
    assert not thread.is_alive()
    assert not loop.is_running()
    assert loop.is_closed()


def __test_shutdown_never_closes_a_running_loop__():
    """
    If the pump thread does not exit within the timeout the loop must be left
    open — never closed while running — so the CLI exit cannot crash.
    """
    loop, thread = _run_loop_on_thread()

    # Wedge the loop thread inside a synchronous callback that blocks on an
    # event. While it blocks, the loop cannot process the ``stop`` callback
    # scheduled after it, so ``run_forever`` keeps running and the join times
    # out — reproducing the "thread not stopped before close()" race.
    release = threading.Event()
    blocking = threading.Event()

    def _block() -> None:
        blocking.set()
        release.wait(timeout=10.0)

    loop.call_soon_threadsafe(_block)
    assert blocking.wait(timeout=5.0), "wedge callback did not start"

    try:
        closed = _shutdown_broker_event_loop(loop, thread, timeout=0.5)
        assert closed is False
        assert loop.is_running()
        assert not loop.is_closed()
    finally:
        # Release the wedge and shut the loop down cleanly for the test process.
        release.set()
        assert _shutdown_broker_event_loop(loop, thread, timeout=5.0) is True


def __test_shutdown_timeout_zero_waits_forever__():
    """
    ``--shutdown-timeout 0`` means *wait forever* (see the CLI option help and
    ``live_runner``, which maps non-positive values to an unlimited wait). A
    non-positive timeout must therefore not degrade into "do not wait at all":
    the drain has to run to completion and the loop must still be closed.
    """
    loop, thread = _run_loop_on_thread()

    cleaned_up = threading.Event()

    async def _slow_to_unwind() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            # Cleanup that outlives an immediate (zero-second) wait.
            await asyncio.sleep(0.2)
        finally:
            cleaned_up.set()

    spawned = threading.Event()

    def _spawn() -> None:
        loop.create_task(_slow_to_unwind())
        spawned.set()

    loop.call_soon_threadsafe(_spawn)
    assert spawned.wait(timeout=5.0), "task was not spawned"

    closed = _shutdown_broker_event_loop(loop, thread, timeout=0.0)

    assert closed is True, "timeout=0 must wait for the drain, not skip it"
    assert cleaned_up.is_set(), "background task was not awaited to completion"
    assert loop.is_closed()
    assert not thread.is_alive()


def __test_timed_out_shutdown_still_stops_the_loop_later__():
    """
    Giving up on the wait must not strand the teardown. The drain and the
    following ``loop.stop`` are chained on the loop thread, so once whatever
    wedged the loop clears, the loop stops on its own — without a second
    teardown call and without abandoning a half-scheduled drain.
    """
    loop, thread = _run_loop_on_thread()

    release = threading.Event()
    blocking = threading.Event()

    def _block() -> None:
        blocking.set()
        release.wait(timeout=10.0)

    loop.call_soon_threadsafe(_block)
    assert blocking.wait(timeout=5.0), "wedge callback did not start"

    assert _shutdown_broker_event_loop(loop, thread, timeout=0.5) is False

    # Clear the wedge; the previously scheduled drain -> stop chain must now run
    # to completion on its own.
    release.set()
    thread.join(timeout=5.0)
    assert not thread.is_alive(), "loop was never stopped after the wedge cleared"
    assert not loop.is_running()
    loop.close()


def __test_shutdown_with_no_thread_closes_idle_loop__():
    """A never-started pump (thread is None) still closes an idle loop."""
    loop = asyncio.new_event_loop()
    assert not loop.is_running()
    closed = _shutdown_broker_event_loop(loop, None, timeout=1.0)
    assert closed is True
    assert loop.is_closed()


def __test_shutdown_cancels_and_awaits_pending_tasks__():
    """
    Long-lived background tasks (the broker's private order-event stream and its
    WebSocket receive/ping loops) must be cancelled AND awaited before the loop
    is closed. Without the drain they are destroyed mid-await, producing
    ``Task was destroyed but it is pending`` + ``Event loop is closed`` — the
    exact failure observed when a strategy exception skips the normal teardown.
    """
    loop, thread = _run_loop_on_thread()

    cleaned_up = threading.Event()

    async def _fake_ws_recv_loop() -> None:
        # Mimics ``BybitWebSocket._recv_loop``: blocks forever on the transport
        # until cancelled, then unwinds cleanly in its ``finally``.
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            pass
        finally:
            cleaned_up.set()

    # Spawn the task ON the loop thread and capture the handle so we can assert
    # its terminal state after the loop is closed.
    task_box: list[asyncio.Task] = []
    spawned = threading.Event()

    def _spawn() -> None:
        task_box.append(loop.create_task(_fake_ws_recv_loop()))
        spawned.set()

    loop.call_soon_threadsafe(_spawn)
    assert spawned.wait(timeout=5.0), "task was not spawned"

    closed = _shutdown_broker_event_loop(loop, thread, timeout=5.0)

    assert closed is True
    assert loop.is_closed()
    # The task ran its cancellation to completion (finally observed) rather than
    # being abandoned pending at loop close.
    assert cleaned_up.is_set(), "background task was not awaited to completion"
    task = task_box[0]
    assert task.done(), "background task left pending at loop close"
    assert not thread.is_alive()
