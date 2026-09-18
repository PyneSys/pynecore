"""
Regression: a chart wait must never block once a child's death is registered.

The chart's per-bar waits are UNTIMED (``_wait_with_liveness``) and the death
watcher sets every wait event exactly ONCE. ``_settle_round`` loops: it wakes,
sees ``rounds_done`` still short of ``rounds_launched``, CLEARS the event and
waits again. The dying child releases the chart twice -- once from its own
``finally`` and once from the watcher -- so when both land before that second
wait, the chart's own ``clear()`` wipes the watcher's only set and the untimed
wait parks forever.

MEASURED on ``test_081``'s script with the wake-to-clear window widened: the
chart sat in ``_settle_round -> _wait_with_liveness(done_event)`` with
``failed_children={'sec...0'}``, ``rounds_launched=6``, ``rounds_done=5`` and
the merged child already exited with code 1. The registry check BEFORE the
block is what turns that into the RuntimeError the chart has to raise, which is
why it cannot be reduced back to a post-wait check.
"""
import multiprocessing
import sys
import threading
import time

import pytest

from pynecore.core.security import _wait_with_liveness, watch_security_child


class _DeadProcess:
    """The little of a died child process the raise path reads."""
    exitcode = 1


def __test_a_registered_death_raises_instead_of_blocking__():
    """An already-registered death must raise on an event nobody will set again.

    This is the chart's second loop turn: the event was cleared after the wake,
    the registry is filled, and nothing is going to set the event a second time.
    """
    event = multiprocessing.Event()  # deliberately NOT set
    box = {}

    def chart():
        try:
            _wait_with_liveness(event, 'sid', {'sid': _DeadProcess()}, {'sid'})
        except BaseException as exc:  # noqa: BLE001 - the raise IS the result
            box['raised'] = exc

    worker = threading.Thread(target=chart, daemon=True)
    worker.start()
    worker.join(10)
    assert not worker.is_alive(), \
        "the chart blocked on an event the death watcher had already set once"
    raised = box.get('raised')
    assert isinstance(raised, RuntimeError), f"expected a RuntimeError, got {raised!r}"
    assert "died" in str(raised), f"unexpected error message: {raised}"


def __test_a_peer_death_raises_on_this_context_s_wait__(log):
    """ANY child's death releases the wait, not only the awaited context's.

    A dead consumer leaves a producer blocked and the chart waiting on that
    producer, so the wait reports whichever child actually died.
    """
    event = multiprocessing.Event()
    box = {}

    def chart():
        try:
            _wait_with_liveness(event, 'alive_sid',
                                {'alive_sid': _DeadProcess(),
                                 'dead_sid': _DeadProcess()}, {'dead_sid'})
        except BaseException as exc:  # noqa: BLE001 - the raise IS the result
            box['raised'] = exc

    worker = threading.Thread(target=chart, daemon=True)
    worker.start()
    worker.join(10)
    assert not worker.is_alive(), "a peer's death left this context's wait blocked"
    assert "dead_sid" in str(box.get('raised')), \
        f"the wait did not name the child that died: {box.get('raised')!r}"
    log.info("peer death surfaced as: %s", box['raised'])


class _RecordingEvent:
    """An event that records what the registry held when it was set."""

    def __init__(self, registry):
        self.registry = registry
        self.registered_at_set = None
        self.set_count = 0

    def set(self):
        self.registered_at_set = set(self.registry)
        self.set_count += 1


def __test_the_watcher_fills_the_registry_before_it_wakes_the_chart__(log):
    """The registry must be filled BEFORE the wait events are set.

    The order is the whole point: a chart woken by the watcher re-checks the
    registry, and a set that arrives first would be answered with "no death
    yet" and a fresh untimed wait.
    """
    failed: set[str] = set()
    ctx = multiprocessing.get_context('spawn')
    # ``sys.exit`` is picklable, so the child needs no module-level target of
    # its own -- and a non-zero exit is exactly what the watcher reacts to.
    proc = ctx.Process(target=sys.exit, args=(1,))
    proc.start()
    wait_event = _RecordingEvent(failed)
    stop_event = _RecordingEvent(failed)
    try:
        # No ``proc.join()``: joining closes the sentinel the watcher polls.
        watch_security_child('sid', proc, failed, (wait_event,), (stop_event,))
        deadline = time.monotonic() + 30
        while wait_event.set_count == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        proc.join(10)

    assert wait_event.set_count == 1, \
        f"the watcher did not release the chart's wait: {wait_event.set_count}"
    assert wait_event.registered_at_set == {'sid'}, (
        "the wait event was set before the death was registered: "
        f"{wait_event.registered_at_set}")
    assert stop_event.registered_at_set == {'sid'}, (
        "a sibling's stop event was set before the death was registered: "
        f"{stop_event.registered_at_set}")
    log.info("watcher registered %s before setting %d events",
             wait_event.registered_at_set, wait_event.set_count + stop_event.set_count)


@pytest.mark.parametrize('sec_id', ['sid', 'other'])
def __test_a_context_without_a_process_keeps_the_plain_wait__(sec_id):
    """Chart-served and ignored contexts have no process and no registry check.

    Their signalling is driven by the chart itself, so an already-set event is
    all they ever wait for -- and an empty registry must not turn that into an
    error.
    """
    event = multiprocessing.Event()
    event.set()
    _wait_with_liveness(event, sec_id, None, set())
    _wait_with_liveness(event, sec_id, {}, set())
