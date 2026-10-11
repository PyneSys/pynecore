"""
Deferred stop requests (:mod:`pynecore.core.broker.interrupt_guard`).
"""
import pytest

from pynecore.core.broker import interrupt_guard
from pynecore.core.broker.interrupt_guard import (
    critical_section,
    interrupt_pending,
    request_interrupt,
)


@pytest.fixture(autouse=True)
def _reset_guard():
    interrupt_guard._depth = 0  # noqa
    interrupt_guard._pending = False  # noqa
    yield
    interrupt_guard._depth = 0  # noqa
    interrupt_guard._pending = False  # noqa


def __test_request_outside_section_interrupts_at_once__():
    """With no section running, a stop request raises immediately."""
    with pytest.raises(KeyboardInterrupt):
        request_interrupt()
    assert not interrupt_pending()


def __test_request_inside_section_waits_for_its_end__():
    """A stop inside a section lets the section finish, then raises."""
    done: list[str] = []

    @critical_section
    def dispatch():
        request_interrupt()
        done.append("recorded")
        return "ok"

    with pytest.raises(KeyboardInterrupt):
        dispatch()
    assert done == ["recorded"]
    assert not interrupt_pending()


def __test_nested_sections_raise_only_at_the_outermost_end__():
    """Inner section boundaries do not fire the deferred stop."""
    trail: list[str] = []

    @critical_section
    def inner():
        request_interrupt()
        trail.append("inner")

    @critical_section
    def outer():
        inner()
        trail.append("outer")

    with pytest.raises(KeyboardInterrupt):
        outer()
    assert trail == ["inner", "outer"]


def __test_repeated_request_forces_the_stop__():
    """A second stop while one is pending raises inside the section."""
    trail: list[str] = []

    @critical_section
    def wedged():
        request_interrupt()
        trail.append("first")
        request_interrupt()
        trail.append("unreachable")

    with pytest.raises(KeyboardInterrupt):
        wedged()
    assert trail == ["first"]


def __test_section_error_is_not_masked_and_stop_stays_pending__():
    """An error ending the section propagates; the stop fires at the next boundary."""

    @critical_section
    def failing():
        request_interrupt()
        raise ValueError("venue refused")

    @critical_section
    def next_sync():
        return None

    with pytest.raises(ValueError):
        failing()
    assert interrupt_pending()
    with pytest.raises(KeyboardInterrupt):
        next_sync()
    assert not interrupt_pending()
