"""Shared fixtures for DNSE plugin unit tests — no live network, no real filesystem.

Every DNSE call funnels through the client wrapper, so a single fake client injected
as ``broker._client`` / ``provider._client`` intercepts the whole REST surface. Set
per-method canned ``(status, body)`` replies; drive ``async`` methods with
``asyncio.run`` and monkeypatch ``time.sleep`` / ``asyncio.sleep`` so poll loops don't
wait. Test functions use the repo convention ``__test_*__`` (see ``pytest.ini``).
"""
import asyncio

import pytest
import pynecore.lib as lib

lib.bar_index = 0  # let the [BROKER] log formatter render during broker._emit()


class _FakeClient:
    """Configurable stand-in for the DNSE client wrapper.

    ``_FakeClient(post_order=(201, {"id": "1"}),
                  get_positions=lambda *a, **k: (200, {"positions": [...]}))``

    A response may be a canned ``(status, body)`` tuple or a callable receiving the
    call's args. Any unset method returns ``(200, {})``. Every call is recorded in
    ``.calls`` as ``(method_name, args, kwargs)``; ``.count(name)`` tallies one method.
    """

    def __init__(self, **responses):
        self._responses = responses
        self.calls = []

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)

        def _call(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            resp = self._responses.get(name)
            if callable(resp):
                return resp(*args, **kwargs)
            return resp if resp is not None else (200, {})

        return _call

    def count(self, method):
        return sum(1 for call in self.calls if call[0] == method)


@pytest.fixture
def fake_client():
    """Factory for a :class:`_FakeClient`; inject as ``broker._client`` / ``provider._client``."""
    return lambda **responses: _FakeClient(**responses)


@pytest.fixture
def collect():
    """Collect up to ``n`` items from an async generator, then close it: ``collect(gen, n)``."""
    def _collect(agen, n, *, timeout=1.0):
        async def _run():
            out = []
            try:
                for _ in range(n):
                    out.append(await asyncio.wait_for(agen.__anext__(), timeout))
            except (StopAsyncIteration, asyncio.TimeoutError):
                pass
            finally:
                await agen.aclose()
            return out
        return asyncio.run(_run())
    return _collect
