"""
@pyne
"""
from pynecore.lib import close, plot, script


@script.indicator("HA developing snapshot proxy")
def main():
    plot(close)


def __test_heikinashi_developing_snapshot__(log):
    """The Heikin Ashi carried state rides ``RootVarSnapshot``: a developing HTF
    bar recomputes from the fixed prior-close baseline (no drift), and only a
    period close commits the new baseline.

    This reproduces the security child's save/restore/commit sequence over the
    REAL ``instance_state`` + ``RootVarSnapshot`` + ``_heikinashi_step``, so a
    regression in the snapshot wiring (a missing ``restore()``, a wrong commit
    ordering) surfaces as Heikin Ashi drift here — without spawning a subprocess.
    The live security loop's own branches (@910/@968) are covered by the e2e
    path; this is the deterministic guard for the rollback discipline itself.
    """
    import math
    from pynecore.core import instance_state
    from pynecore.core.instance_state import RootVarSnapshot
    from pynecore.core.security_process import _heikinashi_step

    key = '__heikinashi__proxytest'
    ha_root = instance_state.create_root(key, {
        'init': (None, None), 'series': (), 'varip': (), 'children': (),
        'names': ('prevHaOpen', 'prevHaClose'),
    })
    try:
        pending: list = [None, None]

        def ha_apply(_o, _h, _lo, _c):
            _ho, _hh, _hl, _hc = _heikinashi_step(ha_root[0], ha_root[1], _o, _h, _lo, _c)
            pending[0], pending[1] = _ho, _hc
            return _ho, _hh, _hl, _hc

        def ha_commit():
            ha_root[0], ha_root[1] = pending[0], pending[1]

        # Registering two var slots makes the snapshot active for an otherwise
        # var-less plain-OHLCV context — the exact mechanism the child relies on.
        snap = RootVarSnapshot([key])
        assert snap.has_vars, "the two HA slots must be var slots for the snapshot"

        # ── P0: one confirmed (historical) bar establishes the first HA close ──
        ho0, _, _, hc0 = ha_apply(100.0, 105.0, 99.0, 104.0)
        ha_commit()
        snap.save()  # batch save after the historical bar
        assert ha_root == [ho0, hc0]
        assert math.isclose(ho0, (100.0 + 104.0) / 2.0)                 # seed open
        assert math.isclose(hc0, (100.0 + 105.0 + 99.0 + 104.0) / 4.0)  # ohlc4

        # ── P1: one developing HTF period, re-run on three growing ticks ──
        base = (ho0 + hc0) / 2.0  # the haOpen every developing tick MUST produce
        dev_ticks = [
            (104.0, 106.0, 103.0, 105.0),
            (104.0, 108.0, 102.0, 107.0),
            (104.0, 109.0, 101.0, 103.0),
        ]
        dev_open_seen = []
        for i, (o, h, lo, c) in enumerate(dev_ticks):
            if i == 0:
                # New dev period: transform reads the baseline, ``save()`` captures
                # it BEFORE the commit, then commit writes the developing HA.
                ho, _, _, _ = ha_apply(o, h, lo, c)
                snap.save()
                ha_commit()
            else:
                # Same-period re-tick: restore the baseline, reset, re-transform.
                snap.restore()
                instance_state.reset()
                ho, _, _, _ = ha_apply(o, h, lo, c)
                ha_commit()
            dev_open_seen.append(ho)

        # DRIFT GUARD: haOpen is identical on every developing tick and equals the
        # fixed P0-close baseline. Without ``snap.restore()`` between re-ticks,
        # tick 1's haOpen would become (dev_tick0_open + dev_tick0_close) / 2.
        for i, ho in enumerate(dev_open_seen):
            assert math.isclose(ho, base), \
                f"developing tick {i}: haOpen drifted to {ho}, expected {base}"

        # ── P1 close: restore baseline, run confirmed final, commit + save ──
        snap.restore()
        instance_state.reset()
        fo, _fh, _fl, fc = ha_apply(*dev_ticks[-1])
        ha_commit()
        snap.save()
        exp_open = base
        exp_close = sum(dev_ticks[-1]) / 4.0
        assert math.isclose(ha_root[0], exp_open) and math.isclose(ha_root[1], exp_close), \
            f"closed baseline {ha_root} != expected [{exp_open}, {exp_close}]"
        assert math.isclose(fo, exp_open) and math.isclose(fc, exp_close)

        log.info("Heikin Ashi developing-bar snapshot proxy passed — no drift, clean commit")
    finally:
        instance_state.discard_root(key)
