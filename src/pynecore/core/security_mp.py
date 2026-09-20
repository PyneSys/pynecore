"""
Multiprocessing context for the security child processes.

Security children used to be started from the platform default context, which
is ``fork`` on Linux up to Python 3.13. That is the one unsafe choice for us:
``live_runner`` starts the ``live-provider`` thread and blocks on its connect
BEFORE warmup, and security children start lazily during the run, so every live
child was forked from a multi-threaded parent. The child then inherits locks
held by threads that do not exist in it — CPython says so itself, in a warning
that is hidden because ``DeprecationWarning`` is silent outside ``__main__``::

    popen_fork.py:67: DeprecationWarning: This process (pid=...) is
    multi-threaded, use of fork() may lead to deadlocks in the child.

``forkserver`` does not have the problem: its server process is exec'd fresh
(``spawnv_passfds`` in ``multiprocessing/forkserver.py``), so it is
single-threaded and has never touched the provider, and the children fork from
it. macOS has needed the same property since Mojave (bpo-33725) and defaults to
``spawn`` for it. Windows has no ``forkserver`` at all, so ``spawn`` is the
fallback there.

Preloading the child entry module in the server pays the pynecore import once
per run instead of once per child. Measured time from ``Process()`` to a child
that has ``security_process`` imported, median of 10:

==============  ========  =========  ============  ====================
Environment         fork      spawn    forkserver    forkserver+preload
==============  ========  =========  ============  ====================
3.11 Linux        2.0 ms    78.7 ms       61.4 ms                8.9 ms
3.13 Linux        1.5 ms    79.6 ms       67.2 ms               10.0 ms
3.14 Linux        1.3 ms    81.3 ms       68.0 ms                7.1 ms
3.13 macOS        1.4 ms   112.2 ms       82.0 ms                5.1 ms
==============  ========  =========  ============  ====================

Everything that travels to a child must be created from this same context.
``SemLock`` unlinks its named semaphore immediately under ``fork`` and keeps it
linked under ``spawn``/``forkserver``, so an event built on a fork context
cannot be reopened by a forkserver child.
"""
import os
from multiprocessing import get_all_start_methods, get_context

#: ``spawn`` only on Windows, where ``forkserver`` does not exist.
START_METHOD = 'forkserver' if 'forkserver' in get_all_start_methods() else 'spawn'

#: The context every security process — and every primitive handed to one — is
#: created from.
mp_context = get_context(START_METHOD)


def preload_in_forkserver(module_name: str) -> None:
    """
    Pre-import a module in the forkserver, so children fork from a server that
    already holds it.

    Must run before the first ``Process().start()``, which is what starts the
    server: a later call is accepted and then ignored. The forkserver also
    swallows a failing preload import (``except ImportError: pass``), leaving
    children slow with nothing logged, so pass a name taken from an object that
    is already imported rather than a literal.

    :param module_name: Importable module name to preload.
    """
    if START_METHOD == 'forkserver':
        mp_context.set_forkserver_preload([module_name])


#: Prefix of every environment switch a run configures its children with.
_RUN_ENV_PREFIX = 'PYNE'


def child_run_env() -> dict[str, str]:
    """
    Snapshot the run's ``PYNE*`` environment for a child process.

    Nothing carries the environment to a child on its own: ``spawn`` re-execs
    with whatever the runner holds right now, but a ``forkserver`` child is
    forked from a server that was exec'd ONCE, at the first ``Process.start()``,
    and ``spawn.get_preparation_data`` transfers ``sys.path``, the authkey and
    the main module — never ``os.environ``. Every later change is invisible to
    the children.

    The runner does change these per run: ``cli/commands/run._pin_timenow``
    writes ``PYNE_TIMENOW_MS`` for each bounded replay, and the equivalence
    tests flip ``PYNE_NO_SECURITY_SLICE`` between runs inside one interpreter.
    A second run in the same process would otherwise give the chart its new
    settings and the security expressions the first run's — a silent, purely
    time-dependent divergence.

    :return: The ``PYNE``-prefixed variables of the runner's environment.
    """
    return {k: v for k, v in os.environ.items() if k.startswith(_RUN_ENV_PREFIX)}


def apply_child_run_env(run_env: dict[str, str]) -> None:
    """
    Make the child's ``PYNE*`` environment the snapshot, exactly.

    The snapshot is authoritative, not additive: a run that *clears* a switch
    is as common as one that sets it — ``_pin_timenow`` only pins a bounded
    replay and the equivalence tests ``os.environ.pop`` the slicing flag for
    their second run — and a forkserver child starts from the server's frozen
    environment. Merging alone would leave the value the server was exec'd with
    in place, so the child keeps running sliced-off or clock-pinned after the
    runner stopped asking for it. Variables outside the prefix are the child's
    own and stay untouched.

    :param run_env: The runner's snapshot (:func:`child_run_env`).
    """
    for key in [k for k in os.environ if k.startswith(_RUN_ENV_PREFIX) and k not in run_env]:
        del os.environ[key]
    os.environ.update(run_env)
