"""
A source file's bytes paired with the stat fingerprint they belong to.
"""
import os
from pathlib import Path

__all__ = ['stable_source']

#: How many times a stable read is retried before it gives up. A file being
#: rewritten under the reader settles within a rename or two; anything past
#: that is churn no number of retries would outlast.
_STABLE_READ_ATTEMPTS = 3


def stable_source(path: Path) -> tuple[bytes, tuple[int, int]] | None:
    """
    A source's bytes and the fingerprint they belong to, as one pair.

    The fingerprint is what every later check compares against, so it may only
    ever be paired with the bytes it actually describes. Stat'ing around a read
    does NOT give that: an atomic replace landing between the read and the stat
    pairs one version's bytes with another version's fingerprint, and an
    interface built from that pairing is stale in a way the fingerprint check
    then certifies as fresh. Every reader that publishes a fingerprint
    alongside signatures it read goes through this -- the loader transforming
    a module, the analyser deriving one's types, the enum capture proving an
    imported constant.

    The pairing is taken through ONE open file: ``fstat`` before the read and
    again after it, both on the same descriptor, so a replacement is either
    wholly outside the pair (the descriptor keeps reading the file it was
    opened on, whose fingerprint is the one returned) or shows up as a
    difference between the two stats, which is retried. A pair that never
    settles, like a file that cannot be opened at all, has no fingerprint to
    give -- see ``module_interface.NO_FINGERPRINT``, which no real file's stat
    matches.

    :param path: Path to the source file
    :return: Its bytes and their ``(mtime_ns, size)``, or None when no such
             pair could be had
    """
    for _ in range(_STABLE_READ_ATTEMPTS):
        try:
            with open(path, 'rb') as handle:
                before = os.fstat(handle.fileno())
                data = handle.read()
                after = os.fstat(handle.fileno())
        except OSError:
            return None
        if (before.st_mtime_ns, before.st_size) == (after.st_mtime_ns, after.st_size):
            return data, (before.st_mtime_ns, before.st_size)
    return None
