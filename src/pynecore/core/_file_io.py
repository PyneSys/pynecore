"""Cross-platform file primitives used by atomic OHLCV publication."""

import os
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Iterator, cast


_THREAD_LOCKS: dict[str, threading.RLock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


# The platform test is a literal ``sys.platform == "win32"`` comparison at every
# use site: type checkers prune the foreign-platform branch only in that form.
if sys.platform == "win32":
    import ctypes
    import msvcrt
    from ctypes import wintypes

    _DELETE = 0x00010000
    _SYNCHRONIZE = 0x00100000
    _GENERIC_READ = 0x80000000
    _GENERIC_WRITE = 0x40000000
    _FILE_SHARE_READ = 0x00000001
    _FILE_SHARE_WRITE = 0x00000002
    _FILE_SHARE_DELETE = 0x00000004
    _CREATE_ALWAYS = 2
    _OPEN_EXISTING = 3
    _OPEN_ALWAYS = 4
    _FILE_ATTRIBUTE_NORMAL = 0x00000080
    # FILE_INFO_BY_HANDLE_CLASS::FileRenameInfoEx (winbase.h ordinal 22). The
    # native NT FILE_INFORMATION_CLASS uses 65 for the same concept, but
    # SetFileInformationByHandle rejects that numbering with ERROR_INVALID_PARAMETER.
    _FILE_RENAME_INFO_EX = 22
    _FILE_RENAME_REPLACE_IF_EXISTS = 0x00000001
    _FILE_RENAME_POSIX_SEMANTICS = 0x00000002
    _LOCKFILE_EXCLUSIVE_LOCK = 0x00000002
    _ERROR_NOT_SUPPORTED = 50
    _ERROR_INVALID_PARAMETER = 87

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.CreateFileW.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    _kernel32.CreateFileW.restype = wintypes.HANDLE
    _kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    _kernel32.CloseHandle.restype = wintypes.BOOL
    _kernel32.SetFileInformationByHandle.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    )
    _kernel32.SetFileInformationByHandle.restype = wintypes.BOOL

    class _OverlappedOffset(ctypes.Structure):
        _fields_ = (("offset", wintypes.DWORD), ("offset_high", wintypes.DWORD))

    class _OverlappedLocation(ctypes.Union):
        _fields_ = (("position", _OverlappedOffset), ("pointer", wintypes.LPVOID))

    class _Overlapped(ctypes.Structure):
        _anonymous_ = ("location",)
        _fields_ = (
            ("internal", ctypes.c_size_t),
            ("internal_high", ctypes.c_size_t),
            ("location", _OverlappedLocation),
            ("event", wintypes.HANDLE),
        )

    _kernel32.LockFileEx.argtypes = (
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(_Overlapped),
    )
    _kernel32.LockFileEx.restype = wintypes.BOOL
    _kernel32.UnlockFileEx.argtypes = (
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(_Overlapped),
    )
    _kernel32.UnlockFileEx.restype = wintypes.BOOL

    _INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

    class _FileRenameInfo(ctypes.Structure):
        _fields_ = (
            ("flags", wintypes.DWORD),
            ("root_directory", wintypes.HANDLE),
            ("file_name_length", wintypes.DWORD),
            ("file_name", wintypes.WCHAR * 1),
        )


def _thread_lock(path: str | Path) -> threading.RLock:
    """Return the process-local lock paired with a lock-file path."""
    key = os.path.normcase(os.path.abspath(str(path)))
    with _THREAD_LOCKS_GUARD:
        lock = _THREAD_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _THREAD_LOCKS[key] = lock
        return lock


def open_shared_binary(path: str | Path, mode: str) -> BinaryIO:
    """Open an unbuffered binary file with delete sharing on Windows.

    Windows' normal Python file opens do not share delete access. OHLCV readers
    deliberately keep immutable snapshots open, so atomic publication needs those
    handles to remain valid while the path is rebound to a replacement file.

    :param path: File to open.
    :param mode: One of ``rb``, ``r+b``, ``w+b``, or ``a+b``.
    :return: Unbuffered binary file object.
    """
    if sys.platform != "win32":
        return cast(BinaryIO, open(path, mode, buffering=0))

    mode_settings = {
        "rb": (_GENERIC_READ, _OPEN_EXISTING, os.O_RDONLY | os.O_BINARY),
        "r+b": (
            _GENERIC_READ | _GENERIC_WRITE,
            _OPEN_EXISTING,
            os.O_RDWR | os.O_BINARY,
        ),
        "w+b": (
            _GENERIC_READ | _GENERIC_WRITE,
            _CREATE_ALWAYS,
            os.O_RDWR | os.O_BINARY,
        ),
        "a+b": (
            _GENERIC_READ | _GENERIC_WRITE,
            _OPEN_ALWAYS,
            os.O_RDWR | os.O_APPEND | os.O_BINARY,
        ),
    }
    try:
        desired_access, creation_disposition, descriptor_flags = mode_settings[mode]
    except KeyError as error:
        raise ValueError(f"Unsupported shared binary mode: {mode!r}") from error

    handle = _kernel32.CreateFileW(
        str(path),
        desired_access,
        _FILE_SHARE_READ | _FILE_SHARE_WRITE | _FILE_SHARE_DELETE,
        None,
        creation_disposition,
        _FILE_ATTRIBUTE_NORMAL,
        None,
    )
    if handle == _INVALID_HANDLE_VALUE:
        raise ctypes.WinError(ctypes.get_last_error())

    try:
        file_descriptor = msvcrt.open_osfhandle(handle, descriptor_flags)
    except Exception:
        _kernel32.CloseHandle(handle)
        raise
    try:
        return cast(
            BinaryIO,
            open(file_descriptor, mode, buffering=0, closefd=True),
        )
    except Exception:
        os.close(file_descriptor)
        raise


def replace_file(source: str | Path, destination: str | Path) -> None:
    """Atomically replace a path while PyneCore readers keep old snapshots open.

    Windows 10 version 1709 and later expose POSIX-style rename semantics through
    ``FileRenameInfoEx``. Existing handles to the replaced file keep referencing the
    old contents, while later opens resolve to the replacement. Unsupported filesystems
    fall back to :func:`os.replace`, which remains sufficient when no destination handle
    is open.

    :param source: Finished private file to publish.
    :param destination: Canonical path to replace.
    """
    if sys.platform != "win32":
        os.replace(source, destination)
        return

    try:
        _replace_file_windows(source, destination)
    except OSError as error:
        if error.winerror not in (_ERROR_NOT_SUPPORTED, _ERROR_INVALID_PARAMETER):
            raise
        os.replace(source, destination)


if sys.platform == "win32":

    def _replace_file_windows(source: str | Path, destination: str | Path) -> None:
        """Publish one file with Windows POSIX-style rename semantics."""
        source_handle = _kernel32.CreateFileW(
            str(source),
            _DELETE | _SYNCHRONIZE,
            _FILE_SHARE_READ | _FILE_SHARE_WRITE | _FILE_SHARE_DELETE,
            None,
            _OPEN_EXISTING,
            _FILE_ATTRIBUTE_NORMAL,
            None,
        )
        if source_handle == _INVALID_HANDLE_VALUE:
            raise ctypes.WinError(ctypes.get_last_error())

        destination_name = os.path.abspath(str(destination))
        encoded_name = destination_name.encode("utf-16-le")
        file_name_offset = _FileRenameInfo.file_name.offset
        buffer = ctypes.create_string_buffer(file_name_offset + len(encoded_name))
        rename_info = ctypes.cast(buffer, ctypes.POINTER(_FileRenameInfo)).contents
        rename_info.flags = (
            _FILE_RENAME_REPLACE_IF_EXISTS | _FILE_RENAME_POSIX_SEMANTICS
        )
        rename_info.root_directory = None
        rename_info.file_name_length = len(encoded_name)
        ctypes.memmove(
            ctypes.addressof(buffer) + file_name_offset,
            encoded_name,
            len(encoded_name),
        )

        try:
            if not _kernel32.SetFileInformationByHandle(
                source_handle,
                _FILE_RENAME_INFO_EX,
                buffer,
                len(buffer),
            ):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            _kernel32.CloseHandle(source_handle)


@contextmanager
def exclusive_file_lock(path: str | Path) -> Iterator[None]:
    """Hold one process-local and cross-process exclusive file lock.

    :param path: Sidecar lock-file path.
    """
    local_lock = _thread_lock(path)
    with local_lock:
        with open_shared_binary(path, "a+b") as lock_file:
            if os.fstat(lock_file.fileno()).st_size == 0:
                lock_file.write(b"\0")
                lock_file.flush()
            lock_file.seek(0)

            if sys.platform == "win32":
                overlapped = _Overlapped()
                handle = msvcrt.get_osfhandle(lock_file.fileno())
                if not _kernel32.LockFileEx(
                    handle,
                    _LOCKFILE_EXCLUSIVE_LOCK,
                    0,
                    1,
                    0,
                    ctypes.byref(overlapped),
                ):
                    raise ctypes.WinError(ctypes.get_last_error())
                try:
                    yield
                finally:
                    if not _kernel32.UnlockFileEx(
                        handle,
                        0,
                        1,
                        0,
                        ctypes.byref(overlapped),
                    ):
                        raise ctypes.WinError(ctypes.get_last_error())
                return

            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
