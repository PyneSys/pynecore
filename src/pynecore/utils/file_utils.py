"""
File utilities
"""
import os
from pathlib import Path


def copy_mtime(source_path: Path, target_path: Path) -> bool:
    """Copy modification time from source to target file.

    :param source_path: Source file to copy mtime from
    :param target_path: Target file to set mtime on
    :return: True if successful, False otherwise
    """
    try:
        source_mtime = source_path.stat().st_mtime
    except (OSError, FileNotFoundError):
        return False

    try:
        os.utime(target_path, (source_mtime, source_mtime))
        return True
    except (OSError, FileNotFoundError):
        return False


def is_updated(src_path: Path, dest_path: Path) -> bool:
    """
    Check if a file is updated using modification time comparison.

    :param src_path: Path to the .pine file
    :param dest_path: Path to the compiled .py file
    :return: True if compilation is needed, False otherwise
    :raises FileNotFoundError: If pine file doesn't exist
    :raises OSError: If we can't get modification times
    """

    # If output file doesn't exist, compilation is needed
    if not dest_path.exists():
        return True
    # If source file doesn't exist, assume compilation is needed
    if not src_path.exists():
        raise FileNotFoundError(f"Source file not found: {src_path}")

    # Get modification times
    src_mtime = os.path.getmtime(src_path)
    dst_mtime = os.path.getmtime(dest_path)

    # If source is newer than output, compilation is needed
    return src_mtime > dst_mtime
