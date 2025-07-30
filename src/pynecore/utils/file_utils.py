"""File modification time utilities for API operations."""

import os
from pathlib import Path
from typing import Optional
from datetime import datetime


def get_file_mtime(file_path: Path) -> Optional[float]:
    """Get file modification time as timestamp.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Modification time as timestamp, or None if file doesn't exist
    """
    try:
        return file_path.stat().st_mtime
    except (OSError, FileNotFoundError):
        return None


def set_file_mtime(file_path: Path, mtime: float) -> bool:
    """Set file modification time.
    
    Args:
        file_path: Path to the file
        mtime: Modification time as timestamp
        
    Returns:
        True if successful, False otherwise
    """
    try:
        os.utime(file_path, (mtime, mtime))
        return True
    except (OSError, FileNotFoundError):
        return False


def is_file_newer(source_path: Path, target_path: Path) -> bool:
    """Check if source file is newer than target file.
    
    Args:
        source_path: Path to source file
        target_path: Path to target file
        
    Returns:
        True if source is newer than target, or if target doesn't exist
    """
    source_mtime = get_file_mtime(source_path)
    target_mtime = get_file_mtime(target_path)
    
    if source_mtime is None:
        return False
    
    if target_mtime is None:
        return True
    
    return source_mtime > target_mtime


def should_compile(pine_path: Path, py_path: Path, force: bool = False) -> bool:
    """Determine if Pine Script should be compiled based on modification times.
    
    Args:
        pine_path: Path to Pine Script file
        py_path: Path to Python output file
        force: Force compilation regardless of modification times
        
    Returns:
        True if compilation should proceed
    """
    if force:
        return True
    
    if not py_path.exists():
        return True
    
    return is_file_newer(pine_path, py_path)


def preserve_mtime(source_path: Path, target_path: Path) -> bool:
    """Copy modification time from source to target file.
    
    Args:
        source_path: Source file to copy mtime from
        target_path: Target file to set mtime on
        
    Returns:
        True if successful, False otherwise
    """
    source_mtime = get_file_mtime(source_path)
    if source_mtime is None:
        return False
    
    return set_file_mtime(target_path, source_mtime)