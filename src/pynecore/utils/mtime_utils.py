"""File modification time utilities for smart compilation of .pine files."""

import os
from pathlib import Path


def file_needs_compilation(pine_file_path: Path, output_file_path: Path) -> bool:
    """Check if a .pine file needs compilation based on modification time.
    
    Args:
        pine_file_path: Path to the .pine file
        output_file_path: Path to the compiled .py file
        
    Returns:
        True if compilation is needed, False otherwise
    """
    # If output file doesn't exist, compilation is needed
    if not output_file_path.exists():
        return True
    
    # If source file doesn't exist, assume compilation is needed
    if not pine_file_path.exists():
        return True
    
    try:
        # Get modification times
        source_mtime = os.path.getmtime(pine_file_path)
        output_mtime = os.path.getmtime(output_file_path)
        
        # If source is newer than output, compilation is needed
        return source_mtime > output_mtime
        
    except OSError:
        # If we can't get modification times, assume compilation is needed
        return True