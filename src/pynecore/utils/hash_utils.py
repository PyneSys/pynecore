"""Hash utilities for smart compilation of .pine files."""

import hashlib
from pathlib import Path
from typing import Optional


def calculate_file_md5(file_path: Path, chunk_size: int = 65536) -> str:
    """Calculate MD5 hash of file content.
    
    Uses Python's built-in hashlib module for optimal performance.
    MD5 is 2-3x faster than SHA256 for change detection (not security).
    Memory efficient with chunk-based reading for large files.
    
    Args:
        file_path: Path to the file to hash
        chunk_size: Size of chunks to read (64KB default for optimal performance)
        
    Returns:
        MD5 hash as hexadecimal string
        
    Raises:
        FileNotFoundError: If file doesn't exist
        IOError: If file cannot be read
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    hash_md5 = hashlib.md5()
    
    try:
        with open(file_path, 'rb') as f:
            # Read file in chunks for memory efficiency
            while chunk := f.read(chunk_size):
                hash_md5.update(chunk)
    except IOError as e:
        raise IOError(f"Error reading file {file_path}: {e}")
    
    return hash_md5.hexdigest()


def get_hash_file_path(pine_file_path: Path) -> Path:
    """Get the path for the hash file corresponding to a .pine file.
    
    Args:
        pine_file_path: Path to the .pine file
        
    Returns:
        Path to the corresponding .pine.hash file
    """
    return pine_file_path.with_suffix(pine_file_path.suffix + '.hash')


def get_stored_hash(pine_file_path: Path) -> Optional[str]:
    """Get stored hash from .pine.hash file.
    
    Args:
        pine_file_path: Path to the .pine file
        
    Returns:
        Stored MD5 hash string or None if hash file doesn't exist or is invalid
    """
    hash_file_path = get_hash_file_path(pine_file_path)
    
    if not hash_file_path.exists():
        return None
    
    try:
        with open(hash_file_path, 'r', encoding='utf-8') as f:
            stored_hash = f.read().strip()
            
        # Validate hash format (32 character hexadecimal string)
        if len(stored_hash) == 32 and all(c in '0123456789abcdef' for c in stored_hash.lower()):
            return stored_hash
        else:
            return None
            
    except (IOError, UnicodeDecodeError):
        return None


def store_hash(pine_file_path: Path, hash_value: str) -> None:
    """Store hash in .pine.hash file.
    
    Args:
        pine_file_path: Path to the .pine file
        hash_value: MD5 hash to store
        
    Raises:
        IOError: If hash file cannot be written
    """
    hash_file_path = get_hash_file_path(pine_file_path)
    
    try:
        # Ensure parent directory exists
        hash_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(hash_file_path, 'w', encoding='utf-8') as f:
            f.write(hash_value)
            
    except IOError as e:
        raise IOError(f"Error writing hash file {hash_file_path}: {e}")


def file_needs_compilation(pine_file_path: Path, output_file_path: Path) -> bool:
    """Check if a .pine file needs compilation based on MD5 hash comparison.
    
    Args:
        pine_file_path: Path to the .pine file
        output_file_path: Path to the compiled .py file
        
    Returns:
        True if compilation is needed, False otherwise
    """
    # If output file doesn't exist, compilation is needed
    if not output_file_path.exists():
        return True
    
    # Calculate current hash of .pine file
    try:
        current_hash = calculate_file_md5(pine_file_path)
    except (FileNotFoundError, IOError):
        # If we can't read the .pine file, assume compilation is needed
        return True
    
    # Get stored hash
    stored_hash = get_stored_hash(pine_file_path)
    
    # If no stored hash or hashes don't match, compilation is needed
    return stored_hash != current_hash


def update_hash_after_compilation(pine_file_path: Path) -> None:
    """Update the hash file after successful compilation.
    
    Args:
        pine_file_path: Path to the .pine file that was compiled
        
    Raises:
        IOError: If hash cannot be calculated or stored
    """
    current_hash = calculate_file_md5(pine_file_path)
    store_hash(pine_file_path, current_hash)