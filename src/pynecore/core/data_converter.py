"""Automatic data file to OHLCV conversion functionality.

This module provides automatic detection and conversion of CSV, TXT, and JSON files
to OHLCV format when needed, eliminating the manual step of running pyne data convert.
"""

from __future__ import annotations

import json
import struct
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import time
from pathlib import Path
from typing import Literal

from pynecore.core.ohlcv_file import OHLCVWriter, OHLCVReader
from pynecore.utils.file_utils import copy_mtime, is_updated
from ..lib.timeframe import from_seconds
from .syminfo import SymInfo, SymInfoInterval, SymInfoSession


@dataclass
class ConversionResult:
    """Result of a conversion operation.
    
    :param converted: Whether conversion was performed
    :param ohlcv_path: Path to the OHLCV file
    :param source_path: Path to the source file
    :param was_renamed: Whether the file was renamed from incorrect .ohlcv
    """
    converted: bool
    ohlcv_path: Path
    source_path: Path
    was_renamed: bool = False


class DataFormatError(Exception):
    """Raised when file format cannot be detected or is unsupported."""
    pass


class ConversionError(Exception):
    """Raised when conversion fails."""
    pass


class DataConverter:
    """Main class for automatic data file conversion.
    
    Provides both CLI and programmatic interfaces for converting
    CSV, TXT, and JSON files to OHLCV format automatically.
    """

    SUPPORTED_FORMATS = {'csv', 'txt', 'json'}

    def __init__(self):
        pass

    @staticmethod
    def auto_detect_symbol_from_filename(file_path: Path) -> str | None:
        """Auto-detect symbol from filename.
        
        :param file_path: Path to the data file
        :return: Symbol name or None if not detected
        """
        filename = file_path.stem  # Filename without extension

        # Common patterns to try:
        # 1. BTCUSD_1h.csv, AAPL_daily.csv, EUR_USD_4h.csv
        # 2. BTCUSD-1h.csv, AAPL-daily.csv
        # 3. btcusd.csv, aapl.csv (simple symbol only)
        # 4. BTC_USD_1D.csv, EUR_USD_4H.csv (with separators)

        # Normalize separators
        normalized = filename.replace('-', '_').upper()
        parts = normalized.split('_')

        if len(parts) >= 1:
            potential_symbol = parts[0]

            # Check if it looks like a valid symbol (3+ chars, alphanumeric)
            if len(potential_symbol) >= 3 and potential_symbol.isalnum():
                # Handle common forex patterns like EUR_USD -> EURUSD
                if len(parts) >= 2 and len(parts[1]) == 3 and parts[1].isalpha():
                    # Likely forex pair: EUR_USD -> EURUSD
                    return potential_symbol + parts[1]
                else:
                    return potential_symbol

        # Try the whole filename if it's a simple symbol
        if len(filename) >= 3 and filename.replace('_', '').replace('-', '').isalnum():
            return filename.upper().replace('_', '').replace('-', '')

        return None

    @staticmethod
    def detect_format(file_path: Path) -> Literal['csv', 'txt', 'json', 'ohlcv', 'unknown']:
        """Detect file format by extension and content inspection.
        
        :param file_path: Path to the file to analyze
        :return: Detected format
        :raises FileNotFoundError: If file doesn't exist
        :raises DataFormatError: If file cannot be read
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Check extension first
        ext = file_path.suffix.lower()
        if ext == '.ohlcv':
            # Using OHLCVReader to validate. If it raises an error, it's not a valid OHLCV file.
            try:
                with OHLCVReader(file_path):
                    # If we can open it successfully, it's a valid OHLCV file
                    return 'ohlcv'
            except (ValueError, OSError, IOError):
                # Not a valid OHLCV file, likely renamed - detect by content
                return DataConverter._detect_content_format(file_path)

        elif ext == '.csv':
            return 'csv'
        elif ext == '.txt':
            return 'txt'
        elif ext == '.json':
            return 'json'

        # No extension or unknown extension, inspect content
        return DataConverter._detect_content_format(file_path)

    @staticmethod
    def _detect_content_format(file_path: Path) -> Literal['csv', 'txt', 'json', 'unknown']:
        """Detect format by inspecting file content.
        
        :param file_path: Path to the file
        :return: Detected format based on content
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Read first few lines for analysis
                first_line = f.readline().strip()

                # Try JSON first (most specific)
                f.seek(0)
                try:
                    json.load(f)
                    return 'json'
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass

                # Check for CSV patterns (only if we have content)
                if first_line and ',' in first_line:
                    # Count commas to see if it looks like structured data
                    comma_count = first_line.count(',')
                    if comma_count >= 4:  # At least OHLC columns
                        return 'csv'

                # Check for other delimiters (TXT)
                if first_line and any(delim in first_line for delim in ['\t', ';', '|']):
                    return 'txt'

                # Default to CSV if it has some structure
                if first_line and ',' in first_line:
                    return 'csv'

                return 'unknown'

        except (OSError, IOError, UnicodeDecodeError):
            return 'unknown'

    @staticmethod
    def is_conversion_required(source_path: Path, ohlcv_path: Path | None = None) -> bool:
        """Check if conversion is required based on file freshness.
        
        :param source_path: Path to the source file
        :param ohlcv_path: Path to the OHLCV file (auto-generated if None)
        :return: True if conversion is needed
        """
        if ohlcv_path is None:
            ohlcv_path = source_path.with_suffix('.ohlcv')

        # If OHLCV file doesn't exist, conversion is needed
        if not ohlcv_path.exists():
            return True

        # Use existing file utility to check if source is newer
        return is_updated(source_path, ohlcv_path)

    def convert_if_needed(
            self,
            file_path: Path,
            *,
            force: bool = False,
            provider: str = "custom",
            symbol: str | None = None,
            timeframe: str = "1D",
            timezone: str = "UTC"
    ) -> ConversionResult:
        """Convert file to OHLCV format if needed.
        
        :param file_path: Path to the data file
        :param force: Force conversion even if OHLCV file is up-to-date
        :param provider: Data provider name for OHLCV file naming
        :param symbol: Symbol for OHLCV file naming
        :param timeframe: Timeframe for conversion
        :param timezone: Timezone for timestamp conversion
        :return: ConversionResult with conversion details
        :raises FileNotFoundError: If source file doesn't exist
        :raises DataFormatError: If file format is unsupported
        :raises ConversionError: If conversion fails
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Source file not found: {file_path}")

        # Detect file format
        detected_format = self.detect_format(file_path)

        # Handle incorrectly renamed files
        was_renamed = False
        if file_path.suffix == '.ohlcv' and detected_format in self.SUPPORTED_FORMATS:
            # File was incorrectly renamed, fix it
            original_path = file_path.with_suffix(f'.{detected_format}')
            file_path.rename(original_path)
            file_path = original_path
            was_renamed = True

        # If it's already OHLCV, no conversion needed
        if detected_format == 'ohlcv':
            return ConversionResult(
                converted=False,
                ohlcv_path=file_path,
                source_path=file_path,
                was_renamed=was_renamed
            )

        # Check if format is supported
        if detected_format not in self.SUPPORTED_FORMATS:
            raise DataFormatError(
                f"Unsupported file format '{detected_format}' for file: {file_path}"
            )

        # Determine OHLCV output path
        ohlcv_path = file_path.with_suffix('.ohlcv')

        # Check if conversion is needed
        if not force and not self.is_conversion_required(file_path, ohlcv_path):
            return ConversionResult(
                converted=False,
                ohlcv_path=ohlcv_path,
                source_path=file_path,
                was_renamed=was_renamed
            )

        # Perform conversion
        self._convert_file(
            source_path=file_path,
            ohlcv_path=ohlcv_path,
            format_type=detected_format,
            provider=provider,
            symbol=symbol,
            timeframe=timeframe,
            timezone=timezone
        )

        return ConversionResult(
            converted=True,
            ohlcv_path=ohlcv_path,
            source_path=file_path,
            was_renamed=was_renamed
        )

    def _convert_file(
            self,
            source_path: Path,
            ohlcv_path: Path,
            format_type: str,
            provider: str,  # Currently unused but kept for future extensibility
            symbol: str | None,
            timeframe: str,
            timezone: str
    ) -> None:
        """Perform the actual file conversion.
        
        :param source_path: Path to source file
        :param ohlcv_path: Path to output OHLCV file
        :param format_type: Detected format type
        :param provider: Data provider name (reserved for future use)
        :param symbol: Symbol name
        :param timeframe: Timeframe (will be auto-detected if "AUTO")
        :param timezone: Timezone
        :raises ConversionError: If conversion fails
        """
        # Create temporary file for atomic operation
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                    mode='wb',
                    suffix='.ohlcv',
                    dir=ohlcv_path.parent,
                    delete=False
            ) as temp_file:
                temp_path = Path(temp_file.name)

            # Perform conversion using existing OHLCV writer
            with OHLCVWriter(temp_path) as ohlcv_writer:
                if format_type == 'csv':
                    ohlcv_writer.load_from_csv(source_path, tz=timezone)
                elif format_type == 'json':
                    ohlcv_writer.load_from_json(source_path, tz=timezone)
                elif format_type == 'txt':
                    # Treat TXT files as CSV with different delimiters
                    ohlcv_writer.load_from_csv(source_path, tz=timezone)
                else:
                    raise ConversionError(f"Unsupported format for conversion: {format_type}")

            # Auto-detect timeframe if needed
            detected_timeframe = self._detect_timeframe_from_ohlcv(temp_path, timeframe)

            # Atomic rename to final location
            temp_path.replace(ohlcv_path)
            temp_path = None  # Prevent cleanup

            # Copy modification time from source to maintain freshness
            copy_mtime(source_path, ohlcv_path)

            # Generate TOML symbol info file if needed
            self._create_symbol_info_file(
                source_path=source_path,
                symbol=symbol,
                timeframe=detected_timeframe,
                timezone=timezone
            )

        except Exception as e:
            # Clean up temporary file on error
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            raise ConversionError(f"Failed to convert {source_path}: {e}") from e

    def _create_symbol_info_file(
            self,
            source_path: Path,
            symbol: str | None,
            timeframe: str,
            timezone: str
    ) -> None:
        """
        Create TOML symbol info file using SymInfo class if it doesn't exist or is outdated.
        
        :param source_path: Path to the source data file
        :param symbol: Symbol name (e.g., 'BTCUSD')
        :param timeframe: Timeframe (e.g., '1h', '1D')
        :param timezone: Timezone for the symbol
        """
        toml_path = source_path.with_suffix('.toml')

        # Skip if TOML file exists and is newer than source
        if toml_path.exists() and not is_updated(source_path, toml_path):
            return

        # Generate symbol info if symbol is provided
        if not symbol:
            return

        # Analyze price data to calculate trading parameters
        price_analysis = self._analyze_price_data(source_path)

        # Determine symbol type based on symbol name patterns
        symbol_upper = symbol.upper()
        symbol_type, currency, base_currency = self._detect_symbol_type(symbol_upper)

        # Get calculated trading parameters
        mintick = price_analysis['tick_size']
        pricescale = price_analysis['price_scale']
        minmove = price_analysis['min_move']
        pointvalue = self._get_default_pointvalue(symbol_type)

        # Create opening hours based on symbol type
        opening_hours = []
        if symbol_type == 'crypto':
            # 24/7 trading for crypto
            for day in range(1, 8):
                opening_hours.append(SymInfoInterval(
                    day=day,
                    start=time(0, 0, 0),
                    end=time(23, 59, 59)
                ))
        else:
            # Business hours for stocks/forex (Mon-Fri)
            for day in range(1, 6):
                opening_hours.append(SymInfoInterval(
                    day=day,
                    start=time(9, 30, 0),
                    end=time(16, 0, 0)
                ))

        # Create session starts and ends
        session_starts = [SymInfoSession(day=1, time=time(0, 0, 0))]
        session_ends = [SymInfoSession(day=7, time=time(23, 59, 59))]

        # Create SymInfo instance
        syminfo = SymInfo(
            prefix="CUSTOM",
            description=f"{symbol} - Auto-generated symbol info",
            ticker=symbol_upper,
            currency=currency,
            basecurrency=base_currency or "EUR",
            period=timeframe,
            type=symbol_type if symbol_type in ["stock", "fund", "dr", "right", "bond",
                                                "warrant", "structured", "index", "forex",
                                                "futures", "spread", "economic", "fundamental",
                                                "crypto", "spot", "swap", "option", "commodity",
                                                "other"] else "other",
            mintick=mintick,
            pricescale=int(pricescale),
            minmove=int(minmove),
            pointvalue=pointvalue,
            opening_hours=opening_hours,
            session_starts=session_starts,
            session_ends=session_ends,
            timezone=timezone,
            volumetype="base"
        )

        # Save using SymInfo's built-in method
        try:
            syminfo.save_toml(toml_path)
            # Copy modification time from source to maintain consistency
            copy_mtime(source_path, toml_path)
        except (OSError, IOError):
            # Don't fail the entire conversion if TOML creation fails
            pass

    def _analyze_price_data(self, source_path: Path) -> dict[str, float]:
        """Analyze price data to calculate trading parameters.
        
        :param source_path: Path to the source data file
        :return: Dictionary with tick_size, price_scale, and min_move
        """
        # Always use the basic method for maximum precision without external dependencies
        return self._analyze_price_data_basic(source_path)

    @staticmethod
    def _analyze_price_data_basic(source_path: Path) -> dict[str, float]:
        """Basic price data analysis by converting to OHLCV and using OHLCVReader.
        
        :param source_path: Path to the source data file
        :return: Dictionary with tick_size, price_scale, and min_move
        """
        try:
            # Convert file to temporary OHLCV format first
            with tempfile.NamedTemporaryFile(suffix='.ohlcv', delete=False) as temp_file:
                temp_ohlcv_path = Path(temp_file.name)

            try:
                # Use existing conversion logic
                with OHLCVWriter(temp_ohlcv_path) as ohlcv_writer:
                    if source_path.suffix.lower() == '.csv':
                        ohlcv_writer.load_from_csv(source_path, tz='UTC')
                    elif source_path.suffix.lower() == '.json':
                        ohlcv_writer.load_from_json(source_path, tz='UTC')
                    else:
                        # Treat as CSV with different delimiters
                        ohlcv_writer.load_from_csv(source_path, tz='UTC')

                # Now analyze price data using OHLCVReader
                with OHLCVReader(temp_ohlcv_path) as reader:
                    if reader.size == 0:
                        return {'tick_size': 0.01, 'price_scale': 100, 'min_move': 1}

                    # Analyze first 100 records (or all if less)
                    decimal_places = 0
                    sample_size = min(100, reader.size)

                    for i in range(sample_size):
                        ohlcv = reader.read(i)
                        # Check all price fields for decimal precision
                        for price in [ohlcv.open, ohlcv.high, ohlcv.low, ohlcv.close]:
                            # Detect actual decimal precision without artificial limits
                            if price != int(price):  # Has decimal component
                                # Use high precision to capture actual decimal places
                                price_str = f"{price:.15f}".rstrip('0').rstrip('.')
                                if '.' in price_str:
                                    current_decimals = len(price_str.split('.')[1])
                                    decimal_places = max(decimal_places, current_decimals)

                    # Calculate parameters preserving full precision
                    decimal_places = max(decimal_places, 2)  # At least 2 decimal places for safety
                    price_scale = 10 ** decimal_places
                    tick_size = 1.0 / price_scale
                    min_move = 1

                    return {
                        'tick_size': tick_size,
                        'price_scale': price_scale,
                        'min_move': min_move
                    }

            finally:
                # Clean up temporary file
                if temp_ohlcv_path.exists():
                    temp_ohlcv_path.unlink()

        except (OSError, IOError, ValueError, ConversionError):
            return {'tick_size': 0.01, 'price_scale': 100, 'min_move': 1}

    @staticmethod
    def _detect_symbol_type(symbol_upper: str) -> tuple[str, str, str | None]:
        """Detect symbol type and extract currency information.
        
        :param symbol_upper: Uppercase symbol string
        :return: Tuple of (symbol_type, currency, base_currency)
        """
        if any(crypto in symbol_upper for crypto in ['BTC', 'ETH', 'USD', 'USDT', 'USDC']):
            symbol_type = 'crypto'
            currency = 'USD'
            base_currency = symbol_upper.replace('USD', '').replace('USDT', '').replace('USDC', '')
            if not base_currency:  # If nothing left after removing USD variants
                base_currency = None
        elif '/' in symbol_upper:
            symbol_type = 'forex'
            parts = symbol_upper.split('/')
            currency = parts[1] if len(parts) > 1 else 'USD'
            base_currency = parts[0] if len(parts) > 0 else None
        else:
            symbol_type = 'stock'
            currency = 'USD'
            base_currency = None

        return symbol_type, currency, base_currency

    @staticmethod
    def _get_default_pointvalue(symbol_type: str) -> float:
        """Get default point value based on symbol type.
        
        :param symbol_type: Type of symbol (crypto, forex, stock)
        :return: Default point value
        """
        if symbol_type == 'forex':
            return 10.0
        else:  # crypto, stock, or unknown
            return 1.0

    def _detect_timeframe_from_ohlcv(self, ohlcv_path: Path, requested_timeframe: str) -> str:
        """Detect timeframe from OHLCV data by analyzing timestamp differences.
        
        :param ohlcv_path: Path to the OHLCV file
        :param requested_timeframe: Requested timeframe ("AUTO" for auto-detection)
        :return: Detected or original timeframe
        """
        if requested_timeframe.upper() != "AUTO":
            return requested_timeframe

        try:
            # Read first few records to analyze timestamp differences
            with open(ohlcv_path, 'rb') as f:
                # Read first 10 records (or less if file is smaller)
                timestamps = []
                for _ in range(10):
                    record = f.read(24)  # OHLCV record size
                    if len(record) < 24:
                        break

                    # Unpack timestamp (first 4 bytes as uint32)
                    timestamp = struct.unpack('<I', record[:4])[0]
                    timestamps.append(timestamp)

            if len(timestamps) < 2:
                return "1D"  # Default fallback

            # Calculate differences between consecutive timestamps
            differences = []
            for i in range(1, len(timestamps)):
                diff = timestamps[i] - timestamps[i - 1]
                if diff > 0:  # Only positive differences
                    differences.append(diff)

            if not differences:
                return "1D"  # Default fallback

            # Find the most common difference (mode)
            diff_counts = Counter(differences)
            most_common_diff = diff_counts.most_common(1)[0][0]

            # Convert seconds to timeframe string (timestamps are in seconds, not nanoseconds)
            return self._seconds_to_timeframe(most_common_diff)

        except (OSError, IOError, struct.error, IndexError, ValueError):
            # If detection fails, return default
            return "1D"

    @staticmethod
    def _seconds_to_timeframe(seconds: int) -> str:
        """
        Convert seconds difference to timeframe string using TV-compatible timeframe module.
        
        :param seconds: Time difference in seconds
        :return: Timeframe string (e.g., '1m', '5m', '1h', '1D')
        """
        # Handle edge cases
        if seconds <= 0:
            return "1D"  # Default fallback

        # Use TV-compatible timeframe conversion
        try:
            return from_seconds(seconds)
        except (ValueError, AssertionError):
            # Fallback to closest standard timeframe if conversion fails
            return "1D"

    def _detect_timeframe_from_data(self, df, requested_timeframe: str) -> str:
        """Detect timeframe from DataFrame by analyzing timestamp differences.
        
        :param df: DataFrame with timestamp column
        :param requested_timeframe: Requested timeframe ("AUTO" for auto-detection)
        :return: Detected or original timeframe
        """
        if requested_timeframe.upper() != "AUTO":
            return requested_timeframe

        try:
            if len(df) < 2:
                return "1D"  # Default fallback

            # Calculate differences between consecutive timestamps
            timestamps = df['timestamp'].values
            differences = []

            for i in range(1, min(len(timestamps), 10)):  # Check first 10 records
                diff = timestamps[i] - timestamps[i - 1]
                if diff > 0:  # Only positive differences
                    differences.append(diff)

            if not differences:
                return "1D"  # Default fallback

            # Find the most common difference (mode)
            diff_counts = Counter(differences)
            most_common_diff = diff_counts.most_common(1)[0][0]

            # Convert seconds to timeframe string (assuming timestamps are in seconds)
            return self._seconds_to_timeframe(most_common_diff)

        except (KeyError, IndexError, ValueError, TypeError):
            # If detection fails, return default
            return "1D"
