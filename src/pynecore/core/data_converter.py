"""Automatic data file to OHLCV conversion functionality.

This module provides automatic detection and conversion of CSV, TXT, and JSON files
to OHLCV format when needed, eliminating the manual step of running pyne data convert.
"""

from __future__ import annotations

import json
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pynecore.core.ohlcv_file import OHLCVWriter, STRUCT_FORMAT
from pynecore.utils.file_utils import copy_mtime, is_updated


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
    OHLCV_MAGIC_BYTES = b'\x00\x00\x00\x00'  # First 4 bytes pattern for binary OHLCV

    def __init__(self):
        pass

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
            # Validate if it's actually binary OHLCV format
            try:
                with open(file_path, 'rb') as f:
                    # Check if file size is multiple of 24 (OHLCV record size)
                    file_size = f.seek(0, 2)  # Seek to end
                    f.seek(0)  # Back to start

                    if file_size == 0:
                        return 'ohlcv'  # Empty OHLCV file is valid

                    if file_size % 24 != 0:
                        # Not a valid OHLCV file, likely renamed
                        return DataConverter._detect_content_format(file_path)

                    # Read first record to validate structure
                    first_record = f.read(24)
                    if len(first_record) == 24:
                        try:
                            # Try to unpack as OHLCV record
                            struct.unpack(STRUCT_FORMAT, first_record)
                            return 'ohlcv'
                        except struct.error:
                            # Invalid binary format, likely renamed
                            return DataConverter._detect_content_format(file_path)

                    return 'ohlcv'
            except (OSError, IOError) as e:
                raise DataFormatError(f"Cannot read file {file_path}: {e}")

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
                if not first_line:
                    return 'unknown'

                # Try JSON first (most specific)
                f.seek(0)
                try:
                    json.load(f)
                    return 'json'
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass

                # Check for CSV patterns
                if ',' in first_line:
                    # Count commas to see if it looks like structured data
                    comma_count = first_line.count(',')
                    if comma_count >= 4:  # At least OHLC columns
                        return 'csv'

                # Check for other delimiters (TXT)
                if any(delim in first_line for delim in ['\t', ';', '|']):
                    return 'txt'

                # Default to CSV if it has some structure
                if ',' in first_line:
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
        """Create TOML symbol info file if it doesn't exist or is outdated.
        
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

        # Create TOML content with warnings
        toml_content = f"""# WARNING: This file was auto-generated from price data analysis.
# Please review and adjust the following values according to your broker's specifications:
# - mintick: Minimum price movement (currently: {mintick})
# - pricescale: Price scale factor (currently: {pricescale})
# - minmove: Minimum move in price scale units (currently: {minmove})
# - pointvalue: Market value per price scale unit (currently: {pointvalue})
# - opening_hours: Trading hours may vary by broker and market

[symbol]
prefix = "CUSTOM"
description = "{symbol} - Auto-generated symbol info"
ticker = "{symbol_upper}"
currency = "{currency}"
"""

        # Use smart defaults for currency fields
        if base_currency:
            toml_content += f'basecurrency = "{base_currency}"\n'
        else:
            toml_content += 'basecurrency = "EUR"  # Default base currency - change if needed\n'

        toml_content += f"""period = "{timeframe}"
type = "{symbol_type}"
mintick = {mintick}
pricescale = {pricescale}
minmove = {minmove}
pointvalue = {pointvalue}
timezone = "{timezone}"
volumettype = "base"
#avg_spread =
#taker_fee =
#maker_fee =
#target_price_average =
#target_price_high =
#target_price_low =
#target_price_date =

# Opening hours (24/7 for crypto, business hours for others)
"""

        if symbol_type == 'crypto':
            # 24/7 trading for crypto
            for day in range(1, 8):
                toml_content += f"""[[opening_hours]]
day = {day}
start = "00:00:00"
end = "23:59:59"

"""
        else:
            # Business hours for stocks/forex (Mon-Fri)
            for day in range(1, 6):
                toml_content += f"""[[opening_hours]]
day = {day}
start = "09:30:00"
end = "16:00:00"

"""

        toml_content += """# Session starts
[[session_starts]]
day = 1
time = "00:00:00"

# Session ends
[[session_ends]]
day = 7
time = "23:59:59"
"""

        # Write TOML file
        try:
            with open(toml_path, 'w', encoding='utf-8') as f:
                f.write(toml_content)

            # Copy modification time from source to maintain consistency
            copy_mtime(source_path, toml_path)

        except (OSError, IOError):
            # Don't fail the entire conversion if TOML creation fails
            # Just log the issue (could be added later)
            pass

    def _analyze_price_data(self, source_path: Path) -> dict[str, float]:
        """Analyze price data to calculate trading parameters.
        
        :param source_path: Path to the source data file
        :return: Dictionary with tick_size, price_scale, and min_move
        """
        try:
            # Try to import pandas for data analysis
            try:
                import pandas as pd  # type: ignore
            except ImportError:
                # Pandas not available, use basic CSV analysis
                return self._analyze_price_data_basic(source_path)

            # Determine file format and read data
            if source_path.suffix.lower() == '.csv':
                df = pd.read_csv(source_path, nrows=1000)  # Sample first 1000 rows
            elif source_path.suffix.lower() == '.json':
                df = pd.read_json(source_path, lines=True, nrows=1000)
            else:
                # Default fallback values
                return {'tick_size': 0.01, 'price_scale': 100, 'min_move': 1}

            # Find price columns (common names)
            price_cols = []
            for col in df.columns:
                col_lower = col.lower()
                if any(price_name in col_lower for price_name in ['close', 'price', 'high', 'low', 'open']):
                    price_cols.append(col)

            if not price_cols:
                # No price columns found, use defaults
                return {'tick_size': 0.01, 'price_scale': 100, 'min_move': 1}

            # Use the first price column for analysis
            prices = pd.to_numeric(df[price_cols[0]], errors='coerce').dropna()

            if len(prices) == 0:
                return {'tick_size': 0.01, 'price_scale': 100, 'min_move': 1}

            # Calculate decimal places from price data
            decimal_places = 0
            for price in prices.head(100):  # Check first 100 valid prices
                if pd.notna(price):
                    price_str = str(float(price))
                    if '.' in price_str and not price_str.endswith('.0'):
                        current_decimals = len(price_str.split('.')[1])
                        decimal_places = max(decimal_places, current_decimals)

            # Calculate price scale (10^decimal_places)
            # Use at least 2 decimal places for reasonable trading precision
            decimal_places = max(decimal_places, 2)
            price_scale = 10 ** decimal_places

            # Calculate minimum tick size
            tick_size = 1.0 / price_scale

            # Min move is typically 1 in price scale units
            min_move = 1

            return {
                'tick_size': tick_size,
                'price_scale': price_scale,
                'min_move': min_move
            }

        except (ImportError, ValueError, KeyError, TypeError):
            # If analysis fails, return sensible defaults
            return {'tick_size': 0.01, 'price_scale': 100, 'min_move': 1}

    @staticmethod
    def _analyze_price_data_basic(source_path: Path) -> dict[str, float]:
        """Basic price data analysis without pandas dependency.
        
        :param source_path: Path to the source data file
        :return: Dictionary with tick_size, price_scale, and min_move
        """
        try:
            import csv

            if source_path.suffix.lower() != '.csv':
                return {'tick_size': 0.01, 'price_scale': 100, 'min_move': 1}

            with open(source_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)

                # Find price column
                price_col = None
                for col in reader.fieldnames or []:
                    col_lower = col.lower()
                    if any(price_name in col_lower for price_name in ['close', 'price', 'high', 'low', 'open']):
                        price_col = col
                        break

                if not price_col:
                    return {'tick_size': 0.01, 'price_scale': 100, 'min_move': 1}

                # Analyze first 100 rows
                decimal_places = 0
                count = 0
                for row in reader:
                    if count >= 100:
                        break

                    try:
                        price = float(row[price_col])
                        price_str = f"{price:.10f}".rstrip('0')
                        if '.' in price_str:
                            current_decimals = len(price_str.split('.')[1])
                            decimal_places = max(decimal_places, current_decimals)
                        count += 1
                    except (ValueError, KeyError):
                        continue

                # Calculate parameters
                price_scale = 10 ** decimal_places
                tick_size = 1.0 / price_scale if decimal_places > 0 else 1.0
                min_move = 1

                return {
                    'tick_size': tick_size,
                    'price_scale': price_scale,
                    'min_move': min_move
                }

        except (OSError, IOError, ValueError, KeyError, TypeError):
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
                    
                    # Unpack timestamp (first 8 bytes as uint64)
                    timestamp = struct.unpack('<Q', record[:8])[0]
                    timestamps.append(timestamp)

            if len(timestamps) < 2:
                return "1D"  # Default fallback

            # Calculate differences between consecutive timestamps
            differences = []
            for i in range(1, len(timestamps)):
                diff = timestamps[i] - timestamps[i-1]
                if diff > 0:  # Only positive differences
                    differences.append(diff)

            if not differences:
                return "1D"  # Default fallback

            # Find the most common difference (mode)
            from collections import Counter
            diff_counts = Counter(differences)
            most_common_diff = diff_counts.most_common(1)[0][0]

            # Convert nanoseconds to timeframe string
            return self._nanoseconds_to_timeframe(most_common_diff)

        except (OSError, IOError, struct.error, IndexError, ValueError):
            # If detection fails, return default
            return "1D"

    @staticmethod
    def _nanoseconds_to_timeframe(nanoseconds: int) -> str:
        """Convert nanoseconds difference to timeframe string.
        
        :param nanoseconds: Time difference in nanoseconds
        :return: Timeframe string (e.g., '1m', '5m', '1h', '1D')
        """
        # Convert to seconds
        seconds = nanoseconds / 1_000_000_000

        # Define common timeframes in seconds
        timeframes = [
            (60, "1m"),
            (300, "5m"),
            (900, "15m"),
            (1800, "30m"),
            (3600, "1h"),
            (14400, "4h"),
            (86400, "1D"),
            (604800, "1W"),
            (2592000, "1M"),  # Approximate month
        ]

        # Find the closest match
        best_match = "1D"  # Default
        min_diff = float('inf')
        
        for tf_seconds, tf_string in timeframes:
            diff = abs(seconds - tf_seconds)
            if diff < min_diff:
                min_diff = diff
                best_match = tf_string

        return best_match

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
                diff = timestamps[i] - timestamps[i-1]
                if diff > 0:  # Only positive differences
                    differences.append(diff)
                    
            if not differences:
                return "1D"  # Default fallback
                
            # Find the most common difference (mode)
            from collections import Counter
            diff_counts = Counter(differences)
            most_common_diff = diff_counts.most_common(1)[0][0]
            
            # Convert nanoseconds to timeframe string
            return self._nanoseconds_to_timeframe(most_common_diff)
            
        except (KeyError, IndexError, ValueError, TypeError):
            # If detection fails, return default
            return "1D"
