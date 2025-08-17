"""
Automatic data file to OHLCV conversion functionality.

This module provides automatic detection and conversion of CSV, TXT, and JSON files
to OHLCV format when needed, eliminating the manual step of running pyne data convert.
"""
from __future__ import annotations

import json
from enum import Enum
from datetime import time
from pathlib import Path
from typing import Literal

from pynecore.core.ohlcv_file import OHLCVWriter, OHLCVReader
from pynecore.utils.file_utils import copy_mtime, is_updated
from ..lib.timeframe import from_seconds
from .syminfo import SymInfo, SymInfoInterval, SymInfoSession


class DataFormatError(Exception):
    """Raised when file format cannot be detected or is unsupported."""
    pass


class ConversionError(Exception):
    """Raised when conversion fails."""
    pass


class SupportedFormats(Enum):
    """Supported data file formats."""
    CSV = 'csv'
    TXT = 'txt'
    JSON = 'json'


class DataConverter:
    """
    Main class for automatic data file conversion.

    Provides both CLI and programmatic interfaces for converting
    CSV, TXT, and JSON files to OHLCV format automatically.
    """

    @staticmethod
    def is_conversion_required(source_path: Path, ohlcv_path: Path | None = None) -> bool:
        """
        Check if conversion is required based on file freshness.

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

    def convert_to_ohlcv(
            self,
            file_path: Path,
            *,
            force: bool = False,
            provider: str | None = None,
            symbol: str | None = None,
            timezone: str = "UTC"
    ) -> None:
        """
        Convert multiple file formats to OHLCV format.

        :param file_path: Path to the data file
        :param force: Force conversion even if OHLCV file is up-to-date
        :param provider: Data provider name for OHLCV file naming
        :param symbol: Symbol for OHLCV file naming
        :param timezone: Timezone for timestamp conversion
        :raises FileNotFoundError: If source file doesn't exist
        :raises DataFormatError: If file format is unsupported
        :raises ConversionError: If conversion fails
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Source file not found: {file_path}")

        # Detect file format
        detected_format = self.detect_format(file_path)

        # If it's already OHLCV, no conversion needed
        if detected_format == 'ohlcv':
            raise ConversionError(f"Source file is already in OHLCV format: {file_path}")

        # Check if format is supported
        if detected_format not in SupportedFormats:
            raise DataFormatError(f"Unsupported file format '{detected_format}' for file: {file_path}")

        # Determine OHLCV output path
        ohlcv_path = file_path.with_suffix('.ohlcv')

        # Check if conversion is needed
        if not force and not self.is_conversion_required(file_path, ohlcv_path):
            return

        # Auto-detect symbol and provider from filename if not provided
        if symbol is None or provider is None:
            detected_symbol, detected_provider = self.guess_symbol_from_filename(file_path)
            if symbol is None:
                symbol = detected_symbol
            if provider is None and detected_provider is not None:
                provider = detected_provider

        # Use default provider if not specified
        if provider is None:
            provider = "CUSTOM"

        analyzed_tick_size = None
        analyzed_price_scale = None
        analyzed_min_move = None
        detected_timeframe = None

        try:
            # Perform conversion directly to target file with truncate to clear existing data
            with OHLCVWriter(ohlcv_path, truncate=True) as ohlcv_writer:
                if detected_format == 'csv':
                    ohlcv_writer.load_from_csv(file_path, tz=timezone)
                elif detected_format == 'json':
                    ohlcv_writer.load_from_json(file_path, tz=timezone)
                # TODO: Add support for TXT!
                # elif detected_format == 'txt':
                #     # Treat TXT files as CSV with different delimiters
                #     ohlcv_writer.load_from_csv(file_path, tz=timezone)
                else:
                    raise ConversionError(f"Unsupported format for conversion: {detected_format}")

                # Get timeframe directly from writer
                if ohlcv_writer.interval is None:
                    raise ConversionError("Cannot determine timeframe from OHLCV file (less than 2 records)")
                try:
                    detected_timeframe = from_seconds(ohlcv_writer.interval)
                except (ValueError, AssertionError):
                    raise ConversionError(
                        f"Cannot convert interval {ohlcv_writer.interval} seconds to valid timeframe")

                # Get analyzed tick size data from writer
                analyzed_tick_size = ohlcv_writer.analyzed_tick_size
                analyzed_price_scale = ohlcv_writer.analyzed_price_scale
                analyzed_min_move = ohlcv_writer.analyzed_min_move

            # Copy modification time from source to maintain freshness
            copy_mtime(file_path, ohlcv_path)

            # Generate TOML symbol info file if needed
            toml_path = file_path.with_suffix('.toml')

            # Skip if TOML file exists and is newer than source (unless force is True)
            if symbol and (force or not toml_path.exists() or is_updated(file_path, toml_path)):
                # Use analyzed values from OHLCVWriter
                if analyzed_tick_size:
                    mintick = analyzed_tick_size
                    pricescale = analyzed_price_scale or int(round(1.0 / analyzed_tick_size))
                    minmove = analyzed_min_move or 1
                else:
                    # Fallback to safe defaults if analysis failed
                    mintick = 0.01
                    pricescale = 100
                    minmove = 1

                # Determine symbol type based on symbol name patterns
                symbol_upper = symbol.upper()
                symbol_type, currency, base_currency = self.guess_symbol_type(symbol_upper)

                # Point value cannot be detected from data, always use 1.0
                # Users can manually adjust in the generated TOML file if needed
                pointvalue = 1.0

                # Get opening hours from OHLCVWriter
                analyzed_opening_hours = ohlcv_writer.analyzed_opening_hours

                if analyzed_opening_hours:
                    # Use automatically detected opening hours
                    opening_hours = analyzed_opening_hours
                else:
                    # Fallback to default based on symbol type (insufficient data or analysis failed)
                    opening_hours = self.get_default_opening_hours(symbol_type)

                # Create session starts and ends
                session_starts = [SymInfoSession(day=1, time=time(0, 0, 0))]
                session_ends = [SymInfoSession(day=7, time=time(23, 59, 59))]

                # Create SymInfo instance
                # Use provider as prefix (uppercase), default to "CUSTOM" if not provided
                prefix = provider.upper() if provider else "CUSTOM"
                syminfo = SymInfo(
                    prefix=prefix,
                    description=f"{symbol}",
                    ticker=symbol_upper,
                    currency=currency,
                    basecurrency=base_currency or "USD",
                    period=detected_timeframe,
                    type=symbol_type,
                    mintick=mintick,
                    pricescale=int(pricescale),
                    minmove=int(minmove),
                    pointvalue=pointvalue,
                    opening_hours=opening_hours,
                    session_starts=session_starts,
                    session_ends=session_ends,
                    timezone=timezone,
                )

                # Save using SymInfo's built-in method
                try:
                    syminfo.save_toml(toml_path)
                    # Copy modification time from source to maintain consistency
                    copy_mtime(file_path, toml_path)
                except (OSError, IOError):
                    # Don't fail the entire conversion if TOML creation fails
                    pass

        except Exception as e:
            # Clean up output file on error
            if ohlcv_path.exists():
                try:
                    ohlcv_path.unlink()
                except OSError:
                    pass
            raise ConversionError(f"Failed to convert {file_path}: {e}") from e

    @staticmethod
    def detect_format(file_path: Path) -> Literal['csv', 'txt', 'json', 'ohlcv', 'unknown']:
        """
        Detect file format by content inspection.

        :param file_path: Path to the file to analyze
        :return: Detected format
        :raises FileNotFoundError: If file doesn't exist
        :raises DataFormatError: If file cannot be read
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # First check if it's a valid OHLCV file (binary format)
        try:
            with OHLCVReader(file_path):
                # If we can open it successfully, it's a valid OHLCV file
                return 'ohlcv'
        except (ValueError, OSError, IOError):
            # Not a valid OHLCV file, detect by content
            pass

        # Detect text-based formats by content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Read first line for initial analysis
                first_line = f.readline().strip()

                # Quick JSON check - look for JSON indicators
                if first_line and (first_line.startswith('{') or first_line.startswith('[')):
                    # Verify it's valid JSON by reading the whole file
                    f.seek(0)
                    try:
                        json.load(f)
                        return 'json'
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        pass
                    # Reset for further analysis if not JSON
                    f.seek(0)
                    first_line = f.readline().strip()

                # Check for CSV patterns
                if first_line and ',' in first_line:
                    # Count commas to see if it looks like structured data
                    comma_count = first_line.count(',')
                    if comma_count >= 4:  # At least OHLC columns
                        return 'csv'

                # Check for other delimiters (TXT)
                if first_line and any(delim in first_line for delim in ['\t', ';', '|']):
                    return 'txt'

                # Default to CSV if it has any commas
                if first_line and ',' in first_line:
                    return 'csv'

                return 'unknown'

        except (OSError, IOError, UnicodeDecodeError):
            return 'unknown'

    @staticmethod
    def get_default_opening_hours(symbol_type: str) -> list[SymInfoInterval]:
        """
        Get default opening hours based on symbol type.

        :param symbol_type: Type of symbol ('crypto', 'forex', 'stock', or 'other')
        :return: List of SymInfoInterval objects representing default trading hours
        """
        opening_hours = []

        if symbol_type == 'crypto':
            # 24/7 trading for crypto
            for day in range(1, 8):
                opening_hours.append(SymInfoInterval(
                    day=day,
                    start=time(0, 0, 0),
                    end=time(23, 59, 59)
                ))
        elif symbol_type == 'forex':
            # Forex markets: Sunday 5 PM ET to Friday 5 PM ET (roughly)
            # Using Monday-Friday 00:00-23:59 as approximation
            for day in range(1, 6):
                opening_hours.append(SymInfoInterval(
                    day=day,
                    start=time(0, 0, 0),
                    end=time(23, 59, 59)
                ))
        else:
            # Stock markets and others: typical business hours (Mon-Fri 9:30 AM - 4:00 PM)
            for day in range(1, 6):
                opening_hours.append(SymInfoInterval(
                    day=day,
                    start=time(9, 30, 0),
                    end=time(16, 0, 0)
                ))

        return opening_hours

    @staticmethod
    def guess_symbol_from_filename(file_path: Path) -> tuple[str | None, str | None]:
        """
        Guess symbol and provider from filename based on common patterns.

        :param file_path: Path to the data file
        :return: Tuple of (symbol, provider) or (None, None) if not detected
        """
        filename = file_path.stem  # Filename without extension
        filename_upper = filename.upper()

        # Known provider patterns - these will be detected first
        provider_patterns = {
            'capitalcom': ['CAPITALCOM'],
            'capital.com': ['CAPITAL.COM', 'CAPITAL_COM'],
            'ccxt': ['CCXT'],
            'tradingview': ['TRADINGVIEW', 'TV'],
            'mt4': ['MT4', 'METATRADER4'],
            'mt5': ['MT5', 'METATRADER5'],
            'binance': ['BINANCE'],
            'bybit': ['BYBIT'],
            'coinbase': ['COINBASE'],
            'kraken': ['KRAKEN'],
            'oanda': ['OANDA'],
            'ib': ['IB', 'INTERACTIVE_BROKERS'],
        }

        # Exchange names for crypto detection
        exchange_names = ['BINANCE', 'BYBIT', 'COINBASE', 'KRAKEN', 'BITFINEX', 'HUOBI', 'OKEX', 'FTX']

        # Crypto bases and quotes for pair detection
        crypto_bases = ['BTC', 'ETH', 'XRP', 'ADA', 'DOT', 'LINK', 'LTC', 'BCH', 'UNI', 'MATIC',
                        'SOL', 'AVAX', 'LUNA', 'ATOM', 'FTM', 'NEAR', 'ALGO', 'VET', 'FIL', 'ICP']
        # Order matters! Longer suffixes first to avoid false matches (USDT before USD)
        crypto_quotes = ['USDT', 'USDC', 'BUSD', 'TUSD', 'DAI', 'USD', 'EUR', 'GBP', 'JPY', 'BTC', 'ETH']

        detected_provider = None
        detected_symbol = None

        # Step 1: Try exchange-based detection first (handles BINANCE_BTC_USDT, BYBIT:BTC:USDT, etc.)
        # Clean separators and split
        cleaned = filename.replace(':', '_').replace('/', '_').replace(',', '_')
        parts = [p.strip() for p in cleaned.split('_') if p.strip()]

        if len(parts) >= 2 and parts[0].upper() in exchange_names:
            # Exchange detected at the beginning
            detected_provider = parts[0].lower()

            if len(parts) >= 3:
                # Could be EXCHANGE_BASE_QUOTE format
                if parts[1].upper() in crypto_bases and parts[2].upper() in crypto_quotes:
                    # Format: BINANCE_BTC_USDT
                    detected_symbol = f"{parts[1].upper()}/{parts[2].upper()}"
                elif parts[1].upper() in crypto_bases:
                    # Maybe compact format in later parts: CCXT_BYBIT_BTC_USDT_USDT_1
                    # Look for quote in remaining parts
                    for i in range(2, len(parts)):
                        if parts[i].upper() in crypto_quotes:
                            detected_symbol = f"{parts[1].upper()}/{parts[i].upper()}"
                            break
                    if not detected_symbol:
                        # No quote found, use the second part as-is
                        detected_symbol = parts[1].upper()
                else:
                    # Check if second part is a compact pair (BTCUSDT)
                    potential = parts[1].upper()
                    for quote in crypto_quotes:
                        if potential.endswith(quote):
                            base = potential[:-len(quote)]
                            if base in crypto_bases:
                                detected_symbol = f"{base}/{quote}"
                                break
                    if not detected_symbol:
                        # Use second part as-is
                        detected_symbol = parts[1].upper()
            elif len(parts) == 2:
                # EXCHANGE_SYMBOL format
                potential = parts[1].upper()
                # Try to detect compact crypto pair
                for quote in crypto_quotes:
                    if potential.endswith(quote):
                        base = potential[:-len(quote)]
                        if base in crypto_bases:
                            detected_symbol = f"{base}/{quote}"
                            break
                if not detected_symbol:
                    detected_symbol = potential

            if detected_symbol:
                return detected_symbol, detected_provider

        # Step 2: Check for explicit provider patterns (handles CAPITALCOM_EURUSD, TV_BTCUSD, etc.)
        # Special case for ccxt_EXCHANGE pattern
        if filename_upper.startswith('CCXT_'):
            # Remove CCXT_ prefix and try to detect exchange and symbol
            temp = filename[5:]  # Remove "CCXT_"
            temp_parts = temp.replace(':', '_').replace('/', '_').split('_')
            if len(temp_parts) >= 2 and temp_parts[0].upper() in exchange_names:
                # Format: CCXT_EXCHANGE_... - provider is the exchange name, not 'ccxt'
                detected_provider = temp_parts[0].lower()  # Use exchange name as provider
                # Try to extract symbol from remaining parts
                if len(temp_parts) >= 3:
                    # Try BASE/QUOTE detection
                    for i in range(1, len(temp_parts) - 1):
                        if temp_parts[i].upper() in crypto_bases:
                            for j in range(i + 1, len(temp_parts)):
                                if temp_parts[j].upper() in crypto_quotes:
                                    detected_symbol = f"{temp_parts[i].upper()}/{temp_parts[j].upper()}"
                                    return detected_symbol, detected_provider
                # Fallback to simple extraction
                detected_symbol = '_'.join(temp_parts[1:]) if len(temp_parts) > 1 else None
                if detected_symbol:
                    return detected_symbol.upper(), detected_provider
            else:
                # No recognized exchange after ccxt_, just use ccxt as provider
                detected_provider = 'ccxt'
                detected_symbol = '_'.join(temp_parts) if temp_parts else None
                if detected_symbol:
                    return detected_symbol.upper(), detected_provider

        for provider, patterns in provider_patterns.items():
            for pattern in patterns:
                if pattern in filename_upper:
                    detected_provider = provider
                    # Remove provider pattern from filename for symbol detection
                    temp_filename = filename
                    for p in patterns:
                        temp_filename = temp_filename.replace(p, '').replace(p.lower(), '').replace(p.capitalize(), '')
                    temp_filename = temp_filename.strip('_').strip('-').strip(',').strip().strip()

                    # TradingView format might have extra parts like ", 30_cbf9d"
                    # First remove everything after comma if present
                    if ',' in temp_filename:
                        temp_filename = temp_filename.split(',')[0].strip()

                    if '_' in temp_filename:
                        temp_parts = temp_filename.split('_')
                        # Filter out hash-like strings and pure numbers
                        symbol_parts = []
                        for part in temp_parts:
                            part = part.strip()
                            if not part:
                                continue
                            # Skip if looks like a hash or timeframe
                            if len(part) <= 6 and any(c.isdigit() for c in part) and any(c.isalpha() for c in part):
                                continue
                            if part.isdigit():
                                continue
                            if part.upper() in ['1M', '5M', '15M', '30M', '60M', '1H', '4H', '1D', '1W', 'DAILY',
                                                'HOURLY', 'WEEKLY']:
                                continue
                            symbol_parts.append(part)
                        if symbol_parts:
                            temp_filename = '_'.join(symbol_parts)

                    if temp_filename:
                        # Try to parse the symbol
                        temp_upper = temp_filename.upper()

                        # Check for forex pair (6 chars, all letters)
                        if len(temp_upper) == 6 and temp_upper.isalpha():
                            detected_symbol = temp_upper
                        # Check for crypto pair
                        elif any(base in temp_upper for base in crypto_bases):
                            for quote in crypto_quotes:
                                if temp_upper.endswith(quote):
                                    base = temp_upper[:-len(quote)]
                                    if base in crypto_bases:
                                        detected_symbol = f"{base}/{quote}"
                                        break
                            if not detected_symbol:
                                detected_symbol = temp_upper
                        else:
                            detected_symbol = temp_upper
                    break
            if detected_provider:
                break

        # Step 3: If no provider detected, try to infer from symbol pattern
        if not detected_provider and not detected_symbol:
            # Remove common suffixes and prefixes
            clean_name = filename
            for suffix in ['_1M', '_5M', '_15M', '_30M', '_60M', '_1H', '_4H', '_1D', '_1W', '_DAILY', '_HOURLY',
                           '_WEEKLY']:
                if clean_name.upper().endswith(suffix):
                    clean_name = clean_name[:len(clean_name) - len(suffix)]
                    break

            clean_upper = clean_name.upper()

            # First check for crypto patterns (more specific)
            for quote in crypto_quotes:
                if clean_upper.endswith(quote):
                    base = clean_upper[:-len(quote)]
                    if base in crypto_bases:
                        detected_symbol = f"{base}/{quote}"
                        detected_provider = 'ccxt'
                        break

            # If not crypto, check for 6-letter forex pair
            if not detected_symbol and len(clean_upper) == 6 and clean_upper.isalpha():
                detected_symbol = clean_upper
                detected_provider = 'forex'

            # If still no match, check for separator-based pairs
            if not detected_symbol:
                # Try underscore or dash separator
                if '_' in clean_name:
                    parts = clean_name.split('_')
                elif '-' in clean_name:
                    parts = clean_name.split('-')
                else:
                    parts = []

                if len(parts) == 2:
                    if len(parts[0]) == 3 and len(parts[1]) == 3 and parts[0].isalpha() and parts[1].isalpha():
                        # Likely forex: EUR_USD or EUR-USD
                        detected_symbol = parts[0].upper() + parts[1].upper()
                        detected_provider = 'forex'
                    elif parts[0].upper() in crypto_bases and parts[1].upper() in crypto_quotes:
                        # Crypto: BTC_USDT or BTC-USDT
                        detected_symbol = f"{parts[0].upper()}/{parts[1].upper()}"
                        detected_provider = 'ccxt'

            # Last resort - if it's a known ticker (must have at least one letter)
            if not detected_symbol and len(clean_upper) >= 3 and clean_upper.isalnum() and any(
                    c.isalpha() for c in clean_upper):
                detected_symbol = clean_upper

        return detected_symbol, detected_provider

    @staticmethod
    def guess_symbol_type(symbol_upper: str) -> tuple[Literal["forex", "crypto", "other"], str, str | None]:
        """
        Guess symbol type and extract currency information based on common patterns.

        :param symbol_upper: Uppercase symbol string
        :return: Tuple of (symbol_type, currency, base_currency)
        """
        # Common forex pairs - check these first for accurate detection
        forex_pairs = {
            'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',
            'EURGBP', 'EURJPY', 'GBPJPY', 'EURCHF', 'EURAUD', 'EURCAD', 'EURNZD',
            'GBPCHF', 'GBPAUD', 'GBPCAD', 'GBPNZD', 'AUDJPY', 'AUDCHF', 'AUDCAD',
            'AUDNZD', 'CADJPY', 'CADCHF', 'NZDJPY', 'NZDCHF', 'NZDCAD', 'CHFJPY',
            'EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF', 'AUD/USD', 'USD/CAD', 'NZD/USD'
        }

        # Common crypto symbols
        crypto_symbols = {
            'BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'DOT', 'DOGE', 'AVAX', 'LUNA', 'SHIB',
            'MATIC', 'UNI', 'LINK', 'LTC', 'ALGO', 'BCH', 'XLM', 'VET', 'ATOM', 'FIL',
            'TRX', 'ETC', 'XMR', 'MANA', 'SAND', 'HBAR', 'EGLD', 'THETA', 'FTM', 'XTZ',
            'AAVE', 'AXS', 'CAKE', 'CRO', 'NEAR', 'KSM', 'ENJ', 'CHZ', 'SUSHI', 'SNX'
        }

        # Initialize default values
        symbol_type: Literal["forex", "crypto", "other"] = 'other'
        currency = 'USD'
        base_currency: str | None = None

        # Clean up separators
        clean_symbol = symbol_upper.replace('_', '').replace('-', '').replace(':', '').strip()

        # Check if it's a direct forex pair match (check both with and without slash)
        if clean_symbol in forex_pairs or symbol_upper in forex_pairs or \
                any(pair.replace('/', '') in clean_symbol for pair in forex_pairs):
            symbol_type = 'forex'
            # Extract currencies from forex pair - more robust extraction
            matched = False
            for pair in forex_pairs:
                clean_pair = pair.replace('/', '')
                # Check both versions
                if clean_pair in clean_symbol or pair == symbol_upper:
                    # Found exact match
                    if '/' in pair:
                        parts = pair.split('/')
                        base_currency = parts[0]
                        currency = parts[1]
                    else:
                        base_currency = pair[:3]
                        currency = pair[3:6]
                    matched = True
                    break

            if not matched:
                # Fallback extraction for forex
                if 'EUR' in clean_symbol:
                    base_currency = 'EUR'
                    remaining = clean_symbol.replace('EUR', '')
                    currency = remaining[:3] if len(remaining) >= 3 else 'USD'
                elif 'GBP' in clean_symbol:
                    base_currency = 'GBP'
                    remaining = clean_symbol.replace('GBP', '')
                    currency = remaining[:3] if len(remaining) >= 3 else 'USD'
                elif clean_symbol.startswith('USD'):
                    base_currency = 'USD'
                    currency = clean_symbol[3:6] if len(clean_symbol) >= 6 else 'EUR'
                else:
                    # Try to extract 3-letter codes
                    base_currency = clean_symbol[:3] if len(clean_symbol) >= 3 else 'EUR'
                    currency = clean_symbol[3:6] if len(clean_symbol) >= 6 else 'USD'

        # Check if symbol contains '/' separator (explicit format)
        elif '/' in symbol_upper:
            parts = symbol_upper.split('/')
            if len(parts) == 2:
                left_part = parts[0].strip()
                right_part = parts[1].strip()

                # Check if it's crypto (contains crypto symbols or stable coins)
                if any(crypto in left_part for crypto in crypto_symbols) or \
                        right_part in ['USDT', 'USDC', 'BUSD', 'DAI', 'UST', 'TUSD']:
                    symbol_type = 'crypto'
                    currency = right_part
                    base_currency = left_part
                # Check if it's forex (both parts are 3-letter currency codes)
                elif len(left_part) == 3 and len(right_part) == 3 and \
                        left_part.isalpha() and right_part.isalpha():
                    symbol_type = 'forex'
                    base_currency = left_part
                    currency = right_part
                else:
                    # Default to crypto for slash notation
                    symbol_type = 'crypto'
                    currency = right_part
                    base_currency = left_part

        # Check if it's crypto by matching known crypto symbols
        elif any(crypto in clean_symbol for crypto in crypto_symbols):
            symbol_type = 'crypto'
            # Try to extract the quote currency
            if 'USDT' in clean_symbol:
                currency = 'USDT'
                base_currency = clean_symbol.replace('USDT', '')
            elif 'USDC' in clean_symbol:
                currency = 'USDC'
                base_currency = clean_symbol.replace('USDC', '')
            elif 'BUSD' in clean_symbol:
                currency = 'BUSD'
                base_currency = clean_symbol.replace('BUSD', '')
            elif 'USD' in clean_symbol:
                currency = 'USD'
                base_currency = clean_symbol.replace('USD', '')
            else:
                # Try to find the crypto part
                for crypto in crypto_symbols:
                    if crypto in clean_symbol:
                        base_currency = crypto
                        currency = clean_symbol.replace(crypto, '') or 'USDT'
                        break
                else:
                    currency = 'USDT'
                    base_currency = clean_symbol

            if not base_currency or base_currency == currency:
                base_currency = clean_symbol[:3] if len(clean_symbol) >= 3 else 'BTC'

        # Check if it looks like a forex pair (6 letters, no special chars)
        elif len(clean_symbol) == 6 and clean_symbol.isalpha():
            # Could be forex like EURUSD or crypto like BTCUSD
            potential_base = clean_symbol[:3]
            potential_quote = clean_symbol[3:6]

            # Common forex currencies
            forex_currencies = {'EUR', 'USD', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD'}

            if potential_base in forex_currencies and potential_quote in forex_currencies:
                symbol_type = 'forex'
                base_currency = potential_base
                currency = potential_quote
            else:
                # Default to other for unknown 6-letter symbols
                symbol_type = 'other'
                currency = 'USD'
                base_currency = None

        else:
            # Default to other for everything else
            symbol_type = 'other'
            currency = 'USD'
            base_currency = None

        return symbol_type, currency, base_currency
