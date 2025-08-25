"""
Example FreqTrade Strategy using PyneCore indicators.

This demonstrates how to use PyneCore's Pine Script indicators
in a real FreqTrade strategy with proper data downloading.
"""

import json
from pathlib import Path

import pandas as pd

try:
    from freqtrade.strategy import IStrategy

    FREQTRADE_AVAILABLE = True
except ImportError:
    # Allow testing without FreqTrade
    FREQTRADE_AVAILABLE = False


    class IStrategy:
        """Dummy IStrategy for testing without FreqTrade."""
        INTERFACE_VERSION = 3
        logger = None

from pynecore_integration import run_pynecore, IntegrationError


class PyneCoreStrategy(IStrategy):
    """
    FreqTrade strategy powered by PyneCore Pine Script indicators.
    
    This strategy calculates multiple indicators using a single
    PyneCore script and generates buy/sell signals based on them.
    """

    # Strategy configuration
    INTERFACE_VERSION = 3
    minimal_roi = {"0": 0.10, "30": 0.05, "60": 0.02, "120": 0.01}
    stoploss = -0.05
    timeframe = '1h'

    # Optional: Add these for better compatibility
    can_short = False
    startup_candle_count = 50  # Number of candles needed before generating signals

    def populate_indicators(
            self, dataframe: pd.DataFrame, metadata: dict
    ) -> pd.DataFrame:
        """
        Calculate all indicators using PyneCore.

        :param dataframe: OHLCV DataFrame from exchange
        :param metadata: Additional information (pair, timeframe)
        :return: DataFrame with indicators added
        """
        pair = metadata.get('pair', 'BTC/USDT')

        # Define all indicators in ONE Pine Script
        indicators_script = '''"""
@pyne
"""
from pynecore.lib import script, close, high, low, volume, ta

@script.indicator(title="Strategy Indicators", overlay=False)
def main():
    # Trend indicators
    ema_fast = ta.ema(close, 20)
    ema_slow = ta.ema(close, 50)
    
    # Momentum
    rsi = ta.rsi(close, 14)
    
    # MACD
    macd_line = ta.ema(close, 12) - ta.ema(close, 26)
    macd_signal = ta.ema(macd_line, 9)
    macd_histogram = macd_line - macd_signal
    
    # Bollinger Bands
    bb_basis = ta.sma(close, 20)
    bb_dev = 2.0 * ta.stdev(close, 20)
    bb_upper = bb_basis + bb_dev
    bb_lower = bb_basis - bb_dev
    
    # Volume indicator
    volume_sma = ta.sma(volume, 20)
    
    # ATR for stops
    atr = ta.atr(14)
    
    # Stochastic
    lowest = ta.lowest(low, 14)
    highest = ta.highest(high, 14)
    stoch_k = 100 * (close - lowest) / (highest - lowest)
    stoch_d = ta.sma(stoch_k, 3)
    
    return {
        "ema_fast": ema_fast,
        "ema_slow": ema_slow,
        "rsi": rsi,
        "macd": macd_line,
        "macd_signal": macd_signal,
        "macd_histogram": macd_histogram,
        "bb_upper": bb_upper,
        "bb_middle": bb_basis,
        "bb_lower": bb_lower,
        "volume_sma": volume_sma,
        "atr": atr,
        "stoch_k": stoch_k,
        "stoch_d": stoch_d
    }
'''

        try:
            # Run PyneCore and get all indicators at once
            results = run_pynecore(
                dataframe, indicators_script,
                pair=pair, timeframe=self.timeframe
            )

            # Add all indicators to dataframe
            for indicator_name, indicator_values in results.items():
                dataframe[indicator_name] = indicator_values

            # Calculate additional signals
            dataframe['trend_up'] = dataframe['ema_fast'] > dataframe['ema_slow']

        except IntegrationError as exception:
            if self.logger:
                self.logger.error(f"PyneCore integration failed: {exception}")
            else:
                print(f"⚠️ PyneCore integration failed: {exception}")
            # Fallback to basic indicators if needed
            dataframe['rsi'] = 50  # Neutral RSI
            dataframe['trend_up'] = False

        return dataframe

    @staticmethod
    def populate_entry_trend(
            dataframe: pd.DataFrame,
            metadata: dict
    ) -> pd.DataFrame:
        """
        Define entry signals (buy/long) based on PyneCore indicators.

        This is the new v3 interface method name (replaces populate_buy_trend).

        :param dataframe: DataFrame with indicators
        :param metadata: Additional information
        :return: DataFrame with enter_long column
        """
        # Check if required columns exist
        required = ['rsi', 'bb_lower', 'macd', 'macd_signal', 'trend_up', 'volume_sma']
        if all(col in dataframe.columns for col in required):
            dataframe.loc[
                (
                        (dataframe['rsi'] < 30) &
                        (dataframe['close'] <= dataframe['bb_lower'] * 1.01) &
                        (dataframe['macd'] > dataframe['macd_signal']) &
                        (dataframe['trend_up'] == True) &
                        (dataframe['volume'] > dataframe['volume_sma'])
                ),
                'enter_long'  # Changed from 'buy' to 'enter_long'
            ] = 1
        else:
            # No entry signals if indicators are missing
            dataframe['enter_long'] = 0

        # If you want to support shorting in the future:
        # dataframe['enter_short'] = 0

        return dataframe

    @staticmethod
    def populate_exit_trend(
            dataframe: pd.DataFrame, metadata: dict
    ) -> pd.DataFrame:
        """
        Define exit signals (sell) based on PyneCore indicators.

        This is the new v3 interface method name (replaces populate_sell_trend).

        :param dataframe: DataFrame with indicators
        :param metadata: Additional information
        :return: DataFrame with exit_long column
        """
        # Check if required columns exist
        required = ['rsi', 'bb_upper', 'macd', 'macd_signal']
        if all(col in dataframe.columns for col in required):
            dataframe.loc[
                (
                        (dataframe['rsi'] > 70) &
                        (dataframe['close'] >= dataframe['bb_upper'] * 0.99) &
                        (dataframe['macd'] < dataframe['macd_signal'])
                ),
                'exit_long'  # Changed from 'sell' to 'exit_long'
            ] = 1
        else:
            # No exit signals if indicators are missing
            dataframe['exit_long'] = 0

        # If you want to support shorting in the future:
        # dataframe['exit_short'] = 0

        return dataframe


# Note: Data download functions removed - use command line instead
# See TESTING_GUID.md for data download instructions


def load_freqtrade_data(
        data_dir: Path,
        pair: str = "BTC/USDT",
        timeframe: str = "1h"
) -> pd.DataFrame:
    """
    Load data from FreqTrade's format (supports both .feather and .json).

    FreqTrade saves data as .feather or .json files in user_data/data/<exchange>/.

    :param data_dir: Path to FreqTrade data directory
    :param pair: Trading pair
    :param timeframe: Timeframe
    :return: DataFrame with OHLCV columns
    :raises ValueError: When data file not found
    """
    try:
        # FreqTrade file naming: BTC_USDT-1h.feather or BTC_USDT-1h.json
        pair_formatted = pair.replace("/", "_")
        feather_path = data_dir / f"{pair_formatted}-{timeframe}.feather"
        json_path = data_dir / f"{pair_formatted}-{timeframe}.json"

        # Try .feather first (newer FreqTrade format)
        if feather_path.exists():
            dataframe = pd.read_feather(feather_path)
            # Feather format already has proper datetime index
            if 'date' in dataframe.columns:
                dataframe['date'] = pd.to_datetime(dataframe['date'])
                dataframe.set_index('date', inplace=True)
            return dataframe

        # Fallback to .json format (older FreqTrade format)
        elif json_path.exists():
            with open(json_path, 'r') as f:
                data = json.load(f)

            dataframe = pd.DataFrame(data)
            dataframe['date'] = pd.to_datetime(dataframe['date'], unit='ms')
            dataframe.set_index('date', inplace=True)
            return dataframe

        else:
            raise ValueError(f"FreqTrade data file not found: {feather_path} or {json_path}")

    except Exception as exception:
        raise ValueError(f"Failed to load FreqTrade data: {exception}") from exception


# Standalone testing
if __name__ == "__main__":
    print("=" * 70)
    print("🤖 Testing FreqTrade Strategy with PyneCore")
    print("=" * 70)

    # Try to load existing data
    df = None

    print("\n📥 Looking for existing data...")
    print("   To download data, use commands from TESTING_GUIDE.md")

    # Option 1: Try to load FreqTrade data if it exists
    freqtrade_feather_path = Path("user_data/data/binance/BTC_USDT-1h.feather")
    freqtrade_json_path = Path("user_data/data/binance/BTC_USDT-1h.json")

    if freqtrade_feather_path.exists() or freqtrade_json_path.exists():
        try:
            data_path = freqtrade_feather_path if freqtrade_feather_path.exists() else freqtrade_json_path
            print(f"\n📊 Found FreqTrade data: {data_path}")
            df = load_freqtrade_data(data_path.parent, "BTC/USDT", "1h")
            print(f"   Loaded {len(df)} bars")
            print(f"   Date range: {df.index[0]} to {df.index[-1]}")
        except Exception as e:
            print(f"   ⚠️ Failed to load FreqTrade data: {e}")

    # Option 2: Use sample data if no data found
    if df is None:
        print("\n📊 Using sample data for demonstration...")

        # Fallback to sample data for testing
        import numpy as np

        dates = pd.date_range('2025-01-01', periods=500, freq='h')

        # Generate more realistic price data
        np.random.seed(42)
        close_prices = 50000 + np.cumsum(np.random.randn(500) * 100)

        df = pd.DataFrame({
            'open': close_prices * (1 + np.random.randn(500) * 0.001),
            'high': close_prices * (1 + np.abs(np.random.randn(500)) * 0.005),
            'low': close_prices * (1 - np.abs(np.random.randn(500)) * 0.005),
            'close': close_prices,
            'volume': 1000000 * (1 + np.random.rand(500))
        }, index=dates)

        # Ensure OHLC relationships are valid
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)

    # Test strategy
    if FREQTRADE_AVAILABLE:
        # FreqTrade requires a config dict
        config = {
            'strategy': 'PyneCoreStrategy',
            'stake_currency': 'USDT',
            'dry_run': True
        }
        strategy = PyneCoreStrategy()
    else:
        # Use without config for testing
        strategy = PyneCoreStrategy()

    print("\n📈 Running strategy...")
    df = strategy.populate_indicators(df, {'pair': 'BTC/USDT'})

    # Use the new v3 interface methods for testing
    df = strategy.populate_entry_trend(df, {'pair': 'BTC/USDT'})
    df = strategy.populate_exit_trend(df, {'pair': 'BTC/USDT'})

    # Results - check for new column names
    buy_signals = df['enter_long'].sum() if 'enter_long' in df else 0
    sell_signals = df['exit_long'].sum() if 'exit_long' in df else 0

    print(f"\n📊 Results:")
    print(f"   • Total bars: {len(df)}")
    print(f"   • Entry (buy) signals: {buy_signals}")
    print(f"   • Exit (sell) signals: {sell_signals}")

    if 'rsi' in df and 'bb_upper' in df:
        print(f"\n📊 Latest indicators:")
        print(f"   • RSI: {df['rsi'].iloc[-1]:.2f}")
        print(f"   • BB Upper: ${df['bb_upper'].iloc[-1]:,.2f}")
        print(f"   • BB Lower: ${df['bb_lower'].iloc[-1]:,.2f}")

    print("\n" + "=" * 70)
    print("✅ Strategy ready for FreqTrade!")
    print("=" * 70)
