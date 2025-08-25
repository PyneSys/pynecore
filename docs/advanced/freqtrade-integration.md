<!--
---
weight: 1005
title: "FreqTrade Integration"
description: "How to integrate PyneCore indicators with FreqTrade trading bot"
icon: "integration_instructions"
date: "2025-08-25"
lastmod: "2025-08-25"
draft: false
toc: true
categories: ["Advanced", "Integration"]
tags: ["freqtrade", "integration", "trading-bot", "indicators", "run_iter"]
---
-->

# FreqTrade Integration

**Pine Script meets automated trading.** FreqTrade is one of the most popular open-source cryptocurrency trading bots, known for its reliability and extensive feature set. By integrating PyneCore with FreqTrade, you can leverage the power of Pine Script-compatible indicators and strategies directly in your automated trading workflows.

This integration brings together PyneCore's precise technical analysis calculations with FreqTrade's robust execution engine, giving you the best of both worlds. The integration works through PyneCore's `run_iter()` method, which allows bar-by-bar calculation of indicators - perfectly matching FreqTrade's dataframe-based approach to strategy evaluation.

## Why Integrate PyneCore with FreqTrade

**Superior indicator accuracy with Pine Script compatibility** - PyneCore provides 14-15 digits of precision matching TradingView's Pine Script calculations exactly. This means your backtest results in FreqTrade will align perfectly with what you see on TradingView charts. Traditional Python TA libraries often have subtle calculation differences that can lead to strategy divergence - PyneCore eliminates this problem entirely.

**State management for complex indicators** - Many advanced indicators require maintaining state between calculations - something that's challenging with traditional pandas operations. PyneCore's Persistent variables handle this automatically, making complex indicators like adaptive moving averages or custom oscillators straightforward to implement.

**Port existing Pine Script strategies** - If you've developed a successful strategy on TradingView, you can now run it live through FreqTrade with minimal modifications.

## Typical Workflow

PyneCore fits seamlessly into your existing FreqTrade development process:

1. **Install FreqTrade** - Set up your trading bot environment
2. **Configure exchange API keys** - For live trading (optional for backtesting)
3. **Download historical data** - For backtesting your strategies
4. **Create/modify strategy with indicators** ← **PyneCore fits HERE**
5. **Backtest strategy** - Test on historical data
6. **Optimize parameters** - Fine-tune your strategy
7. **Run live** - Deploy your strategy

PyneCore enhances step 4 by letting you use Pine Script indicators directly in your FreqTrade strategies, giving you access to TradingView's indicator ecosystem.

## Quick Start

1. **Convert** FreqTrade DataFrame → PyneCore OHLCV format
2. **Execute** Pine Script using `run_iter()` method  
3. **Return** results as pandas Series for FreqTrade

## Integration Approach

You have two options for integrating PyneCore with FreqTrade:

1. **Self-contained** (recommended): Add the integration code directly to your strategy class
2. **Modular**: Use the separate `pynecore_integration.py` module from `examples/freqtrade/`

The self-contained approach shown below is simpler - everything you need is in one file.

## FreqTrade Strategy Example

Add PyneCore integration directly to your FreqTrade strategy file. No separate imports needed - everything is self-contained:

```python
from freqtrade.strategy import IStrategy
import pandas as pd
import os
import tempfile
from pathlib import Path
from pynecore.core.script_runner import ScriptRunner
from pynecore.core.syminfo import SymInfo
from pynecore.types.ohlcv import OHLCV

class PyneCoreStrategy(IStrategy):
    INTERFACE_VERSION = 3
    minimal_roi = {"0": 0.10, "30": 0.05, "60": 0.02, "120": 0.01}
    stoploss = -0.05
    timeframe = '1h'
    startup_candle_count = 50

    def run_pynecore(self, dataframe: pd.DataFrame, script_content: str, 
                     pair: str = "BTCUSDT", timeframe_minutes: int = 60) -> dict[str, pd.Series]:
        """
        Run PyneCore script on FreqTrade DataFrame.
        
        :param dataframe: FreqTrade DataFrame with OHLCV data
        :param script_content: Pine Script code to execute
        :param pair: Trading pair (e.g., "BTCUSDT")
        :param timeframe_minutes: Timeframe in minutes (60 for 1h, 240 for 4h, etc.)
        :return: Dictionary mapping indicator names to pandas Series
        """
        # Convert DataFrame to OHLCV format
        ohlcv_list = []
        for idx, row in dataframe.iterrows():
            timestamp = int(idx.timestamp()) if isinstance(idx, pd.Timestamp) else idx
            ohlcv = OHLCV(
                timestamp=timestamp,
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=float(row['volume'])
            )
            ohlcv_list.append(ohlcv)
        
        # Create SymInfo for the trading pair
        syminfo = SymInfo(
            prefix="FREQTRADE",
            description=f"Crypto pair {pair}",
            ticker=pair,
            currency="USDT" if "USDT" in pair else "USD",
            period=str(timeframe_minutes),  # SymInfo expects string
            type="crypto",
            mintick=0.01,
            pricescale=100,
            pointvalue=1.0,
            opening_hours=[],
            session_starts=[],
            session_ends=[]
        )
        
        # Write script to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write(script_content)
            temp_path = temp_file.name
        
        try:
            # Create ScriptRunner and execute with run_iter()
            runner = ScriptRunner(
                script_path=Path(temp_path),
                ohlcv_iter=ohlcv_list,
                syminfo=syminfo,
                last_bar_index=len(ohlcv_list) - 1
            )
            
            # Collect results using run_iter() - the key integration method
            results = {}
            for ohlcv, plot_data in runner.run_iter():
                for key, value in plot_data.items():
                    if key not in results:
                        results[key] = []
                    results[key].append(value)
            
            # Convert to pandas Series
            return {
                key: pd.Series(values, index=dataframe.index[:len(values)])
                for key, values in results.items()
            }
        finally:
            os.unlink(temp_path)

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Pine Script with multiple indicators
        indicators_script = '''
        """@pyne"""
        from pynecore.lib import script, close, ta

        @script.indicator(title="Strategy Indicators")
        def main():
            ema_fast = ta.ema(close, 20)
            ema_slow = ta.ema(close, 50)
            rsi = ta.rsi(close, 14)
            macd_line = ta.ema(close, 12) - ta.ema(close, 26)
            macd_signal = ta.ema(macd_line, 9)
            bb_basis = ta.sma(close, 20)
            bb_dev = 2.0 * ta.stdev(close, 20)
            
            return {
                "ema_fast": ema_fast,
                "ema_slow": ema_slow,
                "rsi": rsi,
                "macd": macd_line,
                "macd_signal": macd_signal,
                "bb_upper": bb_basis + bb_dev,
                "bb_lower": bb_basis - bb_dev
            }
        '''

        # Run PyneCore and add indicators
        # For 1h timeframe, pass 60 minutes. Adjust based on your strategy's timeframe
        results = self.run_pynecore(dataframe, indicators_script, metadata['pair'], timeframe_minutes=60)
        for name, values in results.items():
            dataframe[name] = values
        
        dataframe['trend_up'] = dataframe['ema_fast'] > dataframe['ema_slow']
        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[
            (
                (dataframe['rsi'] < 30) &
                (dataframe['close'] <= dataframe['bb_lower'] * 1.01) &
                (dataframe['macd'] > dataframe['macd_signal']) &
                (dataframe['trend_up'])
            ),
            'enter_long'
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[
            (
                (dataframe['rsi'] > 70) &
                (dataframe['close'] >= dataframe['bb_upper'] * 0.99) &
                (dataframe['macd'] < dataframe['macd_signal'])
            ),
            'exit_long'
        ] = 1
        return dataframe
```

## Key Integration Steps

1. **Data Conversion**: Convert FreqTrade DataFrame to PyneCore OHLCV format
2. **Symbol Info**: Create SymInfo object for the trading pair
3. **Script Execution**: Use `run_iter()` to process data bar-by-bar
4. **Result Collection**: Convert PyneCore outputs back to pandas Series

## Benefits

- Use existing Pine Script indicators without modification
- Access PyneCore's comprehensive indicator library
- Consistent results with TradingView Pine Script
- Easy to add new indicators

## Error Handling

Include proper error handling in your strategy:

```python
try:
    results = run_pynecore(dataframe, script, pair)
except Exception as e:
    self.logger.error(f"PyneCore failed: {e}")
    # Use fallback indicators
```

For more advanced usage, see [AST Transformations](./ast-transformations.md) to understand how PyneCore processes Pine Script syntax.