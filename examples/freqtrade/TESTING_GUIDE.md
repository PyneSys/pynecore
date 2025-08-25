# Testing Guide for PyneCore-FreqTrade Integration

This guide shows you how to run and test the PyneCore-FreqTrade integration files.

## Files Overview

### `pynecore_integration.py`

The core integration module that bridges PyneCore and FreqTrade:

- Contains the `run_pynecore()` function
- Converts FreqTrade DataFrames to PyneCore OHLCV format
- Uses PyneCore's `run_iter()` method for proper state management
- Handles Pine Script execution and result collection
- Provides error handling with custom `IntegrationError` exceptions

### `freqtrade_strategy.py`

A complete FreqTrade strategy example that demonstrates:

- How to use PyneCore indicators in a real strategy
- Multiple indicators calculated in a single Pine Script
- Buy/sell signal generation based on Pine Script indicators
- Fallback handling when PyneCore integration fails
- Data loading from both FreqTrade and PyneCore formats

## Goto examples/freqtrade directory

```bash
cd examples/freqtrade
```

## Prerequisites

Before testing, ensure you have:

```bash
#Using pip (x86/Linux)
pip install freqtrade

# Recommended for ARM/Apple Silicon M1/M2/M3:
# install packages
brew install gettext libomp
# download freqtrade
git clone https://github.com/freqtrade/freqtrade.git
# enter downloaded directory 'freqtrade'
cd freqtrade
git checkout stable
# --install, Install freqtrade from scratch
./setup.sh -i
# this will also create a new virtual environment, activate it
source .venv/bin/activate

# Install PyneCore
pip install "pynesys-pynecore[all]"

# Install pandas (required for FreqTrade)
pip install pandas
```

## Testing the Integration

### 1. Test the Strategy (`freqtrade_strategy.py`)

The strategy can be tested in multiple ways:

#### Downloaded Data

**Step 1: Download FreqTrade data**

```bash
# Create FreqTrade user directory
freqtrade create-userdir --userdir user_data

# Download BTC/USDT data
freqtrade download-data --exchange binance --pairs BTC/USDT --timeframe 1h --days 30
```

**Step 2: Run strategy with real data**

```bash
python freqtrade_strategy.py
```

**Step 3: Run FreqTrade backtest**

### 3.1 Copy Strategy Files

```bash
# Copy strategy to FreqTrade strategies folder
cp freqtrade_strategy.py user_data/strategies/PyneCoreStrategy.py
cp pynecore_integration.py user_data/strategies/pynecore_integration.py
```

### 3.2 Generate Configuration

```bash
# Generate a basic configuration file
freqtrade new-config --config user_data/config.json
```

### 3.3 Update Configuration

Edit the generated `user_data/config.json` file:

**Replace the pairlists section:**

```json
// FROM:
"pairlists": [
    {
        "method": "VolumePairList",
        ...
    }
]

// TO:
"pairlists": [
    {
        "method": "StaticPairList"
    }
]
```

**Add a static pair whitelist in the exchange section:**

```json
"exchange": {
    ...
    "pair_whitelist": [
        "BTC/USDT",
        "ETH/USDT"
    ]
}
```

### 3.4 Run Backtest

```bash
# Run backtest with the configured strategy
freqtrade backtesting --strategy PyneCoreStrategy --timeframe 1h
```

## Expected Output

### Successful Integration Test

```
======================================================================
🤖 Testing FreqTrade Strategy with PyneCore
======================================================================

📥 Looking for existing data...
   To download data, use commands from TESTING_GUIDE.md

📊 Found FreqTrade data: user_data/data/binance/BTC_USDT-1h.feather
   Loaded 730 bars
   Date range: 2025-07-26 00:00:00+00:00 to 2025-08-25 09:00:00+00:00

📈 Running strategy...

📊 Results:
   • Total bars: 730
   • Buy signals: 0.0
   • Sell signals: 0.0

📊 Latest indicators:
   • RSI: 32.86
   • BB Upper: $115,029.31
   • BB Lower: $111,102.07

======================================================================
✅ Strategy ready for FreqTrade!
======================================================================
```

### Integration Error Example

```
⚠️ PyneCore integration failed: Invalid Pine Script: missing @pyne decorator
```

## Testing Different Scenarios

### Test Custom Pine Script

Modify the script in `freqtrade_strategy.py` to test your own indicators:

```python
# Replace the indicators_script variable with your Pine Script
indicators_script = '''"""@pyne
"""
from pynecore.lib import script, close, ta

@script.indicator(title="My Custom Indicators", overlay=False)
def main():
    # Your custom indicators here
    sma_20 = ta.sma(close, 20)
    rsi_21 = ta.rsi(close, 21)
    
    return {
        "sma_20": sma_20,
        "rsi_21": rsi_21
    }
'''
```

### Test Error Handling

Test the integration's error handling:

```python
# Test with invalid Pine Script
from pynecore_integration import run_pynecore, IntegrationError

try:
    results = run_pynecore(df, "invalid script")
except IntegrationError as e:
    print(f"✅ Error handling works: {e}")

# Test with empty DataFrame
try:
    results = run_pynecore(pd.DataFrame(), valid_script)
except IntegrationError as e:
    print(f"✅ Empty DataFrame handling works: {e}")
```

## Troubleshooting

### Common Issues

**"No module named 'pynecore'"**

```bash
pip install "pynesys-pynecore[all]>=6.3.2"
```

**"No module named 'freqtrade'"**

- For basic testing: This is optional, the strategy will work without FreqTrade
- For full testing: Please install FreqTrade as per the instructions in https://www.freqtrade.io/en/stable/installation/

**"Invalid Pine Script: missing @pyne decorator"**

- Ensure your Pine Script starts with `"""@pyne"""`
- Check that the script syntax is correct

## Performance Notes

- The integration processes data bar-by-bar using `run_iter()` for proper Pine Script state management
- For large datasets (>10,000 bars), expect processing time of 1-5 seconds
- Multiple indicators in one Pine Script are more efficient than separate scripts
- Results are cached as pandas Series for fast FreqTrade integration
