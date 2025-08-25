"""
Minimal PyneCore-FreqTrade Integration.

This module provides a simple bridge between PyneCore's Pine Script engine
and FreqTrade's DataFrame-based workflow using the run_iter() method.
"""

import os
import tempfile
from pathlib import Path

import pandas as pd
from pynecore.core.script_runner import ScriptRunner
from pynecore.core.syminfo import SymInfo
from pynecore.types.ohlcv import OHLCV


class IntegrationError(Exception):
    """Custom exception for PyneCore-FreqTrade integration errors."""
    pass


def run_pynecore(
    dataframe: pd.DataFrame,
    script_content: str,
    pair: str = "BTCUSDT",
    timeframe: str = "1h"
) -> dict[str, pd.Series]:
    """
    Run a PyneCore Pine Script on a FreqTrade DataFrame.
    
    :param dataframe: FreqTrade DataFrame with OHLCV columns
    :param script_content: Pine Script code as a string
    :param pair: Trading pair (e.g., BTCUSDT)
    :param timeframe: Timeframe (e.g., 1h, 4h, 1d)
    :return: Dictionary of indicator results as pandas Series
    :raises IntegrationError: When DataFrame conversion or script execution fails
    """
    if dataframe.empty:
        raise IntegrationError("Cannot process empty DataFrame")
    
    if not script_content or "@pyne" not in script_content:
        raise IntegrationError("Invalid Pine Script: missing @pyne decorator")
    
    try:
        # Convert DataFrame to OHLCV list
        ohlcv_data = _dataframe_to_ohlcv(dataframe)
        
        # Create SymInfo for crypto
        syminfo = _create_syminfo(pair, timeframe)
        
        # Execute script and collect results
        return _execute_script(script_content, ohlcv_data, syminfo, dataframe.index)
        
    except Exception as e:
        raise IntegrationError(f"Failed to run PyneCore: {e}") from e


def _dataframe_to_ohlcv(dataframe: pd.DataFrame) -> list[OHLCV]:
    """
    Convert FreqTrade DataFrame to PyneCore OHLCV format.
    
    :param dataframe: DataFrame with OHLC columns
    :return: List of OHLCV namedtuples
    :raises ValueError: When required columns are missing
    """
    ohlcv_list = []
    
    for idx, row in dataframe.iterrows():
        timestamp = int(idx.timestamp()) if isinstance(idx, pd.Timestamp) else idx
        
        ohlcv = OHLCV(
            timestamp=timestamp,
            open=float(row.get('open', row.get('Open', 0))),
            high=float(row.get('high', row.get('High', 0))),
            low=float(row.get('low', row.get('Low', 0))),
            close=float(row.get('close', row.get('Close', 0))),
            volume=float(row.get('volume', row.get('Volume', 0)))
        )
        ohlcv_list.append(ohlcv)
    
    return ohlcv_list


def _create_syminfo(pair: str, timeframe: str) -> SymInfo:
    """
    Create SymInfo object for cryptocurrency trading.
    
    :param pair: Trading pair
    :param timeframe: Timeframe string
    :return: Configured SymInfo object
    """
    # Convert timeframe to minutes
    timeframe_minutes = {
        "1m": "1", "5m": "5", "15m": "15", "30m": "30",
        "1h": "60", "4h": "240", "1d": "1440"
    }.get(timeframe, "60")
    
    return SymInfo(
        prefix="FREQTRADE",
        description=f"Crypto pair {pair}",
        ticker=pair,
        currency="USDT" if "USDT" in pair else "USD",
        period=timeframe_minutes,
        type="crypto",
        mintick=0.01,
        pricescale=100,
        pointvalue=1.0,
        opening_hours=[],  # 24/7 trading
        session_starts=[],
        session_ends=[]
    )


def _execute_script(
    script_content: str,
    ohlcv_data: list[OHLCV],
    syminfo: SymInfo,
    index: pd.Index
) -> dict[str, pd.Series]:
    """
    Execute Pine Script and collect results using run_iter().
    
    :param script_content: Pine Script code
    :param ohlcv_data: List of OHLCV data
    :param syminfo: Symbol information
    :param index: Original DataFrame index
    :return: Dictionary of indicator results
    :raises RuntimeError: When script execution fails
    """
    # Write script to temporary file
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.py', delete=False
    ) as temp_file:
        temp_file.write(script_content)
        temp_path = temp_file.name
    
    try:
        # Create ScriptRunner with run_iter support
        runner = ScriptRunner(
            script_path=Path(temp_path),
            ohlcv_iter=ohlcv_data,
            syminfo=syminfo,
            last_bar_index=len(ohlcv_data) - 1
        )
        
        # Collect results using run_iter() - key integration method
        results: dict[str, list] = {}
        for ohlcv, plot_data in runner.run_iter():
            for key, value in plot_data.items():
                if key not in results:
                    results[key] = []
                results[key].append(value)
        
        # Convert to pandas Series
        return {
            key: pd.Series(values, index=index[:len(values)])
            for key, values in results.items()
        }
        
    except Exception as e:
        raise RuntimeError(f"Script execution failed: {e}") from e
        
    finally:
        # Clean up temp file
        os.unlink(temp_path)