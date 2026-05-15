"""Multi-stock batch prediction example using Kronos.

This script demonstrates how to run predictions for multiple stocks
and compare their forecasted trends side-by-side.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from kronos import Kronos
except ImportError:
    print("Kronos not found. Please install or add to PYTHONPATH.")
    sys.exit(1)


DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output', 'multi_stock')

# Default number of forecast days; 10 felt too short for my use case
DEFAULT_PREDICT_DAYS = 20


def ensure_output_dir():
    """Create output directory if it does not exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_stock_csv(stock_code: str) -> pd.DataFrame | None:
    """Load stock OHLCV data from a CSV file.

    Args:
        stock_code: Stock ticker symbol used as the filename prefix.

    Returns:
        DataFrame with a DatetimeIndex and at least a 'close' column,
        or None if the file is not found.
    """
    filepath = os.path.join(DATA_DIR, f"{stock_code}.csv")
    if not os.path.exists(filepath):
        print(f"[WARN] Data file not found: {filepath}")
        return None

    df = pd.read_csv(filepath, parse_dates=['date'], index_col='date')
    df.sort_index(inplace=True)
    df.dropna(subset=['close'], inplace=True)
    return df


def run_prediction(stock_code: str, predict_days: int = DEFAULT_PREDICT_DAYS) -> dict | None:
    """Run Kronos prediction for a single stock.

    Args:
        stock_code: Stock ticker symbol.
        predict_days: Number of trading days to forecast.

    Returns:
        Dictionary with keys 'code', 'history', 'forecast', 'dates'
        or None if data is unavailable.
    """
    df = load_stock_csv(stock_code)
    if df is None or len(df) < 60:
        print(f"[WARN] Insufficient data for {stock_code}, skipping.")
        return None

    close_prices = df['close'].values.astype(np.float64)

    model = Kronos()
    model.fit(close_prices)
    forecast = model.predict(steps=predict_days)

    last_date = df.index[-1]
    future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=predict_days)

    return {
        'code': stock_code,
        'history': close_prices[-60:],          # last 60 days for display
        'history_dates': df.index[-60:],
        'forecast': forecast,
        'forecast_dates': future_dates,
    }


def plot_multi_stock(results: list[dict], save: bool = True):
    """Plot forecasts for multiple stocks in a grid layout.

    Args:
        results: List of prediction result dicts from run_prediction().
        save: If True, save the figure to OUTPUT_DIR.
    """
    n = len(results)
    cols = 2
    rows = (n + 1) // cols

    fig = plt.figure(figsize=(14, 5 * rows))
    fig.suptitle('Multi-Stock Kronos Forecast', fontsize=16, fontweight='