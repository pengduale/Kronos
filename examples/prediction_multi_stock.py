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


def run_prediction(stock_code: str, predict_days: int = 10) -> dict | None:
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
    fig.suptitle('Multi-Stock Kronos Forecast', fontsize=16, fontweight='bold', y=1.01)
    gs = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.45, wspace=0.3)

    for idx, res in enumerate(results):
        ax = fig.add_subplot(gs[idx // cols, idx % cols])

        ax.plot(res['history_dates'], res['history'], color='steelblue', linewidth=1.2, label='History')
        ax.plot(res['forecast_dates'], res['forecast'], color='tomato',
                linewidth=1.5, linestyle='--', marker='o', markersize=3, label='Forecast')

        # Shade forecast region
        ax.axvspan(res['forecast_dates'][0], res['forecast_dates'][-1],
                   alpha=0.08, color='tomato')

        ax.set_title(res['code'], fontsize=12, fontweight='bold')
        ax.set_xlabel('Date', fontsize=8)
        ax.set_ylabel('Price', fontsize=8)
        ax.tick_params(axis='x', labelrotation=30, labelsize=7)
        ax.legend(fontsize=7)
        ax.grid(True, linestyle=':', alpha=0.5)

    # Hide unused subplots
    for idx in range(n, rows * cols):
        fig.add_subplot(gs[idx // cols, idx % cols]).set_visible(False)

    plt.tight_layout()

    if save:
        ensure_output_dir()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = os.path.join(OUTPUT_DIR, f'multi_stock_forecast_{timestamp}.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'[INFO] Figure saved to {out_path}')

    plt.show()


def main():
    """Entry point: predict and plot multiple A-share stocks."""
    stock_codes = [
        '000001',  # Ping An Bank
        '000651',  # Gree Electric
        '600036',  # China Merchants Bank
        '600519',  # Kweichow Moutai
    ]

    predict_days = 10
    results = []

    for code in stock_codes:
        print(f'[INFO] Running prediction for {code} ...')
        res = run_prediction(code, predict_days=predict_days)
        if res is not None:
            results.append(res)
            last_price = res['history'][-1]
            end_price = res['forecast'][-1]
            change_pct = (end_price - last_price) / last_price * 100
            direction = '▲' if change_pct >= 0 else '▼'
            print(f"  {direction} {abs(change_pct):.2f}% over {predict_days} days "
                  f"({last_price:.2f} → {end_price:.2f})")

    if not results:
        print('[ERROR] No valid predictions generated. Check your data directory.')
        return

    plot_multi_stock(results, save=True)


if __name__ == '__main__':
    main()
