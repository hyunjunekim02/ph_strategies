#!/usr/bin/env python3
"""
Persistent Homology–Driven Trading Strategies
==============================================
CLI entry point.

Examples
--------
    python main.py --list-assets
    python main.py --asset snp500 --window 45
    python main.py --asset msft --all-windows
    python main.py --asset ionq -w 30 --data-dir ./my_data
"""

import argparse
import sys
import os

import numpy as np
import pandas as pd

from src.core import compute_features_for_window
from src.strategies import (
    strategy_bh, strategy1, strategy2, strategy3, strategy4,
    strategy5, optimize_strategy5,
)
from src.visualization import plot_results, plot_strategy
from src.evaluation import run_strategy, compare_all_strategies, plot_best_strategy


# ── Configuration ─────────────────────────────────────────────────────────

ASSETS = {
    "snp500": {"file": "snp500.csv", "name": "S&P 500"},
    "nasdaq": {"file": "nasdaq.csv", "name": "NASDAQ"},
    "aapl":   {"file": "aapl.csv",   "name": "Apple"},
    "msft":   {"file": "msft.csv",   "name": "Microsoft"},
    "ionq":   {"file": "ionq.csv",   "name": "IonQ"},
    "pltr":   {"file": "pltr.csv",   "name": "Palantir"},
}

VALID_WINDOWS = [15, 30, 45, 60, 75, 90]

STRATEGY_REGISTRY = {
    "Strategy 1": strategy1,
    "Strategy 2": strategy2,
    "Strategy 3": strategy3,
    "Strategy 4": strategy4,
    # Strategy 5 is handled separately (needs grid search)
}


# ── Data loading ──────────────────────────────────────────────────────────

def load_data(asset_key, data_dir="data"):
    """Load and preprocess a CSV price file."""
    info = ASSETS[asset_key]
    path = os.path.join(data_dir, info["file"])
    if not os.path.isfile(path):
        sys.exit(
            f"[ERROR] File not found: {path}\n"
            f"        Place your CSV in the '{data_dir}/' directory."
        )

    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Price"] = (
        df["Price"].astype(str).str.replace(",", "").str.strip().astype(float)
    )
    df = df.sort_values("Date").reset_index(drop=True)
    return df, info["name"]


# ── Strategy execution ────────────────────────────────────────────────────

def _run_all_strategies(res, verbose=True):
    """Run S1–S5 and return {name: DataFrame}."""
    strats = {}
    for name, func in STRATEGY_REGISTRY.items():
        strats[name] = run_strategy(res, strategy_bh, func)

    # S5: grid-search for optimal buy/sell fractions
    grid_df, best_buy, best_sell = optimize_strategy5(res["df_window"])
    if verbose:
        print(f"  S5 grid search → buy={best_buy:.1f}, sell={best_sell:.1f}")

    strats["Strategy 5"] = run_strategy(
        res, strategy_bh,
        lambda dfw, cap: strategy5(dfw, cap, best_buy, best_sell),
    )
    return strats


def _best_from(strats):
    """Return (name, final_value, bh_value) of the best strategy."""
    best_name, best_val, bh_val = None, -np.inf, None
    for name, df_s in strats.items():
        val = df_s["Strategy_Value"].iloc[-1]
        if val > best_val:
            best_name, best_val = name, val
            bh_val = df_s["BH_Value"].iloc[-1]
    return best_name, best_val, bh_val


# ── Run modes ─────────────────────────────────────────────────────────────

def _print_header(asset_name, subtitle, df_raw):
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  {asset_name}  —  {subtitle}")
    print(f"  {df_raw['Date'].iloc[0].date()} ~ {df_raw['Date'].iloc[-1].date()}"
          f"  ({len(df_raw):,} points)")
    print(sep)


def run_single(asset_key, W, data_dir="data"):
    """Full pipeline for one (asset, window) pair."""
    df_raw, name = load_data(asset_key, data_dir)
    _print_header(name, f"W = {W}", df_raw)

    print("\n[1/3] Computing PH features …")
    res = compute_features_for_window(df_raw, W)
    n_anom = int((res["preds"] == -1).sum()) if len(res["preds"]) else 0
    print(f"       {n_anom} anomalies detected")

    print("\n[2/3] Plotting anomaly diagnostics …")
    plot_results(
        df_raw, res["window_dates"], res["distances"],
        res["anomaly_idx"], res["scores"], W, name,
    )

    print("\n[3/3] Running strategies S1–S5 …")
    strats = _run_all_strategies(res)
    compare_all_strategies(strats)
    plot_best_strategy(strats, W, name)


def run_all_windows(asset_key, data_dir="data"):
    """Sweep all window sizes and summarise."""
    df_raw, name = load_data(asset_key, data_dir)
    _print_header(name, "Full Window Sweep", df_raw)

    rows = []
    for W in VALID_WINDOWS:
        print(f"\n── W = {W} {'─' * 40}")
        res = compute_features_for_window(df_raw, W)
        strats = _run_all_strategies(res, verbose=False)
        bname, bval, bhval = _best_from(strats)
        ratio = (bval - bhval) / bhval * 100
        rows.append({"W": W, "Best": bname, "BH": bhval,
                      "Value": bval, "ratio": ratio})
        print(f"  → {bname}  {ratio:+.2f}% vs BH")

    # Summary table
    sweep = pd.DataFrame(rows)
    print(f"\n{'=' * 60}")
    print(f"  WINDOW SIZE COMPARISON — {name}")
    print("=" * 60)
    fmt = sweep.copy()
    fmt["BH"] = fmt["BH"].map("${:,.2f}".format)
    fmt["Value"] = fmt["Value"].map("${:,.2f}".format)
    fmt["Outperformance"] = fmt["ratio"].map("{:+.2f}%".format)
    print(fmt[["W", "Best", "BH", "Value", "Outperformance"]].to_string(index=False))

    # Plot the overall best
    best_row = sweep.loc[sweep["ratio"].idxmax()]
    best_W = int(best_row["W"])
    print(f"\n>>> Overall best: W={best_W}, {best_row['Best']} "
          f"({best_row['ratio']:+.2f}%)")
    print(">>> Plotting best configuration …\n")

    res = compute_features_for_window(df_raw, best_W)
    strats = _run_all_strategies(res, verbose=False)
    plot_best_strategy(strats, best_W, name)


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Persistent Homology–Driven Trading Strategies",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--asset", type=str,
                        help=f"Asset key: {', '.join(ASSETS)}")
    parser.add_argument("--window", "-w", type=int, default=45,
                        help=f"Window size (default 45). Options: {VALID_WINDOWS}")
    parser.add_argument("--all-windows", action="store_true",
                        help="Sweep all window sizes for the given asset.")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="CSV data directory (default: data/).")
    parser.add_argument("--list-assets", action="store_true",
                        help="Print available asset keys and exit.")

    args = parser.parse_args()

    if args.list_assets:
        print("\nAvailable assets:")
        print("-" * 35)
        for key, info in ASSETS.items():
            print(f"  {key:10s}  {info['name']}")
        print(f"\nCSV files should be in '{args.data_dir}/'.")
        return

    if args.asset is None:
        parser.print_help()
        sys.exit("\n[ERROR] --asset is required. Use --list-assets for options.")

    if args.asset not in ASSETS:
        sys.exit(f"[ERROR] Unknown asset '{args.asset}'. "
                 f"Use --list-assets for options.")

    if args.window not in VALID_WINDOWS:
        sys.exit(f"[ERROR] Invalid window {args.window}. "
                 f"Options: {VALID_WINDOWS}")

    if args.all_windows:
        run_all_windows(args.asset, args.data_dir)
    else:
        run_single(args.asset, args.window, args.data_dir)


if __name__ == "__main__":
    main()
