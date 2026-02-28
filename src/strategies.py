"""
Trading Strategies Module
=========================
PH-driven trading strategies (S0–S5) and grid-search optimisation for S5.

Design hierarchy:
  S0  Buy & Hold (baseline)
  S1  Full exit on any anomaly
  S2  Exit on positive anomaly; re-enter on negative anomaly
  S3  Delayed re-entry with fixed cool-down
  S4  Adaptive re-entry (volatility-based cool-down)
  S5  Partial buy/sell with adaptive cool-down
"""

import numpy as np
import pandas as pd


# ── Shared helpers ────────────────────────────────────────────────────────

def _label_anomalies(df):
    """
    Attach ``anomaly_type`` column to *df* (modified in-place & returned).
      +1  anomaly after positive price change  (likely local top)
      -1  anomaly after negative price change  (likely local bottom)
       0  normal day or non-anomaly
    """
    df["pct_change"] = df["Price"].diff()
    df["anomaly_type"] = 0
    is_anomaly = df["pred"] == -1
    df.loc[is_anomaly & (df["pct_change"] > 0), "anomaly_type"] = 1
    df.loc[is_anomaly & (df["pct_change"] < 0), "anomaly_type"] = -1
    return df


def _add_adaptive_threshold(df, base=5, short_window=60, long_window=252,
                            lower=3, upper=15):
    """
    Attach ``adaptive_threshold`` column based on rolling-volatility ratio.
        T = clip(base × σ_short / σ_long, lower, upper)
    """
    returns = df["Price"].pct_change()
    vol_short = returns.rolling(short_window).std()
    vol_long = returns.rolling(long_window).std()
    ratio = (vol_short / vol_long).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    df["adaptive_threshold"] = np.ceil(base * ratio).clip(lower=lower, upper=upper).astype(int)
    return df


def _portfolio_value(capital, shares, price):
    """Current portfolio value."""
    return capital + shares * price


# ── Strategy 0: Passive Buy & Hold ────────────────────────────────────────

def strategy_bh(df_window, initial_capital=1000):
    """Simple Buy & Hold baseline."""
    first_price = df_window["Price"].iloc[0]
    shares = initial_capital / first_price
    return shares * df_window["Price"]


# ── Strategy 1: Full Exit on Any Anomaly ──────────────────────────────────

def strategy1(df_window, initial_capital=1000):
    """Exit entirely on any anomaly; re-enter when normal."""
    prices = df_window["Price"].values
    preds = df_window["pred"].values

    capital, shares, in_market = initial_capital, 0.0, False
    values = np.empty(len(prices))

    for i, (price, pred) in enumerate(zip(prices, preds)):
        if pred == -1:
            if in_market:
                capital, shares, in_market = shares * price, 0.0, False
        else:
            if not in_market:
                shares, in_market = capital / price, True
                capital = 0.0
        values[i] = shares * price if in_market else capital

    return pd.Series(values, index=df_window.index, name="Strategy_Value")


# ── Strategy 2: Exit on Positive Anomaly Only ────────────────────────────

def strategy2(df_window, initial_capital=1000):
    """Sell on positive anomaly; re-enter after recovery to normal."""
    df = _label_anomalies(df_window.copy())
    signals = df["anomaly_type"].values
    prices = df["Price"].values

    capital, shares = initial_capital, 0.0
    in_market = wait_mode = False
    values = np.empty(len(prices))

    for i, (price, sig) in enumerate(zip(prices, signals)):
        if not in_market and not wait_mode and sig == 0:
            shares, in_market = capital / price, True
            capital = 0.0

        if sig == 1:
            if in_market:
                capital, shares, in_market = shares * price, 0.0, False
            wait_mode = True
        elif sig == 0 and wait_mode:
            shares, in_market, wait_mode = capital / price, True, False
            capital = 0.0

        values[i] = shares * price if in_market else capital

    return pd.Series(values, index=df.index, name="Strategy_Value")


# ── Strategy 3: Delayed Re-Entry (Fixed Cool-Down) ───────────────────────

def strategy3(df_window, initial_capital=1000, threshold_days=5):
    """Sell on positive anomaly; re-enter after *threshold_days* clean days."""
    df = _label_anomalies(df_window.copy())
    signals = df["anomaly_type"].values
    prices = df["Price"].values

    capital, shares, in_market, streak = initial_capital, 0.0, False, 0
    values = np.empty(len(prices))

    for i, (price, sig) in enumerate(zip(prices, signals)):
        # First-day entry
        if i == 0 and not in_market:
            shares, in_market = capital / price, True
            capital = 0.0

        if sig != 0:
            streak = 0
            if sig == 1 and in_market:
                capital, shares, in_market = shares * price, 0.0, False
            elif sig == -1 and not in_market:
                shares, in_market = capital / price, True
                capital = 0.0
        else:
            streak += 1
            if streak >= threshold_days and not in_market:
                shares, in_market = capital / price, True
                capital = 0.0

        values[i] = shares * price if in_market else capital

    return pd.Series(values, index=df.index, name="Strategy_Value")


# ── Strategy 4: Adaptive Re-Entry (Volatility-Based) ─────────────────────

def strategy4(df_window, initial_capital=1000):
    """Like S3 but cool-down length adapts to rolling volatility ratio."""
    df = _label_anomalies(df_window.copy())
    df = _add_adaptive_threshold(df)
    signals = df["anomaly_type"].values
    prices = df["Price"].values
    thresholds = df["adaptive_threshold"].values

    capital, shares, in_market, streak = initial_capital, 0.0, False, 0
    values = np.empty(len(prices))

    for i, (price, sig, thr) in enumerate(zip(prices, signals, thresholds)):
        if i == 0 and not in_market:
            shares, in_market = capital / price, True
            capital = 0.0

        if sig != 0:
            streak = 0
            if sig == 1 and in_market:
                capital, shares, in_market = shares * price, 0.0, False
            elif sig == -1 and not in_market:
                shares, in_market = capital / price, True
                capital = 0.0
        else:
            streak += 1
            if streak >= thr and not in_market:
                shares, in_market = capital / price, True
                capital = 0.0

        values[i] = shares * price if in_market else capital

    return pd.Series(values, index=df.index, name="Strategy_Value")


# ── Strategy 5: Partial Buy/Sell + Adaptive Threshold ────────────────────

def strategy5(df_window, initial_capital=1000,
              buy_fraction=0.90, sell_fraction=0.40):
    """Partial scaling on anomaly signals with volatility-adaptive cool-down."""
    df = _label_anomalies(df_window.copy())
    df = _add_adaptive_threshold(df)
    signals = df["anomaly_type"].values
    prices = df["Price"].values
    thresholds = df["adaptive_threshold"].values

    capital, shares, streak = initial_capital, 0.0, 0
    values = np.empty(len(prices))

    for i, (price, sig, thr) in enumerate(zip(prices, signals, thresholds)):
        # First-day: go fully in
        if i == 0 and shares == 0:
            shares, capital = capital / price, 0.0

        if sig != 0:
            streak = 0
            if sig == 1 and shares > 0:            # partial sell
                sell_qty = shares * sell_fraction
                capital += sell_qty * price
                shares -= sell_qty
            elif sig == -1 and capital > 0:         # partial buy
                buy_amt = capital * buy_fraction
                shares += buy_amt / price
                capital -= buy_amt
        else:
            streak += 1
            if streak >= thr and capital > 0:       # reinvest idle cash
                shares += capital / price
                capital = 0.0

        values[i] = _portfolio_value(capital, shares, price)

    return pd.Series(values, index=df.index, name="Strategy_Value")


# ── Grid Search for Strategy 5 ───────────────────────────────────────────

def optimize_strategy5(df, buy_range=None, sell_range=None, initial_capital=1000):
    """
    Brute-force grid search over buy/sell fractions for Strategy 5.

    Returns (results_df, best_buy, best_sell).
    """
    if buy_range is None:
        buy_range = np.round(np.arange(0.1, 1.0, 0.1), 2)
    if sell_range is None:
        sell_range = np.round(np.arange(0.1, 1.0, 0.1), 2)

    rows = []
    for bf in buy_range:
        for sf in sell_range:
            val = strategy5(df.copy(), initial_capital, bf, sf).iloc[-1]
            rows.append((bf, sf, val))

    results_df = (
        pd.DataFrame(rows, columns=["Buy_Fraction", "Sell_Fraction", "Final_Value"])
        .sort_values("Final_Value", ascending=False)
        .reset_index(drop=True)
    )
    best = results_df.iloc[0]
    return results_df, float(best["Buy_Fraction"]), float(best["Sell_Fraction"])
