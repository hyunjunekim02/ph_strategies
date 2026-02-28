"""
Visualization Module
====================
Plotting functions for landscape distances, SVM scores, anomalies,
and strategy-vs-baseline performance.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── Style constants ───────────────────────────────────────────────────────

_FIG_SIZE = (12, 4)
_LINE_WIDTH = 1.8
_ANOMALY_COLOR = "red"
_ANOMALY_ALPHA = 0.15
_GRID = True


def _style_ax(ax, title, xlabel="Date", ylabel=None):
    """Apply consistent styling to an axes object."""
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(_GRID, alpha=0.3)
    ax.legend()


# ── Anomaly overview plots ───────────────────────────────────────────────

def plot_results(df, window_dates, distances, anomaly_idx, scores, W,
                 asset_name="Asset"):
    """
    Three-panel figure:
      1. Persistence landscape L2 distance + anomaly markers
      2. SVM decision score + threshold line
      3. Full price chart with anomaly highlights
    """
    distances = np.asarray(distances)
    anomaly_idx = np.asarray(anomaly_idx)

    # Panel 1 — Landscape distance
    fig, ax = plt.subplots(figsize=_FIG_SIZE)
    ax.plot(window_dates, distances, lw=_LINE_WIDTH, label="Landscape Distance")
    if len(anomaly_idx):
        ax.scatter(
            window_dates.iloc[anomaly_idx], distances[anomaly_idx],
            c=_ANOMALY_COLOR, zorder=5, label="Anomaly",
        )
    _style_ax(ax, f"[{asset_name}] Persistent Landscape & Anomalies (W={W})",
              ylabel="Distance")
    fig.tight_layout(); plt.show()

    # Panel 2 — SVM score
    fig, ax = plt.subplots(figsize=_FIG_SIZE)
    ax.plot(window_dates, scores, lw=_LINE_WIDTH, label="SVM Score")
    ax.axhline(0, color=_ANOMALY_COLOR, ls="--", label="Anomaly Threshold")
    _style_ax(ax, f"[{asset_name}] SVM Confidence Score (W={W})",
              ylabel="Score")
    fig.tight_layout(); plt.show()

    # Panel 3 — Price + anomaly points
    price_anom_idx = anomaly_idx + (1 + W)
    fig, ax = plt.subplots(figsize=_FIG_SIZE)
    ax.plot(df["Date"], df["Price"], lw=2, label=asset_name)
    if len(price_anom_idx):
        ax.scatter(
            df["Date"].iloc[price_anom_idx], df["Price"].iloc[price_anom_idx],
            c=_ANOMALY_COLOR, zorder=5, s=18, label="Detected Anomaly",
        )
    _style_ax(ax, f"[{asset_name}] Price Chart w/ Anomaly Points (W={W})",
              ylabel="Price")
    fig.tight_layout(); plt.show()


# ── Strategy performance plot ────────────────────────────────────────────

def plot_strategy(df_strategy, window_size, asset_name="Asset"):
    """
    Strategy vs Buy & Hold with anomaly regions highlighted.
    Also prints a short performance summary.
    """
    final_bh = df_strategy["BH_Value"].iloc[-1]
    final_st = df_strategy["Strategy_Value"].iloc[-1]
    diff = final_st - final_bh
    ratio = diff / final_bh * 100

    print(f"\n  Buy & Hold : ${final_bh:>12,.2f}")
    print(f"  Strategy   : ${final_st:>12,.2f}")
    print(f"  Difference : ${diff:>12,.2f}  ({ratio:+.2f}%)")

    fig, ax = plt.subplots(figsize=_FIG_SIZE)
    ax.plot(df_strategy["Date"], df_strategy["BH_Value"], label="Buy & Hold")
    ax.plot(df_strategy["Date"], df_strategy["Strategy_Value"], label="Strategy")

    # Highlight anomaly regions (vectorised)
    anom_mask = df_strategy["pred"].values == -1
    if anom_mask.any():
        anom_dates = df_strategy["Date"].values[anom_mask]
        for d in anom_dates:
            ax.axvline(d, color=_ANOMALY_COLOR, alpha=0.04, lw=0.5)

    _style_ax(ax, f"Strategy Comparison | W={window_size} | {asset_name}",
              ylabel="Portfolio Value ($)")
    fig.tight_layout(); plt.show()
