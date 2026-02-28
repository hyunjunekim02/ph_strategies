"""
Evaluation Module
=================
Helpers to run strategies, tabulate comparisons, and identify the best one.
"""

import pandas as pd
from .visualization import plot_strategy


def run_strategy(result_dict, baseline_func, strategy_func, initial_capital=1000):
    """
    Execute *strategy_func* alongside the *baseline_func* (Buy & Hold).

    Parameters
    ----------
    result_dict : dict
        Output of ``compute_features_for_window()``.
    baseline_func, strategy_func : callable(df, capital) → pd.Series
    initial_capital : float

    Returns
    -------
    pd.DataFrame  with columns BH_Value, Strategy_Value, pred, Date, Price, …
    """
    df = result_dict["df_window"].copy()
    df["pred"] = result_dict["preds"]
    df["BH_Value"] = baseline_func(df, initial_capital)
    df["Strategy_Value"] = strategy_func(df, initial_capital)
    return df


def compare_all_strategies(strategy_dfs):
    """
    Build & print a comparison table of final portfolio values.

    Returns a sorted ``pd.DataFrame``.
    """
    rows = []
    for name, df_s in strategy_dfs.items():
        bh = df_s["BH_Value"].iloc[-1]
        st = df_s["Strategy_Value"].iloc[-1]
        diff = st - bh
        rows.append({
            "Strategy": name,
            "BH ($)": f"{bh:,.2f}",
            "Strategy ($)": f"{st:,.2f}",
            "Diff ($)": f"{diff:+,.2f}",
            "vs BH (%)": f"{diff / bh * 100:+.2f}%",
            "_sort": st,
        })

    df = (
        pd.DataFrame(rows)
        .sort_values("_sort", ascending=False)
        .drop(columns="_sort")
        .reset_index(drop=True)
    )
    print("\n===== Strategy Comparison =====")
    print(df.to_string(index=False))
    return df


def plot_best_strategy(strategy_dfs, window_size, asset_name="Asset"):
    """Find and plot the single best-performing strategy."""
    best_name, best_val, best_df = None, -float("inf"), None
    for name, df_s in strategy_dfs.items():
        val = df_s["Strategy_Value"].iloc[-1]
        if val > best_val:
            best_name, best_val, best_df = name, val, df_s

    if best_df is not None:
        print(f"\n>>> Best: {best_name} <<<")
        plot_strategy(best_df, window_size,
                      asset_name=f"{asset_name} — {best_name}")
