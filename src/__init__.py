"""
ph-trading-strategies
=====================
Persistent Homology–Driven Trading Strategies Under Financial Market Anomalies.

Public API
----------
    core.compute_features_for_window
    strategies.{strategy_bh, strategy1, ..., strategy5, optimize_strategy5}
    visualization.{plot_results, plot_strategy}
    evaluation.{run_strategy, compare_all_strategies, plot_best_strategy}
"""

from .core import compute_features_for_window
from .strategies import (
    strategy_bh,
    strategy1,
    strategy2,
    strategy3,
    strategy4,
    strategy5,
    optimize_strategy5,
)
from .visualization import plot_results, plot_strategy
from .evaluation import run_strategy, compare_all_strategies, plot_best_strategy

__all__ = [
    "compute_features_for_window",
    "strategy_bh", "strategy1", "strategy2", "strategy3",
    "strategy4", "strategy5", "optimize_strategy5",
    "plot_results", "plot_strategy",
    "run_strategy", "compare_all_strategies", "plot_best_strategy",
]
