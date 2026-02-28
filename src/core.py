"""
Core Computation Module
=======================
Persistent Landscape extraction and One-Class SVM anomaly detection.

Pipeline: raw prices → log-returns → sliding-window point clouds
        → Vietoris–Rips persistence diagrams → persistence landscapes
        → L2 landscape distances → One-Class SVM anomaly labels
"""

import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import LocalOutlierFactor
import gudhi as gd
from gudhi.representations import Landscape


# ── Embedding ─────────────────────────────────────────────────────────────

def _delay_embedding(data, dim, delay):
    """Build a (N, dim) delay-coordinate point cloud from a 1-D array."""
    n_pts = len(data) - (dim - 1) * delay
    indices = np.arange(dim) * delay
    rows = np.arange(n_pts)[:, None] + indices
    return data[rows]


def _sliding_window_clouds(log_returns, W, dim, delay, step):
    """Produce a list of point clouds via sliding windows over log-returns."""
    return [
        _delay_embedding(log_returns[i : i + W], dim, delay)
        for i in range(0, len(log_returns) - W, step)
    ]


# ── Persistent Homology ──────────────────────────────────────────────────

def _compute_persistence_diagrams(point_clouds, max_edge=1.0, max_dim=2):
    """Compute Vietoris–Rips persistence diagrams for each point cloud."""
    diagrams = []
    for pc in point_clouds:
        rips = gd.RipsComplex(points=pc, max_edge_length=max_edge)
        st = rips.create_simplex_tree(max_dimension=max_dim)
        diagrams.append(st.persistence())
    return diagrams


def _diagrams_to_landscapes(diagrams, num_landscapes=1, resolution=200):
    """Convert persistence diagrams into persistence landscapes."""
    transformer = Landscape(num_landscapes=num_landscapes, resolution=resolution)
    landscapes = []
    for diag in diagrams:
        pairs = np.array([p[1] for p in diag if p[1][1] != float("inf")])
        if len(pairs) == 0:
            pairs = np.array([[0.0, 0.0]])
        landscapes.append(transformer.fit_transform([pairs])[0])
    return np.array(landscapes)


def _consecutive_l2_distances(landscapes):
    """L2 distance between consecutive landscape vectors (vectorised)."""
    if len(landscapes) < 2:
        return np.array([])
    return np.linalg.norm(np.diff(landscapes, axis=0), axis=1)


# ── Anomaly Detection ────────────────────────────────────────────────────

_NORMAL_RATIO_THRESHOLD = 0.8


def _detect_anomalies(distances, nu=0.05):
    """
    One-Class SVM on log-scaled distances; falls back to LOF when the SVM
    labels too many points as anomalies.

    Returns (predictions, decision_scores).
    """
    if len(distances) == 0:
        return np.array([]), np.array([])

    X_scaled = RobustScaler().fit_transform(
        np.log1p(distances).reshape(-1, 1)
    )

    clf = OneClassSVM(kernel="rbf", gamma="auto", nu=nu)
    clf.fit(X_scaled)
    preds = clf.predict(X_scaled)
    scores = clf.decision_function(X_scaled)

    if (preds == 1).mean() < _NORMAL_RATIO_THRESHOLD:
        lof = LocalOutlierFactor(n_neighbors=20, contamination=nu)
        preds = lof.fit_predict(X_scaled)
        scores = -lof.negative_outlier_factor_

    return preds, scores


# ── Public API ───────────────────────────────────────────────────────────

def compute_features_for_window(df_raw, W, dim=3, delay=2, step=1, nu=0.05):
    """
    Full PH pipeline for a single window size.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Must contain ``Date`` and ``Price`` columns.
    W : int
        Sliding-window size.
    dim, delay, step : int
        Embedding hyper-parameters (defaults: 3, 2, 1).
    nu : float
        SVM ν controlling anomaly proportion (default 0.05).

    Returns
    -------
    dict
        W, distances, preds, scores, df_window, window_dates,
        anomaly_idx, price_anomaly_idx
    """
    close = df_raw["Price"].values
    log_returns = np.log(close[1:] / close[:-1])

    clouds = _sliding_window_clouds(log_returns, W, dim, delay, step)
    diagrams = _compute_persistence_diagrams(clouds)
    landscapes = _diagrams_to_landscapes(diagrams)
    distances = _consecutive_l2_distances(landscapes)
    preds, scores = _detect_anomalies(distances, nu=nu)

    # Align with original timestamps
    start_idx = 1 + W
    L = len(distances)
    df_window = df_raw.iloc[start_idx : start_idx + L].copy().reset_index(drop=True)

    if L > 0:
        df_window["distance"] = distances
        df_window["pred"] = preds.astype(int)
        df_window["score"] = scores
    else:
        for col, dtype in [("distance", float), ("pred", int), ("score", float)]:
            df_window[col] = pd.Series(dtype=dtype)

    anomaly_idx = np.flatnonzero(preds == -1) if L > 0 else np.array([], dtype=int)

    return {
        "W": W,
        "distances": distances,
        "preds": preds,
        "scores": scores,
        "df_window": df_window,
        "window_dates": df_window["Date"],
        "anomaly_idx": anomaly_idx,
        "price_anomaly_idx": anomaly_idx + start_idx,
    }
