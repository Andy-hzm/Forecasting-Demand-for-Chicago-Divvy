import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def test_stationarity(series: pd.Series, name: str = "") -> None:
    """Run ADF and KPSS tests and print a clean summary."""
    label = f" [{name}]" if name else ""
    print(f"Stationarity Tests{label}")
    print("-" * 45)

    # ADF — H0: unit root (non-stationary)
    adf_stat, adf_p, _, _, adf_crit, _ = adfuller(series.dropna())
    print(f"ADF  stat={adf_stat:.4f}  p={adf_p:.4f}  → {'stationary' if adf_p < 0.05 else 'non-stationary'}")

    # KPSS — H0: stationary
    kpss_stat, kpss_p, _, kpss_crit = kpss(series.dropna(), regression="c", nlags="auto")
    print(f"KPSS stat={kpss_stat:.4f}  p={kpss_p:.4f}  → {'non-stationary' if kpss_p < 0.05 else 'stationary'}")
    print()


def error_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Return MAE, MSE, MAPE, sMAPE, R² as a dict."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    # avoid divide-by-zero in MAPE
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))
    smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"MAE": mae, "MSE": mse, "MAPE": mape, "sMAPE": smape, "R2": r2}


def plot_ts(series: pd.Series, title: str = "", figsize: tuple = (14, 4)) -> None:
    """Simple time series line plot."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(series.index, series.values)
    ax.set_title(title)
    ax.set_xlabel("Date")
    plt.tight_layout()
    plt.show()


def compare_ts(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    title: str = "",
    figsize: tuple = (14, 4),
) -> None:
    """Dual-axis time series plot for two columns sharing a date index."""
    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()
    ax1.plot(df.index, df[col1], color="steelblue", label=col1)
    ax2.plot(df.index, df[col2], color="tomato", alpha=0.7, label=col2)
    ax1.set_ylabel(col1, color="steelblue")
    ax2.set_ylabel(col2, color="tomato")
    ax1.set_title(title)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_acf_pacf(series: pd.Series, lags: int = 50, title: str = "") -> None:
    """Side-by-side ACF and PACF plots."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(series.dropna(), lags=lags, ax=axes[0], title=f"ACF — {title}")
    plot_pacf(series.dropna(), lags=lags, ax=axes[1], title=f"PACF — {title}")
    plt.tight_layout()
    plt.show()
