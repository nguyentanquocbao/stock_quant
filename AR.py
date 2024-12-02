import numpy as np
import pandas as pd
from dataclasses import dataclass
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
import itertools


def info_criteria_selection(timeseries, max_lag=20):
    """
    Select best lag using Information Criteria (AIC, BIC)

    Parameters:
    -----------
    timeseries : array-like
        Time series data

    Returns:
    --------
    dict
        Lag selection results based on AIC and BIC
    """
    aic_scores = []
    bic_scores = []

    for lag in range(1, max_lag + 1):
        model = AutoReg(timeseries, lags=lag)
        results = model.fit()
        aic_scores.append((lag, results.aic))
        bic_scores.append((lag, results.bic))

    # Find lags with minimum AIC and BIC
    best_aic_lag = min(aic_scores, key=lambda x: x[1])
    best_bic_lag = min(bic_scores, key=lambda x: x[1])

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(*zip(*aic_scores), marker="o")
    plt.title("AIC Scores")
    plt.xlabel("Lag")
    plt.ylabel("AIC")

    plt.subplot(1, 2, 2)
    plt.plot(*zip(*bic_scores), marker="o")
    plt.title("BIC Scores")
    plt.xlabel("Lag")
    plt.ylabel("BIC")

    plt.tight_layout()
    plt.show()

    return {
        "best_aic_lag": best_aic_lag[0],
        "best_bic_lag": best_bic_lag[0],
        "aic_scores": dict(aic_scores),
        "bic_scores": dict(bic_scores),
    }


def plot_best_lag_model(
    timeseries, metric="mse", max_lags=30, min_data_length=252
):
    """
    Select best lag and plot actual vs predicted for the best performing model

    Parameters:
    -----------
    timeseries : array-like
        Time series data
    metric : str, optional (default='mse')
        Performance metric to use (mse, mae, mape)
    max_lags : int, optional (default=30)
        Maximum number of lags to consider
    min_data_length : int, optional (default=252)
        Minimum length of training data

    Returns:
    --------
    dict
        Detailed results of the best lag model
    """
    # Ensure input is numpy array
    timeseries = np.asarray(timeseries)

    # Metric mapping
    metrics = {
        "mse": mean_squared_error,
        "mae": mean_absolute_error,
        "mape": mean_absolute_percentage_error,
    }
    chosen_metric = metrics[metric]

    # Performance tracking
    performance = {}
    detailed_results = {}

    # Iterate through possible lag values
    for lag in range(1, max_lags + 1):
        fold_scores = []
        fold_reals = []
        fold_predictions = []

        # Ensure minimum training size
        start_index = max(lag + 1, min_data_length)

        # Rolling forecast validation
        for end_index in range(start_index, len(timeseries)):
            # Prepare training and test data
            train = timeseries[:end_index]
            test_point = timeseries[end_index]

            # Fit AR model
            model = AutoReg(train, lags=lag)
            model_fit = model.fit()

            # Forecast
            forecast = model_fit.forecast(steps=1)

            # Calculate performance
            score = chosen_metric([test_point], forecast)
            fold_scores.append(score)

            # Store for detailed analysis
            fold_reals.append(test_point)
            fold_predictions.append(forecast[0])

        # Average performance across folds
        if fold_scores:
            performance[lag] = np.mean(fold_scores)
            detailed_results[lag] = {
                "reals": fold_reals,
                "predictions": fold_predictions,
            }

    # Find best lag
    best_lag = min(performance, key=performance.get)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(
        detailed_results[best_lag]["reals"],
        label="Actual Values",
        color="blue",
        marker="o",
    )
    plt.plot(
        detailed_results[best_lag]["predictions"],
        label=f"Predicted Values (Lag {best_lag})",
        color="red",
        linestyle="--",
        marker="x",
    )

    plt.title(
        f"Actual vs Predicted Values for Best Lag Model\n(Lag {best_lag}, {metric.upper()} = {performance[best_lag]:.4f})"
    )
    plt.xlabel("Time Steps")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

    return {
        "best_lag": best_lag,
        "best_performance": performance[best_lag],
        "reals": detailed_results[best_lag]["reals"],
        "predictions": detailed_results[best_lag]["predictions"],
    }


def partial_autocorrelation_selection(timeseries, max_lags=30):
    """
    Select lag using Partial Autocorrelation Function (PACF)

    Parameters:
    -----------
    timeseries : array-like
        Time series data

    Returns:
    --------
    int
        Suggested number of lags
    """
    from statsmodels.graphics.tsaplots import plot_pacf

    plt.figure(figsize=(10, 5))
    pacf_plot = plot_pacf(timeseries, lags=max_lags)
    plt.title("Partial Autocorrelation Function (PACF)")
    plt.show()

    # Typically, select lags where PACF significantly crosses confidence interval
    # This requires manual interpretation of the plot
    return pacf_plot


# Example usage
