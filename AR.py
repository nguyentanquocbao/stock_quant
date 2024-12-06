"""
_summary_
Functions for developing auto regression model for time-series
Returns:
    _type_: _description_
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from icecream import ic
from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_squared_error,
)
from statsmodels.tsa.ar_model import AutoReg


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
    best_aic_lag = min(aic_scores, key=lambda x: x[1])
    best_bic_lag = min(bic_scores, key=lambda x: x[1])
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(*zip(*aic_scores), marker="o")
    plt.title("AIC Scores")
    plt.xlabel("Lag")
    plt.ylabel("AIC")
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(*zip(*bic_scores), marker="x")
    plt.title("BIC Scores")
    plt.xlabel("Lag")
    plt.ylabel("BIC")
    plt.tight_layout()
    plt.grid(True)
    plt.show()
    return {
        "best_aic_lag": best_aic_lag,
        "best_bic_lag": best_bic_lag,
    }


def plot_best_lag_model(timeseries, max_lags=30, min_data_length=252):
    """
    Select best lag and plot actual vs predicted for the best performing model

    Parameters:
    -----------
    timeseries : array-like
        Time series data
    metric : str, optional (default='mse')
        Performance metric to use (mse, mae, mape)
    max_lags : int, optional (default=30)
    min_data_length : int, optional (default=252)
        Minimum length of training data

    Returns:
    --------
    dict
        Detailed results of the best lag model
    """
    # Ensure input is numpy array
    timeseries = np.asarray(timeseries)
    results = pd.DataFrame()
    for lag in range(1, max_lags + 1):
        # fold_scores = []
        fold_reals = []
        fold_predictions = []
        start_index = max(lag + 1, min_data_length)
        # Rolling forecast validation
        for end_index in range(
            start_index,
            min(len(timeseries), start_index + min_data_length),
        ):  # Prepare training and test data
            train = timeseries[:end_index]
            test_point = timeseries[end_index]

            # Fit AR model
            model = AutoReg(train, lags=lag)
            model_fit = model.fit()

            # Forecast
            forecast = model_fit.forecast(steps=1)

            # Calculate performance
            # score = chosen_metric([test_point], forecast)
            # fold_scores.append(score)

            # Store for detailed analysis
            fold_reals.append(test_point)
            fold_predictions.append(forecast[0])
        result = pd.DataFrame(
            {
                "mse": [
                    mean_squared_error(
                        y_true=fold_reals, y_pred=fold_predictions
                    )
                ],
                "mape": [
                    mean_absolute_percentage_error(
                        y_true=fold_reals, y_pred=fold_predictions
                    )
                ],
            }
        )
        result["code"] = lag
        results = pd.concat([results, result], axis=0)
    plt.figure(figsize=(10, 5))
    # First subplot for MSE
    plt.subplot(1, 2, 1)
    plt.plot(
        results["code"], results["mse"], label="MSE", color="blue"
    )
    plt.title("Mean Squared Error")
    plt.xlabel("Code")
    plt.ylabel("MSE Value")
    plt.legend()
    plt.grid(True)
    # Second subplot for MAPE
    plt.subplot(1, 2, 2)
    plt.plot(
        results["code"], results["mape"], label="MAPE", color="red"
    )
    plt.title("Mean Absolute Percentage Error")
    plt.xlabel("Code")
    plt.ylabel("MAPE Value")
    plt.legend()
    plt.grid(True)
    min_mse_index = results[results["mse"] == results["mse"].min()][
        ["mse", "code"]
    ]
    min_mape_index = results[
        results["mape"] == results["mape"].min()
    ][["mape", "code"]]
    ic(min_mse_index, min_mape_index)
    return results


def visualize(
    series: pd.Series, lag: int, min_data_length: int = 252
):
    """give the visualize for prediction and real data as the lag has already selected

    Args:
        series (pd.Series): _description_
        lag (int): _description_
        min_data_length (int, optional): _description_. Defaults to 252.
    """
    start_index = max(lag + 1, min_data_length)
    fold_reals = []
    fold_predictions = []
    for end_index in range(
        start_index,
        min(len(series), start_index + min_data_length),
    ):  # Prepare training and test data
        try:
            train = series[:end_index]
            test_point = series[end_index]
        except ValueError:
            ic(end_index, start_index)
        # Fit AR model
        model = AutoReg(train, lags=lag)
        model_fit = model.fit()
        # Forecast
        forecast = model_fit.forecast(steps=1)
        fold_reals.append(test_point)
        fold_predictions.append(forecast[0])
    plt.figure(figsize=(10, 5))
    # First subplot for MSE
    plt.plot(fold_reals, label="real", color="blue")
    plt.plot(fold_predictions, label="predict", color="red")
    plt.legend()
    plt.grid(True)
    plt.show()
