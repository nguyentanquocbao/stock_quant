"""_summary_
Moving average analysis for time series data
Returns:
    _type_: _description_
"""

from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_squared_error,
)
from statsmodels.tsa.arima.model import ARIMA


def info_criteria_selection_ma(timeseries, max_order=30):
    """
    Select best MA order using Information Criteria (AIC, BIC)

    Parameters:
    -----------
    timeseries : array-like
        Time series data
    max_order : int, optional (default=10)
        Maximum MA order to consider

    Returns:
    --------
    dict
        MA order selection results based on AIC and BIC
    """
    aic_scores = []
    bic_scores = []

    for order in range(1, max_order + 1):
        # Use ARIMA with MA component, keeping AR and I components as 0
        model = ARIMA(timeseries, order=(0, 0, order))
        results = model.fit(method="innovations_mle")
        aic_scores.append((order, results.aic))
        bic_scores.append((order, results.bic))

    # Find orders with minimum AIC and BIC
    best_aic_order = min(aic_scores, key=lambda x: x[1])
    best_bic_order = min(bic_scores, key=lambda x: x[1])

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(*zip(*aic_scores), marker="o")
    plt.title("AIC Scores for MA Models")
    plt.xlabel("MA Order")
    plt.ylabel("AIC")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(*zip(*bic_scores), marker="x")
    plt.title("BIC Scores for MA Models")
    plt.xlabel("MA Order")
    plt.ylabel("BIC")
    plt.tight_layout()
    plt.grid(True)
    plt.show()

    return {
        "best_aic_order": best_aic_order,
        "best_bic_order": best_bic_order,
    }


def ma_for_all_data(
    series: pd.Series,
    start_index: int,
    ma_order: List[int],
    min_data_length: int,
) -> pd.DataFrame:
    """_summary_
    compute arima for all available data and give back 1 prediction for subsequent time point
    Args:
        series (pd.Series): _description_
        start_index (int): _description_
        ma_order (List[int]): _description_
        min_data_length (int): _description_

    Returns:
        _type_: _description_
    """
    end_index = min(len(series), start_index + min_data_length)
    train = series[:end_index]
    test_point = series[end_index]

    # Fit MA model
    model = ARIMA(train, order=(0, 0, ma_order))
    model_fit = model.fit(method="innovations_mle")
    # Forecast
    forecast = model_fit.forecast(steps=1)
    return test_point, forecast[0]  # type: ignore


def plot_best_ma_model(
    timeseries, max_orders=10, min_data_length=252
):
    """
    Select best MA order and plot actual vs predicted for the best performing model

    Parameters:
    -----------
    timeseries : array-like
        Time series data
    max_orders : int, optional (default=10)
        Maximum MA order to consider
    min_data_length : int, optional (default=252)
        Minimum length of training data

    Returns:
    --------
    pd.DataFrame
        Performance results for different MA orders
    """
    # Ensure input is numpy array
    timeseries = np.asarray(timeseries)

    # Performance tracking

    results = pd.DataFrame()
    for order in range(1, max_orders + 1):
        start_indices = range(len(timeseries) - min_data_length)
        output = Parallel(n_jobs=-7)(
            delayed(ma_for_all_data)(
                timeseries, start_index, order, min_data_length
            )
            for start_index in start_indices
        )
        fold_reals, fold_predictions = zip(*output)
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
        result["order"] = order
        results = pd.concat([results, result], axis=0)

    plt.figure(figsize=(10, 5))

    # MSE subplot
    plt.subplot(1, 2, 1)
    plt.plot(
        results["order"], results["mse"], label="MSE", color="blue"
    )
    plt.title("Mean Squared Error")
    plt.xlabel("MA Order")
    plt.ylabel("MSE Value")
    plt.legend()
    plt.grid(True)

    # MAPE subplot
    plt.subplot(1, 2, 2)
    plt.plot(
        results["order"], results["mape"], label="MAPE", color="red"
    )
    plt.title("Mean Absolute Percentage Error")
    plt.xlabel("MA Order")
    plt.ylabel("MAPE Value")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Find minimum values and corresponding orders
    min_mse_index = results[results["mse"] == results["mse"].min()][
        ["mse", "order"]
    ]
    min_mape_index = results[
        results["mape"] == results["mape"].min()
    ][["mape", "order"]]
    print(min_mse_index, min_mape_index)
    return results


def visualize_ma(series, ma_order, min_data_length=252):
    """
    Visualize predictions for a selected MA model

    Parameters:
    -----------
    series : pd.Series
        Time series data
    ma_order : int
        Moving Average order to use
    min_data_length : int, optional (default=252)
        Minimum length of training data
    """
    start_indices = range(len(series) - min_data_length)
    output = Parallel(n_jobs=-7)(
        delayed(ma_for_all_data)(
            series, start_index, ma_order, min_data_length
        )
        for start_index in start_indices
    )
    fold_reals, fold_predictions = zip(*output)
    plt.figure(figsize=(10, 5))
    plt.plot(fold_reals, label="Real", color="blue")
    plt.plot(fold_predictions, label="Predict", color="red")
    plt.title(f"MA({ma_order}) Model: Real vs Predicted")
    plt.legend()
    plt.grid(True)
    plt.show()
