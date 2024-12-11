"""_summary_
Functions for developing ARIMA model for time-series
Returns:
    _type_: _description_
"""

import itertools
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# from icecream import ic
from joblib import Parallel, delayed
from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_squared_error,
)
from statsmodels.tsa.arima.model import ARIMA


def get_combo(*args):
    """
    Generate all possible combinations of ARIMA model parameters, excluding (0,0,0).

    Parameters:
    -----------
    max_p : int
        Maximum value for the autoregressive (AR) term (p)
    max_d : int
        Maximum value for the differencing term (d)
    max_q : int
        Maximum value for the moving average (MA) term (q)

    Returns:
    --------
    list of tuples
        List of all possible (p,d,q) combinations, excluding (0,0,0)

    Example:
    --------
    >>> combinations = generate_arima_combinations(2, 1, 2)
    >>> print(len(combinations))  # Will print total number of combinations
    >>> print(combinations[:5])  # Will print first 5 combinations
    """
    # Generate parameter ranges
    max_p, max_d, max_q = args[0]
    d_range = range(max_d + 1)
    p_range = range(max_p + 1)
    q_range = range(max_q + 1)

    # Create all possible combinations, filtering out (0,0,0)
    arima_combinations = [
        combo
        for combo in itertools.product(p_range, d_range, q_range)
        if not (combo[0] == 0 and combo[1] == 0 and combo[2] == 0)
    ]

    return arima_combinations


def visualize_arima_results(orders, values):
    """
    Create a 3D scatter plot to visualize ARIMA model results.

    Parameters:
    -----------
    orders : list of tuples
        List of ARIMA model parameter tuples (p, d, q)
        Each tuple should contain (p, d, q) values

    values : list
        List of corresponding performance metric values

    Returns:
    --------
    matplotlib.figure.Figure
        The created 3D visualization figure

    Example:
    --------
    orders = [(1,0,1), (1,1,1), (2,0,2), (0,1,1), (2,1,2)]
    values = [100.5, 95.2, 98.7, 102.3, 93.6]
    fig = visualize_arima_results(orders, values)
    plt.show()
    """
    # Validate input
    if len(orders) != len(values):
        raise ValueError(
            "Length of orders and values must be the same"
        )

    # Extract x, y, z coordinates from the order tuples
    p_values = [order[0] for order in orders]
    d_values = [order[1] for order in orders]
    q_values = [order[2] for order in orders]

    # Create the 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    # Create scatter plot
    scatter = ax.scatter(
        p_values,
        d_values,
        q_values,
        c=values,
        cmap="viridis",
        s=100,  # marker size # type: ignore
        alpha=0.7,
    )

    # Customize the plot
    ax.set_xlabel("p (Autoregressive Term)")
    ax.set_ylabel("d (Differencing Term)")
    ax.set_zlabel("q (Moving Average Term)")  # type: ignore
    ax.set_title("ARIMA Model Parameter Visualization")

    # Add a color bar
    plt.colorbar(scatter, ax=ax, label="Performance Metric Value")

    # Adjust layout to prevent cutting off labels
    plt.tight_layout()
    plt.show()


def info_criteria_selection_arima(
    timeseries, max_order=(5, 2, 5), min_data_length=252
):
    """
    Select best MA order using Information Criteria (AIC, BIC)

    Parameters:
    -----------
    timeseries : array-like
        Time series data
    max_order : int, optional (default=(5,2,5))
        Maximum MA order to consider

    Returns:
    --------
    dict
        MA order selection results based on AIC and BIC
    """
    aic_scores = []
    aic_scores0 = []
    bic_scores = []
    bic_scores0 = []
    name0 = []
    for order in get_combo(max_order):
        # Use ARIMA with MA component, keeping AR and I components as 0
        try:
            model = ARIMA(timeseries[-min_data_length:], order=order)
            results = model.fit(method="innovations_mle")
            aic_scores.append(results.aic)
            aic_scores0.append((order, results.aic))
            bic_scores.append(results.bic)
            bic_scores0.append((order, results.aic))
            name0.append(order)
        except ValueError:
            print(f"{order} fail to converge and eliminated")
    # Find orders with minimum AIC and BIC
    best_aic_order = min(aic_scores0, key=lambda x: x[1])
    best_bic_order = min(bic_scores0, key=lambda x: x[1])
    visualize_arima_results(name0, aic_scores)
    visualize_arima_results(name0, bic_scores)
    return {
        "best_aic_order": best_aic_order,
        "best_bic_order": best_bic_order,
    }


def arima_for_all_data(
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
    try:
        model = ARIMA(train, order=ma_order)
        model_fit = model.fit(method="innovations_mle")
        # Forecast
        forecast = model_fit.forecast(steps=1)
        return test_point, forecast[0]  # type: ignore
    except ValueError:
        return pd.DataFrame()


def plot_best_arima_model(
    timeseries, max_orders=(5, 2, 5), min_data_length=252
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
    for order in get_combo(max_orders):
        start_indices = range(len(timeseries) - min_data_length)
        output = Parallel(n_jobs=-7)(
            delayed(arima_for_all_data)(
                timeseries, start_index, order, min_data_length
            )
            for start_index in start_indices
        )
        filtered_output = [
            (real, pred)
            for real, pred in output  # type: ignore
            if real is not None and pred is not None
        ]

        fold_reals, fold_predictions = zip(*filtered_output)
        try:
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
        except Exception as e:
            print(e)
            print(order)
        result["order"] = [order] * len(result)
        results = pd.concat([results, result], axis=0)
    visualize_arima_results(results["order"], results["mse"])
    visualize_arima_results(results["order"], results["mape"])
    min_mse_index = results[results["mse"] == results["mse"].min()][
        ["mse", "order"]
    ]
    min_mape_index = results[
        results["mape"] == results["mape"].min()
    ][["mape", "order"]]
    print(min_mse_index, min_mape_index)
    return results
