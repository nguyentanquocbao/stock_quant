"""
_summary_
module to create visualization and test hypotheses
"""

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss


def simple_moving_average_visual(
    data: pd.DataFrame, col: str, windows: list
) -> None:
    """_summary_
    give visual for simple moving average with given list of windows
    Args:
        data (pd.DataFrame): dataframe contain data
        col (str): value column
        windows (list): list of window period
    """

    for i in windows:
        data[col + str(i)] = data[col].rolling(window=i).mean()
        plt.plot(data["time"], data[col + str(i)], label=str(i))

        # Customize plot elements
    plt.xlabel("Time")
    plt.ylabel("Rolling Mean Return")
    plt.title("Simple moving average return")
    plt.legend()
    plt.grid(True)  # Add grid for better readability

    plt.show()


def cumulative_visual(
    data: pd.DataFrame, col: str, windows: list
) -> None:
    """_summary_
    give visual for simple moving average with given list of windows
    Args:
        data (dataframe): dataframe
        col (str): value column
        windows (list): list of int (window period)
    """

    _, ax = plt.subplots()
    for i in windows:
        data[col + str(i)] = data[col].rolling(window=i).sum()
        ax.plot(data["time"], data[col + str(i)], label=str(i))

    ax.legend()
    plt.title("Cumulative Log Return")
    plt.show()


def exponential_moving_average_visual(
    data: pd.DataFrame, col: str, windows: list
) -> None:
    """_summary_
    Give visualization for Exponential Moving Average
    Args:
        data (pd.DataFrame): dataframe
        col (str): value column
        windows (list): list of window period (int)
    """
    for i in windows:
        data[col + str(i)] = data[col].ewm(span=i, adjust=True).mean()
        plt.plot(data["time"], data[col + str(i)], label=str(i))

        # Customize plot elements
    plt.xlabel("Time")
    plt.ylabel("Rolling Mean Return")
    plt.title("Exponential Mean Return")
    plt.legend()
    plt.grid(True)  # Add grid for better readability

    plt.show()


def auto_regression_visualize(data: pd.Series, lag: int, period: int):
    """_summary_
    Create estimation of whole data set
    Args:
        data (Series): dataframe input
        lag (int):number of lag period used
        period (int): number of data trading days to be used to estimate
    Returns:
        _type_: None
    """
    data = data[-period:]
    ar_model = AutoReg(data, lags=lag).fit()
    ar_predictions = ar_model.predict(start=1, end=period)
    print(ar_model.pvalues)
    print(ar_model.params)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            y=data,
            mode="lines",
            name="Original Series",
        )
    )
    fig.add_trace(
        go.Scatter(
            y=ar_predictions,
            mode="lines",
            name="AR(1) Predictions",
            line=dict(color="red"),
        )
    )
    fig.update_layout(
        title="AR(1) Model Predictions",
        xaxis_title="Time",
        yaxis_title="Value",
    )
    fig.show()


def arima_visualize(
    data: pd.DataFrame, orders: list, period: int, metric="aic"
):
    """_summary_
    Choose the best parameter in orders to create estimation of whole data set
    Args:
        data (pd.DataFrame): dataframe input
        orders (list): a list of parameters for [autoregression, differencing term, moving average]
        period (int): number of data trading days to be used to estimate
        metric (str, optional): _description_.
            Defaults to "aic", other choice are: 'bic', 'rmse' and 'mape'
    Raises:
        ValueError: error when estimate by ARIMA function

    Returns:
        _type_: best order
    """
    data = data[-period:]
    metric_values = []  # Store metric values for each model
    models = {}  # Dictionary to store fitted models for each order

    # Loop through each order and fit the ARIMA model
    for order in orders:
        # Fit the ARIMA model
        arima_model = ARIMA(data, order=order).fit()
        predictions = arima_model.predict(start=0, end=period - 1)
        # Calculate the chosen metric
        if metric == "mape":
            metric_value = calculate_mape(data, predictions)
        elif metric == "rmse":
            metric_value = calculate_rmse(data, predictions)
        elif metric == "aic":
            metric_value = arima_model.aic
        elif metric == "bic":
            metric_value = arima_model.bic
        else:
            raise ValueError(
                "Invalid metric. Choose from 'mape', 'rmse', 'aic', or 'bic'."
            )
        # Store metric and model
        metric_values.append(metric_value)
        models[order] = arima_model
        print(f"Order {order}: {metric.upper()} = {metric_value}")
    # Find the best model with the smallest metric value
    best_index = metric_values.index(min(metric_values))
    best_order = orders[best_index]
    best_model = models[best_order]
    best_predictions = best_model.predict(start=0, end=period - 1)
    # Print best model details
    print(f"\nBest Model Order: {best_order}")
    print(f"Best Model {metric.upper()}:", metric_values[best_index])
    print("P-values:", best_model.pvalues)
    print("Parameters:", best_model.params)
    # Visualization
    fig = go.Figure()
    # Original series as a bar chart
    fig.add_trace(
        go.Bar(
            y=data,
            name="Original Series",
            marker=dict(color="blue"),
        )
    )
    # ARIMA predictions of the best model as a line chart
    fig.add_trace(
        go.Scatter(
            y=best_predictions,
            mode="lines",
            name=f"ARIMA{best_order} Predictions",
            line=dict(color="red"),
        )
    )
    fig.update_layout(
        title=f"Best ARIMA Model (Order {best_order}) Predictions",
        xaxis_title="Time",
        yaxis_title="Value",
    )
    fig.show()

    # Return the metric values and the best model order
    return metric_values, best_order


def auto_regression_forecast_1_step(
    data: pd.DataFrame,
    val_col: str,
    length: int,
    time0: datetime,
    time_col: str,
    lag: int,
) -> pd.DataFrame:
    """_summary_
    Function to forecast by AR with given condition
    Args:
        data (pd.DataFrame): data frame that contain time and value columns
        val_col (str): name of value column
        length (int): trend time to rollback for each estimation
        time0 (datetime): time point to begin rollback
        time_col (str): time column name
        lag (int): number of lag used to put in model
    Returns:
        pd.DataFrame: dataframe that contain predicted values
    """
    temp = data[data[time_col] <= time0]
    temp = temp.tail(length)
    ar_model = AutoReg(temp[val_col], lags=lag).fit()
    return pd.DataFrame(
        {"time": time0, "predict": ar_model.forecast(steps=1)}
    )


def real_time_auto_regression_visualize(
    data: pd.DataFrame,
    val_col: str,
    time_col: str,
    length: int,
    lag: int,
) -> pd.DataFrame:
    """_summary_
    Get predicted data and visualization of AR model
    The function would predict on each time point on only data that available in that time point,
        instead of using all data to detect changes.
        In other words, each prediction might use different model (parameters) to predict,
            only models' hyperparameter is the same.
    TODO: add function to check for model parameter and quality over all time-point
    Args:
        data (pd.DataFrame): data frame that contain time and value columns
        val_col (str): name of value column
        length (int): trend time to rollback for each estimation
        time0 (datetime): time point to begin rollback
        time_col (str): time column name
        lag (int): number of lag used to put in model
        data (pd.DataFrame): _description_
    Returns:
        pd.DataFrame: dataframe that contain predicted values for all given dates
    """
    data.sort_values(by=time_col, inplace=True)
    time_list = data[time_col][length:]
    predicted = pd.DataFrame()
    for i in time_list:
        predicted = pd.concat(
            [
                predicted,
                auto_regression_forecast_1_step(
                    data, val_col, length, i, time_col, lag
                ),
            ],
            axis=0,
        )
    predicted = predicted.merge(
        data[[val_col, time_col]], how="inner", on=time_col
    )
    predicted.set_index(time_col, inplace=True, drop=True)
    _, ax = plt.subplots()

    # Plot the original series
    ax.plot(
        predicted.index, predicted[val_col], label="Original Series"
    )

    # Plot the AR model predictions
    ax.plot(
        predicted.index,
        predicted["predict"],
        label=f"AR( {lag} ) Predictions",
        color="red",
    )
    # Add title and labels
    ax.set_title(f"AR( {lag} ) Model Predictions")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    # Add legend
    ax.legend()
    # Show the plot
    plt.show()
    return predicted


def arima_forecast_1_step(
    data: pd.DataFrame,
    time_col: str,
    time0: datetime,
    val_col: str,
    order: list,
) -> pd.DataFrame:
    """_summary_
    Function to forecast by AR with given condition
    Args:
        data (pd.DataFrame): data frame that contain time and value columns
        val_col (str): name of value column
        length (int): trend time to rollback for each estimation
        time0 (datetime): time point to begin rollback
        time_col (str): time column name
        lag (int): number of lag used to put in model
    Returns:
        pd.DataFrame: dataframe that contain predicted values
    """
    data[time_col] = pd.to_datetime(data[time_col], errors="coerce")
    data = data.set_index(time_col, drop=True)
    arima_model = ARIMA(data[val_col].values, order=order).fit()
    return pd.DataFrame(
        {
            time_col: time0,
            "predict": arima_model.forecast(steps=1),
        }
    )


def real_time_arima_visualize(
    data: pd.DataFrame,
    val_col: str,
    time_col: str,
    length: int,
    order: list,
    para=False,
) -> pd.DataFrame:
    """_summary_
    get predicted data and visualization of ARIMA model
        The function would predict on each time point on only data that available
            in that time point, instead of using all data to detect changes.
        In other words, each prediction might use different model (parameters) to predict,
            only models' hyperparameter is the same.
    TODO: add function to check for model parameter and quality over all time-point
    Args:
        data (pd.DataFrame): data frame that contain time and value columns
        val_col (str): name of value column
        length (int): trend time to rollback for each estimation
        time0 (datetime): time point to begin rollback
        time_col (str): time column name
        lag (int): number of lag used to put in model
        data (pd.DataFrame): _description_
    Returns:
        pd.DataFrame: dataframe that contain predicted values for all given dates
    """
    data.sort_values(by=time_col, inplace=True)
    time_list = data[time_col][length:]
    if not para:
        predicted = pd.DataFrame()
        for i in time_list:

            predicted = pd.concat(
                [
                    predicted,
                    arima_forecast_1_step(
                        data[data[time_col] <= i].tail(length),
                        time_col,
                        i,
                        val_col,
                        order,
                    ),
                ],
                axis=0,
            )
    else:
        predicted = Parallel(n_jobs=-1)(
            delayed(arima_forecast_1_step)(
                data[data[time_col] <= i].tail(length),
                time_col,
                i,
                val_col,
                order,
            )
            for i in time_list
        )
        predicted = pd.concat(predicted, axis=0)
    predicted = predicted.merge(
        data[[val_col, time_col]], how="inner", on=time_col
    )
    predicted.set_index(time_col, inplace=True, drop=True)
    _, ax = plt.subplots()

    # Plot the original series
    ax.plot(
        predicted.index, predicted[val_col], label="Original Series"
    )

    # Plot the AR model predictions
    ax.plot(
        predicted.index,
        predicted["predict"],
        label=f"AR( {order} ) Predictions",
        color="red",
    )

    # Add title and labels
    ax.set_title(f"AR( {order} ) Model Predictions")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")

    # Add legend
    ax.legend()

    # Show the plot
    plt.show()
    return predicted


def efficient_frontier_visual(
    data: pd.DataFrame,
    time_col: str,
    rank: int,
    time_length: int,
    stock_list=None,
    num_portfolios: int = 1000000,
) -> None:
    """_summary_
    Draw Efficient Frontier line
    Args:
        data (pd.DataFrame): data
        time_col (str): datetime columns name
        rank (int): order of maximum data point used (-1 as the latest data)
        time_length (int): how long from the maximum point to be computed
        stock_list (list, optional): list of stock to compute,
            if not given auto filter0 stocks that has enough observation
                thought out the period.
            Defaults to [].
    Returns:
        None
    """
    if stock_list is None:
        stock_list = []
    time_list = data[time_col].unique()
    time_list = sorted(time_list)[
        time_length:
    ]  # Ensure enough data points after sorting
    auto = (
        rank if rank < 0 else len(time_list) + rank
    )  # Handle negative rank
    point = time_list[auto]
    lower_point = time_list[auto - time_length]
    print(f"visualization come from {lower_point} to {point}")

    sub_data = data[
        (data[time_col] < point) & (data[time_col] > lower_point)
    ]
    sub_data.dropna(subset="close", inplace=True)

    filter0 = (
        sub_data.groupby("ticker").time.agg("nunique").reset_index()
    )
    if len(stock_list) == 0:
        list0 = filter0[
            filter0[time_col] == filter0[time_col].max()
        ].ticker
        sub_data = sub_data[sub_data["ticker"].isin(list0)]
    else:
        sub_data = sub_data[sub_data["ticker"].isin(stock_list)]

    sub_data[time_col].value_counts()
    sub_data = sub_data.pivot(
        columns="ticker", values="close", index=time_col
    )
    tickers = len(sub_data.columns)
    log_returns = np.log(sub_data / sub_data.shift(1))[1:]
    mean_returns = log_returns.mean()
    cov_matrix = log_returns.cov()
    results = np.zeros((4, num_portfolios))

    # Calculate market value weighted portfolio
    market_weights = (
        data[data[time_col] == point]
        .set_index("ticker")["market_weight"]
        .reindex(sub_data.columns)
    )
    market_weights /= market_weights.sum()  # Normalize weights
    market_portfolio_return = (
        np.sum(mean_returns * market_weights) * 252
    )
    market_portfolio_volatility = np.sqrt(
        np.dot(market_weights.T, np.dot(cov_matrix, market_weights))
    ) * np.sqrt(252)

    for i in range(num_portfolios):
        weights = np.random.random(tickers)
        weights /= np.sum(weights)
        portfolio_return = np.sum(mean_returns * weights) * 252
        portfolio_volatility = np.sqrt(
            np.dot(weights.T, np.dot(cov_matrix, weights))
        ) * np.sqrt(252)

        results[0, i] = portfolio_return
        results[1, i] = portfolio_volatility
        results[2, i] = results[0, i] / results[1, i]
        results[3, i] = i

    results_frame = pd.DataFrame(
        results.T,
        columns=["Return", "Volatility", "Sharpe Ratio", "Index"],
    )
    max_sharpe_portfolio = results_frame.loc[
        results_frame["Sharpe Ratio"].idxmax()
    ]
    min_volatility_portfolio = results_frame.loc[
        results_frame["Volatility"].idxmin()
    ]

    plt.figure(figsize=(10, 7))
    plt.scatter(
        results_frame["Volatility"],
        results_frame["Return"],
        c=results_frame["Sharpe Ratio"],
        cmap="YlGnBu",
        marker="o",
    )
    plt.colorbar(label="Sharpe Ratio")
    plt.scatter(
        max_sharpe_portfolio[1],
        max_sharpe_portfolio[0],
        marker="*",
        color="r",
        s=200,
        label="Max Sharpe Ratio",
    )
    plt.scatter(
        min_volatility_portfolio[1],
        min_volatility_portfolio[0],
        marker="*",
        color="g",
        s=200,
        label="Min Volatility",
    )

    # Plot market value weighted portfolio
    plt.scatter(
        market_portfolio_volatility,
        market_portfolio_return,
        marker="*",
        color="b",
        s=200,
        label="Market Portfolio",
    )

    plt.title("Efficient Frontier")
    plt.xlabel("Volatility")
    plt.ylabel("Return")
    plt.legend(labelspacing=0.8)
    plt.show()


def adf_test(series: pd.Series, significance_level: float = 0.05):
    """_summary_
    Args:
        series (series): series data need to be test, series of float
        significance_level (float, optional): _description_. Defaults to 0.05.
    """
    # pylint: disable=undefined-variable
    # pylint: disable=no-name-in-module
    results = adfuller(series)
    adf_statistic = results[0]
    p_value = results[1]
    critical_values = results[4]
    # Print ADF results
    print("ADF Test Results:")
    print("ADF Statistic:", adf_statistic)
    print("p-value:", p_value)
    for key, value in critical_values.items():
        print(f"Critical Value ({key}): {value}")

    # Make judgment based on p-value
    if p_value < significance_level:
        print(
            "Judgment: The series is likely stationary (Reject null hypothesis)."
        )
    else:
        print(
            "Judgment: The series is likely non-stationary (Fail to reject null hypothesis)."
        )
    print("\n" + "=" * 40 + "\n")


def calculate_mape(actual: pd.Series, predicted: pd.Series):
    """_summary_
    compute mean absolute percentage error
    Args:
        actual (series):
        predicted (series):

    Returns:
        _type_:
    """
    return np.mean(np.abs((actual - predicted) / actual)) * 100


# Root Mean Square Error
def calculate_rmse(actual: pd.Series, predicted: pd.Series):
    """_summary_
    compute mean squared error of the prediction
    Args:
        actual (series):
        predicted (series):

    Returns:
        _type_: value
    """
    return np.sqrt(mean_squared_error(actual, predicted))


def kpss_test(series: pd.Series, significance_level=0.05):
    """_summary_
    Test for stationary with:
        H0: given series is not stationary
    Args:
        series (series): input series
        significance_level (float, optional):  Defaults to 0.05.
    """

    result = kpss(series, regression="c")
    kpss_statistic, p_value, _, critical_values = result

    # Print KPSS results
    print("KPSS Test Results:")
    print("KPSS Statistic:", kpss_statistic)
    print("p-value:", p_value)
    for key, value in critical_values.items():
        print(f"Critical Value ({key}): {value}")

    # Make judgment based on p-value
    if p_value < significance_level:
        print(
            "Judgment: The series is likely non-stationary (Reject null hypothesis)."
        )
    else:
        print(
            "Judgment: The series is likely stationary (Fail to reject null hypothesis)."
        )
    print("\n" + "=" * 40 + "\n")
