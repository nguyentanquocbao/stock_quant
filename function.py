import pytz
import shutil
import os
import time
import pyarrow as pa
import pyarrow.parquet as pq
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt

from bao import *
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg
from vnstock3 import Vnstock as vn
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats.mstats import winsorize
from sklearn.metrics import mean_squared_error


def get_ticker(path, dictionary) -> pd.DataFrame:
    """_summary_
    Update and return stock list, when there is no data in the given path --> start download the whole dataset
    Args:
        path (_type_): path saved parquet data
    Returns:
        _type_: pd.dataframe: stock indentifier data
    """
    try:
        data = pd.read_parquet(path)
        data = update_ticker(data, path, dictionary)
    except Exception as e:
        print("getting all ticker again", e)
        get_full_ticker(path, dictionary)
        data = pd.read_parquet(path)
    return data


def plot_acf_0(data):
    plot_acf(data, lags=15)
    plt.show()


def draw_SMA(data, col, windows) -> None:
    """_summary_
    give visual for simple moving average with given list of windows
    Args:
        data (_type_): dataframe
        col (_type_): value column
        windows (_type_): list of window period
    """

    for i in windows:
        data[col + str(i)] = data[col].rolling(window=i).mean()
    fig = go.Figure()
    for i in windows:
        fig.add_trace(
            go.Scatter(
                x=data["time"],  # Adjust if your index is not time
                y=data[col + str(i)],
                mode="lines",
                name=str(i),
            )
        )

        # Update layout
        fig.update_layout(
            title="Simple moving average return",
            xaxis_title="Time",
            yaxis_title="Rolling Mean Return",
            showlegend=True,
        )
    fig.show()


def draw_cummulative(data, col, windows) -> None:
    """_summary_
    give visual for simple moving average with given list of windows
    Args:
        data (_type_): dataframe
        col (_type_): value column
        windows (_type_): list of window period
    """

    for i in windows:
        data[col + str(i)] = data[col].rolling(window=i).sum()
    fig = go.Figure()
    for i in windows:
        fig.add_trace(
            go.Scatter(
                x=data["time"],  # Adjust if your index is not time
                y=data[col + str(i)],
                mode="lines",
                name=str(i),
            )
        )

        # Update layout
        fig.update_layout(
            title="Cumulative return rolling back",
            xaxis_title="Time",
            yaxis_title="Rolling Mean Return",
            showlegend=True,
        )
    fig.show()


def draw_EMA(data, col, windows) -> None:
    """_summary_
    Give visualization for Exponential Moving Average
    Args:
        data (_type_): dataframe
        col (_type_): value column
        windows (_type_): list of window period
    """
    for i in windows:
        data[col + str(i)] = data[col].ewm(span=i, adjust=True).mean()
    fig = go.Figure()
    for i in windows:
        fig.add_trace(
            go.Scatter(
                x=data["time"],  # Adjust if your index is not time
                y=data[col + str(i)],
                mode="lines",
                name=str(i),
            )
        )
        # Update layout
        fig.update_layout(
            title="Exponential Moving Average Return", showlegend=True
        )
    fig.show()


def update_ticker(ticker, path, dictionary) -> pd.DataFrame:
    """_summary_
    update and add removed flag if there is missing data
    Args:
        ticker (_type_): old data
        path (_type_): path_for saving

    Returns:
        _type_: full data with new tickers
    """
    stock = vn(show_log=False).stock(symbol="ABC", source="VCI")
    new_ticker = stock.listing.symbols_by_exchange()
    ticker.loc[
        ~ticker[dictionary["ticker"]].isin(
            new_ticker[dictionary["ticker"]]
        ),
        "dropped",
    ] = datetime.today()
    new_ticker = new_ticker[
        ~new_ticker[dictionary["ticker"]].isin(
            ticker[dictionary["ticker"]]
        )
    ]
    if new_ticker.shape[0] > 0:
        for i in new_ticker[dictionary["ticker"]]:
            try:
                new_ticker.loc[
                    new_ticker[dictionary["ticker"]] == i,
                    "total_outstanding",
                ] = stock.trading.price_board([i])[
                    dictionary["share_outstanding"][0]
                ][
                    dictionary["share_outstanding"][1]
                ][
                    0
                ]
            except Exception as e:
                print(i, e)
        ticker = pd.concat([ticker, new_ticker])
        print("updated")
        clean_backup_data(path, clean=True)
        ticker.to_parquet(path, index=False)
    return ticker


def get_full_ticker(path, dictionary) -> None:
    """_summary_
    create whole new ticker list
    Args:
        path (_type_): _description_
    """
    stock = vn(show_log=False).stock(symbol="ABC", source="VCI")
    new_ticker = stock.listing.symbols_by_exchange()
    for i in new_ticker[dictionary["ticker"]]:
        try:
            new_ticker.loc[
                new_ticker[dictionary["ticker"]] == i,
                "total_outstanding",
            ] = stock.trading.price_board([i])[
                dictionary["share_outstanding"][0]
            ][
                dictionary["share_outstanding"][1]
            ][
                0
            ]
            time.sleep(
                1
            )  # api might not respond when the request speed is too high

        except Exception as e:
            print(i, e)
    if os.path.exists(path):
        # Remove the directory or file
        shutil.rmtree(path)
        print(f"{path} has been removed.")
    else:
        print(f"{path} does not exist.")
    new_ticker.to_parquet(
        path, index=False, partition_cols=[dictionary["exchange"]]
    )


def read_1_ticker(
    ticker, start_date, end_date, dictionary
) -> pd.DataFrame:
    """_summary_
    read data daily for 1 ticker
    Args:
        ticker (_type_): ticker name
        start_date (_type_):
        end_date (_type_):
        dictionary (_type_): dictionary for column name

    Returns:
        _type_: _description_
    """
    return (
        vn(show_log=False)
        .stock(symbol=ticker, source=dictionary["source"])
        .quote.history(start=start_date, end=end_date, interval="1D")
    )


def read_1_ticker_intra(ticker, dictionary) -> None:
    """_summary_
    Read and save intradata daily
    Args:
        ticker (_type_): ticker list
        dictionary (_type_): column name

    Returns:
        _type_: None
    """
    return (
        vn(show_log=False)
        .stock(symbol=ticker, source=dictionary["source"])
        .quote.intraday(
            symbol=ticker, page_size=10000000, show_log=False
        )
    )


def get_past_Friday() -> datetime.date:
    """_summary_
    create the lastest Friday in the past to get new data when needed
    Returns:
        _type_: _description_
    """
    date0 = datetime.today().date() - pd.offsets.DateOffset(1)
    while date0.weekday() != 4:  # 5 = Saturday, 4 = Friday
        date0 -= pd.offsets.DateOffset(1)
    return date0.strftime("%Y-%m-%d")


def get_full_data(path_ticker, path_out, dictionary):
    """_summary_
    down load whole data for all given ticker list
    Args:
        path_ticker (_type_): for ticker parquet files
        path_out (_type_): path for saving data by appending parquet data
        dictionary (_type_): column names
    """
    fal_tick = []
    tickers = get_ticker(path_ticker, dictionary)
    data = pd.DataFrame()
    for tick, ex in zip(
        tickers[dictionary["ticker"]], tickers[dictionary["exchange"]]
    ):
        try:
            temp = read_1_ticker(
                tick, "2013-01-01", get_past_Friday(), dictionary
            )
            temp["ticker"] = tick
            temp["exchange"] = ex
            data = pd.concat([data, temp], axis=0)
        except Exception as e:
            fal_tick.append(tick)
    clean_backup_data(path_out)
    table = pa.Table.from_pandas(data)
    pq.write_to_dataset(table, path_out)
    print("sucessful create full data")


def clean_backup_data(path, clean=False) -> None:
    if os.path.exists(path):
        # Remove the directory or file
        source_dir = os.path.dirname(path)

        # Create the backup directory within the same structure
        backup_dir = os.path.join(source_dir, "backup")
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)

        # Get the filename of the source file
        filename = os.path.basename(path)

        # Construct the full path for the backup file
        backup_path = os.path.join(backup_dir, filename)

        # Copy the source file to the backup location
        shutil.copy2(path, backup_path)
        if clean:
            shutil.rmtree(path)
            print(f"{path} has been removed.")


def get_data(path_out, path_ticker, dictionary) -> pd.DataFrame:
    """_summary_
    * Function to read all data, only download full when run the first time, the later use would have it appended into current path
    * Function would update data to nearest Friday (weekly) - not included to day.
    Args:
        path_out (_type_): path to append or first time save dât
        path_ticker (_type_): ticker list path
        dictionary (_type_): colunm name

    Returns:
        _type_: pd.DataFrame()
    """
    try:
        data = pd.read_parquet(path_out)
        lastest_data_date = data["time"].max()
    except Exception as e:
        print("Getting full data")
        get_full_data(path_out, path_ticker, dictionary)
        data = pd.read_parquet(path_out)
        lastest_data_date = data["time"].max()

    latest_friday = get_past_Friday()
    tickers = get_ticker(path_ticker, dictionary)
    if lastest_data_date.strftime("%Y-%m-%d") < latest_friday:
        lastest_data_date = lastest_data_date + pd.offsets.DateOffset(
            1
        )
        data = pd.DataFrame()
        fal_tick = []
        for tick, ex in zip(tickers.symbol, tickers.exchange):
            try:
                temp = read_1_ticker(
                    tick,
                    lastest_data_date.strftime("%Y-%m-%d"),
                    latest_friday,
                )
                temp["ticker"] = tick
                temp["exchange"] = ex
                data = pd.concat([data, temp], axis=0)
            except Exception as e:
                fal_tick.append(tick)
        clean_backup_data(path_out, clean=True)
        table = pa.Table.from_pandas(data)
        pq.write_to_dataset(table, path_out)
    return data, tickers


def read_intra_data(path_ticker, path, dictionary, update=False):
    """_summary_
    Get data set of transaction if it's pass 4pm and not in weekends, update the whole dataset when set update=True
    Args:
        path_ticker (_type_): path of ticker parquet
        path (_type_): path of intraday data
        dictionary (_type_): column name
        update (bool, optional): need update or not . Defaults to False.
    Returns:
        _type_: pd.DataFrame
    """
    vietnam_time = datetime.now(
        pytz.timezone("Asia/Ho_Chi_Minh")
    ).time()
    target_time = datetime.strptime("16:00:00", "%H:%M:%S").time()
    date0 = datetime.today().date()
    if (
        (vietnam_time > target_time)
        and (update)
        and date0.weekday() not in [5, 6]
    ):

        update_intra_data(path_ticker, path, dictionary)
    data = pd.read_parquet(path)
    return data


def update_intra_data(path_ticker, path, dictionary):
    """_summary_
    update intrad day data for all tickers
    Args:
        path_ticker (_type_): path of ticker parquet storage
        path (_type_): path of instraday storage
        dictionary (_type_): column name
    """
    tickers = get_ticker(path_ticker, dictionary)
    fal_tick = []
    data = pd.DataFrame()
    for tick, ex in zip(tickers.symbol, tickers.exchange):
        try:
            temp = read_1_ticker_intra(tick, dictionary)
            temp["ticker"] = tick
            temp["exchange"] = ex
            data = pd.concat([data, temp], axis=0)
        except Exception as e:
            fal_tick.append(tick)
    clean_backup_data(path)
    table = pa.Table.from_pandas(data)
    pq.write_to_dataset(table, path)


def clean_daily_data(
    path_stock_data, path_ticker, dictionary
) -> pd.DataFrame:
    """_summary_
    Data cleaning for daily stock data
    Args:
        data_in (_type_): list contain 2 dataframe: stock data and stock info
        dictionary (_type_): dictionary for some columns's name that might change due to vnstock
    Returns:
        _type_: _description_
    """
    stock_data, stock_info = get_data(
        path_stock_data, path_ticker, dictionary
    )
    stock_info = stock_info[stock_info["total_outstanding"] > 0]
    stock_data = stock_data[
        stock_data["ticker"].isin(stock_info[dictionary["ticker"]])
    ]
    stock_data = stock_data[
        stock_data["ticker"].isin(stock_info[dictionary["ticker"]])
    ]
    stock_data = stock_data.merge(
        stock_info[[dictionary["ticker"], "total_outstanding"]],
        how="inner",
        left_on=["ticker"],
        right_on=[dictionary["ticker"]],
    )
    stock_data.loc[stock_data["close"] == 0, "close"] = None
    stock_data["close"] = stock_data.groupby("ticker")[
        "close"
    ].fillna(method="ffill")
    stock_data = stock_data.dropna(subset="close")
    stock_data["market_value"] = (
        stock_data["close"] * stock_data["total_outstanding"]
    )
    stock_data["return"] = (
        stock_data["close"]
        / stock_data.groupby("ticker")["close"].shift(1)
        - 1
    )
    stock_data["log_return"] = np.log(
        stock_data["close"]
        / stock_data.groupby("ticker")["close"].shift(1)
    )
    stock_data.sort_values(["ticker", "time"], inplace=True)

    stock_data["market_weight"] = stock_data[
        "market_value"
    ] / stock_data.groupby("exchange")["market_value"].transform(
        "sum"
    )
    stock_data["return_weighted"] = (
        stock_data["market_weight"] * stock_data["return"]
    ).fillna(0)
    stock_data["log_return_weighted"] = (
        stock_data["market_weight"] * stock_data["log_return"]
    ).fillna(0)
    return stock_data


def adf_test(series, significance_level=0.05):

    result = adfuller(series)
    adf_statistic, p_value, _, _, critical_values, _ = result

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


def AR_visualize(data, lag, period):
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


def calculate_mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100


# Root Mean Square Error
def calculate_rmse(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))


def kpss_test(series, significance_level=0.05):
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


def ARIMA_visualize(data, orders, period, metric="aic"):
    data = data[-period:]
    metric_values = []  # Store metric values for each model
    models = {}  # Dictionary to store fitted models for each order

    # Loop through each order and fit the ARIMA model
    for order in orders:
        try:
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

        except Exception as e:
            print(
                f"Error fitting ARIMA model with order {order}: {e}"
            )
            metric_values.append(
                float("inf")
            )  # Assign a high value if model fails

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


if __name__ == "__main__":
    ## update data process
    from static import *
    from bao import *

    with open(path_dictionary, "r") as a:
        name_dict = json.load(a)
    df = read_intra_data(
        path_ticker, path_transaction, name_dict, update=True
    )
