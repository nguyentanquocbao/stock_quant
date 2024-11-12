"""
_summary_
Module contain function to create or update data from vnstock
and store as parquet file locally

"""

import json
import os
import shutil
import time
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytz
from vnstock3 import Vnstock as vn


def get_ticker(path: str, dictionary: dict) -> pd.DataFrame:
    """_summary_
    Update and return stock list
    When there is no data in the given path --> start download the whole dataset
    Args:
        path (str): path saved parquet data
    Returns:
        _type_: pd.dataframe: stock identifier data
    """
    try:
        data = pd.read_parquet(path)
        data = update_ticker(data, path, dictionary)
    except FileNotFoundError as e:
        print("getting all ticker again", e)
        get_full_ticker(path, dictionary)
        data = pd.read_parquet(path)
    return data


def update_ticker(
    ticker: str, path: str, dictionary: dict
) -> pd.DataFrame:
    """_summary_
    update and add removed flag if there is missing data
    Args:
        ticker (pd.DataFrame): data of stored ticker information
        path (str): path_for saving

    Returns:
        _type_: full data with new tickers
    """
    stock = vn(show_log=False).stock(
        symbol="ABC", source=dictionary["source"]
    )
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
        ticker = pd.concat([ticker, new_ticker])
        print("updated")
        clean_backup_data(path, clean=True)
        ticker.to_parquet(
            path, index=False, partition_cols="exchange"
        )
        # table = pa.Table.from_pandas(ticker)
        # pq.write_to_dataset(table, path)
    return ticker


def get_full_ticker(path: str, dictionary: dict) -> None:
    """_summary_
    create whole new ticker list
    Args:
        path (str): path to save ticker information
    """
    stock = vn(show_log=False).stock(symbol="ABC", source="VCI")
    new_ticker = stock.listing.symbols_by_exchange()
    for i in new_ticker[dictionary["ticker"]]:
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
    ticker: str,
    start_date: datetime.strftime,
    end_date: datetime.strftime,
    dictionary: dict,
) -> pd.DataFrame:
    """_summary_
    read data daily for 1 ticker
    Args:
        ticker (str): ticker name
        start_date (str):
        end_date (str):
        dictionary (dict): dictionary for column name

    Returns:
        _type_: _description_
    """
    return (
        vn(show_log=False)
        .stock(symbol=ticker, source=dictionary["source"])
        .quote.history(start=start_date, end=end_date, interval="1D")
    )


def read_1_ticker_intra(ticker: str, dictionary: dict) -> None:
    """_summary_
    Read and save intra data daily
    Args:
        ticker (str): ticker list
        dictionary (dict): column name

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


def get_past_friday() -> datetime.date:
    """_summary_
    create the latest Friday in the past to get new data when needed
    Returns:
        _type_: datetime.datetime.strftime("%Y-%m-%d")
    """
    date0 = datetime.today().date() - pd.offsets.DateOffset(1)
    while date0.weekday() != 4:  # 5 = Saturday, 4 = Friday
        date0 -= pd.offsets.DateOffset(1)
    return date0.strftime("%Y-%m-%d")


def get_full_data(path_out: str, path_ticker: str, dictionary: dict):
    """_summary_
    down load whole data for all given ticker list
    Args:
        path_ticker (str): for ticker parquet files
        path_out (str): path for saving data by appending parquet data
        dictionary (dict): column names
    """
    fal_tick = []
    tickers = get_ticker(path_ticker, dictionary)
    data = pd.DataFrame()
    for tick, ex in zip(
        tickers[dictionary["ticker"]], tickers[dictionary["exchange"]]
    ):
        try:
            temp = read_1_ticker(
                tick, "2013-01-01", get_past_friday(), dictionary
            )
            temp["ticker"] = tick
            temp["exchange"] = ex
            data = pd.concat([data, temp], axis=0)
        except ValueError:
            fal_tick.append(tick)
        except TypeError:
            fal_tick.append(tick)
        except KeyError:
            fal_tick.append(tick)
    clean_backup_data(path_out)
    table = pa.Table.from_pandas(data)
    pq.write_to_dataset(table, path_out)
    print("Successful create full data")


def clean_backup_data(path: str, clean=False) -> None:
    """_summary_

    Args:
        path (str, optional): path to clean and backup. Defaults to ''.
        clean (bool, optional): clean=True mean remove the whole path,
            otherwise just backup the data to a subfolder named "backup"+current name
            .Defaults to False.
    """
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

        if os.path.exists(backup_path):
            shutil.rmtree(backup_path)
        shutil.copytree(path, backup_path)
        if clean:
            shutil.rmtree(path)
            print(f"{path} has been removed.")
    else:
        print("path cần backup không tồn tại")


def get_data(
    path_stock_data: str, path_ticker: str, dictionary: dict
) -> pd.DataFrame:
    """_summary_
    * Function to read all data, only download full when run the first time,
    The later use would have it appended into current path
    * Function would update data to nearest Friday (weekly) - not included to day.
    Args:
        path_stock_data (str): path to append or first time save dât
        path_ticker (str): ticker list path
        dictionary (dict): column name

    Returns:
        _type_: pd.DataFrame()
    """
    try:
        data = pd.read_parquet(path_stock_data)
        latest_data_date = data["time"].max()
    except FileNotFoundError:
        print("Getting full data")
        get_full_data(path_stock_data, path_ticker, dictionary)
        data = pd.read_parquet(path_stock_data)
        latest_data_date = data["time"].max()

    print(f"latest data date is {latest_data_date}")
    latest_friday = get_past_friday()
    tickers = get_ticker(path_ticker, dictionary)
    if latest_data_date.strftime("%Y-%m-%d") < latest_friday:
        print("getting new data")
        latest_data_date = latest_data_date + pd.offsets.DateOffset(1)
        data = pd.DataFrame()
        fal_tick = []
        for tick, ex in zip(tickers.symbol, tickers.exchange):
            try:
                temp = read_1_ticker(
                    tick,
                    latest_data_date.strftime("%Y-%m-%d"),
                    latest_friday,
                    dictionary,
                )
                temp["ticker"] = tick
                temp["exchange"] = ex
                print(temp.shape)
                data = pd.concat([data, temp], axis=0)
            except ValueError:
                fal_tick.append(tick)
            except TypeError:
                fal_tick.append(tick)
            except KeyError:
                fal_tick.append(tick)
        if data.shape[0] > 0:
            clean_backup_data(path_stock_data, clean=False)
            table = pa.Table.from_pandas(data)
            pq.write_to_dataset(table, path_stock_data)
    return data, tickers


def read_intra_data(
    path_ticker: str, path: str, dictionary: dict, update=False
):
    """_summary_
    Get data set of transaction if it's pass 4pm and not in weekends,
        update the whole dataset when set update=True
    Args:
        path_ticker (str): path of ticker parquet
        path (str): path of intraday data
        dictionary (dict): column name
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


def update_intra_data(path_ticker: str, path: str, dictionary: dict):
    """_summary_
    update intra day data for all tickers
    Args:
        path_ticker (str): path of ticker parquet storage
        path (str): path of intraday storage
        dictionary (dict): column name
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
    path_stock_data: str,
    path_ticker: str,
    dictionary=dict,
) -> pd.DataFrame:
    """_summary_
    Args:
        path_stock_data (str, optional): _description_. Defaults to path_stock_data.
        path_ticker (str, optional): _description_. Defaults to path_ticker.
        dictionary (dict, optional): _description_. Defaults to {}.

    Returns:
        pd.DataFrame: _description_
    """
    stock_data, stock_info = get_data(
        path_stock_data, path_ticker, dictionary
    )
    stock_info = stock_info[stock_info["total_outstanding"] > 0]
    stock_data = stock_data[
        stock_data["ticker"].isin(stock_info[dictionary["ticker"]])  # type: ignore
    ]
    stock_data = stock_data[
        stock_data["ticker"].isin(stock_info[dictionary["ticker"]])  # type: ignore
    ]
    stock_data = stock_data.merge(
        stock_info[[dictionary["ticker"], "total_outstanding", "type"]],  # type: ignore
        how="inner",
        left_on=["ticker"],
        right_on=[dictionary["ticker"]],  # type: ignore
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


if __name__ == "__main__":
    ## update data process
    from static import PATH_DICTIONARY, PATH_TICKER, PATH_TRANSACTION

    with open(PATH_DICTIONARY, "r", encoding="utf-8") as a:
        name_dict = json.load(a)
    df = read_intra_data(
        PATH_TICKER, PATH_TRANSACTION, name_dict, update=True
    )
