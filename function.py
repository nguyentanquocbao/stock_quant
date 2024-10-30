from bao import *
from vnstock3 import Vnstock as vn
import dask.dataframe as dd
import shutil
import os
import pyarrow as pa
import pyarrow.parquet as pq
import json
import time


def get_ticker(path, dictionary):
    """_summary_
    Update and return stock list
    Args:
        path (_type_): path saved parquet data
    Returns:
        _type_: pd.dataframe: stock indentifier data
    """
    data = pd.read_parquet(path)
    data = update_ticker(data, path, dictionary)
    return data


def update_ticker(ticker, path, dictionary):
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
        shutil.rmtree(path)
        print(f"{path} has been removed.")
        ticker.to_parquet(
            path, index=False, partition_cols=[dictionary("exchange")]
        )
    return ticker


def get_full_ticker(path, dictionary):
    """_summary_
    create whole new ticker list
    Args:
        path (_type_): _description_
    """
    if os.path.exists(path):
        # Remove the directory or file
        shutil.rmtree(path)
        print(f"{path} has been removed.")
    else:
        print(f"{path} does not exist.")
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
    new_ticker.to_parquet(
        path, index=False, partition_cols=[dictionary["exchange"]]
    )


def read_1_ticker(ticker, start_date, end_date, dictionary):
    stock = (
        vn(show_log=False)
        .stock(symbol=ticker, source=dictionary["source"])
        .quote.history(start=start_date, end=end_date, interval="1D")
    )
    return stock


def get_past_Friday():
    date0 = datetime.today().date()
    while date0.weekday() != 4:  # 5 = Saturday, 6 = Sunday
        date0 -= pd.offsets.DateOffset(1)
    return date0.strftime("%Y-%m-%d")


def get_full_data(path_ticker, path_out, dictionary):
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
    if os.path.exists(path_out):
        shutil.rmtree(path_out)
        print(f"{path_out} has been removed.")
    table = pa.Table.from_pandas(data)
    pq.write_to_dataset(table, path_out)


def get_data(path_out, path_ticker, dictionary):
    data = pd.read_parquet(path_out)
    today = get_past_Friday()
    tickers = get_ticker(path_ticker, dictionary)
    lastest_data_date = data["time"].max()
    if lastest_data_date.strftime("%Y-%m-%d") < today:
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
                    today(),
                )
                temp["ticker"] = tick
                temp["exchange"] = ex
                data = pd.concat([data, temp], axis=0)
            except Exception as e:
                fal_tick.append(tick)
        print(data.head(2))
        print(data["ticker"].value_counts())
        if os.path.exists(path_out):
            # Remove the directory or file
            shutil.rmtree(path_out)
            print(f"{path_out} has been removed.")
        table = pa.Table.from_pandas(data)
        pq.write_to_dataset(table, path_out)
    return data


if __name__ == "__main__":
    get_data()
