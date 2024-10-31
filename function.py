from bao import *
from vnstock3 import Vnstock as vn
import dask.dataframe as dd
import shutil
import os
import pyarrow as pa
import pyarrow.parquet as pq
import time
import pytz

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
        ~ticker[dictionary["ticker"]].isin(new_ticker[dictionary["ticker"]]),
        "dropped",
    ] = datetime.today()
    new_ticker = new_ticker[~new_ticker[dictionary["ticker"]].
                            isin(ticker[dictionary["ticker"]])]
    if new_ticker.shape[0] > 0:
        for i in new_ticker[dictionary["ticker"]]:
            try:
                new_ticker.loc[
                    new_ticker[dictionary["ticker"]] == i,
                    "total_outstanding",
                ] = stock.trading.price_board(
                    [i])[dictionary["share_outstanding"][0]][
                        dictionary["share_outstanding"][1]][0]
            except Exception as e:
                print(i, e)
        ticker = pd.concat([ticker, new_ticker])
        print("updated")
        shutil.rmtree(path)
        print(f"{path} has been removed.")
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
            ] = stock.trading.price_board(
                [i])[dictionary["share_outstanding"][0]][
                    dictionary["share_outstanding"][1]][0]
            time.sleep(
                1)  # api might not respond when the request speed is too high

        except Exception as e:
            print(i, e)
    if os.path.exists(path):
        # Remove the directory or file
        shutil.rmtree(path)
        print(f"{path} has been removed.")
    else:
        print(f"{path} does not exist.")
    new_ticker.to_parquet(path,
                          index=False,
                          partition_cols=[dictionary["exchange"]])


def read_1_ticker(ticker, start_date, end_date, dictionary) -> pd.DataFrame:
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
    return  (vn(show_log=False).stock(
        symbol=ticker,
        source=dictionary["source"]).quote.history(start=start_date,
                                                    end=end_date,
                                                    interval="1D"))
def read_1_ticker_intra(ticker,dictionary):
    return (vn(show_log=False).stock(
        symbol=ticker,
        source=dictionary['source']).quote.intraday(symbol=ticker,page_size=10000000, show_log=False))

def get_past_Friday() -> datetime.date:
    """_summary_
    create the lastest Friday in the past to get new data when needed
    Returns:
        _type_: _description_
    """
    date0 = datetime.today().date()-pd.offsets.DateOffset(1)
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
    for tick, ex in zip(tickers[dictionary["ticker"]],
                        tickers[dictionary["exchange"]]):
        try:
            temp = read_1_ticker(tick, "2013-01-01", get_past_Friday(),
                                dictionary)
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
        lastest_data_date = lastest_data_date + pd.offsets.DateOffset(1)
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
        if os.path.exists(path_out):
            # Remove the directory or file
            shutil.rmtree(path_out)
            print(f"{path_out} has been removed.")
        table = pa.Table.from_pandas(data)
        pq.write_to_dataset(table, path_out)
    return data


def read_intra_data(path_ticker, path, dictionary, update=False):
    vietnam_time = datetime.now(pytz.timezone('Asia/Ho_Chi_Minh')).time()
    target_time = datetime.strptime('16:00:00', '%H:%M:%S').time()
    if vietnam_time > target_time and update:
        update_intra_data(path_ticker, path, dictionary)
    data = pd.read_parquet(path)
    return data


def update_intra_data(path_ticker, path, dictionary):
    tickers = get_ticker(path_ticker, dictionary)
    fal_tick = []
    for tick, ex in zip(tickers.symbol, tickers.exchange):
        try:
            temp = read_1_ticker_intra(tick, dictionary)
            temp["ticker"] = tick
            temp["exchange"] = ex
            data = pd.concat([data, temp], axis=0)
        except Exception as e:
            fal_tick.append(tick)
    table = pa.Table.from_pandas(data)
    pq.write_to_dataset(table, path)

if __name__ == "__main__":
    get_data()
