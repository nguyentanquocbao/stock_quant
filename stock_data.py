"""
_summary_
Module contain function to create or update data from vnstock
and store as parquet file locally
"""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from rich import print as pprint
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)
from vnstock3 import Vnstock as vn


def get_past_friday() -> str:
    """_summary_
    create the latest Friday in the past to get new data when needed
    Returns:
        _type_: datetime.datetime.strftime("%Y-%m-%d")
    """
    date0 = datetime.today().date() - pd.offsets.DateOffset(n=1)
    while date0.weekday() != 4:  # 5 = Saturday, 4 = Friday
        date0 -= pd.offsets.DateOffset(n=1)
    return date0.strftime("%Y-%m-%d")


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """_summary_
    clean dataset and interpolate missing value
    Args:
        data (pd.DataFrame): daily stock or indices data

    Returns:
        pd.DataFrame: cleaned dataset
    """
    data.loc[data["close"] == 0, "close"] = None
    data["close"] = data.groupby("ticker")["close"].fillna(
        method="ffill"
    )
    data = data.dropna(subset="close")
    data["market_value"] = data["close"] * data["total_outstanding"]
    data["return"] = (
        data["close"] / data.groupby("ticker")["close"].shift(1) - 1
    )
    data["log_return"] = np.log(
        data["close"] / data.groupby("ticker")["close"].shift(1)
    )
    data.sort_values(["ticker", "time"], inplace=True)

    data["market_weight"] = data["market_value"] / data.groupby(
        "exchange"
    )["market_value"].transform("sum")
    data["return_weighted"] = (
        data["market_weight"] * data["return"]
    ).fillna(0)
    data["log_return_weighted"] = (
        data["market_weight"] * data["log_return"]
    ).fillna(0)
    return data


@dataclass
class StockData:
    """_summary_
    class for updating stock data
    """

    path: str
    path_dictionary: str

    def __post_init__(self):
        """Creates the directory path if it doesn't exist."""
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            print("created a new path for stock data")
        with open(self.path_dictionary, "r", encoding="utf-8") as a:
            self.dictionary = json.load(a)

    def get_ticker(self) -> pd.DataFrame:
        """_summary_
        Get data downloaded and download the first-time if there is not
        Returns:
            _type_: _description_
        """
        if not os.path.exists(
            self.path + self.dictionary["path_ticker_list"]
        ):
            pprint("create data for first time")
            self.reload_ticker_list()
        return pd.read_parquet(
            self.path + self.dictionary["path_ticker_list"]
        )

    def get_indices_data(self) -> List[pd.DataFrame]:
        """_summary_
        get data for 2 major indices in Vietnam: vni and vn30
        Args:
            dictionary (_type_): dictionary

        Returns:
            pd.DataFrame: dictionary
        """
        stock = vn(show_log=False).stock(
            symbol="ABC", source=self.dictionary["source"]
        )
        vni = stock.quote.history(
            symbol="VNINDEX",
            end="2024-01-02",
            start="2013-01-01",
            interval="1D",
        )
        vni["exchange"] = "vni"
        vni["ticker"] = "vni"
        vni["total_outstanding"] = 1
        vn30 = stock.quote.history(
            symbol="VN30",
            end="2024-01-02",
            start="2013-01-01",
            interval="1D",
        )
        vn30["total_outstanding"] = 1
        vn30["ticker"] = "vn30"
        vn30["exchange"] = "vn30"
        return [clean_data(vni), clean_data(vn30)]

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1))
    def read_outstanding_1_stock(
        self, ticker: str, data: pd.DataFrame
    ) -> pd.DataFrame:
        """_summary_
        download outstanding_stock for 1 ticker.
        This function would be used in parallel function to speed up download data
        Args:
            ticker (_type_): _description_
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        data = data[data[self.dictionary["ticker"]] == ticker]
        stock = vn(show_log=False).stock(
            symbol="ABC", source=self.dictionary["source"]
        )
        data_copy = data.copy()
        try:

            data_copy.loc[
                :, "total_outstanding"
            ] = stock.trading.price_board([ticker])[
                self.dictionary["share_outstanding"][0]
            ][
                self.dictionary["share_outstanding"][1]
            ][
                0
            ]
        except (KeyError, IndexError):
            data_copy.loc[
                (data_copy[self.dictionary["ticker"]] == ticker)
                & (data_copy["exchange"] != "BOND"),
                "exchange",
            ] = "DELISTED"
        except ConnectionError:
            pprint(f"Connection error on {ticker}")
            raise
        return data_copy

    def reload_ticker_list(self) -> None:
        """
        _summary_
        create whole new ticker list
        Args:
            path (str): path to save ticker information
        """
        stock = vn(show_log=False).stock(
            symbol="ABC", source=self.dictionary["source"]
        )  # symbol in this step do not affect get all ticker api
        new_ticker = stock.listing.symbols_by_exchange()

        # add outstanding data for each stock
        all_ticker = Parallel(n_jobs=-2)(
            delayed(self.read_outstanding_1_stock)(stock, new_ticker)
            for stock in new_ticker[
                self.dictionary["ticker"]
            ].unique()
        )
        all_ticker = pd.concat(all_ticker)
        all_ticker["created_time"] = (
            datetime.today().date().strftime("%Y-%m-%d")
        )
        if not os.path.exists(
            self.path + self.dictionary["path_ticker_list"]
        ):
            all_ticker.to_parquet(
                self.path + self.dictionary["path_ticker_list"]
            )
        else:
            old_ticker = pd.read_parquet(
                self.path + self.dictionary["path_ticker_list"]
            )
            self.concat_new_and_old(
                old_ticker,
                all_ticker,
                "created_time",
                self.dictionary["path_ticker_list"],
            )

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_fixed(3),
        retry=retry_if_exception_type((ConnectionError)),
        after=lambda retry_state: logging.warning(
            "Retry attempt %s for data retrieval",
            retry_state.attempt_number,
        ),
    )
    def read_1_stock_data(
        self,
        ticker: str,
        end_date: str,
        start_date: str = "2013-01-01",
    ) -> pd.DataFrame:
        """_summary_
        get data for 1 stock only
        !Limitation:
            * Platform error when getting data for delisted stock and exchanged stock
            * Platform error when getting data bond and other instrument
        Args:
            ticker (str): _description_
            end_date (datetime.strftime): _description_
            start_date (datetime.strftime, optional): _description_. Defaults to "2013-01-01".

        Returns:
            pd.DataFrame: _description_
        """

        try:
            data = (
                vn(show_log=False)
                .stock(
                    symbol=ticker, source=self.dictionary["source"]
                )
                .quote.history(
                    start=start_date, end=end_date, interval="1D"
                )
            )
            data["ticker"] = ticker
            return data
        except ValueError:
            return pd.DataFrame()
        except TypeError:
            return pd.DataFrame()
        except KeyError:
            return pd.DataFrame()

    def get_data(self) -> pd.DataFrame:
        """_summary_
        Download all or update stock data
        Returns:
            _type_: _description_
        """
        ticker = self.get_ticker()
        path_temp = (
            self.path + self.dictionary["path_ticker_daily_data"]
        )
        if os.path.exists(path_temp):
            data = pd.read_parquet(path_temp)
            latest_data_date = data["time"].max()
            if (
                latest_data_date.strftime("%Y-%m-%d")
                < get_past_friday()
            ):
                latest_data_date = (
                    latest_data_date + pd.offsets.DateOffset(n=1)
                )
                pprint("update data")
                all_ticker = Parallel(n_jobs=-15)(
                    delayed(self.read_1_stock_data)(
                        stock, get_past_friday()
                    )
                    for stock in ticker[
                        self.dictionary["ticker"]
                    ].unique()
                )
                all_ticker = pd.concat(all_ticker)
                all_ticker.to_parquet(path_temp)
            else:
                all_ticker = data.copy()
        else:
            pprint("Downloading whole data set")
            all_ticker = Parallel(n_jobs=-15)(
                delayed(self.read_1_stock_data)(
                    stock, get_past_friday()
                )
                for stock in ticker[
                    self.dictionary["ticker"]
                ].unique()
            )
            all_ticker = pd.concat(all_ticker)
            all_ticker.to_parquet(path_temp)
        all_ticker = all_ticker.merge(
            ticker[
                [
                    self.dictionary["ticker"],
                    "exchange",
                    "total_outstanding",
                ]
            ],
            how="inner",
            left_on=["ticker"],
            right_on=[self.dictionary["ticker"]],
        )
        return clean_data(all_ticker)

    def concat_new_and_old(
        self,
        old: pd.DataFrame,
        new: pd.DataFrame,
        date_col: list,
        path_out: str,
    ):
        """_summary_
        function to concat data that has conflict with another except some specific columns
        Args:
            old (pd.DataFrame): _description_
            new (pd.DataFrame): _description_
            date_col (list): _description_
            path_out (str): _description_

        Returns:
            _type_: _description_
        """
        data = pd.concat([old, new], axis=0)
        data.sort_values(by=date_col, ascending=True)
        subset_cols = [
            col for col in data.columns if col not in date_col
        ]
        data.drop_duplicates(
            subset=subset_cols,
            inplace=True,
            keep="first",
        )
        if data.size > old.size:
            pprint("there some new data,updated data")
            data.to_parquet(self.path + path_out)
