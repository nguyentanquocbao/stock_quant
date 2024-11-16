"""
_summary_
"""

import os
import shutil
from dataclasses import dataclass

import pandas as pd
import requests


def create_path(path: str, sub_path: str = None) -> None:
    """_summary_
    create a main path and sub_path to store data
    Args:
        path (str): _description_
        sub_path (str): _description_
    """
    if not os.path.exists(path):
        os.mkdir(path)
    if sub_path is not None:
        if not os.path.exists(path + sub_path):
            os.mkdir(path + sub_path)


def save_parquet(path, data, partition=None, replace=False):
    """_summary_
    create new path for storing and store parquet data
    Args:
        path (_type_): _description_
        data (_type_): _description_
        partition (_type_, optional): _description_. Defaults to None.
    """
    if replace:
        shutil.rmtree(path)
    create_path(path)
    if partition is not None:
        data.to_parquet(
            path,
            partition_cols=partition,
            index=None,
        )
        print(f"save file with partition in path: {path}")
    else:
        data.to_parquet(path)
        print(f"save file without partition in path: {path}")


@dataclass
class MacroData:
    """_summary_

    Returns:
        _type_: _description_
    """

    path: str
    url_imf: str = "http://dataservices.imf.org/REST/SDMX_JSON.svc/"

    def __post_init__(self):
        """Creates the directory path if it doesn't exist."""
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            print("created a new path for macro data")

    def cpi(
        self,
        key: str = "CompactData/IFS/M..PCPI_IX",
        data_partition: list = "country_name",
        cpi_path: str = "/cpi",
    ):
        """_summary_

        Args:
            key (str, optional): _description_. Defaults to "CompactData/IFS/M..PCPI_IX".

        Returns:
            _type_: _description_
        """
        # path_cpi = self.path + cpi_path
        data = requests.get(
            f"{self.url_imf}{key}", timeout=500
        ).json()["CompactData"]["DataSet"]["Series"]
        all_data = []
        for dataset in data:
            # observations = dataset["Obs"]
            observations = dataset.get("Obs", [])
            # Create a DataFrame for the current dataset
            country_name = dataset.get("@REF_AREA", None)
            base_year = dataset.get("@BASE_YEAR", None)
            try:
                temp_df = pd.DataFrame(observations)
            except ValueError:
                continue
            temp_df.rename(
                columns={
                    "@TIME_PERIOD": "TIME_PERIOD",
                    "@OBS_VALUE": "value",
                },
                inplace=True,
            )
            temp_df["country_name"] = country_name
            temp_df["base_year"] = base_year
            try:
                temp_df["value"] = pd.to_numeric(temp_df["value"])
            except KeyError:
                continue
            all_data.append(temp_df)
        final_df = pd.concat(all_data, ignore_index=True)
        if os.path.exists(self.path + cpi_path):
            print(
                "There is some data in the given path, need to check, wait for results"
            )
            final_df, update = self.check_data(
                cpi_path, final_df, data_partition
            )
            if update:
                save_parquet(
                    self.path + cpi_path,
                    final_df,
                    partition=data_partition,
                )
        else:
            print("create new data for CPI")
            save_parquet(
                self.path + cpi_path,
                final_df,
                partition=data_partition,
            )
        return final_df

    def check_data(
        self, sub_path, new_df, data_partition: str = None
    ) -> pd.DataFrame:
        """_summary_

        Args:
            sub_path (_type_): _description_
            new_df (_type_): _description_
            data_partition (list, optional): _description_. Defaults to [].
        Returns:
            pd.DataFrame: _description_
        """
        old_df = pd.read_parquet(self.path + sub_path)
        merged_df = pd.merge(
            old_df, new_df, how="outer", indicator=True
        )
        print(old_df.shape, new_df.shape, merged_df.shape)
        # Check for conflicting values
        new_rows = merged_df[merged_df["_merge"] == "right_only"]
        conflict_old = merged_df[merged_df["_merge"] == "left_only"]
        if new_rows.empty:
            conflicts = pd.concat([old_df, new_df], axis=0)
            conflicts.drop_duplicates(inplace=True, keep=False)
            if not conflicts.empty:
                print("get conflict when update data")
                path_conflict = self.path + "/conflict" + sub_path
                create_path(self.path, "/conflict")
                create_path(self.path, "/conflict" + sub_path)
                save_parquet(path_conflict, conflicts, data_partition)
                print(f"Conflicts found and saved to {path_conflict}")
            else:
                print("CPI: no conflict")
                # Save the updated data to the old file
            print("CPI: no new data")
            update = False
            return old_df, update
        else:
            # Concatenate new rows to the old data
            updated_df = pd.concat(
                [old_df, new_rows.drop("_merge", axis=1)],
                ignore_index=True,
                axis=0,
            )
            if not conflict_old.empty:
                print("get conflict when update data")
                path_conflict = self.path + "/conflict" + sub_path
                create_path(self.path, "/conflict")
                create_path(self.path, "/conflict" + sub_path)
                save_parquet(path_conflict, conflicts, data_partition)
                print(f"Conflicts found and saved to {path_conflict}")
            update = True
            return updated_df, update
