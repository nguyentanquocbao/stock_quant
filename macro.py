"""
_summary_
"""

import os
import shutil
import json
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
    macro_path: str

    def __post_init__(self):
        """Creates the directory path if it doesn't exist."""
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            print("created a new path for macro data")
        with open(self.macro_path, "r", encoding="utf-8") as a:
            self.macro_dictionary = json.load(a)
        self.url_imf = self.macro_dictionary["URL_IMF"]

    def get_imf(self, key):
        """_summary_
        get data from imf api with given string
        Args:
            key (_type_): _description_
            additional_col (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        data = requests.get(
            f"{self.url_imf}{key[0]}", timeout=500
        ).json()["CompactData"]["DataSet"]["Series"]
        all_data = []
        for dataset in data:
            observations = dataset.get("Obs", [])
            country_name = dataset.get("@REF_AREA", None)
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
            try:
                temp_df["value"] = pd.to_numeric(temp_df["value"])
            except KeyError:
                continue
            if len(key) > 1:
                for col in key[1].items():
                    temp_df[col[1]] = dataset.get(col[0], None)
            all_data.append(temp_df)
        final_df = pd.concat(all_data, ignore_index=True)
        return final_df

    def check_for_update(self, data, path_check):
        """_summary_
        Check error in new data and update if there already has some data
        Args:
            data (_type_): _description_
            path_check (_type_): _description_

        Returns:
            _type_: _description_
        """
        if os.path.exists(self.path + path_check):
            print(
                "There is some data in the given path, need to check"
            )
            data, update = self.check_data(
                path_check, data, ["country_name"]
            )
            if update:
                save_parquet(
                    self.path + path_check,
                    data,
                    partition=["country_name"],
                )
        else:
            print("create new data for CPI")
            save_parquet(
                self.path + path_check,
                data,
                partition=["country_name"],
            )
        return data

    def create_or_update_all(self):
        """_summary_
        create or update all data
        """
        for i in self.macro_dictionary["DATA"].items():
            self.check_for_update(self.get_imf(i[1]), i[0])
            print(f"{i[0]} finished")

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
                print(f"{sub_path}: no conflict in the new data")
                # Save the updated data to the old file
            print(f"{sub_path}: no new data")
            return old_df, False
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
            else:
                print(
                    f"{sub_path}: no conflict in old data when update data"
                )
            return updated_df, True
