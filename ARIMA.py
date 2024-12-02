from dataclasses import dataclass
from typing import List

import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_squared_error,
)
from statsmodels.tsa.arima.model import ARIMA
import itertools


@dataclass
class ArimaForecast:
    """_summary_
    A class to apply ARIMA into forecast
    Returns:
        _type_: plot and predicted-data
    """

    data: pd.DataFrame
    time_col: str
    val_col: str
    min_length: int
    max_p: int = 5
    max_d: int = 2
    max_q: int = 5

    def __post_init__(self):
        self.data.sort_values(by=self.time_col, inplace=True)
        self.data[self.time_col] = pd.to_datetime(
            self.data[self.time_col]
        )
        self.data.index = pd.date_range(
            start=self.data[self.time_col].min(),
            periods=len(self.data),
            freq="B",
        )

    def _find_best_order(self):
        """
        Find best ARIMA order using grid search and model selection criteria

        Parameters:
        -----------
        data : array-like
            Time series data

        Returns:
        --------
        tuple
            Best (p,d,q) order
        """
        # Generate all possible order combinations
        orders = list(
            itertools.product(
                range(self.max_p + 1),
                range(self.max_d + 1),
                range(self.max_q + 1),
            )
        )

        # Filter out non-stationary combinations if needed
        results = []
        for order in orders:
            model = ARIMA(self.data[self.val_col], order=order)
            model_fit = model.fit()
            results.append(
                {
                    "order": order,
                    "aic": model_fit.aic,
                    "bic": model_fit.bic,
                    "model": model_fit,
                }
            )

        # Sort by multiple criteria
        results_sorted = sorted(
            results, key=lambda x: (x["aic"], x["bic"])
        )
        return results_sorted[0]["order"]

    def ARIMA_walk_step(self, para=False):
        def suppress_warnings(func):
            def wrapper(*args, **kwargs):
                # Capture and suppress all warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return func(*args, **kwargs)

            return wrapper

        @suppress_warnings
        def ARIMA_all_data(time0):
            data_subset = self.data[
                self.data[self.time_col] <= time0
            ].tail(self.length)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=FutureWarning
                )
                warnings.filterwarnings(
                    "ignore", category=UserWarning
                )
            try:
                arima_model = ARIMA(
                    data_subset[self.val_col].values, order=self.order
                ).fit()
            except Exception as e:
                # Log or handle specific model fitting errors
                print(f"Model fitting failed for {time0}: {e}")
                return pd.DataFrame()
            return pd.DataFrame(
                {
                    self.time_col: pd.date_range(
                        start=time0, freq="B", periods=1
                    ),
                    "predict": arima_model.forecast(steps=1),
                }
            ).set_index(self.time_col)

        time_list = self.data[self.time_col][self.length :]
        if para:
            from joblib import Parallel, delayed

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                predicted = Parallel(n_jobs=-7)(
                    delayed(ARIMA_all_data)(i) for i in time_list
                )

            # Filter out any empty DataFrames
            predicted = [p for p in predicted if not p.empty]
            predicted = pd.concat(predicted, axis=0)
        else:
            predicted = pd.DataFrame()
            for i in time_list:
                predicted = pd.concat(
                    [
                        predicted,
                        ARIMA_all_data(
                            i,
                        ),
                    ],
                    axis=0,
                )
        return predicted

    def visualize_ARIMA(self, para=False, order=None):
        if order is not None:
            self.order = order
        predicted = self.ARIMA_walk_step(para)
        predicted = predicted.merge(
            self.data[[self.val_col, self.time_col]],
            how="left",
            left_index=True,
            right_index=True,
        )
        predicted.set_index(self.time_col, inplace=True, drop=True)
        _, ax = plt.subplots()

        # Plot the original series
        ax.plot(
            predicted.index,
            predicted[self.val_col],
            label="Original Series",
        )

        # Plot the AR model predictions
        ax.plot(
            predicted.index,
            predicted["predict"],
            label=f"AR( {self.order} ) Predictions",
            color="red",
        )

        # Add title and labels
        ax.set_title(f"AR( {self.order} ) Model Predictions")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")

        # Add legend
        ax.legend()

        # Show the plot
        plt.show()
        return predicted
