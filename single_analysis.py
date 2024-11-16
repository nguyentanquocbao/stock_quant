"""
_summary_
module to create visualization and test hypotheses for single stock
"""

import dataclasses

import numpy as np
import pandas as pd


@dataclasses
class single_analysis:
    dictionary: dict
    data: pd.DataFrame

    def sharp_ratio(self):
        data
