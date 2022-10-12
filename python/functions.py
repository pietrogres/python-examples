"""
Common functions script
"""
import re

import pandas as pd


def clean_whitespaces(str_: str) -> str:
    # remove every occurrence of multiple whitespaces in a string or whitespace at the beginning and at the end of it
    return re.sub(r'\s+', ' ', str_.strip())


def select_num_cols(df: pd.DataFrame, types: list = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']):
    """Extract list of numerical features from DataFrame"""
    return df.select_dtypes(include=types).columns.tolist()
