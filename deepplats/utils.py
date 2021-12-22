"""Utilities for deepplats library.
"""
import pandas as pd

DATA_ROOT = (
    "https://raw.githubusercontent.com/GuillaumeDMMarion/deep-plats/master/data/{}.csv"
)


def get_data(filename: str = "example1") -> pd.DataFrame:
    """Get example data.

    Args:
        filename: Name of the file.
    """
    url = DATA_ROOT.format(filename)
    return pd.read_csv(url, index_col=0)
