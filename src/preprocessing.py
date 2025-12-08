# src/preprocessing.py

from __future__ import annotations

from typing import Iterable, List, Sequence
import pandas as pd


def fill_numeric_with_train_median(
    train_df: pd.DataFrame,
    other_dfs: Sequence[pd.DataFrame],
    numeric_cols: Iterable[str],
) -> pd.Series:
    """
    Impute numeric columns using medians computed *only* from the train_df,
    and apply the same medians to validation / test dataframes.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training dataframe. medians are computed from this.
    other_dfs : Sequence[pd.DataFrame]
        Other dataframes (e.g. [val_df, test_df]) to impute with the same medians.
    numeric_cols : Iterable[str]
        Columns that should be treated as numeric and imputed.

    Returns
    -------
    pd.Series
        Series of medians indexed by column name (useful for logging/debugging).
    """
    numeric_cols = list(numeric_cols)
    missing_cols = [c for c in numeric_cols if c not in train_df.columns]
    if missing_cols:
        raise KeyError(f"Columns not found in train_df: {missing_cols}")

    medians = train_df[numeric_cols].median()

    # Train
    train_df[numeric_cols] = train_df[numeric_cols].fillna(medians)

    # Other splits
    for df in other_dfs:
        # Skip if df is None
        if df is None:
            continue

        # Only fill columns that are present in df
        cols_in_df = [c for c in numeric_cols if c in df.columns]
        if cols_in_df:
            df[cols_in_df] = df[cols_in_df].fillna(medians[cols_in_df])

    return medians
