"""This module provides cross-validation utilities for model evaluation."""

import math

import narwhals as nw
from narwhals.typing import IntoFrameT, IntoSeriesT


def ceil_div(col: nw.Expr, divisor: int) -> nw.Expr:
    """Perform ceiling division on a column.

    This function divides each element in the given column by the specified divisor
    and returns the smallest integer greater than or equal to the result of the division.

    Args:
        col (nw.Expr): A numeric expression to be divided.
        divisor (int): A numeric value by which to divide the column.

    Returns:
        A column where each element is the result of the ceiling division.
    """
    return col // divisor + (col % divisor > 0)


def add_cv_fold_id_column_k_fold(X: IntoFrameT, k: int = 5) -> IntoFrameT:
    """Add a column `fold_id` to the DataFrame indicating the fold ID for k-fold cross-validation.

    This function divides the input DataFrame into k folds, ensuring that each fold
    has approximately the same number of samples. The fold IDs are assigned in a way
    that the first few folds may have one extra sample if the total number of samples
    is not perfectly divisible by k.

    Args:
        X (DataFrame): The input DataFrame to which the fold ID column will be added.
        k (int): The number of folds. Defaults to 5.

    Returns:
        DataFrame: The input DataFrame with an additional column for fold IDs.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'A': range(10)})
        >>> result = add_cv_fold_id_column_k_fold(df, k=3)
        >>> print(result)
           A  fold_id
        0  0        1
        1  1        1
        2  2        1
        3  3        1
        4  4        2
        5  5        2
        6  6        2
        7  7        3
        8  8        3
        9  9        3

    """
    n_folds_with_more_elements = len(X) % k
    n_elements_per_fold = math.floor(len(X) / k)

    return (
        X.with_columns(one=nw.lit(1))
        .with_columns(row_number=nw.col("one").cum_count())
        .with_columns(fold_id=ceil_div(nw.col("row_number"), n_elements_per_fold + 1))
        .with_columns(
            fold_id=nw.when(nw.col("fold_id") <= n_folds_with_more_elements)
            .then(nw.col("fold_id"))
            .otherwise(
                n_folds_with_more_elements
                + ceil_div(
                    nw.col("row_number")
                    - n_folds_with_more_elements * (n_elements_per_fold + 1),
                    n_elements_per_fold,
                )
            )
            - 1
        )
        .drop("one", "row_number")
    )


def add_cv_fold_id_column_stratified_k_fold(
    X: IntoFrameT, y: IntoSeriesT, k: int = 5
) -> IntoFrameT:
    """Add a `fold_id` column to the DataFrame indicating the fold ID for stratified k-fold CV.

    This function ensures that each fold has approximately the same proportion of each class
    as the original dataset. It calculates the fold IDs based on the distribution of the target
    variable, ensuring that each fold is representative of the overall dataset.

    Args:
        X (DataFrame): The input DataFrame to which the fold ID column will be added.
        y (Series): The target variable used for stratification.
        k (int): The number of folds. Defaults to 5.

    Returns:
        DataFrame: The input DataFrame with an additional column for fold IDs.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'A': range(10)})
        >>> y = pd.Series([0, 0, 1, 1, 0, 1, 0, 1, 0, 1])
        >>> result = add_cv_fold_id_column_stratified_k_fold(df, y, k=3)
        >>> print(result)
           A  fold_id
        0  0        1
        1  1        1
        2  2        1
        3  3        1
        4  4        2
        5  5        2
        6  6        2
        7  7        2
        8  8        3
        9  9        3
    """
    return (
        X.with_columns(one=nw.lit(1), target=y)
        .with_columns(
            count_per_class=nw.len().over("target"),
            row_number_per_class=nw.col("one").cum_count().over("target"),
        )
        .with_columns(
            n_folds_with_more_elements_per_class=nw.col("count_per_class") % k,
            n_elements_per_fold_per_class=nw.col("count_per_class") // k,
        )
        .with_columns(
            fold_id=ceil_div(
                nw.col("row_number_per_class"),
                nw.col("n_elements_per_fold_per_class") + 1,
            )
        )
        .with_columns(
            fold_id=nw.when(
                nw.col("fold_id") <= nw.col("n_folds_with_more_elements_per_class")
            )
            .then(nw.col("fold_id"))
            .otherwise(
                nw.col("n_folds_with_more_elements_per_class")
                + ceil_div(
                    nw.col("row_number_per_class")
                    - nw.col("n_folds_with_more_elements_per_class")
                    * (nw.col("n_elements_per_fold_per_class") + 1),
                    nw.col("n_elements_per_fold_per_class"),
                )
            )
            - 1
        )
        .drop(
            "target",
            "one",
            "count_per_class",
            "row_number_per_class",
            "n_folds_with_more_elements_per_class",
            "n_elements_per_fold_per_class",
        )
    )
