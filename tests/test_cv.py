import pandas as pd
import polars as pl
import pytest

from sklearo.cv import (
    add_cv_fold_id_column_k_fold,
    add_cv_fold_id_column_stratified_k_fold,
)


@pytest.mark.parametrize(
    "DataFrame", [pd.DataFrame, pl.DataFrame], ids=["pandas", "polars"]
)
class TestCVFunctions:

    def test_add_cv_fold_id_column_k_fold(self, DataFrame):
        data = {
            "A": range(10),
        }
        df = DataFrame(data)
        result = add_cv_fold_id_column_k_fold(df, k=3)

        assert result["fold_id"].to_list() == [1, 1, 1, 1, 2, 2, 2, 3, 3, 3]

    def test_add_cv_fold_id_column_stratified_k_fold(self, DataFrame):
        data = {
            "A": range(10),
        }
        target = [0, 0, 1, 1, 0, 1, 0, 1, 0, 1]
        df = DataFrame(data)
        y = DataFrame({"target": target})

        result = add_cv_fold_id_column_stratified_k_fold(df, y["target"], k=3)

        assert result["fold_id"].to_list() == [1, 1, 1, 1, 2, 2, 2, 2, 3, 3]

    def test_add_cv_fold_id_column_k_fold_divisible(self, DataFrame):
        data = {
            "A": range(9),
        }
        df = DataFrame(data)
        result = add_cv_fold_id_column_k_fold(df, k=3)

        assert result["fold_id"].to_list() == [1, 1, 1, 2, 2, 2, 3, 3, 3]

    def test_add_cv_fold_id_column_stratified_k_fold_divisible(self, DataFrame):
        data = {
            "A": range(9),
        }
        target = [0, 0, 1, 1, 0, 0, 0, 1, 0]
        df = DataFrame(data)
        y = DataFrame({"target": target})

        result = add_cv_fold_id_column_stratified_k_fold(df, y["target"], k=3)

        assert result["fold_id"].to_list() == [1, 1, 1, 2, 2, 2, 3, 3, 3]