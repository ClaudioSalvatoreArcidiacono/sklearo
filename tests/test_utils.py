import narwhals as nw
import pandas as pd
import polars as pl
import pytest

from sklearo.utils import (
    infer_target_type,
    select_columns,
    select_columns_by_regex_pattern,
    select_columns_by_types,
)

select_columns_by_regex_pattern = nw.narwhalify(select_columns_by_regex_pattern)
select_columns_by_types = nw.narwhalify(select_columns_by_types)
select_columns = nw.narwhalify(select_columns)


@pytest.mark.parametrize(
    "DataFrame", [pd.DataFrame, pl.DataFrame], ids=["pandas", "polars"]
)
class TestSelectColumns:

    @pytest.fixture
    def sample_data(self):
        data = {
            "A_col": [1, 2, 3],
            "B_col": [4, 5, 6],
            "C_col": [7, 8, 9],
        }
        return data

    def test_select_columns_by_regex_pattern(self, sample_data, DataFrame):
        df = DataFrame(sample_data)
        pattern = r"A_.*"
        selected_columns = list(select_columns_by_regex_pattern(df, pattern))
        assert selected_columns == ["A_col"]

    def test_select_columns_by_regex_pattern_select_all(self, sample_data, DataFrame):
        df = DataFrame(sample_data)
        pattern = r".*"
        selected_columns = list(select_columns_by_regex_pattern(df, pattern))
        assert selected_columns == ["A_col", "B_col", "C_col"]

    def test_select_columns_by_types(self, sample_data, DataFrame):
        df = DataFrame(sample_data)
        dtypes = [nw.dtypes.Int64]
        selected_columns = list(select_columns_by_types(df, dtypes))
        assert selected_columns == ["A_col", "B_col", "C_col"]

    def test_select_columns_with_string(self, sample_data, DataFrame):
        df = DataFrame(sample_data)
        selected_columns = list(select_columns(df, "A_.*"))
        assert selected_columns == ["A_col"]

    def test_select_columns_with_list_of_types(self, sample_data, DataFrame):
        df = DataFrame(sample_data)
        dtypes = [nw.dtypes.Int64]
        selected_columns = list(select_columns(df, dtypes))
        assert selected_columns == ["A_col", "B_col", "C_col"]

    def test_select_columns_with_list_of_strings(self, sample_data, DataFrame):
        df = DataFrame(sample_data)
        selected_columns = list(select_columns(df, ["A_col", "B_col"]))
        assert selected_columns == ["A_col", "B_col"]

    def test_select_columns_empty_list(self, sample_data, DataFrame):
        df = DataFrame(sample_data)
        selected_columns = list(select_columns(df, []))
        assert selected_columns == []

    def test_select_columns_empty_tuple(self, sample_data, DataFrame):
        df = DataFrame(sample_data)
        selected_columns = list(select_columns(df, ()))
        assert selected_columns == []

    def test_select_columns_invalid_type(self, sample_data, DataFrame):
        df = DataFrame(sample_data)
        with pytest.raises(ValueError, match="Invalid columns type"):
            list(select_columns(df, [1, 2]))


@pytest.mark.parametrize("Series", [pd.Series, pl.Series], ids=["pandas", "polars"])
class TestTypeOfTarget:

    @pytest.mark.parametrize(
        "data, expected",
        [
            ([1, 2, 3], "multiclass"),
            ([1, 2, 1], "binary"),
            ([1, 2, 4], "multiclass"),
            (["a", "b", "c"], "multiclass"),
            (["a", "b", "a"], "binary"),
            ([1.0, 2.0, 3.5], "continuous"),
            ([1.0, 2.0, 4.0], "continuous"),
            ([1.0, 3.5, 3.5], "continuous"),
            ([1.0, 2.0, 3.0], "multiclass"),
            ([1.0, 2.0, 1.0], "binary"),
            ([1.0, 4.0, 4.0], "binary"),
        ],
    )
    def test_type_of_target(self, Series, data, expected):
        series = Series(data)
        assert infer_target_type(series) == expected

    def test_type_of_target_unknown(self, Series):
        data = [None, None, None]
        series = Series(data)
        assert infer_target_type(series) == "unknown"
