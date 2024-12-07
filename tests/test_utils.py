import pandas as pd
import polars as pl
import pytest
import narwhals as nw
from sklearo.utils import (
    select_columns_by_regex_pattern,
    select_columns_by_types,
    select_columns,
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