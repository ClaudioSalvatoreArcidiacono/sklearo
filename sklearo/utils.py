import inspect
import re
from typing import Sequence

import narwhals as nw
from narwhals.typing import IntoSeriesT

INTEGER_DTYPES = [
    nw.Int8,
    nw.Int16,
    nw.Int32,
    nw.Int64,
    nw.UInt8,
    nw.UInt16,
    nw.UInt32,
    nw.UInt64,
]

FLOAT_DTYPES = [nw.Float32, nw.Float64]


def select_columns_by_regex_pattern(df: nw.DataFrame, pattern: str):
    for column in df.columns:
        if re.search(pattern, column):
            yield column


def select_columns_by_types(df: nw.DataFrame, dtypes: list[nw.dtypes.DType]):
    for column, dtype in zip(df.schema.names(), df.schema.dtypes()):
        if dtype in dtypes:
            yield column


def select_columns(df: nw.DataFrame, columns: Sequence[nw.typing.DTypes | str] | str):
    if isinstance(columns, str):
        yield from select_columns_by_regex_pattern(df, columns)

    elif (isinstance(columns, list) or isinstance(columns, tuple)) and columns:
        if inspect.isclass(columns[0]) and issubclass(columns[0], nw.dtypes.DType):
            yield from select_columns_by_types(df, columns)
        elif isinstance(columns[0], str):
            yield from columns
        else:
            raise ValueError("Invalid columns type")


@nw.narwhalify
def infer_type_of_target(y: IntoSeriesT) -> str:
    """Infer the type of target variable based on the input series.

    This function determines the type of target variable based on the unique values and data type
    of the input series.

    Args:
        y (nw.Series): The target variable series.

    Returns:
        str: The inferred type of target variable, which can be one of the following:

            - `"binary"`: Returned when the target variable contains exactly two unique values and
              is of an integer, boolean, string or categorical data type or it's floating point with
              no decimal digits (e.g. `[0.0, 1.0]`).
            - `"multiclass"`: Returned when the target variable has more than two unique values and
              is of an integer, boolean, string or categorical data type or it's floating point with
              no decimal digits (e.g. `[0.0, 1.0, 2.0]`). In case of floating point data type, the
              unique values should be consecutive integers.
            - `"continuous"`: Returned when the target variable is of a floating-point data type and
              contains at least one non-integer value or the unique values are not consecutive
              integers.
            - `"unknown"`: Returned when the input series is none of the above types.

    Examples:
        >>> type_of_target(pd.Series([1, 2, 3])
        "multiclass"
        >>> type_of_target(pd.Series([1, 2, 1])
        "binary"
        >>> type_of_target(pd.Series([1, 2, 4])
        "multiclass"
        >>> type_of_target(pd.Series(["a", "b", "c"])
        "multiclass"
        >>> type_of_target(pd.Series(["a", "b", "a"])
        "binary"
        >>> type_of_target(pd.Series([1.0, 2.0, 3.5])
        "continuous"
        >>> type_of_target(pd.Series([1.0, 3.5, 3.5])
        "continuous"
        >>> type_of_target(pd.Series([1.0, 2.0, 4.0])
        "continuous"
        >>> type_of_target(pd.Series([1.0, 4.0, 4.0])
        "binary"
        >>> type_of_target(pd.Series([1.0, 2.0, 3.0])
        "multiclass"
        >>> type_of_target(pd.Series([1.0, 2.0, 1.0])
        "binary"

    """
    if y.dtype == nw.Boolean:
        return "binary"

    if y.dtype in INTEGER_DTYPES or y.dtype in (nw.String, nw.Categorical):
        if len(y.unique().to_list()) == 2:
            return "binary"
        else:
            return "multiclass"

    if y.dtype in FLOAT_DTYPES:
        if (y % 1 != 0).any():
            return "continuous"

        else:
            unique = y.unique()
            if len(unique.to_list()) == 2:
                return "binary"
            sorted_unique = unique.sort()
            labels_diff = sorted_unique - sorted_unique.shift(1)
            if labels_diff.max() == labels_diff.min() == 1.0:
                return "multiclass"
            else:
                return "continuous"

    return "unknown"
