import re
import narwhals as nw
from narwhals.typing import IntoFrameT

def select_columns_by_regex_pattern(df: IntoFrameT, pattern: str):
    for column in df.columns:
        if re.search(pattern, column):
            yield column


def select_columns_by_types(df: IntoFrameT, dtypes: list[nw.dtypes.DType]):
    for column, dtype in zip(df.schema.names(), df.schema.dtypes()):
        if dtype in dtypes:
            yield column

def select_columns(df: IntoFrameT, columns):
    if isinstance(columns, str):
        yield from select_columns_by_regex_pattern(df, columns)

    if (isinstance(columns, list) or isinstance(columns, tuple)) and columns:
        if issubclass(columns[0], nw.dtypes.DType):
            yield from select_columns_by_types(df, columns)
        elif isinstance(columns[0], str):
            yield from columns
        else:
            raise ValueError("Invalid columns type")
