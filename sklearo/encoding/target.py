from typing import Literal, Sequence

import narwhals as nw
from narwhals.typing import IntoFrameT
from pydantic import validate_call

from sklearo.encoding.base import BaseTargetEncoder


class TargetEncoder(BaseTargetEncoder):
    """
    Target Encoder for categorical features.

    This class provides functionality to encode categorical features using the Target Encoding
    technique. Target Encoding replaces each category with the mean of the target variable for that
    category. This method is particularly useful for handling categorical variables in machine
    learning models, especially when the number of categories is large.

    Args:
        columns (str, list[str], list[nw.typing.DTypes]): List of columns to encode.

            - If a list of strings is passed, it is treated as a list of column names to encode.
            - If a single string is passed instead, it is treated as a regular expression pattern to
              match column names.
            - If a list of
              [`narwhals.typing.DTypes`](https://narwhals-dev.github.io/narwhals/api-reference/dtypes/)
              is passed, it will select all columns matching the specified dtype.

        unseen (str): Strategy to handle categories that appear during the `transform` step but were
            never encountered in the `fit` step.

            - If `'raise'`, an error is raised when unseen categories are found.
            - If `'ignore'`, the unseen categories are encoded with the fill_value_unseen.

        fill_value_unseen (int, float, None | Literal["mean"]): Fill value to use for unseen
            categories. Defaults to `"mean"`, which will use the mean of the target variable.

        missing_values (str): Strategy to handle missing values.

            - If `'encode'`, missing values are initially replaced with a specified fill value and
              the mean is computed as if it were a regular category.
            - If `'ignore'`, missing values are left as is.
            - If `'raise'`, an error is raised when missing values are found.

        target_type (str): Type of the target variable.

            - If `'auto'`, the type is inferred from the target variable using
                [`infer_target_type`][sklearo.utils.infer_target_type].
            - If `'binary'`, the target variable is binary.
            - If `'multiclass'`, the target variable is multiclass.
            - If `'continuous'`, the target variable is continuous.

    Attributes:
        columns_ (list[str]): List of columns to be encoded, learned during fit.
        encoding_map_ (dict[str, float]): Mapping of categories to their mean target values, learned
            during fit.

    Examples:
        ```python
        import pandas as pd
        from sklearo.encoding import TargetEncoder
        data = {
            "category": ["A", "A", "B", "B", "C", "C"],
            "target": [1, 0, 1, 0, 1, 0],
        }
        df = pd.DataFrame(data)
        encoder = TargetEncoder()
        encoder.fit(df[["category"]], df["target"])
        encoded = encoder.transform(df[["category"]])
        print(encoded)
        category
        0 0.5
        1 0.5
        2 0.5
        3 0.5
        4 0.5
        5 0.5
        ```
    """

    _encoder_name = "mean_target"
    _allowed_types_of_target = ["binary", "multiclass", "continuous"]

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        columns: Sequence[nw.typing.DTypes | str] | str = (
            nw.Categorical,
            nw.String,
        ),
        unseen: Literal["raise", "ignore"] = "raise",
        fill_value_unseen: int | float | None | Literal["mean"] = "mean",
        missing_values: Literal["encode", "ignore", "raise"] = "encode",
        target_type: Literal["auto", "binary", "multiclass", "continuous"] = "auto",
    ) -> None:

        self.columns = columns
        self.missing_values = missing_values
        self.unseen = unseen
        self.fill_value_unseen = fill_value_unseen
        self.target_type = target_type

    def _calculate_target_statistic(
        self, x_y: IntoFrameT, target_col: str, column: str
    ) -> dict:
        mean_target_all_categories = (
            x_y.group_by(column).agg(nw.col(target_col).mean()).rows()
        )
        mean_target = dict(mean_target_all_categories)
        return mean_target
