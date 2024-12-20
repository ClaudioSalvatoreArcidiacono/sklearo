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
        unseen: Literal["raise", "ignore", "fill"] = "raise",
        fill_value_unseen: float | None | Literal["mean"] = "mean",
        missing_values: Literal["encode", "ignore", "raise"] = "encode",
        underrepresented_categories: Literal["raise", "fill"] = "raise",
        fill_values_underrepresented: float | None | Literal["mean"] = "mean",
        target_type: Literal["auto", "binary", "multiclass", "continuous"] = "auto",
        smooth: Literal["auto"] | float = "auto",
    ) -> None:

        self.columns = columns
        self.missing_values = missing_values
        self.unseen = unseen
        self.fill_value_unseen = fill_value_unseen
        self.target_type = target_type
        self.smooth = smooth
        self.underrepresented_categories = underrepresented_categories
        self.fill_values_underrepresented = fill_values_underrepresented

    def _calculate_target_statistic(
        self, x_y: IntoFrameT, target_col: str, column: str
    ) -> dict:

        if column in (
            "category_count",
            "sum_target",
            "std_target",
            "smoothing",
            "shrinkage",
            "smoothed_target",
        ):
            # rename the column to avoid conflict
            original_column_name = column
            x_y = x_y.rename(columns={column: f"{column}_original"})
            column = f"{column}_original"
        else:
            original_column_name = column

        x_y_grouped = x_y.group_by(column, drop_null_keys=True).agg(
            category_count=nw.col(target_col).count(),
            sum_target=nw.col(target_col).sum(),
            **(
                {"std_target": nw.col(target_col).std()}
                if self.smooth == "auto"
                else {}
            ),
        )
        underrepresented_categories = x_y_grouped.filter(nw.col("category_count") == 1)[
            column
        ].to_list()
        if underrepresented_categories:
            if self.underrepresented_categories == "raise":
                raise ValueError(
                    f"Found underrepresented categories for the column {original_column_name}: "
                    f"{underrepresented_categories}. Please consider handling underrepresented "
                    "categories by using a RareLabelEncoder. Alternatively, set "
                    "underrepresented_categories to 'fill'."
                )
            else:
                if self.fill_values_underrepresented == "mean":
                    fill_values_underrepresented = x_y[target_col].mean()
                else:
                    fill_values_underrepresented = self.fill_values_underrepresented

                x_y_grouped = x_y_grouped.filter(
                    ~nw.col(column).is_in(underrepresented_categories)
                )
                encoding_dict = {
                    category: fill_values_underrepresented
                    for category in underrepresented_categories
                }
        else:
            encoding_dict = {}

        if self.smooth == "auto":
            target_std = x_y[target_col].std()
            x_y_grouped = x_y_grouped.with_columns(
                smoothing=nw.col("std_target") / target_std
            )
        else:
            x_y_grouped = x_y_grouped.with_columns(smoothing=nw.lit(self.smooth))

        categories_encoding_as_list = (
            x_y_grouped.with_columns(
                shrinkage=nw.col("category_count")
                / (nw.col("category_count") + nw.col("smoothing"))
            )
            .with_columns(
                smoothed_target=nw.col("shrinkage")
                * nw.col("sum_target")
                / nw.col("category_count")
                + (1 - nw.col("shrinkage"))
                * nw.col("sum_target")
                / nw.col("category_count")
            )
            .select(column, "smoothed_target")
            .rows()
        )

        encoding_dict.update(dict(categories_encoding_as_list))

        return encoding_dict
