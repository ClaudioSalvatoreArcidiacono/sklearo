"""TargetEncoder class for encoding categorical features using the Target Encoding technique."""

from typing import Literal, Sequence

import narwhals as nw
from narwhals.typing import IntoFrameT
from pydantic import Field, validate_call
from typing_extensions import Annotated

from sklearo.encoding.base import BaseTargetEncoder


class TargetEncoder(BaseTargetEncoder):
    """Target Encoder for categorical features.

    This class provides functionality to encode categorical features using the Target Encoding
    technique. Target Encoding replaces each category with the mean of the target variable for that
    category. This method is particularly useful for handling categorical variables in machine
    learning models, especially when the number of categories is large.

    The mean target per category is blended with the overall mean target using a smoothing
    parameter. The smoothing parameter is calculated as explained [here](https://scikit-learn.org/1.5/modules/preprocessing.html#target-encoder).

    Notes:
        ## Cross-fitting 🏋️‍♂️

        This implementation uses an internal cross-fitting strategy to calculate the mean target
        values for the `fit_transform` method. **This means that calling `.fit(X, y).transform(X)`
        will not return the same result as calling `.fit_transform(X, y)`.** When calling
        `.fit_transform(X, y)`, the dataset is initially split into k folds (configurable via the
        `cv` parameter) then for each fold the mean target values are calculated using the data from
        all other folds. Finally, the transformer is fitted on the entire dataset. This is done to
        prevent leakage of the target information into the training data. This idea has been taken
        from [scikit-learn's implementation of
        TargetEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html).
        The reader is encouraged to learn more about cross-fitting on the [scikit-learn
        documentation](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_target_encoder_cross_val.html#sphx-glr-auto-examples-preprocessing-plot-target-encoder-cross-val-py).

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

        underrepresented_categories (str): Strategy to handle categories that are underrepresented in
            the training data.

            - If `'raise'`, an error is raised when underrepresented categories are found.
            - If `'fill'`, underrepresented categories are filled with a specified fill value.

        fill_values_underrepresented (float, None | Literal["mean"]): Fill value to use for underrepresented
            categories. Defaults to `"mean"`, which will use the mean of the target variable.

        target_type (str): Type of the target variable.

            - If `'auto'`, the type is inferred from the target variable using
                [`infer_target_type`][sklearo.utils.infer_target_type].
            - If `'binary'`, the target variable is binary.
            - If `'multiclass'`, the target variable is multiclass.
            - If `'continuous'`, the target variable is continuous.

        smooth (float, Literal["auto"]): Smoothing parameter to avoid overfitting. If `'auto'`, the
            smoothing parameter is calculated based on the variance of the target variable.

        cv (int): Number of cross-validation folds to use for calculating the target encoding.



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
        cv: Annotated[int, Field(ge=2)] = 5,
    ) -> None:
        """Class constructor for TargetEncoder."""
        self.columns = columns
        self.missing_values = missing_values
        self.unseen = unseen
        self.fill_value_unseen = fill_value_unseen
        self.target_type = target_type
        self.smooth = smooth
        self.underrepresented_categories = underrepresented_categories
        self.fill_values_underrepresented = fill_values_underrepresented
        self.cv = cv

    def _calculate_target_statistic(
        self, x_y: IntoFrameT, target_col: str, column: str
    ) -> dict:
        if column in (
            "count_per_category",
            "sum_target_per_category",
            "std_target_per_category",
            "smoothing",
            "shrinkage",
            "smoothed_target",
        ):
            # rename the column to avoid conflict
            original_column_name = column
            x_y = x_y.rename(mapping={column: f"{column}_original"})
            column = f"{column}_original"
        else:
            original_column_name = column

        mean_target = x_y[target_col].mean()

        x_y_grouped = x_y.group_by(column, drop_null_keys=True).agg(
            count_per_category=nw.col(target_col).count(),
            sum_target_per_category=nw.col(target_col).sum(),
            **(
                {"var_target_per_category": nw.col(target_col).var()}
                if self.smooth == "auto"
                else {}
            ),
        )
        underrepresented_categories = x_y_grouped.filter(
            nw.col("count_per_category") == 1
        )[column].to_list()
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
                    fill_values_underrepresented = mean_target
                else:
                    fill_values_underrepresented = self.fill_values_underrepresented

                x_y_grouped = x_y_grouped.filter(
                    ~nw.col(column).is_in(underrepresented_categories)
                )
                fill_values_underrepresented_dict = {
                    category: fill_values_underrepresented
                    for category in underrepresented_categories
                }
        else:
            fill_values_underrepresented_dict = {}

        if self.smooth == "auto":
            var_target = x_y[target_col].var()
            x_y_grouped = x_y_grouped.with_columns(
                smoothing=nw.col("var_target_per_category") / var_target
            )
        else:
            x_y_grouped = x_y_grouped.with_columns(smoothing=nw.lit(self.smooth))

        categories_encoding_as_list = (
            x_y_grouped.with_columns(
                shrinkage=nw.col("count_per_category")
                / (nw.col("count_per_category") + nw.col("smoothing"))
            )
            .with_columns(
                smoothed_target=nw.col("shrinkage")
                * nw.col("sum_target_per_category")
                / nw.col("count_per_category")
                + (1 - nw.col("shrinkage")) * mean_target
            )
            .select(column, "smoothed_target")
            .rows()
        )

        encoding_dict = dict(categories_encoding_as_list)
        encoding_dict.update(fill_values_underrepresented_dict)
        return encoding_dict
