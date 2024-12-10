import math
import warnings
from collections import defaultdict
from typing import Any, Literal, Sequence

import narwhals as nw
from narwhals.typing import IntoFrameT, IntoSeriesT
from pydantic import validate_call

from sklearo.encoding.base import BaseTargetEncoder
from sklearo.utils import infer_target_type, select_columns
from sklearo.validation import check_if_fitted, check_type_of_target, check_X_y


class WOEEncoder(BaseTargetEncoder):
    """Weight of Evidence (WOE) Encoder with support for multiclass classification.

    This class provides functionality to encode categorical features using the Weight of Evidence
    (WOE) technique. WOE is commonly used in credit scoring and other binary classification problems
    to transform categorical variables into continuous variables, however it can easily be extended
    to all sort of classification problems, including multiclass classification.

    WOE is defined as the natural logarithm of the ratio of the distribution of events for a class
    over the distribution of non-events for that class.

    ```
    WOE = ln((% of events) / (% of non events))
    ```

    Some articles explain it as `ln((% of non events) / (% of events))`, but in this way the WOE
    will be inversely correlated to the target variable. In this implementation, the WOE is
    calculated as the first formula, making it directly correlated to the target variable. I
    personally think that it makes the interpretation of the WOE easier and it won't affect the
    performance of the model.

    So let's say that the event to predict is default on a loan (class 1) and the non-event is
    not defaulting on a loan (class 0). The WOE for a category is calculated as follows:

    ```
    WOE = ln((% of defaults with the category) / (% of non-defaults in the category))
        = ln(
            (number of defaults from the category / total number of defaults) /
            (number of non-defaults from the category / total number of non-defaults)
          )
    ```

    The WOE value defined like this will be positive if the category is more likely to be default
    (positive class) and negative if it is more likely to be repaid (positive class).

    The WOE encoding is useful for logistic regression and other linear models, as it transforms
    the categorical variables into continuous variables that can be used as input features.

    Args:
        columns (str, list[str], list[nw.typing.DTypes]): list of columns to encode.

            - If a list of strings is passed, it is treated as a list of column names to encode.
            - If a single string is passed instead, it is treated as a regular expression pattern to
                match column names.
            - If a list of [`narwhals.typing.DTypes`](https://narwhals-dev.github.io/narwhals/api-reference/dtypes/)
                is passed, it will select all columns matching the specified dtype.

            Defaults to `[narwhals.Categorical, narwhals.String]`, meaning that all categorical
            and string columns are selected by default.

        underrepresented_categories (str): Strategy to handle underrepresented categories.
            Underrepresented categories in this context are categories that are never associated
            with one of the target classes. In this case the WOE is undefined (mathematically it
            would be either -inf or inf).

            - If `'raise'`, an error is raised when a category is underrepresented.
            - If `'fill'`, the underrepresented categories are encoded using the
                fill_values_underrepresented values.

        fill_values_underrepresented (list[int, float, None]): Fill values to use for
            underrepresented categories. The first value is used when the category has no events
            (e.g. defaults) and the second value is used when the category has no non-events (e.g.
            non defaults). Only used when `underrepresented_categories='fill'`.

        unseen (str): Strategy to handle categories that appear during the `transform` step but
            where never encountered in the `fit` step.

            - If `'raise'`, an error is raised when unseen categories are found.
            - If `'ignore'`, the unseen categories are encoded with the fill_value_unseen.

        fill_value_unseen (int, float, None): Fill value to use for unseen categories. Only used
            when `unseen='ignore'`.

        missing_values (str): Strategy to handle missing values.

            - If `'encode'`, missing values are initially replaced with `'MISSING'` and the WOE is
            computed as if it were a regular category.
            - If `'ignore'`, missing values are left as is.
            - If `'raise'`, an error is raised when missing values are found.

    Attributes:
        columns_ (list[str]): List of columns to be encoded, learned during fit.
        encoding_map_ (dict[str, dict[str, float]]): Nested dictionary mapping columns to their WOE
            values for each class, learned during fit.
        is_zero_one_target_ (bool): Whether the target variable is exactly 0 or 1 or not,
            learned during fit.
        feature_names_in_ (list[str]): List of feature names seen during fit.

    Examples:
        ```python
        import pandas as pd
        from sklearo.encoding import WOEEncoder
        data = {
            "category": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
            "target": [1, 0, 0, 1, 1, 0, 1, 1, 0],
        }
        df = pd.DataFrame(data)
        encoder = WOEEncoder()
        encoder.fit(df[["category"]], df["target"])
        encoded = encoder.transform(df[["category"]])
        print(encoded)
        category
        0 -0.223144
        1 -0.223144
        2 -0.223144
        3  1.029619
        4  1.029619
        5  1.029619
        6  1.029619
        7  1.029619
        8  1.029619
        ```
    """

    _encoder_name = "WOE"
    _allowed_types_of_target = ["binary", "multiclass"]

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        columns: Sequence[nw.typing.DTypes | str] | str = (
            nw.Categorical,
            nw.String,
        ),
        underrepresented_categories: Literal["raise", "fill"] = "raise",
        fill_values_underrepresented: Sequence[float | None] = (
            -999.0,
            999.0,
        ),
        unseen: Literal["raise", "ignore"] = "raise",
        fill_value_unseen: float | None = 0.0,
        missing_values: Literal["encode", "ignore", "raise"] = "encode",
    ) -> None:
        self.columns = columns
        self.underrepresented_categories = underrepresented_categories
        self.missing_values = missing_values
        self.fill_values_underrepresented = fill_values_underrepresented or (None, None)
        self.unseen = unseen
        self.fill_value_unseen = fill_value_unseen

    def _calculate_target_statistic(
        self, x_y: IntoFrameT, target_col: str, column: str
    ) -> dict[str, dict[str, float | int | None]]:
        """Calculate the Weight of Evidence for a column."""
        total_number_of_events = x_y[target_col].sum()
        total_number_of_non_events = x_y.shape[0] - total_number_of_events
        total_number_of_events_per_category = (
            x_y.group_by(column, drop_null_keys=True)
            .agg(
                n_events=nw.col(target_col).sum(), n_elements=nw.col(target_col).count()
            )
            .rows(named=True)
        )

        woe_dict = {}
        for row in total_number_of_events_per_category:
            n_events = row["n_events"]
            n_non_events = row["n_elements"] - n_events

            if n_events == 0:
                # the dist_ratio is 0 which would mean a woe of -inf
                if self.underrepresented_categories == "raise":
                    raise ValueError(
                        f"Underrepresented category {row[column]} found for the column {column}. "
                        "Please handle underrepresented categories for example by using a "
                        "RareLabelEncoder. Alternatively, set underrepresented_categories to "
                        "'fill'."
                    )
                else:  # fill
                    woe_dict[row[column]] = self.fill_values_underrepresented[0]
            elif n_non_events == 0:
                # the dist_ratio (and woe) would be infinite
                if self.underrepresented_categories == "raise":
                    raise ValueError(
                        f"Underrepresented category {row[column]} found for the column {column}. "
                        "Please handle underrepresented categories for example by using a "
                        "RareLabelEncoder. Alternatively, set underrepresented_categories to "
                        "'fill'."
                    )
                else:
                    woe_dict[row[column]] = self.fill_values_underrepresented[1]
            else:
                woe_dict[row[column]] = math.log(
                    (n_events / total_number_of_events)
                    / (n_non_events / total_number_of_non_events)
                )
        return woe_dict
