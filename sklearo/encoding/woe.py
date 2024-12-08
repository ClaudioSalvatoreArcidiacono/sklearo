import math
import warnings
from collections import defaultdict
from typing import Any, Literal, Sequence

import narwhals as nw
from narwhals.typing import IntoFrameT, IntoSeriesT
from pydantic import validate_call

from sklearo.base import BaseTransformer
from sklearo.utils import select_columns
from sklearo.validation import check_if_fitted, check_X_y


class WOEEncoder(BaseTransformer):
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

        fill_value_unseen (int, float, None): Fill value to use for unseen categories. Only used when
            `unseen='ignore'`.

        missing_values (str): Strategy to handle missing values.

            - If `'encode'`, missing values are initially replaced with `'MISSING'` and the WOE is
            computed as if it were a regular category.
            - If `'ignore'`, missing values are left as is.
            - If `'raise'`, an error is raised when missing values are found.

    Attributes:
        columns_ (list[str]): List of columns to be encoded, learned during fit.
        encoding_map_ (dict[str, dict[str, float]]): Nested dictionary mapping columns to their WOE
            values for each class, learned during fit.
        is_binary_target_ (bool): Whether the target variable is binary (exactly 0 or 1) or not,
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

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        columns: Sequence[nw.typing.DTypes | str] | str = (
            nw.Categorical,
            nw.String,
        ),
        underrepresented_categories: Literal["raise", "fill"] = "raise",
        fill_values_underrepresented: Sequence[int | float | None] = (
            -999.0,
            999.0,
        ),
        unseen: Literal["raise", "ignore"] = "raise",
        fill_value_unseen: int | float | None = 0.0,
        missing_values: Literal["encode", "ignore", "raise"] = "encode",
    ) -> None:
        self.columns = columns
        self.underrepresented_categories = underrepresented_categories
        self.missing_values = missing_values
        self.fill_values_underrepresented = fill_values_underrepresented or (None, None)
        self.unseen = unseen
        self.fill_value_unseen = fill_value_unseen

    def _handle_missing_values(self, X: IntoFrameT) -> IntoFrameT:
        if self.missing_values == "ignore":
            return X
        if self.missing_values == "raise":
            if max(X[self.columns_].null_count().row(0)) > 0:
                raise ValueError(
                    f"Some columns have missing values. "
                    "Please handle missing values before encoding or set "
                    "missing_values to either 'ignore' or 'encode'."
                )
            return X
        if self.missing_values == "encode":
            # fillna does not work with categorical columns, so we use this
            # workaround
            return X.with_columns(
                nw.when(nw.col(column).is_null())
                .then(nw.lit("MISSING"))
                .otherwise(nw.col(column))
                .alias(column)
                for column in self.columns_
            )

    def _calculate_woe(
        self, x: IntoSeriesT, y: IntoSeriesT, unique_classes: list[Any]
    ) -> dict[str, dict[str, float | int | None]]:
        """Calculate the Weight of Evidence for a column."""
        unique_categories = x.unique().to_list()
        if x.name == "target":
            target_col_name = "target_"
        else:
            target_col_name = "target"

        categories_class_info_as_rows = (
            x.to_frame()
            .with_columns(**{target_col_name: y})
            .with_columns(
                total_events_per_class=nw.col(x.name).count().over(target_col_name),
                total_elements_per_category=nw.col(target_col_name)
                .count()
                .over(x.name),
            )
            .group_by(x.name, target_col_name)
            .agg(
                n_events_per_category=nw.col(target_col_name).count(),
                total_events_per_class=nw.col("total_events_per_class").max(),
                total_elements_per_category=nw.col("total_elements_per_category").max(),
            )
            .with_columns(
                distribution_of_events_per_category=nw.col("n_events_per_category")
                / nw.col("total_events_per_class"),
                n_non_events_per_category=nw.col("total_elements_per_category")
                - nw.col("n_events_per_category"),
                total_number_of_non_events=x.shape[0] - nw.col("n_events_per_category"),
            )
            .with_columns(
                distribution_of_non_events_per_category=nw.col(
                    "n_non_events_per_category"
                )
                / nw.col("total_number_of_non_events"),
            )
            .with_columns(
                dist_ratio=nw.col("distribution_of_events_per_category")
                / nw.col("distribution_of_non_events_per_category"),
            )
            .select(
                [
                    x.name,
                    target_col_name,
                    "dist_ratio",
                    "n_events_per_category",
                    "n_non_events_per_category",
                ]
            )
            .rows(named=True)
        )

        categories_class_info_as_dict = defaultdict(dict)

        for row in categories_class_info_as_rows:
            categories_class_info_as_dict[row[x.name]][row[target_col_name]] = {
                "dist_ratio": row["dist_ratio"],
                "n_events_per_category": row["n_events_per_category"],
                "n_non_events_per_category": row["n_non_events_per_category"],
            }
        # categories_class_info_as_dict = dict(categories_class_info_as_dict)
        # categories_class_info_as_dict
        woe_dict_per_category = defaultdict(dict)
        underrepresented_category_per_class = list()

        for category in sorted(cat for cat in unique_categories if cat is not None):
            for class_ in sorted(unique_classes):
                category_class_info = categories_class_info_as_dict[category].get(
                    class_, {}
                )
                if not category_class_info:
                    # This means that the n_events_per_category is 0
                    # and that we have only non-events in this category
                    # the dist_ratio is 0 which would mean a woe of -inf
                    if self.underrepresented_categories == "raise":
                        underrepresented_category_per_class.append(
                            {
                                "category": category,
                                "class": class_,
                            }
                        )
                    else:  # fill
                        woe_dict_per_category[class_][category] = (
                            self.fill_values_underrepresented[0]
                        )
                        underrepresented_category_per_class.append(
                            {
                                "category": category,
                                "class": class_,
                                "fill_value": self.fill_values_underrepresented[0],
                            }
                        )
                elif category_class_info["n_non_events_per_category"] == 0:
                    # This means that the n_non_events_per_category is 0
                    # and that we have only events in this category
                    # the dist_ratio (and woe) would be infinite
                    if self.underrepresented_categories == "raise":
                        underrepresented_category_per_class.append(
                            {
                                "category": category,
                                "class": class_,
                            }
                        )
                    else:  # fill
                        woe_dict_per_category[class_][category] = (
                            self.fill_values_underrepresented[1]
                        )
                        underrepresented_category_per_class.append(
                            {
                                "category": category,
                                "class": class_,
                                "fill_value": self.fill_values_underrepresented[1],
                            }
                        )
                else:
                    woe_dict_per_category[class_][category] = math.log(
                        category_class_info["dist_ratio"]
                    )
        if underrepresented_category_per_class:
            if self.underrepresented_categories == "raise":
                raise ValueError(
                    f"Underrepresented categories {underrepresented_category_per_class} found for "
                    f"the column {x.name}. "
                    "Please handle underrepresented categories for example by using a "
                    "RareLabelEncoder. Alternatively, set underrepresented_categories to 'fill'."
                )
            else:  # Fill
                warnings.warn(
                    f"Underrepresented categories found for the column {x.name}. "
                    "Please handle underrepresented categories for example by using a "
                    "RareLabelEncoder. These categories will be encoded using the fill value as: \n"
                    f"{underrepresented_category_per_class}."
                )
        return dict(woe_dict_per_category)

    @nw.narwhalify
    @check_X_y
    def fit(self, X: IntoFrameT, y: IntoSeriesT) -> "WOEEncoder":
        """Fit the encoder.

        Args:
            X (DataFrame): The input data.
            y (Series): The target variable.
        """

        self.columns_ = list(select_columns(X, self.columns))
        X = self._handle_missing_values(X)
        self.encoding_map_ = {}

        if not self.columns_:
            return self

        unique_classes = sorted(y.unique().to_list())
        self.unqiue_classes_ = unique_classes

        if len(unique_classes) == 2:
            unique_classes = [unique_classes[1]]

            try:
                greatest_class_as_int = int(unique_classes[0])
            except ValueError:
                self.is_binary_target_ = False
            else:
                if greatest_class_as_int == 1:
                    self.is_binary_target_ = True
                else:
                    self.is_binary_target_ = False
        else:
            self.is_binary_target_ = False

        for column in self.columns_:
            self.encoding_map_[column] = self._calculate_woe(
                X[column], y, unique_classes
            )

        self.feature_names_in_ = list(X.columns)
        return self

    @nw.narwhalify
    @check_if_fitted
    def transform(self, X: IntoFrameT) -> IntoFrameT:
        """Transform the data.

        Args:
            X (DataFrame): The input data.
        """
        X = self._handle_missing_values(X)
        unseen_per_col = {}
        for column, mapping in self.encoding_map_.items():
            uniques = X[column].unique()
            unseen_cats = uniques.filter(
                (
                    ~uniques.is_in(next(iter(mapping.values())).keys())
                    & ~uniques.is_null()
                )
            ).to_list()
            if unseen_cats:
                unseen_per_col[column] = unseen_cats

        if unseen_per_col:
            if self.unseen == "raise":
                raise ValueError(
                    f"Unseen categories {unseen_per_col} found during transform. "
                    "Please handle unseen categories for example by using a RareLabelEncoder. "
                    "Alternatively, set unseen to 'ignore'."
                )
            else:
                warnings.warn(
                    f"Unseen categories {unseen_per_col} found during transform. "
                    "Please handle unseen categories for example by using a RareLabelEncoder. "
                    f"These categories will be encoded as {self.fill_value_unseen}."
                )

        X_out = X.with_columns(
            nw.col(column)
            .replace_strict(
                {
                    **mapping,
                    **{cat: self.fill_value_unseen for cat in unseen_cats},
                }
            )
            .alias(
                f"{column}"
                if self.is_binary_target_
                else f"{column}_WOE_class_{class_}"
            )
            for column, classes_mapping in self.encoding_map_.items()
            for class_, mapping in classes_mapping.items()
        )

        # In case of binary target, the original columns are replaced with the encoded columns.
        # If it is not a binary target, the original columns need to be dropped before returning.
        if not self.is_binary_target_:
            X_out = X_out.drop(*self.columns_)

        return X_out

    @check_if_fitted
    def get_feature_names_out(self) -> list[str]:
        """Get the feature names after encoding."""
        if self.is_binary_target_:
            return self.feature_names_in_
        else:
            return [
                feat for feat in self.feature_names_in_ if feat not in self.columns_
            ] + [
                f"{column}_WOE_class_{class_}"
                for column, classes_mapping in self.encoding_map_.items()
                for class_ in classes_mapping
            ]
