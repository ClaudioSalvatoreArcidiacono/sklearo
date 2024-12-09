import math
import warnings
from collections import defaultdict
from typing import Any, Literal, Sequence

import narwhals as nw
from narwhals.typing import IntoFrameT, IntoSeriesT
from pydantic import validate_call

from sklearo.encoding.base import BaseOneToOneEncoder
from sklearo.utils import infer_type_of_target, select_columns
from sklearo.validation import check_if_fitted, check_X_y


class TargetEncoder(BaseOneToOneEncoder):

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
        fill_value_unseen: int | float | None | Literal["mean"] = "mean",
        missing_values: Literal["encode", "ignore", "raise"] = "encode",
        type_of_target: Literal["auto", "binary", "multiclass", "continuous"] = "auto",
    ) -> None:
        self.columns = columns
        self.underrepresented_categories = underrepresented_categories
        self.missing_values = missing_values
        self.fill_values_underrepresented = fill_values_underrepresented or (None, None)
        self.unseen = unseen
        self.fill_value_unseen = fill_value_unseen
        self.type_of_target = type_of_target

    def _calculate_mean_target(
        self, x_y: IntoFrameT, target_cols: Sequence[str], column: str
    ) -> dict:
        mean_target_all_categories = (
            x_y.group_by(column)
            .agg(nw.col(target_col).mean() for target_col in target_cols)
            .rows(named=True)
        )

        if len(target_cols) == 1:
            mean_target = {}
            [target_column_name] = target_cols
            for mean_target_per_category in mean_target_all_categories:
                mean_target[mean_target_per_category[column]] = (
                    mean_target_per_category[target_column_name]
                )
        else:
            mean_target = defaultdict(dict)
            for target_column in target_cols:
                class_ = target_column.split("_")[-1]
                for mean_target_per_category in mean_target_all_categories:
                    mean_target[class_][mean_target_per_category[column]] = (
                        mean_target_per_category[target_column]
                    )
            mean_target = dict(mean_target)

        return mean_target

    @nw.narwhalify
    @check_X_y
    def fit(self, X: IntoFrameT, y: IntoSeriesT) -> "TargetEncoder":
        """Fit the encoder.

        Args:
            X (DataFrame): The input data.
            y (Series): The target variable.
        """

        self.columns_ = list(select_columns(X, self.columns))
        if not self.columns_:
            return self

        X = self._handle_missing_values(X)

        if self.type_of_target == "auto":
            self.type_of_target_ = infer_type_of_target(y)
        else:
            self.type_of_target_ = self.type_of_target

        if self.type_of_target_ == "binary":
            unique_classes = sorted(y.unique().to_list())
            try:
                greatest_class_as_int = int(unique_classes[1])
            except ValueError:
                self.is_zero_one_target_ = False
            else:
                if greatest_class_as_int == 1:
                    self.is_zero_one_target_ = True
                else:
                    self.is_zero_one_target_ = False

            if not self.is_zero_one_target_:
                y = y.replace_strict({unique_classes[0]: 0, unique_classes[1]: 1})

        else:
            self.is_zero_one_target_ = False

        X = X[self.columns_]

        if "target" in X.columns:
            target_col_name = "__target__"

        else:
            target_col_name = "target"

        X_y = X.with_columns(**{target_col_name: y})

        if self.type_of_target_ == "multiclass":
            unique_classes = y.unique().sort().to_list()
            self.unique_classes_ = unique_classes

            X_y = X_y.with_columns(
                nw.when(nw.col(target_col_name) == class_)
                .then(1)
                .otherwise(0)
                .alias(f"{target_col_name}_is_class_{class_}")
                for class_ in unique_classes
            )
            target_cols = [
                f"{target_col_name}_is_class_{class_}" for class_ in unique_classes
            ]

            if self.unseen == "fill" and self.fill_value_unseen == "mean":
                mean_targets = [X_y[target_cols].mean().rows(named=True)]
                mean_target_per_class = {}
                for target_col, class_ in zip(target_cols, unique_classes):
                    mean_target_per_class[class_] = mean_targets[target_col]
                self.mean_target_ = mean_target_per_class

        else:
            target_cols = [target_col_name]
            if self.unseen == "fill" and self.fill_value_unseen == "mean":
                self.mean_target_ = X_y[target_col_name].mean()

        self.encoding_map_ = {}
        for column in self.columns_:
            self.encoding_map_[column] = self._calculate_mean_target(
                X_y[target_cols + [column]], target_cols=target_cols, column=column
            )

        self.feature_names_in_ = list(X.columns)
        return self

    def _transform_binary_continuous(
        self, X: nw.DataFrame, unseen_per_col: dict
    ) -> IntoFrameT:
        fill_value_unseen = (
            self.fill_value_unseen
            if self.fill_value_unseen != "mean"
            else self.mean_target_
        )
        return X.with_columns(
            nw.col(column).replace_strict(
                {
                    **mapping,
                    **{
                        cat: fill_value_unseen for cat in unseen_per_col.get(column, [])
                    },
                }
            )
            for column, mapping in self.encoding_map_.items()
        )

    def _transform_multiclass(
        self, X: nw.DataFrame, unseen_per_col: dict
    ) -> IntoFrameT:
        fill_value_unseen = (
            {class_: self.fill_value_unseen for class_ in self.unique_classes_}
            if self.fill_value_unseen != "mean"
            else self.mean_target_
        )
        return X.with_columns(
            nw.col(column).replace_strict(
                {
                    **mapping,
                    **{
                        cat: fill_value_unseen[class_]
                        for cat in unseen_per_col.get(column, [])
                    },
                }
            )
            for column, class_mapping in self.encoding_map_.items()
            for class_, mapping in class_mapping.items()
        )

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

        if self.type_of_target_ in ("binary", "continuous"):
            return self._transform_binary_continuous(X, unseen_per_col)

        else:  # multiclass
            return self._transform_multiclass(X, unseen_per_col)
