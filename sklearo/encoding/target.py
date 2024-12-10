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
        unseen: Literal["raise", "ignore"] = "raise",
        fill_value_unseen: int | float | None | Literal["mean"] = "mean",
        missing_values: Literal["encode", "ignore", "raise"] = "encode",
        type_of_target: Literal["auto", "binary", "multiclass", "continuous"] = "auto",
    ) -> None:
        self.columns = columns
        self.missing_values = missing_values
        self.unseen = unseen
        self.fill_value_unseen = fill_value_unseen
        self.type_of_target = type_of_target

    def _calculate_mean_target(
        self, x_y: IntoFrameT, target_col: Sequence[str], column: str
    ) -> dict:
        debug_df = x_y.to_native()
        mean_target_all_categories = (
            x_y.group_by(column).agg(nw.col(target_col).mean()).rows(named=True)
        )
        mean_target = {}
        for mean_target_per_category in mean_target_all_categories:
            mean_target[mean_target_per_category[column]] = mean_target_per_category[
                target_col
            ]

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
        self.encoding_map_ = {}

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

        if "target" in X.columns:
            target_col_name = "__target__"

        else:
            target_col_name = "target"

        if not self.columns_:
            return self

        X_y = X[self.columns_].with_columns(**{target_col_name: y})

        if self.type_of_target_ == "multiclass":
            unique_classes = y.unique().sort().to_list()
            self.unique_classes_ = unique_classes
            self.encoding_map_ = defaultdict(dict)
            if self.unseen == "fill" and self.fill_value_unseen == "mean":
                self.mean_target_ = {}
            for class_ in unique_classes:
                X_y_binarized = X_y.with_columns(
                    nw.when(nw.col(target_col_name) == class_)
                    .then(1)
                    .otherwise(0)
                    .alias(target_col_name)
                )
                for column in self.columns_:
                    debug_df = X_y_binarized[[column, target_col_name]].to_native()
                    self.encoding_map_[column][class_] = self._calculate_mean_target(
                        X_y_binarized[[column, target_col_name]],
                        target_col=target_col_name,
                        column=column,
                    )
                if self.unseen == "fill" and self.fill_value_unseen == "mean":
                    self.mean_target_[class_] = X_y_binarized[target_col_name].mean()

        else:
            for column in self.columns_:
                self.encoding_map_[column] = self._calculate_mean_target(
                    X_y[[column, target_col_name]],
                    target_col=target_col_name,
                    column=column,
                )

        self.feature_names_in_ = list(X.columns)
        return self

    def _transform_binary_continuous(
        self, X: nw.DataFrame, unseen_per_col: dict
    ) -> IntoFrameT:
        fill_value_unseen = (
            self.fill_value_unseen
            if self.fill_value_unseen != "mean" or self.unseen != "fill"
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
            if self.fill_value_unseen != "mean" or self.unseen != "fill"
            else self.mean_target_
        )
        return X.with_columns(
            nw.col(column)
            .replace_strict(
                {
                    **mapping,
                    **{
                        cat: fill_value_unseen[class_]
                        for cat in unseen_per_col.get(column, [])
                    },
                }
            )
            .alias(f"{column}_mean_target_class_{class_}")
            for column, class_mapping in self.encoding_map_.items()
            for class_, mapping in class_mapping.items()
        ).drop(self.columns_)

    @check_if_fitted
    def get_feature_names_out(self) -> list[str]:
        if self.type_of_target_ in ("binary", "continuous"):
            return self.feature_names_in_

        else:  # multiclass
            return [
                feat for feat in self.feature_names_in_ if feat not in self.columns_
            ] + [
                f"{column}_mean_target_class_{class_}"
                for column in self.columns_
                for class_ in self.unique_classes_
            ]

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
            if self.type_of_target_ in ("binary", "continuous"):
                seen_categories = mapping.keys()
            else:
                seen_categories = next(iter(mapping.values())).keys()

            uniques = X[column].unique()
            unseen_cats = uniques.filter(
                (~uniques.is_in(seen_categories) & ~uniques.is_null())
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
