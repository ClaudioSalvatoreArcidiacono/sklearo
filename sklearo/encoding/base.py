import warnings
from abc import abstractmethod
from collections import defaultdict

import narwhals as nw
from narwhals.typing import IntoFrameT, IntoSeriesT

from sklearo.base import BaseTransformer
from sklearo.utils import infer_target_type, select_columns
from sklearo.validation import check_if_fitted, check_X_y


class BaseOneToOneEncoder(BaseTransformer):

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


class BaseTargetEncoder(BaseOneToOneEncoder):

    @abstractmethod
    def _calculate_target_statistic(
        self, x_y: IntoFrameT, target_col: str, column: str
    ) -> dict[str, float | int | None]:
        """Calculate the target statistic for a column."""
        raise NotImplementedError

    @nw.narwhalify
    @check_X_y
    def fit(self, X: IntoFrameT, y: IntoSeriesT) -> "BaseTargetEncoder":
        """Fit the encoder.

        Args:
            X (DataFrame): The input data.
            y (Series): The target variable.
        """

        self.columns_ = list(select_columns(X, self.columns))
        self.encoding_map_ = {}

        X = self._handle_missing_values(X)

        if not hasattr(self, "target_type") or self.target_type == "auto":
            self.target_type_ = infer_target_type(y)
        else:
            self.target_type_ = self.target_type

        if self.target_type_ not in self._allowed_types_of_target:
            raise ValueError(
                f"Invalid type of target '{self.target_type_}'. "
                f"Allowed types are {self._allowed_types_of_target}."
            )

        if self.target_type_ == "binary":
            unique_classes = sorted(y.unique().to_list())
            if unique_classes != [0, 1]:
                y = y.replace_strict({unique_classes[0]: 0, unique_classes[1]: 1})

        if "target" in X.columns:
            target_col_name = "__target__"

        else:
            target_col_name = "target"

        if not self.columns_:
            return self

        X_y = X[self.columns_].with_columns(**{target_col_name: y})

        if self.target_type_ == "multiclass":
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
                    self.encoding_map_[column][class_] = (
                        self._calculate_target_statistic(
                            X_y_binarized[[column, target_col_name]],
                            target_col=target_col_name,
                            column=column,
                        )
                    )
                if self.unseen == "fill" and self.fill_value_unseen == "mean":
                    self.mean_target_[class_] = X_y_binarized[target_col_name].mean()

        else:
            for column in self.columns_:
                self.encoding_map_[column] = self._calculate_target_statistic(
                    X_y[[column, target_col_name]],
                    target_col=target_col_name,
                    column=column,
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
            if self.target_type_ in ("binary", "continuous"):
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

        if self.target_type_ in ("binary", "continuous"):
            return self._transform_binary_continuous(X, unseen_per_col)

        else:  # multiclass
            return self._transform_multiclass(X, unseen_per_col)

    @check_if_fitted
    def get_feature_names_out(self) -> list[str]:
        if self.target_type_ in ("binary", "continuous"):
            return self.feature_names_in_

        else:  # multiclass
            return [
                feat for feat in self.feature_names_in_ if feat not in self.columns_
            ] + [
                f"{column}_{self._encoder_name}_class_{class_}"
                for column in self.columns_
                for class_ in self.unique_classes_
            ]

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
            .alias(f"{column}_{self._encoder_name}_class_{class_}")
            for column, class_mapping in self.encoding_map_.items()
            for class_, mapping in class_mapping.items()
        ).drop(self.columns_)
