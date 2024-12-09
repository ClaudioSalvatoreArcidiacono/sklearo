import warnings

import narwhals as nw
from narwhals.typing import IntoFrameT

from sklearo.base import BaseTransformer
from sklearo.validation import check_if_fitted


class BaseOneToOneEncoder(BaseTransformer):
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
