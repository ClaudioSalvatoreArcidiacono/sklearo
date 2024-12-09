import warnings

import narwhals as nw
from narwhals.typing import IntoFrameT

from sklearo.base import BaseTransformer


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
