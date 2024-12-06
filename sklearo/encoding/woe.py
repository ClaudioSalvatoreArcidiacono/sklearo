import narwhals as nw
from narwhals.typing import IntoFrameT, IntoSeriesT
import warnings
import math
from typing import Sequence, Literal, Optional

from pydantic import validate_arguments
from sklearo.utils import select_columns


class WOEEncoder:
    """Weight of Evidence (WOE) Encoder.

    This class provides functionality to encode categorical features using the Weight of Evidence
    (WOE) technique. WOE is commonly used in credit scoring and other binary classification problems
    to transform categorical variables into continuous variables.

    WOE is defined as the natural logarithm of the ratio of the distribution of goods (i.e. the
    negative class, 0) to the distribution of bads (i.e. the positive class, 1) for a
    given category.

    ```
    WOE = ln((% of goods) / (% of bads))
    ```

    The WOE value is positive if the category is more likely to be good (negative class) and
    negative if it is more likely to be bad (positive class). This means that the WOE should be
    inversely correlated to the target variable.

    The WOE encoding is useful for
    logistic regression and other linear models, as it transforms the categorical variables into
    continuous variables that can be used as input features.

    Args:
        columns (str, list[str], list[nw.typing.DTypes]): list of columns to encode.
            If a single string is passed instead, it is treated as a regular expression pattern to
            match column names. If a list of `narwhals.typing.DTypes` is passed, it will select
            all columns matching the specified dtype. Defaults to [narwhals.Categorical,
            narwhals.String].
        underrepresented_categories (str): Strategy to handle underrepresented categories.
            If 'raise', an error is raised when a category is missing one of the target classes. If
            'fill', the missing categories are encoded using the fill_values_underrepresented
            values.
        fill_values_underrepresented (list[int, float]): Fill values to use for underrepresented
            categories. The first value is used when there are no goods and the second value when
            there are no bads. Only used when underrepresented_categories is set to 'fill'.
            Optional, Defaults to (-999.0, 999.0).
        unseen (str): Strategy to handle unseen categories. If 'raise', an error is raised when
            unseen categories are found. If 'ignore', the unseen categories are encoded with the
            fill_value_unseen.
        fill_value_unseen (int, float): Fill value to use for unseen categories. Only used when
            unseen is set to 'ignore'. Optional, Defaults to 0.0.
        missing_values (str): Strategy to handle missing values. If 'encode', missing values are
            initially encoded as 'MISSING' and the WOE is computed as if it were a regular category.
            If 'ignore', missing values are left as is. If 'raise', an error is raised when missing
            values are found.
        suffix (str): Suffix to append to the column names of the encoded columns. If an empty
            string is passed, the original column names are replaced. Optional, Defaults to "".

    Attributes:
        columns_ (list): List of columns to be encoded, learned during fit.
        encoding_map_ (dict): Dictionary mapping columns to their WOE values, learned during fit.

    Examples:
        ```python
        import pandas as pd
        from sklearo.encoding import WOEEncoder

        data = {
            "category": ["A", "B", "A", "C", "B", "C", "A", "B", "C"],
            "target": [0, 0, 1, 0, 1, 0, 1, 0, 1],
        }
        df = pd.DataFrame(data)

        encoder = WOEEncoder()
        encoder.fit(df[["category"]], df["target"])
        encoded = encoder.transform(df[["category"]])
        print(encoded)
           category
        0 -0.693147
        1  0.693147
        2 -0.693147
        3  0.693147
        4  0.693147
        5  0.693147
        6 -0.693147
        7  0.693147
        8  0.693147
        ```
    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        columns: Sequence[nw.typing.DTypes | str] | str = (
            nw.Categorical,
            nw.String,
        ),
        underrepresented_categories: Literal["raise", "fill"] = "raise",
        fill_values_underrepresented: Sequence[int | float | None] | None = (
            -999.0,
            999.0,
        ),
        unseen: Literal["raise", "ignore"] = "raise",
        fill_value_unseen: int | float | None = 0.0,
        missing_values: Literal["encode", "ignore", "raise"] = "encode",
        suffix: str = "",
    ) -> None:
        self.columns = columns
        self.underrepresented_categories = underrepresented_categories
        self.missing_values = missing_values
        self.fill_values_underrepresented = fill_values_underrepresented or (None, None)
        self.unseen = unseen
        self.fill_value_unseen = fill_value_unseen
        self.suffix = suffix

    def _handle_missing_values(self, x: IntoSeriesT) -> IntoSeriesT:
        if self.missing_values == "ignore":
            return x
        if self.missing_values == "raise":
            if x.null_count() > 0:
                raise ValueError(
                    f"Column {x.name} has missing values. "
                    "Please handle missing values before encoding or set "
                    "missing_values to either 'ignore' or 'encode'."
                )
        if self.missing_values == "encode":
            return x.fill_null("MISSING")

    def _calculate_woe(
        self, x: IntoSeriesT, y: IntoSeriesT, total_goods: int, total_bads: int
    ) -> dict[str, dict[str, float | int | None]]:
        """Calculate the Weight of Evidence for a column."""

        categories_n_goods_n_bads_dist_ratio = (
            x.to_frame()
            .with_columns(y)
            .group_by(x.name)
            .agg(
                n_total=nw.col(y.name).count(),
                n_bads=nw.col(y.name).sum(),
            )
            .with_columns(n_goods=nw.col("n_total") - nw.col("n_bads"))
            .with_columns(
                perc_goods=nw.col("n_goods") / total_goods,
                perc_bads=nw.col("n_bads") / total_bads,
            )
            .with_columns(
                dist_ratio=nw.col("perc_bads") / nw.col("perc_goods")
            )
            .select(x.name, "n_goods", "n_bads", "dist_ratio")
            .rows()
        )
        categories, n_goods, n_bads, dist_ratios = zip(*categories_n_goods_n_bads_dist_ratio)

        total_goods = sum(n_goods)
        total_bads = sum(n_bads)

        if any(n_good == 0 for n_good in n_goods) or any(
            n_bad == 0 for n_bad in n_bads
        ):
            problematic_categories = [
                cat
                for cat, n_good, n_bad in zip(categories, n_goods, n_bads)
                if n_good == 0 or n_bad == 0
            ]
            msg = (
                f"The categories {problematic_categories} for the column {x.name} "
                "are missing one of the target classes. For WOE to be defined, all categories "
                "should have at least one observation of each target class. Please consider "
                "removing infrequent categories using a RareLabelEncoder"
            )
            if self.underrepresented_categories == "raise":
                raise ValueError(
                    msg + " or by setting underrepresented_categories to 'fill'."
                )

            else:  # fill
                warnings.warn(
                    msg + ". The infrequent categories will be encoded as "
                    f"{self.fill_values_underrepresented[0]} "
                    f"when there are no goods and with {self.fill_values_underrepresented[1]} when "
                    "there are no bads."
                )

        woes = []
        for dist_ratio, n_good, n_bad in zip(dist_ratios, n_goods, n_bads):
            if n_good == 0:
                # means there are only bads
                woes.append(self.fill_values_underrepresented[0])
            elif n_bad == 0:
                # means there are only goods
                woes.append(self.fill_values_underrepresented[1])
            else:
                woes.append(math.log(dist_ratio))

        return dict(zip(categories, woes))

    @nw.narwhalify
    def fit(self, X: IntoFrameT, y: IntoSeriesT) -> "WOEEncoder":
        """Fit the encoder."""

        self.columns_ = select_columns(X, self.columns)
        self.encoding_map_ = {}

        total_bads = y.sum()
        total_goods = y.count() - total_bads
        for column in self.columns_:
            self.encoding_map_[column] = self._calculate_woe(
                self._handle_missing_values(X[column]), y, total_goods, total_bads
            )
        return self

    @nw.narwhalify
    def transform(self, X: IntoFrameT) -> IntoFrameT:
        """Transform the data."""

        unseen_per_col = {}
        for column, mapping in self.encoding_map_.items():
            uniques = X[column].unique()
            unseen_cats = uniques.filter(~uniques.is_in(mapping.keys())).to_list()
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

        return X.with_columns(
            nw.col(column)
            .pipe(self._handle_missing_values)
            .replace_strict(
                {
                    **mapping,
                    **{cat: self.fill_value_unseen for cat in unseen_cats},
                }
            )
            .alias(f"{column}{self.suffix}")
            for column, mapping in self.encoding_map_.items()
        )
