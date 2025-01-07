import re

import numpy as np
import pandas as pd
import polars as pl
import pytest

from sklearo.encoding import TargetEncoder


def test_target_encoder_fit_transform_comparison_with_scikit_learn():
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import TargetEncoder as TargetEncoderSKlearn

    X = np.array([["dog"] * 20 + ["cat"] * 30 + ["snake"] * 38], dtype=object).T
    y = [90.3] * 5 + [80.1] * 15 + [20.4] * 5 + [20.1] * 25 + [21.2] * 8 + [49] * 30

    X = pd.DataFrame({"col": X[:, 0]})
    y = pd.Series(y)

    kf = KFold(n_splits=5)

    X_trans_check = X.copy()
    for train_index, test_index in kf.split(X):
        X_train = X.loc[train_index]
        X_test = X.loc[test_index]
        y_train = y[train_index]
        enc_sklearn = TargetEncoderSKlearn()
        enc_sklearn.set_output(transform="pandas")
        X_trans_check.loc[test_index, "col"] = enc_sklearn.fit(
            X_train, y_train
        ).transform(X_test)["col"]

    X_trans_check["col"] = X_trans_check["col"].astype(float)
    enc = TargetEncoder()
    X_trans_sklearo = enc.fit_transform(X, y)
    np.testing.assert_allclose(
        X_trans_sklearo["col"].to_list(), X_trans_check["col"].to_list(), rtol=1e-3
    )


@pytest.mark.parametrize(
    "DataFrame", [pd.DataFrame, pl.DataFrame], ids=["pandas", "polars"]
)
class TestTargetEncoder:
    @pytest.fixture
    def binary_class_data(self):
        data = {
            "category": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
            "target": [1, 0, 0, 1, 1, 0, 1, 1, 0],
        }
        return data

    @pytest.fixture
    def multi_class_data(self):
        data = {
            "category": ["A"] * 5 + ["B"] * 5,
            "target": [1, 1, 2, 3, 3, 1, 2, 2, 3, 3],
        }
        return data

    @pytest.fixture
    def regression_data(self):
        data = {
            "category": ["A"] * 4 + ["B"] * 6,
            "target": [
                100.0,
                200.0,
                300.0,
                400.0,
                500.0,
                600.0,
                700.0,
                800.0,
                900.0,
                10000.0,
            ],
        }
        return data

    def test_target_encoder_fit_binary(self, binary_class_data, DataFrame):
        binary_class_data = DataFrame(binary_class_data)
        encoder = TargetEncoder()
        encoder.fit(binary_class_data[["category"]], binary_class_data["target"])

        assert encoder.columns_ == ["category"]
        assert "category" in encoder.encoding_map_

    def test_target_encoder_fit_regression(self, regression_data, DataFrame):
        regression_data = DataFrame(regression_data)
        encoder = TargetEncoder()
        encoder.fit(regression_data[["category"]], regression_data["target"])

        assert encoder.columns_ == ["category"]
        assert "category" in encoder.encoding_map_

    def test_target_encoder_unseen_value_fill_unseen_multiclass(
        self, multi_class_data, DataFrame
    ):
        multi_class_data = DataFrame(multi_class_data)
        encoder = TargetEncoder(unseen="fill", fill_value_unseen="mean")
        encoder.fit(multi_class_data[["category"]], multi_class_data["target"])

        new_data = DataFrame({"category": ["A", "B", "D"]})
        with pytest.warns(UserWarning, match="Unseen categories"):
            transformed = encoder.transform(new_data)

        np.testing.assert_allclose(
            transformed["category_mean_target_class_1"].to_list(),
            [0.379545, 0.214634, 0.3],
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            transformed["category_mean_target_class_2"].to_list(),
            [0.214634, 0.379545, 0.3],
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            transformed["category_mean_target_class_3"].to_list(),
            [0.4, 0.4, 0.4],
            rtol=1e-5,
        )

    def test_target_encoder_fit_multiclass_non_int_target(
        self, binary_class_data, DataFrame
    ):
        binary_class_data = DataFrame(binary_class_data)
        encoder = TargetEncoder(columns=["target"])
        encoder.fit(binary_class_data[["target"]], binary_class_data["category"])

        assert encoder.columns_ == ["target"]
        assert "target" in encoder.encoding_map_

        transformed_data = encoder.transform(binary_class_data[["target"]])
        np.testing.assert_allclose(
            transformed_data["target_mean_target_class_A"].to_list(),
            [
                0.218391,
                0.458333,
                0.458333,
                0.218391,
                0.218391,
                0.458333,
                0.218391,
                0.218391,
                0.458333,
            ],
            rtol=1e-5,
        )

    def test_target_encoder_fit_binary_non_int_target(
        self, multi_class_data, DataFrame
    ):
        multi_class_data = DataFrame(multi_class_data)
        encoder = TargetEncoder(columns=["target"])
        encoder.fit(multi_class_data[["target"]], multi_class_data["category"])

        assert encoder.columns_ == ["target"]
        assert "target" in encoder.encoding_map_

        transformed_data = encoder.transform(multi_class_data[["target"]])

        assert (
            encoder.get_feature_names_out()
            == ["target"]
            == list(transformed_data.columns)
        )
        np.testing.assert_allclose(
            transformed_data["target"].to_list(),
            [
                0.380952,
                0.380952,
                0.619048,
                0.5,
                0.5,
                0.380952,
                0.619048,
                0.619048,
                0.5,
                0.5,
            ],
            rtol=1e-5,
        )

    def test_target_encoder_fit_binary_non_int_target_classes_1_and_2(
        self, binary_class_data, DataFrame
    ):
        binary_class_data["target"] = [
            1 if x == 1 else 2 for x in binary_class_data["target"]
        ]
        binary_class_data = DataFrame(binary_class_data)
        encoder = TargetEncoder()
        encoder.fit(binary_class_data[["category"]], binary_class_data["target"])

        assert encoder.columns_ == ["category"]
        assert "category" in encoder.encoding_map_

        transformed_data = encoder.transform(binary_class_data[["category"]])

        assert (
            encoder.get_feature_names_out()
            == ["category"]
            == list(transformed_data.columns)
        )

        np.testing.assert_allclose(
            transformed_data["category"].to_list(),
            [
                0.603175,
                0.603175,
                0.603175,
                0.365079,
                0.365079,
                0.365079,
                0.365079,
                0.365079,
                0.365079,
            ],
            rtol=1e-5,
        )

    def test_target_encoder_fit_with_target_in_X_binary(
        self, binary_class_data, DataFrame
    ):
        binary_class_data = DataFrame(binary_class_data)
        encoder = TargetEncoder(columns=["category", "target"])

        encoder.fit(binary_class_data, binary_class_data["target"])

        assert encoder.columns_ == ["category", "target"]
        assert "category" in encoder.encoding_map_

    def test_target_encoder_fit_with_target_in_X_multi_class(
        self, multi_class_data, DataFrame
    ):
        multi_class_data = DataFrame(multi_class_data)
        encoder = TargetEncoder(columns=["category", "target"])

        encoder.fit(multi_class_data, multi_class_data["target"])

        assert encoder.columns_ == ["category", "target"]
        assert "category" in encoder.encoding_map_

    def test_target_encoder_fit_with_empty_columns(self, multi_class_data, DataFrame):
        multi_class_data = DataFrame(multi_class_data)
        encoder = TargetEncoder(columns=[])
        encoder.fit(multi_class_data[["category"]], multi_class_data["target"])

        assert encoder.columns_ == []
        assert encoder.encoding_map_ == {}

    def test_target_encoder_fit_multi_class(self, multi_class_data, DataFrame):
        multi_class_data = DataFrame(multi_class_data)
        encoder = TargetEncoder()
        encoder.fit(multi_class_data[["category"]], multi_class_data["target"])

        assert encoder.columns_ == ["category"]
        assert "category" in encoder.encoding_map_

    def test_target_encoder_transform_binary(self, binary_class_data, DataFrame):
        binary_class_data = DataFrame(binary_class_data)
        encoder = TargetEncoder()
        encoder.fit(binary_class_data[["category"]], binary_class_data["target"])
        transformed = encoder.transform(binary_class_data[["category"]])

        expected_values = [
            0.396825,
            0.396825,
            0.396825,
            0.634921,
            0.634921,
            0.634921,
            0.634921,
            0.634921,
            0.634921,
        ]
        np.testing.assert_allclose(
            transformed["category"].to_list(), expected_values, rtol=1e-5
        )
        assert isinstance(transformed, DataFrame)

    def test_target_encoder_transform_binary_reserved_column(
        self, binary_class_data, DataFrame
    ):
        binary_class_data = DataFrame(
            {
                "smoothing": binary_class_data["category"],
                "target": binary_class_data["target"],
            }
        )
        encoder = TargetEncoder()
        encoder.fit(binary_class_data[["smoothing"]], binary_class_data["target"])
        transformed = encoder.transform(binary_class_data[["smoothing"]])

        expected_values = [
            0.396825,
            0.396825,
            0.396825,
            0.634921,
            0.634921,
            0.634921,
            0.634921,
            0.634921,
            0.634921,
        ]
        np.testing.assert_allclose(
            transformed["smoothing"].to_list(), expected_values, rtol=1e-5
        )
        assert isinstance(transformed, DataFrame)
        assert transformed.columns == ["smoothing"]

    def test_target_encoder_transform_binary_bool(self, binary_class_data, DataFrame):
        binary_class_data = DataFrame(
            {
                "category": binary_class_data["category"],
                "target": [bool(target) for target in binary_class_data["target"]],
            }
        )
        encoder = TargetEncoder()
        encoder.fit(binary_class_data[["category"]], binary_class_data["target"])
        transformed = encoder.transform(binary_class_data[["category"]])

        expected_values = [
            0.396825,
            0.396825,
            0.396825,
            0.634921,
            0.634921,
            0.634921,
            0.634921,
            0.634921,
            0.634921,
        ]
        np.testing.assert_allclose(
            transformed["category"].to_list(), expected_values, rtol=1e-5
        )
        assert isinstance(transformed, DataFrame)

    def test_target_encoder_transform_regression(self, regression_data, DataFrame):
        regression_data = DataFrame(regression_data)
        encoder = TargetEncoder()
        encoder.fit(regression_data[["category"]], regression_data["target"])
        transformed = encoder.transform(regression_data[["category"]])

        expected_values = [
            250.549702,
            250.549702,
            250.549702,
            250.549702,
            2082.60129,
            2082.60129,
            2082.60129,
            2082.60129,
            2082.60129,
            2082.60129,
        ]
        np.testing.assert_allclose(
            transformed["category"].to_list(), expected_values, rtol=1e-5
        )
        assert isinstance(transformed, DataFrame)

    def test_target_encoder_transform_multi_class(self, multi_class_data, DataFrame):
        multi_class_data = DataFrame(multi_class_data)
        encoder = TargetEncoder()
        encoder.fit(multi_class_data[["category"]], multi_class_data["target"])
        transformed = encoder.transform(multi_class_data[["category"]])

        assert list(transformed.columns) == [
            "category_mean_target_class_1",
            "category_mean_target_class_2",
            "category_mean_target_class_3",
        ]

        np.testing.assert_allclose(
            transformed["category_mean_target_class_1"],
            # For class 1 A counts : 2/5, B counts : 1/5
            [
                0.379545,
                0.379545,
                0.379545,
                0.379545,
                0.379545,
                0.214634,
                0.214634,
                0.214634,
                0.214634,
                0.214634,
            ],
            rtol=1e-5,
        )

        np.testing.assert_allclose(
            transformed["category_mean_target_class_2"],
            # For class 2 A counts : 1/5, B counts : 2/5
            [
                0.214634,
                0.214634,
                0.214634,
                0.214634,
                0.214634,
                0.379545,
                0.379545,
                0.379545,
                0.379545,
                0.379545,
            ],
            rtol=1e-5,
        )

        np.testing.assert_allclose(
            transformed["category_mean_target_class_3"],
            # For class 3 A counts : 2/5, B counts : 2/5
            [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
            rtol=1e-5,
        )

    def test_target_encoder_underrepresented_categories_binary(
        self, binary_class_data, DataFrame
    ):
        binary_class_data["category"][0] = None
        binary_class_data = DataFrame(binary_class_data)

        encoder = TargetEncoder(missing_values="encode")

        with pytest.raises(
            ValueError,
            match=re.escape(
                "Found underrepresented categories for the column category: "
                "['MISSING']. Please consider handling underrepresented categories by using a "
                "RareLabelEncoder. Alternatively, set underrepresented_categories to 'fill'."
            ),
        ):
            encoder.fit(binary_class_data[["category"]], binary_class_data["target"])

    def test_target_encoder_underrepresented_categories_multi_class(
        self, multi_class_data, DataFrame
    ):
        multi_class_data["category"][0] = None
        multi_class_data = DataFrame(multi_class_data)

        encoder = TargetEncoder(missing_values="encode")
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Found underrepresented categories for the column category: "
                "['MISSING']. Please consider handling underrepresented categories by using a "
                "RareLabelEncoder. Alternatively, set underrepresented_categories to 'fill'."
            ),
        ):
            encoder.fit(multi_class_data[["category"]], multi_class_data["target"])

    def test_target_encoder_handle_missing_values_binary(
        self, binary_class_data, DataFrame
    ):
        binary_class_data["category"][0] = None
        binary_class_data = DataFrame(binary_class_data)

        encoder = TargetEncoder(
            missing_values="encode", underrepresented_categories="fill"
        )
        encoder.fit(binary_class_data[["category"]], binary_class_data["target"])

        transformed = encoder.transform(binary_class_data[["category"]])
        np.testing.assert_allclose(
            transformed["category"].to_list(),
            [
                0.555556,
                0.0,
                0.0,
                0.634921,
                0.634921,
                0.634921,
                0.634921,
                0.634921,
                0.634921,
            ],
            rtol=1e-5,
        )

    def test_target_encoder_handle_missing_values_multi_class(
        self, multi_class_data, DataFrame
    ):
        multi_class_data["category"][0] = None
        multi_class_data = DataFrame(multi_class_data)

        encoder = TargetEncoder(
            missing_values="encode", underrepresented_categories="fill"
        )

        encoder.fit(multi_class_data[["category"]], multi_class_data["target"])

        transformed = encoder.transform(multi_class_data[["category"]])
        np.testing.assert_allclose(
            transformed["category_mean_target_class_1"].to_list(),
            [
                0.3,
                0.260563,
                0.260563,
                0.260563,
                0.260563,
                0.214634,
                0.214634,
                0.214634,
                0.214634,
                0.214634,
            ],
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            transformed["category_mean_target_class_2"].to_list(),
            [
                0.3,
                0.260563,
                0.260563,
                0.260563,
                0.260563,
                0.379545,
                0.379545,
                0.379545,
                0.379545,
                0.379545,
            ],
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            transformed["category_mean_target_class_3"].to_list(),
            [0.4, 0.47619, 0.47619, 0.47619, 0.47619, 0.4, 0.4, 0.4, 0.4, 0.4],
            rtol=1e-5,
        )

    def test_target_encoder_unnderrepresented_categories_binary_fill_binary_set_value(
        self, binary_class_data, DataFrame
    ):
        binary_class_data["category"][0] = None
        binary_class_data = DataFrame(binary_class_data)

        encoder = TargetEncoder(
            missing_values="encode",
            underrepresented_categories="fill",
            fill_values_underrepresented=999,
        )
        encoder.fit(binary_class_data[["category"]], binary_class_data["target"])

        transformed = encoder.transform(binary_class_data[["category"]])
        np.testing.assert_allclose(
            transformed["category"].to_list(),
            [
                9.990000e02,
                0.000000e00,
                0.000000e00,
                6.349206e-01,
                6.349206e-01,
                6.349206e-01,
                6.349206e-01,
                6.349206e-01,
                6.349206e-01,
            ],
            rtol=1e-5,
        )

    def test_target_encoder_unseen_category_binary(self, binary_class_data, DataFrame):
        binary_class_data = DataFrame(binary_class_data)
        encoder = TargetEncoder(unseen="ignore", fill_value_unseen=-999)
        encoder.fit(binary_class_data[["category"]], binary_class_data["target"])

        new_data = DataFrame({"category": ["A", "B", "D"]})
        with pytest.warns(UserWarning):
            transformed = encoder.transform(new_data)

        np.testing.assert_allclose(
            transformed["category"].to_list(),
            [3.968254e-01, 6.349206e-01, -9.990000e02],
            rtol=1e-5,
        )

    def test_target_encoder_unseen_category_binary_raise(
        self, binary_class_data, DataFrame
    ):
        binary_class_data = DataFrame(binary_class_data)
        encoder = TargetEncoder(unseen="raise")
        encoder.fit(binary_class_data[["category"]], binary_class_data["target"])

        new_data = DataFrame({"category": ["A", "B", "D"]})
        with pytest.raises(ValueError, match="Unseen categories"):
            encoder.transform(new_data)

    def test_get_feature_names_out_binary(self, binary_class_data, DataFrame):
        binary_class_data["num_1"] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        binary_class_data["num_2"] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        binary_class_data = DataFrame(binary_class_data)

        encoder = TargetEncoder()
        encoder.fit(
            binary_class_data[["num_1", "category", "num_2"]],
            binary_class_data["target"],
        )
        transformed = encoder.transform(
            binary_class_data[["num_1", "category", "num_2"]]
        )
        assert (
            list(transformed.columns)
            == ["num_1", "category", "num_2"]
            == encoder.get_feature_names_out()
        )

    def test_get_feature_names_out_multi_class(self, multi_class_data, DataFrame):
        multi_class_data["num_1"] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        multi_class_data["num_2"] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        multi_class_data = DataFrame(multi_class_data)

        encoder = TargetEncoder()
        encoder.fit(
            multi_class_data[["num_1", "category", "num_2"]],
            multi_class_data["target"],
        )
        transformed = encoder.transform(
            multi_class_data[["num_1", "category", "num_2"]]
        )
        assert (
            list(transformed.columns)
            == [
                "num_1",
                "num_2",
                "category_mean_target_class_1",
                "category_mean_target_class_2",
                "category_mean_target_class_3",
            ]
            == encoder.get_feature_names_out()
        )

    def test_transform_called_before_fitting(self, binary_class_data, DataFrame):
        binary_class_data = DataFrame(binary_class_data)
        encoder = TargetEncoder()
        with pytest.raises(ValueError):
            encoder.transform(binary_class_data[["category"]])

    def test_get_feature_names_out_before_fitting(self, binary_class_data, DataFrame):
        binary_class_data = DataFrame(binary_class_data)
        encoder = TargetEncoder()
        with pytest.raises(ValueError):
            encoder.get_feature_names_out()

    def test_X_y_wrong_size(self, binary_class_data, DataFrame):
        binary_class_data = DataFrame(binary_class_data)
        encoder = TargetEncoder()
        with pytest.raises(ValueError):
            encoder.fit(
                binary_class_data[["category"]].head(), binary_class_data["target"]
            )

    def test_target_encoder_handle_missing_values_raise_in_fit(
        self, binary_class_data, DataFrame
    ):
        binary_class_data["category"][0] = None
        binary_class_data = DataFrame(binary_class_data)

        encoder = TargetEncoder(missing_values="raise")
        with pytest.raises(ValueError, match="Some columns have missing values."):
            encoder.fit(binary_class_data[["category"]], binary_class_data["target"])

    def test_target_encoder_handle_missing_values_raise_in_transform(
        self, binary_class_data, DataFrame
    ):
        binary_class_data_df = DataFrame(binary_class_data)

        encoder = TargetEncoder(missing_values="raise")
        encoder.fit(binary_class_data_df[["category"]], binary_class_data_df["target"])

        binary_class_data["category"][0] = None
        binary_class_data_df = DataFrame(binary_class_data)

        with pytest.raises(ValueError, match="Some columns have missing values."):
            encoder.transform(binary_class_data_df[["category"]])

    def test_target_encoder_handle_missing_values_ignore_in_fit(
        self, binary_class_data, DataFrame
    ):
        binary_class_data["category"][1] = None
        binary_class_data = DataFrame(binary_class_data)

        encoder = TargetEncoder(missing_values="ignore")
        encoder.fit(binary_class_data[["category"]], binary_class_data["target"])
        # Ensure that fitting does not raise an error
        assert encoder is not None

    def test_target_encoder_handle_missing_values_ignore_in_transform(
        self, binary_class_data, DataFrame
    ):
        binary_class_data_df = DataFrame(binary_class_data)

        encoder = TargetEncoder(missing_values="ignore")
        encoder.fit(binary_class_data_df[["category"]], binary_class_data_df["target"])

        binary_class_data["category"][1] = None
        binary_class_data_df = DataFrame(binary_class_data)

        # Transform should not raise an error
        transformed = encoder.transform(binary_class_data_df[["category"]])
        assert transformed is not None

    def test_missing_values_in_target_variable(self, binary_class_data, DataFrame):
        binary_class_data["target"][1] = None
        binary_class_data = DataFrame(binary_class_data)

        encoder = TargetEncoder()
        with pytest.raises(ValueError, match="y contains missing values."):
            encoder.fit(binary_class_data[["category"]], binary_class_data["target"])

    def test_target_encoder_fit_transform(self, binary_class_data, DataFrame):
        binary_class_data = DataFrame(
            {
                "category": binary_class_data["category"] * 2,
                "target": binary_class_data["target"] * 2,
            }
        )
        encoder = TargetEncoder(
            cv=3,
        )
        transformed = encoder.fit_transform(
            binary_class_data[["category"]], binary_class_data["target"]
        )

        assert transformed["category"].to_list() == [
            0.39473684210526316,
            0.39473684210526316,
            0.39473684210526316,
            0.631578947368421,
            0.631578947368421,
            0.631578947368421,
            0.5901639344262295,
            0.7182080924855492,
            0.7182080924855492,
            0.4071588366890381,
            0.4071588366890381,
            0.4071588366890381,
            0.5969245107176142,
            0.7101449275362318,
            0.7101449275362318,
            0.6292134831460674,
            0.6292134831460674,
            0.6292134831460674,
        ]

    def test_target_encoder_fit_transform_set_smoothing(
        self, binary_class_data, DataFrame
    ):
        binary_class_data = DataFrame(
            {
                "category": binary_class_data["category"] * 2,
                "target": binary_class_data["target"] * 2,
            }
        )
        encoder = TargetEncoder(cv=3, smooth=10)
        transformed = encoder.fit_transform(
            binary_class_data[["category"]], binary_class_data["target"]
        )

        assert transformed["category"].to_list() == [
            0.49650349650349646,
            0.49650349650349646,
            0.49650349650349646,
            0.5734265734265733,
            0.5734265734265733,
            0.5734265734265733,
            0.5636363636363637,
            0.6309523809523809,
            0.6309523809523809,
            0.5256410256410257,
            0.5256410256410257,
            0.5256410256410257,
            0.588888888888889,
            0.5989010989010989,
            0.5989010989010989,
            0.5680473372781065,
            0.5680473372781065,
            0.5680473372781065,
        ]

    def test_target_encoder_explicitly_set_target_type(
        self, multi_class_data, DataFrame
    ):
        multi_class_data = DataFrame(multi_class_data)
        encoder = TargetEncoder(target_type="continuous")
        encoder.fit(multi_class_data[["category"]], multi_class_data["target"])

        assert encoder.columns_ == ["category"]
        assert "category" in encoder.encoding_map_

        transformed_data = encoder.transform(multi_class_data[["category"]])

        assert (
            encoder.get_feature_names_out()
            == ["category"]
            == list(transformed_data.columns)
        )

        np.testing.assert_allclose(
            transformed_data["category"].to_list(),
            [
                2.02069,
                2.02069,
                2.02069,
                2.02069,
                2.02069,
                2.184559,
                2.184559,
                2.184559,
                2.184559,
                2.184559,
            ],
            rtol=1e-5,
        )
