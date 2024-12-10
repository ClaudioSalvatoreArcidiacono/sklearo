import numpy as np
import pandas as pd
import polars as pl
import pytest

from sklearo.encoding import TargetEncoder


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

    def test_woe_encoder_fit_binary(self, binary_class_data, DataFrame):
        binary_class_data = DataFrame(binary_class_data)
        encoder = TargetEncoder()
        encoder.fit(binary_class_data[["category"]], binary_class_data["target"])

        assert encoder.columns_ == ["category"]
        assert "category" in encoder.encoding_map_
        assert encoder.is_zero_one_target_ is True

    def test_woe_encoder_fit_multiclass_non_int_target(
        self, binary_class_data, DataFrame
    ):
        binary_class_data = DataFrame(binary_class_data)
        encoder = TargetEncoder(columns=["target"])
        encoder.fit(binary_class_data[["target"]], binary_class_data["category"])

        assert encoder.columns_ == ["target"]
        assert "target" in encoder.encoding_map_
        assert encoder.is_zero_one_target_ is False

        transformed_data = encoder.transform(binary_class_data[["target"]])
        np.testing.assert_allclose(
            transformed_data["target_mean_target_class_A"].to_list(),
            [0.2, 0.5, 0.5, 0.2, 0.2, 0.5, 0.2, 0.2, 0.5],
            rtol=1e-5,
        )

    def test_woe_encoder_fit_binary_non_int_target(self, multi_class_data, DataFrame):
        multi_class_data = DataFrame(multi_class_data)
        encoder = TargetEncoder(columns=["target"])
        encoder.fit(multi_class_data[["target"]], multi_class_data["category"])

        assert encoder.columns_ == ["target"]
        assert "target" in encoder.encoding_map_
        assert encoder.is_zero_one_target_ is False

        transformed_data = encoder.transform(multi_class_data[["target"]])

        assert (
            encoder.get_feature_names_out()
            == ["target"]
            == list(transformed_data.columns)
        )
        np.testing.assert_allclose(
            transformed_data["target"].to_list(),
            [
                0.333333,
                0.333333,
                0.666667,
                0.5,
                0.5,
                0.333333,
                0.666667,
                0.666667,
                0.5,
                0.5,
            ],
            rtol=1e-5,
        )

    def test_woe_encoder_fit_binary_non_int_target_classes_1_and_2(
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
        assert encoder.is_zero_one_target_ is False

        transformed_data = encoder.transform(binary_class_data[["category"]])

        assert (
            encoder.get_feature_names_out()
            == ["category"]
            == list(transformed_data.columns)
        )

        np.testing.assert_allclose(
            transformed_data["category"].to_list(),
            [
                0.666667,
                0.666667,
                0.666667,
                0.333333,
                0.333333,
                0.333333,
                0.333333,
                0.333333,
                0.333333,
            ],
            rtol=1e-5,
        )

    def test_woe_encoder_fit_with_target_in_X_binary(
        self, binary_class_data, DataFrame
    ):
        binary_class_data = DataFrame(binary_class_data)
        encoder = TargetEncoder(columns=["category", "target"])

        encoder.fit(binary_class_data, binary_class_data["target"])

        assert encoder.columns_ == ["category", "target"]
        assert "category" in encoder.encoding_map_
        assert encoder.is_zero_one_target_ is True

    def test_woe_encoder_fit_with_target_in_X_multi_class(
        self, multi_class_data, DataFrame
    ):
        multi_class_data = DataFrame(multi_class_data)
        encoder = TargetEncoder(columns=["category", "target"])

        encoder.fit(multi_class_data, multi_class_data["target"])

        assert encoder.columns_ == ["category", "target"]
        assert "category" in encoder.encoding_map_
        assert encoder.is_zero_one_target_ is False

    def test_woe_encoder_fit_with_empty_columns(self, multi_class_data, DataFrame):
        multi_class_data = DataFrame(multi_class_data)
        encoder = TargetEncoder(columns=[])
        encoder.fit(multi_class_data[["category"]], multi_class_data["target"])

        assert encoder.columns_ == []
        assert encoder.encoding_map_ == {}

    def test_woe_encoder_fit_multi_class(self, multi_class_data, DataFrame):
        multi_class_data = DataFrame(multi_class_data)
        encoder = TargetEncoder()
        encoder.fit(multi_class_data[["category"]], multi_class_data["target"])

        assert encoder.columns_ == ["category"]
        assert "category" in encoder.encoding_map_
        assert encoder.is_zero_one_target_ is False

    def test_woe_encoder_transform_binary(self, binary_class_data, DataFrame):
        binary_class_data = DataFrame(binary_class_data)
        encoder = TargetEncoder()
        encoder.fit(binary_class_data[["category"]], binary_class_data["target"])
        transformed = encoder.transform(binary_class_data[["category"]])

        expected_values = [
            0.333333,
            0.333333,
            0.333333,
            0.666667,
            0.666667,
            0.666667,
            0.666667,
            0.666667,
            0.666667,
        ]
        np.testing.assert_allclose(
            transformed["category"].to_list(), expected_values, rtol=1e-5
        )
        assert isinstance(transformed, DataFrame)

    def test_woe_encoder_transform_multi_class(self, multi_class_data, DataFrame):
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
            [0.4, 0.4, 0.4, 0.4, 0.4, 0.2, 0.2, 0.2, 0.2, 0.2],
            rtol=1e-5,
        )

        np.testing.assert_allclose(
            transformed["category_mean_target_class_2"],
            # For class 2 A counts : 1/5, B counts : 2/5
            [0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.4, 0.4, 0.4, 0.4],
            rtol=1e-5,
        )

        np.testing.assert_allclose(
            transformed["category_mean_target_class_3"],
            # For class 3 A counts : 2/5, B counts : 2/
            [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
            rtol=1e-5,
        )

    def test_woe_encoder_handle_missing_values_binary(
        self, binary_class_data, DataFrame
    ):
        binary_class_data["category"][0] = None
        binary_class_data = DataFrame(binary_class_data)

        encoder = TargetEncoder(missing_values="encode")

        encoder.fit(binary_class_data[["category"]], binary_class_data["target"])
        transformed = encoder.transform(binary_class_data[["category"]])

        assert "MISSING" in encoder.encoding_map_["category"]

    def test_woe_encoder_handle_missing_values_multi_class(
        self, multi_class_data, DataFrame
    ):
        multi_class_data["category"][0] = None
        multi_class_data = DataFrame(multi_class_data)

        encoder = TargetEncoder(missing_values="encode")
        encoder.fit(multi_class_data[["category"]], multi_class_data["target"])
        transformed = encoder.transform(multi_class_data[["category"]])

        assert "MISSING" in encoder.encoding_map_["category"][1]

    def test_woe_encoder_unseen_category_binary(self, binary_class_data, DataFrame):
        binary_class_data = DataFrame(binary_class_data)
        encoder = TargetEncoder(unseen="ignore", fill_value_unseen=-999)
        encoder.fit(binary_class_data[["category"]], binary_class_data["target"])

        new_data = DataFrame({"category": ["A", "B", "D"]})
        with pytest.warns(UserWarning):
            transformed = encoder.transform(new_data)

        np.testing.assert_allclose(
            transformed["category"].to_list(), [0.3333333, 0.6666667, -999], rtol=1e-5
        )

    def test_woe_encoder_unseen_category_binary_raise(
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

    def test_woe_encoder_handle_missing_values_raise_in_fit(
        self, binary_class_data, DataFrame
    ):
        binary_class_data["category"][0] = None
        binary_class_data = DataFrame(binary_class_data)

        encoder = TargetEncoder(missing_values="raise")
        with pytest.raises(ValueError, match="Some columns have missing values."):
            encoder.fit(binary_class_data[["category"]], binary_class_data["target"])

    def test_woe_encoder_handle_missing_values_raise_in_transform(
        self, binary_class_data, DataFrame
    ):
        binary_class_data_df = DataFrame(binary_class_data)

        encoder = TargetEncoder(missing_values="raise")
        encoder.fit(binary_class_data_df[["category"]], binary_class_data_df["target"])

        binary_class_data["category"][0] = None
        binary_class_data_df = DataFrame(binary_class_data)

        with pytest.raises(ValueError, match="Some columns have missing values."):
            encoder.transform(binary_class_data_df[["category"]])

    def test_woe_encoder_handle_missing_values_ignore_in_fit(
        self, binary_class_data, DataFrame
    ):
        binary_class_data["category"][1] = None
        binary_class_data = DataFrame(binary_class_data)

        encoder = TargetEncoder(missing_values="ignore")
        encoder.fit(binary_class_data[["category"]], binary_class_data["target"])
        # Ensure that fitting does not raise an error
        assert encoder is not None

    def test_woe_encoder_handle_missing_values_ignore_in_transform(
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

    def test_woe_encoder_fit_transform(self, binary_class_data, DataFrame):
        binary_class_data_df = DataFrame(binary_class_data)

        encoder = TargetEncoder()
        transformed = encoder.fit_transform(
            binary_class_data_df[["category"]], binary_class_data_df["target"]
        )

        # Ensure that the transformed data is not None and has the expected shape
        assert transformed is not None
        assert transformed.shape[0] == binary_class_data_df.shape[0]
