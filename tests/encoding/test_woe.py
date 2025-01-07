import numpy as np
import pandas as pd
import polars as pl
import pytest

from sklearo.encoding.woe import WOEEncoder


@pytest.mark.parametrize(
    "DataFrame", [pd.DataFrame, pl.DataFrame], ids=["pandas", "polars"]
)
class TestWOEEncoder:

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
        encoder = WOEEncoder()
        encoder.fit(binary_class_data[["category"]], binary_class_data["target"])

        assert encoder.columns_ == ["category"]
        assert "category" in encoder.encoding_map_

    def test_woe_encoder_fit_multiclass_non_int_target(
        self, binary_class_data, DataFrame
    ):
        binary_class_data = DataFrame(binary_class_data)
        encoder = WOEEncoder(columns=["target"])
        encoder.fit(binary_class_data[["target"]], binary_class_data["category"])

        assert encoder.columns_ == ["target"]
        assert "target" in encoder.encoding_map_

        transformed_data = encoder.transform(binary_class_data[["target"]])
        np.testing.assert_allclose(
            transformed_data["target_WOE_class_A"].to_list(),
            [
                -0.693147,
                0.693147,
                0.693147,
                -0.693147,
                -0.693147,
                0.693147,
                -0.693147,
                -0.693147,
                0.693147,
            ],
            rtol=1e-5,
        )

    def test_woe_encoder_fit_binary_non_int_target(self, multi_class_data, DataFrame):
        multi_class_data = DataFrame(multi_class_data)
        encoder = WOEEncoder(columns=["target"])
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
                -0.693147,
                -0.693147,
                0.693147,
                0.0,
                0.0,
                -0.693147,
                0.693147,
                0.693147,
                0.0,
                0.0,
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
        encoder = WOEEncoder()
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
                0.916291,
                0.916291,
                0.916291,
                -0.470004,
                -0.470004,
                -0.470004,
                -0.470004,
                -0.470004,
                -0.470004,
            ],
            rtol=1e-5,
        )

    def test_woe_encoder_fit_with_target_in_X_binary(
        self, binary_class_data, DataFrame
    ):
        binary_class_data = DataFrame(binary_class_data)
        encoder = WOEEncoder(
            columns=["category", "target"], underrepresented_categories="fill"
        )

        encoder.fit(binary_class_data, binary_class_data["target"])

        assert encoder.columns_ == ["category", "target"]
        assert "category" in encoder.encoding_map_

    def test_woe_encoder_fit_with_target_in_X_multi_class(
        self, multi_class_data, DataFrame
    ):
        multi_class_data = DataFrame(multi_class_data)
        encoder = WOEEncoder(
            columns=["category", "target"], underrepresented_categories="fill"
        )

        encoder.fit(multi_class_data, multi_class_data["target"])

        assert encoder.columns_ == ["category", "target"]
        assert "category" in encoder.encoding_map_

    def test_woe_encoder_fit_with_target_in_X_multi_class_raise_underrepresented(
        self, multi_class_data, DataFrame
    ):
        multi_class_data = DataFrame(multi_class_data)
        encoder = WOEEncoder(
            columns=["category", "target"], underrepresented_categories="raise"
        )

        with pytest.raises(ValueError, match="Underrepresented category"):
            encoder.fit(multi_class_data, multi_class_data["target"])

    def test_woe_encoder_fit_with_empty_columns(self, multi_class_data, DataFrame):
        multi_class_data = DataFrame(multi_class_data)
        encoder = WOEEncoder(columns=[])
        encoder.fit(multi_class_data[["category"]], multi_class_data["target"])

        assert encoder.columns_ == []
        assert encoder.encoding_map_ == {}

    def test_woe_encoder_fit_multi_class(self, multi_class_data, DataFrame):
        multi_class_data = DataFrame(multi_class_data)
        encoder = WOEEncoder()
        encoder.fit(multi_class_data[["category"]], multi_class_data["target"])

        assert encoder.columns_ == ["category"]
        assert "category" in encoder.encoding_map_

    def test_woe_encoder_transform_binary(self, binary_class_data, DataFrame):
        binary_class_data = DataFrame(binary_class_data)
        encoder = WOEEncoder()
        encoder.fit(binary_class_data[["category"]], binary_class_data["target"])
        transformed = encoder.transform(binary_class_data[["category"]])

        # for category A:
        #   log((1/5)/(2/4)) = -0.916291...
        # for categories B and C:
        #   log((2/5)/(1/4)) = 0.470004...
        expected_values = [
            -0.916291,
            -0.916291,
            -0.916291,
            0.470004,
            0.470004,
            0.470004,
            0.470004,
            0.470004,
            0.470004,
        ]
        np.testing.assert_allclose(
            transformed["category"].to_list(), expected_values, rtol=1e-5
        )
        assert isinstance(transformed, DataFrame)

    def test_woe_encoder_transform_multi_class(self, multi_class_data, DataFrame):
        multi_class_data = DataFrame(multi_class_data)
        encoder = WOEEncoder()
        encoder.fit(multi_class_data[["category"]], multi_class_data["target"])
        transformed = encoder.transform(multi_class_data[["category"]])

        assert list(transformed.columns) == [
            "category_WOE_class_1",
            "category_WOE_class_2",
            "category_WOE_class_3",
        ]

        np.testing.assert_allclose(
            transformed["category_WOE_class_1"],
            # For class 1 A counts : 2, B counts : 1
            [
                0.441833,
                0.441833,
                0.441833,
                0.441833,
                0.441833,
                -0.538997,
                -0.538997,
                -0.538997,
                -0.538997,
                -0.538997,
            ],
            rtol=1e-5,
        )

        np.testing.assert_allclose(
            transformed["category_WOE_class_2"],
            # For class 2 A counts : 1, B counts : 2
            [
                -0.538997,
                -0.538997,
                -0.538997,
                -0.538997,
                -0.538997,
                0.441833,
                0.441833,
                0.441833,
                0.441833,
                0.441833,
            ],
            rtol=1e-5,
        )

        np.testing.assert_allclose(
            transformed["category_WOE_class_3"],
            # For class 3 A counts : 2, B counts : 2
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            rtol=1e-5,
        )

    def test_woe_encoder_handle_missing_values_binary(
        self, binary_class_data, DataFrame
    ):
        binary_class_data["category"][0] = None
        binary_class_data = DataFrame(binary_class_data)

        encoder = WOEEncoder(
            missing_values="encode", underrepresented_categories="fill"
        )
        encoder.fit(binary_class_data[["category"]], binary_class_data["target"])

        assert "MISSING" in encoder.encoding_map_["category"]

    def test_woe_encoder_handle_missing_values_multi_class(
        self, multi_class_data, DataFrame
    ):
        multi_class_data["category"][0] = None
        multi_class_data = DataFrame(multi_class_data)

        encoder = WOEEncoder(
            missing_values="encode", underrepresented_categories="fill"
        )
        encoder.fit(multi_class_data[["category"]], multi_class_data["target"])
        transformed = encoder.transform(multi_class_data[["category"]])

        assert "MISSING" in encoder.encoding_map_["category"][1]

    def test_woe_encoder_unseen_category_binary(self, binary_class_data, DataFrame):
        binary_class_data = DataFrame(binary_class_data)
        encoder = WOEEncoder(unseen="ignore", fill_value_unseen=-999)
        encoder.fit(binary_class_data[["category"]], binary_class_data["target"])

        new_data = DataFrame({"category": ["A", "B", "D"]})
        with pytest.warns(UserWarning):
            transformed = encoder.transform(new_data)

        np.testing.assert_allclose(
            transformed["category"].to_list(),
            [-0.9162907, 0.4700036, -999.0],
            rtol=1e-5,
        )

    def test_woe_encoder_unseen_category_binary_raise(
        self, binary_class_data, DataFrame
    ):
        binary_class_data = DataFrame(binary_class_data)
        encoder = WOEEncoder(unseen="raise")
        encoder.fit(binary_class_data[["category"]], binary_class_data["target"])

        new_data = DataFrame({"category": ["A", "B", "D"]})
        with pytest.raises(ValueError, match="Unseen categories"):
            encoder.transform(new_data)

    def test_woe_encoder_underrepresented_category_binary(
        self, binary_class_data, DataFrame
    ):
        binary_class_data["category"] = [
            "A",
            "A",
            "A",
            "B",
            "B",
            "B",
            "C",
            "C",
            "D",
        ]
        binary_class_data = DataFrame(binary_class_data)

        encoder = WOEEncoder(
            underrepresented_categories="fill", fill_values_underrepresented=(-999, 999)
        )
        encoder.fit(
            binary_class_data[["category"]],
            binary_class_data["target"],
        )
        transformed = encoder.transform(binary_class_data[["category"]])

        assert transformed["category"].to_list()[-1] == -999

    def test_get_feature_names_out_binary(self, binary_class_data, DataFrame):
        binary_class_data["num_1"] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        binary_class_data["num_2"] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        binary_class_data = DataFrame(binary_class_data)

        encoder = WOEEncoder()
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

        encoder = WOEEncoder()
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
                "category_WOE_class_1",
                "category_WOE_class_2",
                "category_WOE_class_3",
            ]
            == encoder.get_feature_names_out()
        )

    def test_transform_called_before_fitting(self, binary_class_data, DataFrame):
        binary_class_data = DataFrame(binary_class_data)
        encoder = WOEEncoder()
        with pytest.raises(ValueError):
            encoder.transform(binary_class_data[["category"]])

    def test_get_feature_names_out_before_fitting(self, binary_class_data, DataFrame):
        binary_class_data = DataFrame(binary_class_data)
        encoder = WOEEncoder()
        with pytest.raises(ValueError):
            encoder.get_feature_names_out()

    def test_X_y_wrong_size(self, binary_class_data, DataFrame):
        binary_class_data = DataFrame(binary_class_data)
        encoder = WOEEncoder()
        with pytest.raises(ValueError):
            encoder.fit(
                binary_class_data[["category"]].head(), binary_class_data["target"]
            )

    def test_woe_encoder_handle_missing_values_raise_in_fit(
        self, binary_class_data, DataFrame
    ):
        binary_class_data["category"][0] = None
        binary_class_data = DataFrame(binary_class_data)

        encoder = WOEEncoder(missing_values="raise")
        with pytest.raises(ValueError, match="Some columns have missing values."):
            encoder.fit(binary_class_data[["category"]], binary_class_data["target"])

    def test_woe_encoder_handle_missing_values_raise_in_transform(
        self, binary_class_data, DataFrame
    ):
        binary_class_data_df = DataFrame(binary_class_data)

        encoder = WOEEncoder(missing_values="raise")
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

        encoder = WOEEncoder(missing_values="ignore")
        encoder.fit(binary_class_data[["category"]], binary_class_data["target"])
        # Ensure that fitting does not raise an error
        assert encoder is not None

    def test_woe_encoder_handle_missing_values_ignore_in_transform(
        self, binary_class_data, DataFrame
    ):
        binary_class_data_df = DataFrame(binary_class_data)

        encoder = WOEEncoder(missing_values="ignore")
        encoder.fit(binary_class_data_df[["category"]], binary_class_data_df["target"])

        binary_class_data["category"][1] = None
        binary_class_data_df = DataFrame(binary_class_data)

        # Transform should not raise an error
        transformed = encoder.transform(binary_class_data_df[["category"]])
        assert transformed is not None

    def test_missing_values_in_target_variable(self, binary_class_data, DataFrame):
        binary_class_data["target"][1] = None
        binary_class_data = DataFrame(binary_class_data)

        encoder = WOEEncoder()
        with pytest.raises(ValueError, match="y contains missing values."):
            encoder.fit(binary_class_data[["category"]], binary_class_data["target"])

    def test_woe_encoder_fit_transform(self, binary_class_data, DataFrame):

        binary_class_data = DataFrame(
            {
                "category": binary_class_data["category"] * 2,
                "target": binary_class_data["target"] * 2,
            }
        )
        encoder = WOEEncoder(
            cv=3,
        )
        transformed = encoder.fit_transform(
            binary_class_data[["category"]], binary_class_data["target"]
        )

        assert transformed["category"].to_list() == [
            -0.8754687373539001,
            -0.8754687373539001,
            -0.8754687373539001,
            0.5108256237659906,
            0.5108256237659906,
            0.5108256237659906,
            0.22314355131420976,
            0.7621400520468967,
            0.7621400520468967,
            -1.0296194171811583,
            -1.0296194171811583,
            -1.0296194171811583,
            0.06899287148695142,
            0.9444616088408515,
            0.9444616088408515,
            0.5389965007326869,
            0.5389965007326869,
            0.5389965007326869,
        ]

    def test_woe_encoder_with_regression_target_type_raises_error(self, DataFrame):
        regression_data = {
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
        regression_data_df = DataFrame(regression_data)

        encoder = WOEEncoder()
        with pytest.raises(ValueError, match="Invalid type of target 'continuous'."):
            encoder.fit(regression_data_df[["category"]], regression_data_df["target"])
