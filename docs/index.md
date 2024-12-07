# Homepage

## Sklearo

A versatile Python package featuring scikit-learn like transformers for feature preprocessing, compatible with all kind of DataFrames thanks to narwhals.

## Installation

To install the package, use pip:

```bash
pip install sklearo
```

## Usage

Here's a basic example of how to use the package with the `WOEEncoder`:

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

## Features

- ∫ **Easy Integration**: built on top of [narwhals](https://narwhals-dev.github.io/narwhals/), meaning it can work with any kind of dataframe supported by [narwhals](https://narwhals-dev.github.io/narwhals/extending/) like pandas, polars and much more!
- 🌸 **Scikit-learn Compatibility**: Designed to work with scikit-learn pipelines.
- ✅ tested against pandas and Polars dataframes.

## Contributing

We welcome contributions! Please check the [development guides](development_guide.md) for more details.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or suggestions, please open an issue on GitHub.

## Why `sklearo`?

The name `sklearo` is a combination of `sklearn` and omni (`o`), which means all. This package is designed to work with all kinds of dataframes, hence the name `sklearo`.
