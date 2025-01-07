"""This module provides base classes for transformers in the encoding process."""

from abc import ABC, abstractmethod

from narwhals.typing import IntoFrameT, IntoSeriesT


class BaseTransformer(ABC):
    """Abstract base class for all transformers."""

    @abstractmethod
    def fit(self, X: IntoFrameT, y: IntoSeriesT | None = None) -> None:
        """Fits the transformer to the data."""
        pass

    @abstractmethod
    def transform(self, X: IntoFrameT) -> IntoFrameT:
        """Transforms the data."""
        pass

    def fit_transform(self, X: IntoFrameT, y: IntoSeriesT | None = None) -> IntoFrameT:
        """Fits and transforms the data."""
        self.fit(X, y)
        return self.transform(X)
