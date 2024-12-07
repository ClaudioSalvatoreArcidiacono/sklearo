from abc import ABC, abstractmethod

from narwhals.typing import IntoFrameT, IntoSeriesT


class BaseTransformer(ABC):

    @abstractmethod
    def fit(self, X: IntoFrameT, y: IntoSeriesT | None = None) -> None:
        pass

    @abstractmethod
    def transform(self, X: IntoFrameT) -> IntoFrameT:
        pass

    def fit_transform(self, X: IntoFrameT, y: IntoSeriesT | None = None) -> IntoFrameT:
        self.fit(X, y)
        return self.transform(X)
