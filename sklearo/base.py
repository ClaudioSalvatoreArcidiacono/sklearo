"""This module provides base classes for transformers in the encoding process."""

from sklearn.base import BaseEstimator, TransformerMixin


class BaseTransformer(TransformerMixin, BaseEstimator):
    """Abstract base class for all transformers."""

    def __sklearn_clone__(self):
        """Clone the transformer."""
        # This method is called by sklearn when cloning the transformer.
        # It should return a new instance of the transformer with the same parameters.
        return self.__class__(**self.get_params())
