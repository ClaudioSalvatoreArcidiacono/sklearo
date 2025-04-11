"""This module provides base classes for transformers in the encoding process."""

from sklearn.base import BaseEstimator, TransformerMixin


class BaseTransformer(TransformerMixin, BaseEstimator):
    """Abstract base class for all transformers."""
