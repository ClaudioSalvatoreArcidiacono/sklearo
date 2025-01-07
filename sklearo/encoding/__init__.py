"""This module provides encoding techniques for categorical features."""

from .target import TargetEncoder
from .woe import WOEEncoder

__all__ = ["WOEEncoder", "TargetEncoder"]
