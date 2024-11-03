from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Literal
import numpy as np

from autoop.core.ml.dataset import Dataset


class Feature(BaseModel):
    """
    Class representing a feature of our dataset.

    Attributes:
        name: string representing the feature's name.
        type: string representing the feature's type (numerical or categorical)
    """

    # attributes here
    name: str = Field()
    type: str = Field()

    def to_tuple(self) -> tuple[str, str]:
        return (self.name, self.type)

    @classmethod
    def from_tuple(cls, tuple_self: tuple[str, str]) -> Feature:
        return cls(name=tuple_self[0], type=tuple_self[1])

    def __str__(self):
        return f"Name: {self.name} of type: {self.type}"
