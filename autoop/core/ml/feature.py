from __future__ import annotations
from pydantic import BaseModel, Field


class Feature(BaseModel):
    """
    Class representing a feature of our dataset.

    Attributes:
        name: string representing the feature's name.
        type: string representing the feature's type (numerical or categorical)
    """

    name: str = Field()
    type: str = Field()

    def to_tuple(self) -> tuple[str, str]:
        """
        Turn the feature into a tuple.

        Returns:
            tuple: the string's tuple representation
        """
        return (self.name, self.type)

    @classmethod
    def from_tuple(cls, tuple_self: tuple[str, str]) -> Feature:
        """
        Construct a feature object from a tuple.

        Args:
            tuple_self: tuple representing the feature

        Rturns:
            Feature: feature created from the given tuple
        """
        return cls(name=tuple_self[0], type=tuple_self[1])

    def __str__(self) -> str:
        """
        Define feature's string representation.

        Returns:
            str: feature's string representation.
        """
        return f"Name: {self.name} of type: {self.type}"
