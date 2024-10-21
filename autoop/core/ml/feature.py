
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

    def __str__(self):
        raise NotImplementedError("To be implemented.")