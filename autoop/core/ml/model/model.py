from abc import abstractmethod, ABC
from autoop.core.ml.artifact import Artifact
import numpy as np
from copy import deepcopy
from typing import Literal


class Model(ABC):
    """
    Abstract class for machine learning models.

    Attributes:
        _parameters: dictionary holding learned parameters.
    """

    _type: str
    _name: str

    def __init__(self):
        self._parameters: dict = None

    @property
    def type(self) -> str:
        return self._type

    @property
    def name(self) -> str:
        return self._name

    @property
    def parameters(self):
        """
        Return object's parameters.

        Returns:
            dictionary containing the object's parameters.
        """
        return deepcopy(self._parameters)

    def __str__(self):
        return self._name

    @abstractmethod
    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """Update the model's parameters based on new observations."""
        pass

    @abstractmethod
    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict ground truth for the observations based on the model's
        parameters.
        """
        pass
