from abc import abstractmethod, ABC
import numpy as np
from copy import deepcopy


class Model(ABC):
    """
    Abstract class for machine learning models.

    Attributes:
        _type(str): static attribute representing the Model's type: regression
            or classification
        _name(str): static attribute representing the model's name
        _parameters(dict): dictionary holding learned parameters.
    """

    _type: str
    _name: str

    def __init__(self) -> None:
        """Initialise Model instance"""
        self._parameters: dict = None

    @property
    def type(self) -> str:
        """
        Return model's type.

        Returns:
            str: type of the model
        """
        return self._type

    @property
    def name(self) -> str:
        """
        Return model's name.

        Returns:
            str: name of model
        """
        return self._name

    @property
    def parameters(self):
        """
        Return object's parameters.

        Returns:
            dict: dictionary containing the object's parameters.
        """
        return deepcopy(self._parameters)

    def __str__(self):
        return self._name

    @abstractmethod
    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Update the model's parameters based on new observations.

        Args:
            observations: the observations to fit
            ground_truth: ground truth to fit

        Returns:
            None
        """
        pass

    @abstractmethod
    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict ground truth for the observations based on the model's
        parameters.

        Args;
            observations: observations on which to conduct the prediction

        Returns:
            ndarray: array containing the predictions
        """
        pass
