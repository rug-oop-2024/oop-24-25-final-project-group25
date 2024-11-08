from sklearn.linear_model import LinearRegression
from autoop.core.ml.model.model import Model
import numpy as np


class MultipleLinearRegression(Model):
    """
    Facade class for the sklearn's LinearRegression model.

    Attributes:
        _type(str): static attribute representing the Model's type: regression
        _name(str): static attribute representing the model's name
        _model(LinearRegression): LinearRegression model used for the
            calculations
    """

    _type = "regression"
    _name: str = "multiple linear regression"

    def __init__(self) -> None:
        """Initialise an instance of the MultipleLinearRegression class"""
        self._model = LinearRegression()

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Update the parameters based on the new provided data.

        Args:
            observations: observations to fit
            ground_truth: ground truth to fit

        Returns:
            None
        """

        self._model.fit(observations, ground_truth)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Make a prediction based on the given observations.

        Args:
            observations: observations for prediction

        Returns:
            ndarray: array containing the predictions
        """
        return self._model.predict(observations)

    @property
    def parameters(self) -> dict:
        """
        Return the model's parameters.

        Returns:
            dict: model's parameters
        """
        return self._model.get_params()
