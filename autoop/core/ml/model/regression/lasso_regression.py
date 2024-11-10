import numpy as np
from sklearn import linear_model
from autoop.core.ml.model.model import Model


class Lasso(Model):
    """
    Facade class for the sklean Lasso model.

    Attributes:
        _type(str): static attribute representing the Model's type: regression
        _name(str): static attribute representing the model's name
        _lasso: linear_model.Lasso model used for the calculations.

    """

    _type: str = "regression"
    _name: str = "lasso regression"

    def __init__(self) -> None:
        """Inititalise instance of the Lasso class"""
        self._lasso: linear_model.Lasso = linear_model.Lasso()

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Adjust model's parameters in accordance with the observations and
        ground truth.

        Args:
            observations: ndarray of observations to be fitted
            ground_truth: ndarray containing ground truth to be fitted

        Returns:
            None
        """
        self._lasso.fit(observations, ground_truth)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict the ground truth based on given observations.

        Args:
            observations: ndarray of observations used to reach prediction

        Returns:
            ndarray: the predictions for the given observations
        """
        return self._lasso.predict(observations)

    @property
    def parameters(self) -> dict:
        """
        Return model's parameters.

        Returns:
            dict: model's parameters
        """
        return self._lasso.get_params()
