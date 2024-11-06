import numpy as np
from sklearn import linear_model
from autoop.core.ml.model.model import Model


class Ridge(Model):
    """
    Facade class for the sklean Ridge model.

    Attributes:

        _type(str): static attribute representing the Model's type: regression
        _name(str): static attribute representing the model's name
        _ridge(Ridge): linear_model.Ridge model used for the calculations
    """

    _type: str = "regression"
    _name: str = "ridge regression"

    def __init__(self) -> None:
        """
        Initialise instance of the Ridge class.

        Returns:
            None
        """
        self._ridge: linear_model.Ridge = linear_model.Ridge()

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Adjust model's parameters in accordance with the observations and
        ground truth.

        Args:
            observations: ndarray of observations to be fitted.
            ground_truth: ndarray containing ground truth to be fitted.

        Returns:
            None
        """
        self._ridge.fit(observations, ground_truth)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict the ground truth based on given observations.

        Args:
            observations: ndarray of observations used to reach prediction.

        Returns:
            ndarray of the predictions for the given observations.
        """
        return self._ridge.predict(observations)

    @property
    def parameters(self) -> dict:
        """
        Return model's paranmeters.

        Returns:
            dict: model's parameters
        """
        return self._ridge.get_params()
