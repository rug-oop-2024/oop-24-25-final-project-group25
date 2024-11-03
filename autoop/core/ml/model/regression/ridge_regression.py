import numpy as np
from sklearn import linear_model
from autoop.core.ml.model.model import Model


class Ridge(Model):
    """
    Facade class for the sklean Ridge model.

    Attributes:
        _ridge: linear_model.Ridge model used for the calculations.
        _parameters: dictionary of the parameters, containing the
                    coeficient and intercept.
    """

    _type: str = "regression"
    _name: str = "ridge regression"

    def __init__(self):
        self._ridge: linear_model.Ridge = linear_model.Ridge()

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray):
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
        self.parameters["coef_"] = self._ridge.coef_
        self.parameters["intercept_"] = self._ridge.intercept_

    def predict(self, observations: np.ndarray):
        """
        Predict the ground truth based on given observations.

        Args:
            observations: ndarray of observations used to reach prediction.

        Returns:
            ndarray of the predictions for the given observations.
        """
        return self._ridge.predict(observations)

    @property
    def parameters(self):
        return self._ridge.get_params()
