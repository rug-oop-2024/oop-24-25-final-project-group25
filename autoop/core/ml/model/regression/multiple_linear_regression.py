from sklearn.linear_model import LinearRegression
from autoop.core.ml.model.model import Model
import numpy as np


class MultipleLinearRegression(Model):

    _type = "regression"
    _name: str = "multiple linear regression"

    def __init__(self):
        self._model = LinearRegression()

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray):
        self._model.fit(observations, ground_truth)

    def predict(self, observations: np.ndarray):
        return self._model.predict(observations)

    @property
    def parameters(self):
        return self._model.get_params()
