import numpy as np
from sklearn.ensemble import RandomForestClassifier
from autoop.core.ml.model.model import Model


class RandomForestClassifierModel(Model):
    """
    Random Forest Classifier for classification tasks.

    Attributes:
        _type (str): Model type (classification)
        _name (str): Model name
        _model (RandomForestClassifier): Instance of RandomForestClassifier
    """

    _type: str = "classification"
    _name: str = "random forest classifier"

    def __init__(self) -> None:
        """Initialize an instance of the RandomForestClassifierModel class."""
        self._model = RandomForestClassifier()

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Train the model using observations and ground truth.

        Args:
            observations: array of observations
            fground_truth: array of ground truth

        Returns:
            None
        """
        self._model.fit(observations, ground_truth)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict based on the provided observations.

        Args:
            observations: observations used for prediction

        Returns:
            ndarray: array of predictions
        """
        return self._model.predict(observations)

    @property
    def parameters(self) -> dict:
        """
        Return model's parameters.

        Returns:
            dict: model's parameters
        """
        return self._model.get_params()