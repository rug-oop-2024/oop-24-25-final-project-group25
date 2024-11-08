import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from autoop.core.ml.model.model import Model

class KNNClassifierModel(Model):
    """
    K-Nearest Neighbors Classifier for classification tasks.

    Attributes:
        _type (str): Model type (classification)
        _name (str): Model name
        _model (KNeighborsClassifier): Instance of KNeighborsClassifier
    """

    _type: str = "classification"
    _name: str = "k-nearest neighbors classifier"

    def __init__(self) -> None:
        """Initialize an instance of the KNNClassifierModel class."""
        self._model = KNeighborsClassifier()

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """Train the model using observations and ground truth."""
        self._model.fit(observations, ground_truth)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Predict based on the provided observations."""
        return self._model.predict(observations)

    @property
    def parameters(self) -> dict:
        """Return model's parameters."""
        return self._model.get_params()
