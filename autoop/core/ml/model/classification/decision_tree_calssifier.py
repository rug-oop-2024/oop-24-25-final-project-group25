from sklearn.tree import DecisionTreeClassifier as SkDecisionTreeClassifier
import numpy as np


class DecisionTreeClassifierModel:
    """
    Facade class for the sklean Lasso model.

    Attributes:
        _type(str): static attribute representing the Model's type:
            classification
        _name(str): static attribute representing the model's name
        _model(DecisionTreeClassifier): sklearn's DecisionTreeClassifier
            model used
            for the calculations.

    """

    _type: str = "classification"
    _name: str = "decision tree classifier"

    def __init__(self) -> None:
        """Initialise an instance of the class"""
        self._model = SkDecisionTreeClassifier()

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the decision tree classifier model to the data.

        Args:
            x: array of observations.
            y: array of ground truth

        Returns:
            None
        """
        self._model.fit(x, y)

    def predict(self, x) -> np.ndarray:
        """
        Makes predictions for the given observations.

        Args:
            x: observations on which to make prediction

        Returns:
            ndarray: array containing the predictions
        """
        return self._model.predict(x)

    @property
    def parameters(self) -> dict:
        """
        Returns the model parameters.

        Returns:
            dict: model's parameters
        """
        return self._model.get_params()
