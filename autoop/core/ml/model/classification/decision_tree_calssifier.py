from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier

class DecisionTreeClassifierModel:

    _type: str = "classification"

    def __init__(self, **kwargs):
        self._model = SklearnDecisionTreeClassifier(**kwargs)
        self._parameters = {}

    def fit(self, X, y):
        """Fits the decision tree classifier model to the data."""
        self._model.fit(X, y)
        self._parameters = {
            'max_depth': self._model.get_depth(),
            'min_samples_split': self._model.min_samples_split
        }

    def predict(self, X):
        """Makes predictions using the fitted model."""
        return self._model.predict(X)

    @property
    def parameters(self):
        """Returns the model parameters."""
        return self._parameters

