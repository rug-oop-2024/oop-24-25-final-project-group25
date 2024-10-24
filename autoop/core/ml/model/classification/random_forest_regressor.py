from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier

class RandomForestClassifierModel:

    _type: str = "classification"

    def __init__(self, **kwargs):
        self._model = SklearnRandomForestClassifier(**kwargs)
        self._parameters = {}

    def fit(self, X, y):
        """Fits the random forest classifier model to the data."""
        self._model.fit(X, y)
        self._parameters = {
            'n_estimators': self._model.n_estimators,
            'max_depth': self._model.max_depth
        }

    def predict(self, X):
        """Makes predictions using the fitted model."""
        return self._model.predict(X)

    def predict_proba(self, X):
        """Returns the predicted probabilities for each class."""
        return self._model.predict_proba(X)

    def get_params(self):
        """Returns the model parameters."""
        return self._parameters

        