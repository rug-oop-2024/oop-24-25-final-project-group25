from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

class LogisticRegressionModel:

    _type: str = "classification"

    def __init__(self, **kwargs):
        self._model = SklearnLogisticRegression(**kwargs)
        self._parameters = {}

    def fit(self, X, y):
        """Fits the logistic regression model to the data."""
        self._model.fit(X, y)
        self._parameters = {
            'coef': self._model.coef_,
            'intercept': self._model.intercept_
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

