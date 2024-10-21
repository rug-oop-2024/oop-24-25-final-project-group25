from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier

class RandomForestClassifierModel:
    def __init__(self, **kwargs):
        self.model = SklearnRandomForestClassifier(**kwargs)
        self.parameters = {}

    def fit(self, X, y):
        """Fits the random forest classifier model to the data."""
        self.model.fit(X, y)
        self.parameters = {
            'n_estimators': self.model.n_estimators,
            'max_depth': self.model.max_depth
        }

    def predict(self, X):
        """Makes predictions using the fitted model."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Returns the predicted probabilities for each class."""
        return self.model.predict_proba(X)

    def get_params(self):
        """Returns the model parameters."""
        return self.parameters

        