from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier

class DecisionTreeClassifierModel:
    def __init__(self, **kwargs):
        self.model = SklearnDecisionTreeClassifier(**kwargs)
        self.parameters = {}

    def fit(self, X, y):
        """Fits the decision tree classifier model to the data."""
        self.model.fit(X, y)
        self.parameters = {
            'max_depth': self.model.get_depth(),
            'min_samples_split': self.model.min_samples_split
        }

    def predict(self, X):
        """Makes predictions using the fitted model."""
        return self.model.predict(X)

    def get_params(self):
        """Returns the model parameters."""
        return self.parameters

