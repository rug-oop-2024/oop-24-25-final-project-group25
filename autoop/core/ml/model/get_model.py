from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression.multiple_linear_regression import (
    MultipleLinearRegression,
)
from autoop.core.ml.model.regression.lasso_regression import Lasso
from autoop.core.ml.model.regression.ridge_regression import Ridge
from autoop.core.ml.model.classification.decision_tree_calssifier import (
    DecisionTreeClassifierModel
)
from autoop.core.ml.model.classification.knn_classifier import (
    KNNClassifierModel
)
from autoop.core.ml.model.classification.random_forest_classifier import (
    RandomForestClassifier
)

REGRESSION_MODELS = [
    "lasso regression",
    "multiple linear regression",
    "ridge regression",
]

CLASSIFICATION_MODELS = [
    "decision tree classifier",
    "k-nearest neighbors classifier",
    "random forest classifier"
]


def get_model(model_name: str) -> Model:
    """
    Return a model from a given name.

    Args:
        model_name: name of the model top return

    Returns:
        Model: model named by the given name
    """
    match model_name:
        case "lasso regression":
            return Lasso()
        case "multiple linear regression":
            return MultipleLinearRegression()
        case "ridge regression":
            return Ridge()
        case "decision tree classifier":
            return DecisionTreeClassifierModel()
        case "k-nearest neighbors classifier":
            return KNNClassifierModel()
        case "random forest classifier":
            return RandomForestClassifier()
        case _:
            raise KeyError("NO SUCH MODEL FOUND")
