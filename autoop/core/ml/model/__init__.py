
from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression.multiple_linear_regression import MultipleLinearRegression
from autoop.core.ml.model.regression.lasso_regression import Lasso
from autoop.core.ml.model.regression.ridge_regression import Ridge

REGRESSION_MODELS = [
    "lasso regression",
    "multiple linear regression",
    "ridge regression"
]

CLASSIFICATION_MODELS = [
    "decision tree classifier"
]


def get_model(model_name: str) -> Model:
    match model_name:
        case "lasso regression":
            return Lasso()
        case "multiple linear regression":
            return MultipleLinearRegression()
        case "ridge regression":
            return Ridge()
        # case "decision tree classifier":
        #     return DecisionTreeClassifierModel()
        case _:
            raise KeyError("NO SUCH MODEL FOUND")