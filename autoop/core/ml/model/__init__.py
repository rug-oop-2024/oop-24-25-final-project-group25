"""
Package containing classification and regression models
and tools for their handling.

Subpackages:
    - classification: package for classification machine learning models
    - regression: package for regression machine learning models

Modules:
    - get_model: allows to get a ML model by its name
    - model: defines abstract Model class that stands at the base of the other
        ML models
"""
__all__ = [
    "classification",
    "regression",
    "get_model",
    "model"
]
