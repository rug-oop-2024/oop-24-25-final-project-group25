"""
Package for machine learning related classes and functionalities.

Subpackages:
    - model: Package for the available machine learning models

Modules:
    - artifact: alows for the creation of artifact object to represent other
        objects
    - dataset: allows for the creation of a dataset object to represent a
        database
    - feature: defines the feature type to identify data by
    - metric: defines multiple classification and regression metrigs by which
        to analyse the predictions
    - pipeline: allows for the creation of a machine learning pipeline
"""

__all__ = [
    "model",
    "artifact",
    "dataset",
    "feature",
    "metric",
    "pipeline"
]
