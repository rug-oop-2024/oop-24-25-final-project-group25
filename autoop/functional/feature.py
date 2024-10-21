
from typing import List
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature

def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """Assumption: only categorical and numerical features and no NaN values.
    Args:
        dataset: Dataset
    Returns:
        List[Feature]: List of features with their types.
    """
    features = []
    dataframe = dataset.read()
    column_values = dataframe.dtypes

    for name in column_values.index:
        type = column_values[name]
        if type == object:
            type = "categorical"
        else:
            type = "numerical"
        features.append(Feature(name = name, type = type))

    return features

