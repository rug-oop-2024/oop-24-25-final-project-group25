from typing import List
import numpy as np
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature


def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """
    Detect the features of a given dataset.
    Assumption: only categorical and numerical features and no NaN values.
    Args:
        dataset: Dataset containing the features.
    Returns:
        List[Feature]: List of features belonging to the dataset.
    """
    features = []
    dataframe = dataset.read()

    numericals = dataframe.select_dtypes(include=np.number)
    numericals_names = numericals.columns

    for name in numericals_names:
        features.append(Feature(name=name, type="numerical"))

    for name in dataframe.columns:
        if name not in numericals_names:
            features.append(Feature(name=name, type="categorical"))

    return features
