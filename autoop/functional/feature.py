
from typing import List
import numpy as np
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

    numericals = dataframe.select_dtypes(include=np.number)
    print(numericals)
    numericals_names = numericals.columns

    for name in numericals_names:
        features.append(Feature(name = name, type = "numerical"))

    for name in dataframe.columns:
        if not name in numericals_names:
            features.append(Feature(name = name, type = "categorical"))

    return features

