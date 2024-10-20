from abc import ABC, abstractmethod
from typing import Any
import numpy as np

METRICS = [
    "mean_squared_error",
    "accuracy",
    "precision_macro",
    "recall_macro"
] # add the names (in strings) of the metrics you implement

class Metric(ABC):
    """
    Abstract base class for all metrics.
    """
    # your code here
    # remember: metrics take ground truth and prediction as input and return a real number
    @abstractmethod
    @staticmethod
    def evaluate(predictions: list, actual: list) -> float:
        """
        Evaluate the predictions based on the given ground truth.

        Args:
            predictions: ???? the predictions generated by our model.
            actual: ???? the actual grounbd truth.

        Returns:
            float number representing our evaluated metric.
        """
        pass

    def __call__(self, predictions, actual) -> float:
        return self.evaluate(predictions, actual)

# add here concrete implementations of the Metric class

class MetricMSE(Metric):
    """
    Class representing the mean-squared-error metric.
    """
    @staticmethod
    def evaluate(predictions: np.ndarray, actual: np.ndarray) -> float:
        """
        Evaluate the predictions based on the given ground truth.

        Args:
            predictions: ???? the predictions generated by our model.
            actual: ???? the actual grounbd truth.

        Returns:
            float number representing our evaluated metric.
        """
        n = len(predictions)
        sum = 0

        for i in range(n):
            sum += (predictions[i] - actual[i])**2

        return sum/n

class MetricAccuracy(Metric):
    """
    Class representing the accuracy metric.
    """
    @staticmethod
    def evaluate(predictions: np.ndarray, actual: np.ndarray):
        n = len(predictions)
        sum = 0

        for i in range(n):
            sum += int(predictions[i] == actual[i])

        return sum/n


class MetricPrecisionMacro(Metric):
    """
    Class representing the macro precision metric.
    """
    @staticmethod
    def evaluate(predictions: np.ndarray, actual: np.ndarray):

        categories = np.unique(actual)
        macro_sum = 0
        for category in categories:
            TP = 0
            FP = 0
            for i in range(len(predictions)):
                if category == predictions[i]:
                    if predictions[i] == actual[i]:
                        TP += 1
                    else:
                        FP += 1
            macro_sum += TP/(TP+FP)

        return macro_sum/len(categories)

class MetricRecallMacro(Metric):
    """
    Class representing the macro recall metric.
    """
    @staticmethod
    def evaluate(predictions: np.ndarray, actual: np.ndarray):


        categories = np.unique(actual)
        macro_sum = 0
        for category in categories:
            TP = 0
            FN = 0
            for i in range(len(predictions)):
                if category == predictions[i]:
                    if predictions[i] == actual[i]:
                        TP += 1
                else:
                    if predictions[i] != actual[i]:
                        FN += 1

            macro_sum += TP/(TP+FN)

        return macro_sum/len(categories)


def get_metric(name: str) -> Metric:
    # Factory function to get a metric by name.
    # Return a metric instance given its str name.
    match name:
        case "mean_squared_error":
            return MetricMSE()
        case "accuracy":
            return MetricAccuracy()
        case "precision_macro":
            return MetricPrecisionMacro()
        case "recall_macro":
            return MetricRecallMacro()
        case _:
            raise KeyError("NO SUCH METRIC FOUND")