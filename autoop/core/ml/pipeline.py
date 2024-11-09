from typing import List
import pickle

from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.model.model import Model
from autoop.core.ml.model.get_model import get_model
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Metric, get_metric
from app.core.system import ArtifactRegistry
from autoop.functional.preprocessing import preprocess_features
import numpy as np


class Pipeline:
    """
    Class representing a machine learning pipeline.

    Attributes:
        metrics(List[Metric]): list of metrics to be applies on the predictions
        dataset(Dataset): dataset containing the data
        model(Model): the used machine learning model
        input_features(List[Feature]): list of features acting as input
        target_feature(Feature): target feature
        split(float): how much of the data that will be used for training
    """

    def __init__(
        self,
        metrics: List[Metric],
        dataset: Dataset,
        model: Model,
        input_ftrs: List[Feature],
        target_ftr: Feature,
        split: float = 0.8,
    ) -> None:
        """
        Initialize a Pipeline object.

        Args:
            metrics(List[Metric]): list of metrics to be applies on the
                predictions
            dataset(Dataset): dataset containing the data
            model(Model): the used machine learning model
            input_features(List[Feature]): list of features acting as input
            target_feature(Feature): target feature
            split(float): how much of the data that will be used for training

        Returns:
            None
        """
        self._dataset = dataset
        self._model = model
        self._input_features = input_ftrs
        self._target_feature = target_ftr
        self._metrics = metrics
        self._artifacts = {}
        self._split = split
        if (
            target_ftr.type == "categorical" and model.type != "classification"
        ):
            raise ValueError(
                "Model type must be classification for a categorical target"
            )
        if target_ftr.type == "numerical" and model.type != "regression":
            raise ValueError(
                "Model type must be regression for continuous target feature"
            )

    def __str__(self) -> str:
        """
        Define the object's string representation.

        Returns:
            str: string representation of the object
        """
        return f"""
Pipeline(
    model = {self._model.name} (of type {self._model.type}),
    dataset = '{self._dataset.name}',
    input_features = {list(map(str, self._input_features))},
    target_feature = {str(self._target_feature)},
    split = {self._split},
    metrics = {list(map(str, self._metrics))},
)
"""

    @property
    def model(self) -> Model:
        """
        Return the object's model.

        Returns:
            Model: object's model
        """
        return self._model

    @property
    def artifacts(self) -> List[Artifact]:
        """
        Used to get the artifacts generated during the pipeline execution to
        be saved.

        Returns:
            List[Artifact]
        """
        artifacts = []
        for name, artifact in self._artifacts.items():
            artifact_type = artifact.get("type")
            if artifact_type in ["OneHotEncoder"]:
                data = artifact["encoder"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
            if artifact_type in ["StandardScaler"]:
                data = artifact["scaler"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
        pipeline_data = {
            "input_features": self._input_features,
            "target_feature": self._target_feature,
            "split": self._split,
        }
        artifacts.append(
            Artifact(name="pipeline_config", data=pickle.dumps(pipeline_data))
        )
        artifacts.append(
            self._model.to_artifact(name=f"pipeline_model_{self._model.type}")
        )
        return artifacts

    def _register_artifact(self, name: str, artifact) -> None:
        """Add an artufact to the _artifacts attribute"""

        self._artifacts[name] = artifact

    def _preprocess_features(self) -> None:
        """Preprocess the features"""

        (target_feature_name, target_data, artifact) = preprocess_features(
            [self._target_feature], self._dataset
        )[0]
        self._register_artifact(target_feature_name, artifact)
        input_results = preprocess_features(
            self._input_features, self._dataset
        )
        for feature_name, data, artifact in input_results:
            self._register_artifact(feature_name, artifact)
        # Get the input vectors and output vector, sort by feature name
        self._output_vector = target_data
        self._input_vectors = [
            data for (feature_name, data, artifact) in input_results
        ]

    def _split_data(self) -> None:
        """
        Split the data into training and testing sets baset on
        the _split attribute.
        """

        split = self._split
        self._train_X = [
            vector[:int(split * len(vector))] for vector in self._input_vectors
        ]
        self._test_X = [
            vector[int(split * len(vector)):] for vector in self._input_vectors
        ]
        self._train_y = self._output_vector[
            :int(split * len(self._output_vector))
        ]
        self._test_y = self._output_vector[
            int(split * len(self._output_vector)):
        ]

    def _compact_vectors(self, vectors: List[np.array]) -> np.array:
        """
        Concatenate the given vectors on the first axis.

        Args:
            vectors: vectors to concatenate

        Returns:
            np.ndarray: the array containing the concatenated list of arrays
        """

        return np.concatenate(vectors, axis=1)

    def _train(self) -> None:
        """Train the model."""

        X = self._compact_vectors(self._train_X)
        Y = self._train_y
        self._model.fit(X, Y)

    def _evaluate(
            self,
            X: np.ndarray,
            Y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluate the predictions compared to the actual ground truth based on
        the metrics.

        Args:
            X:numpy ndarray holding the data on which to make the predictions.
            Y: actual ground truth.

        Returns:
            tuple: two arrays containing the
                predictions and the results of the metrics
        """

        self._predictions = self._model.predict(X)
        self._metrics_results = []
        for metric in self._metrics:
            result = metric.evaluate(self._predictions, Y)
            self._metrics_results.append((metric, result))
        return self._predictions, self._metrics_results

    def execute(self) -> dict[str, np.ndarray]:
        """
        Execute the pipeline: preprocess the data, train the model (if
        only_test_data is False), run the data through the model,
        evaluate the results based on the metrics.

        Args:
            only_test_data: indicates if we should train the model (False), or
                run just it on the entire dataset (True)

        Returns:
            dict: dictionary contain
        """
        self._preprocess_features()
        self._split_data()
        self._train()
        train_predictions, train_metrics = self._evaluate(
            self._compact_vectors(self._train_X), self._train_y
        )

        test_predictions, test_metrics = self._evaluate(
            self._compact_vectors(self._test_X), self._test_y
        )

        return {
            "train_metrics": train_metrics,
            "train_predictions": train_predictions,
            "test_metrics": test_metrics,
            "test_predictions": test_predictions,
        }

    @staticmethod
    def results_as_string(results: dict[str, np.ndarray]) -> str:
        """
        Turn the results of the execute method into a nicely formatted
        string.

        Args:
            results: the results to turn into a string

        Returns:
            str: the formatted results
        """
        results_string = "Here are your results:\n\n"
        results_string += "Train metrics:\n"

        for metric in results["train_metrics"]:
            results_string += f"{metric[0]}: {metric[1]},\n"

        train_pred = results["train_predictions"]
        results_string += f"\nTrain predictions:\n{train_pred}\n\n"
        results_string += "Test metrics:\n"

        for metric in results["test_metrics"]:
            results_string += f"{metric[0]}: {metric[1]},\n"

        test_pred = results["test_predictions"]
        results_string += f"\nTest predictions:\n{test_pred}"

        return results_string

    def predict_new_data(
        self, dataset: Dataset, input_features: List[Feature], as_str=False
    ) -> dict[str, np.ndarray] | str:
        """
        Make predictions based on the new new data.

        Args:
            dataset: the dataset on which to perform the prediction.
            input_features: the input features
            as_str: if the results should be returned as a string

        Returns:
            dict|str: representation of the predictions, str if as_str is True,
                dict otherwise
        """
        new_pipeline = Pipeline(
            metrics=self._metrics,
            dataset=dataset,
            model=self._model,
            input_features=input_features,
            target_feature=self._target_feature,
            split=0,
        )
        new_pipeline._preprocess_features()

        if not as_str:
            return {
                "Predictions": new_pipeline._model.predict(
                    new_pipeline._compact_vectors(new_pipeline._input_vectors)
                )
            }

        return "Predictions: " + str(
            new_pipeline._model.predict(
                new_pipeline._compact_vectors(new_pipeline._input_vectors)
            )
        )

    def to_artifact(
            self,
            name: str,
            id: str,
            path: str,
            version="1.0.0"
    ) -> Artifact:
        """
        Turn the current pipeline into an artifact.

        Args:
            name: name of the artifact
            id: id of the artifact into the database
            path: path of the artifact
            version: version of the artifact

        Returns:
            Artifact: the generated artifact
        """
        data = {
            "metrics": [metric.name for metric in self._metrics],
            "dataset": self._dataset.id,
            "model": self._model.name,
            "input_features": [
                feature.to_tuple() for feature in self._input_features
            ],
            "target_feature": self._target_feature.to_tuple(),
            "split": self._split,
        }
        return Artifact(
            name=name,
            type="pipeline",
            asset_path=path,
            data=pickle.dumps(data),
            version=version,
            id=id,
        )

    @classmethod
    def from_artefact(
        cls,
        artifact: Artifact,
        registry: ArtifactRegistry,
    ) -> "Pipeline":
        """
        Classmethod to turn the given artifact into a pipeline.

        Args:
            artifact: Artifact to turn
            registry: registry containing the artifact

        Return:
            Pipeline
        """
        data = pickle.loads(artifact.data)

        return cls(
            metrics=[get_metric(metric) for metric in data.get("metrics")],
            dataset=registry.get(data.get("dataset")),
            model=get_model(data.get("model")),
            input_ftrs=[
                Feature.from_tuple(feature) for
                feature in data.get("input_features")
            ],
            target_ftr=Feature.from_tuple(data.get("target_feature")),
            split=data.get("split"),
        )
