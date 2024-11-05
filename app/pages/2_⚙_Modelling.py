import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.pipeline import Pipeline
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.metric import Metric, METRICS, get_metric
from autoop.core.ml.model.get_model import (
    CLASSIFICATION_MODELS,
    REGRESSION_MODELS,
    get_model,
)


st.set_page_config(page_title="Modelling", page_icon="📈")
st.logo("app\images\logo.png", size="large")


def write_helper_text(text: str):
    st.write(f'<p style="color: #888;">{text}</p>', unsafe_allow_html=True)


st.write("# ⚙ Modelling")
write_helper_text(
    "In this section, you can design a machine learning pipeline to train a model on a dataset."
)

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")
pipelines = automl.registry.list(type="pipeline")

# print existing pipelines:
st.write("Your pipelines:")

for pipeline in pipelines:
    st.write(pipeline.name)

# select action (adding or deleting a pipeline):
action = st.selectbox(
    label="What do you want to do?",
    options=["Configure new pipeline", "Delete existing pipeline"],
    index=None,
)

if action == "Configure new pipeline":
    # creating a new pipeline

    st.write("Configure your pipeline below:")

    dataset = st.selectbox(
        label="Choose the desired dataset",
        options=datasets,
        index=None,
        format_func=lambda artifact: artifact.name,
    )

    if not dataset is None:

        features = detect_feature_types(dataset)

        input_features = st.multiselect(label="select input features", options=features)

        target_feature = st.selectbox(
            label="select target feature", options=features, index=None
        )

        if not target_feature is None and not input_features is None:
            if target_feature in input_features:
                st.write("Target feature cannot be among the input features.")
            else:
                target_type = target_feature.type
                st.write(f"Detected task type: {target_type} ")

                if target_type == "numerical":
                    models_list = REGRESSION_MODELS
                elif target_type == "categorical":
                    models_list = CLASSIFICATION_MODELS

                model_name = st.selectbox(
                    label="Select desired model", options=models_list, index=None
                )
                if not model_name is None:
                    model = get_model(model_name)

                    split = st.number_input(
                        label="Choose the desired split percentage",
                        min_value=0,
                        max_value=100,
                        step=10,
                        value=80,
                    )

                    if split is not None:

                        metric_names = st.multiselect(
                            label="Select compatible desired metrics", options=METRICS
                        )

                        metrics = [
                            get_metric(metric_name) for metric_name in metric_names
                        ]

                        if any([metric.type != target_type for metric in metrics]):
                            st.write(
                                f"One or more of the chosen metrics is not for type {target_type}"
                            )
                        else:
                            pipeline = Pipeline(
                                metrics,
                                dataset,
                                model,
                                input_features,
                                target_feature,
                                split / 100,
                            )

                            st.write(pipeline)

                            if st.button(label="Run"):
                                st.write(pipeline.results_as_string(pipeline.execute()))

                            name = st.text_input(
                                label="Enter name of pipeline to save", value=None
                            )

                            version = st.text_input(
                                label="Enter version of the pipeline to save",
                                value="1.0.0",
                            )
                            if not name is None and not version is None:
                                if st.button(label="Save pieline"):
                                    st.write("Pipeline saved!")
                                    id = "p" + str(len(pipelines))
                                    artifact = pipeline.to_artifact(
                                        name=name,
                                        id=id,
                                        path=("pipeline" + name + ".bin"),
                                        version=version,
                                    )
                                    automl.registry.register(artifact)
                                    st.rerun()

elif action == "Delete existing pipeline":
    # delete a pipeline

    pipeline_to_delete = st.selectbox(
        label="Selecte dataset to delete",
        options=pipelines,
        index=None,
        format_func=lambda artifact: artifact.name,
    )
    if pipeline_to_delete is not None:
        if st.button(label="Delete"):
            automl.registry.delete(pipeline_to_delete.id)
            st.rerun()
