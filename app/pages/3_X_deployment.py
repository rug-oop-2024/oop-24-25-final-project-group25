import streamlit as st
from app.core.system import AutoMLSystem
from autoop.core.ml.pipeline import Pipeline
from autoop.functional.feature import detect_feature_types


st.set_page_config(page_title="Deployment", page_icon="📈")
st.logo("app\\images\\logo.png", size="large")


def write_helper_text(text: str):
    """
    Writing the given text in a specific style.

    Args:
        text: text to write

    Return:
        None
    """
    st.write(f'<p style="color: #888;">{text}</p>', unsafe_allow_html=True)


st.write("# X Deployment")
write_helper_text(
    "In this section, you can deploy a saved machine learning pipeline."
)

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")
pipelines = automl.registry.list(type="pipeline")

selected_pipeline = st.selectbox(
    label="Which pipeline would you like?",
    options=pipelines,
    index=None,
    format_func=lambda artifact: artifact.name,
)

if selected_pipeline is not None:
    pipeline = Pipeline.from_artefact(selected_pipeline, automl.registry)
    pipeline.execute()
    st.write(pipeline)

    if not st.toggle(label="Run with new data"):

        if st.button(label="Run on predefined data"):
            st.write(pipeline.results_as_string(pipeline.execute()))

    else:
        dataset = st.selectbox(
            label="Choose the desired dataset",
            options=datasets,
            index=None,
            format_func=lambda artifact: artifact.name,
        )
        if dataset is not None:

            features = detect_feature_types(dataset)
            input_features = st.multiselect(
                label="select input features", options=features
            )

            if input_features is not None:
                if st.button(label="Run on new data"):
                    st.write(
                        pipeline.predict_new_data(
                            dataset,
                            input_features,
                            as_str=True
                        )
                    )
