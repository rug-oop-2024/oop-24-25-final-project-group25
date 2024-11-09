import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

st.set_page_config(page_title="Datasets", page_icon="📈")
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


st.write("📊 Datasets")
write_helper_text("In this section, you can add, delete, and view datasets.")

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")

# select one of the available tabs
manage_datasets, view_datasets = st.tabs(
    ["Manage Datasets", "View and Edit Datasets"]
)

with manage_datasets:
    # add or delete a dataset

    st.write("Your datasets:")

    for dataset in datasets:
        st.write(dataset.name)

    action = st.selectbox(
        label="What do you want to do?",
        options=["Add dataset", "Delete dataset"],
        index=None,
    )

    if action == "Add dataset":
        # add a dataset

        new_csv = st.file_uploader(label="Upload new dataset", type="csv")

        if new_csv is not None:

            new_dataframe = pd.read_csv(new_csv)

            name_dataframe = st.text_input(
                label="The name of your dataframe",
                value=None
            )

            if st.button(label="Submit csv"):
                if name_dataframe is None:
                    st.write("Enter name!")
                else:
                    automl.registry.register(
                        Dataset.from_dataframe(
                            new_dataframe,
                            name_dataframe,
                            name_dataframe + ".csv",
                            id=str(len(datasets)),
                        )
                    )
                    st.write("File submitted!")
                    disable_uploader = False
                    new_csv = None
                    st.rerun()

    elif action == "Delete dataset":
        # delete a dataset

        dataset_to_delete = st.selectbox(
            label="Selecte dataset to delete",
            options=datasets,
            index=None,
            format_func=lambda artifact: artifact.name,
        )
        if dataset_to_delete is not None:
            if st.button(label="Delete"):
                automl.registry.delete(dataset_to_delete.id)
                st.rerun()

with view_datasets:
    # view available datasets

    selection = st.selectbox(
        label="Which databse would you like to view?",
        options=datasets,
        index=None,
        format_func=lambda artifact: artifact.name,
    )

    if selection is not None:
        new_data = st.data_editor(selection.read())
        if new_data is not None:
            name_dataset = st.text_input(
                label="Name of your edited dataset",
                value=None
            )

            if st.button(label="Save edited dataset"):
                if name_dataset is None:
                    st.write("Enter name!")
                else:
                    automl.registry.register(
                        Dataset.from_dataframe(
                            new_data,
                            name_dataset,
                            name_dataset + ".csv",
                            id=str(len(datasets)),
                        )
                    )
                    st.write("Saved!")
                    disable_uploader = False
                    new_csv = None
                    st.rerun()

        # st.dataframe(selection.read())
