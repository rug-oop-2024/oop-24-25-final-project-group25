from autoop.core.ml.artifact import Artifact
import streamlit as st

st.set_page_config(
    page_title="Instructions",
    page_icon="👋",
)
st.logo("app\images\logo.png", size="large")
st.markdown(open("INSTRUCTIONS.md").read())
