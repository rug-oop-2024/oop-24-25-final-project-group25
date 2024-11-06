import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)
st.logo("app\images\logo.png", size="large")
st.sidebar.success("Select a page above.")


def write_helper_text(text: str):
    st.write(f'<p style="color: #888;">{text}</p>', unsafe_allow_html=True)


st.write("# Welcome! ")
write_helper_text("In this section, we welcome you as our beloved guest.")
