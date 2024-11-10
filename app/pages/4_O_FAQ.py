import streamlit as st

st.set_page_config(page_title="FAQ", page_icon="?")
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


st.write("# ? Deployment")
write_helper_text(
    "In this section, you can quench the thirst of that little curious mind"
)

with st.expander(label="What OS was used for this assignment?"):
    st.write("Vivat Windows!")

with st.expander(label="Can I statistics?"):
    st.write("We all have that question.")

with st.expander(label="Does your code have an HTML documentation?"):
    st.write("Quite so, in the 'docs' foder")
