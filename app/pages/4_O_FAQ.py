import streamlit as st

st.set_page_config(page_title="FAQ", page_icon="?")
st.logo("app\\images\\logo.png", size="large")


def write_helper_text(text: str):
    st.write(f'<p style="color: #888;">{text}</p>', unsafe_allow_html=True)


st.write("# ? Deployment")
write_helper_text(
    "In this section, you can quench the thirst of that little" +
    " curious mind you have."
)

with st.expander(label="What OS was used for this assignment?"):
    st.write("Vivat Windows!")

with st.expander(label="Can I statistics?"):
    st.write("We all have that question.")
