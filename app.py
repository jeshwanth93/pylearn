import streamlit as st
from transformers import pipeline
from huggingface_hub import login

# Secure login with Hugging Face token from secrets.toml
login(st.secrets["huggingface"]["token"])

# Load the LLM model
@st.cache_resource
def get_pipeline():
    return pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-beta")

llm = get_pipeline()

# Streamlit App UI
st.title("PYlearn - Python Learning Assistant")
st.markdown("Ask me anything about Python! I’ll help you learn, debug, and build with clear and simple explanations.")

user_input = st.text_input("Your question here:")

if user_input:
    with st.spinner("Thinking..."):
        response = llm(user_input, max_new_tokens=200)[0]['generated_text']
        st.markdown("**PYlearn:** " + response)
