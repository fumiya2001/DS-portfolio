import streamlit as st
import requests

API_URL = "http://localhost:8000/ask"

st.title("Simple RAG Demo")


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question"):
    st.session_state.messages.append({"role":"user", "content":prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    response = requests.post(API_URL, json={"query": prompt})
    answer = response.json().get("answer")

    st.session_state.messages.append({"role":"assistant", "content":answer})

    with st.chat_message("assistant"):
        st.markdown(answer)
   
