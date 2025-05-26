#import langchain dependancies
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter 

#bring in streamlit for UI
import streamlit as st

# #bring in watsonx interface
# from watsonxlangchain import LangChainInterface
# Correct import for IBM Watsonx support
from langchain_ibm import WatsonxLLM

#set up credentials dictionary for the LLM
creds = {
    api_key: "YOUR_API_KEY",
}

#set up app title
st.title("Law Chatbot")

#Set up a session state message to store the LLM old messages
if 'messages' not in st.session_state:
    st.session_state.messages = []

#Display the chat history in the UI
for message in st.session_state.messages:
    if message["role"] == "user":
        st.chat_message("user").markdown(message["content"])
    else:
        st.chat_message("assistant").markdown(message["content"])

#Build a prompt for the LLM
prompt = st.chat_input("Ask me a question about the law")
if prompt:
    #Display the prompt in the UI
    st.chat_message('user').markdown(prompt)
    #Store the prompt in the session state messages
    st.session_state.messages.append({"role": "user", "content": prompt})
