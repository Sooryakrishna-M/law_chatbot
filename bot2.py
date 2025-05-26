# Import necessary libraries
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# Set Streamlit app title
st.title("📚 Law Chatbot (Offline - No API)")

# Load PDF and build the knowledge base
@st.cache_resource(show_spinner=True)
def load_knowledge_base():
    # Load PDF
    loader = PyPDFLoader("law_doc.pdf")  # replace with your actual file name
    documents = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    # Embed documents
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)

    return vectorstore

# Load the vector DB
with st.spinner("Loading knowledge base..."):
    vectorstore = load_knowledge_base()

# Load Hugging Face model for local inference
@st.cache_resource(show_spinner=False)
def load_llm():
    hf_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=256
    )
    return HuggingFacePipeline(pipeline=hf_pipeline)

llm = load_llm()

# Create RetrievalQA chain
qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# Initialize message history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)

# User input
prompt = st.chat_input("Ask me a legal question from the document...")
if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get response from the LLM
    with st.spinner("Thinking..."):
        response = qa.run(prompt)

    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
