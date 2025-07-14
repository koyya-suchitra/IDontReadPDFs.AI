import streamlit as st
import os
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Load your API key from .env
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="📚 IDontReadPDFs")
st.title("📄 IDontReadPDFs – Let AI Read for You!")

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("temp.pdf")
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(pages)

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)

    retriever = db.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo"),
        retriever=retriever
    )

    query = st.text_input("Ask a question about the PDF:")

    if query:
        with st.spinner("Let me think..."):
            result = qa_chain.run(query)
            st.write("### 🤖 Answer:")
            st.write(result)
