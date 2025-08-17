import streamlit as st
import os
import asyncio
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Load API key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Ensure event loop exists (for Gemini async gRPC)
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Page Setup
st.set_page_config(page_title="📚 IDontReadPDFs.AI", page_icon="📄", layout="wide")
# ===== Custom CSS for Chat Bubbles =====
st.markdown("""
<style>
.chat-bubble-user {
    background: #e1f5fe;
    color: #000;
    padding: 12px 18px;
    border-radius: 14px;
    margin: 0.6rem 0;
    font-size: 20px;
    font-weight: 600;
    box-shadow: 0 2px 4px rgba(0,0,0,0.08);
    max-width: 80%;
    word-wrap: break-word;
}
.chat-bubble-ai {
    background: #edf2f7;
    color: #1a202c;
    padding: 12px 18px;
    border-radius: 14px;
    margin: 0.6rem 0;
    font-size: 18px;
    font-weight: 500;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    max-width: 80%;
    word-wrap: break-word;
}
</style>
""", unsafe_allow_html=True)
# Custom CSS
st.markdown("""
<style>
/* General App Styling */
body {
    background-color: #f9fafc;
    font-family: 'Segoe UI', sans-serif;
}

/* Title */
h1 {
    text-align: center;
    color: #222;
    font-size: 2.2rem;
    margin-bottom: 0.2rem;
}
h3 {
    text-align: center;
    font-weight: 400;
    color: #4a5568;
    font-size: 1.1rem;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #fff;
    padding: 1rem;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}

/* Chat Input */
[data-testid="stChatInput"] {
    border: 1px solid #cbd5e0 !important;
    border-radius: 20px !important;
    padding: 0.5rem !important;
}

/* Answer bubbles */
.answer-bubble {
    background: #edf2f7;
    color: #1a202c;
    padding: 1rem;
    border-radius: 12px;
    margin: 0.5rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

st.title("📚 IDontReadPDFs.AI")
st.markdown("### Upload a PDF & chat with it. Clean, fast, and powered by **Gemini 1.5 Flash** ✨")


st.write("")

# Upload Box
st.markdown("<div class='upload-box'>📂 Drag & Drop your PDF below</div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["pdf"], label_visibility="collapsed")

if uploaded_file:
    st.markdown(
        f"<p style='color:white; font-weight:600; font-size:16px;'>📄 {uploaded_file.name}</p>",
        unsafe_allow_html=True
    )
    # Save file
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Load PDF
    loader = PyPDFLoader("temp.pdf")
    pages = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = splitter.split_documents(pages)

    # Embeddings + FAISS
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY
    )
    db = FAISS.from_documents(docs, embeddings)

    # QA Chain
    retriever = db.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=GEMINI_API_KEY
        ),
        retriever=retriever,
        return_source_documents=True
    )

    # Chat Interface
    st.markdown("### 💬 Chat with your PDF")
    query = st.chat_input("Type your question here...")

    if query:
        with st.spinner("Thinking..."):
            result = qa_chain.invoke({"query": query})

            # User bubble
            st.markdown(f"<div class='chat-bubble-user'>🙋 {query}</div>", unsafe_allow_html=True)

            # AI bubble
            st.markdown(f"<div class='chat-bubble-ai'>🤖 {result['result']}</div>", unsafe_allow_html=True)

            # Sources
            with st.expander("📖 View Sources"):
                for doc in result["source_documents"]:
                    st.markdown(f"<div class='source-box'>{doc.page_content[:250]}...</div>", unsafe_allow_html=True)
