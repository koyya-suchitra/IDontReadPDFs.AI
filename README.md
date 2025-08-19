# 📚 IDontReadPDFs.AI

*An intelligent retrieval-augmented AI agent for context-aware, document-grounded interactions.*

---

## 🚀 Overview

Reading large PDFs can be cognitively exhausting and inefficient.
This project explores how **retrieval-augmented generation (RAG)** pipelines can act as **intelligent reading companions**, enabling contextual question answering without requiring users to manually parse through dense documents.

The system is designed with a **minimal, chat-style interface** built on Streamlit while leveraging a **LangChain pipeline + FAISS vector database + Gemini 1.5 Flash** for reasoning.

Rather than positioning it as a polished “product,” this repo documents an **exploration into agentic architectures** that augment human comprehension at scale.

---

## ✨ Core Capabilities

* **Context-Aware Q\&A**: Retrieve semantically relevant chunks from large PDFs and condition the LLM for precise answers.
* **Scalable Vector Search**: FAISS enables efficient retrieval even with hundreds of pages.
* **Fast & Intelligent Responses**: Powered by Gemini 1.5 Flash for low-latency, high-quality answers.
* **Minimal Frontend**: Streamlit UI offers a clean chat interface with expandable source tracing.
* **Agentic Flexibility**: Modular design allows swapping components (embeddings, LLMs, vector stores).

---

## 🏗️ Architecture

```
                ┌──────────────────┐
                │   PDF Document    │
                └────────┬─────────┘
                         │
                         ▼
               ┌─────────────────────┐
               │ PyPDFLoader (LangChain)│
               └─────────┬──────────┘
                         │
             Chunking via RecursiveCharacterTextSplitter
                         │
                         ▼
              ┌───────────────────────┐
              │  FAISS Vector Store    │
              │ (Google Embeddings)    │
              └──────────┬────────────┘
                         │
              Query → Semantic Retrieval
                         │
                         ▼
            ┌────────────────────────────┐
            │ Gemini 1.5 Flash (LLM)     │
            │ Conditioned on Retrieved   │
            │ Chunks (RAG pipeline)      │
            └───────────┬────────────────┘
                        │
                        ▼
             ┌─────────────────────────┐
             │  Streamlit Chat UI      │
             └─────────────────────────┘
```

---

## ⚙️ Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/idontreadpdfs-ai.git
cd idontreadpdfs-ai
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
```

> 🔑 You can get your Gemini API key from [Google AI Studio](https://ai.google.dev/).

### 5. Run the App

```bash
streamlit run app.py
```

---

## 🖥️ Usage

1. Upload a **PDF file** via the Streamlit interface.
2. Ask **natural language questions** in the chat box.
3. The system retrieves **relevant passages** and generates context-aware responses.
4. Expand the **“Sources” section** to trace which document chunks were used.

---

## 🧩 Applications

* **Knowledge Extraction** → Quickly parse technical or academic papers.
* **Business / Legal Docs** → Summarize and query contracts, agreements, or reports.
* **Domain-Specific Assistants** → Prototype industry-focused copilots (finance, healthcare, law).
* **Learning Aid** → Students/researchers can interactively query dense study material.

---

## 🔬 Technical Insights

* **Retrieval** → Done via FAISS vector similarity search.
* **Embeddings** → GoogleGenerativeAIEmbeddings (`models/embedding-001`).
* **LLM** → Gemini 1.5 Flash for low-latency, scalable inference.
* **Chunking** → RecursiveCharacterTextSplitter ensures semantic coherence.
* **Pipeline** → RetrievalQA chain (LangChain) orchestrates retrieval + reasoning.

---

## 📂 Project Structure

```
idontreadpdfs-ai/
│
├── app.py                # Streamlit frontend + pipeline integration
├── requirements.txt      # Python dependencies
├── .env.example          # Example env file
└── README.md             # Project documentation
```

---

## 🔮 Future Directions

* 🔍 Multi-PDF support (query across multiple documents).
* 🧠 Advanced reranking (to improve context precision).
* 🖼️ Multimodal extensions (combine PDFs + images + tables).
* ☁️ Deployment on cloud (for enterprise-scale document intelligence).
* 🔗 API layer for integration with external apps.

---

## 📜 License

This project is released under the **MIT License**.

---

## 🤝 Acknowledgements

* [LangChain](https://www.langchain.com/)
* [FAISS](https://github.com/facebookresearch/faiss)
* [Google Gemini](https://ai.google.dev/)
* [Streamlit](https://streamlit.io/)
✍️ Read the full story behind this project on my blog: [My Medium Blog](https://medium.com/@suchitrakoyya/from-pdf-overload-to-ai-magic-how-i-built-idontreadpdfs-ai-ef28a3d2e9ed)

