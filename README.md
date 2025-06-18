# 🧠 Enterprise Knowledge Assistant (RAG + LLM)

A smart, domain-aware chatbot that retrieves and summarizes enterprise knowledge documents using **LLMs** and **vector search**. Built with LangChain, FAISS, and Streamlit.

> 🚀 Ask questions about your enterprise documents — get accurate, contextual answers grounded in your data.

---

## 📌 Features

- 🗂️ **Document Ingestion**: Supports PDF, TXT files for enterprise knowledge bases.
- 🔍 **Semantic Search**: Uses Sentence Transformers + FAISS for high-speed retrieval.
- 🧠 **LLM-Powered QA**: Responses are generated using a Retrieval-Augmented Generation (RAG) pipeline.
- 💬 **Streamlit Chat UI**: Lightweight and interactive frontend for real-time conversation.

---

## 🛠️ Tech Stack

| Layer        | Technology                            |
|-------------|----------------------------------------|
| Embeddings  | `all-MiniLM-L6-v2` via `SentenceTransformers` |
| Vector DB   | FAISS                                  |
| LLM         | OpenAI / HuggingFace                   |
| Framework   | LangChain + Streamlit                  |
| LangChain Modules | `DocumentLoader`, `TextSplitter`, `Retriever`, `ConversationalRetrievalChain` |

---

## ⚙️ Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/your-username/enterprise-knowledge-assistant.git
   cd enterprise-knowledge-assistant
   
2. **Create virtual env (optional, I did that)**
   ```bash
    conda create -n rag-assistant python=3.10 -y
    conda activate rag-assistant

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt

4. **Add Your Documents**
   Place .pdf, .txt, or web docs files in a data/ folder.

5. **Run Ingestion Script**
   A succesfull ingestion message will be received. If not, debug the ingestion process, possible errors might be wrong folder structure, incorrect file format.

6. **Run Sreamlit App**

## All Set!

## Tech Stack
🔍 LangChain: For chaining retrieval + LLM
🧠 HuggingFace Transformers: Sentence embeddings
📦 FAISS: Vector database
🎨 Streamlit: Simple UI
💬 LLM: OpenAI or HuggingFace models (pluggable), used groq here

## 🧠 What I Learned
  Implementing end-to-end RAG pipelines
  Using FAISS for dense vector similarity search
  Tokenization strategies and chunk overlap logic
  Creating interactive ML UIs with Streamlit

##  Future Scope
🔐 Add authentication for enterprise users
☁️ Deploy on Streamlit Cloud or HuggingFace Spaces
🧩 Plug in custom LLMs or fine-tuned models
📎 Support for more file types (.csv, .pptx)