from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
import os

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder=None  
    )

def load_and_split_docs(path: str):
    loader = TextLoader(path)
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_documents(docs)

def load_vectorstore(doc_path: str, persist_dir: str = "vectorstore"):
    os.makedirs(persist_dir, exist_ok=True)
    index_path = os.path.join(persist_dir, "faiss_index")

    if os.path.exists(index_path):
        return FAISS.load_local(index_path, get_embeddings(), allow_dangerous_deserialization=True)

    docs = load_and_split_docs(doc_path)
    vectorstore = FAISS.from_documents(docs, get_embeddings())
    vectorstore.save_local(index_path)
    return vectorstore