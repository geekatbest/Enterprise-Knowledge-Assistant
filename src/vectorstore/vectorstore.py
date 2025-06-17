# vectorstore.py

import os
import faiss
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings

load_dotenv()

def get_embedder():
    embed_type = os.getenv("EMBEDDING_TYPE", "huggingface").lower()
    
    if embed_type == "openai":
        return OpenAIEmbeddings()
    
    if embed_type == "huggingface":
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    raise ValueError("Unsupported EMBEDDING_TYPE")


def build_faiss_index(documents, save_path="vectorstore/faiss_index"):
    embedder = get_embedder()
    vectordb = FAISS.from_documents(documents, embedder)
    vectordb.save_local(save_path)
    print(f"FAISS index saved at: {save_path}")
    return vectordb


def load_faiss_index(save_path="vectorstore/faiss_index"):
    embedder = get_embedder()
    vectordb = FAISS.load_local(save_path, embedder)
    return vectordb
