import os
import requests
from bs4 import BeautifulSoup

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_pdf_documents(pdf_dir):
    documents = []
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_dir, filename))
            documents.extend(loader.load())
    return documents


def load_txt_documents(txt_dir):
    documents = []
    for filename in os.listdir(txt_dir):
        if filename.endswith(".txt"):
            loader = TextLoader(os.path.join(txt_dir, filename))
            documents.extend(loader.load())
    return documents


def load_web_documents(url_file):
    documents = []
    with open(url_file, 'r') as f:
        urls = f.read().splitlines()

    for url in urls:
        try:
            html = requests.get(url, timeout=5).text
            soup = BeautifulSoup(html, "html.parser")
            text = soup.get_text()
            documents.append(Document(page_content=text, metadata={"source": url}))
        except Exception as e:
            print(f"Failed to load {url}: {e}")
    return documents


def chunk_documents(documents, chunk_size=500, chunk_overlap=75):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)


def ingest_all_documents(pdf_dir="data/pdfs", txt_dir="data/txts", url_file="data/urls.txt"):
    pdf_docs = load_pdf_documents(pdf_dir)
    txt_docs = load_txt_documents(txt_dir)
    web_docs = load_web_documents(url_file)

    all_docs = pdf_docs + txt_docs + web_docs
    print(f"Loaded {len(all_docs)} raw documents")

    chunked_docs = chunk_documents(all_docs)
    print(f"Chunked into {len(chunked_docs)} total segments")

    return chunked_docs
