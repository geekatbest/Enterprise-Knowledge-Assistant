import os
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS

from vectorstore.vectorstore import load_faiss_index  # Corrected import
from llm_wrapper import get_llm

load_dotenv()

def build_rag_chain(vectorstore_path="vectorstore/faiss_index", k=4):
    vectordb = load_faiss_index(vectorstore_path)

    retriever = vectordb.as_retriever(search_kwargs={"k": k})

    prompt_template = """You are a helpful AI assistant for answering enterprise document questions.
Use only the following context to answer the question.
If you don't know, say you don't know.

Context:
{context}

Question: {question}
"""

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    )

    llm = get_llm()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain


def query_pipeline(question, vectorstore_path="vectorstore/faiss_index", k=4):
    chain = build_rag_chain(vectorstore_path, k)
    result = chain({"query": question})

    return {
        "answer": result["result"],
        "sources": [doc.metadata["source"] for doc in result["source_documents"]]
    }

if __name__ == "__main__":
    res = query_pipeline("What is the refund policy?")
    print("Answer:", res["answer"])
    print("Sources:", res["sources"])
