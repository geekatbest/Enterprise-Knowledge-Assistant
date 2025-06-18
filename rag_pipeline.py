from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from src.llm_wrapper import get_llm
from src.vectorstore import load_vectorstore

def build_prompt():
    template = """
    You are an enterprise knowledge assistant. Use the following context to answer the user's question concisely.

    Context:
    {context}

    Question: {question}
    """
    return PromptTemplate(template=template.strip(), input_variables=["context", "question"])

def get_retriever():
    vectorstore = load_vectorstore("vectorstore/faiss_index")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    return retriever

def build_rag_chain():
    retriever = get_retriever()
    llm = get_llm()
    prompt = build_prompt()

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    return rag_chain
