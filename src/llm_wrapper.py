from langchain_groq import ChatGroq
from langchain_core.language_models.chat_models import BaseChatModel
import os
from dotenv import load_dotenv

load_dotenv()

def get_llm() -> BaseChatModel:
    return ChatGroq(
        model="llama3-70b-8192",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2,
    )
