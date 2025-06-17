import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI

load_dotenv()

def get_llm():
    model_name = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment.")

    return ChatOpenAI(
        temperature=0.0,
        model_name=model_name,
        openai_api_key=api_key
    )
