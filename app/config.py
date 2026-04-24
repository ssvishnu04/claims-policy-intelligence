import os
from dotenv import load_dotenv

load_dotenv()

try:
    import streamlit as st
    STREAMLIT_SECRETS_AVAILABLE = hasattr(st, "secrets")
except Exception:
    st = None
    STREAMLIT_SECRETS_AVAILABLE = False


def get_config_value(key: str, default: str = "") -> str:
    if STREAMLIT_SECRETS_AVAILABLE:
        try:
            if key in st.secrets:
                return st.secrets[key]
        except Exception:
            pass

    return os.getenv(key, default)


GROQ_API_KEY = get_config_value("GROQ_API_KEY", "")
GROQ_MODEL = get_config_value("GROQ_MODEL", "llama-3.1-8b-instant")
EMBEDDING_MODEL = get_config_value(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-MiniLM-L3-v2"
)

RAW_DATA_DIR = "data/raw"
FAISS_INDEX_DIR = "vectorstore/faiss_index"