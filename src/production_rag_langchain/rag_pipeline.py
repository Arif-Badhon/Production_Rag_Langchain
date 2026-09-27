"""
Install with: UV add langchain langchain-chroma langchain-huggingface sentence-transformers langchain-google-genai
"""

from langchain_google_genai import ChatGoogleGenerativeAI   
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
import tempfile

load_dotenv()

# Get Embedding Model
def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    """Load from local cache first; download only if not found locally."""
    try:
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"local_files_only": True}
        )
    except Exception:
        print(f"Model '{model_name}' not found locally. Downloading from Hugging Face Hub...")
        return HuggingFaceEmbeddings(model_name=model_name)

# 1. Initialize the lightweight "mini" embedding model
embedding_model = get_embedding_model("all-MiniLM-L6-v2")


def create_kb():
    """ create a vector store from knowledge base"""

    #split the knowledge base into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "], # list of separators to use for splitting the text
        chunk_size=500, # size of each chunk
        chunk_overlap=50, # overlap between chunks
        length_function=len, # function to use for calculating the length of each chunk
        is_separator_regex=False, # if True, the separators are treated as regular expressions
    )
    #load. knowledge base
    with open("docs/Arif_Uz_Zaman_AI_Resume.pdf", "r") as f:
        knowledge_base = f.read()
    doc = Document(page_content=knowledge_base,
    metadata={"source": "./docs/Arif_Uz_Zaman_AI_Resume.pdf"})

    chunks = text_splitter.split_documents([doc])

    # create vector store from the documents

    vector_store = Chroma.from_documents(chunks, embedding=embedding_model, persist_directory="./chroma_mini_db")
    return vector_store
    

def demo_basic_rag():
    vector_store = create_kb()





    

    


