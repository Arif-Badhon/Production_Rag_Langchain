import os
import tempfile
from dotenv import load_dotenv
# FIX: Use the modern, active TextLoader package to resolve the DeprecationWarning
from langchain_core.documents import Document
from langchain_core.document_loaders.blob_loaders import Blob
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

def load_text_file():
    # Create an in-memory binary stream structure exactly like a file would provide
    blob = Blob.from_data("Hello, this is a sample text file...", mime_type="text/plain")
    
    # Generate the unified document format natively
    documents = [
        Document(
            page_content=blob.as_string(), 
            metadata={"source": "in_memory_blob", "type": "text"}
        )
    ]
    print(f"\nLoaded {len(documents)} documents from in-memory stream")
    for i, doc in enumerate(documents):
        print(f"\n--- Document {i+1} ---")
        print(f"Page Content:\n{doc.page_content}")
        print(f"Metadata: {doc.metadata}")
    
    # This list is fully prepared for text splitting or vector storage!


def load_pdf_file(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"\nLoaded {len(documents)} documents from PDF file")
    for i, doc in enumerate(documents):
        print(f"\n--- Document {i+1} ---")
        print(f"Page Content:\n{doc.page_content}")
        print(f"Metadata: {doc.metadata}")


if __name__ == "__main__":
    load_text_file()
    load_pdf_file("docs/Arif_Uz_Zaman_AI_Resume.pdf")
