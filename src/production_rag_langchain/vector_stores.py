from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

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

# 2. Mock documents with metadata to store
texts = [
    "LangChain handles orchestration for LLM applications.",
    "Chroma is a lightweight AI-native vector database.",
    "MiniLM models are fast, compact, and run completely locally."
]
metadatas = [
    {"source": "langchain_doc", "category": "orchestration", "year": 2023},
    {"source": "chroma_doc", "category": "database", "year": 2023},
    {"source": "huggingface_doc", "category": "model", "year": 2021}
]

# 3. Create the Chroma database using the mini embeddings and metadata
vector_store = Chroma.from_texts(
    texts=texts,
    embedding=embedding_model,
    metadatas=metadatas,
    persist_directory="./chroma_mini_db"  # Remove this argument to keep it purely in-memory
)

def search_vector_store(query: str, k: int = 1, filter_dict: dict | None = None):
    results = vector_store.similarity_search(query, k=k, filter=filter_dict)
    result_score = vector_store.similarity_search_with_score(query, k=k, filter=filter_dict)
    return results, result_score



if __name__ == "__main__":
    # 4. Query with metadata filter
    # Example 1: Search only within category == 'database'
    results, result_score = search_vector_store(
        query="Tell me about VectorDB",
        k=1,
        filter_dict={"category": "database"}
    )

    print(f"Results: {results}")
    print(f"Most relevant document: {result_score[0][0].page_content}")
    print(f"Metadata: {result_score[0][0].metadata}")
    print(f"Similarity score: {result_score[0][1]}")
