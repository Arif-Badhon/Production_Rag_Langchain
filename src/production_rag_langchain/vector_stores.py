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

# 2. Mock documents to store
texts = [
    "LangChain handles orchestration for LLM applications.",
    "Chroma is a lightweight AI-native vector database.",
    "MiniLM models are fast, compact, and run completely locally."
]

# 3. Create the Chroma database using the mini embeddings
vector_store = Chroma.from_texts(
    texts=texts,
    embedding=embedding_model,
    persist_directory="./chroma_mini_db"  # Remove this argument to keep it purely in-memory
)

def search_vector_store(query: str, k: int = 1):
    results = vector_store.similarity_search(query, k=k)
    result_score = vector_store.similarity_search_with_score(query, k=k)
    return results, result_score



if __name__ == "__main__":
    # 4. Query the vector store
    results, result_score = search_vector_store("Tell me about VectorDB")

    print(f"Results: {results}")
    print(f"Most relevant document: {result_score[0][0].page_content}")
    print(f"Similarity score: {result_score[0][1]}")
