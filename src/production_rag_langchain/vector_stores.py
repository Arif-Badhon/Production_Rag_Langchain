from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Initialize the lightweight "mini" embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

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

# 4. Query the vector store
query = "Tell me about langchain"
results = vector_store.similarity_search(query, k=1)

result_score = vector_store.similarity_search_with_score(query, k=1)

print(f"Results: {results}")
print(f"Most relevant document: {result_score[0][0].page_content}")
print(f"Similarity score: {result_score[0][1]}")
