import chromadb
chroma_client = chromadb.Client()


collection_name = "Test_Collection"
collection = chroma_client.get_or_create_collection(name=collection_name)



documents = [
    {"id": "doc1", "text": "Hello, World!", "metadata": {"source": "A"}},
    {"id": "doc2", "text": "Hello, Universe!", "metadata": {"source": "B"}},
    {"id": "doc3", "text": "Hello, World Again!", "metadata": {"source": "C"}}
]


for doc in documents:
    collection.upsert(
        ids=doc["id"],
        documents=[doc["text"]],
        metadatas=[doc["metadata"]]
    )
        


query_text = "Hi. How are you?"

results = collection.query(
    query_texts=[query_text],
    n_results=2
)

print(results)