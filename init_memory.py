from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

print("Connecting to Qdrant server...")
client = QdrantClient(url="http://localhost:6333")

collection_name = "arwen_episodic_memory"

if not client.collection_exists(collection_name):
    print(f"Creating collection '{collection_name}'...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
    print("Success! Arwen's memory is ready to be written.")
else:
    print(f"Collection '{collection_name}' already exists. No action needed.")
