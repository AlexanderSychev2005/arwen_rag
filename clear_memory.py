from qdrant_client import QdrantClient

from config import QDRANT_URL


def clear_episodic_memory() -> None:
    """
    Connects to the local Qdrant database and completely deletes
    the episodic memory collection (dialogue history),
    preventing context poisoning from past failed interactions.
    """
    print("Initiating Memory Wipe Protocol...")

    try:
        client = QdrantClient(url=QDRANT_URL)
        print(f"Successfully connected to Qdrant at {QDRANT_URL}")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not connect to Qdrant database. Details: {e}")
        print("Ensure your Docker container with Qdrant is running.")
        return

    collection_name: str = "arwen_episodic_memory"

    if client.collection_exists(collection_name=collection_name):
        print(f"Found corrupted memory sphere: '{collection_name}'")

        client.delete_collection(collection_name=collection_name)
        print(f"SUCCESS: The collection '{collection_name}' has been obliterated.")
        print("The Agent's episodic memory is now a clean slate.")
    else:
        print(f"INFO: The collection '{collection_name}' does not exist.")
        print("No memory wipe required. The slate is already clean.")


if __name__ == "__main__":
    clear_episodic_memory()
