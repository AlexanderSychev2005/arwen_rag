import os
from typing import List

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

QDRANT_URL = "http://localhost:6333"
EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"
VECTOR_SIZE = 384  # Size for all-MiniLM-L6-v2 embeddings


def init_collection(client: QdrantClient, collection_name: str) -> None:
    """
    Checks if a Qdrant collection exists and creates it if it does not.
    """
    if not client.collection_exists(collection_name):
        print(f"[*] Creating new collection: '{collection_name}'...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
    else:
        print(f"[*] Collection '{collection_name}' already exists.")


def build_vector_database(folder_path: str, collection_name: str) -> None:
    """
    Loads text files from a directory (including subdirectories), splits them into chunks,
    generates embeddings, and uploads them to the specified Qdrant collection.
    """
    if not os.path.exists(folder_path):
        print(f"[Error] Directory '{folder_path}' does not exist. Aborting.")
        return

    print(f"\nProcessing knowledge from: {folder_path}")

    print("[1/5] Loading documents from directories...")
    loader = DirectoryLoader(
        folder_path,
        glob="**/*.txt",  # The '**' pattern ensures all subfolders are scanned
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()
    print(f"-> Loaded {len(documents)} documents.")

    if not documents:
        print("[!] No documents found. Please check your folder structure.")
        return

    # 2. Split the text
    print("[2/5] Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150, length_function=len
    )
    chunks: List[Document] = text_splitter.split_documents(documents)
    print(f"-> Text splitted into {len(chunks)} fragments.")

    # 3. Initialize Qdrant Client and Collection
    print("[3/5] Connecting to Qdrant and initializing collections...")
    qdrant_client = QdrantClient(url=QDRANT_URL)
    init_collection(client=qdrant_client, collection_name=collection_name)

    # 4. Load the local embeddings model
    print("[4/5] Loading the embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)

    # 5. Upload to Database
    print(f"[5/5] Uploading vectors to collection '{collection_name}'...")
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        embedding=embeddings,
    )

    # add_documents appends new vectors safely to the existing collection
    vector_store.add_documents(chunks)
    print(
        f"Success! Knowledge from '{folder_path}' is now stored in '{collection_name}'.\n"
    )


if __name__ == "__main__":
    # Building the Middle Earth lore database
    # build_vector_database(folder_path="books", collection_name="middle_earth_lore")

    # Initializing empty memory collection
    client = QdrantClient(url=QDRANT_URL)
    init_collection(client=client, collection_name="arwen_episodic_memory")
    print("Empty memory collection is ready for conversations!\n")
