import datetime

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

from config import QDRANT_URL

print("Connecting to Qdrant Database...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Connection to the Lore DB
vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="middle_earth_lore",
    url=QDRANT_URL,
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Connection to the Episodic Memory DB
memory_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="arwen_episodic_memory",
    url=QDRANT_URL,
)


def save_to_memory(user_text: str, bot_response: str) -> None:
    """
    Saves the user's input and Arwen's response to the episodic memory collection.
    """
    memory_text = f"User said: {user_text}\nArwen replied: {bot_response}"
    doc = Document(
        page_content=memory_text,
        metadata={
            "source": "past_conversation",
            "timestamp": datetime.datetime.now().isoformat(),
        },
    )
    memory_store.add_documents([doc])
