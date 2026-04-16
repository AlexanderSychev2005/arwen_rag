from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter


def build_vector_database():
    print("Loading the books...")

    loader = DirectoryLoader(
        "books",
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()

    print(f"Loaded documents: {len(documents)}")

    print("Splitting the text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)
    print(f"Text is splitted into {len(chunks)} fragments (chunks).")

    print("Loading the local embeddings model...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("Creating the Qdrant database and saving...")

    QdrantVectorStore.from_documents(
        chunks,
        embeddings,
        url="http://localhost:6333",
        collection_name="middle_earth_lore",
    )
    print("Success! The vector base is saved as 'qdrant_db'.")


if __name__ == "__main__":
    build_vector_database()
