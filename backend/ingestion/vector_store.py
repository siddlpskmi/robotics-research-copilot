import chromadb
from chromadb.config import Settings

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="./chroma_db")

def get_or_create_collection(collection_name: str = "research_papers"):
    """
    Get existing collection or create a new one.
    """
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def store_chunks(chunks: list[str], embeddings, collection_name: str = "research_papers"):
    """
    Store text chunks and their embeddings in ChromaDB.
    """
    collection = get_or_create_collection(collection_name)
    
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    
    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=chunks
    )
    
    print(f"Stored {len(chunks)} chunks in ChromaDB")
    return collection


def query_collection(query_embedding, collection_name: str = "research_papers", n_results: int = 3):
    """
    Find most similar chunks to a query embedding.
    """
    collection = get_or_create_collection(collection_name)
    
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results
    )
    
    return results["documents"][0]


if __name__ == "__main__":
    from embedder import get_embeddings

    sample_chunks = [
        "The RPN shares convolutional features with the detection network.",
        "Faster R-CNN achieves real-time object detection using region proposals.",
        "The model uses anchor boxes of different scales and aspect ratios.",
        "The system processes images at 5 fps on a GPU."
    ]

    print("Generating embeddings...")
    embeddings = get_embeddings(sample_chunks)

    print("Storing in ChromaDB...")
    store_chunks(sample_chunks, embeddings)

    print("\nQuerying ChromaDB...")
    query = get_embeddings(["how fast is the detection system?"])
    results = query_collection(query[0])

    print("\nTop matching chunks:")
    for i, result in enumerate(results):
        print(f"\n{i+1}. {result}")