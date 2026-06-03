from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import os
from dotenv import load_dotenv
import uuid

load_dotenv()

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

VECTOR_SIZE = 384

def get_or_create_collection(collection_name: str):
    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )
        print(f"✅ Created Qdrant collection: {collection_name}")
    return collection_name


def store_chunks_qdrant(chunks: list[str], embeddings, collection_name: str):
    get_or_create_collection(collection_name)
    
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embeddings[i].tolist(),
            payload={"text": chunks[i]}
        )
        for i in range(len(chunks))
    ]
    
    client.upsert(collection_name=collection_name, points=points)
    print(f"✅ Stored {len(chunks)} chunks in Qdrant: {collection_name}")
    return len(chunks)


def query_qdrant(query_embedding, collection_name: str, n_results: int = 3) -> list[str]:
    get_or_create_collection(collection_name)
    
    results = client.query_points(
        collection_name=collection_name,
        query=query_embedding.tolist(),
        limit=n_results
    )
    
    return [r.payload["text"] for r in results.points]