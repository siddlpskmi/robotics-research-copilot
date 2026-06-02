from sentence_transformers import SentenceTransformer

# Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embeddings(chunks: list[str]) -> list:
    """
    Convert text chunks into embedding vectors.
    """
    embeddings = model.encode(chunks, show_progress_bar=True)
    return embeddings


if __name__ == "__main__":
    sample_chunks = [
        "The RPN shares convolutional features with the detection network.",
        "Faster R-CNN achieves real-time object detection using region proposals.",
        "The model uses anchor boxes of different scales and aspect ratios."
    ]
    
    embeddings = get_embeddings(sample_chunks)
    print(f"Number of chunks: {len(embeddings)}")
    print(f"Embedding size: {len(embeddings[0])}")
    print(f"\nFirst embedding (first 5 values): {embeddings[0][:5]}")