def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping chunks.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks


if __name__ == "__main__":
    sample = "This is a test. " * 100
    chunks = chunk_text(sample)
    print(f"Total chunks: {len(chunks)}")
    print(f"\nFirst chunk:\n{chunks[0]}")
    print(f"\nSecond chunk:\n{chunks[1]}")
    print(f"\nOverlap between chunk 1 and 2:")
    print(f"End of chunk 1: ...{chunks[0][-50:]}")
    print(f"Start of chunk 2: {chunks[1][:50]}...")