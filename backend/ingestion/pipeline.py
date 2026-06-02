import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.pdf_loader import load_pdf
from ingestion.chunker import chunk_text
from ingestion.embedder import get_embeddings
from ingestion.vector_store import store_chunks, query_collection
from retrieval.gemini_qa import answer_question


def ingest_pdf(pdf_path: str, collection_name: str = "research_papers"):
    """
    Full ingestion pipeline:
    PDF → Extract → Chunk → Embed → Store
    """
    print(f"\n📄 Loading PDF: {pdf_path}")
    text = load_pdf(pdf_path)
    print(f"✅ Extracted {len(text)} characters")

    print("\n✂️  Chunking text...")
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    print(f"✅ Created {len(chunks)} chunks")

    print("\n🧠 Generating embeddings...")
    embeddings = get_embeddings(chunks)
    print(f"✅ Generated {len(embeddings)} embeddings")

    print("\n💾 Storing in ChromaDB...")
    store_chunks(chunks, embeddings, collection_name)
    print(f"✅ Stored in collection: {collection_name}")

    return len(chunks)


def ask(question: str, collection_name: str = "research_papers"):
    """
    Full retrieval pipeline:
    Question → Embed → Search ChromaDB → Answer with Groq
    """
    print(f"\n🔍 Searching for relevant chunks...")
    query_embedding = get_embeddings([question])
    relevant_chunks = query_collection(query_embedding[0], collection_name)
    print(f"✅ Found {len(relevant_chunks)} relevant chunks")

    print("\n🤖 Asking Groq...")
    answer = answer_question(question, relevant_chunks)
    return answer


if __name__ == "__main__":
    # Step 1 — Ingest the PDF
    ingest_pdf("test.pdf", collection_name="faster_rcnn")

    # Step 2 — Ask questions from the actual paper
    questions = [
        "What is the main contribution of this paper?",
        "How does the Region Proposal Network work?",
        "What were the results on the PASCAL VOC dataset?"
    ]

    for question in questions:
        print(f"\n{'='*50}")
        print(f"❓ Question: {question}")
        answer = ask(question, collection_name="faster_rcnn")
        print(f"\n💡 Answer: {answer}")