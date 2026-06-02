import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import arxiv
from groq import Groq
from dotenv import load_dotenv
from ingestion.pipeline import ingest_pdf, ask
import urllib.request

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── TOOLS ─────────────────────────────────────────────────────────────────────

def search_arxiv(query: str, max_results: int = 3) -> list[dict]:
    """
    Search arXiv for papers matching the query.
    """
    client_arxiv = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    papers = []
    for result in client_arxiv.results(search):
        papers.append({
            "title": result.title,
            "authors": [a.name for a in result.authors[:3]],
            "summary": result.summary[:300],
            "pdf_url": result.pdf_url,
            "paper_id": result.entry_id.split("/")[-1]
        })
    
    return papers


def download_and_ingest_paper(pdf_url: str, paper_id: str) -> str:
    """
    Download a paper from arXiv and ingest it into ChromaDB.
    """
    pdf_path = f"temp_{paper_id}.pdf"
    urllib.request.urlretrieve(pdf_url, pdf_path)
    num_chunks = ingest_pdf(pdf_path, collection_name=paper_id)
    os.remove(pdf_path)
    return f"Ingested {num_chunks} chunks from paper {paper_id}"


def query_paper(question: str, paper_id: str) -> str:
    """
    Ask a question about a specific ingested paper.
    """
    return ask(question, collection_name=paper_id)


# ── AGENT ─────────────────────────────────────────────────────────────────────

def run_agent(user_query: str) -> str:
    """
    Simple agentic loop:
    1. Search arXiv for relevant papers
    2. Download and ingest the top paper
    3. Answer the user's query from the paper
    """
    print(f"\n🤖 Agent starting for query: '{user_query}'")
    
    # Step 1 — Search arXiv
    print("\n🔍 Step 1: Searching arXiv...")
    papers = search_arxiv(user_query, max_results=2)
    
    if not papers:
        return "No papers found on arXiv for this query."
    
    print(f"✅ Found {len(papers)} papers:")
    for i, p in enumerate(papers):
        print(f"  {i+1}. {p['title']}")
    
    # Step 2 — Download and ingest top paper
    top_paper = papers[0]
    print(f"\n📥 Step 2: Downloading: {top_paper['title'][:50]}...")
    ingest_result = download_and_ingest_paper(
        top_paper["pdf_url"],
        top_paper["paper_id"]
    )
    print(f"✅ {ingest_result}")
    
    # Step 3 — Answer the query from the paper
    print(f"\n💡 Step 3: Answering your query from the paper...")
    answer = query_paper(user_query, top_paper["paper_id"])
    
    final_response = f"""
📄 Paper: {top_paper['title']}
👥 Authors: {', '.join(top_paper['authors'])}

💡 Answer: {answer}
"""
    return final_response


if __name__ == "__main__":
    result = run_agent("visual SLAM for mobile robots")
    print(result)