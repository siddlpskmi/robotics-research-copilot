# Robotics Research Copilot

A domain-specific retrieval-augmented generation system combining agentic AI, multimodal vision, and semantic search for Robotics and Computer Vision research.

## Overview

The system enables researchers to upload academic papers, query their contents through natural language, and autonomously retrieve relevant literature from arXiv. A LangGraph-based agent orchestrates tool selection across document retrieval, semantic search, and visual understanding — eliminating the need for manual literature management.

The pipeline supports PDF ingestion, raw text input, and image-based figure explanation, making it applicable across the full spectrum of research workflows.

## Architecture
User Query
↓
LangGraph Agent — reasons about which tool to invoke
├── search_arxiv() — fetches and ingests papers from arXiv API on demand
├── query_rag() — semantic retrieval from Qdrant Cloud vector store
└── explain_figure() — Groq Llama 4 Scout vision model for figures and diagrams
↓
RAG Pipeline
PDF / Text Input → Extraction → Chunking → Sentence Transformer Embeddings → Qdrant Cloud → Semantic Retrieval → Groq LLM → Answer
## Key Components

**Retrieval-Augmented Generation Pipeline**
Documents are parsed using PyMuPDF, split into overlapping chunks to preserve semantic continuity across boundaries, and embedded using the all-MiniLM-L6-v2 Sentence Transformer model. Embeddings are stored in Qdrant Cloud for persistent retrieval across sessions.

**Agentic Workflow with LangGraph**
A LangGraph agent implements a ReAct-style reasoning loop — it evaluates the user query, selects the appropriate tool, observes the result, and iterates until it has sufficient context to respond. This enables multi-step autonomous behavior beyond simple question answering.

**arXiv Integration**
The search_arxiv tool queries the arXiv API by topic, downloads the top-ranked paper, runs it through the full ingestion pipeline, and makes it immediately queryable — without any manual intervention from the user.

**Multimodal Vision**
Research paper figures, architecture diagrams, and result charts are processed by Groq Llama 4 Scout, a vision-language model capable of semantic understanding of visual content rather than simple OCR.

**REST API**
A FastAPI backend exposes the ingestion and retrieval pipeline as HTTP endpoints with auto-generated OpenAPI documentation, enabling integration with external systems or frontends.

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Groq — LLaMA 3.3 70B Versatile |
| Vision Model | Groq — Llama 4 Scout |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| Vector Database | Qdrant Cloud (eu-central-1) |
| Agent Framework | LangGraph |
| Backend | FastAPI |
| Frontend | Gradio |
| Paper Retrieval | arXiv API |
| PDF Parsing | PyMuPDF |
| Language | Python 3.11 |

## Setup

```bash
git clone https://github.com/siddlpskmi/robotics-research-copilot.git
cd robotics-research-copilot
conda create -n research-copilot python=3.11 -y
conda activate research-copilot
python -m pip install -r requirements.txt
```

Create a `.env` file:
GROQ_API_KEY=your_groq_api_key
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_api_key
Run the Gradio interface:
```bash
python frontend/app.py
```

Run the FastAPI backend:
```bash
uvicorn backend.api.main:app --reload --port 8000
```

Live demo available at: https://huggingface.co/spaces/siddahuja/robotics-research-copilot

## Author

Siddharth Ahuja
M.Eng. Intelligent Robotics — Technische Hochschule Deggendorf
sidd.ahuja14@gmail.com
GitHub: https://github.com/siddlpskmi