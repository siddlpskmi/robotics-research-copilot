---
title: Robotics Research Copilot
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "6.15.2"
app_file: app.py
pinned: false
---

# 🤖 Robotics Research Copilot
> Multimodal RAG + Agentic AI system for Robotics & Computer Vision researchers

[![Live Demo](https://img.shields.io/badge/🤗-Live%20Demo-orange)](https://huggingface.co/spaces/siddahuja/robotics-research-copilot)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black)](https://github.com/siddlpskmi/robotics-research-copilot)

## 🎯 What it does
Upload robotics/CV research papers, ask questions, and let the AI agent automatically fetch papers from arXiv — powered by RAG, Agentic AI, and Multimodal vision.

## ✨ Features
- �� **PDF Ingestion** — Upload any research paper and ask questions from it
- 🔍 **Semantic Search** — Finds relevant content by meaning, not keywords
- 🤖 **Agentic AI** — Automatically fetches and ingests papers from arXiv
- 🖼️ **Multimodal Vision** — Understands figures, charts, and architecture diagrams
- 📝 **Raw Text Input** — Paste any text and ask questions from it
- ⚡ **REST API** — FastAPI backend with auto-generated Swagger docs

## 🛠️ Tech Stack
| Layer | Technology |
|---|---|
| LLM + Vision | Groq (Llama 3.3 + Llama 4 Scout) |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| Vector DB | Qdrant Cloud |
| Agent | LangGraph |
| Backend | FastAPI |
| UI | Gradio |
| Paper Fetching | arXiv API |

## 🚀 Quick Start
```bash
git clone https://github.com/siddlpskmi/robotics-research-copilot.git
cd robotics-research-copilot
conda create -n research-copilot python=3.11 -y
conda activate research-copilot
python -m pip install -r requirements.txt
echo "GROQ_API_KEY=your_key" > .env
echo "QDRANT_URL=your_url" >> .env
echo "QDRANT_API_KEY=your_key" >> .env
python frontend/app.py
```

## 👤 Author
**Siddharth Ahuja** — M.Eng. Intelligent Robotics, THD Deggendorf
- 📧 sidd.ahuja14@gmail.com
- 🐙 [GitHub](https://github.com/siddlpskmi)
