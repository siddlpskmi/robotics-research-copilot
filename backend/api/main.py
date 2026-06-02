import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import shutil

from ingestion.pipeline import ingest_pdf, ask

app = FastAPI(
    title="Robotics Research Copilot",
    description="Multimodal RAG + Agentic AI for Robotics & CV Research",
    version="1.0.0"
)

# ── Request Models ────────────────────────────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str
    collection_name: str = "research_papers"

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    return {"status": "online", "message": "Robotics Research Copilot is running!"}


@app.post("/ingest")
async def ingest_paper(file: UploadFile = File(...)):
    """
    Upload a PDF and ingest it into the vector database.
    """
    # Save uploaded file temporarily
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Ingest it
    collection_name = file.filename.replace(".pdf", "").replace(" ", "_")
    num_chunks = ingest_pdf(temp_path, collection_name=collection_name)

    # Clean up temp file
    os.remove(temp_path)

    return {
        "status": "success",
        "filename": file.filename,
        "chunks_stored": num_chunks,
        "collection": collection_name
    }


@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """
    Ask a question and get an answer from the vector database.
    """
    answer = ask(request.question, request.collection_name)
    return {
        "question": request.question,
        "answer": answer,
        "collection": request.collection_name
    }
    @app.post("/explain-figure")
async def explain_figure_endpoint(
    file: UploadFile = File(...),
    question: str = None
):
    """
    Upload an image and get an AI explanation of the figure.
    """
    # Save uploaded image temporarily
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Explain the figure
    from multimodal.vision import explain_figure
    explanation = explain_figure(temp_path, question)

    # Clean up
    os.remove(temp_path)

    return {
        "filename": file.filename,
        "question": question,
        "explanation": explanation
    }
