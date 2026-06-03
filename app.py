import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'backend'))

import gradio as gr
from ingestion.pipeline import ingest_pdf, ask
from ingestion.chunker import chunk_text
from ingestion.embedder import get_embeddings
from ingestion.vector_store import store_chunks
from agent.research_agent import run_agent
from multimodal.vision import explain_figure

# ── FUNCTIONS ─────────────────────────────────────────────────────────────────

def handle_pdf_upload(pdf_file, question):
    if pdf_file is None:
        return "Please upload a PDF first."
    collection = os.path.basename(pdf_file).replace(".pdf", "").replace(" ", "_")
    ingest_pdf(pdf_file, collection_name=collection)
    if question:
        answer = ask(question, collection_name=collection)
        return f"✅ Paper ingested!\n\n💡 Answer:\n{answer}"
    return f"✅ Paper ingested as '{collection}'! Now ask a question."


def handle_question(question, collection):
    if not question:
        return "Please enter a question."
    if not collection:
        return "Please enter a collection name."
    return ask(question, collection_name=collection)


def handle_agent(query):
    if not query:
        return "Please enter a query."
    return run_agent(query)


def handle_image(image_file, question):
    if image_file is None:
        return "Please upload an image."
    return explain_figure(image_file, question if question else None)


def handle_text(text, question):
    if not text:
        return "Please paste some text first."
    if not question:
        return "Please enter a question."
    chunks = chunk_text(text)
    embeddings = get_embeddings(chunks)
    store_chunks(chunks, embeddings, collection_name="text_input")
    return ask(question, collection_name="text_input")


# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Robotics Research Copilot", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("""
    # 🤖 Robotics Research Copilot
    ### Multimodal RAG + Agentic AI for Robotics & CV Research
    """)
    
    with gr.Tabs():
        
        # Tab 1 — Upload & Ask
        with gr.Tab("📄 Upload Paper & Ask"):
            gr.Markdown("Upload a robotics/CV research paper and ask questions about it.")
            with gr.Row():
                pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
                question_input = gr.Textbox(label="Your Question", placeholder="What is the main contribution?")
            upload_btn = gr.Button("Ingest & Ask", variant="primary")
            upload_output = gr.Textbox(label="Answer", lines=8)
            upload_btn.click(handle_pdf_upload, inputs=[pdf_input, question_input], outputs=upload_output)
        
        # Tab 2 — Ask from existing collection
        with gr.Tab("💬 Ask from Ingested Paper"):
            gr.Markdown("Ask questions from an already ingested paper.")
            collection_input = gr.Textbox(label="Collection Name", placeholder="e.g. faster_rcnn")
            question_input2 = gr.Textbox(label="Your Question", placeholder="How does the model work?")
            ask_btn = gr.Button("Ask", variant="primary")
            ask_output = gr.Textbox(label="Answer", lines=8)
            ask_btn.click(handle_question, inputs=[question_input2, collection_input], outputs=ask_output)
        
        # Tab 3 — Agent
        with gr.Tab("🤖 Research Agent"):
            gr.Markdown("Let the agent automatically find and answer from arXiv papers.")
            agent_input = gr.Textbox(label="Research Query", placeholder="Find me papers on visual SLAM")
            agent_btn = gr.Button("Run Agent", variant="primary")
            agent_output = gr.Textbox(label="Agent Response", lines=12)
            agent_btn.click(handle_agent, inputs=agent_input, outputs=agent_output)
        
        # Tab 4 — Multimodal
        with gr.Tab("🖼️ Explain Figure"):
            gr.Markdown("Upload a research paper figure or chart for AI explanation.")
            with gr.Row():
                image_input = gr.Image(label="Upload Figure", type="filepath")
                figure_question = gr.Textbox(label="Question (optional)", placeholder="What does this chart show?")
            figure_btn = gr.Button("Explain Figure", variant="primary")
            figure_output = gr.Textbox(label="Explanation", lines=8)
            figure_btn.click(handle_image, inputs=[image_input, figure_question], outputs=figure_output)

        # Tab 5 — Raw Text
        with gr.Tab("📝 Paste Text"):
            gr.Markdown("Paste any text and ask questions from it.")
            text_input = gr.Textbox(
                label="Paste your text here",
                placeholder="Paste a paragraph, abstract, or any text...",
                lines=8
            )
            text_question = gr.Textbox(
                label="Your Question",
                placeholder="What does this text say about..."
            )
            text_btn = gr.Button("Ingest & Ask", variant="primary")
            text_output = gr.Textbox(label="Answer", lines=6)
            text_btn.click(handle_text, inputs=[text_input, text_question], outputs=text_output)

if __name__ == "__main__":
    demo.launch(share=True)