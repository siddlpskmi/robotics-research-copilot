import fitz  # PyMuPDF
import os

def load_pdf(pdf_path: str) -> str:
    """
    Extract all text from a PDF file.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    doc = fitz.open(pdf_path)
    full_text = ""
    
    for page_num, page in enumerate(doc):
        text = page.get_text()
        full_text += f"\n--- Page {page_num + 1} ---\n{text}"
    
    doc.close()
    return full_text


if __name__ == "__main__":
    # Quick test
    text = load_pdf("test.pdf")
    print(text[:500])
    print(f"\nTotal characters extracted: {len(text)}")