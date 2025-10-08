import fitz  # pip install PyMuPDF
import os
from backend.database import QdrantDB
from embeddings import EmbeddingGenerator

# -----------------------------
# Config
# -----------------------------
PDF_FOLDER = "pdfs"  # folder where your PDFs are stored
CHUNK_SIZE = 500      # number of characters per chunk

# -----------------------------
# Initialize DB and Embeddings
# -----------------------------
db = QdrantDB()
embedder = EmbeddingGenerator()

# -----------------------------
# Helper function: extract text from a PDF
# -----------------------------
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# -----------------------------
# Main function to process PDFs
# -----------------------------
def populate_qdrant_from_pdfs(folder_path):
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"No PDFs found in {folder_path}")
        return

    chunk_id = 0
    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)
        print(f"Processing: {pdf_path}")
        text = extract_text_from_pdf(pdf_path)

        # Split into chunks
        chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]

        # Insert chunks into Qdrant
        for chunk in chunks:
            embedding = embedder.generate_embedding(chunk)
            db.insert(id=chunk_id, vector=embedding, payload={"text_chunk": chunk})
            chunk_id += 1

    print(f"Inserted {chunk_id} chunks into Qdrant.")

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    populate_qdrant_from_pdfs(PDF_FOLDER)
