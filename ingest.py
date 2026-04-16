import os
import pickle
from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer

DOCS_DIR = "docs/"
CHROMA_DIR = "chroma_db"
CHUNKS_FILE = "chunks.pkl"

def load_pdfs():
    texts = []
    for filename in os.listdir(DOCS_DIR):
        if filename.endswith(".pdf"):
            reader = PdfReader(os.path.join(DOCS_DIR, filename))
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text.strip():
                    texts.append({
                        "text": text,
                        "source": filename,
                        "page": i + 1
                    })
    return texts

def chunk_texts(pages, chunk_size=500, overlap=50):
    chunks = []
    for page in pages:
        text = page["text"]
        words = text.split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append({
                    "text": chunk,
                    "source": page["source"],
                    "page": page["page"]
                })
    return chunks

def ingest():
    print("PDFs load ho rahe hain...")
    pages = load_pdfs()
    chunks = chunk_texts(pages)
    print(f"Total chunks: {len(chunks)}")

    # chunks save karo BM25 ke liye
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)

    # dense embeddings banao
    print("Embeddings ban rahi hain...")
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    # Chroma mein store karo
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    
    try:
        client.delete_collection("docs")
    except:
        pass
    
    collection = client.create_collection("docs")
    collection.add(
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=[{"source": c["source"], "page": c["page"]} for c in chunks],
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )

    print("Done! chroma_db aur chunks.pkl ban gaye.")

if __name__ == "__main__":
    ingest()
