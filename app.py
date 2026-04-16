import os
import pickle
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

load_dotenv()

CHROMA_DIR = "chroma_db"
CHUNKS_FILE = "chunks.pkl"

st.set_page_config(page_title="Hybrid RAG", page_icon="🔍", layout="wide")
st.title("Hybrid RAG — LLaMA 3 + Groq")
st.caption("Apne documents se seedha jawab pao")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Documents Upload karo")
    uploaded_files = st.file_uploader(
        "PDF files select karo",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("Index Karo", type="primary"):
            with st.spinner("Documents index ho rahe hain..."):
                try:
                    os.makedirs("docs", exist_ok=True)
                    for file in uploaded_files:
                        with open(os.path.join("docs", file.name), "wb") as f:
                            f.write(file.getbuffer())

                    from ingest import ingest
                    ingest()

                    st.cache_resource.clear()
                    st.success(f"{len(uploaded_files)} file(s) index ho gayi!")

                except Exception as e:
                    st.error(f"Error: {str(e)}")

    st.divider()
    st.markdown("**Indexed files:**")
    if os.path.exists("docs"):
        files = [f for f in os.listdir("docs") if f.endswith(".pdf")]
        if files:
            for f in files:
                st.markdown(f"- {f}")
        else:
            st.caption("Koi file nahi abhi")

# ── Load Resources ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Model load ho raha hai...")
def load_resources():
    embed_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection("docs")

    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)

    tokenized = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)

    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    return embed_model, collection, chunks, bm25, groq_client

# ── Hybrid Search ─────────────────────────────────────────────────────────────
def hybrid_search(query, embed_model, collection, chunks, bm25, top_k=5):
    # dense search
    query_embedding = embed_model.encode([query])[0].tolist()
    dense_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    dense_ids = [int(id.split("_")[1]) for id in dense_results["ids"][0]]

    # sparse search BM25
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    sparse_ids = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:top_k]

    # RRF fusion
    rrf_scores = {}
    for rank, idx in enumerate(dense_ids):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (60 + rank + 1)
    for rank, idx in enumerate(sparse_ids):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (60 + rank + 1)

    top_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]
    return [chunks[i] for i in top_ids]

# ── Get Answer ────────────────────────────────────────────────────────────────
def get_answer(query, context_chunks, groq_client):
    context = "\n\n".join([c["text"] for c in context_chunks])
    prompt = f"""Neeche diya gaya context use karke question ka jawab do.
Agar context mein answer nahi hai toh "Mujhe document mein yeh information nahi mili." kaho.
Apna jawab clear aur concise rakho.

Context:
{context}

Question: {query}

Answer:"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content

# ── Chat UI ───────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if query := st.chat_input("Apna sawaal yahan likhein..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Dhundh raha hoon..."):
            try:
                # check karo ki index bana hai ya nahi
                if not os.path.exists(CHUNKS_FILE):
                    st.warning("Pehle sidebar mein PDF upload karo aur Index Karo button dabao!")
                    st.stop()

                embed_model, collection, chunks, bm25, groq_client = load_resources()
                top_chunks = hybrid_search(query, embed_model, collection, chunks, bm25)
                answer = get_answer(query, top_chunks, groq_client)

                st.markdown(answer)

                with st.expander("Sources dekho"):
                    seen = set()
                    for chunk in top_chunks:
                        key = f"{chunk['source']}_p{chunk['page']}"
                        if key not in seen:
                            seen.add(key)
                            st.markdown(f"- `{chunk['source']}` — page {chunk['page']}")

            except Exception as e:
                answer = f"Error aaya: {str(e)}"
                st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
