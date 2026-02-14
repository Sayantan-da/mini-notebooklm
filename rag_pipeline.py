from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load text generation model
generator = pipeline("text2text-generation", model="google/flan-t5-base")


def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content
    return text


def split_text(text, chunk_size=500):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks


def create_embeddings(chunks):
    embeddings = embed_model.encode(chunks)
    return np.array(embeddings)


def store_embeddings(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def search(query, index, chunks, k=3):
    query_emb = embed_model.encode([query])
    distances, indices = index.search(np.array(query_emb), k)
    results = [chunks[i] for i in indices[0]]
    return results

def generate_answer(query, context):
    prompt = f"""
    Answer the question based only on the context.

    Context:
    {context}

    Question:
    {query}
    """
    result = generator(prompt, max_length=200)
    return result[0]["generated_text"]
