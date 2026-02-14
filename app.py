import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from pypdf import PdfReader

st.title("📚 Mini NotebookLM (Basic Version)")

# Load models once
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")
    return embed_model, qa_pipeline

embed_model, qa_pipeline = load_models()

# Extract text from PDF
def load_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

# Chunk text
def chunk_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Create FAISS index
def create_faiss_index(chunks):
    embeddings = embed_model.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index, chunks

# Retrieve relevant chunks
def retrieve(query, index, chunks, top_k=3):
    query_vector = embed_model.encode([query])
    distances, indices = index.search(np.array(query_vector), top_k)
    return [chunks[i] for i in indices[0]]

# Generate answer
def generate_answer(context, question):
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    result = qa_pipeline(prompt, max_length=200)
    return result[0]["generated_text"]

# UI
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with st.spinner("Processing document..."):
        text = load_pdf(uploaded_file)
        chunks = chunk_text(text)
        index, chunks = create_faiss_index(chunks)

    st.success("Document ready! Ask your question 👇")

    query = st.text_input("Ask a question")

    if query:
        with st.spinner("Thinking..."):
            retrieved_chunks = retrieve(query, index, chunks)
            context = " ".join(retrieved_chunks)
            answer = generate_answer(context, query)

        st.subheader("Answer:")
        st.write(answer)