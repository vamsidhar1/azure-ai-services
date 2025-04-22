import fitz  # PyMuPDF for PDF text extraction
import faiss
import numpy as np
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Global Variables
chunks = []
index = None

def extract_text_from_pdf(pdf_path):
    """ Extracts text from a PDF file. """
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])
    return text

def chunk_text(text):
    """ Splits extracted text into manageable chunks. """
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)

def embed_text_openai(text_chunks):
    """ Uses OpenAI to embed text chunks. """
    embeddings = []
    for chunk in text_chunks:
        response = openai.Embedding.create(
            input=chunk,
            model="text-embedding-3-small"
        )
        embeddings.append(response["data"][0]["embedding"])
    return np.array(embeddings).astype("float32")

def build_faiss_index(embeddings):
    """ Creates a FAISS index for efficient retrieval. """
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def search_faiss(query, top_k=3):
    """ Searches FAISS for the most relevant chunks. """
    query_embedding = openai.Embedding.create(
        input=query,
        model="text-embedding-3-small"
    )["data"][0]["embedding"]
    
    query_vector = np.array([query_embedding], dtype="float32")
    distances, indices = index.search(query_vector, top_k)
    return [chunks[i] for i in indices[0]]

def ask_openai(question, context):
    """ Uses OpenAI GPT-4 to generate an answer based on context. """
    prompt = f"Context: {context}\n\nQuestion: {question}"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300
    )
    return response["choices"][0]["message"]["content"]

def process_pdf_and_index(file_path):
    """ Processes PDF and builds FAISS index. """
    global chunks, index
    text = extract_text_from_pdf(file_path)
    chunks = chunk_text(text)
    embeddings = embed_text_openai(chunks)
    index = build_faiss_index(embeddings)
    print(f"Processed {len(chunks)} chunks from PDF.")

def run_query(query):
    """ Runs a query against the indexed data. """
    if not chunks or index is None:
        return "No document has been processed yet!"
    
    top_chunks = search_faiss(query)
    context = " ".join(top_chunks)
    answer = ask_openai(query, context)
    return answer

# Example usage
if __name__ == "__main__":
    pdf_path = "example.pdf"  # Replace with your actual PDF file path
    process_pdf_and_index(pdf_path)
    
    while True:
        user_query = input("\nAsk a question (or type 'exit'): ")
        if user_query.lower() == "exit":
            break
        response = run_query(user_query)
        print(f"\nAnswer: {response}")
