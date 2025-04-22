import os
import fitz  # PyMuPDF for PDF text extraction
import requests
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import Vector
from azure.core.credentials import AzureKeyCredential

# Load environment variables for Azure
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX")
AZURE_FORM_RECOGNIZER_ENDPOINT = os.getenv("AZURE_FORM_RECOGNIZER_ENDPOINT")
AZURE_FORM_RECOGNIZER_KEY = os.getenv("AZURE_FORM_RECOGNIZER_KEY")

# Initialize Azure AI Search
search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX,
    credential=AzureKeyCredential(AZURE_SEARCH_KEY)
)

def extract_text_from_pdf(pdf_path):
    """Extract text using Azure AI Document Intelligence."""
    url = f"{AZURE_FORM_RECOGNIZER_ENDPOINT}/formrecognizer/documentModels/prebuilt-read:analyze?api-version=2023-07-31"
    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_FORM_RECOGNIZER_KEY,
        "Content-Type": "application/pdf"
    }
    
    with open(pdf_path, "rb") as f:
        response = requests.post(url, headers=headers, data=f.read())

    response_json = response.json()
    text = "\n".join([page["content"] for page in response_json["analyzeResult"]["pages"]])
    
    return text

def chunk_text(text):
    """Splits extracted text into manageable chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)

def get_embeddings(texts):
    """Get embeddings from Azure OpenAI Service."""
    url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/text-embedding-ada-002/embeddings?api-version=2023-05-15"
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_KEY
    }
    data = {"input": texts}
    response = requests.post(url, headers=headers, json=data)
    return [item["embedding"] for item in response.json()["data"]]

def index_chunks(chunks):
    """Stores chunk embeddings in Azure AI Search."""
    batch = []
    embeddings = get_embeddings(chunks)

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        batch.append({
            "id": str(i),
            "content": chunk,
            "vector": Vector(embedding)
        })
    
    search_client.upload_documents(documents=batch)

def search_chunks(query):
    """Searches Azure AI Search for relevant chunks."""
    query_embedding = get_embeddings([query])[0]
    
    results = search_client.search(
        search_text="",
        vectors=[Vector(value=query_embedding, k=3, fields="vector")]
    )

    return [result["content"] for result in results]

def ask_gpt4(query, context):
    """Uses Azure OpenAI GPT-4 Turbo to generate a response."""
    url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/gpt-4-turbo/chat/completions?api-version=2023-05-15"
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_KEY
    }
    data = {
        "messages": [
            {"role": "system", "content": "You are an AI assistant that answers questions based on provided context."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
        ],
        "max_tokens": 200
    }

    response = requests.post(url, headers=headers, json=data)
    return response.json()["choices"][0]["message"]["content"]

# Example function to process a PDF and index the chunks
def process_pdf(file_path):
    """Process the PDF, extract text, split into chunks, and index."""
    text = extract_text_from_pdf(file_path)
    chunks = chunk_text(text)
    index_chunks(chunks)
    return len(chunks)

# Example function to ask a question based on a query
def ask_question(query):
    """Search for relevant chunks and generate an answer using GPT-4."""
    top_chunks = search_chunks(query)
    context = " ".join(top_chunks)
    answer = ask_gpt4(query, context)
    return answer

# Example usage
if __name__ == "__main__":
    # Example: Process a PDF
    pdf_path = "your_pdf_file.pdf"
    total_chunks = process_pdf(pdf_path)
    print(f"PDF processed and indexed successfully! Total chunks: {total_chunks}")
    
    # Example: Ask a question
    question = "What is the main topic of the document?"
    answer = ask_question(question)
    print(f"Answer: {answer}")
