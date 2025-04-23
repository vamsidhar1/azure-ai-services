import os
from dotenv import load_dotenv

from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.openai import OpenAIClient
from azure.search.documents import SearchClient

# Load environment variables
load_dotenv()

# Environment vars
FORM_RECOGNIZER_ENDPOINT = os.getenv("FORM_RECOGNIZER_ENDPOINT")
FORM_RECOGNIZER_KEY = os.getenv("FORM_RECOGNIZER_KEY")

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_ADMIN_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")

# Initialize clients
form_client = DocumentAnalysisClient(FORM_RECOGNIZER_ENDPOINT, AzureKeyCredential(FORM_RECOGNIZER_KEY))
openai_client = OpenAIClient(AZURE_OPENAI_ENDPOINT, AzureKeyCredential(AZURE_OPENAI_KEY))
search_client = SearchClient(AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_INDEX_NAME, AzureKeyCredential(AZURE_SEARCH_ADMIN_KEY))

# Chunking function
def chunk_text(text, chunk_size=500, overlap=50):
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# Embedding helper
def get_embedding(text):
    response = openai_client.get_embeddings(
        deployment_id=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        input=[text]
    )
    return response.embeddings[0]

# PDF processing
def process_pdf(pdf_path):
    with open(pdf_path, "rb") as f:
        poller = form_client.begin_analyze_document("prebuilt-read", document=f)
        result = poller.result()

    extracted_text = "\n".join(page.content for page in result.pages)
    chunks = chunk_text(extracted_text)

    print(f"[INFO] Extracted and chunked {len(chunks)} pieces of text.")

    docs_to_index = []
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        docs_to_index.append({
            "id": str(i),
            "content": chunk,
            "embedding": embedding
        })

    result = search_client.upload_documents(documents=docs_to_index)
    print("[INFO] Indexed all chunks to Azure Cognitive Search.")
    return True

# Ask a question
def ask_question(query):
    embedding = get_embedding(query)
    vector = {"value": embedding, "fields": "embedding", "k": 3}
    results = search_client.search(search_text="", vector=vector)
    context = " ".join([doc["content"] for doc in results])

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer."},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
    ]

    response = openai_client.get_chat_completions(
        deployment_id=AZURE_OPENAI_CHAT_DEPLOYMENT,
        messages=messages
    )
    answer = response.choices[0].message.content
    print(f"🤖 {answer}\n")

# Main loop
if __name__ == "__main__":
    pdf_path = "example.pdf"
    if process_pdf(pdf_path):
        print("\n🧠 Ask me anything about the document! Type 'exit' to quit.\n")
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("👋 Goodbye!")
                break
            ask_question(user_input)
