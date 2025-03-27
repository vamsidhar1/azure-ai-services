import os
from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn

# ---------- Azure Form Recognizer ------------
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential

# ---------- Text Chunking (Custom Function) ------------
def chunk_text(text, chunk_size=500, overlap=50):
    """
    Splits text into chunks of fixed size with overlap.
    (In a production solution you might use Cognitive Skills or more advanced logic.)
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# ---------- Azure OpenAI Client ------------
from azure.core.credentials import AzureKeyCredential
from azure.ai.openai import OpenAIClient

# ---------- Azure Cognitive Search ------------
from azure.search.documents import SearchClient
# The vector search parameter is sent as a JSON object via the 'vector' parameter.

# ---------- Initialize FastAPI ------------
app = FastAPI()

# ---------- Environment Variables ------------
FORM_RECOGNIZER_ENDPOINT = os.environ.get("FORM_RECOGNIZER_ENDPOINT")
FORM_RECOGNIZER_KEY = os.environ.get("FORM_RECOGNIZER_KEY")

AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.environ.get("AZURE_OPENAI_KEY")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4")

AZURE_SEARCH_ENDPOINT = os.environ.get("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_ADMIN_KEY = os.environ.get("AZURE_SEARCH_ADMIN_KEY")
AZURE_SEARCH_INDEX_NAME = os.environ.get("AZURE_SEARCH_INDEX_NAME", "document-index")

# ---------- Initialize Azure Clients ------------
form_recognizer_client = DocumentAnalysisClient(
    endpoint=FORM_RECOGNIZER_ENDPOINT, 
    credential=AzureKeyCredential(FORM_RECOGNIZER_KEY)
)

openai_client = OpenAIClient(
    endpoint=AZURE_OPENAI_ENDPOINT,
    credential=AzureKeyCredential(AZURE_OPENAI_KEY)
)

search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT, 
    index_name=AZURE_SEARCH_INDEX_NAME, 
    credential=AzureKeyCredential(AZURE_SEARCH_ADMIN_KEY)
)

# ---------- Global (for simplicity) ------------
# We store the list of text chunks locally after processing.
indexed_chunks = []  # List of chunk texts

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # Save the uploaded file temporarily.
        contents = await file.read()
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(contents)

        # Extract text using Azure Form Recognizer (prebuilt-read model)
        with open(temp_path, "rb") as f:
            poller = form_recognizer_client.begin_analyze_document("prebuilt-read", document=f)
            result = poller.result()
        os.remove(temp_path)  # Clean up the temporary file

        # Concatenate extracted text from all pages
        extracted_text = "\n".join(page.content for page in result.pages)

        # Chunk the text
        chunks = chunk_text(extracted_text)
        global indexed_chunks
        indexed_chunks = chunks  # Store locally for retrieval fallback

        # Define a helper function to get embeddings via Azure OpenAI
        def get_embedding(text):
            response = openai_client.get_embeddings(
                deployment_id=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
                input=[text]
            )
            return response.embeddings[0]

        # Prepare documents for Azure Cognitive Search
        documents_to_index = []
        for i, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            doc = {
                "id": str(i),
                "content": chunk,
                "embedding": embedding  # Must be a list of floats
            }
            documents_to_index.append(doc)

        # Upload documents to the Cognitive Search index (index should be pre-created)
        upload_result = search_client.upload_documents(documents=documents_to_index)
        if not upload_result[0].succeeded:
            raise Exception("Indexing failed for one or more documents.")
        return {"message": "PDF processed and indexed successfully!", "total_chunks": len(chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ask/")
async def ask(query: str):
    try:
        # Helper function to get embedding for query
        def get_embedding(text):
            response = openai_client.get_embeddings(
                deployment_id=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
                input=[text]
            )
            return response.embeddings[0]

        query_embedding = get_embedding(query)

        # Perform vector search in Azure Cognitive Search
        # (The API expects the vector parameter as a dict with keys: "value", "fields", and "k")
        vector = {
            "value": query_embedding,
            "fields": "embedding",
            "k": 3
        }
        results = search_client.search(search_text="", vector=vector)
        retrieved_chunks = []
        for result in results:
            retrieved_chunks.append(result["content"])
        context = " ".join(retrieved_chunks)

        # Generate answer using Azure OpenAI Chat endpoint
        messages = [
            {"role": "system", "content": "You are an AI assistant that answers questions based on the provided context."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
        ]
        chat_response = openai_client.get_chat_completions(
            deployment_id=AZURE_OPENAI_CHAT_DEPLOYMENT,
            messages=messages
        )
        answer = chat_response.choices[0].message.content
        return {"query": query, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
