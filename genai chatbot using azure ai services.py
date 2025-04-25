```python
"""
End-to-End Azure AI Search Pipeline
Steps:
1. Extract text from PDF
2. Chunk text
3. Create Azure Search index
4. Compute embeddings
5. Upload chunks + embeddings
6. Query: compute embedding & search
7. Handle low-confidence/no results
8. Query LLM with context
"""

import re
import uuid
import fitz  # PyMuPDF
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex, SimpleField, SearchableField, edm
from azure.search.documents import SearchClient
from azure.ai.openai import OpenAIClient

# --- Configuration ---
PDF_PATH = "document.pdf"
SEARCH_ENDPOINT = "https://<YOUR-SEARCH-SERVICE>.search.windows.net"
SEARCH_KEY = "<YOUR-ADMIN-KEY>"
INDEX_NAME = "documents-index"
OPENAI_ENDPOINT = "https://<YOUR-OPENAI-ENDPOINT>"
OPENAI_KEY = "<YOUR-API-KEY>"
EMBEDDING_MODEL = "text-embedding-ada-002"
LLM_MODEL = "gpt-4"
TOP_K = 5
CONFIDENCE_THRESHOLD = 0.7

# --- Step 1: Extract text from PDF ---
def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    texts = [page.get_text() for page in doc]
    return "\n".join(texts)

# --- Step 2: Chunk the text ---
def chunk_text(text: str, max_tokens: int = 300) -> list[str]:
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks, current, count = [], [], 0
    for sent in sentences:
        words = sent.split()
        if count + len(words) > max_tokens and current:
            chunks.append(" ".join(current))
            current, count = [], 0
        current.append(sent)
        count += len(words)
    if current:
        chunks.append(" ".join(current))
    return chunks

# --- Step 3: Create Azure Search index ---
def create_search_index():
    client = SearchIndexClient(endpoint=SEARCH_ENDPOINT, credential=AzureKeyCredential(SEARCH_KEY))
    fields = [
        SimpleField(name="id", type=edm.String, key=True),
        SearchableField(name="content", type=edm.String, analyzer_name="en.lucene"),
        SimpleField(name="source", type=edm.String, filterable=True),
        SimpleField(name="embedding", type=edm.Collection(edm.Single))
    ]
    idx = SearchIndex(name=INDEX_NAME, fields=fields)
    client.create_index(idx)
    print(f"Index '{INDEX_NAME}' created.")

# --- Step 4: Compute embeddings ---
openai_client = OpenAIClient(endpoint=OPENAI_ENDPOINT, credential=AzureKeyCredential(OPENAI_KEY))

def compute_embedding(text: str) -> list[float]:
    resp = openai_client.embeddings(model=EMBEDDING_MODEL, input=text)
    return resp['data'][0]['embedding']

# --- Step 5: Upload chunks + embeddings ---

def upload_chunks(chunks: list[str]):
    search_client = SearchClient(endpoint=SEARCH_ENDPOINT, index_name=INDEX_NAME, credential=AzureKeyCredential(SEARCH_KEY))
    docs = []
    for chunk in chunks:
        emb = compute_embedding(chunk)
        docs.append({
            "id": str(uuid.uuid4()),
            "content": chunk,
            "source": PDF_PATH,
            "embedding": emb
        })
    result = search_client.upload_documents(documents=docs)
    success = len([r for r in result if r.succeeded])
    print(f"Uploaded {success}/{len(docs)} chunks.")

# --- Step 6: Query Azure Search ---

def search_query(query: str):
    q_emb = compute_embedding(query)
    client = SearchClient(endpoint=SEARCH_ENDPOINT, index_name=INDEX_NAME, credential=AzureKeyCredential(SEARCH_KEY))
    return list(client.search(
        search_text="*",
        vector=q_emb,
        vector_field_name="embedding",
        top=TOP_K
    ))

# --- Step 7: Handle results confidence ---

def handle_results(results):
    if not results or results[0].score < CONFIDENCE_THRESHOLD:
        return None
    return results

# --- Step 8: Query LLM with context ---

def query_llm(query: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(context_chunks)
    prompt = f"Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {query}"  
    resp = openai_client.completion(model=LLM_MODEL, prompt=prompt, max_tokens=200)
    return resp['choices'][0]['text']

# --- Main Execution Flow ---
if __name__ == "__main__":
    print("Extracting text...")
    text = extract_text_from_pdf(PDF_PATH)

    print("Chunking text...")
    chunks = chunk_text(text)

    print("Creating search index...")
    create_search_index()

    print("Uploading chunks...")
    upload_chunks(chunks)

    # Interactive query loop
    while True:
        q = input("Enter your question (or 'exit' to quit): ")
        if q.lower() == 'exit':
            break
        print("Searching...")
        res = search_query(q)
        valid = handle_results(res)
        if not valid:
            print("No data found.")
            continue
        top_texts = [r['content'] for r in valid]
        print("Querying LLM...")
        ans = query_llm(q, top_texts)
        print("Answer:\n", ans)
```
