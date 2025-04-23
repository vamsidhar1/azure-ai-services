import os
import uuid
from datetime import datetime

from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.search.documents import SearchClient
from azure.search.documents.models import Vector, QueryType
from azure.ai.openai import OpenAIClient, ChatMessage
from azure.cosmos import CosmosClient, PartitionKey

# ─── Config from .env ───────────────────────────────────────────────────────────
FORM_ENDPOINT = os.getenv("AZURE_FORM_RECOGNIZER_ENDPOINT")
FORM_KEY      = os.getenv("AZURE_FORM_RECOGNIZER_KEY")
SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_KEY      = os.getenv("AZURE_SEARCH_KEY")
OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_KEY      = os.getenv("AZURE_OPENAI_KEY")
COSMOS_ENDPOINT = os.getenv("AZURE_COSMOS_ENDPOINT")
COSMOS_KEY      = os.getenv("AZURE_COSMOS_KEY")

INDEX_NAME      = "documents-index"
EMBED_DEPLOY    = "text-embedding-ada-002"
CHAT_DEPLOYMENT = "gpt-4"
COSMOS_DB       = "ChatBotDB"
COSMOS_CONTAINER= "Conversations"

# ─── Clients ─────────────────────────────────────────────────────────────────────
doc_client = DocumentAnalysisClient(FORM_ENDPOINT, AzureKeyCredential(FORM_KEY))
search_client = SearchClient(SEARCH_ENDPOINT, INDEX_NAME, AzureKeyCredential(SEARCH_KEY))
openai_client = OpenAIClient(OPENAI_ENDPOINT, AzureKeyCredential(OPENAI_KEY))

cosmos = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)
db     = cosmos.create_database_if_not_exists(COSMOS_DB)
container = db.create_container_if_not_exists(
    id=COSMOS_CONTAINER,
    partition_key=PartitionKey(path="/conversation_id"),
    offer_throughput=400
)

# ─── Helpers ─────────────────────────────────────────────────────────────────────
def extract_text(file_path):
    poller = doc_client.begin_analyze_document("prebuilt-document", open(file_path, "rb"))
    result = poller.result()
    return "\n".join(line.content for page in result.pages for line in page.lines)

def chunk_text(text, max_tokens=800, overlap=100):
    tokens = text.split()
    chunks, i = [], 0
    while i < len(tokens):
        j = i + max_tokens
        chunks.append(" ".join(tokens[i:j]))
        i = j - overlap
    return chunks

def embed(text):
    resp = openai_client.get_embeddings(engine=EMBED_DEPLOY, input=text)
    return resp.data[0].embedding

def index_file(path):
    text = extract_text(path)
    for chunk in chunk_text(text):
        doc = {"id": str(uuid.uuid4()), "content": chunk, "embedding": embed(chunk)}
        search_client.upload_documents(documents=[doc])
    print(f"Indexed chunks from {os.path.basename(path)}")

def retrieve(query, k=5):
    q_emb = embed(query)
    results = search_client.search(
        search_text="*",
        vector=Vector(value=q_emb, fields="embedding", k=k),
        query_type=QueryType.VECTORS
    )
    return [hit["content"] for hit in results]

def generate(query, docs):
    system = ChatMessage(role="system", content=(
        "You are an expert assistant. Use the provided context to answer the question."
    ))
    context = "\n\n---\n\n".join(docs)
    user    = ChatMessage(role="user", content=f"Context:\n{context}\n\nQuestion: {query}")
    resp    = openai_client.get_chat_completions(
        deployment_id=CHAT_DEPLOYMENT,
        messages=[system, user],
        max_tokens=512,
        temperature=0.0
    )
    return resp.choices[0].message.content

def store_conv(user_msg, bot_msg):
    container.create_item({
        "conversation_id": str(uuid.uuid4()),
        "user_message": user_msg,
        "assistant_message": bot_msg,
        "timestamp": datetime.utcnow().isoformat()
    })

# ─── Main ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Index new docs
    for f in os.listdir("sample_docs"):
        if f.lower().endswith((".pdf", ".png", ".jpg")):
            index_file(os.path.join("sample_docs", f))

    print("Chatbot ready. Type 'exit' to quit.")
    while True:
        q = input("You: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        docs   = retrieve(q)
        answer = generate(q, docs)
        store_conv(q, answer)
        print("Bot:", answer)
