import os
import uuid
from datetime import datetime

from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SemanticSettings,
    SemanticConfiguration,
    PrioritizedFields,
    VectorField,
    VectorSearch,
    VectorSearchAlgorithmConfiguration
)
from azure.ai.openai import OpenAIClient, ChatCompletionsOptions, ChatMessage
from azure.cosmos import CosmosClient, PartitionKey
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ─── Configuration ──────────────────────────────────────────────────────────────
# Form Recognizer
FORM_RECOGNIZER_ENDPOINT = os.getenv("AZURE_FORM_RECOGNIZER_ENDPOINT")
FORM_RECOGNIZER_KEY      = os.getenv("AZURE_FORM_RECOGNIZER_KEY")

# Azure AI Search
SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_KEY      = os.getenv("AZURE_SEARCH_KEY")
INDEX_NAME      = "documents-index"

# Azure OpenAI
OPENAI_ENDPOINT  = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_KEY       = os.getenv("AZURE_OPENAI_KEY")
EMBEDDING_MODEL  = "text-embedding-ada-002"
CHAT_MODEL       = "gpt-4"

# Cosmos DB (for conversation memory)
COSMOS_ENDPOINT = os.getenv("AZURE_COSMOS_ENDPOINT")
COSMOS_KEY      = os.getenv("AZURE_COSMOS_KEY")
COSMOS_DB       = "ChatBotDB"
COSMOS_CONTAINER= "Conversations"

# ─── Clients ────────────────────────────────────────────────────────────────────
# Document extraction
doc_client = DocumentAnalysisClient(
    endpoint=FORM_RECOGNIZER_ENDPOINT,
    credential=AzureKeyCredential(FORM_RECOGNIZER_KEY)
)

# Search index & vector config
index_client = SearchIndexClient(endpoint=SEARCH_ENDPOINT, credential=AzureKeyCredential(SEARCH_KEY))
search_client = SearchClient(endpoint=SEARCH_ENDPOINT, index_name=INDEX_NAME, credential=AzureKeyCredential(SEARCH_KEY))

# OpenAI
openai_client = OpenAIClient(endpoint=OPENAI_ENDPOINT, credential=AzureKeyCredential(OPENAI_KEY))

# Cosmos DB
cosmos_client = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)
cosmos_db = cosmos_client.create_database_if_not_exists(COSMOS_DB)
cosmos_container = cosmos_db.create_container_if_not_exists(
    id=COSMOS_CONTAINER,
    partition_key=PartitionKey(path="/conversation_id"),
    offer_throughput=400
)

# ─── 1. Ensure Search Index ─────────────────────────────────────────────────────
def ensure_search_index():
    try:
        idx = index_client.get_index(INDEX_NAME)
    except:
        fields = [
            SimpleField(name="id", type="Edm.String", key=True),
            SearchableField(name="content", type="Edm.String", analyzer_name="en.lucene"),
            VectorField(
                name="embedding",
                type="Collection(Edm.Single)",
                searchable=True,
                vector_search_configuration="default"
            )
        ]
        semantic_settings = SemanticSettings(
            configurations=[
                SemanticConfiguration(
                    name="default",
                    prioritized_fields=PrioritizedFields(content_fields=["content"])
                )
            ]
        )
        vs_config = VectorSearch(
            algorithm_configurations=[
                VectorSearchAlgorithmConfiguration(name="default", kind="hnsw")
            ]
        )
        index = SearchIndex(
            name=INDEX_NAME,
            fields=fields,
            semantic_settings=semantic_settings,
            vector_search=vs_config
        )
        index_client.create_index(index)

# ─── 2. Document Extraction ─────────────────────────────────────────────────────
def extract_text(file_path: str) -> str:
    poller = doc_client.begin_analyze_document("prebuilt-document", document=open(file_path, "rb"))
    result = poller.result()
    lines = [line.content for page in result.pages for line in page.lines]
    return "\n".join(lines)

# ─── 3. Chunking ─────────────────────────────────────────────────────────────────
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""]
)
def chunk_text(text: str):
    return text_splitter.split_text(text)

# ─── 4. Embedding ───────────────────────────────────────────────────────────────
def embed_text(text: str):
    resp = openai_client.get_embeddings(engine=EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding

# ─── 5. Indexing ─────────────────────────────────────────────────────────────────
def index_file(file_path: str):
    raw = extract_text(file_path)
    chunks = chunk_text(raw)
    actions = []
    for chunk in chunks:
        actions.append({
            "@search.action": "upload",
            "id": str(uuid.uuid4()),
            "content": chunk,
            "embedding": embed_text(chunk)
        })
    search_client.index_documents(actions)
    print(f"Indexed {len(actions)} chunks from {file_path}")

# ─── 6. Retrieval ────────────────────────────────────────────────────────────────
from azure.search.documents.models import QueryType, Vector
def retrieve_documents(query: str, top_k: int = 5):
    q_emb = embed_text(query)
    results = search_client.search(
        search_text="*",
        vector=Vector(value=q_emb, fields="embedding", k=top_k),
        query_type=QueryType.VECTORS
    )
    return [ {"content": hit["content"], "score": hit["@search.score"]} for hit in results ]

# ─── 7. Answer Generation ────────────────────────────────────────────────────────
def generate_answer(query: str, docs: list):
    sys = ChatMessage(role="system", content="You are an expert assistant. Use the context to answer or say you don't know.")
    context = "\n\n---\n\n".join(d["content"] for d in docs)
    usr = ChatMessage(role="user", content=f"Context:\n{context}\n\nQuestion: {query}")
    comp = openai_client.get_chat_completions(
        deployment_id=CHAT_MODEL,
        messages=[sys, usr],
        max_tokens=512,
        temperature=0.0
    )
    return comp.choices[0].message.content

# ─── 8. Conversation Storage ────────────────────────────────────────────────────
def store_conversation(user_msg: str, assistant_msg: str):
    cid = str(uuid.uuid4())
    item = {
        "conversation_id": cid,
        "user_message": user_msg,
        "assistant_message": assistant_msg,
        "timestamp": datetime.utcnow().isoformat()
    }
    cosmos_container.create_item(body=item)

# ─── 9. Full Query Handler ──────────────────────────────────────────────────────
def answer_query(query: str):
    docs = retrieve_documents(query)
    answer = generate_answer(query, docs)
    store_conversation(query, answer)
    return answer

# ─── Main Entrypoint ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ensure_search_index()

    # 1) Index all docs in sample_docs/
    folder = "sample_docs/"
    for fname in os.listdir(folder):
        if fname.lower().endswith((".pdf", ".png", ".jpg")):
            index_file(os.path.join(folder, fname))

    # 2) Interactively answer questions
    print("RAG chatbot ready—type your question (or 'exit'):")
    while True:
        q = input("You: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        resp = answer_query(q)
        print("Bot:", resp)
