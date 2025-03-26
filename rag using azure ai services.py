#pip install azure-ai-formrecognizer openai azure-search-documents
import openai
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import *

# Azure AI Document Intelligence (Form Recognizer)
FORM_RECOGNIZER_ENDPOINT = "YOUR_FORM_RECOGNIZER_ENDPOINT"
FORM_RECOGNIZER_KEY = "YOUR_FORM_RECOGNIZER_KEY"

# Azure OpenAI
openai.api_key = "YOUR_OPENAI_API_KEY"
OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"

# Azure AI Search
SEARCH_ENDPOINT = "YOUR_SEARCH_ENDPOINT"
SEARCH_KEY = "YOUR_SEARCH_ADMIN_KEY"
INDEX_NAME = "legal-doc-index"

#Extract Text from PDF (Azure AI Document Intelligence)

def extract_text_from_pdf(pdf_path):
    client = DocumentAnalysisClient(FORM_RECOGNIZER_ENDPOINT, AzureKeyCredential(FORM_RECOGNIZER_KEY))

    with open(pdf_path, "rb") as file:
        poller = client.begin_analyze_document("prebuilt-document", document=file)
        result = poller.result()

    extracted_data = []
    for page in result.pages:
        for paragraph in page.paragraphs:
            extracted_data.append(paragraph.content)
    
    return extracted_data  # List of text chunks


#Convert Text to Vector Embeddings (Azure OpenAI)

def generate_embeddings(text_list):
    response = openai.Embedding.create(
        input=text_list,
        model=OPENAI_EMBEDDING_MODEL
    )
    
    return [data["embedding"] for data in response["data"]]  # List of embeddings


#Create Azure AI Search Index




def create_search_index():
    index_client = SearchIndexClient(SEARCH_ENDPOINT, AzureKeyCredential(SEARCH_KEY))
    
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SearchField(name="contentVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), searchable=True, vector_search_configuration="default")
    ]
    
    index = SearchIndex(
        name=INDEX_NAME,
        fields=fields,
        vector_search=VectorSearch(
            algorithm_configurations=[
                HnswAlgorithmConfiguration(name="default", kind="hnsw", parameters={"m": 4, "efConstruction": 400})
            ]
        )
    )
    
    index_client.create_or_update_index(index)

#Upload Extracted Text & Embeddings to Azure AI Search

def upload_documents_to_search(text_chunks, embeddings):
    search_client = SearchClient(SEARCH_ENDPOINT, INDEX_NAME, AzureKeyCredential(SEARCH_KEY))
    
    documents = []
    for i, (text, embedding) in enumerate(zip(text_chunks, embeddings)):
        documents.append({
            "id": str(i),
            "content": text,
            "contentVector": embedding
        })

    search_client.upload_documents(documents)

#Perform Hybrid Search (Keyword + Vector)

def hybrid_search(query):
    search_client = SearchClient(SEARCH_ENDPOINT, INDEX_NAME, AzureKeyCredential(SEARCH_KEY))
    
    # Generate vector embedding for query
    query_embedding = generate_embeddings([query])[0]

    # Perform hybrid search
    search_results = search_client.search(
        search=query,  # Keyword search
        vectors=[VectorizedQuery(vector=query_embedding, k=3, fields="contentVector")],  # Vector search
        top=3  # Return top 3 results
    )

    return [doc for doc in search_results]

#Run the Full Pipeline

# 1️⃣ Extract text from PDF
pdf_text_chunks = extract_text_from_pdf("constitution_of_india.pdf")

# 2️⃣ Generate embeddings
text_embeddings = generate_embeddings(pdf_text_chunks)

# 3️⃣ Create Azure AI Search index (only needed once)
create_search_index()

# 4️⃣ Upload extracted text & embeddings to AI Search
upload_documents_to_search(pdf_text_chunks, text_embeddings)

# 5️⃣ Perform Hybrid Search
query = "How are new states formed in India?"
results = hybrid_search(query)

# 6️⃣ Display Results
for result in results:
    print(f"🔹 Matched Content: {result['content']}\n")

#Augment Search Results with LLM Explanation


from openai import ChatCompletion

def augment_search_results(query, search_results):
    # Combine search results into a context for LLM
    context = "\n\n".join([f"Article: {doc['content']}" for doc in search_results])

    # Prompt for the LLM
    prompt = f"""
    You are a legal expert. Based on the following legal text, provide a concise and clear explanation for the query: "{query}"

    Legal Documents:
    {context}

    Explanation:
    """

    # Call the LLM (e.g., GPT-4)
    response = ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are an expert in constitutional law."},
                  {"role": "user", "content": prompt}]
    )

    return response["choices"][0]["message"]["content"]

# Example Usage
query = "How are new states formed in India?"
search_results = hybrid_search(query)  # Retrieve AI Search results
llm_response = augment_search_results(query, search_results)

print("🔹 LLM Augmented Answer:\n", llm_response)
