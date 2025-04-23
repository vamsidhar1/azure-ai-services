import os
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchableField,
    SemanticSettings, SemanticConfiguration, PrioritizedFields,
    VectorSearch, VectorSearchAlgorithmConfiguration, VectorField
)

# Load from .env (if using python-dotenv)
# from dotenv import load_dotenv
# load_dotenv()

endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
admin_key = os.getenv("AZURE_SEARCH_KEY")
index_name = "documents-index"

client = SearchIndexClient(endpoint=endpoint, credential=AzureKeyCredential(admin_key))

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

semantic = SemanticSettings(
    configurations=[
        SemanticConfiguration(
            name="default",
            prioritized_fields=PrioritizedFields(content_fields=["content"])
        )
    ]
)

vector_cfg = VectorSearch(
    algorithm_configurations=[
        VectorSearchAlgorithmConfiguration(name="default", kind="hnsw")
    ]
)

index = SearchIndex(
    name=index_name,
    fields=fields,
    semantic_settings=semantic,
    vector_search=vector_cfg
)

# Create or update the index
client.create_or_update_index(index)
print(f"Index '{index_name}' created/updated.")
