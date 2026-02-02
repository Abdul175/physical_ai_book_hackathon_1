import cohere
from qdrant_client import QdrantClient

# Initialize Cohere client
cohere_client = cohere.Client ("a65RloxIg0U8mDuIwb9zlHyRqDrUHm1zlKNjkJpj")

# Connect to Qdrant
qdrant = QdrantClient(
    url="https://cc400596-72e6-40e3-8bde-2c6a07ec191c.europe-west3-0.gcp.cloud.qdrant.io:6333", 
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.rMVjAO-0k9xxGEPCvnY7_TRRu94phcC9r7X8fKXg1xU",
)

def get_embedding(text):
    """Get embedding vector for cobere embed v3.0 model"""
    response = cohere_client.embed(
    model="embed-english-v3.0",
    input_type="search query",  # Use Search Query for Query Embeddings
    texts=[text],
)
    return response.embeddings[0]  # Return the first embedding

def retrieve(query):
    embedding = get_embedding(query)
    reults = qdrant.query_points(
        collection_name="physical-ai-book",
        query = embedding,
        limit=5
    )
    return [point.payload ["text"] for point in reults.points]
print(retrieve ("What data do you have?"))


