from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI
from agents import set_tracing_disabled, function_tool
import os
from dotenv import load_dotenv
import cohere
from qdrant_client import QdrantClient
from agents import enable_verbose_stdout_logging

enable_verbose_stdout_logging(True)

load_dotenv()
set_tracing_disabled(True)

gemini_api_key = os.getenv("AIzaSyC2SvAQiO2XTJJBdT-DrEentAkOuAbIJYg")
provider = AsyncOpenAI(
    api_key=gemini_api_key, 
    base_url="https://genrativelanguage.googleapis.com/v1beta/openai/"
)
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=provider,
)



# Initialize Cohere client
cohere_client = cohere.Client ("a65RloxIg0U8mDuIwb9zlHyRqDrUHm1zlKNjkJpj")

# Connect to Qdrant
qdrant = QdrantClient(
    url="https://cf0c2cf9-f719-40f0-a4a9-1d7f09c9c08c.us-east4-0.gcp.cloud.qdrant.io",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.Fyza-OXLPtAifvX76-wigNDZr8yMDzpSx1oR-g_rJVI"
)

def get_embedding(text):
    """Get embedding vector for cobere embed v3.0 model"""
    response = cohere_client.embed(
    model="embed-english-v3.0",
    input_type="search query",  # Use Search Query for Query Embeddings
    texts=[text],
)
    return response.embeddings[0]  # Return the first embedding

@function_tool
def retrieve(query):
    embedding = get_embedding(query)
    reults = qdrant.query_points(
        collection_name="physical-ai-book",
        query = embedding,
        limit=5
    )
    return [point.payload ["text"] for point in reults.points]


agent = Agent(
    name = "Assistant",
    instructions = """ You are an AI tutor for the Physical AI text Book.
    to answer user questions. first call the tools 'retrieve' with the user query.
    use only the returned content from 'retrieve' to construct your answer.
    if the answer is not found in the retrieved content, respond i don't know.
    """,
    model = model,
    tools = [retrieve]
)
result = Runner.run_sync(
    agent,
    input = "What is Physical AI?"
)
print("result.final_output")