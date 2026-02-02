import requests
import xml.etree.ElementTree as ET
import trafilatura
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import cohere


# Configuration (can be overridden via environment variables)
SITEMAP_URL =("https://rag-chatbot-system.vercel.app/sitemap.xml")
COLLECTION_NAME = "physical_ai_book"

cohere_client = cohere.Client ("ep4o8CFtHGg8Cie1ycc0z3D7gZq3O5Gj7vzKntpM")
EMBED_MODEL = "embed-english-v3.0"

# Connect to Qdrant Cloud
qdrant_client = QdrantClient(
    url="https://7d712024-30f4-4d4c-8351-9867bf02fa60.sa-east-1-0.aws.cloud.qdrant.io:6333", 
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.Kal4_mrNjL0isWCrmGRH5hbqOQSgkI6PXjGjdgEJs-0",
)

# STEP 1: Extract URLs from Sitemap
def get_all_urls(sitemap_url):
    xml = requests.get(sitemap_url).text
    root = ET.fromstring(xml)

    urls = []
    for child in root:
        loc_tag = child.find("{http://www.sitemaps.org/schemas/sitemap/0.9}loc")
        if loc_tag is not None:
            urls.append(loc_tag.text)

    print("\nFound URLS:")
    for u in urls:
        print(" -", u)
    return urls

# STEP 2: Download Page + extract text
def extract_text_from_url(url):
    html = requests.get(url).text
    text = trafilatura.extract(html)

    if not text:
        print("[WARNING] No text found Extracted from :", url)
        return None

    return text

# STEP 3: Chunk text into smaller pieces
def chunk_text(text, max_chars=1200):
    chunks = []
    while len(text) > max_chars:
        split_pos = text[:max_chars].rfind(". ")
        if split_pos == -1:
            split_pos = max_chars
        chunks.append(text[:split_pos])
        text = text[split_pos:]
    chunks.append(text)
    return chunks

# STEP 4: Generate embeddings for each chunk
def embed(text):
    response = cohere_client.embed(
        model=EMBED_MODEL,
        input_type="search query", # Use Search Query for Query Embeddings
        texts=[text],
    )
    return response.embeddings[0] # Return the first embedding

# STEP 5: Store embeddings in Qdrant
def create_collection():
    print("\nCreating Collection in Qdrant...")
    qdrant_client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=1024,               # Cohere embed-english-v3.0 dimension
            distance=Distance.COSINE
        )
    )

def save_chunk_to_qdrant(chunk, chunk_id, url):
    vector = embed(chunk)
    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=chunk_id,
                vector=vector,
                payload={
                    "url": url,
                    "text": chunk,
                    "chunk_id": chunk_id
                }
            )
        ]
    )

# Main Ingestion Pipeline
def ingest_book():
    urls = get_all_urls(SITEMAP_URL)
    if not urls:
        print("No URLs found in sitemap.")
        return

    create_collection()
    global_id = 1

    for url in urls:
        print("\nProcessing URL:", url)
        text = extract_text_from_url(url)
        if not text:
            continue
        chunks = chunk_text(text)

        for ch in chunks:
            save_chunk_to_qdrant(ch, global_id, url)
            print(f"Saved chunk {global_id}")
            global_id += 1

    print("\nIngestion Completed!")
    print("Total chunks saved:", global_id - 1)

if __name__ == "__main__":
    ingest_book()