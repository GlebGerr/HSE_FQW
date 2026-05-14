from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from src.config import EMBEDDING_MODEL, QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME


model = SentenceTransformer(EMBEDDING_MODEL)
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def search_vector(query: str, limit: int = 5):
    query_vector = model.encode(query, normalize_embeddings=True).tolist()

    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=limit
    )

    return [
        {
            "id": r.id,
            "score": r.score,
            "title": r.payload.get("title"),
            "category": r.payload.get("category"),
            "description": r.payload.get("description")
        }
        for r in results
    ]


if __name__ == "__main__":
    results = search_vector("stock market technology", 5)
    for r in results:
        print(r["id"], r["category"], r["title"], r["score"])