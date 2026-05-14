import sys
import numpy as np
import pandas as pd

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from src.config import QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME


DEFAULT_CORPUS = "russian_docs"


def get_paths(corpus_name: str):
    csv_path = f"data/{corpus_name}/processed/russian_chunks.csv"
    embeddings_path = f"data/{corpus_name}/processed/russian_chunks_embeddings.npy"
    return csv_path, embeddings_path


def get_collection_name(corpus_name: str):
    if corpus_name == "hse_site":
        return "hse_site_embeddings"
    return COLLECTION_NAME


def main():
    corpus_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CORPUS

    csv_path, embeddings_path = get_paths(corpus_name)
    collection_name = get_collection_name(corpus_name)

    print(f"[INFO] Using corpus: {corpus_name}")
    print(f"[INFO] CSV path: {csv_path}")
    print(f"[INFO] Embeddings path: {embeddings_path}")
    print(f"[INFO] Qdrant collection: {collection_name}")

    df = pd.read_csv(csv_path)
    embeddings = np.load(embeddings_path)

    if len(df) != len(embeddings):
        raise ValueError(
            f"Rows count and embeddings count mismatch: {len(df)} rows vs {len(embeddings)} embeddings"
        )

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    vector_size = embeddings.shape[1]

    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
        ),
    )

    batch_size = 128
    total = len(df)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)

        points = []

        for idx in range(start, end):
            row = df.iloc[idx]

            payload = {
                "chunk_id": int(row["chunk_id"]),
                "document_id": int(row["document_id"]),
                "document_title": row.get("document_title", ""),
                "filename": row.get("filename", ""),
                "source_url": row.get("source_url", ""),
                "doc_type": row.get("doc_type", ""),
                "topic": row.get("topic", ""),
                "chunk_order": int(row["chunk_order"]),
                "chunk_text": row.get("chunk_text", ""),
            }

            if "page_value" in df.columns:
                payload["page_value"] = row.get("page_value", "")

            points.append(
                PointStruct(
                    id=int(row["chunk_id"]),
                    vector=embeddings[idx].tolist(),
                    payload=payload,
                )
            )

        client.upsert(
            collection_name=collection_name,
            points=points,
        )

        print(f"[INFO] Batch {start // batch_size + 1}: indexed chunks {start + 1}-{end}")

    print(f"[DONE] Indexed {total} chunks into Qdrant collection {collection_name}")


if __name__ == "__main__":
    main()