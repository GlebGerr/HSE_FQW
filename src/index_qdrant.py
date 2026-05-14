import math
import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from src.config import QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME

CSV_PATH = "data/processed/ag_news_sample.csv"
EMB_PATH = "data/processed/ag_news_embeddings.npy"
BATCH_SIZE = 200


def build_points(df: pd.DataFrame, embeddings: np.ndarray, start_idx: int, end_idx: int):
    points = []

    batch_df = df.iloc[start_idx:end_idx]
    for _, row in batch_df.iterrows():
        idx = int(row["id"]) - 1
        points.append(
            PointStruct(
                id=int(row["id"]),
                vector=embeddings[idx].tolist(),
                payload={
                    "title": str(row["title"]),
                    "description": str(row["description"]),
                    "category": str(row["category"]),
                },
            )
        )

    return points


def main():
    df = pd.read_csv(CSV_PATH)
    embeddings = np.load(EMB_PATH)

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    vector_size = embeddings.shape[1]

    # Если коллекция уже есть — удаляем и создаём заново
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(collection_name=COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    total_rows = len(df)
    total_batches = math.ceil(total_rows / BATCH_SIZE)

    for batch_num in range(total_batches):
        start_idx = batch_num * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, total_rows)

        points = build_points(df, embeddings, start_idx, end_idx)

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )

        print(
            f"Batch {batch_num + 1}/{total_batches}: "
            f"indexed rows {start_idx + 1}-{end_idx}"
        )

    print(f"Indexed {total_rows} vectors into Qdrant")


if __name__ == "__main__":
    main()