import sys
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL


DEFAULT_CORPUS = "russian_docs"


def get_paths(corpus_name: str):
    csv_path = f"data/{corpus_name}/processed/russian_chunks.csv"
    embeddings_path = f"data/{corpus_name}/processed/russian_chunks_embeddings.npy"
    return csv_path, embeddings_path


def main():
    corpus_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CORPUS

    csv_path, embeddings_path = get_paths(corpus_name)

    print(f"[INFO] Using corpus: {corpus_name}")
    print(f"[INFO] CSV path: {csv_path}")
    print(f"[INFO] Embeddings path: {embeddings_path}")
    print(f"[INFO] Loading model: {EMBEDDING_MODEL}")

    df = pd.read_csv(csv_path)

    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = df["chunk_text"].fillna("").tolist()

    print(f"[INFO] Building embeddings for {len(texts)} chunks...")

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    np.save(embeddings_path, embeddings)

    print(f"[DONE] Saved embeddings {embeddings.shape} to {embeddings_path}")


if __name__ == "__main__":
    main()