import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL

CSV_PATH = "data/processed/ag_news_sample.csv"
OUTPUT_PATH = "data/processed/ag_news_embeddings.npy"


def main():
    df = pd.read_csv(CSV_PATH)
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = df["full_text"].fillna("").tolist()
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=64,
        normalize_embeddings=True
    )

    np.save(OUTPUT_PATH, embeddings)
    print(f"Saved embeddings: {embeddings.shape} -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()