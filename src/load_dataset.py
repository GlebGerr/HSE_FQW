from datasets import load_dataset
import pandas as pd
from pathlib import Path

LABELS = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech"
}

OUTPUT_PATH = Path("data/processed/ag_news_sample.csv")


def main(sample_size: int = 5000):
    dataset = load_dataset("ag_news", split="train")
    df = dataset.to_pandas()

    df["category"] = df["label"].map(LABELS)
    df = df.rename(columns={"text": "description"})
    df["title"] = df["description"].str.split(".").str[0].str.slice(0, 120)
    df["full_text"] = df["title"].fillna("") + ". " + df["description"].fillna("")
    df = df[["category", "title", "description", "full_text"]].copy()

    df = df.head(sample_size).reset_index(drop=True)
    df["id"] = df.index + 1

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"Saved {len(df)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()