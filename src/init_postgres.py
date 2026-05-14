import pandas as pd
from sqlalchemy import create_engine, text
from src.config import DATABASE_URL

CSV_PATH = "data/processed/ag_news_sample.csv"


def main():
    engine = create_engine(DATABASE_URL)
    df = pd.read_csv(CSV_PATH)

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM news_articles"))

    df[["id", "title", "description", "category", "full_text"]].to_sql(
        "news_articles",
        engine,
        if_exists="append",
        index=False
    )

    with engine.begin() as conn:
        conn.execute(text("""
            UPDATE news_articles
            SET search_vector = to_tsvector('english', coalesce(title,'') || ' ' || coalesce(description,''));
        """))

    print(f"Inserted {len(df)} rows into PostgreSQL")


if __name__ == "__main__":
    main()