import sys
import pandas as pd
from sqlalchemy import create_engine, text

from src.config import DATABASE_URL


DEFAULT_CORPUS = "russian_docs"


def get_paths(corpus_name: str):
    csv_path = f"data/{corpus_name}/processed/russian_chunks.csv"
    return csv_path


def get_table_name(corpus_name: str):
    if corpus_name == "hse_site":
        return "hse_document_chunks"
    return "document_chunks"


def main():
    corpus_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CORPUS

    csv_path = get_paths(corpus_name)
    table_name = get_table_name(corpus_name)

    print(f"[INFO] Using corpus: {corpus_name}")
    print(f"[INFO] CSV path: {csv_path}")
    print(f"[INFO] PostgreSQL table: {table_name}")

    df = pd.read_csv(csv_path)

    engine = create_engine(DATABASE_URL)

    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {table_name};"))

        conn.execute(text(f"""
            CREATE TABLE {table_name} (
                chunk_id INTEGER PRIMARY KEY,
                document_id INTEGER,
                document_title TEXT,
                filename TEXT,
                source_url TEXT,
                doc_type TEXT,
                topic TEXT,
                chunk_order INTEGER,
                chunk_text TEXT,
                page_value TEXT
            );
        """))

        for _, row in df.iterrows():
            conn.execute(
                text(f"""
                    INSERT INTO {table_name} (
                        chunk_id,
                        document_id,
                        document_title,
                        filename,
                        source_url,
                        doc_type,
                        topic,
                        chunk_order,
                        chunk_text,
                        page_value
                    )
                    VALUES (
                        :chunk_id,
                        :document_id,
                        :document_title,
                        :filename,
                        :source_url,
                        :doc_type,
                        :topic,
                        :chunk_order,
                        :chunk_text,
                        :page_value
                    );
                """),
                {
                    "chunk_id": int(row["chunk_id"]),
                    "document_id": int(row["document_id"]),
                    "document_title": row.get("document_title", ""),
                    "filename": row.get("filename", ""),
                    "source_url": row.get("source_url", ""),
                    "doc_type": row.get("doc_type", ""),
                    "topic": row.get("topic", ""),
                    "chunk_order": int(row["chunk_order"]),
                    "chunk_text": row.get("chunk_text", ""),
                    "page_value": row.get("page_value", None),
                }
            )

    print(f"[DONE] Inserted {len(df)} chunks into PostgreSQL table {table_name}")


if __name__ == "__main__":
    main()