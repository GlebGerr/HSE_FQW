from sqlalchemy import create_engine, text
from src.config import DATABASE_URL


def main():
    engine = create_engine(DATABASE_URL)

    sql = """
    UPDATE news_articles
    SET search_vector = to_tsvector(
        'english',
        coalesce(title, '') || ' ' || coalesce(description, '')
    );
    """

    with engine.begin() as conn:
        conn.execute(text(sql))

    print("search_vector updated")


if __name__ == "__main__":
    main()