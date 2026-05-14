import re
from sqlalchemy import create_engine, text
from src.config import DATABASE_URL

STOP_WORDS = {
    "and", "or", "the", "a", "an", "of", "to", "in", "on", "for", "with", "by", "at", "is"
}


def normalize_terms(query: str):
    terms = re.findall(r"[A-Za-z]+", query.lower())
    terms = [t for t in terms if len(t) > 2 and t not in STOP_WORDS]
    return terms


def search_keyword(query: str, limit: int = 5):
    engine = create_engine(DATABASE_URL)
    terms = normalize_terms(query)

    if not terms:
        return []

    conditions = []
    params = {"limit": limit}

    for i, term in enumerate(terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(title ILIKE :{param_name} OR description ILIKE :{param_name})"
        )
        params[param_name] = f"%{term}%"

    score_expr = " + ".join(
        [
            f"CASE WHEN (title ILIKE :term_{i} OR description ILIKE :term_{i}) THEN 1 ELSE 0 END"
            for i in range(len(terms))
        ]
    )

    sql = text(f"""
        SELECT
            id,
            title,
            category,
            description,
            ({score_expr}) AS score
        FROM news_articles
        WHERE {" OR ".join(conditions)}
        ORDER BY score DESC, id ASC
        LIMIT :limit
    """)

    with engine.connect() as conn:
        rows = conn.execute(sql, params).mappings().all()

    return [dict(row) for row in rows]


if __name__ == "__main__":
    results = search_keyword("stock market company profits", 5)
    for r in results:
        print(r["id"], r["category"], r["title"], r["score"])