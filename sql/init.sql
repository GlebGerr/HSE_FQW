CREATE TABLE IF NOT EXISTS news_articles (
    id BIGINT PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    category TEXT NOT NULL,
    full_text TEXT NOT NULL,
    search_vector tsvector
);

CREATE INDEX IF NOT EXISTS idx_news_search_vector
ON news_articles
USING GIN(search_vector);