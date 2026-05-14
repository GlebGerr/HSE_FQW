from src.keyword_search import search_keyword
from src.vector_search import search_vector

TEST_QUERIES = [
    "global politics and international conflict",
    "football championship results",
    "stock market company profits",
    "new smartphone and software release",
    "business merger and acquisition",
    "oil prices and world economy",
    "mobile phone security software",
    "bank acquisition and merger deal",
    "sports team victory and match result",
    "international terrorism and war",
]


def main():
    for q in TEST_QUERIES:
        print("=" * 80)
        print(f"QUERY: {q}\n")

        print("KEYWORD SEARCH:")
        kw = search_keyword(q, limit=3)
        for item in kw:
            print(f"- [{item['category']}] {item['title']}")

        print("\nVECTOR SEARCH:")
        vec = search_vector(q, limit=3)
        for item in vec:
            print(f"- [{item['category']}] {item['title']} (score={item['score']:.4f})")
        print()


if __name__ == "__main__":
    main()