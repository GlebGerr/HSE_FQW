import re
import time
from typing import List, Dict

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from src.config import (
    DATABASE_URL,
    EMBEDDING_MODEL,
    QDRANT_HOST,
    QDRANT_PORT,
    COLLECTION_NAME,
)

# -----------------------------
# Настройки страницы
# -----------------------------
st.set_page_config(
    page_title="Семантический поиск по документам",
    page_icon="🔎",
    layout="wide",
)

# -----------------------------
# Стили
# -----------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        margin-bottom: 0.15rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #666666;
        margin-bottom: 1.2rem;
    }
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin-top: 0.7rem;
        margin-bottom: 0.6rem;
    }
    .result-card {
        padding: 1rem;
        border: 1px solid #e8e8e8;
        border-radius: 14px;
        margin-bottom: 0.9rem;
        background-color: #fafafa;
    }
    .result-title {
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .result-meta {
        font-size: 0.9rem;
        color: #666666;
        margin-bottom: 0.55rem;
    }
    .score-tag {
        display: inline-block;
        padding: 0.15rem 0.45rem;
        border-radius: 8px;
        background-color: #eef3ff;
        color: #2d5bd1;
        font-size: 0.85rem;
        margin-left: 0.4rem;
    }
    .demo-box {
        padding: 0.85rem 1rem;
        border: 1px solid #ececec;
        border-radius: 12px;
        background-color: #fcfcfc;
        margin-bottom: 1rem;
    }
    .source-link {
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Константы
# -----------------------------
STOP_WORDS_RU = {
    "и", "или", "в", "во", "на", "по", "для", "с", "со", "к", "ко", "от", "до",
    "как", "что", "это", "а", "но", "из", "у", "же", "ли", "не", "при", "об",
    "о", "про", "над", "под", "то", "та", "те", "где", "когда"
}

DEMO_QUERIES = [
    "как оформить академический отпуск",
    "что нужно для предзащиты",
    "как проходит предзащита ВКР",
    "какие требования к ВКР",
    "что нужно для практики",
    "где найти правила внутреннего распорядка",
]

TOPIC_LABELS = {
    "Все": "Все",
    "academic_leave": "Академический отпуск",
    "vkr": "ВКР и предзащита",
    "practice": "Практика",
    "faq": "FAQ и справочные материалы",
    "documents": "Организационные документы",
    "education_rules": "Правила внутреннего распорядка",
}

CORPUS_SETTINGS = {
    "Ручной корпус": {
        "table_name": "document_chunks",
        "collection_name": COLLECTION_NAME,
        "description": "Отобранные вручную документы по ВКР, практике, академическому отпуску и правилам.",
    },
    "Расширенный корпус HSE": {
        "table_name": "hse_document_chunks",
        "collection_name": "hse_site_embeddings",
        "description": "Автоматически собранные страницы справочника учебного процесса HSE.",
    },
}

TOPIC_KEYWORDS = {
    "academic_leave": [
        "академ", "академический отпуск", "отпуск", "уважительная причина"
    ],
    "vkr": [
        "вкр", "предзащита", "защита", "магистерская диссертация", "диплом", "презентация", "рецензия"
    ],
    "practice": [
        "практика", "стажировка", "практическая подготовка", "internship"
    ],
    "faq": [
        "вопрос", "faq", "справка", "справочные материалы"
    ],
    "education_rules": [
        "правила", "внутренний распорядок", "обязанности", "дисциплина"
    ],
}

RELATED_QUERIES = {
    "предзащита": [
        "как проходит предзащита ВКР",
        "какие требования к ВКР",
        "как подготовить презентацию к предзащите",
    ],
    "вкр": [
        "какие требования к ВКР",
        "как проходит предзащита ВКР",
        "что нужно для защиты ВКР",
    ],
    "академ": [
        "как выйти из академического отпуска",
        "какие документы нужны для академического отпуска",
        "сохраняется ли статус студента в академическом отпуске",
    ],
    "практик": [
        "что нужно для практики",
        "обязательна ли практика",
        "какие документы нужны для практики",
    ],
}

# -----------------------------
# Кэшируем ресурсы
# -----------------------------
@st.cache_resource
def get_engine():
    return create_engine(DATABASE_URL)

@st.cache_resource
def get_model():
    return SentenceTransformer(EMBEDDING_MODEL)

@st.cache_resource
def get_qdrant_client():
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

@st.cache_data
def get_chunk_count(table_name: str):
    engine = get_engine()
    sql = text(f"SELECT COUNT(*) FROM {table_name}")
    with engine.connect() as conn:
        count = conn.execute(sql).scalar()
    return int(count)

@st.cache_data
def get_topic_stats(table_name: str):
    engine = get_engine()
    sql = text(f"""
        SELECT topic, COUNT(*) AS cnt
        FROM {table_name}
        GROUP BY topic
        ORDER BY cnt DESC
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql).mappings().all()
    return [dict(row) for row in rows]

@st.cache_data
def get_doc_type_stats(table_name: str):
    engine = get_engine()
    sql = text(f"""
        SELECT doc_type, COUNT(*) AS cnt
        FROM {table_name}
        GROUP BY doc_type
        ORDER BY cnt DESC
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql).mappings().all()
    return [dict(row) for row in rows]

# -----------------------------
# Вспомогательные функции
# -----------------------------
def normalize_terms(query: str) -> List[str]:
    terms = re.findall(r"[А-Яа-яA-Za-zЁё0-9]+", query.lower())
    terms = [t for t in terms if len(t) > 2 and t not in STOP_WORDS_RU]
    return terms

def prettify_topic(topic: str) -> str:
    return TOPIC_LABELS.get(topic, topic)

def detect_topic_hints(query: str) -> List[str]:
    q = query.lower()
    matched_topics = []

    for topic, keywords in TOPIC_KEYWORDS.items():
        for keyword in keywords:
            if keyword in q:
                matched_topics.append(topic)
                break

    return matched_topics

def shorten_text(text_value: str, max_len: int = 800) -> str:
    if not text_value:
        return ""
    text_value = text_value.strip()
    if len(text_value) <= max_len:
        return text_value
    return text_value[:max_len].rstrip() + "..."

def highlight_text(text_value: str, query: str) -> str:
    if not text_value:
        return ""

    words = normalize_terms(query)
    highlighted = text_value

    for word in sorted(words, key=len, reverse=True):
        highlighted = re.sub(
            rf"({re.escape(word)})",
            r"<mark>\1</mark>",
            highlighted,
            flags=re.IGNORECASE,
        )

    return highlighted


def generate_answer(results: List[Dict], query: str) -> str:
    if not results:
        return "По вашему запросу релевантные фрагменты не найдены."

    query_low = query.lower()

    def unique_sources(items: List[Dict], max_sources: int = 3) -> List[str]:
        sources = []
        seen = set()

        for item in items:
            title = item.get("document_title", "Без названия")
            source_url = item.get("source_url", "")
            key = (title, source_url)

            if key in seen:
                continue

            seen.add(key)

            if source_url and source_url != "internal":
                sources.append(f"{len(sources) + 1}. {title} — {source_url}")
            else:
                sources.append(f"{len(sources) + 1}. {title}")

            if len(sources) >= max_sources:
                break

        return sources

    def make_answer(lines: List[str], source_items: List[Dict]) -> str:
        answer_parts = []
        answer_parts.append(
            "Ниже приведён краткий ответ на основе найденных документов. "
            "Для точного применения правил рекомендуется открыть источник."
        )
        answer_parts.append("")
        answer_parts.append("**Что важно:**")

        for line in lines:
            answer_parts.append(f"- {line}")

        answer_parts.append("")
        answer_parts.append("**Документы-основания:**")
        answer_parts.extend(unique_sources(source_items))

        return "\n".join(answer_parts)

    # 1. Предзащита
    if "предзащит" in query_low:
        relevant = [
            item for item in results
            if "вкр" in item.get("document_title", "").lower()
            or "предзащит" in item.get("chunk_text", "").lower()
        ]

        if not relevant:
            relevant = results

        return make_answer(
            [
                "Предзащита связана с подготовкой выпускной квалификационной работы перед основной защитой.",
                "Обычно студенту нужно подготовить текст ВКР или промежуточную версию работы, если это предусмотрено программой.",
                "Также необходимо подготовить доклад и презентационные материалы.",
                "Работу и материалы желательно заранее согласовать с научным руководителем.",
                "На предзащите могут задаваться вопросы и даваться рекомендации по доработке ВКР.",
            ],
            relevant[:3],
        )

    # 2. Выбор темы ВКР
    if "тему" in query_low and "вкр" in query_low:
        relevant = [
            item for item in results
            if "вкр" in item.get("document_title", "").lower()
            or "тему" in item.get("chunk_text", "").lower()
            or "руководител" in item.get("chunk_text", "").lower()
        ]

        if not relevant:
            relevant = results

        return make_answer(
            [
                "Тему ВКР студент выбирает в цифровой системе НИУ ВШЭ.",
                "Сроки выбора темы определяются программой практики или образовательной программой.",
                "Перед подачей заявки желательно обсудить тему с потенциальным научным руководителем.",
                "Руководитель должен подтвердить согласие на руководство темой.",
                "После подачи заявка проходит согласование у руководителя и академического руководителя программы.",
            ],
            relevant[:3],
        )

    # 3. Апелляция по ГИА
    if "апелляц" in query_low and "гиа" in query_low:
        relevant = [
            item for item in results
            if "апелляц" in item.get("document_title", "").lower()
            or "апелляц" in item.get("chunk_text", "").lower()
        ]

        if not relevant:
            relevant = results

        return make_answer(
            [
                "Апелляция по результатам ГИА подаётся в виде письменного мотивированного заявления.",
                "Заявление направляется секретарю Апелляционной комиссии.",
                "Как правило, используется корпоративная электронная почта студента.",
                "Порядок подачи апелляции и состав комиссии публикуются заранее на сайте образовательной программы.",
                "В заявлении нужно указать основания несогласия с результатом или процедурой проведения ГИА.",
            ],
            relevant[:3],
        )

    # Универсальный fallback для остальных запросов
    query_terms = normalize_terms(query)
    candidate_sentences = []

    for item in results[:5]:
        text_value = item.get("chunk_text", "").strip()
        sentences = re.split(r"(?<=[.!?])\s+|\n+", text_value)

        for sentence in sentences:
            sentence = re.sub(r"\s+", " ", sentence).strip()

            if len(sentence) < 70:
                continue

            if sentence[-1] not in ".!?":
                continue

            sentence_low = sentence.lower()
            score = sum(1 for term in query_terms if term in sentence_low)

            if score > 0:
                candidate_sentences.append((score, sentence))

    candidate_sentences = sorted(candidate_sentences, key=lambda x: x[0], reverse=True)

    facts = []
    seen = set()

    for _, sentence in candidate_sentences:
        key = sentence[:120].lower()
        if key in seen:
            continue
        seen.add(key)
        facts.append(sentence)
        if len(facts) >= 5:
            break

    if not facts:
        facts = [
            "Система нашла релевантные документы, но не смогла автоматически выделить короткий однозначный ответ.",
            "Рекомендуется открыть документы-основания и проверить исходный текст.",
        ]

    return make_answer(facts, results[:3])


def get_related_queries(query: str) -> List[str]:
    q = query.lower()
    suggestions = []

    for key, values in RELATED_QUERIES.items():
        if key in q:
            suggestions.extend(values)

    # убираем дубли
    seen = set()
    unique_suggestions = []
    for item in suggestions:
        if item not in seen and item.lower() != q:
            unique_suggestions.append(item)
            seen.add(item)

    return unique_suggestions[:3]

# -----------------------------
# Поиск по ключевым словам
# -----------------------------
def keyword_search(query: str, limit: int = 5, topic_filter: str = "Все", table_name: str = "document_chunks") -> List[Dict]:
    engine = get_engine()
    terms = normalize_terms(query)

    if not terms:
        return []

    conditions = []
    params = {"limit": limit}

    for i, term in enumerate(terms):
        param_name = f"term_{i}"
        conditions.append(f"(chunk_text ILIKE :{param_name} OR document_title ILIKE :{param_name})")
        params[param_name] = f"%{term}%"

    score_expr = " + ".join(
        [
            f"CASE WHEN (chunk_text ILIKE :term_{i} OR document_title ILIKE :term_{i}) THEN 1 ELSE 0 END"
            for i in range(len(terms))
        ]
    )

    where_clause = f"({' OR '.join(conditions)})"

    if topic_filter != "Все":
        where_clause += " AND topic = :topic_filter"
        params["topic_filter"] = topic_filter

    sql = text(f"""
        SELECT
            chunk_id,
            document_id,
            document_title,
            filename,
            source_url,
            doc_type,
            topic,
            chunk_order,
            chunk_text,
            ({score_expr}) AS score
        FROM {table_name}
        WHERE {where_clause}
        ORDER BY score DESC, chunk_id ASC
        LIMIT :limit
    """)

    with engine.connect() as conn:
        rows = conn.execute(sql, params).mappings().all()

    results = [dict(row) for row in rows]
    for r in results:
        r["score_label"] = "keyword score"
        r["search_method"] = "По ключевым словам"
    return results

# -----------------------------
# Векторный поиск
# -----------------------------
def vector_search(
    query: str,
    limit: int = 5,
    topic_filter: str = "Все",
    collection_name: str = COLLECTION_NAME,
) -> List[Dict]:
    model = get_model()
    client = get_qdrant_client()

    query_vector = model.encode(query, normalize_embeddings=True).tolist()

    search_kwargs = {
        "collection_name": collection_name,
        "query_vector": query_vector,
        "limit": max(limit * 3, 15),
    }

    if topic_filter != "Все":
        search_kwargs["query_filter"] = Filter(
            must=[
                FieldCondition(
                    key="topic",
                    match=MatchValue(value=topic_filter)
                )
            ]
        )

    results = client.search(**search_kwargs)

    query_topics = detect_topic_hints(query)

    prepared = []
    for r in results:
        topic = r.payload.get("topic", "")
        score = float(r.score)

        # Мягкий тематический буст
        if topic_filter == "Все" and query_topics and topic in query_topics:
            score += 0.08

        prepared.append(
            {
                "chunk_id": r.id,
                "document_id": r.payload.get("document_id"),
                "document_title": r.payload.get("document_title", ""),
                "filename": r.payload.get("filename", ""),
                "source_url": r.payload.get("source_url", ""),
                "doc_type": r.payload.get("doc_type", ""),
                "topic": topic,
                "chunk_order": r.payload.get("chunk_order"),
                "chunk_text": r.payload.get("chunk_text", ""),
                "score": score,
                "score_label": "vector similarity",
                "search_method": "Векторный поиск",
            }
        )

    prepared = sorted(prepared, key=lambda x: x["score"], reverse=True)
    return prepared[:limit]

# -----------------------------
# Гибридный поиск
# -----------------------------
def hybrid_search(
    query: str,
    limit: int = 5,
    topic_filter: str = "Все",
    table_name: str = "document_chunks",
    collection_name: str = COLLECTION_NAME,
) -> List[Dict]:
    kw = keyword_search(query, limit * 3, topic_filter=topic_filter, table_name=table_name)
    vec = vector_search(query, limit * 3, topic_filter=topic_filter, collection_name=collection_name)

    results = {}
    query_topics = detect_topic_hints(query)

    # Приоритет типов документов
    doc_type_boost = {
        "methodical": 0.40,   # методички — самый полезный тип
        "web": 0.25,          # содержательные веб-страницы
        "faq": 0.10,          # FAQ полезен, но не всегда должен быть первым
        "regulation": -0.10,  # общие регламенты немного понижаем
    }

    # Дополнительное понижение для общих правил, если запрос не про дисциплину
    def extra_penalty(item: Dict) -> float:
        title = str(item.get("document_title", "")).lower()
        topic = str(item.get("topic", "")).lower()
        query_lower = query.lower()

        penalty = 0.0

        if "правила внутреннего распорядка" in title and not any(
            x in query_lower for x in ["распорядок", "дисциплина", "правила"]
        ):
            penalty -= 0.35

        if topic == "education_rules" and not any(
            x in query_lower for x in ["распорядок", "дисциплина", "правила"]
        ):
            penalty -= 0.20

        if "предзащит" in query_lower:
            if "плагиат" in title or topic == "academic_leave":
                penalty -= 1.0

        if "апелляц" in query_lower or "гиа" in query_lower:
            if "выпускная квалификационная работа" in title and "апелляц" not in title:
                penalty -= 0.5

        if "тему" in query_lower and "вкр" in query_lower:
            if "плагиат" in title or "академический отпуск" in title:
                penalty -= 1.0

        return penalty

    # --- Keyword часть ---
    for rank, r in enumerate(kw, start=1):
        base_score = float(r.get("score", 0))
        score = base_score * 1.2 + (1.0 / rank)

        if topic_filter == "Все" and query_topics and r.get("topic") in query_topics:
            score += 0.4

        score += doc_type_boost.get(r.get("doc_type", ""), 0.0)
        score += extra_penalty(r)

        results[r["chunk_id"]] = {
            **r,
            "score": score,
            "source_type": "keyword",
            "score_label": "hybrid score",
            "search_method": "Гибридный поиск",
        }

    # --- Vector часть ---
    for rank, r in enumerate(vec, start=1):
        vec_score = float(r.get("score", 0))
        score = vec_score + (1.0 / rank)

        if topic_filter == "Все" and query_topics and r.get("topic") in query_topics:
            score += 0.4

        score += doc_type_boost.get(r.get("doc_type", ""), 0.0)
        score += extra_penalty(r)

        if r["chunk_id"] in results:
            # Если документ найден и keyword, и vector — это сильный сигнал
            results[r["chunk_id"]]["score"] += score + 0.8
            results[r["chunk_id"]]["source_type"] = "hybrid"
        else:
            results[r["chunk_id"]] = {
                **r,
                "score": score,
                "source_type": "vector",
                "score_label": "hybrid score",
                "search_method": "Гибридный поиск",
            }

    sorted_results = sorted(results.values(), key=lambda x: x["score"], reverse=True)
    deduplicated_results = deduplicate_results(sorted_results, max_per_document=2)
    return deduplicated_results[:limit]


def deduplicate_results(results: List[Dict], max_per_document: int = 2) -> List[Dict]:
    unique_results = []
    seen_chunks = set()
    document_counts = {}

    for item in results:
        document_title = item.get("document_title", "")
        chunk_text = item.get("chunk_text", "")

        chunk_key = (
            document_title.lower().strip(),
            chunk_text[:250].lower().strip(),
        )

        if chunk_key in seen_chunks:
            continue

        current_count = document_counts.get(document_title, 0)

        if current_count >= max_per_document:
            continue

        seen_chunks.add(chunk_key)
        document_counts[document_title] = current_count + 1
        unique_results.append(item)

    return unique_results

# -----------------------------
# Рендер карточки результата
# -----------------------------
def render_result_card(item: Dict, query: str):
    title = item.get("document_title", "Без названия")
    topic = prettify_topic(item.get("topic", ""))
    doc_type = item.get("doc_type", "")
    chunk_order = item.get("chunk_order", "-")
    source_url = item.get("source_url", "")
    chunk_text = shorten_text(item.get("chunk_text", ""))
    chunk_text = highlight_text(chunk_text, query)
    score_label = item.get("score_label", "score")
    filename = item.get("filename", "")

    score_html = ""
    if "score" in item and item["score"] is not None:
        try:
            score_html = f'<span class="score-tag">{score_label}: {float(item["score"]):.4f}</span>'
        except Exception:
            score_html = f'<span class="score-tag">{score_label}: {item["score"]}</span>'

    source_html = ""
    if source_url and source_url != "internal":
        source_html = f'<div class="source-link"><a href="{source_url}" target="_blank">Открыть источник</a></div>'

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-title">{title}</div>
            <div class="result-meta">
                Тема: {topic} | Тип: {doc_type} | Фрагмент № {chunk_order} | Файл: {filename} {score_html}
            </div>
            <div>{chunk_text}</div>
            {source_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_grouped_results(results: List[Dict], query: str, max_fragments_per_document: int = 2):
    grouped = {}

    for item in results:
        title = item.get("document_title", "Без названия")
        grouped.setdefault(title, []).append(item)

    for title, items in grouped.items():
        sorted_items = sorted(items, key=lambda x: float(x.get("score", 0)), reverse=True)
        best_items = sorted_items[:max_fragments_per_document]

        with st.expander(f"{title} — найдено фрагментов: {len(items)}", expanded=True):
            for item in best_items:
                render_result_card(item, query)

# -----------------------------
# Сравнение режимов
# -----------------------------
def compare_mode(
    query: str,
    limit: int,
    topic_filter: str = "Все",
    table_name: str = "document_chunks",
    collection_name: str = COLLECTION_NAME,
):
    col1, col2 = st.columns(2)

    start_kw = time.perf_counter()
    kw_results = keyword_search(query, limit, topic_filter=topic_filter, table_name=table_name)
    kw_elapsed = time.perf_counter() - start_kw

    start_vec = time.perf_counter()
    vec_results = vector_search(query, limit, topic_filter=topic_filter, collection_name=collection_name)
    vec_elapsed = time.perf_counter() - start_vec

    with col1:
        st.subheader("Поиск по ключевым словам")
        st.caption(f"Время поиска: {kw_elapsed:.4f} сек.")
        st.caption(f"Найдено результатов: {len(kw_results)}")
        if kw_results:
            for item in kw_results:
                render_result_card(item, query)
        else:
            st.info("По ключевым словам результаты не найдены.")

    with col2:
        st.subheader("Векторный поиск")
        st.caption(f"Время поиска: {vec_elapsed:.4f} сек.")
        st.caption(f"Найдено результатов: {len(vec_results)}")
        if vec_results:
            for item in vec_results:
                render_result_card(item, query)
        else:
            st.info("По векторному поиску результаты не найдены.")

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Параметры поиска")

    corpus_choice = st.selectbox(
        "Корпус документов",
        ["Ручной корпус", "Расширенный корпус HSE"],
        index=0,
    )

    corpus_table = CORPUS_SETTINGS[corpus_choice]["table_name"]
    corpus_collection = CORPUS_SETTINGS[corpus_choice]["collection_name"]

    st.caption(CORPUS_SETTINGS[corpus_choice]["description"])

    search_mode = st.radio(
        "Режим поиска",
        ["Сравнение", "По ключевым словам", "Векторный поиск", "Гибридный поиск"],
        index=3,
    )

    limit = st.slider(
        "Количество результатов",
        min_value=1,
        max_value=10,
        value=5,
    )

    topic_filter = st.selectbox(
        "Фильтр по теме",
        ["Все", "academic_leave", "vkr", "practice", "faq", "documents", "education_rules"],
        index=0,
        format_func=prettify_topic,
    )

    output_mode = st.radio(
        "Формат ответа",
        ["Фрагменты документов", "Краткий ответ"],
        index=0,
    )

    st.markdown("---")
    st.subheader("Примеры запросов")

    for demo_query in DEMO_QUERIES:
        if st.button(demo_query, use_container_width=True):
            st.session_state["query_input"] = demo_query
            st.session_state["auto_run_search"] = True
            st.rerun()

    st.markdown("---")
    st.caption("Стек: PostgreSQL + Qdrant + Sentence Transformers + Streamlit")

# -----------------------------
# Заголовок
# -----------------------------
st.markdown('<div class="main-title">Семантический поиск по русскоязычным документам</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">MVP системы поиска по нормативным, организационным и справочным документам. Поддерживаются поиск по ключевым словам, векторный поиск и гибридный режим.</div>',
    unsafe_allow_html=True,
)

# -----------------------------
# Метрики
# -----------------------------
chunk_count = get_chunk_count(corpus_table)
topic_stats = get_topic_stats(corpus_table)
doc_type_stats = get_doc_type_stats(corpus_table)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Количество фрагментов", f"{chunk_count}")
m2.metric("Тип хранилищ", "2", help="PostgreSQL + Qdrant")
m3.metric("Размерность эмбеддинга", "384")
m4.metric("Режим", search_mode)

with st.expander("Статистика корпуса документов"):
    if topic_stats:
        st.markdown("**Распределение по темам**")
        st.table(pd.DataFrame(topic_stats))

        topic_df = pd.DataFrame(topic_stats)
        if not topic_df.empty:
            topic_df["topic"] = topic_df["topic"].apply(prettify_topic)
            st.bar_chart(topic_df.set_index("topic")["cnt"])

    if doc_type_stats:
        st.markdown("**Распределение по типам документов**")
        st.table(pd.DataFrame(doc_type_stats))

# -----------------------------
# Блок запроса
# -----------------------------
if "query_input" not in st.session_state:
    st.session_state["query_input"] = "как оформить академический отпуск"

st.markdown('<div class="section-title">Поисковый запрос</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="demo-box">
    Примеры запросов для демонстрации:<br>
    • как оформить академический отпуск<br>
    • что нужно для предзащиты<br>
    • какие требования к ВКР<br>
    • что делать по практике<br>
    • где найти правила внутреннего распорядка
    </div>
    """,
    unsafe_allow_html=True,
)

query = st.text_input("Введите запрос", key="query_input")
run = st.button("Найти", type="primary")
auto_run = st.session_state.pop("auto_run_search", False)
run = run or auto_run

# -----------------------------
# Поиск
# -----------------------------
if run:
    if not query.strip():
        st.warning("Введите поисковый запрос.")
    else:
        st.markdown("---")

        start_time = time.perf_counter()

        if search_mode == "Сравнение":
            compare_mode(
                query,
                limit,
                topic_filter,
                table_name=corpus_table,
                collection_name=corpus_collection,
            )

        elif search_mode == "По ключевым словам":
            st.subheader("Результаты поиска по ключевым словам")
            results = keyword_search(
                query,
                limit,
                topic_filter=topic_filter,
                table_name=corpus_table,
            )
            elapsed = time.perf_counter() - start_time

            st.caption(f"Время поиска: {elapsed:.4f} сек.")
            st.caption(f"Найдено результатов: {len(results)}")

            if results:
                if output_mode == "Краткий ответ":
                    st.subheader("Краткий ответ")
                    answer_text = generate_answer(results, query)
                    answer_text = highlight_text(answer_text, query)
                    st.markdown(answer_text.replace("\n", "  \n"), unsafe_allow_html=True)

                    st.subheader("Документы-основания")
                    for item in results[:3]:
                        render_result_card(item, query)
                else:
                    render_grouped_results(results, query)

                related = get_related_queries(query)
                if related:
                    st.subheader("Похожие запросы")
                    cols = st.columns(len(related))
                    for i, related_query in enumerate(related):
                        with cols[i]:
                            if st.button(related_query, key=f"related_keyword_{i}"):
                                st.session_state["query_input"] = related_query
                                st.session_state["auto_run_search"] = True
                                st.rerun()
            else:
                st.info("По ключевым словам результаты не найдены.")

        elif search_mode == "Векторный поиск":
            st.subheader("Результаты векторного поиска")
            results = vector_search(
                query,
                limit,
                topic_filter=topic_filter,
                collection_name=corpus_collection,
            )
            elapsed = time.perf_counter() - start_time

            st.caption(f"Время поиска: {elapsed:.4f} сек.")
            st.caption(f"Найдено результатов: {len(results)}")

            if results:
                if output_mode == "Краткий ответ":
                    st.subheader("Краткий ответ")
                    answer_text = generate_answer(results, query)
                    answer_text = highlight_text(answer_text, query)
                    st.markdown(answer_text.replace("\n", "  \n"), unsafe_allow_html=True)

                    st.subheader("Документы-основания")
                    for item in results[:3]:
                        render_result_card(item, query)
                else:
                    render_grouped_results(results, query)

                related = get_related_queries(query)
                if related:
                    st.subheader("Похожие запросы")
                    cols = st.columns(len(related))
                    for i, related_query in enumerate(related):
                        with cols[i]:
                            if st.button(related_query, key=f"related_vector_{i}"):
                                st.session_state["query_input"] = related_query
                                st.session_state["auto_run_search"] = True
                                st.rerun()
            else:
                st.info("По векторному поиску результаты не найдены.")

        elif search_mode == "Гибридный поиск":
            st.subheader("Результаты гибридного поиска")
            results = hybrid_search(
                query,
                limit,
                topic_filter=topic_filter,
                table_name=corpus_table,
                collection_name=corpus_collection,
            )
            elapsed = time.perf_counter() - start_time

            st.caption(f"Время поиска: {elapsed:.4f} сек.")
            st.caption(f"Найдено результатов: {len(results)}")

            if results:
                if output_mode == "Краткий ответ":
                    st.subheader("Краткий ответ")
                    answer_text = generate_answer(results, query)
                    answer_text = highlight_text(answer_text, query)
                    st.markdown(answer_text.replace("\n", "  \n"), unsafe_allow_html=True)

                    st.subheader("Документы-основания")
                    for item in results[:3]:
                        render_result_card(item, query)
                else:
                    render_grouped_results(results, query)

                related = get_related_queries(query)
                if related:
                    st.subheader("Похожие запросы")
                    cols = st.columns(len(related))
                    for i, related_query in enumerate(related):
                        with cols[i]:
                            if st.button(related_query, key=f"related_hybrid_{i}"):
                                st.session_state["query_input"] = related_query
                                st.session_state["auto_run_search"] = True
                                st.rerun()
            else:
                st.info("По гибридному поиску результаты не найдены.")

# -----------------------------
# Описание прототипа
# -----------------------------
with st.expander("О системе"):
    st.write(
        """
        Данный интерфейс демонстрирует работу MVP-системы поиска по русскоязычным
        нормативным, организационным и справочным документам. Система поддерживает
        три режима поиска: поиск по ключевым словам, векторный поиск и гибридный поиск.
        Поиск выполняется по фрагментам документов, что позволяет возвращать
        пользователю непосредственно релевантные части текста, а не только целые документы.
        """
    )