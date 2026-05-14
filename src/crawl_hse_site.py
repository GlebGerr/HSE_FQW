import csv
import re
import time
import hashlib
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

RAW_DIR = Path("data/hse_site/raw")
LOG_DIR = Path("data/hse_site/crawl_logs")
METADATA_PATH = Path("data/hse_site/metadata.csv")

START_URLS = [
    "https://www.hse.ru/studyspravka/",
    "https://www.hse.ru/studyspravka/vkr",
    "https://www.hse.ru/studyspravka/practice",
    "https://www.hse.ru/studyspravka/academotpusk",
]

ALLOWED_PREFIXES = [
    "https://www.hse.ru/studyspravka/",
]

EXCLUDED_SLUG_HINTS = [
    "handbook",
    "studyspravka_std",
    "studyspravka_tch",
    "nauchsotrud",
    "academhead",
    "uchassistant",
    "adm",
    "serv_pps",
    "newprogram",
]

LOW_VALUE_SLUG_HINTS = [
    "loc",
    "normdokmon",
    "kontr",
    "syllabus",
    "new_regulations",
    "academnormy",
    "acmob",
    "ob_sluzeniem",
    "iga",
    "ai_guidelines",
    "ai_guidelines_pps",
    "instpaliat",
]

EXCLUDED_TITLE_HINTS = [
    "студентам — справочник учебного процесса",
    "преподавателям — справочник учебного процесса",
    "научным сотрудникам",
    "учебные офисы",
    "академические руководители",
    "учебные, цифровые ассистенты",
]

CONTENT_KEYWORDS = [
    "документы",
    "порядок",
    "необходимо",
    "студент",
    "заявление",
    "экзамен",
    "пересдач",
    "защита",
    "вкр",
    "гиа",
    "практик",
    "отпуск",
    "график",
    "апелля",
]

MAX_PAGES = 50
REQUEST_DELAY = 3.0

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; VKR-HSE-Crawler/1.0)"
}


def url_allowed(url: str) -> bool:
    return any(url.startswith(prefix) for prefix in ALLOWED_PREFIXES)


def url_should_skip(url: str) -> bool:
    lowered = url.lower()

    bad_parts = [
        "#",
        "?",
        "mailto:",
        "tel:",
        "/search/",
        "/ajax/",
        "/api/",
        "javascript:",
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".svg",
        ".zip",
        ".rar",
        ".mp4",
        ".mp3",
    ]

    return any(part in lowered for part in bad_parts)


def normalize_url(base_url: str, href: str) -> str:
    absolute = urljoin(base_url, href)
    parsed = urlparse(absolute)
    clean = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    return clean.rstrip("/")


def slugify_url(url: str) -> str:
    digest = hashlib.md5(url.encode("utf-8")).hexdigest()[:10]
    parsed = urlparse(url)
    path = parsed.path.strip("/").replace("/", "_")
    if not path:
        path = "home"
    path = re.sub(r"[^a-zA-Z0-9_а-яА-ЯёЁ-]+", "_", path)
    return f"{path}_{digest}.txt"


def looks_like_index_page(url: str, title: str, text: str) -> bool:
    lowered_url = url.lower()
    lowered_title = title.lower()
    lowered_text = text.lower()

    # Явные служебные / оглавительные страницы
    if any(hint in lowered_url for hint in EXCLUDED_SLUG_HINTS):
        return True

    if any(hint in lowered_title for hint in EXCLUDED_TITLE_HINTS):
        return True

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return True

    short_lines = sum(1 for line in lines if len(line) < 60)
    long_lines = sum(1 for line in lines if len(line) >= 120)

    short_ratio = short_lines / len(lines)

    # Очень много коротких строк и почти нет длинных абзацев — похоже на меню/каталог
    if short_ratio > 0.90 and long_lines < 5:
        return True

    # Проверяем наличие содержательных слов
    content_hits = sum(1 for kw in CONTENT_KEYWORDS if kw in lowered_text)

    if content_hits < 3 and long_lines < 8:
        return True

    return False


def classify_page_value(url: str, title: str, text: str) -> str:
    lowered_url = url.lower()
    lowered_title = title.lower()
    lowered_text = text.lower()

    if any(hint in lowered_url for hint in EXCLUDED_SLUG_HINTS):
        return "exclude"

    if any(hint in lowered_title for hint in EXCLUDED_TITLE_HINTS):
        return "exclude"

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "exclude"

    short_lines = sum(1 for line in lines if len(line) < 60)
    long_lines = sum(1 for line in lines if len(line) >= 120)

    short_ratio = short_lines / len(lines)
    content_hits = sum(1 for kw in CONTENT_KEYWORDS if kw in lowered_text)

    if short_ratio > 0.90 and long_lines < 5:
        return "exclude"

    if content_hits < 3 and long_lines < 8:
        return "exclude"

    if any(hint in lowered_url for hint in LOW_VALUE_SLUG_HINTS):
        return "low_value"

    return "keep"


def extract_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    title = soup.title.get_text(" ", strip=True) if soup.title else ""

    main_block = soup.find("main") or soup.find("article") or soup.body
    if main_block:
        text = main_block.get_text("\n", strip=True)
    else:
        text = soup.get_text("\n", strip=True)

    text = f"{title}\n\n{text}" if title else text
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


def detect_topic(url: str, title: str) -> str:
    u = url.lower()
    t = title.lower()

    if "academotpusk" in u or "академ" in t:
        return "academic_leave"
    if "vkr" in u or "вкр" in t or "предзащит" in t:
        return "vkr"
    if "practice" in u or "практик" in t:
        return "practice"
    if "faq" in u or "вопрос" in t:
        return "faq"
    return "documents"


def crawl():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    visited = set()
    queue = deque(START_URLS)
    rows = []
    doc_id = 1

    while queue and len(visited) < MAX_PAGES:
        url = queue.popleft()

        if url in visited:
            continue

        if not url_allowed(url) or url_should_skip(url):
            continue

        try:
            print(f"[INFO] Fetching: {url}")
            response = requests.get(url, headers=HEADERS, timeout=20)
            response.raise_for_status()
            html = response.text
        except Exception as e:
            print(f"[WARN] Failed to fetch {url}: {e}")
            continue

        visited.add(url)

        text = extract_text_from_html(html)

        soup = BeautifulSoup(html, "html.parser")
        title = soup.title.get_text(" ", strip=True) if soup.title else url
        topic = detect_topic(url, title)

        page_value = classify_page_value(url, title, text)

        if page_value == "exclude":
            print(f"[SKIP] Index page / low-value page: {url}")
        else:
            filename = slugify_url(url)
            file_path = RAW_DIR / filename
            file_path.write_text(text, encoding="utf-8")

            rows.append(
                {
                    "document_id": doc_id,
                    "title": title,
                    "filename": filename,
                    "source_url": url,
                    "doc_type": "web",
                    "topic": topic,
                    "page_value": page_value,
                }
            )
            doc_id += 1

        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            new_url = normalize_url(url, href)

            if new_url not in visited and url_allowed(new_url) and not url_should_skip(new_url):
                queue.append(new_url)

        time.sleep(REQUEST_DELAY)

    with open(METADATA_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["document_id", "title", "filename", "source_url", "doc_type", "topic", "page_value"]
        )
        writer.writeheader()
        writer.writerows(rows)

    visited_log = LOG_DIR / "visited_urls.txt"
    visited_log.write_text("\n".join(sorted(visited)), encoding="utf-8")

    print(f"[DONE] Crawled pages: {len(visited)}")
    print(f"[DONE] Metadata saved to: {METADATA_PATH}")


if __name__ == "__main__":
    crawl()