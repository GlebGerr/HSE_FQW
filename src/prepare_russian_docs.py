import re
import sys
import math
from pathlib import Path

import pandas as pd
from pypdf import PdfReader
from docx import Document

DEFAULT_CORPUS = "russian_docs"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


def get_corpus_paths(corpus_name: str):
    base_dir = Path("data") / corpus_name
    raw_dir = base_dir / "raw"
    processed_dir = base_dir / "processed"
    metadata_path = base_dir / "metadata.csv"
    output_path = processed_dir / "russian_chunks.csv"

    return base_dir, raw_dir, processed_dir, metadata_path, output_path


def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n".join(pages)


def read_docx(path: Path) -> str:
    doc = Document(str(path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def read_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return read_txt(path)
    if suffix == ".pdf":
        return read_pdf(path)
    if suffix == ".docx":
        return read_docx(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\xa0", " ")

    # Удаляем script/style
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", text)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)

    # Удаляем комментарии
    text = re.sub(r"(?is)<!--.*?-->", " ", text)

    # Удаляем HTML-теги
    text = re.sub(r"(?is)<[^>]+>", "\n", text)

    # Удаляем URL
    text = re.sub(r"https?://\S+", " ", text)

    # Нормализуем пробелы
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def remove_boilerplate_lines(text: str) -> str:
    lines = [line.strip() for line in text.split("\n")]
    cleaned = []

    boilerplate_exact = {
        "✖",
        "A",
        "А",
        "АБВ",
        "РУС",
        "EN",
        "Меню",
        "Найти",
        "здесь",
        "Важно",
        "Документы",
        "Студентам",
        "Преподавателям",
        "Научным сотрудникам",
        "Учебным офисам",
        "Департаментам",
        "О ВЫШКЕ",
        "ОБРАЗОВАНИЕ",
        "НАУКА",
    }

    boilerplate_patterns = [
        r"в старых версиях браузеров",
        r"мы используем файлы cookies",
        r"обычная версия сайта",
        r"версия для слабовидящих",
        r"расширенный поиск",
        r"национальный исследовательский университет",
        r"высшая школа экономики",
        r"справочник учебного процесса",
        r"дирекция основных образовательных программ",
        r"нашли\s+опечатку",
        r"выделите её",
        r"ctrl\+enter",
        r"сервис предназначен только",
        r"продолжая пользоваться сайтом",
        r"правила обработки персональных данных",
        r"корпоративные информационные системы",
        r"справочник сотрудника",
        r"справочник исследователя",
        r"оргструктура",
        r"обратная связь",
        r"цифровые сервисы",
        r"про деньги",
        r"перемещения студентов",
        r"учебный процесс",
        r"сессии и экзамены",
        r"сессии и гиа",
        r"выпуск из вышки",
        r"студентам с овз",
        r"контроль успеваемости",
        r"учебная нагрузка",
        r"трудоустройство",
        r"памятка преподавателя",
        r"локальные положения и регламенты",
        r"федеральные нормативные документы",
        r"дополнительное образование",
        r"единая платежная страница",
        r"фонд целевого капитала",
        r"противодействие коррупции",
        r"сведения об образовательной организации",
        r"людям с ограниченными возможностями",
        r"работа в вышке",
        r"центр развития карьеры",
        r"бизнес-инкубатор",
    ]

    for line in lines:
        if not line:
            continue

        normalized = re.sub(r"\s+", " ", line).strip()
        low = normalized.lower()

        if normalized in boilerplate_exact:
            continue

        if len(normalized) <= 2:
            continue

        if any(re.search(pattern, low) for pattern in boilerplate_patterns):
            continue

        if low.count("/") > 3:
            continue

        if "javascript" in low or "stylesheet" in low or "viewport" in low:
            continue

        if "favicon" in low or "canonical" in low or "gtm" in low:
            continue

        cleaned.append(normalized)

    deduped = []
    seen = set()

    for line in cleaned:
        key = line.lower()

        if key in seen:
            continue

        seen.add(key)
        deduped.append(line)

    return "\n".join(deduped).strip()


def is_low_quality_chunk(chunk: str) -> bool:
    low = chunk.lower().strip()

    if len(low) < 120:
        return True

    bad_patterns = [
        r"cookies",
        r"обычная версия сайта",
        r"расширенный поиск",
        r"версия для слабовидящих",
        r"ctrl\+enter",
        r"нашли\s+опечатку",
        r"национальный исследовательский университет",
        r"дирекция основных образовательных программ",
        r"справочник учебного процесса",
        r"о вышке",
        r"образование",
        r"наука",
        r"бизнес-инкубатор",
        r"центр развития карьеры",
        r"фонд целевого капитала",
        r"противодействие коррупции",
        r"сведения об образовательной организации",
    ]

    bad_hits = sum(1 for pattern in bad_patterns if re.search(pattern, low))

    # Если в чанке много служебной навигации — выбрасываем
    if bad_hits >= 2:
        return True

    # Очень много коротких строк — похоже на меню
    lines = [line.strip() for line in chunk.split("\n") if line.strip()]
    if lines:
        short_lines = sum(1 for line in lines if len(line) < 60)
        short_ratio = short_lines / len(lines)

        if short_ratio > 0.85 and len(lines) > 10:
            return True

    return False


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()

        # Пытаемся резать по ближайшей точке/переводу строки
        if end < text_len:
            cut_candidates = [
                chunk.rfind(". "),
                chunk.rfind("; "),
                chunk.rfind(": "),
                chunk.rfind(" "),
            ]
            best_cut = max(cut_candidates)
            if best_cut > int(chunk_size * 0.6):
                chunk = chunk[:best_cut + 1].strip()
                end = start + len(chunk)

        if chunk:
            chunks.append(chunk)

        if end >= text_len:
            break

        start = max(end - overlap, start + 1)

    return chunks


def main():
    corpus_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CORPUS

    base_dir, raw_dir, processed_dir, metadata_path, output_path = get_corpus_paths(corpus_name)

    print(f"[INFO] Using corpus: {corpus_name}")
    print(f"[INFO] Base dir: {base_dir}")

    processed_dir.mkdir(parents=True, exist_ok=True)

    try:
        metadata = pd.read_csv(metadata_path, encoding="utf-8")
    except UnicodeDecodeError:
        try:
            metadata = pd.read_csv(metadata_path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            metadata = pd.read_csv(metadata_path, encoding="cp1251")

    if "page_value" in metadata.columns:
        metadata = metadata[metadata["page_value"] == "keep"].copy()

    rows = []
    chunk_global_id = 1

    for _, meta in metadata.iterrows():
        filename = meta["filename"]
        file_path = raw_dir / filename

        if not file_path.exists():
            print(f"[WARN] File not found: {file_path}")
            continue

        print(f"[INFO] Reading: {filename}")
        raw_text = read_file(file_path)
        cleaned_text = clean_text(raw_text)
        cleaned_text = remove_boilerplate_lines(cleaned_text)

        chunks = chunk_text(cleaned_text)
        chunks = [c for c in chunks if not is_low_quality_chunk(c)]

        print(f"[INFO] Chunks created: {len(chunks)}")

        for chunk_order, chunk in enumerate(chunks, start=1):
            row = {
                "chunk_id": chunk_global_id,
                "document_id": int(meta["document_id"]),
                "document_title": meta["title"],
                "filename": meta["filename"],
                "source_url": meta["source_url"],
                "doc_type": meta["doc_type"],
                "topic": meta["topic"],
                "chunk_order": chunk_order,
                "chunk_text": chunk,
            }

            if "page_value" in meta.index:
                row["page_value"] = meta["page_value"]

            rows.append(row)
            chunk_global_id += 1

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"[DONE] Saved {len(df)} chunks to {output_path}")
    if not df.empty:
        print(df.head(3).to_string())


if __name__ == "__main__":
    main()