"""
Ingest pdfs then convert embedding to save database.
"""

try:
    from config import SETTINGS  # Settings: API_KEY, MODEL_EMBED, BASE_URL, etc.
    from db import create_table, insert_chunks
    from scraper import fetch_links_selenium, fetch_pdf_text, extract_filename_from_url
    from preprocess_text import split_into_chunks
except Exception:
    from .config import SETTINGS
    from .db import create_table, insert_chunks
    from .scraper import fetch_links_selenium, fetch_pdf_text, extract_filename_from_url
    from .preprocess_text import split_into_chunks

import logging
# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def ingest(
    url: str = SETTINGS.DATA_URL,
    pages: int = SETTINGS.DEFAULT_MAX_PAGES,
    pause: int = 5,
):
    create_table()
    links = fetch_links_selenium(url=url, end=pages, pause_s=pause)
    for i, link in enumerate(links, start=1):
        text = fetch_pdf_text(link)
        chunks = split_into_chunks(text, source_link=link, source_name=extract_filename_from_url(link))
        insert_chunks(chunks, i)
        logger.info(f"Ingested {link} (Group {i})")


if __name__ == '__main__':
    ingest()