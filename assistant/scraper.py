from __future__ import annotations  # Forward type hints (e.g., Union[str, list[str]])
import os
import io  # For in-memory file streams
import json  # Read/write JSON
import time  # Sleep/backoff
from pathlib import Path  # Path manipulation
from typing import Union  # Type hint for single or list of URLs

import requests  # HTTP requests
from urllib.parse import urlparse  # URL parsing
from bs4 import BeautifulSoup  # Optional static HTML parsing

import pymupdf  # PyMuPDF for PDF text extraction

# Optional selenium import kept isolated for dynamic pages
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    HAVE_SELENIUM = True
except Exception:
    HAVE_SELENIUM = False

import logging
# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

try:
    from config import SETTINGS  # Settings: API_KEY, MODEL_EMBED, BASE_URL, etc.
except Exception:
    from .config import SETTINGS

# ------------------------
# Headers for HTTP requests
# ------------------------
# HEADERS = {"User-Agent": "rag-bot/1.0"}
HEADERS = {
    # "User-Agent": "Mozilla/5.0",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/117.0 Safari/537.36",
    "Accept": "application/pdf",  # Ensure we request PDFs
}


# ------------------------
# Selenium-based dynamic page scraping
# ------------------------
def fetch_links_selenium(
    url: str = SETTINGS.DATA_URL,
    start: int = 1,
    end: int = SETTINGS.DEFAULT_MAX_PAGES,
    pause_s: float = 1.0,
    save_json: bool = True,
    output_dir: str = "data",
    ) -> "set[str]":
    """
    Scrape .pdf links from paginated pages using Selenium.
    Requires Chrome & chromedriver in PATH.

    Fetch at first all links of relevant documents.
    Start is always page 1, this cannot change
    """
    if not HAVE_SELENIUM:
        raise RuntimeError(
            "Selenium not available. Install selenium & driver or use fetch_links_static()."
        )

    options = Options()
    # options.add_argument("--headless=new")
    options.add_argument("--headless")  # Run browser without GUI
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(options=options)
    driver.get(url)
    links: "set[str]" = set()

    for page in range(start, end + 1):
        try:
            input_field = WebDriverWait(driver, 10).until(
                # EC.presence_of_element_located((By.TAG_NAME, "body"))
                EC.presence_of_element_located((By.CSS_SELECTOR, 'input.k-textbox'))
            )
            input_field.clear()
            input_field.send_keys(str(page))
            time.sleep(SETTINGS.SE_SLOWMO_MS / 1000)  # Slow down to avoid server overload
            # naive: collect all .pdf anchors on page
            # links = driver.find_elements(By.XPATH, "//a[contains(@href, '.pdf')]")
            anchors = driver.find_elements(By.XPATH, "//a[contains(@href, '.pdf')]")
            for a in anchors:
                href = a.get_attribute("href")
                if href:
                    links.add(href)  # Keep only unique URLs
            logger.info(f"Loaded page: {page} found: {len(links)} .pdf in set.")
            # Attempt to click 'Next' button if exists
            try:
                next_btn = driver.find_element(By.CSS_SELECTOR, 'a[aria-label="Go to the next page"]')
                next_btn.click()
                time.sleep(pause_s)
            except Exception:
                break  # No next button; exit pagination loop
        except Exception:
            break  # Timeout or other Selenium errors; exit
    driver.quit()
    # Optionally save links to JSON
    if save_json and links:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        file_path = Path(output_dir) / 'pdfs_link_who.json'
        with open(file_path, 'wt', encoding='utf-8') as jsonf:
            json.dump({'urls': sorted(links)}, jsonf, indent=2, ensure_ascii=False)
    logger.info(f"Totally fetched/scraped {len(sorted(links))} .pdf links!")
    return sorted(links)


# ------------------------
# Extract text from a single PDF
# ------------------------
def fetch_pdf_text(url: str = SETTINGS.DATA_URL) -> str:
    """Download PDF and extract its text using PyMuPDF."""
    resp = requests.get(url, headers=HEADERS, timeout=SETTINGS.TIMEOUT_PDF_REQUEST)
    resp.raise_for_status()
    file_stream = io.BytesIO(resp.content)

    with pymupdf.open(stream=file_stream, filetype="pdf") as doc:
        # return "\n".join(page.get_text() for page in doc)  # Extract text page by page
        texts = []
        for page in doc:
            texts.append(page.get_text())
    return "\n".join(texts)


# ------------------------
# Extract filename from WHO-style URL
# ------------------------
def extract_filename_from_url(url: str) -> str:
    """Extract filename from a WHO-style URL."""
    parsed = urlparse(url)
    filename = Path(parsed.path).name
    # Remove query parameters like ?sequence=1
    return filename.split("?")[0] if "?" in filename else filename


# ------------------------
# Download PDFs given single or list of URLs
# ------------------------
def download_pdfs(
    urls: "Union[str, list[str]]",
    output_dir: str = "data",
    max_retries: int = 5,
    backoff_base: int = 2,
) -> "list[Path]":
    """
    Download PDFs, saving them with filename extracted from URL.
    Handles rate limiting (HTTP 429) with exponential backoff.

    Args:
        urls (str | List[str]): Single URL or list of URLs.
        output_dir (str): Directory where PDFs will be saved.
        max_retries (int): Maximum retries for failed requests.
        backoff_base (int): Exponential backoff base for rate limiting.

    Returns:
        List[Path]: List of file paths for successfully downloaded PDFs.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if urls is None:  # Load from default JSON if None
        try:
            with open(f'{output_dir}/pdf_links_who.json', 'rt') as f_in:
                urls = json.load(f_in)["urls"]
        except Exception:
            urls = []

    if isinstance(urls, str):
        urls = [urls]

    saved_files = []
    for i, url in enumerate(urls, start=1):
        filename = extract_filename_from_url(url)
        # filename = Path(output_dir) / f"file_{i}.pdf"
        file_path = Path(output_dir) / filename

        for attempt in range(max_retries):
            try:
                with requests.get(url, headers=HEADERS, timeout=30, stream=True) as r:
                    if r.status_code == 429:  # Too many requests, rate limited
                        wait = backoff_base ** attempt
                        print(f"[{url}] Rate limited. Retrying in {wait}s...")
                        time.sleep(wait)
                        continue

                    r.raise_for_status()
                    # Skip non-PDF responses
                    if not r.headers.get("Content-Type", "").startswith("application/pdf"):
                        print(f"[{url}] Skipped: not a PDF (got {r.headers.get('Content-Type')})")
                        break

                    # Write PDF to disk
                    with open(file_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)

                    print(f"[{url}] Saved to {file_path}")
                    saved_files.append(file_path)
                    break
            except Exception as e:
                print(f"[{url}] Error: {e}")
                if attempt < max_retries - 1:
                    wait = backoff_base ** attempt
                    print(f"Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"[{url}] Failed after {max_retries} attempts")

    return saved_files


# ------------------------
# Load PDF text from local file
# ------------------------
def load_pdf_text(
    file_path: str = "data/WHO-EURO-2025-6904-46670-80597-eng.pdf",  # Health needs assessment of the adult population in Ukraine Survey report April 2025
) -> str:
    """Extract text from a local PDF file."""
    with pymupdf.open(file_path, filetype="pdf") as doc:
        # return "\n".join(page.get_text() for page in doc)  # Extract text page by page
        texts = []
        for page in doc:
            texts.append(page.get_text())
    return "\n".join(texts)
