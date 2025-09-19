from __future__ import annotations  # Enables forward references in type hints
import hashlib  # For generating SHA256 hash of text
import re  # For regex-based text parsing
from typing import Optional  # For optional type hints

# ------------------------
# NLTK imports with fallback
# ------------------------
try:    
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize  # Sentence & word tokenizers
    from nltk.corpus import stopwords  # Stopwords for potential future use
    nltk.data.path.append("./nltk_data")  # Append local folder for nltk data
    try:
        ## punkt → the pre-trained Punkt sentence tokenizer models
        ## Ensure Punkt sentence tokenizer is available (pre-trained model for sentence splitting)
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        ## Download 'punkt' and 'punkt_tab' if missing, store locally
        # nltk.download('punkt_tab', download_dir='./nltk_data')
        nltk.download(['punkt', 'punkt_tab'], download_dir="./nltk_data")
except Exception:
    sent_tokenize = None  # Fallback if nltk fails to load

try:
    from config import SETTINGS  # Settings: API_KEY, MODEL_EMBED, BASE_URL, etc.
except Exception:
    from .config import SETTINGS

# Regular expression patterns for headings and bullet points
_BULLET = re.compile(r"^\s*[-•]\s*(?P<bullet>.+)")  # r'^\s*[-•]\s*(?P<bullet>.+)' Match bullet points starting with '-' or '•'
_HEADING = re.compile(r"^(?P<heading>[A-Z][^\n:]{2,}):\s*$")  # r'^(?P<heading>[A-Z].+):$' Match capitalized headings ending with ':'
# For Unicode-heavy texts, consider using \u00A0 (non-breaking space) normalization too:
_WS = re.compile(r"\s+|\u00A0")  # Normalize/collapse whitespace + non-breaking space (NBSP) characters


# ------------------------
# Whitespace normalization
# ------------------------
def normalize_ws(text: str) -> str:
    """Normalize whitespace (collapse runs of spaces, tabs, newlines, NBSPs into one space)."""
    # return re.sub(r"\s+", " ", text).strip()
    return _WS.sub(" ", text).strip()


# ------------------------
# Sentence splitting with fallback
# ------------------------
def sentence_split_fallback(text: str) -> "list[str]":
    """
    Split text into sentences using NLTK if available,
    otherwise fall back to a simple regex-based splitter.
    """
    clean = normalize_ws(text)  # Normalize whitespace
    if sent_tokenize:
        return sent_tokenize(clean)  # Use NLTK sentence tokenizer
    # Lightweight Regex fallback: split on punctuation followed by + space + capital/number
    # (?<=[.!?]) → positive lookbehind: ensures the split happens after ., !, or ?.
    # \s+ → requires at least one space after the punctuation.
    # (?=[A-Z0-9]) → positive lookahead: ensures the next character is a capital letter or digit (common start of a new sentence).
    return re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", clean)


# ------------------------
# Paragraph splitting with overlap
# ------------------------
def split_large_paragraph(
    paragraph: str,
    max_tokens: int = SETTINGS.CHUNK_MAX_TOKENS,
    overlap: int = SETTINGS.CHUNK_OVERLAP_TOKENS,
) -> "list[str]":
    """
    Split a paragraph into overlapping chunks of at most `max_tokens` words.
    Sentences are preserved where possible.

    Args:
        paragraph: Input text block.
        max_tokens: Max words allowed in a chunk.
        overlap: Number of words to overlap between consecutive chunks.

    Returns:
        List of text chunks.
    """
    # filters out any empty strings or whitespace-only sentences.
    sents = [s for s in sentence_split_fallback(paragraph) if s.strip()]  # Clean sentences

    chunks: "list[str]" = []  # Accumulated chunks
    cur: "list[str]" = []  # Current chunk words
    words = 0  # Current chunk word count

    for sent in sents:
        w = len(sent.split())  # Count words in sentence
        # If adding this sentence would exceed the chunk size
        if words + w > max_tokens and cur:  # Exceeding max_tokens
            chunks.append(" ".join(cur).strip())  # Save current chunk
            # Keep last `overlap` words for continuity
            cur = " ".join(cur).split()[-overlap:]  # Keep overlap words
            words = len(cur)

        cur.append(sent)  # Add sentence to current chunk
        words += w

    if cur:
        chunk_text = " ".join(cur).strip()  # Final chunk
        if chunk_text:  # avoid empty chunk
            chunks.append(chunk_text)

    # Fallback: if a single sentence or chunk is too long, split by words
    final_chunks = []
    step = max(1, max_tokens - overlap)
    for ch in chunks:
        words = ch.split()
        if len(words) > max_tokens:  # Overlength chunk
            for i in range(0, len(words), step):
                final_chunks.append(" ".join(words[i:i + max_tokens]))
        else:
            final_chunks.append(ch)

    return final_chunks


# ------------------------
# Full text splitting into structured chunks
# ------------------------
def split_into_chunks(
    text: str,
    max_tokens: int = SETTINGS.CHUNK_MAX_TOKENS,
    overlap: int = SETTINGS.CHUNK_OVERLAP_TOKENS,
    source_link: str = "",
    source_name : str = "",
    source_note: str = "",
) -> "list[dict[str, Optional[str]]]":
    """
    Split text into structured chunks by heading, bullet points, and paragraphs.
    Paragraphs are further split into overlapping word chunks.

    Returns:
        List of dictionaries with keys: section, type, content
    """
    text = normalize_ws(text)  # Normalize all text whitespace
    # Strip whitespace and remove empty lines
    lines = [line.strip() for line in text.splitlines() if line.strip()]  # Non-empty lines

    # Initialize the final list of structured chunks
    chunks: "list[dict[str, Optional[str]]]" = []  # Final chunk list
    # Temporary buffer to accumulate lines belonging to the current paragraph
    cur: "list[str]" = []  # Current paragraph buffer
    # Current section (heading) name
    section: "str | None" = None  # Current section heading

    # ------------------------
    # Helper to flush paragraph buffer into structured chunks
    # ------------------------
    def flush_paragraph():
        """Convert accumulated paragraph lines into structured chunks and reset buffer."""
        nonlocal cur
        if not cur:
            return  # nothing to flush

        # Join lines into a single paragraph string and normalize whitespace
        para_text = normalize_ws(" ".join(cur))  # Join and normalize paragraph

        # Split the paragraph into smaller chunks based on max_tokens and overlap
        for sub in split_large_paragraph(para_text, max_tokens, overlap):
            chunks.append({
                "type": "paragraph",     # Mark type as Paragraph type
                "section": section,      # Associate chunk with current section
                "content": sub,          # Store chunk text
                "source_link": source_link,          # Store source_link
                "source_name": source_name,          # Store source_name
                "source_note": source_note,          # Store source_note
            })

        # Clear the current paragraph buffer
        cur = []  # Reset buffer

    # ------------------------
    # Process each line one by one
    # ------------------------
    # for line in map(str.strip, lines):
    for line in lines:
        # Check if the line is a heading (all caps or capitalized line ending with colon)
        m_h = _HEADING.match(line)
        if m_h:  # Heading detected
            flush_paragraph()  # Save any previous paragraph before starting new section
            # New heading detected
            section = m_h.group("heading")  # Update current section
            chunks.append({
                "type": "heading",
                "section": section,
                "content": line,  # Store heading line
                "source_link": source_link,          # Store source_link
                "source_name": source_name,          # Store source_name
                "source_note": source_note,          # Store source_note
            })
            continue  # Move to next line

        # Check if the line is a bullet point (starts with '-' or '•')
        m_b = _BULLET.match(line)
        if m_b:  # Bullet detected
            flush_paragraph()  # Save any previous paragraph
            chunks.append({
                "type": "bullet",
                "section": section,
                "content": m_b.group("bullet"),  # Store bullet text only
                "source_link": source_link,          # Store source_link
                "source_name": source_name,          # Store source_name
                "source_note": source_note,          # Store source_note
            })
            continue  # Move to next line

        # If line is neither heading nor bullet, it belongs to a paragraph
        cur.append(line)  # Accumulate paragraph lines

    # ------------------------
    # Flush any remaining paragraph lines after processing all lines
    # ------------------------
    flush_paragraph()  # Flush any remaining paragraph

    # Return the fully structured list of chunks
    return chunks


# ------------------------
# Utility function to hash text
# ------------------------
def hash_text(text: str) -> str:
    """Return SHA256 hash of input text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()