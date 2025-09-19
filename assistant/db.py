import psycopg2
from psycopg2.extras import execute_values

import logging
# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

try:
    from config import DB_CONFIG, SETTINGS
    from embeddings import generate_embedding
    from preprocess_text import hash_text
except Exception:
    from .config import DB_CONFIG, SETTINGS
    from .embeddings import generate_embedding
    from .preprocess_text import hash_text


def get_connection():
    """
    Returns a new connection to the PostgreSQL database using DB_CONFIG.
    Ensure DB_CONFIG has keys: host, port, dbname (database), user, password
    """
    return psycopg2.connect(**DB_CONFIG)


def create_table():
    """
    Creates a table for storing embeddings and associated text chunks.
    Uses vector column for storing 1536-dim embeddings (Postgres pgvector extension).
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(f'''
        -- CREATE EXTENSION IF NOT EXISTS vector;
        CREATE TABLE IF NOT EXISTS {SETTINGS.POSTGRES_TABLE} (
            id SERIAL PRIMARY KEY,
            section_content TEXT,
            type TEXT,
            vector VECTOR(768),           -- (1536) pgvector column adjust this based on your OpenAI model
            hash TEXT UNIQUE,             -- content hash for deduplication
            group_id INT,
            source_link TEXT,
            source_name TEXT,
            source_note TEXT,
            timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
        ''')
        conn.commit()
        logger.info(f"{SETTINGS.POSTGRES_TABLE} Table created if not exist!")


def insert_chunks(chunks, group_id: int):
    """
    Inserts a list of structured chunks into the database.
    - Generates embedding per chunk
    - Computes hash for deduplication
    - Uses execute_values for batch insert
    - Skips duplicates based on hash
    """
    with get_connection() as conn, conn.cursor() as cur:
        rows = []
        for chunk in chunks:
            # Combine section and content for embedding
            section_content = (chunk['section'] or "") + " " + chunk['content']
            # Generate vector embedding
            vec = generate_embedding([section_content])  # returns list[float]
            # Compute hash for deduplication
            h = hash_text(section_content)
            # Append row tuple
            rows.append((
                section_content,
                chunk['type'],
                vec,
                h,
                group_id,
                chunk['source_link'],
                chunk['source_name'],
                chunk['source_note'],
            ))
            
        
        # Batch insert using execute_values
        execute_values(cur, f'''
            INSERT INTO {SETTINGS.POSTGRES_TABLE} (section_content, type, vector, hash, group_id, source_link, source_name, source_note)
            VALUES %s
            ON CONFLICT (hash) DO NOTHING;
        ''', rows)
        conn.commit()
        logger.info(f"(Group {group_id}) In table: {SETTINGS.POSTGRES_TABLE} All Embedding/Vector data added!")