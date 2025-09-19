from collections import defaultdict

import psycopg2

import numpy as np
from scipy.spatial.distance import cosine
import faiss

try:
    from config import SETTINGS, DB_CONFIG  # Settings: API_KEY, MODEL_EMBED, BASE_URL, etc.
except Exception:
    from .config import SETTINGS, DB_CONFIG


# ---------------------------
# Vector utilities
# ---------------------------
def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """
    Normalize a vector to unit length.
    Ensures fair comparison for cosine/dot similarity.
    """
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def _to_numpy_vector(vec_raw) -> np.ndarray:
    """
    Convert a Postgres-stored vector (string or list) to a NumPy array.
    Handles both pgvector's array type and text-based "[...]" storage.
    """
    if isinstance(vec_raw, str):  # Case: stored as string "[0.1, 0.2, ...]"
        # np.array([float(x) for x in vec_raw[1:-1].split(',')], dtype=np.float32)
        return np.fromstring(vec_raw.strip("[]"), sep=",", dtype=np.float32)
    return np.array(vec_raw, dtype=np.float32)  # Already list-like


def _fetch_vectors():
    """
    Fetch all stored vectors and metadata from Postgres.
    Returns: list of (id, text, vector_raw, group_id).
    """
    with psycopg2.connect(**DB_CONFIG) as conn, conn.cursor() as cur:
        # Fetch vectors and their corresponding text chunks from the database
        cur.execute(f"SELECT id, section_content, vector, group_id, source_link, source_name, source_note FROM {SETTINGS.POSTGRES_TABLE}")
        rows = cur.fetchall()
        return rows

# ---------------------------
# Postgres scipy cosine search
# ---------------------------
def search_scipy_cosine(query_vector, top_n=5):
    """
    Compute cosine similarity in Python (scipy) directly from vectors stored in Postgres.
    NOTE: Scales poorly with many rows → consider FAISS or pgvector SQL ops for production.
    """
    rows = _fetch_vectors()
    if not rows:
        return []

    # Normalize query vector
    query_vector = normalize_vector(np.array(query_vector, dtype=np.float32))
    # Calculate cosine similarity between query vector and each chunk vector
    similarities = []
    for row in rows:
        chunk_id, section_content, vec_raw_str, group_id, source_link, source_name, source_note = row
        # Convert string representation of vector to numpy array
        chunk_vector = normalize_vector(_to_numpy_vector(vec_raw_str))
        # scipy cosine returns distance, so subtract from 1 to get similarity
        similarity = 1 - cosine(query_vector, chunk_vector)
        similarities.append((chunk_id, section_content, float(similarity), group_id, source_link))

    # Sort descending (higher similarity = better)
    return sorted(similarities, key=lambda x: x[2], reverse=True)[:top_n]


# ---------------------------
# FAISS L2
# ---------------------------
def search_faiss_l2(query_vector, top_n=5):
    """
    Euclidean (L2) distance search using FAISS.
    Lower distance = better, so we invert to similarity for consistency.
    """
    rows = _fetch_vectors()
    if not rows:
        return []

    # Prepare FAISS index
    d = len(query_vector)  # Dimension of the embeddings
    index = faiss.IndexFlatL2(d)  # Using Euclidean (L2) distance

    vectors, ids, contents, groups, source_links = [], [], [], [], []

    # Assuming vector_str is a string representation of a list of floats, e.g., "[0.1, 0.2, ...]"
    for row in rows:
        chunk_id, section_content, vec_raw_str, group_id, source_link, source_name, source_note = row
        try:
            # Safely convert the string representation of the vector to a NumPy array
            chunk_vector = _to_numpy_vector(vec_raw_str)
        except Exception as e:
            print(f"Error converting vector for ID {chunk_id}: {e}")
            continue

        # Check if chunk_vector is a valid numpy array
        if chunk_vector.shape[0] != d:
            print(f"Skipping invalid vector (dim mismatch) for ID {chunk_id}")
            continue
        vectors.append(chunk_vector)
        ids.append(chunk_id)
        contents.append(section_content)
        groups.append(group_id)
        source_links.append(source_link)

    # Convert list to numpy array and check if it's valid
    if not vectors:  # Check if vectors is not empty
        return []

    # Build FAISS index
    # Check if vectors_np is indeed a NumPy array
    index.add(np.vstack(vectors).astype(np.float32))
    # Convert query_vector to NumPy array
    # Search the query in FAISS
    qvec = np.array(query_vector, dtype=np.float32).reshape(1, -1)
    distances, indices = index.search(qvec, top_n)

    # Convert L2 distance to negative score so "higher is better"
    return [(ids[i], contents[i], -float(distances[0][n]), groups[i], source_links[i]) for n, i in enumerate(indices[0])]

# ---------------------------
# FAISS dot product
# ---------------------------
def search_faiss_dot(query_vector, top_n=5):
    """
    Dot-product similarity search using FAISS.
    Requires normalized vectors if you want cosine-equivalent results.
    """
    rows = _fetch_vectors()
    if not rows:
        return []

    # Prepare FAISS index
    d = len(query_vector)  # Dimension of the embeddings
    index = faiss.IndexFlatIP(d)  # Inner product = dot similarity

    vectors, ids, contents, groups, source_links = [], [], [], [], []

    # Assuming vector_str is a string representation of a list of floats, e.g., "[0.1, 0.2, ...]"
    for row in rows:
        chunk_id, section_content, vec_raw_str, group_id, source_link, source_name, source_note = row
        # Safely convert the string representation of the vector to a NumPy array
        vec = normalize_vector(_to_numpy_vector(vec_raw_str))
        vectors.append(vec)
        ids.append(chunk_id)
        contents.append(section_content)
        groups.append(group_id)
        source_links.append(source_link)

    # Convert list to numpy array and add to FAISS
    index.add(np.vstack(vectors).astype(np.float32))
    # Normalize query vector for fair comparison  ??
    qvec = normalize_vector(np.array(query_vector, dtype=np.float32)).reshape(1, -1)
    sims, indices = index.search(qvec, top_n)

    return [(ids[i], contents[i], float(sims[0][n]), groups[i], source_links[i]) for n, i in enumerate(indices[0])]


# ---------------------------
# Canberra distance
# ---------------------------
def _canberra(vec1, vec2):
    """
    Canberra distance between two vectors.
    Lower distance = more similar.
    """
    return np.sum(np.abs(vec1 - vec2) / (np.abs(vec1) + np.abs(vec2) + 1e-10))  # Avoid /0 division by zero

def search_canberra(query_vector, top_n=5):
    """
    Compute Canberra distance similarity.
    Returns inverted distance (negative) so "higher is better".
    """
    rows = _fetch_vectors()
    if not rows:
        return []

    # Calculate Canberra distance between query vector and each chunk vector
    sims = []
    for row in rows:
        chunk_id, section_content, vec_raw_str, group_id, source_link, source_name, source_note = row
        vec = _to_numpy_vector(vec_raw_str)
        dist = _canberra(query_vector, vec)
        sims.append((chunk_id, section_content, -float(dist), group_id, source_link))  # negative for consistency ??
    # Sort by distance (lower is better) and get top N results
    return sorted(sims, key=lambda x: x[2])[:top_n]


# ---------------------------
# Ensemble ranking
# ---------------------------
def compare_top_ids(scipy_cosine, faiss_l2, faiss_dot, canberra, top_n=5, ensemble_top_k=3):
    """
    Combine rankings from multiple similarity measures.
    Each rank contributes a weighted score (higher rank = more points).
    Returns top combined IDs.
    """
    # Initialize a dictionary to store the total score for each ID
    id_scores, id_groups, source_links = defaultdict(int), {}, {}
    # Function to assign scores based on rank
    def assign(results):
        for rank, r in enumerate(results[:top_n], start=1):
            chunk_id, _, _, group_id, source_link = r
            id_scores[chunk_id] += top_n - rank + 1  # Higher rank gives higher score (e.g., rank 1 gets top_n points, rank 2 gets top_n-1, etc.)
            id_groups[chunk_id] = group_id
            source_links[chunk_id] = source_link
    # Merge scores from all methods
    assign(scipy_cosine)
    assign(faiss_l2)
    assign(faiss_dot)
    assign(canberra)
    # Sort by total score, higher = better
    # Sort the IDs by their total score, higher score is better
    # Extract the top 3 IDs based on the highest scores
    top = sorted(id_scores.items(), key=lambda x: -x[1])[:ensemble_top_k]
    return [(cid, id_groups[cid], score, source_links[cid]) for cid, score in top]


# ---------------------------
# Direct DB fetch
# ---------------------------
def fetch_chunk_by_id(chunk_id):
    """
    Fetch the chunk text directly from Postgres given its ID.
    Useful for retrieving full text once a top ID is chosen.
    """
    with psycopg2.connect(**DB_CONFIG) as conn, conn.cursor() as cur:
        cur.execute(
            f"SELECT section_content FROM {SETTINGS.POSTGRES_TABLE} WHERE id = %s",
            (chunk_id,)
        )
        row = cur.fetchone()
        # Return the text content of the chunk
        # In case the chunk ID is not found
        return row[0] if row else None
    

def display_results(
    results, 
    method_name: str, 
    score_label: str = "score",  # Similarity
    max_text_len: int | None = 300
) -> None:
    """
    Pretty-print top search results.

    Args:
        results: List of tuples (id, text, score, group_id).
        method_name: Name of the search method (e.g., "FAISS L2").
        score_label: Label for the numeric metric (default: "score").
        max_text_len: If set, truncate long text for readability.

    display_results(results, "Scipy Cosine", score_label="similarity")
    display_results(results, "FAISS L2", score_label="distance")
    """
    print(f"\n=== Top Results ({method_name}) ===")
    if not results:
        print("No results found.")
        return
    for i, (chunk_id, section_content, similarity, group_id, source_link) in enumerate(results, start=1):
        text = section_content.strip().replace("\n", " ")
        if max_text_len and len(text) > max_text_len:
            text = text[:max_text_len] + "..."
        print(f"{i}. ID: {chunk_id} | {score_label}: {similarity:.4f} | Group_Id: {group_id} | Source_Link: {source_link}")
        print(f"   Text: {text}\n")