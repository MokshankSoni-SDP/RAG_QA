import numpy as np
from typing import List, Dict

from app.database import fetch_all_chunks


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    numerator = np.dot(vec1, vec2)
    denominator = np.linalg.norm(vec1) * np.linalg.norm(vec2)

    if denominator == 0.0:
        return 0.0

    return float(numerator / denominator)

def retrieve_top_k_chunks(
    query_embedding: np.ndarray,
    top_k: int = 3
) -> List[Dict]:
    

    rows = fetch_all_chunks()
    scored_chunks = []

    for document_name, chunk_id, chunk_text, embedding_blob in rows:
        chunk_embedding = np.frombuffer(
            embedding_blob, dtype=np.float32
        )

        score = cosine_similarity(query_embedding, chunk_embedding)

        scored_chunks.append({
            "document": document_name,
            "chunk_id": chunk_id,
            "text": chunk_text,
            "score": score
        })

    # Sort by similarity score (descending)
    scored_chunks.sort(key=lambda x: x["score"], reverse=True)

    return scored_chunks[:top_k]
