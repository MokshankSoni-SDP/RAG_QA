import numpy as np
from sentence_transformers import SentenceTransformer

# Load once (VERY IMPORTANT)
_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def generate_embedding(text: str) -> bytes:
    """
    Generate embedding for a single chunk and return as bytes.
    """
    vector = _model.encode(text)
    vector = vector.astype("float32")
    return vector.tobytes()
