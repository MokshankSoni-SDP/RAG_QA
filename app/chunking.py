import re
from typing import List


def semantic_chunk_text(text: str, max_words: int = 150) -> List[str]:
    """
    Split text into semantic chunks based on sentences,
    grouping them until max_words is reached.
    """

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Split into sentences (simple but effective)
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        words = sentence.split()
        if current_word_count + len(words) > max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_word_count = 0

        current_chunk.append(sentence)
        current_word_count += len(words)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
