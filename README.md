## 🧩 Chunking Strategy

The system uses **sentence-based semantic chunking**.

### How it works

1. **Normalization:** The input document text is normalized (extra whitespace removed).
2. **Splitting:** The text is split into sentences using punctuation boundaries.
3. **Grouping:** Consecutive sentences are grouped together until a target word limit (≈150 words) is reached.
4. **Creation:** Each group of sentences forms one chunk.

### Why this approach

- **Preserves semantic meaning** (unlike fixed character or line-based chunking).
- **Avoids breaking sentences** mid-thought.
- **Produces coherent chunks** that are self-contained and suitable for embedding and retrieval.
- **Lightweight and deterministic**, making it easy to debug and explain.

> _This approach provides a good balance between simplicity and retrieval quality for small-to-medium text documents._

---

## 🧠 Embedding Choice

The project uses the Hugging Face embedding model:

`sentence-transformers/all-MiniLM-L6-v2`

### Reasons for choosing this model

- **Purpose-built:** Designed specifically for sentence-level semantic similarity.
- **Efficiency:** Lightweight and CPU-friendly (no GPU required).
- **Stability:** Produces stable embeddings well-suited for cosine similarity.
- **Adoption:** Widely used and well-documented in RAG and semantic search tasks.

### Trade-offs

- **Dimensionality:** The embedding dimension (384) is smaller than larger models, potentially capturing slightly less nuance.
- **Language:** Not optimized for multilingual scenarios.

> _Given the project constraints (SQLite storage, manual similarity, CPU execution), this model provides the best balance of performance, simplicity, and reliability._

---

## 📊 Confidence Logic

Confidence is **computed**, not hardcoded.

### How confidence is calculated

1. For a given question, the **top-k** most relevant chunks are retrieved using cosine similarity.
2. Each retrieved chunk has an associated **similarity score**.
3. The final confidence value is calculated as the **average cosine similarity** of the retrieved chunks.

### Why this works

- **Direct Correlation:** Cosine similarity directly reflects semantic closeness between the question and the retrieved context.
- **Groundedness:** Higher similarity → higher confidence in the answer being grounded in the source text.
- **Transparency:** Simple, explainable, and quantitatively justified.

_Note: If retrieval scores are too low, the system returns a zero confidence value._

---

## 🚫 Hallucination Prevention

The system is explicitly designed to **prevent hallucinations**.

### Measures taken

1. **Strict Prompt Construction:** The language model is instructed to answer _only_ using the retrieved context.
2. **Forced Fallback Response:** If the answer is not present in the provided context, the model must respond exactly with:
   > "I don't know based on the provided context."
3. **Retrieval Threshold Check:** If no retrieved chunk exceeds a minimum similarity threshold, the system bypasses the LLM and directly returns the fallback response.

### Result

- The system **never** generates unsupported answers.
- Every answer is **grounded** in retrieved document chunks.
- **Evidence** is always returned alongside the answer.

---

## ⚠️ Limitations

While the system fulfills all functional requirements, it has the following limitations:

- **Scalability:** Uses SQLite for storage, which may not scale well for very large document collections.
- **Language Support:** The embedding model is optimized for English text.
- **Dependencies:** LLM quality depends on external API availability and rate limits.

_These trade-offs were made intentionally to keep the system simple, transparent, and aligned with the assignment constraints._

---

## ✅ Summary

This project demonstrates a complete, end-to-end RAG pipeline with:

- [x] Semantic chunking
- [x] Hugging Face embeddings
- [x] Manual cosine similarity
- [x] Evidence-backed answers
- [x] Computed confidence
- [x] Explicit hallucination control

The implementation prioritizes **correctness, explainability, and adherence to engineering constraints** over unnecessary complexity.
