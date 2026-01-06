import numpy as np
from pydantic import BaseModel

from app.embeddings import generate_embedding
from app.retrieval import retrieve_top_k_chunks
from app.qa import build_prompt, generate_answer

from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.database import create_tables
from fastapi import UploadFile, File, HTTPException
from app.chunking import semantic_chunk_text
from app.embeddings import generate_embedding
from app.database import insert_chunk


class QuestionRequest(BaseModel):
    question: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    create_tables()
    yield
    # Shutdown logic (not needed now)


app = FastAPI(
    title="RAG QA System",
    description="Text-based Question Answering with Evidence and Confidence",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "API is running correctly"
    }


@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are supported")

    content = await file.read()
    text = content.decode("utf-8").strip()

    if not text:
        raise HTTPException(status_code=400, detail="Empty document")

    chunks = semantic_chunk_text(text)

    for idx, chunk in enumerate(chunks):
        embedding_bytes = generate_embedding(chunk)
        insert_chunk(
            document_name=file.filename,
            chunk_id=idx,
            chunk_text=chunk,
            embedding=embedding_bytes
        )

    return {
        "status": "success",
        "document": file.filename,
        "chunks_created": len(chunks)
    }

@app.post("/ask")
def ask_question(request: QuestionRequest):
    question = request.question.strip()

    if not question:
        return {
            "question": question,
            "answer": "I don't know based on the provided context.",
            "confidence": 0.0,
            "evidence": []
        }

    # Generate embedding for question
    q_emb_bytes = generate_embedding(question)
    q_emb = np.frombuffer(q_emb_bytes, dtype=np.float32)

    # Retrieve top-k relevant chunks
    top_chunks = retrieve_top_k_chunks(q_emb, top_k=3)

    # If retrieval confidence is too low → no answer
    if not top_chunks or top_chunks[0]["score"] < 0.2:
        return {
            "question": question,
            "answer": "I don't know based on the provided context.",
            "confidence": 0.0,
            "evidence": []
        }

    # Build strict prompt
    prompt = build_prompt(question, top_chunks)

    # Generate answer using LLM
    answer = generate_answer(prompt)

    # Compute confidence (average similarity)
    confidence = sum(c["score"] for c in top_chunks) / len(top_chunks)

    # Evidence
    evidence = [
        {
            "document": c["document"],
            "chunk_id": c["chunk_id"],
            "text": c["text"]
        }
        for c in top_chunks
    ]

    return {
        "question": question,
        "answer": answer,
        "confidence": round(confidence, 3),
        "evidence": evidence
    }
