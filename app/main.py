from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.database import create_tables
from fastapi import UploadFile, File, HTTPException
from app.chunking import semantic_chunk_text
from app.embeddings import generate_embedding
from app.database import insert_chunk


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
