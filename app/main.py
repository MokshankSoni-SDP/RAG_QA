from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.database import create_tables


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
