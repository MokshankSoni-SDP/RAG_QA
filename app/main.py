from fastapi import FastAPI

app = FastAPI(
    title="RAG QA System",
    description="Text-based Question Answering with Evidence and Confidence",
    version="1.0.0"
)

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "API is running correctly"
    }
