
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field

from app.retriever import load_model
from app.vector_store import EvidenceVectorStore


app = FastAPI(
    title="PlotTwist AI",
    description="AI-powered detective evidence retrieval API",
    version="0.2.0"
)

# Load embedding model once
model = load_model()

# Initialize persistent vector database
vector_store = EvidenceVectorStore(model)


class QuestionRequest(BaseModel):
    question: str
    top_k: int = Field(default=3, ge=1)


@app.get("/")
def home():
    return {
        "message": "Welcome to PlotTwist AI!",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "evidence_count": vector_store.count_evidence()
    }


@app.post("/documents")
async def upload_document(file: UploadFile = File(...)):

    # Read uploaded document
    content = await file.read()

    # Decode UTF-8 text
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Please upload a UTF-8 encoded text file."
        )

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Split text into paragraphs
    chunks = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="The uploaded document is empty."
        )

    # Save evidence in ChromaDB
    stored_count = vector_store.add_evidence(
        chunks=chunks,
        filename=file.filename or "unknown.txt"
    )

    return {
        "filename": file.filename,
        "size": len(content),
        "text": text,
        "chunks": chunks,
        "chunk_count": len(chunks),
        "stored_count": stored_count,
        "total_evidence": vector_store.count_evidence()
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    if vector_store.count_evidence() == 0:
        return {
            "question": request.question,
            "message": "No evidence uploaded yet.",
            "results": []
        }

    # Retrieve evidence from ChromaDB
    results = vector_store.search_evidence(
        question=request.question,
        top_k=request.top_k
    )

    return {
        "question": request.question,
        "results": results
    }
