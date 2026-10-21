
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field

from app.retriever import load_model
from app.vector_store import EvidenceVectorStore
from app.llm import generate_answer

from ollama import ResponseError
from httpx import ConnectError


app = FastAPI(
    title="PlotTwist AI",
    description="AI-powered detective assistant using RAG",
    version="0.3.0"
)

# Load the embedding model once
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
    content = await file.read()

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

    # Step 1: Retrieve relevant evidence
    evidence = vector_store.search_evidence(
        question=request.question,
        top_k=request.top_k
    )

    if not evidence:
        return {
            "question": request.question,
            "answer": "There is not enough evidence to answer this question.",
            "results": []
        }

    # Step 2: Generate an answer using the local LLM
    try:
        answer = generate_answer(
            question=request.question,
            evidence=evidence
        )
    except (ConnectError, ResponseError) as error:
        raise HTTPException(
            status_code=503,
            detail=f"Ollama is unavailable: {error}"
        )

    # Step 3: Return the answer and its supporting evidence
    return {
        "question": request.question,
        "answer": answer,
        "results": evidence
    }
