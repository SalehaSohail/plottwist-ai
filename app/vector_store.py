
import chromadb
from pathlib import Path
from uuid import uuid4

from sentence_transformers import SentenceTransformer


# Directory where ChromaDB will save data
DB_PATH = Path("data/chroma_db")

# Collection name for detective evidence
COLLECTION_NAME = "detective_evidence"


class EvidenceVectorStore:
    def __init__(self, model: SentenceTransformer):
        self.model = model

        # Create a persistent ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(DB_PATH)
        )

        # Create or load the evidence collection
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    def add_evidence(
        self,
        chunks: list[str],
        filename: str
    ) -> int:

        if not chunks:
            return 0

        # Generate embeddings for evidence chunks
        embeddings = self.model.encode(chunks).tolist()

        # Generate unique IDs for each chunk
        ids = [str(uuid4()) for _ in chunks]

        # Store source filename for each chunk
        metadatas = [
            {
                "filename": filename,
                "chunk_index": index
            }
            for index in range(len(chunks))
        ]

        # Save evidence in ChromaDB
        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )

        return len(chunks)

    def search_evidence(
        self,
        question: str,
        top_k: int = 3
    ) -> list[dict]:

        if not question.strip() or top_k <= 0:
            return []

        # Check whether evidence exists
        total_chunks = self.collection.count()

        if total_chunks == 0:
            return []

        # Convert the question into an embedding
        question_embedding = self.model.encode(
            question
        ).tolist()

        # Search for the most similar evidence
        search_results = self.collection.query(
            query_embeddings=[question_embedding],
            n_results=min(top_k, total_chunks),
            include=["documents", "distances", "metadatas"]
        )

        results = []

        # Convert ChromaDB results into API-friendly format
        for document, distance, metadata in zip(
            search_results["documents"][0],
            search_results["distances"][0],
            search_results["metadatas"][0]
        ):
            results.append({
                "evidence": document,
                "score": round(1 - distance, 4),
                "source": metadata["filename"],
                "chunk_index": metadata["chunk_index"]
            })

        return results

    def count_evidence(self) -> int:
        return self.collection.count()
