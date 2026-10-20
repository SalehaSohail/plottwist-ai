
from sentence_transformers import SentenceTransformer, util

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

EVIDENCE = [
    "Marcus left the laboratory at 9:15 PM.",
    "A red bicycle was found near the park.",
    "The laboratory security camera stopped recording at 9:00 PM.",
    "Sarah entered the laboratory at 8:45 PM.",
    "A broken window was discovered in the laboratory."
]


def retrieve_evidence(
    question: str,
    model: SentenceTransformer,
    evidence: list[str],
    evidence_embeddings,
    top_k: int = 3
) -> list[dict]:
    """Retrieve the most relevant evidence for a detective question."""

    if not question.strip():
        return []

    # Convert the question into an embedding.
    question_embedding = model.encode(question)

    # Compare the question against all evidence embeddings.
    similarities = util.cos_sim(
        question_embedding,
        evidence_embeddings
    )[0]

    # Get the indices of the highest similarity scores.
    top_k = min(top_k, len(evidence))
    top_results = similarities.topk(k=top_k)

    results = []

    for score, index in zip(
        top_results.values,
        top_results.indices
    ):
        evidence_index = index.item()

        results.append({
            "evidence": evidence[evidence_index],
            "score": round(score.item(), 4)
        })

    return results


def main():
    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Generating evidence embeddings...")
    evidence_embeddings = model.encode(EVIDENCE)

    print("\nWelcome to PlotTwist AI - Evidence Retrieval")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Detective, ask your question: ").strip()

        if question.lower() == "exit":
            print("Investigation closed. Goodbye!")
            break

        if not question:
            print("Please enter a question.\n")
            continue

        results = retrieve_evidence(
            question=question,
            model=model,
            evidence=EVIDENCE,
            evidence_embeddings=evidence_embeddings,
            top_k=3
        )

        print("\nMost Relevant Evidence:")

        for position, result in enumerate(results, start=1):
            print(f"\n{position}. {result['evidence']}")
            print(f"   Similarity Score: {result['score']}")

        print("\n" + "-" * 50)


if __name__ == "__main__":
    main()
