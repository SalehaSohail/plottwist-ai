
from ollama import Client

MODEL_NAME = "llama3.2:3b"

client = Client(host="http://localhost:11434")


def generate_answer(question: str, evidence: list[dict]) -> str:

    if not evidence:
        return "There is not enough evidence to answer this question."

    # Sort evidence by source and original position
    ordered_evidence = sorted(
        evidence,
        key=lambda item: (
            item["source"],
            item["chunk_index"]
        )
    )

    # Combine evidence
    evidence_text = "\n".join(
        item["evidence"]
        for item in ordered_evidence
    )

    # Minimal prompt, similar to our successful terminal test
    prompt = f"""Read this witness statement and answer the question.

{evidence_text}

Question: {question}
Answer briefly."""

    print("\nPROMPT SENT TO LLAMA:\n")
    print(prompt)

    response = client.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0
        }
    )

    answer = response.message.content or ""

    print("\nLLAMA RESPONSE:\n")
    print(answer)

    return answer

