from sentence_transformers import SentenceTransformer,util

MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"

def load_model():
    return SentenceTransformer(MODEL_NAME)


def retrieve_evidence(question:str,model:SentenceTransformer,evidence:list[str],evidence_embeddings,top_k:int=3)->list[dict]:
    if not question.strip():
        return []
    question_embedding=model.encode(question)
    similarities=util.cos_sim(question_embedding,evidence_embeddings)[0]
    top_k = min(top_k, len(evidence))
    top_results = similarities.topk(k=top_k)
    results=[]
    for score, index in zip(top_results.values, top_results.indices):
        evidence_index = index.item()

        results.append({
            "evidence": evidence[evidence_index],
            "score": round(score.item(), 4)
        })

    return results
