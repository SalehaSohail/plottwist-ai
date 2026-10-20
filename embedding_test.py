from sentence_transformers import SentenceTransformer
from sentence_transformers import util
model=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

sentences = [
    "Marcus left the laboratory at 9:15 PM.",
    "Marcus exited the lab at 9:15 PM.",
    "The detective found a red bicycle near the park."
]
embedding=model.encode(sentences)

print("Embedding shape: " , embedding.shape)
similarity=util.cos_sim(embedding,embedding)

print("Similarity Matrix:")
print(similarity)