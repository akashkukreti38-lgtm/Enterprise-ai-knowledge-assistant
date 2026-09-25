from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

text_a = "Employees receive 24 days of annual leave."
text_b = "Workers get 24 vacation days every year."
text_c = "The database stores information using PostgreSQL."

embedding_a = model.encode([text_a])
embedding_b = model.encode([text_b])
embedding_c = model.encode([text_c])

similarity_ab = cosine_similarity(embedding_a, embedding_b)[0][0]
similarity_ac = cosine_similarity(embedding_a, embedding_c)[0][0]

print("Similarity between A and B:")
print(similarity_ab)

print("\nSimilarity between A and C:")
print(similarity_ac)