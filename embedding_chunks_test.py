from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

chunks = [
    "Employees receive 24 days of annual leave.",
    "Workers get 24 vacation days every year.",
    "The database stores information using PostgreSQL."
]

embeddings = model.encode(chunks)

print("Number of chunks:", len(chunks))
print("Number of embeddings:", len(embeddings))
print("Dimensions of each embedding:", len(embeddings[0]))