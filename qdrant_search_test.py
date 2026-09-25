from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


# Connect to our local Qdrant database
client = QdrantClient(path="qdrant_storage")


# Load the same embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# User's question
query = "How many vacation days do employees receive?"


# Convert the question into a vector
query_embedding = model.encode(query)


# Search Qdrant
results = client.query_points(
    collection_name="enterprise_documents",
    query=query_embedding.tolist(),
    limit=1
).points


# Display the result
for result in results:
    print("Similarity score:", result.score)
    print("Payload:", result.payload)


# Close Qdrant
client.close()