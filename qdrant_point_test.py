from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer


# Connect to our local Qdrant database
client = QdrantClient(path="qdrant_storage")


# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Text that we want to store
text = "Employees receive 24 days of annual leave."


# Convert text into a 384-dimensional vector
embedding = model.encode(text)


# Create a point
point = PointStruct(
    id=1,
    vector=embedding.tolist(),
    payload={
        "text": text,
        "source": "employee_policy.pdf"
    }
)


# Store the point in Qdrant
client.upsert(
    collection_name="enterprise_documents",
    points=[point]
)


print("Point inserted successfully!")


# Close Qdrant
client.close()