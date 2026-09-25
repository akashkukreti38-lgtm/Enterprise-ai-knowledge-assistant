from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


# Create a local Qdrant database
client = QdrantClient(path="qdrant_storage")


# Create our collection
client.create_collection(
    collection_name="enterprise_documents",
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    )
)

print("Collection created successfully!")


# Close Qdrant
client.close()