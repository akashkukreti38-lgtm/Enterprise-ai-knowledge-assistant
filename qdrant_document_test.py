from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import uuid
from qdrant_client.models import (
    PointStruct,
    Distance,
    VectorParams
)

from backend.main import extract_pages_from_pdf, chunk_text


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

PDF_PATH = "backend/uploads/HR Policy TCS.pdf"
COLLECTION_NAME = "enterprise_documents_test"


# --------------------------------------------------
# 2. Connect to Qdrant
# --------------------------------------------------

client = QdrantClient(location=":memory:")

# --------------------------------------------------
# Reset collection
# --------------------------------------------------


client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    )
)

print("Fresh collection created.")

count = client.count(
    collection_name=COLLECTION_NAME,
    exact=True
)

print("Points in fresh collection:", count.count)


# --------------------------------------------------
# 3. Load embedding model
# --------------------------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")


# --------------------------------------------------
# 4. Extract text from PDF
# --------------------------------------------------

pages = extract_pages_from_pdf(PDF_PATH)

print("Number of pages:", len(pages))


# --------------------------------------------------
# 5. Create chunks
# --------------------------------------------------

document_id = str(uuid.uuid4())

chunks = chunk_text(
    pages,
    document_id,
    "HR Policy TCS.pdf"
)

print("Number of chunks:", len(chunks))


# --------------------------------------------------
# 6. Create embeddings
# --------------------------------------------------

chunk_texts = [
    chunk["text"]
    for chunk in chunks
]

embeddings = model.encode(chunk_texts)

print("Number of embeddings:", len(embeddings))
print("Embedding dimensions:", len(embeddings[0]))

print("\nChecking embeddings:")
for i, embedding in enumerate(embeddings):
    print(
        f"Embedding {i}: "
        f"{embedding[:5]}"
    )

print("\nChecking chunks:")
for chunk in chunks:
    print(
        f"Chunk {chunk['chunk_index']} | "
        f"Pages {chunk['start_page']}-{chunk['end_page']} | "
        f"Words {len(chunk['text'].split())}"
    )


# --------------------------------------------------
# 7. Create Qdrant points
# --------------------------------------------------

points = []

for chunk, embedding in zip(chunks, embeddings):

    point_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"{document_id}-{chunk['chunk_index']}"
        )
    )

    point = PointStruct(
        id=point_id,
        vector=embedding.tolist(),
        payload={
            "document_id": chunk["document_id"],
            "filename": chunk["filename"],
            "chunk_index": chunk["chunk_index"],
            "start_page": chunk["start_page"],
            "end_page": chunk["end_page"],
            "text": chunk["text"]
        }
    )

    points.append(point)


print("\nChecking points before Qdrant insertion:")

for point in points:
    print(
        f"ID: {point.id} | "
        f"Chunk: {point.payload['chunk_index']} | "
        f"Pages: {point.payload['start_page']}-"
        f"{point.payload['end_page']}"
    )


# --------------------------------------------------
# 8. Store points in Qdrant
# --------------------------------------------------

client.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)

print("Points inserted into Qdrant:", len(points))

print("\nChecking points inside Qdrant:")

stored_points, _ = client.scroll(
    collection_name=COLLECTION_NAME,
    limit=20,
    with_payload=True,
    with_vectors=False
)

for point in stored_points:
    print(
        f"ID: {point.id} | "
        f"Chunk: {point.payload['chunk_index']} | "
        f"Pages: {point.payload['start_page']}-"
        f"{point.payload['end_page']}"
    )


# --------------------------------------------------
# 9. Test semantic search
# --------------------------------------------------

queries = [
    "What is the leave policy for employees?",
    "How does TCS recruit employees?",
    "What benefits are provided to employees?",
    "How does the company handle employee absenteeism?"
]

for query in queries:

    query_embedding = model.encode(query)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding.tolist(),
        limit=3
    ).points

    print("\n" + "=" * 80)
    print("QUERY:", query)
    print("=" * 80)

    for rank, result in enumerate(results, start=1):
        print(
            f"Rank {rank} | "
            f"Score: {result.score:.4f} | "
            f"Chunk: {result.payload['chunk_index']} | "
            f"Pages: {result.payload['start_page']}-"
            f"{result.payload['end_page']}"
        )


# --------------------------------------------------
# 10. Display results
# --------------------------------------------------

print("\nQuery:")
print(query)

print("\nTop 3 results:\n")

for rank, result in enumerate(results, start=1):
    print(f"Rank {rank}")
    print(f"ID: {result.id}")
    print(f"Similarity: {result.score:.4f}")
    print(f"Chunk index: {result.payload['chunk_index']}")
    print(f"Pages: {result.payload['start_page']}-{result.payload['end_page']}")
    print("-" * 80)

# --------------------------------------------------
# 11. Close Qdrant
# --------------------------------------------------

client.close()