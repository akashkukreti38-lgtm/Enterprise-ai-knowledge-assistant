from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from sentence_transformers import SentenceTransformer
import uuid
import hashlib

from backend.main import extract_pages_from_pdf, chunk_text


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

def get_file_hash(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        while True:
            data = file.read(8192)

            if not data:
                break

            sha256.update(data)

    return sha256.hexdigest()

PDF_FILES = [
    "backend/uploads/HR Policy TCS.pdf",
    "backend/uploads/IT-Security-Policy.pdf",
    "backend/uploads/Employee-Handbook.pdf"
]

COLLECTION_NAME = "enterprise_documents_test"


# --------------------------------------------------
# 2. Create fresh in-memory Qdrant
# --------------------------------------------------

client = QdrantClient(location=":memory:")

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    )
)

print("Fresh Qdrant collection created.")

count = client.count(
    collection_name=COLLECTION_NAME,
    exact=True
)

print("Initial points:", count.count)


# --------------------------------------------------
# 3. Load embedding model
# --------------------------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")


# --------------------------------------------------
# 4. Process each PDF
# --------------------------------------------------

all_points = []

for pdf_path in PDF_FILES:

    print("\n" + "=" * 80)
    print("Processing:", pdf_path)
    print("=" * 80)

    # Extract pages
    pages = extract_pages_from_pdf(pdf_path)

    print("Number of pages:", len(pages))

    # Create document ID
    document_id = get_file_hash(pdf_path)

    # Get filename
    filename = pdf_path.split("/")[-1]

    # Create chunks
    chunks = chunk_text(
        pages,
        document_id,
        filename
    )

    print("Number of chunks:", len(chunks))

    # Extract chunk text
    chunk_texts = [
        chunk["text"]
        for chunk in chunks
    ]

    # Create embeddings
    embeddings = model.encode(chunk_texts)

    print("Number of embeddings:", len(embeddings))


    # --------------------------------------------------
    # 5. Create Qdrant points
    # --------------------------------------------------

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

        all_points.append(point)


# --------------------------------------------------
# 6. Insert everything into Qdrant
# --------------------------------------------------


# --------------------------------------------------
# 7. Test cross-document search
# --------------------------------------------------

for run in range(2):

    print("\n" + "=" * 80)
    print("INGESTION RUN:", run + 1)
    print("=" * 80)

    all_points = []

    for pdf_path in PDF_FILES:

        print("\nProcessing:", pdf_path)

        pages = extract_pages_from_pdf(pdf_path)

        document_id = get_file_hash(pdf_path)

        filename = pdf_path.split("/")[-1]

        print("Document ID:", document_id[:16] + "...")

        chunks = chunk_text(
            pages,
            document_id,
            filename
        )

        chunk_texts = [
            chunk["text"]
            for chunk in chunks
        ]

        embeddings = model.encode(chunk_texts)

        for chunk, embedding in zip(chunks, embeddings):

            point_id = str(uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{document_id}-{chunk['chunk_index']}"
            ))

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

            all_points.append(point)

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=all_points
    )

    count = client.count(
        collection_name=COLLECTION_NAME,
        exact=True
    )

    print("\nPoints after run", run + 1, ":", count.count)

queries = [
    "What is the leave policy for employees?",
    "What are the IT security rules?",
    "What benefits are provided to employees?"
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
            f"Document: {result.payload['filename']} | "
            f"Chunk: {result.payload['chunk_index']} | "
            f"Pages: "
            f"{result.payload['start_page']}-"
            f"{result.payload['end_page']}"
        )


# --------------------------------------------------
# 8. Close Qdrant
# --------------------------------------------------

client.close()