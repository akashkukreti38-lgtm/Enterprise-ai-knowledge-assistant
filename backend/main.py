from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import os
import fitz
import pytesseract
from PIL import Image
import uuid
import hashlib

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from sentence_transformers import SentenceTransformer

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


app = FastAPI()

COLLECTION_NAME = "enterprise_documents"

qdrant_client = QdrantClient(path="qdrant_storage")

if not qdrant_client.collection_exists(COLLECTION_NAME):
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE
        )
    )

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

def get_file_hash(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        while True:
            data = file.read(8192)

            if not data:
                break

            sha256.update(data)

    return sha256.hexdigest()


@app.get("/")
def home():
    return {
        "message": "Enterprise AI Knowledge Assistant API is running"
    }

@app.get("/hello")
def hello(name:str):
    return{
        "message": f"hello {name}"
    }

class ChatRequest(BaseModel):
    question: str


@app.post("/chat")
def chat(request: ChatRequest):
    return {
        "message": f"You asked: {request.question}"
    }

class SearchRequest(BaseModel):
    question: str
    top_k: int = 3

@app.post("/search")
def search_documents(request: SearchRequest):

    # Convert the user's question into an embedding
    query_embedding = embedding_model.encode(
        request.question
    )

    # Search Qdrant
    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding.tolist(),
        limit=request.top_k
    ).points

    search_results = []

    for result in results:

        search_results.append({
            "filename": result.payload["filename"],
            "pages": (
                f"{result.payload['start_page']}-"
                f"{result.payload['end_page']}"
            ),
            "chunk_index": result.payload["chunk_index"],
            "score": result.score,
            "text": result.payload["text"]
        })

    return {
        "question": request.question,
        "results": search_results
    }


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):

    if file.content_type != "application/pdf":
        return {
            "error": "Only PDF files are allowed"
        }

    file_path = os.path.join(
        "backend",
        "uploads",
        file.filename
    )

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # Create stable document ID
    document_id = get_file_hash(file_path)

    # Extract pages
    pages = extract_pages_from_pdf(file_path)

    # Create chunks
    chunks = chunk_text(
        pages,
        document_id,
        file.filename
    )

    # Create embeddings
    chunk_texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        chunk_texts
    )

    # Create Qdrant points
    points = []

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

        points.append(point)

    # Store in Qdrant
    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    # Check current number of points
    count = qdrant_client.count(
        collection_name=COLLECTION_NAME,
        exact=True
    )

    return {
        "message": "Document processed successfully",
        "document_id": document_id,
        "filename": file.filename,
        "pages": len(pages),
        "chunks": len(chunks),
        "points_in_qdrant": count.count
    }

def extract_pages_from_pdf(file_path):
    document = fitz.open(file_path)

    text = ""
    pages=[]

    for page in document:
        page_text = page.get_text()

        if page_text.strip():
            text = page_text
        else:
            pix = page.get_pixmap()
            image = Image.frombytes(
                "RGB",
                [pix.width, pix.height],
                pix.samples
            )

            text = pytesseract.image_to_string(image)
        pages.append({"page_number":page.number+1,
                          "text":text})

    document.close()

    return pages

def chunk_text(pages, document_id, filename, chunk_size=500, overlap=50):
    chunks = []

    # Combine pages while keeping track of which page each word came from
    all_words = []

    for page in pages:
        words = page["text"].split()

        for word in words:
            all_words.append({
                "word": word,
                "page_number": page["page_number"]
            })

    chunk_index = 0
    start = 0

    while start < len(all_words):

        end = min(start + chunk_size, len(all_words))

        chunk_items = all_words[start:end]

        chunk = " ".join(
            item["word"] for item in chunk_items
        )

        start_page = chunk_items[0]["page_number"]
        end_page = chunk_items[-1]["page_number"]

        chunks.append({
            "document_id": document_id,
            "filename": filename,
            "start_page": start_page,
            "end_page": end_page,
            "chunk_index": chunk_index,
            "text": chunk
        })

        chunk_index += 1

        # Move forward while keeping the overlap
        start += chunk_size - overlap

    return chunks