from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import os
import fitz
import pytesseract
from PIL import Image
import uuid

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


app = FastAPI()


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

@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):

    if file.content_type != "application/pdf":
        return {
            "error": "Only PDF files are allowed"
        }

    file_path = os.path.join("backend", "uploads", file.filename)

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    document_id=str(uuid.uuid4())

    pages=extract_pages_from_pdf(file_path)

    chunks=chunk_text(pages,document_id,file.filename)

    full_text = "\n".join(page["text"] for page in pages)

    return {
        "message": "File uploaded successfully",
        "document_id":document_id,
        "filename": file.filename,
        "text_preview": full_text[:500],
        "chunk_count": len(chunks),
        "first_chunk": chunks[0]["text"] if chunks else ""
    }

def extract_pages_from_pdf(file_path):
    document = fitz.open(file_path)

    text = ""
    pages=[]

    for page in document:
        page_text = page.get_text()

        if page_text.strip():
            text += page_text
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
    chunk_index = 0

    for page in pages:
        words = page["text"].split()
        start = 0

        while start < len(words):
            end = start + chunk_size

            chunk = " ".join(words[start:end])

            chunks.append({
                "document_id": document_id,
                "filename": filename,
                "page_number": page["page_number"],
                "chunk_index": chunk_index,
                "text": chunk
            })

            chunk_index += 1
            start += chunk_size - overlap

    return chunks