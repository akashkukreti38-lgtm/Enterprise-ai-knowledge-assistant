from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import os
import fitz
import pytesseract
from PIL import Image

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

    text=extract_text_from_pdf(file_path)

    return {
        "message": "File uploaded successfully",
        "filename": file.filename,
        "text_preview": text[:500]
    }

"""def extract_text_from_pdf(file_path):
    document=fitz.open(file_path)
    text=""
    for page in document:
        text+=page.get_text()
    document.close()
    return text"""

def extract_text_from_pdf(file_path):
    document = fitz.open(file_path)

    text = ""

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

            ocr_text = pytesseract.image_to_string(image)
            text += ocr_text

    document.close()

    return text