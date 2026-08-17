from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import os

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
    file_path=os.path.join("backend","uploads", file.filename)
    with open(file_path,"wb") as buffer:
        buffer.write(await file.read())
    return {
        "filename": file.filename,
        "content_type": file.content_type
    }