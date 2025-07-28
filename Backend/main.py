from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os
sys.path.append(os.path.dirname(__file__))
from atc_transcriber import get_transcriber

app = FastAPI()

# Allow frontend (adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for request body
class TranscriptionRequest(BaseModel):
    audio_path: str

@app.post("/transcribe")
def transcribe_audio(request: TranscriptionRequest):
    audio_file = request.audio_path.lstrip("/")
    full_path = os.path.normpath(os.path.join("Frontend", "public", audio_file.lstrip("/")))

    if not os.path.exists(full_path):
        return {"error": f"Audio file not found: {full_path}"}

    transcriber = get_transcriber()
    result = transcriber.transcribe_audio_file(full_path)
    return {"transcription": result}