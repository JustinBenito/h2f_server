from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import whisper
import os
import uvicorn
from pydub import AudioSegment
from tempfile import NamedTemporaryFile
import ollama
import torch

model = whisper.load_model("tiny.en")

app = FastAPI()

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        # Save the incoming file to a temporary location
        temp_file = NamedTemporaryFile(delete=False, suffix=".tmp")
        file_location = temp_file.name
        print(temp_file.name)
        with open(file_location, "wb") as f:
            f.write(await file.read())
        
        # Convert the file to WAV format
        wav_file_location = file_location + ".wav"
        try:
            audio = AudioSegment.from_file(file_location)
            audio.export(wav_file_location, format="wav")
        except Exception as e:
            print(f"Error converting file to WAV: {e}")
            return JSONResponse(content={"error": "Failed to convert file to WAV"}, status_code=500)
        
        # Process the WAV file with Whisper
        try:
            result = model.transcribe(wav_file_location)
            transcription = str(result["text"]) + " You are a model that is very friendly, you only give one sentence replies and repsonses in a friendly way."
            print(str(transcription))

            resp = ollama.chat(model='llama3.2:1b', messages=[
            {
                'role': 'user',
                'content': transcription,
            },
            ])

            return JSONResponse(content={"transcription": resp}, status_code=200) 
        except Exception as e:
            print(f"Error transcribing file: {e}")
            return JSONResponse(content={"error": "Failed to transcribe file"}, status_code=500)
        
    finally:
        print("hi")



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
