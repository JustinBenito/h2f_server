from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import base64
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import soundfile as sf
from pydub import AudioSegment
import os
import requests
import base64
from groq import Groq

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models and processors globally
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
model = WhisperForConditionalGeneration.from_pretrained("shReYas0363/whisper-tiny-fine-tuned")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

groq_client = Groq(api_key='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')


def convertaudio_64(input_path, output_path):
    audio = AudioSegment.from_file(input_path, format="m4a")
    audio = audio.set_frame_rate(16000)
    audio.export(output_path, format="wav")
    return True

async def transcribe_audio(audio_path: str):
    try:
        audio_input, sampling_rate = sf.read(audio_path)
        
        if len(audio_input.shape) > 1:
            audio_input = audio_input.mean(axis=1)
        
        input_features = processor(audio_input, sampling_rate=sampling_rate, return_tensors="pt").input_features
        input_features = input_features.to(device)
        
        predicted_ids = model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        print(transcription)
        return transcription
    except Exception as e:
        print(f"Error in transcription: {e}")
        return None

async def generate_chat_response(transcription: str):
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": f'''You are a medical assistant give a very friendly one line response for this query: {transcription} Remember to give a single reply in just one reply'''}],
            model="llama-3.2-1b-preview",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Error in chat response: {e}")
        return None

@app.post("/process-audio/")
async def process_audio(file: UploadFile = File(...)):
    try:
        # Save the uploaded file temporarily
        input_path = file.filename
        output_path = input_path.replace(".m4a", ".wav")
        
        with open(input_path, "wb") as f:
            f.write(await file.read())
        
        # Convert audio and transcribe
        convertaudio_64(input_path, output_path)
        transcription = await transcribe_audio(output_path)
        
        if transcription:
            response = await generate_chat_response(transcription)
            if response:
                # Get TTS audio as a response
                tts_url = "http://[::1]:5002/api/tts"
                tts_params = {"text": response, "speaker_id": "p374"}
                tts_response = requests.get(tts_url, params=tts_params)
                
                if tts_response.status_code == 200:
                    output_audio_path = "response_audio.wav"
                    with open(output_audio_path, "wb") as f:
                        f.write(tts_response.content)
                    with open(output_audio_path, "rb") as audio_file:
                        encoded_string = base64.b64encode(audio_file.read()).decode('utf-8')
                    return {"message": "Success", "chat": response, "encode": encoded_string, "transcription":transcription}
                else:
                    raise HTTPException(status_code=500, detail="TTS service error.")
            else:
                raise HTTPException(status_code=500, detail="Failed to generate chat response.")
        else:
            raise HTTPException(status_code=500, detail="Failed to transcribe audio.")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
