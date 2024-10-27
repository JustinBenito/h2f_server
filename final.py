from fastapi import FastAPI, WebSocket, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Dict
import base64
from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoTokenizer
import torch
import soundfile as sf
import subprocess
import os
from pydub import AudioSegment
import time
from groq import Groq
# from parler_tts import ParlerTTSForConditionalGeneration
import asyncio
import json

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

# Initialize Groq client
groq_client = Groq(api_key='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')

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def send_message(self, message: str, client_id: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(message)

manager = ConnectionManager()

def convertaudio_64(input, output):
    audio = AudioSegment.from_file(input, format="m4a")
    audio = audio.set_frame_rate(16000)
    audio.export(output, format="wav")
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

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            data = json.loads(data)
            
            # if data["type"] == "audio":
            #     # Save base64 audio to temporary file
            #     audio_data = base64.b64decode(data["audio"])
            #     temp_output = f"audios/{client_id}.m4a"
            #     # temp_output = f"temp_output_{client_id}.wav"
                
            #     with open(temp_output, "wb") as f:
            #         f.write(audio_data)


            if data["type"] == "audio":
                # Process the audio
                success = await manager.send_message("Converting audio...", client_id)
                await manager.send_message("Transcribing audio...", client_id)
                success =  convertaudio_64("sample.m4a", "iam.wav")
                if success: 
                    transcription = await transcribe_audio("iam.wav")
                    print(transcription)
                #yet to define
                if transcription:
                    await manager.send_message(f"Transcription: {transcription}", client_id)
                    
                    await manager.send_message("Generating response...", client_id)
                    response = await generate_chat_response(transcription)
                    
                    if response:
                        await manager.send_message(f"Response: {response}", client_id)
                        
                        # Generate TTS response using the external API
                        try:
                            import requests
                            url = "http://[::1]:5002/api/tts"
                            params = {
                                "text": response,
                                "speaker_id": "p374",
                                "style_wav": "",
                                "language_id": ""
                            }
                            
                            tts_response = requests.get(url, params=params)
                            if tts_response.status_code == 200:
                                audio_base64 = base64.b64encode(tts_response.content).decode('utf-8')
                                await manager.send_message(json.dumps({
                                    "type": "audio_response",
                                    "audio": audio_base64
                                }), client_id)
                                break
                        except Exception as e:
                            await manager.send_message(f"Error in TTS: {str(e)}", client_id)
                                
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        manager.disconnect(client_id)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)