from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoTokenizer
import torch
import soundfile as sf
import subprocess
<<<<<<< HEAD
=======
import os
>>>>>>> 0d9eef4 (second or third time it is)
import time
from groq import Groq
import torch
from parler_tts import ParlerTTSForConditionalGeneration



start_time = time.time()

#Converting any audio file to WAV file
def convert_audio_to_wav(input_path, output_path="output.wav", target_sample_rate=16000):
    try:
        # Use FFmpeg command to convert audio
        command = [
            "ffmpeg", "-i", input_path,       # Input file
            "-ar", str(target_sample_rate),   # Set sample rate to 16000 Hz
            "-ac", "1",                       # Set to mono (1 channel)
            output_path                       # Output file
        ]
        subprocess.run(command, check=True)
        print(f"Converted {input_path} to {output_path} at {target_sample_rate} Hz.")
    except subprocess.CalledProcessError as e:
        print("An error occurred during conversion:", e)

# Example usage
<<<<<<< HEAD
input_audio_file = "testing_input.m4a"  # Replace with your audio file path
=======
input_audio_file = "sample.m4a"  # Replace with your audio file path
>>>>>>> 0d9eef4 (second or third time it is)
output_wav_file = "outputFile.wav"       # Desired output path
convert_audio_to_wav(input_audio_file, output_wav_file)


#Transcribing the converted WAV file
def transcribe_audio(audio_path, model_repo):
 
<<<<<<< HEAD
    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
=======
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
>>>>>>> 0d9eef4 (second or third time it is)
    model = WhisperForConditionalGeneration.from_pretrained(model_repo)
    
    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    
    audio_input, sampling_rate = sf.read(audio_path)
    
    # Convert to mono if stereo
    if len(audio_input.shape) > 1:
        audio_input = audio_input.mean(axis=1)
    

    input_features = processor(
        audio_input, 
        sampling_rate=sampling_rate, 
        return_tensors="pt"
    ).input_features
    input_features = input_features.to(device)
    
   
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(
        predicted_ids, 
        skip_special_tokens=True
    )[0]
    
    return transcription


model_repo = "shReYas0363/whisper-tiny-fine-tuned"  # Your HF repository name
audio_file = "outputFile.wav"
transcription = transcribe_audio(audio_file, model_repo)
print(transcription)




#GROQ
client = Groq(
    api_key='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',
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",

<<<<<<< HEAD
"content": f'''Here’s a concise version of the prompt:

---

You are "Saathi," a respectful and polite Indian English voice assistant. Respond in clear, concise Indian English with warmth and formality, using culturally relevant examples when helpful. Answer as short as possible, keeping language simple, polite, and respectful. Confirm steps if guidance is needed. 

**User Query:** "{transcription}"''',
=======
"content": f'''You are an indian medical assistant give a very friendly one line response in Indian English for this query: {transcription} Remember to give a reply in just one reply and in Indian English''',
>>>>>>> 0d9eef4 (second or third time it is)
        }
    ],
    # model="llama3-8b-8192",
    model = "llama-3.2-1b-preview",
)


print(chat_completion.choices[0].message.content)


device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

# prompt = "Namastey sir. how are you doing today?"
prompt = chat_completion.choices[0].message.content
description = "Laura Female Indian voice normal"
input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
audio_arr = generation.cpu().numpy().squeeze()
sf.write("parler_tts_out.wav", audio_arr, model.config.sampling_rate)

end=time.time()
print(f"The total time taken is: {end-start_time}")
<<<<<<< HEAD
=======
os.remove('outputFile.wav')
>>>>>>> 0d9eef4 (second or third time it is)
