import gradio as gr
from transformers import pipeline
import numpy as np
from openai import OpenAI
from elevenlabs.client import ElevenLabs
import tempfile
import apikeys 

client = OpenAI(api_key=apikeys.openaiAPI)
tts_client = ElevenLabs(api_key=apikeys.elevenlabsAPI)


#Initializing ASR model
whisper = pipeline("automatic-speech-recognition", "openai/whisper-large-v3")

#Handling transcription
def transcribe_respond(audio):
    sr, y = audio
    y = y.astype(np.float32)
    y /= np.max(np.abs(y)) #Normalizing

    transcription = whisper({"sampling_rate": sr, "raw": y})["text"]
    print("transcribed text:", transcription)

    #Sending transcript to GPT model and getting a reponse
    response = client.chat.completions.create(
        model = "gpt-3.5-turbo",
        messages=[
            {
                "role": "system", "content": "You are a highly intelligent AI system called Sarah that keeps its answer short and under 1000 characters."},
                { "role": "user", "content": transcription}
        ]
        
    )
    txtResponse =response.choices[0].message.content
    print(txtResponse)

    #Converting response into audio
    audioResponse = b"".join(tts_client.generate(
    text=txtResponse,
    voice="Sarah",
    model="eleven_multilingual_v2"
    ))
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        temp_audio.write(audioResponse)
        tempAudioPath = temp_audio.name

    return tempAudioPath

inter = gr.Interface(
    fn = transcribe_respond,
    inputs=gr.Audio(sources=["microphone"]),
    outputs=gr.Audio(type="filepath", autoplay=True)
)

if __name__ == "__main__":
    inter.launch()