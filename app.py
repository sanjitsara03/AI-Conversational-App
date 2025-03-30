import gradio as gr
from transformers import pipeline
import numpy as np
from openai import OpenAI
import elevenlabs 
import tempfile
import apikeys 

client = OpenAI(api_key=apikeys.openaiAPI)
elevenlabs.api_key = apikeys.elevenlabsAPI


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
    response = client.chat.completion.create(
        model = "gpt-4o-mini",
        messages=[
            {
                "role": "system", "content": "You are a highly intelligent AI system, that keeps its answer short and max 1000 characters"},
                { "role": "user", "content": transcription}
        ]
        
    )
    txtResponse =response.choices[0].message.content
    print(txtResponse)

    #Converting response into audio
    audioResponse = elevenlabs.generate(text=txtResponse, voice = "Rachel")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        temp_audio.write(audioResponse)
        tempAudioPath = temp_audio.name
    return tempAudioPath

inter = gr.Interface(
    fn = transcribe_respond,
    inputs=gr.Audio(sources=["microphone"]),
    outputs=gr.Audio(sources=["upload"])
)

if __name__ == "__main":
    inter.launch()