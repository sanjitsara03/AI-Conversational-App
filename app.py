import gradio as gr
from transformers import pipeline
import numpy as np
from openai import OpenAI
import elevenlabs
import tempfile
import apikeys 

client = OpenAI(api_key=apikeys.openaiAPI)
elevenlabs.set_api_key(apikeys.elevenlabsAPI)