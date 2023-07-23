import streamlit as st
import stable_whisper
import json
import torch
import soundfile as sf
from io import BytesIO

# Create a dropdown to select the model
model_name = st.selectbox("Select a model", ["base", "small", "medium", "large", "large-v2"])

# Load the selected model
model = stable_whisper.load_model(model_name)

# Create a file uploader for the audio file
audiofile = st.file_uploader("Upload an audio file", type=["mp3", "wav"])

# Create a button to run the prediction
if st.button('Transcribe'):
    if audiofile is not None:
        # Read the audio file into a numpy array
        audio_data, _ = sf.read(BytesIO(audiofile.read()))
        # Convert the audio data to float
        audio_data = torch.from_numpy(audio_data).float()
        # Transcribe the audio file
        result = model.transcribe(audio_data)
        # Convert the result to JSON and display it
        if isinstance(result, stable_whisper.WhisperResult):
            result_json = result.to_dict()  # replace with actual method if exists
        else:
            result_json = json.loads(result)
        st.json(result_json)
    else:
        st.write("Please upload an audio file.")