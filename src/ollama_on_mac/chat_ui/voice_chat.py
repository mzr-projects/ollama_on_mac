
import mlx_whisper
import tempfile
import streamlit as st
import os

WHISPER_MODEL = "mlx-community/whisper-small-mlx"

st.title("Voice Chat")
# record the audio input
audio_input = st.audio_input("Record your message")

if audio_input is not None:
    # write the audio input to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio_input.read())
        temp_file_path = f.name
    with st.spinner("Transcribing..."):
        result = mlx_whisper.transcribe(
            temp_file_path, path_or_hf_repo=WHISPER_MODEL
        )
        transcribed_text = result["text"].strip() # type: ignore

    os.unlink(temp_file_path)
    st.success(f"You said: {transcribed_text}")

