
from torch import mode
import mlx_whisper
import tempfile
import streamlit as st
import os
import ollama

WHISPER_MODEL = "mlx-community/whisper-small-mlx"
OLLAMA_MODEL = "llama3.1:8b"

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

    if not transcribed_text:
        st.warning("No text was transcribed. Please try again.")
    else:
        st.session_state.messages.append({"role": "user", "content": transcribed_text})
        with st.chat_message("user"):
            st.markdown(transcribed_text)
        with st.chat_message("assistant"):
            response_text=""
            placeholder_text=st.empty()
            stream = ollama.chat(model=OLLAMA_MODEL, messages=st.session_state.messages,stream=True)
        for chunk in stream:
            response_text += chunk["message"]["content"]
            placeholder_text.markdown(response_text + "▌")
        placeholder_text.markdown(response_text)
    
        st.session_state.messages.append({"role": "assistant", "content": response_text})

        

