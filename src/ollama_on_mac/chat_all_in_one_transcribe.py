import pathlib

import mlx_whisper
import numpy
import numpy as np
import sounddevice
import scipy.io.wavfile as wavfile
import tempfile
import os
import argparse
from pathlib import Path

SAMPLE_RATE = 16000
DURATION = 5 
OUTPUT_FILE = pathlib.Path(__file__).with_name("output.wav")
WHISPER_MODEL = "mlx-community/whisper-small-mlx"

print("mlx-whisper:",mlx_whisper.__version__)
print("sound device:",sounddevice.__version__)
print("numpy:",numpy.__version__)

def record_audio(duration: int = DURATION, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    print(f"Recording {duration} seconds of audio to {OUTPUT_FILE.name}...")
    audio = sounddevice.rec(
         duration * sample_rate,
         samplerate=sample_rate, 
         channels=1, 
         dtype="float32"
    )
    sounddevice.wait()
    print("Recording complete.")
    return audio.squeeze()


def transcribe_audio(audio: np.ndarray, model: str = WHISPER_MODEL,sample_rate: int = SAMPLE_RATE) -> str:
    # write the audio to a temporary file
    # the audio is a numpy array with shape (duration * sample_rate)
    # we need to write it to a file so that the model can use it
   with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
      temp_path = f.name
      wavfile.write(temp_path, sample_rate, (audio * 32767).astype(np.int16))
   try:
    result = mlx_whisper.transcribe(temp_path, path_or_hf_repo=model)
    return result["text"].strip() # type: ignore
   finally:
    os.unlink(temp_path)

def transcribe_file(path: str, model: str = WHISPER_MODEL) -> str:
    result = mlx_whisper.transcribe(path, path_or_hf_repo=model)
    return result["text"].strip() # type: ignore 

def record_and_transcribe(duration: int = DURATION, model: str = WHISPER_MODEL) -> str:
    audio = record_audio(duration)
    return transcribe_audio(audio, model)   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record and transcribe audio")
    parser.add_argument("--duration", type=int, default=DURATION, help="Duration of the audio to record")
    parser.add_argument("--model", type=str, default=WHISPER_MODEL, help="Model to use for transcription")
    args = parser.parse_args()
    text = record_and_transcribe(args.duration, args.model)
    print(f"\nTranscript:\n{text}")