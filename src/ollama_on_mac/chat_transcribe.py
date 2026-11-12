import pathlib
import mlx_whisper
from pathlib import Path

AUDIO_FILE = pathlib.Path(__file__).with_name("output.wav")
WHISPER_MODEL = "mlx-community/whisper-small-mlx"

def transcribe_audio(audio_path: Path  , model: str = WHISPER_MODEL) -> str:
    if not Path(audio_path).is_file():
        raise FileNotFoundError(f"Audio file {audio_path} not found")
    print(f"Transcribing audio from {audio_path} with model {model}...")
    # path_or_hf_repo is the path to the model or the name of the model on the Hugging Face Hub
    result   = mlx_whisper.transcribe(str(audio_path), path_or_hf_repo=model)
    return result["text"].strip() # type: ignore

if __name__ == "__main__":
    text = transcribe_audio(AUDIO_FILE)
    print(f"\nTranscript:\n{text}")