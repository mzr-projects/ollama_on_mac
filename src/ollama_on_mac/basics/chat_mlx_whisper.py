import pathlib

import mlx_whisper
import numpy
import numpy as np
import sounddevice
import scipy.io.wavfile as wavfile

print("mlx-whisper:",mlx_whisper.__version__)
print("sound device:",sounddevice.__version__)
print("numpy:",numpy.__version__)

SAMPLE_RATE = 16000 #Hz
DURATION = 5 #seconds
OUTPUT_FILE = pathlib.Path(__file__).with_name("output.wav")

def record_audio(duration: int = DURATION, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    print(f"Recording {duration} seconds of audio to {OUTPUT_FILE.name}...")
    # capture audio from the microphone
    # duration * sample_rate is the number of samples to capture
    # samplerate is the sample rate of the audio
    # channels is the number of channels of the audio
    # dtype is the data type of the audio
    audio = sounddevice.rec(
         duration * sample_rate,
         samplerate=sample_rate, 
         channels=1, 
         dtype="float32"
    )
    sounddevice.wait()
    print("Recording complete.")
    # squeeze the audio to remove the extra dimension
    # the audio is a 2D array with shape (1, duration * sample_rate)
    # we want to remove the first dimension
    # so that the audio is a 1D array with shape (duration * sample_rate)
    return audio.squeeze()

def save_audio(audio: np.ndarray, output_file: pathlib.Path,sample_rate: int = SAMPLE_RATE) -> None:
    print(f"Saving a float32 audio with sample rate {sample_rate} to {output_file.name}...")
    # convert the audio to int16
    # 32767 is the maximum value of the int16 data type
    # we multiply the audio by 32767 to convert it to int16
    # we then cast the audio to int16
    audui_int16 = (audio * 32767).astype(np.int16)
    # write the audio to a wav file
    # sample_rate is the sample rate of the audio
    # output_file is the path to the wav file
    wavfile.write(output_file, sample_rate, audui_int16)
    print("Saving complete.")

if __name__ == "__main__":
    audio = record_audio(DURATION)
    save_audio(audio, OUTPUT_FILE, SAMPLE_RATE)
