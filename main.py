from utils.mask import Mask
from utils.VAD import SpeechVADTranscriber
import warnings

warnings.filterwarnings("ignore")

# this is the initialization of the SpeechVADTranscriber and mask classes
# it loads the Silero VAD model and Whisper model into the memory for transcription
transcriber = SpeechVADTranscriber(
    whisper_model_size="turbo"
)  # "tiny", "base", "small", "medium", "large", "turbo"
mask = Mask(t5_model="base")  # "small", "base", "large", "3b", "11b"

print("Models loaded successfully")

audio_file_paths = [
    "sample_audio/sample.mp3",
    "sample_audio/output_muted.mp3",
    "sample_audio/output_muted_400.mp3",
]

for each in audio_file_paths:
    try:
        # Process the audio file

        transcription = transcriber.process_audio_file(
            each,
            vad_threshold=0.4,  # Adjust sensitivity (0.0-1.0)
            min_speech_duration_ms=250,  # Minimum speech duration
            min_silence_duration_ms=100,  # Minimum silence duration
        )

        print("transcript = ", transcription)

        result = mask.fill_masks(transcription)
        print(f"Reconstructed sentence: {result}")

    except FileNotFoundError:
        print(f"Error: Audio file '{each}' not found.")
        print("Please update the audio_file_path variable with a valid audio file.")
    except Exception as e:
        print(f"Error processing audio: {e}")
