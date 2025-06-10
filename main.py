from VAD import SpeechVADTranscriber

# this is the initialization of the SpeechVADTranscriber class
# it loads the Silero VAD model and Whisper model into the memory for transcription
transcriber = SpeechVADTranscriber(whisper_model_size="turbo")

audio_file_paths = [
    "sample.mp3",
    "output_muted.mp3",
    "output_muted_400.mp3",
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

        print("FINAL TRANSCRIPTION:")
        print(transcription)

    except FileNotFoundError:
        print(f"Error: Audio file '{each}' not found.")
        print("Please update the audio_file_path variable with a valid audio file.")
    except Exception as e:
        print(f"Error processing audio: {e}")
