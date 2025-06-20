import gradio as gr
import requests
import io
from scipy.io import wavfile

FASTAPI_URL = "http://127.0.0.1:8000/process-and-synthesize"


def process_and_synthesize_gradio(
    audio_filepath, vad_threshold, min_speech_duration_ms, min_silence_duration_ms
):
    """
    This function takes inputs from the UI,
    calls the FastAPI backend, and returns the results to be displayed.
    """

    if audio_filepath is None:
        return "Please upload an audio file to begin.", "", None

    try:

        # prepare params for request
        with open(audio_filepath, "rb") as audio_file:
            files = {"file": ("audio.wav", audio_file, "audio/wav")}
            data = {
                "vad_threshold": vad_threshold,
                "min_speech_duration_ms": min_speech_duration_ms,
                "min_silence_duration_ms": min_silence_duration_ms,
            }
            print(f"Sending request to {FASTAPI_URL} with parameters: {data}")

            response = requests.post(FASTAPI_URL, files=files, data=data)

            # Check if the request was successful
            response.raise_for_status()

        # process the response

        transcript = response.headers.get(
            "X-Transcript", "Transcript not found in response headers."
        )
        reconstructed = response.headers.get(
            "X-Reconstructed", "Reconstructed sentence not found in response headers."
        )
        audio_bytes = response.content

        # To make the audio playable in Gradio, we need to convert the bytes
        sample_rate, audio_data = wavfile.read(io.BytesIO(audio_bytes))

        print("Successfully processed audio and received response.")

        # Return the results to be displayed in the Gradio output components
        return transcript, reconstructed, (sample_rate, audio_data)

    except requests.exceptions.RequestException as e:
        error_message = f"API Request Failed: Could not connect to the backend at {FASTAPI_URL}. Details: {e}"
        print(error_message)
        return error_message, "", None

    except Exception as e:
        error_message = f"An error occurred: {e}"
        print(error_message)
        if "response" in locals() and response.text:
            error_message += f"\n\nBackend Response: {response.text}"
        return error_message, "", None


# web UI
demo_interface = gr.Interface(
    fn=process_and_synthesize_gradio,
    inputs=[
        gr.Audio(type="filepath", label="Upload Audio File"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.4,
            step=0.05,
            label="VAD Threshold",
            info="The sensitivity of the voice activity detection. Higher values are more sensitive.",
        ),
        gr.Slider(
            minimum=50,
            maximum=1000,
            value=250,
            step=10,
            label="Min Speech Duration (ms)",
            info="Minimum duration for a speech segment to be considered valid.",
        ),
        gr.Slider(
            minimum=50,
            maximum=1000,
            value=100,
            step=10,
            label="Min Silence Duration (ms)",
            info="Minimum duration of silence between speech segments.",
        ),
    ],
    outputs=[
        gr.Textbox(label="Original Transcript (from Whisper)"),
        gr.Textbox(label="Reconstructed Sentence (from T5)"),
        gr.Audio(label="Synthesized Audio (from F5 TTS)"),
    ],
    title="VoicePatch Pipeline Interactive Demo",
    description=(
        "Upload an audio file and adjust the VAD parameters to see how they affect the transcription and final synthesized audio."
    ),
    examples=[["sample_audio/output_muted_400.mp3", 0.4, 250, 100]],
    allow_flagging="never",
    theme=gr.themes.Ocean(),
)

if __name__ == "__main__":
    demo_interface.launch()
