import warnings
import wave
import base64
import io
import os
import asyncio
from datetime import datetime
from contextlib import asynccontextmanager
from pydub import AudioSegment

from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
    UploadFile,
    File,
    HTTPException,
)
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from f5_tts.api import F5TTS
from utils.audiotools import FFmpegAudio, StandardAudio
from utils.mask import Mask
from utils.VAD import SpeechVADTranscriber

warnings.filterwarnings("ignore")

# Global variables to store models
transcriber = None
mask = None
f5_tts_instance = None


# Pydantic models for request/response
class ProcessAudioRequest(BaseModel):
    vad_threshold: float = 0.4
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 100


class ProcessAudioResponse(BaseModel):
    transcript: str
    reconstructed_sentence: str
    message: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup - Initialize models
    global transcriber, mask, f5_tts_instance

    # Initialize the SpeechVADTranscriber and mask classes
    # Loads the Silero VAD model and Whisper model into memory for transcription
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("temp", exist_ok=True)

    transcriber = SpeechVADTranscriber(
        whisper_model_size="turbo"
    )  # "tiny", "base", "small", "medium", "large", "turbo"

    mask = Mask(t5_model="base")  # "small", "base", "large", "3b", "11b"

    f5_tts_instance = F5TTS(model="E2TTS_Base")

    print("Models loaded successfully")

    yield


# Create FastAPI app with lifespan
app = FastAPI(
    title="Audio Processing API",
    description="API for audio transcription, mask filling, and TTS synthesis",
    version="1.0.0",
    lifespan=lifespan,
)

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Mount static directory for serving generated files ---
os.makedirs("outputs", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")


# --- HTTP Endpoints ---


@app.get("/")
async def read_root():
    return {"status": "ok", "message": "VoicePatch backend is running."}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": all([transcriber, mask, f5_tts_instance]),
    }


@app.post("/process-audio", response_model=ProcessAudioResponse)
async def process_audio_file(
    file: UploadFile = File(...),
    vad_threshold: float = 0.4,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
):
    """
    Process an uploaded audio file: transcribe, fill masks, and return results
    """
    if not all([transcriber, mask, f5_tts_instance]):
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    temp_file_path = None
    try:
        file_content = await file.read()

        file_extension = os.path.splitext(file.filename)[1] if file.filename else ".mp3"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        temp_filename = f"temp_audio_{timestamp}{file_extension}"
        temp_file_path = os.path.join("temp", temp_filename)

        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(file_content)

        transcription, _ = transcriber.process_audio_file(
            temp_file_path,
            vad_threshold=vad_threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
        )

        reconstructed_sentence = mask.fill_masks(transcription)

        return ProcessAudioResponse(
            transcript=transcription,
            reconstructed_sentence=reconstructed_sentence,
            message="Audio processed successfully",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.post("/process-and-synthesize")
async def process_and_synthesize_audio(
    file: UploadFile = File(...),
    vad_threshold: float = 0.4,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
):
    """
    Process an uploaded audio file and return synthesized audio
    """
    if not all([transcriber, mask, f5_tts_instance]):
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    temp_file_path = None
    try:
        file_content = await file.read()

        file_extension = os.path.splitext(file.filename)[1] if file.filename else ".mp3"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[
            :-3
        ]  # Include milliseconds
        temp_filename = f"temp_audio_{timestamp}{file_extension}"
        temp_file_path = os.path.join("temp", temp_filename)

        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(file_content)

        try:
            transcription, _ = transcriber.process_audio_file(
                temp_file_path,
                vad_threshold=vad_threshold,
                min_speech_duration_ms=min_speech_duration_ms,
                min_silence_duration_ms=min_silence_duration_ms,
            )

            reconstructed_sentence = mask.fill_masks(transcription)

            audio = StandardAudio.from_ffmpeg_audio(
                FFmpegAudio.from_audio_object(temp_file_path)
            )
            wave_bytes, sr = audio.infer(f5_tts_instance, reconstructed_sentence)

            output_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            original_filename = (
                os.path.splitext(file.filename)[0] if file.filename else "audio"
            )
            output_filename = f"{original_filename}_synthesized_{output_timestamp}.wav"
            output_path = os.path.join("outputs", output_filename)

            with wave.open(output_path, "wb") as wf:
                wf.setframerate(sr)
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.writeframes(wave_bytes.tobytes())

            print(f"Synthesized audio saved to: {output_path}")

            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, "wb") as wf:
                wf.setframerate(sr)
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.writeframes(wave_bytes.tobytes())

            wav_buffer.seek(0)

            return StreamingResponse(
                wav_buffer,
                media_type="audio/wav",
                headers={
                    "Content-Disposition": f"attachment; filename={output_filename}",
                    "X-Saved-File": output_path,
                    "X-Transcript": transcription,
                    "X-Reconstructed": reconstructed_sentence,
                },
            )
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)


# --- WebSocket Endpoint for Real-time Processing ---
@app.websocket("/ws/process")
async def process_audio_websocket(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_json(
        {"step": "engine_ready", "message": "Backend models loaded and ready."}
    )
    print("Client connected via WebSocket, engine ready signal sent.")
    try:
        while True:
            payload = await websocket.receive_json()
            temp_audio_path = None

            try:
                audio_data_b64 = payload.get("audio_data")
                file_name = payload.get("file_name")
                vad_params = payload.get("vad_params", {})

                header, encoded = audio_data_b64.split(",", 1)
                audio_bytes = base64.b64decode(encoded)

                original_extension = os.path.splitext(file_name)[1]

                in_memory_file = io.BytesIO(audio_bytes)

                # Load the audio from the in-memory file, explicitly setting the format
                audio_segment = AudioSegment.from_file(
                    in_memory_file, format=original_extension.replace(".", "")
                )

                # --- Export as a standardized WAV file for your ML models ---
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                temp_audio_path = os.path.join("temp", f"temp_audio_{timestamp}.wav")

                # Ensure it's mono, 16kHz, and 16-bit for your models
                audio_segment = audio_segment.set_channels(1)
                audio_segment = audio_segment.set_frame_rate(16000)
                audio_segment = audio_segment.set_sample_width(2)

                audio_segment.export(temp_audio_path, format="wav")

                print(f"Successfully converted {file_name} to {temp_audio_path}")

                # Stage 1: Transcription
                print(f"WS Step 1: Starting transcription on {temp_audio_path}...")

                transcript_result, vad_gaps = transcriber.process_audio_file(
                    temp_audio_path,
                    vad_threshold=float(vad_params.get("threshold", 0.5)),
                    min_speech_duration_ms=int(vad_params.get("minSpeechMs", 250)),
                    min_silence_duration_ms=int(vad_params.get("minSilenceMs", 100)),
                )
                await websocket.send_json(
                    {
                        "step": "transcription",
                        "data": {"transcript": transcript_result, "vadGaps": vad_gaps},
                    }
                )
                print(f"WS Step 1: Transcription complete. Sent: {transcript_result}")
                await asyncio.sleep(0.1)

                # Stage 2: Reconstruction
                print("WS Step 2: Starting reconstruction...")
                reconstructed_sentence = mask.fill_masks(transcript_result)
                original_words = transcript_result.split()
                final_words = reconstructed_sentence.split()
                reconstructed_only_words = " ".join(
                    [word for word in final_words if word not in original_words]
                )
                await websocket.send_json(
                    {
                        "step": "reconstruction",
                        "data": {
                            "reconstructed_words": reconstructed_only_words,
                            "full_reconstructed_text": reconstructed_sentence,
                        },
                    }
                )
                print(
                    f"WS Step 2: Reconstruction complete. Sent: {reconstructed_sentence}"
                )
                await asyncio.sleep(0.1)

                # Stage 3: Synthesis
                print("WS Step 3: Starting synthesis...")
                audio = StandardAudio.from_ffmpeg_audio(
                    FFmpegAudio.from_audio_object(temp_audio_path)
                )
                wave_bytes, sr = audio.infer(f5_tts_instance, reconstructed_sentence)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"synthesized_{timestamp}.wav"
                output_path = os.path.join("outputs", output_filename)

                with wave.open(output_path, "wb") as wf:
                    wf.setframerate(sr)
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.writeframes(wave_bytes.tobytes())

                await websocket.send_json(
                    {
                        "step": "synthesis",
                        "data": {
                            "audio_url": f"/{output_path.replace(os.path.sep, '/')}",
                            "synthesisFilename": output_filename,
                        },
                    }
                )
                print(f"WS Step 3: Synthesis complete. File saved to {output_path}")
                print("WS processing finished. Waiting for next request.")

            finally:
                if temp_audio_path and os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)
                    print(f"Cleaned up WebSocket temporary file: {temp_audio_path}")

    except WebSocketDisconnect:
        print("WebSocket client disconnected.")
    except Exception as e:
        print(f"An error occurred in WebSocket: {e}")
        if not websocket.client_state.name == "DISCONNECTED":
            await websocket.send_json({"error": str(e)})
