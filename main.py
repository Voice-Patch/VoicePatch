import warnings
import wave
import io
import os
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Audio Processing API is running"}


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

    try:
        file_content = await file.read()

        file_extension = os.path.splitext(file.filename)[1] if file.filename else ".mp3"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        temp_filename = f"temp_audio_{timestamp}{file_extension}"
        temp_file_path = os.path.join("temp", temp_filename)

        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(file_content)

        try:
            transcription = transcriber.process_audio_file(
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
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")


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
            transcription = transcriber.process_audio_file(
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
