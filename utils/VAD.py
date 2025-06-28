import torch
import torchaudio
import whisper
import numpy as np
from typing import List, Tuple


class SpeechVADTranscriber:
    def __init__(self, whisper_model_size: str = "base"):
        """
        Initialize the SpeechVADTranscriber with Silero VAD and Whisper models.
        Args:
            whisper_model_size (str): Size of the Whisper model to use
                Options: "tiny", "base", "small", "medium", "large", "turbo"
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.vad_model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )
        self.vad_model = self.vad_model.to(self.device)

        self.get_speech_timestamps = utils[0]
        self.save_audio = utils[1]
        self.read_audio = utils[2]
        self.VADIterator = utils[3]
        self.collect_chunks = utils[4]

        self.whisper_model = whisper.load_model(whisper_model_size, device=self.device)

    def load_audio(self, audio_path: str, target_sr: int = 16000) -> torch.Tensor:
        """
        Load audio file and resample to target sample rate.

        Args:
            audio_path: Path to audio file
            target_sr: Target sample rate (Silero VAD expects 16kHz)

        Returns:
            Audio tensor
        """
        waveform, sample_rate = torchaudio.load(audio_path)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
            waveform = resampler(waveform)

        return waveform.squeeze(0)

    def detect_speech_segments(
        self,
        audio: torch.Tensor,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
    ) -> List[dict]:
        """
        Detect speech segments using Silero VAD.

        Args:
            audio: Audio tensor
            sample_rate: Sample rate of audio
            threshold: VAD threshold (0.0 - 1.0)
            min_speech_duration_ms: Minimum speech duration in ms
            min_silence_duration_ms: Minimum silence duration in ms

        Returns:
            List of speech segments with start and end timestamps
        """

        audio = audio.to(self.device)

        speech_timestamps = self.get_speech_timestamps(
            audio,
            self.vad_model,
            threshold=threshold,
            sampling_rate=sample_rate,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            return_seconds=True,
        )

        return speech_timestamps

    def create_masked_audio_segments(
        self, audio: torch.Tensor, speech_segments: List[dict], sample_rate: int = 16000
    ) -> List[Tuple[torch.Tensor, float, float, bool]]:
        """
        Create segments with speech/non-speech labels for transcription.

        Args:
            audio: Original audio tensor
            speech_segments: List of speech segments from VAD
            sample_rate: Sample rate

        Returns:
            List of tuples (audio_segment, start_time, end_time, is_speech)
        """
        segments = []
        audio_duration = len(audio) / sample_rate

        current_time = 0.0

        for speech_seg in speech_segments:
            start_time = speech_seg["start"]
            end_time = speech_seg["end"]

            if current_time < start_time:
                non_speech_start_sample = int(current_time * sample_rate)
                non_speech_end_sample = int(start_time * sample_rate)
                non_speech_audio = audio[non_speech_start_sample:non_speech_end_sample]

                if len(non_speech_audio) > 0:
                    segments.append((non_speech_audio, current_time, start_time, False))

            speech_start_sample = int(start_time * sample_rate)
            speech_end_sample = int(end_time * sample_rate)
            speech_audio = audio[speech_start_sample:speech_end_sample]

            if len(speech_audio) > 0:
                segments.append((speech_audio, start_time, end_time, True))

            current_time = end_time

        if current_time < audio_duration:
            non_speech_start_sample = int(current_time * sample_rate)
            non_speech_audio = audio[non_speech_start_sample:]

            if len(non_speech_audio) > 0:
                segments.append((non_speech_audio, current_time, audio_duration, False))

        return segments

    def transcribe_audio_segment(
        self, audio_segment: torch.Tensor, sample_rate: int = 16000
    ) -> str:
        """
        Transcribe a single audio segment using Whisper.

        Args:
            audio_segment: Audio tensor segment
            sample_rate: Sample rate

        Returns:
            Transcribed text
        """
        # Convert to numpy array for Whisper
        audio_np = audio_segment.cpu().numpy().astype(np.float32)

        # Whisper expects audio to be normalized
        if audio_np.max() > 1.0:
            audio_np = audio_np / np.abs(audio_np).max()
        try:
            result = self.whisper_model.transcribe(audio_np, fp16=False)
            return result["text"].strip()
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""

    def process_audio_file(
        self,
        audio_path: str,
        vad_threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
    ) -> Tuple[str, List[dict]]:
        """
        Main processing function. Combines VAD, transcription, and calculates silent gaps.
        Returns:
            A tuple containing:
            - Final transcription with [MASK] for non-speech segments (str)
            - A list of dictionaries representing the silent gaps (List[dict])
        """
        audio = self.load_audio(audio_path)
        sample_rate = 16000
        audio_duration = len(audio) / sample_rate

        speech_segments = self.detect_speech_segments(
            audio,
            sample_rate,
            threshold=vad_threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
        )

        all_segments = self.create_masked_audio_segments(
            audio, speech_segments, sample_rate
        )

        final_transcription = []
        for i, (audio_segment, start_time, end_time, is_speech) in enumerate(all_segments):
            if is_speech:
                if len(audio_segment) > 0:
                    transcription = self.transcribe_audio_segment(audio_segment, sample_rate)
                    if transcription:
                        final_transcription.append(transcription)
                    else:
                        final_transcription.append("[MASK]")
                else:
                    final_transcription.append("[MASK]")
            else:
                final_transcription.append("[MASK]")

        # --- New Logic to Calculate Silent Gaps ---
        vad_gaps = []
        last_speech_end = 0.0
        for seg in speech_segments:
            # If there's a gap between the last speech end and the current start
            if seg["start"] > last_speech_end:
                vad_gaps.append({"start": last_speech_end, "end": seg["start"]})
            last_speech_end = seg["end"]
        # Check for a final gap after the last speech segment
        if audio_duration > last_speech_end:
            vad_gaps.append({"start": last_speech_end, "end": audio_duration})
        
        # --- End of New Logic ---

        result = " ".join(final_transcription)
        result = result.replace(". [MASK]", " [MASK]").replace(".[MASK]", " [MASK]")

        return result, vad_gaps
