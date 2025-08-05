import contextlib
import io
import json
import shutil
import subprocess
import typing as t
import wave

import numpy as np
import pydub
from f5_tts.api import F5TTS

ffprobe_bin = shutil.which("ffprobe")
ffobject_type = t.Union[str, bytes, t.IO[bytes]]
ffmpeg_bin = shutil.which("ffmpeg")


def audio_repr(ffobject: ffobject_type):
    if isinstance(ffobject, str):
        return f"<Source url={ffobject!r}>"

    if isinstance(ffobject, (bytes, bytearray, memoryview)):
        return f"<Source hex={ffobject[:10].hex()!r} size={len(ffobject)}>"

    if ffobject.seekable():
        previous_state = ffobject.tell()
        ffobject.seek(0)
        initial_bytes = ffobject.read(10)
        size = ffobject.seek(0, io.SEEK_END)
        ffobject.seek(previous_state)

        return f"<Source hex={initial_bytes.hex()!r} size={size} in-memory binary>"

    return "<Source unseekable in-memory binary>"


def parse_audio(ffobject: ffobject_type):
    pipe_options = {"stdout": subprocess.PIPE}

    if isinstance(ffobject, str):
        url = ffobject
    else:
        url = "pipe:0"
        pipe_options["stdin"] = subprocess.PIPE

    process = subprocess.Popen(
        [
            ffprobe_bin,
            "-v",
            "quiet",
            "-i",
            url,
            "-show_streams",
            "-select_streams",
            "a",
            "-show_format",
            "-show_optional_fields",
            "1",
            "-of",
            "json=compact=1",
        ],
        **pipe_options,
    )

    if isinstance(ffobject, str):
        stdout, _ = process.communicate()
    else:
        if isinstance(ffobject, bytes):
            stdout, _ = process.communicate(ffobject)
        else:
            previous_state = ffobject.tell()
            stdout, _ = process.communicate(ffobject.read())
            if ffobject.seekable():
                ffobject.seek(previous_state)

    if stdout:
        return json.loads(stdout)


class FFmpegAudio:
    """
    Audio class created specifically for conversions with ffmpeg
    """

    def __init__(
        self,
        audio: ffobject_type,
        audio_stream_index: int,
        codec_name: str,
        codec_long_name: str,
        sample_rate: int,
        channels: int,
        start_time: float,
        duration: float,
        bit_rate: int,
    ):
        self.__audio = audio
        self.__audio_stream_index = audio_stream_index

        self.codec_name = codec_name
        self.codec_long_name = codec_long_name
        self.sample_rate = sample_rate
        self.channels = channels
        self.start_time = start_time
        self.duration = duration
        self.bit_rate = bit_rate

    @property
    def audio(self):
        return self.__audio

    @contextlib.contextmanager
    def raw_reader(
        self, codec: str = "f32le", sample_rate: int = 16000, channels: int = 1
    ) -> t.Generator[t.IO[bytes], None, int]:
        """
        Load a reader that automatically handles transcoding audio
        to the default audio format supported by `whisper` for the
        most ideal results.

        >>> import numpy as np
        >>> with Audio.from_audio_object(...).raw_reader() as reader:
        ...     audio = np.frombuffer(reader.read(), np.float32).flatten()
        """
        pipe_options = {"stdout": subprocess.PIPE}

        if isinstance(self.__audio, str):
            url = self.__audio
        else:
            url = "pipe:0"
            pipe_options["stdin"] = subprocess.PIPE

        if codec in {
            "s16le",
            "s32le",
            "s8",
            "u8",
            "u16le",
            "u32le",
            "f32le",
            "f64le",
        }:
            acodec = f"pcm_{codec}"
        else:
            acodec = None

        args = [
            ffmpeg_bin,
            "-v",
            "quiet",
            "-i",
            url,
            "-map",
            f"0:a:{self.__audio_stream_index}",
            "-f",
            codec,
            *(("-acodec", acodec) if acodec is not None else ()),
            "-vn",
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            "pipe:1",
        ]

        process = subprocess.Popen(args, **pipe_options)

        if isinstance(self.__audio, str):
            ...
        else:
            if isinstance(self.__audio, bytes):
                process.stdin.write(self.__audio)
            else:
                previous_state = self.__audio.tell()
                process.stdin.write(self.__audio.read())
                if self.__audio.seekable():
                    self.__audio.seek(previous_state)

        yield process.stdout
        status_code = process.wait()
        return status_code

    @classmethod
    def from_audio_object(cls, ffobject: ffobject_type, *, audio_stream_index: int = 0):
        parsed_data = parse_audio(ffobject)

        if parsed_data is None:
            raise ValueError("Invalid object received: %s", audio_repr(ffobject))

        if not parsed_data["streams"]:
            raise ValueError(
                "No audio streams available in the object: %s", audio_repr(ffobject)
            )

        if audio_stream_index > len(parsed_data["streams"]):
            raise ValueError(
                "No audio streams available at index %d in the object: %s",
                audio_stream_index,
                audio_repr(ffobject),
            )

        stream = parsed_data["streams"][audio_stream_index]

        return cls(
            ffobject,
            audio_stream_index,
            stream["codec_name"],
            stream["codec_long_name"],
            int(stream["sample_rate"]),
            int(stream["channels"]),
            float(stream["start_time"] if stream["start_time"] != "N/A" else 0),
            float(stream["duration"]),
            int(stream["bit_rate"]),
        )


class StandardAudio:
    STANDARD_SR = 16000
    STANDARD_CHANNELS = 1

    def __init__(self, audio: pydub.AudioSegment):
        self.__audio = audio

        self.__f5: t.Optional["F5TTS"] = None

    def infer(
        self,
        f5_instance: F5TTS,
        text: str,
        *,
        reference_text: t.Optional[str] = None,
        transcription_model: str = "base",
    ):
        from f5_tts.infer.utils_infer import chunk_text, infer_batch_process

        if reference_text is None:
            whisper_transcription = self.transcribe(transcription_model)["text"]
            reference_text: str = whisper_transcription.strip()

        sr = self.__audio.frame_rate
        audio = self.as_tensor_f32le.unsqueeze(0)

        max_chars = round(
            len(reference_text.encode("utf-8"))
            / (audio.shape[-1] / sr)
            * (22 - audio.shape[-1] / sr)
        )

        wav, sr, _ = next(
            infer_batch_process(
                (audio, sr),
                reference_text,
                chunk_text(text, max_chars=max_chars),
                f5_instance.ema_model,
                f5_instance.vocoder,
                device=f5_instance.device,
            )
        )

        wav *= 32768.0
        wav = wav.astype(np.int16)

        return wav, sr

    @classmethod
    def from_ffmpeg_audio(cls, ffmpeg_audio_class: FFmpegAudio):
        if isinstance(ffmpeg_audio_class.audio, str):
            segment: pydub.AudioSegment = pydub.AudioSegment.from_file(
                ffmpeg_audio_class.audio
            )
            return cls(
                segment.set_frame_rate(cls.STANDARD_SR).set_channels(
                    cls.STANDARD_CHANNELS
                )
            )

        with ffmpeg_audio_class.raw_reader(
            codec="wav", sample_rate=cls.STANDARD_SR, channels=cls.STANDARD_CHANNELS
        ) as raw_reader:
            return cls(pydub.AudioSegment(raw_reader.read()))

    def with_slience_at(self, start_at: float, duration: float):
        return StandardAudio(
            pydub.AudioSegment(
                self.__audio[: round(start_at * 1000)].raw_data
                + b"\x00"
                * self.__audio.sample_width
                * (round(duration * self.__audio.frame_rate))
                + self.__audio[round((start_at + duration) * 1000) :].raw_data,
                metadata={
                    "channels": self.__audio.channels,
                    "sample_width": self.__audio.sample_width,
                    "frame_rate": self.__audio.frame_rate,
                    "frame_width": self.__audio.frame_width,
                },
            ),
        )

    def buffer(self) -> bytes:
        # Convert with `ffmpeg -f s16le -ar 16000 -ac 1 -i .\raw.pcm_s16le raw.mp3`
        # Or, just use .as_wav() method for wav audio.
        return self.__audio.raw_data

    def as_wav(self, file: t.IO[bytes]):
        if not file.writable():
            raise ValueError("File object is not writable.")

        with wave.open(file, "wb") as wf:
            wf.setsampwidth(self.__audio.sample_width)
            wf.setnchannels(self.__audio.channels)
            wf.setframerate(self.__audio.frame_rate)
            wf.setnframes(int(self.__audio.frame_count()))
            wf.writeframesraw(self.buffer())

        if file.seekable():
            file.seek(0)

    @property
    def as_tensor_f32le(self):
        import torch

        return torch.from_numpy(self.as_numpy_f32le)

    def export(self, *args, **kwargs):
        return self.__audio.export(*args, **kwargs)

    def slice_at(self, start: float, end: float):
        return StandardAudio(self.__audio[round(start * 1000) : round(end * 1000)])

    @property
    def as_numpy_s16le(self):
        return np.frombuffer(self.buffer(), dtype=np.int16).flatten()

    @property
    def as_numpy_f32le(self):
        return self.as_numpy_s16le.astype(np.float32) / 32768.0

    def transcribe(self, model: str = "base"):
        import whisper_timestamped as whisper

        return whisper.transcribe_timestamped(
            whisper.load_model(model, device="cuda", download_root="./models/"),
            self.as_numpy_f32le,
        )


if __name__ == "__main__":
    audio = StandardAudio(pydub.AudioSegment.from_wav("..."))

    # audio.configure_f5_tts(vocoder_local_path="./models/vocos/")
    f5_instance = F5TTS(model="E2TTS_Base", vocoder_local_path="./models/vocos/")

    wav, sr = audio.infer(
        "Maybe someday, someday maybe.",
    )

    with wave.open("./output.wav", "wb") as wf:
        wf.setframerate(sr)
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.writeframes(wav.tobytes())
