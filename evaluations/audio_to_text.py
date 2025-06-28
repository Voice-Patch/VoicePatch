import csv
import json
import os
import random
import re
import shutil
import subprocess
import typing as t

import faster_whisper
import numpy as np
import torch
from silero_vad.utils_vad import get_speech_timestamps
from transformers import T5ForConditionalGeneration, T5Tokenizer

ffmpeg_bin = shutil.which("ffmpeg")


def iter_audio_raw(file: str, *, channels=1, sample_rate=16000, outf: str = "s16le"):
    args = [
        ffmpeg_bin,
        "-v",
        "quiet",
        "-i",
        file,
        "-f",
        outf,
        "-vn",
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "pipe:1",
    ]

    proc = subprocess.Popen(args, stdout=subprocess.PIPE)

    return_status: t.Optional[bool] = yield from proc.stdout

    if return_status:
        yield proc.wait()


torch.hub.set_dir("./models/torch_hub/")

cuda = torch.device("cuda")

vad_model, _ = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False,
    onnx=False,
)

vad_model = vad_model.to(cuda)

audio_file = "eval_audio/vp_audio.mp3"
audio_np = np.frombuffer(
    b"".join(iter_audio_raw(audio_file, outf="f32le")), dtype=np.float32
)


def clean_word(word):
    # Convert to lowercase
    word_lower = word.lower()

    # Remove all non-letter characters
    cleaned = re.sub(r"[^a-z \',]", "", word_lower)

    # Handle special cases
    if cleaned == "":
        if word.strip() == ".":
            return ""
        elif word.strip() == "":
            return ""
        else:
            return ""  # Any other case that results in empty string

    return cleaned


def clean_word_list(word_list):
    return [clean_word(word) for word in word_list]


def predict_masked_words_t5(
    masked_text: str,
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    *,
    device: str = cuda.type,
):
    counter = 0
    while "[MASK]" in masked_text:
        masked_text = masked_text.replace("[MASK]", f"<extra_id_{counter}>", 1)
        counter += 1

    inputs = tokenizer(masked_text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(**inputs)

    generated_tokens = re.split(
        r"<extra_id_\d+>",
        (
            tokenizer.decode(outputs[0], skip_special_tokens=False)
            .replace("<pad>", "")
            .replace("</s>", "")
            .replace("<s>", "")
        ).strip(),
    )

    return [t for t in map(str.strip, generated_tokens) if t][:counter]


def randomly_scattered_safe(start: int, end: int, k: int, min_distance: int):
    if k == 0:
        return []

    required_span = (k - 1) * (min_distance + 1)
    available_span = end - start
    if available_span < required_span:
        """
        We can't fit k points with that distance but will try our best.
        """

    points = []

    for _ in range(k):
        valid_choices = [
            num
            for num in range(start, end)
            if all(abs(num - p) > min_distance for p in points)
        ]

        if valid_choices:
            new_point = random.choice(valid_choices)
            points.append(new_point)

    return points


os.environ["HF_HOME"] = "./hf_cache/"


audio_np = np.frombuffer(
    b"".join(iter_audio_raw(audio_file, outf="f32le")), dtype=np.float32
)


whisper_model = faster_whisper.WhisperModel(
    "large",
    "cuda",
    download_root="./models/",
)
t5_model = T5ForConditionalGeneration.from_pretrained("t5-base").to("cuda")
t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")


def iter_audio_slices(
    audio_np: np.ndarray,
    *,
    device: torch.device = cuda,
    keep_durations_above: float = 10.0,
):
    audio_tensor = torch.Tensor(audio_np).to(device)

    cache = np.array([], dtype=np.float32)

    for speech_timestamp in get_speech_timestamps(audio_tensor, vad_model):
        segment = audio_np[speech_timestamp["start"] : speech_timestamp["end"]]
        cache = np.concatenate((cache, segment))

        if len(cache) / 16000 > keep_durations_above:
            yield cache

            cache = np.array([], dtype=np.float32)


with open("./out.csv", "w") as csv_f:
    writer = csv.writer(csv_f)

    writer.writerow(
        (
            "Original",
            "Masked",
            "Replacement Masks",
            "LLM Replaced masks",
            "LLM Replaced text",
        )
    )

    for audio_slice in iter_audio_slices(audio_np):
        segments, _ = whisper_model.transcribe(audio_slice, "en", word_timestamps=True)

        words = []

        for segment in segments:
            words.extend(word.word.strip() for word in segment.words)

        if len(words) < 10:
            continue

        set_masks = {}

        *masks, null_mask = randomly_scattered_safe(
            start=0, end=len(words), k=3, min_distance=5
        )

        words_ = words.copy()

        if random.random() < 1 / 5:
            words_.insert(null_mask, "[MASK]")
            set_masks[null_mask] = ""
        else:
            masks.append(null_mask)

        for n, mask in enumerate(masks):
            if words_[mask] == "[MASK]":
                set_masks[mask] = ""
            else:
                set_masks[mask] = words_[mask]

            words_[mask] = "[MASK]"

        set_masks_items = list(set_masks.items())
        set_masks_items.sort(key=lambda x: x[0])

        original_script = " ".join(words)
        masked_script = " ".join(words_)

        INIT_replaced_words = [v for _, v in set_masks_items]
        INIT_replacement_words = list(
            predict_masked_words_t5(" ".join(words_), t5_model, t5_tokenizer)
        )

        replaced_words = clean_word_list(INIT_replaced_words)
        replacement_words = clean_word_list(INIT_replacement_words)

        llm_output = " ".join(words_)

        for item in replacement_words:
            llm_output = llm_output.replace("[MASK]", item, 1)

        writer.writerow(
            (
                original_script,
                masked_script,
                json.dumps(replaced_words),
                json.dumps(replacement_words),
                llm_output,
            )
        )

        csv_f.flush()
