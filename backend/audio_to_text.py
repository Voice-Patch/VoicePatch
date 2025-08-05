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
from transformers import (
    BartForConditionalGeneration,
    T5ForConditionalGeneration,
    T5Tokenizer,
    BartTokenizer,
)

ffmpeg_bin = shutil.which("ffmpeg")


def extract_mask_replacements(masked_text, generated_text):
    """
    Extract replacements for [MASK] tokens by comparing masked and generated text.
    Returns only the actual replacement words, not parts of original text.
    """
    import re

    # Clean and split texts
    masked_words = masked_text.split()
    generated_words = generated_text.split()

    replacements = []
    gen_idx = 0

    for i, masked_word in enumerate(masked_words):
        if masked_word == "[MASK]":
            replacement_words = []

            # Find the next non-mask word in masked text for boundary detection
            next_boundary = None
            for j in range(i + 1, len(masked_words)):
                if masked_words[j] != "[MASK]":
                    next_boundary = masked_words[j].lower().rstrip('.,!?;:"')
                    break

            # Collect replacement words until we hit the boundary or end
            while gen_idx < len(generated_words):
                current_gen_word = generated_words[gen_idx].lower().rstrip('.,!?;:"')

                # If we hit the next boundary word, stop collecting
                if next_boundary and current_gen_word == next_boundary:
                    break

                # Add to replacement
                replacement_words.append(generated_words[gen_idx])
                gen_idx += 1

            # Join replacement words or empty string if none
            replacement = " ".join(replacement_words) if replacement_words else ""
            replacements.append(replacement)

        else:
            # For non-mask words, advance in generated text to find matching word
            target_word = masked_word.lower().rstrip('.,!?;:"')

            # Skip ahead in generated text to find this word
            while gen_idx < len(generated_words):
                current_gen_word = generated_words[gen_idx].lower().rstrip('.,!?;:"')
                if current_gen_word == target_word:
                    gen_idx += 1  # Move past the matched word
                    break
                gen_idx += 1

    return replacements


def analyze_differences(masked_text, generated_text):
    """
    More robust approach: find actual differences between texts
    """
    import re

    # Normalize texts for comparison
    def normalize_word(word):
        return word.lower().rstrip(".,!?;:\"'")

    masked_words = masked_text.split()
    generated_words = generated_text.split()

    replacements = []
    gen_idx = 0

    for mask_idx, masked_word in enumerate(masked_words):
        if masked_word == "[MASK]":
            replacement_parts = []

            # Find what comes after this mask
            next_word_after_mask = None
            for next_idx in range(mask_idx + 1, len(masked_words)):
                if masked_words[next_idx] != "[MASK]":
                    next_word_after_mask = normalize_word(masked_words[next_idx])
                    break

            # Collect words from generated text until we find the next expected word
            while gen_idx < len(generated_words):
                current_gen_normalized = normalize_word(generated_words[gen_idx])

                # Stop if we found the next expected word
                if (
                    next_word_after_mask
                    and current_gen_normalized == next_word_after_mask
                ):
                    break

                replacement_parts.append(generated_words[gen_idx])
                gen_idx += 1

            replacements.append(
                " ".join(replacement_parts) if replacement_parts else ""
            )

        else:
            # Skip matching words in generated text
            masked_normalized = normalize_word(masked_word)

            # Find and skip past this word in generated text
            while gen_idx < len(generated_words):
                gen_normalized = normalize_word(generated_words[gen_idx])
                if gen_normalized == masked_normalized:
                    gen_idx += 1
                    break
                gen_idx += 1

    return replacements


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

audio_file = "evaluations/eval_audio/p_audio.mp3"
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


def predict_masked_words_bart(
    masked_text: str,
    model: BartForConditionalGeneration,
    tokenizer: BartTokenizer,
    *,
    device: str = cuda.type,
):
    # Replace [MASK] tokens with BART's <mask> token
    masked_text_bart = masked_text.replace("[MASK]", "<mask>")

    inputs = tokenizer(masked_text_bart, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=inputs["input_ids"].shape[1] + 50)

    # Decode the generated output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    replacements = analyze_differences(masked_text, generated_text)

    return replacements


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
# t5_model = T5ForConditionalGeneration.from_pretrained("t5-base").to("cuda")
# t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")


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


with open("./evaluations/out1_bart.csv", "w") as csv_f:
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
            predict_masked_words_bart(
                " ".join(words_),
                BartForConditionalGeneration.from_pretrained("facebook/bart-base").to(
                    "cuda"
                ),
                BartTokenizer.from_pretrained("facebook/bart-base"),
            )
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
