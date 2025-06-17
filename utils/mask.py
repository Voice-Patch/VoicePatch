import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    T5Tokenizer,
    T5ForConditionalGeneration,
)
import re


class Mask:
    def __init__(self, t5_model: str) -> None:
        """
        Args:
            t5_model (str): The T5 model to use "small", "base", "large", "3b", "11b".
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.tokenizer = T5Tokenizer.from_pretrained(f"t5-{t5_model}")
        self.model = T5ForConditionalGeneration.from_pretrained(f"t5-{t5_model}")

        self.model = self.model.to(self.device)

    def convert_masks_to_t5_format(self, text: str) -> str:
        """
        Convert [MASK] tokens to T5's expected <extra_id_N> format
        """
        mask_count = 0

        def replace_mask(match):
            nonlocal mask_count
            replacement = f"<extra_id_{mask_count}>"
            mask_count += 1
            return replacement

        converted_text = re.sub(r"\[MASK\]", replace_mask, text)
        return converted_text

    def generate(self, sentence: str) -> str:
        inputs = self.tokenizer(sentence, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model.generate(
            **inputs, max_new_tokens=50, num_beams=4, early_stopping=True
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=False)

    def reconstruct_text(self, original_text: str, generated_text: str) -> str:
        """Reconstruct the original text with filled masks"""
        import re

        cleaned_generated = (
            generated_text.replace("<pad>", "").replace("</s>", "").strip()
        )

        mask_pattern = r"<extra_id_(\d+)>"
        # index number of that pattern
        original_masks = re.findall(mask_pattern, original_text)

        if not original_masks:
            return original_text

        generated_parts = re.split(r"<extra_id_\d+>", cleaned_generated)

        # list of generated words
        fills = [part.strip() for part in generated_parts if part.strip()]

        # Reconstruct the sentence
        reconstructed = original_text

        for i, mask_num in enumerate(original_masks):
            mask_token = f"<extra_id_{mask_num}>"
            if i < len(fills):
                reconstructed = reconstructed.replace(mask_token, fills[i], 1)

        return reconstructed

    def fill_masks(self, sentence: str) -> str:
        # Convert [MASK] to T5 format first
        t5_formatted = self.convert_masks_to_t5_format(sentence)
        # print(f"Original: {sentence}")
        # print(f"T5 formatted: {t5_formatted}")

        generated = self.generate(t5_formatted)
        # print(f"Generated: {generated}")

        reconstructed = self.reconstruct_text(t5_formatted, generated)

        return reconstructed
