"""GravityOCR-KO inference. Usage: python ocr.py <image> [text|table|formula]"""
import sys
import torch
from transformers import AutoProcessor, GlmOcrForConditionalGeneration

REPO = "trillionlabs/GravityOCR-KO"
# ponytail: GLM-OCR task prompts; the KO card documents none, so these are inherited from the base model
PROMPTS = {"text": "Text Recognition:", "table": "Table Recognition:", "formula": "Formula Recognition:"}
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

_model = _processor = None


def load():
    global _model, _processor
    if _model is None:
        _model = GlmOcrForConditionalGeneration.from_pretrained(REPO, dtype=torch.bfloat16, device_map=DEVICE)
        _processor = AutoProcessor.from_pretrained(REPO)
    return _model, _processor


def ocr(image, task="text", max_new_tokens=4096):
    """image: file path or PIL.Image."""
    model, processor = load()
    img = {"type": "image", "url": image} if isinstance(image, str) else {"type": "image", "image": image}
    messages = [{"role": "user", "content": [img, {"type": "text", "text": PROMPTS[task]}]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    ).to(model.device)
    inputs.pop("token_type_ids", None)
    # repetition_penalty: without it, delivery receipts loop on "27,000" until max_new_tokens (126s vs 21s)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.1)
    return processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


if __name__ == "__main__":
    print(ocr(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "text"))
