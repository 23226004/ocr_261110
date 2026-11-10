"""OCR inference (GLM-OCR family or general VLMs like Qwen3.5). Usage: python ocr.py <image> [text|table|formula]"""
import os
import sys
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

# same architecture: zai-org/GLM-OCR, trillionlabs/GravityOCR, trillionlabs/GravityOCR-KO
REPO = os.environ.get("OCR_MODEL", "trillionlabs/GravityOCR-KO")
# ponytail: GLM-OCR task prompts; the KO card documents none, so these are inherited from the base model
PROMPTS = {"text": "Text Recognition:", "table": "Table Recognition:", "formula": "Formula Recognition:"}
# general VLMs don't know the GLM-OCR task tags; same intent in plain words
VLM_PROMPTS = {
    "text": "이미지에 있는 모든 텍스트를 위에서 아래 순서대로 빠짐없이 그대로 옮겨 적으세요. 설명이나 요약 없이 텍스트만 출력하세요.",
    "table": "이미지의 표를 HTML <table>로 옮겨 적으세요. 설명 없이 표만 출력하세요.",
    "formula": "이미지의 수식을 LaTeX로 옮겨 적으세요. 설명 없이 수식만 출력하세요.",
}
IS_GLM_OCR = "OCR" in REPO
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

_model = _processor = None


def load():
    global _model, _processor
    if _model is None:
        _model = AutoModelForImageTextToText.from_pretrained(REPO, dtype=torch.bfloat16, device_map=DEVICE)
        _processor = AutoProcessor.from_pretrained(REPO)
    return _model, _processor


def ocr(image, task="text", max_new_tokens=4096):
    """image: file path or PIL.Image."""
    model, processor = load()
    img = {"type": "image", "url": image} if isinstance(image, str) else {"type": "image", "image": image}
    messages = [{"role": "user", "content": [img, {"type": "text", "text": (PROMPTS if IS_GLM_OCR else VLM_PROMPTS)[task]}]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
        # Qwen3.5 thinks by default; not passed to GLM-OCR, whose template would append /nothink and change the benchmarked prompt
        **({} if IS_GLM_OCR else {"enable_thinking": False}),
    ).to(model.device)
    inputs.pop("token_type_ids", None)
    # repetition_penalty: without it, delivery receipts loop on "27,000" until max_new_tokens (126s vs 21s)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.1)
    return processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


if __name__ == "__main__":
    print(ocr(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "text"))
