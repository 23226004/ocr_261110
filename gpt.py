"""Shared OpenAI client and image encoding for the GPT-based modules. Needs OPENAI_API_KEY (env or .env, gitignored)."""
import base64, mimetypes, os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
_client = None
# reasoning effort ("none" turns it off; gpt-6.1-terra supports none, -sol/-astra only low+). Unset = model default.
REASONING = os.environ.get("GPT_REASONING")


def client():
    global _client
    _client = _client or OpenAI()
    return _client


def create(**kw):
    if REASONING:
        kw["reasoning"] = {"effort": REASONING}
    return client().responses.create(**kw)


def image_part(path):
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    b64 = base64.b64encode(Path(path).read_bytes()).decode()
    return {"type": "input_image", "image_url": f"data:{mime};base64,{b64}", "detail": "high"}


OCR_RULES = "이미지에 있는 모든 글자(손글씨 포함)를 위에서 아래 순서대로 빠짐없이 그대로 옮겨 적어라. 줄바꿈을 유지하고, 설명 없이 텍스트만 출력한다."


def ocr(image_path, model, instructions=OCR_RULES):
    """Plain-text OCR of one image. Returns (text, usage)."""
    r = create(model=model, instructions=instructions,
               input=[{"role": "user", "content": [image_part(image_path)]}])
    return r.output_text.strip(), r.usage
