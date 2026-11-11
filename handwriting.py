"""Handwriting OCR via GPT.
gpt-6.1-terra: CER 0.06% on 18 notebook pages, vs 12.08% for local GravityOCR-KO (see docs/gravityocr-ko.md).
Usage: python handwriting.py <image>"""
import os, sys
import gpt

MODEL = os.environ.get("HANDWRITING_MODEL", "gpt-6.1-terra")


def recognize(image_path, model=MODEL):
    """Returns the transcribed text, line breaks kept."""
    return gpt.ocr(image_path, model)[0]


if __name__ == "__main__":
    print(recognize(sys.argv[1]))
