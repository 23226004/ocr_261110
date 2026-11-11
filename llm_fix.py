"""Correct OCR text with the OpenAI API (text-only or image + text).
Needs OPENAI_API_KEY in the environment or in .env (gitignored).
Usage: python llm_fix.py [text|image|ocr] [model]   (ocr = GPT reads the image alone, no OCR text)   -> CER before/after on samples/handwriting"""
import json, os, sys, time
from pathlib import Path
import gpt

MODEL = os.environ.get("OCR_FIX_MODEL", "gpt-5.4-mini")

RULES = """너는 OCR 교정기다. OCR 결과에서 잘못 인식된 글자만 고친다.
- 오탈자, 없는 단어, 문맥상 맞지 않는 단어만 고친다.
- 문장을 다듬거나 바꿔 쓰지 않는다. 원문에 없는 내용을 더하지 않는다.
- 줄 구조와 줄 수를 그대로 유지한다.
- 숫자는 확실한 근거가 없으면 바꾸지 않는다.
- 설명 없이 교정된 텍스트만 출력한다."""


def fix(text, image_path=None, model=MODEL):
    content = [{"type": "input_text", "text": f"OCR 결과:\n{text}"}]
    if image_path:
        content.insert(0, gpt.image_part(image_path))
        content.append({"type": "input_text", "text": "이미지와 대조해서, 이미지에 실제로 적힌 글자와 다른 부분만 고쳐라."})
    r = gpt.create(model=model, instructions=RULES, input=[{"role": "user", "content": content}])
    return r.output_text.strip(), r.usage


if __name__ == "__main__":
    import yaml
    from hw_bench import cer, edit_distance, norm
    mode = sys.argv[1] if len(sys.argv) > 1 else "text"
    model = sys.argv[2] if len(sys.argv) > 2 else MODEL
    root = Path("samples/handwriting")
    gt = {g["file"]: g["text"] for g in yaml.safe_load((root / "groundtruth.yaml").read_text())["samples"]}
    rows, err_before, err_after, n, tok_in, tok_out = [], 0, 0, 0, 0, 0
    for r in json.loads(Path("hw_result_GravityOCR-KO.json").read_text()):
        t0 = time.time()
        fixed, usage = gpt.ocr(root / r["file"], model) if mode == "ocr" else fix(r["text"], root / r["file"] if mode == "image" else None, model)
        ref = gt[r["file"]]
        b, a = edit_distance(norm(r["text"]), norm(ref)), edit_distance(norm(fixed), norm(ref))
        err_before, err_after, n = err_before + b, err_after + a, n + len(norm(ref))
        tok_in, tok_out = tok_in + usage.input_tokens, tok_out + usage.output_tokens
        rows.append({"file": r["file"], "cer_before": cer(r["text"], ref), "cer_after": cer(fixed, ref), "text": fixed})
        print(f"{r['file']} CER {b / len(norm(ref)):.1%} -> {a / len(norm(ref)):.1%}  {time.time() - t0:.1f}s", flush=True)
    print(f"\n[{mode} / {model}] TOTAL CER {err_before / n:.2%} -> {err_after / n:.2%}  tokens in={tok_in} out={tok_out}")
    Path(f"hw_fix_result_{mode}_{model}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
