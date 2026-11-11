"""Handwriting OCR: character error rate vs groundtruth.yaml (whitespace ignored).
Usage: python hw_bench.py [samples/handwriting]"""
import json, sys, time, unicodedata
from pathlib import Path
import yaml


def norm(s):
    return "".join(unicodedata.normalize("NFC", s).split())


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(pred, ref):
    p, r = norm(pred), norm(ref)
    return edit_distance(p, r) / max(len(r), 1)


assert edit_distance("kitten", "sitting") == 3
assert cer("사랑 의", "사랑의") == 0 and cer("", "가나") == 1

if __name__ == "__main__":
    import ocr
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "samples/handwriting")
    rows = []
    for g in yaml.safe_load((root / "groundtruth.yaml").read_text())["samples"]:
        t0 = time.time()
        text = ocr.ocr(str(root / g["file"]))
        r = {"file": g["file"], "cer": round(cer(text, g["text"]), 4), "time": round(time.time() - t0, 1), "text": text}
        rows.append(r)
        print(r["file"], f"CER={r['cer']:.1%}", f"{r['time']}s", flush=True)
    tot = sum(edit_distance(norm(r["text"]), norm(g["text"])) for r, g in zip(rows, yaml.safe_load((root / "groundtruth.yaml").read_text())["samples"]))
    n = sum(len(norm(g["text"])) for g in yaml.safe_load((root / "groundtruth.yaml").read_text())["samples"])
    print(f"\nTOTAL CER={tot / n:.2%} ({tot}/{n} chars) avg {sum(r['time'] for r in rows) / len(rows):.1f}s")
    Path(f"hw_result_{ocr.REPO.split('/')[-1]}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
