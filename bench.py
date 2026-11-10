"""Does the ground-truth value appear in GravityOCR's raw text? (OCR recall, upper bound for field extraction)
Usage: python bench.py [samples/real]"""
import json, re, sys, time, unicodedata
from pathlib import Path
import yaml
import ocr

root = Path(sys.argv[1] if len(sys.argv) > 1 else "samples/real")
gt = yaml.safe_load((root / "groundtruth.yaml").read_text())["samples"]
nfc = lambda s: unicodedata.normalize("NFC", s)
squash = lambda s: re.sub(r"[\s().·\-,]+", "", nfc(s)).lower()
# files may sit in subdirs (server: samples/extracted/) and macOS-made names are NFD, yaml is NFC
files = {nfc(p.name): p for p in root.rglob("*") if p.is_file()}


def check(text, g):
    t = nfc(text)
    flat = t.replace(",", "").replace(" ", "")
    dates = re.sub(r"\s*[./년월]\s*", "-", t)  # 2026.04.08 / 2026년 04월 -> 2026-04-08
    d, hm = g["transaction_datetime"].split()[0], g["transaction_datetime"].split()[1][:5]
    store = squash(g["store_name"])
    return {
        "amount": str(g["total_amount"]) in flat,
        "date": d in dates or d[2:] in dates,
        "minute": hm in t,
        "store_exact": store in squash(t),
        "store_partial": any(len(w) >= 2 and squash(w) in squash(t) for w in re.split(r"[\s()]+", g["store_name"])),
    }


rows = []
for g in gt:
    t0 = time.time()
    text = ocr.ocr(str(files[nfc(g["file"])]))
    r = {"file": g["file"], **check(text, g), "time": round(time.time() - t0, 1), "text": text}
    rows.append(r)
    print(r["file"], {k: v for k, v in r.items() if k not in ("file", "text")}, flush=True)

keys = ["amount", "date", "minute", "store_exact", "store_partial"]
print("\nTOTAL", {k: sum(r[k] for r in rows) for k in keys}, len(rows), "avg", round(sum(r["time"] for r in rows) / len(rows), 1), "s")
Path(f"bench_result_{ocr.REPO.split('/')[-1]}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
